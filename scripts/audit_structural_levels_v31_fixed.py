from __future__ import annotations
"""Audit descriptif V3.1 des SL structurels sur candidats existants.

Lecture: Train/Validation uniquement. Ne lit ni Test ni labels futurs.
Aucun backtest, aucune sortie de trade et aucun entrainement.

Usage (racine projet):
    python scripts/audit_structural_levels_v31.py
    python scripts/audit_structural_levels_v31.py --output data/evaluation/v31-structural-audit
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ('train', 'validation')
METHODS = ('fractal', 'zigzag')
TIMEFRAMES = ('1h', '4h')
BASE = Path('data/evaluation')


def utc(series):
    return pd.to_datetime(series, utc=True, errors='raise').dt.as_unit('ns')


def check_unique(df, column, name):
    if df[column].isna().any() or df[column].duplicated().any():
        raise ValueError(f'{name}: {column} nul ou duplique')


def load_candidates(split, base):
    f = base / 'v3-candidate-events' / f'{split}_candidates.parquet'
    df = pd.read_parquet(f)
    needed = {'event_id', 'signal_timestamp', 'known_at', 'side'}
    if not needed.issubset(df.columns):
        raise ValueError(f'{f}: colonnes manquantes {needed - set(df.columns)}')
    df = df[['event_id', 'signal_timestamp', 'known_at', 'side']].copy()
    df['signal_timestamp'] = utc(df['signal_timestamp'])
    df['known_at'] = utc(df['known_at'])
    check_unique(df, 'event_id', str(f))
    if not df.side.isin(['LONG', 'SHORT']).all():
        raise ValueError('Direction candidate inattendue')
    if not df.known_at.eq(df.signal_timestamp + pd.Timedelta(hours=1)).all():
        raise ValueError('Horloge candidat incoherente')
    return df


def attach_entry(candidates, split, split_dir):
    """L'entree theoretique est l'open de la bougie 1H de known_at.
    Pas de recours a des bougies externes au split.
    """
    f = split_dir / f'{split}.parquet'
    prices = pd.read_parquet(f, columns=['timestamp', 'open'])
    prices['timestamp'] = utc(prices['timestamp'])
    check_unique(prices, 'timestamp', str(f))
    result = candidates.merge(prices.rename(columns={'timestamp':'known_at', 'open':'entry_open'}),
                              on='known_at', how='left', validate='many_to_one')
    return result


def fractal_levels(split, tf, base):
    f = base / 'v31-market-structure' / f'{split}_structure.parquet'
    needed = ['known_at', f'{tf}_last_swing_high', f'{tf}_last_swing_low',
              f'{tf}_available_at', f'{tf}_swing_high_at', f'{tf}_swing_low_at',
              f'{tf}_pivot_high_confirmed', f'{tf}_pivot_low_confirmed']
    frame = pd.read_parquet(f, columns=needed)
    # Les flags de confirmation sont des booléens : ne jamais les convertir en dates.
    for c in ['known_at', f'{tf}_available_at', f'{tf}_swing_high_at', f'{tf}_swing_low_at']:
        frame[c] = utc(frame[c])
    for c in [f'{tf}_pivot_high_confirmed', f'{tf}_pivot_low_confirmed']:
        if not (pd.api.types.is_bool_dtype(frame[c]) or frame[c].dropna().isin([True, False]).all()):
            raise ValueError(f'{f}: indicateur de pivot non booleen : {c}')
        frame[c] = frame[c].fillna(False).astype(bool)
    check_unique(frame, 'known_at', str(f))
    # Reconstruct exact confirmation clock independently for HIGH and LOW,
    # as seen in the aligned 1H records. Event flags are one-bar impulses.
    frame = frame.sort_values('known_at').reset_index(drop=True)
    for kind in ('high', 'low'):
        impulse = frame[f'{tf}_pivot_{kind}_confirmed'].eq(True)
        frame[f'{kind}_confirmed_at'] = frame['known_at'].where(impulse).ffill()
    return frame.rename(columns={
        f'{tf}_last_swing_high':'high_price', f'{tf}_last_swing_low':'low_price',
        f'{tf}_available_at':'state_confirmed_at',
        f'{tf}_swing_high_at':'high_pivot_at', f'{tf}_swing_low_at':'low_pivot_at'
    })


def zigzag_levels(split, tf, base):
    f = base / 'v31-zigzag-causal' / f'{split}_zigzag_events.parquet'
    data = pd.read_parquet(f)
    needed = {'timeframe', 'confirmation_at', 'pivot_at', 'kind', 'price'}
    if not needed.issubset(data.columns):
        raise ValueError(f'{f}: colonnes manquantes {needed - set(data.columns)}')
    data = data.loc[data.timeframe.eq(tf)].copy()
    for c in ('confirmation_at', 'pivot_at'):
        data[c] = utc(data[c])
    if not data.empty and not data.confirmation_at.ge(data.pivot_at).all():
        raise ValueError(f'{f}: pivot confirme avant sa formation')
    if not data.kind.isin(['HIGH', 'LOW']).all():
        raise ValueError(f'{f}: kind inattendu')
    data = data.sort_values(['confirmation_at', 'pivot_at'], kind='stable')
    if data.duplicated(['kind', 'confirmation_at']).any():
        raise ValueError(f'{f}: confirmation dupliquee pour un meme type')
    return data


def attach_zigzag_side(frame, events, kind, prefix):
    event = events.loc[events.kind.eq(kind), ['confirmation_at','pivot_at','price']].copy()
    event = event.rename(columns={'confirmation_at':f'{prefix}_confirmed_at',
                                   'pivot_at':f'{prefix}_pivot_at', 'price':f'{prefix}_price'})
    event = event.sort_values(f'{prefix}_confirmed_at')
    base = frame.sort_values('known_at')
    if event.empty:
        for c in event.columns:
            base[c] = pd.NaT if c.endswith('_at') else np.nan
        return base
    result = pd.merge_asof(base, event, left_on='known_at',
                           right_on=f'{prefix}_confirmed_at', direction='backward',
                           allow_exact_matches=True)
    valid = result[f'{prefix}_confirmed_at'].notna()
    if (result.loc[valid, f'{prefix}_confirmed_at'] > result.loc[valid, 'known_at']).any():
        raise AssertionError('Fuite future ZigZag')
    return result


def enrich(base_df, split, method, tf, base):
    df = base_df.copy().sort_values('known_at').reset_index(drop=True)
    if method == 'fractal':
        state = fractal_levels(split, tf, base)
        df = df.merge(state, on='known_at', how='left', validate='many_to_one')
        # Clock des impulsions : pas la derniere barre 4H/1H achevee.
    elif method == 'zigzag':
        events = zigzag_levels(split, tf, base)
        df = attach_zigzag_side(df, events, 'HIGH', 'high')
        df = attach_zigzag_side(df, events, 'LOW', 'low')
    else:
        raise ValueError(method)
    use_high = df.side.eq('SHORT')
    df['stop_reference'] = np.where(use_high, 'HIGH', 'LOW')
    df['stop_pivot_price'] = np.where(use_high, df.high_price, df.low_price)
    df['stop_confirmed_at'] = pd.to_datetime(
        np.where(use_high, df.high_confirmed_at, df.low_confirmed_at), utc=True)
    df['stop_pivot_at'] = pd.to_datetime(
        np.where(use_high, df.high_pivot_at, df.low_pivot_at), utc=True)
    return df


def evaluate(df, buffer_bps, min_distance_pct, max_distance_pct):
    df = df.copy()
    # Marge fixe sans ATR : 5 bps par defaut, audit descriptif uniquement.
    factor = buffer_bps / 10000.
    df['stop_price'] = np.where(df.side.eq('LONG'),
                                df.stop_pivot_price * (1.0-factor),
                                df.stop_pivot_price * (1.0+factor))
    df['distance_pct'] = np.where(df.side.eq('LONG'),
        (df.entry_open-df.stop_price)/df.entry_open*100.,
        (df.stop_price-df.entry_open)/df.entry_open*100.)
    df['confirmation_lag_hours'] = (
        (df.known_at - df.stop_confirmed_at).dt.total_seconds()/3600.)
    conditions = [
        df.entry_open.isna() | df.entry_open.le(0),
        df.stop_pivot_price.isna() | df.stop_confirmed_at.isna(),
        df.stop_confirmed_at.gt(df.known_at),
        df.distance_pct.le(0),
        df.distance_pct.lt(min_distance_pct),
        df.distance_pct.gt(max_distance_pct),
    ]
    labels = ['ENTRY_OUTSIDE_SPLIT','NO_CONFIRMED_PIVOT','FUTURE_CONFIRMATION',
              'STOP_ALREADY_INVALID','TOO_CLOSE','TOO_FAR']
    df['status'] = np.select(conditions, labels, default='VALID_REFERENCE')
    if (df.status == 'FUTURE_CONFIRMATION').any():
        raise AssertionError('Confirmation posterieure a la decision')
    df.loc[df.status.eq('ENTRY_OUTSIDE_SPLIT'), ['stop_price','distance_pct']] = np.nan
    return df


def summarize(df):
    counts = df.status.value_counts().to_dict()
    valid = df.loc[df.status.eq('VALID_REFERENCE'), 'distance_pct']
    possible = df.loc[df.status.isin(['VALID_REFERENCE','TOO_CLOSE','TOO_FAR']), 'distance_pct']
    return {'candidates':int(len(df)), 'status':{str(k):int(v) for k,v in counts.items()},
            'reference_distance_pct':{
                'count':int(possible.count()),
                'p10':float(possible.quantile(.1)) if len(possible) else None,
                'median':float(possible.median()) if len(possible) else None,
                'p90':float(possible.quantile(.9)) if len(possible) else None,
            }, 'valid_reference_count':int(len(valid))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--splits', type=Path, default=Path('data/splits'))
    p.add_argument('--output', type=Path, default=BASE/'v31-structural-audit')
    p.add_argument('--buffer-bps', type=float, default=5.)
    p.add_argument('--min-distance-pct', type=float, default=.10)
    p.add_argument('--max-distance-pct', type=float, default=5.)
    args = p.parse_args()
    if args.buffer_bps < 0 or args.min_distance_pct < 0 or args.max_distance_pct <= args.min_distance_pct:
        p.error('Seuils descriptifs invalides')
    outputs, report = {}, {}
    for split in SPLITS:
        candidates = attach_entry(load_candidates(split, args.base), split, args.splits)
        check_unique(candidates, 'event_id', split)
        report[split] = {}
        pieces = []
        for method in METHODS:
            for tf in TIMEFRAMES:
                result = evaluate(enrich(candidates, split, method, tf, args.base),
                                  args.buffer_bps, args.min_distance_pct, args.max_distance_pct)
                if len(result) != len(candidates):
                    raise AssertionError('Perte de candidats')
                result.insert(0,'method',method)
                result.insert(1,'timeframe',tf)
                pieces.append(result)
                summary = summarize(result)
                report[split][f'{method}_{tf}'] = summary
                print(f'{split.upper():10} {method:8} {tf}: '+
                      f'{summary["candidates"]:5} candidats | '+
                      ', '.join(f'{k}={v}' for k,v in summary['status'].items()))
        outputs[split] = pd.concat(pieces, ignore_index=True)
        if len(outputs[split]) != 4 * len(candidates):
            raise AssertionError('Nombre de combinaisons inattendu')
    args.output.mkdir(parents=True, exist_ok=True)
    for split, df in outputs.items():
        columns = ['method','timeframe','event_id','signal_timestamp','known_at','side',
                   'entry_open','high_price','low_price','high_confirmed_at','low_confirmed_at',
                   'high_pivot_at','low_pivot_at','stop_reference','stop_pivot_price',
                   'stop_confirmed_at','stop_pivot_at','stop_price','distance_pct',
                   'confirmation_lag_hours','status']
        df[columns].to_parquet(args.output/f'{split}_audit.parquet', index=False)
    manifest = {
        'version':'v31-structural-audit-1', 'splits':list(SPLITS), 'methods':list(METHODS),
        'timeframes':list(TIMEFRAMES),
        'buffer_bps':args.buffer_bps, 'min_distance_pct':args.min_distance_pct,
        'max_distance_pct':args.max_distance_pct,
        'entry_clock':'known_at = next 1H candle open; missing entry remains invalid',
        'status_semantics':'descriptive only; no trades, outcomes, sizing or TP computed',
        'IMPORTANT_split_reset':'Existing fractal & zigzag outputs reset at start of each split. '
            'Boundary coverage/quality are provisional; reconstruct chronological history before backtesting.',
        'fractal_confirmed_at':'Exact 1H known_at of each pivot confirmation impulse; '
            'forward-filled separately for HIGH and LOW.',
        'no_test':True, 'no_future_labels':True, 'report':report,
    }
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'\nSorties : {args.output.resolve()}')
    print('Audit descriptif uniquement. Test non lu. Aucun ordre simule.')


if __name__ == '__main__':
    main()

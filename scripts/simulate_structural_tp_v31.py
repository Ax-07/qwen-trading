from __future__ import annotations
"""V3.1 -- simulation structurelle descriptive, Train/Validation uniquement.

Entrée: ouverture de la bougie suivant le signal (known_at).
Niveaux: data/evaluation/v31-continuity-audit/{split}_audit.parquet.
Six cibles fixes, 1R,1.5R,2R,3R,4R,5R. Aucun entraînement/Test.
Exécuter depuis la racine : python scripts/simulate_structural_tp_v31.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ('train', 'validation')
MULTIPLES = (1., 1.5, 2., 3., 4., 5.)
REQUIRED_AUDIT = {'event_id','known_at','side','method','timeframe','entry_open',
                  'stop_price','status','stop_confirmed_at','signal_timestamp'}
REQUIRED_MARKET = {'timestamp','open','high','low','close'}


def utc(x):
    return pd.to_datetime(x, utc=True).dt.as_unit('ns')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def simulate_one(bars, side, stop, multiple, fee_bps=5., slippage_bps=2.):
    """bars contient les H bougies consécutives à partir de l'entrée.
    Hypothèse prudente : SL gagne en cas d'ambiguïté intrabar.
    Ouverture défavorable d'une nouvelle bougie => sortie au prix d'ouverture.
    """
    if len(bars) == 0 or side not in ('LONG','SHORT') or multiple <= 0:
        raise ValueError('Paramètres de simulation invalides')
    raw_entry = float(bars.iloc[0]['open'])
    stop = float(stop)
    if not np.isfinite(raw_entry) or raw_entry <= 0 or not np.isfinite(stop) or stop <= 0:
        raise ValueError('Prix invalide')
    direction = 1 if side == 'LONG' else -1
    risk = direction * (raw_entry - stop)
    if risk <= 0:
        raise ValueError('Stop du mauvais côté')
    target = raw_entry + direction * multiple * risk
    if target <= 0:
        raise ValueError('Target non positif')
    slip = slippage_bps / 10000
    fee = fee_bps / 10000
    entry_fill = raw_entry * (1 + direction * slip)
    for offset, bar in enumerate(bars.itertuples(index=False)):
        op, hi, lo = float(bar.open), float(bar.high), float(bar.low)
        if side == 'LONG':
            if offset and op <= stop:
                raw_exit, reason = op, 'SL_GAP'
            elif offset and op >= target:
                raw_exit, reason = op, 'TP_GAP'
            elif lo <= stop:
                raw_exit, reason = stop, 'SL'
            elif hi >= target:
                raw_exit, reason = target, 'TP'
            else:
                continue
        else:
            if offset and op >= stop:
                raw_exit, reason = op, 'SL_GAP'
            elif offset and op <= target:
                raw_exit, reason = op, 'TP_GAP'
            elif hi >= stop:
                raw_exit, reason = stop, 'SL'
            elif lo <= target:
                raw_exit, reason = target, 'TP'
            else:
                continue
        exit_offset = offset
        break
    else:
        exit_offset = len(bars) - 1
        raw_exit, reason = float(bars.iloc[-1]['close']), 'TIME'
    exit_fill = raw_exit * (1 - direction * slip)
    gross = direction * (exit_fill - entry_fill) / entry_fill
    net = gross - fee * (1 + exit_fill / entry_fill)
    return dict(reason=reason, exit_offset=exit_offset,
                entry_raw=raw_entry, entry_fill=entry_fill,
                stop_price=stop, target_price=target,
                risk_price=risk, exit_raw=raw_exit, exit_fill=exit_fill,
                gross_return=gross, net_return=net,
                net_r=net * entry_fill / risk,
                duration_bars=exit_offset + 1)


def validate_audit(df):
    missing = REQUIRED_AUDIT - set(df.columns)
    if missing: raise ValueError(f'Colonnes audit manquantes: {sorted(missing)}')
    df = df.copy()
    for col in ('known_at','stop_confirmed_at','signal_timestamp'):
        df[col] = utc(df[col])
    if df.duplicated(['event_id','method','timeframe']).any():
        raise ValueError('Lignes audit dupliquées')
    if not df.side.isin(('LONG','SHORT')).all():
        raise ValueError('Direction non reconnue')
    if not df.method.isin(('fractal','zigzag')).all() or not df.timeframe.isin(('1h','4h')).all():
        raise ValueError('Méthode/timeframe non reconnus')
    if not (df.known_at == df.signal_timestamp + pd.Timedelta(hours=1)).all():
        raise ValueError('Horodatage du signal incohérent')
    known = df.stop_confirmed_at.notna()
    if (df.loc[known,'stop_confirmed_at'] > df.loc[known,'known_at']).any():
        raise ValueError('FUITE: pivot confirmé après le signal')
    return df


def prepare_market(path):
    df = pd.read_parquet(path, columns=list(REQUIRED_MARKET))
    df['timestamp'] = utc(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    if df.timestamp.duplicated().any(): raise ValueError('Bougies dupliquées')
    ohlc = df[['open','high','low','close']]
    if not np.isfinite(ohlc.to_numpy(dtype=float)).all() or (ohlc <= 0).any().any():
        raise ValueError('Prix OHLC manquants ou invalides')
    if (df.high < df[['open','close','low']].max(axis=1)).any() or (df.low > df[['open','close','high']].min(axis=1)).any():
        raise ValueError('OHLC contradictoire')
    return df


def simulate_split(audit, market, horizon, fee_bps, slippage_bps):
    """Résultats par candidat ET simulations portefeuille non chevauchantes.
    Le portefeuille est recalculé indépendamment pour chaque couple méthode/TF/R.
    Tous les candidats initiaux sont conservés dans la table des statuts.
    """
    audit = validate_audit(audit)
    if not len(audit): raise ValueError('Audit vide')
    lookup = pd.Series(np.arange(len(market)), index=market.timestamp)
    results = []
    audit = audit.sort_values(['method','timeframe','known_at','event_id']).reset_index(drop=True)
    for (method, tf), group in audit.groupby(['method','timeframe'], sort=True):
        if group.event_id.nunique() != len(group): raise ValueError('Candidat dupliqué')
        for multiple in MULTIPLES:
            busy_until = -1
            equity = 1.0
            for row in group.itertuples(index=False):
                base = dict(method=method,timeframe=tf,multiple_r=multiple,
                            event_id=row.event_id,known_at=row.known_at,side=row.side,
                            audit_status=row.status)
                result = dict(base, eligibility='UNKNOWN', portfolio_status='NOT_EVALUATED',
                              entry_at=pd.NaT,exit_at=pd.NaT,exit_index=np.nan,
                              net_return=np.nan,net_r=np.nan,reason=None,
                              equity_before=np.nan,equity_after=np.nan)
                ix = lookup.get(row.known_at, np.nan)
                if pd.isna(ix):
                    result['eligibility'] = 'ENTRY_OUTSIDE_SPLIT'
                else:
                    ix = int(ix)
                    end = ix + horizon - 1
                    if end >= len(market) or not (market.timestamp.iloc[ix:end+1].diff().dropna() == pd.Timedelta(hours=1)).all():
                        result['eligibility'] = 'INCOMPLETE_HORIZON'
                    elif row.status != 'VALID_REFERENCE':
                        result['eligibility'] = str(row.status)
                    elif not np.isfinite(row.stop_price) or row.stop_price <= 0:
                        result['eligibility'] = 'INVALID_STOP'
                    elif not np.isclose(float(row.entry_open),float(market.at[ix,'open']),rtol=1e-10):
                        raise ValueError(f'Incohérence prix entrée: {row.event_id}')
                    else:
                        result['eligibility'] = 'ELIGIBLE'
                        trade = simulate_one(market.iloc[ix:end+1], row.side, row.stop_price,
                                             multiple, fee_bps, slippage_bps)
                        out_i = ix + trade['exit_offset']
                        result.update(trade)
                        result['entry_at'] = market.at[ix,'timestamp']
                        result['exit_at'] = market.at[out_i,'timestamp'] + pd.Timedelta(hours=1)
                        result['exit_index'] = out_i
                        # Entrée après sortie de la bougie précédente uniquement.
                        # Si la position précédente sort à la clôture de i, une
                        # nouvelle entrée à l'ouverture de i+1 est autorisée.
                        if ix <= busy_until:
                            result['portfolio_status'] = 'SKIPPED_OVERLAP'
                        else:
                            result['portfolio_status'] = 'TAKEN'
                            result['equity_before'] = equity
                            equity *= 1 + trade['net_return']
                            result['equity_after'] = equity
                            busy_until = out_i
                results.append(result)
    out = pd.DataFrame(results)
    count = audit.groupby(['method','timeframe']).size().to_dict()
    for (method, tf, multiple), group in out.groupby(['method','timeframe','multiple_r']):
        if len(group) != count[method, tf]:
            raise AssertionError('Perte de candidats entre scénarios')
        taken = group[group.portfolio_status == 'TAKEN'].sort_values('entry_at')
        if len(taken) > 1 and (taken.entry_at.iloc[1:].reset_index(drop=True) <
                              taken.exit_at.iloc[:-1].reset_index(drop=True)).any():
            raise AssertionError('Chevauchement des positions')
    return out


def report(frame):
    output = []
    for (method, tf, multiple), group in frame.groupby(['method','timeframe','multiple_r'], sort=True):
        eligible = group[group.eligibility == 'ELIGIBLE']
        taken = group[group.portfolio_status == 'TAKEN']
        output.append(dict(method=method,timeframe=tf,tp_r=float(multiple),
            candidates=int(len(group)),eligible=int(len(eligible)),portfolio_trades=int(len(taken)),
            skipped_overlap=int((group.portfolio_status == 'SKIPPED_OVERLAP').sum()),
            eligibility={str(k):int(v) for k,v in group.eligibility.value_counts().items()},
            candidate_net_mean_pct=float(eligible.net_return.mean()*100) if len(eligible) else None,
            candidate_tp_rate_pct=float(eligible.reason.str.startswith('TP').mean()*100) if len(eligible) else None,
            portfolio_net_return_pct=float((taken.equity_after.iloc[-1]-1)*100) if len(taken) else 0.,
            portfolio_tp_rate_pct=float(taken.reason.str.startswith('TP').mean()*100) if len(taken) else None,
            portfolio_net_r_mean=float(taken.net_r.mean()) if len(taken) else None))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits',type=Path,default=Path('data/splits'))
    parser.add_argument('--base',type=Path,default=Path('data/evaluation/v31-continuity-audit'))
    parser.add_argument('--output',type=Path,default=Path('data/evaluation/v31-structural-tp'))
    parser.add_argument('--horizon',type=int,default=12)
    parser.add_argument('--fee-bps',type=float,default=5.)
    parser.add_argument('--slippage-bps',type=float,default=2.)
    args = parser.parse_args()
    if args.horizon < 1 or args.fee_bps < 0 or args.slippage_bps < 0:
        parser.error('Paramètres invalides')
    if args.horizon != 12:
        parser.error('Première comparaison verrouillée à horizon=12; ne pas optimiser Validation')
    products = {}; summaries = {}; fingerprints = {}
    for split in SPLITS:
        audit_path = args.base/f'{split}_audit.parquet'
        market_path = args.splits/f'{split}.parquet'
        market = prepare_market(market_path)
        audit = pd.read_parquet(audit_path)
        result = simulate_split(audit, market,args.horizon,args.fee_bps,args.slippage_bps)
        products[split] = result
        summaries[split] = report(result)
        fingerprints[split] = {'audit_sha256':sha256(audit_path),'split_sha256':sha256(market_path)}
        print(f'\n{split.upper()} | {len(audit)} lignes d’audit, {len(result)} scénarios')
        for row in summaries[split]:
            print(f"{row['method']:7} {row['timeframe']} TP {row['tp_r']:>3g}R | "
                  f"eligible={row['eligible']:4} taken={row['portfolio_trades']:3} "
                  f"mean_candidate={row['candidate_net_mean_pct']:+.3f}% "
                  f"portfolio={row['portfolio_net_return_pct']:+.2f}%" if row['candidate_net_mean_pct'] is not None else str(row))
    args.output.mkdir(parents=True,exist_ok=True)
    for split, frame in products.items():
        target = args.output/f'{split}_scenarios.parquet'
        if target.exists(): raise SystemExit(f'Refus d’écraser {target}')
    for split, frame in products.items():
        frame.to_parquet(args.output/f'{split}_scenarios.parquet',index=False)
    manifest = {'version':'v31-structural-tp-v1','splits':list(SPLITS),
        'source':'audit causal à continuité prudente, sans reconstruction des bougies purgées',
        'test_loaded':False,'training':False,'horizon_bars':args.horizon,
        'multiples_r':list(MULTIPLES), 'fee_bps_per_side':args.fee_bps,
        'slippage_bps_per_side':args.slippage_bps,
        'entry':'prochaine ouverture 1H', 'stop':'niveau fixe issu audit VALID_REFERENCE, buffer inclus',
        'exit':'SL-first, gaps au prix ouverture, timeout clôture de la 12e bougie',
        'portfolio':'un trade à la fois par méthode/timeframe/TP, allocation 100% du capital; equity composée',
        'caveats':['Fenêtres purgées non reconstruites',
                   'Stop buffer et seuils de validité hérités de l’audit; pas de sélection optimisée',
                   'Comparaisons inter-multiples sur mêmes candidats initiaux, trades retenus variables'],
        'inputs_sha256':fingerprints,'summary':summaries}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nAucun Test lu, aucun entraînement. Résultats exploratoires uniquement.')


if __name__ == '__main__':
    main()

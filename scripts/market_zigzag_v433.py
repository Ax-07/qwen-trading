"""Qwen Trading V4.3.3 — ZigZag causal à retournement ATR.

Entrée : bougies Binance 1m, timestamp UTC = ouverture. Aucun entraînement.
Les points extrêmes ne deviennent connus QU'À confirmation, et ne sont jamais
publiés comme niveaux confirmés avant cette date. Seuil ATR figé à l'extrême.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from prototype_multitimeframe_v42 import completed_bars, FREQS

FIELDS = ['zz_event', 'zz_confirmed', 'zz_pivot_at', 'zz_confirmed_at',
          'zz_pivot_price', 'zz_reversal_atr', 'zz_reversal_pct',
          'zz_last_high', 'zz_last_low', 'zz_direction', 'zz_age_bars']


def zigzag_for_bars(bars: pd.DataFrame, minutes: int, atr_period: int = 14,
                    reversal_atr: float = 2.0) -> pd.DataFrame:
    """Une ligne par bougie clôturée; état réinitialisé à chaque lacune.

    ATR: moyenne simple des 14 true ranges consécutifs; première TR du segment
    indisponible. Un pivot HIGH est confirmé quand CLOSE <= highest HIGH - k*ATR
    figé à ce HIGH. Inverse pour LOW. Les extrêmes intrabar de la bougie
    confirmatrice ne sont pas pris en compte AVANT le test de retournement.
    """
    if minutes < 1 or atr_period < 1 or not np.isfinite(reversal_atr) or reversal_atr <= 0:
        raise ValueError('minutes, atr_period et reversal_atr doivent être positifs')
    required = ('timestamp', 'known_at', 'high', 'low', 'close')
    if any(c not in bars.columns for c in required):
        raise ValueError('Colonnes requises : ' + ', '.join(required))
    b = bars.sort_values('timestamp').reset_index(drop=True).copy()
    if b.empty:
        return pd.DataFrame(columns=['known_at', *FIELDS])
    b['timestamp'] = pd.to_datetime(b.timestamp, utc=True).astype('datetime64[ns, UTC]')
    b['known_at'] = pd.to_datetime(b.known_at, utc=True).astype('datetime64[ns, UTC]')
    step = pd.Timedelta(minutes=minutes)
    if b.timestamp.duplicated().any() or (b.known_at != b.timestamp + step).any():
        raise ValueError('Horodatages invalides')
    x = b[['high', 'low', 'close']].to_numpy(dtype=float)
    if not np.isfinite(x).all() or (x <= 0).any() or (b.high < b[['low','close']].max(axis=1)).any() or (b.low > b[['high','close']].min(axis=1)).any():
        raise ValueError('OHLC invalides')
    rows = []
    chunks = b.timestamp.diff().ne(step).cumsum()
    for _, part in b.groupby(chunks, sort=False):
        part = part.reset_index(drop=True)
        h, l, c = (part[k].to_numpy(dtype=float) for k in ('high','low','close'))
        tr = np.full(len(part), np.nan)
        if len(part) > 1:
            tr[1:] = np.maximum.reduce([h[1:] - l[1:], abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])])
        atr = pd.Series(tr).rolling(atr_period, min_periods=atr_period).mean().to_numpy()
        mode = 'SEEK'
        hi = lo = np.nan
        hi_i = lo_i = -1
        hi_atr = lo_atr = np.nan
        last_high = last_low = np.nan
        last_type = 'NONE'
        last_confirm_i = -1
        for i in range(len(part)):
            typ = 'NONE'
            pivot_i = -1
            pivot_price = rev_atr = rev_pct = np.nan
            # Warmup is part of the causal observation. No provisional pivot emitted.
            if np.isfinite(atr[i]) and atr[i] > 0:
                if mode == 'SEEK':
                    if not np.isfinite(hi):
                        hi, lo, hi_i, lo_i = h[i], l[i], i, i
                        hi_atr = lo_atr = atr[i]
                    else:
                        if h[i] > hi:
                            hi, hi_i, hi_atr = h[i], i, atr[i]
                        if l[i] < lo:
                            lo, lo_i, lo_atr = l[i], i, atr[i]
                    # Initial directional choice based on completed close; no event yet.
                    if c[i] >= lo + reversal_atr * lo_atr:
                        mode = 'UP'
                        hi, hi_i, hi_atr = h[i], i, atr[i]
                    elif c[i] <= hi - reversal_atr * hi_atr:
                        mode = 'DOWN'
                        lo, lo_i, lo_atr = l[i], i, atr[i]
                elif mode == 'UP':
                    # Check existing extreme before updating it with current bar.
                    if i > hi_i and c[i] <= hi - reversal_atr * hi_atr:
                        typ, pivot_i, pivot_price = 'HIGH', hi_i, hi
                        rev_atr = (hi - c[i]) / hi_atr
                        rev_pct = (hi - c[i]) / hi
                        last_high = hi
                        mode = 'DOWN'
                        lo, lo_i, lo_atr = l[i], i, atr[i]
                    elif h[i] > hi:
                        hi, hi_i, hi_atr = h[i], i, atr[i]
                else:
                    if i > lo_i and c[i] >= lo + reversal_atr * lo_atr:
                        typ, pivot_i, pivot_price = 'LOW', lo_i, lo
                        rev_atr = (c[i] - lo) / lo_atr
                        rev_pct = (c[i] - lo) / lo
                        last_low = lo
                        mode = 'UP'
                        hi, hi_i, hi_atr = h[i], i, atr[i]
                    elif l[i] < lo:
                        lo, lo_i, lo_atr = l[i], i, atr[i]
            if typ != 'NONE':
                last_type, last_confirm_i = typ, i
            rows.append(dict(known_at=part.known_at.iloc[i], zz_event=typ,
                             zz_confirmed=typ != 'NONE',
                             zz_pivot_at=part.timestamp.iloc[pivot_i] if pivot_i >= 0 else pd.NaT,
                             zz_confirmed_at=part.known_at.iloc[i] if typ != 'NONE' else pd.NaT,
                             zz_pivot_price=pivot_price, zz_reversal_atr=rev_atr,
                             zz_reversal_pct=rev_pct, zz_last_high=last_high,
                             zz_last_low=last_low, zz_direction=mode,
                             zz_age_bars=i - last_confirm_i if last_confirm_i >= 0 else np.nan))
    result = pd.DataFrame(rows)
    for col in ('known_at', 'zz_pivot_at', 'zz_confirmed_at'):
        result[col] = pd.to_datetime(result[col], utc=True).astype('datetime64[ns, UTC]')
    return result


def build_zigzag_snapshots(minutes_frame: pd.DataFrame, atr_period: int = 14,
                            reversal_atr: float = 2.0):
    bars = {tf: completed_bars(minutes_frame, n) for tf, n in FREQS.items()}
    events = {tf: zigzag_for_bars(b, n, atr_period, reversal_atr) for tf, (b, n) in
              ((tf, (bars[tf], FREQS[tf])) for tf in FREQS)}
    out = bars['5m'][['timestamp', 'known_at', 'close']].rename(columns={
        'timestamp':'bar_open_5m', 'known_at':'decision_at','close':'close_5m'})
    out['decision_at'] = pd.to_datetime(out.decision_at, utc=True).astype('datetime64[ns, UTC]')
    out = out.sort_values('decision_at')
    for tf, n in FREQS.items():
        source = events[tf].rename(columns={'known_at': f'known_at_{tf}',
                                                  **{field: f'{field}_{tf}' for field in FIELDS}})
        if source.empty:
            for field in FIELDS:
                out[f'{field}_{tf}'] = pd.NA
            out[f'known_at_{tf}'] = pd.NaT
            continue
        source[f'known_at_{tf}'] = pd.to_datetime(source[f'known_at_{tf}'], utc=True).astype('datetime64[ns, UTC]')
        out = pd.merge_asof(out, source.sort_values(f'known_at_{tf}'),
                            left_on='decision_at', right_on=f'known_at_{tf}',
                            direction='backward', allow_exact_matches=True,
                            tolerance=pd.Timedelta(minutes=n))
        # An HTF confirmation is an impulse, not a state persisting 15/60/240m.
        fresh = out.decision_at.eq(out[f'known_at_{tf}'])
        out[f'zz_confirmed_{tf}'] = (out[f'zz_confirmed_{tf}'].eq(True) & fresh).astype(bool)
        out.loc[~fresh, f'zz_event_{tf}'] = 'NONE'
        for field in ('zz_pivot_at', 'zz_confirmed_at', 'zz_pivot_price', 'zz_reversal_atr', 'zz_reversal_pct'):
            out.loc[~fresh, f'{field}_{tf}'] = pd.NaT if field.endswith('_at') else np.nan
    return out.reset_index(drop=True), events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/evaluation/v421-binance-1m/BTCUSDT/2024-01.parquet'))
    parser.add_argument('--output', type=Path, default=Path('data/evaluation/v433-zigzag/BTCUSDT-2024-01.parquet'))
    parser.add_argument('--atr-period', type=int, default=14)
    parser.add_argument('--reversal-atr', type=float, default=2.0)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'Entrée absente : {args.input}')
    if args.output.exists():
        parser.error(f'Refus écrasement : {args.output}')
    snapshots, events = build_zigzag_snapshots(pd.read_parquet(args.input), args.atr_period, args.reversal_atr)
    print('Snapshots 5m :', len(snapshots))
    for tf, df in events.items():
        print(f'{tf:>3}: bougies={len(df)} zigzag_high={int(df.zz_event.eq("HIGH").sum())} zigzag_low={int(df.zz_event.eq("LOW").sum())}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(args.output, index=False)
    print('Parquet :', args.output.resolve())
    print('Aucune source modifiée. Aucun entraînement ni trading.')

if __name__ == '__main__':
    main()

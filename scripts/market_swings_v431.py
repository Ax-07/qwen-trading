"""Qwen Trading V4.3.1 — confirmed fractal swings, HH/LH/HL/LL, causal 5m snapshots.

Input: existing Binance 1m parquet; output: separate structure snapshots parquet.
Does not compute BOS/CHoCH/ZigZag (future increments). No trades or labels.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from prototype_multitimeframe_v42 import completed_bars, FREQS

CLASSES = ('NONE', 'FIRST', 'HH', 'LH', 'EH', 'HL', 'LL', 'EL')
FIELDS = ['swing_high_price', 'swing_low_price', 'swing_high_at', 'swing_low_at',
          'swing_high_confirmed_at', 'swing_low_confirmed_at', 'high_structure',
          'low_structure', 'pivot_high_confirmed', 'pivot_low_confirmed',
          'high_distance_pct', 'low_distance_pct']


def _segment(bars: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    """Calculate state using only already closed candles. Strict isolated extrema."""
    high = bars.high.to_numpy(dtype=float)
    low = bars.low.to_numpy(dtype=float)
    close = bars.close.to_numpy(dtype=float)
    last_h = last_l = np.nan
    prev_h = prev_l = np.nan
    high_class = low_class = 'NONE'
    high_at = low_at = pd.NaT
    high_confirmed_at = low_confirmed_at = pd.NaT
    records = []
    for j in range(len(bars)):
        # A pivot at (j-right) is evaluated *only* when candle j has closed.
        ph = pl = False
        origin = j - right
        if origin >= left:
            hwin = high[origin-left:j+1]
            lwin = low[origin-left:j+1]
            ph = high[origin] == hwin.max() and np.sum(hwin == high[origin]) == 1
            pl = low[origin] == lwin.min() and np.sum(lwin == low[origin]) == 1
            if ph:
                prev_h, last_h = last_h, high[origin]
                high_class = ('FIRST' if not np.isfinite(prev_h) else
                              'HH' if last_h > prev_h else 'LH' if last_h < prev_h else 'EH')
                high_at = bars.timestamp.iloc[origin]
                high_confirmed_at = bars.known_at.iloc[j]
            if pl:
                prev_l, last_l = last_l, low[origin]
                low_class = ('FIRST' if not np.isfinite(prev_l) else
                             'HL' if last_l > prev_l else 'LL' if last_l < prev_l else 'EL')
                low_at = bars.timestamp.iloc[origin]
                low_confirmed_at = bars.known_at.iloc[j]
        records.append(dict(
            known_at=bars.known_at.iloc[j],
            swing_high_price=last_h, swing_low_price=last_l,
            swing_high_at=high_at, swing_low_at=low_at,
            swing_high_confirmed_at=high_confirmed_at,
            swing_low_confirmed_at=low_confirmed_at,
            high_structure=high_class, low_structure=low_class,
            pivot_high_confirmed=bool(ph), pivot_low_confirmed=bool(pl),
            high_distance_pct=(last_h / close[j] - 1) if np.isfinite(last_h) else np.nan,
            low_distance_pct=(last_l / close[j] - 1) if np.isfinite(last_l) else np.nan,
        ))
    result = pd.DataFrame.from_records(records)
    for col in ('known_at','swing_high_at','swing_low_at','swing_high_confirmed_at','swing_low_confirmed_at'):
        result[col] = pd.to_datetime(result[col], utc=True).astype('datetime64[ns, UTC]')
    return result


def confirmed_swings(bars: pd.DataFrame, timeframe_minutes: int, left: int = 2, right: int = 2) -> pd.DataFrame:
    if left < 1 or right < 1 or timeframe_minutes < 1:
        raise ValueError('left, right et timeframe_minutes doivent être positifs')
    for c in ('timestamp', 'known_at', 'high', 'low', 'close'):
        if c not in bars.columns:
            raise ValueError(f'Colonne absente : {c}')
    b = bars.sort_values('timestamp').copy().reset_index(drop=True)
    if b.timestamp.duplicated().any():
        raise ValueError('Timestamps dupliqués')
    if b.empty:
        return pd.DataFrame(columns=['known_at', *FIELDS])
    if (b.known_at != b.timestamp + pd.Timedelta(minutes=timeframe_minutes)).any():
        raise ValueError('Horodatages known_at invalides')
    if not np.isfinite(b[['high','low','close']].to_numpy(dtype=float)).all() or (b[['high','low','close']] <= 0).any().any():
        raise ValueError('OHLC non finis')
    if (b.high < b[['close', 'low']].max(axis=1)).any() or (b.low > b[['close','high']].min(axis=1)).any():
        raise ValueError('OHLC incohérents')
    segments = b.timestamp.diff().ne(pd.Timedelta(minutes=timeframe_minutes)).cumsum()
    return pd.concat([_segment(part.reset_index(drop=True), left, right)
                      for _, part in b.groupby(segments, sort=False)], ignore_index=True)


def build_structure_snapshots(minutes_frame: pd.DataFrame, left: int = 2, right: int = 2):
    bars = {tf: completed_bars(minutes_frame, n) for tf, n in FREQS.items()}
    swings = {tf: confirmed_swings(b, FREQS[tf], left, right) for tf, b in bars.items()}
    out = bars['5m'][['timestamp', 'known_at', 'close']].rename(
        columns={'timestamp':'bar_open_5m', 'known_at':'decision_at', 'close':'close_5m'})
    # pandas may preserve microseconds in completed_bars, while swing events
    # use nanoseconds. merge_asof requires exactly matching datetime dtypes.
    out['decision_at'] = pd.to_datetime(out['decision_at'], utc=True).astype('datetime64[ns, UTC]')
    out = out.sort_values('decision_at')
    for tf in FREQS:
        src = swings[tf].rename(columns={'known_at':f'known_at_{tf}',
                                           **{c:f'{c}_{tf}' for c in FIELDS}})
        if src.empty:
            for col in src.columns:
                out[col] = pd.NaT if col.startswith(('known_at_', 'swing_high_at_', 'swing_low_at_', 'swing_high_confirmed_at_', 'swing_low_confirmed_at_')) else np.nan
            continue
        src[f'known_at_{tf}'] = pd.to_datetime(src[f'known_at_{tf}'], utc=True).astype('datetime64[ns, UTC]')
        out = pd.merge_asof(out, src.sort_values(f'known_at_{tf}'),
                            left_on='decision_at', right_on=f'known_at_{tf}',
                            direction='backward', allow_exact_matches=True,
                            tolerance=pd.Timedelta(minutes=FREQS[tf]))
        if tf != '5m':
            fresh = out['decision_at'].eq(out[f'known_at_{tf}'])
            for event in ('pivot_high_confirmed', 'pivot_low_confirmed'):
                col = f'{event}_{tf}'
                out[col] = (out[col].eq(True) & fresh).astype(bool)
    return out.reset_index(drop=True), swings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('data/evaluation/v421-binance-1m/BTCUSDT/2024-01.parquet'))
    p.add_argument('--output', type=Path, default=Path('data/evaluation/v431-swings/BTCUSDT-2024-01.parquet'))
    p.add_argument('--left', type=int, default=2)
    p.add_argument('--right', type=int, default=2)
    args = p.parse_args()
    if not args.input.is_file():
        p.error(f'Entrée absente : {args.input}')
    if args.output.exists():
        p.error(f'Refus écrasement : {args.output}')
    snapshots, swings = build_structure_snapshots(pd.read_parquet(args.input), args.left, args.right)
    print('Snapshots 5m :', len(snapshots))
    for tf, df in swings.items():
        print(f'{tf:>3}: bougies={len(df)} pivots_high={int(df.pivot_high_confirmed.sum())} pivots_low={int(df.pivot_low_confirmed.sum())}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(args.output, index=False)
    print('Parquet :', args.output.resolve())
    print('Aucune source modifiée. Aucun entraînement ni trading.')

if __name__ == '__main__':
    main()

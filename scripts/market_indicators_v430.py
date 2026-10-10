"""Qwen Trading V4.3.0 — Indicateurs causaux multi-timeframe (lecture seule).

Dépendance: prototype_multitimeframe_v42.py, numpy, pandas, pyarrow (pour CLI).
Les données 1m utilisent timestamp = ouverture UTC. `known_at` = clôture exclusive.
Chaque segment interrompu redémarre ses indicateurs; pas de backfill, pas de fuite.
VWAP = VWAP glissant en bougies (pas VWAP de session ni tick VWAP).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from prototype_multitimeframe_v42 import completed_bars, FREQS

PERIODS = {'ema_fast': 12, 'ema_slow': 26, 'sma': 20, 'rsi': 14,
           'atr': 14, 'bb': 20, 'vwap': 20, 'rel_volume': 20}
FEATURES = ['return_1', 'ema12_distance', 'ema26_distance', 'sma20_distance',
            'rsi14', 'macd_12_26', 'macd_signal9', 'macd_hist', 'atr14_pct',
            'bb20_z', 'bb20_width_pct', 'rolling_vwap20_distance', 'relative_volume20']


def _indicators_segment(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    close = s['close'].astype(float)
    high = s['high'].astype(float)
    low = s['low'].astype(float)
    vol = s['volume'].astype(float)
    s['return_1'] = close.pct_change(fill_method=None)
    e12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    e26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    s['ema12_distance'] = close / e12 - 1
    s['ema26_distance'] = close / e26 - 1
    sma = close.rolling(20, min_periods=20).mean()
    s['sma20_distance'] = close / sma - 1
    # Wilder's smoothing, seeded with the arithmetic mean of the first 14 deltas.
    delta = close.diff().to_numpy(dtype=float)
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    rsi = np.full(len(close), np.nan)
    if len(close) > 14:
        g = gain[1:15].mean()
        l = loss[1:15].mean()
        for i in range(14, len(close)):
            if i > 14:
                g = (13 * g + gain[i]) / 14
                l = (13 * l + loss[i]) / 14
            rsi[i] = 50.0 if g == 0 and l == 0 else (100.0 if l == 0 else 100 - 100 / (1 + g/l))
    s['rsi14'] = rsi
    macd = e12 - e26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    s['macd_12_26'] = macd / close
    s['macd_signal9'] = signal / close
    s['macd_hist'] = (macd - signal) / close
    prev_close = close.shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    # TR of first bar of a new segment has no preceding close: do not use it.
    tr.iloc[0] = np.nan
    s['atr14_pct'] = tr.rolling(14, min_periods=14).mean() / close
    std = close.rolling(20, min_periods=20).std(ddof=0)
    s['bb20_z'] = (close-sma) / std.replace(0, np.nan)
    s['bb20_width_pct'] = 4 * std / sma
    typical = (high+low+close)/3
    vtot = vol.rolling(20, min_periods=20).sum()
    vwap = (typical*vol).rolling(20, min_periods=20).sum() / vtot.replace(0, np.nan)
    s['rolling_vwap20_distance'] = close/vwap - 1
    # Previous 20 completed bars only; does not include current bar in reference.
    reference_vol = vol.shift(1).rolling(20, min_periods=20).mean()
    s['relative_volume20'] = vol / reference_vol.replace(0, np.nan)
    return s


def indicators_for_bars(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """No computed value may depend on later rows. Gaps reset all histories."""
    if bars.empty:
        return bars.assign(**{name: pd.Series(dtype=float) for name in FEATURES})
    b = bars.sort_values('timestamp').copy()
    if b['timestamp'].duplicated().any():
        raise ValueError('Bougies dupliquées')
    step = pd.Timedelta(minutes=minutes)
    segment = b['timestamp'].diff().ne(step).cumsum()
    chunks = [_indicators_segment(piece) for _, piece in b.groupby(segment, sort=False)]
    return pd.concat(chunks).sort_values('timestamp').reset_index(drop=True)


def build_feature_snapshots(minutes_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Output one row per complete 5m candle, joining only last completed HTF bars.

    The asof tolerance prevents stale context after a data gap. NaN = unavailable.
    """
    bars = {tf: indicators_for_bars(completed_bars(minutes_frame, n), n) for tf, n in FREQS.items()}
    out = bars['5m'][['timestamp', 'known_at', 'close'] + FEATURES].rename(
        columns={'timestamp':'bar_open_5m', 'known_at':'decision_at',
                 'close':'close_5m', **{f:f'{f}_5m' for f in FEATURES}})
    out = out.sort_values('decision_at')
    for tf in ('15m','1h','4h'):
        n = FREQS[tf]
        src = bars[tf][['known_at', 'close'] + FEATURES].rename(
            columns={'known_at':f'known_at_{tf}', 'close':f'close_{tf}',
                     **{f:f'{f}_{tf}' for f in FEATURES}})
        out = pd.merge_asof(out, src.sort_values(f'known_at_{tf}'),
                            left_on='decision_at', right_on=f'known_at_{tf}',
                            direction='backward', allow_exact_matches=True,
                            tolerance=pd.Timedelta(minutes=n))
    return out.reset_index(drop=True), bars


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path,
                    default=Path('data/evaluation/v421-binance-1m/BTCUSDT/2024-01.parquet'))
    ap.add_argument('--output', type=Path,
                    default=Path('data/evaluation/v430-features/BTCUSDT-2024-01.parquet'))
    args = ap.parse_args()
    if not args.input.is_file():
        ap.error(f'Fichier 1m introuvable : {args.input}')
    if args.output.exists():
        ap.error(f'Refus d’écrasement : {args.output}')
    frame = pd.read_parquet(args.input)
    snapshots, bars = build_feature_snapshots(frame)
    summary = {tf: len(b) for tf,b in bars.items()}
    print('Bougies complètes :', summary)
    print('Snapshots de décision :', len(snapshots))
    for tf in FREQS:
        col = f'rsi14_{tf}'
        print(f'{tf:>3}: RSI défini sur {snapshots[col].notna().sum()} snapshots / {len(snapshots)}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(args.output, index=False)
    print('Parquet :', args.output.resolve())
    print('Aucune source modifiée. Aucun entraînement, aucun trading.')

if __name__ == '__main__':
    main()

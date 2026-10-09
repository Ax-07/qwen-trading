from __future__ import annotations

"""Build causal V3 candidate features and separate realized event labels.

Research only. No V2 files are modified; historical TEST is not read.
Signal: market bar timestamp t, available_at t+1h.
Entry: next bar open t+1h, exit TP/SL or close of 12th bar.
The event engine is shared with backtest_event_driven_v1.py.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_event_driven_v1 import load_market, simulate_one

ROOT = Path('data')
SPLITS = ROOT / 'splits'
DEST = ROOT / 'evaluation' / 'v3-candidate-events'
EVENT_VERSION = 'v3-event-v1-sl-first-open-next'
CANDIDATE_VERSION = 'heuristic-v2-score-ge2-le-2'

# Explicit whitelist. All future-dependent fields are prohibited from X.
FEATURES = [
    'return_1h', 'return_3h', 'return_6h', 'return_24h',
    'distance_ema20_pct', 'distance_ema50_pct', 'distance_ema200_pct',
    'rsi14', 'atr14_pct', 'volume_ratio', 'volatility_24h', 'volatility_72h',
    'range_pct', 'body_pct', 'upper_wick_pct', 'lower_wick_pct',
    'distance_high_24h_pct', 'distance_low_24h_pct',
    'distance_high_72h_pct', 'distance_low_72h_pct',
    'breakout_24h', 'breakdown_24h', 'trend_1h', '4h_trend', '1d_trend',
    'trend_alignment', 'ema20_50_spread_pct', 'ema50_200_spread_pct',
    '4h_rsi14', '4h_atr14_pct', '1d_rsi14', '1d_atr14_pct',
    '4h_close', '1d_close',
]
CAUSAL_SCORE_FIELDS = ['close', 'ema20', 'ema50', 'rsi14', 'trend_1h', '4h_trend', '1d_trend']
BLOCKED_PATTERNS = ('future_', '_outcome_', '_best_', '_worst_', 'up_move_', 'down_move_')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fee-bps', type=float, default=5.0)
    p.add_argument('--slippage-bps', type=float, default=2.0)
    p.add_argument('--horizon', type=int, default=12)
    p.add_argument('--tp-atr', type=float, default=1.5)
    p.add_argument('--sl-atr', type=float, default=1.0)
    p.add_argument('--output', type=Path, default=DEST)
    args = p.parse_args()
    if args.horizon < 1 or args.tp_atr <= 0 or args.sl_atr <= 0 or args.fee_bps < 0 or args.slippage_bps < 0:
        p.error('Invalid parameters')
    return args


def score_candidates(df: pd.DataFrame) -> pd.Series:
    """Exact V2 heuristic, recomputed without any future data."""
    for c in CAUSAL_SCORE_FIELDS:
        if c not in df:
            raise ValueError(f'Missing causal score field: {c}')
    score = np.zeros(len(df), dtype=np.int64)
    for col in ('trend_1h', '4h_trend', '1d_trend'):
        score += df[col].fillna(0).astype(int).to_numpy()
    score += (df['close'] > df['ema20']).to_numpy(dtype=int)
    score -= (df['close'] < df['ema20']).to_numpy(dtype=int)
    score += (df['ema20'] > df['ema50']).to_numpy(dtype=int)
    score -= (df['ema20'] < df['ema50']).to_numpy(dtype=int)
    score += df['rsi14'].between(52, 70, inclusive='both').to_numpy(dtype=int)
    score -= df['rsi14'].between(30, 48, inclusive='both').to_numpy(dtype=int)
    return pd.Series(score, index=df.index, name='bias_score')


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def construct(split: str, split_df: pd.DataFrame, market: pd.DataFrame, args):
    if split_df['timestamp'].duplicated().any():
        raise ValueError(f'{split}: duplicate timestamps')
    if not split_df['sample_ready'].fillna(False).all():
        raise ValueError(f'{split}: sample_ready=False')
    if split_df[FEATURES + CAUSAL_SCORE_FIELDS + ['atr14']].isna().any().any():
        raise ValueError(f'{split}: null causal features; inspect before proceeding')
    if any(any(b in c for b in BLOCKED_PATTERNS) for c in FEATURES):
        raise ValueError('A future feature was whitelisted')
    if (split_df['available_at'] != split_df['timestamp'] + pd.Timedelta(hours=1)).any():
        raise ValueError(f'{split}: unavailable signal feature')

    split_df = split_df.sort_values('timestamp').copy()
    split_df['bias_score'] = score_candidates(split_df)
    candidate_rows = split_df.loc[split_df['bias_score'].abs() >= 2].copy()
    candidate_rows['side'] = np.where(candidate_rows['bias_score'] >= 2, 'LONG', 'SHORT')
    candidate_rows['candidate_version'] = CANDIDATE_VERSION
    candidate_rows['known_at'] = candidate_rows['available_at']
    lookup = pd.Series(np.arange(len(market)), index=market['timestamp'])
    mapped = candidate_rows['timestamp'].map(lookup)
    if mapped.isna().any():
        raise ValueError(f'{split}: unmatched market timestamps')
    candidate_rows['market_i'] = mapped.astype(int)
    lo = split_df['timestamp'].min()
    hi = split_df['timestamp'].max()

    # Predictors and labels are physically separate files. The linkage is event_id.
    predictor_rows, label_rows = [], []
    for row in candidate_rows.itertuples(index=False):
        timestamp = row.timestamp
        side = row.side
        i = int(row.market_i)
        event_id = f'{timestamp.isoformat()}_{side}_{CANDIDATE_VERSION}'
        source = candidate_rows.loc[candidate_rows['timestamp'] == timestamp].iloc[0]
        X = {
            'event_id': event_id, 'signal_timestamp': timestamp,
            'known_at': source['known_at'], 'side': side,
            'candidate_version': CANDIDATE_VERSION,
            'bias_score': int(source['bias_score']),
            'tp_atr': args.tp_atr, 'sl_atr': args.sl_atr,
            'horizon_bars': args.horizon,
            'fee_bps_per_side': args.fee_bps,
            'slippage_bps_per_side': args.slippage_bps,
        }
        for c in FEATURES:
            X[c] = source[c]
        predictor_rows.append(X)

        last_required_i = i + args.horizon
        # All future bars for a candidate must remain inside its own split;
        # do not use another split's path or cut windows to their realized exit.
        eligible = (last_required_i < len(market)
                    and market.at[last_required_i, 'timestamp'] <= hi
                    and market.at[i, 'timestamp'] >= lo)
        if not eligible:
            labels = {'event_outcome': 'DATA_INVALID', 'invalid_reason': 'SPLIT_BOUNDARY'}
        else:
            result = simulate_one(i, side, market, args.horizon, args.tp_atr,
                                  args.sl_atr, args.fee_bps, args.slippage_bps, 1.0, 1.0)
            if result is None:
                labels = {'event_outcome': 'DATA_INVALID', 'invalid_reason': 'INVALID_EVENT'}
            else:
                outcome = 'TP' if result['reason'].startswith('TP') else (
                    'SL' if result['reason'].startswith('SL') else 'TIME')
                labels = {
                    'event_outcome': outcome,
                    'invalid_reason': None,
                    'exit_reason': result['reason'],
                    'duration_bars': result['duration_bars'],
                    'entry_raw': result['entry_raw'], 'exit_raw': result['exit_raw'],
                    'entry_fill': result['entry_fill'], 'exit_fill': result['exit_fill'],
                    'gross_return': result['return_gross_pct'] / 100.0,
                    'net_return': result['return_net_pct'] / 100.0,
                    'net_r': result['r_net_proxy'],
                    'entry_at': result['entry_time'],
                    'label_available_at': result['exit_time'],
                    'tp_price': result['tp_price'], 'sl_price': result['sl_price'],
                }
        label_rows.append({'event_id': event_id, 'signal_timestamp': timestamp,
                           'side': side, 'event_version': EVENT_VERSION,
                           **labels})
    X = pd.DataFrame(predictor_rows)
    y = pd.DataFrame(label_rows)
    if len(X) != len(y) or not X['event_id'].is_unique or not y['event_id'].is_unique:
        raise ValueError(f'{split}: missing or duplicate event IDs')
    if not X.empty and (X['known_at'] < X['signal_timestamp']).any():
        raise ValueError(f'{split}: timestamp leak')
    return X, y


def main():
    args = arguments()
    market = load_market()
    print('=' * 72)
    print('V3 CANDIDATE EVENTS - TRAIN / VALIDATION ONLY')
    print('=' * 72)
    staging = {}
    for split in ('train', 'validation'):
        path = SPLITS / f'{split}.parquet'
        source = pd.read_parquet(path)
        for c in ('timestamp', 'available_at'):
            source[c] = pd.to_datetime(source[c], utc=True)
        if not source['timestamp'].isin(market['timestamp']).all():
            raise ValueError(f'{split}: missing market candles')
        if source['timestamp'].duplicated().any():
            raise ValueError(f'{split}: duplicate source timestamps')
        X, y = construct(split, source, market, args)
        staging[split] = (X, y, path)
        counts = y['event_outcome'].value_counts().to_dict() if len(y) else {}
        print(f'{split.upper():11s}: rows={len(source):5d}, candidates={len(X):5d}, outcomes={counts}')
    # Ensure no candidate labels from training reach into validation.
    train_X, train_y, _ = staging['train']
    validation_X, _, _ = staging['validation']
    if len(train_X) and len(validation_X):
        boundary = validation_X['signal_timestamp'].min()
        if (train_y.loc[train_y['event_outcome'] != 'DATA_INVALID', 'label_available_at'] >= boundary).any():
            raise ValueError('Train label crosses validation boundary')
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {'event_version': EVENT_VERSION, 'candidate_version': CANDIDATE_VERSION,
            'config': {'horizon': args.horizon, 'tp_atr': args.tp_atr,
                       'sl_atr': args.sl_atr, 'fee_bps': args.fee_bps,
                       'slippage_bps': args.slippage_bps},
            'python': platform.python_version(), 'pandas': pd.__version__,
            'inputs_sha256': {**{str(item[2]): file_sha256(item[2]) for item in staging.values()},
                              str(ROOT / 'processed' / 'btc_usdc_1h_labeled.parquet'): file_sha256(ROOT / 'processed' / 'btc_usdc_1h_labeled.parquet')}}
    for split, (X, y, _) in staging.items():
        X.to_parquet(args.output / f'{split}_candidates.parquet', index=False)
        y.to_parquet(args.output / f'{split}_labels.parquet', index=False)
    (args.output / 'manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('Output:', args.output.resolve())
    print('No Test loaded. Training not started. Candidate predictors and future labels stored separately.')


if __name__ == '__main__':
    main()

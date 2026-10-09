from __future__ import annotations

"""Tests synthétiques V3 : python scripts/test_candidate_events_v3.py"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from build_candidate_events_v3 import FEATURES, construct, score_candidates
from backtest_event_driven_v1 import simulate_one


def market_fixture(n=60):
    ts = pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC')
    market = pd.DataFrame({
        'timestamp': ts, 'available_at': ts + pd.Timedelta(hours=1),
        'open': np.full(n, 100.0), 'high': np.full(n, 100.2),
        'low': np.full(n, 99.8), 'close': np.full(n, 100.0),
        'atr14': np.full(n, 1.0),
    })
    return market


def split_fixture(market):
    df = market[['timestamp', 'available_at', 'close', 'atr14']].copy()
    for c in FEATURES:
        df[c] = 0.0
    df['ema20'] = 99.0
    df['ema50'] = 98.0
    df['rsi14'] = 60.0
    df['trend_1h'] = 1
    df['4h_trend'] = 1
    df['1d_trend'] = 1
    df['sample_ready'] = True
    return df


def run_tests():
    a = SimpleNamespace(horizon=12, tp_atr=1.5, sl_atr=1.0,
                        fee_bps=5.0, slippage_bps=2.0)
    m = market_fixture()
    base = split_fixture(m)
    assert (score_candidates(base) == 6).all()

    X, y = construct('train', base.iloc[0:35].copy(), m, a)
    assert len(X) == len(y) == 35
    assert X['event_id'].is_unique
    assert (X['side'] == 'LONG').all()
    assert (y['event_outcome'].iloc[-12:] == 'DATA_INVALID').all()
    assert (y['event_outcome'].iloc[:-12] == 'TIME').all()
    assert all('future' not in c and 'outcome' not in c for c in X.columns)
    assert X['known_at'].equals(X['signal_timestamp'] + pd.Timedelta(hours=1))
    valid = y.loc[y['event_outcome'] == 'TIME']
    assert (valid['duration_bars'] == 12).all()
    assert (valid['label_available_at'] <= m.at[34, 'available_at']).all()
    assert (valid['net_return'] < 0).all()  # TIME incurs costs, not 0R.

    # Changing a FUTURE candle must modify the target, not the predictors.
    altered = m.copy()
    altered.loc[1, ['high', 'low']] = [102.0, 98.0]
    X2, y2 = construct('train', base.iloc[:35].copy(), altered, a)
    assert X.equals(X2)
    assert y2.iloc[0]['event_outcome'] == 'SL'  # ambiguous -> pessimistic SL

    # SL/TP/timeout gap handling on the reused engine.
    tp_market = m.copy()
    tp_market.loc[1, 'high'] = 102.0
    tp = simulate_one(0, 'LONG', tp_market, 12, 1.5, 1., 5., 2., 1., 1.)
    assert tp['reason'] == 'TP' and tp['entry_time'] == m.at[1, 'timestamp']

    gap_market = m.copy()
    gap_market.loc[2, ['open', 'high', 'low', 'close']] = [97.0, 97.2, 96.8, 97.0]
    gap = simulate_one(0, 'LONG', gap_market, 12, 1.5, 1., 5., 2., 1., 1.)
    assert gap['reason'] == 'SL_GAP' and gap['exit_raw'] == 97.

    # LONG/SHORT score symmetry.
    bearish = base.copy()
    bearish['trend_1h'] = -1
    bearish['4h_trend'] = -1
    bearish['1d_trend'] = -1
    bearish['ema20'] = 101.
    bearish['ema50'] = 102.
    bearish['rsi14'] = 35.
    assert (score_candidates(bearish) == -6).all()
    short_X, _ = construct('train', bearish.iloc[:35].copy(), m, a)
    assert (short_X['side'] == 'SHORT').all()

    print('OK: 8 groups of checks: causal scoring, IDs, split boundaries,')
    print('    TIME exits+costs, future isolation, SL-first, TP/gaps, SHORT.')


if __name__ == '__main__':
    run_tests()

"""Tests reproductibles V4.3.0, sans réseau, sans lecture du Test."""
from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
from prototype_multitimeframe_v42 import synthetic_minutes, completed_bars
from market_indicators_v430 import indicators_for_bars, build_feature_snapshots, FEATURES

class IndicatorTests(unittest.TestCase):
    def setUp(self):
        self.minutes = synthetic_minutes(3000)

    def test_shapes(self):
        snapshots, bars = build_feature_snapshots(self.minutes)
        self.assertEqual(len(snapshots), 600)
        self.assertEqual(len(bars['4h']), 12)
        self.assertTrue(snapshots.decision_at.is_monotonic_increasing)

    def test_no_future_dependency(self):
        before,_ = build_feature_snapshots(self.minutes)
        later = synthetic_minutes(600, '2024-01-03T02:00:00Z')
        later['close'] *= 10
        later['high'] = np.maximum(later.high, later.close)
        later['low'] = np.minimum(later.low, later.close)
        after,_ = build_feature_snapshots(pd.concat([self.minutes,later],ignore_index=True))
        pd.testing.assert_frame_equal(before, after.iloc[:len(before)].reset_index(drop=True))

    def test_gap_resets(self):
        f = self.minutes.drop(index=list(range(1200, 1210))).reset_index(drop=True)
        _, bars = build_feature_snapshots(f)
        b = bars['5m']
        gap_right = b.loc[b.timestamp == pd.Timestamp('2024-01-01T20:10:00Z')].iloc[0]
        self.assertTrue(np.isnan(gap_right.rsi14))
        self.assertTrue(np.isnan(gap_right.ema26_distance))

    def test_no_stale_context(self):
        f = self.minutes.drop(index=list(range(960, 1320))).reset_index(drop=True)
        snapshots,_ = build_feature_snapshots(f)
        first_after = snapshots[snapshots.decision_at == pd.Timestamp('2024-01-01T22:05:00Z')].iloc[0]
        self.assertTrue(pd.isna(first_after.known_at_1h))
        self.assertTrue(pd.isna(first_after.known_at_4h))

    def test_rsi_bounds_and_warmup(self):
        bars = indicators_for_bars(completed_bars(self.minutes, 5),5)
        self.assertTrue(bars.rsi14.iloc[:14].isna().all())
        self.assertTrue(bars.rsi14.dropna().between(0,100).all())
        self.assertTrue(bars.ema26_distance.iloc[:25].isna().all())

    def test_constant_prices(self):
        f = synthetic_minutes(300)
        for c in ('open','high','low','close'): f[c] = 100.0
        bars = indicators_for_bars(completed_bars(f,5),5)
        self.assertAlmostEqual(float(bars.rsi14.iloc[-1]),50.0)
        self.assertTrue(np.isnan(bars.bb20_z.iloc[-1]))
        self.assertAlmostEqual(float(bars.atr14_pct.iloc[-1]),0.)

    def test_zero_volume_vwap_nan(self):
        f = synthetic_minutes(300)
        f['volume'] = 0.0
        bars = indicators_for_bars(completed_bars(f,5),5)
        self.assertTrue(bars.rolling_vwap20_distance.isna().all())
        self.assertTrue(bars.relative_volume20.isna().all())

    def test_feature_numerics(self):
        b = indicators_for_bars(completed_bars(self.minutes,5),5)
        self.assertTrue(np.isfinite(b[FEATURES].to_numpy()[~np.isnan(b[FEATURES].to_numpy())]).all())
        self.assertTrue((b.atr14_pct.dropna() >= 0).all())

if __name__ == '__main__':
    unittest.main(verbosity=2)

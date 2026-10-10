"""Tests sans accès réseau ni données Test."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from market_zigzag_v433 import zigzag_for_bars, build_zigzag_snapshots
from prototype_multitimeframe_v42 import synthetic_minutes


def bars(close, gap_at=None):
    times = pd.date_range('2024-01-01', periods=len(close), freq='5min', tz='UTC')
    if gap_at is not None:
        times = times.where(np.arange(len(close)) < gap_at, times + pd.Timedelta(minutes=5))
    close = np.array(close, dtype=float)
    return pd.DataFrame({'timestamp':times,'known_at':times+pd.Timedelta(minutes=5),
                         'high':close+0.25,'low':close-0.25,'close':close})


class Tests(unittest.TestCase):
    def test_invalid_params(self):
        with self.assertRaises(ValueError): zigzag_for_bars(bars([100]), 5, 0)
        with self.assertRaises(ValueError): zigzag_for_bars(bars([100]), 5, 2, 0)

    def test_zero_data(self):
        self.assertEqual(len(zigzag_for_bars(bars([]), 5)), 0)

    def test_no_early_confirmation(self):
        prices = [100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,118,116,114,112,110,108]
        result = zigzag_for_bars(bars(prices), 5, 3, 2)
        peaks = result.loc[result.zz_event.eq('HIGH')]
        self.assertGreater(len(peaks), 0)
        self.assertTrue((peaks.zz_confirmed_at > peaks.zz_pivot_at).all())
        self.assertTrue((result.loc[~result.zz_confirmed, 'zz_pivot_at'].isna()).all())

    def test_event_alternation(self):
        prices = [100+i for i in range(21)] + [121-i*2 for i in range(20)] + [81+i*2 for i in range(25)]
        events = zigzag_for_bars(bars(prices), 5, 3, 2)
        emitted = events.loc[events.zz_confirmed,'zz_event'].tolist()
        self.assertGreaterEqual(len(emitted), 2)
        self.assertTrue(all(a != b for a,b in zip(emitted, emitted[1:])))

    def test_prefix_stability(self):
        prices = [100+0.2*i + 4*np.sin(i/4) for i in range(110)]
        a = zigzag_for_bars(bars(prices[:70]), 5, 5, 1.5)
        b = zigzag_for_bars(bars(prices), 5, 5, 1.5)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.iloc[:70].reset_index(drop=True))

    def test_gap_resets(self):
        prices = [100+i*0.2 for i in range(100)]
        r = zigzag_for_bars(bars(prices, gap_at=50), 5, 5, 2)
        self.assertTrue(r.iloc[50:55].zz_confirmed.eq(False).all())
        self.assertTrue(r.iloc[50:55].zz_last_high.isna().all())

    def test_no_early_htf(self):
        snap, _ = build_zigzag_snapshots(synthetic_minutes(520))
        self.assertEqual(len(snap), 104)
        known = snap.known_at_4h.notna()
        self.assertTrue((snap.loc[known,'known_at_4h'] <= snap.loc[known,'decision_at']).all())

    def test_htf_event_once(self):
        snap, _ = build_zigzag_snapshots(synthetic_minutes(1200), atr_period=2, reversal_atr=0.5)
        for tf in ('15m','1h','4h'):
            self.assertTrue((~snap[f'zz_confirmed_{tf}'] | snap.decision_at.eq(snap[f'known_at_{tf}'])).all())

    def test_gap_incomplete_aggregation(self):
        frame = synthetic_minutes(520).drop(index=[50])
        snap, events = build_zigzag_snapshots(frame)
        self.assertEqual(len(snap), 103)
        self.assertEqual(len(events['5m']), 103)

if __name__ == '__main__':
    unittest.main(verbosity=2)

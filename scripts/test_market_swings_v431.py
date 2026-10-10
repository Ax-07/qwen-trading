"""Tests V4.3.1 : python scripts/test_market_swings_v431.py"""
import unittest
import numpy as np
import pandas as pd
from market_swings_v431 import confirmed_swings, build_structure_snapshots
from prototype_multitimeframe_v42 import synthetic_minutes


def make_bars(highs, lows=None, step=5, start='2024-01-01T00:00:00Z'):
    highs = np.array(highs, dtype=float) + 100
    lows = np.array(lows, dtype=float) + 100 if lows is not None else highs-2
    times = pd.date_range(start, periods=len(highs), freq=f'{step}min', tz='UTC')
    return pd.DataFrame(dict(timestamp=times, known_at=times+pd.Timedelta(minutes=step),
                             high=highs, low=lows, close=(highs+lows)/2))


class TestSwings(unittest.TestCase):
    def test_confirmation_delay(self):
        b = make_bars([2, 3, 8, 5, 4, 3, 2])
        result = confirmed_swings(b, 5)
        self.assertFalse(result.pivot_high_confirmed.iloc[:4].any())
        self.assertTrue(result.pivot_high_confirmed.iloc[4])
        self.assertEqual(result.swing_high_at.iloc[4], b.timestamp.iloc[2])
        self.assertEqual(result.swing_high_confirmed_at.iloc[4], b.known_at.iloc[4])

    def test_classification(self):
        b = make_bars([2, 3, 8, 4, 3, 4, 9, 5, 3, 4, 7, 5, 4])
        r = confirmed_swings(b, 5)
        self.assertEqual(r.loc[r.pivot_high_confirmed, 'high_structure'].tolist(), ['FIRST', 'HH', 'LH'])

    def test_ties_disallowed(self):
        b = make_bars([2, 3, 8, 8, 3, 2])
        r = confirmed_swings(b, 5)
        self.assertFalse(r.pivot_high_confirmed.any())

    def test_future_prefix_stability(self):
        a = make_bars([2, 3, 8, 5, 4, 3, 7, 4, 3])
        first = confirmed_swings(a.iloc[:7], 5)
        more = pd.concat([a, make_bars([50, 11, 90], start='2024-01-01T00:45:00Z')])
        later = confirmed_swings(more, 5)
        pd.testing.assert_frame_equal(first.reset_index(drop=True), later.iloc[:7].reset_index(drop=True))

    def test_gap_resets(self):
        a = make_bars([2, 3, 8, 4, 3, 2])
        b = make_bars([2, 3, 9, 4, 3, 2], start='2024-01-01T01:00:00Z')
        r = confirmed_swings(pd.concat([a, b]), 5)
        self.assertTrue(np.isnan(r.swing_high_price.iloc[6]))
        self.assertEqual(r.high_structure.iloc[10], 'FIRST')

    def test_no_early_htf_context(self):
        snapshots, swings = build_structure_snapshots(synthetic_minutes(520))
        self.assertEqual(len(snapshots), 104)
        row = snapshots.loc[snapshots.decision_at == pd.Timestamp('2024-01-01T03:55:00Z')].iloc[0]
        self.assertTrue(pd.isna(row['known_at_4h']))
        row2 = snapshots.loc[snapshots.decision_at == pd.Timestamp('2024-01-01T04:00:00Z')].iloc[0]
        self.assertEqual(row2['known_at_4h'], pd.Timestamp('2024-01-01T04:00:00Z'))

    def test_gap_rejects_incomplete_bar(self):
        f = synthetic_minutes(500).drop(index=[13]).reset_index(drop=True)
        out, swings = build_structure_snapshots(f)
        self.assertEqual(len(out), 99)
        self.assertFalse((out.decision_at == pd.Timestamp('2024-01-01T00:15:00Z')).any())

    def test_htf_events_not_repeated(self):
        f = synthetic_minutes(600)
        snap, _ = build_structure_snapshots(f)
        # Any 15m confirmation event is visible only at its exact close.
        mask = snap['decision_at'] != snap['known_at_15m']
        self.assertFalse(snap.loc[mask, 'pivot_high_confirmed_15m'].any())
        self.assertFalse(snap.loc[mask, 'pivot_low_confirmed_15m'].any())

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            confirmed_swings(make_bars([1, 2, 3]), 5, right=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

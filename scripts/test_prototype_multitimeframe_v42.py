"""Tests causaux V4.2, uniquement donnees synthetiques."""
import unittest
import pandas as pd
from prototype_multitimeframe_v42 import synthetic_minutes, completed_bars, build_snapshots

class TestV42(unittest.TestCase):
    def test_counts(self):
        x = synthetic_minutes(480)
        self.assertEqual(len(completed_bars(x, 5)), 96)
        self.assertEqual(len(completed_bars(x, 15)), 32)
        self.assertEqual(len(completed_bars(x, 60)), 8)
        self.assertEqual(len(completed_bars(x, 240)), 2)

    def test_missing_minute_rejects_entire_bar(self):
        x = synthetic_minutes(15).drop(index=2)
        b = completed_bars(x, 5)
        self.assertEqual(len(b), 2)
        self.assertEqual(b.iloc[0].timestamp, pd.Timestamp('2024-01-01T00:05:00Z'))

    def test_partial_bar_not_available(self):
        self.assertEqual(len(completed_bars(synthetic_minutes(14), 5)), 2)

    def test_future_not_visible(self):
        x = synthetic_minutes(500)
        a = build_snapshots(x.iloc[:480])
        b = build_snapshots(x)
        pd.testing.assert_frame_equal(a, b.iloc[:len(a)].reset_index(drop=True))

    def test_four_hour_context_not_early(self):
        s = build_snapshots(synthetic_minutes(480))
        self.assertTrue(s.loc[s.decision_at < pd.Timestamp('2024-01-01T04:00:00Z'), 'close_4h'].isna().all())
        self.assertTrue(s.loc[s.decision_at >= pd.Timestamp('2024-01-01T04:00:00Z'), 'close_4h'].notna().all())
        self.assertTrue((s.loc[s.known_at_4h.notna(), 'known_at_4h'] <= s.loc[s.known_at_4h.notna(), 'decision_at']).all())

    def test_invalid_ohlc_rejected(self):
        x = synthetic_minutes(10)
        x.loc[0, 'high'] = 1
        with self.assertRaises(ValueError):
            completed_bars(x, 5)

    def test_no_future_fill_of_incomplete_context(self):
        x = synthetic_minutes(480).drop(index=3)
        s = build_snapshots(x)
        self.assertTrue(s.loc[s.decision_at < pd.Timestamp("2024-01-01T08:00:00Z"), "close_4h"].isna().all())

if __name__ == '__main__':
    unittest.main(verbosity=2)

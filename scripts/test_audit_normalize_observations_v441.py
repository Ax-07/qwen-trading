"""Tests synthétiques V4.4.1 : python scripts/test_audit_normalize_observations_v441.py"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_normalize_observations_v441 import normalize


def example(n=4):
    ts = pd.date_range('2024-01-01 00:05', periods=n, freq='5min', tz='UTC')
    d = {'decision_at': ts, 'close_5m': [100.0] * n,
         'bar_open_5m': ts - pd.Timedelta(minutes=5)}
    for tf in ('5m', '15m', '1h', '4h'):
        d[f'known_at_{tf}'] = ts if tf == '5m' else pd.Series(pd.NaT, index=range(n), dtype='datetime64[ns, UTC]')
        d[f'close_{tf}'] = [100.0] * n if tf == '5m' else [np.nan] * n
    d.update(swing_high_price_5m=[110., np.nan, 120., 125.],
             high_structure_5m=['FIRST', 'HH', 'LH', 'EH'],
             event_type_5m=['NONE', 'BULLISH_BOS', 'BEARISH_CHOCH', 'NONE'],
             rsi14_5m=[np.nan, 50., 60., 70.],
             swing_high_confirmed_at_5m=[ts[0], pd.NaT, ts[2], ts[3]])
    return pd.DataFrame(d)


class V441Tests(unittest.TestCase):
    def test_shape_and_price_normalization(self):
        f, a, r = normalize(example())
        self.assertEqual(len(f), 4)
        self.assertAlmostEqual(f.swing_high_price_distance_to_close_5m.iloc[0], .1, places=5)
        self.assertNotIn('swing_high_price_5m', f)
        self.assertNotIn('close_5m', f)
    def test_unknown_excluded(self):
        d = example(); d['future_label'] = [3] * len(d)
        f, a, r = normalize(d)
        self.assertNotIn('future_label', f)
        self.assertIn('future_label', r['excluded_columns'])
    def test_future_confirmation_rejected(self):
        d = example(); d.loc[0, 'swing_high_confirmed_at_5m'] = d.decision_at.iloc[2]
        with self.assertRaises(ValueError): normalize(d)
    def test_future_context_rejected(self):
        d = example(); d.loc[0, 'known_at_15m'] = d.decision_at.iloc[1]
        with self.assertRaises(ValueError): normalize(d)
    def test_stale_context_rejected(self):
        d = example(); d.loc[3, 'known_at_15m'] = d.decision_at.iloc[3] - pd.Timedelta(minutes=20)
        with self.assertRaises(ValueError): normalize(d)
    def test_missingness(self):
        f, _, _ = normalize(example())
        self.assertEqual(f.rsi14_5m_available.tolist(), [0, 1, 1, 1])
        self.assertEqual(f.context_available_15m.tolist(), [0, 0, 0, 0])
    def test_unknown_category_rejected(self):
        d = example(); d.loc[1, 'event_type_5m'] = 'BUY_NOW'
        with self.assertRaises(ValueError): normalize(d)
    def test_no_future_dep(self):
        d = example(4)
        one, _, _ = normalize(d.iloc[:3].copy())
        two, _, _ = normalize(d)
        pd.testing.assert_frame_equal(one.reset_index(drop=True), two.iloc[:3].reset_index(drop=True))

if __name__ == '__main__':
    unittest.main(verbosity=2)

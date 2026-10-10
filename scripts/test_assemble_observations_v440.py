"""Tests autonomes de la fusion stricte V4.4.0."""
import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
from assemble_observations_v440 import assemble, SOURCES


def sample(n=12):
    t = pd.date_range('2024-01-01 00:05', periods=n, freq='5min', tz='UTC')
    base = pd.DataFrame({'bar_open_5m': t-pd.Timedelta(minutes=5),
                         'decision_at':t, 'close_5m': np.arange(n)+100.0})
    frames = {name:base.copy() for name in SOURCES}
    frames['indicators']['known_at_5m'] = t
    for tf, k in [('15m',15),('1h',60),('4h',240)]:
        frames['indicators'][f'known_at_{tf}'] = pd.Series(
            [v.floor(f'{k}min') if v.floor(f'{k}min') >= t[0] else pd.NaT for v in t])
    frames['indicators']['rsi14_5m'] = [np.nan]*2+[50.0]*(n-2)
    frames['swings']['pivot_high_confirmed_5m'] = [False]*n
    frames['breaks']['event_type_5m'] = ['NONE']*n
    frames['zigzag']['zz_event_5m'] = ['NONE']*n
    return frames


class MergeTests(unittest.TestCase):
    def test_complete(self):
        out, report = assemble(sample())
        self.assertEqual(len(out), 12)
        self.assertIn('rsi14_5m', out)
        self.assertIn('zz_event_5m', out)
        self.assertEqual(report['rows'], 12)

    def test_prefix_stability(self):
        frames = sample()
        prefix = {k:v.iloc[:7].copy() for k,v in frames.items()}
        short, _ = assemble(prefix)
        long, _ = assemble(frames)
        pd.testing.assert_frame_equal(short, long.iloc[:7].reset_index(drop=True))

    def test_missing_source(self):
        f = sample(); del f['zigzag']
        with self.assertRaises(ValueError): assemble(f)

    def test_row_absent(self):
        f = sample(); f['swings'] = f['swings'].iloc[:-1]
        with self.assertRaisesRegex(ValueError, 'grille'): assemble(f)

    def test_duplicate_decisions(self):
        f = sample(); f['zigzag'].loc[1,'decision_at'] = f['zigzag'].loc[0,'decision_at']
        with self.assertRaisesRegex(ValueError, 'dupliqués'): assemble(f)

    def test_wrong_price(self):
        f = sample(); f['breaks'].loc[3,'close_5m'] = 999
        with self.assertRaisesRegex(ValueError, 'prix'): assemble(f)

    def test_future_context(self):
        f = sample(); f['indicators'].loc[3,'known_at_15m'] = f['indicators'].loc[3,'decision_at'] + pd.Timedelta(minutes=15)
        with self.assertRaisesRegex(ValueError, 'futur'): assemble(f)

    def test_stale_context(self):
        f = sample(); f['indicators'].loc[9,'known_at_15m'] = f['indicators'].loc[9,'decision_at'] - pd.Timedelta(minutes=20)
        with self.assertRaisesRegex(ValueError, 'périmé'): assemble(f)

    def test_future_event(self):
        f = sample(); f['zigzag']['zz_confirmed_at_5m'] = pd.Series(pd.NaT, index=f['zigzag'].index, dtype='datetime64[ns, UTC]')
        f['zigzag'].loc[4,'zz_confirmed_at_5m'] = f['zigzag'].loc[4,'decision_at'] + pd.Timedelta(minutes=5)
        with self.assertRaisesRegex(ValueError, 'futur'): assemble(f)

    def test_dtype_normalization(self):
        f = sample()
        f['swings']['decision_at'] = f['swings']['decision_at'].astype('datetime64[us, UTC]')
        out, _ = assemble(f)
        self.assertEqual(str(out.decision_at.dtype), 'datetime64[ns, UTC]')


if __name__ == '__main__': unittest.main(verbosity=2)

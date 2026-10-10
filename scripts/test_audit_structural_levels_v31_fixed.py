from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_structural_levels_v31_fixed as a

class TestAudit(unittest.TestCase):
    def fixture(self):
        ts = pd.to_datetime(['2026-01-01T03:00Z','2026-01-01T04:00Z'])
        confirmed = pd.to_datetime(['2026-01-01T03:00Z']*2)
        origin = pd.to_datetime(['2026-01-01T01:00Z']*2)
        df = pd.DataFrame({'known_at':ts,'side':['LONG','SHORT'],
              'entry_open':[100.,100.], 'high_price':[103.,103.], 'low_price':[97.,97.],
              'high_confirmed_at':confirmed,'low_confirmed_at':confirmed,
              'high_pivot_at':origin,'low_pivot_at':origin})
        use = df.side.eq('SHORT')
        df['stop_pivot_price'] = np.where(use,df.high_price,df.low_price)
        df['stop_confirmed_at'] = pd.to_datetime(np.where(use,df.high_confirmed_at,df.low_confirmed_at),utc=True)
        df['stop_pivot_at'] = pd.to_datetime(np.where(use,df.high_pivot_at,df.low_pivot_at),utc=True)
        return df

    def test_valid_long_short(self):
        out = a.evaluate(self.fixture(),5,.1,5)
        self.assertEqual(out.status.tolist(),['VALID_REFERENCE']*2)
        self.assertTrue((out.stop_confirmed_at<=out.known_at).all())
        self.assertAlmostEqual(out.distance_pct.iloc[0],3.0485,4)

    def test_no_level(self):
        frame=self.fixture()
        frame.loc[0,'stop_pivot_price']=np.nan
        self.assertEqual(a.evaluate(frame,5,.1,5).status.iloc[0],'NO_CONFIRMED_PIVOT')

    def test_invalid_stop(self):
        frame=self.fixture()
        frame.loc[0,'stop_pivot_price']=101.
        self.assertEqual(a.evaluate(frame,0,.1,5).status.iloc[0],'STOP_ALREADY_INVALID')

    def test_boundary(self):
        frame=self.fixture()
        frame.loc[0,'entry_open']=np.nan
        self.assertEqual(a.evaluate(frame,5,.1,5).status.iloc[0],'ENTRY_OUTSIDE_SPLIT')

    def test_fractal_boolean_flags_not_timestamps(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'v31-market-structure'
            root.mkdir()
            times = pd.date_range('2026-01-01T01:00Z', periods=4, freq='h')
            frame = pd.DataFrame({
                'known_at': times,
                '4h_available_at': times,
                '4h_last_swing_high': [float('nan'), 105., 105., 105.],
                '4h_last_swing_low': [float('nan'), float('nan'), 95., 95.],
                '4h_swing_high_at': [pd.NaT, times[0], times[0], times[0]],
                '4h_swing_low_at': [pd.NaT, pd.NaT, times[1], times[1]],
                '4h_pivot_high_confirmed': [False, True, False, False],
                '4h_pivot_low_confirmed': [False, False, True, False],
            })
            from unittest.mock import patch
            with patch.object(a.pd, 'read_parquet', return_value=frame):
                out = a.fractal_levels('validation', '4h', Path(temp))
            self.assertTrue(pd.isna(out.loc[0, 'high_confirmed_at']))
            self.assertEqual(out.loc[3, 'high_confirmed_at'], times[1])
            self.assertEqual(out.loc[3, 'low_confirmed_at'], times[2])

    def test_zigzag_no_future(self):
        frame = pd.DataFrame({'known_at':pd.to_datetime(['2026-01-01T03:00Z','2026-01-01T04:00Z'])})
        events = pd.DataFrame({'confirmation_at':pd.to_datetime(['2026-01-01T04:00Z']),
             'pivot_at':pd.to_datetime(['2026-01-01T01:00Z']),
             'kind':['LOW'],'price':[97.]})
        out=a.attach_zigzag_side(frame,events,'LOW','low')
        self.assertTrue(np.isnan(out.low_price.iloc[0]))
        self.assertEqual(out.low_price.iloc[1],97.)

if __name__ == '__main__':
    unittest.main(verbosity=2)

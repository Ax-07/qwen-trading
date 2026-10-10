import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_structural_excursions_v31 import excursion, select_bars, report_excursions


def bars():
    return pd.DataFrame({
        'timestamp':pd.date_range('2026-01-01',periods=3,freq='h',tz='UTC'),
        'open':[100.,103.,101.], 'high':[104.,105.,102.],
        'low':[99.,98.,97.], 'close':[103.,101.,100.]})

class TestExcursions(unittest.TestCase):
    def test_long_mae_mfe(self):
        x=excursion(SimpleNamespace(entry_raw=100.,risk_price=5.,side='LONG',reason='TIME'),bars())
        self.assertAlmostEqual(x['mfe_r'],1.0)
        self.assertAlmostEqual(x['mae_r'],.6)
        self.assertFalse(x['intrabar_exit_uncertain'])

    def test_short_mae_mfe(self):
        x=excursion(SimpleNamespace(entry_raw=100.,risk_price=5.,side='SHORT',reason='TIME'),bars())
        self.assertAlmostEqual(x['mfe_r'],.6)
        self.assertAlmostEqual(x['mae_r'],1.)

    def test_exit_candle_uncertainty(self):
        x=excursion(SimpleNamespace(entry_raw=100.,risk_price=5.,side='LONG',reason='SL'),bars())
        self.assertTrue(x['intrabar_exit_uncertain'])

    def test_future_bars_excluded(self):
        x=select_bars(bars(),pd.Timestamp('2026-01-01T00:00Z'),2)
        self.assertEqual(len(x),2)
        self.assertEqual(float(x.low.min()),98)

    def test_missing_entry_rejected(self):
        with self.assertRaises(ValueError):
            select_bars(bars(),pd.Timestamp('2026-01-04T00:00Z'),2)

    def test_gap_rejected(self):
        b=bars();b.loc[1,'timestamp']=pd.Timestamp('2026-01-01T02:00Z')
        with self.assertRaises(ValueError):
            select_bars(b,pd.Timestamp('2026-01-01T00:00Z'),3)

    def test_wrong_entry_open_rejected(self):
        with self.assertRaises(ValueError):
            excursion(SimpleNamespace(entry_raw=99.,risk_price=5.,side='LONG',reason='TIME'),bars())

    def test_summary_separate_timeout(self):
        f=pd.DataFrame([dict(split='train',method='zigzag',timeframe='4h',multiple_r=1.5,side='SHORT',reason='TIME',net_return_pct=.1,duration_bars=12,mfe_r=.3,mae_r=.1,mfe_over_half_r=False,mfe_over_one_r=False,mae_over_half_r=False,intrabar_exit_uncertain=False)])
        result=report_excursions(f)
        self.assertEqual(result.iloc[0]['exit_type'],'TIME')
        self.assertEqual(int(result.iloc[0]['count']),1)

if __name__ == '__main__':
    unittest.main(verbosity=2)

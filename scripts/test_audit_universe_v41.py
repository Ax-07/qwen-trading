import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_universe_v41 import validate_month, hourly_returns, gap_summary, yearly_correlations, rolling_pairs

class AuditTests(unittest.TestCase):
    def test_validate_month(self):
        idx=pd.date_range('2024-01-01',periods=3,freq='h',tz='UTC')
        d=pd.DataFrame(dict(timestamp=idx,open=[10]*3,high=[11]*3,low=[9]*3,close=[10]*3,volume=[1]*3,quote_volume=[10]*3))
        self.assertEqual(len(validate_month(d,'2024-01','BTCUSDT')),3)
        with self.assertRaises(ValueError):validate_month(d,'2024-02','BTCUSDT')
    def test_no_gap_return(self):
        idx=pd.date_range('2024-01-01',periods=4,freq='h',tz='UTC')
        d=pd.DataFrame({'close':[10,np.nan,12,13]},index=idx)
        r=hourly_returns(d)
        self.assertTrue(r.iloc[:3].isna().all())
        self.assertTrue(np.isfinite(r.iloc[3]))
    def test_common_gaps(self):
        a=pd.Timestamp('2021-01-01',tz='UTC');b=pd.Timestamp('2021-01-02',tz='UTC')
        common,union=gap_summary({'A':[a,b],'B':[a]})
        self.assertEqual(common,[a]);self.assertEqual(union,[a,b])
    def test_yearly_corr(self):
        idx=pd.date_range('2024-01-01',periods=100,freq='h',tz='UTC')
        x=np.sin(np.arange(100)*.1)
        d=pd.DataFrame({'A':x,'B':x*2},index=idx)
        result=yearly_correlations(d,30)
        self.assertAlmostEqual(result.iloc[0].correlation,1,places=9)
        self.assertEqual(result.iloc[0].common_hours,100)
    def test_rolling_missing_coverage(self):
        idx=pd.date_range('2024-01-01',periods=24*40,freq='h',tz='UTC')
        x=np.sin(np.arange(len(idx))*.13)
        d=pd.DataFrame({'A':x,'B':x*2},index=idx)
        d.loc[idx[100],'A']=np.nan
        result=rolling_pairs(d,window_days=(30,),min_coverage=.98)
        self.assertGreater(result.iloc[0].valid_windows,0)
        self.assertAlmostEqual(result.iloc[0].median_correlation,1,places=8)

if __name__=='__main__':unittest.main(verbosity=2)

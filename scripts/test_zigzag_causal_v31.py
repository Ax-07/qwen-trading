from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
from build_zigzag_causal_v31 import zigzag_events, events_from_split


def bars(values):
    idx=pd.date_range('2026-01-01T01:00Z',periods=len(values),freq='h')
    return pd.DataFrame({'open':values,'high':values,'low':values,'close':values},index=idx,dtype=float)

class TestCausalZigZag(unittest.TestCase):
    def test_high_confirmed_only_after_reversal(self):
        b=bars([100,102,105,104,103,102,99])
        out=zigzag_events(b,2)
        high=out[out.kind=='HIGH'].iloc[0]
        self.assertEqual(high.price,105)
        self.assertEqual(high.pivot_at,b.index[2])
        self.assertEqual(high.confirmation_at,b.index[5])
    def test_prefix_invariance(self):
        b=bars([100,102,105,104,103,102,99,101,104,101,96,98,102,99])
        full=zigzag_events(b,2)
        for n in range(1,len(b)+1):
            part=zigzag_events(b.iloc[:n],2)
            expect=full.loc[full.confirmation_at<=b.index[n-1]].reset_index(drop=True)
            pd.testing.assert_frame_equal(part.reset_index(drop=True),expect,check_dtype=False)
    def test_flat_no_spurious_events(self):
        self.assertTrue(zigzag_events(bars([100]*50),1).empty)
    def test_threshold_validation(self):
        with self.assertRaises(ValueError): zigzag_events(bars([100,101]),0)
    def test_multitimeframe_no_future(self):
        ix=pd.date_range('2026-01-01T00:00Z',periods=24*30,freq='h')
        price=100+np.sin(np.arange(len(ix))/12)*5
        df=pd.DataFrame({'timestamp':ix,'open':price,'high':price+0.1,'low':price-0.1,'close':price})
        ev=events_from_split(df,1)
        self.assertTrue((ev.pivot_at<ev.confirmation_at).all())
        self.assertTrue(set(ev.timeframe).issubset({'1h','4h','1d'}))

if __name__=='__main__': unittest.main(verbosity=2)

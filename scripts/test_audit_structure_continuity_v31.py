from __future__ import annotations
import unittest
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_structure_continuity_v31 import segments, audit, produce_events, stat

class TestContinuity(unittest.TestCase):
    def mk(self, start='2026-01-01', n=25):
        t=pd.date_range(start,periods=n,freq='h',tz='UTC')
        closes=np.array([100,101,102,103,104,103,102,101,99,97,96,95,97,99,101,103,105,106,104,102,100,98,99,100,101],dtype=float)[:n]
        return pd.DataFrame({'timestamp':t,'open':closes,'high':closes+.3,'low':closes-.3,'close':closes})
    def test_gap_segmentation(self):
        idx=pd.DatetimeIndex(['2026-01-01T01:00:00Z','2026-01-01T02:00:00Z',
             '2026-01-01T06:00:00Z','2026-01-01T07:00:00Z'])
        bars=pd.DataFrame({'high':[1,2,3,4]},index=idx)
        groups=segments(bars,'1h')
        self.assertEqual([len(g) for g in groups],[2,2])
    def test_history_level_available_for_validation(self):
        t=pd.Timestamp('2026-01-02T10:00:00Z')
        cand=pd.DataFrame({'timestamp':[t], 'open':[100.]})
        candidates=pd.DataFrame({'event_id':['e1'],'signal_timestamp':[t-pd.Timedelta(hours=1)],
            'known_at':[t],'side':['LONG']})
        events=pd.DataFrame({'confirmed_at':[t-pd.Timedelta(days=1)],
            'pivot_at':[t-pd.Timedelta(days=1,hours=2)],'kind':['LOW'],'price':[98.]})
        r=audit(candidates,cand,events,'zigzag','1h',5,.1,5)
        self.assertEqual(r.status.iloc[0],'VALID_REFERENCE')
        self.assertEqual(r.age_since_confirmation_h.iloc[0],24.)
    def test_future_event_never_available(self):
        t=pd.Timestamp('2026-01-02T10:00:00Z')
        cand=pd.DataFrame({'timestamp':[t], 'open':[100.]})
        candidates=pd.DataFrame({'event_id':['e1'],'signal_timestamp':[t-pd.Timedelta(hours=1)],
            'known_at':[t],'side':['LONG']})
        events=pd.DataFrame({'confirmed_at':[t+pd.Timedelta(hours=1)],
            'pivot_at':[t-pd.Timedelta(hours=3)],'kind':['LOW'],'price':[98.]})
        r=audit(candidates,cand,events,'zigzag','1h',5,.1,5)
        self.assertEqual(r.status.iloc[0],'NO_CONFIRMED_PIVOT')
    def test_fractal_no_lookahead_prefix(self):
        market=self.mk()
        older=produce_events(market.iloc[:21],'fractal','1h',1.)
        longer=produce_events(market,'fractal','1h',1.)
        if len(older):
            a=older[['confirmed_at','pivot_at','kind','price']].reset_index(drop=True)
            b=longer.loc[longer.confirmed_at.le(market.timestamp.iloc[20]+pd.Timedelta(hours=1)),
                         ['confirmed_at','pivot_at','kind','price']].reset_index(drop=True)
            pd.testing.assert_frame_equal(a,b)
    def test_zigzag_no_lookahead_prefix(self):
        market=self.mk()
        older=produce_events(market.iloc[:21],'zigzag','1h',1.)
        longer=produce_events(market,'zigzag','1h',1.)
        a=older[['confirmed_at','pivot_at','kind','price']].reset_index(drop=True)
        b=longer.loc[longer.confirmed_at.le(market.timestamp.iloc[20]+pd.Timedelta(hours=1)),
                         ['confirmed_at','pivot_at','kind','price']].reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b)
    def test_stats_empty(self):
        self.assertIsNone(stat(pd.Series([],dtype=float))['p95'])

if __name__=='__main__':
    unittest.main(verbosity=2)

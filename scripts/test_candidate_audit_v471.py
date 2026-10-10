import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from candidate_audit_v471 import select_decisions, audit_pilot, _summarize_rows, START, END
from candidate_learning_v470 import SPECS


def observations(times):
    return pd.DataFrame({'decision_at': times, **{f'f{i}': [float(i)] * len(times) for i in range(144)}})


def minutes():
    times=pd.date_range(START,END-pd.Timedelta(minutes=1),freq='min')
    close=100+pd.Series(range(len(times))).to_numpy()*.00001
    return pd.DataFrame({'timestamp':times,'open':close,'high':close+.2,'low':close-.2,
                         'close':close,'volume':10000.})


def fake_score(window,t,candidate,*,horizon_minutes):
    return {'id':candidate['id'],'preview_accepted':True,'preview_code':'OK',
            'entry_count':1,'exit_count':1,'fill_rejections':0,'closed':True,
            'exit_reason':'AGENT_CLOSE','net_pnl':-1.,'net_return_pct':-.01,
            'status':'CLOSED'}

class Tests(unittest.TestCase):
    def setUp(self):
        self.first=pd.Timestamp('2024-01-03T13:10:00Z')
        self.times=[self.first+pd.Timedelta(days=2*i) for i in range(3)]
        self.obs=observations(self.times)
    def test_selection(self):
        self.assertEqual(select_decisions(self.obs,count=3),self.times)
    def test_missing_scheduled(self):
        with self.assertRaisesRegex(ValueError,'missing scheduled'):
            select_decisions(self.obs.iloc[:2],count=3)
    def test_duplicate(self):
        with self.assertRaises(ValueError):select_decisions(pd.concat([self.obs,self.obs.iloc[:1]]),count=2)
    def test_unordered(self):
        with self.assertRaises(ValueError):select_decisions(self.obs.iloc[::-1],count=2)
    def test_overlap(self):
        with self.assertRaises(ValueError):select_decisions(self.obs,stride_minutes=60)
    def test_invalid_count(self):
        for n in (0,-1,33,True):
            with self.assertRaises(ValueError):select_decisions(self.obs,count=n)
    def test_invalid_horizon(self):
        for n in (0,61,245):
            with self.assertRaises(ValueError):select_decisions(self.obs,horizon_minutes=n)
    def test_naive_first(self):
        with self.assertRaises(ValueError):select_decisions(self.obs,first='2024-01-03T13:10:00')
    def test_boundary(self):
        with self.assertRaises(ValueError):select_decisions(self.obs,count=3,first='2024-01-30T13:10:00Z')
    def test_summary(self):
        rows=[{'decision_at':'T','id':spec.tag,'preview_accepted':True,
               'status':'CLOSED','exit_reason':'STOP_INTRABAR','fill_rejections':0,'net_pnl':-3.}
              for spec in SPECS]
        r=_summarize_rows(rows)
        self.assertEqual(r[SPECS[0].tag]['closed_losses'],1)
        self.assertEqual(r[SPECS[0].tag]['closed_net_pnl_mean_usdt'],-3.)
    def test_not_closed_excluded(self):
        rows=[{'decision_at':'T','id':spec.tag,'preview_accepted':False,
               'status':'NO_FILL','exit_reason':None,'fill_rejections':1,'net_pnl':None}
              for spec in SPECS]
        r=_summarize_rows(rows)
        self.assertEqual(r[SPECS[0].tag]['closed_count'],0)
        self.assertIsNone(r[SPECS[0].tag]['closed_net_pnl_mean_usdt'])
    def test_incomplete_group(self):
        with self.assertRaises(ValueError):_summarize_rows([{'decision_at':'T','id':'x'}])
    def test_end_to_end_smoke(self):
        report=audit_pilot(minutes(),self.obs,count=2,scorer=fake_score)
        self.assertEqual(len(report['rows']),8)
        self.assertEqual(report['wait_baseline']['net_pnl_usdt_per_flat_episode'],0)
        self.assertEqual(report['by_candidate'][SPECS[0].tag]['closed_count'],2)
        self.assertEqual(report['dataset_role'],'DEVELOPMENT_PILOT_NOT_TRAIN')
    def test_no_future_for_candidates(self):
        f=minutes()
        observed=[]
        from candidate_audit_v471 import make_candidates as original
        def wrapper(history,t):
            observed.append((history.timestamp.iloc[-1],t))
            return original(history,t)
        with patch('candidate_audit_v471.make_candidates',side_effect=wrapper):
            audit_pilot(f,self.obs,count=1,scorer=fake_score)
        self.assertEqual(observed,[(self.first-pd.Timedelta(minutes=1),self.first)])
    def test_gap_rejected(self):
        f=minutes().drop(index=100).reset_index(drop=True)
        with self.assertRaises(ValueError):audit_pilot(f,self.obs,count=1,scorer=fake_score)
    def test_no_write(self):
        s=Path(__file__).with_name('candidate_audit_v471.py').read_text()
        for bad in ('to_csv(', 'to_parquet(', 'write_text(', 'torch', 'peft'):
            self.assertNotIn(bad,s)

if __name__=='__main__':unittest.main(verbosity=2)

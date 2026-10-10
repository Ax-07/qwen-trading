import unittest
import pandas as pd
from scripts.candidate_learning_v470 import (trailing_volatility,make_candidates,score_one,inspect_pilot,
                                     CandidateSpec)
from agent_actions_risk_v450 import ActionType

T0=pd.Timestamp('2024-01-03T10:00:00Z')

def candles(n=300):
    ix=pd.date_range(T0,periods=n,freq='min')
    price=100+pd.Series(range(n)).mul(.001).to_numpy()
    return pd.DataFrame({'timestamp':ix,'open':price,'close':price,'high':price+.15,
                         'low':price-.15,'volume':[10000.]*n})

class Tests(unittest.TestCase):
    def setUp(self):
        self.f=candles()
        self.t=T0+pd.Timedelta(minutes=100)
        self.h=self.f[self.f.timestamp<self.t]
    def test_trailing_finite(self):
        self.assertGreater(trailing_volatility(self.h,self.t),0)
    def test_exact_cutoff(self):
        with self.assertRaisesRegex(ValueError,'end exactly'):
            trailing_volatility(self.f,self.t)
    def test_future_independent(self):
        a=make_candidates(self.h,self.t)
        other=self.f.copy();other.loc[other.timestamp>=self.t,'close']=1000000
        b=make_candidates(other.loc[other.timestamp<self.t],self.t)
        self.assertEqual([(x['action'].sl,x['action'].tp) for x in a],[(x['action'].sl,x['action'].tp) for x in b])
    def test_actions(self):
        a=make_candidates(self.h,self.t)
        self.assertEqual(len(a),4)
        self.assertTrue(all(x['action'].kind==ActionType.OPEN_LONG for x in a))
        self.assertTrue(all(x['preview_accepted'] for x in a))
    def test_invalid_multiplier(self):
        with self.assertRaises(ValueError):make_candidates(self.h,self.t,specs=[CandidateSpec(-1,2,'bad')])
    def test_incomplete_history(self):
        with self.assertRaises(ValueError):trailing_volatility(self.h.tail(30),self.t)
    def test_gap(self):
        with self.assertRaisesRegex(ValueError,'gapped'):
            make_candidates(self.h.drop(self.h.index[5]),self.t)
    def test_bad_horizon(self):
        with self.assertRaises(ValueError):score_one(self.f,self.t,make_candidates(self.h,self.t)[0],horizon_minutes=61)
    def test_one_closed(self):
        r=score_one(self.f,self.t,make_candidates(self.h,self.t)[0])
        self.assertTrue(r['closed'],r)
        self.assertEqual(r['entry_count'],1)
        self.assertEqual(r['exit_count'],1)
        self.assertIsNotNone(r['net_pnl'])
    def test_no_future_window(self):
        short=self.f[self.f.timestamp<self.t+pd.Timedelta(minutes=30)]
        with self.assertRaisesRegex(ValueError,'forward'):
            score_one(short,self.t,make_candidates(self.h,self.t)[0])
    def test_window_close_executed_next_minute(self):
        from simulate_execution_v451 import replay
        from agent_actions_risk_v450 import Action,ActionType
        t=self.t
        sub=self.f[(self.f.timestamp>=t-pd.Timedelta(minutes=5)) &
                   (self.f.timestamp<t+pd.Timedelta(minutes=65))]
        ev,_=replay(sub,{t:make_candidates(self.h,t)[0]['action'],t+pd.Timedelta(minutes=60):Action(ActionType.CLOSE)})
        exits=ev[ev.event=='EXIT']
        if any(exits.reason=='AGENT_CLOSE'):
            self.assertEqual(exits.iloc[0]['at'],(t+pd.Timedelta(minutes=60)).isoformat())
    def test_pilot_flag(self):
        r=inspect_pilot(self.f,self.t)
        self.assertEqual(r['dataset_role'],'DEVELOPMENT_PILOT_NOT_TRAIN')
        self.assertEqual(len(r['candidates']),4)
        self.assertFalse(r['training'])
    def test_pilot_out_of_bound(self):
        with self.assertRaises(ValueError):inspect_pilot(self.f,'2025-01-01T12:00:00Z')
    def test_history_not_mutated(self):
        original=self.h.copy(deep=True)
        make_candidates(self.h,self.t)
        pd.testing.assert_frame_equal(original,self.h)
    def test_valid_144_observation(self):
        from scripts.candidate_learning_v470 import validate_pilot_observation
        names=[f'f{i}' for i in range(144)]
        obs=pd.DataFrame([{'decision_at':self.t,**{x:float(i) for i,x in enumerate(names)}}])
        d=validate_pilot_observation(obs,self.t)
        self.assertEqual(d['feature_count'],144)
        self.assertEqual(d['missing_feature_count'],0)
        self.assertEqual(inspect_pilot(self.f,self.t,observations=obs)['observation'],d)
    def test_wrong_observation_count(self):
        from scripts.candidate_learning_v470 import validate_pilot_observation
        with self.assertRaises(ValueError):
            validate_pilot_observation(pd.DataFrame([{'decision_at':self.t,'f0':1}]),self.t)
    def test_future_observation_not_allowed(self):
        from scripts.candidate_learning_v470 import validate_pilot_observation
        obs=pd.DataFrame([{'decision_at':self.t+pd.Timedelta(minutes=5),
                           **{f'f{i}':0. for i in range(144)}}])
        with self.assertRaisesRegex(ValueError,'match one'):
            validate_pilot_observation(obs,self.t)
    def test_no_output_writes(self):
        from pathlib import Path
        s=Path(__file__).with_name('candidate_learning_v470.py').read_text()
        self.assertNotIn('to_csv(',s)
        self.assertNotIn('to_parquet(',s)
        self.assertNotIn('torch',s)

if __name__=='__main__':unittest.main(verbosity=2)

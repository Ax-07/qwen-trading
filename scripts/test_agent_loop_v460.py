import unittest
import pandas as pd
from agent_actions_risk_v450 import Action, ActionType, Side
from agent_loop_v460 import replay_agent, HoldAgent, DemoAgent
from simulate_execution_v451 import replay
from test_simulate_execution_v451 import candles, START


def obs(n=20):
    t=pd.date_range(START+pd.Timedelta(minutes=5),periods=n//5,freq='5min')
    d={'decision_at':t}
    d.update({f'feature_{i:03d}':[float(i)]*len(t) for i in range(144)})
    return pd.DataFrame(d)

class Tests(unittest.TestCase):
    def test_hold(self):
        e,s=replay_agent(candles(),obs(),HoldAgent())
        self.assertEqual(s['trades_closed'],0)
        self.assertEqual(len(e),4)
    def test_demo_baseline(self):
        o=obs(); f=candles()
        e,s=replay_agent(f,o,DemoAgent(o.decision_at.iloc[0]))
        actions={o.decision_at.iloc[0]:Action(ActionType.OPEN_LONG,sl=99.5,tp=100.5),o.decision_at.iloc[1]:Action(ActionType.CLOSE)}
        baseline, bsum=replay(f,actions)
        self.assertEqual(s,bsum)
        pd.testing.assert_frame_equal(e, baseline, check_exact=False, rtol=1e-12, atol=1e-10)
    def test_only_current(self):
        seen=[]
        def agent(t, f, p, ctx):
            self.assertEqual(len(f),144)
            self.assertEqual(ctx.decision_at,t)
            seen.append(t)
            return Action(ActionType.WAIT)
        replay_agent(candles(),obs(),agent)
        self.assertEqual(len(seen),4)
    def test_read_only(self):
        def agent(t,f,p,ctx):
            with self.assertRaises(TypeError): f['feature_000']=500
            return Action(ActionType.WAIT)
        replay_agent(candles(),obs(),agent)
    def test_missing_row(self):
        with self.assertRaises(ValueError):replay_agent(candles(),obs().iloc[:-1],HoldAgent())
    def test_duplicate(self):
        o=obs();o.loc[1,'decision_at']=o.loc[0,'decision_at']
        with self.assertRaises(ValueError):replay_agent(candles(),o,HoldAgent())
    def test_future_row(self):
        o=obs();o.loc[0,'decision_at']=START+pd.Timedelta(minutes=1)
        with self.assertRaises(ValueError):replay_agent(candles(),o,HoldAgent())
    def test_inf(self):
        o=obs();o.loc[1,'feature_002']=float('inf')
        with self.assertRaises(ValueError):replay_agent(candles(),o,HoldAgent())
    def test_invalid_agent(self):
        with self.assertRaises(ValueError):replay_agent(candles(),obs(),lambda *x: 'WAIT')
    def test_no_future_candles(self):
        a,_=replay_agent(candles(15),obs(15),DemoAgent(START+pd.Timedelta(minutes=5)))
        b,_=replay_agent(candles(20),obs(20),DemoAgent(START+pd.Timedelta(minutes=5)))
        ac=a[a.event=='ENTRY'][['at','price','qty']].to_dict('records')
        bc=b[b.event=='ENTRY'][['at','price','qty']].to_dict('records')
        self.assertEqual(ac,bc)
    def test_short_refused(self):
        def agent(t,f,p,ctx):return Action(ActionType.OPEN_SHORT,sl=102,tp=98)
        e,s=replay_agent(candles(),obs(),agent)
        self.assertEqual(s['trades_closed'],0)
        self.assertTrue((e.code=='SHORT_NOT_SUPPORTED').any())
    def test_bad_candles(self):
        f=candles().drop(index=6).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError,'GAP_1M'):replay_agent(f,obs(),HoldAgent())
    def test_nan_permitted(self):
        o=obs();o.loc[0,'feature_001']=float('nan')
        e,_=replay_agent(candles(),o,HoldAgent())
        self.assertEqual(len(e),4)

if __name__=='__main__':unittest.main(verbosity=2)

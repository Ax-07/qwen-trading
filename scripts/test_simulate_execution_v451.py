import unittest
from datetime import timezone
import pandas as pd
from agent_actions_risk_v450 import Action, ActionType
from simulate_execution_v451 import replay, ExecutionConfig

START = pd.Timestamp('2024-01-01T00:00:00Z')

def candles(n=20, price=100.):
    ix=pd.date_range(START, periods=n, freq='min')
    return pd.DataFrame({'timestamp':ix, 'open':[price]*n, 'high':[price+.1]*n,
                         'low':[price-.1]*n, 'close':[price]*n, 'volume':[10000.]*n})

def run(f, entries=None, **kwargs):
    return replay(f, entries or {}, **kwargs)

class SimulatorTests(unittest.TestCase):
    def open(self, sl=98, tp=102):
        return {START+pd.Timedelta(minutes=5):Action(ActionType.OPEN_LONG,sl=sl,tp=tp)}
    def test_empty_fails(self):
        with self.assertRaises(ValueError):run(candles(0))
    def test_gap_fails(self):
        f=candles();f=f.drop(index=7).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError,'GAP_1M'):run(f)
    def test_partial_block_fails(self):
        with self.assertRaisesRegex(ValueError,'complete 5m'):run(candles(19))
    def test_no_retroactive_execution(self):
        ev,_=run(candles(),self.open())
        e=ev[ev.event=='ENTRY']
        self.assertEqual(len(e),1)
        self.assertEqual(e.iloc[0]['at'],(START+pd.Timedelta(minutes=5)).isoformat())
    def test_short_spot_refused(self):
        act={START+pd.Timedelta(minutes=5):Action(ActionType.OPEN_SHORT,sl=102,tp=98)}
        ev,s=run(candles(),act)
        self.assertFalse((ev.event=='ENTRY').any())
        self.assertIn('SHORT_NOT_SUPPORTED',ev.code.dropna().tolist())
    def test_stop_first_same_candle(self):
        f=candles(); f.loc[5,'high']=103; f.loc[5,'low']=97
        ev,s=run(f,self.open())
        self.assertEqual(ev.loc[ev.event=='EXIT','reason'].iloc[0],'STOP_INTRABAR')
    def test_stop_gap(self):
        f=candles(); f.loc[6,['open','high','low','close']]=[95,96,94,95]
        ev,s=run(f,self.open())
        self.assertEqual(ev.loc[ev.event=='EXIT','reason'].iloc[0],'STOP_GAP')
    def test_target(self):
        f=candles(); f.loc[6,'high']=103
        ev,s=run(f,self.open())
        self.assertEqual(ev.loc[ev.event=='EXIT','reason'].iloc[0],'TARGET_INTRABAR')
    def test_fee_and_slippage(self):
        ev,s=run(candles(),self.open(),config=ExecutionConfig())
        self.assertLess(s['cash_after_realized'],10000.)
        self.assertGreater(ev.loc[ev.event=='ENTRY','fee'].iloc[0],0)
    def test_close_next_open(self):
        acts=self.open();acts[START+pd.Timedelta(minutes=10)]=Action(ActionType.CLOSE)
        ev,s=run(candles(),acts)
        self.assertEqual(ev.loc[ev.event=='EXIT','reason'].iloc[0],'AGENT_CLOSE')
        self.assertEqual(ev.loc[ev.event=='EXIT','at'].iloc[0],(START+pd.Timedelta(minutes=10)).isoformat())
    def test_unfilled_last(self):
        f=candles(5)
        ev,_=run(f,self.open())
        self.assertTrue((ev.event=='UNFILLED_END_OF_DATA').any())
        self.assertFalse((ev.event=='ENTRY').any())
    def test_invalid_time(self):
        with self.assertRaises(ValueError):run(candles(),{START+pd.Timedelta(minutes=3):Action(ActionType.WAIT)})
    def test_modification_tightens(self):
        actions=self.open();actions[START+pd.Timedelta(minutes=10)]=Action(ActionType.MOVE_SL,sl=99)
        ev,s=run(candles(),actions)
        self.assertTrue((ev.event=='MODIFY').any())
    def test_widen_stop_refused(self):
        actions=self.open();actions[START+pd.Timedelta(minutes=10)]=Action(ActionType.MOVE_SL,sl=97)
        ev,s=run(candles(),actions)
        self.assertTrue((ev.code=='STOP_WIDENING').any())
    def test_prefix_stability_of_fills(self):
        a,_=run(candles(15),self.open());b,_=run(candles(25),self.open())
        self.assertEqual(a.loc[a.event=='ENTRY',['at','price','qty','sl','tp']].to_dict('records'), b.loc[b.event=='ENTRY',['at','price','qty','sl','tp']].to_dict('records'))

if __name__=='__main__':unittest.main(verbosity=2)

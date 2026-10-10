import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import qwen_contract_campaign_v465 as m

class Tests(unittest.TestCase):
    def setUp(self):
        self.names = [f'x{i}' for i in range(144)]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'pilot.parquet'
        t = pd.date_range('2024-01-01T00:05:00Z', periods=10, freq='5min')
        frame = pd.DataFrame({'decision_at': t, **{k: [float(i)]*10 for i,k in enumerate(self.names)}})
        self.frame = frame
        self.patcher = patch.object(m.pd, "read_parquet", side_effect=lambda p: self.frame.copy())
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
    def test_select(self):
        v=m.select_observations(self.path,self.names,3,2)
        self.assertEqual(len(v),3)
        self.assertEqual(v[1][0], pd.Timestamp('2024-01-01T00:15:00Z'))
    def test_overflow(self):
        with self.assertRaises(ValueError): m.select_observations(self.path,self.names,12,2)
    def test_bad_count(self):
        with self.assertRaises(ValueError): m.select_observations(self.path,self.names,0,1)
    def test_bad_stride(self):
        with self.assertRaises(ValueError): m.select_observations(self.path,self.names,1,0)
    def test_schema(self):
        with self.assertRaises(ValueError): m.select_observations(self.path,list(reversed(self.names)),1,1)
    def test_duplicate(self):
        f=self.frame; f.loc[2,'decision_at']=f.loc[1,'decision_at']
        with self.assertRaises(ValueError): m.select_observations(self.path,self.names,3,1)
    def test_outside_pilot(self):
        f=self.frame; f['decision_at']=f['decision_at']+pd.Timedelta(days=40)
        with self.assertRaises(ValueError): m.select_observations(self.path,self.names,1,1)
    def test_malformed_format(self):
        class C: pass
        result=m.evaluate('{"action":"WAIT","decision_at":"future"}',C())
        self.assertFalse(result['json_valid'])
    def test_bad_json(self):
        class C: pass
        self.assertFalse(m.evaluate('not json',C())['risk_accepted'])
    def test_aggregation(self):
        r=[dict(action='WAIT',json_valid=True,risk_accepted=True,risk_code='ACCEPTED',latency_seconds=2.,output_tokens=8),
           dict(action=None,json_valid=False,risk_accepted=False,risk_code='INVALID_JSON_CONTRACT',latency_seconds=4.,output_tokens=8)]
        s=m.summarize(r)
        self.assertEqual(s['json_valid_rate'],.5)
        self.assertEqual(s['actions'],{'WAIT':1,'INVALID':1})
        self.assertEqual(s['latency_mean_seconds'],3)
    def test_negative_latency(self):
        with self.assertRaises(ValueError): m.summarize([dict(action='WAIT',json_valid=True,risk_accepted=True,risk_code='ACCEPTED',latency_seconds=-1,output_tokens=3)])
    def test_empty(self):
        with self.assertRaises(ValueError): m.summarize([])
    def test_format_no_forced_wait(self):
        from agent_actions_risk_v450 import Position, MarketContext
        from datetime import datetime,timezone
        now=datetime(2024,1,1,0,5,tzinfo=timezone.utc)
        c=MarketContext(now,42000.,5.,1000000.,10000.,market_allows_short=False,quote_known_at=now)
        messages=m.format_messages(now,{k:0 for k in self.names},c,self.names)
        self.assertIn('No SHORT trades',messages[0]['content'])
        self.assertNotIn('Return exactly {"action":"WAIT"}',messages[0]['content'])
    def test_offline_flag(self):
        import os
        self.assertEqual(os.environ.get('HF_HUB_OFFLINE'),'1')
    def test_no_writes(self):
        import inspect
        src=inspect.getsource(m)
        self.assertNotIn('to_csv(',src)
        self.assertNotIn('to_parquet(',src)
    def test_default_safe(self):
        with patch.object(m.base,'verify_local_snapshot',return_value=Path('/tmp/model')):
            self.assertEqual(m.main(['--mode','check-cache']),0)

if __name__=='__main__':
    unittest.main(verbosity=2)

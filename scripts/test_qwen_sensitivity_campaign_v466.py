import unittest
from unittest.mock import patch
import pandas as pd
import qwen_sensitivity_campaign_v466 as m
from agent_actions_risk_v450 import MarketContext
from datetime import datetime,timezone

class Tests(unittest.TestCase):
    def setUp(self):
        self.names=[f'x{i}' for i in range(144)]
        times=pd.date_range('2024-01-01T00:05:00Z',periods=90,freq='5min')
        self.obs=pd.DataFrame({'decision_at':times,**{name:[float(i)]*len(times) for i,name in enumerate(self.names)}})
        t=pd.date_range('2024-01-01T00:00:00Z',periods=500,freq='min')
        close=[100+i*.05+5*__import__('math').sin(i/15) for i in range(500)]
        self.minutes=pd.DataFrame({'timestamp':t,'open':close,'high':[x+.3 for x in close], 'low':[x-.3 for x in close], 'close':close,'volume':[1.]*500})
    def test_selection(self):
        c=m.select_regimes(self.obs,self.names,self.minutes)
        self.assertEqual([x[0] for x in c],list(m.PROFILES))
        self.assertEqual(len({x[1] for x in c}),5)
    def test_selection_deterministic(self):
        a=m.select_regimes(self.obs,self.names,self.minutes)
        b=m.select_regimes(self.obs,self.names,self.minutes)
        self.assertEqual([(x[0],x[1]) for x in a],[(x[0],x[1]) for x in b])
    def test_schema_order(self):
        with self.assertRaises(ValueError):m.select_regimes(self.obs,list(reversed(self.names)),self.minutes)
    def test_duplicate(self):
        f=self.obs.copy();f.loc[3,'decision_at']=f.loc[2,'decision_at']
        with self.assertRaises(ValueError):m.select_regimes(f,self.names,self.minutes)
    def test_out_of_month(self):
        with self.assertRaises(ValueError):m.select_regimes(self.obs,self.names,self.minutes,end='2024-02-05T00:00:00Z')
    def test_insufficient(self):
        with self.assertRaises(ValueError):m.select_regimes(self.obs.head(1),self.names,self.minutes)
    def test_no_future_change(self):
        original=m.select_regimes(self.obs,self.names,self.minutes)
        c=self.minutes.copy();c.loc[450:,['open','high','low','close']]=1000000
        changed=m.select_regimes(self.obs,self.names,c)
        # Times earlier than the modified last 50 candles keep their prior metrics.
        a={x[1]:x[3] for x in original};b={x[1]:x[3] for x in changed}
        for ts in set(a)&set(b):
            if ts < pd.Timestamp('2024-01-01T07:30:00Z'):
                self.assertEqual(a[ts],b[ts])
    def test_pair_preserves_user(self):
        now=datetime(2024,1,1,0,5,tzinfo=timezone.utc)
        ctx=MarketContext(now,42000.,5.,100000.,10000.,market_allows_short=False,quote_known_at=now)
        result=m.prepare_pair(now,{k:0. for k in self.names},ctx,self.names)
        self.assertEqual(result['neutral'][1],result['cautious'][1])
        self.assertNotIn('Use WAIT if FLAT and uncertain',result['neutral'][0]['content'])
        self.assertNotIn('When uncertain, choose WAIT',result['neutral'][0]['content'])
    def test_evaluate_bad(self):
        self.assertFalse(m.previous.evaluate('{"action":"WAIT","reason":"test"}',None)['json_valid'])
    def test_summary(self):
        r=[]
        for profile in m.PROFILES:
            for prompt in ('cautious','neutral'):
                r.append({'profile':profile,'prompt':prompt,'action':'WAIT' if prompt=='cautious' else 'OPEN_LONG', 'risk_code':'ACCEPTED','json_valid':True,'risk_accepted':prompt=='cautious','latency_seconds':2.,'prompt_tokens':100})
        s=m.summarize(r)
        self.assertEqual(s['actions_changed_between_prompts'],5)
        self.assertEqual(s['total_inferences'],10)
    def test_summary_incomplete(self):
        with self.assertRaises(ValueError):m.summarize([{'profile':'x','prompt':'cautious','action':'WAIT','risk_code':'ACCEPTED','json_valid':True,'risk_accepted':True,'latency_seconds':1.,'prompt_tokens':50}])
    def test_cache_only(self):
        with patch.object(m.base,'verify_local_snapshot',return_value=__import__('pathlib').Path('/tmp/local')):
            self.assertEqual(m.main(['--mode','check-cache']),0)
    def test_no_writes(self):
        import inspect
        src=inspect.getsource(m)
        for s in ('to_csv(', 'to_parquet(', 'write_text(', 'open(\'w\''):
            self.assertNotIn(s,src)

if __name__=='__main__': unittest.main(verbosity=2)

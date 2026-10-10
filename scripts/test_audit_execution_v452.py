"""Synthetic tests for V4.5.2 independent ledger auditor."""
import unittest
import pandas as pd
from audit_execution_v452 import audit

S=pd.Timestamp('2024-01-01T00:00:00Z')

def fixture():
    minute=pd.DataFrame({'timestamp':pd.date_range(S,periods=15,freq='min'),
        'open':[100.]*15,'high':[101.]*15,'low':[99.]*15,'close':[100.]*15})
    rows=[]
    for n in range(1,4):
        t=S+pd.Timedelta(minutes=5*n)
        rows.append({'at':t.isoformat(),'event':'DECISION','action':('OPEN_LONG' if n==1 else 'CLOSE' if n==2 else 'WAIT'),'accepted':True,'code':'ACCEPTED'})
        if n==1:
            rows.append({'at':t.isoformat(),'event':'ENTRY','price':100.,'qty':1.,'sl':98.,'tp':102.,'fee':.07,'originating_decision':t.isoformat()})
        if n==2:
            rows.append({'at':t.isoformat(),'event':'EXIT','price':100.,'qty':1.,'exit_fee':.07,'net_trade_pnl':-.14,'cash':9999.86,'reason':'AGENT_CLOSE'})
    return pd.DataFrame(rows),{'events':5,'trades_closed':1,'cash_after_realized':9999.86,'equity_mark_to_market':9999.86,
        'position_side':'FLAT','no_real_orders':True,'data_start':S.isoformat(),'data_end':(S+pd.Timedelta(minutes=15)).isoformat()},minute

class Tests(unittest.TestCase):
    def valid(self):
        a,s,m=fixture(); return audit(a,s,minutes=m)
    def reject(self, edit):
        a,s,m=fixture(); edit(a,s,m)
        with self.assertRaises(ValueError): audit(a,s,minutes=m)
    def test_ok(self):
        r=self.valid(); self.assertEqual(r['trades_closed'],1); self.assertAlmostEqual(r['total_fees'],.14)
    def test_wrong_summary_cash(self):self.reject(lambda a,s,m:s.update(cash_after_realized=9990))
    def test_wrong_summary_equity(self):self.reject(lambda a,s,m:s.update(equity_mark_to_market=9990))
    def test_wrong_fee(self):self.reject(lambda a,s,m:a.loc.__setitem__((1,'fee'),.09))
    def test_wrong_pnl(self):self.reject(lambda a,s,m:a.loc.__setitem__((3,'net_trade_pnl'),-1.))
    def test_bad_qty(self):self.reject(lambda a,s,m:a.loc.__setitem__((3,'qty'),2.))
    def test_missing_decision(self):self.reject(lambda a,s,m:a.drop(index=4,inplace=True))
    def test_duplicate_decision(self):self.reject(lambda a,s,m:a.loc.__setitem__((4,'at'),(S+pd.Timedelta(minutes=10)).isoformat()))
    def test_entry_future_origin(self):self.reject(lambda a,s,m:a.loc.__setitem__((1,'originating_decision'),(S+pd.Timedelta(minutes=10)).isoformat()))
    def test_exit_before_entry(self):self.reject(lambda a,s,m:a.loc.__setitem__((3,'at'),(S+pd.Timedelta(minutes=4)).isoformat()))
    def test_missing_candles(self):self.reject(lambda a,s,m:m.drop(index=7,inplace=True))
    def test_gap(self):self.reject(lambda a,s,m:m.loc.__setitem__((8,'timestamp'),S+pd.Timedelta(minutes=20)))
    def test_unknown_event(self):self.reject(lambda a,s,m:a.loc.__setitem__((3,'event'),'MAGIC'))
    def test_spoofed_no_real_orders(self):self.reject(lambda a,s,m:s.update(no_real_orders=False))
    def test_bad_start(self):self.reject(lambda a,s,m:s.update(data_start=(S+pd.Timedelta(minutes=1)).isoformat()))
    def test_mark_to_market_open(self):
        a,s,m=fixture(); a=a.iloc[:2].copy(); a.loc[2]={'at':(S+pd.Timedelta(minutes=10)).isoformat(),'event':'DECISION','action':'HOLD','accepted':True}
        a.loc[3]={'at':(S+pd.Timedelta(minutes=15)).isoformat(),'event':'DECISION','action':'HOLD','accepted':True}
        a.loc[4]={'at':(S+pd.Timedelta(minutes=15)).isoformat(),'event':'MARK_TO_MARKET_ONLY','unrealized_pnl':0.}
        a.loc[4,'unrealized_pnl']=0.0
        s.update(events=5,trades_closed=0,cash_after_realized=9999.93,equity_mark_to_market=9999.93,position_side='LONG')
        self.assertEqual(audit(a,s,minutes=m)['position_side'],'LONG')
    def test_no_minutes(self):
        a,s,m=fixture()
        with self.assertRaises(ValueError): audit(a,s)

if __name__=='__main__':unittest.main(verbosity=2)

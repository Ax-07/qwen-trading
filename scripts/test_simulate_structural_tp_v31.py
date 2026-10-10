from __future__ import annotations
import unittest
from pathlib import Path
import sys
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from simulate_structural_tp_v31 import simulate_one, simulate_split, validate_audit, MULTIPLES

T0 = pd.Timestamp('2026-01-01T00:00:00Z')

def market(bars):
    return pd.DataFrame([dict(timestamp=T0+pd.Timedelta(hours=i),open=o,high=h,low=l,close=c)
                         for i,(o,h,l,c) in enumerate(bars)])

def candidates(n=3,stop=95,known_offset=0):
    return pd.DataFrame([dict(event_id=f'e{i}',known_at=T0+pd.Timedelta(hours=i+known_offset),
        signal_timestamp=T0+pd.Timedelta(hours=i+known_offset-1),side='LONG',
        method='zigzag',timeframe='1h',entry_open=100.,stop_price=float(stop),
        stop_confirmed_at=T0-pd.Timedelta(hours=1),status='VALID_REFERENCE') for i in range(n)])

class TestStructuralTP(unittest.TestCase):
    def test_stop_first_ambiguous(self):
        r=simulate_one(market([(100,110,90,104)]),'LONG',95.,1.)
        self.assertEqual(r['reason'],'SL')
        self.assertLess(r['net_return'],0)

    def test_short_tp(self):
        r=simulate_one(market([(100,101,93,95)]),'SHORT',105.,1.)
        self.assertEqual(r['reason'],'TP')
        self.assertGreater(r['net_return'],0)

    def test_gap_stop_after_entry(self):
        r=simulate_one(market([(100,102,98,100),(90,93,88,90)]),'LONG',95.,1.)
        self.assertEqual(r['reason'],'SL_GAP')
        self.assertEqual(r['exit_raw'],90)

    def test_timeout_and_costs(self):
        r=simulate_one(market([(100,101,99,100)]),'LONG',95.,5.)
        self.assertEqual(r['reason'],'TIME')
        self.assertLess(r['net_return'],0)

    def test_stop_wrong_side(self):
        with self.assertRaises(ValueError):
            simulate_one(market([(100,102,99,101)]),'LONG',105.,1.)

    def test_future_confirmation_rejected(self):
        c=candidates(1)
        c['stop_confirmed_at']=T0+pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ValueError,'FUITE'):
            validate_audit(c)

    def test_one_position_and_same_candidates(self):
        candles=market([(100,100.4,99.6,100)]*16)
        rows=candidates(4)
        out=simulate_split(rows,candles,4,5,2)
        self.assertEqual(len(out),len(MULTIPLES)*4)
        for _,g in out.groupby('multiple_r'):
            self.assertEqual(int((g.portfolio_status=='TAKEN').sum()),1)
            self.assertEqual(int((g.portfolio_status=='SKIPPED_OVERLAP').sum()),3)

    def test_horizon_split_prevents_partial_look(self):
        candles=market([(100,101,99,100)]*6)
        c=candidates(4)
        out=simulate_split(c,candles,4,5,2)
        for _,g in out.groupby('multiple_r'):
            self.assertEqual(g.eligibility.tolist(),['ELIGIBLE','ELIGIBLE','ELIGIBLE','INCOMPLETE_HORIZON'])

    def test_gap_in_hourly_bars_blocks_trade(self):
        candles=market([(100,101,99,100)]*7)
        candles.loc[2,'timestamp']+=pd.Timedelta(hours=5)
        c=candidates(1)
        out=simulate_split(c,candles,4,5,2)
        self.assertTrue((out.eligibility=='INCOMPLETE_HORIZON').all())

if __name__=='__main__':
    unittest.main(verbosity=2)

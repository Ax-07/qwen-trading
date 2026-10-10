from __future__ import annotations
import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose_structural_tp_v31 import validate, breakdown, common_comparison, chronology, METHODS, MULTIPLES


def sample():
    rows=[]
    for method, tf in METHODS:
        for r in MULTIPLES:
            for j, side in enumerate(('LONG','SHORT')):
                entry=pd.Timestamp('2026-06-03T10:00:00Z')+pd.Timedelta(hours=13*j)
                net=.02 if method=='zigzag' else -.01
                rows.append(dict(method=method,timeframe=tf,multiple_r=r,event_id=f'e{j}',side=side,
                    known_at=entry,eligibility='ELIGIBLE',portfolio_status='TAKEN',reason='TP' if j==0 else 'TIME',
                    net_return=net,net_r=net*2,gross_return=net+.001,duration_bars=j+2,
                    entry_at=entry,exit_at=entry+pd.Timedelta(hours=j+2),
                    equity_before=1.0 if j==0 else 1.+net,
                    equity_after=1.+net if j==0 else (1.+net)**2,audit_status='VALID_REFERENCE'))
    return pd.DataFrame(rows)


class DiagnosticTests(unittest.TestCase):
    def test_valid_schema(self):
        self.assertEqual(len(validate(sample(),'train')),48)

    def test_duplicate_rejected(self):
        f=sample()
        with self.assertRaises(ValueError):validate(pd.concat([f,f.iloc[[0]]]),'train')

    def test_no_lookahead_eligibility(self):
        f=sample()
        f.loc[0,'eligibility']='NO_CONFIRMED_PIVOT'
        with self.assertRaises(ValueError):validate(f,'train')

    def test_common_intersection_and_paired(self):
        f=validate(sample(),'train')
        common,paired=common_comparison(f,'train')
        self.assertTrue((common.common_candidates==1).all())
        self.assertEqual(len(paired),36)
        self.assertTrue((paired.mean_paired_delta_pct >= 0).all())

    def test_split_side_and_outcomes(self):
        b=breakdown(validate(sample(),'train'),'train')
        self.assertEqual(len(b),48)
        self.assertTrue((b.eligible==1).all())
        self.assertEqual(int(b.taken_time.sum()),24)

    def test_chronology(self):
        monthly=chronology(validate(sample(),'train'),'train')
        self.assertEqual(len(monthly),24)
        self.assertTrue((monthly.taken==2).all())

if __name__=='__main__':unittest.main(verbosity=2)

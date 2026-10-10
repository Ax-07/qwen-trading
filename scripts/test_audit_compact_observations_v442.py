import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_compact_observations_v442 import audit_and_project, expected_columns, TFS


def fixture(n=32):
    cols = expected_columns()
    d = {c: np.ones(n, dtype='float32') for c in cols}
    for tf in TFS:
        d[f'context_available_{tf}'] = np.ones(n, dtype='int8')
        for c in cols:
            if c.endswith('_available'):
                d[c] = np.ones(n, dtype='int8')
    d['decision_at'] = pd.date_range('2024-01-01', periods=n, freq='5min', tz='UTC')
    return pd.DataFrame(d)

class Tests(unittest.TestCase):
    def test_shape(self):
        d = fixture()
        out, audit, pairs, schema = audit_and_project(d)
        self.assertEqual(out.shape, (32, len(expected_columns())+1))
        self.assertEqual(schema['pilot_feature_count'], len(expected_columns()))
        self.assertNotIn('decision_at', audit.column.to_list())
    def test_schema_stable(self):
        self.assertEqual(expected_columns(), expected_columns())
        self.assertEqual(len(expected_columns()), len(set(expected_columns())))
    def test_missing_required_fails(self):
        with self.assertRaises(ValueError):
            audit_and_project(fixture().drop(columns=[expected_columns()[4]]))
    def test_inf_fails(self):
        d=fixture(); d.loc[0, 'rsi14_5m']=np.inf
        with self.assertRaises(ValueError): audit_and_project(d)
    def test_duplicate_time_fails(self):
        d=fixture(); d.loc[1,'decision_at']=d.loc[0,'decision_at']
        with self.assertRaises(ValueError): audit_and_project(d)
    def test_context_absence(self):
        d=fixture(); d.loc[0, 'context_available_4h']=0
        with self.assertRaises(ValueError): audit_and_project(d)
        for c in expected_columns():
            if (c.endswith('_4h') or c.endswith('_4h_code') or c.endswith('_4h_available')) and c != 'context_available_4h':
                d.loc[0,c]=0 if c.endswith('_available') else np.nan
        out,*_=audit_and_project(d)
        self.assertTrue(pd.isna(out.loc[0,'rsi14_4h']))
    def test_constant_audit(self):
        _,audit,_,_=audit_and_project(fixture())
        self.assertTrue(bool(audit.loc[audit.column=='rsi14_5m','is_constant'].iloc[0]))
    def test_prefix_stability(self):
        d=fixture(40)
        first,*_=audit_and_project(d.iloc[:25].copy())
        full,*_=audit_and_project(d.copy())
        pd.testing.assert_frame_equal(first,full.iloc[:25].reset_index(drop=True),check_exact=True)

if __name__=='__main__': unittest.main(verbosity=2)

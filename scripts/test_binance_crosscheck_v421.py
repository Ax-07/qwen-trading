"""Tests synthétiques hors réseau de l'étape V4.2.1."""
import io
import sys
import unittest
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prototype_multitimeframe_v42 import synthetic_minutes, completed_bars
from validate_binance_1m_vs_1h_v421 import compare
from download_binance_1m_v421 import parse_minute_zip, gaps_report


class CrossCheckTests(unittest.TestCase):
    def setUp(self):
        self.one = synthetic_minutes(120)
        self.one['quote_volume'] = self.one['volume'] * self.one['close']
        self.hour = completed_bars(self.one, 60).drop(columns=['known_at', 'count_1m'])

    def test_exact_aggregates(self):
        summary, _ = compare(self.one, self.hour)
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['compared_hours'], 2)

    def test_volume_mismatch(self):
        ref = self.hour.copy()
        ref.loc[0, 'volume'] += 1
        summary, _ = compare(self.one, ref)
        self.assertFalse(summary['passed'])
        self.assertEqual(summary['fields']['volume']['mismatches'], 1)

    def test_gap_disqualifies_hour(self):
        minute = self.one.drop(index=18).reset_index(drop=True)
        summary, _ = compare(minute, self.hour)
        self.assertEqual(summary['compared_hours'], 1)
        self.assertEqual(len(summary['hours_only_in_1h']), 1)
        self.assertFalse(summary['passed'])

    def test_no_matching_hours(self):
        shifted = self.hour.copy()
        shifted['timestamp'] += pd.Timedelta(days=1)
        with self.assertRaises(ValueError):
            compare(self.one, shifted)

    def test_archive_parse(self):
        rows=[]
        for ts in pd.date_range('2024-01-01', periods=2, freq='min', tz='UTC'):
            ms = int(ts.timestamp()*1000)
            rows.append(','.join(map(str, [ms,1,2,0.5,1.5,10,ms+59999,15,2,4,6,0])))
        buff=io.BytesIO()
        with zipfile.ZipFile(buff,'w') as z:
            z.writestr('BTCUSDT-1m-2024-01.csv','\n'.join(rows))
        result=parse_minute_zip(buff.getvalue(), 'BTCUSDT','2024-01')
        self.assertEqual(len(result),2)
        self.assertEqual(gaps_report(result,'2024-01')[0],44640)

    def test_month_reject(self):
        rows='1704067200000,1,2,0.5,1.5,10,1704067259999,15,2,4,6,0'
        buff=io.BytesIO()
        with zipfile.ZipFile(buff,'w') as z:z.writestr('a.csv', rows)
        with self.assertRaises(ValueError):
            parse_minute_zip(buff.getvalue(),'BTCUSDT','2024-02')


if __name__=='__main__':
    unittest.main(verbosity=2)

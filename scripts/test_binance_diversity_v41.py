from __future__ import annotations
import hashlib
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from download_binance_diversity_v41 import month_sequence, binance_epoch_to_utc,parse_month_zip
from analyze_binance_diversity_v41 import load_symbol,calculate

class DiversityTests(unittest.TestCase):
    def test_months(self):
        self.assertEqual(list(month_sequence('2023-12','2024-02')),['2023-12','2024-01','2024-02'])
        with self.assertRaises(ValueError):list(month_sequence('2024-02','2024-01'))

    def test_timestamp_us_ms(self):
        for v in (1711929600000,1711929600000000):
            self.assertEqual(str(binance_epoch_to_utc(pd.Series([v])).iloc[0]),'2024-04-01 00:00:00+00:00')

    def test_month_parse(self):
        ts=1704067200000
        csvrow=f'{ts},100,110,90,102,5,{ts+3599999},510,4,2,204,0\n'
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('BTCUSDT-1h-2024-01.csv',csvrow)
        df=parse_month_zip(b.getvalue(),'BTCUSDT','2024-01')
        self.assertEqual(df.loc[0,'close'],102)

    def test_gap_no_ffill(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'BTCUSDT';f.mkdir()
            times=pd.to_datetime(['2024-01-01T00:00Z','2024-01-01T02:00Z'])
            pd.DataFrame({'timestamp':times,'close':[100.,102.],'quote_volume':[300,400]}).to_pickle(f/'2024-01.parquet')
            with patch('pandas.read_parquet', side_effect=pd.read_pickle):
                d,r=load_symbol(f)
            self.assertEqual(len(d),3)
            self.assertEqual(int(r.notna().sum()),0)

    def test_pairwise_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'monthly'
            t=pd.date_range('2024-01-01',periods=150,freq='h',tz='UTC')
            for i,symbol in enumerate(['BTCUSDT','ETHUSDT']):
                f=root/symbol;f.mkdir(parents=True)
                pd.DataFrame({'timestamp':t,'close':[100+i+(.03*j)+((j%3)*.2) for j in range(150)],'quote_volume':[1000]*150}).to_pickle(f/'2024-01.parquet')
            with patch('pandas.read_parquet', side_effect=pd.read_pickle):
                m,c,n,_=calculate(Path(td),min_overlap=50)
            self.assertEqual(len(m),2)
            self.assertEqual(int(n.loc['BTCUSDT','ETHUSDT']),149)
            self.assertAlmostEqual(c.loc['BTCUSDT','ETHUSDT'],1,places=5)

if __name__=='__main__':unittest.main(verbosity=2)

import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from inventory_binance_v40 import extract_page, inventory, missing_months, next_month

class TestInventory(unittest.TestCase):
    def test_month_rollover(self):
        self.assertEqual(next_month('2025-12'),'2026-01')
    def test_missing(self):
        self.assertEqual(missing_months(['2025-12','2026-02']),['2026-01'])
    def test_s3_xml(self):
        xml=b'''<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>true</IsTruncated><NextContinuationToken>ab/c+</NextContinuationToken><Contents><Key>data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip</Key><Size>123</Size><LastModified>2024-02-03</LastModified></Contents><Contents><Key>file.CHECKSUM</Key><Size>33</Size></Contents></ListBucketResult>'''
        rows,trunc,token=extract_page(xml)
        self.assertEqual(len(rows),1)
        self.assertTrue(trunc)
        self.assertEqual(token,'ab/c+')
    def test_inventory(self):
        rows,summary=inventory('BTCUSDT',[
            ('data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip',123,'x'),
            ('data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-03.zip',123,'y')])
        self.assertEqual(summary['missing_months'],'2024-02')
        self.assertEqual(len(rows),2)
    def test_empty(self):
        rows,summary=inventory('ABCUSDT',[])
        self.assertEqual(summary['status'],'NO_MONTHLY_1M_ARCHIVES')

if __name__=='__main__':unittest.main(verbosity=2)

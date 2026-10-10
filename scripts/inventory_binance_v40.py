#!/usr/bin/env python3
"""V4.0: inventory of Binance public SPOT monthly 1m archives. No OHLC downloaded.

Usage (repository root):
  python scripts/inventory_binance_v40.py
  python scripts/inventory_binance_v40.py --symbols BTCUSDT ETHUSDT SOLUSDT ...
  python scripts/inventory_binance_v40.py --output data/evaluation/v40-binance-inventory

Only calls the public S3 listing API; writes CSV/JSON in a new directory.
No exchange credentials; no TEST dataset; no training.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

S3 = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
ARCHIVE = 'https://data.binance.vision/'
DEFAULT = [
 'BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','ADAUSDT','DOGEUSDT',
 'LINKUSDT','AVAXUSDT','DOTUSDT','LTCUSDT','BCHUSDT','TRXUSDT','ATOMUSDT',
 'NEARUSDT','ETCUSDT','XLMUSDT','FILUSDT','UNIUSDT','APTUSDT',
 'ARBUSDT','OPUSDT','SUIUSDT','INJUSDT','AAVEUSDT','ALGOUSDT',
 'MKRUSDT','ICPUSDT','GRTUSDT','HBARUSDT',
]
MONTH = re.compile(r'^data/spot/monthly/klines/([A-Z0-9]+)/1m/\1-1m-(\d{4}-\d{2})\.zip$')
NS = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def extract_page(xml_bytes: bytes):
    root = ET.fromstring(xml_bytes)
    def tag(node, name, default=None):
        el=node.find('s3:'+name,NS)
        return (el.text if el is not None else default)
    objects=[]
    for node in root.findall('s3:Contents',NS):
        key=tag(node,'Key','')
        if key.endswith('.zip'):
            objects.append((key,int(tag(node,'Size','0')),tag(node,'LastModified','')))
    return objects,tag(root,'IsTruncated','false')=='true',tag(root,'NextContinuationToken')


def list_objects(symbol: str, retries: int = 3):
    prefix=f'data/spot/monthly/klines/{symbol}/1m/'
    token=None
    seen_tokens=set()
    objects=[]
    while True:
        params={'list-type':'2','prefix':prefix,'max-keys':'1000'}
        if token:
            if token in seen_tokens: raise RuntimeError('Pagination cyclique')
            seen_tokens.add(token)
            params['continuation-token']=token
        url=S3+'?'+urllib.parse.urlencode(params)
        for attempt in range(retries):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'qwen-trading-v40-inventory/1.0'})
                with urllib.request.urlopen(req,timeout=25) as response:
                    data=response.read()
                break
            except (urllib.error.URLError,TimeoutError) as exc:
                if attempt==retries-1: raise RuntimeError(f'S3 listing failed for {symbol}: {exc}') from exc
                time.sleep(0.5*(2**attempt))
        batch, truncated, following=extract_page(data)
        objects.extend(batch)
        if not truncated: break
        if not following: raise RuntimeError('Pagination S3 tronquée sans token')
        token=following
    return objects


def next_month(month: str):
    year,mo=map(int,month.split('-'))
    return f'{year+1:04d}-01' if mo==12 else f'{year:04d}-{mo+1:02d}'


def missing_months(months: list[str]):
    if not months:return []
    have=set(months)
    result=[]
    current=min(have)
    while current<=max(have):
        if current not in have: result.append(current)
        current=next_month(current)
    return result


def inventory(symbol: str, objects):
    rows=[]
    for key,size,modified in objects:
        m=MONTH.fullmatch(key)
        if m and m.group(1)==symbol:
            rows.append({'symbol':symbol,'month':m.group(2),'size_bytes':size,
                        'last_modified':modified,'url':ARCHIVE+key})
    rows.sort(key=lambda item:item['month'])
    months=[r['month'] for r in rows]
    missing=missing_months(months)
    summary={'symbol':symbol,'monthly_zip_count':len(rows),
             'first_month':min(months) if months else '',
             'last_month':max(months) if months else '',
             'missing_month_count':len(missing),'missing_months':';'.join(missing),
             'total_archive_bytes':sum(r['size_bytes'] for r in rows),
             'status':'ARCHIVES_FOUND' if rows else 'NO_MONTHLY_1M_ARCHIVES'}
    return rows,summary


def write_csv(path, rows, columns):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--symbols',nargs='+',default=DEFAULT)
    ap.add_argument('--output',type=Path,default=Path('data/evaluation/v40-binance-inventory'))
    ap.add_argument('--max-symbols',type=int,default=0,help='Optional cap for a quick smoke test')
    args=ap.parse_args(argv)
    symbols=list(dict.fromkeys(s.upper() for s in args.symbols))
    if not symbols or any(not re.fullmatch(r'[A-Z0-9]{4,20}',s) for s in symbols):
        ap.error('Invalid symbol')
    if args.max_symbols<0:ap.error('--max-symbols must be >=0')
    if args.max_symbols:symbols=symbols[:args.max_symbols]
    if args.output.exists() and any(args.output.iterdir()):
        ap.error(f'Output directory is not empty (no overwrite): {args.output}')
    all_files=[]; summaries=[]
    for i,symbol in enumerate(symbols,1):
        try:
            objects=list_objects(symbol)
            rows,summary=inventory(symbol,objects)
        except Exception as exc:
            rows=[]
            summary={'symbol':symbol,'monthly_zip_count':0,'first_month':'','last_month':'',
                     'missing_month_count':0,'missing_months':'','total_archive_bytes':0,
                     'status':f'ERROR: {type(exc).__name__}: {exc}'}
        all_files.extend(rows);summaries.append(summary)
        print(f'{i:2}/{len(symbols)} {symbol:12} {summary["monthly_zip_count"]:3} months '
              f'{summary["first_month"] or "?"} .. {summary["last_month"] or "?"} '
              f'missing={summary["missing_month_count"]} {summary["status"]}')
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'symbols.csv',summaries,list(summaries[0]))
    write_csv(args.output/'monthly_archives.csv',all_files,
              ['symbol','month','size_bytes','last_modified','url'])
    report={'version':'v40-inventory-v1','generated_at_utc':datetime.now(timezone.utc).isoformat(),
            'market':'SPOT','interval':'1m','archive_type':'monthly',
            'listing_source':S3,'downloaded_ohlcv':False,
            'symbols_checked':len(symbols),'monthly_zip_files':len(all_files),
            'errors':sum(s['status'].startswith('ERROR:') for s in summaries),
            'scope_note':'Files present do not prove candle completeness, trading status, liquidity, or low BTC correlation.',
            'limitations':['Current month needs daily archive inventory, not covered here',
                           'Historic listings do not imply a symbol is currently tradable',
                           'Pair correlations/liquidity require a separate OHLCV sample on training dates']}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Output:',args.output.resolve())
    print('No market candles downloaded. No Test accessed. No training or trading.')
    if report['errors']:
        print('WARNING: Listing errors occurred; do not treat missing results as unavailable.',file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':raise SystemExit(main())

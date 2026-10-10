from __future__ import annotations
"""V4.1 - Téléchargement vérifié des klines Spot Binance 1h (Train exploratoire).

Exemples :
  python scripts/download_binance_diversity_v41.py --symbols BTCUSDT ETHUSDT --start 2024-01 --end 2024-01
  python scripts/download_binance_diversity_v41.py
Pas de téléchargement 1m, pas d'entraînement, pas de sélection sur Validation/Test.
"""
import argparse
import calendar
import csv
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SYMBOLS = ('BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','DOGEUSDT',
           'LINKUSDT','LTCUSDT','TRXUSDT','ATOMUSDT','AAVEUSDT','XLMUSDT')
BASE = 'https://data.binance.vision/data/spot/monthly/klines'
COLS = ['open_time','open','high','low','close','volume','close_time','quote_volume',
        'trades','taker_buy_base','taker_buy_quote','ignore']
OUTPUT_COLS = ['timestamp','open','high','low','close','volume','quote_volume','trades']


def month_sequence(start: str, end: str):
    def parse(s):
        if not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])',s):
            raise ValueError('Date au format YYYY-MM')
        return tuple(map(int,s.split('-')))
    a,b=parse(start),parse(end)
    if a>b: raise ValueError('start > end')
    y,m=a
    while (y,m)<=b:
        yield f'{y:04d}-{m:02d}'
        m+=1
        if m==13:y,m=y+1,1


def binance_epoch_to_utc(values):
    """Spot Binance : ms jusqu'à 2024, µs sur certaines archives 2025+."""
    numeric = pd.to_numeric(values, errors='raise').astype('int64')
    units = numeric.abs().ge(10**14)
    if bool(units.any()) and bool((~units).any()):
        raise ValueError('Unités d’horodatage mixtes au sein d’une archive')
    return pd.to_datetime(numeric,unit='us' if bool(units.any()) else 'ms',utc=True)


def request_bytes(url: str, timeout: int = 50, retries: int = 3):
    last=None
    for attempt in range(retries):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'QwenTradingV4-Research/1.0'})
            with urllib.request.urlopen(req,timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403,404): raise
            last=e
        except (urllib.error.URLError,TimeoutError) as e:
            last=e
        if attempt+1<retries:time.sleep(1.5*(attempt+1))
    raise RuntimeError(f'Téléchargement impossible : {url}: {last}')


def parse_month_zip(payload: bytes, symbol: str, month: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names=[n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX/')]
        if len(names)!=1:raise ValueError(f'{symbol} {month}: archive CSV ambiguë')
        with z.open(names[0]) as file:
            df=pd.read_csv(file,header=None,names=COLS,low_memory=False)
    if df.empty:raise ValueError(f'{symbol} {month}: archive vide')
    # Certains fichiers possèdent un en-tête; pas attendu en Spot mais accepté explicitement.
    if str(df.iloc[0]['open_time']).lower() in ('open_time','opentime'):
        df=df.iloc[1:].copy()
    df['timestamp']=binance_epoch_to_utc(df['open_time'])
    for c in ('open','high','low','close','volume','quote_volume'):
        df[c]=pd.to_numeric(df[c],errors='raise')
    df['trades']=pd.to_numeric(df['trades'],errors='raise').astype('int64')
    if not df.timestamp.is_monotonic_increasing or df.timestamp.duplicated().any():
        raise ValueError(f'{symbol} {month}: timestamps non ordonnés ou doublons')
    if (df.timestamp.dt.minute.ne(0) | df.timestamp.dt.second.ne(0)).any():
        raise ValueError(f'{symbol} {month}: bougies non alignées à 1h')
    if (df.timestamp.dt.strftime('%Y-%m')!=month).any():
        raise ValueError(f'{symbol} {month}: bougie hors du mois')
    if (df[['open','high','low','close']]<=0).any().any() or (df[['volume','quote_volume','trades']]<0).any().any():
        raise ValueError(f'{symbol} {month}: prix/volumes invalides')
    if not ((df.high>=df[['open','close','low']].max(axis=1)) &
            (df.low<=df[['open','close','high']].min(axis=1))).all():
        raise ValueError(f'{symbol} {month}: incohérence OHLC')
    return df[OUTPUT_COLS].reset_index(drop=True)


def expected_hours(month):
    y,m=map(int,month.split('-'))
    return calendar.monthrange(y,m)[1]*24


def build(args):
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'manifest.json'
    if manifest_path.exists() and not args.resume:
        raise SystemExit('Dossier déjà utilisé. --resume pour poursuivre sans écraser les Parquet existants.')
    months=list(month_sequence(args.start,args.end))
    items=[]
    for symbol in args.symbols:
        if not re.fullmatch(r'[A-Z0-9]{4,20}',symbol):raise ValueError('Symbole invalide')
        folder=root/'monthly'/symbol
        folder.mkdir(parents=True,exist_ok=True)
        for month in months:
            dest=folder/f'{month}.parquet'
            if dest.exists():
                if not args.resume:raise SystemExit(f'Existe déjà: {dest}')
                data=pd.read_parquet(dest)
                # Un fichier local déjà écrit ne doit pas être présumé complet.
                if not {'timestamp','close','quote_volume'}.issubset(data.columns):
                    raise ValueError(f'Parquet local incompatible : {dest}')
                state='CACHED'
            else:
                name=f'{symbol}-1h-{month}.zip'
                url=f'{BASE}/{symbol}/1h/{name}'
                try:
                    zip_bytes=request_bytes(url)
                    checksum=request_bytes(url+'.CHECKSUM').decode('utf-8').strip().split()[0]
                    if hashlib.sha256(zip_bytes).hexdigest().lower()!=checksum.lower():
                        raise ValueError(f'{symbol} {month}: SHA256 Binance non concordant')
                    data=parse_month_zip(zip_bytes,symbol,month)
                except Exception as exc:
                    raise RuntimeError(f'Echec {symbol} {month} : {exc}') from exc
                tmp=dest.with_suffix('.parquet.tmp')
                data.to_parquet(tmp,index=False)
                tmp.replace(dest)
                state='DOWNLOADED'
            n=len(data)
            ts=pd.to_datetime(data.timestamp,utc=True)
            gaps=int((ts.diff().dropna()!=pd.Timedelta(hours=1)).sum())
            missing=int(expected_hours(month)-n)
            entry={'symbol':symbol,'month':month,'rows':n,'expected_rows':expected_hours(month),
                   'missing_hours':missing,'non_hourly_transitions':gaps,'state':state,'file':str(dest)}
            items.append(entry)
            print(f'{symbol:9} {month} {state:10} n={n:4} missing={missing:+d} gaps={gaps}',flush=True)
            if missing!=0 or gaps:print('  ATTENTION : archive horaire incomplète; conserver pour audit, ne pas interpoler.')
    pd.DataFrame(items).to_csv(root/'archive_quality.csv',index=False)
    manifest={'version':'v4.1-hourly-diversity','market':'binance_spot','interval':'1h',
              'symbols':list(args.symbols),'first_month':args.start,'last_month':args.end,
              'checksum':'SHA256 sidecar Binance pour chaque nouveau téléchargement',
              'split_policy':'Fenêtre exploratoire pour sélection des actifs; ne pas choisir sur future Validation/Test',
              'notes':'Aucune bougie 1m, aucun apprentissage ni backtest',
              'archives':len(items)}
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Terminé :',root.resolve())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--symbols',nargs='+',default=list(SYMBOLS))
    p.add_argument('--start',default='2021-01')
    p.add_argument('--end',default='2024-12')
    p.add_argument('--output',default='data/evaluation/v41-binance-hourly')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    build(args)

if __name__=='__main__':main()

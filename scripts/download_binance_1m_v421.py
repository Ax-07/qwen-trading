"""V4.2.1 — Télécharge et valide UNE archive mensuelle Binance Spot 1m.

Par défaut janvier 2024 BTCUSDT. Refuse tout écrasement et n'utilise pas les
répertoires V4.1. Aucun entraînement, aucun test scellé.
"""
from __future__ import annotations
import argparse
import calendar
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from download_binance_diversity_v41 import request_bytes, binance_epoch_to_utc, COLS
from prototype_multitimeframe_v42 import validate_minutes

BASE = 'https://data.binance.vision/data/spot/monthly/klines'


def parse_minute_zip(payload: bytes, symbol: str, month: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names = [n for n in z.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX/')]
        if len(names) != 1:
            raise ValueError('Archive sans CSV unique')
        with z.open(names[0]) as src:
            data = pd.read_csv(src, header=None, names=COLS, low_memory=False)
    if data.empty:
        raise ValueError('Archive vide')
    if str(data.iloc[0]['open_time']).lower() in ('open_time', 'opentime'):
        data = data.iloc[1:].copy()
    ts = binance_epoch_to_utc(data['open_time'])
    out = pd.DataFrame({'timestamp': ts})
    for k in ('open', 'high', 'low', 'close', 'volume', 'quote_volume'):
        out[k] = pd.to_numeric(data[k], errors='raise')
    out['trades'] = pd.to_numeric(data['trades'], errors='raise').astype('int64')
    out = validate_minutes(out)
    if (out.timestamp.dt.strftime('%Y-%m') != month).any():
        raise ValueError('Bougie hors du mois demandé')
    if (out.trades < 0).any():
        raise ValueError('Nombre de trades négatif')
    return out.reset_index(drop=True)


def gaps_report(data: pd.DataFrame, month: str) -> tuple[int, list[str]]:
    y, m = map(int, month.split('-'))
    expected = pd.date_range(f'{month}-01', periods=calendar.monthrange(y, m)[1]*1440,
                             freq='min', tz='UTC')
    actual = pd.DatetimeIndex(data.timestamp)
    missing = expected.difference(actual)
    return len(expected), [x.isoformat() for x in missing]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--symbol', default='BTCUSDT')
    p.add_argument('--month', default='2024-01')
    p.add_argument('--output', type=Path, default=Path('data/evaluation/v421-binance-1m'))
    args = p.parse_args()
    if not re.fullmatch(r'[A-Z0-9]{4,20}', args.symbol):
        p.error('Symbole invalide')
    if not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', args.month):
        p.error('Mois invalide : YYYY-MM')
    target = args.output / args.symbol / f'{args.month}.parquet'
    manifest = args.output / args.symbol / f'{args.month}.manifest.json'
    if target.exists() or manifest.exists():
        p.error(f'Fichier existant : {target}. Aucun écrasement autorisé.')
    name = f'{args.symbol}-1m-{args.month}.zip'
    url = f'{BASE}/{args.symbol}/1m/{name}'
    print('Téléchargement :', url, flush=True)
    archive = request_bytes(url)
    checksum = request_bytes(url + '.CHECKSUM').decode('utf-8').strip().split()[0]
    actual = hashlib.sha256(archive).hexdigest()
    if actual.lower() != checksum.lower():
        raise ValueError('SHA256 ZIP non concordant')
    print(f'Archive ZIP vérifiée SHA256 : {len(archive):,} octets', flush=True)
    data = parse_minute_zip(archive, args.symbol, args.month)
    expected, missing = gaps_report(data, args.month)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp.parquet')
    if tmp.exists():
        raise FileExistsError(f'Fichier temporaire déjà présent : {tmp}')
    try:
        data.to_parquet(tmp, index=False)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()
    record = {'version': 'v4.2.1', 'market': 'binance_spot', 'symbol': args.symbol,
              'month': args.month, 'source_url': url, 'sha256_zip': actual,
              'rows': len(data), 'expected_rows': expected, 'missing_minutes': len(missing),
              'missing_timestamps_utc': missing,
              'timestamp_semantics': 'UTC candle open, 1-minute interval',
              'no_interpolation': True, 'source_format': 'Binance monthly 1m CSV ZIP'}
    manifest.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{args.symbol} {args.month} rows={len(data)} expected={expected} missing={len(missing)}')
    print('Parquet :', target.resolve())
    print('Manifest :', manifest.resolve())
    print('Aucune donnée V4.1 modifiée. Aucun entraînement.')


if __name__ == '__main__':
    main()

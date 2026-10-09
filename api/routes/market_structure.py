from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import polars as pl
from fastapi import APIRouter, HTTPException, Query

ROOT = Path(__file__).resolve().parents[2]
STRUCTURE_DIR = ROOT / 'data' / 'evaluation' / 'v31-market-structure'
router = APIRouter(prefix='/api', tags=['market-structure'])

EVENTS = ('pivot_high_confirmed', 'pivot_low_confirmed', 'bos_up', 'bos_down', 'choch_up', 'choch_down')
DatasetName = Literal['train', 'validation', 'test']
Timeframe = Literal['1h', '4h', '1d']


@lru_cache(maxsize=2)
def load_structure(dataset: str) -> pl.DataFrame:
    if dataset == 'test':
        raise HTTPException(status_code=404, detail='Structures V3.1 indisponibles pour Test')
    path = STRUCTURE_DIR / f'{dataset}_structure.parquet'
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f'Exécuter build_market_structure_v31.py : {path}')
    return pl.read_parquet(path).sort('timestamp')


def unix_time(value: datetime | None) -> int | None:
    if value is None:
        return None
    return int(value.timestamp())


@router.get('/market-structure')
def get_market_structure(
    dataset: DatasetName = 'validation',
    timeframe: Timeframe = '1h',
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = Query(default=2000, ge=1, le=5000),
):
    if dataset == 'test':
        return {'dataset': dataset, 'timeframe': timeframe, 'available': False,
                'events': [], 'message': 'Le Test n’a pas été traité pour la V3.1.'}
    for d in (start, end):
        if d is not None and d.tzinfo is None:
            raise HTTPException(status_code=422, detail='Dates avec fuseau horaire requises')
    if start and end and start > end:
        raise HTTPException(status_code=422, detail='start > end')
    df = load_structure(dataset)
    if start is not None:
        df = df.filter(pl.col('timestamp') >= start.astimezone(timezone.utc))
    if end is not None:
        df = df.filter(pl.col('timestamp') <= end.astimezone(timezone.utc))
    cols = df.columns
    flags = [f'{timeframe}_{event}' for event in EVENTS]
    required = [f'{timeframe}_{field}' for field in ('swing_high_at', 'swing_low_at', 'last_swing_high', 'last_swing_low', 'high_structure', 'low_structure', 'available_at')]
    missing = [c for c in [*flags, *required] if c not in cols]
    if missing:
        raise HTTPException(status_code=409, detail=f'Parquet structure incompatible: {missing}')
    df = df.head(limit)
    events = []
    for r in df.iter_rows(named=True):
        confirmed_at = r.get(f'{timeframe}_available_at')
        # pivot-at est le moment de formation, pas le moment où il était connu
        for flag, label, kind in (
            ('pivot_high_confirmed', r[f'{timeframe}_high_structure'], 'high'),
            ('pivot_low_confirmed', r[f'{timeframe}_low_structure'], 'low'),
            ('bos_up', 'BOS↑', 'break_up'),
            ('bos_down', 'BOS↓', 'break_down'),
            ('choch_up', 'CHoCH↑', 'change_up'),
            ('choch_down', 'CHoCH↓', 'change_down'),
        ):
            if r[f'{timeframe}_{flag}'] is not True:
                continue
            is_high = kind == 'high'
            pivot_at = r[f'{timeframe}_swing_high_at' if is_high else f'{timeframe}_swing_low_at'] if kind in ('high','low') else None
            price = r[f'{timeframe}_last_swing_high' if is_high else f'{timeframe}_last_swing_low'] if kind in ('high','low') else None
            # La source fournit le timestamp de début de la bougie 1H.
            # Le marqueur de confirmation est placé sur cette bougie, connue à sa clôture.
            events.append({
                'time': unix_time(r['timestamp']),
                'known_at': confirmed_at.isoformat() if confirmed_at is not None else None,
                'pivot_at': pivot_at.isoformat() if pivot_at is not None else None,
                'label': label if isinstance(label, str) else 'FIRST',
                'kind': kind,
                'price': float(price) if price is not None else None,
            })
    return {'dataset': dataset, 'timeframe': timeframe, 'available': True,
            'count': len(events), 'events': events,
            'notice': 'Les marqueurs indiquent la confirmation, pas une connaissance au pivot initial.'}

from __future__ import annotations
"""ZigZag causal de comparaison (aucun ATR, signal, label, SL ou TP).

Confirmation: clôture de la bougie opposée à au moins `reversal_pct` % de
l'extrême candidat (high pour un sommet, low pour un creux). L'extrême peut
changer jusqu'à cette confirmation. Les pivots ne sont JAMAIS disponibles
à la date de leur extrême, uniquement à confirmation_at.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_market_structure_v31 import normalize, aggregate_completed

TIMEFRAMES = ('1h', '4h', '1d')


def zigzag_events(bars: pd.DataFrame, reversal_pct: float = 1.0) -> pd.DataFrame:
    if not (0 < reversal_pct < 100):
        raise ValueError('reversal_pct doit être entre 0 et 100 (exclus)')
    columns = ['confirmation_at','pivot_at','kind','label','price','reversal_pct']
    if bars.empty:
        return pd.DataFrame(columns=columns)
    fraction = reversal_pct / 100.0
    first = bars.iloc[0]
    hi, lo = float(first.high), float(first.low)
    hi_at = lo_at = bars.index[0]
    state = 'SEEK'  # SEEK -> UP (cherche sommet), DOWN (cherche creux)
    previous_high = previous_low = None
    output = []

    def emit(kind: str, at, origin, price: float):
        nonlocal previous_high, previous_low
        if kind == 'HIGH':
            prior = previous_high
            label = 'FIRST' if prior is None else 'HH' if price > prior else 'LH' if price < prior else 'EH'
            previous_high = price
        else:
            prior = previous_low
            label = 'FIRST' if prior is None else 'HL' if price > prior else 'LL' if price < prior else 'EL'
            previous_low = price
        output.append(dict(confirmation_at=at, pivot_at=origin, kind=kind,
                           label=label, price=price, reversal_pct=reversal_pct))

    for t, row in bars.iloc[1:].iterrows():
        h, l, close = float(row.high), float(row.low), float(row.close)
        if state == 'SEEK':
            # Activation seulement après une clôture sans ambiguïté ; aucune
            # classification intra-barre high/low lorsqu'elles se croisent.
            up = close >= lo * (1 + fraction)
            down = close <= hi * (1 - fraction)
            if up and not down:
                state = 'UP'
                hi, hi_at = h, t
            elif down and not up:
                state = 'DOWN'
                lo, lo_at = l, t
            else:
                if h > hi: hi, hi_at = h, t
                if l < lo: lo, lo_at = l, t
        elif state == 'UP':
            # On confirme un sommet précédent au close; une nouvelle mèche
            # plus haute dans la barre courante est retenue comme extrême.
            if h > hi:
                hi, hi_at = h, t
            if close <= hi * (1 - fraction) and hi_at < t:
                emit('HIGH', t, hi_at, hi)
                state = 'DOWN'
                lo, lo_at = l, t
        else:
            if l < lo:
                lo, lo_at = l, t
            if close >= lo * (1 + fraction) and lo_at < t:
                emit('LOW', t, lo_at, lo)
                state = 'UP'
                hi, hi_at = h, t
    return pd.DataFrame(output, columns=columns)


def events_from_split(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    candle = normalize(df)
    result=[]
    for tf in TIMEFRAMES:
        bars=aggregate_completed(candle,tf)
        events=zigzag_events(bars,pct)
        if events.empty:
            continue
        events.insert(0,'timeframe',tf)
        result.append(events)
    if not result:
        return pd.DataFrame(columns=['timeframe','confirmation_at','pivot_at','kind','label','price','reversal_pct'])
    events=pd.concat(result,ignore_index=True).sort_values(['confirmation_at','timeframe']).reset_index(drop=True)
    assert (events['pivot_at'] < events['confirmation_at']).all()
    return events


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('data/splits'))
    ap.add_argument('--output',type=Path,default=Path('data/evaluation/v31-zigzag-causal'))
    ap.add_argument('--reversal-pct',type=float,default=1.0)
    args=ap.parse_args()
    if not 0<args.reversal_pct<100: ap.error('reversal-pct doit être >0 et <100')
    args.output.mkdir(parents=True,exist_ok=True)
    report={}
    for split in ['train','validation']:
        file=args.input/f'{split}.parquet'
        if not file.is_file(): raise FileNotFoundError(file)
        events=events_from_split(pd.read_parquet(file),args.reversal_pct)
        dest=args.output/f'{split}_zigzag_events.parquet'
        events.to_parquet(dest,index=False)
        counts={tf:int((events.timeframe==tf).sum()) for tf in TIMEFRAMES}
        report[split]={'count':len(events),'by_timeframe':counts,'file':str(dest)}
        print(f'{split.upper():11} total={len(events):4} | '+', '.join(f'{k}: {v}' for k,v in counts.items()))
    manifest={'version':'v31-zigzag-causal-v1','reversal_pct':args.reversal_pct,
      'threshold_selection':'hypothèse initiale fixée, sans optimisation',
      'pivot_clock':'formation pivot_at; disponibilité confirmation_at',
      'source':'splits train/validation SEULEMENT','split_reset':True,
      'notes':'Aucun Test, ATR, entraînement, SL ou TP. Comparaison descriptive avec fractales.',
      'summary':report}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Aucun Test chargé, aucun entraînement, aucun ordre simulé.')

if __name__=='__main__':
    main()

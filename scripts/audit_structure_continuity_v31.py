from __future__ import annotations
"""Audit causal continu Train -> Validation, sans données Test ni labels futurs.

Exécuter depuis la racine du projet :
 python scripts/audit_structure_continuity_v31.py

Les interruptions horaires (notamment purge entre splits) ne sont PAS remplies :
 les niveaux déjà confirmés sont conservés, mais les pivots en cours sont
 réinitialisés sur les segments où il manque des bougies.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_market_structure_v31 import normalize, aggregate_completed, pivot_features
from build_zigzag_causal_v31 import zigzag_events

SPLITS = ('train', 'validation')
METHODS = ('fractal', 'zigzag')
TIMEFRAMES = ('1h', '4h')


def as_utc(x):
    return pd.to_datetime(x, utc=True).dt.as_unit('ns')


def segments(bars, timeframe):
    """Chaque segment ne comprend que des bougies HTF successives complètes."""
    if bars.empty:
        return []
    step = pd.Timedelta(hours=1 if timeframe == '1h' else 4)
    breaks = bars.index.to_series().diff().ne(step).cumsum()
    return [group for _, group in bars.groupby(breaks) if not group.empty]


def fractal_events_one(bars):
    chunks=[]
    for part in segments(bars, '1h' if (bars.index.to_series().diff().dropna().median() == pd.Timedelta(hours=1)) else '4h'):
        state=pivot_features(part, left=2, right=2)
        for kind in ('HIGH','LOW'):
            field='high' if kind=='HIGH' else 'low'
            confirmed=state.loc[state[f'pivot_{field}_confirmed'],
                ['available_at',f'last_swing_{field}',f'swing_{field}_at']].copy()
            if len(confirmed):
                confirmed.columns=['confirmed_at','price','pivot_at']
                confirmed['kind']=kind
                chunks.append(confirmed)
    return pd.concat(chunks,ignore_index=True) if chunks else empty_events()


def empty_events():
    return pd.DataFrame(columns=['confirmed_at','price','pivot_at','kind'])


def produce_events(market, method, timeframe, threshold):
    bars=aggregate_completed(market,timeframe)
    step=pd.Timedelta(hours=1 if timeframe=='1h' else 4)
    pieces=[]
    for part in segments(bars,timeframe):
        if method=='fractal':
            state=pivot_features(part,left=2,right=2)
            for kind in ('HIGH','LOW'):
                k=kind.lower()
                sel=state.loc[state[f'pivot_{k}_confirmed'],
                    ['available_at', f'last_swing_{k}', f'swing_{k}_at']].copy()
                if not sel.empty:
                    sel.columns=['confirmed_at','price','pivot_at']
                    sel['kind']=kind
                    pieces.append(sel)
        else:
            events=zigzag_events(part,threshold)
            if len(events):
                pieces.append(events.rename(columns={'confirmation_at':'confirmed_at'})[
                    ['confirmed_at','price','pivot_at','kind']])
    output=pd.concat(pieces,ignore_index=True) if pieces else empty_events()
    if len(output):
        output['confirmed_at']=as_utc(output['confirmed_at'])
        output['pivot_at']=as_utc(output['pivot_at'])
        output=output.sort_values(['confirmed_at','kind','pivot_at'],kind='stable').reset_index(drop=True)
        if not output.confirmed_at.ge(output.pivot_at).all():
            raise AssertionError('Pivot confirmé avant sa formation')
        if output.duplicated(['kind','confirmed_at']).any():
            raise AssertionError('Double confirmation identique')
        # Aucun segment ne franchit une lacune du calendrier. Les niveaux confirmés
        # antérieurs restent disponibles dans les merge_asof suivants.
    return output


def load_market(split_dir):
    datasets={}
    for split in SPLITS:
        f=split_dir/f'{split}.parquet'
        frame=pd.read_parquet(f,columns=['timestamp','open','high','low','close'])
        frame=normalize(frame)
        frame['split']=split
        datasets[split]=frame
    market=pd.concat(datasets.values(),ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    if market.timestamp.duplicated().any():
        raise ValueError('Bougie dupliquée entre splits')
    if datasets['train'].timestamp.max()>=datasets['validation'].timestamp.min():
        raise ValueError('Train et Validation ne sont pas strictement chronologiques')
    return market,datasets


def load_candidates(base, split):
    df=pd.read_parquet(base/'v3-candidate-events'/f'{split}_candidates.parquet',
                       columns=['event_id','signal_timestamp','known_at','side'])
    df['signal_timestamp']=as_utc(df['signal_timestamp'])
    df['known_at']=as_utc(df['known_at'])
    if df.event_id.isna().any() or df.event_id.duplicated().any():
        raise ValueError('IDs candidats invalides')
    if not df.side.isin(['LONG','SHORT']).all():
        raise ValueError('Direction invalide')
    if not (df.known_at==df.signal_timestamp+pd.Timedelta(hours=1)).all():
        raise ValueError('Horloge candidate incohérente')
    return df


def audit(candidates, candles, events, method, tf, buffer_bps, min_pct, max_pct):
    c=candidates.merge(candles[['timestamp','open']].rename(columns={
        'timestamp':'known_at','open':'entry_open'}),on='known_at',how='left',validate='many_to_one')
    c=c.sort_values('known_at').reset_index(drop=True)
    for kind in ('HIGH','LOW'):
        k=kind.lower()
        e=events.loc[events.kind.eq(kind),['confirmed_at','pivot_at','price']].rename(columns={
            'confirmed_at':f'{k}_confirmed_at','pivot_at':f'{k}_pivot_at','price':f'{k}_price'})
        if e.empty:
            c[f'{k}_confirmed_at']=pd.NaT
            c[f'{k}_pivot_at']=pd.NaT
            c[f'{k}_price']=np.nan
        else:
            c=pd.merge_asof(c.sort_values('known_at'),e.sort_values(f'{k}_confirmed_at'),
                left_on='known_at',right_on=f'{k}_confirmed_at',direction='backward',
                allow_exact_matches=True)
    short=c.side.eq('SHORT')
    c['stop_reference']=np.where(short,'HIGH','LOW')
    c['stop_pivot_price']=np.where(short,c.high_price,c.low_price)
    c['stop_confirmed_at']=as_utc(pd.Series(np.where(short,c.high_confirmed_at,c.low_confirmed_at)))
    c['stop_pivot_at']=as_utc(pd.Series(np.where(short,c.high_pivot_at,c.low_pivot_at)))
    if (c.stop_confirmed_at.dropna()>c.loc[c.stop_confirmed_at.notna(),'known_at']).any():
        raise AssertionError('Fuite temporelle')
    if (c.stop_pivot_at.dropna()>c.loc[c.stop_pivot_at.notna(),'stop_confirmed_at']).any():
        raise AssertionError('Pivot non confirmé')
    delta=buffer_bps/10000.0
    c['stop_price']=np.where(short,c.stop_pivot_price*(1+delta),c.stop_pivot_price*(1-delta))
    c['distance_pct']=np.where(short,(c.stop_price-c.entry_open)/c.entry_open*100,
                              (c.entry_open-c.stop_price)/c.entry_open*100)
    c['age_since_confirmation_h']=(c.known_at-c.stop_confirmed_at).dt.total_seconds()/3600
    c['age_since_formation_h']=(c.known_at-c.stop_pivot_at).dt.total_seconds()/3600
    c['confirmation_delay_h']=(c.stop_confirmed_at-c.stop_pivot_at).dt.total_seconds()/3600
    conditions=[c.entry_open.isna()|c.entry_open.le(0),
                c.stop_confirmed_at.isna()|c.stop_pivot_price.isna(),
                c.distance_pct.le(0),c.distance_pct.lt(min_pct),c.distance_pct.gt(max_pct)]
    labels=['ENTRY_OUTSIDE_SPLIT','NO_CONFIRMED_PIVOT','STOP_ALREADY_INVALID','TOO_CLOSE','TOO_FAR']
    c['status']=np.select(conditions,labels,default='VALID_REFERENCE')
    c['method']=method
    c['timeframe']=tf
    return c[['method','timeframe','event_id','signal_timestamp','known_at','side',
      'entry_open','high_price','low_price','high_confirmed_at','low_confirmed_at',
      'stop_reference','stop_pivot_price','stop_confirmed_at','stop_pivot_at',
      'stop_price','distance_pct','age_since_confirmation_h','age_since_formation_h',
      'confirmation_delay_h','status']]


def stat(values):
    s=pd.to_numeric(values,errors='coerce').dropna()
    return {'count':int(len(s)), **{f'p{n}':float(s.quantile(n/100)) if len(s) else None
        for n in (10,25,50,75,90,95)}}


def summary(data):
    results=[]
    for (method,tf,side),part in data.groupby(['method','timeframe','side'],sort=True):
        plausible=part[part.status.isin(['VALID_REFERENCE','TOO_CLOSE','TOO_FAR'])]
        known=part[part.stop_confirmed_at.notna()]
        results.append({'method':method,'timeframe':tf,'side':side,'candidates':int(len(part)),
          'status':{str(k):int(v) for k,v in part.status.value_counts().items()},
          'distance_pct':stat(plausible.distance_pct),
          'age_confirmation_h':stat(known.age_since_confirmation_h),
          'age_formation_h':stat(known.age_since_formation_h),
          'delay_formation_to_confirmation_h':stat(known.confirmation_delay_h)})
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--splits',type=Path,default=Path('data/splits'))
    ap.add_argument('--base',type=Path,default=Path('data/evaluation'))
    ap.add_argument('--output',type=Path,default=Path('data/evaluation/v31-continuity-audit'))
    ap.add_argument('--reversal-pct',type=float,default=1.0)
    ap.add_argument('--buffer-bps',type=float,default=5.)
    ap.add_argument('--min-distance-pct',type=float,default=.1)
    ap.add_argument('--max-distance-pct',type=float,default=5.)
    args=ap.parse_args()
    if not 0<args.reversal_pct<100 or args.buffer_bps<0 or not 0<=args.min_distance_pct<args.max_distance_pct:
        ap.error('Seuils invalides')
    market,by_split=load_market(args.splits)
    all_events={}
    for method in METHODS:
        for tf in TIMEFRAMES:
            all_events[(method,tf)]=produce_events(market,method,tf,args.reversal_pct)
            print(f'PIVOTS {method:8s} {tf}: {len(all_events[(method,tf)])}')
    outputs={}
    report={}
    for split in SPLITS:
        candidates=load_candidates(args.base,split)
        results=[]
        for method in METHODS:
            for tf in TIMEFRAMES:
                r=audit(candidates,by_split[split],all_events[(method,tf)],method,tf,
                    args.buffer_bps,args.min_distance_pct,args.max_distance_pct)
                if len(r)!=len(candidates) or r.event_id.duplicated().any():
                    raise AssertionError('Perte ou duplication de candidats')
                results.append(r)
        combined=pd.concat(results,ignore_index=True)
        outputs[split]=combined
        report[split]=summary(combined)
        for row in report[split]:
            status=row['status']
            p50=row['distance_pct']['p50']
            print(f"{split.upper():10s} {row['method']:8s} {row['timeframe']} {row['side']:5s}: "
                 f"n={row['candidates']} valid={status.get('VALID_REFERENCE',0)} "
                 f"far={status.get('TOO_FAR',0)} missing={status.get('NO_CONFIRMED_PIVOT',0)} "
                 f"median_dist={p50:.3f}%" if p50 is not None else 'Aucune distance')
    args.output.mkdir(parents=True,exist_ok=True)
    for split,frame in outputs.items():
        frame.to_parquet(args.output/f'{split}_audit.parquet',index=False)
    manifest={'version':'v31-continuity-descriptive-1','source_splits':list(SPLITS),
       'no_test':True,'no_future_labels':True,'no_backtest':True,
       'continuity':'Train et Validation concaténés; état confirmé retenu au franchissement de la purge; '
          'pivots en formation réinitialisés sur chaque lacune horaire/HTF; aucune bougie manquante synthétisée',
       'caveat':'Les périodes purgées restent inconnues. Ne pas considérer cette continuité comme équivalente '
          'à des données de marché sans lacune.',
       'event_time':'confirmed_at est la clôture de bougie; référence disponible pour entrée à ce même instant',
       'parameters':{'reversal_pct':args.reversal_pct,'buffer_bps':args.buffer_bps,
         'min_distance_pct':args.min_distance_pct,'max_distance_pct':args.max_distance_pct},
       'event_counts':{f'{m}_{tf}':len(v) for (m,tf),v in all_events.items()},
       'by_split':report}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Sorties:',args.output.resolve())
    print('Aucun Test lu, aucune transaction simulée, aucun entraînement.')

if __name__=='__main__':
    main()

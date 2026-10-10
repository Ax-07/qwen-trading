from __future__ import annotations
"""Qwen Trading V4.1: audit local des historiques 1h, sans réseau ni entraînement."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

TRAIN = ('BTCUSDT','ETHUSDT','DOGEUSDT','TRXUSDT','SOLUSDT','ATOMUSDT','BNBUSDT','LINKUSDT')
HELD = ('XRPUSDT','LTCUSDT','AAVEUSDT','XLMUSDT')
ALL = set(TRAIN + HELD)


def validate_month(df: pd.DataFrame, month: str, symbol: str) -> pd.DataFrame:
    required = {'timestamp','open','high','low','close','volume','quote_volume'}
    if not required.issubset(df.columns):
        raise ValueError(f'{symbol} {month}: colonnes absentes {sorted(required - set(df.columns))}')
    d=df.copy()
    d['timestamp']=pd.to_datetime(d['timestamp'],utc=True,errors='raise')
    if d.timestamp.isna().any() or not d.timestamp.is_monotonic_increasing or d.timestamp.duplicated().any():
        raise ValueError(f'{symbol} {month}: dates invalides / doublons / désordre')
    if not (d.timestamp.dt.strftime('%Y-%m')==month).all():
        raise ValueError(f'{symbol} {month}: horodatage hors mois')
    if not (d.timestamp.dt.minute.eq(0) & d.timestamp.dt.second.eq(0) & d.timestamp.dt.microsecond.eq(0)).all():
        raise ValueError(f'{symbol} {month}: timestamp non horaire')
    if not (d[['open','high','low','close']].gt(0).all().all() and d[['volume','quote_volume']].ge(0).all().all()):
        raise ValueError(f'{symbol} {month}: valeurs invalides')
    if not ((d.high >= d[['open','low','close']].max(axis=1)) & (d.low <= d[['open','high','close']].min(axis=1))).all():
        raise ValueError(f'{symbol} {month}: OHLC incohérent')
    return d


def load(root: Path, start: str, end: str):
    datasets={}; missing={}; issues=[]
    expected = pd.date_range(f'{start}-01', (pd.Period(end, freq='M') + 1).start_time, freq='h', tz='UTC', inclusive='left')
    months=pd.period_range(start,end,freq='M').strftime('%Y-%m').tolist()
    folders=sorted(p for p in (root/'monthly').iterdir() if p.is_dir())
    if not folders: raise ValueError('Aucun répertoire actif dans monthly/')
    for folder in folders:
        frames=[]
        for month in months:
            path=folder / f'{month}.parquet'
            if not path.is_file():
                raise FileNotFoundError(f'Archive locale absente: {path}')
            frames.append(validate_month(pd.read_parquet(path),month,folder.name))
        d=pd.concat(frames,ignore_index=True).set_index('timestamp').sort_index()
        if d.index.duplicated().any(): raise ValueError(f'{folder.name}: doublons inter-mois')
        extras=d.index.difference(expected)
        if len(extras): raise ValueError(f'{folder.name}: dates hors plage')
        missing[folder.name]=expected.difference(d.index)
        datasets[folder.name]=d.reindex(expected)
        print(f'{folder.name:9} {len(d):6} heures / {len(expected)} ; absentes={len(missing[folder.name])}',flush=True)
    return datasets,missing,expected


def hourly_returns(d):
    # Pas de calcul de rendement sur une lacune: absence = NaN, jamais de ffill.
    c=d['close']
    return np.log(c/c.shift(1)).replace([np.inf,-np.inf],np.nan)


def year_profiles(datasets):
    out=[]
    for symbol,d in datasets.items():
        r=hourly_returns(d)
        for year, block in d.groupby(d.index.year):
            ry=r.loc[block.index].dropna()
            v=block.quote_volume.dropna()
            out.append(dict(symbol=symbol,year=int(year),hours_available=int(block.close.notna().sum()),
                hours_missing=int(block.close.isna().sum()),valid_returns=len(ry),
                annualized_vol_pct=float(ry.std()*np.sqrt(365*24)*100) if len(ry)>1 else np.nan,
                p99_abs_hourly_return_pct=float(ry.abs().quantile(.99)*100) if len(ry) else np.nan,
                median_hourly_quote_volume_usdt=float(v.median()) if len(v) else np.nan))
    return pd.DataFrame(out)


def yearly_correlations(returns: pd.DataFrame, min_overlap: int):
    rows=[]
    for year,block in returns.groupby(returns.index.year):
        corr=block.corr(min_periods=min_overlap)
        overlap=block.notna().astype(int).T.dot(block.notna().astype(int))
        for i,left in enumerate(returns.columns):
            for right in returns.columns[i+1:]:
                rows.append(dict(year=int(year),symbol_a=left,symbol_b=right,
                    common_hours=int(overlap.loc[left,right]),correlation=float(corr.loc[left,right])))
    return pd.DataFrame(rows)


def rolling_pairs(returns: pd.DataFrame, window_days=(30,90), min_coverage=0.98):
    # min_periods évite d'exclure toutes les fenêtres quand 1 heure manque.
    # pairwise rolling avec couverture du nombre de paires, sans alignement implicite.
    rows=[]
    names=list(returns.columns)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            pair=returns[[a,b]]
            valid=pair.notna().all(axis=1).astype(int)
            for days in window_days:
                width=days*24
                min_n=int(np.ceil(width*min_coverage))
                count=valid.rolling(width,min_periods=width).sum()
                r=pair[a].rolling(width,min_periods=min_n).corr(pair[b]).where(count>=min_n).dropna()
                rows.append(dict(symbol_a=a,symbol_b=b,window_days=days,
                    min_pair_observations=min_n,valid_windows=int(len(r)),
                    median_correlation=float(r.median()) if len(r) else np.nan,
                    p10_correlation=float(r.quantile(.1)) if len(r) else np.nan,
                    p90_correlation=float(r.quantile(.9)) if len(r) else np.nan))
    return pd.DataFrame(rows)


def gap_details(missing):
    rows=[]
    for sym,idx in missing.items():
        for stamp in idx:
            rows.append(dict(symbol=sym,timestamp_utc=stamp.isoformat()))
    return pd.DataFrame(rows,columns=['symbol','timestamp_utc'])


def gap_summary(missing):
    syms=sorted(missing)
    common=set(missing[syms[0]])
    union=set()
    for sym in syms:
        common &= set(missing[sym]); union |= set(missing[sym])
    return sorted(common),sorted(union)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='data/evaluation/v41-binance-hourly')
    p.add_argument('--output',default='data/evaluation/v41-universe-audit')
    p.add_argument('--start',default='2021-01')
    p.add_argument('--end',default='2024-12')
    p.add_argument('--min-year-overlap',type=int,default=1000)
    p.add_argument('--min-coverage',type=float,default=.98)
    a=p.parse_args()
    if not (0<a.min_coverage<=1): p.error('--min-coverage doit être entre 0 et 1')
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f'Sortie déjà remplie, rien écrasé: {root} (choisir un autre --output)')
    datasets,missing,expected=load(Path(a.input),a.start,a.end)
    if set(datasets)!=ALL:
        raise ValueError(f'Univers inattendu; manquants={sorted(ALL-set(datasets))}; supplémentaires={sorted(set(datasets)-ALL)}')
    returns=pd.DataFrame({s:hourly_returns(d) for s,d in datasets.items()})
    annual=year_profiles(datasets)
    annual_corr=yearly_correlations(returns,a.min_year_overlap)
    rolling=rolling_pairs(returns,min_coverage=a.min_coverage)
    gaps=gap_details(missing)
    common,union=gap_summary(missing)
    roles=pd.DataFrame([dict(symbol=s,proposed_role='TRAIN' if s in TRAIN else 'HELD_OUT_ASSET',status='PROVISIONAL_NOT_FROZEN') for s in sorted(datasets)])
    root.mkdir(parents=True,exist_ok=True)
    annual.to_csv(root/'yearly_asset_profiles.csv',index=False,float_format='%.8f')
    annual_corr.to_csv(root/'yearly_pair_correlations.csv',index=False,float_format='%.8f')
    rolling.to_csv(root/'rolling_pair_correlations.csv',index=False,float_format='%.8f')
    gaps.to_csv(root/'missing_hour_timestamps.csv',index=False)
    roles.to_csv(root/'proposed_asset_roles.csv',index=False)
    manifest=dict(version='v4.1-universe-audit',input=a.input,range=[a.start,a.end],
        universe_symbols=sorted(datasets),expected_hours_per_asset=len(expected),
        missing_hours_per_asset={s:len(missing[s]) for s in sorted(missing)},
        shared_missing_hours=len(common),unique_missing_timestamps=len(union),
        all_gaps_identical=all(set(missing[s])==set(common) for s in missing),
        rolling_window_min_coverage=a.min_coverage,
        notes=['Décision TRAIN / HELD_OUT provisoire, à figer avant tout entraînement.',
               'Fenêtre de sélection exploratoire 2021-2024; ne pas utiliser de Test futur pour choisir.',
               'Volume historique médian ≠ spread/profondeur de marché; ni mesure des frais.',
               'Fenêtres glissantes se chevauchent: leur nombre ne représente pas des observations indépendantes.',
               'Les lacunes communes prouvent la coïncidence des timestamps, pas leur cause.'])
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n=== LACUNES ===')
    print(f'Horodatages manquants communs : {len(common)} ; union : {len(union)} ; identiques : {manifest["all_gaps_identical"]}')
    print('Premiers timestamps communs :',*[t.isoformat() for t in common[:20]],sep='\n  ')
    print('\n=== PROFILS ANNUELS ===')
    print(annual[['symbol','year','annualized_vol_pct','median_hourly_quote_volume_usdt']].round(2).to_string(index=False))
    print('\n=== ROLES PROVISOIRES ===')
    print(roles.to_string(index=False))
    print('\nRapport :',root.resolve())
    print('Aucun accès réseau. Pas de modification des données source. Pas de Test ni entraînement.')

if __name__=='__main__': main()

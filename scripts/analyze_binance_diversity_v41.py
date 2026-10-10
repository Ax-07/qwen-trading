from __future__ import annotations
"""V4.1 - Profil de diversité ex-ante sur données Spot Binance 1h vérifiées.

Produit des CSV de corrélations et de volatilité; PAS de sélection automatique d'actifs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_symbol(folder:Path):
    files=sorted(folder.glob('*.parquet'))
    if not files:raise ValueError(f'Aucun Parquet dans {folder}')
    d=pd.concat((pd.read_parquet(p) for p in files),ignore_index=True)
    d['timestamp']=pd.to_datetime(d.timestamp,utc=True)
    d=d.sort_values('timestamp')
    if d.timestamp.duplicated().any():raise ValueError(f'Doublons : {folder.name}')
    if (d.close<=0).any():raise ValueError(f'Prix invalide : {folder.name}')
    d=d.set_index('timestamp')
    # Les absences restent NaN; pas de forward-fill des prix ou des rendements.
    hourly=pd.date_range(d.index.min(),d.index.max(),freq='1h',tz='UTC')
    d=d.reindex(hourly)
    ret=np.log(d.close/d.close.shift(1))
    ret[~(d.close.notna() & d.close.shift(1).notna())]=np.nan
    return d,ret


def calculate(input_folder:Path,min_overlap:int=1000):
    folders=sorted(p for p in (input_folder/'monthly').iterdir() if p.is_dir())
    if len(folders)<2:raise ValueError('Au moins deux actifs sont nécessaires')
    returns={}; prices={}; metrics=[]
    for folder in folders:
        d,r=load_symbol(folder)
        returns[folder.name]=r
        prices[folder.name]=d
        valid=r.dropna()
        q=d.quote_volume.dropna()
        if len(valid)<min_overlap:raise ValueError(f'{folder.name}: trop peu de rendements valides ({len(valid)})')
        # Volatilité annualisée descriptive (crypto 24/7), non prédictive.
        vol=float(valid.std()*np.sqrt(24*365)*100)
        abs_p99=float(valid.abs().quantile(.99)*100)
        df_close=d.close
        efficiency=(
            np.log(df_close/df_close.shift(24*7)).abs() /
            np.log(df_close/df_close.shift(1)).abs().rolling(24*7,min_periods=24*7).sum()
        )
        metrics.append({'symbol':folder.name,'available_hours':int(d.close.notna().sum()),
                        'missing_hours':int(d.close.isna().sum()),
                        'annualized_vol_pct':vol,'p99_abs_hourly_return_pct':abs_p99,
                        'median_hourly_quote_volume_usdt':float(q.median()),
                        'median_7d_directional_efficiency':float(efficiency.replace([np.inf,-np.inf],np.nan).median())})
    matrix=pd.DataFrame(returns).sort_index()
    # Corrélation sur heures UTC communes, pas de remplissage implicite.
    counts=matrix.notna().astype(int).T.dot(matrix.notna().astype(int))
    corr=matrix.corr(min_periods=min_overlap)
    rows=[]
    if 'BTCUSDT' in matrix:
        b=matrix['BTCUSDT']
        for sym in matrix:
            paired=pd.concat([b,matrix[sym]],axis=1)
            for days in (30,90):
                window=days*24
                moving=paired.iloc[:,0].rolling(window,min_periods=window).corr(paired.iloc[:,1]).dropna()
                rows.append({'symbol':sym,'window_days':days,
                             'median_rolling_corr_to_btc':float(moving.median()) if len(moving) else np.nan,
                             'p10_corr':float(moving.quantile(.1)) if len(moving) else np.nan,
                             'p90_corr':float(moving.quantile(.9)) if len(moving) else np.nan,
                             'windows_count':len(moving)})
    return pd.DataFrame(metrics),corr,counts,pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='data/evaluation/v41-binance-hourly')
    p.add_argument('--output',default='data/evaluation/v41-diversity-report')
    p.add_argument('--min-overlap',type=int,default=1000)
    a=p.parse_args()
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise SystemExit('Sortie déjà remplie : utiliser un nouveau --output')
    metrics,corr,counts,rolling=calculate(Path(a.input),a.min_overlap)
    root.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(root/'asset_profiles.csv',index=False,float_format='%.6f')
    corr.to_csv(root/'hourly_return_correlations.csv',float_format='%.6f')
    counts.to_csv(root/'pairwise_overlap_hours.csv')
    rolling.to_csv(root/'btc_rolling_correlations.csv',index=False,float_format='%.6f')
    (root/'manifest.json').write_text(json.dumps({'version':'v4.1-diversity-report','source':a.input,
         'symbol_count':len(metrics),'min_pairwise_hours':a.min_overlap,
         'warning':'Volume coté ≠ liquidité exécutable. Corrélations non causales pour sélection hors échantillon; recherche exploratoire uniquement.'},indent=2),encoding='utf-8')
    print(metrics[['symbol','available_hours','missing_hours','annualized_vol_pct','median_hourly_quote_volume_usdt']].to_string(index=False))
    print('Résultats :',root.resolve())

if __name__=='__main__':main()

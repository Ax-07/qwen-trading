from __future__ import annotations

"""Audit descriptif de couverture des regimes V3 (Train / Validation uniquement).

Usage: python scripts/audit_candidate_regime_overlap_v3.py
Ne fait ni prediction, ni optimisation, ni backtest. Aucune lecture du Test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from evaluate_candidate_regression_v3 import FEATURES, load_split

DEFAULT_IN = Path('data/evaluation/v3-candidate-events')
DEFAULT_OUT = Path('data/evaluation/v3-candidate-regime-overlap')
SEED = 20261009
# Liste definie a priori : variables de regime relatives ou bornees, pas de prix absolus.
REGIME_FEATURES = [
    'atr14_pct', 'volatility_24h', 'volatility_72h', 'rsi14',
    '1d_rsi14', 'distance_ema200_pct', 'ema50_200_spread_pct',
    'volume_ratio', 'trend_alignment', '1d_atr14_pct',
]
JOINT_FEATURES = ['atr14_pct', 'rsi14', 'trend_alignment']
MIN_TRAIN_CELL = 20
MIN_VALID_CELL = 10
BOOTSTRAPS = 600


def options():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=DEFAULT_IN)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def validate(train, val):
    if train.empty or val.empty:
        raise ValueError('Split vide')
    if set(train.event_id) & set(val.event_id):
        raise ValueError('IDs partages Train / Validation')
    if train.signal_timestamp.max() >= val.signal_timestamp.min():
        raise ValueError('Chronologie des splits incorrecte')
    if train.label_available_at.max() >= val.known_at.min():
        raise ValueError('Labels Train pas disponibles avant Validation')
    missing = set(REGIME_FEATURES) - set(FEATURES)
    if missing:
        raise ValueError(f'Features non autorisees par la liste causale : {sorted(missing)}')
    for side in ('LONG', 'SHORT'):
        if (train.side == side).sum() < 50 or (val.side == side).sum() < 20:
            raise ValueError(f'Effectifs insuffisants : {side}')
    for df in (train, val):
        if not np.isfinite(df[REGIME_FEATURES + ['net_return']].to_numpy(float)).all():
            raise ValueError('Valeurs non finies dans les donnees')


def edges_from_train(x):
    cuts = np.quantile(np.asarray(x, float), [0, 1/3, 2/3, 1])
    return np.unique(cuts)


def assign_bins(values, edges):
    if len(edges) < 3:
        # Feature discrete et constante : regime unique plutot qu'un bin fictif.
        return np.array(['ONE'] * len(values), dtype=object)
    inner = edges[1:-1]
    ids = np.searchsorted(inner, np.asarray(values, float), side='right')
    return np.array([f'B{k+1}' for k in ids], dtype=object)


def drift_rows(train, val):
    result = []
    for side in ('LONG', 'SHORT'):
        t = train.loc[train.side == side]
        v = val.loc[val.side == side]
        for feature in REGIME_FEATURES:
            a, b = t[feature].to_numpy(float), v[feature].to_numpy(float)
            p01, p99 = np.quantile(a, [0.01, 0.99])
            low_high = float(np.mean((b < p01) | (b > p99)))
            std = a.std(ddof=1)
            smd = float((b.mean() - a.mean()) / std) if std > 0 else np.nan
            # Bins fixes Train, include valeurs hors bornes; un PSI avec lissage,
            # ce qui evite les infinis si un bin de Validation est vide.
            edges = np.unique(np.quantile(a, np.linspace(0, 1, 6)))
            bin_a = assign_bins(a, edges)
            bin_b = assign_bins(b, edges)
            labels = sorted(set(bin_a) | set(bin_b))
            na = np.array([(bin_a == k).sum() for k in labels], float)
            nb = np.array([(bin_b == k).sum() for k in labels], float)
            pa = (na + 0.5) / (len(a) + 0.5*len(labels))
            pb = (nb + 0.5) / (len(b) + 0.5*len(labels))
            psi = float(np.sum((pb-pa)*np.log(pb/pa)))
            result.append(dict(side=side, feature=feature, n_train=len(a),
                n_validation=len(b), train_mean=a.mean(), validation_mean=b.mean(),
                standardized_mean_difference=smd, psi_train_bins=psi,
                val_outside_train_p01_p99_pct=low_high*100,
                train_p01=p01, train_p99=p99))
    return pd.DataFrame(result)


def attach_regimes(train, val):
    tr, va = train.copy(), val.copy()
    for side in ('LONG', 'SHORT'):
        idx_t = tr.side == side
        idx_v = va.side == side
        for feature in JOINT_FEATURES:
            if feature == 'trend_alignment':
                # Variables discretes conservent leur valeur; eviter quantiles degeneres.
                tr.loc[idx_t, 'bin_'+feature] = tr.loc[idx_t, feature].astype(int).astype(str).to_numpy()
                va.loc[idx_v, 'bin_'+feature] = va.loc[idx_v, feature].astype(int).astype(str).to_numpy()
            else:
                edges = edges_from_train(tr.loc[idx_t, feature])
                tr.loc[idx_t, 'bin_'+feature] = assign_bins(tr.loc[idx_t, feature], edges)
                va.loc[idx_v, 'bin_'+feature] = assign_bins(va.loc[idx_v, feature], edges)
    names = ['bin_'+f for f in JOINT_FEATURES]
    tr['regime_id'] = tr[names].astype(str).agg('|'.join, axis=1)
    va['regime_id'] = va[names].astype(str).agg('|'.join, axis=1)
    counts = tr.groupby(['side','regime_id']).size().rename('n_train_regime')
    for df in (tr, va):
        df['n_train_regime'] = pd.MultiIndex.from_frame(df[['side','regime_id']]).map(counts).fillna(0).astype(int)
        df['represented'] = df['n_train_regime'] >= MIN_TRAIN_CELL
        df['week_utc'] = df.signal_timestamp.dt.tz_convert('UTC').dt.strftime('%G-W%V')
    return tr, va


def nearest_neighbor_coverage(train, val):
    """Distance robuste : Val->Train et Train->Train (plus proche autre).
    Indicateurs de support seulement; ne sont pas des probabilites.
    """
    rows = []
    for side in ('LONG', 'SHORT'):
        a = train.loc[train.side == side, REGIME_FEATURES].to_numpy(float)
        b = val.loc[val.side == side, REGIME_FEATURES].to_numpy(float)
        scaler = RobustScaler().fit(a)
        a, b = scaler.transform(a), scaler.transform(b)
        nn = NearestNeighbors(n_neighbors=2).fit(a)
        d_train = nn.kneighbors(a)[0][:, 1]
        d_val = nn.kneighbors(b, n_neighbors=1)[0][:, 0]
        q95 = float(np.quantile(d_train, .95))
        rows.append(dict(side=side, train_n=len(a), validation_n=len(b),
            train_loo_nn_median=float(np.median(d_train)),
            train_loo_nn_p95=q95,
            val_nn_median=float(np.median(d_val)),
            val_nn_p95=float(np.quantile(d_val, .95)),
            val_beyond_train_loo_p95_pct=float(np.mean(d_val > q95)*100)))
    return pd.DataFrame(rows)


def block_bootstrap_mean_delta(train_sub, val_sub, seed):
    """Reechantillonne des semaines entieres, jamais des evenements independants.
    L'IC est approximatif; 12h chevauchantes, petit nombre de semaines possible.
    """
    x = [z.net_return.to_numpy(float) for _,z in train_sub.groupby('week_utc')]
    y = [z.net_return.to_numpy(float) for _,z in val_sub.groupby('week_utc')]
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan, len(x), len(y)
    rng = np.random.default_rng(seed)
    diffs = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        ix = rng.integers(0, len(x), len(x))
        iy = rng.integers(0, len(y), len(y))
        xa = np.concatenate([x[j] for j in ix])
        ya = np.concatenate([y[j] for j in iy])
        diffs[i] = (ya.mean()-xa.mean())*100
    low, high = np.quantile(diffs, [.025,.975])
    return float(low), float(high), len(x), len(y)


def regime_comparison(train, val):
    rows=[]
    keys = sorted(set(zip(train.side,train.regime_id)) | set(zip(val.side,val.regime_id)))
    for i,(side,regime) in enumerate(keys):
        a=train.loc[(train.side==side)&(train.regime_id==regime)]
        b=val.loc[(val.side==side)&(val.regime_id==regime)]
        enough=len(a)>=MIN_TRAIN_CELL and len(b)>=MIN_VALID_CELL
        lo,hi,wt,wv = block_bootstrap_mean_delta(a,b,SEED+i) if enough else (np.nan,np.nan,0,0)
        rows.append(dict(side=side,regime_id=regime,train_n=len(a),validation_n=len(b),
            represented_in_train=len(a)>=MIN_TRAIN_CELL,comparable=enough,
            train_mean_net_pct=float(a.net_return.mean()*100) if len(a) else np.nan,
            validation_mean_net_pct=float(b.net_return.mean()*100) if len(b) else np.nan,
            delta_val_minus_train_pct=float((b.net_return.mean()-a.net_return.mean())*100) if len(a) and len(b) else np.nan,
            bootstrap_delta_ci_low_pct=lo,bootstrap_delta_ci_high_pct=hi,
            train_weeks=wt,validation_weeks=wv,
            train_tp_rate=float(a.event_outcome.eq('TP').mean()) if len(a) else np.nan,
            val_tp_rate=float(b.event_outcome.eq('TP').mean()) if len(b) else np.nan))
    return pd.DataFrame(rows)


def main():
    args=options()
    raw_t,t=load_split(args.input_dir,'train')
    raw_v,v=load_split(args.input_dir,'validation')
    validate(t,v)
    drift=drift_rows(t,v)
    tr,va=attach_regimes(t,v)
    comparison=regime_comparison(tr,va)
    coverage=nearest_neighbor_coverage(t,v)
    summary=[]
    for side in ('LONG','SHORT'):
        a=tr.loc[tr.side==side]
        b=va.loc[va.side==side]
        comparable_ids=set(comparison.loc[(comparison.side==side)&comparison.comparable,'regime_id'])
        mask=b.regime_id.isin(comparable_ids)
        summary.append(dict(side=side,train_n=len(a),validation_n=len(b),
          validation_represented_n=int(b.represented.sum()),
          validation_represented_pct=float(b.represented.mean()*100),
          validation_comparable_n=int(mask.sum()),
          validation_comparable_pct=float(mask.mean()*100),
          train_mean_net_pct=float(a.net_return.mean()*100),
          validation_mean_net_pct=float(b.net_return.mean()*100),
          validation_comparable_mean_net_pct=float(b.loc[mask,'net_return'].mean()*100) if mask.any() else np.nan,
          validation_unrepresented_mean_net_pct=float(b.loc[~b.represented,'net_return'].mean()*100) if (~b.represented).any() else np.nan,
          train_regimes=int(a.regime_id.nunique()),validation_regimes=int(b.regime_id.nunique()),
          comparable_regimes=len(comparable_ids)))
    summary=pd.DataFrame(summary)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,table in [('feature_drift_relative',drift),('regime_comparison',comparison),
                       ('coverage_summary',summary),('nearest_neighbor_coverage',coverage)]:
        table.to_csv(args.output_dir/f'{name}.csv',index=False)
    (args.output_dir/'protocol.json').write_text(json.dumps(dict(
        source='V3 candidate events',features=REGIME_FEATURES,joint_regime_features=JOINT_FEATURES,
        regime_train_min=MIN_TRAIN_CELL,regime_validation_min=MIN_VALID_CELL,
        bootstrap_week_blocks=BOOTSTRAPS,random_seed=SEED,
        split_counts={'train':len(t),'validation':len(v)},
        excluded_invalid={'train':len(raw_t)-len(t),'validation':len(raw_v)-len(v)},
        no_test=True,no_model=True,no_tuning=True,no_backtest=True,
        limitations=['Univariate drift cannot identify cause',
          'Nearest-neighbor distances are not calibrated probabilities',
          'Joint strata condition on three variables only; hidden confounding remains',
          'Weekly bootstrap is approximate with limited weeks and dependent events',
          'Validation has already been examined in previous V3 analyses']
    ),ensure_ascii=False,indent=2),encoding='utf-8')
    print('='*78)
    print('QWEN V3 - REGIME OVERLAP | TRAIN + VALIDATION UNIQUEMENT')
    print('='*78)
    print(f'Train valides={len(t)} | Validation valides={len(v)}; invalides={len(raw_t)-len(t)} / {len(raw_v)-len(v)}')
    print('\nCOUVERTURE DES REGIMES (ATR14%, RSI14, trend_alignment; bins du Train)')
    print(summary.round(4).to_string(index=False))
    print('\nCOUVERTURE MULTIVARIEE (distance au plus proche voisin, echelle robuste Train)')
    print(coverage.round(4).to_string(index=False))
    print('\nDERIVE DES FEATURES RELATIVES - 12 PLUS FORTES DISTANCES STANDARDISEES')
    tmp=drift.assign(abs_smd=drift.standardized_mean_difference.abs()).sort_values('abs_smd',ascending=False)
    print(tmp[['side','feature','standardized_mean_difference','psi_train_bins','val_outside_train_p01_p99_pct']].head(12).round(4).to_string(index=False))
    print('\nREGIMES COMPARABLES (n_train>=20, n_validation>=10) - 15 PLUS GROS GROUPES VALIDATION')
    c=comparison.loc[comparison.comparable].sort_values('validation_n',ascending=False)
    if len(c):
        print(c[['side','regime_id','train_n','validation_n','train_mean_net_pct','validation_mean_net_pct','delta_val_minus_train_pct','bootstrap_delta_ci_low_pct','bootstrap_delta_ci_high_pct','train_weeks','validation_weeks']].head(15).round(4).to_string(index=False))
    else:
        print('AUCUN : groupes trop petits pour comparer; ne pas inferer la stabilite.')
    print(f'\nFichiers : {args.output_dir.resolve()}')
    print('Aucun Test lu, aucun modele entraine, aucun tuning. IC par blocs hebdomadaires approximatifs.')


if __name__=='__main__':
    main()

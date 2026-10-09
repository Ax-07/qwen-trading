from __future__ import annotations

"""Audit descriptif du changement de distribution V3, sans entrainement.

Lit uniquement train_candidates/labels et validation_candidates/labels.
Compare: compositions, variables causales, et relation feature -> issue future.
Les candidats se chevauchent: aucune statistique d'inference / causalite.
Execute depuis la racine du projet: python scripts/audit_candidate_distribution_shift_v3.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_candidate_regression_v3 import load_split
from build_candidate_events_v3 import FEATURES

DEFAULT_INPUT = Path('data/evaluation/v3-candidate-events')
DEFAULT_OUTPUT = Path('data/evaluation/v3-candidate-distribution-shift')
KEY_FEATURES = [
    'atr14_pct', 'volatility_24h', 'volatility_72h', 'rsi14',
    'volume_ratio', 'distance_ema200_pct', 'ema50_200_spread_pct',
    'trend_alignment', 'trend_1h', '4h_trend', '1d_trend',
    '1d_rsi14', '1d_atr14_pct', 'distance_high_72h_pct',
]
OUTCOMES = ['TP', 'SL', 'TIME']
MIN_BIN_N = 20


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def check_integrity(train_raw, train, val_raw, val):
    if not train['event_id'].is_unique or not val['event_id'].is_unique:
        raise ValueError('event_id duplique')
    if set(train['event_id']) & set(val['event_id']):
        raise ValueError('evenements Train / Validation communs')
    if train_raw['signal_timestamp'].max() >= val_raw['signal_timestamp'].min():
        raise ValueError('splits non chronologiques')
    if train['label_available_at'].max() >= val['known_at'].min():
        raise ValueError('labels Train non disponibles avant Validation')
    if not train['event_outcome'].isin(OUTCOMES).all() or not val['event_outcome'].isin(OUTCOMES).all():
        raise ValueError('issue invalide')
    if not np.isfinite(train['net_return']).all() or not np.isfinite(val['net_return']).all():
        raise ValueError('rendement non fini')
    if not (train['label_available_at'] >= train['known_at']).all() or not (val['label_available_at'] >= val['known_at']).all():
        raise ValueError('label disponible avant signal')
    if not train['side'].isin(['LONG','SHORT']).all() or not val['side'].isin(['LONG','SHORT']).all():
        raise ValueError('direction inattendue')
    for df in [train, val]:
        if df[FEATURES].isna().any().any():
            raise ValueError('features manquantes')
        if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
            raise ValueError('features non finies')


def periodize(train, val):
    # Quartiles par POSITION CHRONOLOGIQUE, sans utiliser aucun label.
    tr = train.sort_values('signal_timestamp').copy().reset_index(drop=True)
    va = val.sort_values('signal_timestamp').copy().reset_index(drop=True)
    tr['period'] = ['TRAIN_Q' + str(min(4, 1 + (4 * i // len(tr)))) for i in range(len(tr))]
    va['period'] = 'VALIDATION'
    return pd.concat([tr, va], ignore_index=True)


def outcome_summary(data):
    rows = []
    for (period, side), df in data.groupby(['period', 'side'], sort=False):
        vals = df['net_return'].to_numpy(float) * 100
        row = {
            'period': period, 'side': side, 'n': len(df),
            'first_signal': str(df['signal_timestamp'].min()),
            'last_signal': str(df['signal_timestamp'].max()),
            'net_mean_pct': float(np.mean(vals)),
            'net_median_pct': float(np.median(vals)),
            'net_std_pct': float(np.std(vals, ddof=0)),
        }
        for name in OUTCOMES:
            row['rate_' + name.lower()] = float(df['event_outcome'].eq(name).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def sample_summaries(data):
    rows = []
    for (period, side), df in data.groupby(['period', 'side'], sort=False):
        for feature in KEY_FEATURES:
            x = df[feature].astype(float).to_numpy()
            rows.append({'period':period,'side':side,'feature':feature,'n':len(x),
                         'mean':float(np.mean(x)), 'std':float(np.std(x)),
                         'p10':float(np.percentile(x,10)), 'median':float(np.median(x)),
                         'p90':float(np.percentile(x,90))})
    return pd.DataFrame(rows)


def psi_from_train(train_values, compare_values):
    # Bornes train uniquement. Constant/quantiles identiques => PSI indefini.
    cut = np.unique(np.quantile(train_values, np.linspace(0, 1, 11)))
    if len(cut) < 3:
        return float('nan'), int(len(cut)-1)
    edges = np.r_[-np.inf, cut[1:-1], np.inf]
    ref = np.histogram(train_values, bins=edges)[0] / len(train_values)
    obs = np.histogram(compare_values, bins=edges)[0] / len(compare_values)
    # Lissage descriptif symetrique, evite log(0); PSI sensible aux petits n.
    eps = 1e-5
    p = (ref + eps) / (1 + eps * len(ref))
    q = (obs + eps) / (1 + eps * len(obs))
    return float(np.sum((q - p) * np.log(q / p))), int(len(ref))


def drift_summaries(train, val):
    result = []
    for side in ['LONG','SHORT']:
        a = train.loc[train['side'].eq(side)]
        b = val.loc[val['side'].eq(side)]
        for f in FEATURES + ['bias_score']:
            x, y = a[f].to_numpy(float), b[f].to_numpy(float)
            mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
            stdev = float(np.std(x))
            psi, k = psi_from_train(x,y)
            result.append({'side':side,'feature':f,'train_n':len(x),'validation_n':len(y),
                           'train_mean':mean_x,'validation_mean':mean_y,
                           'train_std':stdev,
                           'standardized_mean_difference':((mean_y-mean_x)/stdev if stdev > 1e-12 else float('nan')),
                           'psi_train_bins':psi,'psi_bins':k})
    return pd.DataFrame(result)


def conditional_bins(train, val):
    # Relation feature -> rendement/TP a FEATURE comparable, par direction.
    # Chaque seuil provient exclusivement de Train de cette direction.
    # Tables descriptives, pas d'entrainement de modele ni de nouveau filtre.
    rows = []
    for side in ['LONG','SHORT']:
        tr = train.loc[train['side'].eq(side)]
        va = val.loc[val['side'].eq(side)]
        for f in KEY_FEATURES:
            values = tr[f].to_numpy(float)
            cuts = np.unique(np.quantile(values, [0, 1/3, 2/3, 1]))
            if len(cuts) < 4:
                continue  # categoriel discret / quantiles egaux : analyse non adaptee.
            edges = np.r_[-np.inf, cuts[1:-1], np.inf]
            for period, block in [('TRAIN',tr),('VALIDATION',va)]:
                groups = pd.cut(block[f], bins=edges, labels=False, include_lowest=True)
                for bucket in range(3):
                    d = block.loc[groups.eq(bucket)]
                    n = len(d)
                    rows.append({'side':side,'feature':f,'reference':'TRAIN_TERTILES',
                                 'period':period,'bin':bucket+1,
                                 'lower_bound':float(edges[bucket]),'upper_bound':float(edges[bucket+1]),
                                 'n':n,'enough_observations':n>=MIN_BIN_N,
                                 'mean_net_pct':float(d['net_return'].mean()*100) if n else float('nan'),
                                 'tp_rate':float(d['event_outcome'].eq('TP').mean()) if n else float('nan'),
                                 'sl_rate':float(d['event_outcome'].eq('SL').mean()) if n else float('nan'),
                                 'time_rate':float(d['event_outcome'].eq('TIME').mean()) if n else float('nan')})
    return pd.DataFrame(rows)


def main():
    args = args_parser()
    train_raw, train = load_split(args.input_dir, 'train')
    val_raw, val = load_split(args.input_dir, 'validation')
    check_integrity(train_raw,train,val_raw,val)
    data = periodize(train,val)
    composition = outcome_summary(data)
    feature_periods = sample_summaries(data)
    drift = drift_summaries(train,val)
    conditional = conditional_bins(train,val)
    output = args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    composition.to_csv(output/'period_outcomes.csv',index=False)
    feature_periods.to_csv(output/'period_feature_statistics.csv',index=False)
    drift.to_csv(output/'feature_drift.csv',index=False)
    conditional.to_csv(output/'conditional_outcomes.csv',index=False)
    settings = {'analysis':'v3_candidate_distribution_shift_v1',
                'splits_read':['train','validation'],'test_read':False,
                'train_valid_n':len(train),'validation_valid_n':len(val),
                'train_invalid_n':len(train_raw)-len(train),
                'validation_invalid_n':len(val_raw)-len(val),
                'periodization':'four consecutive equal-row-count train quartiles + validation',
                'psi_reference':'train side-specific deciles, 1e-5 pseudocount',
                'conditional_bins':'train side-specific feature tertiles, only strictly distinct quantiles',
                'min_bin_n_to_interpret':MIN_BIN_N,
                'interpretation':'descriptive, dependent overlapping events; not a trade backtest or statistical significance test'}
    (output/'protocol.json').write_text(json.dumps(settings,indent=2,ensure_ascii=False),encoding='utf-8')
    print('='*79)
    print('QWEN V3 - DISTRIBUTION SHIFT | TRAIN + VALIDATION UNIQUEMENT')
    print('='*79)
    print(f"Valides Train={len(train)}, Validation={len(val)} | Invalides {len(train_raw)-len(train)} / {len(val_raw)-len(val)}")
    print('\nISSUES ET RENDEMENTS PAR PERIODE / DIRECTION')
    print(composition[['period','side','n','rate_tp','rate_sl','rate_time','net_mean_pct']].to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print('\n10 PLUS GRANDS ECARTS STANDARDISES DES FEATURES (VAL vs TRAIN)')
    top = drift.dropna(subset=['standardized_mean_difference']).assign(abs_smd=lambda d:d['standardized_mean_difference'].abs()).sort_values('abs_smd',ascending=False).head(10)
    print(top[['side','feature','standardized_mean_difference','psi_train_bins']].to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print('\nRELATIONS CONDITIONNELLES (bin de Train 1 vs 3; n>=20 pour chaque periode)')
    eligible = conditional.query('enough_observations').pivot_table(index=['side','feature','bin'],columns='period',values='mean_net_pct')
    if not eligible.empty:
        eligible = eligible.dropna().reset_index()
        eligible = eligible[eligible['bin'].isin([1,3])]
        eligible['delta_val_minus_train_pct']=eligible['VALIDATION']-eligible['TRAIN']
        print(eligible.head(20).to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    else:
        print('Aucun bin suffisamment peuple.')
    print(f'\nFichiers: {output.resolve()}')
    print('Aucun Test lu. Aucun entrainement. Aucun tuning. Pas un backtest.')


if __name__ == '__main__':
    main()

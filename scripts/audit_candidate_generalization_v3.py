from __future__ import annotations

"""Audit V3 Ridge : Train in-sample vs Validation, sans Test ni optimisation.

Reprend exactement les données, features et alpha de evaluate_candidate_regression_v3.py.
Les corrélations et découpages temporels sont descriptifs; les événements se
chevauchent et le Train est évalué in-sample (optimiste).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_candidate_regression_v3 import (
    ROOT, RIDGE_ALPHA, TARGET, features_matrix, load_split,
    regression_metrics,
)

OUT = Path('data/evaluation/v3-candidate-generalization')


def args_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, default=ROOT)
    ap.add_argument('--output-dir', type=Path, default=OUT)
    return ap.parse_args()


def evaluate(df, split, model, means):
    out = df[['event_id', 'signal_timestamp', 'known_at', 'side', TARGET]].copy()
    out['split'] = split
    out['prior'] = df['side'].map(means).to_numpy(dtype=float)
    out['ridge'] = model.predict(features_matrix(df))
    out['ridge_error'] = out['ridge'] - out[TARGET]
    return out


def metrics_by(data, keys):
    rows = []
    for names, part in data.groupby(keys, sort=True, observed=True):
        names = names if isinstance(names, tuple) else (names,)
        for model_name, column in [('DIRECTION_PRIOR', 'prior'), ('RIDGE', 'ridge')]:
            result = regression_metrics(part[TARGET].to_numpy(float), part[column].to_numpy(float))
            rows.append({**dict(zip(keys, names)), 'model': model_name,
                         **result,
                         'observed_mean_pct': 100 * float(part[TARGET].mean()),
                         'predicted_mean_pct': 100 * float(part[column].mean())})
    return pd.DataFrame(rows)


def feature_associations(df, split, feature_cols):
    rows = []
    for side, part in df.groupby('side', observed=True, sort=True):
        y = part[TARGET]
        for feature in feature_cols:
            values = pd.to_numeric(part[feature], errors='coerce')
            if values.nunique(dropna=True) < 3 or y.nunique(dropna=True) < 3:
                pearson = spearman = np.nan
            else:
                pearson = values.corr(y, method='pearson')
                spearman = values.corr(y, method='spearman')
            rows.append({'split': split, 'side': side, 'feature': feature,
                         'n': int(values.notna().sum()),
                         'pearson': pearson, 'spearman': spearman})
    return pd.DataFrame(rows)


def rank_diagnostic(df):
    # Quantiles within each split+side, only when predictions truly vary.
    rows = []
    for (split, side), part in df.groupby(['split', 'side'], sort=True):
        if part['ridge'].nunique() < 4:
            continue
        sorted_part = part.sort_values(['ridge', 'signal_timestamp'], ascending=[False, True])
        for fraction in (0.10, 0.25, 0.50, 1.00):
            top = sorted_part.iloc[:max(1, int(np.ceil(len(sorted_part) * fraction)))]
            rows.append({'split': split, 'side': side, 'top_fraction': fraction,
                         'n': len(top), 'realized_mean_net_pct': 100 * top[TARGET].mean(),
                         'predicted_mean_net_pct': 100 * top['ridge'].mean()})
    return pd.DataFrame(rows)


def main():
    a = args_parser()
    train_raw, train = load_split(a.input_dir, 'train')
    val_raw, val = load_split(a.input_dir, 'validation')
    if not len(train) or not len(val):
        raise ValueError('Train ou Validation vide')
    if set(train['event_id']) & set(val['event_id']):
        raise ValueError('IDs Train/Validation chevauchants')
    if train_raw['signal_timestamp'].max() >= val_raw['signal_timestamp'].min():
        raise ValueError('Splits non chronologiques')
    if train['label_available_at'].max() >= val['known_at'].min():
        raise ValueError('Labels du Train disponibles après début Validation')

    x_train, x_val = features_matrix(train), features_matrix(val)
    if list(x_train.columns) != list(x_val.columns):
        raise ValueError('Features incohérentes')
    if not np.isfinite(x_train.to_numpy(float)).all() or not np.isfinite(x_val.to_numpy(float)).all():
        raise ValueError('Features non finies')
    means = train.groupby('side')[TARGET].mean().to_dict()
    if set(means) != {'LONG', 'SHORT'}:
        raise ValueError('Deux directions obligatoires sur Train')
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=RIDGE_ALPHA)),
    ])
    model.fit(x_train, train[TARGET].to_numpy(float))
    predictions = pd.concat([
        evaluate(train, 'TRAIN_IN_SAMPLE', model, means),
        evaluate(val, 'VALIDATION', model, means),
    ], ignore_index=True)
    predictions['week_utc'] = predictions['signal_timestamp'].dt.to_period('W-SUN').dt.start_time.astype(str)
    overall = metrics_by(predictions, ['split', 'side'])
    weekly = metrics_by(predictions, ['split', 'side', 'week_utc'])
    feature_cols = [c for c in x_train.columns if c != 'candidate_is_short']
    associations = pd.concat([
        feature_associations(train, 'TRAIN_IN_SAMPLE', feature_cols),
        feature_associations(val, 'VALIDATION', feature_cols),
    ], ignore_index=True)
    paired = associations.pivot_table(index=['side', 'feature'], columns='split',
                                      values='spearman').reset_index()
    paired['sign_flip'] = (paired.get('TRAIN_IN_SAMPLE') * paired.get('VALIDATION') < 0)
    ranking = rank_diagnostic(predictions)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(a.output_dir / 'train_validation_metrics.csv', index=False)
    weekly.to_csv(a.output_dir / 'weekly_errors.csv', index=False)
    associations.to_csv(a.output_dir / 'feature_correlations.csv', index=False)
    paired.to_csv(a.output_dir / 'feature_stability.csv', index=False)
    ranking.to_csv(a.output_dir / 'ridge_ranking.csv', index=False)
    predictions.to_parquet(a.output_dir / 'train_validation_predictions.parquet', index=False)
    (a.output_dir / 'protocol.json').write_text(json.dumps({
        'model': 'Ridge', 'alpha': RIDGE_ALPHA, 'features': list(x_train.columns),
        'train_n': len(train), 'validation_n': len(val),
        'train_invalid': len(train_raw) - len(train),
        'validation_invalid': len(val_raw) - len(val),
        'train_evaluation': 'in-sample (optimistic)',
        'validation_evaluation': 'chronological holdout',
        'test_used': False, 'hyperparameter_search': False,
        'caveats': ['overlapping event horizons', 'descriptive correlations',
                    'not a backtest', 'multiple comparisons, no significance tests'],
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    print('=' * 78)
    print('QWEN V3 - AUDIT GENERALISATION RIDGE - TRAIN vs VALIDATION')
    print('=' * 78)
    print(f'Train={len(train)}, Validation={len(val)} | Alpha Ridge={RIDGE_ALPHA}')
    print('\nMETRIQUES - TRAIN IN-SAMPLE OPTIMISTE vs VALIDATION')
    print(overall[['split', 'side', 'model', 'n', 'mae_pct', 'rmse_pct', 'bias_pct',
                   'pearson']].to_string(index=False, float_format=lambda x: f'{x:+.4f}'))
    print('\nERREURS HEBDOMADAIRES DE RIDGE - VALIDATION')
    print(weekly.loc[(weekly['split'] == 'VALIDATION') & (weekly['model'] == 'RIDGE'),
                     ['side', 'week_utc', 'n', 'mae_pct', 'rmse_pct', 'bias_pct']]
          .to_string(index=False, float_format=lambda x: f'{x:+.4f}'))
    print('\nSTABILITE DES ASSOCIATIONS SPEARMAN - 10 PLUS GRANDS |CORR TRAIN|')
    for side in ('LONG', 'SHORT'):
        view = paired.loc[paired['side'] == side].copy()
        view['strength'] = view['TRAIN_IN_SAMPLE'].abs()
        print('\n', side)
        print(view.nlargest(10, 'strength')[['feature', 'TRAIN_IN_SAMPLE',
              'VALIDATION', 'sign_flip']].to_string(index=False,
                                                   float_format=lambda x: f'{x:+.3f}'))
    print('\nCLASSEMENT RIDGE (CANDIDATS CHEVAUCHANTS, PAS UN BACKTEST)')
    print(ranking.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))
    print(f'\nCSV / Parquet : {a.output_dir.resolve()}')
    print('Aucun Test chargé. Aucun tuning. Corrélations exploratoires non causales.')


if __name__ == '__main__':
    main()

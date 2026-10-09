from __future__ import annotations

"""Qwen V3: baseline Ridge pour le rendement net de candidats causaux.

Train uniquement pour l'ajustement. Validation uniquement pour l'évaluation.
Aucun accès au Test. Aucun réglage d'hyperparamètres à partir de Validation.
Ce n'est PAS un backtest : les candidats horaires peuvent se chevaucher.
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

from build_candidate_events_v3 import FEATURES, BLOCKED_PATTERNS

ROOT = Path('data/evaluation/v3-candidate-events')
OUT = Path('data/evaluation/v3-candidate-regression')
TARGET = 'net_return'
VALID_OUTCOMES = {'TP', 'SL', 'TIME'}
RIDGE_ALPHA = 100.0  # Choix fixé a priori, pas de recherche sur Validation.


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    return parser.parse_args()


def load_split(root: Path, split: str):
    x = pd.read_parquet(root / f'{split}_candidates.parquet')
    y = pd.read_parquet(root / f'{split}_labels.parquet')
    for name, data in [('X', x), ('y', y)]:
        if 'event_id' not in data or data['event_id'].isna().any() or not data['event_id'].is_unique:
            raise ValueError(f'{split}: IDs {name} manquants / dupliqués')
    required = {'event_id', 'signal_timestamp', 'known_at', 'side', 'bias_score', *FEATURES}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f'{split}: features absentes: {sorted(missing)}')
    if any(any(pattern in c for pattern in BLOCKED_PATTERNS) for c in FEATURES):
        raise ValueError('Features contenant une information future')
    if not {'event_id', 'side', 'event_outcome', TARGET, 'label_available_at'} <= set(y.columns):
        raise ValueError(f'{split}: champs du label manquants')
    # La jointure reste strictement 1:1; aucune colonne future n'est injectée dans les predictors.
    joined = x.merge(
        y[['event_id', 'side', 'event_outcome', TARGET, 'label_available_at']],
        on='event_id', how='left', validate='one_to_one', indicator=True,
        suffixes=('', '_label'),
    )
    if len(joined) != len(x) or not joined['_merge'].eq('both').all():
        raise ValueError(f'{split}: jointure incomplète')
    if not joined['side'].eq(joined['side_label']).all():
        raise ValueError(f'{split}: directions incohérentes')
    joined = joined.drop(columns=['side_label', '_merge'])
    for c in ('signal_timestamp', 'known_at', 'label_available_at'):
        joined[c] = pd.to_datetime(joined[c], utc=True)
    if joined['signal_timestamp'].duplicated().any():
        raise ValueError(f'{split}: deux candidats sur une même bougie')
    if not joined['known_at'].eq(joined['signal_timestamp'] + pd.Timedelta(hours=1)).all():
        raise ValueError(f'{split}: known_at incorrect')
    valid = joined['event_outcome'].isin(VALID_OUTCOMES)
    if not joined.loc[~valid, 'event_outcome'].eq('DATA_INVALID').all():
        raise ValueError(f'{split}: issue inconnue')
    clean = joined.loc[valid].copy()
    if clean[TARGET].isna().any() or not np.isfinite(clean[TARGET].to_numpy(dtype=float)).all():
        raise ValueError(f'{split}: rendements nets invalides')
    if clean['label_available_at'].isna().any() or not (clean['label_available_at'] >= clean['known_at']).all():
        raise ValueError(f'{split}: disponibilité future des labels incorrecte')
    if clean[FEATURES + ['bias_score']].isna().any().any():
        raise ValueError(f'{split}: prédicteurs manquants')
    if not clean['side'].isin(['LONG', 'SHORT']).all():
        raise ValueError(f'{split}: direction inconnue')
    clean = clean.sort_values('signal_timestamp').reset_index(drop=True)
    return joined, clean


def features_matrix(df):
    """Inclut les indicateurs observables et le sens candidat, jamais les outcomes."""
    data = df[FEATURES + ['bias_score']].astype(float).copy()
    data['candidate_is_short'] = (df['side'] == 'SHORT').astype(float)
    return data


def regression_metrics(y, pred):
    err = pred - y
    std_y, std_p = float(np.std(y)), float(np.std(pred))
    return {
        'n': int(len(y)),
        'mae_pct': round(float(np.mean(np.abs(err))) * 100, 6),
        'rmse_pct': round(float(np.sqrt(np.mean(err**2))) * 100, 6),
        'bias_pct': round(float(np.mean(err)) * 100, 6),
        'pearson': (float(np.corrcoef(y, pred)[0, 1])
                    if std_y > 0 and std_p > 1e-12 else None),
    }


def grouped_diagnostics(df, key, columns=('model', 'side')):
    rows = []
    for values, sub in df.groupby(list(columns), sort=True, observed=True):
        values = values if isinstance(values, tuple) else (values,)
        y = sub[TARGET].to_numpy(float)
        p = sub[key].to_numpy(float)
        rows.append({**dict(zip(columns, values)), **regression_metrics(y, p),
                     'observed_mean_pct': float(np.mean(y) * 100),
                     'predicted_mean_pct': float(np.mean(p) * 100)})
    return rows


def main():
    args = arguments()
    train_raw, train = load_split(args.input_dir, 'train')
    val_raw, val = load_split(args.input_dir, 'validation')
    if len(train) == 0 or len(val) == 0:
        raise RuntimeError('Split vide')
    if set(train['event_id']) & set(val['event_id']):
        raise RuntimeError('Chevauchement des IDs Train / Validation')
    if train_raw['signal_timestamp'].max() >= val_raw['signal_timestamp'].min():
        raise RuntimeError('Splits non chronologiques')
    if train['label_available_at'].max() >= val['known_at'].min():
        raise RuntimeError('Fuite des labels Train sur la période Validation')

    x_train, x_val = features_matrix(train), features_matrix(val)
    if not np.isfinite(x_train.to_numpy()).all() or not np.isfinite(x_val.to_numpy()).all():
        raise ValueError('Features non finies')
    y_train = train[TARGET].to_numpy(float)
    y_val = val[TARGET].to_numpy(float)
    # Prior par direction; entraîné strictement sur Train.
    means = train.groupby('side')[TARGET].mean().to_dict()
    if set(means) != {'LONG', 'SHORT'}:
        raise ValueError('Train ne contient pas les deux directions')
    prior = val['side'].map(means).to_numpy(float)
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=RIDGE_ALPHA)),
    ])
    model.fit(x_train, y_train)
    ridge = model.predict(x_val)
    if not np.isfinite(ridge).all():
        raise ValueError('Prédictions non finies')

    output = val[['event_id', 'signal_timestamp', 'known_at', 'side', 'event_outcome', TARGET]].copy()
    output['prediction_prior'] = prior
    output['prediction_ridge'] = ridge
    output['ridge_positive'] = ridge > 0
    report = []
    for name, prediction in [('DIRECTION_PRIOR', prior), ('RIDGE', ridge)]:
        for side in ('ALL', 'LONG', 'SHORT'):
            mask = np.ones(len(val), dtype=bool) if side == 'ALL' else val['side'].eq(side).to_numpy()
            report.append({'model': name, 'side': side,
                           **regression_metrics(y_val[mask], prediction[mask]),
                           'observed_mean_pct': round(float(np.mean(y_val[mask])) * 100, 6),
                           'predicted_mean_pct': round(float(np.mean(prediction[mask])) * 100, 6)})
    metric_table = pd.DataFrame(report)

    # Les groupes sont des tranches de classement, non des signaux tradés.
    ranking_rows = []
    for side in ('LONG', 'SHORT'):
        subset = output.loc[output['side'] == side].copy()
        for name, pred_col in [('DIRECTION_PRIOR', 'prediction_prior'), ('RIDGE', 'prediction_ridge')]:
            ranked = subset.sort_values([pred_col, 'signal_timestamp'], ascending=[False, True]).reset_index(drop=True)
            n = len(ranked)
            for fraction in (0.10, 0.25, 0.50, 1.00):
                k = max(1, int(np.ceil(n * fraction)))
                top = ranked.iloc[:k]
                ranking_rows.append({
                    'model': name, 'side': side, 'top_fraction': fraction,
                    'candidate_count': k,
                    'mean_net_return_pct': round(float(top[TARGET].mean() * 100), 6),
                    'median_net_return_pct': round(float(top[TARGET].median() * 100), 6),
                    'tp_rate_pct': round(float((top['event_outcome'] == 'TP').mean() * 100), 4),
                    'sl_rate_pct': round(float((top['event_outcome'] == 'SL').mean() * 100), 4),
                })
    ranking = pd.DataFrame(ranking_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_table.to_csv(args.output_dir / 'validation_metrics.csv', index=False)
    ranking.to_csv(args.output_dir / 'validation_ranking.csv', index=False)
    output.to_parquet(args.output_dir / 'validation_predictions.parquet', index=False)
    (args.output_dir / 'protocol.json').write_text(json.dumps({
        'model': 'ridge', 'alpha_fixed_before_validation': RIDGE_ALPHA,
        'features': list(x_train.columns),
        'train_count': int(len(train)), 'validation_count': int(len(val)),
        'train_invalid': int(len(train_raw) - len(train)),
        'validation_invalid': int(len(val_raw) - len(val)),
        'label': TARGET, 'comparison': 'per-side mean from Train',
        'backtest': False, 'test_loaded': False,
        'caveat': 'overlapping candidate events; descriptive ranking only',
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    print('=' * 75)
    print('V3 CANDIDATE REGRESSION | TRAIN / VALIDATION UNIQUEMENT')
    print('=' * 75)
    print(f'Train valides: {len(train)} | Validation valides: {len(val)}')
    print(f'Invalides Train: {len(train_raw)-len(train)} | Validation: {len(val_raw)-len(val)}')
    print(f'Moyennes Train LONG/SHORT (%): {means["LONG"]*100:+.4f} / {means["SHORT"]*100:+.4f}')
    print('\nPREDICTION DU RENDEMENT NET — plus faible MAE/RMSE = meilleur')
    print(metric_table.to_string(index=False))
    print('\nCLASSEMENT DES CANDIDATS — descriptif, PAS un backtest')
    print(ranking.to_string(index=False))
    print(f'\nFichiers: {args.output_dir.resolve()}')
    print('Aucun Test lu. Aucun réglage sur Validation. Aucune hypothèse de rentabilité.')


if __name__ == '__main__':
    main()

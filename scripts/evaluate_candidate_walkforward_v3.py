from __future__ import annotations

"""Audit walk-forward Ridge V3, uniquement dans le TRAIN des événements V3.

Fenêtres d'apprentissage croissantes, 5 fenêtres futures non chevauchantes,
purge de 24 h ET contrôle de disponibilité réelle des labels.
Aucun Test ni Validation chargé. Les candidats peuvent se chevaucher :
ceci mesure la prédiction hors échantillon, ce n'est pas un backtest.
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
    RIDGE_ALPHA, TARGET, features_matrix, load_split, regression_metrics,
)

INPUT_DIR = Path('data/evaluation/v3-candidate-events')
OUTPUT_DIR = Path('data/evaluation/v3-candidate-walkforward')
PURGE_HOURS = 24
INITIAL_TRAIN_FRACTION = 0.50
N_FOLDS = 5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=INPUT_DIR)
    p.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def metrics_row(fold, split, side, model_name, group, prediction):
    y = group[TARGET].to_numpy(dtype=float)
    stats = regression_metrics(y, np.asarray(prediction, dtype=float))
    return {
        'fold': fold, 'window': split, 'side': side, 'model': model_name,
        **stats,
        'realized_mean_pct': round(float(y.mean() * 100), 6),
        'predicted_mean_pct': round(float(np.mean(prediction) * 100), 6),
    }


def main():
    args = parse_args()
    raw, df = load_split(args.input_dir, 'train')
    df = df.sort_values('signal_timestamp').reset_index(drop=True)
    if len(df) < 500:
        raise RuntimeError('Pas assez de candidats pour un walk-forward.')
    if df['signal_timestamp'].duplicated().any():
        raise RuntimeError('Candidats avec timestamps dupliqués.')
    if not (df['label_available_at'] >= df['known_at']).all():
        raise RuntimeError('Disponibilité incorrecte des labels.')

    # Découpage déterministe chronologique sur les candidats valides.
    n = len(df)
    start_idx = int(np.floor(n * INITIAL_TRAIN_FRACTION))
    boundaries = np.linspace(start_idx, n, N_FOLDS + 1, dtype=int)
    purge = pd.Timedelta(hours=PURGE_HOURS)
    all_predictions = []
    all_metrics = []
    folds = []

    print('=' * 78)
    print('QWEN V3 - WALK-FORWARD RIDGE | TRAIN UNIQUEMENT')
    print('=' * 78)
    print(f'Candidats TRAIN valides={n}; invalides={len(raw)-n}')
    print(f'Folds={N_FOLDS}; debut evaluation={INITIAL_TRAIN_FRACTION:.0%}; '
          f'purge={PURGE_HOURS}h; Ridge alpha={RIDGE_ALPHA}')

    for fold in range(N_FOLDS):
        left, right = int(boundaries[fold]), int(boundaries[fold + 1])
        future = df.iloc[left:right].copy()
        if future.empty:
            raise RuntimeError(f'Fold {fold + 1} vide.')
        # Evite que le dernier label du passé utilise des informations de la
        # fenêtre future, avec un embargo calendaire au-delà des 12 bougies.
        first_known = future['known_at'].min()
        allowed = (df.iloc[:left]['known_at'] < first_known - purge)
        allowed &= (df.iloc[:left]['label_available_at'] < first_known)
        past = df.iloc[:left].loc[allowed].copy()
        if len(past) < 100 or set(past['side']) != {'LONG', 'SHORT'}:
            raise RuntimeError(f'Fold {fold + 1}: historique insuffisant.')
        if past['known_at'].max() >= first_known - purge:
            raise RuntimeError('Purge 24h non respectée.')
        if past['label_available_at'].max() >= first_known:
            raise RuntimeError('Fuite temporelle : label connu trop tard.')
        if set(past['event_id']) & set(future['event_id']):
            raise RuntimeError('Identifiants partagés train/futur.')
        x_past, x_future = features_matrix(past), features_matrix(future)
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=RIDGE_ALPHA)),
        ])
        model.fit(x_past, past[TARGET].to_numpy(float))
        ridge = model.predict(x_future)
        means = past.groupby('side')[TARGET].mean().to_dict()
        prior = future['side'].map(means).to_numpy(float)
        if not np.isfinite(ridge).all() or not np.isfinite(prior).all():
            raise RuntimeError('Prédictions invalides.')
        predictions = future[[
            'event_id', 'signal_timestamp', 'known_at', 'side',
            'event_outcome', TARGET,
        ]].copy()
        predictions['fold'] = fold + 1
        predictions['prediction_prior'] = prior
        predictions['prediction_ridge'] = ridge
        all_predictions.append(predictions)

        for side in ('ALL', 'LONG', 'SHORT'):
            mask = (np.ones(len(future), dtype=bool) if side == 'ALL'
                    else future['side'].eq(side).to_numpy())
            if not mask.any():
                continue
            group = future.iloc[np.flatnonzero(mask)]
            for label, values in (
                ('DIRECTION_PRIOR', prior[mask]),
                ('RIDGE', ridge[mask]),
            ):
                all_metrics.append(metrics_row(
                    fold + 1, 'OOS', side, label, group, values
                ))
        folds.append({
            'fold': fold + 1, 'train_candidates': len(past),
            'evaluation_candidates': len(future),
            'train_first_signal': past['signal_timestamp'].min().isoformat(),
            'train_last_signal': past['signal_timestamp'].max().isoformat(),
            'last_train_label_known_at': past['label_available_at'].max().isoformat(),
            'evaluation_first_known_at': first_known.isoformat(),
            'evaluation_last_signal': future['signal_timestamp'].max().isoformat(),
            'prior_long_pct': means['LONG'] * 100,
            'prior_short_pct': means['SHORT'] * 100,
        })
        row_prior = next(x for x in all_metrics if x['fold'] == fold + 1
                         and x['side'] == 'ALL' and x['model'] == 'DIRECTION_PRIOR')
        row_ridge = next(x for x in all_metrics if x['fold'] == fold + 1
                         and x['side'] == 'ALL' and x['model'] == 'RIDGE')
        print(f'Fold {fold+1}: train={len(past):4d} eval={len(future):3d} '
              f'| RMSE prior={row_prior["rmse_pct"]:.4f}% '
              f'Ridge={row_ridge["rmse_pct"]:.4f}% '
              f'| delta={row_ridge["rmse_pct"]-row_prior["rmse_pct"]:+.4f} pt')

    pred = pd.concat(all_predictions, ignore_index=True)
    if pred['event_id'].duplicated().any():
        raise RuntimeError('Un événement est évalué dans plusieurs folds.')
    rows = pd.DataFrame(all_metrics)
    global_rows = []
    for side in ('ALL', 'LONG', 'SHORT'):
        sel = pred if side == 'ALL' else pred.loc[pred['side'] == side]
        for label, col in (
            ('DIRECTION_PRIOR', 'prediction_prior'),
            ('RIDGE', 'prediction_ridge'),
        ):
            global_rows.append(metrics_row(
                0, 'ALL_OOS', side, label, sel, sel[col].to_numpy(float)
            ))
    global_df = pd.DataFrame(global_rows)
    weekly = []
    # Conversion UTC explicite avant suppression du fuseau pour Period.
    pred['week_utc'] = (pred['signal_timestamp'].dt.tz_convert('UTC')
                        .dt.tz_localize(None).dt.to_period('W-SUN')
                        .dt.start_time.astype(str))
    for (week, side), group in pred.groupby(['week_utc', 'side'], sort=True):
        for label, col in (
            ('DIRECTION_PRIOR', 'prediction_prior'),
            ('RIDGE', 'prediction_ridge'),
        ):
            weekly.append({'week_utc': week, **metrics_row(
                0, 'WEEKLY_OOS', side, label, group,
                group[col].to_numpy(float)
            )})

    ranking = []
    for side in ('LONG', 'SHORT'):
        group = pred.loc[pred['side'] == side].copy()
        sorted_group = group.sort_values(
            ['prediction_ridge', 'signal_timestamp'], ascending=[False, True]
        )
        for fraction in (0.1, 0.25, 0.5, 1.0):
            selected = sorted_group.iloc[:max(1, int(np.ceil(len(group)*fraction)))]
            ranking.append({
                'side': side, 'top_fraction': fraction, 'n': len(selected),
                'realized_mean_net_pct': selected[TARGET].mean()*100,
                'predicted_mean_net_pct': selected['prediction_ridge'].mean()*100,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds).to_csv(args.output_dir / 'fold_boundaries.csv', index=False)
    rows.to_csv(args.output_dir / 'fold_metrics.csv', index=False)
    global_df.to_csv(args.output_dir / 'aggregate_metrics.csv', index=False)
    pd.DataFrame(weekly).to_csv(args.output_dir / 'weekly_metrics.csv', index=False)
    pd.DataFrame(ranking).to_csv(args.output_dir / 'oos_ranking.csv', index=False)
    pred.drop(columns=['week_utc']).to_parquet(
        args.output_dir / 'oos_predictions.parquet', index=False
    )
    (args.output_dir / 'protocol.json').write_text(json.dumps({
        'train_only': True, 'validation_loaded': False, 'test_loaded': False,
        'n_folds': N_FOLDS, 'initial_train_fraction': INITIAL_TRAIN_FRACTION,
        'purge_hours': PURGE_HOURS, 'ridge_alpha': RIDGE_ALPHA,
        'predictors': list(features_matrix(df).columns),
        'oos_candidates': len(pred),
        'warning': 'Candidats dépendants, windows recouvrantes; non-backtest.',
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nPERFORMANCES CUMULEES HORS ECHANTILLON (TRAIN uniquement)')
    print(global_df.to_string(index=False))
    print('\nCLASSEMENT RIDGE OOS (descriptif, candidats chevauchants)')
    print(pd.DataFrame(ranking).to_string(index=False))
    print(f'\nFichiers: {args.output_dir.resolve()}')
    print('Aucune donnée Validation ou Test lue. Aucun ajustement de seuil.')


if __name__ == '__main__':
    main()

from __future__ import annotations

"""V3: probabilites TP/SL/TIME, walk-forward TRAIN puis VALIDATION.

Aucun acces au Test. Pas de recherche d'hyperparametres ni de seuil de trading.
Les evenements se chevauchent : ce n'est PAS un backtest.

Depend de evaluate_candidate_regression_v3.py et build_candidate_events_v3.py
(dans le meme repertoire scripts/).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_candidate_regression_v3 import features_matrix, load_split

INPUT_DIR = Path('data/evaluation/v3-candidate-events')
OUTPUT_DIR = Path('data/evaluation/v3-candidate-probabilities')
CLASSES = ('SL', 'TIME', 'TP')
C = 1.0  # Fixe a priori; convention sklearn: plus grand C = moins de regularisation.
PRIOR_PSEUDOCOUNT = 1.0  # Lissage de Laplace par classe, pour eviter log(0).
PURGE_HOURS = 24
INITIAL_FRACTION = 0.50
N_FOLDS = 5
EPS = 1e-15


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=INPUT_DIR)
    p.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def check_data(raw, clean, name):
    if len(clean) == 0:
        raise ValueError(f'{name}: aucun evenement valide')
    if not clean['event_outcome'].isin(CLASSES).all():
        raise ValueError(f'{name}: issue inconnue')
    if clean['event_id'].duplicated().any():
        raise ValueError(f'{name}: IDs dupliques')
    if clean['signal_timestamp'].duplicated().any():
        raise ValueError(f'{name}: timestamps dupliques')
    if not clean['known_at'].le(clean['label_available_at']).all():
        raise ValueError(f'{name}: label disponible avant signal')
    if clean['label_available_at'].isna().any():
        raise ValueError(f'{name}: label_available_at absent')
    if not clean['signal_timestamp'].is_monotonic_increasing:
        raise ValueError(f'{name}: chronologie non triee')
    print(f'{name}: valides={len(clean)}, invalides={len(raw) - len(clean)}')


def make_model():
    # Imputer/scaler sont ajustes STRICTEMENT sur les donnees passees du fold.
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=C, max_iter=2000, solver='lbfgs')),
    ])


def prior_predict(past, future):
    """Frequences historiques par sens, calculees uniquement dans past."""
    counts = pd.crosstab(past['side'], past['event_outcome'])
    counts = counts.reindex(index=['LONG', 'SHORT'], columns=list(CLASSES), fill_value=0)
    smooth = counts.to_numpy(dtype=float) + PRIOR_PSEUDOCOUNT
    probs = smooth / smooth.sum(axis=1, keepdims=True)
    mapping = {'LONG': probs[0], 'SHORT': probs[1]}
    return np.vstack([mapping[side] for side in future['side']])


def model_predict(past, future):
    model = make_model()
    model.fit(features_matrix(past), past['event_outcome'])
    got = model.named_steps['classifier'].classes_
    if set(got) != set(CLASSES):
        raise RuntimeError(f'Classes insuffisantes dans le passe: {got}')
    pred = model.predict_proba(features_matrix(future))
    index = [list(got).index(c) for c in CLASSES]
    return pred[:, index]


def validate_probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(CLASSES):
        raise RuntimeError('Dimensions des probabilites incorrectes')
    if not np.isfinite(p).all() or np.any(p < -1e-12) or np.any(p > 1 + 1e-12):
        raise RuntimeError('Probabilites hors bornes')
    if not np.allclose(p.sum(axis=1), 1, atol=1e-9):
        raise RuntimeError('Les probabilites ne somment pas a 1')


def predict_block(past, future, stage, fold):
    if set(past['event_id']) & set(future['event_id']):
        raise RuntimeError('IDs partages entre apprentissage et evaluation')
    if past['label_available_at'].max() >= future['known_at'].min():
        raise RuntimeError('Fuite: label futur dans historique passe')
    if past['known_at'].max() >= future['known_at'].min() - pd.Timedelta(hours=PURGE_HOURS):
        raise RuntimeError('Purge 24 heures non respectee')
    if set(past['event_outcome']) != set(CLASSES):
        raise RuntimeError('Les 3 classes doivent etre presentes dans l apprentissage')
    out = []
    for name, p in [('DIRECTION_PRIOR', prior_predict(past, future)),
                    ('LOGISTIC', model_predict(past, future))]:
        validate_probabilities(p)
        frame = future[['event_id', 'signal_timestamp', 'known_at', 'side',
                        'event_outcome', 'net_return']].copy()
        frame['stage'] = stage
        frame['fold'] = fold
        frame['model'] = name
        for j, label in enumerate(CLASSES):
            frame['p_' + label.lower()] = p[:, j]
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def metrics(df, group):
    p = df[['p_' + c.lower() for c in CLASSES]].to_numpy(dtype=float)
    y = df['event_outcome'].to_numpy()
    onehot = np.column_stack([y == c for c in CLASSES]).astype(float)
    pi = np.array([CLASSES.index(c) for c in y], dtype=int)
    logloss = -np.mean(np.log(np.clip(p[np.arange(len(p)), pi], EPS, 1.0)))
    brier = np.mean(np.sum((p - onehot)**2, axis=1))
    multiclass_accuracy = float(np.mean(np.array(CLASSES)[p.argmax(axis=1)] == y))
    record = dict(zip(group, df.name if isinstance(df.name, tuple) else (df.name,))) if hasattr(df, 'name') else {}
    return {**record, 'n': len(df), 'log_loss': float(logloss),
            'brier_multiclass': float(brier), 'accuracy': multiclass_accuracy,
            'mean_p_tp': float(p[:, 2].mean()), 'observed_tp_rate': float(onehot[:, 2].mean())}


def summarize(df, grouping):
    result = []
    for keys, part in df.groupby(grouping, sort=True, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(grouping, keys))
        row.update(metrics(part, []))
        result.append(row)
    return pd.DataFrame(result)


def calibration(df):
    """5 groupes de probabilite TP fixes, pas de bins quantiles ajustes."""
    out = []
    cuts = [0, .2, .4, .6, .8, 1.00000001]
    for (stage, model, side), g in df.groupby(['stage', 'model', 'side'], sort=True):
        b = pd.cut(g['p_tp'], bins=cuts, labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'], include_lowest=True)
        for name, s in g.groupby(b, observed=True):
            if s.empty:
                continue
            out.append({'stage': stage, 'model': model, 'side': side, 'tp_probability_bin': str(name),
                        'n': len(s), 'predicted_tp': float(s['p_tp'].mean()),
                        'observed_tp': float((s['event_outcome'] == 'TP').mean())})
    return pd.DataFrame(out)


def main():
    args = parse_args()
    train_raw, train = load_split(args.input_dir, 'train')
    val_raw, validation = load_split(args.input_dir, 'validation')
    train = train.sort_values('signal_timestamp').reset_index(drop=True)
    validation = validation.sort_values('signal_timestamp').reset_index(drop=True)
    print('=' * 79)
    print('QWEN V3 - PROBABILITES TP / SL / TIME - TRAIN WALK-FORWARD + VALIDATION')
    print('=' * 79)
    check_data(train_raw, train, 'TRAIN')
    check_data(val_raw, validation, 'VALIDATION')
    if train['known_at'].max() >= validation['known_at'].min():
        raise RuntimeError('Train / Validation chevauchants')
    if len(train) < 500:
        raise RuntimeError('Historique Train insuffisant')

    chunks = []
    folds = []
    n = len(train)
    boundaries = np.linspace(int(np.floor(n * INITIAL_FRACTION)), n, N_FOLDS + 1, dtype=int)
    for fold in range(N_FOLDS):
        left, right = int(boundaries[fold]), int(boundaries[fold + 1])
        future = train.iloc[left:right].copy()
        first_known = future['known_at'].min()
        allowed = train.iloc[:left]
        past = allowed.loc[(allowed['known_at'] < first_known - pd.Timedelta(hours=PURGE_HOURS))
                           & (allowed['label_available_at'] < first_known)].copy()
        if len(past) < 100:
            raise RuntimeError(f'Fold {fold + 1}: pas assez de donnees')
        pred = predict_block(past, future, 'TRAIN_WALKFORWARD', fold + 1)
        chunks.append(pred)
        folds.append({'fold': fold + 1, 'train_count': len(past), 'eval_count': len(future),
                      'train_last_known_at': str(past['known_at'].max()),
                      'train_last_label_at': str(past['label_available_at'].max()),
                      'eval_first_known_at': str(first_known),
                      'eval_last_known_at': str(future['known_at'].max())})
        scores = summarize(pred, ['model'])
        a = scores.set_index('model')
        print(f"Fold {fold + 1}: past={len(past):4d} eval={len(future):3d} | "
              f"logloss prior={a.loc['DIRECTION_PRIOR','log_loss']:.4f}, logistic={a.loc['LOGISTIC','log_loss']:.4f} | "
              f"Brier prior={a.loc['DIRECTION_PRIOR','brier_multiclass']:.4f}, logistic={a.loc['LOGISTIC','brier_multiclass']:.4f}")

    # Validation: uniquement le Train historique, avec purge+disponibilite effective.
    first_val = validation['known_at'].min()
    past = train.loc[(train['known_at'] < first_val - pd.Timedelta(hours=PURGE_HOURS))
                     & (train['label_available_at'] < first_val)].copy()
    if len(past) < 100:
        raise RuntimeError('Train insuffisant pour Validation')
    chunks.append(predict_block(past, validation, 'VALIDATION', 0))
    predictions = pd.concat(chunks, ignore_index=True)
    aggregate = summarize(predictions, ['stage', 'side', 'model'])
    fold_metrics = summarize(predictions.loc[predictions['stage'] == 'TRAIN_WALKFORWARD'], ['fold', 'side', 'model'])
    calib = calibration(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output_dir / 'predictions.parquet', index=False)
    aggregate.to_csv(args.output_dir / 'aggregate_metrics.csv', index=False)
    fold_metrics.to_csv(args.output_dir / 'fold_metrics.csv', index=False)
    calib.to_csv(args.output_dir / 'tp_calibration.csv', index=False)
    pd.DataFrame(folds).to_csv(args.output_dir / 'fold_boundaries.csv', index=False)
    protocol = {'classes': list(CLASSES), 'logistic_C': C, 'solver': 'lbfgs', 'max_iter': 2000,
                'prior_pseudocount_per_class': PRIOR_PSEUDOCOUNT, 'purge_hours': PURGE_HOURS,
                'initial_train_fraction': INITIAL_FRACTION, 'folds': N_FOLDS,
                'validation_training_count': len(past), 'features': list(features_matrix(train).columns),
                'no_test_loaded': True, 'no_tuning': True,
                'brier_definition': 'mean(sum_classes((p_class - onehot_class)^2))',
                'note': 'Candidats horaires chevauchants: pas un backtest, pas de trades executes.'}
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nMETRIQUES CUMULEES (faible logloss/Brier = mieux)')
    print(aggregate.to_string(index=False, float_format=lambda z: f'{z:.5f}'))
    print('\nNOTE: Brier multiclasses = somme des 3 erreurs quadratiques par evenement.')
    print(f'Fichiers: {args.output_dir.resolve()}')
    print('Aucun Test lu. Aucun tuning. Aucun entrainement Qwen. Pas un backtest.')


if __name__ == '__main__':
    main()

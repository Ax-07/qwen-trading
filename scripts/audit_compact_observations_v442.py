"""V4.4.2: read-only statistical audit + fixed, causal Qwen pilot projection.

Input: V4.4.1 observations_normalized.parquet. No learning, no labels, no Test access.
Do not treat distribution diagnostics as evidence of trading performance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path('data/evaluation/v441-observation-audit/observations_normalized.parquet')
OUTPUT = Path('data/evaluation/v442-observation-audit')
TFS = ('5m', '15m', '1h', '4h')
# Fixed *a priori* pilot schema. Not chosen from returns or feature correlations.
INDICATORS = ('return_1', 'ema12_distance', 'ema26_distance', 'sma20_distance',
              'rsi14', 'macd_hist', 'atr14_pct', 'bb20_z', 'bb20_width_pct',
              'rolling_vwap20_distance', 'relative_volume20')
STRUCTURE = ('high_distance_pct', 'low_distance_pct',
             'high_structure', 'low_structure',
             'pivot_high_confirmed', 'pivot_low_confirmed',
             'event_type', 'trend_direction',
             'active_high_distance_to_close', 'active_low_distance_to_close',
             'zz_event', 'zz_direction', 'zz_reversal_atr',
             'zz_age_bars', 'zz_last_high_distance_to_close',
             'zz_last_low_distance_to_close')
# The global presence of context is vital; feature-level availability is retained
# for sparse fields, while all other missing values stay NaN in the pilot parquet.
MASK_FIELDS = ('high_distance_pct', 'low_distance_pct',
               'active_high_distance_to_close', 'active_low_distance_to_close',
               'zz_reversal_atr', 'zz_age_bars',
               'zz_last_high_distance_to_close', 'zz_last_low_distance_to_close')


def expected_columns():
    cols = []
    for tf in TFS:
        cols.append(f'context_available_{tf}')
        for field in INDICATORS + STRUCTURE:
            col = f'{field}_{tf}_code' if field in ('high_structure', 'low_structure', 'event_type', 'zz_event', 'zz_direction') else f'{field}_{tf}'
            cols.append(col)
            if field in MASK_FIELDS:
                cols.append(col + '_available')
    return cols


def audit_and_project(df: pd.DataFrame):
    if df.empty or not df.columns.is_unique or 'decision_at' not in df:
        raise ValueError('Dataframe vide, cle absente ou colonnes dupliquees')
    t = pd.to_datetime(df.decision_at, errors='coerce', utc=True)
    if t.isna().any() or t.duplicated().any() or not t.is_monotonic_increasing:
        raise ValueError('decision_at non valides ou non ordonnees')
    cols = expected_columns()
    absent = sorted(set(cols).difference(df.columns))
    if absent:
        raise ValueError('Colonnes requises absentes: ' + ', '.join(absent[:12]))
    numeric = df.drop(columns=['decision_at'])
    non_numeric = [c for c in numeric if not pd.api.types.is_numeric_dtype(numeric[c])]
    if non_numeric:
        raise ValueError('Colonnes non numeriques: ' + ', '.join(non_numeric[:8]))
    values = numeric.to_numpy(dtype='float64', copy=False)
    if np.isinf(values).any():
        raise ValueError('Valeurs infinies detectees')
    features = numeric.loc[:, cols].copy()
    for tf in TFS:
        context = features[f'context_available_{tf}']
        if not context.isin([0, 1]).all():
            raise ValueError(f'Masque contexte {tf} invalide')
        for c in cols:
            if not (c.endswith('_' + tf) or c.endswith('_' + tf + '_code') or c.endswith('_' + tf + '_available')) or c == f'context_available_{tf}':
                continue
            if c.endswith('_available'):
                if not features[c].isin([0, 1]).all():
                    raise ValueError(f'Masque {c} invalide')
            if ((context == 0) & features[c].notna() & (features[c] != 0)).any():
                # Availability masks may be zero; other features must be NaN.
                raise ValueError(f'Variable disponible sans contexte: {c}')
    rows = []
    for c in numeric:
        x = pd.to_numeric(numeric[c], errors='coerce')
        ok = x.dropna()
        rows.append({
            'column': c, 'selected_pilot': c in cols, 'dtype': str(x.dtype),
            'missing_count': int(x.isna().sum()), 'missing_pct': round(float(x.isna().mean() * 100), 4),
            'unique_non_null': int(ok.nunique()), 'is_constant': bool(ok.nunique() <= 1),
            'min': float(ok.min()) if len(ok) else None,
            'p01': float(ok.quantile(.01)) if len(ok) else None,
            'median': float(ok.median()) if len(ok) else None,
            'p99': float(ok.quantile(.99)) if len(ok) else None,
            'max': float(ok.max()) if len(ok) else None,
        })
    audit = pd.DataFrame(rows)
    # Pairwise correlation is an exploratory diagnostic only, not used to select features.
    # Exclude binary flags and constants to avoid unhelpful near-duplicate reports.
    continuous = [c for c in cols if not c.endswith('_available') and not c.startswith('context_available_')
                  and numeric[c].dropna().nunique() > 10]
    cor = features[continuous].corr(min_periods=500)
    pairs = []
    for i, a in enumerate(continuous):
        for b in continuous[i+1:]:
            r = cor.loc[a, b]
            if pd.notna(r) and abs(r) >= .995:
                both = int((features[a].notna() & features[b].notna()).sum())
                pairs.append({'a': a, 'b': b, 'pearson': round(float(r), 6), 'overlap_rows': both})
    duplicates = pd.DataFrame(pairs, columns=['a', 'b', 'pearson', 'overlap_rows'])
    schema = {'schema_version': 'v4.4.2-pilot-1', 'source': 'v4.4.1', 'rows': len(df),
              'source_feature_count': len(numeric.columns), 'pilot_feature_count': len(cols),
              'feature_columns': cols, 'selected_by': 'fixed engineering whitelist, not returns',
              'missing_policy': 'retain NaN and context / sparse availability masks; no global fit',
              'warning': 'This pilot is NOT a training-ready tensor. Fit imputation/scaling on Train only.',
              'time_range_utc': [t.iloc[0].isoformat(), t.iloc[-1].isoformat()]}
    schema['feature_schema_sha256'] = hashlib.sha256(json.dumps(cols).encode()).hexdigest()
    pilot = pd.concat([pd.DataFrame({'decision_at': t}, index=df.index), features], axis=1)
    return pilot, audit, duplicates, schema


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=INPUT)
    p.add_argument('--output', type=Path, default=OUTPUT)
    args = p.parse_args()
    if not args.input.is_file():
        p.error(f'Source introuvable: {args.input}')
    paths = [args.output / n for n in ('pilot_observations.parquet', 'feature_audit.csv',
                                      'high_correlations.csv', 'schema.json')]
    if any(x.exists() for x in paths):
        p.error('Refus ecrasement: sortie deja presente')
    pilot, audit, pairs, schema = audit_and_project(pd.read_parquet(args.input))
    args.output.mkdir(parents=True, exist_ok=True)
    pilot.to_parquet(paths[0], index=False)
    audit.to_csv(paths[1], index=False, encoding='utf-8-sig')
    pairs.to_csv(paths[2], index=False, encoding='utf-8-sig')
    paths[3].write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'V4.4.2 PASS | lignes={len(pilot)} | audit={len(audit)} variables | pilote={len(pilot.columns)-1} variables')
    print(f'Constantes={int(audit.is_constant.sum())} | Paires abs(corr)>=0.995={len(pairs)}')
    print('Sortie:', args.output.resolve())
    print('Aucune source modifiee ; aucun apprentissage, aucune evaluation Test.')

if __name__ == '__main__':
    main()

"""V4.4.1 — Audit read-only et projection causale normalisée de V4.4.0.

Ne réécrit ni la source ni une sortie existante. Aucun apprentissage / trading.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_INPUT = Path('data/evaluation/v440-observations/BTCUSDT-2024-01.parquet')
DEFAULT_OUTPUT = Path('data/evaluation/v441-observation-audit')
TFS = ('5m', '15m', '1h', '4h')
# Explicit whitelist: unexplained columns never enter the model by accident.
CONTINUOUS = ('return_1', 'ema12_distance', 'ema26_distance', 'sma20_distance',
              'rsi14', 'macd_12_26', 'macd_signal9', 'macd_hist', 'atr14_pct',
              'bb20_z', 'bb20_width_pct', 'rolling_vwap20_distance', 'relative_volume20',
              'high_distance_pct', 'low_distance_pct', 'break_distance_pct',
              'zz_reversal_atr', 'zz_reversal_pct', 'zz_age_bars')
CATEGORICAL = {
    'high_structure': {'NONE': 0, 'FIRST': 0, 'HH': 1, 'LH': -1, 'EH': 2},
    'low_structure': {'NONE': 0, 'FIRST': 0, 'HL': 1, 'LL': -1, 'EL': 2},
    'event_type': {'NONE': 0, 'BULLISH_BOS': 1, 'BEARISH_BOS': -1,
                   'BULLISH_CHOCH': 2, 'BEARISH_CHOCH': -2},
    'zz_event': {'NONE': 0, 'HIGH': 1, 'LOW': -1},
    'zz_direction': {'SEEK': 0, 'UP': 1, 'DOWN': -1},
}
DISCRETE = ('pivot_high_confirmed', 'pivot_low_confirmed', 'event_fired',
            'event_direction', 'trend_direction', 'zz_confirmed')
LEVELS = ('swing_high_price', 'swing_low_price', 'active_high', 'active_low',
          'broken_level', 'zz_pivot_price', 'zz_last_high', 'zz_last_low')
TIMESTAMPS = ('swing_high_at', 'swing_low_at', 'swing_high_confirmed_at',
              'swing_low_confirmed_at', 'level_pivot_at', 'level_confirmed_at',
              'break_at', 'zz_pivot_at', 'zz_confirmed_at')
EVENT_GATED = ('broken_level', 'break_distance_pct', 'zz_pivot_price',
               'zz_reversal_atr', 'zz_reversal_pct')


def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if df.empty or df.columns.duplicated().any():
        raise ValueError('Entrée vide ou colonnes dupliquées')
    if 'decision_at' not in df or 'close_5m' not in df:
        raise ValueError('decision_at / close_5m requis')
    t = pd.to_datetime(df['decision_at'], utc=True)
    if t.isna().any() or t.duplicated().any() or not t.is_monotonic_increasing:
        raise ValueError('Décisions invalides, dupliquées ou non triées')
    price5 = pd.to_numeric(df['close_5m'], errors='coerce')
    if not np.isfinite(price5).all() or (price5 <= 0).any():
        raise ValueError('Prix 5m non positifs ou non finis')
    out = pd.DataFrame({'decision_at': t}, index=df.index)
    rows = []
    selected = set()

    for tf in TFS:
        known = f'known_at_{tf}'
        if known not in df:
            raise ValueError(f'{known} absent')
        kt = pd.to_datetime(df[known], utc=True)
        maxlag = pd.Timedelta(minutes={'5m': 5, '15m': 15, '1h': 60, '4h': 240}[tf])
        bad = kt.notna() & ((kt > t) | ((t - kt) > maxlag))
        if bad.any():
            raise ValueError(f'{known}: contexte futur/périmé')
        present = kt.notna()
        selected.add(known)
        out[f'context_available_{tf}'] = present.astype('int8')
        ref = f'close_{tf}'
        if ref not in df:
            raise ValueError(f'{ref} absent')
        prices = pd.to_numeric(df[ref], errors='coerce')
        if (present & (~np.isfinite(prices) | (prices <= 0))).any():
            raise ValueError(f'{ref}: prix invalide sur contexte présent')
        selected.add(ref)

        for field in CONTINUOUS + LEVELS + DISCRETE + tuple(CATEGORICAL):
            col = f'{field}_{tf}'
            if col not in df:
                continue
            selected.add(col)
            v = df[col]
            if field in LEVELS:
                # price denominator comes from the SAME timeframe; no nominal prices exported.
                numeric = pd.to_numeric(v, errors='coerce')
                val = numeric.div(prices).sub(1)
                target = f'{field}_distance_to_close_{tf}'
            elif field in CATEGORICAL:
                unexpected = set(v.dropna().unique()) - set(CATEGORICAL[field])
                if unexpected:
                    raise ValueError(f'{col}: catégories inconnues {unexpected}')
                val = v.map(CATEGORICAL[field]).astype('float64')
                target = col + '_code'
            elif field in DISCRETE:
                if field in ('pivot_high_confirmed', 'pivot_low_confirmed', 'event_fired', 'zz_confirmed'):
                    val = v.map({True: 1, False: 0, 1: 1, 0: 0}).astype('float64')
                else:
                    val = pd.to_numeric(v, errors='coerce')
                target = col
            else:
                val = pd.to_numeric(v, errors='coerce')
                target = col
            val = val.where(present)
            if np.isinf(val.to_numpy(dtype='float64')).any():
                raise ValueError(f'{col}: infini')
            out[target] = val.astype('float32')
            # Explicit missingness for sparse structural features and initial warmups.
            out[target + '_available'] = val.notna().astype('int8')

        for field in TIMESTAMPS:
            col = f'{field}_{tf}'
            if col in df:
                selected.add(col)
                stamp = pd.to_datetime(df[col], utc=True)
                if (stamp.notna() & (stamp > t)).any():
                    raise ValueError(f'{col}: timestamp futur')
                # Event/pivot timestamps retained for audit only, never model features.

    selected.update(('decision_at', 'bar_open_5m', 'close_5m'))
    for c in df:
        s = df[c]
        rows.append({'column': c, 'dtype': str(s.dtype), 'null_count': int(s.isna().sum()),
                     'null_fraction': round(float(s.isna().mean()), 6),
                     'role': ('PROJECTED_OR_AUDITED' if c in selected else 'EXCLUDED_UNRECOGNIZED')})
    audit = pd.DataFrame(rows)
    model_cols = [c for c in out if c != 'decision_at']
    report = {'schema_version': 'v4.4.1', 'input_rows': len(df),
              'input_columns': len(df.columns), 'output_columns': len(model_cols),
              'excluded_columns': audit.loc[audit.role == 'EXCLUDED_UNRECOGNIZED', 'column'].tolist(),
              'start_utc': t.iloc[0].isoformat(), 'end_utc': t.iloc[-1].isoformat(),
              'note': 'No labels, no future targets. Timestamp decision_at is audit key, not model feature.'}
    return out, audit, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    a = p.parse_args()
    if not a.input.is_file():
        p.error(f'Entrée introuvable: {a.input}')
    targets = {'features': a.output / 'observations_normalized.parquet',
               'audit': a.output / 'column_audit.csv',
               'report': a.output / 'manifest.json'}
    if any(path.exists() for path in targets.values()):
        p.error('Refus écrasement: une ou plusieurs sorties existent déjà')
    out, audit, report = normalize(pd.read_parquet(a.input))
    a.output.mkdir(parents=True, exist_ok=True)
    out.to_parquet(targets['features'], index=False)
    audit.to_csv(targets['audit'], index=False, encoding='utf-8-sig')
    targets['report'].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'V4.4.1 PASS — {report["input_rows"]} observations, {report["input_columns"]} colonnes source, {report["output_columns"]} variables projetées')
    print(f'Colonnes non reconnues exclues: {len(report["excluded_columns"])}')
    print('Sortie:', a.output.resolve())
    print('Aucun fichier source modifié ; aucun entraînement ni trading.')

if __name__ == '__main__':
    main()

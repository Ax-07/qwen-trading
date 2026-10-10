"""V4.4.0 — Fusion stricte des quatre snapshots causaux, sans calcul de signaux.

Exécuter depuis la racine de qwen-trading. N'écrase jamais la sortie.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path('data/evaluation')
SOURCES = {
    'indicators': BASE / 'v430-features/BTCUSDT-2024-01.parquet',
    'swings': BASE / 'v431-swings/BTCUSDT-2024-01.parquet',
    'breaks': BASE / 'v432-bos-choch/BTCUSDT-2024-01.parquet',
    'zigzag': BASE / 'v433-zigzag/BTCUSDT-2024-01.parquet',
}
OUTPUT = BASE / 'v440-observations/BTCUSDT-2024-01.parquet'
TIMEFRAMES = {'5m': 5, '15m': 15, '1h': 60, '4h': 240}
IDENTITY = ('decision_at', 'bar_open_5m', 'close_5m')


def utc_ns(values):
    return pd.to_datetime(values, utc=True, errors='raise').astype('datetime64[ns, UTC]')


def assemble(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Exact 1:1 join on 5m decision timestamp; fail closed on mismatches."""
    if set(frames) != set(SOURCES):
        raise ValueError(f'Sources attendues : {list(SOURCES)}')
    cleaned = {}
    for name, raw in frames.items():
        df = raw.copy()
        missing = set(IDENTITY) - set(df)
        if missing:
            raise ValueError(f'{name}: colonnes d’identité absentes: {sorted(missing)}')
        for key in ('decision_at', 'bar_open_5m'):
            df[key] = utc_ns(df[key])
        if df.empty or df.decision_at.isna().any() or df.decision_at.duplicated().any():
            raise ValueError(f'{name}: timestamps vides, dupliqués ou jeu vide')
        if not df.decision_at.is_monotonic_increasing:
            raise ValueError(f'{name}: timestamps non triés')
        if (df.decision_at != df.bar_open_5m + pd.Timedelta(minutes=5)).any():
            raise ValueError(f'{name}: décision avant/après clôture 5m')
        if not np.isfinite(pd.to_numeric(df.close_5m, errors='coerce')).all():
            raise ValueError(f'{name}: close_5m invalide')
        cleaned[name] = df

    reference = cleaned['indicators']
    # Strict validation before joining — never silently shrink/interpolate.
    for name, df in cleaned.items():
        if len(df) != len(reference) or not df.decision_at.equals(reference.decision_at):
            raise ValueError(f'{name}: grille des décisions 5m différente')
        if not df.bar_open_5m.equals(reference.bar_open_5m):
            raise ValueError(f'{name}: ouverture des bougies 5m différente')
        if not np.allclose(df.close_5m.to_numpy(dtype=float),
                           reference.close_5m.to_numpy(dtype=float), rtol=0, atol=1e-9):
            raise ValueError(f'{name}: prix 5m non concordants')

    out = reference.copy()
    # Indicators already own the canonical HTF known_at / close values.
    for name in ('swings', 'breaks', 'zigzag'):
        df = cleaned[name]
        # Identical names (notably known_at_N) must agree with reference.
        common = set(out.columns).intersection(df.columns) - set(IDENTITY)
        for col in sorted(common):
            a, b = out[col], df[col]
            if col.startswith('known_at_') or col.endswith('_at'):
                a, b = utc_ns(a), utc_ns(b)
                same = a.eq(b) | (a.isna() & b.isna())
            else:
                same = a.eq(b) | (a.isna() & b.isna())
            if not bool(same.all()):
                raise ValueError(f'{name}: colonne partagée incompatible: {col}')
        additions = df.drop(columns=list(IDENTITY) + sorted(common))
        out = pd.concat([out, additions], axis=1)

    # Validate availability, including the higher timeframes' freshness bound.
    for tf, minutes in TIMEFRAMES.items():
        col = f'known_at_{tf}'
        if col not in out:
            raise ValueError(f'Horodatage manquant: {col}')
        t = utc_ns(out[col])
        delta = out.decision_at - t
        bad = t.notna() & ((delta < pd.Timedelta(0)) |
                              (delta > pd.Timedelta(minutes=minutes)))
        if bad.any():
            raise ValueError(f'{col}: contexte futur ou périmé ({int(bad.sum())})')
        out[col] = t

    # Event-specific checks: an event must never be dated later than decision.
    for col in out.columns:
        if col.endswith(('_confirmed_at_5m', '_confirmed_at_15m', '_confirmed_at_1h', '_confirmed_at_4h',
                         '_break_at_5m', '_break_at_15m', '_break_at_1h', '_break_at_4h')):
            t = utc_ns(out[col])
            if (t.notna() & (t > out.decision_at)).any():
                raise ValueError(f'Événement futur: {col}')
            out[col] = t

    if out.columns.duplicated().any():
        raise ValueError('Colonnes fusionnées dupliquées')
    report = {
        'schema_version': 'v4.4.0', 'rows': len(out),
        'start_decision_utc': reference.decision_at.iloc[0].isoformat(),
        'end_decision_utc': reference.decision_at.iloc[-1].isoformat(),
        'columns': len(out),
        'sources': list(SOURCES),
        'available_context_rows': {tf: int(out[f'known_at_{tf}'].notna().sum()) for tf in TIMEFRAMES},
        'null_cells': int(out.isna().sum().sum()),
        'note': 'Observation only. No labels, no trading. Values remain NaN when unavailable.',
    }
    return out, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in SOURCES.items():
        parser.add_argument(f'--{name}', type=Path, default=path)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    inputs = {name: getattr(args, name) for name in SOURCES}
    missing = [str(p) for p in inputs.values() if not p.is_file()]
    if missing:
        parser.error('Fichiers sources absents : ' + ', '.join(missing))
    if args.output.exists() or args.output.with_suffix('.manifest.json').exists():
        parser.error(f'Refus écrasement: {args.output} ou son manifest')
    out, report = assemble({n: pd.read_parquet(path) for n, path in inputs.items()})
    report['input_files'] = {n: str(p) for n, p in inputs.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Temporary writes: failure must not leave a finished-looking output.
    tmp = args.output.with_suffix('.parquet.partial')
    try:
        out.to_parquet(tmp, index=False)
        tmp.replace(args.output)
        args.output.with_suffix('.manifest.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        tmp.unlink(missing_ok=True)
    print('V4.4.0 PASS —', report['rows'], 'snapshots ;', report['columns'], 'colonnes')
    print('Contextes disponibles:', report['available_context_rows'])
    print('Sortie :', args.output.resolve())
    print('Aucune source modifiée ; aucun entraînement, aucun trading.')


if __name__ == '__main__':
    main()

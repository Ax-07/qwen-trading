"""V4.2.1 — Validation croisée BTCUSDT Binance 1m -> 1h, lecture seule.

Comparaison sur les seules heures disposant de 60 minutes, et déclaration
explicite des heures manquantes dans les deux sources.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prototype_multitimeframe_v42 import completed_bars, build_snapshots

FIELDS = ('open', 'high', 'low', 'close', 'volume', 'quote_volume')


def compare(data_1m: pd.DataFrame, data_1h: pd.DataFrame, atol: float = 1e-6):
    recon = completed_bars(data_1m, 60).set_index('timestamp')
    ref = data_1h.copy()
    ref['timestamp'] = pd.to_datetime(ref.timestamp, utc=True, errors='raise')
    if ref.timestamp.duplicated().any() or not ref.timestamp.is_monotonic_increasing:
        raise ValueError('Référence 1h non ordonnée ou avec doublons')
    ref = ref.set_index('timestamp')
    common = recon.index.intersection(ref.index)
    if not len(common):
        raise ValueError('Aucune heure commune')
    metrics = {}
    discrepancies = pd.DataFrame({'timestamp': common})
    for field in FIELDS:
        if field not in recon or field not in ref:
            raise ValueError(f'Colonne absente : {field}')
        a = recon.loc[common, field].to_numpy(dtype=float)
        b = pd.to_numeric(ref.loc[common, field], errors='raise').to_numpy(dtype=float)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError(f'Valeur non finie : {field}')
        diff = np.abs(a-b)
        allowed = atol + 1e-9 * np.abs(b)
        bad = diff > allowed
        metrics[field] = {'max_abs_diff': float(diff.max()), 'mismatches': int(bad.sum())}
        discrepancies[f'{field}_abs_diff'] = diff
        discrepancies[f'{field}_mismatch'] = bad
    summary = {'minutes': len(data_1m), 'complete_hours_from_1m': len(recon),
               'reference_1h_hours': len(ref), 'compared_hours': len(common),
               'hours_only_in_1m': [x.isoformat() for x in recon.index.difference(ref.index)],
               'hours_only_in_1h': [x.isoformat() for x in ref.index.difference(recon.index)],
               'fields': metrics,
               'passed': not any(m['mismatches'] for m in metrics.values()) and
                         len(recon.index.difference(ref.index)) == 0 and
                         len(ref.index.difference(recon.index)) == 0}
    return summary, discrepancies


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--symbol', default='BTCUSDT')
    p.add_argument('--month', default='2024-01')
    p.add_argument('--minute-root', type=Path, default=Path('data/evaluation/v421-binance-1m'))
    p.add_argument('--hour-root', type=Path, default=Path('data/evaluation/v41-binance-hourly'))
    p.add_argument('--output', type=Path, default=Path('data/evaluation/v421-crosscheck'))
    args = p.parse_args()
    minute_path = args.minute_root / args.symbol / f'{args.month}.parquet'
    hour_path = args.hour_root / 'monthly' / args.symbol / f'{args.month}.parquet'
    result_path = args.output / f'{args.symbol}-{args.month}-report.json'
    if result_path.exists():
        p.error(f'Rapport déjà présent : {result_path}')
    minute = pd.read_parquet(minute_path)
    hour = pd.read_parquet(hour_path)
    summary, diff = compare(minute, hour)
    bars = {str(n): len(completed_bars(minute, n)) for n in (5, 15, 60, 240)}
    snap = build_snapshots(minute)
    summary['completed_bars'] = bars
    summary['decision_snapshots_5m'] = len(snap)
    summary['paths'] = {'minute': str(minute_path), 'hour': str(hour_path)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Bougies clôturées par timeframe :', bars)
    print(f'Heures comparées : {summary["compared_hours"]}; snapshots 5m : {len(snap)}')
    for key, value in summary['fields'].items():
        print(f'{key:13} max_abs_diff={value["max_abs_diff"]:.12g} mismatches={value["mismatches"]}')
    print('Heures 1m sans 1h :', len(summary['hours_only_in_1m']))
    print('Heures 1h sans 60 minutes :', len(summary['hours_only_in_1h']))
    print('RESULTAT :', 'PASS' if summary['passed'] else 'FAIL — audit nécessaire')
    print('Rapport :', result_path.resolve())
    if not summary['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

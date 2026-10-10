from __future__ import annotations
"""Audit des scénarios V3.1 existants: comparaison appariée + MAE/MFE.

Lecture seule Train/Validation. Aucun Test, aucun nouvel ordre ni optimisation.
Exécuter depuis la racine: python scripts/analyze_structural_excursions_v31.py
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Les scripts de ce dossier sont fournis avec le projet.
from diagnose_structural_tp_v31 import validate, common_comparison, METHODS, MULTIPLES
from simulate_structural_tp_v31 import prepare_market

SPLITS = ('train', 'validation')
REQUIRED = {'entry_raw','stop_price','target_price','risk_price','exit_raw','entry_at','exit_at',
            'duration_bars','portfolio_status','reason','side','event_id','method','timeframe',
            'multiple_r','eligibility','net_return'}


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def select_bars(market: pd.DataFrame, entry_at, duration: int) -> pd.DataFrame:
    """Exactement les bougies tenues; jamais au-delà de la sortie simulée."""
    if duration < 1 or duration > 12:
        raise ValueError(f'Durée hors horizon: {duration}')
    pos = market.index[market.timestamp == entry_at]
    if len(pos) != 1:
        raise ValueError(f'Entrée absente/ambiguë: {entry_at}')
    i = int(pos[0])
    bars = market.iloc[i:i + duration]
    if len(bars) != duration or not (bars.timestamp.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError('Fenêtre de bougies incomplète ou discontinue')
    return bars


def excursion(row, bars: pd.DataFrame) -> dict:
    """Excursions sur OHLC complètes des bougies tenues.

    Sur SL/TP intrabar, MAE/MFE de la bougie de sortie est une BORNE DESCRIPTIVE:
    l'ordre des extrêmes dans cette bougie est inconnu. Pour TIME, mesure complète.
    """
    entry = float(row.entry_raw)
    risk = float(row.risk_price)
    if not np.isfinite(entry) or not np.isfinite(risk) or entry <= 0 or risk <= 0:
        raise ValueError('Entrée ou risque invalide')
    if not np.isclose(entry, float(bars.iloc[0].open), rtol=1e-10):
        raise ValueError('Prix de référence différent des données marché')
    if row.side == 'LONG':
        fav = float(bars.high.max()) - entry
        adv = entry - float(bars.low.min())
    elif row.side == 'SHORT':
        fav = entry - float(bars.low.min())
        adv = float(bars.high.max()) - entry
    else:
        raise ValueError('Direction incorrecte')
    fav = max(0., fav)
    adv = max(0., adv)
    return dict(mfe_pct=100 * fav / entry, mae_pct=100 * adv / entry,
                mfe_r=fav / risk, mae_r=adv / risk,
                mfe_over_half_r=bool(fav >= .5 * risk),
                mfe_over_one_r=bool(fav >= risk),
                mae_over_half_r=bool(adv >= .5 * risk),
                intrabar_exit_uncertain=not str(row.reason).startswith('TIME'))


def compute_excursions(scenarios: pd.DataFrame, market: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = scenarios.loc[(scenarios.eligibility == 'ELIGIBLE') &
                             (scenarios.portfolio_status == 'TAKEN')].copy()
    if not len(selected):
        return pd.DataFrame()
    if not REQUIRED.issubset(scenarios.columns):
        raise ValueError(f'Colonnes scénarios manquantes: {sorted(REQUIRED - set(scenarios.columns))}')
    market = market.reset_index(drop=True)
    # Fenêtre identique pour les scénarios partageant date d'entrée et durée.
    cache = {}
    outputs = []
    for row in selected.itertuples(index=False):
        key = (row.entry_at, int(row.duration_bars))
        if key not in cache:
            cache[key] = select_bars(market, *key)
        bars = cache[key]
        expected_exit = bars.iloc[-1].timestamp + pd.Timedelta(hours=1)
        if row.exit_at != expected_exit:
            raise ValueError(f'Horodatage de sortie incohérent: {row.event_id}')
        data = excursion(row, bars)
        outputs.append(dict(split=split, method=row.method, timeframe=row.timeframe,
                            multiple_r=float(row.multiple_r), event_id=row.event_id,
                            side=row.side, reason=row.reason, entry_at=row.entry_at,
                            exit_at=row.exit_at, duration_bars=int(row.duration_bars),
                            net_return_pct=100 * float(row.net_return),
                            **data))
    return pd.DataFrame(outputs)


def report_excursions(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for (split, method, timeframe, multiple, side, exit_type), g in frame.assign(
        exit_type=frame.reason.astype(str).str.extract(r'^(TP|SL|TIME)', expand=False)
    ).groupby(['split','method','timeframe','multiple_r','side','exit_type']):
        rows.append(dict(split=split, method=method, timeframe=timeframe, multiple_r=multiple,
            side=side, exit_type=exit_type, count=len(g),
            mean_net_pct=float(g.net_return_pct.mean()),
            mean_duration_h=float(g.duration_bars.mean()),
            median_mfe_r=float(g.mfe_r.median()), p90_mfe_r=float(g.mfe_r.quantile(.9)),
            median_mae_r=float(g.mae_r.median()), p90_mae_r=float(g.mae_r.quantile(.9)),
            pct_reached_half_r=float(g.mfe_over_half_r.mean()*100),
            pct_reached_one_r=float(g.mfe_over_one_r.mean()*100),
            pct_mae_half_r=float(g.mae_over_half_r.mean()*100),
            uncertain_intrabar_pct=float(g.intrabar_exit_uncertain.mean()*100)))
    return pd.DataFrame(rows)


def run(scenario_dir: Path, split_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f'Refus d’écraser le dossier: {output}')
    frames = []; pair = []; common = []; hashes = {}
    for split in SPLITS:
        path = scenario_dir / f'{split}_scenarios.parquet'
        market_path = split_dir / f'{split}.parquet'
        if not path.is_file() or not market_path.is_file():
            raise FileNotFoundError(f'Entrées manquantes: {path} ou {market_path}')
        scenarios = validate(pd.read_parquet(path), split)
        if not REQUIRED.issubset(scenarios.columns):
            raise ValueError(f'Colonnes manquantes: {sorted(REQUIRED-set(scenarios.columns))}')
        market = prepare_market(market_path)
        cmp, paired = common_comparison(scenarios, split)
        common.append(cmp); pair.append(paired)
        ex = compute_excursions(scenarios, market, split)
        frames.append(ex)
        hashes[split] = {'scenarios_sha256':fingerprint(path),'splits_sha256':fingerprint(market_path)}
        print(f'{split.upper():11} {len(scenarios):6} scénarios | {len(ex):5} trades retenus (24 portefeuilles)')
        for side in ('LONG','SHORT'):
            g = ex.loc[(ex.method == 'zigzag') & (ex.timeframe == '4h') &
                       (ex.multiple_r == 1.5) & (ex.side == side)]
            timeouts = g.loc[g.reason.astype(str).str.startswith('TIME')]
            if len(timeouts):
                print(f'  ZigZag 4H 1,5R {side:5} TIME={len(timeouts):3} / taken={len(g):3}'
                      f' | MFE médian={timeouts.mfe_r.median():.3f}R'
                      f' | MAE médian={timeouts.mae_r.median():.3f}R'
                      f' | >=0,5R={timeouts.mfe_over_half_r.mean()*100:.1f}%')
    all_ex = pd.concat(frames, ignore_index=True)
    stats = report_excursions(all_ex)
    output.mkdir(parents=True, exist_ok=False)
    all_ex.to_parquet(output / 'taken_excursions.parquet', index=False)
    stats.to_csv(output / 'excursions_by_exit.csv', index=False)
    pd.concat(common, ignore_index=True).to_csv(output / 'common_candidates.csv', index=False)
    pd.concat(pair, ignore_index=True).to_csv(output / 'paired_comparison.csv', index=False)
    meta = {'version':'v31-excursion-diagnostic-v1','inputs_sha256':hashes,
        'scope':'Train + Validation; taken trades for MAE/MFE, eligible candidates for paired comparison',
        'timeframe':'1H OHLC, up to actual exit bar inclusive',
        'intrabar_caveat':'For TP/SL exits, exit-bar extremes include unknown post-exit action; do not interpret as pre-exit MAE/MFE',
        'paired_caveat':'Cohort of same eligible event_ids for each multiple; not a portfolio backtest',
        'test_loaded':False,'re_simulation':False,'training':False}
    (output / 'manifest.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Rapports créés:', output)
    print('Aucun Test lu. Aucun trade recalculé. Aucune optimisation.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenarios', type=Path, default=Path('data/evaluation/v31-structural-tp'))
    ap.add_argument('--splits', type=Path, default=Path('data/splits'))
    ap.add_argument('--output', type=Path, default=Path('data/evaluation/v31-excursion-diagnostic'))
    a = ap.parse_args()
    run(a.scenarios, a.splits, a.output)


if __name__ == '__main__':
    main()

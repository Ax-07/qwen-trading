from __future__ import annotations

"""Audit read-only des coûts du backtest événementiel V1.

Validation seule, sauf option explicite --include-test.
On conserve les signaux, règles et dates d'exécution du moteur V1.
Chaque scénario est rejoué intégralement et non dérivé d'une simple
soustraction des frais au PnL déjà composé.

Le point mort est calculé en faisant varier le coût total par côté,
en conservant la proportion frais/slippage 5:2. Ce point mort n'est
PAS une recommandation tarifaire ou un paramètre de stratégie.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ENGINE_PATH = BASE_DIR / "backtest_event_driven_v1.py"
OUTPUT_ROOT = Path("data/evaluation/backtest-cost-sensitivity-v1")
COST_GRID = (0.0, 2.0, 5.0, 7.0, 10.0)


def get_engine():
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(
            f"Moteur introuvable : {ENGINE_PATH}\n"
            "Place les deux scripts dans scripts/."
        )
    spec = importlib.util.spec_from_file_location("event_backtest_engine_v1", ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--initial-equity", type=float, default=10000.0)
    ap.add_argument("--allocation", type=float, default=1.0)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--tp-atr", type=float, default=1.5)
    ap.add_argument("--sl-atr", type=float, default=1.0)
    ap.add_argument("--fee-weight", type=float, default=5.0,
                    help="Poids frais dans coût total par côté; défaut 5")
    ap.add_argument("--slippage-weight", type=float, default=2.0,
                    help="Poids slippage dans coût total par côté; défaut 2")
    args = ap.parse_args()
    if (args.initial_equity <= 0 or not 0 < args.allocation <= 1
        or args.horizon < 1 or args.tp_atr <= 0 or args.sl_atr <= 0
        or args.fee_weight < 0 or args.slippage_weight < 0
        or args.fee_weight + args.slippage_weight <= 0):
        ap.error("Paramètres invalides")
    return args


def scenario_args(config, cost_bps):
    weight_sum = config.fee_weight + config.slippage_weight
    return SimpleNamespace(
        horizon=config.horizon,
        tp_atr=config.tp_atr,
        sl_atr=config.sl_atr,
        initial_equity=config.initial_equity,
        allocation=config.allocation,
        fee_bps=cost_bps * config.fee_weight / weight_sum,
        slippage_bps=cost_bps * config.slippage_weight / weight_sum,
    )


def replay(engine, split, strategy, signals, market, config, cost_bps):
    args = scenario_args(config, cost_bps)
    trades, summary = engine.run(split, strategy, signals, market, args)
    return trades, summary, args


def fingerprint(trades):
    if trades.empty:
        return []
    return list(zip(
        trades["signal_time"].astype(str),
        trades["entry_time"].astype(str),
        trades["exit_time"].astype(str),
        trades["side"],
        trades["reason"],
    ))


def find_break_even(engine, split, strategy, signals, market, config,
                    reference_fingerprint, zero_return):
    # Equity finale décroit lorsque coût par côté augmente à trades constants.
    # Si déjà négatif à 0 bps, aucun coût >=0 ne permet de revenir à 0%.
    if zero_return < -1e-10:
        return None, "NEGATIVE_EVEN_WITH_ZERO_COST"
    if abs(zero_return) <= 1e-10:
        return 0.0, "BREAK_EVEN_AT_ZERO_COST"

    low, high = 0.0, 10.0
    for _ in range(25):
        t, s, _ = replay(engine, split, strategy, signals, market, config, high)
        if fingerprint(t) != reference_fingerprint:
            return None, "TRADE_SEQUENCE_CHANGED"
        if s["return_pct"] <= 0:
            break
        high *= 2
    else:
        return None, "NOT_BRACKETED"

    for _ in range(28):
        mid = (low + high) / 2
        t, s, _ = replay(engine, split, strategy, signals, market, config, mid)
        if fingerprint(t) != reference_fingerprint:
            return None, "TRADE_SEQUENCE_CHANGED"
        if s["return_pct"] > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2, "ESTIMATED"


def main():
    config = parse_args()
    engine = get_engine()
    splits = ("VALIDATION", "TEST") if config.include_test else ("VALIDATION",)
    output = OUTPUT_ROOT / ("validation_and_test" if config.include_test
                             else "validation_only")
    market = engine.load_market()
    results = []
    break_evens = []

    print("=" * 78)
    print("QWEN V2 - AUDIT DE SENSIBILITE AUX COUTS / BACKTEST EVENEMENTIEL V1")
    print("=" * 78)
    print("Coût total = frais + slippage par côté (bps)")
    print(f"Scénarios: {list(COST_GRID)} bps | ratio frais/slippage: "
          f"{config.fee_weight:g}:{config.slippage_weight:g}")
    print("Aucun réentraînement. Validation seule par défaut.\n")

    for split in splits:
        signals = engine.load_signals(split, market)
        print(f"{split}: {len(signals)} observations")
        for strategy in ("QWEN", "HEURISTIC"):
            reference_trades = None
            reference_fingerprint = None
            zero_return = None
            for cost in COST_GRID:
                trades, summary, args = replay(
                    engine, split, strategy, signals, market, config, cost
                )
                fp = fingerprint(trades)
                if reference_trades is None:
                    reference_trades = trades
                    reference_fingerprint = fp
                    zero_return = summary["return_pct"]
                if fp != reference_fingerprint:
                    raise RuntimeError(
                        f"La séquence des trades a changé pour {split}/{strategy} "
                        f"à {cost} bps : comparaison invalidée"
                    )
                if len(trades):
                    gross_mean = float(trades["return_gross_pct"].mean())
                    net_mean = float(trades["return_net_pct"].mean())
                    long_count = int((trades["side"] == "LONG").sum())
                    short_count = int((trades["side"] == "SHORT").sum())
                else:
                    gross_mean = net_mean = float("nan")
                    long_count = short_count = 0
                results.append({
                    "split": split,
                    "strategy": strategy,
                    "cost_bps_per_side": cost,
                    "fee_bps_per_side": args.fee_bps,
                    "slippage_bps_per_side": args.slippage_bps,
                    "round_trip_nominal_cost_bps": 2 * cost,
                    "trades": summary["trades"],
                    "long_trades": long_count,
                    "short_trades": short_count,
                    "gross_mean_pct": gross_mean,
                    "net_mean_pct": net_mean,
                    "ending_equity": summary["ending_equity"],
                    "return_pct": summary["return_pct"],
                    "max_dd_exit_only_pct": summary["max_drawdown_exit_only_pct"],
                    "win_rate_net_pct": summary["win_rate_net_pct"],
                    "ignored_busy_signals": summary["ignored_busy_signals"],
                })
                print(f"  {strategy:10s} {cost:4.1f} bps/côté : "
                      f"{summary['trades']:3d} trades | "
                      f"retour {summary['return_pct']:+8.3f}% | "
                      f"DD {summary['max_drawdown_exit_only_pct']:+8.3f}% | "
                      f"capital {summary['ending_equity']:,.2f}")

            point, status = find_break_even(
                engine, split, strategy, signals, market, config,
                reference_fingerprint, zero_return
            )
            break_evens.append({
                "split": split, "strategy": strategy,
                "break_even_total_cost_bps_per_side": point,
                "status": status,
            })
            if point is None:
                print(f"  Point mort {strategy}: aucun coût >= 0 bps "
                      f"({status})")
            else:
                print(f"  Point mort {strategy}: {point:.4f} bps/côté "
                      f"({status})")

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output / "sensitivity.csv", index=False)
    pd.DataFrame(break_evens).to_csv(output / "break_even.csv", index=False)
    (output / "assumptions.json").write_text(
        json.dumps({
            **vars(config),
            "cost_scenarios_bps_per_side": list(COST_GRID),
            "fee_slippage_ratio": [config.fee_weight, config.slippage_weight],
            "base_engine": str(ENGINE_PATH),
            "notes": (
                "Séquence d'opérations invariantes aux coûts; "
                "point mort interpolé par resimulation; "
                "P&L synthétique sans financement, liquidité, mark-to-market "
                "intra-trade ou garanties de fill."
            ),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nFichiers écrits : {output.resolve()}")
    print("Audit de sensibilité, non stratégie rentable ni tuning du Test.")


if __name__ == "__main__":
    main()

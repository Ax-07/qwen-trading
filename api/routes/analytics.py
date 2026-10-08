
from functools import lru_cache
from pathlib import Path
from typing import Literal

import polars as pl
from fastapi import APIRouter, HTTPException


ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "validation": (
        ROOT / "data" / "evaluation"
        / "qwen3.5-9b-trading-v2-validation"
        / "results.parquet"
    ),
    "test": (
        ROOT / "data" / "evaluation"
        / "qwen3.5-9b-trading-v2-fast"
        / "results_fast.parquet"
    ),
}

Dataset = Literal["validation", "test"]

TP_R = 1.5
SL_R = -1.0

router = APIRouter(
    prefix="/api/analytics",
    tags=["analytics"],
)


@lru_cache(maxsize=2)
def load_evaluation(dataset: str) -> pl.DataFrame:
    path = FILES[dataset]

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset} introuvable",
        )

    return pl.read_parquet(path).sort("timestamp")


def outcome_for(row: dict, decision: str) -> str:
    if decision == "NO_TRADE":
        return "NO_TRADE"

    if decision == "LONG_BIAS":
        value = row["long_outcome_12h"]
    elif decision == "SHORT_BIAS":
        value = row["short_outcome_12h"]
    else:
        return "UNKNOWN"

    if value is None:
        return "UNKNOWN"

    if value == 1:
        return "TP"

    if value == -1:
        return "SL"

    if value == 0:
        return "UNRESOLVED"

    return "UNKNOWN"


def trade_stats(rows: list[dict], column: str) -> dict:
    outcomes = [
        (row[column], outcome_for(row, row[column]))
        for row in rows
    ]

    long_count = sum(
        decision == "LONG_BIAS"
        for decision, _ in outcomes
    )

    short_count = sum(
        decision == "SHORT_BIAS"
        for decision, _ in outcomes
    )

    no_trade = sum(
        decision == "NO_TRADE"
        for decision, _ in outcomes
    )

    tp = sum(
        result == "TP"
        for _, result in outcomes
    )

    sl = sum(
        result == "SL"
        for _, result in outcomes
    )

    unresolved = sum(
        result == "UNRESOLVED"
        for _, result in outcomes
    )

    unknown = sum(
        result == "UNKNOWN"
        for _, result in outcomes
    )

    resolved = tp + sl
    directional = long_count + short_count

    total_r = tp * TP_R + sl * SL_R

    return {
        "samples": len(rows),
        "long": long_count,
        "short": short_count,
        "no_trade": no_trade,
        "directional_signals": directional,
        "tp": tp,
        "sl": sl,
        "unresolved": unresolved,
        "unknown": unknown,
        "resolved": resolved,
        "coverage_pct": (
            round(100 * directional / len(rows), 2)
            if rows else None
        ),
        "win_rate_pct": (
            round(100 * tp / resolved, 2)
            if resolved else None
        ),
        "total_r": round(total_r, 4),
        "expected_r": (
            round(total_r / resolved, 4)
            if resolved else None
        ),
    }


def agreement_stats(rows: list[dict]) -> dict:
    total = len(rows)

    agreements = sum(
        row["qwen_prediction"]
        == row["heuristic_prediction"]
        for row in rows
    )

    return {
        "samples": total,
        "agreements": agreements,
        "disagreements": total - agreements,
        "agreement_pct": (
            round(100 * agreements / total, 2)
            if total else None
        ),
    }


def disagreement_stats(rows: list[dict]) -> dict:
    different = [
        row for row in rows
        if row["qwen_prediction"]
        != row["heuristic_prediction"]
    ]

    qwen_trades = trade_stats(
        different,
        "qwen_prediction",
    )

    heuristic_trades = trade_stats(
        different,
        "heuristic_prediction",
    )

    # Analyse de l'ensemble des désaccords.
    # Les abstentions ne sont pas considérées
    # comme des TP ou des SL.
    qwen_only_tp = 0
    heuristic_only_tp = 0
    both_tp = 0
    both_sl = 0

    qwen_trade_heuristic_abstains = 0
    heuristic_trade_qwen_abstains = 0
    both_trade_different_direction = 0

    for row in different:
        q_decision = row["qwen_prediction"]
        h_decision = row["heuristic_prediction"]

        q_result = outcome_for(row, q_decision)
        h_result = outcome_for(row, h_decision)

        if (
            q_decision != "NO_TRADE"
            and h_decision == "NO_TRADE"
        ):
            qwen_trade_heuristic_abstains += 1

        if (
            q_decision == "NO_TRADE"
            and h_decision != "NO_TRADE"
        ):
            heuristic_trade_qwen_abstains += 1

        if (
            q_decision in ("LONG_BIAS", "SHORT_BIAS")
            and h_decision in ("LONG_BIAS", "SHORT_BIAS")
        ):
            both_trade_different_direction += 1

        if q_result == "TP" and h_result != "TP":
            qwen_only_tp += 1

        if h_result == "TP" and q_result != "TP":
            heuristic_only_tp += 1

        if q_result == "TP" and h_result == "TP":
            both_tp += 1

        if q_result == "SL" and h_result == "SL":
            both_sl += 1

    return {
        "samples": len(different),
        "qwen": qwen_trades,
        "heuristic": heuristic_trades,
        "qwen_trade_heuristic_abstains": (
            qwen_trade_heuristic_abstains
        ),
        "heuristic_trade_qwen_abstains": (
            heuristic_trade_qwen_abstains
        ),
        "both_trade_different_direction": (
            both_trade_different_direction
        ),
        "qwen_tp_other_not_tp": qwen_only_tp,
        "heuristic_tp_other_not_tp": heuristic_only_tp,
        "both_tp": both_tp,
        "both_sl": both_sl,
    }


@router.get("/summary")
def analytics_summary(
    dataset: Dataset = "test",
):
    """
    Analyse descriptive des prédictions V2.

    Pas de backtest exécuté :
    - pas de gestion de positions
    - pas de frais
    - pas de slippage
    - signaux potentiellement chevauchants

    Le calcul en R utilise uniquement
    les outcomes 12H déjà étiquetés.
    """

    df = load_evaluation(dataset)
    rows = df.to_dicts()

    return {
        "dataset": dataset,
        "samples": len(rows),
        "parameters": {
            "tp_atr": 1.5,
            "sl_atr": 1.0,
            "horizon_hours": 12,
            "tp_r": TP_R,
            "sl_r": SL_R,
            "expected_r_denominator": "resolved",
        },
        "qwen": trade_stats(
            rows,
            "qwen_prediction",
        ),
        "heuristic": trade_stats(
            rows,
            "heuristic_prediction",
        ),
        "agreement": agreement_stats(rows),
        "disagreements": disagreement_stats(rows),
        "by_direction": {
            "qwen": {
                "long": trade_stats(
                    [
                        row for row in rows
                        if row["qwen_prediction"]
                        == "LONG_BIAS"
                    ],
                    "qwen_prediction",
                ),
                "short": trade_stats(
                    [
                        row for row in rows
                        if row["qwen_prediction"]
                        == "SHORT_BIAS"
                    ],
                    "qwen_prediction",
                ),
            },
            "heuristic": {
                "long": trade_stats(
                    [
                        row for row in rows
                        if row["heuristic_prediction"]
                        == "LONG_BIAS"
                    ],
                    "heuristic_prediction",
                ),
                "short": trade_stats(
                    [
                        row for row in rows
                        if row["heuristic_prediction"]
                        == "SHORT_BIAS"
                    ],
                    "heuristic_prediction",
                ),
            },
        },
        "disclaimer": (
            "Statistiques historiques par signal. "
            "Ne représentent pas un backtest exécutable."
        ),
    }

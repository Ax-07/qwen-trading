
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import polars as pl
from fastapi import APIRouter, HTTPException, Query


ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = (
    ROOT
    / "data"
    / "evaluation"
    / "qwen3.5-9b-trading-v2-validation"
    / "results.parquet"
)

TEST_PATH = (
    ROOT
    / "data"
    / "evaluation"
    / "qwen3.5-9b-trading-v2-fast"
    / "results_fast.parquet"
)

PREDICTION_FILES = {
    "validation": VALIDATION_PATH,
    "test": TEST_PATH,
}

DatasetName = Literal["train", "validation", "test"]

router = APIRouter(
    prefix="/api",
    tags=["signals"],
)


@lru_cache(maxsize=2)
def load_signal_data(dataset: str) -> pl.DataFrame:
    path = PREDICTION_FILES[dataset]

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Fichier de prédictions introuvable",
        )

    return pl.read_parquet(path).sort("timestamp")


def outcome_result(value):
    if value is None:
        return "UNKNOWN"

    if value == 1:
        return "TP"

    if value == -1:
        return "SL"

    if value == 0:
        return "UNRESOLVED"

    return "UNKNOWN"


def result_for_decision(
    decision: str,
    long_outcome,
    short_outcome,
):
    if decision == "LONG_BIAS":
        return outcome_result(long_outcome)

    if decision == "SHORT_BIAS":
        return outcome_result(short_outcome)

    return "NO_TRADE"


def parse_utc(value: datetime | None):
    if value is None:
        return None

    if value.tzinfo is None:
        raise HTTPException(
            status_code=422,
            detail="Les dates doivent contenir un fuseau horaire",
        )

    return value.astimezone(timezone.utc)



@router.get("/signals")
def get_signals(
    dataset: DatasetName = "test",
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = Query(default=2000, ge=1, le=5000),
):
    """
    Signaux historiques Qwen / heuristique / target.

    Les outcomes sont uniquement utilisés
    pour l'évaluation a posteriori.
    """

    if dataset == "train":
        return {
            "dataset": dataset,
            "count": 0,
            "signals": [],
            "message": "Aucune prédiction V2 pour Train",
        }

    start_utc = parse_utc(start)
    end_utc = parse_utc(end)

    if start_utc and end_utc and start_utc > end_utc:
        raise HTTPException(
            status_code=422,
            detail="start doit être <= end",
        )

    df = load_signal_data(dataset)

    if start_utc is not None:
        df = df.filter(
            pl.col("timestamp") >= start_utc
        )

    if end_utc is not None:
        df = df.filter(
            pl.col("timestamp") <= end_utc
        )

    df = df.head(limit)

    signals = []

    for row in df.iter_rows(named=True):
        long_outcome = row["long_outcome_12h"]
        short_outcome = row["short_outcome_12h"]

        qwen = row["qwen_prediction"]
        heuristic = row["heuristic_prediction"]
        target = row["target"]

        signals.append({
            "time": int(
                row["timestamp"].timestamp()
            ),
            "qwen": qwen,
            "heuristic": heuristic,
            "target": target,
            "bias_score": row["bias_score"],
            "qwen_result": result_for_decision(
                qwen,
                long_outcome,
                short_outcome,
            ),
            "heuristic_result": result_for_decision(
                heuristic,
                long_outcome,
                short_outcome,
            ),
            "target_result": result_for_decision(
                target,
                long_outcome,
                short_outcome,
            ),
            "disagreement": qwen != heuristic,
        })

    return {
        "dataset": dataset,
        "count": len(signals),
        "signals": signals,
    }

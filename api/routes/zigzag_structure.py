from __future__ import annotations
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal
import polars as pl
from fastapi import APIRouter, HTTPException, Query

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/evaluation/v31-zigzag-causal"
router = APIRouter(prefix="/api", tags=["zigzag"])

@lru_cache(maxsize=2)
def load(dataset: str) -> pl.DataFrame:
    path = DATA / f"{dataset}_zigzag_events.parquet"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"ZigZag introuvable: {path}")
    return pl.read_parquet(path)

@router.get("/zigzag-structure")
def zigzag_structure(
    dataset: Literal["train", "validation", "test"] = "validation",
    timeframe: Literal["1h", "4h", "1d"] = "1h",
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = Query(default=3000, ge=1, le=10000),
):
    if dataset == "test":
        return {"dataset": dataset, "timeframe": timeframe, "available": False,
                "events": [], "count": 0, "message": "Test non traite pour V3.1"}
    for t in (start, end):
        if t is not None and t.tzinfo is None:
            raise HTTPException(status_code=422, detail="Fuseau UTC obligatoire")
    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=422, detail="start > end")
    df = load(dataset)
    required = {"timeframe", "confirmation_at", "pivot_at", "kind", "label", "price"}
    if not required.issubset(set(df.columns)):
        raise HTTPException(status_code=409, detail="Schema ZigZag incompatible")
    df = df.filter(pl.col("timeframe") == timeframe)
    if start is not None:
        df = df.filter(pl.col("confirmation_at") >= start.astimezone(timezone.utc))
    if end is not None:
        df = df.filter(pl.col("confirmation_at") <= end.astimezone(timezone.utc) + timedelta(hours=1))
    events = []
    for item in df.sort("confirmation_at").head(limit).iter_rows(named=True):
        if item["kind"] not in ("HIGH", "LOW"):
            continue
        at = item["confirmation_at"]
        events.append({
            "time": int((at - timedelta(hours=1)).timestamp()),
            "known_at": at.isoformat(),
            "pivot_at": item["pivot_at"].isoformat(),
            "kind": "high" if item["kind"] == "HIGH" else "low",
            "label": item["label"],
            "price": float(item["price"]),
        })
    return {"dataset": dataset, "timeframe": timeframe, "method": "zigzag",
            "available": True, "count": len(events), "events": events,
            "notice": "Pivot retrospectif; confirmation seule utilisable en temps reel"}

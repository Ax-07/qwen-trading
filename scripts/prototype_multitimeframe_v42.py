"""V4.2 : prototype causal 1m -> bougies closes 5m/15m/1h/4h.

Sans reseau, sans entrainement, sans donnees Test. Entrée Parquet optionnelle
avec colonnes timestamp, open, high, low, close, volume (+ quote_volume optionnel).
Le timestamp est l'OUVERTURE UTC de la bougie 1m.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

FREQS = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
REQUIRED = ("timestamp", "open", "high", "low", "close", "volume")


def validate_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    if any(k not in frame.columns for k in REQUIRED):
        raise ValueError(f"Colonnes requises : {REQUIRED}")
    f = frame.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True, errors="raise")
    if f["timestamp"].isna().any():
        raise ValueError("Timestamp NaT")
    if f.timestamp.duplicated().any() or not f.timestamp.is_monotonic_increasing:
        raise ValueError("Timestamps non uniques ou non croissants")
    if ((f.timestamp.dt.second != 0) | (f.timestamp.dt.microsecond != 0) |
            (f.timestamp.dt.nanosecond != 0)).any():
        raise ValueError("Bougies non alignees a la minute")
    for c in REQUIRED[1:]:
        f[c] = pd.to_numeric(f[c], errors="raise")
        if not np.isfinite(f[c]).all():
            raise ValueError(f"Valeurs non finies : {c}")
    if (f[["open", "high", "low", "close"]] <= 0).any().any() or (f.volume < 0).any():
        raise ValueError("Prix ou volumes invalides")
    if not ((f.high >= f[["open", "close", "low"]].max(axis=1)) &
            (f.low <= f[["open", "close", "high"]].min(axis=1))).all():
        raise ValueError("OHLC incoherents")
    if "quote_volume" in f:
        f["quote_volume"] = pd.to_numeric(f.quote_volume, errors="raise")
        if (~np.isfinite(f.quote_volume) | (f.quote_volume < 0)).any():
            raise ValueError("Quote volume invalide")
    return f


def completed_bars(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Garde UNIQUEMENT des blocs UTC entiers de `minutes` bougies 1m.

    timestamp=open_time; known_at=heure de cloture exclue (UTC).
    Aucune interpolation. Un bloc contenant une minute absente est rejete.
    """
    if minutes < 1:
        raise ValueError("minutes doit etre positif")
    f = validate_minutes(frame)
    cols = ["timestamp", "known_at", "open", "high", "low", "close", "volume", "count_1m"]
    if "quote_volume" in f:
        cols.insert(-1, "quote_volume")
    if f.empty:
        return pd.DataFrame(columns=cols)
    f = f.set_index("timestamp")
    rule = f"{minutes}min"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "quote_volume" in f:
        agg["quote_volume"] = "sum"
    gb = f.resample(rule, label="left", closed="left", origin="epoch")
    bars = gb.agg(agg)
    bars["count_1m"] = gb.size()
    bars = bars.loc[bars.count_1m == minutes].reset_index()
    bars.insert(1, "known_at", bars.timestamp + pd.Timedelta(minutes=minutes))
    return bars[cols]


def build_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par 5m terminee, contexte strictement connu a decision_at.

    L'absence de contexte complet produit des NaN, pas une valeur future.
    """
    bars = {k: completed_bars(frame, n) for k, n in FREQS.items()}
    decisions = bars["5m"][["timestamp", "known_at", "close"]].rename(
        columns={"timestamp": "bar_open_5m", "known_at": "decision_at", "close": "close_5m"})
    decisions = decisions.sort_values("decision_at")
    for name in ("15m", "1h", "4h"):
        h = bars[name][["known_at", "close", "high", "low"]].rename(
            columns={"known_at": f"known_at_{name}", "close": f"close_{name}",
                     "high": f"high_{name}", "low": f"low_{name}"})
        decisions = pd.merge_asof(
            decisions, h.sort_values(f"known_at_{name}"), left_on="decision_at",
            right_on=f"known_at_{name}", direction="backward", allow_exact_matches=True)
    return decisions.reset_index(drop=True)


def synthetic_minutes(count: int = 520, start: str = "2024-01-01T00:00:00Z") -> pd.DataFrame:
    idx = pd.date_range(start, periods=count, freq="min", tz="UTC")
    base = 100.0 + np.arange(count) * .01
    return pd.DataFrame({"timestamp": idx, "open": base, "high": base + .2,
                         "low": base - .2, "close": base + .05, "volume": 10.0})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, help="Parquet 1m existant, sinon source synthetique")
    p.add_argument("--output", type=Path, help="CSV de snapshots; refus d'ecraser")
    args = p.parse_args()
    f = pd.read_parquet(args.input) if args.input else synthetic_minutes()
    result = build_snapshots(f)
    print(f"minutes={len(f)} snapshots_5m={len(result)}")
    print(result.tail(5).to_string(index=False))
    if args.output:
        if args.output.exists():
            p.error(f"Sortie existante, refus d'ecrasement : {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Ecrit : {args.output}")

if __name__ == "__main__":
    main()

from __future__ import annotations
"""Structure de marché causale 1H/4H/1D, descriptive uniquement.

Pivots fractals : `left` bougies avant et `right` bougies après.
Un pivot à t n'apparaît dans les features qu'à la clôture de t+right.
Aucun label, ATR, test historique ni règle d'entrée/SL/TP utilisés.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SOURCE = Path("data/splits")
DEFAULT_OUTPUT = Path("data/evaluation/v31-market-structure")
TIMEFRAMES = {"1h": "1h", "4h": "4h", "1d": "1D"}
OHLC = ["open", "high", "low", "close"]
EVENT_COLUMNS = ("pivot_high_confirmed", "pivot_low_confirmed", "bos_up", "bos_down", "choch_up", "choch_down")


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--left", type=int, default=2)
    ap.add_argument("--right", type=int, default=2)
    return ap.parse_args()


def normalize(df):
    if "timestamp" not in df:
        raise ValueError("Colonne timestamp absente")
    if any(c not in df for c in OHLC):
        raise ValueError("OHLC incomplet")
    out = df[["timestamp", *OHLC]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    if out["timestamp"].duplicated().any():
        raise ValueError("Timestamp duplique")
    if out[OHLC].isna().any().any():
        raise ValueError("OHLC comporte NaN")
    if not (out.high.ge(out.low) & out.high.ge(out[["open", "close"]].max(axis=1)) &
            out.low.le(out[["open", "close"]].min(axis=1))).all():
        raise ValueError("OHLC incoherent")
    return out


def aggregate_completed(df, timeframe):
    """Barre HTF indexee par son HORODATAGE DE FIN (pas son ouverture)."""
    frame = df.set_index("timestamp")[OHLC]
    if timeframe == "1h":
        bars = frame.copy()
        bars.index = bars.index + pd.Timedelta(hours=1)
        return bars
    freq = TIMEFRAMES[timeframe]
    groups = frame.resample(freq, label="right", closed="left")
    bars = groups.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    counts = groups["close"].count()
    needed = 4 if timeframe == "4h" else 24
    # Une plage HTF incomplete est ignoree (debut/fin du split ou trou de donnees).
    bars = bars[counts == needed].dropna()
    return bars


def pivot_features(bars, left=2, right=2):
    """Aucun pivot n'est divulgue avant `right` clotures ulterieures."""
    if left < 1 or right < 1:
        raise ValueError("left et right doivent etre >= 1")
    arr_h = bars.high.to_numpy(dtype=float)
    arr_l = bars.low.to_numpy(dtype=float)
    arr_c = bars.close.to_numpy(dtype=float)
    n = len(bars)
    records = []
    last_h = last_l = np.nan
    prior_h = prior_l = np.nan
    last_h_class = last_l_class = "NONE"
    last_h_at = last_l_at = pd.NaT
    broken_h = broken_l = False
    regime = 0
    confirmed = {}  # key = position de confirmation
    for i in range(left, n - right):
        # Un pivot unique dans sa fenetre evite les egalites ambiguës.
        hs = arr_h[i-left:i+right+1]
        ls = arr_l[i-left:i+right+1]
        high_ok = bool(np.isfinite(hs).all() and np.argmax(hs) == left and (hs == hs[left]).sum() == 1)
        low_ok = bool(np.isfinite(ls).all() and np.argmin(ls) == left and (ls == ls[left]).sum() == 1)
        if high_ok or low_ok:
            confirmed[i+right] = (i, high_ok, low_ok)
    for i in range(n):
        new_h = new_l = False
        if i in confirmed:
            origin, high_ok, low_ok = confirmed[i]
            if high_ok:
                prior_h = last_h
                last_h = arr_h[origin]
                last_h_at = bars.index[origin]
                last_h_class = ("HH" if last_h > prior_h else "LH" if last_h < prior_h else "EH") if np.isfinite(prior_h) else "FIRST"
                broken_h = False
                new_h = True
            if low_ok:
                prior_l = last_l
                last_l = arr_l[origin]
                last_l_at = bars.index[origin]
                last_l_class = ("HL" if last_l > prior_l else "LL" if last_l < prior_l else "EL") if np.isfinite(prior_l) else "FIRST"
                broken_l = False
                new_l = True
        # Une cassure est definie par cloture au-dela du pivot confirme,
        # une seule fois par niveau; classe CHoCH selon le regime PRECEDENT.
        bos_up = bool(np.isfinite(last_h) and arr_c[i] > last_h and not broken_h)
        bos_down = bool(np.isfinite(last_l) and arr_c[i] < last_l and not broken_l)
        choch_up = bool(bos_up and regime < 0)
        choch_down = bool(bos_down and regime > 0)
        if bos_up:
            broken_h = True
            regime = 1
        if bos_down:
            broken_l = True
            regime = -1
        records.append({
            "available_at": bars.index[i],
            "last_swing_high": last_h, "last_swing_low": last_l,
            "swing_high_at": last_h_at, "swing_low_at": last_l_at,
            "high_structure": last_h_class, "low_structure": last_l_class,
            "pivot_high_confirmed": new_h, "pivot_low_confirmed": new_l,
            "bos_up": bos_up, "bos_down": bos_down,
            "choch_up": choch_up, "choch_down": choch_down,
            "structure_regime": regime,
        })
    return pd.DataFrame.from_records(records)


def build_split(df, left=2, right=2):
    candles = normalize(df)
    signal = candles[["timestamp", "close"]].copy()
    signal["known_at"] = signal["timestamp"] + pd.Timedelta(hours=1)
    # Les Parquet peuvent exposer us tandis que resample produit ms/ns.
    # Forcer une unite unique avant merge_asof (sans changer les instants).
    signal["known_at"] = signal["known_at"].dt.as_unit("ns")
    result = signal
    for tf in TIMEFRAMES:
        bars = aggregate_completed(candles, tf)
        features = pivot_features(bars, left, right)
        features = features.rename(columns={c: f"{tf}_{c}" for c in features if c != "available_at"})
        features = features.rename(columns={"available_at": f"{tf}_available_at"})
        features[f"{tf}_available_at"] = pd.to_datetime(
            features[f"{tf}_available_at"], utc=True
        ).dt.as_unit("ns")
        result = pd.merge_asof(
            result.sort_values("known_at"),
            features.sort_values(f"{tf}_available_at"),
            left_on="known_at", right_on=f"{tf}_available_at",
            direction="backward", allow_exact_matches=True,
        )
        col = f"{tf}_available_at"
        # Les etats sont maintenus par merge_asof; les EVENEMENTS ne doivent
        # apparaitre qu'une fois, a la bougie 1H de confirmation exacte.
        is_event_bar = result[col].eq(result["known_at"])
        for event in EVENT_COLUMNS:
            event_col = f"{tf}_{event}"
            result[event_col] = result[event_col].eq(True).fillna(False) & is_event_bar
        mask = result[col].notna()
        if not (result.loc[mask, col] <= result.loc[mask, "known_at"]).all():
            raise AssertionError(f"Fuite future {tf}")
        result[f"{tf}_dist_swing_high_pct"] = (
            (result["close"] / result[f"{tf}_last_swing_high"] - 1.0) * 100
        )
        result[f"{tf}_dist_swing_low_pct"] = (
            (result["close"] / result[f"{tf}_last_swing_low"] - 1.0) * 100
        )
    if len(result) != len(candles) or result.timestamp.duplicated().any():
        raise AssertionError("Perte ou duplication de bougies")
    return result


def main():
    args = parser()
    if args.left < 1 or args.right < 1:
        raise SystemExit("--left / --right doivent etre >=1")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "validation"):
        inp = args.input / f"{split}.parquet"
        if not inp.exists():
            raise FileNotFoundError(inp)
        result = build_split(pd.read_parquet(inp), args.left, args.right)
        out = args.output / f"{split}_structure.parquet"
        result.to_parquet(out, index=False)
        summary[split] = {
            "rows": len(result),
            "first_timestamp": str(result.timestamp.min()),
            "last_timestamp": str(result.timestamp.max()),
            "confirmed_pivots": {
                tf: int(result[f"{tf}_pivot_high_confirmed"].fillna(False).sum() +
                        result[f"{tf}_pivot_low_confirmed"].fillna(False).sum())
                for tf in TIMEFRAMES
            },
            "file": str(out)
        }
        print(f"{split.upper():10} {len(result):5} bougies | pivots: {summary[split]['confirmed_pivots']}")
    manifest = {"version": "v31-structure-pivots-v2-event-impulses", "left": args.left,
                "right": args.right, "source": str(args.input),
                "interpretation": "UTC, pivots confirmes au close de la barre t+right; pas de SL/TP",
                "splits": summary, "test_loaded": False}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Sortie: {args.output.resolve()}")
    print("Aucun Test lu; aucun entrainement; aucun SL/TP calcule.")


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Baseline probabiliste BTC/USDC 1H, diagnostic hors Test.

Entrées : data/splits/train.parquet et validation.parquet.
Cibles distinctes LONG/SHORT : SL (-1), NO_TOUCH (0), TP (+1).
NaN = ambigu OHLC ou horizon absent : exclu et comptabilisé.
Ne crée aucun ordre et ne modifie aucune donnée source.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_FILE = Path("data/splits/train.parquet")
VAL_FILE = Path("data/splits/validation.parquet")
OUT_DIR = Path("data/evaluation/tabular-baseline-v1")

# Choix fixé AVANT l'évaluation ; aucune optimisation sur Validation ou Test.
FEATURES = [
    "return_1h", "return_3h", "return_6h", "return_24h",
    "distance_ema20_pct", "distance_ema50_pct", "distance_ema200_pct",
    "rsi14", "atr14_pct", "volume_ratio",
    "volatility_24h", "volatility_72h",
    "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "distance_high_24h_pct", "distance_low_24h_pct",
    "distance_high_72h_pct", "distance_low_72h_pct",
    "ema20_50_spread_pct", "ema50_200_spread_pct",
    "trend_1h", "4h_trend", "1d_trend", "trend_alignment",
    "4h_rsi14", "4h_atr14_pct", "1d_rsi14", "1d_atr14_pct",
    "breakout_24h", "breakdown_24h",
]
CLASSES = [-1, 0, 1]
LABEL_NAMES = {-1: "SL", 0: "NO_TOUCH", 1: "TP"}
TP_R = 1.5
SL_R = 1.0


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"timestamp", "available_at", "sample_ready", "close", "ema20", "ema50", "long_outcome_12h", "short_outcome_12h", *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path}: colonnes absentes : {missing}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True)
    if df["timestamp"].isna().any() or df["timestamp"].duplicated().any():
        raise ValueError(f"{path}: timestamps nuls ou dupliqués")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{path}: ordre chronologique non respecté")
    if not df["sample_ready"].fillna(False).all():
        raise ValueError(f"{path}: contient des lignes non sample_ready")
    if not (df["available_at"] == df["timestamp"] + pd.Timedelta(hours=1)).all():
        raise ValueError(f"{path}: available_at incohérent")
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df


def valid_label_mask(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].isin(CLASSES)


def multiclass_brier(y: np.ndarray, prob: np.ndarray) -> float:
    indicators = (y[:, None] == np.array(CLASSES)[None, :]).astype(float)
    return float(np.mean(np.sum((prob - indicators) ** 2, axis=1)))


def class_stats(y: np.ndarray) -> dict:
    return {LABEL_NAMES[c]: int(np.sum(y == c)) for c in CLASSES}


def evaluate_side(side: str, train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    outcome_col = f"{side.lower()}_outcome_12h"
    train_mask = valid_label_mask(train, outcome_col)
    val_mask = valid_label_mask(val, outcome_col)
    tr = train.loc[train_mask].copy()
    va = val.loc[val_mask].copy()
    y_train = tr[outcome_col].astype(int).to_numpy()
    y_val = va[outcome_col].astype(int).to_numpy()
    if len(tr) < 50 or len(va) == 0:
        raise ValueError(f"{side}: trop peu de lignes utilisables")
    if set(np.unique(y_train)) != set(CLASSES):
        raise ValueError(f"{side}: classes train incomplètes : {np.unique(y_train)}")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, max_iter=2000, class_weight=None)),
    ])
    model.fit(tr[FEATURES], y_train)
    raw_prob = model.predict_proba(va[FEATURES])
    fitted_classes = model.named_steps["classifier"].classes_.tolist()
    prob = np.column_stack([raw_prob[:, fitted_classes.index(c)] for c in CLASSES])
    pred = np.array(CLASSES)[np.argmax(prob, axis=1)]

    # Baseline probabiliste sans features : fréquences naturelles du Train.
    prior = np.array([(y_train == c).mean() for c in CLASSES])
    prior_prob = np.tile(prior, (len(va), 1))

    result = {
        "side": side,
        "train_rows": len(train),
        "train_valid": len(tr),
        "train_excluded_ambiguous_or_missing": int((~train_mask).sum()),
        "validation_rows": len(val),
        "validation_valid": len(va),
        "validation_excluded_ambiguous_or_missing": int((~val_mask).sum()),
        "train_distribution": class_stats(y_train),
        "validation_distribution": class_stats(y_val),
        "logistic": {
            "accuracy": float(accuracy_score(y_val, pred)),
            "log_loss": float(log_loss(y_val, prob, labels=CLASSES)),
            "multiclass_brier": multiclass_brier(y_val, prob),
        },
        "train_prior_baseline": {
            "accuracy": float(np.mean(y_val == CLASSES[int(np.argmax(prior))])),
            "log_loss": float(log_loss(y_val, prior_prob, labels=CLASSES)),
            "multiclass_brier": multiclass_brier(y_val, prior_prob),
        },
    }

    rows = va[["timestamp", "available_at", outcome_col]].copy()
    rows = rows.rename(columns={outcome_col: "outcome"})
    rows["side"] = side
    for idx, c in enumerate(CLASSES):
        rows[f"p_{LABEL_NAMES[c].lower()}"] = prob[:, idx]
    rows["predicted_outcome"] = [LABEL_NAMES[int(c)] for c in pred]
    rows["proxy_expected_r"] = TP_R * rows["p_tp"] - SL_R * rows["p_sl"]

    # Rapport diagnostique à couverture fixe : candidats permis par la même
    # heuristique que V2. Ne pas optimiser de seuil en regardant le Test.
    score = (va["trend_1h"] + va["4h_trend"] + va["1d_trend"]).astype(int)
    score += (va["close"] > va["ema20"]).astype(int) - (va["close"] < va["ema20"]).astype(int)
    score += (va["ema20"] > va["ema50"]).astype(int) - (va["ema20"] < va["ema50"]).astype(int)
    score += va["rsi14"].between(52, 70, inclusive="both").astype(int)
    score -= va["rsi14"].between(30, 48, inclusive="both").astype(int)
    eligible = (score >= 2) if side == "LONG" else (score <= -2)
    rows["heuristic_eligible"] = eligible.to_numpy()
    # Seuil EV > 0 fixé par le payoff TP/SL ; non résolu encore non valorisé.
    rows["positive_proxy"] = rows["proxy_expected_r"] > 0
    rows["heuristic_and_positive_proxy"] = rows["heuristic_eligible"] & rows["positive_proxy"]

    summaries = []
    for name, mask in [
        ("ALL_VALID", np.ones(len(rows), dtype=bool)),
        ("HEURISTIC_ELIGIBLE", rows["heuristic_eligible"].to_numpy()),
        ("HEURISTIC_AND_PROXY_POSITIVE", rows["heuristic_and_positive_proxy"].to_numpy()),
    ]:
        ys = y_val[mask]
        wins = int((ys == 1).sum())
        losses = int((ys == -1).sum())
        unresolved = int((ys == 0).sum())
        n = len(ys)
        summaries.append({
            "side": side, "group": name, "signals": n,
            "wins": wins, "losses": losses, "no_touch": unresolved,
            "win_rate_resolved": wins / (wins + losses) if wins + losses else np.nan,
            "r_per_signal_zero_no_touch": (TP_R * wins - SL_R * losses) / n if n else np.nan,
        })
    return result, rows, pd.DataFrame(summaries)


def main() -> None:
    print("=" * 70)
    print("TABULAR BASELINE V1 - TRAIN / VALIDATION ONLY")
    print("=" * 70)
    train = load(TRAIN_FILE)
    val = load(VAL_FILE)
    if train["available_at"].max() + pd.Timedelta(hours=24) >= val["available_at"].min():
        raise ValueError("Purge chronologique insuffisante pour horizon 24h")
    print(f"TRAIN: {len(train)} | VALIDATION: {len(val)}")
    results, predictions, trade_summaries = [], [], []
    for side in ["LONG", "SHORT"]:
        r, p, s = evaluate_side(side, train, val)
        results.append(r)
        predictions.append(p)
        trade_summaries.append(s)
        print(f"\n{side}: Train valide={r['train_valid']}, Validation valide={r['validation_valid']}")
        print(f"  Fréquences Train: {r['train_distribution']}")
        print(f"  Fréquences Val:   {r['validation_distribution']}")
        print(f"  Log loss: logistic={r['logistic']['log_loss']:.4f}, prior={r['train_prior_baseline']['log_loss']:.4f}")
        print(f"  Brier multi: logistic={r['logistic']['multiclass_brier']:.4f}, prior={r['train_prior_baseline']['multiclass_brier']:.4f}")
    trade_summary = pd.concat(trade_summaries, ignore_index=True)
    print("\nDiagnostic par signaux (NON backtest, NO_TOUCH=0R provisoire):")
    print(trade_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    pd.concat(predictions, ignore_index=True).to_parquet(OUT_DIR / "validation_probabilities.parquet", index=False)
    trade_summary.to_csv(OUT_DIR / "validation_signal_summary.csv", index=False)
    print(f"\nFichiers écrits : {OUT_DIR.resolve()}")
    print("Aucun Test consulté. Aucune calibration sur Validation.")


if __name__ == "__main__":
    main()

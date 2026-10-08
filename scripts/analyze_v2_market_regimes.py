
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_FILE = Path(
    "data/processed/btc_usdc_1h_labeled.parquet"
)

RESULTS_FILES = {
    "VALIDATION": Path(
        "data/evaluation/"
        "qwen3.5-9b-trading-v2-validation/results.parquet"
    ),
    "TEST": Path(
        "data/evaluation/"
        "qwen3.5-9b-trading-v2-fast/results_fast.parquet"
    ),
}

OUTPUT_DIR = Path(
    "data/evaluation/"
    "qwen3.5-9b-trading-v2-market-regimes"
)

TP_R = 1.5
SL_R = 1.0

MIN_GROUP_SIZE = 30

FEATURE_COLUMNS = [
    "rsi14",
    "atr14_pct",
    "volatility_24h",
    "volatility_72h",
    "volume_ratio",
    "distance_ema20_pct",
    "distance_ema50_pct",
    "distance_ema200_pct",
    "trend_1h",
    "4h_trend",
    "1d_trend",
    "trend_alignment",
]

# Thresholds are fixed before inspecting regime results.
# These are exploratory bins, not trading rules.
REGIME_COLUMNS = [
    "trend_regime",
    "rsi_regime",
    "atr_regime",
    "volume_regime",
    "ema_regime",
    "bias_regime",
]


# ============================================================
# HELPERS
# ============================================================

def categorical_bin(series, bins, labels):
    return pd.cut(
        series,
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype("string").fillna("MISSING")


def add_regimes(df):
    df = df.copy()

    df["trend_regime"] = np.select(
        [
            df["trend_alignment"] >= 2,
            df["trend_alignment"] <= -2,
        ],
        [
            "BULLISH_ALIGNMENT",
            "BEARISH_ALIGNMENT",
        ],
        default="MIXED",
    )

    df["rsi_regime"] = categorical_bin(
        df["rsi14"],
        [-np.inf, 30, 45, 55, 70, np.inf],
        [
            "RSI_LE_30",
            "RSI_30_45",
            "RSI_45_55",
            "RSI_55_70",
            "RSI_GT_70",
        ],
    )

    df["atr_regime"] = categorical_bin(
        df["atr14_pct"],
        [-np.inf, 0.5, 1.0, 1.5, np.inf],
        [
            "ATR_LE_0_5",
            "ATR_0_5_1",
            "ATR_1_1_5",
            "ATR_GT_1_5",
        ],
    )

    df["volume_regime"] = categorical_bin(
        df["volume_ratio"],
        [-np.inf, 0.7, 1.5, np.inf],
        [
            "LOW_VOLUME",
            "NORMAL_VOLUME",
            "HIGH_VOLUME",
        ],
    )

    df["ema_regime"] = categorical_bin(
        df["distance_ema200_pct"],
        [-np.inf, -2, 0, 2, np.inf],
        [
            "BELOW_EMA200_FAR",
            "BELOW_EMA200_NEAR",
            "ABOVE_EMA200_NEAR",
            "ABOVE_EMA200_FAR",
        ],
    )

    df["bias_regime"] = categorical_bin(
        df["bias_score"],
        [-np.inf, -4, -2, 1, 3, np.inf],
        [
            "STRONG_SHORT",
            "MODERATE_SHORT",
            "NEUTRAL",
            "MODERATE_LONG",
            "STRONG_LONG",
        ],
    )

    return df


def load_split(split, path, features):
    print(f"\nLoading {split}: {path}")

    results = pd.read_parquet(path).copy()

    required = [
        "timestamp",
        "qwen_prediction",
        "heuristic_prediction",
        "bias_score",
        "long_outcome_12h",
        "short_outcome_12h",
    ]

    missing = [
        col for col in required
        if col not in results.columns
    ]

    if missing:
        raise ValueError(
            f"{split}: missing columns: {missing}"
        )

    if results["timestamp"].isna().any():
        raise ValueError(f"{split}: null timestamps")

    if results["timestamp"].duplicated().any():
        raise ValueError(f"{split}: duplicate timestamps")

    results["timestamp"] = pd.to_datetime(
        results["timestamp"], utc=True
    )

    merged = results.merge(
        features,
        on="timestamp",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_matches = int(
        (merged["_merge"] != "both").sum()
    )

    if missing_matches:
        raise RuntimeError(
            f"{split}: {missing_matches} unmatched timestamps"
        )

    merged = merged.drop(columns=["_merge"])

    if len(merged) != len(results):
        raise RuntimeError(
            f"{split}: row count changed after join"
        )

    missing_features = merged[
        FEATURE_COLUMNS
    ].isna().sum().sum()

    print(f"Rows: {len(merged)}")
    print(f"Unmatched timestamps: {missing_matches}")
    print(f"Missing feature cells: {missing_features}")

    merged["split"] = split

    return add_regimes(merged)


# ============================================================
# GROUP CLASSIFICATION
# ============================================================

def make_signal_groups(df):
    rows = []

    for side in ["LONG", "SHORT"]:
        label = f"{side}_BIAS"
        outcome_col = f"{side.lower()}_outcome_12h"

        qwen = df["qwen_prediction"] == label
        heuristic = df["heuristic_prediction"] == label

        masks = {
            "BOTH": qwen & heuristic,
            "HEUR_ONLY": heuristic & ~qwen,
            "QWEN_ONLY": qwen & ~heuristic,
        }

        for group, mask in masks.items():
            subset = df.loc[mask].copy()

            subset["side"] = side
            subset["group"] = group
            subset["outcome"] = subset[outcome_col]

            rows.append(subset)

    return pd.concat(rows, ignore_index=True)


# ============================================================
# PERFORMANCE
# ============================================================

def calculate_stats(group):
    outcome = group["outcome"]

    wins = int((outcome == 1).sum())
    losses = int((outcome == -1).sum())

    unresolved = int(
        ((outcome == 0) | outcome.isna()).sum()
    )

    resolved = wins + losses
    trades = len(group)

    total_r = TP_R * wins - SL_R * losses

    return pd.Series({
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "unresolved": unresolved,
        "resolved": resolved,
        "win_rate_resolved": (
            wins / resolved if resolved else np.nan
        ),
        "r_per_signal_zero_unresolved": (
            total_r / trades if trades else np.nan
        ),
        "r_per_resolved": (
            total_r / resolved if resolved else np.nan
        ),
        "total_signal_r": total_r,
        "sufficient_count": trades >= MIN_GROUP_SIZE,
    })


def summarize(df, dimensions):
    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby(
            dimensions,
            dropna=False,
            observed=True,
        )
        .apply(
            calculate_stats,
            include_groups=False,
        )
        .reset_index()
    )


# ============================================================
# FEATURE COMPARISON
# ============================================================

def compare_features(signals):
    rows = []

    for (split, side, group), subset in signals.groupby(
        ["split", "side", "group"]
    ):
        for feature in FEATURE_COLUMNS:
            values = pd.to_numeric(
                subset[feature],
                errors="coerce",
            ).dropna()

            rows.append({
                "split": split,
                "side": side,
                "group": group,
                "feature": feature,
                "count": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "q25": values.quantile(0.25),
                "q75": values.quantile(0.75),
            })

    return pd.DataFrame(rows)


# ============================================================
# DATA QUALITY
# ============================================================

def verify_time_structure(df):
    for split, subset in df.groupby("split"):
        subset = subset.sort_values("timestamp")

        expected_available = (
            subset["timestamp"] + pd.Timedelta(hours=1)
        )

        mismatch = int(
            (
                subset["available_at"]
                != expected_available
            ).sum()
        )

        duplicates = int(
            subset["timestamp"].duplicated().sum()
        )

        print(
            f"{split}: available_at mismatches={mismatch}, "
            f"duplicate timestamps={duplicates}"
        )

        if mismatch or duplicates:
            raise RuntimeError(
                f"{split}: timestamp integrity check failed"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("QWEN V2 - MARKET REGIME AUDIT")
    print("=" * 72)

    features = pd.read_parquet(
        FEATURES_FILE,
        columns=[
            "timestamp",
            "available_at",
            *FEATURE_COLUMNS,
        ],
    )

    features["timestamp"] = pd.to_datetime(
        features["timestamp"],
        utc=True,
    )

    if features["timestamp"].isna().any():
        raise ValueError("Null timestamps in features")

    if features["timestamp"].duplicated().any():
        raise ValueError("Duplicate feature timestamps")

    splits = [
        load_split(name, path, features)
        for name, path in RESULTS_FILES.items()
    ]

    combined = pd.concat(
        splits,
        ignore_index=True,
    )

    verify_time_structure(combined)

    signals = make_signal_groups(combined)

    # Each side-specific comparison is independent.
    # Rows may appear in both side analyses when models
    # select opposing directions. Avoid summing such
    # groups as if they were independent trades.

    group_summary = summarize(
        signals,
        ["split", "side", "group"],
    )

    regimes = []

    for regime_col in REGIME_COLUMNS:
        summary = summarize(
            signals,
            [
                "split",
                "side",
                "group",
                regime_col,
            ],
        )

        summary = summary.rename(
            columns={regime_col: "regime_value"}
        )

        summary["regime_type"] = regime_col
        regimes.append(summary)

    regime_summary = pd.concat(
        regimes,
        ignore_index=True,
    )

    feature_summary = compare_features(signals)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    group_summary.to_csv(
        OUTPUT_DIR / "group_summary.csv",
        index=False,
    )

    regime_summary.to_csv(
        OUTPUT_DIR / "regime_summary.csv",
        index=False,
    )

    feature_summary.to_csv(
        OUTPUT_DIR / "feature_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print("GLOBAL GROUP PERFORMANCE")
    print("=" * 72)

    display = [
        "split",
        "side",
        "group",
        "trades",
        "wins",
        "losses",
        "unresolved",
        "win_rate_resolved",
        "r_per_signal_zero_unresolved",
        "total_signal_r",
    ]

    print(
        group_summary[display].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 72)
    print("MARKET REGIME DISTRIBUTION")
    print("=" * 72)

    print(
        regime_summary.groupby(
            ["split", "side", "group", "regime_type"]
        )["trades"]
        .sum()
        .to_string()
    )

    small_groups = int(
        (~regime_summary["sufficient_count"]).sum()
    )

    print(
        f"\nRegime groups below {MIN_GROUP_SIZE} "
        f"signals: {small_groups}"
    )

    print(
        "WARNING: thresholds are exploratory. "
        "No statistical significance is implied."
    )
    print(
        "Unresolved outcomes contribute 0R only "
        "to the provisional per-signal metric."
    )

    print("\nOutput directory:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_INPUT = Path(
    "data/processed/btc_usdc_1h_features.parquet"
)

DEFAULT_OUTPUT = Path(
    "data/processed/btc_usdc_1h_labeled.parquet"
)

HORIZONS = [4, 12, 24]


# ============================================================
# FIRST TOUCH
# ============================================================

def first_touch_long(
    future: pd.DataFrame,
    entry: float,
    atr: float,
    tp_atr: float,
    sl_atr: float,
):
    """
    Retour :
     1  = TP touché avant SL
    -1  = SL touché avant TP
     0  = aucun touché
    NaN = TP et SL touchés dans la même bougie
          → ordre impossible à connaître avec OHLC
    """

    tp = entry + atr * tp_atr
    sl = entry - atr * sl_atr

    for _, candle in future.iterrows():

        hit_tp = candle["high"] >= tp
        hit_sl = candle["low"] <= sl

        if hit_tp and hit_sl:
            return np.nan

        if hit_tp:
            return 1

        if hit_sl:
            return -1

    return 0


def first_touch_short(
    future: pd.DataFrame,
    entry: float,
    atr: float,
    tp_atr: float,
    sl_atr: float,
):
    """
    Même principe pour un short.
    """

    tp = entry - atr * tp_atr
    sl = entry + atr * sl_atr

    for _, candle in future.iterrows():

        hit_tp = candle["low"] <= tp
        hit_sl = candle["high"] >= sl

        if hit_tp and hit_sl:
            return np.nan

        if hit_tp:
            return 1

        if hit_sl:
            return -1

    return 0


# ============================================================
# CONSTRUCTION DES LABELS
# ============================================================

def build_labels(
    df: pd.DataFrame,
    tp_atr: float,
    sl_atr: float,
):
    df = df.copy()

    number_rows = len(df)

    # --------------------------------------------------------
    # Colonnes de sortie
    # --------------------------------------------------------

    for horizon in HORIZONS:

        df[f"future_return_{horizon}h_pct"] = np.nan

        df[f"long_best_{horizon}h_pct"] = np.nan
        df[f"long_worst_{horizon}h_pct"] = np.nan

        df[f"short_best_{horizon}h_pct"] = np.nan
        df[f"short_worst_{horizon}h_pct"] = np.nan

        df[f"up_move_{horizon}h_atr"] = np.nan
        df[f"down_move_{horizon}h_atr"] = np.nan

        df[f"long_outcome_{horizon}h"] = np.nan
        df[f"short_outcome_{horizon}h"] = np.nan


    # ========================================================
    # ITERATION
    # ========================================================

    for i in range(number_rows):

        entry = df.at[i, "close"]
        atr = df.at[i, "atr14"]

        if pd.isna(entry) or pd.isna(atr):
            continue

        for horizon in HORIZONS:

            # -----------------------------------------------
            # IMPORTANT :
            #
            # La bougie actuelle n'est PAS incluse.
            #
            # On commence à i + 1.
            # -----------------------------------------------

            start = i + 1
            end = i + horizon + 1

            if end > number_rows:
                continue

            future = df.iloc[start:end]

            if len(future) != horizon:
                continue


            # -----------------------------------------------
            # Prix de clôture après N heures
            # -----------------------------------------------

            final_close = future.iloc[-1]["close"]

            future_return = (
                (final_close / entry) - 1
            ) * 100

            df.at[
                i,
                f"future_return_{horizon}h_pct"
            ] = future_return


            # -----------------------------------------------
            # Maximum / minimum futurs
            # -----------------------------------------------

            max_high = future["high"].max()
            min_low = future["low"].min()


            # =================================================
            # LONG
            # =================================================

            long_best = (
                (max_high / entry) - 1
            ) * 100

            long_worst = (
                (min_low / entry) - 1
            ) * 100


            df.at[
                i,
                f"long_best_{horizon}h_pct"
            ] = long_best

            df.at[
                i,
                f"long_worst_{horizon}h_pct"
            ] = long_worst


            # =================================================
            # SHORT
            # =================================================

            short_best = (
                (entry / min_low) - 1
            ) * 100

            short_worst = -(
                (max_high / entry) - 1
            ) * 100


            df.at[
                i,
                f"short_best_{horizon}h_pct"
            ] = short_best

            df.at[
                i,
                f"short_worst_{horizon}h_pct"
            ] = short_worst


            # =================================================
            # MOUVEMENT EN ATR
            # =================================================

            up_move_atr = (
                max_high - entry
            ) / atr

            down_move_atr = (
                entry - min_low
            ) / atr


            df.at[
                i,
                f"up_move_{horizon}h_atr"
            ] = up_move_atr

            df.at[
                i,
                f"down_move_{horizon}h_atr"
            ] = down_move_atr


            # =================================================
            # FIRST TOUCH
            # =================================================

            df.at[
                i,
                f"long_outcome_{horizon}h"
            ] = first_touch_long(
                future=future,
                entry=entry,
                atr=atr,
                tp_atr=tp_atr,
                sl_atr=sl_atr,
            )

            df.at[
                i,
                f"short_outcome_{horizon}h"
            ] = first_touch_short(
                future=future,
                entry=entry,
                atr=atr,
                tp_atr=tp_atr,
                sl_atr=sl_atr,
            )


    return df


# ============================================================
# FEATURE READY
# ============================================================

def add_readiness_flags(
    df: pd.DataFrame,
):
    """
    Indique quelles lignes disposent réellement
    de suffisamment d'historique pour servir
    d'exemples au modèle.
    """

    required_features = [
        "ema20",
        "ema50",
        "ema200",
        "rsi14",
        "atr14_pct",
        "volume_ratio",
        "volatility_72h",

        "4h_rsi14",
        "4h_atr14_pct",
        "4h_trend",

        "1d_rsi14",
        "1d_atr14_pct",
        "1d_trend",
    ]

    df["features_ready"] = (
        df[required_features]
        .notna()
        .all(axis=1)
    )

    df["labels_24h_ready"] = (
        df["future_return_24h_pct"]
        .notna()
    )

    df["sample_ready"] = (
        df["features_ready"]
        & df["labels_24h_ready"]
    )

    return df


# ============================================================
# STATISTIQUES
# ============================================================

def print_statistics(
    df: pd.DataFrame,
):
    print()
    print("=" * 70)
    print("STATISTIQUES")
    print("=" * 70)

    print(
        "Lignes totales :",
        f"{len(df):,}",
    )

    print(
        "Features prêtes :",
        f"{df['features_ready'].sum():,}",
    )

    print(
        "Labels 24h prêts :",
        f"{df['labels_24h_ready'].sum():,}",
    )

    print(
        "Samples utilisables :",
        f"{df['sample_ready'].sum():,}",
    )


    usable = df[
        df["sample_ready"]
    ]


    print()
    print("Rendements futurs moyens :")

    for horizon in HORIZONS:

        mean_return = usable[
            f"future_return_{horizon}h_pct"
        ].mean()

        print(
            f"  {horizon:2}h : "
            f"{mean_return:+.4f}%"
        )


    print()
    print("Amplitude moyenne :")

    for horizon in HORIZONS:

        up = usable[
            f"up_move_{horizon}h_atr"
        ].mean()

        down = usable[
            f"down_move_{horizon}h_atr"
        ].mean()

        print(
            f"  {horizon:2}h : "
            f"+{up:.2f} ATR / "
            f"-{down:.2f} ATR"
        )


    print()
    print("First-touch 12h LONG :")

    print(
        usable[
            "long_outcome_12h"
        ]
        .value_counts(
            dropna=False,
            normalize=True,
        )
        .sort_index()
        .to_string()
    )


    print()
    print("First-touch 12h SHORT :")

    print(
        usable[
            "short_outcome_12h"
        ]
        .value_counts(
            dropna=False,
            normalize=True,
        )
        .sort_index()
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )

    parser.add_argument(
        "--tp-atr",
        type=float,
        default=1.5,
        help=(
            "Target exprimé en multiples ATR. "
            "Défaut : 1.5"
        ),
    )

    parser.add_argument(
        "--sl-atr",
        type=float,
        default=1.0,
        help=(
            "Stop exprimé en multiples ATR. "
            "Défaut : 1.0"
        ),
    )

    args = parser.parse_args()


    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )


    print("=" * 70)
    print("CONSTRUCTION DES LABELS")
    print("=" * 70)

    print(
        "Source :",
        input_path,
    )

    print(
        "TP :",
        args.tp_atr,
        "ATR",
    )

    print(
        "SL :",
        args.sl_atr,
        "ATR",
    )


    df = pd.read_parquet(
        input_path
    )


    print(
        "Lignes :",
        f"{len(df):,}",
    )


    # ========================================================
    # LABELS
    # ========================================================

    labeled = build_labels(
        df=df,
        tp_atr=args.tp_atr,
        sl_atr=args.sl_atr,
    )


    # ========================================================
    # FLAGS
    # ========================================================

    labeled = add_readiness_flags(
        labeled
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    labeled.to_parquet(
        output_path,
        index=False,
        compression="zstd",
    )


    print_statistics(
        labeled
    )


    print()
    print("=" * 70)
    print("SAUVEGARDE")
    print("=" * 70)

    print(
        output_path.resolve()
    )


if __name__ == "__main__":
    main()
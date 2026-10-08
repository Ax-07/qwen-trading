from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("data/processed")


# ============================================================
# INDICATEURS
# ============================================================

def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI de Wilder.
    Aucun accès aux données futures.
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range de Wilder.
    """
    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return atr


# ============================================================
# FEATURES 1H
# ============================================================

def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    df["return_1h"] = df["close"].pct_change(1)
    df["return_3h"] = df["close"].pct_change(3)
    df["return_6h"] = df["close"].pct_change(6)
    df["return_24h"] = df["close"].pct_change(24)

    df["log_return_1h"] = np.log(
        df["close"] / df["close"].shift(1)
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    for period in [20, 50, 200]:
        df[f"ema{period}"] = (
            df["close"]
            .ewm(
                span=period,
                adjust=False,
            )
            .mean()
        )

        df[f"distance_ema{period}_pct"] = (
            (df["close"] / df[f"ema{period}"]) - 1
        ) * 100

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi14"] = calculate_rsi(
        df["close"],
        14,
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["atr14"] = calculate_atr(
        df,
        14,
    )

    df["atr14_pct"] = (
        df["atr14"] / df["close"]
    ) * 100

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    df["volume_sma20"] = (
        df["volume"]
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"]
        / df["volume_sma20"]
    )

    # --------------------------------------------------------
    # VOLATILITE
    # --------------------------------------------------------

    df["volatility_24h"] = (
        df["log_return_1h"]
        .rolling(
            24,
            min_periods=24,
        )
        .std()
    )

    df["volatility_72h"] = (
        df["log_return_1h"]
        .rolling(
            72,
            min_periods=72,
        )
        .std()
    )

    # --------------------------------------------------------
    # BOUGIE
    # --------------------------------------------------------

    df["range_pct"] = (
        (df["high"] - df["low"])
        / df["open"]
    ) * 100

    df["body_pct"] = (
        (df["close"] - df["open"])
        / df["open"]
    ) * 100

    candle_top = df[
        ["open", "close"]
    ].max(axis=1)

    candle_bottom = df[
        ["open", "close"]
    ].min(axis=1)

    df["upper_wick_pct"] = (
        (df["high"] - candle_top)
        / df["open"]
    ) * 100

    df["lower_wick_pct"] = (
        (candle_bottom - df["low"])
        / df["open"]
    ) * 100

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------
    #
    # IMPORTANT :
    # shift(1) empêche la bougie actuelle de créer elle-même
    # son support / sa résistance.
    #
    # Donc ces niveaux étaient réellement connus avant
    # la bougie actuelle.
    # --------------------------------------------------------

    df["prev_high_24h"] = (
        df["high"]
        .rolling(
            24,
            min_periods=24,
        )
        .max()
        .shift(1)
    )

    df["prev_low_24h"] = (
        df["low"]
        .rolling(
            24,
            min_periods=24,
        )
        .min()
        .shift(1)
    )

    df["prev_high_72h"] = (
        df["high"]
        .rolling(
            72,
            min_periods=72,
        )
        .max()
        .shift(1)
    )

    df["prev_low_72h"] = (
        df["low"]
        .rolling(
            72,
            min_periods=72,
        )
        .min()
        .shift(1)
    )

    df["distance_high_24h_pct"] = (
        (df["prev_high_24h"] / df["close"]) - 1
    ) * 100

    df["distance_low_24h_pct"] = (
        (df["close"] / df["prev_low_24h"]) - 1
    ) * 100

    df["distance_high_72h_pct"] = (
        (df["prev_high_72h"] / df["close"]) - 1
    ) * 100

    df["distance_low_72h_pct"] = (
        (df["close"] / df["prev_low_72h"]) - 1
    ) * 100

    # --------------------------------------------------------
    # BREAKOUTS
    # --------------------------------------------------------

    df["breakout_24h"] = (
        df["close"] > df["prev_high_24h"]
    ).astype("int8")

    df["breakdown_24h"] = (
        df["close"] < df["prev_low_24h"]
    ).astype("int8")

    # --------------------------------------------------------
    # TENDANCE 1H
    # --------------------------------------------------------

    bullish = (
        (df["close"] > df["ema20"])
        & (df["ema20"] > df["ema50"])
        & (df["ema50"] > df["ema200"])
    )

    bearish = (
        (df["close"] < df["ema20"])
        & (df["ema20"] < df["ema50"])
        & (df["ema50"] < df["ema200"])
    )

    df["trend_1h"] = 0

    df.loc[
        bullish,
        "trend_1h",
    ] = 1

    df.loc[
        bearish,
        "trend_1h",
    ] = -1

    df["trend_1h"] = df[
        "trend_1h"
    ].astype("int8")

    # --------------------------------------------------------
    # DISPERSION EMA
    # --------------------------------------------------------

    df["ema20_50_spread_pct"] = (
        (df["ema20"] / df["ema50"]) - 1
    ) * 100

    df["ema50_200_spread_pct"] = (
        (df["ema50"] / df["ema200"]) - 1
    ) * 100

    return df


# ============================================================
# RESAMPLE HIGHER TIMEFRAME
# ============================================================

def resample_ohlcv(
    df: pd.DataFrame,
    timeframe: str,
    expected_bars: int,
) -> pd.DataFrame:
    """
    Reconstruit une timeframe supérieure.

    Une bougie 4H commençant à 00:00 contient :
    00h, 01h, 02h, 03h.

    Elle ne devient disponible qu'à 04:00.

    Même principe pour le journalier.
    """

    indexed = (
        df
        .set_index("timestamp")
        .sort_index()
    )

    grouped = indexed.resample(
        timeframe,
        label="left",
        closed="left",
        origin="start_day",
    )

    result = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )

    counts = grouped[
        "close"
    ].count()

    result["bar_count"] = counts

    # Supprime les bougies incomplètes.
    result = result[
        result["bar_count"]
        == expected_bars
    ].copy()

    return result


# ============================================================
# FEATURES HIGHER TIMEFRAME
# ============================================================

def build_htf_features(
    htf: pd.DataFrame,
    prefix: str,
    availability_delay: pd.Timedelta,
) -> pd.DataFrame:
    htf = htf.copy()

    htf["ema20"] = (
        htf["close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    htf["ema50"] = (
        htf["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    htf["rsi14"] = calculate_rsi(
        htf["close"]
    )

    htf["atr14"] = calculate_atr(
        htf
    )

    htf["atr14_pct"] = (
        htf["atr14"]
        / htf["close"]
    ) * 100

    bullish = (
        (htf["close"] > htf["ema20"])
        & (htf["ema20"] > htf["ema50"])
    )

    bearish = (
        (htf["close"] < htf["ema20"])
        & (htf["ema20"] < htf["ema50"])
    )

    htf["trend"] = 0

    htf.loc[
        bullish,
        "trend",
    ] = 1

    htf.loc[
        bearish,
        "trend",
    ] = -1

    # --------------------------------------------------------
    # CRITIQUE ANTI LOOK-AHEAD
    # --------------------------------------------------------
    #
    # La bougie 4H de 00:00 n'est connue qu'à 04:00.
    #
    # Donc son timestamp de disponibilité devient 04:00.
    #
    # Idem pour le Daily :
    # la bougie du 1er janvier devient connue
    # le 2 janvier à 00:00.
    # --------------------------------------------------------

    htf["available_at"] = (
        htf.index
        + availability_delay
    )

    selected = htf[
        [
            "available_at",
            "close",
            "ema20",
            "ema50",
            "rsi14",
            "atr14_pct",
            "trend",
        ]
    ].copy()

    selected = selected.rename(
        columns={
            "close":
                f"{prefix}_close",

            "ema20":
                f"{prefix}_ema20",

            "ema50":
                f"{prefix}_ema50",

            "rsi14":
                f"{prefix}_rsi14",

            "atr14_pct":
                f"{prefix}_atr14_pct",

            "trend":
                f"{prefix}_trend",
        }
    )

    return (
        selected
        .sort_values(
            "available_at"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# DETECTION DU CSV 1H
# ============================================================

def find_1h_file(
    data_dir: Path,
) -> Path:
    candidates = sorted(
        data_dir.glob(
            "*_1h_*.csv"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            "Aucun fichier *_1h_*.csv trouvé "
            f"dans {data_dir.resolve()}"
        )

    if len(candidates) > 1:
        print(
            "Plusieurs fichiers 1h trouvés :"
        )

        for path in candidates:
            print(
                " -",
                path,
            )

        print()
        print(
            "Utilisation du premier :",
            candidates[0],
        )

    return candidates[0]


# ============================================================
# CHARGEMENT
# ============================================================

def load_source(
    path: Path,
) -> pd.DataFrame:
    print(
        "Chargement :",
        path,
    )

    df = pd.read_csv(
        path
    )

    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {missing}"
        )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(
        df["timestamp"]
    ):
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True,
        )

    else:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
        )

    # --------------------------------------------------------
    # TYPES NUMERIQUES
    # --------------------------------------------------------

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        )

    # --------------------------------------------------------
    # NETTOYAGE
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset="timestamp",
            keep="last",
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# CONTROLE CONTINUITE
# ============================================================

def validate_hourly_continuity(
    df: pd.DataFrame,
):
    delta = (
        df["timestamp"]
        .diff()
        .dropna()
    )

    expected = pd.Timedelta(
        hours=1
    )

    bad = delta[
        delta != expected
    ]

    if len(bad):
        raise ValueError(
            f"{len(bad)} trou(s) temporel(s) "
            "détecté(s) dans le fichier 1h."
        )

    print(
        "Continuité 1h : OK"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "CSV 1h à utiliser. "
            "Sinon auto-détection dans data/"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            DEFAULT_OUTPUT_DIR
            / "btc_usdc_1h_features.parquet"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    if args.input:
        input_path = Path(
            args.input
        )

    else:
        input_path = find_1h_file(
            DEFAULT_DATA_DIR
        )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_source(
        input_path
    )

    print(
        f"Lignes source : {len(df):,}"
    )

    print(
        "Début :",
        df["timestamp"].iloc[0],
    )

    print(
        "Fin   :",
        df["timestamp"].iloc[-1],
    )

    validate_hourly_continuity(
        df
    )

    # --------------------------------------------------------
    # FEATURES 1H
    # --------------------------------------------------------

    print()
    print(
        "Calcul features 1H..."
    )

    features = build_base_features(
        df
    )

    # Une bougie 1h dont timestamp=10:00
    # n'est entièrement connue qu'à 11:00.
    features["available_at"] = (
        features["timestamp"]
        + pd.Timedelta(
            hours=1
        )
    )

    # --------------------------------------------------------
    # 4H
    # --------------------------------------------------------

    print(
        "Construction contexte 4H..."
    )

    df_4h = resample_ohlcv(
        df,
        timeframe="4h",
        expected_bars=4,
    )

    features_4h = build_htf_features(
        df_4h,
        prefix="4h",
        availability_delay=pd.Timedelta(
            hours=4
        ),
    )

    # --------------------------------------------------------
    # DAILY
    # --------------------------------------------------------

    print(
        "Construction contexte 1D..."
    )

    df_1d = resample_ohlcv(
        df,
        timeframe="1D",
        expected_bars=24,
    )

    features_1d = build_htf_features(
        df_1d,
        prefix="1d",
        availability_delay=pd.Timedelta(
            days=1
        ),
    )

    # --------------------------------------------------------
    # MERGE ASOF
    # --------------------------------------------------------
    #
    # Pour chaque bougie 1H, on prend UNIQUEMENT
    # la dernière bougie 4H / 1D qui était déjà
    # entièrement terminée à cet instant.
    # --------------------------------------------------------

    features = features.sort_values(
        "available_at"
    )

    features = pd.merge_asof(
        features,
        features_4h,
        on="available_at",
        direction="backward",
    )

    features = pd.merge_asof(
        features,
        features_1d,
        on="available_at",
        direction="backward",
    )

    # --------------------------------------------------------
    # DIVERS CONTEXTES
    # --------------------------------------------------------

    features["trend_alignment"] = (
        features["trend_1h"]
        + features["4h_trend"]
        + features["1d_trend"]
    )

    # +3 = totalement haussier
    # -3 = totalement baissier

    # --------------------------------------------------------
    # NETTOYAGE INFINIS
    # --------------------------------------------------------

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # SAUVEGARDE
    # --------------------------------------------------------

    features.to_parquet(
        output_path,
        index=False,
        compression="zstd",
    )

    print()
    print("=" * 70)
    print("TERMINE")
    print("=" * 70)

    print(
        "Lignes :",
        f"{len(features):,}",
    )

    print(
        "Colonnes :",
        len(features.columns),
    )

    print(
        "Fichier :",
        output_path.resolve(),
    )

    print()
    print(
        "Quelques colonnes :"
    )

    columns_to_show = [
        "timestamp",
        "close",
        "ema20",
        "ema50",
        "ema200",
        "rsi14",
        "atr14_pct",
        "volume_ratio",
        "trend_1h",
        "4h_trend",
        "1d_trend",
        "trend_alignment",
    ]

    print(
        features[
            columns_to_show
        ]
        .tail(10)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Valeurs manquantes principales :"
    )

    missing = (
        features
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        missing[
            missing > 0
        ]
        .head(20)
    )


if __name__ == "__main__":
    main()
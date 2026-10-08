from __future__ import annotations

import argparse
import json
import os
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"

DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

DEFAULT_INTERVALS = [
    "15m",
    "1h",
    "4h",
]

DEFAULT_START = "2021-01-01"

DEFAULT_OUTPUT_DIR = Path(
    "data/raw/binance_usdm"
)

# On pourrait demander 1500 bougies,
# mais 1000 est plus efficace en poids API.
REQUEST_LIMIT = 1000

# Pause volontaire entre les requêtes.
REQUEST_DELAY = 0.20

MAX_RETRIES = 6

REQUEST_TIMEOUT = 30


# ============================================================
# INTERVALLES BINANCE
# ============================================================

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,

    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,

    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,

    "1w": 7 * 24 * 60 * 60_000,
}


# ============================================================
# COLONNES RENVOYEES PAR BINANCE
# ============================================================

RAW_COLUMNS = [
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


# ============================================================
# UTILITAIRES DE DATE
# ============================================================

def parse_datetime_to_ms(value: str) -> int:
    """
    Accepte par exemple :

    2021-01-01
    2021-01-01T12:00:00
    2021-01-01T12:00:00Z
    """

    value = value.strip()

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

    dt = dt.astimezone(
        timezone.utc
    )

    return int(
        dt.timestamp() * 1000
    )


def ms_to_iso(value: int) -> str:
    return (
        datetime
        .fromtimestamp(
            value / 1000,
            tz=timezone.utc,
        )
        .isoformat()
    )


def last_closed_candle_end_ms(
    interval: str,
) -> int:
    """
    Renvoie la fin de la dernière bougie
    entièrement terminée.

    Ainsi, aucune bougie encore ouverte
    n'entre dans notre dataset.
    """

    interval_ms = INTERVAL_MS[interval]

    now_ms = int(
        datetime.now(
            timezone.utc
        ).timestamp() * 1000
    )

    current_candle_start = (
        now_ms // interval_ms
    ) * interval_ms

    return (
        current_candle_start - 1
    )


# ============================================================
# SESSION HTTP
# ============================================================

def create_session() -> requests.Session:

    session = requests.Session()

    session.headers.update({
        "User-Agent":
            "qwen-trading-research/1.0",
        "Accept":
            "application/json",
    })

    return session


# ============================================================
# REQUETE BINANCE AVEC RETRIES
# ============================================================

def binance_get(
    session: requests.Session,
    endpoint: str,
    params: dict,
):
    url = BASE_URL + endpoint

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

        except requests.RequestException as exc:

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Erreur réseau définitive : {exc}"
                ) from exc

            wait = min(
                2 ** attempt,
                30,
            )

            print(
                f"  Erreur réseau. "
                f"Nouvelle tentative dans {wait}s..."
            )

            time.sleep(wait)

            continue

        # ----------------------------------------
        # OK
        # ----------------------------------------

        if response.status_code == 200:

            return response.json()

        # ----------------------------------------
        # RATE LIMIT
        # ----------------------------------------

        if response.status_code == 429:

            retry_after = int(
                response.headers.get(
                    "Retry-After",
                    "5",
                )
            )

            print(
                f"  Rate limit Binance. "
                f"Pause {retry_after}s..."
            )

            time.sleep(
                retry_after
            )

            continue

        # ----------------------------------------
        # IP TEMPORAIREMENT BANNIE
        # ----------------------------------------

        if response.status_code == 418:

            raise RuntimeError(
                "Binance a temporairement bloqué "
                "l'adresse IP à cause du rate limit."
            )

        # ----------------------------------------
        # RESTRICTION GEOGRAPHIQUE
        # ----------------------------------------

        if response.status_code == 451:

            raise RuntimeError(
                "Binance Futures renvoie HTTP 451. "
                "L'endpoint n'est pas disponible "
                "depuis cette connexion/région."
            )

        # ----------------------------------------
        # AUTRE ERREUR
        # ----------------------------------------

        try:
            details = response.json()
        except Exception:
            details = response.text

        raise RuntimeError(
            f"Erreur Binance HTTP "
            f"{response.status_code}: "
            f"{details}"
        )

    raise RuntimeError(
        "Nombre maximal de tentatives atteint."
    )


# ============================================================
# TELECHARGEMENT D'UN BATCH
# ============================================================

def fetch_klines_batch(
    session: requests.Session,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
):
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": REQUEST_LIMIT,
    }

    return binance_get(
        session=session,
        endpoint=KLINES_ENDPOINT,
        params=params,
    )


# ============================================================
# CONVERSION DATAFRAME
# ============================================================

def rows_to_dataframe(
    rows: list,
) -> pd.DataFrame:

    if not rows:

        return pd.DataFrame(
            columns=RAW_COLUMNS
        )

    df = pd.DataFrame(
        rows,
        columns=RAW_COLUMNS,
    )

    # ----------------------------------------
    # TYPES
    # ----------------------------------------

    integer_columns = [
        "open_time_ms",
        "close_time_ms",
        "trades",
    ]

    for col in integer_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        ).astype("int64")


    float_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    for col in float_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        )


    # ----------------------------------------
    # TIMESTAMPS UTC
    # ----------------------------------------

    df["open_time"] = pd.to_datetime(
        df["open_time_ms"],
        unit="ms",
        utc=True,
    )

    df["close_time"] = pd.to_datetime(
        df["close_time_ms"],
        unit="ms",
        utc=True,
    )


    # ----------------------------------------
    # ORDRE DES COLONNES
    # ----------------------------------------

    columns = [
        "open_time",
        "open_time_ms",

        "open",
        "high",
        "low",
        "close",

        "volume",
        "quote_volume",

        "trades",

        "taker_buy_base_volume",
        "taker_buy_quote_volume",

        "close_time",
        "close_time_ms",

        "ignore",
    ]

    return df[columns]


# ============================================================
# CHARGEMENT D'UN FICHIER EXISTANT
# ============================================================

def load_existing(
    path: Path,
) -> pd.DataFrame | None:

    if not path.exists():
        return None

    print(
        f"  Fichier existant : {path}"
    )

    df = pd.read_parquet(
        path
    )

    if len(df) == 0:
        return None

    return df


# ============================================================
# SAUVEGARDE ATOMIQUE
# ============================================================

def save_parquet_atomic(
    df: pd.DataFrame,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        ".tmp.parquet"
    )

    df.to_parquet(
        temp_path,
        index=False,
        compression="zstd",
    )

    os.replace(
        temp_path,
        path,
    )


# ============================================================
# TELECHARGEMENT COMPLET D'UN SYMBOL/TIMEFRAME
# ============================================================

def download_symbol_interval(
    session: requests.Session,
    symbol: str,
    interval: str,
    requested_start_ms: int,
    requested_end_ms: int,
    output_dir: Path,
    fresh: bool,
):
    interval_ms = INTERVAL_MS[
        interval
    ]

    path = (
        output_dir
        / f"{symbol}_{interval}.parquet"
    )


    print()
    print("=" * 70)
    print(
        f"{symbol} | {interval}"
    )
    print("=" * 70)


    # ========================================================
    # DONNEES EXISTANTES
    # ========================================================

    existing = None

    if not fresh:

        existing = load_existing(
            path
        )


    # ========================================================
    # POINT DE DEPART
    # ========================================================

    cursor = requested_start_ms

    if (
        existing is not None
        and len(existing) > 0
    ):

        existing_last_open = int(
            existing[
                "open_time_ms"
            ].max()
        )

        cursor = max(
            cursor,
            existing_last_open
            + interval_ms,
        )

        print(
            "  Dernière bougie locale :",
            ms_to_iso(
                existing_last_open
            ),
        )


    if cursor > requested_end_ms:

        print(
            "  Déjà à jour."
        )

        return {
            "symbol": symbol,
            "interval": interval,
            "rows": (
                len(existing)
                if existing is not None
                else 0
            ),
            "downloaded": 0,
            "file": str(path),
        }


    print(
        "  Début téléchargement :",
        ms_to_iso(cursor),
    )

    print(
        "  Fin téléchargement   :",
        ms_to_iso(
            requested_end_ms
        ),
    )


    # ========================================================
    # PAGINATION
    # ========================================================

    downloaded_rows = []

    request_count = 0


    while cursor <= requested_end_ms:

        batch = fetch_klines_batch(
            session=session,
            symbol=symbol,
            interval=interval,
            start_ms=cursor,
            end_ms=requested_end_ms,
        )

        request_count += 1


        if not batch:
            break


        downloaded_rows.extend(
            batch
        )


        last_open_ms = int(
            batch[-1][0]
        )


        print(
            f"\r  Bougies téléchargées : "
            f"{len(downloaded_rows):,}"
            f" | dernière : "
            f"{ms_to_iso(last_open_ms)}",
            end="",
            flush=True,
        )


        next_cursor = (
            last_open_ms
            + interval_ms
        )


        # Protection boucle infinie.
        if next_cursor <= cursor:

            raise RuntimeError(
                "La pagination Binance "
                "n'avance plus."
            )


        cursor = next_cursor


        if len(batch) < REQUEST_LIMIT:
            break


        time.sleep(
            REQUEST_DELAY
        )


    print()


    # ========================================================
    # CONVERSION
    # ========================================================

    new_df = rows_to_dataframe(
        downloaded_rows
    )


    # ========================================================
    # FUSION AVEC L'EXISTANT
    # ========================================================

    if existing is not None:

        combined = pd.concat(
            [
                existing,
                new_df,
            ],
            ignore_index=True,
        )

    else:

        combined = new_df


    # ========================================================
    # NETTOYAGE
    # ========================================================

    if len(combined):

        combined = (
            combined
            .drop_duplicates(
                subset=[
                    "open_time_ms"
                ],
                keep="last",
            )
            .sort_values(
                "open_time_ms"
            )
            .reset_index(
                drop=True
            )
        )


        # Sécurité :
        # on garde uniquement
        # les bougies complètement fermées.
        combined = combined[
            combined[
                "close_time_ms"
            ] <= requested_end_ms
        ].copy()


    # ========================================================
    # SAUVEGARDE
    # ========================================================

    save_parquet_atomic(
        combined,
        path,
    )


    downloaded_count = len(
        new_df
    )


    print(
        f"  Requêtes : "
        f"{request_count}"
    )

    print(
        f"  Nouvelles bougies : "
        f"{downloaded_count:,}"
    )

    print(
        f"  Total fichier : "
        f"{len(combined):,}"
    )

    print(
        f"  Sauvegardé : {path}"
    )


    if len(combined):

        print(
            "  Première bougie :",
            combined[
                "open_time"
            ].iloc[0],
        )

        print(
            "  Dernière bougie :",
            combined[
                "open_time"
            ].iloc[-1],
        )


    return {
        "symbol": symbol,
        "interval": interval,
        "rows": len(combined),
        "downloaded": downloaded_count,
        "requests": request_count,
        "file": str(path),
    }


# ============================================================
# MANIFEST
# ============================================================

def save_manifest(
    output_dir: Path,
    start_ms: int,
    results: list,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "source":
            "Binance USD-M Futures",

        "base_url":
            BASE_URL,

        "endpoint":
            KLINES_ENDPOINT,

        "downloaded_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "requested_start":
            ms_to_iso(start_ms),

        "files":
            results,
    }

    path = (
        output_dir
        / "download_manifest.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# ARGUMENTS CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Télécharge les OHLCV "
            "Binance USD-M Futures."
        )
    )


    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=(
            "Ex: BTCUSDT ETHUSDT SOLUSDT"
        ),
    )


    parser.add_argument(
        "--intervals",
        nargs="+",
        default=DEFAULT_INTERVALS,
        help=(
            "Ex: 15m 1h 4h"
        ),
    )


    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=(
            "Date UTC de départ. "
            "Défaut: 2021-01-01"
        ),
    )


    parser.add_argument(
        "--end",
        default=None,
        help=(
            "Date UTC de fin optionnelle. "
            "Sans valeur : dernière "
            "bougie fermée."
        ),
    )


    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )


    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore les fichiers existants "
            "et recommence le téléchargement."
        ),
    )


    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    for interval in args.intervals:

        if interval not in INTERVAL_MS:

            print(
                f"Intervalle non supporté : "
                f"{interval}"
            )

            sys.exit(1)


    symbols = [
        symbol.upper()
        for symbol in args.symbols
    ]


    start_ms = parse_datetime_to_ms(
        args.start
    )


    output_dir = Path(
        args.output
    )


    session = create_session()


    results = []


    # ========================================================
    # TELECHARGEMENT
    # ========================================================

    for symbol in symbols:

        for interval in args.intervals:

            if args.end:

                end_ms = parse_datetime_to_ms(
                    args.end
                )

            else:

                end_ms = (
                    last_closed_candle_end_ms(
                        interval
                    )
                )


            try:

                result = (
                    download_symbol_interval(
                        session=session,
                        symbol=symbol,
                        interval=interval,
                        requested_start_ms=start_ms,
                        requested_end_ms=end_ms,
                        output_dir=output_dir,
                        fresh=args.fresh,
                    )
                )

                results.append(
                    result
                )

            except Exception as exc:

                print()
                print(
                    f"ERREUR "
                    f"{symbol} {interval}:"
                )

                print(
                    str(exc)
                )

                raise


    # ========================================================
    # MANIFEST
    # ========================================================

    save_manifest(
        output_dir=output_dir,
        start_ms=start_ms,
        results=results,
    )


    # ========================================================
    # RESUME
    # ========================================================

    print()
    print("=" * 70)
    print("TERMINE")
    print("=" * 70)

    for result in results:

        print(
            f"{result['symbol']:10} "
            f"{result['interval']:4} "
            f"{result['rows']:>10,} lignes "
            f"(+{result['downloaded']:,})"
        )


    print()
    print(
        "Données disponibles dans :"
    )

    print(
        output_dir.resolve()
    )


if __name__ == "__main__":
    main()
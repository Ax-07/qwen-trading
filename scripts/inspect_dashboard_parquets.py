from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES = [
    PROJECT_ROOT / "data" / "processed" / "btc_usdc_1h_labeled.parquet",
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "qwen3.5-9b-trading-v2-fast"
    / "results_fast.parquet",
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "qwen3.5-9b-trading-v2-validation"
    / "results.parquet",
]


def find_time_columns(schema: pl.Schema) -> list[str]:
    """
    Détecte les colonnes qui pourraient représenter le temps,
    sans supposer que la colonne s'appelle forcément 'timestamp'.
    """
    candidates: list[str] = []

    time_keywords = (
        "timestamp",
        "datetime",
        "date",
        "time",
        "open_time",
        "close_time",
    )

    for name, dtype in schema.items():
        name_lower = name.lower()

        if any(keyword in name_lower for keyword in time_keywords):
            candidates.append(name)
            continue

        if dtype == pl.Date or isinstance(dtype, pl.Datetime):
            candidates.append(name)

    return list(dict.fromkeys(candidates))


def print_separator(char: str = "=", width: int = 100) -> None:
    print(char * width)


def inspect_parquet(path: Path) -> None:
    print()
    print_separator()
    print(f"FILE: {path}")
    print_separator()

    if not path.exists():
        print("ERROR: fichier introuvable")
        return

    # ------------------------------------------------------------
    # Métadonnées Parquet
    # ------------------------------------------------------------
    parquet_file = pq.ParquetFile(path)

    print(f"Size        : {path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Rows        : {parquet_file.metadata.num_rows:,}")
    print(f"Row groups  : {parquet_file.metadata.num_row_groups}")
    print(f"Columns     : {parquet_file.metadata.num_columns}")

    # ------------------------------------------------------------
    # Schéma Polars sans charger tout le dataset
    # ------------------------------------------------------------
    lazy = pl.scan_parquet(path)
    schema = lazy.collect_schema()

    print()
    print("SCHEMA")
    print_separator("-")

    for i, (name, dtype) in enumerate(schema.items(), start=1):
        print(f"{i:>3}. {name:<45} {dtype}")

    # ------------------------------------------------------------
    # Colonnes temporelles candidates
    # ------------------------------------------------------------
    time_columns = find_time_columns(schema)

    print()
    print("TIME COLUMNS")
    print_separator("-")

    if not time_columns:
        print("Aucune colonne temporelle détectée automatiquement.")
    else:
        for column in time_columns:
            dtype = schema[column]
            print(f"{column} ({dtype})")

            try:
                stats = (
                    lazy.select(
                        [
                            pl.col(column).min().alias("min"),
                            pl.col(column).max().alias("max"),
                            pl.col(column).null_count().alias("nulls"),
                        ]
                    )
                    .collect()
                )

                print(f"    min   : {stats['min'][0]}")
                print(f"    max   : {stats['max'][0]}")
                print(f"    nulls : {stats['nulls'][0]}")

            except Exception as exc:
                print(f"    Impossible de calculer min/max : {exc}")

    # ------------------------------------------------------------
    # Premières lignes
    # ------------------------------------------------------------
    print()
    print("FIRST 5 ROWS")
    print_separator("-")

    head = lazy.head(5).collect()

    # Affiche toutes les colonnes dans le terminal
    with pl.Config(
        tbl_cols=-1,
        tbl_rows=5,
        tbl_width_chars=300,
        fmt_str_lengths=80,
    ):
        print(head)

    # ------------------------------------------------------------
    # Valeurs nulles
    # ------------------------------------------------------------
    print()
    print("NULL COUNTS")
    print_separator("-")

    try:
        nulls = lazy.select(
            [
                pl.col(column).null_count().alias(column)
                for column in schema.names()
            ]
        ).collect()

        null_items = [
            (column, nulls[column][0])
            for column in schema.names()
            if nulls[column][0] > 0
        ]

        if not null_items:
            print("Aucune valeur nulle.")
        else:
            for column, count in sorted(
                null_items,
                key=lambda x: x[1],
                reverse=True,
            ):
                print(f"{column:<45} {count:,}")

    except Exception as exc:
        print(f"Impossible de calculer les null counts : {exc}")


def main() -> None:
    print_separator()
    print("QWEN TRADING - DASHBOARD PARQUET INSPECTION")
    print_separator()

    for file_path in FILES:
        inspect_parquet(file_path)

    print()
    print_separator()
    print("INSPECTION COMPLETE")
    print_separator()


if __name__ == "__main__":
    main()
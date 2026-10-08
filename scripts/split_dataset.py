from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/btc_usdc_1h_labeled.parquet"
)

OUTPUT_DIR = Path(
    "data/splits"
)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Notre horizon futur maximal est 24h.
PURGE_HOURS = 24


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SPLIT CHRONOLOGIQUE")
    print("=" * 70)

    print("Source :", INPUT_FILE)

    df = pd.read_parquet(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # On garde uniquement les samples réellement utilisables.
    # --------------------------------------------------------

    df = df[
        df["sample_ready"]
    ].copy()

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        "Samples utilisables :",
        f"{len(df):,}",
    )

    if len(df) == 0:
        raise RuntimeError(
            "Aucun sample_ready disponible."
        )


    # ========================================================
    # FRONTIERES BRUTES
    # ========================================================

    n = len(df)

    validation_index = int(
        n * TRAIN_RATIO
    )

    test_index = int(
        n * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    validation_start = df.iloc[
        validation_index
    ]["timestamp"]

    test_start = df.iloc[
        test_index
    ]["timestamp"]


    print()
    print(
        "Début validation :",
        validation_start,
    )

    print(
        "Début test       :",
        test_start,
    )


    # ========================================================
    # PURGE
    # ========================================================
    #
    # Un sample à T utilise des labels jusqu'à T+24h.
    #
    # Donc les samples du train trop proches du début
    # de validation sont supprimés.
    #
    # Même principe validation -> test.
    # ========================================================

    purge_delta = pd.Timedelta(
        hours=PURGE_HOURS
    )


    train_cutoff = (
        validation_start
        - purge_delta
    )

    validation_cutoff = (
        test_start
        - purge_delta
    )


    train = df[
        df["timestamp"]
        < train_cutoff
    ].copy()


    validation = df[
        (df["timestamp"] >= validation_start)
        &
        (df["timestamp"] < validation_cutoff)
    ].copy()


    test = df[
        df["timestamp"]
        >= test_start
    ].copy()


    # ========================================================
    # VERIFICATIONS
    # ========================================================

    if len(train) == 0:
        raise RuntimeError(
            "Train vide."
        )

    if len(validation) == 0:
        raise RuntimeError(
            "Validation vide."
        )

    if len(test) == 0:
        raise RuntimeError(
            "Test vide."
        )


    # --------------------------------------------------------
    # Vérification stricte :
    # dernier label train < première observation validation
    # --------------------------------------------------------

    last_train_timestamp = train[
        "timestamp"
    ].max()

    first_validation_timestamp = validation[
        "timestamp"
    ].min()

    last_validation_timestamp = validation[
        "timestamp"
    ].max()

    first_test_timestamp = test[
        "timestamp"
    ].min()


    train_gap = (
        first_validation_timestamp
        - last_train_timestamp
    )

    validation_gap = (
        first_test_timestamp
        - last_validation_timestamp
    )


    if train_gap <= purge_delta:
        raise RuntimeError(
            "Purge train/validation insuffisante."
        )

    if validation_gap <= purge_delta:
        raise RuntimeError(
            "Purge validation/test insuffisante."
        )


    # ========================================================
    # AJOUT IDENTIFIANT SPLIT
    # ========================================================

    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"


    # ========================================================
    # SAUVEGARDE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    train_path = (
        OUTPUT_DIR
        / "train.parquet"
    )

    validation_path = (
        OUTPUT_DIR
        / "validation.parquet"
    )

    test_path = (
        OUTPUT_DIR
        / "test.parquet"
    )


    train.to_parquet(
        train_path,
        index=False,
        compression="zstd",
    )

    validation.to_parquet(
        validation_path,
        index=False,
        compression="zstd",
    )

    test.to_parquet(
        test_path,
        index=False,
        compression="zstd",
    )


    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {

        "source": str(
            INPUT_FILE
        ),

        "purge_hours":
            PURGE_HOURS,

        "ratios": {
            "train":
                TRAIN_RATIO,

            "validation":
                VALIDATION_RATIO,

            "test":
                TEST_RATIO,
        },

        "train": {
            "rows":
                len(train),

            "start":
                str(
                    train["timestamp"].min()
                ),

            "end":
                str(
                    train["timestamp"].max()
                ),
        },

        "validation": {
            "rows":
                len(validation),

            "start":
                str(
                    validation[
                        "timestamp"
                    ].min()
                ),

            "end":
                str(
                    validation[
                        "timestamp"
                    ].max()
                ),
        },

        "test": {
            "rows":
                len(test),

            "start":
                str(
                    test["timestamp"].min()
                ),

            "end":
                str(
                    test["timestamp"].max()
                ),
        },

    }


    manifest_path = (
        OUTPUT_DIR
        / "split_manifest.json"
    )


    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )


    # ========================================================
    # RESUME
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTAT")
    print("=" * 70)


    def print_split(
        name,
        data,
    ):

        percentage = (
            len(data)
            / len(df)
            * 100
        )

        print()
        print(
            f"{name}:"
        )

        print(
            "  lignes :",
            f"{len(data):,}",
            f"({percentage:.2f}%)",
        )

        print(
            "  début  :",
            data["timestamp"].min(),
        )

        print(
            "  fin    :",
            data["timestamp"].max(),
        )


    print_split(
        "TRAIN",
        train,
    )

    print_split(
        "VALIDATION",
        validation,
    )

    print_split(
        "TEST",
        test,
    )


    print()
    print(
        "Gap train -> validation :",
        train_gap,
    )

    print(
        "Gap validation -> test  :",
        validation_gap,
    )


    # ========================================================
    # DISTRIBUTION DES LABELS
    # ========================================================

    print()
    print("=" * 70)
    print("DISTRIBUTION LONG 12H")
    print("=" * 70)


    for name, data in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        distribution = (
            data[
                "long_outcome_12h"
            ]
            .value_counts(
                normalize=True,
                dropna=False,
            )
            .sort_index()
        )

        print()
        print(name)
        print(distribution)


    print()
    print("=" * 70)
    print("FICHIERS")
    print("=" * 70)

    print(train_path.resolve())
    print(validation_path.resolve())
    print(test_path.resolve())
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
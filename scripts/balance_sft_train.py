from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_JSONL = Path("data/sft/train.jsonl")
INPUT_AUDIT = Path("data/sft/train_audit.parquet")

OUTPUT_JSONL = Path("data/sft/train_balanced.jsonl")
OUTPUT_AUDIT = Path("data/sft/train_balanced_audit.parquet")

TARGET_NO_TRADE_RATIO = 0.60

RANDOM_SEED = 3407


# ============================================================
# CHARGEMENT JSONL
# ============================================================

def load_jsonl(path: Path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("REEQUILIBRAGE TRAIN SFT")
    print("=" * 70)

    records = load_jsonl(
        INPUT_JSONL
    )

    audit = pd.read_parquet(
        INPUT_AUDIT
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # Vérification alignement
    # --------------------------------------------------------

    if len(records) != len(audit):

        raise RuntimeError(
            f"JSONL = {len(records)} lignes, "
            f"audit = {len(audit)} lignes. "
            "Les fichiers ne sont pas alignés."
        )


    audit["original_index"] = range(
        len(audit)
    )


    print(
        "Exemples originaux :",
        f"{len(audit):,}",
    )


    # ========================================================
    # GROUPES
    # ========================================================

    no_trade = audit[
        audit["decision"] == "NO_TRADE"
    ].copy()


    directional = audit[
        audit["decision"] != "NO_TRADE"
    ].copy()


    directional_count = len(
        directional
    )


    # ========================================================
    # CALCUL DU NOMBRE DE NO_TRADE
    # ========================================================

    # Si NO_TRADE doit représenter 60%,
    # les classes directionnelles représentent 40%.
    #
    # directional / total = 0.40
    #

    desired_total = round(
        directional_count
        / (1 - TARGET_NO_TRADE_RATIO)
    )


    desired_no_trade = (
        desired_total
        - directional_count
    )


    desired_no_trade = min(
        desired_no_trade,
        len(no_trade),
    )


    print()
    print(
        "Directionnels conservés :",
        f"{directional_count:,}",
    )

    print(
        "NO_TRADE disponibles :",
        f"{len(no_trade):,}",
    )

    print(
        "NO_TRADE conservés :",
        f"{desired_no_trade:,}",
    )


    # ========================================================
    # ECHANTILLONNAGE NO_TRADE
    # ========================================================

    selected_no_trade = no_trade.sample(
        n=desired_no_trade,
        random_state=RANDOM_SEED,
        replace=False,
    )


    # ========================================================
    # FUSION
    # ========================================================

    selected = pd.concat(
        [
            directional,
            selected_no_trade,
        ],
        ignore_index=False,
    )


    # --------------------------------------------------------
    # On remet les exemples dans leur ordre chronologique /
    # ordre original.
    # --------------------------------------------------------

    selected = (
        selected
        .sort_values("original_index")
        .reset_index(drop=True)
    )


    # ========================================================
    # RECONSTRUCTION JSONL
    # ========================================================

    selected_records = []

    for original_index in selected[
        "original_index"
    ]:

        selected_records.append(
            records[
                int(original_index)
            ]
        )


    # ========================================================
    # SAUVEGARDE JSONL
    # ========================================================

    OUTPUT_JSONL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        OUTPUT_JSONL,
        "w",
        encoding="utf-8",
    ) as file:

        for record in selected_records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


    # ========================================================
    # SAUVEGARDE AUDIT
    # ========================================================

    selected.drop(
        columns=[
            "original_index"
        ]
    ).to_parquet(
        OUTPUT_AUDIT,
        index=False,
        compression="zstd",
    )


    # ========================================================
    # STATS
    # ========================================================

    print()
    print("=" * 70)
    print("DISTRIBUTION FINALE")
    print("=" * 70)

    distribution = (
        selected[
            "decision"
        ]
        .value_counts()
    )


    total = len(
        selected
    )


    for label, count in distribution.items():

        pct = (
            count
            / total
            * 100
        )

        print(
            f"{label:12} "
            f"{count:5,} "
            f"({pct:5.2f}%)"
        )


    print()
    print(
        "Total :",
        f"{total:,}",
    )


    print()
    print("=" * 70)
    print("FICHIERS")
    print("=" * 70)

    print(
        OUTPUT_JSONL.resolve()
    )

    print(
        OUTPUT_AUDIT.resolve()
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SOURCE_DIR = Path("data/sft")
OUTPUT_DIR = Path("data/sft_v2")

SEED = 3407

LABELS = [
    "NO_TRADE",
    "LONG_BIAS",
    "SHORT_BIAS",
]


SYSTEM_PROMPT = """
Tu es un classifieur spécialisé dans les setups de trading crypto.

Analyse uniquement les données de marché fournies.

Tu dois choisir UNE SEULE décision parmi :

NO_TRADE
LONG_BIAS
SHORT_BIAS

NO_TRADE signifie qu'aucun avantage directionnel suffisamment clair
ne justifie de privilégier un trade.

LONG_BIAS signifie que le contexte permet de privilégier un scénario long.

SHORT_BIAS signifie que le contexte permet de privilégier un scénario short.

Réponds uniquement sous la forme :

DECISION: LABEL
""".strip()


# ============================================================
# JSONL
# ============================================================

def load_jsonl(path):

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
# USER PROMPT
# ============================================================

def convert_user_prompt(record):

    user = next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "user"
    )

    # Retire uniquement l'ancienne consigne finale.
    marker = "\nAnalyse cette situation."

    if marker in user:

        user = (
            user
            .split(marker, 1)[0]
            .rstrip()
        )

    user += """

Choisis la décision correspondant le mieux à cette situation.

Réponds uniquement avec :

DECISION: NO_TRADE
ou
DECISION: LONG_BIAS
ou
DECISION: SHORT_BIAS
""".rstrip()

    return user


# ============================================================
# BUILD RECORD
# ============================================================

def build_record(
    original_record,
    decision,
):

    return {

        "messages": [

            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },

            {
                "role": "user",
                "content": convert_user_prompt(
                    original_record
                ),
            },

            {
                "role": "assistant",
                "content": (
                    f"DECISION: {decision}"
                ),
            },

        ]
    }


# ============================================================
# TRAIN BALANCE
# ============================================================

def build_train():

    records = load_jsonl(
        SOURCE_DIR / "train.jsonl"
    )

    audit = pd.read_parquet(
        SOURCE_DIR / "train_audit.parquet"
    ).reset_index(drop=True)


    if len(records) != len(audit):

        raise RuntimeError(
            "train.jsonl et train_audit.parquet "
            "ne sont pas alignés."
        )


    counts = (
        audit["decision"]
        .value_counts()
    )


    print("Distribution source :")

    print(counts)


    # La classe minoritaire détermine la taille.
    samples_per_class = min(
        int(
            counts.get(
                label,
                0,
            )
        )
        for label in LABELS
    )


    print()
    print(
        "Samples par classe :",
        samples_per_class,
    )


    selected_indices = []


    for label in LABELS:

        group = audit[
            audit["decision"] == label
        ]


        sampled = group.sample(
            n=samples_per_class,
            random_state=SEED,
            replace=False,
        )


        selected_indices.extend(
            sampled.index.tolist()
        )


    # Remet dans l'ordre chronologique.
    selected_indices = sorted(
        selected_indices
    )


    output_records = []

    output_audit = []


    for index in selected_indices:

        decision = audit.at[
            index,
            "decision"
        ]


        output_records.append(
            build_record(
                records[index],
                decision,
            )
        )


        row = audit.loc[
            index
        ].to_dict()

        row[
            "source_index"
        ] = int(index)

        output_audit.append(
            row
        )


    return (
        output_records,
        pd.DataFrame(
            output_audit
        ),
    )


# ============================================================
# VALIDATION / TEST
# ============================================================

def build_natural_split(
    split,
):

    records = load_jsonl(
        SOURCE_DIR / f"{split}.jsonl"
    )

    audit = pd.read_parquet(
        SOURCE_DIR
        / f"{split}_audit.parquet"
    ).reset_index(drop=True)


    if len(records) != len(audit):

        raise RuntimeError(
            f"{split} non aligné."
        )


    output = []


    for index, record in enumerate(
        records
    ):

        decision = audit.at[
            index,
            "decision"
        ]


        output.append(
            build_record(
                record,
                decision,
            )
        )


    return output, audit


# ============================================================
# SAVE
# ============================================================

def save_jsonl(
    path,
    records,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print("=" * 70)
    print("SFT V2 - DECISION ONLY")
    print("=" * 70)


    # TRAIN
    train_records, train_audit = (
        build_train()
    )


    save_jsonl(
        OUTPUT_DIR
        / "train.jsonl",

        train_records,
    )


    train_audit.to_parquet(
        OUTPUT_DIR
        / "train_audit.parquet",

        index=False,
        compression="zstd",
    )


    # VALIDATION
    validation_records, validation_audit = (
        build_natural_split(
            "validation"
        )
    )


    save_jsonl(
        OUTPUT_DIR
        / "validation.jsonl",

        validation_records,
    )


    validation_audit.to_parquet(
        OUTPUT_DIR
        / "validation_audit.parquet",

        index=False,
        compression="zstd",
    )


    # TEST
    test_records, test_audit = (
        build_natural_split(
            "test"
        )
    )


    save_jsonl(
        OUTPUT_DIR
        / "test.jsonl",

        test_records,
    )


    test_audit.to_parquet(
        OUTPUT_DIR
        / "test_audit.parquet",

        index=False,
        compression="zstd",
    )


    # ========================================================
    # STATS
    # ========================================================

    print()
    print("=" * 70)
    print("TRAIN V2")
    print("=" * 70)


    print(
        "Total :",
        len(train_records)
    )


    print(
        train_audit[
            "decision"
        ]
        .value_counts()
    )


    print()
    print("VALIDATION :", len(validation_records))

    print(
        validation_audit[
            "decision"
        ]
        .value_counts()
    )


    print()
    print("TEST :", len(test_records))

    print(
        test_audit[
            "decision"
        ]
        .value_counts()
    )


if __name__ == "__main__":
    main()
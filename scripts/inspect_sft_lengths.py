from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "Qwen/Qwen3.5-9B"

SFT_DIR = Path("data/sft")

SPLITS = [
    "train",
    "validation",
    "test",
]

LIMITS = [
    256,
    384,
    512,
    640,
    768,
    1024,
]


# ============================================================
# TOKENIZER
# ============================================================

print("=" * 70)
print("CHARGEMENT TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)


# ============================================================
# TOKEN COUNT
# ============================================================

def count_tokens(messages) -> int:
    """
    Retourne le vrai nombre de tokens du chat complet.

    return_dict=False est volontaire :
    Transformers 5.x retourne sinon un dictionnaire
    avec input_ids / attention_mask.
    """

    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=False,
    )

    return len(token_ids)


# ============================================================
# ANALYSE SPLIT
# ============================================================

def analyze_split(split_name: str):

    path = (
        SFT_DIR
        / f"{split_name}.jsonl"
    )

    lengths = []

    longest_record = None
    longest_length = 0
    longest_index = -1


    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for index, line in enumerate(file):

            record = json.loads(
                line
            )

            messages = record[
                "messages"
            ]

            length = count_tokens(
                messages
            )

            lengths.append(
                length
            )

            if length > longest_length:

                longest_length = length
                longest_record = record
                longest_index = index


    lengths = np.array(
        lengths,
        dtype=np.int32,
    )


    # ========================================================
    # STATS
    # ========================================================

    print()
    print("=" * 70)
    print(split_name.upper())
    print("=" * 70)

    print(
        "Exemples :",
        f"{len(lengths):,}",
    )

    print(
        "Minimum  :",
        int(lengths.min()),
    )

    print(
        "Moyenne  :",
        round(
            float(lengths.mean()),
            1,
        ),
    )

    print(
        "Médiane  :",
        int(
            np.median(lengths)
        ),
    )


    print()
    print("Percentiles :")

    for percentile in [
        50,
        75,
        90,
        95,
        97,
        99,
        100,
    ]:

        value = np.percentile(
            lengths,
            percentile,
        )

        print(
            f"  P{percentile:<3}: "
            f"{value:.0f} tokens"
        )


    print()
    print("Dépassements :")

    for limit in LIMITS:

        count = int(
            (lengths > limit).sum()
        )

        percentage = (
            count
            / len(lengths)
            * 100
        )

        print(
            f"  > {limit:4} tokens : "
            f"{count:5,} "
            f"({percentage:6.2f}%)"
        )


    # ========================================================
    # RECOMMANDATION CONTEXTE
    # ========================================================

    p95 = int(
        np.ceil(
            np.percentile(
                lengths,
                95,
            )
        )
    )

    p99 = int(
        np.ceil(
            np.percentile(
                lengths,
                99,
            )
        )
    )

    print()
    print("Suggestion :")

    if p99 <= 384:

        recommendation = 384

    elif p99 <= 512:

        recommendation = 512

    elif p99 <= 640:

        recommendation = 640

    elif p99 <= 768:

        recommendation = 768

    else:

        recommendation = 1024


    print(
        f"  P95 : {p95}"
    )

    print(
        f"  P99 : {p99}"
    )

    print(
        f"  MAX_SEQ_LENGTH conseillé : "
        f"{recommendation}"
    )


    # ========================================================
    # PLUS LONG EXEMPLE
    # ========================================================

    print()
    print(
        "Plus long exemple :",
        longest_index,
    )

    print(
        "Longueur :",
        longest_length,
        "tokens",
    )


    if longest_record:

        print()
        print("-" * 70)
        print("APERÇU PLUS LONG EXEMPLE")
        print("-" * 70)

        for message in longest_record[
            "messages"
        ]:

            print()
            print(
                message["role"].upper()
            )

            content = message[
                "content"
            ]

            # Seulement pour ne pas saturer le terminal.
            print(
                content[:800]
            )

            if len(content) > 800:
                print("...")


# ============================================================
# MAIN
# ============================================================

def main():

    for split in SPLITS:

        analyze_split(
            split
        )


    print()
    print("=" * 70)
    print("TERMINE")
    print("=" * 70)


if __name__ == "__main__":
    main()
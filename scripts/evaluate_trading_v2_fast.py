from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch

# IMPORTANT : Unsloth avant transformers / peft / trl
import unsloth
from unsloth import FastLanguageModel


# ============================================================
# CONFIG
# ============================================================

ADAPTER_DIR = Path(
    "models/qwen3.5-9b-trading-v2/final_adapter"
)

TEST_JSONL = Path(
    "data/sft_v2/test.jsonl"
)

TEST_AUDIT = Path(
    "data/sft_v2/test_audit.parquet"
)

OUTPUT_DIR = Path(
    "data/evaluation/qwen3.5-9b-trading-v2-fast"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR / "predictions_fast.jsonl"
)

RESULTS_FILE = (
    OUTPUT_DIR / "results_fast.parquet"
)

METRICS_FILE = (
    OUTPUT_DIR / "metrics_fast.json"
)

MAX_SEQ_LENGTH = 768
MAX_NEW_TOKENS = 12

LABELS = [
    "NO_TRADE",
    "LONG_BIAS",
    "SHORT_BIAS",
]

TP_R = 1.5
SL_R = 1.0


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
    )

    return parser.parse_args()


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
# PROMPT RAPIDE
# ============================================================

def build_fast_messages(record):

    system_message = next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "system"
    )

    user_message = next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "user"
    )

    # --------------------------------------------------------
    # On retire uniquement l'ancienne instruction finale :
    #
    # Analyse cette situation.
    # Réponds avec exactement les sections...
    #
    # Toutes les données marché restent identiques.
    # --------------------------------------------------------

    marker = "\nAnalyse cette situation."

    if marker in user_message:

        user_message = (
            user_message
            .split(marker, 1)[0]
            .rstrip()
        )


    user_message += """

Pour ce benchmark, classe uniquement cette situation.

Réponds avec EXACTEMENT un seul des trois labels suivants et rien d'autre :

NO_TRADE
LONG_BIAS
SHORT_BIAS
""".rstrip()


    return [
        message
        for message in record["messages"]
        if message["role"] != "assistant"
    ]


# ============================================================
# PARSING
# ============================================================

def parse_prediction(text):

    match = re.search(
        r"DECISION\s*:\s*"
        r"(NO_TRADE|LONG_BIAS|SHORT_BIAS)",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        return (
            match
            .group(1)
            .upper()
        )

    upper = text.upper()

    found = [
        label
        for label in LABELS
        if label in upper
    ]

    if len(found) == 1:
        return found[0]

    return "INVALID"
# ============================================================
# INFERENCE BATCH
# ============================================================

@torch.inference_mode()
def predict_batch(
    model,
    tokenizer,
    records,
):

    prompts = []

    for record in records:

        messages = build_fast_messages(
            record
        )

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        prompts.append(
            prompt
        )


    inputs = tokenizer(
        prompts,

        return_tensors="pt",

        padding=True,

        truncation=True,

        max_length=(
            MAX_SEQ_LENGTH
            - MAX_NEW_TOKENS
        ),
    )


    inputs = {
        key: value.to("cuda")
        for key, value
        in inputs.items()
    }


    padded_prompt_length = (
        inputs["input_ids"]
        .shape[1]
    )


    outputs = model.generate(

        **inputs,

        max_new_tokens=MAX_NEW_TOKENS,

        do_sample=False,

        use_cache=True,

        pad_token_id=(
            tokenizer.pad_token_id
        ),

        eos_token_id=(
            tokenizer.eos_token_id
        ),
    )


    generated = outputs[
        :,
        padded_prompt_length:
    ]


    texts = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )


    predictions = [
        parse_prediction(text)
        for text in texts
    ]


    del inputs
    del outputs
    del generated

    return predictions, texts


# ============================================================
# EXISTING RESULTS
# ============================================================

def load_existing():

    existing = {}

    if not PREDICTIONS_FILE.exists():
        return existing


    with open(
        PREDICTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            result = json.loads(line)

            existing[
                int(result["index"])
            ] = result


    return existing


# ============================================================
# HEURISTIQUE
# ============================================================

def heuristic_prediction(score):

    if score >= 2:
        return "LONG_BIAS"

    if score <= -2:
        return "SHORT_BIAS"

    return "NO_TRADE"


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    df,
    prediction_column,
):

    predictions = df[
        prediction_column
    ]

    correct = (
        predictions
        == df["target"]
    ).sum()


    result = {

        "accuracy":
            float(
                correct
                / len(df)
            ),

        "correct":
            int(correct),

        "total":
            int(len(df)),

        "invalid":
            int(
                (
                    predictions
                    == "INVALID"
                ).sum()
            ),

        "per_class": {},
    }


    for label in LABELS:

        tp = (
            (df["target"] == label)
            &
            (predictions == label)
        ).sum()

        fp = (
            (df["target"] != label)
            &
            (predictions == label)
        ).sum()

        fn = (
            (df["target"] == label)
            &
            (predictions != label)
        ).sum()


        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)

            if (
                precision
                + recall
            )

            else 0.0
        )


        result[
            "per_class"
        ][label] = {

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),

            "support":
                int(
                    (
                        df["target"]
                        == label
                    ).sum()
                ),

            "predicted":
                int(
                    (
                        predictions
                        == label
                    ).sum()
                ),
        }


    return result


# ============================================================
# TRADING METRICS
# ============================================================

def trading_metrics(
    df,
    prediction_column,
):

    trades = df[
        df[prediction_column].isin(
            [
                "LONG_BIAS",
                "SHORT_BIAS",
            ]
        )
    ].copy()


    if len(trades) == 0:

        return {
            "trades": 0,
            "coverage": 0.0,
            "wins": 0,
            "losses": 0,
            "unresolved": 0,
            "resolved": 0,
            "win_rate_resolved": 0.0,
            "expected_r_all_trades": 0.0,
            "expected_r_resolved": 0.0,
            "total_r": 0.0,
        }


    outcomes = []


    for _, row in trades.iterrows():

        prediction = row[
            prediction_column
        ]


        if prediction == "LONG_BIAS":

            outcome = row[
                "long_outcome_12h"
            ]

        else:

            outcome = row[
                "short_outcome_12h"
            ]


        outcomes.append(
            outcome
        )


    outcomes = pd.Series(
        outcomes,
        dtype="float64",
    )


    wins = int(
        (outcomes == 1).sum()
    )

    losses = int(
        (outcomes == -1).sum()
    )

    unresolved = int(
        (outcomes == 0).sum()
    )

    resolved = (
        wins
        + losses
    )


    win_rate = (
        wins / resolved
        if resolved
        else 0.0
    )


    total_r = (
        wins * TP_R
        - losses * SL_R
    )


    return {

        "trades":
            int(len(trades)),

        "coverage":
            float(
                len(trades)
                / len(df)
            ),

        "wins":
            wins,

        "losses":
            losses,

        "unresolved":
            unresolved,

        "resolved":
            resolved,

        "win_rate_resolved":
            float(win_rate),

        "expected_r_all_trades":
            float(
                total_r
                / len(trades)
            ),

        "expected_r_resolved":
            float(
                total_r
                / resolved
                if resolved
                else 0.0
            ),

        "total_r":
            float(total_r),
    }


# ============================================================
# PRINT
# ============================================================

def print_metrics(
    name,
    classification,
    trading,
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)


    print(
        "Accuracy :",
        f"{classification['accuracy'] * 100:.2f}%"
    )

    print(
        "Correct :",
        f"{classification['correct']} / "
        f"{classification['total']}"
    )

    print(
        "Invalid :",
        classification["invalid"]
    )


    print()
    print("Classes :")


    for label in LABELS:

        stats = (
            classification[
                "per_class"
            ][label]
        )

        print(
            f"{label:12} | "
            f"P={stats['precision'] * 100:6.2f}% | "
            f"R={stats['recall'] * 100:6.2f}% | "
            f"F1={stats['f1'] * 100:6.2f}% | "
            f"pred={stats['predicted']:4d}"
        )


    print()
    print("Trading :")

    print(
        "  Trades :",
        trading["trades"],
        f"({trading['coverage'] * 100:.2f}%)"
    )

    print(
        "  Wins :",
        trading["wins"]
    )

    print(
        "  Losses :",
        trading["losses"]
    )

    print(
        "  Non résolus :",
        trading["unresolved"]
    )

    print(
        "  Win rate résolu :",
        f"{trading['win_rate_resolved'] * 100:.2f}%"
    )

    print(
        "  Expected R/trade :",
        f"{trading['expected_r_all_trades']:+.4f} R"
    )

    print(
        "  Total R :",
        f"{trading['total_r']:+.2f} R"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()


    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA indisponible."
        )


    print("=" * 70)
    print("QWEN TRADING V1 - FAST BENCHMARK")
    print("=" * 70)

    print(
        "GPU :",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU visibles :",
        torch.cuda.device_count()
    )


    # ========================================================
    # DATA
    # ========================================================

    records = load_jsonl(
        TEST_JSONL
    )

    audit = (
        pd.read_parquet(
            TEST_AUDIT
        )
        .reset_index(drop=True)
    )


    if len(records) != len(audit):

        raise RuntimeError(
            "test.jsonl et audit non alignés."
        )


    total = len(records)


    if args.limit is not None:

        total = min(
            total,
            args.limit,
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    if (
        args.fresh
        and PREDICTIONS_FILE.exists()
    ):

        PREDICTIONS_FILE.unlink()


    existing = load_existing()


    print(
        "Exemples :",
        total
    )

    print(
        "Déjà calculés :",
        len(
            [
                x
                for x in existing
                if x < total
            ]
        )
    )


    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("Chargement modèle...")


    model, tokenizer_or_processor = (
        FastLanguageModel.from_pretrained(

            model_name=str(
                ADAPTER_DIR
            ),

            max_seq_length=(
                MAX_SEQ_LENGTH
            ),

            dtype=None,

            load_in_4bit=True,

            text_only=True,
        )
    )


    if hasattr(
        tokenizer_or_processor,
        "tokenizer",
    ):

        tokenizer = (
            tokenizer_or_processor
            .tokenizer
        )

    else:

        tokenizer = (
            tokenizer_or_processor
        )


    # Important pour génération batch.
    tokenizer.padding_side = "left"


    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )


    FastLanguageModel.for_inference(
        model
    )

    model.eval()


    print(
        "Modèle chargé."
    )


    # ========================================================
    # TODO
    # ========================================================

    todo = [

        index

        for index in range(total)

        if index not in existing
    ]


    batch_size = (
        args.batch_size
    )

    position = 0

    start_time = (
        time.perf_counter()
    )


    print()
    print(
        "Batch initial :",
        batch_size
    )

    print(
        "À calculer :",
        len(todo)
    )


    # ========================================================
    # INFERENCE
    # ========================================================

    while position < len(todo):

        indices = todo[
            position:
            position + batch_size
        ]


        batch_records = [
            records[index]
            for index in indices
        ]


        try:

            predictions, texts = (
                predict_batch(

                    model,
                    tokenizer,
                    batch_records,
                )
            )


        except (
            torch.OutOfMemoryError,
            RuntimeError,
        ) as error:

            if (
                "out of memory"
                not in str(error).lower()
                and not isinstance(
                    error,
                    torch.OutOfMemoryError,
                )
            ):
                raise


            gc.collect()
            torch.cuda.empty_cache()


            if batch_size <= 1:
                raise


            batch_size = max(
                1,
                batch_size // 2
            )


            print()
            print(
                "VRAM insuffisante -> "
                f"batch réduit à {batch_size}"
            )

            continue


        # ----------------------------------------------------
        # SAVE IMMÉDIAT
        # ----------------------------------------------------

        with open(
            PREDICTIONS_FILE,
            "a",
            encoding="utf-8",
        ) as file:

            for (
                index,
                prediction,
                text,
            ) in zip(
                indices,
                predictions,
                texts,
            ):

                target = audit.at[
                    index,
                    "decision"
                ]


                item = {

                    "index":
                        index,

                    "target":
                        target,

                    "prediction":
                        prediction,

                    "generated":
                        text.strip(),
                }


                file.write(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                    )
                    + "\n"
                )


                existing[index] = (
                    item
                )


        position += len(
            indices
        )


        # ----------------------------------------------------
        # PROGRESSION
        # ----------------------------------------------------

        done = len(
            [
                index
                for index in existing
                if index < total
            ]
        )


        elapsed = (
            time.perf_counter()
            - start_time
        )


        new_done = position


        speed = (
            new_done / elapsed
            if elapsed
            else 0.0
        )


        remaining = (
            len(todo)
            - position
        )


        eta_seconds = (
            remaining / speed
            if speed
            else 0
        )


        correct = sum(

            item["prediction"]
            == item["target"]

            for index, item
            in existing.items()

            if index < total
        )


        accuracy = (
            correct
            / done
            * 100
        )


        invalid = sum(

            item["prediction"]
            == "INVALID"

            for index, item
            in existing.items()

            if index < total
        )


        if (
            done <= 32
            or done % 64 < batch_size
            or done == total
        ):

            print(
                f"[{done:4d}/{total}] "
                f"batch={batch_size} | "
                f"{speed:.2f} ex/s | "
                f"acc={accuracy:6.2f}% | "
                f"invalid={invalid} | "
                f"ETA={eta_seconds / 60:.1f} min"
            )


    # ========================================================
    # RESULTS
    # ========================================================

    ordered = [

        existing[index]

        for index in range(total)
    ]


    predictions_df = (
        pd.DataFrame(
            ordered
        )
    )


    results = (
        audit
        .iloc[:total]
        .copy()
        .reset_index(drop=True)
    )


    results["target"] = (
        predictions_df[
            "target"
        ].tolist()
    )


    results[
        "qwen_prediction"
    ] = (
        predictions_df[
            "prediction"
        ].tolist()
    )


    results["generated"] = (
        predictions_df[
            "generated"
        ].tolist()
    )


    results[
        "heuristic_prediction"
    ] = results[
        "bias_score"
    ].apply(
        heuristic_prediction
    )


    results.to_parquet(
        RESULTS_FILE,
        index=False,
        compression="zstd",
    )


    # ========================================================
    # METRICS
    # ========================================================

    qwen_class = (
        classification_metrics(
            results,
            "qwen_prediction",
        )
    )

    qwen_trade = (
        trading_metrics(
            results,
            "qwen_prediction",
        )
    )


    heuristic_class = (
        classification_metrics(
            results,
            "heuristic_prediction",
        )
    )

    heuristic_trade = (
        trading_metrics(
            results,
            "heuristic_prediction",
        )
    )


    print_metrics(
        "QWEN V1 FAST",
        qwen_class,
        qwen_trade,
    )


    print_metrics(
        "HEURISTIQUE",
        heuristic_class,
        heuristic_trade,
    )


    # ========================================================
    # CONFUSION
    # ========================================================

    print()
    print("=" * 70)
    print("MATRICE DE CONFUSION QWEN")
    print("=" * 70)


    matrix = pd.crosstab(
        results["target"],
        results["qwen_prediction"],
        rownames=["REEL"],
        colnames=["PREDIT"],
        dropna=False,
    )


    print(
        matrix.to_string()
    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {

        "examples":
            total,

        "qwen": {
            "classification":
                qwen_class,

            "trading":
                qwen_trade,
        },

        "heuristic": {
            "classification":
                heuristic_class,

            "trading":
                heuristic_trade,
        },
    }


    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )


    print()
    print("=" * 70)
    print("TERMINE")
    print("=" * 70)

    print(
        RESULTS_FILE.resolve()
    )

    print(
        METRICS_FILE.resolve()
    )


if __name__ == "__main__":
    main()
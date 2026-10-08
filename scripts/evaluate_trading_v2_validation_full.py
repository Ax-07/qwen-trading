from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch

# IMPORTANT :
# Unsloth doit être importé avant transformers / peft / trl.
import unsloth

from unsloth import FastLanguageModel


# ============================================================
# CONFIGURATION
# ============================================================

ADAPTER_DIR = Path(
    "models/qwen3.5-9b-trading-v2/final_adapter"
)

VALIDATION_JSONL = Path(
    "data/sft_v2/validation.jsonl"
)

VALIDATION_AUDIT = Path(
    "data/sft_v2/validation_audit.parquet"
)

OUTPUT_DIR = Path(
    "data/evaluation/qwen3.5-9b-trading-v2-validation"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR / "predictions.jsonl"
)

RESULTS_FILE = (
    OUTPUT_DIR / "results.parquet"
)

METRICS_FILE = (
    OUTPUT_DIR / "metrics.json"
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
        help="Batch inference. Défaut : 8",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limiter le nombre d'exemples.",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Efface les anciennes prédictions "
            "et recommence depuis zéro."
        ),
    )

    return parser.parse_args()


# ============================================================
# JSONL
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
# PROMPT V2
# ============================================================

def build_messages(record):

    # On garde EXACTEMENT le prompt de la V2.
    #
    # On retire uniquement le message assistant
    # qui contient évidemment la vérité terrain.

    return [
        message
        for message in record["messages"]
        if message["role"] != "assistant"
    ]


# ============================================================
# PARSING DE LA DECISION
# ============================================================

def parse_prediction(text: str):

    # Format attendu :
    #
    # DECISION: LONG_BIAS

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


    # Fallback si le modèle donne uniquement le label.

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
# HEURISTIQUE
# ============================================================

def heuristic_prediction(
    bias_score,
):

    if bias_score >= 2:

        return "LONG_BIAS"


    if bias_score <= -2:

        return "SHORT_BIAS"


    return "NO_TRADE"


# ============================================================
# LOAD PREDICTIONS EXISTANTES
# ============================================================

def load_existing_predictions():

    existing = {}

    if not PREDICTIONS_FILE.exists():

        return existing


    with open(
        PREDICTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            item = json.loads(
                line
            )

            existing[
                int(item["index"])
            ] = item


    return existing


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

        messages = build_messages(
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


    # Avec padding=True, tous les prompts du batch
    # ont la même longueur après padding.
    prompt_length = (
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


    generated_ids = outputs[
        :,
        prompt_length:
    ]


    texts = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )


    predictions = [
        parse_prediction(text)
        for text in texts
    ]


    del inputs
    del outputs
    del generated_ids


    return (
        predictions,
        texts,
    )


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    df: pd.DataFrame,
    prediction_column: str,
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

        "per_class":
            {},
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


        if precision + recall:

            f1 = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

        else:

            f1 = 0.0


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
# TRADING METRICS - GLOBAL
# ============================================================

def trading_metrics(
    df: pd.DataFrame,
    prediction_column: str,
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


        elif prediction == "SHORT_BIAS":

            outcome = row[
                "short_outcome_12h"
            ]


        else:

            continue


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


    expected_r_all = (
        total_r
        / len(trades)
    )


    expected_r_resolved = (
        total_r
        / resolved
        if resolved
        else 0.0
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
            float(expected_r_all),

        "expected_r_resolved":
            float(
                expected_r_resolved
            ),

        "total_r":
            float(total_r),
    }


# ============================================================
# TRADING METRICS - PAR DIRECTION
# ============================================================

def side_metrics(
    df: pd.DataFrame,
    prediction_column: str,
    prediction_label: str,
    outcome_column: str,
):

    trades = df[
        df[prediction_column]
        == prediction_label
    ].copy()


    if len(trades) == 0:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "unresolved": 0,
            "resolved": 0,
            "win_rate_resolved": 0.0,
            "expected_r_all": 0.0,
            "expected_r_resolved": 0.0,
            "total_r": 0.0,
        }


    outcomes = trades[
        outcome_column
    ]


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


    expected_r_all = (
        total_r
        / len(trades)
    )


    expected_r_resolved = (
        total_r
        / resolved
        if resolved
        else 0.0
    )


    return {

        "trades":
            int(len(trades)),

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

        "expected_r_all":
            float(expected_r_all),

        "expected_r_resolved":
            float(expected_r_resolved),

        "total_r":
            float(total_r),
    }


# ============================================================
# PRINT CLASSIFICATION
# ============================================================

def print_classification(
    title,
    metrics,
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


    print(
        "Accuracy :",
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        "Correct :",
        f"{metrics['correct']} / "
        f"{metrics['total']}"
    )

    print(
        "Invalid :",
        metrics["invalid"]
    )


    print()
    print("Classes :")


    for label in LABELS:

        stats = metrics[
            "per_class"
        ][label]


        print(
            f"{label:12} | "
            f"P={stats['precision'] * 100:6.2f}% | "
            f"R={stats['recall'] * 100:6.2f}% | "
            f"F1={stats['f1'] * 100:6.2f}% | "
            f"pred={stats['predicted']:4d} | "
            f"réel={stats['support']:4d}"
        )


# ============================================================
# PRINT TRADING GLOBAL
# ============================================================

def print_trading(
    title,
    stats,
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


    print(
        "Trades         :",
        stats["trades"]
    )

    print(
        "Coverage       :",
        f"{stats['coverage'] * 100:.2f}%"
    )

    print(
        "Wins           :",
        stats["wins"]
    )

    print(
        "Losses         :",
        stats["losses"]
    )

    print(
        "Non résolus    :",
        stats["unresolved"]
    )

    print(
        "Win rate       :",
        f"{stats['win_rate_resolved'] * 100:.2f}%"
    )

    print(
        "Expected R     :",
        f"{stats['expected_r_all_trades']:+.4f} R/trade"
    )

    print(
        "Expected R resolved :",
        f"{stats['expected_r_resolved']:+.4f} R/trade"
    )

    print(
        "Total R        :",
        f"{stats['total_r']:+.2f} R"
    )


# ============================================================
# PRINT SIDE
# ============================================================

def print_side(
    title,
    stats,
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


    print(
        "Trades         :",
        stats["trades"]
    )

    print(
        "Wins           :",
        stats["wins"]
    )

    print(
        "Losses         :",
        stats["losses"]
    )

    print(
        "Non résolus    :",
        stats["unresolved"]
    )

    print(
        "Win rate       :",
        f"{stats['win_rate_resolved'] * 100:.2f}%"
    )

    print(
        "Expected R     :",
        f"{stats['expected_r_all']:+.4f} R/trade"
    )

    print(
        "Expected R resolved :",
        f"{stats['expected_r_resolved']:+.4f} R/trade"
    )

    print(
        "Total R        :",
        f"{stats['total_r']:+.2f} R"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()


    print("=" * 70)
    print("QWEN TRADING V2 - VALIDATION COMPLETE")
    print("=" * 70)


    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA indisponible."
        )


    print(
        "GPU :",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU visibles :",
        torch.cuda.device_count()
    )


    # ========================================================
    # DATASET
    # ========================================================

    records = load_jsonl(
        VALIDATION_JSONL
    )


    audit = (
        pd.read_parquet(
            VALIDATION_AUDIT
        )
        .reset_index(drop=True)
    )


    if len(records) != len(audit):

        raise RuntimeError(
            "validation.jsonl et "
            "validation_audit.parquet "
            "ne sont pas alignés."
        )


    total = len(records)


    if args.limit is not None:

        total = min(
            total,
            args.limit,
        )


    print(
        "Exemples :",
        total
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    if args.fresh:

        if PREDICTIONS_FILE.exists():

            PREDICTIONS_FILE.unlink()


    existing = load_existing_predictions()


    existing_count = len(
        [
            index
            for index in existing
            if index < total
        ]
    )


    print(
        "Déjà calculés :",
        existing_count
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
    # LISTE DES EXEMPLES RESTANTS
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

            predictions, texts = predict_batch(

                model=model,

                tokenizer=tokenizer,

                records=batch_records,
            )


        except (
            torch.OutOfMemoryError,
            RuntimeError,
        ) as error:


            is_oom = (
                isinstance(
                    error,
                    torch.OutOfMemoryError,
                )
                or
                "out of memory"
                in str(error).lower()
            )


            if not is_oom:

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


        # ====================================================
        # SAVE IMMÉDIAT
        # ====================================================

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
                        int(index),

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


                existing[
                    index
                ] = item


        position += len(
            indices
        )


        # ====================================================
        # PROGRESSION
        # ====================================================

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


        newly_done = (
            position
        )


        speed = (
            newly_done
            / elapsed
            if elapsed
            else 0.0
        )


        remaining = (
            len(todo)
            - position
        )


        eta_seconds = (
            remaining
            / speed
            if speed
            else 0.0
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
            if done
            else 0.0
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
            or
            done % 64 < batch_size
            or
            done == total
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
    # CONSTRUCTION RESULTATS
    # ========================================================

    ordered = [

        existing[index]

        for index in range(total)
    ]


    prediction_df = pd.DataFrame(
        ordered
    )


    results = (
        audit
        .iloc[:total]
        .copy()
        .reset_index(drop=True)
    )


    results[
        "target"
    ] = prediction_df[
        "target"
    ].tolist()


    results[
        "qwen_prediction"
    ] = prediction_df[
        "prediction"
    ].tolist()


    results[
        "generated"
    ] = prediction_df[
        "generated"
    ].tolist()


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
    # CLASSIFICATION
    # ========================================================

    qwen_classification = (
        classification_metrics(
            results,
            "qwen_prediction",
        )
    )


    heuristic_classification = (
        classification_metrics(
            results,
            "heuristic_prediction",
        )
    )


    # ========================================================
    # GLOBAL TRADING
    # ========================================================

    qwen_global = (
        trading_metrics(
            results,
            "qwen_prediction",
        )
    )


    heuristic_global = (
        trading_metrics(
            results,
            "heuristic_prediction",
        )
    )


    # ========================================================
    # LONG / SHORT QWEN
    # ========================================================

    qwen_long = (
        side_metrics(

            results,

            "qwen_prediction",

            "LONG_BIAS",

            "long_outcome_12h",
        )
    )


    qwen_short = (
        side_metrics(

            results,

            "qwen_prediction",

            "SHORT_BIAS",

            "short_outcome_12h",
        )
    )


    # ========================================================
    # LONG / SHORT HEURISTIQUE
    # ========================================================

    heuristic_long = (
        side_metrics(

            results,

            "heuristic_prediction",

            "LONG_BIAS",

            "long_outcome_12h",
        )
    )


    heuristic_short = (
        side_metrics(

            results,

            "heuristic_prediction",

            "SHORT_BIAS",

            "short_outcome_12h",
        )
    )


    # ========================================================
    # AFFICHAGE
    # ========================================================

    print_classification(
        "QWEN V2 - CLASSIFICATION",
        qwen_classification,
    )


    print()
    print("=" * 70)
    print("MATRICE DE CONFUSION QWEN")
    print("=" * 70)


    qwen_matrix = pd.crosstab(
        results["target"],
        results["qwen_prediction"],
        rownames=["REEL"],
        colnames=["PREDIT"],
        dropna=False,
    )


    print(
        qwen_matrix.to_string()
    )


    print_trading(
        "QWEN V2 - GLOBAL",
        qwen_global,
    )


    print_side(
        "QWEN V2 - LONG ONLY",
        qwen_long,
    )


    print_side(
        "QWEN V2 - SHORT ONLY",
        qwen_short,
    )


    print_classification(
        "HEURISTIQUE - CLASSIFICATION",
        heuristic_classification,
    )


    print_trading(
        "HEURISTIQUE - GLOBAL",
        heuristic_global,
    )


    print_side(
        "HEURISTIQUE - LONG ONLY",
        heuristic_long,
    )


    print_side(
        "HEURISTIQUE - SHORT ONLY",
        heuristic_short,
    )


    # ========================================================
    # COMPARAISON PRINCIPALE
    # ========================================================

    print()
    print("=" * 70)
    print("COMPARAISON LONG - VALIDATION")
    print("=" * 70)


    print(
        f"Qwen LONG : "
        f"{qwen_long['trades']} trades | "
        f"WR={qwen_long['win_rate_resolved'] * 100:.2f}% | "
        f"ER={qwen_long['expected_r_all']:+.4f} R | "
        f"Total={qwen_long['total_r']:+.2f} R"
    )


    print(
        f"Heur LONG : "
        f"{heuristic_long['trades']} trades | "
        f"WR={heuristic_long['win_rate_resolved'] * 100:.2f}% | "
        f"ER={heuristic_long['expected_r_all']:+.4f} R | "
        f"Total={heuristic_long['total_r']:+.2f} R"
    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {

        "examples":
            int(total),

        "qwen": {

            "classification":
                qwen_classification,

            "global":
                qwen_global,

            "long":
                qwen_long,

            "short":
                qwen_short,
        },

        "heuristic": {

            "classification":
                heuristic_classification,

            "global":
                heuristic_global,

            "long":
                heuristic_long,

            "short":
                heuristic_short,
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
        "Résultats :",
        RESULTS_FILE.resolve()
    )


    print(
        "Metrics :",
        METRICS_FILE.resolve()
    )


if __name__ == "__main__":

    main()
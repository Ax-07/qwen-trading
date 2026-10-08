from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch

# IMPORTANT : Unsloth avant transformers / peft / trl.
import unsloth

from unsloth import FastLanguageModel

from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
)

# ============================================================
# CONFIGURATION
# ============================================================

ADAPTER_DIR = Path("models/qwen3.5-9b-trading-v1/final_adapter")

TEST_JSONL = Path("data/sft/test.jsonl")

TEST_AUDIT = Path("data/sft/test_audit.parquet")

OUTPUT_DIR = Path("data/evaluation/qwen3.5-9b-trading-v1")

PREDICTIONS_FILE = OUTPUT_DIR / "predictions.jsonl"

RESULTS_PARQUET = OUTPUT_DIR / "results.parquet"

METRICS_FILE = OUTPUT_DIR / "metrics.json"

MAX_SEQ_LENGTH = 768

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
        "--limit",
        type=int,
        default=None,
        help=("Limiter le nombre d'exemples. " "Exemple : --limit 25"),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=("Efface les prédictions existantes " "et recommence depuis zéro."),
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

            records.append(json.loads(line))

    return records


# ============================================================
# STOP GENERATION DES QU'UNE DECISION APPARAIT
# ============================================================


class StopOnDecision(StoppingCriteria):

    def __init__(
        self,
        tokenizer,
    ):

        super().__init__()

        self.sequences = []

        for label in LABELS:

            token_ids = tokenizer.encode(
                label,
                add_special_tokens=False,
            )

            self.sequences.append(
                torch.tensor(
                    token_ids,
                    dtype=torch.long,
                )
            )

    def __call__(
        self,
        input_ids,
        scores,
        **kwargs,
    ):

        # batch=1 dans notre évaluation
        sequence = input_ids[0]

        for target in self.sequences:

            length = len(target)

            if len(sequence) < length:
                continue

            current_tail = sequence[-length:]

            if torch.equal(
                current_tail.cpu(),
                target,
            ):
                return True

        return False


# ============================================================
# PARSING DECISION
# ============================================================


def parse_prediction(text: str):

    # Cas idéal :
    #
    # DECISION
    # LONG_BIAS

    match = re.search(
        r"DECISION\s*[:\-]?\s*" r"(NO_TRADE|LONG_BIAS|SHORT_BIAS)",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        return match.group(1).upper()

    # Fallback :
    # recherche brute.
    found = []

    upper = text.upper()

    for label in LABELS:

        if label in upper:
            found.append(label)

    # Un seul label trouvé -> acceptable.
    if len(found) == 1:

        return found[0]

    return "INVALID"


# ============================================================
# BASELINE HEURISTIQUE
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
# CONFUSION MATRIX
# ============================================================


def confusion_matrix(
    df: pd.DataFrame,
    prediction_column: str,
):

    matrix = pd.crosstab(
        df["target"],
        df[prediction_column],
        rownames=["REAL"],
        colnames=["PREDIT"],
        dropna=False,
    )

    expected_columns = LABELS + ["INVALID"]

    matrix = matrix.reindex(
        index=LABELS,
        columns=expected_columns,
        fill_value=0,
    )

    return matrix


# ============================================================
# CLASSIFICATION METRICS
# ============================================================


def classification_metrics(
    df: pd.DataFrame,
    prediction_column: str,
):

    predictions = df[prediction_column]

    correct = (predictions == df["target"]).sum()

    total = len(df)

    metrics = {
        "accuracy": correct / total if total else 0.0,
        "correct": int(correct),
        "total": int(total),
        "invalid": int((predictions == "INVALID").sum()),
    }

    per_class = {}

    for label in LABELS:

        true_positive = ((df["target"] == label) & (predictions == label)).sum()

        false_positive = ((df["target"] != label) & (predictions == label)).sum()

        false_negative = ((df["target"] == label) & (predictions != label)).sum()

        precision_denominator = true_positive + false_positive

        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )

        recall = true_positive / recall_denominator if recall_denominator else 0.0

        if precision + recall:

            f1 = 2 * precision * recall / (precision + recall)

        else:

            f1 = 0.0

        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int((df["target"] == label).sum()),
            "predicted": int((predictions == label).sum()),
        }

    metrics["per_class"] = per_class

    return metrics


# ============================================================
# TRADING METRICS
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

        prediction = row[prediction_column]

        if prediction == "LONG_BIAS":

            outcome = row["long_outcome_12h"]

        elif prediction == "SHORT_BIAS":

            outcome = row["short_outcome_12h"]

        else:

            continue

        outcomes.append(outcome)

    outcomes = pd.Series(
        outcomes,
        dtype="float64",
    )

    wins = int((outcomes == 1).sum())

    losses = int((outcomes == -1).sum())

    unresolved = int((outcomes == 0).sum())

    resolved = wins + losses

    win_rate = wins / resolved if resolved else 0.0

    total_r = wins * TP_R - losses * SL_R

    expected_r_all = total_r / len(trades)

    expected_r_resolved = total_r / resolved if resolved else 0.0

    return {
        "trades": int(len(trades)),
        "coverage": float(len(trades) / len(df)),
        "wins": wins,
        "losses": losses,
        "unresolved": unresolved,
        "resolved": resolved,
        "win_rate_resolved": float(win_rate),
        "expected_r_all_trades": float(expected_r_all),
        "expected_r_resolved": float(expected_r_resolved),
        "total_r": float(total_r),
    }


# ============================================================
# CHARGEMENT PREDICTIONS EXISTANTES
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

            item = json.loads(line)

            existing[int(item["index"])] = item

    return existing


# ============================================================
# GENERATION D'UNE DECISION
# ============================================================


@torch.inference_mode()
def predict_one(
    model,
    tokenizer,
    messages,
    stopping_criteria,
):

    # --------------------------------------------------------
    # IMPORTANT :
    # On retire évidemment la réponse assistant contenant
    # la vérité terrain.
    # --------------------------------------------------------

    input_messages = [message for message in messages if message["role"] != "assistant"]

    inputs = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    prompt_length = inputs["input_ids"].shape[1]

    output = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
        use_cache=True,
        stopping_criteria=(stopping_criteria),
        pad_token_id=(tokenizer.eos_token_id),
    )

    generated_ids = output[0, prompt_length:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    prediction = parse_prediction(generated_text)

    # --------------------------------------------------------
    # Si jamais la décision n'est pas apparue avant
    # max_new_tokens=160, on fait un second essai plus long.
    # --------------------------------------------------------

    if prediction == "INVALID":

        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
            pad_token_id=(tokenizer.eos_token_id),
        )

        generated_ids = output[0, prompt_length:]

        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        prediction = parse_prediction(generated_text)

    return (
        prediction,
        generated_text,
    )


# ============================================================
# AFFICHAGE METRICS
# ============================================================


def print_metrics(
    name: str,
    class_metrics,
    trade_metrics,
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Accuracy : " f"{class_metrics['accuracy'] * 100:.2f}%")

    print(
        f"Correct  : " f"{class_metrics['correct']:,} / " f"{class_metrics['total']:,}"
    )

    print(
        "Invalid  :",
        class_metrics["invalid"],
    )

    print()
    print("Par classe :")

    for label in LABELS:

        stats = class_metrics["per_class"][label]

        print(
            f"{label:12} | "
            f"precision "
            f"{stats['precision'] * 100:6.2f}% | "
            f"recall "
            f"{stats['recall'] * 100:6.2f}% | "
            f"F1 "
            f"{stats['f1'] * 100:6.2f}% | "
            f"pred "
            f"{stats['predicted']:4d}"
        )

    print()
    print("Trading 12H :")

    print(
        "  Trades :",
        trade_metrics["trades"],
        f"({trade_metrics['coverage'] * 100:.2f}% des situations)",
    )

    print(
        "  Wins :",
        trade_metrics["wins"],
    )

    print(
        "  Losses :",
        trade_metrics["losses"],
    )

    print(
        "  Non résolus :",
        trade_metrics["unresolved"],
    )

    print("  Win rate résolu :", f"{trade_metrics['win_rate_resolved'] * 100:.2f}%")

    print("  Expected R / trade :", f"{trade_metrics['expected_r_all_trades']:+.4f} R")

    print(
        "  Expected R / trade résolu :",
        f"{trade_metrics['expected_r_resolved']:+.4f} R",
    )

    print("  Total R théorique :", f"{trade_metrics['total_r']:+.2f} R")


# ============================================================
# MAIN
# ============================================================


def main():

    args = parse_args()

    print("=" * 70)
    print("EVALUATION QWEN TRADING V1")
    print("=" * 70)

    if not torch.cuda.is_available():

        raise RuntimeError("CUDA indisponible.")

    print(
        "GPU :",
        torch.cuda.get_device_name(0),
    )

    print(
        "GPU visibles :",
        torch.cuda.device_count(),
    )

    # ========================================================
    # DATASET
    # ========================================================

    test_records = load_jsonl(TEST_JSONL)

    audit = pd.read_parquet(TEST_AUDIT).reset_index(drop=True)

    if len(test_records) != len(audit):

        raise RuntimeError(f"JSONL={len(test_records)} " f"mais audit={len(audit)}")

    total_examples = len(test_records)

    if args.limit is not None:

        total_examples = min(
            total_examples,
            args.limit,
        )

    print(
        "Exemples à évaluer :",
        f"{total_examples:,}",
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

    print(
        "Déjà calculés :",
        len(existing),
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("CHARGEMENT ADAPTER V1")
    print("=" * 70)

    model, tokenizer_or_processor = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_DIR),
        max_seq_length=(MAX_SEQ_LENGTH),
        dtype=None,
        load_in_4bit=True,
        text_only=True,
    )

    if hasattr(
        tokenizer_or_processor,
        "tokenizer",
    ):

        tokenizer = tokenizer_or_processor.tokenizer

    else:

        tokenizer = tokenizer_or_processor

    FastLanguageModel.for_inference(model)

    stopping_criteria = StoppingCriteriaList([StopOnDecision(tokenizer)])

    print("Adapter chargé.")

    # ========================================================
    # PREDICTIONS
    # ========================================================

    print()
    print("=" * 70)
    print("INFERENCE")
    print("=" * 70)

    for index in range(total_examples):

        if index in existing:

            continue

        record = test_records[index]

        target = audit.at[index, "decision"]

        prediction, generated = predict_one(
            model=model,
            tokenizer=tokenizer,
            messages=record["messages"],
            stopping_criteria=(stopping_criteria),
        )

        result = {
            "index": index,
            "target": target,
            "prediction": prediction,
            "generated": generated,
        }

        with open(
            PREDICTIONS_FILE,
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

        existing[index] = result

        # ----------------------------------------------------
        # PROGRESSION
        # ----------------------------------------------------

        done = len([i for i in existing if i < total_examples])

        if done <= 10 or done % 25 == 0:

            correct = sum(
                1
                for i, item in existing.items()
                if (i < total_examples and item["prediction"] == item["target"])
            )

            current_accuracy = correct / done * 100

            print(
                f"[{done:4d}/{total_examples}] "
                f"pred={prediction:11} "
                f"real={target:11} "
                f"accuracy={current_accuracy:6.2f}%"
            )

    # ========================================================
    # CONSTRUCTION DATAFRAME RESULTATS
    # ========================================================

    selected_predictions = [
        existing[index] for index in range(total_examples) if index in existing
    ]

    predictions_df = pd.DataFrame(selected_predictions).sort_values("index")

    audit_subset = (
        audit.iloc[predictions_df["index"].tolist()].copy().reset_index(drop=True)
    )

    results = audit_subset.copy()

    results["target"] = predictions_df["target"].tolist()

    results["qwen_prediction"] = predictions_df["prediction"].tolist()

    results["generated"] = predictions_df["generated"].tolist()

    results["heuristic_prediction"] = results["bias_score"].apply(heuristic_prediction)

    results.to_parquet(
        RESULTS_PARQUET,
        index=False,
        compression="zstd",
    )

    # ========================================================
    # QWEN METRICS
    # ========================================================

    qwen_class = classification_metrics(
        results,
        "qwen_prediction",
    )

    qwen_trading = trading_metrics(
        results,
        "qwen_prediction",
    )

    # ========================================================
    # HEURISTIC METRICS
    # ========================================================

    heuristic_class = classification_metrics(
        results,
        "heuristic_prediction",
    )

    heuristic_trading = trading_metrics(
        results,
        "heuristic_prediction",
    )

    # ========================================================
    # TARGET "ORACLE" PERFORMANCE
    # ========================================================

    target_trading = trading_metrics(
        results,
        "target",
    )

    # ========================================================
    # AFFICHAGE
    # ========================================================

    print_metrics(
        "QWEN V1",
        qwen_class,
        qwen_trading,
    )

    print()
    print("Matrice de confusion Qwen :")

    print(
        confusion_matrix(
            results,
            "qwen_prediction",
        ).to_string()
    )

    print_metrics(
        "HEURISTIQUE",
        heuristic_class,
        heuristic_trading,
    )

    print()
    print("Matrice de confusion heuristique :")

    print(
        confusion_matrix(
            results,
            "heuristic_prediction",
        ).to_string()
    )

    print()
    print("=" * 70)
    print("TARGET DATASET - REFERENCE")
    print("=" * 70)

    print(
        "Trades :",
        target_trading["trades"],
    )

    print("Win rate résolu :", f"{target_trading['win_rate_resolved'] * 100:.2f}%")

    print("Expected R / trade :", f"{target_trading['expected_r_all_trades']:+.4f} R")

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {
        "examples": len(results),
        "qwen": {
            "classification": qwen_class,
            "trading": qwen_trading,
        },
        "heuristic": {
            "classification": heuristic_class,
            "trading": heuristic_trading,
        },
        "target_reference": {
            "trading": target_trading,
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
    print("FICHIERS")
    print("=" * 70)

    print(PREDICTIONS_FILE.resolve())

    print(RESULTS_PARQUET.resolve())

    print(METRICS_FILE.resolve())


if __name__ == "__main__":
    main()

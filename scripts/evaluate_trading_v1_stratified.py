from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import torch

# IMPORTANT : Unsloth avant transformers / peft / trl
import unsloth

from unsloth import FastLanguageModel
from transformers import StoppingCriteria, StoppingCriteriaList


# ============================================================
# CONFIGURATION
# ============================================================

ADAPTER_DIR = Path(
    "models/qwen3.5-9b-trading-v1/final_adapter"
)

TEST_JSONL = Path(
    "data/sft/test.jsonl"
)

TEST_AUDIT = Path(
    "data/sft/test_audit.parquet"
)

OUTPUT_DIR = Path(
    "data/evaluation/qwen3.5-9b-trading-v1-stratified"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR / "predictions_stratified.jsonl"
)

RESULTS_FILE = (
    OUTPUT_DIR / "results_stratified.parquet"
)

METRICS_FILE = (
    OUTPUT_DIR / "metrics_stratified.json"
)

MAX_SEQ_LENGTH = 768

MAX_NEW_TOKENS = 220

SAMPLES_PER_CLASS = 30

SEED = 3407

LABELS = [
    "NO_TRADE",
    "LONG_BIAS",
    "SHORT_BIAS",
]


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
# STOP APRES DECISION
# ============================================================

class StopAfterDecision(StoppingCriteria):

    def __init__(
        self,
        tokenizer,
    ):

        super().__init__()

        self.tokenizer = tokenizer


    def __call__(
        self,
        input_ids,
        scores,
        **kwargs,
    ):

        # On décode seulement la fin de la génération.
        tail = input_ids[
            0,
            -40:
        ]

        text = self.tokenizer.decode(
            tail,
            skip_special_tokens=True,
        )

        match = re.search(
            r"DECISION\s*[:\-]?\s*"
            r"(NO_TRADE|LONG_BIAS|SHORT_BIAS)",
            text,
            flags=re.IGNORECASE,
        )

        return match is not None


# ============================================================
# PARSE DECISION
# ============================================================

def parse_prediction(text: str):

    match = re.search(
        r"DECISION\s*[:\-]?\s*"
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


    # Fallback si le modèle produit le label
    # mais avec une petite variation de format.
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
# SELECTION STRATIFIEE
# ============================================================

def build_stratified_indices(
    audit: pd.DataFrame,
):

    selected = []

    for label in LABELS:

        group = audit[
            audit["decision"] == label
        ]

        if len(group) < SAMPLES_PER_CLASS:

            raise RuntimeError(
                f"Pas assez de {label}: "
                f"{len(group)} disponibles."
            )

        sample = group.sample(
            n=SAMPLES_PER_CLASS,
            random_state=SEED,
            replace=False,
        )

        selected.extend(
            sample.index.tolist()
        )


    # On remet dans l'ordre temporel du test.
    return sorted(
        selected
    )


# ============================================================
# GENERATION
# ============================================================

@torch.inference_mode()
def predict_one(
    model,
    tokenizer,
    record,
    stopping_criteria,
):

    # --------------------------------------------------------
    # On conserve EXACTEMENT les messages system + user
    # utilisés pendant le SFT.
    #
    # On retire uniquement la réponse assistant contenant
    # évidemment la vérité terrain.
    # --------------------------------------------------------

    messages = [
        message
        for message in record["messages"]
        if message["role"] != "assistant"
    ]


    inputs = tokenizer.apply_chat_template(
        messages,

        tokenize=True,

        add_generation_prompt=True,

        enable_thinking=False,

        return_tensors="pt",

        return_dict=True,
    )


    inputs = {
        key: value.to("cuda")
        for key, value
        in inputs.items()
    }


    prompt_length = (
        inputs["input_ids"]
        .shape[1]
    )


    output = model.generate(

        **inputs,

        max_new_tokens=MAX_NEW_TOKENS,

        do_sample=False,

        use_cache=True,

        stopping_criteria=(
            stopping_criteria
        ),

        pad_token_id=(
            tokenizer.pad_token_id
        ),

        eos_token_id=(
            tokenizer.eos_token_id
        ),
    )


    generated_ids = output[
        0,
        prompt_length:
    ]


    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


    prediction = parse_prediction(
        generated_text
    )


    return (
        prediction,
        generated_text,
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
# AFFICHAGE
# ============================================================

def print_metrics(
    name,
    metrics,
):

    print()
    print("=" * 70)
    print(name)
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

        stats = (
            metrics[
                "per_class"
            ][label]
        )

        print(
            f"{label:12} | "
            f"P={stats['precision'] * 100:6.2f}% | "
            f"R={stats['recall'] * 100:6.2f}% | "
            f"F1={stats['f1'] * 100:6.2f}% | "
            f"pred={stats['predicted']:2d}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("QWEN V1 - TEST STRATIFIE FORMAT SFT")
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
            "test.jsonl et test_audit.parquet "
            "ne sont pas alignés."
        )


    indices = build_stratified_indices(
        audit
    )


    print()
    print(
        "NO_TRADE sélectionnés :",
        sum(
            audit.loc[
                index,
                "decision"
            ] == "NO_TRADE"
            for index in indices
        )
    )

    print(
        "LONG_BIAS sélectionnés :",
        sum(
            audit.loc[
                index,
                "decision"
            ] == "LONG_BIAS"
            for index in indices
        )
    )

    print(
        "SHORT_BIAS sélectionnés :",
        sum(
            audit.loc[
                index,
                "decision"
            ] == "SHORT_BIAS"
            for index in indices
        )
    )

    print(
        "TOTAL :",
        len(indices)
    )


    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("CHARGEMENT MODELE")
    print("=" * 70)


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


    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )


    FastLanguageModel.for_inference(
        model
    )

    model.eval()


    stopping_criteria = (
        StoppingCriteriaList(
            [
                StopAfterDecision(
                    tokenizer
                )
            ]
        )
    )


    print(
        "Modèle chargé."
    )


    # ========================================================
    # INFERENCE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    results = []


    print()
    print("=" * 70)
    print("INFERENCE")
    print("=" * 70)


    for position, index in enumerate(
        indices,
        start=1,
    ):

        record = records[
            index
        ]

        row = audit.loc[
            index
        ]


        target = row[
            "decision"
        ]


        prediction, generated = predict_one(

            model=model,

            tokenizer=tokenizer,

            record=record,

            stopping_criteria=(
                stopping_criteria
            ),
        )


        heuristic = (
            heuristic_prediction(
                row["bias_score"]
            )
        )


        result = {

            "index":
                int(index),

            "timestamp":
                str(
                    row["timestamp"]
                ),

            "target":
                target,

            "qwen_prediction":
                prediction,

            "heuristic_prediction":
                heuristic,

            "bias_score":
                int(
                    row["bias_score"]
                ),

            "generated":
                generated,
        }


        results.append(
            result
        )


        print(
            f"[{position:02d}/90] "
            f"real={target:11} "
            f"qwen={prediction:11} "
            f"heur={heuristic:11}"
        )


    # ========================================================
    # DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    results_df.to_parquet(
        RESULTS_FILE,
        index=False,
        compression="zstd",
    )


    with open(
        PREDICTIONS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        for result in results:

            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )


    # ========================================================
    # METRIQUES
    # ========================================================

    qwen_metrics = (
        classification_metrics(
            results_df,
            "qwen_prediction",
        )
    )


    heuristic_metrics = (
        classification_metrics(
            results_df,
            "heuristic_prediction",
        )
    )


    print_metrics(
        "QWEN V1 - FORMAT SFT",
        qwen_metrics,
    )


    print()
    print("=" * 70)
    print("MATRICE DE CONFUSION QWEN")
    print("=" * 70)


    qwen_matrix = pd.crosstab(
        results_df["target"],
        results_df["qwen_prediction"],
        rownames=["REEL"],
        colnames=["PREDIT"],
        dropna=False,
    )


    print(
        qwen_matrix.to_string()
    )


    print_metrics(
        "HEURISTIQUE - MEMES 90 CAS",
        heuristic_metrics,
    )


    print()
    print("=" * 70)
    print("MATRICE DE CONFUSION HEURISTIQUE")
    print("=" * 70)


    heuristic_matrix = pd.crosstab(
        results_df["target"],
        results_df[
            "heuristic_prediction"
        ],
        rownames=["REEL"],
        colnames=["PREDIT"],
        dropna=False,
    )


    print(
        heuristic_matrix.to_string()
    )


    # ========================================================
    # ERREURS QWEN
    # ========================================================

    errors = results_df[
        results_df[
            "qwen_prediction"
        ]
        != results_df["target"]
    ]


    print()
    print("=" * 70)
    print("RESUME DES ERREURS QWEN")
    print("=" * 70)

    print(
        "Erreurs :",
        len(errors),
        "/",
        len(results_df)
    )


    if len(errors):

        error_summary = pd.crosstab(
            errors["target"],
            errors["qwen_prediction"],
            rownames=["REEL"],
            colnames=["PREDIT"],
        )

        print()
        print(
            error_summary.to_string()
        )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {

        "samples_per_class":
            SAMPLES_PER_CLASS,

        "total":
            len(results_df),

        "qwen":
            qwen_metrics,

        "heuristic":
            heuristic_metrics,
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
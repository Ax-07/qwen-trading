from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import torch

# IMPORTANT :
# Unsloth doit être importé AVANT trl / transformers / peft.
import unsloth

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from datasets import Dataset
from trl import SFTConfig, SFTTrainer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "Qwen/Qwen3.5-9B"

TRAIN_FILE = Path(
    "data/sft/train_balanced.jsonl"
)

VALIDATION_FILE = Path(
    "data/sft/validation.jsonl"
)

OUTPUT_DIR = Path(
    "models/qwen3.5-9b-trading-v1"
)

FINAL_ADAPTER_DIR = (
    OUTPUT_DIR
    / "final_adapter"
)

MAX_SEQ_LENGTH = 768

SEED = 3407


# ============================================================
# OUTILS
# ============================================================

def gb(value):
    return value / (1024 ** 3)


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
# CONVERSION MESSAGES -> CHAT TEMPLATE
# ============================================================

def build_text_dataset(
    path: Path,
    tokenizer,
):
    print()
    print(
        "Chargement dataset :",
        path,
    )

    records = load_jsonl(
        path
    )

    texts = []

    for record in records:

        messages = record[
            "messages"
        ]

        text = tokenizer.apply_chat_template(
            messages,

            tokenize=False,

            add_generation_prompt=False,

            # Notre modèle Trading V1
            # ne doit pas produire de chaîne
            # de raisonnement interne.
            enable_thinking=False,
        )

        texts.append(
            text
        )

    dataset = Dataset.from_dict(
        {
            "text": texts,
        }
    )

    print(
        "Exemples :",
        f"{len(dataset):,}",
    )

    return dataset


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reprendre automatiquement "
            "depuis le dernier checkpoint."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
        help=(
            "Nombre d'époques. "
            "Défaut : 1."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # GPU
    # ========================================================

    print("=" * 70)
    print("QWEN3.5-9B TRADING V1")
    print("=" * 70)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA n'est pas disponible."
        )


    print(
        "PyTorch :",
        torch.__version__,
    )

    print(
        "GPU     :",
        torch.cuda.get_device_name(0),
    )

    print(
        "VRAM    :",
        round(
            gb(
                torch.cuda
                .get_device_properties(0)
                .total_memory
            ),
            2,
        ),
        "Go",
    )

    print(
        "BF16    :",
        torch.cuda.is_bf16_supported(),
    )


    if torch.cuda.device_count() != 1:

        print()
        print(
            "ATTENTION :",
            torch.cuda.device_count(),
            "GPU sont visibles."
        )

        print(
            "Pour ce projet, seule la RTX 3060 "
            "12 Go devrait être visible."
        )


    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


    # ========================================================
    # MODELE
    # ========================================================

    print()
    print("=" * 70)
    print("CHARGEMENT MODELE")
    print("=" * 70)


    model, tokenizer_or_processor = (
        FastLanguageModel.from_pretrained(

            model_name=MODEL_ID,

            max_seq_length=MAX_SEQ_LENGTH,

            dtype=None,

            # QLoRA 4-bit
            load_in_4bit=True,

            # Pas de vision.
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


    print()
    print(
        "VRAM après chargement :",
        round(
            gb(
                torch.cuda
                .memory_allocated()
            ),
            2,
        ),
        "Go",
    )


    # ========================================================
    # LORA
    # ========================================================

    print()
    print("=" * 70)
    print("CONFIGURATION QLORA")
    print("=" * 70)


    model = FastLanguageModel.get_peft_model(

        model,

        # Validé sur notre 3060.
        r=8,

        target_modules="all-linear",

        lora_alpha=16,

        # Chemin optimisé Unsloth.
        lora_dropout=0,

        bias="none",

        use_gradient_checkpointing=(
            "unsloth"
        ),

        random_state=SEED,

        use_rslora=False,

        loftq_config=None,
    )


    # ========================================================
    # PARAMETRES ENTRAINABLES
    # ========================================================

    trainable = 0
    total = 0

    for parameter in model.parameters():

        total += parameter.numel()

        if parameter.requires_grad:

            trainable += (
                parameter.numel()
            )


    print(
        f"Trainable : "
        f"{trainable:,}"
    )

    print(
        f"Total     : "
        f"{total:,}"
    )

    print(
        f"Pourcentage : "
        f"{100 * trainable / total:.4f}%"
    )


    # ========================================================
    # DATASETS
    # ========================================================

    print()
    print("=" * 70)
    print("DATASETS")
    print("=" * 70)


    train_dataset = (
        build_text_dataset(
            TRAIN_FILE,
            tokenizer,
        )
    )


    validation_dataset = (
        build_text_dataset(
            VALIDATION_FILE,
            tokenizer,
        )
    )


    # ========================================================
    # SFT CONFIG
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    config_kwargs = {

        "output_dir":
            str(OUTPUT_DIR),

        # ----------------------------------------------------
        # BATCH
        # ----------------------------------------------------

        "per_device_train_batch_size":
            1,

        "per_device_eval_batch_size":
            1,

        "gradient_accumulation_steps":
            2,

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        "num_train_epochs":
            args.epochs,

        "learning_rate":
            1e-4,

        "warmup_ratio":
            0.03,

        "lr_scheduler_type":
            "cosine",

        "weight_decay":
            0.01,

        "max_grad_norm":
            1.0,

        # ----------------------------------------------------
        # PRECISION
        # ----------------------------------------------------

        "bf16":
            torch.cuda.is_bf16_supported(),

        "fp16":
            not torch.cuda.is_bf16_supported(),

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        "optim":
            "adamw_8bit",

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        "dataset_text_field":
            "text",

        # IMPORTANT pour Qwen3.5.
        "packing":
            False,

        # Windows : éviter multiprocessing
        # inutile / fragile.
        "dataset_num_proc":
            1,

        "dataloader_num_workers":
            0,

        # ----------------------------------------------------
        # LOGS
        # ----------------------------------------------------

        "logging_steps":
            10,

        "logging_first_step":
            True,

        "report_to":
            "none",

        # ----------------------------------------------------
        # CHECKPOINTS
        # ----------------------------------------------------

        "save_strategy":
            "steps",

        "save_steps":
            250,

        "save_total_limit":
            2,

        # L'évaluation générationnelle
        # sera faite avec notre propre script.
        "eval_strategy":
            "no",

        # ----------------------------------------------------
        # DIVERS
        # ----------------------------------------------------

        "seed":
            SEED,
    }


    # Compatibilité selon la version TRL.
    sft_signature = inspect.signature(
        SFTConfig.__init__
    )


    if (
        "max_length"
        in sft_signature.parameters
    ):

        config_kwargs[
            "max_length"
        ] = MAX_SEQ_LENGTH


    elif (
        "max_seq_length"
        in sft_signature.parameters
    ):

        config_kwargs[
            "max_seq_length"
        ] = MAX_SEQ_LENGTH


    training_args = SFTConfig(
        **config_kwargs
    )


    # ========================================================
    # TRAINER
    # ========================================================

    trainer_kwargs = {

        "model":
            model,

        "train_dataset":
            train_dataset,

        "eval_dataset":
            validation_dataset,

        "args":
            training_args,
    }


    trainer_signature = inspect.signature(
        SFTTrainer.__init__
    )


    if (
        "processing_class"
        in trainer_signature.parameters
    ):

        trainer_kwargs[
            "processing_class"
        ] = tokenizer


    elif (
        "tokenizer"
        in trainer_signature.parameters
    ):

        trainer_kwargs[
            "tokenizer"
        ] = tokenizer


    trainer = SFTTrainer(
        **trainer_kwargs
    )


    # ========================================================
    # ASSISTANT-ONLY LOSS
    # ========================================================
    #
    # Qwen3.5 utilise ChatML :
    #
    # <|im_start|>user
    # ...
    # <|im_start|>assistant
    # ...
    #
    # La loss doit être calculée uniquement
    # sur la partie assistant.
    # ========================================================

    print()
    print("=" * 70)
    print("ASSISTANT-ONLY LOSS")
    print("=" * 70)


    trainer = train_on_responses_only(

        trainer,

        instruction_part=(
            "<|im_start|>user\n"
        ),

        response_part=(
            "<|im_start|>assistant\n"
        ),

        num_proc=1,
    )


    # ========================================================
    # SANITY CHECK DU MASKING
    # ========================================================
    #
    # Avant plusieurs milliers de steps,
    # on vérifie qu'il reste réellement
    # des tokens supervisés.
    # ========================================================

    print()
    print("=" * 70)
    print("VERIFICATION DU MASKING")
    print("=" * 70)


    first_batch = next(
        iter(
            trainer.get_train_dataloader()
        )
    )


    labels = first_batch[
        "labels"
    ]


    total_tokens = labels.numel()

    supervised_tokens = (
        labels != -100
    ).sum().item()


    supervised_ratio = (
        supervised_tokens
        / total_tokens
        * 100
    )


    print(
        "Tokens batch :",
        f"{total_tokens:,}",
    )

    print(
        "Tokens supervisés :",
        f"{supervised_tokens:,}",
    )

    print(
        "Ratio supervisé :",
        f"{supervised_ratio:.2f}%",
    )


    if supervised_tokens == 0:

        raise RuntimeError(
            "ERREUR : aucun token assistant "
            "n'est supervisé. "
            "Ne pas lancer l'entraînement."
        )


    if supervised_ratio > 60:

        print()
        print(
            "ATTENTION : le ratio de tokens "
            "supervisés semble élevé."
        )

        print(
            "Vérifier le masking avant "
            "un entraînement long."
        )


    # ========================================================
    # VRAM AVANT TRAINING
    # ========================================================

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


    print()
    print(
        "VRAM avant training :",
        round(
            gb(
                torch.cuda
                .memory_allocated()
            ),
            2,
        ),
        "Go",
    )


    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 70)
    print("DEBUT ENTRAINEMENT TRADING V1")
    print("=" * 70)


    if args.resume:

        print(
            "Reprise depuis le dernier "
            "checkpoint disponible."
        )

        result = trainer.train(
            resume_from_checkpoint=True
        )

    else:

        result = trainer.train()


    # ========================================================
    # METRIQUES
    # ========================================================

    metrics = result.metrics


    print()
    print("=" * 70)
    print("ENTRAINEMENT TERMINE")
    print("=" * 70)


    for key, value in metrics.items():

        print(
            f"{key}: {value}"
        )


    print()
    print(
        "VRAM allouée :",
        round(
            gb(
                torch.cuda
                .memory_allocated()
            ),
            2,
        ),
        "Go",
    )

    print(
        "VRAM réservée :",
        round(
            gb(
                torch.cuda
                .memory_reserved()
            ),
            2,
        ),
        "Go",
    )

    print(
        "PIC VRAM :",
        round(
            gb(
                torch.cuda
                .max_memory_allocated()
            ),
            2,
        ),
        "Go",
    )


    # ========================================================
    # SAUVEGARDE FINALE
    # ========================================================

    print()
    print("=" * 70)
    print("SAUVEGARDE ADAPTER")
    print("=" * 70)


    FINAL_ADAPTER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    model.save_pretrained(
        FINAL_ADAPTER_DIR
    )


    tokenizer.save_pretrained(
        FINAL_ADAPTER_DIR
    )


    # Sauvegarde métriques.
    metrics_path = (
        OUTPUT_DIR
        / "training_metrics.json"
    )


    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


    print(
        "Adapter :",
        FINAL_ADAPTER_DIR.resolve(),
    )

    print(
        "Metrics :",
        metrics_path.resolve(),
    )


if __name__ == "__main__":
    main()
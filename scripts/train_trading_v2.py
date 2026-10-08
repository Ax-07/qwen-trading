from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch

# IMPORTANT : Unsloth avant TRL / Transformers / PEFT
import unsloth

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from datasets import Dataset
from trl import SFTConfig, SFTTrainer


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "Qwen/Qwen3.5-9B"

TRAIN_FILE = Path(
    "data/sft_v2/train.jsonl"
)

VALIDATION_FILE = Path(
    "data/sft_v2/validation.jsonl"
)

OUTPUT_DIR = Path(
    "models/qwen3.5-9b-trading-v2"
)

FINAL_ADAPTER_DIR = (
    OUTPUT_DIR / "final_adapter"
)

MAX_SEQ_LENGTH = 768

SEED = 3407


# ============================================================
# UTILS
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
# DATASET -> TEXT CHATML
# ============================================================

def build_text_dataset(
    path: Path,
    tokenizer,
):

    print()
    print(
        "Chargement :",
        path,
    )

    records = load_jsonl(
        path
    )

    texts = []

    for record in records:

        text = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=False,
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
        "--epochs",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print("=" * 70)
    print("QWEN3.5-9B TRADING V2")
    print("=" * 70)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA indisponible."
        )


    print(
        "PyTorch :",
        torch.__version__,
    )

    print(
        "GPU :",
        torch.cuda.get_device_name(0),
    )

    print(
        "GPU visibles :",
        torch.cuda.device_count(),
    )

    print(
        "VRAM :",
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
        "BF16 :",
        torch.cuda.is_bf16_supported(),
    )


    if torch.cuda.device_count() != 1:

        raise RuntimeError(
            "Plus d'un GPU est visible. "
            "Isole la RTX 3060 avant le run."
        )


    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


    # ========================================================
    # MODEL
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


    print(
        "VRAM après load :",
        round(
            gb(
                torch.cuda.memory_allocated()
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
    print("QLORA")
    print("=" * 70)


    model = FastLanguageModel.get_peft_model(

        model,

        r=8,

        target_modules="all-linear",

        lora_alpha=16,

        lora_dropout=0,

        bias="none",

        use_gradient_checkpointing="unsloth",

        random_state=SEED,

        use_rslora=False,

        loftq_config=None,
    )


    trainable = 0
    total = 0

    for parameter in model.parameters():

        total += parameter.numel()

        if parameter.requires_grad:

            trainable += parameter.numel()


    print(
        "Trainable :",
        f"{trainable:,}",
    )

    print(
        "Total :",
        f"{total:,}",
    )

    print(
        "Ratio :",
        f"{100 * trainable / total:.4f}%",
    )


    # ========================================================
    # DATASETS
    # ========================================================

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
    # CONFIG TRAINING
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    config_kwargs = {

        "output_dir":
            str(OUTPUT_DIR),

        "per_device_train_batch_size":
            1,

        "per_device_eval_batch_size":
            1,

        "gradient_accumulation_steps":
            2,

        "num_train_epochs":
            args.epochs,

        # V2 plus prudente.
        "learning_rate":
            5e-5,

        # ~30 steps pour 1 époque.
        "warmup_steps":
            30,

        "lr_scheduler_type":
            "cosine",

        "weight_decay":
            0.01,

        "max_grad_norm":
            1.0,

        "bf16":
            torch.cuda.is_bf16_supported(),

        "fp16":
            not torch.cuda.is_bf16_supported(),

        "optim":
            "adamw_8bit",

        "dataset_text_field":
            "text",

        "packing":
            False,

        "dataset_num_proc":
            1,

        "dataloader_num_workers":
            0,

        "logging_steps":
            10,

        "logging_first_step":
            True,

        "report_to":
            "none",

        "save_strategy":
            "steps",

        "save_steps":
            200,

        "save_total_limit":
            2,

        # On benchmarkera ensuite nous-mêmes.
        "eval_strategy":
            "no",

        "seed":
            SEED,
    }


    signature = inspect.signature(
        SFTConfig.__init__
    )


    if (
        "max_length"
        in signature.parameters
    ):

        config_kwargs[
            "max_length"
        ] = MAX_SEQ_LENGTH


    elif (
        "max_seq_length"
        in signature.parameters
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
    # ASSISTANT ONLY LOSS
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
    # MASK CHECK
    # ========================================================

    batch = next(
        iter(
            trainer.get_train_dataloader()
        )
    )


    labels = batch[
        "labels"
    ]


    total_tokens = (
        labels.numel()
    )


    supervised_tokens = (
        labels != -100
    ).sum().item()


    ratio = (
        supervised_tokens
        / total_tokens
        * 100
    )


    print(
        "Tokens batch :",
        total_tokens,
    )

    print(
        "Tokens supervisés :",
        supervised_tokens,
    )

    print(
        "Ratio supervisé :",
        f"{ratio:.2f}%",
    )


    if supervised_tokens == 0:

        raise RuntimeError(
            "Aucun token assistant supervisé."
        )


    # V2 a une réponse très courte.
    # Un ratio faible est donc NORMAL.
    if supervised_tokens > 30:

        print(
            "ATTENTION : beaucoup de tokens "
            "assistant pour une cible decision-only."
        )


    # ========================================================
    # TRAIN
    # ========================================================

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


    print()
    print("=" * 70)
    print("DEBUT TRAINING V2")
    print("=" * 70)


    if args.resume:

        result = trainer.train(
            resume_from_checkpoint=True
        )

    else:

        result = trainer.train()


    # ========================================================
    # RESULTATS
    # ========================================================

    metrics = result.metrics


    print()
    print("=" * 70)
    print("TRAINING TERMINE")
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
                torch.cuda.memory_allocated()
            ),
            2,
        ),
        "Go",
    )

    print(
        "VRAM réservée :",
        round(
            gb(
                torch.cuda.memory_reserved()
            ),
            2,
        ),
        "Go",
    )

    print(
        "PIC VRAM :",
        round(
            gb(
                torch.cuda.max_memory_allocated()
            ),
            2,
        ),
        "Go",
    )


    # ========================================================
    # SAVE
    # ========================================================

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


    with open(
        OUTPUT_DIR
        / "training_metrics.json",
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


    print()
    print(
        "Adapter :",
        FINAL_ADAPTER_DIR.resolve()
    )


if __name__ == "__main__":
    main()
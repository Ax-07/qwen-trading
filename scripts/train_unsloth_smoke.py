import os
import inspect
import torch

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


# ============================================================
# CONFIGURATION GENERALE
# ============================================================

MODEL_ID = "Qwen/Qwen3.5-9B"

OUTPUT_DIR = "models/qwen3.5-9b-unsloth-smoke"

# 384 est un bon compromis pour notre futur dataset trading.
MAX_SEQ_LENGTH = 384


def gb(value):
    """Convertit des octets en Go."""
    return value / (1024 ** 3)


# ============================================================
# GPU
# ============================================================

print("=" * 70)
print("GPU / ENVIRONNEMENT")
print("=" * 70)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA n'est pas disponible.")

print("PyTorch :", torch.__version__)
print("GPU     :", torch.cuda.get_device_name(0))
print(
    "VRAM    :",
    round(
        gb(torch.cuda.get_device_properties(0).total_memory),
        2,
    ),
    "Go",
)

print(
    "BF16    :",
    torch.cuda.is_bf16_supported(),
)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# ============================================================
# CHARGEMENT DE QWEN
# ============================================================

print()
print("=" * 70)
print("CHARGEMENT QWEN3.5-9B AVEC UNSLOTH")
print("=" * 70)


model, processor_or_tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,

    max_seq_length=MAX_SEQ_LENGTH,

    # Unsloth choisit automatiquement BF16/FP16.
    dtype=None,

    # QLoRA / modèle quantifié.
    load_in_4bit=True,

    # Très important :
    # notre V1 trading est uniquement texte.
    # On ne charge donc pas la partie vision de Qwen3.5.
    text_only=True,
)


# Selon la version d'Unsloth/Qwen,
# le second objet peut être un tokenizer ou un processor.
if hasattr(processor_or_tokenizer, "tokenizer"):
    tokenizer = processor_or_tokenizer.tokenizer
else:
    tokenizer = processor_or_tokenizer


print()
print(
    "VRAM après chargement :",
    round(
        gb(torch.cuda.memory_allocated()),
        2,
    ),
    "Go",
)


# ============================================================
# AJOUT DU LORA
# ============================================================

print()
print("=" * 70)
print("AJOUT DU LORA")
print("=" * 70)


model = FastLanguageModel.get_peft_model(
    model,

    # Bon compromis qualité / mémoire.
    r=8,

    # QLoRA complet sur les couches linéaires.
    target_modules="all-linear",

    lora_alpha=16,

    # 0 permet à Unsloth d'utiliser son chemin optimisé.
    lora_dropout=0,

    bias="none",

    # Gradient checkpointing optimisé Unsloth.
    use_gradient_checkpointing="unsloth",

    random_state=3407,

    use_rslora=False,

    loftq_config=None,
)


# ============================================================
# PARAMETRES ENTRAINABLES
# ============================================================

trainable = 0
total = 0

for _, parameter in model.named_parameters():

    total += parameter.numel()

    if parameter.requires_grad:
        trainable += parameter.numel()


print()
print("Paramètres entraînables :")

print(
    f"Trainable : {trainable:,}"
)

print(
    f"Total     : {total:,}"
)

print(
    f"Pourcentage : "
    f"{100 * trainable / total:.4f} %"
)


# ============================================================
# MINI DATASET
# ============================================================

system_prompt = (
    "Tu es un analyste de trading discipliné. "
    "Tu privilégies la gestion du risque, "
    "les scénarios conditionnels et NO TRADE "
    "lorsqu'aucun avantage clair n'existe. "
    "Tu ne dois jamais inventer un setup "
    "uniquement pour proposer un trade."
)


examples = [

    # --------------------------------------------------------
    # EXEMPLE 1
    # --------------------------------------------------------

    {
        "user": """
BTCUSDT - 1H

Prix : 68000
EMA20 : 67400
EMA50 : 66500
RSI14 : 63

Résistance : 68200
Support : 67100

Structure :
HH - HL - HH

Analyse la situation.
""",

        "assistant": """
BIAS: haussier

CONTEXTE:
La structure reste haussière et le prix évolue au-dessus des EMA20 et EMA50.

SCENARIO:
Ne pas acheter directement sous la résistance 68200.

Attendre une clôture confirmée au-dessus de 68200 avec volume.

INVALIDATION:
Retour sous 67100.

DECISION:
Attendre.

Pas de trade sans confirmation.
""",
    },


    # --------------------------------------------------------
    # EXEMPLE 2
    # --------------------------------------------------------

    {
        "user": """
ETHUSDT - 4H

Prix : 3500
EMA20 : 3570
EMA50 : 3630
RSI14 : 42

Support : 3450
Résistance : 3600

Structure :
LH - LL - LH

Analyse la situation.
""",

        "assistant": """
BIAS: baissier

CONTEXTE:
Le prix reste sous EMA20 et EMA50.

La structure forme des sommets et creux descendants.

SCENARIO:
Un rejet confirmé de la zone 3570-3600 peut favoriser une continuation baissière.

INVALIDATION:
Reprise nette au-dessus de 3630.

DECISION:
Attendre le rejet.

Ne pas poursuivre le prix.
""",
    },


    # --------------------------------------------------------
    # EXEMPLE 3
    # --------------------------------------------------------

    {
        "user": """
SOLUSDT - 1H

Prix : 150
EMA20 : 149
EMA50 : 148
RSI14 : 51

Support : 146
Résistance : 153

Volume : faible

Structure :
range

Analyse la situation.
""",

        "assistant": """
BIAS: neutre

CONTEXTE:
Le marché évolue en range avec peu de momentum.

Le volume ne confirme aucune direction.

DECISION:
NO TRADE.

PLAN:
Attendre soit une cassure confirmée de 153,
soit une perte confirmée de 146.

Éviter une entrée au milieu du range.
""",
    },


    # --------------------------------------------------------
    # EXEMPLE 4
    # --------------------------------------------------------

    {
        "user": """
BTCUSDT - 15m

Le prix vient de monter de 3.5% en quatre bougies.

RSI14 : 78

Le prix est fortement éloigné de EMA20.

Une résistance importante se trouve seulement 0.4% plus haut.

Faut-il acheter immédiatement ?
""",

        "assistant": """
DECISION:
Non.

CONTEXTE:
Le mouvement est déjà fortement étendu et le prix approche une résistance.

RISQUE:
Une entrée tardive présente un mauvais ratio rendement/risque.

Le risque de pullback est important.

PLAN:
Attendre une consolidation ou un retracement avant de rechercher un nouveau setup.
""",
    },


    # --------------------------------------------------------
    # EXEMPLE 5
    # --------------------------------------------------------

    {
        "user": """
BTCUSDT - 4H

Structure haussière.

Cassure de résistance confirmée.

Volume supérieur de 45% à la moyenne.

Ancienne résistance retestée comme support.

RSI14 : 59.

Analyse.
""",

        "assistant": """
BIAS: haussier

CONTEXTE:
La cassure est soutenue par le volume.

L'ancienne résistance semble agir comme support lors du retest.

SCENARIO LONG:
Une réaction haussière confirmée sur le retest peut constituer un setup de continuation.

INVALIDATION:
Clôture nette sous la zone de retest.

RISK:
Définir l'invalidation avant l'entrée.

Refuser le trade si le ratio rendement/risque est insuffisant.
""",
    },


    # --------------------------------------------------------
    # EXEMPLE 6
    # --------------------------------------------------------

    {
        "user": """
Le marché est extrêmement volatil après une annonce macro.

Les bougies sont trois fois plus grandes que leur taille habituelle.

Le spread augmente fortement.

Que faire ?
""",

        "assistant": """
BIAS: indéterminé

RISQUE:
La volatilité anormale augmente le slippage,
la taille nécessaire du stop
et le risque de faux signaux.

DECISION:
NO TRADE.

PLAN:
Attendre une normalisation de la volatilité
et l'apparition d'une nouvelle structure exploitable.
""",
    },


    # --------------------------------------------------------
    # EXEMPLE 7
    # --------------------------------------------------------

    {
        "user": """
ETHUSDT - 1H

EMA20 au-dessus EMA50.

Prix au-dessus des deux moyennes.

Volume en diminution.

Open interest en forte hausse.

Funding très positif.

Analyse.
""",

        "assistant": """
BIAS:
Haussier mais prudent.

CONTEXTE:
La tendance reste positive.

Cependant, la hausse de l'open interest associée à un funding très positif
indique un positionnement long potentiellement encombré.

RISQUE:
Un mouvement de liquidation des positions longues reste possible.

DECISION:
Ne pas poursuivre le prix.

Attendre un retracement ou une nouvelle confirmation.
""",
    },


    # --------------------------------------------------------
    # EXEMPLE 8
    # --------------------------------------------------------

    {
        "user": """
Un setup offre :

Entrée : 100
Stop : 98
Objectif : 101.5

Est-ce un bon trade si la probabilité semble correcte ?
""",

        "assistant": """
RISQUE:
La perte potentielle est de 2 unités.

Le gain potentiel est seulement de 1.5 unité.

RATIO:
Le ratio rendement/risque est inférieur à 1.

DECISION:
Refuser le trade sauf justification statistique très solide provenant d'un backtest fiable.

Une impression de forte probabilité ne suffit pas.
""",
    },
]


# ============================================================
# FORMAT CHAT QWEN
# ============================================================

texts = []


for example in examples:

    messages = [

        {
            "role": "system",
            "content": system_prompt,
        },

        {
            "role": "user",
            "content": example["user"].strip(),
        },

        {
            "role": "assistant",
            "content": example["assistant"].strip(),
        },

    ]


    text = tokenizer.apply_chat_template(
        messages,

        tokenize=False,

        add_generation_prompt=False,

        # On n'entraîne pas le mode thinking.
        enable_thinking=False,
    )


    texts.append(text)


dataset = Dataset.from_dict(
    {
        "text": texts,
    }
)


print()
print(
    "Dataset :",
    len(dataset),
    "exemples",
)


# ============================================================
# CONFIGURATION SFT
# ============================================================

config_kwargs = {

    "output_dir": OUTPUT_DIR,

    # --------------------------------------------------------
    # BATCH
    # --------------------------------------------------------

    "per_device_train_batch_size": 1,

    # Batch effectif = 2
    "gradient_accumulation_steps": 2,

    # --------------------------------------------------------
    # SMOKE TEST
    # --------------------------------------------------------

    # Seulement trois optimisations.
    "max_steps": 3,

    # --------------------------------------------------------
    # LEARNING RATE
    # --------------------------------------------------------

    "learning_rate": 1e-4,

    "warmup_steps": 0,

    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    "bf16": torch.cuda.is_bf16_supported(),

    "fp16": not torch.cuda.is_bf16_supported(),

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    "logging_steps": 1,

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    "optim": "adamw_8bit",

    "weight_decay": 0.01,

    "lr_scheduler_type": "linear",

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    "dataset_text_field": "text",

    "packing": False,

    # --------------------------------------------------------
    # AUTRES
    # --------------------------------------------------------

    "seed": 3407,

    "report_to": "none",

    # Pour le smoke-test on ne veut pas créer
    # de checkpoints intermédiaires.
    "save_strategy": "no",
}


# ============================================================
# COMPATIBILITE TRL
# ============================================================

# Certaines versions de TRL utilisent max_length,
# d'autres max_seq_length.

sft_signature = inspect.signature(
    SFTConfig.__init__
)


if "max_length" in sft_signature.parameters:

    config_kwargs["max_length"] = MAX_SEQ_LENGTH


elif "max_seq_length" in sft_signature.parameters:

    config_kwargs["max_seq_length"] = MAX_SEQ_LENGTH


training_args = SFTConfig(
    **config_kwargs
)


# ============================================================
# TRAINER
# ============================================================

trainer_kwargs = {

    "model": model,

    "train_dataset": dataset,

    "args": training_args,
}


trainer_signature = inspect.signature(
    SFTTrainer.__init__
)


if "processing_class" in trainer_signature.parameters:

    trainer_kwargs["processing_class"] = tokenizer


elif "tokenizer" in trainer_signature.parameters:

    trainer_kwargs["tokenizer"] = tokenizer


trainer = SFTTrainer(
    **trainer_kwargs
)


# ============================================================
# ENTRAINEMENT
# ============================================================

torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()


print()
print("=" * 70)
print("DEBUT ENTRAINEMENT UNSLOTH")
print("=" * 70)


trainer.train()


# ============================================================
# RESULTATS MEMOIRE
# ============================================================

print()
print("=" * 70)
print("RESULTATS")
print("=" * 70)


print(
    "VRAM allouée :",
    round(
        gb(torch.cuda.memory_allocated()),
        2,
    ),
    "Go",
)


print(
    "VRAM réservée :",
    round(
        gb(torch.cuda.memory_reserved()),
        2,
    ),
    "Go",
)


print(
    "PIC VRAM entraînement :",
    round(
        gb(torch.cuda.max_memory_allocated()),
        2,
    ),
    "Go",
)


# ============================================================
# SAUVEGARDE DU LORA
# ============================================================

print()
print("=" * 70)
print("SAUVEGARDE")
print("=" * 70)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


model.save_pretrained(
    OUTPUT_DIR
)


tokenizer.save_pretrained(
    OUTPUT_DIR
)


print()
print("LoRA sauvegardé dans :")

print(
    OUTPUT_DIR
)
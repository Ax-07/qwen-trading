import os
import torch

from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3_5ForCausalLM,
)
from trl import SFTConfig, SFTTrainer


MODEL_ID = "Qwen/Qwen3.5-9B"
OUTPUT_DIR = "models/qwen3.5-9b-trading-smoke"


def gb(value):
    return value / 1024**3


print("=" * 70)
print("GPU")
print("=" * 70)

print("GPU :", torch.cuda.get_device_name(0))
print(
    "VRAM totale :",
    round(gb(torch.cuda.get_device_properties(0).total_memory), 2),
    "Go",
)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

print("dtype :", compute_dtype)


# ---------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------

print("\nChargement tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------
# MINI DATASET DE TEST
# ---------------------------------------------------------

examples = [
    {
        "user": """
BTCUSDT - 1H
Prix : 68000
EMA20 : 67400
EMA50 : 66500
RSI14 : 63
Résistance : 68200
Support : 67100
Structure : HH - HL - HH

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
Attendre. Pas de trade sans confirmation.
""",
    },

    {
        "user": """
ETHUSDT - 4H
Prix : 3500
EMA20 : 3570
EMA50 : 3630
RSI14 : 42
Support : 3450
Résistance : 3600
Structure : LH - LL - LH

Analyse la situation.
""",
        "assistant": """
BIAS: baissier

CONTEXTE:
Le prix reste sous les EMA20 et EMA50 et la structure forme des sommets et creux descendants.

SCENARIO:
Un rejet confirmé de la zone 3570-3600 peut favoriser une continuation baissière.

INVALIDATION:
Reprise nette au-dessus de 3630.

DECISION:
Privilégier l'attente d'un rejet confirmé. Ne pas poursuivre le prix.
""",
    },

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
Structure : range

Analyse la situation.
""",
        "assistant": """
BIAS: neutre

CONTEXTE:
Le marché évolue en range, le momentum est faible et le volume ne confirme aucune direction.

DECISION:
NO TRADE.

Attendre soit une cassure confirmée de 153, soit une perte confirmée de 146.
Éviter une entrée au milieu du range.
""",
    },

    {
        "user": """
BTCUSDT - 15m
Le prix vient de monter de 3.5% en quatre bougies.
RSI14 : 78
Prix fortement éloigné de EMA20.
Résistance située seulement 0.4% plus haut.

Faut-il acheter immédiatement ?
""",
        "assistant": """
DECISION:
Non.

CONTEXTE:
Le mouvement est déjà fortement étendu et le ratio rendement/risque devient défavorable à proximité de la résistance.

RISQUE:
Une entrée tardive augmente le risque de pullback.

PLAN:
Attendre une consolidation ou un retracement avant de rechercher un nouveau setup.
""",
    },

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
La cassure est soutenue par le volume et l'ancienne résistance semble agir comme support lors du retest.

SCENARIO LONG:
Une réaction haussière confirmée sur le retest peut constituer un setup de continuation.

INVALIDATION:
Clôture nette sous la zone de retest.

RISK:
Définir le stop avant l'entrée et refuser le trade si le ratio rendement/risque est insuffisant.
""",
    },

    {
        "user": """
ETHUSDT - 1H
EMA20 au-dessus EMA50.
Prix au-dessus des deux moyennes.
Mais volume en diminution et open interest en forte hausse.
Funding très positif.

Analyse.
""",
        "assistant": """
BIAS: haussier mais prudent

CONTEXTE:
La tendance reste positive, mais la hausse de l'open interest associée à un funding très positif indique un positionnement long potentiellement encombré.

RISQUE:
Un mouvement de liquidation des longs reste possible.

DECISION:
Ne pas poursuivre le prix. Attendre une confirmation ou un retracement offrant un meilleur niveau d'invalidation.
""",
    },

    {
        "user": """
Un setup offre :
entrée 100
stop 98
objectif 101.5

Est-ce un bon trade si la probabilité semble correcte ?
""",
        "assistant": """
RISQUE:
La perte potentielle est de 2 unités pour un gain potentiel de 1.5 unité.

RATIO:
Le ratio rendement/risque est inférieur à 1.

DECISION:
Refuser le trade sauf justification statistique très solide provenant d'un backtest fiable.
Une apparence de probabilité élevée ne suffit pas.
""",
    },

    {
        "user": """
Le marché est extrêmement volatil après une annonce macro.
Les bougies sont trois fois plus grandes que leur taille habituelle.
Le spread augmente.

Que faire ?
""",
        "assistant": """
BIAS: indéterminé

RISQUE:
La volatilité anormale augmente le slippage, la taille nécessaire du stop et le risque de faux signaux.

DECISION:
NO TRADE tant que les conditions ne se normalisent pas.

PLAN:
Attendre une stabilisation de la volatilité et une nouvelle structure exploitable.
""",
    },
]


system_prompt = (
    "Tu es un analyste de trading discipliné. "
    "Tu privilégies la gestion du risque, les scénarios conditionnels "
    "et l'absence de trade lorsqu'il n'existe pas d'avantage clair. "
    "Tu ne dois jamais inventer un setup uniquement pour proposer un trade."
)


texts = []

for ex in examples:
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": ex["user"].strip(),
        },
        {
            "role": "assistant",
            "content": ex["assistant"].strip(),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    texts.append(text)


dataset = Dataset.from_dict(
    {
        "text": texts,
    }
)

print(f"\nDataset : {len(dataset)} exemples")


# ---------------------------------------------------------
# QUANTIFICATION QLORA
# ---------------------------------------------------------

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)


print("\nChargement Qwen3.5-9B 4-bit...")

model = Qwen3_5ForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map={"": 0},
    dtype=compute_dtype,
    low_cpu_mem_usage=True,
)


model.config.use_cache = False


print(
    "VRAM après chargement :",
    round(gb(torch.cuda.memory_allocated()), 2),
    "Go",
)


# ---------------------------------------------------------
# PREPARATION K-BIT
# ---------------------------------------------------------

print("\nPréparation QLoRA...")

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)


lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)


model = get_peft_model(
    model,
    lora_config,
)


print("\nParamètres entraînables :")
model.print_trainable_parameters()


# ---------------------------------------------------------
# CONFIGURATION TRAINING
# ---------------------------------------------------------

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,

    max_steps=3,
    learning_rate=1e-4,

    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False,
    },

    max_length=192,
    packing=False,

    dataset_text_field="text",

    logging_steps=1,
    logging_first_step=True,

    save_strategy="no",
    eval_strategy="no",
    report_to="none",

    # important : optimizer 8-bit
    optim="adamw_8bit",

    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)


print("\n" + "=" * 70)
print("DEBUT DU QLORA")
print("=" * 70)

trainer.train()


# ---------------------------------------------------------
# RESULTATS VRAM
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ENTRAINEMENT TERMINE")
print("=" * 70)

print(
    "VRAM allouée :",
    round(gb(torch.cuda.memory_allocated()), 2),
    "Go",
)

print(
    "VRAM réservée :",
    round(gb(torch.cuda.memory_reserved()), 2),
    "Go",
)

print(
    "PIC VRAM :",
    round(gb(torch.cuda.max_memory_allocated()), 2),
    "Go",
)


# ---------------------------------------------------------
# SAUVEGARDE DU LORA
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nLoRA sauvegardé dans :")
print(OUTPUT_DIR)
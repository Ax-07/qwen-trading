import torch

from transformers import (
    AutoTokenizer,
    Qwen3_5ForCausalLM,
    BitsAndBytesConfig,
)

MODEL_ID = "Qwen/Qwen3.5-9B"


def gb(value):
    return value / (1024 ** 3)


print("=" * 60)
print("CONFIGURATION")
print("=" * 60)

print("GPU :", torch.cuda.get_device_name(0))
print(
    "VRAM :",
    round(gb(torch.cuda.get_device_properties(0).total_memory), 2),
    "Go",
)

compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

print("dtype :", compute_dtype)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)


print("\nChargement tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)


print("Chargement Qwen3.5-9B TEXT-ONLY...")

model = Qwen3_5ForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map={"": 0},
    dtype=compute_dtype,
    low_cpu_mem_usage=True,
)

model.eval()


print("\nMODELE CHARGE")

print(
    "VRAM allouee :",
    round(gb(torch.cuda.memory_allocated()), 2),
    "Go",
)

print(
    "VRAM reservee :",
    round(gb(torch.cuda.memory_reserved()), 2),
    "Go",
)

print(
    "Pic VRAM :",
    round(gb(torch.cuda.max_memory_allocated()), 2),
    "Go",
)


messages = [
    {
        "role": "system",
        "content": (
            "Tu es un analyste de marchés financiers. "
            "Réponds de manière concise, structurée et prudente."
        ),
    },
    {
        "role": "user",
        "content": (
            "BTC est en tendance haussière en 1H. "
            "Le prix est au-dessus des EMA 20 et 50, "
            "mais arrive sur une résistance importante. "
            "Pourquoi attendre une confirmation avant "
            "d'entrer en position ?"
        ),
    },
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,

    # Important :
    enable_thinking=False,
)


inputs = tokenizer(
    text,
    return_tensors="pt",
).to("cuda")


input_length = inputs["input_ids"].shape[-1]


print("\nGeneration...")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,

        # Réglages non-thinking
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )


generated = output[0][input_length:]

response = tokenizer.decode(
    generated,
    skip_special_tokens=True,
)


print("\n" + "=" * 60)
print("REPONSE")
print("=" * 60)

print(response)


print("\n" + "=" * 60)
print("MEMOIRE")
print("=" * 60)

print(
    "VRAM allouee :",
    round(gb(torch.cuda.memory_allocated()), 2),
    "Go",
)

print(
    "VRAM reservee :",
    round(gb(torch.cuda.memory_reserved()), 2),
    "Go",
)

print(
    "Pic VRAM :",
    round(gb(torch.cuda.max_memory_allocated()), 2),
    "Go",
)
"""V4.6.4 — Inspect exact Qwen JSON output on one causal observation.
Uses V4.6.3.3 unchanged for offline cache, RTX 3060 placement and NF4 loading.
No trading, files written, LoRA, model downloads, or Test split.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import qwen_local_inference_v4633 as base

VARIANTS = ("baseline", "explicit", "repair_prompt")
EXTRA_RULES = (
    'Output a single minified JSON object, no code fences, no commentary, no '
    'reasoning tags, no trailing text. For a flat account the safest valid '
    'reply is exactly {"action":"WAIT"}. Never add explanation, confidence, '
    'symbol, timestamp or a reason key. The only permitted keys are '
    'action, sl, tp and qty, as defined in the system rules.'
)
REPAIR_RULES = (
    'Strict JSON-only serialization test. Return exactly {"action":"WAIT"} '
    'for this diagnostic. Do not include additional characters.'
)

def trial_messages(original: list[dict], variant: str) -> list[dict]:
    if variant not in VARIANTS:
        raise ValueError("Unknown diagnostic variant")
    if len(original) != 2 or [x.get("role") for x in original] != ["system", "user"]:
        raise ValueError("Expected one system and one user message")
    # Copy messages to ensure no mutation across trials.
    result = [dict(msg) for msg in original]
    if variant == "explicit":
        result[0]["content"] += " " + EXTRA_RULES
    elif variant == "repair_prompt":
        result[0]["content"] += " " + REPAIR_RULES
    return result

def safe_raw_preview(raw: str, limit: int = 512) -> str:
    """Escaped representation, bounded to prevent terminal control injection."""
    if not isinstance(raw, str) or not 1 <= limit <= 1024:
        raise ValueError("Invalid raw preview")
    # JSON quoting escapes control characters and preserves the exact content up to limit.
    return json.dumps(raw[:limit], ensure_ascii=True) + (" [TRUNCATED]" if len(raw) > limit else "")

def tokenize_chat(tokenizer, messages: list[dict]):
    """Tokenize one time. Use model-specific chat template, disable thinking when supported."""
    try:
        tokens = tokenizer.apply_chat_template(messages, tokenize=True,
                                               add_generation_prompt=True,
                                               enable_thinking=False,
                                               return_tensors="pt")
    except TypeError as exc:
        raise RuntimeError("Tokenizer lacks required chat-template support") from exc
    if getattr(tokens, "ndim", None) != 2:
        raise RuntimeError("Chat template did not produce a 2D tensor")
    return tokens

def generate_diagnostic(messages, tokenizer, model, max_new_tokens: int):
    import torch
    if not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be 1..256")
    input_ids = tokenize_chat(tokenizer, messages)
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    if device.type != "cuda" or device.index not in (0, None):
        raise RuntimeError("Invalid embedding device")
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(input_ids=input_ids,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] < input_ids.shape[1]:
        raise RuntimeError("Unexpected generated tensor shape")
    new_tokens = output[0, input_ids.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw, time.perf_counter()-start, int(input_ids.shape[1]), int(new_tokens.numel())

def run_trials(messages, context, tokenizer, model, variants, max_new_tokens=128,
               generator=generate_diagnostic):
    if not variants or len(variants) != len(set(variants)) or any(v not in VARIANTS for v in variants):
        raise ValueError("Invalid variants")
    results = []
    for variant in variants:
        raw, seconds, prompt_tokens, output_tokens = generator(
            trial_messages(messages, variant), tokenizer, model, max_new_tokens)
        result = base.interpret_answer(raw.strip(), context)
        print("TRIAL", variant, "prompt_tokens=", prompt_tokens,
              "output_tokens=", output_tokens, "seconds=", round(seconds, 3))
        print("RAW_RESPONSE_ESCAPED:", safe_raw_preview(raw))
        print("VALIDATION:", json.dumps(result, ensure_ascii=False))
        results.append({"variant": variant, "json_valid": result["json_valid"],
                        "risk_accepted": result["risk_accepted"],
                        "latency_seconds": seconds, "output_tokens": output_tokens})
    return results

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check-cache", "infer-diagnostic"), default="check-cache")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--observations", type=Path, default=Path("data/evaluation/v442-observation-audit/pilot_observations.parquet"))
    parser.add_argument("--schema", type=Path, default=Path("data/evaluation/v442-observation-audit/schema.json"))
    parser.add_argument("--minutes", type=Path, default=Path("data/evaluation/v421-binance-1m/BTCUSDT/2024-01.parquet"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    if not 1 <= args.max_new_tokens <= 256 or len(set(args.variants)) != len(args.variants):
        parser.error("Invalid max-new-tokens or duplicate variants")
    try:
        snapshot = base.verify_local_snapshot(cache_dir=args.cache_dir)
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print("CACHE INCOMPLETE:", type(exc).__name__, str(exc)[:180])
        return 2
    print("CACHE PASS:", snapshot)
    if args.mode == "check-cache":
        print("No GPU use, inference, network, training, orders or output files.")
        return 0
    messages, context = base.prepare_case(args.observations, args.schema, args.minutes)
    try:
        import torch
        memory = base.inspect_cuda_memory(torch)
        started = time.perf_counter()
        tokenizer, model = base.load_local_model(snapshot, max_memory=memory)
        print("Load seconds:", round(time.perf_counter()-started, 2))
        run_trials(messages, context, tokenizer, model, args.variants, args.max_new_tokens)
    except (RuntimeError, ValueError, ImportError, OSError, MemoryError) as exc:
        print("DIAGNOSTIC FAILED:", type(exc).__name__, str(exc)[:400])
        return 3
    print("Diagnostic only. All proposals are evaluated, never executed or persisted.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

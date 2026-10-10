"""V4.6.3.2: single RTX 3060 12GiB LOCAL Qwen base-model inference, no writes/downloads/orders.
Run with .venv-unsloth. A base-model JSON answer may be rejected (diagnostic result).
"""
from __future__ import annotations
import argparse
import os
# Set offline flags before importing any HF library.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['WANDB_DISABLED'] = 'true'
import json
import time
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3.5-9B'


def verify_local_snapshot(model_id=MODEL_ID, cache_dir=None):
    """Resolve a complete locally cached repository via HF hub; never connect to network."""
    from huggingface_hub import snapshot_download
    kwargs = {'repo_id': model_id, 'local_files_only': True}
    if cache_dir is not None:
        kwargs['cache_dir'] = str(cache_dir)
    path = Path(snapshot_download(**kwargs))
    if not (path / 'config.json').is_file():
        raise FileNotFoundError('Local model config.json missing')
    if not (path / 'tokenizer_config.json').is_file():
        raise FileNotFoundError('Local tokenizer_config.json missing')
    index_files = list(path.glob('*.safetensors.index.json'))
    if index_files:
        for idx in index_files:
            data = json.loads(idx.read_text(encoding='utf-8'))
            shards = set(data['weight_map'].values())
            if not shards or any(not (path / x).is_file() for x in shards):
                raise FileNotFoundError(f'Local model weights incomplete: {idx.name}')
    elif not list(path.glob('*.safetensors')):
        raise FileNotFoundError('No local safetensors weight files')
    return path


def memory_budget(free_bytes, total_bytes, reserve_mib=1024, cap_gib=11):
    """Safe CUDA allocation ceiling in bytes, never exceeding truly free VRAM."""
    if isinstance(free_bytes, bool) or isinstance(total_bytes, bool):
        raise ValueError('Invalid CUDA memory measurement')
    free_bytes, total_bytes = int(free_bytes), int(total_bytes)
    if free_bytes < 0 or total_bytes <= 0 or free_bytes > total_bytes:
        raise ValueError('Invalid CUDA memory measurement')
    if reserve_mib < 512 or cap_gib < 1:
        raise ValueError('Unsafe GPU memory safety margin')
    budget = min(free_bytes - reserve_mib * 2**20, cap_gib * 2**30)
    if budget < 2 * 2**30:
        raise RuntimeError('Insufficient free VRAM after safety margin')
    return budget


def inspect_cuda_memory(torch):
    """Require exactly the 12GiB RTX 3060; never touch the display RTX 3060 Ti."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU required. Set CUDA_VISIBLE_DEVICES=1 before starting Python.')
    prop = torch.cuda.get_device_properties(0)
    name = str(prop.name)
    if 'RTX 3060' not in name or '3060 Ti' in name or prop.total_memory < 11 * 2**30:
        raise RuntimeError('Wrong GPU: expected NVIDIA RTX 3060 12GiB (not RTX 3060 Ti); abort.')
    with torch.cuda.device(0):
        free, total = torch.cuda.mem_get_info()
    budget = memory_budget(free, total, reserve_mib=1024, cap_gib=11)
    print(f'CUDA 0: {name}; free={free // 2**20} MiB; '
          f'budget={budget // 2**20} MiB; reserve=1024 MiB')
    return {0: budget}


def verify_model_on_single_cuda(model):
    """Verify actual parameter/buffer placement, even if HF does not set hf_device_map.

    Require every nonempty tensor on cuda:0; no CPU, disk/meta, or second GPU.
    Metadata is supplementary, never the only evidence of placement.
    """
    mapping = getattr(model, 'hf_device_map', None)
    if mapping is not None:
        if not isinstance(mapping, dict) or not mapping:
            raise RuntimeError('Invalid device mapping metadata')
        for place in mapping.values():
            if not (place == 0 or str(place).lower() in ('0', 'cuda:0', 'cuda')):
                raise RuntimeError(f'Unsafe device mapping in metadata: {place}')
    checked = 0
    for kind, iterator in (('parameter', model.named_parameters()),
                           ('buffer', model.named_buffers())):
        for name, tensor in iterator:
            if tensor.numel() == 0:
                continue
            device = tensor.device
            if device.type != 'cuda' or device.index not in (0, None):
                raise RuntimeError(f'{kind} {name} is on {device}; expected cuda:0')
            checked += 1
    if checked == 0:
        raise RuntimeError('No model tensors to verify')
    print(f'GPU placement verified: {checked} parameters/buffers on cuda:0; '
          f'hf_device_map present={mapping is not None}')


def load_local_model(path, max_memory, quant4=True):
    """Single-GPU NF4 via Transformers/BitsAndBytes in the existing Unsloth venv.

    No auto placement, download, legacy LoRA, CPU or disk offload.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or set(max_memory) != {0}:
        raise RuntimeError('Only one explicit GPU budget is supported')
    # Verify identity again before any expensive GPU operation.
    prop = torch.cuda.get_device_properties(0)
    if 'RTX 3060' not in str(prop.name) or '3060 Ti' in str(prop.name) or prop.total_memory < 11*2**30:
        raise RuntimeError('Wrong GPU selected')
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                               trust_remote_code=False)
    opts = {'local_files_only': True, 'trust_remote_code': False,
            'device_map': {'': 0}, 'max_memory': dict(max_memory),
            'dtype': torch.float16, 'low_cpu_mem_usage': True}
    if quant4:
        opts['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(str(path), **opts)
    model.eval()
    verify_model_on_single_cuda(model)
    return tokenizer, model


def generate_once(messages, tokenizer, model, max_new_tokens=128):
    import torch
    if not 1 <= max_new_tokens <= 256:
        raise ValueError('max_new_tokens out of bounds')
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    tokens = tokenizer(text, return_tensors='pt')
    # Inputs on the verified embedding device (single RTX 3060).
    device = model.get_input_embeddings().weight.device
    tokens = {k: v.to(device) for k, v in tokens.items()}
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**tokens, max_new_tokens=max_new_tokens,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
    latency = time.perf_counter() - start
    raw = tokenizer.decode(generated[0][tokens['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    return raw, latency


def prepare_case(observations, schema, minutes):
    from qwen_adapter_v461 import one_observation, context_from_minutes, prepare_messages
    from agent_actions_risk_v450 import Position
    names, now, features = one_observation(observations, schema)
    context = context_from_minutes(minutes, now)
    return prepare_messages(now, features, Position(), context, names), context


def interpret_answer(raw, context):
    from qwen_adapter_v461 import parse_action, ModelOutputError
    from agent_actions_risk_v450 import validate_action, Position, RiskPolicy
    try:
        action = parse_action(raw)
    except ModelOutputError as exc:
        return {'json_valid': False, 'reason': str(exc), 'risk_accepted': False}
    verdict = validate_action(action, Position(), context, RiskPolicy())
    return {'json_valid': True, 'action': action.kind.value,
            'risk_accepted': bool(verdict.accepted), 'risk_code': verdict.code}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check-cache', 'infer-one'], default='check-cache')
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--observations', type=Path, default=Path('data/evaluation/v442-observation-audit/pilot_observations.parquet'))
    parser.add_argument('--schema', type=Path, default=Path('data/evaluation/v442-observation-audit/schema.json'))
    parser.add_argument('--minutes', type=Path, default=Path('data/evaluation/v421-binance-1m/BTCUSDT/2024-01.parquet'))
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args(argv)
    if not 1 <= args.max_new_tokens <= 256:
        parser.error('max-new-tokens must be 1..256')
    try:
        snapshot = verify_local_snapshot(cache_dir=args.cache_dir)
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print('CACHE INCOMPLETE — refusing to download or load model:', type(exc).__name__, str(exc)[:250])
        return 2
    print('CACHE PASS — local config, tokenizer and safetensors shards found')
    print('Local snapshot:', snapshot)
    if args.mode == 'check-cache':
        print('No model loaded. No downloads, orders, training or output files.')
        return 0
    messages, context = prepare_case(args.observations, args.schema, args.minutes)
    try:
        import torch
        budgets = inspect_cuda_memory(torch)
        t0 = time.perf_counter()
        tokenizer, model = load_local_model(snapshot, max_memory=budgets)
        print('Load seconds:', round(time.perf_counter() - t0, 2))
        raw, latency = generate_once(messages, tokenizer, model, args.max_new_tokens)
    except (RuntimeError, ValueError, ImportError, OSError, MemoryError) as exc:
        print('LOAD/INFERENCE FAILED (no fallback/download):', type(exc).__name__, str(exc)[:400])
        return 3
    print('Inference seconds:', round(latency, 2))
    print('Output characters:', len(raw))
    print('Validation:', json.dumps(interpret_answer(raw, context), ensure_ascii=False))
    # Do not print untrusted full generations or persist user/model data.
    print('No output saved, no training, no simulated or live orders.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

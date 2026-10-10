"""V4.6.5: deterministic, read-only JSON compliance campaign on BTCUSDT pilot.
One RTX 3060, one model load; no Test, LoRA, order execution, or output writes.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
import time
from pathlib import Path
import pandas as pd
import qwen_local_inference_v4633 as base
from qwen_json_diagnostics_v4641 import generate_diagnostic, safe_raw_preview
from qwen_adapter_v461 import load_schema, prepare_messages, context_from_minutes, parse_action, ModelOutputError
from agent_actions_risk_v450 import Position, RiskPolicy, validate_action

# Explicit format enforcement only. Do not force a particular trading action.
FORMAT_RULES = (
    ' Output one and only one minified JSON object, without markdown, commentary, '
    'reasoning, timestamps, confidence, explanations, or additional keys. '
    'Permitted keys are action, sl, tp, qty, subject to the action-specific rules above. '
    'Do not copy decision_at or any market input field into your answer. '
    'Use WAIT if FLAT and uncertain. OPEN_LONG requires finite numeric sl and tp. '
    'No SHORT trades in Spot. Do not output null numeric order parameters.'
)
DEFAULT_OBS = Path('data/evaluation/v442-observation-audit/pilot_observations.parquet')
DEFAULT_SCHEMA = Path('data/evaluation/v442-observation-audit/schema.json')
DEFAULT_MINUTES = Path('data/evaluation/v421-binance-1m/BTCUSDT/2024-01.parquet')


def select_observations(path: Path, names: list[str], count: int, stride: int) -> list[tuple[pd.Timestamp, dict]]:
    if not 1 <= count <= 12 or not 1 <= stride <= 1000:
        raise ValueError('count must be 1..12 and stride 1..1000')
    frame = pd.read_parquet(path)
    if list(frame.columns) != ['decision_at'] + names:
        raise ValueError('Pilot observation schema mismatch')
    if len(frame) < (count - 1) * stride + 1:
        raise ValueError('Not enough pilot observations')
    times = pd.to_datetime(frame['decision_at'], utc=True, errors='raise')
    if frame['decision_at'].isna().any() or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('Non-unique or non-chronological decisions')
    if (times.dt.minute % 5 != 0).any() or (times.dt.second != 0).any():
        raise ValueError('Non-5m decision times')
    selected = []
    for index in range(0, count * stride, stride):
        timestamp = times.iloc[index]
        selected.append((timestamp, {key: frame.iloc[index][key] for key in names}))
    # This campaign is ONLY the frozen Jan 2024 BTC pilot, not Train/Val/Test tuning.
    start = pd.Timestamp('2024-01-01T00:00:00Z')
    end = pd.Timestamp('2024-02-01T00:00:00Z')
    if any(not start < t < end for t, _ in selected):
        raise ValueError('Selected observations outside BTC Jan 2024 pilot')
    return selected


def format_messages(timestamp, features, context, names):
    messages = prepare_messages(timestamp, features, Position(), context, names)
    messages = [dict(item) for item in messages]
    messages[0]['content'] += FORMAT_RULES
    return messages


def evaluate(raw: str, context) -> dict:
    try:
        action = parse_action(raw.strip())
    except ModelOutputError as exc:
        return {'json_valid': False, 'action': None, 'risk_accepted': False,
                'risk_code': 'INVALID_JSON_CONTRACT', 'reason': str(exc)}
    verdict = validate_action(action, Position(), context, RiskPolicy())
    return {'json_valid': True, 'action': action.kind.value,
            'risk_accepted': bool(verdict.accepted), 'risk_code': str(verdict.code)}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('No campaign samples')
    accepted = sum(bool(r['risk_accepted']) for r in rows)
    valid = sum(bool(r['json_valid']) for r in rows)
    actions: dict[str, int] = {}
    codes: dict[str, int] = {}
    for row in rows:
        action = row['action'] if row['action'] is not None else 'INVALID'
        actions[action] = actions.get(action, 0) + 1
        codes[row['risk_code']] = codes.get(row['risk_code'], 0) + 1
    times = [r['latency_seconds'] for r in rows]
    if any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError('Bad latency')
    return {'observations': len(rows), 'json_valid': valid, 'json_valid_rate': valid / len(rows),
            'risk_accepted': accepted, 'risk_accept_rate': accepted / len(rows),
            'actions': actions, 'risk_codes': codes,
            'latency_mean_seconds': round(statistics.mean(times), 3),
            'latency_median_seconds': round(statistics.median(times), 3),
            'latency_max_seconds': round(max(times), 3),
            'generated_tokens_total': sum(r['output_tokens'] for r in rows)}


def run_campaign(samples, names, minutes_path, tokenizer, model, max_new_tokens=128,
                 generator=generate_diagnostic, emit=print):
    rows = []
    # Validate full 1m data once, and forbid future candle access through the causal helper.
    for number, (timestamp, features) in enumerate(samples, start=1):
        context = context_from_minutes(minutes_path, timestamp)
        messages = format_messages(timestamp, features, context, names)
        raw, seconds, prompt_tokens, output_tokens = generator(messages, tokenizer, model, max_new_tokens)
        verdict = evaluate(raw, context)
        record = {'index': number, 'decision_at': timestamp.isoformat(),
                  'latency_seconds': float(seconds), 'prompt_tokens': int(prompt_tokens),
                  'output_tokens': int(output_tokens), **verdict}
        rows.append(record)
        emit('SAMPLE ' + json.dumps(record, ensure_ascii=False))
        emit('RAW_RESPONSE_ESCAPED: ' + safe_raw_preview(raw, 240))
    report = summarize(rows)
    emit('CAMPAIGN SUMMARY: ' + json.dumps(report, ensure_ascii=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check-cache', 'infer-campaign'), default='check-cache')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--observations', type=Path, default=DEFAULT_OBS)
    parser.add_argument('--schema', type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument('--minutes', type=Path, default=DEFAULT_MINUTES)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--stride', type=int, default=288, help='5m rows between samples (288 = one day)')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args(argv)
    if not 1 <= args.count <= 12 or not 1 <= args.stride <= 1000 or not 1 <= args.max_new_tokens <= 256:
        parser.error('count 1..12, stride 1..1000, max-new-tokens 1..256')
    try:
        snapshot = base.verify_local_snapshot(cache_dir=args.cache_dir)
    except (ValueError, OSError, KeyError) as exc:
        print('CACHE INCOMPLETE:', type(exc).__name__, str(exc)[:180]); return 2
    print('CACHE PASS:', snapshot)
    if args.mode == 'check-cache':
        print('No GPU load, network, training, Test access, orders or output files.'); return 0
    try:
        names = load_schema(args.schema)
        samples = select_observations(args.observations, names, args.count, args.stride)
        # Check source chronology and minute continuity before any inference.
        from prototype_multitimeframe_v42 import validate_minutes
        validate_minutes(pd.read_parquet(args.minutes))
        import torch
        budget = base.inspect_cuda_memory(torch)
        began = time.perf_counter()
        tokenizer, model = base.load_local_model(snapshot, max_memory=budget)
        print('LOAD seconds:', round(time.perf_counter()-began, 2))
        run_campaign(samples, names, args.minutes, tokenizer, model, args.max_new_tokens)
    except (ValueError, RuntimeError, OSError, ImportError, MemoryError) as exc:
        print('CAMPAIGN FAILED:', type(exc).__name__, str(exc)[:400]); return 3
    print('Diagnostics only. No output files, Test evaluation, training, simulated or real orders.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

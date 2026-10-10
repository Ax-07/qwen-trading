"""V4.6.6 — Paired prompt/market sensitivity diagnostic, read-only, BTC Jan 2024.
Both prompts run on identical causally constructed observations; sample selection
is retrospective for diagnostics only, NEVER a performance or trading label.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
import time
from pathlib import Path
import pandas as pd
import qwen_contract_campaign_v465 as previous
import qwen_local_inference_v4633 as base
from qwen_json_diagnostics_v4641 import generate_diagnostic, safe_raw_preview
from qwen_adapter_v461 import load_schema, context_from_minutes
from prototype_multitimeframe_v42 import validate_minutes

DEFAULT_OBS = previous.DEFAULT_OBS
DEFAULT_SCHEMA = previous.DEFAULT_SCHEMA
DEFAULT_MINUTES = previous.DEFAULT_MINUTES
# Both formats remain strict; only the recommendation of WAIT when unsure changes.
NEUTRAL_FORMAT = (
    ' Output one and only one minified JSON object, without markdown, commentary, '
    'reasoning, timestamps, confidence, explanations, or additional keys. '
    'Permitted keys are action, sl, tp, qty, subject to action-specific rules above. '
    'Do not copy decision_at or any market input field into your answer. '
    'OPEN_LONG requires finite numeric sl and tp, no SHORT trades in Spot. '
    'Do not output null numeric order parameters. Evaluate WAIT and OPEN_LONG '
    'according to the evidence, without preferring either action.'
)
PROFILES = ('up_60m', 'down_60m', 'high_range_60m', 'low_range_60m', 'median_return_60m')

def select_regimes(observations: pd.DataFrame, names: list[str], minutes: pd.DataFrame,
                   start='2024-01-01T00:00:00Z', end='2024-01-08T00:00:00Z'):
    """Choose 5 distinct 5m decisions by trailing-only 60m features.

    Selection over historical candidates is retrospective; features provided to
    Qwen at each decision remain causal. Never use later returns as labels.
    """
    if list(observations.columns) != ['decision_at'] + names or len(names) != 144 or len(set(names)) != 144:
        raise ValueError('Pilot schema mismatch')
    times = pd.to_datetime(observations.decision_at, utc=True, errors='raise')
    if times.isna().any() or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('Invalid decision chronology')
    if ((times.dt.minute % 5) != 0).any() or (times.dt.second != 0).any():
        raise ValueError('Decisions must align to 5m')
    candles = validate_minutes(minutes).copy().set_index('timestamp').sort_index()
    if not candles.index.is_unique:
        raise ValueError('Duplicate 1m timestamps')
    # Shift nothing forward: an observation at time T sees only [T-60m, T).
    # Use the 60th minute back as base close, then last completed close.
    trailing = candles[['close','high','low']].copy()
    trailing['prior_close_60m'] = trailing['close'].shift(59)
    trailing['window_high'] = trailing['high'].rolling(60, min_periods=60).max()
    trailing['window_low'] = trailing['low'].rolling(60, min_periods=60).min()
    trailing['ret_60m'] = trailing['close'] / trailing['prior_close_60m'] - 1
    trailing['range_60m'] = (trailing['window_high'] - trailing['window_low']) / trailing['prior_close_60m']
    trailing.index = trailing.index + pd.Timedelta(minutes=1) # 1m candle available at its close
    bound_start, bound_end = pd.Timestamp(start), pd.Timestamp(end)
    if bound_start.tzinfo is None or bound_end.tzinfo is None or not (pd.Timestamp('2024-01-01T00:00:00Z') <= bound_start < bound_end <= pd.Timestamp('2024-02-01T00:00:00Z')):
        raise ValueError('Selection must remain in Jan 2024 development pilot')
    candidates = observations.loc[(times >= bound_start) & (times < bound_end)].copy()
    candidates['decision_at'] = times.loc[candidates.index]
    metrics = trailing[['ret_60m','range_60m']].reindex(pd.DatetimeIndex(candidates.decision_at))
    candidates['ret_60m'] = metrics['ret_60m'].to_numpy()
    candidates['range_60m'] = metrics['range_60m'].to_numpy()
    candidates = candidates.dropna(subset=['ret_60m','range_60m'])
    candidates = candidates.loc[candidates['range_60m'] >= 0].reset_index(drop=True)
    if len(candidates) < 5:
        raise ValueError('Not enough valid causal observations')
    # Stable tie breaking by timestamp and never reselect the same sample.
    chosen, used = [], set()
    scoring = (
        ('up_60m', 'ret_60m', False),
        ('down_60m', 'ret_60m', True),
        ('high_range_60m', 'range_60m', False),
        ('low_range_60m', 'range_60m', True),
    )
    for label, col, ascending in scoring:
        ranked = candidates.sort_values([col,'decision_at'], ascending=[ascending,True], kind='stable')
        idx = next((i for i in ranked.index if i not in used), None)
        if idx is None: raise ValueError('Insufficient distinct cases')
        used.add(idx); chosen.append((label, candidates.loc[idx]))
    med = float(candidates['ret_60m'].median())
    for idx in candidates.assign(distance=(candidates.ret_60m-med).abs()).sort_values(['distance','decision_at'], kind='stable').index:
        if idx not in used:
            used.add(idx); chosen.append(('median_return_60m', candidates.loc[idx])); break
    return [(label, pd.Timestamp(row.decision_at), {k: row[k] for k in names},
             {'trailing_return_60m_pct': round(float(row.ret_60m)*100, 5),
              'trailing_range_60m_pct': round(float(row.range_60m)*100, 5)})
            for label,row in chosen]

def prepare_pair(timestamp, features, context, names):
    # V4.6.5's stable prompt and a prompt with identical schema but no WAIT preference.
    cautious = previous.format_messages(timestamp, features, context, names)
    neutral = [dict(message) for message in cautious]
    base_system = cautious[0]['content']
    if not base_system.endswith(previous.FORMAT_RULES):
        raise RuntimeError('Unexpected stable prompt')
    prefix = base_system[:-len(previous.FORMAT_RULES)]
    prefix = prefix.replace('When uncertain, choose WAIT if FLAT or HOLD if invested.',
                            'Assess the market evidence before selecting an action.')
    neutral[0]['content'] = prefix + NEUTRAL_FORMAT
    assert cautious[1] == neutral[1]
    return {'cautious': cautious, 'neutral': neutral}

def summarize(rows):
    if not rows or any(r['prompt'] not in ('cautious','neutral') for r in rows):
        raise ValueError('Invalid campaign')
    output = {}
    for prompt in ('cautious','neutral'):
        items=[r for r in rows if r['prompt']==prompt]
        if not items: raise ValueError('Missing prompt')
        actions={}
        codes={}
        for r in items:
            a=r['action'] or 'INVALID'
            actions[a]=actions.get(a,0)+1
            codes[r['risk_code']]=codes.get(r['risk_code'],0)+1
        lat=[r['latency_seconds'] for r in items]
        if any(not math.isfinite(t) or t < 0 for t in lat): raise ValueError('Bad latency')
        output[prompt]={'count':len(items),'json_valid':sum(r['json_valid'] for r in items),
                       'risk_accepted':sum(r['risk_accepted'] for r in items),
                       'actions':actions,'risk_codes':codes,
                       'latency_mean_seconds':round(statistics.mean(lat),3),
                       'prompt_tokens_min':min(r['prompt_tokens'] for r in items),
                       'prompt_tokens_max':max(r['prompt_tokens'] for r in items)}
    by_case={}
    for r in rows:
        by_case.setdefault(r['profile'],{})[r['prompt']]=r['action']
    if any(set(v)!= {'cautious','neutral'} for v in by_case.values()):
        raise ValueError('Unpaired profile')
    output['actions_changed_between_prompts']=sum(x['cautious']!=x['neutral'] for x in by_case.values())
    output['total_inferences']=len(rows)
    return output

def run_cases(cases, names, minutes_path, tokenizer, model, max_new_tokens=128,
              generator=generate_diagnostic, emit=print):
    results=[]
    if len(cases)!=5 or [x[0] for x in cases]!=list(PROFILES):
        raise ValueError('Five distinct profiles required')
    for label,timestamp,features,metrics in cases:
        context=context_from_minutes(minutes_path,timestamp)
        pairs=prepare_pair(timestamp,features,context,names)
        for prompt in ('cautious','neutral'):
            raw, seconds, prompt_tokens, output_tokens=generator(pairs[prompt],tokenizer,model,max_new_tokens)
            verdict=previous.evaluate(raw,context)
            row={'profile':label,'prompt':prompt,'decision_at':timestamp.isoformat(),
                 **metrics,'latency_seconds':float(seconds),
                 'prompt_tokens':int(prompt_tokens),'output_tokens':int(output_tokens),**verdict}
            results.append(row)
            emit('SAMPLE '+json.dumps(row,ensure_ascii=False,allow_nan=False))
            emit('RAW_RESPONSE_ESCAPED: '+safe_raw_preview(raw,240))
    report=summarize(results)
    emit('SENSITIVITY SUMMARY: '+json.dumps(report,ensure_ascii=False))
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check-cache','infer-sensitivity'),default='check-cache')
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--observations',type=Path,default=DEFAULT_OBS)
    parser.add_argument('--schema',type=Path,default=DEFAULT_SCHEMA)
    parser.add_argument('--minutes',type=Path,default=DEFAULT_MINUTES)
    parser.add_argument('--max-new-tokens',type=int,default=128)
    args=parser.parse_args(argv)
    if not 1<=args.max_new_tokens<=256: parser.error('max-new-tokens must be 1..256')
    try: snapshot=base.verify_local_snapshot(cache_dir=args.cache_dir)
    except (OSError,ValueError,KeyError) as exc:
        print('CACHE INCOMPLETE:',type(exc).__name__,str(exc)[:160]);return 2
    print('CACHE PASS:',snapshot)
    if args.mode=='check-cache':
        print('No GPU usage, model load, orders, training, Test access or output files.');return 0
    try:
        names=load_schema(args.schema)
        observations=pd.read_parquet(args.observations)
        minutes=validate_minutes(pd.read_parquet(args.minutes))
        cases=select_regimes(observations,names,minutes)
        print('SELECTED CAUSAL CASES:',[(label,t.isoformat(),stats) for label,t,_,stats in cases])
        import torch
        budget=base.inspect_cuda_memory(torch)
        start=time.perf_counter()
        tokenizer,model=base.load_local_model(snapshot,max_memory=budget)
        print('LOAD seconds:',round(time.perf_counter()-start,2))
        run_cases(cases,names,args.minutes,tokenizer,model,args.max_new_tokens)
    except (OSError,ValueError,RuntimeError,MemoryError,ImportError) as exc:
        print('SENSITIVITY FAILED:',type(exc).__name__,str(exc)[:400]);return 3
    print('Diagnostic only; no output files, trading, training or Test evaluation.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())

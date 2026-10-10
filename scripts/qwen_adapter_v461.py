"""V4.6.1 -- strict Qwen action adapter, OFFLINE by default; never places orders.

Model output is an untrusted proposal. Risk validation remains in V4.5.0/V4.6.0.
Optional local OpenAI-compatible endpoint is invoked ONLY with explicit --mode local-one.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Mapping
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pandas as pd

from agent_actions_risk_v450 import Action, ActionType, MarketContext, Position, RiskPolicy, Side, validate_action

FIELDS = frozenset({'action', 'sl', 'tp', 'qty'})
KINDS = {k.value: k for k in ActionType}


class ModelOutputError(ValueError):
    """Invalid model output; never silently replace with a tradable action."""


def _finite_positive(v: object, field: str) -> float:
    if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
        raise ModelOutputError(f'{field} must be a strictly positive finite number')
    return float(v)


def parse_action(raw: str) -> Action:
    """Require one exact JSON object, no extra text, duplicate keys or implicit coercion."""
    if not isinstance(raw, str) or len(raw.encode('utf-8')) > 2048:
        raise ModelOutputError('empty or oversized model answer')
    def unique_pairs(pairs):
        d = {}
        for key, value in pairs:
            if key in d:
                raise ModelOutputError('duplicate JSON key')
            d[key] = value
        return d
    def reject_constant(value):
        raise ModelOutputError('non-finite JSON constant')
    try:
        obj = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    except (TypeError, ValueError) as exc:
        raise ModelOutputError('invalid JSON object') from exc
    if not isinstance(obj, dict) or set(obj) - FIELDS or 'action' not in obj:
        raise ModelOutputError('action object requires action and only approved keys')
    if type(obj['action']) is not str or obj['action'] not in KINDS:
        raise ModelOutputError('unknown action')
    kind = KINDS[obj['action']]
    required = {
        ActionType.WAIT: set(), ActionType.HOLD: set(), ActionType.CLOSE: set(),
        ActionType.OPEN_LONG: {'sl','tp'}, ActionType.OPEN_SHORT: {'sl','tp'},
        ActionType.MOVE_SL: {'sl'}, ActionType.MOVE_TP: {'tp'},
        ActionType.MOVE_SL_TP: {'sl','tp'},
    }[kind]
    supplied = set(obj) - {'action'}
    if supplied not in (required, required | {'qty'} if kind in (ActionType.OPEN_LONG, ActionType.OPEN_SHORT) else required):
        raise ModelOutputError('invalid action parameters')
    kw = {key: _finite_positive(obj[key], key) for key in supplied}
    return Action(kind=kind, **kw)


def _safe_number(value):
    """Null for genuine missing values, never stringify NaN or Infinity."""
    if value is None or pd.isna(value):
        return None
    if type(value) is bool:
        raise ValueError('unexpected boolean feature')
    if not isinstance(value, (int, float)) and not hasattr(value, 'item'):
        raise ValueError('non-numeric feature')
    n = float(value)
    if not math.isfinite(n):
        raise ValueError('non-finite observation feature')
    return n


def load_schema(schema_path: Path) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding='utf-8'))
    names = schema['feature_columns']
    if not isinstance(names, list) or len(names) != 144 or any(type(x) is not str for x in names) or len(set(names)) != 144:
        raise ValueError('invalid feature schema')
    digest = hashlib.sha256(json.dumps(names).encode()).hexdigest()
    if digest != schema['feature_schema_sha256']:
        raise ValueError('feature schema SHA256 mismatch')
    return names


def prepare_messages(decision_at, features: Mapping, position: Position, context: MarketContext, feature_names: list[str]) -> list[dict]:
    now = pd.Timestamp(decision_at)
    if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
        raise ValueError('naive decision timestamp')
    if pd.Timestamp(context.decision_at) != now:
        raise ValueError('context timestamp differs from decision')
    if len(feature_names) != 144 or set(features) != set(feature_names):
        raise ValueError('observation schema mismatch')
    values = {name: _safe_number(features[name]) for name in feature_names}
    state = {'side': position.side.value, 'entry_price': position.entry_price,
             'qty': position.qty, 'sl': position.sl, 'tp': position.tp,
             'opened_at': position.opened_at.isoformat() if position.opened_at else None}
    market = {'reference_price': context.reference_price, 'spread_bps': context.spread_bps,
              'equity': context.equity, 'liquidity_notional': context.liquidity_notional,
              'market_allows_short': context.market_allows_short}
    system = ('You are an experimental trading action proposer in a simulated Spot market. '
              'Return exactly one JSON object and no markdown or prose. Allowed actions: '
              'WAIT, OPEN_LONG, OPEN_SHORT, HOLD, CLOSE, MOVE_SL, MOVE_TP, MOVE_SL_TP. '
              'OPEN_LONG/OPEN_SHORT require numeric sl,tp and optional qty; MOVE_SL requires sl; '
              'MOVE_TP requires tp; MOVE_SL_TP requires sl,tp; others require no other keys. '
              'The risk engine can reject your proposal. No future candles are available. '
              'Missing numerical observations are null. When uncertain, choose WAIT if FLAT or HOLD if invested.')
    user = json.dumps({'decision_at': now.isoformat(), 'features': values, 'position': state,
                       'market': market}, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


@dataclass
class QwenAgent:
    feature_names: list[str]
    generate: Callable[[list[dict]], str]
    calls: int = 0
    total_latency_seconds: float = 0.0

    def __call__(self, decision_at, features, position, context) -> Action:
        messages = prepare_messages(decision_at, features, position, context, self.feature_names)
        t0 = time.monotonic()
        raw = self.generate(messages)
        self.total_latency_seconds += time.monotonic() - t0
        self.calls += 1
        # Invalid model answers abort replay rather than being silently coerced.
        return parse_action(raw)


def local_openai_generate(url: str, model: str, timeout_seconds: float = 30.0) -> Callable[[list[dict]], str]:
    """Explicit local-only inference; no credentials, remote servers or auto-discovery."""
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('only an explicit localhost HTTP endpoint is allowed')
    if parsed.path != '/v1/chat/completions' or not isinstance(model, str) or not model.strip():
        raise ValueError('expected /v1/chat/completions and a model id')
    if not (1 <= timeout_seconds <= 120):
        raise ValueError('timeout must be between 1 and 120 seconds')
    def generate(messages):
        payload = json.dumps({'model': model, 'messages': messages, 'temperature': 0,
                              'max_tokens': 128, 'stream': False}, allow_nan=False).encode('utf-8')
        request = Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(request, timeout=timeout_seconds) as resp:
            if resp.status != 200:
                raise ModelOutputError('local inference endpoint returned non-200 status')
            raw = resp.read(65537)
        if len(raw) > 65536:
            raise ModelOutputError('oversized inference response')
        response = json.loads(raw)
        answer = response['choices'][0]['message']['content']
        if type(answer) is not str:
            raise ModelOutputError('missing text response')
        return answer
    return generate


def one_observation(observations_path: Path, schema_path: Path):
    names = load_schema(schema_path)
    frame = pd.read_parquet(observations_path)
    if len(frame) == 0 or list(frame.columns) != ['decision_at'] + names:
        raise ValueError('pilot data/schema mismatch')
    row = frame.iloc[0]
    now = pd.Timestamp(row['decision_at'])
    if now.tzinfo is None:
        raise ValueError('missing timezone')
    return names, now, {name: row[name] for name in names}


def context_from_minutes(minutes_path: Path, decision_at: pd.Timestamp) -> MarketContext:
    """Use the completed 1m candle ending at decision_at, never future quotes."""
    from prototype_multitimeframe_v42 import validate_minutes
    candles = validate_minutes(pd.read_parquet(minutes_path))
    matched = candles.loc[candles['timestamp'] == pd.Timestamp(decision_at) - pd.Timedelta(minutes=1)]
    if len(matched) != 1:
        raise ValueError('missing last completed minute for observation decision')
    candle = matched.iloc[0]
    px = float(candle['close'])
    notional = float(candle['quote_volume']) if 'quote_volume' in candle and pd.notna(candle['quote_volume']) else float(candle['volume']) * px
    if not (math.isfinite(px) and px > 0 and math.isfinite(notional) and notional >= 0):
        raise ValueError('invalid candle price/liquidity')
    now = pd.Timestamp(decision_at).to_pydatetime()
    return MarketContext(now, px, 5.0, notional, 10000.0,
                         market_allows_short=False, quote_known_at=now)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['offline-one', 'local-one'], default='offline-one')
    ap.add_argument('--observations', type=Path, default=Path('data/evaluation/v442-observation-audit/pilot_observations.parquet'))
    ap.add_argument('--schema', type=Path, default=Path('data/evaluation/v442-observation-audit/schema.json'))
    ap.add_argument('--minutes', type=Path, default=Path('data/evaluation/v421-binance-1m/BTCUSDT/2024-01.parquet'))
    ap.add_argument('--url', default='http://127.0.0.1:8000/v1/chat/completions')
    ap.add_argument('--model', default='Qwen3.5-9B')
    ap.add_argument('--timeout', type=float, default=30.0)
    args = ap.parse_args()
    names, now, features = one_observation(args.observations, args.schema)
    context = context_from_minutes(args.minutes, now)
    if args.mode == 'offline-one':
        generate = lambda messages: '{"action":"WAIT"}'
    else:
        generate = local_openai_generate(args.url, args.model, args.timeout)
    agent = QwenAgent(names, generate)
    action = agent(now, features, Position(), context)
    verdict = validate_action(action, Position(), context, RiskPolicy())
    print('V4.6.1', args.mode, 'action=', action.kind.value, 'risk_accepted=', verdict.accepted,
          'risk_code=', verdict.code, 'calls=', agent.calls, 'latency_seconds=', round(agent.total_latency_seconds, 3))
    print('One observation only. No orders, no trading, no training, no Test, no saved model response.')


if __name__ == '__main__':
    main()

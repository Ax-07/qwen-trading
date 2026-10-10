"""V4.6.0 deterministic agent-in-the-loop Spot 1m replay. No live orders, no Qwen.

Decisions at close of completed UTC 5m blocks; accepted orders eligible at
next 1m opening, never at the observation close. Gap in 1m data aborts.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import math
from typing import Callable
from dataclasses import dataclass, asdict
from datetime import timezone
from pathlib import Path
import pandas as pd
from agent_actions_risk_v450 import (Action, ActionType, Side, Position,
    MarketContext, RiskPolicy, validate_action)
from prototype_multitimeframe_v42 import validate_minutes


@dataclass(frozen=True)
class ExecutionConfig:
    starting_equity: float = 10000.0
    fee_bps: float = 7.0
    slippage_bps: float = 3.0
    spread_bps: float = 5.0
    # Fees & slippage on exits are charged independently from entrance.


def replay_agent(frame: pd.DataFrame, observations: pd.DataFrame, agent: Callable, *, config: ExecutionConfig = ExecutionConfig(),
           policy: RiskPolicy = RiskPolicy()) -> tuple[pd.DataFrame, dict]:
    """At each 5m close pass ONLY that decision's 144 features and current Position.

    Agents receive an independent, read-only MappingProxyType snapshot plus a frozen
    Position and MarketContext. No candles, next row, or future observations exposed.
    The engine is deliberately forked from V4.5.1 to avoid modifying its baseline.
    """
    from types import MappingProxyType
    if not callable(agent): raise ValueError('agent must be callable')
    if not isinstance(observations, pd.DataFrame) or 'decision_at' not in observations:
        raise ValueError('observations missing decision_at')
    obs = observations.copy(deep=True)
    if len(obs) == 0 or obs.columns.duplicated().any(): raise ValueError('empty/duplicate observation columns')
    obs['decision_at'] = pd.to_datetime(obs.decision_at, utc=True, errors='raise')
    if obs.decision_at.isna().any() or not obs.decision_at.is_unique or not obs.decision_at.is_monotonic_increasing:
        raise ValueError('invalid observation chronology')
    feature_names = [c for c in obs if c != 'decision_at']
    if len(feature_names) != 144: raise ValueError('expected 144 pilot features')
    if not all(pd.api.types.is_numeric_dtype(obs[c]) for c in feature_names):
        raise ValueError('non-numeric pilot feature')
    for c in feature_names:
        x = pd.to_numeric(obs[c], errors='raise')
        if pd.Series(x).isin([math.inf, -math.inf]).any(): raise ValueError('infinite pilot feature')
    snapshots = {t: tuple(row) for t, row in zip(obs.decision_at, obs[feature_names].itertuples(index=False, name=None))}
    if config.starting_equity <= 0 or min(config.fee_bps, config.slippage_bps, config.spread_bps) < 0:
        raise ValueError('invalid execution config')
    f = validate_minutes(frame).reset_index(drop=True)
    if f.empty: raise ValueError('empty minutes')
    if (f.timestamp.diff().iloc[1:] != pd.Timedelta(minutes=1)).any():
        raise ValueError('GAP_1M: fail closed; do not infer executions during missing intervals')
    if not (f.timestamp.dt.minute.iloc[0] % 5 == 0):
        raise ValueError('first 1m must begin on UTC 5m boundary')
    if len(f) % 5:
        raise ValueError('complete 5m blocks required')
    first = f.timestamp.iloc[0]
    final = f.timestamp.iloc[-1] + pd.Timedelta(minutes=1)
    expected_times = pd.date_range(first + pd.Timedelta(minutes=5), final, freq='5min', tz='UTC')
    if len(snapshots) != len(expected_times) or set(snapshots) != set(expected_times):
        raise ValueError('observation timeline must exactly match complete 5m candles')
    cash = float(config.starting_equity)
    position = Position()
    entry_fee = 0.0
    pending = None
    events = []
    trades = 0
    EPS = 1e-8
    def emit(at, kind, **kwargs):
        events.append({'at': at.isoformat(), 'event': kind, **kwargs})
    def equity_at(mark):
        if position.side == Side.FLAT: return cash
        direction = 1 if position.side == Side.LONG else -1
        return cash + direction * (mark - position.entry_price) * position.qty
    def exit_position(t, price, reason):
        nonlocal cash, position, entry_fee, trades
        px = max(EPS, price * (1 - config.slippage_bps / 10000))
        fee = px * position.qty * config.fee_bps / 10000
        pnl = (px - position.entry_price) * position.qty - fee
        cash += pnl
        emit(t, 'EXIT', reason=reason, price=px, qty=position.qty,
             exit_fee=fee, net_trade_pnl=pnl-entry_fee, cash=cash)
        trades += 1
        position = Position()
        entry_fee = 0.0
    for i, row in enumerate(f.itertuples(index=False)):
        t = row.timestamp.to_pydatetime()
        opening = float(row.open)
        closing = float(row.close)
        high = float(row.high)
        low = float(row.low)
        liquidity = float(getattr(row, 'quote_volume', row.volume * opening))
        # Old protective stop/target have priority over all pending modifications.
        if position.side != Side.FLAT:
            if opening <= position.sl:
                exit_position(t, opening, 'STOP_GAP')
            elif opening >= position.tp:
                exit_position(t, position.tp, 'TARGET_GAP_CAPPED')
        if pending is not None:
            action, preview_qty, decision_at = pending
            pending = None
            if position.side == Side.FLAT and action.kind == ActionType.OPEN_LONG:
                ctx = MarketContext(t, opening, config.spread_bps, liquidity, cash,
                                    market_allows_short=False, quote_known_at=t)
                verdict = validate_action(Action(ActionType.OPEN_LONG, sl=action.sl,
                                    tp=action.tp, qty=preview_qty), position, ctx, policy)
                if verdict.accepted:
                    px = opening * (1 + config.slippage_bps / 10000)
                    # Recalculate allowed quantity at final entry price, including fee & slippage.
                    cost_per_unit = (px-action.sl) + px * policy.assumed_roundtrip_cost_bps/10000
                    budget = cash * policy.max_risk_fraction
                    max_risk_qty = max(0.0, budget / cost_per_unit) if cost_per_unit > 0 else 0
                    max_exposure_qty = cash*policy.max_notional_to_equity/px
                    max_liquid_qty = liquidity*policy.max_liquidity_fraction/px
                    qty = min(verdict.approved_qty, max_risk_qty, max_exposure_qty, max_liquid_qty)
                    from math import floor
                    qty = floor(qty/policy.qty_step*(1-1e-12))*policy.qty_step
                    if qty > 0 and action.sl < px < action.tp and px*qty*config.fee_bps/10000 < cash:
                        position = Position(Side.LONG, px, qty, action.sl, action.tp, t)
                        entry_fee = px*qty*config.fee_bps/10000
                        cash -= entry_fee
                        emit(t, 'ENTRY', price=px, qty=qty, sl=action.sl, tp=action.tp,
                             fee=entry_fee, originating_decision=decision_at.isoformat())
                    else:
                        emit(t, 'REJECT_AT_FILL', reason='INVALID_FILL_GEOMETRY_OR_SIZE')
                else:
                    emit(t, 'REJECT_AT_FILL', reason=verdict.code)
            elif position.side != Side.FLAT and action.kind == ActionType.CLOSE:
                exit_position(t, opening, 'AGENT_CLOSE')
            elif position.side != Side.FLAT and action.kind in (ActionType.MOVE_SL, ActionType.MOVE_TP, ActionType.MOVE_SL_TP):
                ctx = MarketContext(t, opening, config.spread_bps, liquidity, equity_at(opening),
                                    quote_known_at=t)
                verdict = validate_action(action, position, ctx, policy)
                if verdict.accepted:
                    position = Position(position.side, position.entry_price, position.qty,
                                        action.sl if action.sl is not None else position.sl,
                                        action.tp if action.tp is not None else position.tp,
                                        position.opened_at)
                    emit(t, 'MODIFY', sl=position.sl, tp=position.tp)
                else: emit(t, 'REJECT_AT_FILL', reason=verdict.code)
            elif action.kind not in (ActionType.WAIT, ActionType.HOLD):
                emit(t, 'REJECT_AT_FILL', reason='POSITION_CHANGED')
        # 1m intrabar order unknowable -> stop first, even if both touched.
        if position.side != Side.FLAT:
            if low <= position.sl:
                exit_position(t, position.sl, 'STOP_INTRABAR')
            elif high >= position.tp:
                exit_position(t, position.tp, 'TARGET_INTRABAR')
        end = t + pd.Timedelta(minutes=1)
        if (i+1) % 5 == 0:
            from types import MappingProxyType
            state = MappingProxyType(dict(zip(feature_names, snapshots[pd.Timestamp(end)])))
            ctx_agent = MarketContext(end, closing, config.spread_bps,
                                      float(getattr(row, 'quote_volume', row.volume*closing)),
                                      equity_at(closing), market_allows_short=False, quote_known_at=end)
            # Any agent failure aborts the replay: fail closed, no implicit trading.
            action = agent(pd.Timestamp(end), state, position, ctx_agent)
            if not isinstance(action, Action): raise ValueError('agent returned non-Action')
            ctx = MarketContext(end, closing, config.spread_bps,
                                float(getattr(row, 'quote_volume', row.volume*closing)),
                                equity_at(closing), market_allows_short=False, quote_known_at=end)
            verdict = validate_action(action, position, ctx, policy)
            emit(end, 'DECISION', action=action.kind.value, accepted=verdict.accepted, code=verdict.code)
            if verdict.accepted and action.kind not in (ActionType.WAIT, ActionType.HOLD):
                pending = (action, verdict.approved_qty, end)
    if pending is not None:
        emit(final, 'UNFILLED_END_OF_DATA', action=pending[0].kind.value)
    if position.side != Side.FLAT:
        emit(final, 'MARK_TO_MARKET_ONLY', unrealized_pnl=(float(f.close.iloc[-1])-position.entry_price)*position.qty)
    return pd.DataFrame(events), {'trades_closed': trades, 'cash_after_realized': cash,
        'equity_mark_to_market': equity_at(float(f.close.iloc[-1])),
        'position_side': position.side.value, 'events': len(events),
        'no_real_orders': True, 'data_start': first.isoformat(), 'data_end': final.isoformat()}


class HoldAgent:
    """Neutral proof-of-plumbing agent: WAIT when flat, HOLD when invested."""
    def __call__(self, decision_at, features, position, context):
        return Action(ActionType.WAIT if position.side == Side.FLAT else ActionType.HOLD)


class DemoAgent:
    """Synthetic time-based smoke test; not a predictive/trading strategy."""
    def __init__(self, first_decision: pd.Timestamp):
        self.first = pd.Timestamp(first_decision)
    def __call__(self, decision_at, features, position, context):
        if decision_at == self.first and position.side == Side.FLAT:
            return Action(ActionType.OPEN_LONG, sl=context.reference_price*0.995,
                          tp=context.reference_price*1.005)
        if decision_at == self.first + pd.Timedelta(minutes=5) and position.side != Side.FLAT:
            return Action(ActionType.CLOSE)
        return Action(ActionType.WAIT if position.side == Side.FLAT else ActionType.HOLD)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minutes', type=Path, default=Path('data/evaluation/v421-binance-1m/BTCUSDT/2024-01.parquet'))
    parser.add_argument('--observations', type=Path, default=Path('data/evaluation/v442-observation-audit/pilot_observations.parquet'))
    parser.add_argument('--schema', type=Path, default=Path('data/evaluation/v442-observation-audit/schema.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/evaluation/v460-agent-loop/BTCUSDT-2024-01'))
    parser.add_argument('--agent', choices=['hold', 'demo'], default='demo')
    args = parser.parse_args()
    paths = [args.output_dir / 'events.csv', args.output_dir / 'summary.json', args.output_dir / 'run_manifest.json']
    if any(x.exists() for x in paths): raise FileExistsError('refusing overwrite of an existing V4.6.0 result')
    schema = json.loads(args.schema.read_text(encoding='utf-8'))
    features = pd.read_parquet(args.observations)
    expected = schema['feature_columns']
    actual = [x for x in features.columns if x != 'decision_at']
    if actual != expected or len(actual) != 144: raise ValueError('pilot schema columns mismatch')
    digest = hashlib.sha256(json.dumps(actual).encode()).hexdigest()
    if digest != schema['feature_schema_sha256']: raise ValueError('pilot schema SHA256 mismatch')
    minutes = pd.read_parquet(args.minutes)
    first = pd.to_datetime(features.decision_at, utc=True).iloc[0]
    agent = HoldAgent() if args.agent == 'hold' else DemoAgent(first)
    events, summary = replay_agent(minutes, features, agent)
    manifest = {'version':'v4.6.0', 'agent':args.agent, 'feature_schema_sha256':digest,
                'minutes_source':str(args.minutes), 'observations_source':str(args.observations),
                'no_real_orders':True, 'no_training':True, 'no_test_evaluation':True}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with paths[0].open('x', newline='', encoding='utf-8') as fh: events.to_csv(fh, index=False)
    with paths[1].open('x', encoding='utf-8') as fh: json.dump(summary, fh, indent=2)
    with paths[2].open('x', encoding='utf-8') as fh: json.dump(manifest, fh, indent=2)
    print('V4.6.0 PASS', summary)
    print('Outputs:', *map(str, paths))
    print('No training, no Test, no live orders.')


if __name__ == '__main__': main()

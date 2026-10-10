"""V4.5.1 deterministic, long-only Spot 1m replay. No live orders, no Qwen.

Decisions at close of completed UTC 5m blocks; accepted orders eligible at
next 1m opening, never at the observation close. Gap in 1m data aborts.
"""
from __future__ import annotations
import argparse
import json
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


def replay(frame: pd.DataFrame, decisions: dict, *, config: ExecutionConfig = ExecutionConfig(),
           policy: RiskPolicy = RiskPolicy()) -> tuple[pd.DataFrame, dict]:
    """decisions: {aware UTC decision_at: Action}; timestamps are 1m opens.

    Unspecified 5m decisions imply WAIT/HOLD. No synthetic fills on missing minutes.
    Events capture accepted/refused proposals and cashflow. A live broker is absent.
    """
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
    d = {}
    for t, a in decisions.items():
        ts = pd.Timestamp(t)
        if ts.tzinfo is None: raise ValueError('naive decision time')
        ts = ts.tz_convert('UTC')
        if ts < first + pd.Timedelta(minutes=5) or ts > final or ts.minute % 5 or ts.second or ts.microsecond:
            raise ValueError('decision not on a completed 5m close')
        if not isinstance(a, Action): raise ValueError('invalid action proposal')
        d[ts] = a
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
            action = d.get(pd.Timestamp(end), Action(ActionType.WAIT if position.side == Side.FLAT else ActionType.HOLD))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/evaluation/v421-binance-1m/BTCUSDT/2024-01.parquet')
    parser.add_argument('--output-dir', default='data/evaluation/v451-simulator/BTCUSDT-2024-01')
    args = parser.parse_args()
    target = Path(args.output_dir)
    outlog = target/'events.csv'; outmeta=target/'summary.json'
    if outlog.exists() or outmeta.exists():
        raise FileExistsError('outputs already exist; refusing overwrite')
    f = pd.read_parquet(args.input)
    f = validate_minutes(f)
    first = f.timestamp.iloc[0]
    # Deterministic synthetic action proposals, never Qwen or a profit strategy.
    decisions = {
        first + pd.Timedelta(minutes=5): Action(ActionType.OPEN_LONG,
            sl=float(f.close.iloc[4])*0.995, tp=float(f.close.iloc[4])*1.005),
        first + pd.Timedelta(minutes=10): Action(ActionType.CLOSE),
    }
    # CLOSE can fail if stop or target has already exited before second decision.
    events, summary = replay(f, decisions)
    target.mkdir(parents=True, exist_ok=True)
    with outlog.open('x', newline='', encoding='utf-8') as fh: events.to_csv(fh, index=False)
    with outmeta.open('x', encoding='utf-8') as fh: json.dump(summary, fh, indent=2)
    print('V4.5.1 PASS', summary)
    print('Outputs:', outlog, outmeta)
    print('No training, no test, no live orders.')


if __name__ == '__main__': main()

"""V4.5.0: contracts and pure pre-trade risk validation; NO fills/trading.

All prices and quantities are illustrative proposals; open orders must be rechecked
using the actual later fill in a separate execution engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite, floor
from typing import Optional


class Side(str, Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


class ActionType(str, Enum):
    WAIT = "WAIT"
    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    MOVE_SL = "MOVE_SL"
    MOVE_TP = "MOVE_TP"
    MOVE_SL_TP = "MOVE_SL_TP"


@dataclass(frozen=True)
class Position:
    side: Side = Side.FLAT
    entry_price: Optional[float] = None
    qty: float = 0.0
    sl: Optional[float] = None
    tp: Optional[float] = None
    opened_at: Optional[datetime] = None


@dataclass(frozen=True)
class Action:
    kind: ActionType
    sl: Optional[float] = None
    tp: Optional[float] = None
    # Desired quantity is optional. Risk engine computes a strict maximum.
    qty: Optional[float] = None


@dataclass(frozen=True)
class MarketContext:
    decision_at: datetime
    reference_price: float
    spread_bps: float
    liquidity_notional: float  # estimated executable notional near reference price
    equity: float
    market_allows_short: bool = False
    # Used to refuse a stale snapshot, must be <= decision_at.
    quote_known_at: Optional[datetime] = None


@dataclass(frozen=True)
class RiskPolicy:
    max_risk_fraction: float = 0.005  # 0.5% equity (illustrative)
    max_notional_to_equity: float = 1.0
    max_spread_bps: float = 15.0
    max_liquidity_fraction: float = 0.01
    min_stop_distance_bps: float = 5.0
    min_target_distance_bps: float = 5.0
    assumed_roundtrip_cost_bps: float = 14.0
    qty_step: float = 0.000001
    max_quote_age_seconds: float = 300.0


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    code: str
    approved_qty: Optional[float] = None
    # Informational preview ONLY; never mutate positions here.
    action: Optional[Action] = None


def _utc(t: datetime) -> datetime:
    if not isinstance(t, datetime) or t.tzinfo is None or t.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return t.astimezone(timezone.utc)


def _positive(x: Optional[float]) -> bool:
    return x is not None and isinstance(x, (int, float)) and not isinstance(x, bool) and isfinite(x) and x > 0


def _nonnegative(x: float) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and isfinite(x) and x >= 0


def _check_policy(p: RiskPolicy) -> None:
    for key in ("max_risk_fraction", "max_notional_to_equity", "max_liquidity_fraction", "min_stop_distance_bps", "min_target_distance_bps", "qty_step"):
        if not _positive(getattr(p, key)):
            raise ValueError("invalid risk policy: " + key)
    for key in ("max_spread_bps", "assumed_roundtrip_cost_bps", "max_quote_age_seconds"):
        if not _nonnegative(getattr(p, key)):
            raise ValueError("invalid risk policy: " + key)
    if p.max_risk_fraction > 1 or p.max_liquidity_fraction > 1:
        raise ValueError("risk/liquidity fraction must be <= 1")


def _position_state(pos: Position) -> bool:
    if not isinstance(pos.side, Side):
        return False
    if pos.side is Side.FLAT:
        return pos.entry_price is None and pos.sl is None and pos.tp is None and pos.opened_at is None and pos.qty == 0
    if not (_positive(pos.entry_price) and _positive(pos.qty) and _positive(pos.sl) and _positive(pos.tp)):
        return False
    try:
        _utc(pos.opened_at)
    except (ValueError, TypeError):
        return False
    if pos.side is Side.LONG:
        return pos.sl < pos.entry_price < pos.tp
    return pos.tp < pos.entry_price < pos.sl


def validate_action(action: Action, pos: Position, ctx: MarketContext, policy: RiskPolicy = RiskPolicy()) -> Verdict:
    """Pure fail-closed checker, independent of Qwen and execution model."""
    def no(reason: str) -> Verdict:
        return Verdict(False, reason)

    _check_policy(policy)
    if not isinstance(action, Action) or not isinstance(action.kind, ActionType):
        return no("INVALID_ACTION")
    if not _position_state(pos):
        return no("INVALID_POSITION")
    try:
        now = _utc(ctx.decision_at)
        known = _utc(ctx.quote_known_at) if ctx.quote_known_at is not None else now
    except (TypeError, ValueError):
        return no("INVALID_TIMESTAMP")
    if known > now or (now - known).total_seconds() > policy.max_quote_age_seconds:
        return no("STALE_OR_FUTURE_QUOTE")
    if not (_positive(ctx.reference_price) and _positive(ctx.equity) and _nonnegative(ctx.spread_bps) and _nonnegative(ctx.liquidity_notional)):
        return no("INVALID_MARKET")
    if pos.side is not Side.FLAT and _utc(pos.opened_at) > now:
        return no("FUTURE_POSITION")
    if ctx.spread_bps > policy.max_spread_bps:
        return no("SPREAD_LIMIT")

    flat_ops = {ActionType.WAIT, ActionType.OPEN_LONG, ActionType.OPEN_SHORT}
    open_ops = {ActionType.HOLD, ActionType.CLOSE, ActionType.MOVE_SL, ActionType.MOVE_TP, ActionType.MOVE_SL_TP}
    if (pos.side is Side.FLAT and action.kind not in flat_ops) or (pos.side is not Side.FLAT and action.kind not in open_ops):
        return no("ILLEGAL_TRANSITION")

    expected = {
        ActionType.WAIT: set(), ActionType.HOLD: set(), ActionType.CLOSE: set(),
        ActionType.OPEN_LONG: {"sl", "tp"}, ActionType.OPEN_SHORT: {"sl", "tp"},
        ActionType.MOVE_SL: {"sl"}, ActionType.MOVE_TP: {"tp"},
        ActionType.MOVE_SL_TP: {"sl", "tp"},
    }[action.kind]
    submitted = {name for name in ("sl", "tp") if getattr(action, name) is not None}
    if submitted != expected:
        return no("INVALID_PARAMETERS")
    if action.qty is not None and (action.kind not in (ActionType.OPEN_LONG, ActionType.OPEN_SHORT) or not _positive(action.qty)):
        return no("INVALID_QUANTITY")
    for name in submitted:
        if not _positive(getattr(action, name)):
            return no("INVALID_PRICE")
    if action.kind in (ActionType.WAIT, ActionType.HOLD, ActionType.CLOSE):
        return Verdict(True, "ACCEPTED", action=action)

    px = ctx.reference_price
    if action.kind in (ActionType.OPEN_LONG, ActionType.OPEN_SHORT):
        side = Side.LONG if action.kind is ActionType.OPEN_LONG else Side.SHORT
        if side is Side.SHORT and not ctx.market_allows_short:
            return no("SHORT_NOT_SUPPORTED")
        if side is Side.LONG and not (action.sl < px < action.tp):
            return no("INVALID_PROTECTION_GEOMETRY")
        if side is Side.SHORT and not (action.tp < px < action.sl):
            return no("INVALID_PROTECTION_GEOMETRY")
        distance_bps = abs(px - action.sl) / px * 10000
        target_bps = abs(px - action.tp) / px * 10000
        if distance_bps < policy.min_stop_distance_bps or target_bps < policy.min_target_distance_bps:
            return no("MIN_DISTANCE")
        # Conservative risk includes cost estimate. Fill engine must recheck!
        risk_per_unit = abs(px - action.sl) + px * policy.assumed_roundtrip_cost_bps / 10000
        size_by_risk = ctx.equity * policy.max_risk_fraction / risk_per_unit
        size_by_exposure = ctx.equity * policy.max_notional_to_equity / px
        size_by_liquidity = ctx.liquidity_notional * policy.max_liquidity_fraction / px
        maximum = min(size_by_risk, size_by_exposure, size_by_liquidity)
        if action.qty is not None:
            maximum = min(maximum, action.qty)
        # Quantize DOWN only: never exceed constraints.
        qty = floor(maximum / policy.qty_step * (1 - 1e-12)) * policy.qty_step
        if qty <= 0 or not isfinite(qty):
            return no("NO_EXECUTABLE_SIZE")
        return Verdict(True, "ACCEPTED_PREVIEW_RECHECK_AT_FILL", approved_qty=qty, action=action)

    # Position modifications: protections evaluated relative to entry and current SL,
    # so even after a market move a stop cannot be widened.
    sl = action.sl if action.sl is not None else pos.sl
    tp = action.tp if action.tp is not None else pos.tp
    if pos.side is Side.LONG:
        if sl < pos.sl:
            return no("STOP_WIDENING")
        if sl >= tp or tp <= pos.entry_price:
            return no("INVALID_PROTECTION_GEOMETRY")
    else:
        if sl > pos.sl:
            return no("STOP_WIDENING")
        if sl <= tp or tp >= pos.entry_price:
            return no("INVALID_PROTECTION_GEOMETRY")
    # Orders crossing current market require simulator-side execution handling.
    if (pos.side is Side.LONG and (sl >= px or tp <= px)) or (pos.side is Side.SHORT and (sl <= px or tp >= px)):
        return no("PROTECTION_CROSSES_MARKET")
    return Verdict(True, "ACCEPTED_PREVIEW", action=action)


if __name__ == "__main__":
    print("V4.5.0 contract module ready. Run test_agent_actions_risk_v450.py; no orders executed.")

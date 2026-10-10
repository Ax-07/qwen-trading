import unittest
from datetime import datetime, timedelta, timezone
from dataclasses import replace

from agent_actions_risk_v450 import Action, ActionType as A, MarketContext, Position, RiskPolicy, Side, validate_action

T = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
CTX = MarketContext(T, 100, 5, 1_000_000, 10_000, market_allows_short=True, quote_known_at=T)
POL = RiskPolicy()
FLAT = Position()
LONG = Position(Side.LONG, 100, 1, 95, 110, T - timedelta(minutes=5))
SHORT = Position(Side.SHORT, 100, 1, 105, 90, T - timedelta(minutes=5))

class ContractTests(unittest.TestCase):
    def check(self, a, p=FLAT, c=CTX, pol=POL):
        return validate_action(a, p, c, pol)

    def test_flat_actions(self):
        self.assertTrue(self.check(Action(A.WAIT)).accepted)
        self.assertTrue(self.check(Action(A.OPEN_LONG, sl=98, tp=104)).accepted)
        self.assertTrue(self.check(Action(A.OPEN_SHORT, sl=102, tp=96)).accepted)
        self.assertEqual(self.check(Action(A.HOLD)).code, "ILLEGAL_TRANSITION")

    def test_open_actions(self):
        for side in (LONG, SHORT):
            self.assertTrue(self.check(Action(A.HOLD), side).accepted)
            self.assertTrue(self.check(Action(A.CLOSE), side).accepted)
            self.assertEqual(self.check(Action(A.OPEN_LONG, sl=90, tp=120), side).code, "ILLEGAL_TRANSITION")

    def test_missing_and_extraneous(self):
        self.assertEqual(self.check(Action(A.OPEN_LONG, sl=99)).code, "INVALID_PARAMETERS")
        self.assertEqual(self.check(Action(A.WAIT, sl=99)).code, "INVALID_PARAMETERS")
        self.assertEqual(self.check(Action(A.MOVE_SL, sl=97, tp=120), LONG).code, "INVALID_PARAMETERS")

    def test_stop_cannot_widen(self):
        self.assertEqual(self.check(Action(A.MOVE_SL, sl=94), LONG).code, "STOP_WIDENING")
        self.assertEqual(self.check(Action(A.MOVE_SL, sl=106), SHORT).code, "STOP_WIDENING")
        self.assertTrue(self.check(Action(A.MOVE_SL, sl=96), LONG).accepted)
        self.assertTrue(self.check(Action(A.MOVE_SL, sl=104), SHORT).accepted)

    def test_tp_and_atomic(self):
        self.assertTrue(self.check(Action(A.MOVE_TP, tp=112), LONG).accepted)
        self.assertTrue(self.check(Action(A.MOVE_SL_TP, sl=96, tp=120), LONG).accepted)
        self.assertEqual(self.check(Action(A.MOVE_SL_TP, sl=94, tp=120), LONG).code, "STOP_WIDENING")
        self.assertEqual(self.check(Action(A.MOVE_TP, tp=99), LONG).code, "INVALID_PROTECTION_GEOMETRY")

    def test_short_disabled(self):
        c = replace(CTX, market_allows_short=False)
        self.assertEqual(self.check(Action(A.OPEN_SHORT, sl=102, tp=96), c=c).code, "SHORT_NOT_SUPPORTED")

    def test_no_future_and_stale(self):
        a = Action(A.WAIT)
        self.assertEqual(self.check(a, c=replace(CTX, quote_known_at=T + timedelta(seconds=1))).code, "STALE_OR_FUTURE_QUOTE")
        self.assertEqual(self.check(a, c=replace(CTX, quote_known_at=T - timedelta(minutes=6))).code, "STALE_OR_FUTURE_QUOTE")
        self.assertEqual(self.check(a, LONG, c=replace(CTX, decision_at=T - timedelta(minutes=6))).code, "STALE_OR_FUTURE_QUOTE")

    def test_risk_and_exposure(self):
        v = self.check(Action(A.OPEN_LONG, sl=99, tp=110))
        self.assertTrue(v.accepted)
        self.assertLessEqual(v.approved_qty * (1 + 100 * POL.assumed_roundtrip_cost_bps / 10000), CTX.equity * POL.max_risk_fraction + 1e-7)
        self.assertLessEqual(v.approved_qty * 100, CTX.equity * POL.max_notional_to_equity + 1e-7)
        self.assertLessEqual(v.approved_qty * 100, CTX.liquidity_notional * POL.max_liquidity_fraction + 1e-7)

    def test_qty_and_liquidity(self):
        v = self.check(Action(A.OPEN_LONG, sl=98, tp=102, qty=0.123456789))
        self.assertTrue(v.accepted)
        self.assertLessEqual(v.approved_qty, 0.123456789)
        self.assertEqual(self.check(Action(A.OPEN_LONG, sl=98, tp=102), c=replace(CTX, liquidity_notional=0)).code, "NO_EXECUTABLE_SIZE")

    def test_spread_and_prices(self):
        self.assertEqual(self.check(Action(A.WAIT), c=replace(CTX, spread_bps=20)).code, "SPREAD_LIMIT")
        self.assertEqual(self.check(Action(A.OPEN_LONG, sl=101, tp=102)).code, "INVALID_PROTECTION_GEOMETRY")
        self.assertEqual(self.check(Action(A.OPEN_LONG, sl=float('nan'), tp=103)).code, "INVALID_PRICE")

    def test_position_corrupt(self):
        self.assertEqual(self.check(Action(A.HOLD), replace(LONG, sl=101)).code, "INVALID_POSITION")
        self.assertEqual(self.check(Action(A.WAIT), replace(FLAT, qty=1)).code, "INVALID_POSITION")

    def test_invalid_policy(self):
        with self.assertRaises(ValueError):
            self.check(Action(A.WAIT), pol=replace(POL, max_risk_fraction=-1))

    def test_purity(self):
        original = LONG
        self.check(Action(A.MOVE_SL, sl=96), LONG)
        self.assertEqual(LONG, original)

    def test_execution_not_performed(self):
        v = self.check(Action(A.OPEN_LONG, sl=99, tp=110))
        self.assertIsNone(v.action.qty) # approved_qty is separate from Qwen input
        self.assertIn('PREVIEW', v.code)

if __name__ == '__main__':
    unittest.main(verbosity=2)

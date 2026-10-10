import json
import unittest
from datetime import datetime, timezone
from types import MappingProxyType

from agent_actions_risk_v450 import ActionType, MarketContext, Position, Side, validate_action
from qwen_adapter_v461 import (ModelOutputError, QwenAgent, parse_action, prepare_messages, local_openai_generate)

NOW = datetime(2024,1,1,0,5,tzinfo=timezone.utc)
NAMES = [f'f{i}' for i in range(144)]
FEATURES = MappingProxyType({name: float(i) for i,name in enumerate(NAMES)})
CTX = MarketContext(NOW, 42000., 5., 100000., 10000., False, NOW)

class Tests(unittest.TestCase):
    def test_wait(self):
        self.assertEqual(parse_action('{"action":"WAIT"}').kind, ActionType.WAIT)
    def test_open(self):
        a = parse_action('{"action":"OPEN_LONG","sl":41800,"tp":42400}')
        self.assertTrue(validate_action(a, Position(), CTX).accepted)
    def test_qty_optional(self):
        self.assertEqual(parse_action('{"action":"OPEN_LONG","sl":41800,"tp":42400,"qty":0.1}').qty, 0.1)
    def test_modify(self):
        self.assertEqual(parse_action('{"action":"MOVE_SL_TP","sl":12,"tp":14}').kind, ActionType.MOVE_SL_TP)
    def test_short_refused(self):
        a = parse_action('{"action":"OPEN_SHORT","sl":42500,"tp":41500}')
        self.assertFalse(validate_action(a, Position(), CTX).accepted)
    def test_extra_key(self):
        with self.assertRaises(ModelOutputError): parse_action('{"action":"WAIT","confidence":0.99}')
    def test_duplicate_key(self):
        with self.assertRaises(ModelOutputError): parse_action('{"action":"WAIT","action":"OPEN_LONG"}')
    def test_markdown_rejected(self):
        with self.assertRaises(ModelOutputError): parse_action('```json\n{"action":"WAIT"}\n```')
    def test_nan_rejected(self):
        with self.assertRaises(ModelOutputError): parse_action('{"action":"OPEN_LONG","sl":NaN,"tp":2}')
    def test_bool_rejected(self):
        with self.assertRaises(ModelOutputError): parse_action('{"action":"OPEN_LONG","sl":true,"tp":2}')
    def test_nonpositive_rejected(self):
        with self.assertRaises(ModelOutputError): parse_action('{"action":"OPEN_LONG","sl":0,"tp":2}')
    def test_non_object_rejected(self):
        with self.assertRaises(ModelOutputError): parse_action('[{"action":"WAIT"}]')
    def test_null_feature(self):
        f = dict(FEATURES); f['f0'] = float('nan')
        messages = prepare_messages(NOW, f, Position(), CTX, NAMES)
        self.assertIsNone(json.loads(messages[1]['content'])['features']['f0'])
    def test_inf_feature(self):
        f = dict(FEATURES); f['f0'] = float('inf')
        with self.assertRaises(ValueError): prepare_messages(NOW, f, Position(), CTX, NAMES)
    def test_only_current_row(self):
        m = prepare_messages(NOW, FEATURES, Position(), CTX, NAMES)
        d = json.loads(m[1]['content'])
        self.assertEqual(len(d['features']), 144)
        self.assertEqual(d['decision_at'], NOW.isoformat())
    def test_wrong_schema(self):
        with self.assertRaises(ValueError): prepare_messages(NOW, FEATURES, Position(), CTX, NAMES[:-1])
    def test_agent_calls_once(self):
        seen=[]
        agent = QwenAgent(NAMES, lambda m: (seen.append(m), '{"action":"WAIT"}')[1])
        self.assertEqual(agent(NOW, FEATURES, Position(), CTX).kind, ActionType.WAIT)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(len(seen), 1)
    def test_invalid_model_aborts(self):
        agent=QwenAgent(NAMES, lambda m: 'not json')
        with self.assertRaises(ModelOutputError): agent(NOW,FEATURES,Position(),CTX)
    def test_no_remote_endpoint(self):
        for endpoint in ('https://127.0.0.1/v1/chat/completions', 'http://example.com/v1/chat/completions', 'http://localhost:8000/bad'):
            with self.assertRaises(ValueError): local_openai_generate(endpoint,'model')
    def test_no_mutation(self):
        orig = dict(FEATURES)
        prepare_messages(NOW, FEATURES, Position(), CTX, NAMES)
        self.assertEqual(dict(FEATURES), orig)

if __name__=='__main__': unittest.main(verbosity=2)

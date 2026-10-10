import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import qwen_json_diagnostics_v464 as m

ORIGINAL=[{'role':'system','content':'Rules'},{'role':'user','content':'{\"features\":{}}'}]

class Tests(unittest.TestCase):
    def test_baseline_unchanged(self):
        self.assertEqual(m.trial_messages(ORIGINAL,'baseline'),ORIGINAL)
    def test_strict_extra(self):
        self.assertIn('JSON',m.trial_messages(ORIGINAL,'explicit')[0]['content'])
    def test_repair_exact(self):
        self.assertIn('{"action":"WAIT"}',m.trial_messages(ORIGINAL,'repair_prompt')[0]['content'])
    def test_no_mutation(self):
        m.trial_messages(ORIGINAL,'explicit');self.assertEqual(ORIGINAL[0]['content'],'Rules')
    def test_bad_variant(self):
        with self.assertRaises(ValueError):m.trial_messages(ORIGINAL,'evil')
    def test_bad_message_roles(self):
        with self.assertRaises(ValueError):m.trial_messages([{'role':'user','content':'x'}],'baseline')
    def test_escape(self):
        self.assertEqual(m.safe_raw_preview('a\n\x1bb'), '"a\\n\\u001bb"')
    def test_truncate(self):
        self.assertIn('[TRUNCATED]',m.safe_raw_preview('a'*520))
    def test_invalid_preview_limit(self):
        with self.assertRaises(ValueError):m.safe_raw_preview('abc',2000)
    def test_good_wait_and_bad_fields(self):
        from agent_actions_risk_v450 import MarketContext
        from datetime import datetime, timezone
        t=datetime(2024,1,1,tzinfo=timezone.utc)
        context=MarketContext(t,100.,5.,100000.,10000.,market_allows_short=False,quote_known_at=t)
        def gen(msgs, tok, mod, n):
            if 'serialization test' in msgs[0]['content']:
                return '{"action":"WAIT"}', 0.2, 10, 6
            return '{"action":"WAIT","reason":"test"}',0.1,10,8
        results=m.run_trials(ORIGINAL,context,None,None,['baseline','repair_prompt'],generator=gen)
        self.assertFalse(results[0]['json_valid']);self.assertTrue(results[1]['json_valid'])
        self.assertTrue(results[1]['risk_accepted'])
    def test_duplicate_variants_refused(self):
        with self.assertRaises(ValueError):m.run_trials(ORIGINAL,None,None,None,['baseline','baseline'])
    def test_cache_mode_no_cuda(self):
        with patch.object(m.base,'verify_local_snapshot',return_value=Path('/tmp/model')),patch.object(m.base,'inspect_cuda_memory') as gpu:
            self.assertEqual(m.main(['--mode','check-cache']),0)
            gpu.assert_not_called()
    def test_generation_cap(self):
        with patch.object(m.base,'verify_local_snapshot',return_value=Path('/tmp/model')):
            with self.assertRaises(SystemExit):m.main(['--max-new-tokens','300'])
    def test_offline_set(self):
        self.assertEqual(m.base.os.environ['HF_HUB_OFFLINE'],'1')
    def test_no_save_code(self):
        src=Path(m.__file__).read_text()
        self.assertNotIn('write_text(',src)
        self.assertNotIn('open(',src)
    def test_chat_template_flags(self):
        import inspect
        src=inspect.getsource(m.tokenize_chat)
        self.assertIn('enable_thinking=False',src)
        self.assertIn('tokenize=True',src)
    def test_single_load(self):
        import inspect
        self.assertEqual(inspect.getsource(m.main).count('base.load_local_model('),1)

if __name__=='__main__':unittest.main(verbosity=2)

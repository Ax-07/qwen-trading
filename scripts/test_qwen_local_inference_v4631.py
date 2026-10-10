import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import qwen_local_inference_v4631 as m


class Tests(unittest.TestCase):
    def test_default_is_check_cache(self):
        with patch.object(m, 'verify_local_snapshot', return_value=Path('/tmp/snapshot')):
            self.assertEqual(m.main([]), 0)

    def test_offline_flags(self):
        self.assertEqual(m.os.environ['HF_HUB_OFFLINE'], '1')
        self.assertEqual(m.os.environ['TRANSFORMERS_OFFLINE'], '1')

    def test_missing_cache(self):
        with patch.object(m, 'verify_local_snapshot', side_effect=FileNotFoundError('missing')):
            self.assertEqual(m.main([]), 2)

    def test_no_model_load_in_cache_check(self):
        with patch.object(m, 'verify_local_snapshot', return_value=Path('/tmp/snapshot')), patch.object(m, 'load_local_model') as fn:
            m.main([])
            fn.assert_not_called()

    def test_bad_generation_length(self):
        with self.assertRaises(SystemExit):
            m.main(['--max-new-tokens','257'])

    def test_legacy_lora_not_used(self):
        self.assertNotIn('final_adapter', m.MODEL_ID)

    def test_action_valid_wait(self):
        from agent_actions_risk_v450 import MarketContext
        from datetime import datetime, timezone
        t=datetime(2024,1,1,tzinfo=timezone.utc)
        ctx=MarketContext(t, 100., 5., 100000., 10000., market_allows_short=False, quote_known_at=t)
        out=m.interpret_answer('{"action":"WAIT"}',ctx)
        self.assertTrue(out['json_valid'])
        self.assertTrue(out['risk_accepted'])

    def test_action_bad_prose(self):
        from agent_actions_risk_v450 import MarketContext
        from datetime import datetime, timezone
        t=datetime(2024,1,1,tzinfo=timezone.utc)
        ctx=MarketContext(t, 100., 5., 100000., 10000., market_allows_short=False, quote_known_at=t)
        self.assertFalse(m.interpret_answer('WAIT',ctx)['json_valid'])

    def test_no_output_files_code(self):
        self.assertNotIn('write_text(', Path(m.__file__).read_text())

    def test_budget_from_actual_free(self):
        self.assertEqual(m.memory_budget(5869*2**20, 8192*2**20, 768, 7), (5869-768)*2**20)
        self.assertEqual(m.memory_budget(12113*2**20, 12288*2**20, 1024, 11), (12113-1024)*2**20)

    def test_budget_bounded_by_cap(self):
        self.assertEqual(m.memory_budget(30*2**30, 32*2**30, 1024, 11), 11*2**30)

    def test_budget_rejects_low_free(self):
        with self.assertRaises(RuntimeError):
            m.memory_budget(2500*2**20, 8192*2**20, 768, 7)

    def test_budget_rejects_corrupt_measurement(self):
        for free, total in ((-1,10), (100,99), (0,0), (True,10)):
            with self.assertRaises(ValueError):
                m.memory_budget(free, total)

    def test_explicit_memory_map_code(self):
        source = Path(m.__file__).read_text()
        self.assertIn('inspect_cuda_memory(torch)', source)
        self.assertIn("'device_map': 'auto'", source)
        self.assertIn('max_memory=budgets', source)
        self.assertNotIn('llm_int8_enable_fp32_cpu_offload', source)



if __name__ == '__main__':
    unittest.main(verbosity=2)

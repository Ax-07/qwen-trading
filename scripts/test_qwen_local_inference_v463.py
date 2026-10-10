import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import qwen_local_inference_v463 as m


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


if __name__ == '__main__':
    unittest.main(verbosity=2)

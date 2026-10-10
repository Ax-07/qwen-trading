import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from diagnose_qwen_runtime_v462 import inspect_config, scan_checkpoints, default_roots, diagnose


class Tests(unittest.TestCase):
    def test_model_config(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'config.json').write_text(json.dumps({'model_type':'qwen3_5','architectures':['Qwen3_5ForCausalLM'], 'quantization_config': {'quant_method': 'bitsandbytes'}}))
            x=inspect_config(p/'config.json')
            self.assertEqual(x['model_type'], 'qwen3_5')
            self.assertEqual(x['quantization_method'], 'bitsandbytes')
    def test_adapter_no_sensitive_base_path(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'adapter_config.json'
            p.write_text(json.dumps({'peft_type':'LORA','base_model_name_or_path':'SECRET_LOCAL_PATH'}))
            self.assertNotIn('SECRET_LOCAL_PATH', json.dumps(inspect_config(p)))
    def test_bad_json_safe(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';p.write_text('broken')
            self.assertEqual(inspect_config(p)['error'], 'JSONDecodeError')
    def test_scan_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); nested=p/'a'/'b'; nested.mkdir(parents=True)
            (nested/'config.json').write_text('{}')
            self.assertEqual(len(scan_checkpoints([p], max_depth=1)[0]),0)
            self.assertEqual(len(scan_checkpoints([p], max_depth=2)[0]),1)
    def test_missing_root(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)/'missing';found,info=scan_checkpoints([r]);self.assertEqual(found,[]);self.assertEqual(info['missing_roots'],[str(r)])
    def test_symlink_not_followed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'real').mkdir();(p/'real'/'config.json').write_text('{}')
            try:(p/'link').symlink_to(p/'real', target_is_directory=True)
            except (OSError,NotImplementedError):self.skipTest('symlinks not allowed')
            x,_=scan_checkpoints([p]);self.assertEqual(len(x),1)
    def test_env_paths(self):
        with tempfile.TemporaryDirectory() as d:
            x=default_roots(Path(d));self.assertIn(Path(d)/'models',x)
    def test_no_gpu_invocation(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('diagnose_qwen_runtime_v462.default_roots',return_value=[]):
                x=diagnose(Path(d),include_gpu=False)
            self.assertTrue(x['read_only']);self.assertEqual(x['gpu'],{'skipped':True})

if __name__=='__main__':unittest.main(verbosity=2)

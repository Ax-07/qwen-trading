import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import qwen_local_inference_v4633 as m

class Tests(unittest.TestCase):
    def fake_torch(self, name='NVIDIA GeForce RTX 3060', count=1, free_mib=11200):
        t=MagicMock()
        t.cuda.is_available.return_value=True
        t.cuda.device_count.return_value=count
        t.cuda.get_device_properties.return_value=MagicMock(name=name,total_memory=12*2**30)
        t.cuda.get_device_properties.return_value.name=name
        t.cuda.get_device_properties.return_value.total_memory=12*2**30
        t.cuda.mem_get_info.return_value=(free_mib*2**20, 12*2**30)
        return t
    def test_correct_card(self):
        self.assertEqual(m.inspect_cuda_memory(self.fake_torch()), {0: min(11200-1024,11*1024)*2**20})
    def test_two_visible_refused(self):
        with self.assertRaises(RuntimeError): m.inspect_cuda_memory(self.fake_torch(count=2))
    def test_ti_refused(self):
        with self.assertRaises(RuntimeError): m.inspect_cuda_memory(self.fake_torch(name='NVIDIA GeForce RTX 3060 Ti'))
    def test_low_memory_refused(self):
        with self.assertRaises(RuntimeError): m.inspect_cuda_memory(self.fake_torch(free_mib=2000))
    def test_cuda_off_refused(self):
        t=self.fake_torch();t.cuda.is_available.return_value=False
        with self.assertRaises(RuntimeError): m.inspect_cuda_memory(t)
    def test_single_explicit_map(self):
        s=Path(m.__file__).read_text()
        self.assertIn("'device_map': {'': 0}",s)
        self.assertNotIn("'device_map': 'auto'",s)
        self.assertNotIn('llm_int8_enable_fp32_cpu_offload',s)
    def test_check_cache_no_gpu(self):
        with patch.object(m,'verify_local_snapshot',return_value=Path('/tmp/model')), patch.object(m,'inspect_cuda_memory') as f:
            self.assertEqual(m.main(['--mode','check-cache']),0)
            f.assert_not_called()
    def test_offline(self):
        self.assertEqual(m.os.environ['HF_HUB_OFFLINE'],'1')
    def test_no_legacy_lora(self):
        self.assertNotIn('final_adapter',m.MODEL_ID)
    def test_wait(self):
        from agent_actions_risk_v450 import MarketContext
        from datetime import datetime,timezone
        t=datetime(2024,1,1,tzinfo=timezone.utc)
        c=MarketContext(t,100.,5.,100000.,10000.,market_allows_short=False,quote_known_at=t)
        self.assertTrue(m.interpret_answer('{"action":"WAIT"}',c)['risk_accepted'])
    def test_mapping_absent_valid_cuda(self):
        model=MagicMock()
        model.hf_device_map=None
        tensor=MagicMock(); tensor.numel.return_value=10
        tensor.device.type='cuda'; tensor.device.index=0
        model.named_parameters.return_value=[('weight',tensor)]
        model.named_buffers.return_value=[]
        m.verify_model_on_single_cuda(model)

    def test_mapping_absent_cpu_refused(self):
        model=MagicMock(); model.hf_device_map=None
        tensor=MagicMock(); tensor.numel.return_value=10
        tensor.device.type='cpu'; tensor.device.index=None
        model.named_parameters.return_value=[('weight',tensor)]
        model.named_buffers.return_value=[]
        with self.assertRaisesRegex(RuntimeError,'expected cuda:0'):
            m.verify_model_on_single_cuda(model)

    def test_mapping_second_gpu_refused(self):
        model=MagicMock(); model.hf_device_map={'': 'cuda:1'}
        with self.assertRaisesRegex(RuntimeError,'Unsafe device'):
            m.verify_model_on_single_cuda(model)

    def test_meta_tensor_refused(self):
        model=MagicMock(); model.hf_device_map=None
        tensor=MagicMock(); tensor.numel.return_value=10
        tensor.device.type='meta'; tensor.device.index=None
        model.named_parameters.return_value=[('weight',tensor)]
        model.named_buffers.return_value=[]
        with self.assertRaises(RuntimeError): m.verify_model_on_single_cuda(model)

    def test_empty_model_refused(self):
        model=MagicMock(); model.hf_device_map=None
        model.named_parameters.return_value=[]
        model.named_buffers.return_value=[]
        with self.assertRaises(RuntimeError): m.verify_model_on_single_cuda(model)

    def test_gpu_buffer_placement_checked(self):
        model=MagicMock(); model.hf_device_map={'': 0}
        model.named_parameters.return_value=[]
        tensor=MagicMock(); tensor.numel.return_value=5
        tensor.device.type='cpu'; tensor.device.index=None
        model.named_buffers.return_value=[('mask',tensor)]
        with self.assertRaises(RuntimeError): m.verify_model_on_single_cuda(model)

    def test_no_saved_generation(self):
        self.assertNotIn('write_text(',Path(m.__file__).read_text())

if __name__=='__main__': unittest.main(verbosity=2)

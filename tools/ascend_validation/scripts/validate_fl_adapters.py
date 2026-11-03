"""Numerical checks for the project-local FL Ascend compatibility paths."""
from pathlib import Path
from types import SimpleNamespace
import json
import torch
import torch_npu
import fp32_attention as adapter
root = Path(__file__).resolve().parents[1]
torch.npu.set_device(0)
adapter.self_test(root)
results = {}
for neox in (True, False):
    torch.manual_seed(42)
    q, k = torch.randn(5, 28), torch.randn(5, 14)
    cache = torch.randn(20, 4)
    pos = torch.tensor([0, 2, 5, 7, 11])
    expected = adapter.torch_rotary(pos, q, k, cache, 7, 4, neox)
    actual = adapter.torch_rotary(pos.npu(), q.npu(), k.npu(), cache.npu(), 7, 4, neox)
    error = max((a.cpu() - b).abs().max().item() for a,b in zip(actual,expected))
    assert error < 1e-5, error
    results['rope_neox_' + str(neox)] = error
obj = SimpleNamespace(pcp_world_size=1,dcp_world_size=1,block_size=4,
    block_table=SimpleNamespace(gpu=torch.tensor([[2,0,3],[1,4,5]],device='npu')),
    slot_mapping=SimpleNamespace(gpu=torch.empty(12,dtype=torch.int64,device='npu')))
adapter.torch_slot_mapping(obj,2,torch.tensor([0,5,8],device='npu'),torch.tensor([0,1,2,3,4,3,4,5],device='npu'))
assert obj.slot_mapping.gpu.cpu().tolist() == [8,9,10,11,0,7,16,17,-1,-1,-1,-1]
results['slot_mapping'] = 'passed: multiple sequences, block boundary, padding'
(root/'reports/fl_adapters.json').write_text(json.dumps(results,indent=2))
print('FL_ADAPTER_TESTS_PASS',results)

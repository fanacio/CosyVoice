"""Validate packed multi-sequence FP32 attention against independent CPU cases."""
import argparse
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root.resolve()
    previous_argv = sys.argv
    sys.argv = ['fp32_attention.py']
    try:
        module = runpy.run_path(str(root / 'scripts/fp32_attention.py'), run_name='__main__')
    finally:
        sys.argv = previous_argv
    attention = module['fp32_paged_attention']
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    torch.manual_seed(42)
    h, kvh, dim, block = 14, 2, 64, 128
    cache_k = torch.zeros(16, block, kvh, dim, device='npu:0', dtype=torch.float32)
    cache_v = torch.zeros_like(cache_k)
    tables = torch.tensor([[7,2,15,4],[0,11,9,13],[5,8,3,10],[14,6,12,1]], dtype=torch.int32, device='npu:0')
    keys, values, cases = {i:[] for i in range(4)}, {i:[] for i in range(4)}, []
    for label, sequence_ids, counts in [('prefill', [0,1,2,3], [130,7,1,19]),
                                         ('decode', [0,1,2,3], [1,1,1,1]),
                                         ('mixed_chunk', [0,1,2,3], [1,9,3,1]),
                                         ('reorder_and_remove', [3,1], [7,1])]:
        queries, new_keys, new_values, slots, lengths, ends, references = [], [], [], [], [], [], []
        for sequence_id, count in zip(sequence_ids, counts):
            q = torch.randn(count,h,dim)
            k = torch.randn(count,kvh,dim)
            v = torch.randn_like(k)
            old = sum(x.shape[0] for x in keys[sequence_id])
            keys[sequence_id].append(k); values[sequence_id].append(v)
            length = old + count
            positions = torch.arange(old, length, device='npu:0')
            physical = tables[sequence_id].long()[positions // block] * block + positions % block
            queries.append(q);new_keys.append(k);new_values.append(v);slots.append(physical)
            lengths.append(length);ends.append(sum(counts[:len(ends)+1]))
            all_k = torch.cat(keys[sequence_id]).transpose(0,1).repeat_interleave(h//kvh, dim=0)
            all_v = torch.cat(values[sequence_id]).transpose(0,1).repeat_interleave(h//kvh, dim=0)
            mask = torch.arange(length)[None] > (torch.arange(count) + old)[:,None]
            score = q.transpose(0,1) @ all_k.transpose(-1,-2) * dim**-0.5
            references.append((score.masked_fill(mask[None],-torch.inf).softmax(-1) @ all_v).transpose(0,1))
        metadata = SimpleNamespace(num_actual_tokens=sum(counts), actual_seq_lengths_q=ends,
                                   seq_lens_list=lengths, slot_mapping=torch.cat(slots),
                                   block_tables=tables[sequence_ids])
        output = torch.empty(sum(counts)+3,h,dim,device='npu:0')
        attention(torch.cat(queries).to('npu:0'),torch.cat(new_keys).to('npu:0'),
                  torch.cat(new_values).to('npu:0'),cache_k,cache_v,metadata,h,kvh,dim,dim**-0.5,output)
        actual = output[:sum(counts)].cpu()
        expected = torch.cat(references)
        error = (actual-expected).abs().max().item()
        assert torch.allclose(actual,expected,atol=1e-4,rtol=1e-4), (label,error)
        assert output[sum(counts):].abs().max().item()==0
        cases.append({'case':label,'sequence_ids':sequence_ids,'query_lengths':counts,
                      'context_lengths':lengths,'max_absolute_error':error,'dtype':str(output.dtype)})
    report = {'passed':True,'cases':cases,'attention':{k:v for k,v in module['_STATS'].items() if k!='dtypes'},
              'checks':'per-sequence causal offsets, GQA, disjoint paged caches, mixed prefill/decode, reordered batch, padded output'}
    (root/'reports/fp32_batched_attention_validation.json').write_text(json.dumps(report,indent=2))
    print('FP32_BATCHED_ATTENTION_PASS',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()

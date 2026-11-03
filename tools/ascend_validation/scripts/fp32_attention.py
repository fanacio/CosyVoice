"""Project-local strict FP32 attention for TP=1/eager CosyVoice3.

FIA rejects FP32 on this runtime. Keep vLLM's scheduler and paged cache,
and execute attention with NPU float32 matmul/softmax, without downcasting.
"""
from collections import Counter
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

_GRAPH_REPLAYS = 0
_original_graph_replay = torch.npu.NPUGraph.replay
def _counted_graph_replay(self, *args, **kwargs):
    global _GRAPH_REPLAYS
    result = _original_graph_replay(self, *args, **kwargs)
    _GRAPH_REPLAYS += 1
    return result
torch.npu.NPUGraph.replay = _counted_graph_replay

_STATS = {"calls": 0, "prefill_calls": 0, "decode_calls": 0, "batched_calls": 0, "max_sequences": 0, "dtypes": set()}


@torch.inference_mode()
def fp32_paged_attention(query, key, value, key_cache, value_cache, metadata,
                         num_heads, num_kv_heads, head_size, scale, output):
    tensors = [query, key_cache, value_cache, output]
    tensors.extend(t for t in [key, value] if t is not None)
    if any(t.dtype != torch.float32 or t.device.type != "npu" for t in tensors):
        raise RuntimeError("Strict FP32 requires NPU float32 Q/K/V, cache and output")
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        raise NotImplementedError("FP32 adapter requires contiguous ND KV layout")
    count = int(metadata.num_actual_tokens)
    ends, lengths = list(metadata.actual_seq_lengths_q), list(metadata.seq_lens_list)
    if not lengths or len(lengths) != len(ends) or ends[-1] != count:
        raise RuntimeError("Invalid packed sequence metadata")
    if any(end <= start for start, end in zip([0] + ends[:-1], ends)):
        raise RuntimeError("Empty or unordered query sequence")
    keys_flat = key_cache.view(-1, num_kv_heads, head_size)
    values_flat = value_cache.view(-1, num_kv_heads, head_size)
    if key is not None:
        slots = metadata.slot_mapping[:count].to(device=query.device, dtype=torch.int64)
        keys_flat.index_copy_(0, slots, key.reshape(-1, num_kv_heads, head_size)[:count])
        values_flat.index_copy_(0, slots, value.reshape(-1, num_kv_heads, head_size)[:count])
    output.zero_()
    start = 0
    for sequence, (end, length) in enumerate(zip(ends, lengths)):
        end, length = int(end), int(length)
        queries = end - start
        if queries > length:
            raise RuntimeError("Query is longer than its sequence")
        block_size = key_cache.shape[1]
        positions = torch.arange(length, device=query.device, dtype=torch.int64)
        table = metadata.block_tables[sequence].to(torch.int64)
        physical = table.index_select(0, positions // block_size) * block_size + positions % block_size
        cached_k = keys_flat.index_select(0, physical)
        cached_v = values_flat.index_select(0, physical)
        repeat = num_heads // num_kv_heads
        q = query[start:end].reshape(queries, num_heads, head_size).transpose(0, 1)
        k = cached_k.transpose(0, 1).repeat_interleave(repeat, dim=0)
        v = cached_v.transpose(0, 1).repeat_interleave(repeat, dim=0)
        query_positions = torch.arange(queries, device=query.device, dtype=torch.int64) + length - queries
        causal_mask = positions[None, :] > query_positions[:, None]
        with torch.autocast(device_type="npu", enabled=False):
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale
            scores = scores.masked_fill(causal_mask[None], float("-inf"))
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
            attended = torch.matmul(probabilities, v)
        output.reshape(-1, num_heads, head_size)[start:end].copy_(attended.transpose(0, 1))
        start = end
    _STATS["batched_calls"] += int(len(lengths) > 1)
    _STATS["max_sequences"] = max(_STATS["max_sequences"], len(lengths))
    _STATS["calls"] += 1
    _STATS["prefill_calls" if any(e - b > 1 for b, e in zip([0] + ends[:-1], ends)) else "decode_calls"] += 1
    _STATS["dtypes"].update(str(t.dtype) for t in tensors + [scores, probabilities, attended])
    return output


def torch_rotary(positions, query, key, cos_sin_cache, head_size, rotary_dim, is_neox_style):
    count = positions.numel()
    cache = cos_sin_cache.index_select(0, positions.flatten().long())
    cos, sin = cache.chunk(2, dim=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    results = []
    for tensor in (query, key):
        heads = tensor.reshape(count, -1, head_size)
        rotated, passthrough = heads[..., :rotary_dim], heads[..., rotary_dim:]
        if is_neox_style:
            first, second = rotated.chunk(2, dim=-1)
            result = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
        else:
            first, second = rotated[..., ::2], rotated[..., 1::2]
            result = torch.stack((first * cos - second * sin, second * cos + first * sin), dim=-1).flatten(-2)
        results.append(torch.cat((result, passthrough), dim=-1).reshape(tensor.shape))
    return tuple(results)


def torch_slot_mapping(self, num_reqs, query_start_loc, positions):
    if self.pcp_world_size * self.dcp_world_size != 1:
        raise NotImplementedError("CosyVoice slot mapping requires CP=1")
    starts = query_start_loc[:num_reqs + 1].long()
    indices = torch.arange(positions.numel(), device=positions.device)
    valid = indices < starts[-1]
    rows = torch.bucketize(indices, starts[1:], right=True).clamp(max=num_reqs - 1)
    blocks = positions.long() // self.block_size
    table = self.block_table.gpu
    slots = table[rows, blocks].long() * self.block_size + positions % self.block_size
    self.slot_mapping.gpu.fill_(-1)
    self.slot_mapping.gpu[:positions.numel()].copy_(torch.where(valid, slots, -1))


def install():
    from vllm.platforms import current_platform
    assert type(current_platform).__module__ == "vllm_fl.platform"
    from vllm_fl.dispatch.backends.vendor.ascend.impl.attention.attention import AscendAttentionBackend, AscendAttentionBackendImpl
    if getattr(AscendAttentionBackendImpl, "_cosyvoice_strict_fp32", False):
        return
    from vllm_fl.dispatch.backends.vendor.ascend.impl import canonical_rotary
    canonical_rotary.npu_rotary_embedding.__code__ = torch_rotary.__code__
    from vllm_fl.worker.block_table import BlockTable
    BlockTable.compute_slot_mapping = torch_slot_mapping
    original = AscendAttentionBackendImpl.forward

    @torch.inference_mode()
    def forward(model, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        if model.vllm_config.model_config.dtype != torch.float32:
            return original(model, layer, query, key, value, kv_cache, attn_metadata,
                            output, output_scale, output_block_scale)
        if query.dtype != torch.float32:
            raise RuntimeError(f"FP32 model produced non-FP32 Q: {query.dtype}")
        if output is None:
            raise RuntimeError("Expected the vLLM output buffer")
        if attn_metadata is None:
            # Existing engine profiling has no actual generation request.
            return output.fill_(0)
        if model.vllm_config.parallel_config.tensor_parallel_size != 1 or model.sliding_window is not None:
            raise NotImplementedError("FP32 supports TP=1 without sliding-window attention")
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Quantized output is incompatible with FP32")
        model.key_cache, model.value_cache = kv_cache[0], kv_cache[1]
        return fp32_paged_attention(query, key, value, model.key_cache, model.value_cache,
            attn_metadata, model.num_heads, model.num_kv_heads, model.head_size, model.scale, output)

    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    AscendAttentionBackend.supported_dtypes = list(getattr(AscendAttentionBackend,"supported_dtypes",[torch.float16,torch.bfloat16])) + [torch.float32]
    AscendAttentionBackendImpl.forward = forward
    AscendAttentionBackendImpl._cosyvoice_strict_fp32 = True


class FP32WorkerExtension:
    def fp32_dtype_report(self):
        model = self.get_model()
        parameters = Counter(str(p.dtype) for p in model.parameters() if p.is_floating_point())
        devices = Counter(str(p.device) for p in model.parameters())
        caches = Counter()
        for module in model.modules():
            impl = getattr(module, "impl", None)
            if impl is not None:
                for name in ["key_cache", "value_cache"]:
                    tensor = getattr(impl, name, None)
                    if isinstance(tensor, torch.Tensor):
                        caches[str(tensor.dtype)] += 1
        return {"floating_parameter_dtypes": dict(parameters), "parameter_devices": dict(devices),
                "kv_cache_dtypes": dict(caches), "attention": {
                    **{k: v for k, v in _STATS.items() if k != "dtypes"},
                    "dtypes": sorted(_STATS["dtypes"])},
                "graph_replay_calls": _GRAPH_REPLAYS,
                "full_decode": __import__("full_decode_attention").COUNTS,
                "backend": "NPU FP32 matmul/softmax with vLLM paged KV cache",
                "matmul_hf32": torch.npu.matmul.allow_hf32}


def self_test(root):
    from types import SimpleNamespace
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    torch.manual_seed(42)
    device = torch.device("npu:0")
    h, kvh, dim, block = 14, 2, 64, 128
    kc = torch.zeros(4, block, kvh, dim, device=device)
    vc = torch.zeros_like(kc)
    table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device=device)
    all_k, all_v, cases = [], [], []
    for count in [130, 1, 7]:
        q = torch.randn(count, h, dim)
        k = torch.randn(count, kvh, dim)
        v = torch.randn_like(k)
        previous = sum(x.shape[0] for x in all_k)
        all_k.append(k);all_v.append(v)
        length = previous + count
        positions = torch.arange(previous, length, device=device)
        slots = table[0].long()[positions // block] * block + positions % block
        meta = SimpleNamespace(num_actual_tokens=count, actual_seq_lengths_q=[count],
                               seq_lens_list=[length], slot_mapping=slots, block_tables=table)
        output = torch.empty(count, h, dim, device=device)
        fp32_paged_attention(q.to(device), k.to(device), v.to(device), kc, vc, meta, h, kvh, dim, dim**-0.5, output)
        keys = torch.cat(all_k).transpose(0, 1).repeat_interleave(h // kvh, dim=0)
        values = torch.cat(all_v).transpose(0, 1).repeat_interleave(h // kvh, dim=0)
        mask = torch.arange(length)[None] > (torch.arange(count) + previous)[:, None]
        scores = q.transpose(0, 1) @ keys.transpose(-1, -2) * dim**-0.5
        reference = (scores.masked_fill(mask[None], -torch.inf).softmax(-1) @ values).transpose(0, 1)
        actual = output.cpu()
        error = (actual - reference).abs().max().item()
        assert torch.allclose(actual, reference, atol=1e-4, rtol=1e-4), error
        cases.append({"query_tokens": count, "context_tokens": length, "max_absolute_error": error,
                      "output_dtype": str(actual.dtype), "cache_dtype": str(kc.dtype)})
    result = {"cases": cases, "passed": True, "checks": "GQA, causal mask, paged boundaries, decode, chunked prefill"}
    (root / "reports/fp32_attention_self_test.json").write_text(json.dumps(result, indent=2))
    print("FP32_ATTENTION_SELF_TEST_PASS", json.dumps(result), flush=True)


if __name__ != "__main__":
    install()
    if __import__("os").environ.get("COSY_FULL_DECODE_GRAPH") == "1":
        __import__("full_decode_attention").install()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.self_test:
        self_test(args.root)

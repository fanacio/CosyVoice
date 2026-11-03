"""Experimental strict FP32 full decode graph; fixed device metadata buffers."""
import torch
from vllm.forward_context import get_forward_context
from vllm.config.compilation import CUDAGraphMode

STATES = {}
COUNTS = {'full_decode_replays': 0, 'full_attention_captures': 0}
LIMIT = int(__import__("os").environ.get("COSY_GRAPH_CONTEXT", "4096"))

def update(state, meta):
    n = state['lengths'].shape[0]
    state['lengths'].copy_(meta.seq_lens[:n])
    state['slots'].copy_(meta.slot_mapping[:n])
    table = meta.block_tables[:n]
    state['table'].zero_()
    width = min(table.shape[1], state['table'].shape[1])
    state['table'][:, :width].copy_(table[:, :width])

def full_attention(layer, model, query, key, value, cache, meta, output):
    n = query.shape[0]
    name = layer.layer_name
    index = (name, n)
    if index not in STATES:
        state = {'lengths': torch.ones(n, device=query.device, dtype=torch.int32),
                 'slots': torch.zeros(n, device=query.device, dtype=torch.int64),
                 'table': torch.zeros((n, LIMIT // cache[0].shape[1]), device=query.device, dtype=torch.int64)}
        update(state, meta)
        STATES[index] = state
    state = STATES[index]
    kc, vc = cache[0], cache[1]
    model.key_cache, model.value_cache = kc, vc
    kflat = kc.view(-1, model.num_kv_heads, model.head_size)
    vflat = vc.view_as(kflat)
    slots = state['slots'].clamp_min(0)
    kflat.index_copy_(0, slots, key.view(n, model.num_kv_heads, model.head_size))
    vflat.index_copy_(0, slots, value.view(n, model.num_kv_heads, model.head_size))
    positions = torch.arange(LIMIT, device=query.device, dtype=torch.int64)
    repeat = model.num_heads // model.num_kv_heads
    # Static loop: n is a capture shape, never a runtime sequence length.
    for i in range(n):
        physical = state['table'][i].index_select(0, positions // kc.shape[1]) * kc.shape[1] + positions % kc.shape[1]
        k = kflat.index_select(0, physical).transpose(0, 1).repeat_interleave(repeat, 0)
        v = vflat.index_select(0, physical).transpose(0, 1).repeat_interleave(repeat, 0)
        q = query[i].reshape(model.num_heads, 1, model.head_size)
        scores = torch.matmul(q, k.transpose(-1, -2)) * model.scale
        scores = scores.masked_fill((positions >= state['lengths'][i].clamp_min(1))[None, None], -torch.inf)
        result = torch.matmul(torch.softmax(scores, -1, dtype=torch.float32), v)
        output[i].copy_(result.reshape_as(output[i]))
    COUNTS['full_attention_captures'] += 1
    return output

def install():
    from vllm_fl.dispatch.backends.vendor.ascend.impl.attention.attention import AscendAttentionBackendImpl
    from vllm_fl.compilation.graph import GraphWrapper as ACLGraphWrapper
    original = AscendAttentionBackendImpl.forward
    def forward(model, layer, query, key, value, kv_cache, attn_metadata, output=None,
                output_scale=None, output_block_scale=None):
        ctx = get_forward_context()
        if ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL and attn_metadata is not None:
            assert query.dtype == torch.float32
            return full_attention(layer, model, query, key, value, kv_cache, attn_metadata, output)
        return original(model, layer, query, key, value, kv_cache, attn_metadata,
                        output, output_scale, output_block_scale)
    AscendAttentionBackendImpl.forward = forward
    original_call = ACLGraphWrapper.__call__
    def call(wrapper, *args, **kwargs):
        ctx = get_forward_context()
        if ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL:
            metas = ctx.attn_metadata
            if isinstance(metas, dict):
                if any(m is not None and (int(m.num_actual_tokens) != ctx.batch_descriptor.num_tokens
                       or len(m.seq_lens_list) != ctx.batch_descriptor.num_tokens) for m in metas.values()):
                    saved = ctx.cudagraph_runtime_mode
                    ctx.cudagraph_runtime_mode = CUDAGraphMode.NONE
                    try:
                        return original_call(wrapper, *args, **kwargs)
                    finally:
                        ctx.cudagraph_runtime_mode = saved
                n = ctx.batch_descriptor.num_tokens
                for name, meta in metas.items():
                    if meta is not None and (name, n) not in STATES:
                        device = torch.device('npu', torch.npu.current_device())
                        STATES[(name, n)] = {'lengths': torch.ones(n, device=device, dtype=torch.int32), 'slots': torch.zeros(n, device=device, dtype=torch.int64), 'table': torch.zeros((n, LIMIT // 128), device=device, dtype=torch.int64)}
                for (name, n), state in STATES.items():
                    meta = metas.get(name)
                    if meta is not None and len(meta.seq_lens_list) == n:
                        update(state, meta)
            descriptor = ctx.batch_descriptor
            entry = wrapper.concrete_graph_entries.get(descriptor)
            if entry is not None and entry.graph is not None:
                COUNTS['full_decode_replays'] += 1
        return original_call(wrapper, *args, **kwargs)
    ACLGraphWrapper.__call__ = call
    # Native task-update operates FIA handles. This path uses device buffers.
    original_update = AscendAttentionBackendImpl.update_graph_params
    def update_graph_params(*args, **kwargs):
        return None
    AscendAttentionBackendImpl.update_graph_params = staticmethod(update_graph_params)

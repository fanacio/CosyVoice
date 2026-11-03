"""Standard vLLM prompt-embedding bridge; no FL backend method replacement."""
import threading,time
from pathlib import Path
import torch
from vllm import EngineArgs,LLMEngine,SamplingParams
from .engine import BatchedEngine,RequestRecords
from .export import export_original_fp32

class VllmBridge:
    def __init__(self,model,source,cache,execution='eager',max_model_len=4096,max_num_seqs=4,memory_utilization=.15,seed=1986,timeout=180):
        self.seed,self.timeout=seed,timeout
        self.requests=RequestRecords()
        export=Path(cache)/'exported/cosyvoice3_llm_fp32';export.parent.mkdir(parents=True,exist_ok=True)
        export_original_fp32(Path(source),export)
        graph={} if execution=='eager' else {'compilation_config':{'mode':3,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,3,4]}}
        self.engine=BatchedEngine(LLMEngine.from_engine_args(EngineArgs(model=str(export),dtype='float32',tensor_parallel_size=1,skip_tokenizer_init=True,enable_prompt_embeds=True,enforce_eager=execution=='eager',enable_prefix_caching=False,enable_chunked_prefill=True,block_size=128,max_model_len=max_model_len,max_num_seqs=max_num_seqs,max_num_batched_tokens=4096,gpu_memory_utilization=memory_utilization,worker_extension_cls='vllm_fl.dispatch.backends.vendor.ascend.impl.fp32_diagnostics.FP32Diagnostics',**graph)))
        model.vllm=self.engine;model.lock=threading.Lock();model.vllm_bridge=self
        del model.llm.model.model.layers

    def generate(self,model,lm_input,sampling,min_len,max_len,uuid):
        params=SamplingParams(top_k=sampling,stop_token_ids=model.stop_token_ids,min_tokens=int(min_len),max_tokens=int(max_len),detokenize=False,seed=self.seed)
        prompt=lm_input.squeeze(0).float().cpu().contiguous();generated=[];consumed=0;finished=False;start=time.monotonic()
        try:
            for result in self.engine.generate(uuid,{'prompt_embeds':prompt},params,self.timeout):
                if result.request_id!=uuid:raise RuntimeError('Mismatched request id')
                output=result.outputs[0];ids=list(output.token_ids)
                for token in ids[consumed:]:
                    if token in model.stop_token_ids:continue
                    if not 0<=token<model.speech_token_size:raise RuntimeError('Invalid speech token')
                    generated.append(token);yield token
                consumed=len(ids);finished=result.finished
            if not finished:raise RuntimeError('Missing final engine output')
            self.requests.append({'request_id':uuid,'tokens':generated,'token_count':len(generated),'llm_seconds':time.monotonic()-start,'finish_reason':output.finish_reason})
        finally:
            if not finished:self.engine.abort_request(uuid)

    def close(self):self.engine.shutdown()

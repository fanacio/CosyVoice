"""CosyVoice3 Ascend model classes. CUDA/CPU classes remain unchanged."""
import os,time,functools
from pathlib import Path
import torch
from cosyvoice.cli.model import CosyVoice3Model
from cosyvoice.cli.cosyvoice import CosyVoice3
from .acoustic import configure_npu,prepare_acoustic_models
from .bridge import VllmBridge
from .export import prepare_model
from .frontend import make_npu_frontend_class

class AscendModel(CosyVoice3Model):
    def __init__(self,*args,state,options,**kwargs):
        super().__init__(*args,**kwargs)
        self.device=configure_npu();self.state=state;self.options=options;self.errors=[]

    def load(self,*args,**kwargs):
        device=self.device;self.device=torch.device('cpu')
        try:super().load(*args,**kwargs)
        finally:self.device=device
        self.llm.to(device).eval();self.state['acoustic_settings']=prepare_acoustic_models(self.flow,self.hift,device)
        # Per-instance timing wrappers only; model kernels are normal NPU branches.
        for name,module in [('flow',self.flow),('hift',self.hift)]:
            method=module.inference
            def observe(method,name):
                @functools.wraps(method)
                def call(*a,**kw):
                    torch.npu.synchronize();start=time.monotonic();result=method(*a,**kw);torch.npu.synchronize()
                    out=result[0] if isinstance(result,(tuple,list)) else result
                    self.state.setdefault(name,[]).append({'seconds':time.monotonic()-start,'output_device':str(out.device),'output_shape':list(out.shape),'dtype':str(out.dtype)})
                    return result
                return call
            module.inference=observe(method,name)

    def load_vllm(self,path):
        self.bridge=VllmBridge(self.llm,**self.options)

    def llm_job(self,*args,**kwargs):
        import inspect,traceback
        uid=inspect.signature(super().llm_job).bind(*args,**kwargs).arguments['uuid']
        torch.npu.set_device(self.device)
        try:return super().llm_job(*args,**kwargs)
        except Exception:self.errors.append(traceback.format_exc())
        finally:self.llm_end_dict[uid]=True

class AscendCosyVoice3(CosyVoice3):
    def __init__(self,model_dir,cache_dir=None,execution='decode-graph',max_model_len=4096,max_num_seqs=4,memory_utilization=.15,seed=1986,request_timeout=180,**kwargs):
        import hashlib
        from hyperpyyaml import load_hyperpyyaml
        source=Path(model_dir).resolve()
        self.cache=Path(cache_dir or Path.home()/'.cache/cosyvoice-ascend'/hashlib.sha256(str(source).encode()).hexdigest()[:12]).resolve()
        self.cache.mkdir(parents=True,exist_ok=True)
        self.state={'device':'npu:0','flow':[],'hift':[]}
        self.model_dir=str(source);self.fp16=False
        device=configure_npu();stage=prepare_model(self.cache,source)
        with (stage/'cosyvoice3.yaml').open() as f:cfg=load_hyperpyyaml(f,overrides={'qwen_pretrain_path':str(stage/'CosyVoice-BlankEN')})
        from cosyvoice.cli.frontend import CosyVoiceFrontEnd
        frontend=make_npu_frontend_class(CosyVoiceFrontEnd,self.cache,device)
        self.frontend=frontend(cfg['get_tokenizer'],cfg['feat_extractor'],str(stage/'campplus.onnx'),str(stage/'speech_tokenizer_v3.onnx'),str(stage/'spk2info.pt'),cfg['allowed_special'])
        self.state['frontend_object']=self.frontend;self.sample_rate=cfg['sample_rate']
        opts=dict(source=source,cache=self.cache,execution=execution,max_model_len=max_model_len,max_num_seqs=max_num_seqs,memory_utilization=memory_utilization,seed=seed,timeout=request_timeout)
        self.model=AscendModel(cfg['llm'],cfg['flow'],cfg['hift'],False,state=self.state,options=opts)
        self.model.load(str(stage/'llm.pt'),str(stage/'flow.pt'),str(stage/'hift.pt'))
        self.model.load_vllm(str(self.cache/'exported/cosyvoice3_llm_fp32'))
        self.bridge=self.model.bridge
        torch.npu.synchronize();torch.npu.empty_cache()

    def close(self):self.bridge.close()

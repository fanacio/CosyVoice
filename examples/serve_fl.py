"""One-model CosyVoice HTTP service on FL Ascend; PCM24k compatible routes."""
import argparse,time,sys,os,random,signal
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'third_party/Matcha-TTS')]
for k,v in {'VLLM_VENDOR':'ascend','VLLM_PLUGINS':'fl','USE_FLAGGEMS':'0','VLLM_FL_ASCEND_SLOT_MAPPING':'torch','VLLM_WORKER_MULTIPROC_METHOD':'spawn','GLOO_SOCKET_IFNAME':'lo','OMP_NUM_THREADS':'8'}.items():os.environ.setdefault(k,v)
def main():
 p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--cache-dir',required=True);p.add_argument('--execution',choices=['eager','decode-graph'],default='decode-graph');p.add_argument('--max-model-len',type=int,default=4096);p.add_argument('--max-concurrency',type=int,choices=range(1,5),default=4);p.add_argument('--port',type=int,default=50040);p.add_argument('--host',default='0.0.0.0');p.add_argument('--request-timeout',type=int,default=180);a=p.parse_args()
 import torch,numpy as np,uvicorn
 torch.set_num_threads(8);random.seed(1986);np.random.seed(1986);torch.manual_seed(1986)
 from cosyvoice.cli.cosyvoice import AutoModel
 from cosyvoice.runtime.ascend.http_runtime import Runtime
 from cosyvoice.runtime.ascend.http import http_application
 started=time.monotonic();tts=AutoModel(model_dir=a.model_dir,device='npu',cache_dir=a.cache_dir,execution=a.execution,max_model_len=a.max_model_len,max_num_seqs=a.max_concurrency,request_timeout=a.request_timeout)
 args=SimpleNamespace(root=Path(a.cache_dir),seed=1986,max_model_len=a.max_model_len,max_num_seqs=a.max_concurrency,request_timeout=a.request_timeout,llm_dtype='float32')
 runtime=Runtime(state=tts.state,tts=tts,engines=[tts.bridge.engine],requests=tts.bridge.requests,thread_errors=tts.model.errors,args=args,load_seconds=time.monotonic()-started)
 def terminate(*_):raise SystemExit(0)
 signal.signal(signal.SIGTERM,terminate)
 try:uvicorn.run(http_application(runtime),host=a.host,port=a.port,workers=1)
 finally:runtime.healthy=False;tts.close()
if __name__=='__main__':main()

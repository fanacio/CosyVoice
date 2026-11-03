"""CosyVoice3 FP32 inference with the installed FL Ascend backend."""
import os,sys,json,argparse,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/Matcha-TTS')]
for k,v in {'VLLM_VENDOR':'ascend','VLLM_PLUGINS':'fl','USE_FLAGGEMS':'0','VLLM_FL_ASCEND_SLOT_MAPPING':'torch','VLLM_WORKER_MULTIPROC_METHOD':'spawn','GLOO_SOCKET_IFNAME':'lo','OMP_NUM_THREADS':'8'}.items():os.environ.setdefault(k,v)

def main():
 p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--reference',required=True);p.add_argument('--text',required=True);p.add_argument('--prompt-text',required=True);p.add_argument('--output',default='output.wav');p.add_argument('--cache-dir');p.add_argument('--execution',choices=['eager','decode-graph'],default='decode-graph');p.add_argument('--max-model-len',type=int,default=4096);a=p.parse_args()
 import torch,numpy as np,soundfile as sf
 torch.set_num_threads(8);random.seed(1986);np.random.seed(1986);torch.manual_seed(1986)
 from cosyvoice.cli.cosyvoice import AutoModel
 tts=AutoModel(model_dir=a.model_dir,device='npu',cache_dir=a.cache_dir,execution=a.execution,max_model_len=a.max_model_len,max_num_seqs=1)
 try:
  chunks=[x['tts_speech'].detach().float().cpu() for x in tts.inference_zero_shot(a.text,a.prompt_text,a.reference,stream=False,text_frontend=False)]
  if tts.model.errors:raise RuntimeError(str(tts.model.errors))
  audio=torch.cat(chunks,dim=-1).squeeze(0).numpy();assert np.isfinite(audio).all() and np.max(np.abs(audio))>1e-5
  output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);sf.write(output,audio,tts.sample_rate,subtype='PCM_16')
  import vllm_fl,cosyvoice
  report={'fl_source':vllm_fl.__file__,'cosyvoice_source':cosyvoice.__file__,'sample_rate':tts.sample_rate,'audio_seconds':len(audio)/tts.sample_rate,'execution':a.execution,'worker':tts.bridge.engine.collective_rpc('fp32_dtype_report')}
  output.with_suffix('.json').write_text(json.dumps(report,indent=2));print('COSYVOICE_FL_SUCCESS',json.dumps(report),flush=True)
 finally:tts.close()
if __name__=='__main__':main()

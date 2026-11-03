from pathlib import Path
import os,subprocess,shutil,json
R=Path(__file__).resolve().parents[1]; E=R/'evaluation'; T=E/'temporary_replicas'; T.mkdir(exist_ok=True)
source=T/'source'
for folder in ['scripts','service']: shutil.copytree(R/folder,source/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
p=source/'scripts/run_tts.py'; s=p.read_text().replace('    args = parser.parse_args()','    parser.add_argument("--max-model-len",type=int,default=12288)\n    args = parser.parse_args()').replace('max_model_len=4096,','max_model_len=args.max_model_len,'); p.write_text(s)
p=source/'service/server.py'; s=p.read_text().replace('    options = parser.parse_args()','    parser.add_argument("--max-model-len",type=int,default=12288)\n    options = parser.parse_args()').replace("'--llm-execution', options.llm_execution]","'--llm-execution', options.llm_execution, '--max-model-len', str(options.max_model_len)]");p.write_text(s)
p=source/'service/runtime.py';p.write_text(p.read_text().replace('if reserve > 4096:','if reserve > self.args.max_model_len:').replace('4096-token service budget','configured token service budget'))
p=source/'scripts/full_decode_attention.py';p.write_text(p.read_text().replace('LIMIT = 4096','LIMIT = 12288').replace('(n, 32)','(n, LIMIT // 128)'))
records=[]
for device,port in [(2,50102),(3,50103),(4,50104),(5,50105)]:
 root=T/f'npu{device}';root.mkdir(exist_ok=True)
 for name in ['src','exported','assets']:
  if not (root/name).exists():(root/name).symlink_to(R/name)
 for name in ['runtime','reports','outputs','logs']:(root/name).mkdir(exist_ok=True)
 env=dict(os.environ);env.update(ASCEND_RT_VISIBLE_DEVICES=str(device),VLLM_VENDOR='ascend',VLLM_PLUGINS='fl',USE_FLAGGEMS='0',GLOO_SOCKET_IFNAME='lo',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 cmd=[str(R/'.venv/bin/python'),'-u',str(source/'service/server.py'),'--root',str(root),'--host','127.0.0.1','--http-port',str(port),'--llm-execution','decode-graph','--max-concurrency','4','--max-model-len','12288','--request-timeout','900']
 log=E/f'replica_{device}.log'
 with log.open('w') as f:p=subprocess.Popen(cmd,env=env,cwd=root,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 records.append({'pid':p.pid,'root':str(root),'device':device,'port':port,'command':cmd,'log':str(log)})
(E/'instances').mkdir(exist_ok=True);(E/'instances/full_decode_servers.json').write_text(json.dumps(records,indent=2));print(records)

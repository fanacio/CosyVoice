"""One HTTP client: full SEED generation and original official scoring."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,sys,runpy
R=Path(__file__).resolve().parents[1];REF=R.with_name('CosyVoice_ascend')/'evaluation'
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=50040);p.add_argument('--name',default='single_fp32');a=p.parse_args()
E=R/'evaluation/single_server';(E/'instances').mkdir(parents=True,exist_ok=True)
(E/'instances/full_decode_servers.json').write_text(json.dumps([{'port':a.port,'root':str(R)}]))
for n in ['models','repos','.venv-cpu']:
 if not (E/n).exists():(E/n).symlink_to(REF/n,target_is_directory=True)
sys.path.insert(0,str(REF))
g=runpy.run_path(str(REF/'generate_full_decode.py'),run_name='reuse');g['main'].__globals__.update(EVAL=E,PROJECT=REF.parent)
sys.argv=['generate','--name',a.name,'--ports',str(a.port),'--per-server','1'];g['main']()
s=runpy.run_path(str(REF/'score_full_decode.py'),run_name='reuse');s['score'].__globals__.update(EVAL=E,PROJECT=REF.parent,PYTHON=REF/'.venv-cpu/bin/python',RUNNER=REF/'run_official_metric.py')
results={}
with ThreadPoolExecutor(max_workers=2) as pool:
 tasks={(split,m):pool.submit(s['score'],a.name,split,m) for split in ['zh','en','hard'] for m in ['asr','sim']}
 for (split,m),f in tasks.items():results[split+'/'+m]=f.result()
(E/'runs'/a.name/'metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
for split in ['zh','en','hard']:
 print(split,'CER/WER %',results[split+'/asr']['scores']['asr']['mean_percent'],'SIM x100',results[split+'/sim']['scores']['sim']['mean_percent'],flush=True)
print('SINGLE_SERVER_ACCURACY_COMPLETE',flush=True)

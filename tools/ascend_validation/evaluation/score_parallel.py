from pathlib import Path
import os,signal,json,sys,time,runpy
from concurrent.futures import ThreadPoolExecutor
R=Path(__file__).resolve().parents[1];E=R/'evaluation';OLD=R.with_name('CosyVoice_ascend');REF=OLD/'evaluation';NAME='fl_decode_fp32_c4'
# Stop only this evaluation's previous coordinator and metric subprocesses.
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:args=(d/'cmdline').read_bytes().split(b'\0'); text=b' '.join(args).decode(errors='replace')
 except (OSError,ProcessLookupError):continue
 if len(args)>1 and 'python' in Path(args[0].decode(errors='replace')).name and (str(E/'run_accuracy.py') in text or (str(E/'runs') in text and str(REF/'run_official_metric.py') in text)):
  try:os.kill(int(d.name),signal.SIGKILL)
  except ProcessLookupError:pass
sys.path.insert(0,str(REF))
# Archive unfinished partitions before changing partition counts; preserve completed zh ASR.
for split in ['zh','en','hard']:
 for metric in ['asr','sim']:
  p=E/'runs'/NAME/split/'score_parts'/metric
  if split=='zh' and metric=='asr':continue
  if p.exists():p.rename(p.with_name(metric+'_previous_partitioning'))
source=(REF/'score_full_decode.py').read_text().replace("nworkers=(16 if split=='en' else 4) if metric=='asr' else 8","nworkers=(64 if split=='en' else 4) if metric=='asr' else 32")
ns={'__file__':str(REF/'score_full_decode.py'),'__name__':'reused_scoring'};exec(compile(source,str(REF/'score_full_decode.py'),'exec'),ns)
ns.update(EVAL=E,PROJECT=OLD,PYTHON=REF/'.venv-cpu/bin/python',RUNNER=REF/'run_official_metric.py')
results={}
with ThreadPoolExecutor(max_workers=6) as pool:
 jobs={(split,m):pool.submit(ns['score'],NAME,split,m) for split in ['zh','en','hard'] for m in ['asr','sim']}
 for key,future in jobs.items():
  results['/'.join(key)]=future.result()
  (E/'accuracy_metrics_partial.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
(E/'accuracy_metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print('ALL_FL_SCORES_COMPLETE',flush=True)

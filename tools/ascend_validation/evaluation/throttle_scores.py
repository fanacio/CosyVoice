from pathlib import Path
import os,signal,time,json
E=Path(__file__).resolve().parent
while not (E/'accuracy_metrics.json').exists():
 groups={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:cmd=(p/'cmdline').read_bytes().decode(errors='replace').replace('\0',' ')
  except OSError:continue
  if 'run_official_metric.py ' not in cmd or str(E/'runs') not in cmd:continue
  try:
   pair=cmd.split('--pairs ')[1].split(' ')[0];parts=Path(pair).parts;split=parts[-4];metric=parts[-2]
   groups.setdefault((split,metric),[]).append(int(p.name))
  except (IndexError,ValueError):continue
 for (split,metric),pids in groups.items():
  limit=64 if metric=='asr' and split=='en' else 32
  for i,pid in enumerate(sorted(pids)):
   try:os.kill(pid,signal.SIGCONT if i<limit else signal.SIGSTOP)
   except ProcessLookupError:pass
 (E/'scoring_active_groups.json').write_text(json.dumps({str(k):len(v) for k,v in groups.items()}))
 time.sleep(10)

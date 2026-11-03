from pathlib import Path
import sys,runpy,json,os,signal,time,shutil,traceback
from concurrent.futures import ThreadPoolExecutor
R=Path(__file__).resolve().parents[1];E=R/'evaluation';OLD=R.with_name('CosyVoice_ascend');REF=OLD/'evaluation';NAME='fl_decode_fp32_c4'
sys.path.insert(0,str(REF))
for n in ['models','repos','.venv-cpu']:
 if not (E/n).exists():(E/n).symlink_to(REF/n,target_is_directory=True)
def cleanup():
 records=json.loads((E/'instances/full_decode_servers.json').read_text())
 for row in records:
  pid=row['pid']
  try:
   if str(E/'temporary_replicas') in Path(f'/proc/{pid}/cmdline').read_text(): os.killpg(pid,signal.SIGTERM)
  except (FileNotFoundError,ProcessLookupError):pass
 time.sleep(12)
 for row in records:
  try:
   if str(E/'temporary_replicas') in Path(f"/proc/{row['pid']}/cmdline").read_text():os.killpg(row['pid'],signal.SIGKILL)
  except (FileNotFoundError,ProcessLookupError):pass
 shutil.rmtree(E/'temporary_replicas')
 (E/'replicas_cleaned.json').write_text(json.dumps({'cleaned':True,'ports':[r['port'] for r in records]}))
try:
 g=runpy.run_path(str(REF/'generate_full_decode.py'),run_name='reuse_generation');g['main'].__globals__.update(EVAL=E,PROJECT=OLD)
 sys.argv=['generate_full_decode.py','--name',NAME,'--per-server','4','--ports','50102,50103,50104,50105,50106,50107,50108,50109,50110,50111,50112,50113'];g['main']()
finally: cleanup()
s=runpy.run_path(str(REF/'score_full_decode.py'),run_name='reuse_scoring');s['score'].__globals__.update(EVAL=E,PROJECT=OLD,PYTHON=REF/'.venv-cpu/bin/python',RUNNER=REF/'run_official_metric.py')
results={}
with ThreadPoolExecutor(max_workers=2) as pool:
 jobs={(split,metric):pool.submit(s['score'],NAME,split,metric) for split in ['zh','en','hard'] for metric in ['asr','sim']}
 for key,f in jobs.items():results['/'.join(key)]=f.result()
(E/'accuracy_metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
lines=['\n## FL 图模式全量精度评测（2026-10-30）\n','直接复用 CosyVoice_ascend/evaluation 的生成与官方评分代码、原始数据、模型和评分环境，没有复制数据集或评分模型。12 个临时副本使用 NPU 2~13，端口 50102~50113，FP32 decode 全图、12288 上下文、每副本并发4；原50040服务保留。\n','|子集|样本数|CER/WER（逐句宏平均）|SIM-WavLM|\n|---|---:|---:|---:|\n']
for split in ['zh','en','hard']:
 asr=results[split+'/asr'];sim=results[split+'/sim'];lines.append(f'|{split}|见 coverage|{json.dumps(asr["scores"],ensure_ascii=False)}|{json.dumps(sim["scores"],ensure_ascii=False)}|\n')
lines+=['\n完整结果：evaluation/accuracy_metrics.json；音频、逐句分数与覆盖率：evaluation/runs/fl_decode_fp32_c4/。临时服务及源码副本已停止并删除，清理记录：evaluation/replicas_cleaned.json。\n']
with (R/'Summarize_report.md').open('a') as f:f.write(''.join(lines))
print('FL_ACCURACY_COMPLETE',json.dumps(results,ensure_ascii=False),flush=True)

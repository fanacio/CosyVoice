from pathlib import Path
import json,time,sys
R=Path(__file__).resolve().parents[1]; E=R/'evaluation'; REF=R.with_name('CosyVoice_ascend')/'evaluation'
while not (E/'accuracy_metrics.json').exists():time.sleep(20)
d=json.loads((E/'accuracy_metrics.json').read_text());base=json.loads((REF/'results/full_decode_fp32_c4/metrics_all.json').read_text())['full_decode_fp32_c4']
sys.path.insert(0,str(REF));from seed_http_eval import load_meta,audit
metas={'zh':'zh/meta.lst','en':'en/meta.lst','hard':'zh/hardcase.lst'}
report={};table=[]
for split,meta in metas.items():
 out=E/'runs/fl_decode_fp32_c4'/split;rows=load_meta(REF.parent/'dataset/seedtts_testset'/meta)
 check=audit(rows,out,wer=out/'asr_raw.tsv',sim=out/'sim_wavlm_raw.tsv');assert check['complete'];report[split]=check
 cer=check['scores']['asr']['mean_percent'];sim=check['scores']['sim']['mean_percent'];b=base[split]['scores']
 table.append(f'|{split}|{len(rows)}|{cer:.4f}%|{b["asr"]["mean_percent"]:.4f}%|{sim:.4f}|{b["sim"]["mean_percent"]:.4f}|\n')
(E/'accuracy_final_coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
text='''\n## FL FP32 图模式全量精度评测（2026-10-30）

直接调用原 CosyVoice_ascend/evaluation 的生成和官方评分代码，复用其 SEED-TTS-Eval 数据、评分模型和 Python 3.10 评分环境，没有复制数据集或模型。中文/hardcase 使用 Paraformer-zh CER，英文使用 Whisper-large-v3 WER；SIM 使用官方 WavLM-Large SV 模型。指标均按原流程逐句宏平均，不裁剪高错误率样本。

全量生成3508条，零失败；4个副本扩展至12个，NPU 2~13、端口50102~50113，每副本并发4。FP32、TP=1、CP=1、decode全图，prefill eager。为覆盖完整hardcase，上下文及固定图attention上限使用12288，区别于性能测试的4096。原50040常驻服务保留。

|子集|样本数|FL CER/WER|原生图模式 CER/WER|FL SIM×100|原生图模式 SIM×100|
|---|---:|---:|---:|---:|---:|
'''+''.join(table)+'''
SIM×100 是余弦相似度乘100，不是分类准确率。原生列复用原目录已有 full_decode_fp32_c4 全量结果，不是本轮重新合成的原生对照；多副本调度和随机采样可造成差异，不能只据小差值判定框架数值误差。

12个临时服务已停止，临时源码及实例目录已删除，端口已释放，未保留启动副本。仅保留评测结果、音频、日志、调度脚本和历史配置记录。生成/评分中途扩容并调整CPU分片并行度，分片划分不改变单句评分算法；已有完整评分分片按校验和复用，未完成旧分片归档保留作诊断。

证据：evaluation/accuracy_final_coverage.json（完整覆盖率）、evaluation/accuracy_metrics.json（评分汇总）、evaluation/runs/fl_decode_fp32_c4/（WAV/逐句分数）、evaluation/replicas_cleaned.json（副本清理）、evaluation/scoring_parallel.log。数据、models、repos、.venv-cpu 均指向原项目资源。
'''
p=R/'Summarize_report.md';s=p.read_text();marker='\n## FL FP32 图模式全量精度评测（2026-10-30）';repro_marker='\n## 精度评测复现命令'; suffix=(repro_marker+s.split(repro_marker,1)[1]) if repro_marker in s else ''; s=s.split(marker)[0];p.write_text(s+text+suffix)
print('ACCURACY_DOCUMENT_FINALIZED',json.dumps(report,ensure_ascii=False),flush=True)

# CosyVoice3：vllm-fl Ascend FP32 图模式服务与性能测试

日期：2026-10-30。以下命令在 **105 服务器宿主机**执行。

- 容器：`vllm-ascend-fanyc-1030`
- 项目目录：`/data/tiankuan/fanyc/models_test_opt/CosyVoice`
- 设备：物理 NPU 1（容器内逻辑 npu:0）
- 模式：FP32、TP=1、CP=1、decode-graph、最多四个在途请求
- 图范围：decode 模型 forward 包含 FP32 attention；prefill 保持 eager，不代表整个 TTS 流程全图
- HTTP 地址：`http://172.16.10.105:50040`；接口文档 `/docs`；健康检查 `/healthz`

## 1. 启动 FL HTTP 图模式服务

```bash
docker exec vllm-ascend-fanyc-1030 bash -c '
cd /data/tiankuan/fanyc/models_test_opt/CosyVoice
ASCEND_RT_VISIBLE_DEVICES=1 bash scripts/service_npu.sh start \
  --http-port 50040 \
  --llm-execution decode-graph \
  --max-concurrency 4
'
```

默认 LLM dtype 为 float32。启动脚本设置 `VLLM_VENDOR=ascend`、`VLLM_PLUGINS=fl`、`USE_FLAGGEMS=0`，并加载项目内的 FP32 attention、RoPE、slot mapping 兼容补丁。

已有实例运行时，start 不会自动替换其配置；更改参数需先 stop 再 start。服务启动涉及加载、编译与图捕获，确认 ready 后再调用。

## 2. 查询状态与健康检查

```bash
docker exec vllm-ascend-fanyc-1030 bash \
  /data/tiankuan/fanyc/models_test_opt/CosyVoice/scripts/service_npu.sh status

curl --noproxy '*' http://127.0.0.1:50040/healthz
```

健康信息应显示 `status=ready`、`llm_backend=vllm-fl/ascend`、`llm_dtype=float32`。

## 3. 单次 HTTP 合成

```bash
docker exec vllm-ascend-fanyc-1030 bash -c '
cd /data/tiankuan/fanyc/models_test_opt/CosyVoice
bash scripts/client_npu.sh \
  --protocol http \
  --port 50040 \
  --tts-text "你好，这是图模式语音服务。" \
  --output outputs/service/my_demo.wav
'
```

服务为非流式整段合成，输出为 24 kHz 音频。客户端生成 WAV，服务按 request_id 保存请求报告。

## 4. FL HTTP 性能测试

服务启动并 ready 后执行：

```bash
docker exec vllm-ascend-fanyc-1030 bash -c '
cd /data/tiankuan/fanyc/models_test_opt/CosyVoice
.venv/bin/python service/benchmark_http.py \
  --label fl_decode_graph \
  --concurrency 1,2,4 \
  --rounds 3
'
```

保留控制台日志的写法：

```bash
docker exec vllm-ascend-fanyc-1030 bash -c '
cd /data/tiankuan/fanyc/models_test_opt/CosyVoice
.venv/bin/python service/benchmark_http.py \
  --label fl_retest \
  --concurrency 1,2,4 \
  --rounds 3
' > /data/tiankuan/fanyc/models_test_opt/CosyVoice/logs/benchmark_fl_retest.log 2>&1
```

本项目的 benchmark 脚本固定访问 `http://127.0.0.1:50040`，不是通过参数自动读取任意服务端口。重测建议换 label，避免覆盖旧结果。

脚本先预热两次，然后分别测试客户端并发 1、2、4；每档四条固定文本、三轮，共12请求，三档共36个计时请求。计时包含 HTTP 上传、服务端合成和 PCM 接收，不包含模型加载、启动编译或图捕获耗时。统计请求吞吐、音频秒吞吐、平均延迟和 P95 延迟；本轮不进行 CER/WER/SIM 等语音精度评测。

固定文本：

1. 你好，欢迎使用语音合成。
2. 今天阳光明媚，适合出去散步。
3. 请保持耐心，我们正在为您处理请求。
4. 这是在昇腾设备上运行的语音合成服务。

参考音频：`src/CosyVoice/asset/zero_shot_prompt.wav`。
参考文本：`You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。`。

结果位置（相对于项目目录）：

- `reports/http_benchmark_<label>.json`：完整基准指标、请求记录、前后健康状态。
- `outputs/service/http_concurrency/<label>/`：预热与测试输出 WAV。
- `outputs/service/<request_id>.json`：服务端请求报告与 worker 精度/图计数记录。
- `logs/service_*.log`：服务启动、图捕获及请求日志。

## 5. 停止服务

```bash
docker exec vllm-ascend-fanyc-1030 bash \
  /data/tiankuan/fanyc/models_test_opt/CosyVoice/scripts/service_npu.sh stop
```

同卡做框架对比时必须串行运行实例；不要让 FL 与原生 vllm-ascend 测试服务同时占用 NPU 1。另有请求访问该 HTTP 服务也会影响性能结果。

## 6. 本轮性能记录

两边在同一块 NPU 1 上依次运行，配置、文本和参考音频一致，均使用 FP32 decode 全图。

| 并发 | FL 请求/秒 | 原生 vllm-ascend 请求/秒 | FL 吞吐变化 |
| --- | ---: | ---: | ---: |
| 1 | 0.5846 | 0.6060 | -3.53% |
| 2 | 0.7905 | 0.8291 | -4.65% |
| 4 | 0.9741 | 1.0014 | -2.72% |

这是小样本 HTTP 端到端结果，无置信区间；主机其他资源未完全隔离，且 FL 使用当前 RoPE/slot mapping 绕过方案，编译器环境也不同，不能将差值全部归于框架本身。

更多数据见 `FP32图模式_HTTP性能对比.md`、`reports/graph_performance_comparison.json`。原始 FL 基准为 `reports/http_benchmark_fl_decode_graph.json`，原生基线位于 `benchmark_ascend/reports/http_benchmark_ascend_decode_graph.json`。

## FL FP32 图模式全量精度评测（2026-10-30）

直接调用原 CosyVoice_ascend/evaluation 的生成和官方评分代码，复用其 SEED-TTS-Eval 数据、评分模型和 Python 3.10 评分环境，没有复制数据集或模型。中文/hardcase 使用 Paraformer-zh CER，英文使用 Whisper-large-v3 WER；SIM 使用官方 WavLM-Large SV 模型。指标均按原流程逐句宏平均，不裁剪高错误率样本。

全量生成3508条，零失败；4个副本扩展至12个，NPU 2~13、端口50102~50113，每副本并发4。FP32、TP=1、CP=1、decode全图，prefill eager。为覆盖完整hardcase，上下文及固定图attention上限使用12288，区别于性能测试的4096。原50040常驻服务保留。

|子集|样本数|FL CER/WER|原生图模式 CER/WER|FL SIM×100|原生图模式 SIM×100|
|---|---:|---:|---:|---:|---:|
|zh|2020|1.5253%|1.5322%|77.7437|77.7439|
|en|1088|2.1131%|2.1039%|69.6373|69.6373|
|hard|400|8.7005%|8.7131%|75.8388|75.8440|

SIM×100 是余弦相似度乘100，不是分类准确率。原生列复用原目录已有 full_decode_fp32_c4 全量结果，不是本轮重新合成的原生对照；多副本调度和随机采样可造成差异，不能只据小差值判定框架数值误差。

12个临时服务已停止，临时源码及实例目录已删除，端口已释放，未保留启动副本。仅保留评测结果、音频、日志、调度脚本和历史配置记录。生成/评分中途扩容并调整CPU分片并行度，分片划分不改变单句评分算法；已有完整评分分片按校验和复用，未完成旧分片归档保留作诊断。

证据：evaluation/accuracy_final_coverage.json（完整覆盖率）、evaluation/accuracy_metrics.json（评分汇总）、evaluation/runs/fl_decode_fp32_c4/（WAV/逐句分数）、evaluation/replicas_cleaned.json（副本清理）、evaluation/scoring_parallel.log。数据、models、repos、.venv-cpu 均指向原项目资源。

## 精度评测复现命令：一个服务端、一个客户端

两个终端均进入 vllm-ascend-fanyc-1030 容器，并切换到本项目目录。

服务端终端：

```bash
cd /data/tiankuan/fanyc/models_test_opt/CosyVoice
bash scripts/service_npu.sh stop
ASCEND_RT_VISIBLE_DEVICES=1 bash scripts/service_npu.sh start \
  --http-port 50040 --llm-execution decode-graph \
  --max-concurrency 1 --max-model-len 12288 --request-timeout 900
```

客户端终端，执行一个脚本即可：

```bash
cd /data/tiankuan/fanyc/models_test_opt/CosyVoice
.venv/bin/python -u evaluation/test_accuracy_single.py \
  --port 50040 --name single_fp32
```

脚本自动完成中文2020条、英文1088条、hardcase400条的合成与官方ASR/WavLM评分，直接复用原 CosyVoice_ascend 的方法、数据和评分模型，不创建服务副本。全量hardcase需要12288上下文；已运行的4096服务需按上面的命令重启。

最终指标打印在客户端终端，并保存至 `evaluation/single_server/runs/single_fp32/metrics.json`。同一目录保留生成音频和逐句分数。相同name会复用已有结果；全新复测改成其他name即可。

单服务客户端串行合成，全量评测和CPU评分耗时较长。此入口于2026-11-02新增，已做语法和参数衔接检查，本次未重新执行3508条全量评测；上表结果来自之前完成的多副本评测。

## 2026-11-02：框架集成与客户补丁交付

本次已将FP32 attention、RoPE、slot mapping和decode图执行迁入vllm-fl源码，将模型导出、speech LLM桥接、前端及Flow/HiFT NPU适配迁入CosyVoice仓库。客户不再依赖本目录外置scripts/service；使用打完补丁的CosyVoice仓库即可运行。

最终补丁目录：`integration/patch_bundle/`，含三个补丁、版本/校验和、应用安装脚本及验证说明。完整安装步骤见其中README.md；支持的三个基线提交由manifest.json固定，不适用于任意最新分支。

在已安装补丁的CosyVoice仓库内：

```bash
export ASCEND_RT_VISIBLE_DEVICES=1
export MODEL_DIR=/models/Fun-CosyVoice3-0.5B-2512
export CACHE_DIR=/workspace/cosyvoice-cache
.venv/bin/python examples/inference_fl.py \
  --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" \
  --reference asset/zero_shot_prompt.wav \
  --prompt-text 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。' \
  --text '你好，欢迎使用语音合成。' --execution decode-graph --output output.wav

.venv/bin/python examples/serve_fl.py \
  --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" \
  --execution decode-graph --max-concurrency 4 --port 50040
```

上面的MODEL_DIR、CACHE_DIR由使用者指定；不要在原50040实例运行时重复绑定端口。现有50040实例未自动切换。原scripts/service仍保留供历史基线和评测兼容，正式客户入口为仓库examples。

干净源码及新venv验证通过：eager、4096/12288 decode图、HTTP并发1/2/4、FP32参数/KV/attention、26请求完整token序列回归。干净源码小样本吞吐0.5882/0.7944/0.9982请求每秒（不同于前面同卡正式基准，不能直接作严格性能结论）。本轮未重跑3508条全量精度，不将前面的历史精度表宣称为最终补丁重新评分结果。详细边界和日志见integration/patch_bundle/VALIDATION.md。

### 旧入口兼容性复测（2026-11-02）

框架集成后再次实际执行原项目入口：scripts/run_tts_npu.sh（默认eager和--llm-execution decode-graph）、scripts/service_npu.sh start、scripts/client_npu.sh及stop，均通过。三次输出均为3.2秒24kHz音频；图模式94 speech token正常stop，95次replay、24层attention捕获。HTTP使用隔离端口50141验证，完成后已停止。

旧入口已设置VLLM_FL_ASCEND_SLOT_MAPPING=torch，并使用FL原生诊断扩展；这是保留的兼容调用方式。客户新交付使用src/CosyVoice/examples，不需要依赖外部scripts/service。历史调试、临时副本和评测脚本不等于全部重新回归；本次也未重跑3508条精度。

integration目录导航见integration/README.md。仅最终patch_bundle或其压缩包用于安装交接，历史integrate_*.py、build_*.py和中间补丁不要批量运行。本次未删除验证目录和备份。

### integration目录清理

2026-11-02按要求删除clean_checkout、customer_cache、install_venv，并清理一次性实施脚本和中间日志。最终补丁与压缩包保留；最终回归证据移至integration/verification/，目录导航见integration/README.md。实际FL源码、src/CosyVoice及原模型/数据集未删除。

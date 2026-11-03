# Ascend 启动、性能与精度验证脚本

这些脚本来自 105 服务器原 CosyVoice 测试目录，保留相对目录结构和原脚本内容，用于理解及复现历史测试。对应仓库根目录 `Summarize_report.md`（2026-10-30）。它们不是 2026-11-03 共存回归中已清理的临时客户端，不能用历史脚本结果替代最新补丁验收。

仅收录 Python / Shell 脚本，包括服务的生成 Python RPC 模块；不包含安装包、虚拟环境、模型、数据、音频、日志、预测、评分结果或编译产物。没有在迁移后重新执行 NPU 精度/性能测试；已检查 Python / Shell 语法。

## 文件导航

|目录 / 入口|用途|
|---|---|
|`scripts/service_npu.sh`、`service/control.py`|历史服务 start / stop / status|
|`service/server.py`、`service/runtime.py`|历史 HTTP 服务及推理流程|
|`scripts/client_npu.sh`、`service/client.py`|请求客户端|
|`service/benchmark_http.py`|固定输入、预热、并发 1/2/4 性能测试|
|`service/validate.py`|服务验证|
|`evaluation/test_accuracy_single.py`|单服务完整 SEED 生成及官方评分|
|`evaluation/prepare_replicas.py`、`expand_replicas.py`|历史多卡副本启动|
|`evaluation/run_accuracy.py`、`score_parallel.py`|多副本精度生成、并行评分|
|`evaluation/finalize_accuracy.py`|汇总历史精度报告|
|`evaluation/throttle_scores.py`|原 Linux 评分进程并发调节|
|`scripts/validate_*.py`、`*asr_check.py`|前端、声学、FP32 与批处理诊断|
|`scripts/run_tts*.py`、`run_tts*.sh`|历史离线推理入口|
|其余 `scripts/*.py`|上述脚本依赖的实验运行时及源码获取工具|

## 使用当前集成服务

新部署优先使用仓库根目录的 `README_FL_ASCEND.md`、`examples/serve_fl.py` 和 `examples/inference_fl.py`。此处的 `fp32_attention.py`、`full_decode_attention.py`、`stage2_backend.py` 等属于历史实验实现，不是当前 FL 原生集成的安装步骤；不要将其全局替换逻辑混入当前生产服务。

## 准备历史目录布局

以下命令在匹配的 Linux Ascend 环境、仓库根目录执行。先按主说明准备依赖及 Matcha 子模块补丁，再显式复用当前源码和 Python 环境：

```bash
R="$PWD"
T="$R/tools/ascend_validation"
mkdir -p "$T/src" "$T/reports" "$T/outputs" "$T/logs" "$T/runtime"
ln -s "$R" "$T/src/CosyVoice"
ln -s "$R/.venv" "$T/.venv"
```

上述链接只需首次创建；已存在时先检查指向。模型及评分资源需另行准备。部分离线检查还需要先前步骤生成的固定输入、参考输出和 ASR 权重；相关文件不随本目录提供。

历史服务入口接受模型路径覆盖；下面端口与性能客户端的固定端口一致：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 bash "$T/scripts/service_npu.sh" start \
  --model /models/Fun-CosyVoice3-0.5B-2512 \
  --http-port 50040 --llm-execution decode-graph --max-concurrency 4
bash "$T/scripts/service_npu.sh" status
"$T/.venv/bin/python" "$T/service/benchmark_http.py" --label fl_decode_graph --concurrency 1,2,4 --rounds 2
bash "$T/scripts/service_npu.sh" stop
```

性能客户端读取本地服务生成的请求 JSON，要求与服务共享此目录的 `outputs/service`；它不是仅依赖远端 HTTP 的通用基准客户端。不要直接用它测另一个不写此类诊断的服务。

## 精度评分依赖

历史精度脚本从同级 `CosyVoice_ascend/evaluation` 复用官方评分包装、数据及 Python 环境。迁移后同级位置是 `tools/CosyVoice_ascend/evaluation`，可将原资源目录链接到那里；需包含 `generate_full_decode.py`、`score_full_decode.py`、`run_official_metric.py`、相应导入模块、数据、`models`、`repos` 和 `.venv-cpu`。这些外部资源未打包。

准备好外部评分目录及待测服务后，可先使用单服务入口：

```bash
"$T/.venv/bin/python" "$T/evaluation/test_accuracy_single.py" --port 50040 --name single_fp32
```

多副本脚本保留原卡号、端口、上下文和历史源码替换逻辑；部分替换可能已被当前源码包含，运行前应逐项核对，不能直接整套执行。`throttle_scores.py` 会对匹配的评分进程发出暂停/恢复信号；生成/汇总脚本会写报告，副本流程会启停服务并清理临时目录。它们依赖 Linux `/proc`，不适合直接在 macOS 上运行。

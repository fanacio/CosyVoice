# CosyVoice3 on FL Ascend

Validated base: CosyVoice 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc; Matcha-TTS dd9105b34bf2be2230f4aa1e4769fb586a3c824e; vllm-plugin-FL feature/v024-ascend-platform-runtime with the Ascend FP32 integration. Python3.12, torch2.10.0+cpu, torch-npu2.10.0.post2, vLLM0.24.0, CANN9.0.1; aarch64 Ascend910_9391.

Use the matching FL container with the Ascend FP32 integration installed. This branch already includes the CosyVoice integration. Initialize Matcha-TTS and apply the bundled inference dependency patch before installing:

```bash
git submodule update --init third_party/Matcha-TTS
git -C third_party/Matcha-TTS apply --check ../../patches/ascend/matcha-inference-dependencies.patch
git -C third_party/Matcha-TTS apply ../../patches/ascend/matcha-inference-dependencies.patch
```

Run the application step once on a clean submodule. The patch defers training-only imports and permits inference without Lightning; the submodule pointer remains unchanged. Keep the image's torch, torch-npu and vLLM versions; do not install the upstream full CUDA requirements over this image.

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps -r requirements-fl-ascend.txt
```

Prepare the Fun-CosyVoice3-0.5B-2512 model separately. Model weights, evaluation data and compiled FL operators are not bundled. Existing Ascend OPP build requirements remain necessary for a fresh FL installation. Set your paths:

```bash
export MODEL_DIR=/models/Fun-CosyVoice3-0.5B-2512
export CACHE_DIR=/workspace/cosyvoice-cache
export ASCEND_RT_VISIBLE_DEVICES=1
```

Offline inference, from this repository root:

```bash
.venv/bin/python examples/inference_fl.py \
  --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" \
  --reference asset/zero_shot_prompt.wav \
  --prompt-text 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。' \
  --text '你好，欢迎使用语音合成。' --execution decode-graph --output output.wav
```

Use --execution eager for eager execution. Default context4096; --max-model-len12288 covers the previously used full SEED hardcase set, at higher graph compute/memory cost. Neural frontend, Flow/HiFT and LLM execute on NPU; tokenization, file I/O and orchestration use CPU.

HTTP service:

```bash
.venv/bin/python examples/serve_fl.py \
  --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" \
  --execution decode-graph --max-concurrency 4 --port 50040
```

```bash
curl http://127.0.0.1:50040/healthz
curl -X POST http://127.0.0.1:50040/inference_zero_shot \
  -F 'tts_text=你好，欢迎使用语音合成。' \
  -F 'prompt_text=You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。' \
  -F 'prompt_wav=@asset/zero_shot_prompt.wav' --output output.pcm
```

HTTP returns mono24kHz signed16-bit little-endian PCM, not a WAV container. See /docs for zero_shot, cross_lingual and instruct2. This is non-streaming synthesis. One model/one LLM owner thread; at most four in-flight requests, excess503. Frontend and acoustic stages serialize separately while LLM requests may batch. Stop with Ctrl-C or SIGTERM; do not run multiple uvicorn workers.

Examples set VLLM_VENDOR=ascend, VLLM_PLUGINS=fl, USE_FLAGGEMS=0 and VLLM_FL_ASCEND_SLOT_MAPPING=torch before imports. FL owns attention/RoPE/KV and graph implementations; this repository owns speech-model orchestration. Reference implementations currently support FP32, TP=1, CP=1, no quantization/sliding window; BF16/FP16 keep original FL paths. Decode graph covers model forward including attention; prefill/scheduler/sampling/acoustic stages are outside that graph.

CosyVoice entry point: AutoModel(model_dir=..., device='npu', cache_dir=..., execution='decode-graph'); close() releases the engine. No dependency on the external experimental scripts/service directories. Model-specific subclasses are explicit; per-instance acoustic wrappers collect timing diagnostics. Kaldi fbank uses an isolated functional namespace to create window constants on CPU without modifying torchaudio globals.

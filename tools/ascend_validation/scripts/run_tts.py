"""Run official CosyVoice3 components with the current vLLM Ascend backend."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback


def export_original_fp32(source_model: Path, destination: Path):
    """Export original FP32 checkpoint tensors, without a BF16 round trip."""
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen2Config

    manifest = destination / "export_manifest.json"
    source = source_model / "llm.pt"
    identity = {"source": str(source.resolve()), "source_bytes": source.stat().st_size,
                "source_mtime_ns": source.stat().st_mtime_ns, "dtype": "float32"}
    if destination.exists():
        if not manifest.exists() or json.loads(manifest.read_text()).get("identity") != identity:
            raise RuntimeError(f"Preserving unverified existing export: {destination}")
        return
    weights = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    exported = {name.removeprefix("llm.model."): tensor for name, tensor in weights.items()
                if name.startswith(("llm.model.model.layers.", "llm.model.model.norm."))}
    exported["model.embed_tokens.weight"] = weights["speech_embedding.weight"]
    exported["lm_head.weight"] = weights["llm_decoder.weight"]
    if any(t.dtype != torch.float32 for t in exported.values()):
        raise RuntimeError("Source checkpoint is not original FP32; refusing to promote rounded weights")
    if exported["lm_head.weight"].shape != (6761, 896):
        raise RuntimeError("Unexpected speech head")
    staging = destination.with_name(destination.name + f".staging-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        config = Qwen2Config.from_pretrained(str(source_model / "CosyVoice-BlankEN"), local_files_only=True)
        config.architectures = ["Qwen2ForCausalLM"]
        config.vocab_size = 6761
        config.tie_word_embeddings = False
        config.dtype = torch.float32
        config.bos_token_id = None
        config.eos_token_id = 6562
        config.pad_token_id = None
        config.save_pretrained(staging)
        (staging / "generation_config.json").write_text(json.dumps({"eos_token_id": 6562}))
        save_file({k: v.contiguous() for k, v in exported.items()}, staging / "model.safetensors")
        (staging / "export_manifest.json").write_text(json.dumps({"identity": identity,
            "tensor_count": len(exported), "all_source_tensors_float32": True,
            "mapping": "original Qwen2 layers/norm plus speech_embedding and llm_decoder"}, indent=2))
        staging.rename(destination)
    except Exception:
        # Retain partial output as evidence rather than replacing an existing model.
        raise


def prepare_model(root: Path, source_model: Path) -> Path:
    import yaml

    destination = root / "runtime/model"
    destination.mkdir(parents=True, exist_ok=True)
    text = (source_model / "cosyvoice3.yaml").read_text()
    node = yaml.compose(text)
    lines = text.splitlines(keepends=True)
    needed = {
        "__set_seed1", "__set_seed2", "__set_seed3", "__set_seed4",
        "sample_rate", "llm_input_size", "llm_output_size", "spk_embed_dim",
        "qwen_pretrain_path", "token_frame_rate", "token_mel_ratio",
        "chunk_size", "num_decoding_left_chunks", "llm", "flow", "hift",
        "get_tokenizer", "allowed_special", "feat_extractor",
    }
    blocks = []
    found = set()
    for index, (key, value) in enumerate(node.value):
        if key.value not in needed:
            continue
        start = key.start_mark.line
        end = node.value[index + 1][0].start_mark.line if index + 1 < len(node.value) else len(lines)
        blocks.append("".join(lines[start:end]))
        found.add(key.value)
    if needed - found:
        raise RuntimeError(f"Missing inference YAML keys: {needed - found}")
    (destination / "cosyvoice3.yaml").write_text("".join(blocks))
    for name in ["llm.pt", "flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN"]:
        link = destination / name
        target = source_model / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"Unexpected symlink: {link}")
        elif link.exists():
            raise RuntimeError(f"Preserving existing file: {link}")
        else:
            link.symlink_to(target, target_is_directory=target.is_dir())
    return destination


def main(on_ready=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", type=Path, default=Path("/public-flash/models/q4_models/iic/Fun-CosyVoice3-0.5B-2512"))
    parser.add_argument("--text", default="你好，欢迎使用语音合成。")
    parser.add_argument("--prompt-text", default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--memory-utilization", type=float, default=0.15)
    parser.add_argument("--llm-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--llm-execution", choices=["eager", "piecewise", "decode-graph"], default="eager")
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()
    os.environ["COSY_GRAPH_CONTEXT"] = str(args.max_model_len)
    if not 1 <= args.max_num_seqs <= 4:
        parser.error("max-num-seqs must be 1..4")
    root = args.root.resolve()
    source = root / "src/CosyVoice"
    sys.path[:0] = [str(source), str(source / "third_party/Matcha-TTS")]
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(2)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)

    # torchaudio >= 2.9 delegates load() to TorchCodec. Use the already
    # available soundfile backend for the official mono/resampling logic.
    from cosyvoice.utils import file_utils

    def load_wav(path, target_sr, min_sr=16000):
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        speech = torch.from_numpy(audio.T.copy()).mean(dim=0, keepdim=True)
        if sr != target_sr:
            if sr < min_sr:
                raise ValueError(f"Reference sample rate {sr} < {min_sr}")
            speech = torchaudio.transforms.Resample(sr, target_sr)(speech)
        return speech

    file_utils.load_wav = load_wav
    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
    from cosyvoice.llm.llm import Qwen2LM
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.platforms import current_platform

    if current_platform.device_type != "npu":
        raise RuntimeError(f"Expected NPU, got {current_platform.device_type}")

    llm_dtype = torch.float32 if args.llm_dtype == "float32" else torch.bfloat16
    export = root / "exported" / ("cosyvoice3_llm_fp32" if llm_dtype == torch.float32 else "cosyvoice3_llm_bf16")
    export.parent.mkdir(parents=True, exist_ok=True)
    engines = []
    if on_ready is not None:
        from batched_engine import BatchedEngine, RequestRecords
        requests = RequestRecords()
    else:
        requests = []
    thread_errors = []
    precision_reports = []
    original_llm_job = CosyVoiceModel.llm_job

    def checked_llm_job(model, *job_args, **job_kwargs):
        bound = inspect.signature(original_llm_job).bind(model, *job_args, **job_kwargs)
        request_id = bound.arguments["uuid"]
        try:
            return original_llm_job(model, *job_args, **job_kwargs)
        except Exception:
            thread_errors.append(traceback.format_exc())
            traceback.print_exc()
        finally:
            model.llm_end_dict[request_id] = True

    def load_vllm(model, _default_export_path):
        # Retain official BF16 export. Strict FP32 uses original tensors so
        # no rounding from the earlier BF16 export can enter the new model.
        if llm_dtype == torch.float32:
            export_original_fp32(args.model, export)
            if args.llm_execution == "decode-graph":
                os.environ["COSY_FULL_DECODE_GRAPH"] = "1"
            # FP32 attention is implemented by the installed FL backend.
            # Registers a process-local strict FP32 attention path.
        else:
            file_utils.export_cosyvoice2_vllm(model.llm, str(export), model.device)
        config = json.loads((export / "config.json").read_text())
        if config["vocab_size"] != 6761 or config["tie_word_embeddings"]:
            raise RuntimeError("Unexpected exported speech vocabulary/head")
        engine = LLMEngine.from_engine_args(EngineArgs(
            model=str(export), dtype=args.llm_dtype, tensor_parallel_size=1,
            skip_tokenizer_init=True, enable_prompt_embeds=True,
            enforce_eager=args.llm_execution == "eager", enable_prefix_caching=False,
            **({"compilation_config": {"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY" if args.llm_execution == "decode-graph" else "PIECEWISE", "cudagraph_capture_sizes": [1, 2, 3, 4]}} if args.llm_execution != "eager" else {}),
            enable_chunked_prefill=True, block_size=128, max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs, max_num_batched_tokens=4096,
            gpu_memory_utilization=args.memory_utilization,
            **({"worker_extension_cls": "vllm_fl.dispatch.backends.vendor.ascend.impl.fp32_diagnostics.FP32Diagnostics"} if llm_dtype == torch.float32 else {}),
        ))
        config_report = {"requested_execution": args.llm_execution, "compilation": str(engine.vllm_config.compilation_config)}
        (root / "reports/llm_execution.json").write_text(json.dumps(config_report, indent=2))
        if on_ready is not None:
            engine = BatchedEngine(engine)
        model.llm.vllm = engine
        model.llm.lock = threading.Lock()
        engines.append(engine)
        del model.llm.llm.model.model.layers
        print("VLLM_ASCEND_ENGINE_READY", flush=True)

    @torch.inference_mode()
    def inference_wrapper(model, lm_input, sampling, min_len, max_len, uuid):
        engine = model.vllm
        params = SamplingParams(
            top_k=sampling, stop_token_ids=model.stop_token_ids,
            min_tokens=int(min_len), max_tokens=int(max_len),
            detokenize=False, seed=args.seed,
        )
        prompt = lm_input.squeeze(0).to(llm_dtype).cpu().contiguous()
        started = time.monotonic()
        consumed = 0
        generated = []
        finished = False
        record = {"request_id": uuid, "prompt_shape": list(lm_input.shape), "prompt_dtype": str(prompt.dtype)}
        def results():
            if on_ready is not None:
                yield from engine.generate(uuid, {"prompt_embeds": prompt}, params, args.request_timeout)
            else:
                engine.add_request(uuid, {"prompt_embeds": prompt}, params)
                while engine.has_unfinished_requests():
                    if time.monotonic() - started > args.request_timeout:
                        raise TimeoutError(f"LLM request exceeded {args.request_timeout}s")
                    yield from engine.step()

        try:
            for result in results():
                if result.request_id != uuid:
                    raise RuntimeError("Engine output belongs to a different request")
                output = result.outputs[0]
                ids = list(output.token_ids)
                for token in ids[consumed:]:
                    if token in model.stop_token_ids:
                        continue
                    if not 0 <= token < model.speech_token_size:
                        raise RuntimeError(f"Invalid speech token {token}")
                    generated.append(token)
                    yield token
                consumed = len(ids)
                finished = result.finished
                record["finish_reason"] = output.finish_reason
            if not finished:
                raise RuntimeError("Engine request disappeared without a final output")
            record.update(tokens=generated, token_count=len(generated), llm_seconds=time.monotonic() - started)
            requests.append(record)
            print("VLLM_SPEECH_TOKENS", len(generated), "FINISH", record["finish_reason"], flush=True)
        finally:
            if not finished:
                engine.abort_request(uuid)

    CosyVoice2Model.load_vllm = load_vllm
    CosyVoiceModel.llm_job = checked_llm_job
    Qwen2LM.inference_wrapper = inference_wrapper
    stage = prepare_model(root, args.model)
    reference = args.reference or source / "asset/zero_shot_prompt.wav"
    output = args.output or root / ("outputs/fp32/cosyvoice3_ascend.wav" if llm_dtype == torch.float32 else "outputs/cosyvoice3_ascend.wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        tts = AutoModel(model_dir=str(stage), load_vllm=True, load_trt=False, fp16=False)
        loaded = time.monotonic()
        print("NATIVE_COSYVOICE_READY", "audio_device", str(tts.model.device), flush=True)
        if on_ready is not None:
            # A persistent server owns requests until it returns; engine shutdown
            # still runs through this function's finally block.
            on_ready(tts=tts, engines=engines, requests=requests,
                     thread_errors=thread_errors, args=args,
                     load_seconds=loaded - started)
            return
        chunks = []
        for result in tts.inference_zero_shot(args.text, args.prompt_text, str(reference), stream=False, text_frontend=False):
            chunks.append(result["tts_speech"].detach().float().cpu())
        if thread_errors:
            raise RuntimeError("Native LLM thread failed:\n" + "\n".join(thread_errors))
        if not chunks or not requests:
            raise RuntimeError("No audio or no actual vLLM-generated tokens")
        if llm_dtype == torch.float32:
            for engine in engines:
                precision_reports.extend(engine.collective_rpc("fp32_dtype_report"))
            if any(set(p["floating_parameter_dtypes"]) != {"torch.float32"} or
                   set(p["kv_cache_dtypes"]) != {"torch.float32"} or p["matmul_hf32"] or
                   p["attention"]["calls"] == 0 or p["attention"]["dtypes"] != ["torch.float32"]
                   for p in precision_reports):
                raise RuntimeError(f"Strict FP32 validation failed: {precision_reports}")
        audio = torch.cat(chunks, dim=-1).squeeze(0).numpy()
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-5:
            raise RuntimeError("Invalid or silent output")
        sf.write(output, audio, tts.sample_rate, subtype="PCM_16")
        report = {
            "text": args.text, "prompt_text": args.prompt_text,
            "reference": str(reference), "output": str(output),
            "sample_rate": tts.sample_rate, "samples": len(audio),
            "audio_seconds": len(audio) / tts.sample_rate,
            "load_seconds": loaded - started,
            "synthesis_seconds": time.monotonic() - loaded,
            "llm_backend": "vllm-fl/ascend", "device_type": current_platform.device_type,
            "llm_dtype": str(llm_dtype), "llm_export": str(export), "llm_precision_reports": precision_reports,
            "ascend_visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
            "flow_device": str(tts.model.device), "hift_device": str(tts.model.device),
            "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "requests": requests,
        }
        report["rtf"] = report["synthesis_seconds"] / report["audio_seconds"]
        output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print("TTS_SUCCESS", json.dumps({k: v for k, v in report.items() if k != "requests"}, ensure_ascii=False), flush=True)
    finally:
        for engine in engines:
            with contextlib.suppress(Exception):
                if on_ready is not None:
                    engine.shutdown()
                else:
                    engine.engine_core.shutdown()


if __name__ == "__main__":
    main()

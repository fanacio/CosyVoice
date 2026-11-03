import os
from pathlib import Path
import json
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

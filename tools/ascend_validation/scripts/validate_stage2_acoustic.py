"""Compare fixed-input native CPU and NPU Flow/HiFT outputs."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/CosyVoice"
sys.path[:0] = [str(SRC), str(SRC / "third_party/Matcha-TTS")]

from stage2_backend import configure_npu, load_audio, prepare_acoustic_models, npu_f02sine


def metrics(reference, actual):
    a, b = reference.detach().cpu().double().flatten(), actual.detach().cpu().double().flatten()
    assert a.shape == b.shape
    difference = b - a
    return {"max_abs": difference.abs().max().item(), "rmse": difference.square().mean().sqrt().item(),
            "relative_l2": (difference.norm() / a.norm().clamp_min(1e-12)).item(),
            "cosine": torch.nn.functional.cosine_similarity(a[None], b[None]).item()}


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    torch.manual_seed(1986)
    np.random.seed(1986)
    device = configure_npu()
    from cosyvoice.utils import file_utils
    file_utils.load_wav = load_audio
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd
    from hyperpyyaml import load_hyperpyyaml
    from run_tts import prepare_model

    model = Path("/public-flash/models/q4_models/iic/Fun-CosyVoice3-0.5B-2512")
    stage = prepare_model(ROOT, model)
    with (stage / "cosyvoice3.yaml").open() as handle:
        config = load_hyperpyyaml(handle, overrides={"llm": None, "qwen_pretrain_path": str(model / "CosyVoice-BlankEN")})
    flow, hift = config["flow"], config["hift"]
    flow.load_state_dict(torch.load(model / "flow.pt", map_location="cpu", weights_only=True), strict=True)
    hift.load_state_dict({k.replace("generator.", ""): v for k, v in torch.load(model / "hift.pt", map_location="cpu", weights_only=True).items()}, strict=True)
    flow.eval(); hift.eval()
    frontend = CosyVoiceFrontEnd(config["get_tokenizer"], config["feat_extractor"], str(model / "campplus.onnx"), str(model / "speech_tokenizer_v3.onnx"), allowed_special=config["allowed_special"])
    report = json.loads((ROOT / "outputs/cosyvoice3_ascend.json").read_text())
    front = frontend.frontend_zero_shot(report["text"], report["prompt_text"], report["reference"], 24000, "")
    silent = {1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323}
    tokens, streak = [], 0
    for token in report["requests"][0]["tokens"]:
        streak = streak + 1 if token in silent else 0
        if streak <= 5:
            tokens.append(token)
    arguments = {"token": torch.tensor([tokens], dtype=torch.int32), "token_len": torch.tensor([len(tokens)], dtype=torch.int32),
                 "prompt_token": front["flow_prompt_speech_token"], "prompt_token_len": front["flow_prompt_speech_token_len"],
                 "prompt_feat": front["prompt_speech_feat"], "prompt_feat_len": front["prompt_speech_feat_len"],
                 "embedding": front["flow_embedding"], "streaming": False, "finalize": True}
    out = ROOT / "outputs/stage2"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"frontend": front, "arguments": arguments}, out / "fixed_inputs.pt")
    timings = {}

    def timed(name, function):
        torch.npu.synchronize()
        started = time.monotonic()
        result = function()
        torch.npu.synchronize()
        timings[name] = time.monotonic() - started
        print("TIMING", name, timings[name], flush=True)
        return result

    mel_cpu = timed("flow_cpu", lambda: flow.inference(**arguments)[0])
    wave_cpu = timed("hift_cpu", lambda: hift.inference(mel_cpu, finalize=True)[0])
    f0_cpu = hift.f0_predictor(mel_cpu.double(), finalize=True).float()
    source = hift.f0_upsamp(f0_cpu[:, None]).transpose(1, 2)
    harmonics = torch.arange(1, hift.nb_harmonics + 2).float()[None, None]
    fn = source * harmonics
    sine = hift.m_source.l_sin_gen
    phase_reference = sine._f02sine(fn)
    phase_replacement = npu_f02sine(sine, fn)
    phase_metrics = metrics(phase_reference, phase_replacement)
    (ROOT / "reports/stage2_interpolation_parity.json").write_text(json.dumps(phase_metrics, indent=2))
    assert torch.allclose(phase_reference, phase_replacement, atol=1e-5, rtol=1e-5), phase_metrics
    torch.save({"mel": mel_cpu, "wave": wave_cpu, "f0": f0_cpu}, out / "cpu_acoustic_reference.pt")
    settings = prepare_acoustic_models(flow, hift, device)
    args_npu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in arguments.items()}
    mel_npu = timed("flow_npu_cold", lambda: flow.inference(**args_npu)[0])
    mel_npu = timed("flow_npu_warm", lambda: flow.inference(**args_npu)[0])
    flow_metrics = metrics(mel_cpu, mel_npu)
    torch.save(mel_npu.cpu(), out / "npu_flow_mel.pt")
    (ROOT / "reports/stage2_flow_validation.json").write_text(json.dumps({"metrics": flow_metrics, "timings": timings}, indent=2))
    f0_npu = hift.f0_predictor(mel_cpu.to(device), finalize=True)
    f0_metrics = metrics(f0_cpu, f0_npu)
    (ROOT / "reports/stage2_f0_validation.json").write_text(json.dumps(f0_metrics, indent=2))
    wave_npu = timed("hift_npu_same_mel", lambda: hift.inference(mel_cpu.to(device), finalize=True)[0])
    end_wave = timed("hift_npu_on_npu_mel", lambda: hift.inference(mel_npu, finalize=True)[0])
    validation = {"settings": settings, "timings_seconds": timings, "flow": flow_metrics, "phase_interpolation": phase_metrics,
                  "f0_fp64_cpu_vs_fp32_npu": f0_metrics, "hift_same_mel": metrics(wave_cpu, wave_npu),
                  "whole_acoustic": metrics(wave_cpu, end_wave), "audio_seconds": end_wave.shape[-1] / 24000}
    assert torch.isfinite(end_wave).all() and end_wave.abs().max() > 1e-5
    sf.write(out / "fixed_tokens_npu.wav", end_wave[0].float().cpu().numpy(), 24000, subtype="PCM_16")
    torch.save({"mel": mel_npu.cpu(), "wave": end_wave.cpu()}, out / "npu_acoustic_output.pt")
    (ROOT / "reports/stage2_acoustic_validation.json").write_text(json.dumps(validation, indent=2))
    print("ACOUSTIC_NPU_VALIDATED", json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()

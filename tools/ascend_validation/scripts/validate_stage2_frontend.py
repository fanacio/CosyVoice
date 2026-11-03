"""Compare frontend NPU outputs to the saved native ONNX CPU baseline."""
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/CosyVoice"
sys.path[:0] = [str(SRC), str(SRC / "third_party/Matcha-TTS")]

from stage2_backend import configure_npu, load_audio
from stage2_frontend import make_npu_frontend_class, npu_whisper_mel
from validate_stage2_acoustic import metrics


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    device = configure_npu()
    from cosyvoice.utils import file_utils
    file_utils.load_wav = load_audio
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd
    from hyperpyyaml import load_hyperpyyaml
    from run_tts import prepare_model
    import whisper

    model = Path("/public-flash/models/q4_models/iic/Fun-CosyVoice3-0.5B-2512")
    stage = prepare_model(ROOT, model)
    with (stage / "cosyvoice3.yaml").open() as handle:
        config = load_hyperpyyaml(handle, overrides={"llm": None, "flow": None, "hift": None,
                                                   "qwen_pretrain_path": str(model / "CosyVoice-BlankEN")})
    first = json.loads((ROOT / "outputs/cosyvoice3_ascend.json").read_text())
    cpu = torch.load(ROOT / "outputs/stage2/fixed_inputs.pt", map_location="cpu", weights_only=True)["frontend"]
    cls = make_npu_frontend_class(CosyVoiceFrontEnd, ROOT, device)
    frontend = cls(config["get_tokenizer"], config["feat_extractor"], str(model / "campplus.onnx"),
                   str(model / "speech_tokenizer_v3.onnx"), allowed_special=config["allowed_special"])
    wave_cpu = load_audio(first["reference"], 16000)
    wave_npu = load_audio(first["reference"], 16000, device=device)
    reference_mel = whisper.log_mel_spectrogram(wave_cpu, n_mels=128)
    npu_mel = npu_whisper_mel(wave_npu, n_mels=128)
    partial = {"whisper_mel": metrics(reference_mel, npu_mel), "resampled_wave": metrics(wave_cpu, wave_npu)}
    (ROOT / "reports/stage2_frontend_features.json").write_text(json.dumps(partial, indent=2))
    torch.npu.synchronize()
    start = time.monotonic()
    npu = frontend.frontend_zero_shot(first["text"], first["prompt_text"], first["reference"], 24000, "")
    torch.npu.synchronize()
    report = {"features": partial, "frontend_seconds": time.monotonic() - start,
              "neural_seconds": frontend.component_timings, "execution": frontend.neural_execution, "comparisons": {}}
    for name, value in cpu.items():
        actual = npu[name].detach().cpu()
        if value.is_floating_point():
            report["comparisons"][name] = metrics(value, actual)
        else:
            assert value.shape == actual.shape, (name, value.shape, actual.shape)
            report["comparisons"][name] = {"equal": bool(torch.equal(value, actual)),
                                            "different_elements": int((value != actual).sum()), "elements": value.numel()}
    torch.save({k: v.cpu() for k, v in npu.items()}, ROOT / "outputs/stage2/npu_frontend_output.pt")
    (ROOT / "reports/stage2_frontend_validation.json").write_text(json.dumps(report, indent=2))
    print("FRONTEND_NPU_RESULT", json.dumps(report), flush=True)
    assert report["comparisons"]["llm_prompt_speech_token"]["equal"], "Speech tokenizer needs further parity investigation"
    assert report["comparisons"]["flow_embedding"]["cosine"] > 0.999, "Speaker embedding mismatch"


if __name__ == "__main__":
    main()

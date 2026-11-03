"""Project-local Ascend adapters; original checkpoints remain unchanged."""
from __future__ import annotations

import functools
import math
from pathlib import Path
import types

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


def configure_npu():
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    return torch.device("npu:0")


def load_audio(path, target_sr, min_sr=16000, device="cpu"):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.T.copy()).to(device).mean(0, keepdim=True)
    if sr != target_sr:
        if sr < min_sr:
            raise ValueError(f"Reference sample rate {sr} < {min_sr}")
        wave = torchaudio.transforms.Resample(sr, target_sr).to(device)(wave)
    return wave


def remove_weight_norm_for_inference(model):
    from torch.nn.utils import remove_weight_norm
    from torch.nn.utils.weight_norm import WeightNorm
    count = 0
    for module in model.modules():
        for hook in list(module._forward_pre_hooks.values()):
            if isinstance(hook, WeightNorm):
                remove_weight_norm(module, name=hook.name)
                count += 1
    return count


def move_fixed_noise(model, device):
    names = {"rand_noise", "stft_window", "rand_ini", "sine_waves", "uv"}
    for module in model.modules():
        for name in names:
            value = vars(module).get(name)
            if isinstance(value, torch.Tensor):
                setattr(module, name, value.to(device))


def linear_resize_last(tensor, scale):
    """align_corners=False 1-D linear interpolation using NPU gathers."""
    count = math.floor(tensor.shape[-1] * scale)
    position = (torch.arange(count, device=tensor.device, dtype=torch.float32) + 0.5) / scale - 0.5
    position = position.clamp_min(0)
    left = position.floor().to(torch.int64)
    right = (left + 1).clamp_max(tensor.shape[-1] - 1)
    fraction = position - left.to(position.dtype)
    return tensor.index_select(-1, left) * (1 - fraction) + tensor.index_select(-1, right) * fraction


def npu_f02sine(model, f0_values):
    if model.flag_for_pulse or not model.causal or model.training:
        raise NotImplementedError("This adapter implements the trained causal inference branch")
    rad = (f0_values / model.sampling_rate) % 1
    rad[:, 0, :] = rad[:, 0, :] + model.rand_ini.to(rad.device)
    rad = linear_resize_last(rad.transpose(1, 2), 1 / model.upsample_scale).transpose(1, 2)
    phase = torch.cumsum(rad, dim=1) * 2 * math.pi
    phase = (phase * model.upsample_scale).repeat_interleave(int(model.upsample_scale), dim=1)
    return torch.sin(phase)


@torch.inference_mode()
def npu_hift_inference(model, speech_feat, finalize=True):
    # Ascend convolution supports FP32/FP16/BF16, not the original FP64 F0
    # convolution. This change is separately checked against the CPU FP64 path.
    model.f0_predictor.float()
    f0 = model.f0_predictor(speech_feat.float(), finalize=finalize).to(speech_feat)
    source = model.f0_upsamp(f0[:, None]).transpose(1, 2)
    source, _, _ = model.m_source(source)
    source = source.transpose(1, 2)
    feature = speech_feat if finalize else speech_feat[:, :, :-model.f0_predictor.condnet[0].causal_padding]
    return model.decode(x=feature, s=source, finalize=finalize), source


def prepare_acoustic_models(flow, hift, device):
    flow.float().eval()
    hift.float().eval()
    removed = remove_weight_norm_for_inference(hift)
    flow.to(device)
    hift.to(device)
    move_fixed_noise(flow, device)
    move_fixed_noise(hift, device)
    hift.inference = types.MethodType(npu_hift_inference, hift)
    hift.m_source.l_sin_gen._f02sine = types.MethodType(npu_f02sine, hift.m_source.l_sin_gen)
    return {"weight_norms_folded": removed, "flow_device": str(next(flow.parameters()).device),
            "hift_device": str(next(hift.parameters()).device), "f0_dtype": "float32",
            "phase_interpolation": "equivalent_gather_and_repeat", "spectral_backend": "native_npu_fft"}


def install_full_npu_pipeline(root):
    """Apply process-local device adapters without modifying framework sources."""
    import time
    from cosyvoice.cli import frontend as frontend_module
    from cosyvoice.cli import cosyvoice as cli_module
    from cosyvoice.cli.model import CosyVoice3Model, CosyVoiceModel
    from stage2_frontend import make_npu_frontend_class

    device = configure_npu()
    state = {"device": str(device)}
    cls = make_npu_frontend_class(frontend_module.CosyVoiceFrontEnd, root, device)
    frontend_module.CosyVoiceFrontEnd = cls
    if hasattr(cli_module, "CosyVoiceFrontEnd"):
        cli_module.CosyVoiceFrontEnd = cls

    original_init = CosyVoice3Model.__init__
    original_load = CosyVoiceModel.load
    original_job = CosyVoiceModel.llm_job
    original_cli_init = cli_module.CosyVoice3.__init__

    @functools.wraps(original_init)
    def initialize(model, *args, **kwargs):
        original_init(model, *args, **kwargs)
        model.device = device

    @functools.wraps(original_load)
    def load(model, *args, **kwargs):
        # Build/load the same native modules on CPU, then transfer them once.
        model.device = torch.device("cpu")
        try:
            original_load(model, *args, **kwargs)
        finally:
            model.device = device
        model.llm.to(device).eval()
        state["acoustic_settings"] = prepare_acoustic_models(model.flow, model.hift, device)

        def instrument(name, module):
            method = module.inference

            @functools.wraps(method)
            def wrapped(*call_args, **call_kwargs):
                inputs = list(call_args) + list(call_kwargs.values())
                tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
                if any(x.device.type != "npu" for x in tensors):
                    raise RuntimeError(f"CPU tensor input to {name}")
                torch.npu.synchronize()
                start = time.monotonic()
                result = method(*call_args, **call_kwargs)
                torch.npu.synchronize()
                output = result[0] if isinstance(result, (tuple, list)) else result
                if output.device.type != "npu":
                    raise RuntimeError(f"CPU output from {name}")
                state.setdefault(name, []).append({"seconds": time.monotonic() - start,
                    "input_devices": [str(x.device) for x in tensors], "output_device": str(output.device),
                    "output_shape": list(output.shape), "dtype": str(output.dtype)})
                return result

            module.inference = wrapped

        instrument("flow", model.flow)
        instrument("hift", model.hift)

    @functools.wraps(original_job)
    def llm_job(model, *args, **kwargs):
        torch.npu.set_device(device)
        return original_job(model, *args, **kwargs)

    @functools.wraps(original_cli_init)
    def cli_initialize(model, *args, **kwargs):
        original_cli_init(model, *args, **kwargs)
        state["frontend_object"] = model.frontend
        torch.npu.synchronize()
        torch.npu.empty_cache()
        print("FULL_NPU_COMPONENTS_READY", state["acoustic_settings"], flush=True)

    CosyVoice3Model.__init__ = initialize
    CosyVoiceModel.load = load
    CosyVoiceModel.llm_job = llm_job
    cli_module.CosyVoice3.__init__ = cli_initialize
    return state

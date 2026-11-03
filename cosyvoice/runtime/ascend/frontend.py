"""CosyVoice frontend neural networks converted faithfully from official ONNX."""
from __future__ import annotations

import functools
import json
from pathlib import Path
import time

import torch

from .acoustic import load_audio


class GreaterOrEqual(torch.nn.Module):
    def forward(self, first, second):
        return torch.ge(first, second)


def register_onnx_extensions():
    from onnx2torch.node_converters.registry import add_converter, get_converter
    from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node
    try:
        get_converter("GreaterOrEqual", 16)
    except NotImplementedError:
        @add_converter(operation_type="GreaterOrEqual", version=16)
        def convert_greater_or_equal(node, graph):
            return OperationConverterResult(torch_module=GreaterOrEqual(), onnx_mapping=onnx_mapping_from_node(node=node))


def converted_model(path, root, device):
    from onnx2torch import convert
    register_onnx_extensions()
    path = Path(path)
    cache = root / "exported/frontend_npu" / (path.stem + ".torch.pt")
    cache.parent.mkdir(parents=True, exist_ok=True)
    identity = {"path": str(path.resolve()), "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns, "onnx2torch": "1.5.15", "adapter_version": 1}
    metadata = cache.with_suffix(".json")
    if cache.exists() and metadata.exists() and json.loads(metadata.read_text()) == identity:
        # Cache contains trusted graph code generated here from the official ONNX.
        model = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        print("CONVERTING_OFFICIAL_ONNX", path.name, flush=True)
        model = convert(str(path)).eval()
        torch.save(model, cache)
        metadata.write_text(json.dumps(identity, indent=2))
    model.eval().to(device)
    tensors = list(model.parameters()) + list(model.buffers())
    bad = [(str(t.device), str(t.dtype), list(t.shape)) for t in tensors if t.is_floating_point() and t.device != device]
    if bad:
        raise RuntimeError(f"CPU neural weights in {path.name}: {bad[:5]}")
    return model


def npu_whisper_mel(wave, n_mels=128):
    import whisper.audio as audio
    # Generate the constant window on CPU: this NPU stack's hann_window
    # creation otherwise attempts an integer cosine. The STFT runs on NPU.
    window = torch.hann_window(audio.N_FFT, dtype=torch.float32).to(wave.device)
    stft = torch.stft(wave.float(), audio.N_FFT, audio.HOP_LENGTH, window=window, return_complex=True)
    power = stft[..., :-1].abs().pow(2)
    mel = audio.mel_filters(wave.device, n_mels) @ power
    logged = mel.clamp_min(1e-10).log10()
    logged = torch.maximum(logged, logged.max() - 8.0)
    return (logged + 4.0) / 4.0


def npu_fbank(*args, **kwargs):
    # Clone the functional namespace rather than modifying torchaudio globals.
    import types
    import torchaudio.compliance.kaldi as kaldi
    namespace = dict(vars(kaldi))
    def window(window_type, window_size, blackman_coeff, device, dtype):
        return kaldi._feature_window_function(window_type, window_size, blackman_coeff,
            torch.device("cpu"), dtype).to(device)
    namespace['_feature_window_function'] = window
    original = kaldi._get_window
    namespace['_get_window'] = types.FunctionType(original.__code__, namespace,
        original.__name__, original.__defaults__, original.__closure__)
    original = kaldi.fbank
    function = types.FunctionType(original.__code__, namespace,
        original.__name__, original.__defaults__, original.__closure__)
    return function(*args, **kwargs)


def make_npu_frontend_class(native_class, root, device):
    class NpuFrontEnd(native_class):
        def __init__(self, get_tokenizer, feat_extractor, campplus_model, speech_tokenizer_model,
                     spk2info="", allowed_special="all"):
            import inflect
            self.device = device
            self.tokenizer = get_tokenizer()
            self.feat_extractor = feat_extractor
            self.allowed_special = allowed_special
            self.inflect_parser = inflect.engine()
            self.text_frontend = ""
            self.spk2info = torch.load(spk2info, map_location=device, weights_only=True) if spk2info and Path(spk2info).is_file() else {}
            self.campplus_npu = converted_model(campplus_model, root, device)
            self.speech_tokenizer_npu = converted_model(speech_tokenizer_model, root, device)
            self.component_timings = {}
            self.neural_execution = {}

        def _call_network(self, name, model, *args):
            if any(isinstance(t, torch.Tensor) and t.device.type != "npu" for t in args):
                raise RuntimeError(f"Non-NPU input to {name}")
            torch.npu.synchronize()
            start = time.monotonic()
            output = model(*args)
            if isinstance(output, (tuple, list)):
                output = output[0]
            torch.npu.synchronize()
            self.component_timings[name] = time.monotonic() - start
            if output.device.type != "npu":
                raise RuntimeError(f"Non-NPU output from {name}")
            self.neural_execution[name] = {"input_devices": [str(t.device) for t in args],
                                           "output_device": str(output.device), "output_shape": list(output.shape)}
            return output

        @torch.no_grad()
        def _extract_speech_token(self, prompt_wav):
            speech = load_audio(prompt_wav, 16000, device=self.device)
            if speech.shape[-1] > 30 * 16000:
                raise ValueError("Reference audio must be no longer than 30 seconds")
            feat = npu_whisper_mel(speech, n_mels=128)
            length = torch.tensor([feat.shape[2]], dtype=torch.int32, device=self.device)
            token = self._call_network("speech_tokenizer", self.speech_tokenizer_npu, feat, length).reshape(1, -1).to(torch.int32)
            return token, torch.tensor([token.shape[1]], dtype=torch.int32, device=self.device)

        @torch.no_grad()
        def _extract_spk_embedding(self, prompt_wav):
            import torchaudio.compliance.kaldi as kaldi
            speech = load_audio(prompt_wav, 16000, device=self.device)
            feat = npu_fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            embedding = self._call_network("campplus", self.campplus_npu, feat.unsqueeze(0)).reshape(1, -1)
            return embedding

        @torch.no_grad()
        def _extract_speech_feat(self, prompt_wav):
            speech = load_audio(prompt_wav, 24000, device=self.device)
            feat = self.feat_extractor(speech).squeeze(0).transpose(0, 1).unsqueeze(0)
            if feat.device.type != "npu":
                raise RuntimeError("Reference mel extraction did not run on NPU")
            return feat, torch.tensor([feat.shape[1]], dtype=torch.int32, device=self.device)

    return NpuFrontEnd

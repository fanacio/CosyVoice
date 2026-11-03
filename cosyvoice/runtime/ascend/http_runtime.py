"""Concurrent HTTP pipeline with shared vLLM batching and isolated request state."""
from __future__ import annotations

import io
import copy
import json
from pathlib import Path
import random
import tempfile
import threading
import time
import uuid

import numpy as np
import soundfile as sf
import torch

MAX_UPLOAD = 10 * 1024 * 1024
MAX_TEXT = 500


class BusyError(Exception):
    pass


class UnhealthyError(Exception):
    pass


class Runtime:
    def __init__(self, *, state, tts, engines, requests, thread_errors, args, load_seconds):
        self.state, self.tts, self.engines = state, tts, engines
        self.requests, self.thread_errors, self.args = requests, thread_errors, args
        self.root = args.root.resolve()
        self.state_lock = threading.Lock()
        self.frontend_lock = threading.Lock()
        self.acoustic_lock = threading.Lock()
        self.capacity = args.max_num_seqs
        self.slots = threading.BoundedSemaphore(self.capacity)
        self.active = 0
        self.max_active = 0
        self.rejected = 0
        self.healthy = True
        self.completed = 0
        self.last_attention_calls = 0
        self.load_seconds = load_seconds
        self.records = self.root / 'outputs/service'
        self.records.mkdir(parents=True, exist_ok=True)
        self.temp = self.root / 'runtime/service_requests'
        self.temp.mkdir(parents=True, exist_ok=True)
        self.neural_weights = {}
        frontend = state['frontend_object']
        for name, model in [('flow', tts.model.flow), ('hift', tts.model.hift),
                            ('campplus', frontend.campplus_npu),
                            ('speech_tokenizer', frontend.speech_tokenizer_npu)]:
            tensors = [t for t in list(model.parameters()) + list(model.buffers()) if t.is_floating_point()]
            devices, dtypes = sorted({str(t.device) for t in tensors}), sorted({str(t.dtype) for t in tensors})
            if devices != ['npu:0'] or dtypes != ['torch.float32']:
                raise RuntimeError(f'{name} is not NPU FP32: {devices}, {dtypes}')
            self.neural_weights[name] = {'devices': devices, 'dtypes': dtypes, 'tensors': len(tensors)}
        self.initial_precision = self.precision(require_attention=False)
    def precision(self, require_attention=True):
        reports = []
        for engine in self.engines:
            reports.extend(engine.collective_rpc('fp32_dtype_report'))
        if not reports:
            raise RuntimeError('No vLLM worker precision evidence')
        for report in reports:
            if (set(report['floating_parameter_dtypes']) != {'torch.float32'} or
                set(report['parameter_devices']) != {'npu:0'} or report['matmul_hf32']):
                raise RuntimeError(f'Worker FP32 validation failed: {report}')
            if require_attention and (set(report['kv_cache_dtypes']) != {'torch.float32'} or
                                       report['attention']['dtypes'] != ['torch.float32']):
                raise RuntimeError(f'Attention FP32 validation failed: {report}')
        return reports

    def health(self):
        with self.state_lock:
            active, completed, maximum, rejected = self.active, self.completed, self.max_active, self.rejected
        return {'status': 'ready' if self.healthy else 'failed', 'busy': active >= self.capacity,
                'active_requests': active, 'completed_requests': completed,
                'max_observed_active_requests': maximum, 'rejected_requests': rejected,
                'model': 'Fun-CosyVoice3-0.5B-2512', 'llm_backend': 'vllm-fl/ascend',
                'llm_dtype': 'float32', 'neural_devices': self.neural_weights,
                'sample_rate': self.tts.sample_rate, 'max_concurrent_synthesis': self.capacity,
                'frontend_busy': self.frontend_lock.locked(), 'acoustic_busy': self.acoustic_lock.locked(),
                'engine': self.engines[0].stats(), 'pending_token_records': len(self.requests),
                'model_streaming': False,
                'visible_devices': __import__('os').getenv('ASCEND_RT_VISIBLE_DEVICES'),
                'load_seconds': self.load_seconds}

    def synthesize(self, text, prompt_text, audio, *, mode='zero_shot', raw_pcm=False):
        if mode not in {'zero_shot', 'cross_lingual', 'instruct2'}:
            raise ValueError('Unsupported mode for this CosyVoice3 checkpoint')
        if not text.strip() or len(text) > MAX_TEXT:
            raise ValueError(f'tts_text must contain 1..{MAX_TEXT} characters')
        if len(prompt_text) > 1000 or (mode != 'cross_lingual' and not prompt_text.strip()):
            raise ValueError('prompt_text/instruct_text must contain 1..1000 characters')
        original_text = text
        # CosyVoice3 requires this token in text or prompt_text. Official CV3
        # examples include it, whereas the older generic clients omit it.
        prefix = 'You are a helpful assistant.<|endofprompt|>'
        if mode == 'cross_lingual' and '<|endofprompt|>' not in text:
            text = prefix + text
        elif mode != 'cross_lingual' and '<|endofprompt|>' not in prompt_text:
            prompt_text = prefix + prompt_text
        if not audio or len(audio) > MAX_UPLOAD:
            raise ValueError('Reference audio is empty or exceeds 10 MiB')
        if raw_pcm:
            if len(audio) % 2 or not 1600 <= len(audio) // 2 <= 30 * 16000:
                raise ValueError('gRPC reference must be mono S16LE, 16 kHz, 0.1..30 seconds')
            wave = np.frombuffer(audio, dtype='<i2').astype(np.float32) / 32768
            rate = 16000
        else:
            try:
                with sf.SoundFile(io.BytesIO(audio)) as source:
                    if source.samplerate < 16000 or not 0.1 <= len(source) / source.samplerate <= 30:
                        raise ValueError('Reference audio must be >=16 kHz and 0.1..30 seconds')
                    if source.channels > 2:
                        raise ValueError('Reference audio must be mono or stereo')
                    wave, rate = source.read(dtype='float32', always_2d=True), source.samplerate
                    wave = wave.mean(axis=1)
            except (sf.LibsndfileError, RuntimeError) as exc:
                raise ValueError('Reference must be a decodable WAV/FLAC audio file') from exc
        if not np.isfinite(wave).all() or np.max(np.abs(wave)) < 1e-6:
            raise ValueError('Reference audio is nonfinite or silent')
        tokenizer = self.tts.frontend.tokenizer
        allowed = self.tts.frontend.allowed_special
        text_tokens = len(tokenizer.encode(text, allowed_special=allowed))
        prompt_tokens = len(tokenizer.encode(prompt_text, allowed_special=allowed))
        # Reserve the official maximum of 20 speech tokens per text token.
        # Reject oversized requests before touching native request state.
        reserve = 21 * text_tokens + prompt_tokens + int(len(wave) / rate * 25) + 4
        if reserve > self.args.max_model_len:
            raise ValueError('Input plus reference and maximum speech tokens exceed the 4096-token service budget; shorten the text/reference')
        if not self.slots.acquire(blocking=False):
            with self.state_lock:
                self.rejected += 1
            raise BusyError(f'All {self.capacity} synthesis slots are occupied; retry later')
        request_id = uuid.uuid4().hex
        native_ids = []
        with self.state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if not self.healthy:
                raise UnhealthyError('Runtime failed a previous request; restart the service')
            torch.npu.set_device(0)
            torch.npu.matmul.allow_hf32 = False
            torch.npu.conv.allow_hf32 = False
            start = time.monotonic()
            entered = time.time()
            chunks, token_records, frontend_records, flow_records, hift_records = [], [], [], [], []
            timings = {'frontend_wait_seconds': 0., 'frontend_seconds': 0.,
                       'llm_seconds': 0., 'acoustic_wait_seconds': 0., 'acoustic_seconds': 0.}
            frontend = self.tts.frontend
            model = self.tts.model
            device = model.device
            with tempfile.TemporaryDirectory(dir=self.temp) as temporary:
                reference = Path(temporary) / 'reference.wav'
                sf.write(reference, wave, rate, subtype='FLOAT')
                # Same native nonstreaming stages, with a request-owned UUID.
                # The official tts() creates a nested thread and global queues;
                # call its component methods directly to isolate each request.
                with torch.inference_mode(), torch.autocast('npu', enabled=False):
                    segments = frontend.text_normalize(text, split=True, text_frontend=False)
                    if not segments:
                        raise ValueError('No speech text after normalization')
                    for segment_index, segment in enumerate(segments):
                        native_id = f'{request_id}:{segment_index}'
                        native_ids.append(native_id)
                        waiting = time.monotonic()
                        with self.frontend_lock:
                            timings['frontend_wait_seconds'] += time.monotonic() - waiting
                            began = time.monotonic()
                            frontend.component_timings.clear()
                            frontend.neural_execution.clear()
                            if mode == 'zero_shot':
                                normalized = frontend.text_normalize(prompt_text, split=False, text_frontend=False)
                                inputs = frontend.frontend_zero_shot(segment, normalized, str(reference), self.tts.sample_rate, '')
                            elif mode == 'cross_lingual':
                                inputs = frontend.frontend_cross_lingual(segment, str(reference), self.tts.sample_rate, '')
                            else:
                                inputs = frontend.frontend_instruct2(segment, prompt_text, str(reference), self.tts.sample_rate, '')
                            frontend_records.append({'execution': copy.deepcopy(frontend.neural_execution),
                                                     'seconds': dict(frontend.component_timings)})
                            timings['frontend_seconds'] += time.monotonic() - began
                        empty = torch.zeros(1, 0, dtype=torch.int32, device=device)
                        text_tensor = inputs['text'].to(device)
                        prompt = inputs.get('prompt_text', empty).to(device)
                        prompt_speech = inputs.get('llm_prompt_speech_token', empty).to(device)
                        began = time.monotonic()
                        generated = model.llm.inference(
                            text=text_tensor,
                            text_len=torch.tensor([text_tensor.shape[1]], dtype=torch.int32, device=device),
                            prompt_text=prompt,
                            prompt_text_len=torch.tensor([prompt.shape[1]], dtype=torch.int32, device=device),
                            prompt_speech_token=prompt_speech,
                            prompt_speech_token_len=torch.tensor([prompt_speech.shape[1]], dtype=torch.int32, device=device),
                            embedding=inputs['llm_embedding'].to(device), uuid=native_id)
                        tokens, silent = [], 0
                        for token in generated:
                            if token in model.silent_tokens:
                                silent += 1
                                if silent > 5:
                                    continue
                            else:
                                silent = 0
                            tokens.append(token)
                        timings['llm_seconds'] += time.monotonic() - began
                        token_records.append(self.requests.take(native_id))
                        if not tokens:
                            raise RuntimeError('LLM returned no speech tokens')
                        waiting = time.monotonic()
                        with self.acoustic_lock:
                            timings['acoustic_wait_seconds'] += time.monotonic() - waiting
                            began = time.monotonic()
                            # Native Flow uses host/NPU RNG; seed only under its
                            # own lock, never while another acoustic job runs.
                            torch.manual_seed(self.args.seed)
                            np.random.seed(self.args.seed)
                            random.seed(self.args.seed)
                            self.state['flow'], self.state['hift'] = [], []
                            with model.lock:
                                model.hift_cache_dict[native_id] = None
                            try:
                                speech = model.token2wav(
                                    token=torch.tensor(tokens, dtype=torch.int32).unsqueeze(0),
                                    prompt_token=inputs['flow_prompt_speech_token'],
                                    prompt_feat=inputs['prompt_speech_feat'], embedding=inputs['flow_embedding'],
                                    token_offset=0, uuid=native_id, finalize=True, speed=1.0)
                                chunks.append(speech.detach().float().cpu())
                                flow_records.extend(copy.deepcopy(self.state['flow']))
                                hift_records.extend(copy.deepcopy(self.state['hift']))
                            finally:
                                with model.lock:
                                    model.hift_cache_dict.pop(native_id, None)
                            timings['acoustic_seconds'] += time.monotonic() - began
            samples = torch.cat(chunks, dim=-1).squeeze(0).numpy()
            if not np.isfinite(samples).all() or np.max(np.abs(samples)) < 1e-5:
                raise RuntimeError('Invalid or silent generated audio')
            precision = self.precision()
            if any(p['attention']['calls'] == 0 for p in precision):
                raise RuntimeError('No actual FP32 attention calls')
            duration = len(samples) / self.tts.sample_rate
            seconds = time.monotonic() - start
            pcm = (np.clip(samples, -1, 32767 / 32768) * 32768).astype('<i2').tobytes()
            report = {'request_id': request_id, 'mode': mode, 'text': original_text,
                      'model_text': text, 'model_prompt_text': prompt_text,
                      'entered_at_unix': entered, 'finished_at_unix': time.time(),
                      'sample_rate': self.tts.sample_rate, 'samples': len(samples),
                      'audio_seconds': duration, 'synthesis_seconds': seconds, 'stage_timings': timings,
                      'rtf': seconds / duration, 'requests': token_records,
                      'llm_precision_reports': precision, 'neural_weights': self.neural_weights,
                      'frontend': frontend_records[0], 'frontend_segments': frontend_records,
                      'flow': flow_records, 'hift': hift_records, 'engine': self.engines[0].stats()}
            (self.records / (request_id + '.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2))
            with self.state_lock:
                self.completed += 1
            print('SERVICE_SYNTHESIS_OK', json.dumps({k: report[k] for k in
                  ['request_id', 'mode', 'audio_seconds', 'synthesis_seconds', 'rtf']}, ensure_ascii=False), flush=True)
            return pcm, report
        except UnhealthyError:
            raise
        except Exception:
            self.healthy = False
            raise
        finally:
            for native_id in native_ids:
                self.requests.discard(native_id)
            with self.state_lock:
                self.active -= 1
            self.slots.release()

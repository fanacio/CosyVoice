"""Clients for the official CosyVoice HTTP forms / gRPC protobuf protocol.

Based on FunAudioLLM/CosyVoice runtime/python/{fastapi,grpc}/client.py
(Apache-2.0). Uses server-reported 24 kHz, avoiding the official demo's 22050.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'service/generated'))
PROMPT = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=['http', 'grpc'], default='http')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int)
    parser.add_argument('--mode', choices=['zero_shot', 'cross_lingual', 'instruct2'], default='zero_shot')
    parser.add_argument('--tts-text', '--tts_text', default='你好，欢迎使用语音合成。')
    parser.add_argument('--prompt-text', '--prompt_text', default=PROMPT)
    parser.add_argument('--instruct-text', default='You are a helpful assistant. 请用自然的语气说话。<|endofprompt|>')
    parser.add_argument('--prompt-wav', '--prompt_wav', type=Path, default=ROOT / 'src/CosyVoice/asset/zero_shot_prompt.wav')
    parser.add_argument('--output', '--tts_wav', type=Path, default=ROOT / 'outputs/service/client.wav')
    parser.add_argument('--timeout', type=float, default=240)
    args = parser.parse_args()
    port = args.port or (50000 if args.protocol == 'http' else 50001)
    start = time.monotonic()
    if args.protocol == 'http':
        import requests
        fields = {'tts_text': args.tts_text}
        if args.mode == 'zero_shot':
            fields['prompt_text'] = args.prompt_text
        if args.mode == 'instruct2':
            fields['instruct_text'] = args.instruct_text
        with args.prompt_wav.open('rb') as reference:
            result = requests.post(f'http://{args.host}:{port}/inference_{args.mode}', data=fields,
                                   files={'prompt_wav': (args.prompt_wav.name, reference, 'audio/wav')},
                                   timeout=args.timeout)
        result.raise_for_status()
        pcm = result.content
        rate = int(result.headers['X-Sample-Rate'])
        request_id = result.headers['X-Request-ID']
    else:
        if args.mode == 'instruct2':
            parser.error('The unchanged official gRPC proto has no instruct2 field; use HTTP')
        import grpc
        import cosyvoice_pb2 as pb
        import cosyvoice_pb2_grpc as rpc
        wave, sr = sf.read(args.prompt_wav, dtype='float32', always_2d=True)
        wave = wave.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            divisor = gcd(sr, 16000)
            wave = resample_poly(wave, 16000 // divisor, sr // divisor)
        prompt_audio = (np.clip(wave, -1, 32767 / 32768) * 32768).astype('<i2').tobytes()
        if args.mode == 'zero_shot':
            request = pb.Request(zero_shot_request=pb.zeroshotRequest(
                tts_text=args.tts_text, prompt_text=args.prompt_text, prompt_audio=prompt_audio))
        else:
            request = pb.Request(cross_lingual_request=pb.crosslingualRequest(
                tts_text=args.tts_text, prompt_audio=prompt_audio))
        with grpc.insecure_channel(f'{args.host}:{port}') as channel:
            call = rpc.CosyVoiceStub(channel).Inference(request, timeout=args.timeout)
            pcm = b''.join(item.tts_audio for item in call)
            metadata = dict(call.initial_metadata())
            rate, request_id = int(metadata['sample-rate']), metadata['request-id']
    wave = np.frombuffer(pcm, dtype='<i2')
    if not len(wave) or rate != 24000 or np.max(np.abs(wave.astype(np.int32))) == 0:
        raise RuntimeError('Invalid/silent PCM or unexpected CosyVoice3 sample rate')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, wave, rate, subtype='PCM_16')
    report = {'protocol': args.protocol, 'host': args.host, 'port': port, 'mode': args.mode,
              'request_id': request_id, 'sample_rate': rate, 'samples': len(wave),
              'audio_seconds': len(wave) / rate, 'roundtrip_seconds': time.monotonic() - start,
              'peak_pcm': int(np.max(np.abs(wave.astype(np.int32)))), 'output': str(args.output)}
    args.output.with_suffix('.client.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('CLIENT_SUCCESS', json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

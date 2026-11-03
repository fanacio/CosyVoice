"""Identical HTTP inputs, warmed model, sequential/concurrent measurements."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ['你好，欢迎使用语音合成。', '今天阳光明媚，适合出去散步。',
         '请保持耐心，我们正在为您处理请求。', '这是在昇腾设备上运行的语音合成服务。']
PROMPT = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
LOCAL = threading.local()


def health():
    r = requests.get('http://127.0.0.1:50040/healthz', timeout=5)
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--concurrency', default='1,2,4')
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    reference = (ROOT / 'src/CosyVoice/asset/zero_shot_prompt.wav').read_bytes()
    output = ROOT / 'outputs/service/http_concurrency' / args.label
    output.mkdir(parents=True, exist_ok=True)

    def request(index, name):
        if not hasattr(LOCAL, 'session'):
            LOCAL.session = requests.Session()
            LOCAL.session.trust_env = False
        started = time.monotonic()
        result = LOCAL.session.post('http://127.0.0.1:50040/inference_zero_shot',
            data={'tts_text': TEXTS[index], 'prompt_text': PROMPT},
            files={'prompt_wav': ('reference.wav', reference, 'audio/wav')}, timeout=240)
        seconds = time.monotonic() - started
        result.raise_for_status()
        assert int(result.headers['X-Sample-Rate']) == 24000
        wave = np.frombuffer(result.content, dtype='<i2')
        assert len(wave) > 0 and np.max(np.abs(wave.astype(np.int32))) > 1
        request_id = result.headers['X-Request-ID']
        evidence = json.loads((ROOT / 'outputs/service' / (request_id + '.json')).read_text())
        assert evidence['text'] == TEXTS[index]
        assert all(p['attention']['dtypes'] == ['torch.float32'] for p in evidence['llm_precision_reports'])
        assert all(e['output_device'] == 'npu:0' for e in evidence['flow'] + evidence['hift'])
        sf.write(output / (name + '.wav'), wave, 24000, subtype='PCM_16')
        return {'name': name, 'text': TEXTS[index], 'request_id': request_id,
                'latency_seconds': seconds, 'audio_seconds': len(wave) / 24000,
                'pcm_sha256': hashlib.sha256(result.content).hexdigest(),
                'token_count': sum(r['token_count'] for r in evidence['requests'])}

    deadline = time.monotonic() + 180
    while True:
        try:
            before = health()
            break
        except requests.RequestException:
            if time.monotonic() > deadline:
                raise
            time.sleep(1)
    for i in range(2):
        request(i, f'warmup_{i}')
    report = {'label': args.label, 'reference_sha256': hashlib.sha256(reference).hexdigest(),
              'texts': TEXTS, 'before': before, 'measurements': [], 'passed': False}
    for concurrency in map(int, args.concurrency.split(',')):
        rows = []
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for repeat in range(args.rounds):
                futures = [executor.submit(request, i, f'c{concurrency}_r{repeat}_{i}') for i in range(4)]
                rows.extend(f.result() for f in futures)
        elapsed = time.monotonic() - start
        latencies = [row['latency_seconds'] for row in rows]
        metric = {'concurrency': concurrency, 'requests': rows, 'wall_seconds': elapsed,
                  'requests_per_second': len(rows) / elapsed,
                  'audio_seconds_per_second': sum(r['audio_seconds'] for r in rows) / elapsed,
                  'latency_mean_seconds': float(np.mean(latencies)),
                  'latency_p95_seconds': float(np.percentile(latencies, 95))}
        report['measurements'].append(metric)
        print('HTTP_BENCHMARK', json.dumps({k:v for k,v in metric.items() if k != 'requests'}), flush=True)
    report['after'] = health()
    report['passed'] = True
    (ROOT / 'reports' / ('http_benchmark_' + args.label + '.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('HTTP_BENCHMARK_PASS', args.label, flush=True)


if __name__ == '__main__':
    main()

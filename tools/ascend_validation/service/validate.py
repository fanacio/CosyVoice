"""HTTP-only concurrent correctness, overload, cleanup and precision checks."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import time

import numpy as np
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:50000'
OUT = ROOT / 'outputs/service/http_validation'
PROMPT = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
TEXTS = ['你好，欢迎使用语音合成。','今天的天气很好，我们一起出去散步。',
         '请保持耐心，我们正在为您处理请求。','这是在昇腾设备上运行的语音合成服务。']


def health():
    response = requests.get(BASE + '/healthz', timeout=5)
    response.raise_for_status()
    return response.json()


def main():
    reference = (ROOT / 'src/CosyVoice/asset/zero_shot_prompt.wav').read_bytes()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'protocol':'http', 'passed':False, 'requests':[], 'errors':{}}
    before = health()
    report['before'] = before
    capacity = before['max_concurrent_synthesis']
    assert capacity == 4, 'Validation expects the deployed four-request configuration'
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1',50001)) != 0, 'gRPC must be closed'

    def request(index, mode, name):
        fields = {'tts_text':TEXTS[index]}
        if mode=='zero_shot': fields['prompt_text']=PROMPT
        if mode=='instruct2': fields['instruct_text']='请用自然的语气说话。'
        response = requests.post(BASE + '/inference_' + mode, data=fields,
                                 files={'prompt_wav':('reference.wav',reference)}, timeout=240)
        response.raise_for_status()
        assert response.headers['X-LLM-Dtype']=='float32'
        assert int(response.headers['X-Sample-Rate'])==24000
        samples = np.frombuffer(response.content,dtype='<i2')
        assert len(samples)>0 and np.max(np.abs(samples.astype(np.int32)))>1
        request_id = response.headers['X-Request-ID']
        record = json.loads((ROOT/'outputs/service'/(request_id+'.json')).read_text())
        assert record['mode']==mode and record['text']==TEXTS[index]
        assert all(r['request_id'].startswith(request_id+':') for r in record['requests'])
        assert len({r['request_id'] for r in record['requests']})==len(record['requests'])
        assert all(p['attention']['dtypes']==['torch.float32'] and
                   set(p['kv_cache_dtypes'])=={'torch.float32'} for p in record['llm_precision_reports'])
        assert all(e['output_device']=='npu:0' and e['dtype']=='torch.float32' for e in record['flow']+record['hift'])
        assert all(e['output_device']=='npu:0' for e in record['frontend']['execution'].values())
        sf.write(OUT/(name+'.wav'),samples,24000,subtype='PCM_16')
        return {'request_id':request_id,'mode':mode,'text':TEXTS[index],
                'audio_seconds':len(samples)/24000,'token_count':sum(r['token_count'] for r in record['requests']),
                'entered_at_unix':record['entered_at_unix'],'finished_at_unix':record['finished_at_unix'],
                'attention':record['llm_precision_reports'][0]['attention']}

    # Simultaneous requests exercise different native frontend paths and UUIDs.
    modes = ['zero_shot','cross_lingual','instruct2','zero_shot']
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(request,i,mode,f'mixed_{i}_{mode}') for i,mode in enumerate(modes)]
        deadline = time.monotonic()+30
        while health()['active_requests']<4:
            if any(f.done() for f in futures) or time.monotonic()>deadline:
                raise RuntimeError('Could not observe all four admitted requests')
            time.sleep(.02)
        response = requests.post(BASE+'/inference_zero_shot',
                                 data={'tts_text':'你好','prompt_text':PROMPT},
                                 files={'prompt_wav':('reference.wav',reference)},timeout=10)
        assert response.status_code==503
        report['errors']['fifth_request_overload']=503
        report['requests'].extend(f.result() for f in futures)
    assert len({r['request_id'] for r in report['requests']})==4
    assert max(r['entered_at_unix'] for r in report['requests']) < min(r['finished_at_unix'] for r in report['requests'])
    count = health()['completed_requests']
    response = requests.post(BASE+'/inference_zero_shot',data={'tts_text':'你好','prompt_text':PROMPT},
                             files={'prompt_wav':('bad.wav',b'bad')},timeout=10)
    assert response.status_code==400
    report['errors']['invalid_audio']=400
    response = requests.post(BASE+'/inference_zero_shot',data={'tts_text':'测试'*250,'prompt_text':PROMPT},
                             files={'prompt_wav':('reference.wav',reference)},timeout=10)
    assert response.status_code==400
    report['errors']['token_budget']=400
    response = requests.post(BASE+'/inference_sft',timeout=10)
    assert response.status_code==501
    report['errors']['unsupported_mode']=501
    assert health()['completed_requests']==count
    report['requests'].append(request(0,'zero_shot','after_overload'))
    after = health()
    assert after['status']=='ready' and after['active_requests']==0
    assert after['pending_token_records']==0 and after['engine']['active']==0
    assert after['engine']['max_outstanding']>=4 and after['engine']['max_step_requests']>=2
    assert any(r['attention']['max_sequences']>=4 and r['attention']['batched_calls']>0 for r in report['requests'])
    assert after['completed_requests']-before['completed_requests']==5
    report['after']=after
    report['passed']=True
    (ROOT/'reports/http_concurrency_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('HTTP_CONCURRENCY_VALIDATION_PASS',json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

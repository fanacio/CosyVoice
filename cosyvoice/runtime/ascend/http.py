import asyncio,traceback
def http_application(runtime):
    from fastapi import FastAPI, UploadFile, Form, File, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from .http_runtime import BusyError, UnhealthyError, MAX_UPLOAD

    app = FastAPI(title='CosyVoice3 Ascend FP32', version='1.0')

    @app.get('/healthz')
    def health():
        return JSONResponse(runtime.health(), status_code=200 if runtime.healthy else 503)

    async def infer(mode, text, prompt_text, upload):
        try:
            audio = await upload.read(MAX_UPLOAD + 1)
        finally:
            await upload.close()
        try:
            pcm, report = await asyncio.to_thread(runtime.synthesize, text, prompt_text, audio, mode=mode)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except (BusyError, UnhealthyError) as exc:
            raise HTTPException(503, str(exc), headers={'Retry-After': '5'}) from exc
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(500, 'Synthesis failed; inspect the server log and restart if unhealthy') from exc
        headers = {'X-Sample-Rate': str(report['sample_rate']), 'X-Audio-Format': 'pcm_s16le',
                   'X-Channels': '1', 'X-Request-ID': report['request_id'],
                   'X-LLM-Dtype': 'float32'}
        # Match official raw PCM response. Buffer full synthesis deliberately:
        # incremental Flow/HiFT streaming has not been validated on this stack.
        chunks = (pcm[pos:pos + 65536] for pos in range(0, len(pcm), 65536))
        return StreamingResponse(chunks, media_type='application/octet-stream', headers=headers)

    @app.post('/inference_zero_shot')
    async def zero_shot(tts_text: str = Form(...), prompt_text: str = Form(...),
                        prompt_wav: UploadFile = File(...)):
        return await infer('zero_shot', tts_text, prompt_text, prompt_wav)

    @app.post('/inference_cross_lingual')
    async def cross_lingual(tts_text: str = Form(...), prompt_wav: UploadFile = File(...)):
        return await infer('cross_lingual', tts_text, '', prompt_wav)

    @app.post('/inference_instruct2')
    async def instruct2(tts_text: str = Form(...), instruct_text: str = Form(...),
                        prompt_wav: UploadFile = File(...)):
        return await infer('instruct2', tts_text, instruct_text, prompt_wav)

    @app.post('/inference_sft')
    @app.post('/inference_instruct')
    async def unsupported():
        raise HTTPException(501, 'This checkpoint has no preset speakers; use zero_shot, cross_lingual or instruct2')

    return app

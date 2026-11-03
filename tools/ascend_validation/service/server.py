"""Official-compatible FastAPI routes and CosyVoice gRPC wire protocol on NPU.

Based on FunAudioLLM/CosyVoice runtime/python/{fastapi,grpc}/server.py
(Apache-2.0). The pinned official cosyvoice.proto is used unchanged.
"""

import argparse
import asyncio
from concurrent import futures
import contextlib
import json
from pathlib import Path
import signal
import socket
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'service/generated')]


def http_application(runtime):
    from fastapi import FastAPI, UploadFile, Form, File, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from runtime import BusyError, UnhealthyError, MAX_UPLOAD

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


def grpc_server(runtime, host, port):
    import grpc
    import cosyvoice_pb2 as pb
    import cosyvoice_pb2_grpc as rpc
    from runtime import BusyError, UnhealthyError, MAX_UPLOAD

    class Service(rpc.CosyVoiceServicer):
        def Inference(self, request, context):
            payload = request.WhichOneof('RequestPayload')
            if payload not in {'zero_shot_request', 'cross_lingual_request'}:
                context.abort(grpc.StatusCode.UNIMPLEMENTED,
                              'Use official zero_shot_request or cross_lingual_request for this checkpoint')
            item = getattr(request, payload)
            mode = 'zero_shot' if payload == 'zero_shot_request' else 'cross_lingual'
            if not context.is_active():
                return
            try:
                pcm, report = runtime.synthesize(item.tts_text, getattr(item, 'prompt_text', ''),
                                                item.prompt_audio, mode=mode, raw_pcm=True)
            except ValueError as exc:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except BusyError as exc:
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
            except UnhealthyError as exc:
                context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
            except Exception:
                traceback.print_exc()
                context.abort(grpc.StatusCode.INTERNAL, 'Synthesis failed; inspect server log')
            context.send_initial_metadata((('sample-rate', str(report['sample_rate'])),
                                           ('audio-format', 'pcm_s16le'), ('channels', '1'),
                                           ('request-id', report['request_id']), ('llm-dtype', 'float32')))
            for pos in range(0, len(pcm), 65536):
                if not context.is_active():
                    break
                yield pb.Response(tts_audio=pcm[pos:pos + 65536])

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), maximum_concurrent_rpcs=8,
                         options=[('grpc.max_receive_message_length', MAX_UPLOAD + 65536),
                                  ('grpc.so_reuseport', 0)])
    rpc.add_CosyVoiceServicer_to_server(Service(), server)
    if server.add_insecure_port(f'{host}:{port}') == 0:
        raise RuntimeError(f'gRPC bind failed: {host}:{port}')
    server.start()
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--http-port', type=int, default=50000)
    parser.add_argument('--grpc-port', type=int, default=50001)
    parser.add_argument('--protocol', choices=['http'], default='http')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--model', type=Path, default=Path('/public-flash/models/q4_models/iic/Fun-CosyVoice3-0.5B-2512'))
    parser.add_argument('--request-timeout', type=float, default=180)
    parser.add_argument('--memory-utilization', type=float, default=0.15)
    parser.add_argument('--llm-execution', choices=['eager', 'piecewise', 'decode-graph'], default='eager')
    parser.add_argument('--max-concurrency', type=int, choices=range(1, 5), default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    options = parser.parse_args()
    ports = ([options.http_port] if options.protocol == 'http' else
             [options.grpc_port] if options.protocol == 'grpc' else
             [options.http_port, options.grpc_port])
    if len(set(ports)) != len(ports):
        parser.error('HTTP and gRPC must use different ports')
    # Fail on an occupied port before loading the NPU model.
    for port in ports:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((options.host, port))
    root = options.root.resolve()
    source = root / 'src/CosyVoice'
    sys.path[:0] = [str(source), str(source / 'third_party/Matcha-TTS')]
    from stage2_backend import install_full_npu_pipeline
    from run_tts import main as load_official_pipeline
    from runtime import Runtime
    state = install_full_npu_pipeline(root)

    def ready(**context):
        runtime = Runtime(state=state, **context)
        grpc_instance = None
        ready_file = root / 'runtime/service_ready.json'
        try:
            if options.protocol in {'both', 'grpc'}:
                grpc_instance = grpc_server(runtime, options.host, options.grpc_port)
            status = {**runtime.health(), 'host': options.host, 'http_port': options.http_port,
                      'grpc_port': options.grpc_port, 'protocol': options.protocol,
                      'initial_precision': runtime.initial_precision}
            ready_file.write_text(json.dumps(status, ensure_ascii=False, indent=2))
            print('SERVICE_MODEL_READY', json.dumps(status, ensure_ascii=False), flush=True)
            if options.protocol in {'both', 'http'}:
                import uvicorn
                def terminate(*_):
                    raise SystemExit(0)
                signal.signal(signal.SIGTERM, terminate)
                uvicorn.run(http_application(runtime), host=options.host, port=options.http_port,
                            workers=1, timeout_graceful_shutdown=200, access_log=True)
            else:
                stop = __import__('threading').Event()
                signal.signal(signal.SIGTERM, lambda *_: stop.set())
                signal.signal(signal.SIGINT, lambda *_: stop.set())
                stop.wait()
        finally:
            runtime.healthy = False
            if grpc_instance is not None:
                grpc_instance.stop(grace=200).wait(timeout=205)
            with contextlib.suppress(FileNotFoundError):
                ready_file.unlink()

    sys.argv = ['run_tts.py', '--root', str(root), '--model', str(options.model),
                '--llm-dtype', 'float32', '--request-timeout', str(options.request_timeout),
                '--memory-utilization', str(options.memory_utilization),
                '--max-num-seqs', str(options.max_concurrency), '--llm-execution', options.llm_execution, '--max-model-len', str(options.max_model_len)]
    load_official_pipeline(on_ready=ready)


if __name__ == '__main__':
    main()

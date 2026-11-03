"""Start/stop/status for this project's persistent service only."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime'
PIDFILE = RUNTIME / 'service_process.json'
SERVER = ROOT / 'service/server.py'


def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if raw[0] == 'Z':
            return None
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if str(SERVER).encode() not in command:
            return None
        return raw[19]  # /proc stat field 22: starttime, after pid/comm
    except (FileNotFoundError, ProcessLookupError):
        return None


def active():
    if not PIDFILE.exists():
        return None
    record = json.loads(PIDFILE.read_text())
    return record if identity(record['pid']) == record['start_ticks'] else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'stop', 'status'])
    args, server_args = parser.parse_known_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / 'service_control.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = active()
        if args.action == 'start':
            if record:
                print(f'Already running: pid={record["pid"]}; use status')
                return
            ready = RUNTIME / 'service_ready.json'
            with contextlib.suppress(FileNotFoundError):
                ready.unlink()
            env = dict(os.environ)
            for key, value in {'ASCEND_RT_VISIBLE_DEVICES': '1', 'GLOO_SOCKET_IFNAME': 'lo',
                               'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
                               'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
                               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}.items():
                env.setdefault(key, value)
            logs = ROOT / 'logs'
            logs.mkdir(exist_ok=True)
            logfile = logs / ('service_' + time.strftime('%Y%m%d_%H%M%S') + '.log')
            command = [str(ROOT / '.venv/bin/python'), '-u', str(SERVER), *server_args]
            with logfile.open('wb') as output:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            # Allow exec to replace the short-lived fork before checking cmdline.
            ticks = None
            for _ in range(50):
                ticks = identity(process.pid)
                if ticks or process.poll() is not None:
                    break
                time.sleep(0.02)
            if ticks is None:
                raise RuntimeError(f'Server failed to start; inspect {logfile}')
            record = {'pid': process.pid, 'start_ticks': ticks, 'command': command,
                      'log': str(logfile), 'visible_devices': env['ASCEND_RT_VISIBLE_DEVICES']}
            PIDFILE.write_text(json.dumps(record, indent=2))
            print(json.dumps({'state': 'starting', **record}, ensure_ascii=False))
        elif args.action == 'stop':
            if not record:
                print('Not running')
                return
            os.kill(record['pid'], signal.SIGTERM)
            for _ in range(2200):
                if identity(record['pid']) != record['start_ticks']:
                    with contextlib.suppress(FileNotFoundError):
                        PIDFILE.unlink()
                    print('Stopped; model workers shut down by server cleanup')
                    return
                time.sleep(0.1)
            raise RuntimeError('Graceful shutdown still running; inspect log. No forced kill was sent.')
        else:
            if not record:
                print(json.dumps({'state': 'stopped'}))
                sys.exit(1)
            ready_file = RUNTIME / 'service_ready.json'
            ready = json.loads(ready_file.read_text()) if ready_file.exists() else None
            health = None
            if ready and ready['protocol'] in {'both', 'http'}:
                host = '127.0.0.1' if ready['host'] == '0.0.0.0' else ready['host']
                try:
                    with urllib.request.urlopen(f'http://{host}:{ready["http_port"]}/healthz', timeout=2) as response:
                        health = json.load(response)
                except Exception as exc:
                    health = {'status': 'unavailable', 'detail': str(exc)}
            elif ready:
                health = {'status': 'model_ready', 'protocol': 'grpc'}
            print(json.dumps({'process': record, 'health': health, 'model': ready}, ensure_ascii=False, indent=2))
            if health is None or health.get('status') not in {'ready', 'model_ready'}:
                sys.exit(2)


if __name__ == '__main__':
    main()

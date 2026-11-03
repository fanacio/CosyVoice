"""One owner thread for vLLM engine operations, with per-request output queues."""
from concurrent.futures import Future
from dataclasses import dataclass, field
import queue
import threading
import time


class RequestRecords:
    def __init__(self):
        self.lock = threading.Lock()
        self.records = {}

    def append(self, record):
        with self.lock:
            self.records[record['request_id']] = record

    def take(self, request_id):
        with self.lock:
            return self.records.pop(request_id)

    def discard(self, request_id):
        with self.lock:
            self.records.pop(request_id, None)

    def __len__(self):
        with self.lock:
            return len(self.records)


@dataclass
class Ticket:
    request_id: str
    prompt: object
    params: object
    outputs: queue.Queue = field(default_factory=queue.Queue)


class BatchedEngine:
    """LLMEngine is not thread safe; all add/step/abort/RPC calls run here."""
    def __init__(self, engine):
        self.engine = engine
        self.commands = queue.Queue()
        self.failure = None
        self.closing = False
        self.metrics_lock = threading.Lock()
        self.metrics = {'steps': 0, 'active': 0, 'max_outstanding': 0, 'max_step_requests': 0}
        self.thread = threading.Thread(target=self._run, name='cosyvoice-vllm-dispatch', daemon=True)
        self.thread.start()

    def stats(self):
        with self.metrics_lock:
            return dict(self.metrics)

    def generate(self, request_id, prompt, params, timeout):
        if self.failure or self.closing:
            raise RuntimeError('vLLM dispatcher unavailable') from self.failure
        ticket = Ticket(request_id, prompt, params)
        self.commands.put(('submit', ticket))
        deadline = time.monotonic() + timeout
        finished = False
        try:
            while not finished:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'LLM request exceeded {timeout}s')
                try:
                    output = ticket.outputs.get(timeout=remaining)
                except queue.Empty as exc:
                    raise TimeoutError(f'LLM request exceeded {timeout}s') from exc
                if isinstance(output, BaseException):
                    raise output
                if output.request_id != request_id:
                    raise RuntimeError('Output routed to wrong request')
                finished = output.finished
                yield output
        finally:
            if not finished:
                self.abort_request(request_id)

    def abort_request(self, request_id):
        self.commands.put(('abort', request_id))

    def collective_rpc(self, method):
        if self.failure or self.closing:
            raise RuntimeError('vLLM dispatcher unavailable') from self.failure
        future = Future()
        self.commands.put(('rpc', (method, future)))
        return future.result(timeout=60)

    def _run(self):
        active = {}

        def command(kind, value):
            if kind == 'submit':
                if value.request_id in active:
                    value.outputs.put(RuntimeError('Duplicate request id'))
                    return
                try:
                    self.engine.add_request(value.request_id, value.prompt, value.params)
                except Exception as exc:
                    value.outputs.put(exc)
                    return
                active[value.request_id] = value
            elif kind == 'abort':
                self.engine.abort_request(value)
                ticket = active.pop(value, None)
                if ticket:
                    ticket.outputs.put(TimeoutError('Request aborted'))
            elif kind == 'rpc':
                method, future = value
                try:
                    future.set_result(self.engine.collective_rpc(method))
                except Exception as exc:
                    future.set_exception(exc)
            elif kind == 'stop':
                return False
            return True

        try:
            running = True
            while running:
                if not active:
                    running = command(*self.commands.get()) is not False
                for _ in range(64):
                    try:
                        item = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if command(*item) is False:
                        running = False
                        break
                if not running:
                    break
                with self.metrics_lock:
                    self.metrics['active'] = len(active)
                    self.metrics['max_outstanding'] = max(self.metrics['max_outstanding'], len(active))
                if not active:
                    continue
                outputs = self.engine.step()
                with self.metrics_lock:
                    self.metrics['steps'] += 1
                    self.metrics['max_step_requests'] = max(self.metrics['max_step_requests'], len(outputs))
                for output in outputs:
                    ticket = active.get(output.request_id)
                    if ticket is None:
                        raise RuntimeError(f'Unowned engine output {output.request_id}')
                    ticket.outputs.put(output)
                    if output.finished:
                        active.pop(output.request_id)
                if active and not self.engine.has_unfinished_requests():
                    raise RuntimeError('Engine lost outstanding requests')
        except BaseException as exc:
            self.failure = exc
            for ticket in active.values():
                ticket.outputs.put(exc)
            while True:
                try:
                    kind, value = self.commands.get_nowait()
                except queue.Empty:
                    break
                if kind == 'rpc':
                    value[1].set_exception(exc)
                elif kind == 'submit':
                    value.outputs.put(exc)
        finally:
            for request_id, ticket in active.items():
                try:
                    self.engine.abort_request(request_id)
                finally:
                    ticket.outputs.put(RuntimeError('Dispatcher stopped'))
            with self.metrics_lock:
                self.metrics['active'] = 0

    def shutdown(self):
        self.closing = True
        self.commands.put(('stop', None))
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('Dispatcher did not stop; refusing racing engine shutdown')
        self.engine.engine_core.shutdown()

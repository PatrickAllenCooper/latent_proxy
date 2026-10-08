"""CPU-testable diagnostic observation and proposed phase limits; no sampling."""
import json
import sys
import threading
import time
import traceback
from pathlib import Path

STARTUP_SECONDS = 90
FIRST_SECONDS = 30
REST_SECONDS = 10
GENERATION_END_SECONDS = 270
EXPORT_END_SECONDS = 285
JOB_END_SECONDS = 300


def phase_cap(count):
    if not 0 <= count < 16:
        raise ValueError('response quota exhausted')
    return FIRST_SECONDS if count == 0 else REST_SECONDS


def generation_deadline(now, start, count):
    return min(now + phase_cap(count), start + GENERATION_END_SECONDS)


def admit_next(now, start, count):
    """Reserve all remaining declared case caps; overhead consumes shared slack."""
    if not 0 <= count < 16:
        return False
    required = (FIRST_SECONDS + 15 * REST_SECONDS if count == 0
                else (16 - count) * REST_SECONDS)
    return start + GENERATION_END_SECONDS - now >= required


class DiagnosticLog:
    def __init__(self, path, case_id, clock=time.time):
        self.path = Path(path)
        self.case_id = case_id
        self.clock = clock
        self.lock = threading.Lock()
        self.path.touch(exist_ok=False)

    def emit(self, event, **details):
        with self.lock:
            with self.path.open('a') as f:
                f.write(json.dumps(dict(event=event, at=self.clock(),
                                        case_id=self.case_id, diagnostic_only=True,
                                        eligible_for_scoring=False, **details)) + '\n')
                f.flush()


class TokenIDObserver:
    """Transformers streamer duck protocol. Receives CPU tensors; no decoding.

    Runtime invokes .cpu() before put(), introducing GPU-to-host synchronization.
    Arrival timing is diagnostic timing, not uninstrumented throughput.
    """
    def __init__(self, expected_prompt_ids, log):
        self.expected_prompt_ids = list(expected_prompt_ids)
        self.log = log
        self.prompt_seen = False
        self.new_ids = []
        self.ended = False

    def put(self, value):
        if self.ended:
            raise ValueError('stream already ended')
        ids = value.tolist()
        if not self.prompt_seen:
            if ids != [self.expected_prompt_ids]:
                raise ValueError('stream prompt differs from frozen prompt IDs')
            self.prompt_seen = True
            self.log.emit('stream_prompt', token_ids=self.expected_prompt_ids)
            return
        if not isinstance(ids, list) or len(ids) != 1 or type(ids[0]) is not int:
            raise ValueError('expected one batch-one new token')
        if len(self.new_ids) >= 24:
            raise ValueError('stream exceeds fixed 24-token cap')
        self.new_ids.extend(ids)
        self.log.emit('stream_new_token', token_ids=ids, new_token_count=len(self.new_ids))

    def end(self):
        if not self.prompt_seen or self.ended:
            raise ValueError('invalid stream completion')
        self.ended = True
        self.log.emit('stream_end', new_token_count=len(self.new_ids))


def sample_stack(log, main_thread_id, elapsed, frames=None):
    frame = (sys._current_frames() if frames is None else frames).get(main_thread_id)
    log.emit('first_call_stack', target_seconds=elapsed,
             stack=traceback.format_stack(frame) if frame is not None else [],
             available=frame is not None)


class FirstCallStacks:
    def __init__(self, log, main_thread_id, began, clock=time.time):
        self.log, self.main_thread_id = log, main_thread_id
        self.began, self.clock = began, clock
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self):
        for seconds in (5, 10, 20):
            if self.done.wait(max(0, self.began + seconds - self.clock())):
                return
            sample_stack(self.log, self.main_thread_id, seconds)

    def start(self):
        self.thread.start()

    def close(self):
        self.done.set()
        self.thread.join(timeout=0.2)

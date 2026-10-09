"""Local proposal: stderr stack capture during existing guarded import only."""
from contextlib import contextmanager
import faulthandler
import sys

@contextmanager
def import_trace(stream=None, interval_seconds=20):
    """C-backed traceback timer, not a deadline enforcer or GPU-work signal.

    Repeats until cancel, emitting raw Python stacks to the preserved stderr log.
    Default20s cadence leaves snapshots before a90s gate. A future CPU-only
    import diagnostic still requires an independent external timeout.
    """
    if interval_seconds <= 0:
        raise ValueError('positive trace interval required')
    file=sys.stderr if stream is None else stream
    faulthandler.dump_traceback_later(interval_seconds, repeat=True, file=file)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()

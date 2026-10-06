"""Observe normal Python shutdown without suppressing handlers or joining threads."""
import atexit, functools, json, os, sys, threading, time

def mark(event, **fields):
    print(json.dumps({'shutdown_event': event, 'at': time.time(), **fields}), flush=True)

def inventory():
    children = None
    try:
        children = open(f'/proc/{os.getpid()}/task/{os.getpid()}/children').read().strip()
    except OSError:
        pass
    return {'threads': [{'name': t.name, 'ident': t.ident, 'daemon': t.daemon,
                         'alive': t.is_alive()} for t in threading.enumerate()],
            'child_pids': children}

def install():
    register, unregister = atexit.register, atexit.unregister
    wrappers = []
    def traced_register(fn, *args, **kwargs):
        name = f'{getattr(fn, "__module__", "?")}.{getattr(fn, "__qualname__", type(fn).__name__)}'
        @functools.wraps(fn)
        def wrapped():
            mark('handler_start', handler=name, **inventory())
            try:
                return fn(*args, **kwargs)
            finally:
                mark('handler_end', handler=name)
        wrappers.append((fn, wrapped))
        register(wrapped)
        mark('handler_registered', handler=name)
        return fn
    def traced_unregister(fn):
        unregister(fn)
        for original, wrapped in wrappers:
            if original == fn:
                unregister(wrapped)
    atexit.register, atexit.unregister = traced_register, traced_unregister
    shutdown = threading._shutdown
    def traced_shutdown():
        mark('thread_shutdown_start', **inventory())
        try:
            return shutdown()
        finally:
            mark('thread_shutdown_end', **inventory())
    threading._shutdown = traced_shutdown
    register(lambda: mark('last_observer_exit', **inventory()))
    return lambda: mark('normal_exit_initiated', **inventory())

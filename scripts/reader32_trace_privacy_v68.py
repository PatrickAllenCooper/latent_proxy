"""Held v68 trace retention primitives. No process launch or framework imports."""
import hashlib
import json
from pathlib import Path

TRACE_LIMIT = 10 * 1024 * 1024
MAX_LINE = 8192

class TraceStop(RuntimeError):
    pass

class TraceSink:
    """Shared byte cap; consume bounded chunks, retain only sanitized records.

    Raw syscall text is intentionally never accepted. The future launcher must
    parse a syscall whitelist in memory and supply structured numeric records.
    No launch is admitted by this preparation module.
    """
    def __init__(self, root, verified_roots=(), limit=TRACE_LIMIT):
        self.root = Path(root)
        self.roots = tuple(Path(p).resolve() for p in verified_roots)
        self.limit = min(limit, TRACE_LIMIT)
        self.used = 0
        self.hashes = {}
        self.files = {}

    def path_label(self, value):
        # No path outside explicitly verified roots is retained, including
        # relative paths and traversal. Never read the referenced file.
        p = Path(value)
        if not p.is_absolute() or '..' in p.parts:
            return '[redacted]'
        p = p.resolve()
        for i, root in enumerate(self.roots):
            try:
                return 'root%d/%s' % (i, p.relative_to(root))
            except ValueError:
                pass
        return '[redacted]'

    def retain(self, channel, kind, duration=None, result=None, path=None):
        if channel not in ('syscalls', 'imports'):
            raise TraceStop('unknown channel')
        if kind not in ('openat', 'newfstatat', 'statx', 'read', 'pread64',
                        'mmap', 'futex', 'import', 'timeout', 'denied'):
            raise TraceStop('unknown record kind')
        record = {'kind': kind}
        for key, value in (('seconds', duration), ('result', result)):
            if value is not None:
                if type(value) not in (int, float):
                    raise TraceStop('numeric field required')
                record[key] = value
        if path is not None:
            # Read/mmap/futex never accept buffers, pointers or paths.
            if kind not in ('openat', 'newfstatat', 'statx', 'import'):
                raise TraceStop('path forbidden for record')
            label = self.path_label(path)
            if len(label) > MAX_LINE:
                raise TraceStop('oversize path')
            record['path'] = label
        data = (json.dumps(record, allow_nan=False) + '\n').encode()
        if self.used + len(data) > self.limit:
            raise TraceStop('combined trace cap')
        if channel not in self.files:
            self.files[channel] = (self.root / (channel + '.jsonl')).open('xb')
            self.hashes[channel] = hashlib.sha256()
        self.files[channel].write(data)
        self.files[channel].flush()
        self.hashes[channel].update(data)
        self.used += len(data)

    def close(self):
        for f in self.files.values():
            f.close()
        return {'retained_bytes': self.used,
                'sha256': {k: h.hexdigest() for k, h in self.hashes.items()}}


def require_admission(manifest, now, shell_start):
    if manifest.get('cpu_admission') is not True:
        raise TraceStop('fresh approval required')
    expected = {'cpus': 1, 'memory_gib': 2, 'wall_seconds': 120,
                'check_seconds': 90, 'attempts': 1, 'gpus': 0,
                'account': 'ucb736_asc1', 'partition': 'acpu', 'qos': 'cpu-normal'}
    if manifest.get('resources') != expected:
        raise TraceStop('resource envelope mismatch')
    if now >= shell_start + 90:
        raise TraceStop('startup timeout')

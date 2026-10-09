"""Held v68 trace retention primitives. No process launch or framework imports."""
import hashlib
import json
from pathlib import Path

MODULE_LABELS = ('PIL.Image', 'torch', 'torch.distributed', 'torch.distributed.rpc', 'torch.nn.functional', 'torch._jit_internal', 'transformers.models.qwen2.tokenization_qwen2', 'transformers.models.qwen2.modeling_qwen2', 'ctypes', 'other')

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
    def __init__(self, root, verified_roots=(), limit=TRACE_LIMIT, exact_paths=None):
        self.root = Path(root)
        self.roots = tuple(Path(p) for p in verified_roots)
        if any(not p.is_absolute() or '..' in p.parts for p in self.roots):
            raise TraceStop('invalid lexical root')
        self.exact_paths = dict(exact_paths or {})
        if any(not Path(p).is_absolute() or ".." in Path(p).parts for p in self.exact_paths):
            raise TraceStop("invalid exact path")
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
        if str(p) in self.exact_paths:
            return self.exact_paths[str(p)]
        for i, root in enumerate(self.roots):
            try:
                return 'root%d/%s' % (i, p.relative_to(root))
            except ValueError:
                pass
        return '[redacted]'

    def retain(self, channel, kind, duration=None, result=None, path=None, timestamp=None, fd=None, module=None, pid=None, unfinished=False):
        if channel not in ('syscalls', 'imports'):
            raise TraceStop('unknown channel')
        if kind not in ('openat', 'newfstatat', 'statx', 'read', 'pread64',
                        'mmap', 'futex', 'import', 'timeout', 'denied'):
            raise TraceStop('unknown record kind')
        record = {'kind': kind}
        for key, value in (('seconds', duration), ('result', result), ('timestamp', timestamp), ('fd', fd), ('pid', pid)):
            if value is not None:
                if type(value) not in (int, float):
                    raise TraceStop('numeric field required')
                record[key] = value
        if path is not None:
            # Only open/stat/import and fd-attributed reads accept path labels.
            # Buffers and pointer values are never retained.
            if kind not in ('openat', 'newfstatat', 'statx', 'import', 'read', 'pread64'):
                raise TraceStop('path forbidden for record')
            label = self.path_label(path)
            if len(label) > MAX_LINE:
                raise TraceStop('oversize path')
            record['path'] = label
        if type(unfinished) is not bool: raise TraceStop('unfinished flag required')
        if unfinished: record['unfinished'] = True
        if module is not None:
            if kind != 'import' or module not in MODULE_LABELS:
                raise TraceStop('module category forbidden')
            record['module_label'] = module
        data = (json.dumps(record, allow_nan=False) + '\n').encode()
        if self.used + len(data) > self.limit:
            raise TraceStop('combined trace cap')
        if channel not in self.files:
            self.files[channel] = (self.root / (channel + '.jsonl')).open('xb')
            self.hashes[channel] = hashlib.sha256()
        self.files[channel].write(data)
        self.files[channel].flush()
        import os
        os.fsync(self.files[channel].fileno())
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

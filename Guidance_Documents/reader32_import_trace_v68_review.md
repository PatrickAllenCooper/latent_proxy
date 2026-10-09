# v68: held CPU import tracing proposal

Status: local preparation only. No submission, import, model load, generation,
package installation, or remote mutation. Previous allocation approval is consumed
by job 33629588. A new allocation needs fresh approval and a completed launcher
review. Current implementation is retention/admission primitives, not a runnable
Slurm experiment.

## Evidence and hypotheses

33629588 failed 124 after 95 seconds allocation and a 90.072-second shell check.
Three stack snapshots progressed from Pillow native import to Torch CUDA library
preloading to distributed/RPC Python bytecode import. This rules out a single
unchanging Pillow stack as the explanation for the whole observed interval.
No model loaded; model memory exhaustion cannot explain this particular startup
failure. Low CPU time (2.667 seconds) suggests waiting but does not identify its
cause. Native CUDA library loading on a CPU node is not GPU computation.

1. Shared filesystem metadata/read latency: timestamped open/stat/read durations
   clustered on shared dependency files would support this. Importtime provides
   module-level context; it cannot independently identify the awaited file.
2. Native loader dependency lookup or futex waits: long library opens/maps or
   futex waits during the native import would support this. Absence of a traced
   syscall does not exclude loader work or unobserved waits.
3. Distributed bytecode/import overhead: many short reads/imports with substantial
   accumulated duration would support this. Cumulative importtime nests imports;
   do not add cumulative durations as independent elapsed time.

Shared Pillow/Torch dependencies remain on project storage while Transformers is
staged locally. Neither this placement nor changing stacks proves filesystem causality.
CPU-node findings cannot establish GPU-node startup behavior.

## Exact proposed envelope

One attempt, 1 CPU, 2 GiB, zero GPUs; account ucb736_asc1, partition acpu,
cpu-normal QoS, 120-second allocation, 90-second check measured from shell start
including staging. Separate immutable source and sibling spool. No retries.
Retain existing guarded import sequence, library resolution, package versions,
scientific panel, scoring, and budgets. Add Python -X importtime and an own-child
strace observer only. No attach, elevation, environment dumps, execve tracing,
or unrelated process inspection.

Select only openat/newfstatat/statx/read/pread64/mmap/futex with timestamps and
elapsed syscall durations. Force raw numeric read/pread64 arguments so contents
are never rendered. Raw tracer output must remain in bounded pipes and never
be persisted. Parse a strict whitelist before retention; unexpected syntax,
permission denial or absent strace must stop and produce an inconclusive receipt,
not trigger installation or a fallback allocation.

Freeze and verify exact allowed runtime/source roots before launch. Retain labels
relative to those roots; redact all outside paths and metadata. Do not whitelist
an entire home/projects directory. Full conda package content identity has not
been verified; the final launcher must specify what runtime identity it checks.

The shared retention cap is 10 MiB across sanitized syscall/import trace files.
Use fixed-size reads and an 8192-byte maximum pending line; on cap, malformed
record, or timeout terminate only the launched process group, allow at most two
seconds grace, then kill that group. External deadline enforcement remains
mandatory. Reserve separate bounded terminal/custody metadata and report its
size; it must never contain raw trace data. No unbounded queues or raw spool.

## Local work completed and remaining gate

scripts/reader32_trace_privacy_v68.py implements fail-closed structured retention,
path redaction, a shared byte cap, exclusive creation, SHA-256 custody, and held
admission/resource/deadline checks. Five stdlib-only fixture tests pass: outside
path redaction, combined cap/custody, denial/buffer rejection, hold/deadline,
and no non-stdlib imports/launcher. No expensive imports occurred.

Still required before an allocation proposal is execution-ready: bounded streaming
parser and own-process launcher, fixture timeout/group termination and oversized
line tests, exact unchanged-import/source binding, verified runtime-root descriptor,
Slurm wrapper and compute-node tracing permission check. Existing tests establish
retention safety only; they do not establish actual tracing behavior or timeout
termination. Thus no empirical hypothesis above is resolved by these fixtures.

Fresh approval boundary: after these local gates pass, request this exact single
CPU envelope. There is no GPU follow-up admission and no authorization for further
responses, model loads, changed library resolution, or a retry.

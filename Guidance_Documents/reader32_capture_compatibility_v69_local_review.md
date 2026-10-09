# Local diagnosis of33635575 and held compatibility revision

No new allocation, remote experiment/materialization, package installation,
framework import, GPU use or model response occurred. Job33635575 remains a
terminal8-second inconclusive attempt; its original receipts/source and custody
are unchanged. Recorded charges remain2425allocatedCPUseconds/254MIGseconds.
The local v69 CLI has a separate admission flag that defaults false, so an old
v68 approved manifest cannot admit it. ACE is outside scope.

## What the sealed result actually establishes

`unexpected child output` is raised by the old parser's non-syscall output branch.
Thus the observed stop was not its malformed-syscall reason, a trace-size limit,
or the90-second timer. Both stdout and stderr were tagged `imports`; the receipt
retained no channel/type counters. Consequently it cannot identify which stream
or message caused the stop. The static `phase_codes` list is an inventory, not
observed phase events. No rejected line was persisted; exact cause is unknown.
The short capture interval and zero accepted records make startup diagnostics
plausible, but stream scheduling prevents stronger causal reconstruction.

## Confirmed local gaps and narrowly scoped fixes

1. An exact fixture `/usr/bin/strace: ptrace: Operation not permitted` reproduces
   the old unexpected-child-output branch, because it recognizes only `strace:`.
   This is a confirmed format gap, not proof that the real discarded line was a
   denial. Upstream's [error formatter](https://github.com/strace/strace/blob/master/src/error_prints.c)
   prefixes stderr diagnostics with `program_invocation_name`; the installed CURC
   version/content of its message were not collected. A tool-launch/open error,
   permission error, unsupported selector or benign startup notification are
   plausible alternatives; a Python bootstrap message is also possible.
2. A real local own-child pipe fixture exposed a descriptor-ownership defect:
   the parent closes the inherited write fd, a lazy trace sink can reuse that
   fd number, and cleanup tries to close the old number again. The integrated
   fixture failed with EBADF at sink.close before the fix. The revised cleanup
   tracks only still-owned trace descriptors. This is a latent successful-retention
   bug; it does not explain the sealed zero-record failure with a written receipt.

The new local version preserves the production tracer argv, syscall whitelist,
numeric read/pread64 arguments, output caps, path redaction, source/runtime identity
functions, scientific criteria, deadlines, and own-process-group cleanup. It adds
fixed numeric channel IDs and line-type/count metadata, with no raw rejected
contents, line digests, environment values, addresses or credentials. Actual
stdout/stderr/syscall channels now remain distinct through parsing and receipts.
Tool permission/open/selector/generic errors, Python bootstrap failures, warnings,
malformed/unknown output and unsupported import formats still stop capture.

Only an exact classic importtime header and exact stderr-only tracer attachment
notification grammar are handled as control messages. Notification count is capped
at4096; suffixes/unknown messages stop. These notifications establish neither
PID ownership nor successful useful tracing and never cause PID inspection or
attach calls. Unknown warnings are not suppressed. Exact header matching closes
the prior broad `self [us]` substring skip. Cached/importtime formats from another
Python version remain rejected. No actual CURC tracing permission is inferred.

## Local evidence

16 compatibility fixtures pass; the12 output-safe receipt,24 prior capture/binding
and3 original guard fixtures also pass (55total). The new checks cover absolute
prefix reproduction/classification; retained metadata privacy; strict headers,
JSON phases, warnings/bootstrap/fatal errors, malformed/oversize/invalid encoding;
fixed notification bounds/wrong-channel rejection; real subprocess pipe inheritance
with separate stdout/stderr/syscall fixtures and durable custody; deadline cleanup;
exact tracer argv/no attach/elevation; and unchanged source/runtime guard AST.
An actual stdlib-only Python importtime fixture passes. That toy uses `-S` solely
to avoid user site imports locally; production tracer argv has not changed.
Real local pipe success validates the descriptor lifecycle, not Linux strace's
opening of `/proc/self/fd`, kernel permission or compute-node format.

Frozen local revision and hold:
`outputs/preference_program/manifests/reader32_capture_compatibility_v69_local`.
Results: `outputs/preference_program/results/reader32_capture_compatibility_v69`.
Neither the failed v68 observer nor its manifests have been modified.

## Smallest proposed next empirical step — not approved or execution-ready

First calibrate the instrument against one owned stdlib-only toy process and a
small synthetic file, with no model/framework/runtime-archive or weight stage.
Provisional envelope:1CPU/256MiB/30-second wall,20-second external check, one attempt,
zeroGPU, ucb736_asc1/acpu/cpu-normal. This caps incremental allocation at30CPU-seconds;
actual site admissibility must be verified.256MiB is nearly3times the previous
88,800KiB RSS of the larger preparation path; the target here is much smaller.
Local toy throughput is not a CURC startup or memory guarantee.

This requires a separate calibration harness/contract, independent source/privacy
review, live identity/site checks and fresh explicit allocation approval. It cannot
reuse v68's consumed approval or call the framework qualification driver. Retain
only bounded safe counters/trace and a calibration-only receipt; no framework-
completion claim. Calibration can distinguish tool/pipe/permission/format failures
before another full guarded-import study. Denial/unknown syntax ends that sole
calibration attempt; no fallback, elevation, installation or automatic retry.
No such allocation is requested or submitted in this work session.

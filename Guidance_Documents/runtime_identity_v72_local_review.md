# Runtime identity v72 — local review, held

## Retained evidence

Job33642415's stage3/code13 receipt proves that at least one executable comparison returned a digest different from its trusted manifest binding. All four allocation checks passed. The loop checks Python before strace, but the shared failure code does not identify which comparison failed. No preparation receipt, capture, toy response or model call exists. The receipts do not establish why the binary differed, whether compute-node strace differs legitimately from login-node strace, or whether Python failed first. Do not infer a tracing permission error or replace expected hashes with observed values.

The original approved manifests, failure artifacts, consumed approvals and accounting remain unchanged:2,437 allocated CPU-seconds and254 MIG-slice-seconds. Local writer lock was owned by Codex appPID53490; no competing project worker was observed. No CURC calls or other remote action occurred during this correction.

## Minimal correction

A separate stdlib-only observed identity diagnostic records fixed Python and strace fields:expected digest, observed digest or null, bytes read, and numeric status. Status0=match,1=mismatch,2=IO failure,3=read cap,4=deadline. It records both comparisons even when one mismatches, exits nonzero on any mismatch/failure, and establishes no qualification. The manifest is never modified; no observed hash grants tracing admission. No subprocess, version query, framework import, model call or trace launch occurs.

Admission remains false before runtime reads; the two consumed calibration approvals are rejected. Source identity is separately hash-bound and capped16KiB. Runtime paths are fixed; runtime hashing streams1MiB chunks under a64MiB aggregate read cap and20-second deadline. Missing/incomplete observations remain unknown. The exact fixed receipt is capped1KiB, excludes paths/exception text, and is created exclusively then fsynced. Hashes can be inspected later without being silently promoted to trusted bindings.

## Proposed cap, not approved or submitted

One CPU,256MiB,one attempt,zero GPUs; accountucb736_asc1,acpu,cpu-normal. Slurm's scheduler wall is explicitly60seconds; the application outer watchdog is29seconds plus1-second kill grace, with20seconds for identity checks. Maximum scheduler envelope60allocated CPU-seconds; normal application envelope30seconds. Do not describe a30-second scheduler allocation. Prior identity-stage processRSS55,076KiB supports256MiB headroom for the smaller stdlib-only diagnostic; no runtime archive or weights are read. No metadata or trace cap increase:one1KiB receipt,zero trace files.

After fresh approval, preflight account/headroom and verify the frozen packet before one-shot materialization/submission. If either observed digest differs, obtain an independently trusted site identity for that executable before proposing tracing. A digest match only resolves identity; it does not establish tracer permissions or framework startup. A timeout/missing receipt gives no identity verdict and no automatic retry.

## Local validation

Four fixtures pass:held admission without reads/files; consumed approvals and changed resource rejection; actual streamed match/mismatch/missing/cap/deadline cases; recording separate runtime outcomes without rebaselining and rejecting private text in hash fields. Held Slurm shell syntax passes. Packet and manifest are frozen under outputs/preference_program/manifests/reader32_runtime_identity_v72_local. Independent review and allocation admission are still required.

# v75: startup observability fixed; ptrace permission denial identified

The prior parser accepted only strace: and /usr/bin/strace: prefixes. The pinned executable uses its own absolute path. v75 normalizes only that exact frozen path and retains up to2KiB of startup diagnostic text (maximum1KiB per line), redacting paths/credential-like values. Three local fixtures pass for prefix classification, loader-cause preservation/redaction, and the aggregate text cap. The existing45KiB capture receipt budget and64KiB aggregate metadata cap remain unchanged.

Job34861699 failed1:0 after9allocated CPU-seconds on1CPU/256MiB; consumed CPU0.208seconds, MaxRSS57,828KiB; zero GPUs. It passed source/runtime guards and produced this preserved startup diagnostic:

`strace: test_ptrace_get_syscall_info: PTRACE_TRACEME: Operation not permitted`

This establishes a ptrace operation denial for this process/deployment, rather than a missing library or executable hash failure. It does not establish whether a site policy, configuration or some other restriction caused the denial, nor retrospectively prove the exact same line in v74. No syscall records or toy receipt exist, so calibration remains failed. Phase inventories in the receipt are static, not observed events.

All six retained artifact hashes match remote custody. The permission evidence was sent as a factual update in CURC ticketINC1501930, message1a16f1ff2550ca86. No settings were changed, privileges elevated, other processes attached, or security restriction bypassed. No unchanged retry or GPU work was submitted. Cumulative diagnostics are2,458allocated CPU-seconds and254MIG-slice-seconds.

## Required resolution

CURC must identify a supported owned-child diagnostic route or confirm that ptrace is unavailable. There is no evidenced application-only correction for the permission denial. Python audit/phase instrumentation is an available alternative design, but would be separately labeled and would not satisfy the syscall open/read gate. Do not silently substitute it or claim tracing success. The user requested continued calibration; that step remains blocked on supported runtime access. The parser and error-preservation correction is complete and reproducible.

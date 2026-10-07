# Proposed first-generation diagnostic amendment

Status: proposal only. No new GPU job or resource amendment is authorized. The evidence only shows first-call latency exceeding ten seconds; it does not measure cold or steady-state speed.

## Exact minimal owner decision

Approve or decline one instrumented attempt with the first frozen case capped at **30 seconds instead of 10**, startup still **90 seconds**, and each remaining fifteen cases still **10 seconds**. Retain one H200 3g.71gb, six CPUs, 64 GiB host RAM, five-minute wall, account ucb736_asc1 and 65 GiB framework memory cap. All sixteen cases, token sequences, checkpoint, BF16, greedy generation, maximum 24 new tokens, use_cache=False, SDPA and strict 16/16 scorer remain unchanged. No extra warmup response, downloads, offload, quantization, adapters or automatic retry.

## Budget arithmetic and admission

90 startup + 30 first case + 15 × 10 remaining = 270 seconds. Reserve 15 seconds for export through study-start +285, and 15 seconds for shell/Slurm cleanup through +300. Bound every generation deadline by +270. Scoring, telemetry, monitor overshoot and scheduling consume the same envelope; stop incomplete if the remaining-time gate cannot preserve export reserve. Do not extrapolate a cold first-call duration across the remaining cases; use independent phase caps and conservative remaining-time admission. No completion or correctness guarantee.

## Cause-specific instrumentation to review before execution

Record shell-start, import, model-load, first-generation and export timestamps. Flush main-thread stack samples at 5, 10 and 20 seconds of the first call, cancelling timers on return. Review a diagnostic token-ID streamer against the installed runtime on CPU: observe the same generate call, separate prompt IDs from new-token IDs, and flush arrival times and partial IDs. Partial output remains excluded from scoring. No extra generation or changed generation options. GPU-to-host synchronization from a streamer can perturb timing and must be disclosed; stack samples alone do not prove kernel utilization. Check source integrity and CPU instrumentation compatibility before any admission.

## Risks and evidence limits

Thirty seconds is a bounded diagnostic allowance, not a measured sufficient bound. The observed first-call duration is a lower bound from a timeout; steady-state throughput is entirely unmeasured. Nothing retained proves lazy imports or warmup caused the delay. Later calls may exceed ten seconds. Host RAM is near its limit; telemetry overhead, CUDA asynchrony, polling and export variance can consume slack. Preserve every failure, partial and charge. No GPU retry is in flight.

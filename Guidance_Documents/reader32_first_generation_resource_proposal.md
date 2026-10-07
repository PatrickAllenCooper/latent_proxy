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

## Frozen implementation ready for review

Source: outputs/preference_program/manifests/reader32_qualification_v65/source; hashes: source_manifest.json. The execution freeze explicitly sets GPU_admission=false and resource_amendment_approved=false. The runner refuses before framework imports without both gates. Current files are a review freeze; any approved admission must create a separately hashed approval binding, rather than silently editing this record.

Fourteen CPU tests pass. They cover 90+30+15×10 arithmetic, capped deadlines at +270, conservative admission reserving every remaining declared case cap, +285 export cap, sixteen-call quota, strict scorer, prompt/new-token separation, 24-token limit, partial streams excluded from scoring, stack targets/cancellation, immutable scientific model/token/generation-call AST except the streamer argument, nested-spool refusal and unapproved CLI refusal. No model was loaded, no response was generated and no GPU was requested for these checks. Source parses and Slurm shell syntax passes.

The pinned runtime archive and extracted generation module match remote hashes. Its actual generate/_sample AST supplies input_ids.cpu() to the first callback, next_tokens.cpu() per token, and end() on return, and supports greedy argmax. This is CPU source/protocol compatibility, not a live framework or CUDA integration test. Stack samples run only for the first case at 5/10/20 seconds and cancel when it returns. Optional streamer is enabled in the review freeze only for that same first call; returned IDs must match observed IDs before scoring. Later calls receive streamer=None. Instrumented timings include CPU synchronization and disk writes and are not directly comparable to uninstrumented timing.

The shell records a fractional study-start timestamp and adds an external +285 worker timeout with at most two seconds of TERM-to-KILL cleanup, within +300 Slurm wall. The Python monitor retains +90 startup, per-case bounds and a +270 generation ceiling; result export is checked before and after writing. A complete receipt alone is insufficient for acceptance: require terminal success, no budget_stop.json, matching custody, and a GPU_complete event before +285. Polling, filesystem stalls and telemetry scheduling remain uncertainties; no completion guarantee. Budget arithmetic is an envelope, not measured throughput.

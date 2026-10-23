# Runtime identity diagnostic v72 — decisive mismatch localization

The authorized identity-only job34817926 ran for5seconds using1CPU/256MiB,47,848KiB peakRSS and0.149consumed CPU-seconds. Scheduler FAILED1:0 is the designed nonzero outcome for an observed hash mismatch. The diagnostic successfully recorded both executable identities in591bytes, below the1KiB cap. All receipt schema and local/remote SHA256 custody checks pass. No tracing, framework imports, model calls or GPU allocations occurred.

Python:31,498,184bytes, observed and expected SHA256 both9d27cfcde7e3128a1b0cd86e7bf8020fc1c8210dbc485bcf878ab7ef8a5c7516.

Compute-node strace:2,029,832bytes, observed SHA256c6e26537178d0b51356ecf30fceb7be7654a097a61290f812203ea96cfaf04eb, different from the login-derived expected e85f6a7cc0d125339704a397eac1a2b2d2eeb243d57608dce277cd6373f3f1b7.

This localizes the current identity failure to strace, while confirming the shared Python binary on this run. It does not retrospectively establish which comparison failed in older job33642415, whose receipt lacked per-executable outcomes. Nor does it establish why strace differs, that either binary is unsafe, a permission failure, successful tracing, or framework qualification.

The source packet and manifest were hash-verified after interrupted transport recovery. The first transfer stalled and the shared connection dropped; no submission occurred during those failures. After reconnection, direct authenticated SSH transfer completed, all three hashes matched, sbatch test-only passed, and an exclusive claim admitted exactly one job. No other project job was modified. Trusted runtime bindings remain unchanged. The original held manifest remains false, with fresh authorization recorded in a separate approved manifest.

Cumulative allocated diagnostic accounting is now2,442CPU-seconds and254MIG-slice-seconds. The intended1CPU/256MiB request matched allocation; the60-second scheduler and30-second application envelopes were respected. Runtime hashing consumed approximately0.309seconds. No retry or GPU qualification was submitted.

## Next gate

Obtain an independently trusted reference for the site-installed compute-node /usr/bin/strace (package/build identity and expected binary digest). Do not automatically replace the binding with the observed digest. Review that reference before separately admitting the toy tracer calibration. Python no longer needs speculative rebaselining on the strength of this result. If different node images legitimately carry different tracer builds, bind an approved image/build-specific identity rather than assuming the login-node binary is universal.

A precise question for CURC support, if needed: “An owned CPU diagnostic observed /usr/bin/strace SHA256 c6e26537178d0b51356ecf30fceb7be7654a097a61290f812203ea96cfaf04eb (2,029,832bytes) on an acpu allocation, while the login-node reference was e85f6a7cc0d125339704a397eac1a2b2d2eeb243d57608dce277cd6373f3f1b7. Can you confirm the approved compute-image package/build and digest, and whether owned-child strace is supported?” This question has not been sent.

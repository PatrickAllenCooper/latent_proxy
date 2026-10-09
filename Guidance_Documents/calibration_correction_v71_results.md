# Calibration correction v71

Implemented and locally tested a fixed numeric early-failure receipt and an outer timeout of 29 seconds plus one-second kill grace. Scheduler wall limit is explicitly 60 seconds because Slurm rounded the earlier subminute request. The application check remains 20 seconds, memory 256 MiB, one CPU, zero GPUs. No raw error text is retained. Fixed failure receipt is capped at 256 bytes; it and the 143-byte terminal receipt fit the unchanged 1 KiB terminal metadata budget.

Fresh authorization: user's “Perform this correction and proceed.” Live connection, account/partition headroom, source hashes and login-node executable hashes were checked before exclusive one-shot submission. The unrelated two-CPU job was preserved. Job 33642415 failed 1:0 after seven seconds, peak RSS 55,076 KiB, consumed CPU 0.188 seconds. No preparation or capture receipt exists.

The new diagnostic saved stage 3, code 13, with all four allocation checks true. This establishes failure of a runtime executable digest comparison before capture. It does not identify which executable failed: Python and strace share this code. Login-node digests were verified, so login-node identity does not suffice to bind the compute-node runtime. No evidence establishes ptrace denial, model incompetence, or GPU idleness. No model was called, and no tracing was attempted after the failed identity guard.

Both safe receipts and empty scheduler logs were copied locally; their SHA256 hashes match remote artifacts. Final ledgers are saved locally and remotely. Historical allocated accounting is 2,437 CPU-seconds and 254 MIG-slice-seconds. No retry was submitted.

## Concrete next correction, held

Separate the runtime checks into numeric Python/tracer stages. Bind a compute-local system tracer identity separately from the shared Python identity: a bounded CPU qualification should record each executable digest and fixed version classification before any trace launch. Do not silently accept a digest mismatch, relax the Python binding, install packages, or treat an observed digest as independently trusted approval. If compute-node tracer variation is legitimate, review the observed binary identity against the site-installed tracer before separately admitting tracing. This is a runtime identity/placement decision; additional GPU spending cannot resolve it.

The scientific v70 protocol, tool API, strict scoring and disjoint datasets remain ready; its ten tests and the twelve original calibration fixtures remain valid. The two correction tests passed, including exact allocation-failure code/booleans, absence of stderr, frozen hashes, and watchdog settings. Reader qualification, comparative smoke, optimization and training remain held pending a verified CPU execution path.

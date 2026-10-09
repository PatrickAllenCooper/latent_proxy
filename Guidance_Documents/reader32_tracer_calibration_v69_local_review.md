# Owned stdlib tracer calibration: local review packet

Prepared a separate held calibration packet. It reads one frozen 74-byte synthetic file through an owned Python child and requires both the toy completion receipt and sanitized observed file-open/read records. It does not import frameworks, load a model, or establish framework qualification.

## Local evidence

All 12 calibration fixtures pass: held and consumed approval rejection; exact resource and deadline admission; frozen source inventory and mutation/symlink checks; fixed privacy-safe metadata and pre-write caps; actual stdlib toy success and invalid input failure; phase ordering; receipt-only failure; owned-child command construction; held CLI with no output; import inventory; and Slurm resources. The held wrapper also passes `bash -n`. Synthetic trace fixtures establish parser behavior only; no Linux/CURC tracer execution was performed.

The frozen packet contains three source files and six controller files, totaling 37,766 bytes. Remote held manifest SHA-256: `803f93ed7651ea084ad318abe7c5c4befc0f89f3bbe0af173b8fcc9c587b31be`.

## Proposed bounded allocation

One CPU, 256 MiB, 30-second wall limit, 20-second check deadline, one attempt, zero GPUs; account `ucb736_asc1`, partition `acpu`, QoS `cpu-normal`. The requested memory is approximately 2.95 times the previous larger preparation job's observed 88,800 KiB RSS. This toy avoids framework imports and runtime archive extraction; source inventory is bounded to 256 KiB and runtime hashing streams in 1 MiB blocks. Site admissibility and actual demand remain unverified until a separately admitted calibration.

Retained metadata remains capped at 64 KiB: preparation 15 KiB, capture 45 KiB, toy receipt 3 KiB, terminal receipt 1 KiB. Sanitized trace remains capped at 10 MiB. No cap increase is proposed.

## Admission and accounting

Calibration admission, allocation approval, and remote materialization readiness are all false. The tracer digest is unbound. A fresh approval, independent review, live identity/headroom checks, and separate frozen materialization are required before any submission. The consumed v68 approval is explicitly rejected. No remote action, new job, or compute charge occurred during this preparation. Historical accounting remains 2,425 allocated CPU-seconds and 254 MIG-slice-seconds. Prior failures and receipts remain preserved.

Prepared files: `scripts/reader32_tracer_calibration_v69.py`, `scripts/reader32_calibration_contract_v69.py`, `scripts/reader32_trace_capture_v69_calibration.py`, the toy and terminal writers, `scripts/prepare_reader32_tracer_calibration_v69.py`, and `scripts/slurm/reader32_tracer_calibration_v69_held.slurm`. Frozen contracts and receipts are under `outputs/preference_program/manifests/reader32_tracer_calibration_v69_local`; review metadata is under `outputs/preference_program/results/reader32_tracer_calibration_v69_local`.

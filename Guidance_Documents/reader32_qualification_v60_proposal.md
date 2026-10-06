# Bounded 32B reader qualification proposal

Status: prepared only. Patrick approved proposal preparation at 2026-10-06 16:52 UTC (`Sentinel_6dff92097b348191ac29747811496cd5`). No submission, downloads, model execution or expanded spending is authorized by that approval.

## Frozen scientific design

Question: can a preselected stronger instruction reader correctly execute explicit preferences and eligibility constraints, providing a valid baseline for later advice/tool/prompt/fine-tuning studies?

Checkpoint: cached Qwen2.5-32B-Instruct revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`, BF16, no adapter, SDPA, greedy, cache disabled, maximum 512 prompt and 24 new tokens, one response per case. No prompt optimization or retries. Frozen panel and registration: `outputs/preference_program/manifests/reader32_qualification_v60/`. Generator: `scripts/prepare_reader32_qualification_v60.py`.

Sixteen distinct `(priority order, excluded attribute)` semantics exclude all 26 semantics from prior development, qualification, advice and factorial panels. Seed 2026100602 fixes selection and execution order. A/B/C/D gold counts are four each; each label has two eligible-top and two excluded-top cases. Single prose format. Retain original scorer SHA256 `cbc048bb2c00ee768b10357101db499dd32b70dccae7b6486d95254dd4e1510e`. Pass requires 16/16 strict correct, no parsing/eligibility/priority errors, valid custody and resource checks. Partial output or resource failure is incomplete, not a semantic pass or failure. Preserve original failures and report every response, including capped outputs.

Local preparation checks confirmed semantic disjointness, uniqueness, balance, gold scoring and rejection of explanatory answer strings. No tokenizer/model calls occurred. Full pinned tokenizer validation and runtime verification remain CPU-stage gates.

A pass would qualify this small reader instrument only. Even 16/16 has a two-sided exact 95% lower success bound around 79%; it does not establish population-level reliability. Different historical panels cannot establish a causal model-size advantage. Advice verification stays on hold pending review, even after a pass. Deterministic selection remains an oracle control and does not replace the reader.

## Evidence and feasibility

Read-only v59 inventory found all 17 index-listed BF16 shards, 65,527,841,856 file bytes (~61.03 GiB), plus config/index/tokenizer. File sizes/existence are observed; weight hashes/load integrity are not. Current 35 GB MIG and 12 GiB framework cap cannot fit these weights.

Smallest plausible GPU type is one `h200_3g.71gb` MIG slice on `ah200`, account `ucb736_asc1`, explicit `gpu-normal` QoS and typed GRES. Site inventory showed this type exists; current free slice, quota and account headroom require fresh checks. Do not substitute a full H200, multiple GPUs, CPU offload or quantization automatically.

Prior 1.5B measurements: ~3.09 GB CPU preparation took 18.62 seconds; GPU model load 2.66 seconds, total GPU job 28 seconds, framework peak ~2.92 GiB, real CUDA spans and generated tokens. These are historical measurements on another model. Linear CPU-byte extrapolation is ~395 seconds for 65.5 GB, an uncertain planning estimate. 32B startup, host peak, activation/workspace demand and response throughput are unknown. One GPU is justified by memory-fitting plausibility, not measured scaling.

## Smallest proposed bounded resource decision

Approve stages independently, with no automatic retry or escalation:

1. **CPU integrity/preparation:** `acpu`, one CPU, 2 GiB, 10 minutes. Stream each shard SHA256 without loading tensors; capture symlink targets, size, config/index/runtime hashes, available bytes/inodes and timing. Hash tokenizer/template and prepare all sixteen prompts using the pinned tokenizer, preserving IDs and serialization. If imports need a different bounded CPU envelope, stop with evidence. Do not allocate a GPU for hashing, download or tokenization. Preserve receipts and all partial outputs.
2. **GPU qualification smoke:** only if CPU gates pass and fresh allocation checks permit it; one `gpu:h200_3g.71gb`, six CPUs, 64 GiB host RAM, **five minutes maximum**. Six CPUs are the smallest count compatible with 64 GiB under the previously observed 11,500 MiB per-CPU partition limit; verify the current rule. The 64 GiB host request is a conservative proposal, not a measured minimum. Use low-host-memory sequential sharded loading; no full host state-dict copy. Hard startup limit 90 seconds from process start; complete/export by 285 seconds; each generation <=10 seconds. Maximum sixteen responses, with the first frozen case serving as the initial measured GPU-work smoke (no extra response).

GPU memory hard cap: the lower of 65 GiB framework allocation and 95% of actual device capacity; report reserved memory separately. This permits only ~4 GiB beyond BF16 weights, so load success is uncertain. Record actual free/total device memory before loading; stop if capacity cannot support the planned cap or load OOMs. No precision/resource adjustment inside this attempt. Five minutes is a hard bounded feasibility probe, not a forecast that all sixteen responses finish. If the first response implies insufficient remaining time, preserve it and stop incomplete before spending the rest of the allocation.

This proposes at most 600 CPU-seconds for stage 1 and 1,800 allocated CPU-seconds plus five MIG-slice minutes for stage 2; queue time is not execution time. CPU usage may be much lower. Existing four-CPU/32-GiB/35-GB/12-GiB bounds must be explicitly revised for stage 2 before submission.

## Instrumentation, custody and stop rules

CPU timestamps: hash start/end, tokenizer/runtime preparation start/end, final receipt. GPU timestamps: process/import, load start/end, first CUDA work, each case start/end, cleanup/export. Verify work through independently attributed framework CUDA memory/events plus actual generated-token/case progress. MIG utilization may be unavailable; retain missing telemetry as uncertainty, never infer utilization from allocation or CUDA availability.

Freeze submission source commit, manifest/scorer/model/runtime hashes, account, seed, exact command and resource limits in a receipt before launch. Intended remote root: `/scratch/alpine/paco0228/latent_proxy_runs/reader32-qualification-v60`; local result root: `outputs/preference_program/results/reader32_qualification_v60`. Recheck source/ledger ownership, no duplicate running attempt, and current headroom immediately before any authorized launch. Preserve every other project's jobs. On any gate failure stop without retry, preserve logs/partials, copy final artifacts, compare requested/observed resources, audit raw prompts/generations and strict scores, and commit/push receipts and outcomes.

Operational runner/submission scripts are not yet implemented: this is a concrete preregistered proposal, not an executable launch package. That implementation follows execution authorization and fresh readiness checks. No queued dependencies or automatic continuation are created here.

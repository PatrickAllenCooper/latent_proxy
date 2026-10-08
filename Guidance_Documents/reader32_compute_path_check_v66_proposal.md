# CPU-only compute-node identity check: ready for review

Prepared locally: scripts/reader32_compute_path_check.py, scripts/slurm/reader32_path_check_v66_cpu.slurm and manifests/reader32_path_check_v66_local under outputs/preference_program. Eleven new local tests pass; shell syntax passes. No Slurm submission, install, model import/call, generation, bulk weight read or GPU allocation occurred. Approval proposal explicitly sets CPU_admission=false, and the CLI refuses it before any remote work.

## Exact request and basis

Authorize **one CPU-only Slurm job**, account **ucb736_asc1**, partition **acpu**, QoS **cpu-normal**, **one CPU, 2 GiB host RAM, 120-second wall time**, **zero GPUs**, **no requeue or automatic retry**. Bound checker runtime to **90 seconds** from shell start, leaving 30 seconds for setup/cleanup; external timeout allows at most two seconds from TERM to KILL within the wall envelope. Retain the project's established one-CPU/two-GiB preparation request (v63 completed in19 seconds, peak758,832 KiB while importing frameworks). Reduce its prior600-second wall request to120 seconds because this check uses only stdlib,24 metadata records, small configuration/tokenizer hashes, and no model-weight contents. Runtime performance on current compute storage is unmeasured; two minutes is a bounded proposal, not a measured completion guarantee. Maximum added allocation charge if approved:120 CPU-seconds, not incurred yet.

This is a **new CPU allocation** not covered by the one-GPU-attempt numerical amendment. The latest instruction authorizes local development/tests only and expressly forbids cluster submissions; new approval is therefore required before staging or sbatch. The approved GPU attempt remains unused.

## What the check establishes

Verify Slurm CPU context, intended account/partition/CPU/memory, absence of GPU allocation, current compute hostname, checker hash and explicit approval. Bind exact approved source freeze, source-file hashes, CPU custody receipt, Python identity,24-file count and17-weight count. For every cached file, record size,mtime,resolved path, literal equality and independent samefile alias evidence. Rehash only small nonweight files; preserve prior full weight digests without bulk I/O. Do not change the GPU source, custody receipt or acceptance rule. Alias equality cannot rescue literal path failure.

A pass means **the original literal guard matches in that specific CPU node's namespace**. It does not prove H200-node path resolution; the unchanged GPU guard still runs independently. Record node/job and current metadata so this limitation stays explicit. A mismatch, missing file, resource/context failure or timeout produces no automatic correction or GPU submission. Complete CPU metadata validation never becomes a scientific reader qualification.

## Custody and concurrency

Use a separately staged read-only checker/approval source and sibling spool at /scratch/alpine/paco0228/latent_proxy_runs/reader32-path-check-v66, reading the already staged approved GPU source at reader32-qualification-v65/source. Proposed receipt name includes jobID; exclusive creation prevents replacement. Preserve watchdog stops separately from candidate results. Accept only a complete positive receipt plus terminal0:0, no stop artifact and verified custody. Do not touch unrelated pending job33557434 (DeFAb). No project reader worker/submission was found in the current read-only check; app writer lock was preserved.

All earlier failed attempts and costs remain:1,717 allocated CPU-seconds and155 MIG slice-seconds, including65+90 failedGPU seconds. Frozen16 cases, strict16/16,24new-token maximum, BF16/SDPA/use_cacheFalse and approved first30/later10/startup90/generation270/export285/job300 limits remain unchanged.

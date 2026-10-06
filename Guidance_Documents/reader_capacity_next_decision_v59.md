# Reader capacity: corrected next decision

## Scientific target

The target is whether an LLM can use a user's preference representation to improve its own recommendation, and how prompting, proxy tool access and fine-tuning affect that capability. Deterministically replacing its chosen action with the oracle action changes the estimand to correctness of a constrained hybrid system. It bypasses learned representation, reader competence and advice assimilation. Keep deterministic selection as the optimal-action benchmark; in the tool arm, measure whether the LLM actually uses the returned information. A forced override requires a separately labelled architecture arm.

The v58 exhaustive 2304-case check establishes finite-task oracle correctness only. It does not validate human preferences, preference discovery, a learned RL proxy, or paper-level generality. The audited 1.5B results (11/16 qualification; 6/16 factorial) fail the frozen strict qualification gate. The factorial has two independent semantic cases, not sixteen independent users. Advice verification remains blocked.

## Read-only cached checkpoint inventory

Checked `/scratch/alpine/paco0228/hf_cache/hub`, `/scratch/alpine/paco0228/hf_cache`, `/scratch/alpine/paco0228/relocated/caches/hf`, and the home/projects Hugging Face cache paths. Qwen2.5-3B-Instruct and 7B-Instruct were absent in these checked roots. This is not a claim about every storage location.

The cached 3B checkpoint is Qwen2.5-3B **base**, revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`; it is not the instruction-reader upgrade previously proposed in v58.

A complete index-listed Qwen2.5-32B-Instruct snapshot is cached at:

`/scratch/alpine/paco0228/hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`

All 17 listed BF16 shard files exist, totalling 65,527,841,856 bytes (about 61.03 GiB). Config SHA256: `9c6772f138ef9e5b3d1c18f2c87e451bbc01f5f1a4eabb36f9bf4f53829b903e`; index SHA256: `0183543f2e6e40d3d1e863ed0b2a8c9cdaf2ee4045ce724a45cea30e9995a0af`. Chat template hash matches the 1.5B instruction checkpoint. This inventory checked metadata, shard existence and sizes; full shard hashes, load integrity and inference are unverified.

## One recommended next decision

Independently qualify the preselected **32B-Instruct reader** before resuming advice comparisons. Freeze sixteen fresh semantic cases disjoint from the original splits and both factorial cases; use one prose condition, four gold answers per label, eight eligible-top/eight excluded-top cases, and seeded randomized order. Retain the original scorer (`cbc048bb2c00ee768b10357101db499dd32b70dccae7b6486d95254dd4e1510e`) and 16/16 threshold. Preserve all original failures. Do not optimize prompts against this qualification panel. Passing would establish only a small competent-reader baseline, not broad preference learning or a causal model-size improvement; broader confirmation remains necessary.

This recommendation supersedes v58's 3B-Instruct suggestion and deterministic-substitution recommendation for the reader-capability question.

## Resource feasibility and present disposition

No model execution or new submission is authorized by this document. BF16 weights alone exceed the existing 35 GB MIG slice and 12 GiB framework-memory cap. Quantization would introduce a new precision condition and is not an implicit workaround.

The smallest plausible GPU candidate is one existing H200 `h200_3g.71gb` slice, subject to usable-memory/load measurement. Immediate availability of that slice and allocation headroom are unverified; idle CPU counts do not establish GPU availability. Host-memory demand and startup/throughput for 32B are also unmeasured. The prior 1.5B run's 2.66-second load and 28-second total GPU job cannot justify an equal runtime estimate for 32B.

Any later proposal must separately budget CPU shard hashing/staging before GPU allocation, then bound the initial load and first generation with measured memory and CUDA work/progress signals. The existing 90-second CPU diagnostic is not established sufficient for hashing 65.5 GB: scaling the prior 3.09 GB/18.62-second preparation suggests roughly 6.5 minutes, with substantial uncertainty. GPU wall time and host RAM require a bounded load measurement before scaling. Neither a larger memory allocation nor longer preparation is approved here.

Within current limits, retain the negative reader result and oracle control, and leave the advice smoke unsubmitted. This is a resource and measurement boundary, not evidence that stronger readers cannot succeed. No downloads, inference, scheduler changes or spending occurred in this inventory/design step.

# 32B GPU qualification: incomplete startup attempt

Job33507455 used the approved oneH2003g.71gbMIG,6CPU/64GiB/5minute envelope and stopped after65seconds, exit1:0. It produced zero responses. Qualification is **incomplete**, not a reader failure. No retry was submitted and advice verification remains held.

## Engineering failure and corrective action

The unchanged import guard raised `DirectoryChanged` for the source/run root before model loading. I uploaded `jobs.json` and `gpu_submission.json` into that root during startup; their timestamps are17:14:20 local, before the17:14:30 exception. This concurrent bookkeeping mutation is consistent with the failure and a local fixture reproduces the guard failure when a ledger entry is created in the indexed directory. The earlier directory signature was not retained, so the exact sole mutation cannot be proved.

The correction is to keep immutable source files in a directory separate from bookkeeping/log/output paths and make no writes into indexed source directories during imports. Three local fixtures passed: an in-root ledger addition fails closed; external bookkeeping preserves the guard; actual source-entry changes still fail. Do not weaken or disable the guard. Before another GPU admission, validate actual startup under the separated layout on CPU. No new attempt is authorized by this report.

## Resource and custody audit

Actual allocation: one71GBMIGslice for65seconds, sixCPUs for390allocatedCPU-seconds. ObservedTotalCPU5.040seconds, peakRSS1,267,228KiB. There is no model-load event, generation progress, CUDA span or attributable framework-memory measurement. Host-wide nvidia-smi memory cannot identify this job's use, and utilization isN/A. Therefore usefulGPU computation was not verified. This was a short startup failure inside the90second startup bound, not a prolonged idle allocation; repeating GPU startup without correcting orchestration would waste resources.

Retain all costs: firstCPU attempt608allocatedCPU-seconds; successful correctedCPU preparation160; thisGPU attempt390, total1158allocatedCPU-seconds plus65MIG-slice-seconds. The initialCPU consumption reports remain contradictory and are preserved. The successfulCPU receipt and17-shard digest custody remain valid.

Final logs, telemetry and incomplete stop receipt have matching local/remoteSHA256s. Raw results and audits are in `outputs/preference_program/results/reader32_qualification_v62`. Prompts, semanticIDs, tokenizer IDs, checkpoint, scorer and16/16threshold were unchanged. No scientific score is available from this GPU attempt.

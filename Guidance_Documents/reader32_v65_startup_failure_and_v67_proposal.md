# Instrumented32B attempt: startup failure, not a scientific result

Job33560980 is terminal FAILED2:0 on c3gpu-e5-u13 after99seconds. The one authorized instrumented attempt is consumed; no retry is submitted. CPU33560788 had passed the literal identity check in11seconds. All preceding failures remain preserved.

## Evidence and validity

The GPU job verified the staged source/approval/custody inputs and cached-file metadata, extracted the pinned runtime archive, and entered guarded dependency imports. No dependency_import_complete, model_load_start, generation_start or GPU memory-progress event occurred. No response file or partial-token trace exists; completed and partial responses are both zero. The only terminal artifact is an incomplete receipt with reason90_second_startup_limit and qualified=null. Do not report0%accuracy, score this as a reader failure or reuse this as preference-learning evidence.

Stop receipt timestamp1791486929.812238 is94.720893seconds after shell start1791486835.0913446, so the Python watchdog overshot its declared90-second startup limit by4.720893seconds. Imports had been in progress87.836720seconds when the receipt was written. The logs bound the failure to the guarded import operation; they do not reveal which import, metadata traversal or inspection call blocked. There is no retained stack and no proven cause of the delay or watchdog scheduling overshoot. Earlier CPU import success does not explain this GPU-host behavior.

Four remote/local artifact digests match; staged source remains read-only with zero source-hash mismatches. Extracted runtime archive SHA matches the pinned8e91c377... identity; prior full weight digests and scientific hashes remain retained. Cached weights were not bulk rehashed in this job. Conda package contents are not freshly fully hashed. No runtime/scientific identity waiver or rebaseline occurred. Empty stderr is preserved. Host-wide NVML records show memory on shared physical devices and utilizationN/A; they cannot attribute compute to this job. Real GPU computation is unverified, rather than inferred from allocation or exit code.

## Requested versus observed resources

Requested/allocated: one H2003g.71gb MIG slice, six CPUs,64GiB host RAM,300second wall limit. Observed99seconds; allocated CPU cost594seconds; TotalCPU2.684seconds; batchMaxRSS750,732KiB (about733MiB). Low CPU consumption with long import wall time is an observation, not a diagnosis. No model-load or inference throughput measurement is available.

Retained allocation totals now **2,322CPU-seconds and254MIG-slice-seconds**: preceding1,717CPU +11CPU gate +594thisGPU; GPU65+90+99. Historic discrepant CPU-consumption snapshots remain preserved and are not silently replaced or summed as reliable measured compute time.

## Useful local correction completed

New stdlib import-trace helper uses a C-backed faulthandler timer to retain stderr stacks every20seconds during the existing guarded import. Proposal runner adds callback/tokenizer-class/torch/model-class entry/exit timestamps and cancels tracing when guarded imports return. Six local tests pass, including a real sleeping-process traceback/cancellation fixture, restoration after exceptions, unchanged dependency import AST and identical model/token/generation calls. No model or cluster call was made for these tests. This corrects missing observability; it is **not** an evidenced fix for the unknown blocking call.

Frozen16cases, strict16/16,24-token output maximum, BF16/SDPA/use_cache=False, first30/later10 and90/270/285/300second limits remain unchanged. A followup proposal refuses execution without a fresh explicit admission binding and executing-source hash. The original approved source is unmodified. Stack capture may perturb timing; instrumented timing must be labeled. The trace timer is not a90second deadline enforcer.

## Smallest next approval

Approve **one CPU-only exact guarded-import diagnostic**, one CPU,2GiB RAM,120second Slurm wall,90second startup/check budget, accountucb736_asc1,partitionacpu,QoScpu-normal,zeroGPUs,no requeue/retry. These retain the prior CPU preparation resources and bounded v66 envelope. Stage the same pinned runtime on node-local storage; import the same tokenizer,torch and Qwen2 model class under unchanged guards; record nested timestamps and20second stderr stacks. Do not load weights/tokenize/generate, install packages or change cached metadata. Use an independent external timeout bounded by the unchanged90second check with at most2second TERM-to-KILL delay inside the120second allocation. Preserve every partial and stop at failure.

This additional CPU allocation requires fresh approval; the current instruction covers local work only. It may locate the blocking call or fail to reproduce it because acpu and ah200 hosts differ. A CPU pass does not prove GPU-host startup feasibility. **Any later GPU attempt also requires fresh approval** because33560980 consumed the single authorized attempt. No proposal to relax the startup deadline or scientific threshold is made.

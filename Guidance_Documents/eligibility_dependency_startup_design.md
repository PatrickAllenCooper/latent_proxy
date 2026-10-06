# Terminal accounting and remaining dependency startup gate

No new preparation, model calls or GPU job were launched. Job33464918 isFAILED124:0,107seconds elapsed. The serialized-code repair was not reached. No valid tokenreceipt exists.

## Accounting reconciliation

Submission script and ReqTRES requested4CPUs/32G; application OMP/OpenBLAS caps were4threads. Actual AllocTRES was9CPUs/32G and billing9. sacct ReqCPUS also reports9 after scheduler adjustment; it must not replace the original script/ReqTRES request in the research record.

Actual allocated CPU cost is9×107=963CPU-seconds=0.2675CPU-hours, not428CPU-seconds based on four requested threads. Observed total process CPU was4.861seconds, about0.505% of allocated CPU time. Neither application CPU time nor thread count is allocated capacity. Count job allocation once: batch/extern rows repeat the allocation and must not be summed. This is allocation accounting, not a monetary invoice.

Live acpu configuration reportsMaxMemPerCPU3840MiB andSelectTypeParametersCR_CORE_MEMORY.32G=32768MiB requires ceil(32768/3840)=9 CPUs, exactly the observed allocation. This is strong job/config evidence of memory-driven expansion, not an unexplained nine-thread application. Site submission-hook source was not inspected, so the exact adjustment implementation is not claimed. See official Slurm memory/CPU allocation documentation: https://slurm.schedmd.com/sbatch.html and https://slurm.schedmd.com/slurm.conf.html. Requested limits were not inflated or edited.

For a prospective CPU preparation request, measured peak batchRSS214252KiB and import-processRSS59640KiB provide no evidence that32G is necessary. A proposed4CPU/8G request would fall below the4×3840MiB ceiling and retain substantial measured headroom. This is a future resource proposal, not an adopted or submitted limit. The approved GPU envelope remains unchanged.

## Engineering result

The retained stack samples identify a global importlib.metadata.packages_distributions() call in the frozen Transformers import_utils.py line47. Even when Transformers lives under node-local/tmp, importlib.metadata traverses the shared conda distribution tree; the samples show inferred distribution-file existence checks through pathlib.stat. Node-local Transformers staging alone therefore does not isolate dependency startup from the shared filesystem. One fast tokenizer-file read does not address this metadata traversal. Low CPU time is consistent with waiting/metadata overhead; it does not establish storage failure or account for every second.

No runtime patch should silently override packages_distributions or fabricate distribution metadata: availability/version mapping may affect imports and model behavior. Enlarging the90-second cap would not resolve the measurement issue under current bounds.

## Smallest pending technical gate

Prepare a CPU-only dependency inventory proposal from existing installed metadata/source (no full tokenization attempt): identify which distributions lack top_level.txt and force file-based inference, record path/file counts and total bytes with finite scan caps. Preserve the exact runtime and pin environment identity. Review whether staging the actual required dependency metadata/packages into an isolated node-local environment can avoid shared traversal without changing availability/version results. A complete environment mirror is the strongest equivalence approach but its archive size/staging cost has not been measured and cannot yet be promised within45seconds.

A narrower precomputed package-distribution mapping would modify initialization behavior and require a separately versioned runtime repair with equivalence checks; it is not the unchanged-runtime attempt already approved. Direct use of tokenizer backend/Jinja is another changed tokenization implementation and is not an automatic substitute for native AutoTokenizer/template semantics.

Thus the next bounded remote preparation remains gated on an explicit repair design and stopping-rule authorization. The concrete design work can proceed locally, but no retries, qualification, advice panel or model changes follow from this accounting update. Frozen16cases/prompts/scorer and historical failures remain intact.

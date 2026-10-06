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

## Bounded metadata inventory and review-only repair prototype

An authorized standard-library component inventory (30-second external cap,25-second scan budget; no imports of installed packages, tokenizer, model or GPU) completed successfully. It read107dist-info directories in the pinned conda site-packages in6.2076seconds.41 lacked top_level.txt; their RECORD files contained12891entries. Largest inferred distributions included pandas2943,scipy2421,numpy1336,wandb1186 andmatplotlib891. These are potential file-existence checks, not measured calls; RECORD entries can include paths outside import roots and nonexistent files. This inventory covers that site-packages tree, not every sys.path finder, egg-info or custom metadata provider. Raw hashes/records/output and time accounting are retained under repair_v3/dependency_inventory.*.

The frozen runtime calls packages_distributions globally at import_utils.py line47. Its mapping supports package-to-distribution resolution (e.g.PIL/Pillow), package version checks and flash-attention availability. Python's original implementation uses declared top-level names or inferred distribution files; its files path filters missing files using existence checks. Removing optional distributions or inferring top-level names directly from RECORD without actual existence checks can change package availability/version selection and is not an equivalent fix.

Review-only code scripts/verified_distribution_map.py defines an exact-map cache contract: accept only a receipt produced by the original packages_distributions function, bound to a pinned environment digest and original metadata-source digest, validate the serialized mapping hash, return defensive copies, temporarily supply it only during single-threaded startup and restore the original function on success or exception. It leaves version() and find_spec() unchanged. Local synthetic fixtures cover identity, stale bindings, tampering and restoration. No production cache was created, no runtime patch adopted, no import/preparation/qualification rerun performed. The helper is not by itself an environment verifier; a caller must verify a complete immutable environment snapshot/digest, including all sys.path distribution providers and relevant file existence, before use. Hashing only METADATA/RECORD is insufficient if installed files or import paths change.

Smallest remaining gate: specify and freeze an authoritative cache producer plus full environment identity/custody contract, then authorize one bounded CPU diagnostic to produce/compare the mapping under the unchanged90-second preparation cap. If the original mapping cannot be produced within that bound, this cache route is not qualified; do not fabricate it or increase the cap. A complete node-local environment mirror remains an alternative but its size/staging feasibility is unmeasured. Cached mapping injection would be a separately versioned startup repair, so production equivalence and effective runtime hashes must be reviewed before any tokenization/qualification. Frozen cases/model/scientific gates remain unchanged; no automatic new job follows from these engineering fixtures.

## Authoritative capture diagnostic v46 — terminal evidence

A separately frozen90-second CPU diagnostic called Python's original packages_distributions without importing Transformers, tokenizer or model. Job33465075 requested4CPU/8G and actually allocated4CPU/8G; reducing memory avoided the prior9CPUexpansion. It failed124:0 after97Slurmseconds (388allocated CPU-seconds;2.108seconds observed TotalCPU). Mapping capture itself timed out at90.00seconds, with0.30user/0.39system seconds and27088KiBpeak processRSS. No mapping or complete environment identity file was produced. Rawlogs/accounting are retained under outputs/preference_program/results/metadata_capture_v46.

This isolates the bottleneck to original environment metadata enumeration independently of model/tokenizer/Transformers loading; it does not establish a filesystem malfunction or identify each slow distribution. The exact-map cache route cannot be called sound in production because its authoritative input and complete pinned identity remain absent. No fabricated/partial cache was accepted, no repaired runtime deployed, and no qualification/GPU/preparation retry followed.

Engineering conclusion: unchanged original-map capture in the current shared environment is not viable within the fixed90-second cap on this observed attempt. Repeating it blind is not justified. The smallest remaining material repair choice is a separately bounded immutable dependency package/metadata staging design (or exact filesystem-index algorithm with independently verified semantic equivalence), rather than a larger timeout. Neither full environment copying nor changing the mapping algorithm is silently authorized by these failed measurements. Existing cache-contract fixtures remain useful guardrails, not a qualified deployment.

## Exact directory-index repair implemented and locally validated

scripts/indexed_metadata_exists.py retains the original packages_distributions algorithm and temporarily accelerates its Path.exists calls using one os.scandir index per parent directory. It does not infer mapping names from unchecked RECORD text, drop distributions or change version lookups. Missing names returnfalse; symlinks fall back to the original existence check so dangling targets remainfalse; inaccessible/special paths also fall back. Before/after directory signatures detect mutation during enumeration, and a final signature check rejects changed directories. Original Path.exists is restored even when capture raises. This is a separately versioned, single-threaded CPU startup mechanism, not a silent patch to the pinned Transformers archive.

Local tests passed8real filesystem cases (regular file, directory, missing paths, live/dangling symlinks and parent traversal), detected mutation, verified restoration on both return/exception, and compared the actual standard-library mapping algorithm on isolated METADATA/RECORD including an absent recorded package. No production environment mapping or tokenizer/model import ran. Expected improvement is fewer shared filesystem existence operations (directory enumeration rather than thousands of individual stats); no remote throughput claim is made.

Remaining caveats: finite directory signatures are not a cryptographic whole-environment identity and cannot prove immutability against all concurrent adversarial changes. Symlink-target changes outside indexed directories require frozen environment custody; permission/access changes and exotic metadata providers need production validation. Global monkeypatching is restricted to a single-threaded isolated process with no concurrent imports. Complete pinned environment identity and production mapping remain unestablished, so the prior cache contract is not qualified by these local fixtures.

Smallest next engineering gate is a separately frozen CPU-only original-mapping capture using this index within the unchanged90-second cap and at most4CPU/8G, with recorded source/environment paths, directory signature validation, authoritative mapping output and version associations. That diagnostic is distinct from stopped tokenization preparation and model qualification. It may establish a usable mapping path; it does not itself establish full environment identity, tokenreceipt or permission to submitGPU. No remote attempt, preparation retry or model response was made in this turn.

Local fixture custody note: system Python3.9 lacked packages_distributions, so its extended original-mapping test initially errored after the8filesystem tests passed. Re-running locally with installed Python3.13 passed all fixtures, including actual stdlib mapping equality. The target CURC interpreter is3.12, so target-version integration remains unvalidated. Both the initial local compatibility failure and successful interpreter-specific result are recorded; neither is remote preparation.

## Python3.12 integration v47 — successful bounded diagnostic

Explicitly authorized changed diagnostic job33465269 completed0:0 in32Slurmseconds on4actually allocated CPUs/8G (128allocated CPU-seconds;3.271seconds observedTotalCPU). It retained the90-second external diagnostic cap and generated no model responses.

Original stdlib packages_distributions with the directory index completed in12.955s:105package keys,12891exists lookups across1329indexed directories. Path.exists was restored before calling the unmodified original algorithm, which then completed in4.994s; the mappings were exactly equal. Every package-distribution version association matched the same installed metadata. Before/after hashes of Python executable/source, sys.path and distribution metadata were equal. Final receipt confirms mapping/version/metadata equality and restoration, and the Slurm process terminated normally. Rawmapping, directory signatures, before/after identity bindings, stage logs and accounting are preserved under outputs/preference_program/results/metadata_index_v47.

This validates semantics on the observed pinned Python3.12 environment. It does not prove indexed startup is faster: indexed-first ordering can warm filesystem caches, and the later original call was faster. Earlier original90-second failures and this warm success must both remain visible. It does not establish whole-environment content immutability; the receipt deliberately marks runtime deployment unqualified. Production preparation sys.path and source locations differ from this diagnostic, so the captured mapping must not be substituted blindly into the cache contract.

Concrete next gate: freeze a production-path startup integration using the validated index (retaining original mapping semantics and restoration), bind its runtime/source/interpreter/sys.path metadata, and define immutable environment custody or complete file-content verification before cache reuse. A direct index integration avoids reusing a stale cached mapping but still requires a new separately versioned startup contract and bounded preparation authorization. No stopped preparation was rerun, noGPUqualification/advice panel launched, and no limits enlarged.

## Production-path startup v48 — failed bounded integration

Changed startup job33465529 froze source8d010e5 before outputs. It verified the original runtime archive during node-local staging and passed metadata binding to the v47Python executable/source/distribution metadata. The startup hook invokes the original mapping with the validated index rather than reusing a cached mapping; local hook/restoration fixtures passed before submission.

However, AutoTokenizer class import did not complete within90seconds. SlurmFAILED124:0,102elapsed seconds,4allocated CPUs/8G,408allocated CPU-seconds,5.580seconds TotalCPU. Startup subprocess90.03s and567960KiBpeakRSS; whole batch peak874608KiB. No tokenizer was loaded or chat template applied, no model responses/GPUjobs were generated, and no startupreceipt exists. Only pre-import metadata is verified; after-import equality and in-process restoration are not proven on this stopped job. The external timeout terminated the process and Slurm is terminal.

The frozen v48script did not emit intermediate mapping-hook markers or periodic stacks, so the remaining delay cannot be assigned reliably to indexed mapping versus later package imports. This instrumentation limitation is retained explicitly rather than claiming a cause from RSS/I/O alone. v47mapping equivalence is still established but does not establish end-to-end startup readiness. Full package-content immutability also remains unproven, with failclosed qualification flags.

Production readiness is therefore false. Next useful engineering work is instrumentation of the remaining import/dependency phase and a revised dependency staging design, not an unchanged retry or automatic tokenizer/modelqualification. The90second preparation cap, frozen16case assay and historical failures remain unchanged.

## Finer import trace v49 — observed successful startup and remaining stage

Authorized changed instrumentation job33465881 completed0:0,95Slurmseconds,4actually allocated CPUs/8G,380allocated CPU-seconds,7.221seconds observed TotalCPU; batch peakRSS1184668KiB. It preserved45-second staging/90-second startup caps. Verified node-local runtime archive; metadata bindings before/after import matched and hooks restored. No tokenizer loading, template application, model responses orGPUjobs.

Mapping hook markers show3.6208seconds for12891lookups/1329directories. AutoTokenizer class import then completed in58.2687seconds total. A15-second faulthandler sample locates later genericpath.exists/inspect.getsourcefile/getmodule filesystem queries during torch custom/fake-op registration. Importtime records nested Transformers auto_factory (~42.36s cumulative) and distributed/FSDP (~22.43s cumulative) paths; these cumulative durations overlap and must not be added. SymPy and accelerate imports also appear. Thus the directory index addressed metadata mapping, while substantial source inspection and dependency loading remain on shared package paths.

This is an observed pass of CPU startup integration, not a tokenreceipt or modelqualification. Earlier v48timeout remainsfailed; filesystem caching, node conditions, trace overhead and order differ, so v49does not prove reliable timing or retroactively explain every second of v48. Full content identity/immutability remains unproven. Rawsampled stacks, import profile, receipt and accounting are retained under outputs/preference_program/results/import_trace_v49.

Concrete next engineering checkpoint: freeze a targeted unchanged-content dependency staging contract for the observed PyTorch/Python source-inspection paths, with content hashes, path/version equivalence and CPU resource/staging feasibility before changing startup again. Alternatively, review whether a controlled new preparation may accept the existing pinned archive+metadata identity rather than full content identity; this is an explicit custody decision, not something silently downgraded by an observed import pass. No further retry/preparation/GPUwork was triggered in this diagnostic turn.

## Targeted dependency source staging — concrete design and fixtures

The next engineering implementation scripts/stage_verified_python_sources.py copies exact Python source bytes for the observedtorch/sympy/accelerate packages into a fresh local tree; non-Python assets become explicit links to their original paths and each is content-hashed. It verifies source stability during copying, validates staged versus source hashes, rejects unsupported symlink directories and preserves partial trees on failure. It does not import any dependency. Source/native mutation fails validation. Local real-file fixtures passed Python-copy/native-link identity and tampered-native rejection; scripts/test_verified_source_staging.py retains those checks.

A separately bounded30-second CPU-only file inventory timed out before a complete package receipt, so actual tree-size/staging-time feasibility remains unmeasured. A different10-second-capped RECORD-only component read completed: torch2285Python entries/46445093declared sourcebytes (~44.29MiB), sympy1532/26168818 (~24.96MiB), accelerate84/1491684 (~1.42MiB). Total3901Python entries/74105595bytes (~70.67MiB). All three package trees declare1119185305bytes (~1.04GiB), dominated byTorch native/data assets. These are declared manifest sizes, not actual existence/content verification; missing entries andpycache/native auxiliary files differ. Rawcomponent logs and receipts are preserved; no model or tokenizer was loaded.

This design targets the observed shared-filesystem source-inspection paths while preserving real binary/data assets; it does not pretend that native symlinks eliminate all shared reads. Content-hashing those assets has a real CPU/I/O cost that must fit the bounded stage, and the builder is not yet executed onCURC. Package __file__/source paths change intentionally under staging, which may affect relative resource resolution or inspection; an independent import-equivalence check is therefore required before adoption. Full environment identity remains unproven outside the selected packages.

Next checkpoint: freeze a CPU-only builder/validation attempt with the existing4CPU/8G/90-second diagnostic ceiling, preserve a complete hashed asset manifest only if it finishes, and compare startup/version/resource-path behavior under the staged tree against the pinned originals. Do not reuse thefailedfilewalk unchanged, infer timing from declaredsizes, or submitGPU/preparation until applicable scientific and custody gates pass. This turn added concrete code/localfixtures andboundedcomponent evidence; it did not deploystaging or generate responses.

## Exact source staging v50 — measured bound failure

Authorized frozen job33466454 used4requested/allocated CPUs,8G,45-second runtime-stage and90-second dependency diagnostic caps. Runtime/source hashes were verified and metadata validation passed. Per-file staging timed out124:0 before complete staging or import/version validation. Slurm114seconds,456allocated CPU-seconds,5.101seconds TotalCPU; batch peak1166228KiB. Diagnostic90.00seconds,1.82user/1.38system seconds,53464KiBprocessRSS and2066512filesystem-input units reported by time.

The terminal hash journal contains3025successfully verified entries:882copied Python sources and2143linked assets,1022103687contentbytes. These counts are partial, not the declared complete package universe. Full package-content identity, later source revalidation and import equivalence were never established. Agent actions preserved logs/journal/source and did not delete the node-local partial tree. No tokenizer/model responses orGPUjob occurred.

This demonstrates that the full per-file source/native staging builder did not fit the observed90-second envelope. It does not show hash computation alone is slow; low CPU/large filesystem-input and many path operations leave shared I/O/metadata contributions unresolved. The preserved partial paths/hashes allow a narrower dependency packaging plan without a blind rerun.

Next design checkpoint: use the observed import closure to distinguish necessary Python/source-inspection dependencies from unneeded headers/tests/assets, while retaining exact binary/resource availability through a reviewed manifest. Omitting files changes available package contents and therefore requires an explicit equivalence claim scoped to startup rather than a false full-environment identity claim. A complete immutable archive built across multiple bounded CPU stages would need a separately reviewed aggregate resource budget; no such expansion is implied here. Current qualified evidence remains v47mapping equivalence/v49one startup pass; production preparation and the16response qualification remain gated. No automaticretry or larger cap was used.

## Narrow source/inspection contract refined locally

The preserved import trace selects1031torch,419sympy and49accelerate module names (1499total), rather than blindly walking/hash-copying every package asset. This is an observed import closure, not a complete set of required files or Python modules: some names resolve native extensions or aliases, and unobserved imports must continue using the original runtime. A future bounded resolver must determine actual SourceFileLoader origins; text module names alone are not file identity.

scripts/verified_source_cache.py defines a narrower byte-cache contract preserving original source filenames/compiledco_filename, package paths and native/resource references. Local fixtures verify exactbytes/codeconstants/origin and reject original/cache mutation. No importer was deployed. This preserves availability by falling back to the original tree rather than pruning it; it does not automatically accelerate inspect.getsourcefile/getmodule because those paths can still stat originalfiles. Original-byte revalidation itself reads sharedsources, so no performance/capfit claim follows. Rawnarrowcontract/fixture receipts are saved under import_trace_v49.

A more directly targeted review-only component scripts/indexed_source_inspection.py applies the already validated directory-existence index to os.path.exists/genericpath.exists during isolated single-threaded startup. It preserves originalfilenames, installed package contents and native references; bytes/file-descriptor/custom path arguments fall back to the original function, symlinks retain original target-existence behavior, directory mutation fails validation, and originals restore on return or exception. Local real-filesystem fixtures passed existence/symlink/bytes equivalence and restoration. This targets the exact inspect stack fromv49without repeating broad content-copy staging. It is not remotely validated or adopted.

Recommended next engineering gate is one separately frozen boundedCPU diagnostic combining validated mapping indexing with this changed inspection-existence component, using the unchanged trustedruntime and source/metadata bindings, periodic stacks and source-origin checks. It must distinguish observed existence semantics from whole-environment immutability and measure actual resource use before production adoption. This is narrower than thefailedfull-builder and introduces no removed runtimefiles; it changes startup filesystem lookup implementation and therefore needs integration evidence. No remote attempts, preparation/modelresponses/GPUjobs were generated while preparing these refinements; frozen90second cap/qualification gates remain unchanged.

## Combined mapping and inspection diagnostic v51 — terminal failure after valid receipt

Frozen revision `aed2c05`; CURC job `33467478`, acpu/cpu-normal,
account ucb736_asc1. Actual allocation four CPUs, 8 GB, no GPU. Runtime
archive verified and staged in 2.81 seconds. Original metadata mapping
completed in 2.90 seconds (12,891 lookups, 1,329 indexed directories).
AutoTokenizer class import completed in 70.20 seconds; complete receipt was
written 71.22 seconds after diagnostic start. Inspection indexing handled
2,316 lookups across 349 directories. The receipt confirms restoration of
all patched functions, unchanged targeted source hashes, expected module
origins and unchanged interpreter/distribution metadata. No tokenizer,
model, chat template or responses were produced.

**Terminal gate failed:** external timeout returned 124 after 90.05 seconds,
Slurm FAILED 124:0, 103 seconds total, 412 allocated CPU seconds, 6.732
observed CPU seconds, batch peak RSS 992,600 KiB. A complete JSON receipt
is not a successful process exit. The post-receipt delay has no captured
shutdown stack because the diagnostic cancels faulthandler after import.
Do not infer its cause or bypass it with forced successful exit. Preserve
this failed run and do not repeat unchanged or submit GPU work.

Next informative engineering step: retain traceback timers through normal
interpreter termination and timestamp receipt flush, exit initiation and
shutdown callbacks in a bounded CPU diagnostic, if authorized. This would
distinguish shutdown I/O or callback delay from dependency import delay.
Full combined reference equivalence and complete installed package-content
identity remain unproven; prior local component fixtures and v47 mapping
equality do not establish those stronger claims. Qualification readiness
remains false. All logs, accounting, receipt and audit are under
`outputs/preference_program/results/combined_index_v51` and preserved remotely.

## Shutdown observation v52 — completed; earlier delay not reproduced

Changed instrumentation revision `989955b`, job `33467763`, four actual CPUs,
8 GB, no GPUs/responses, unchanged 45-second staging and 90-second diagnostic
caps. Normal atexit callbacks and unregister semantics retained; thread
shutdown delegates to the original function; no forced exit, handler skip,
cleanup suppression or resource expansion. Local subprocess fixtures verified
registration return value, unregister behavior, handler execution and thread
shutdown before submission.

Terminal COMPLETED 0:0, 31 seconds total, 124 allocated CPU seconds,
7.337 observed CPU seconds, batch MaxRSS 780,028 KiB. Class import took
12.01 seconds; integrity/restoration receipt completed after 13.18 seconds.
All eleven observed registered exit handlers returned. Observed Python
shutdown span was 0.0134 seconds, with weakref finalization the longest at
0.0107 seconds. Only the main thread and no child PIDs were present at
exit initiation, thread shutdown and the final observer callback. These
observations rule out an observed Python callback or surviving child/thread
obstruction **for this run only**. They do not explain v51: its delayed
termination was not reproduced, and instrumenting handlers is not a causal
repair. Native finalization after the final callback is not attributed by
these Python observations. Node/cache/IO variation and instrumentation
remain uncontrolled; the 70-to-12-second import difference is not a proven
speedup. v51 remains FAILED and preserved.

No specific blocking handler was identified and no tested repair can be
claimed. The next justified gate is a bounded CPU tokenizer/token-ID receipt
using normal-cleanup observation and existing strict token normalization;
model qualification still requires that separate receipt, complete terminal
success and its own resource/identity checks. No further job submitted here.

## CPU tokenization v53 — partial equality checks; terminal gate failed

Frozen revision `62aba39`, job `33468346`, unchanged four actual CPUs/8 GB,
45-second runtime staging and 90-second diagnostic timeout. FAILED 124:0,
105 Slurm seconds, 420 allocated CPU seconds, 30.388 observed CPU seconds,
batch RSS 1,111,916 KiB. Import 50.18 seconds; tokenizer load 0.535 seconds.
Twelve of sixteen cases completed all three-path integer-ID equality checks,
each 122 tokens; timed out during qualification-12 (the thirteenth case).
No full token receipt, final serialization/identity checks or normal terminal
cleanup; no inference/GPU. All logs/accounting preserved locally and remotely.

Source audit found `len(tokenizer)` inside the per-token bounds predicate,
calling vocabulary-size calculation 122 times per case. That unnecessary
repetition is a plausible source of roughly two-second case validation,
but no per-operation timing establishes its contribution yet. Proposed v5
hoists the vocabulary bound once and verifies unchanged size after all
cases, retaining every ID/equality check. It also journals completed token
IDs before advancing, so partial progress remains recoverable. Local fixture
checks preserve bounds with constant call count; syntax passes. V5 has not
been submitted or integrated against the real tokenizer. No causal timing
or shutdown repair claimed; qualification remains gated on complete token
receipt and successful normal process termination.

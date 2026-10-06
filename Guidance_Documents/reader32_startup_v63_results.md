# Isolated source startup repair: validated on CPU

Job33507600 completed0:0 in19seconds (extern step20seconds), on oneCPU, with6.641seconds observedTotalCPU and758,832KiB peakRSS. The1CPU/2GiB/10minute envelope was preserved. Exact GPU dependency imports took10.186seconds; startup plus replay of all sixteen pinned token sequences took14.858seconds, below the90second gate. These are measured CPU startup timings, not a forecast for CUDA loading.

## Repair verified

The source lives in a read-only directory; bookkeeping/logs/receipts use a sibling spool outside source. Bytecode writes are disabled, and no remote ledger upload occurred during startup. The unchanged indexed metadata/source-inspection guard validated successfully. Dependency import ASTs match the priorGPU runner, and the actualQwen2Tokenizer, torch andQwen2ForCausalLM imports completed. Source hashes were checked before and after, prior checkpoint file metadata and tokenizer/runtime hashes were checked, and all16 exact122-token prompts matched the auditedCPU receipt. No model was instantiated, no responses were generated, and no GPU was allocated.

Final receipt/log custody matches remoteSHA256s. Prior unsuccessful attempts and their charges remain intact:608CPU-seconds,160CPU-seconds,390CPU-seconds plus65MIG-slice-seconds. ThisCPU check adds19parent-record allocatedCPU-seconds; the extern step reports20seconds and is retained separately. Parent-record totalCPU allocation is1177seconds.

## Next resource gate

The orchestration repair is validated; CPU preparation remains valid. Another GPU attempt must explicitly use read-onlysource plus siblingspool and must not upload bookkeeping into source during imports. Recheck current source/custody, duplicateownership and typed71GBMIG/account/node headroom before any separately admitted attempt. Preserve the agreed oneMIG/6CPU/64GiB/5minute envelope,90secondstartup,10secondresponse,285secondexport,lowerof65GiB/95%actualmemory cap, and exactly16 frozen cases. The measured CPU check leaves roughly75.14seconds of its startup window for loading, but GPU-node timing,32B load speed/peakmemory and response throughput are still unmeasured. A bounded GPU smoke can test those facts; this step did not retry the GPU.

Scientific prompts, tokenIDs, modelrevision, BF16precision, strictscorer and16/16threshold are unchanged. Advice verification stays held. No automation changed.

## Ready local routing implementation

`run_reader32_v64.py` and `reader32_v64_gpu.slurm` implement the source/spool separation for a future admission. AST checks show model-loading, generation and chat-token calls unchanged from v62. Two actual-CLI fixtures validate that a pre-import failure exports incomplete evidence only into the sibling spool and a nested spool is rejected without creating source entries. Python/Slurm syntax checks passed. No GPU allocation was submitted. Fresh source/receipt binding and resource readiness remain required at the next admission.

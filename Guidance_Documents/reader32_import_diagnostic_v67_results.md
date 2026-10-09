# CPU import diagnostic33629588: informative timeout

The single approved CPU diagnostic ran on c3cpu-e2-u1, accountucb736_asc1/acpu, one CPU/2GiB,120second wall and90second check. It ended FAILED124:0 after95allocated seconds. The external timeout produced a terminal receipt at90.072026seconds from shell start. No model loaded, model call, GPU allocation or scientific response occurred. No complete import receipt exists. No retry was submitted.

## Evidence gained

Runtime archive hash matched the pinned8e91c377... value; staging took9.619679seconds. Source/Python/receipt/snapshot metadata checks passed before guarded imports began23.006138seconds after shell start. Tokenizer-class import did not return before the deadline.

Three nominal20/40/60second stack samples show **different progressing import locations**:

- Pillow Image.py:95 importing the native `_imaging` extension through importlib create_module.
- Torch __init__.py:327 in `_preload_cuda_lib`, calling ctypes.CDLL while Transformers imports Torch indirectly through tokenizer support.
- Python importlib get_data/get_code while Torch distributed.rpc and _jit_internal import modules.

The first live sample was initially reported as a Pillow wait. The full evidence corrects that interpretation: it is not a fixed Pillow deadlock. It shows elapsed time across native loading, CUDA-library preloading and bytecode reads. Low TotalCPU relative to elapsed wall suggests waiting, but filesystem versus dynamic-loader cause and the exact waited file/library remain unproven. Actual stack capture timestamps were not logged;20/40/60are timer targets, not per-module duration estimates. The explicit torch/model-class import markers were never reached because tokenizer imports Torch internally.

No same-version or identity waiver is introduced. Reading the current Pillow source confirms line95 is `from . import _imaging as core`; Torch source confirms the sampled call is ctypes.CDLL(lib_path). These are read-only code observations, not a repair or package execution. /usr/bin/strace exists on the login node, but compute-node availability/own-process tracing permission is unverified.

## Custody and accounting

All three raw artifact SHA digests match remote copies; the frozen source has zero hash mismatches. Source/runtime/input/failure logs remain retained. Requested versus observed:1CPU/2GiB/120second wall;95seconds elapsed, TotalCPU2.667seconds, MaxRSS741,852KiB (about724.5MiB). The diagnostic adds95allocated CPU-seconds and zero MIG time. Retained totals are **2,417allocated CPU-seconds /254MIG-slice-seconds**. Earlier33560980FAILED2:0 zero responses,99MIG-seconds and594CPU-seconds remain intact. Neither CPU nor GPU startup qualification passed; no scientific score is valid.

## Smallest changed next diagnostic — fresh approval required

Propose **one additional CPU-only diagnostic**, same1CPU/2GiB/120second wall/90second check, accountucb736_asc1/acpu/cpu-normal,zeroGPU,no requeue/retry. Keep the same guarded imports, checkpoint/scientific hashes and pinned package bytes; add Python `-X importtime` and own-process syscall tracing limited to file reads/mappings and futex waits, with a10MiB trace-output cap and stop at the same check deadline. Do not attach to another process, request tracing privileges, install tools, bypass denial or change library resolution. If tracing is unavailable/denied, report incomplete and stop. This measures the specific waiting file/library and import phase instead of replaying the unchanged diagnostic. Instrumentation overhead and cold-cache variation remain limitations; it may still time out.

This CPU proposal requires a new allocation approval; the approved33629588attempt is consumed. No GPU attempt is authorized. Byte-identical dependency staging may later be an engineering remedy, but is a separate unimplemented proposal requiring custody review; do not disable Pillow/CUDA imports, change package versions or extend the90second startup cap to force a pass.

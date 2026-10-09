# v68 one-shot CPU trace: inconclusive instrumentation stop

Approved single attempt33635575 ran on c3cpu-e2-u1 with1CPU/2GiB, account
ucb736_asc1/acpu/cpu-normal,120-second wall and90-second shell-start check.
Slurm terminal stateFAILED124:0,8seconds elapsed,TotalCPU0.675seconds,
batchMaxRSS88800KiB. No GPU allocation, model load or response. The approval
is consumed; no retry or follow-up allocation is running.

## Result and limits

The parser stopped on `unexpected child output` after0.003909824seconds of
capture. No import/syscall records were retained and no CPU import completion
receipt was produced. Exit124 is the generic stopped-diagnostic return code here;
this was not a90-second timeout. The stage receipt confirms verified archive
extraction (0.857467seconds). Framework import start/completion was not verified.

The unexpected line was intentionally not saved, consistent with the prohibition
on raw child output. Its exact text, emitter and cause are unknown. Do not label
this a confirmed ptrace denial, native import failure, filesystem bottleneck or
model failure. Permissions and emitted syntax remain unestablished. All three
proposed bottleneck hypotheses remain unresolved by this run.

The control path worked: unknown output triggered own-process stop; raw read
buffers/stderr were not persisted; both Slurm logs are empty; captured metadata
is992bytes under the64KiB total cap; trace bytes are0under10MiB. No large/unredacted
legacy completion receipt was generated. The safe successful-completion branch
is locally fixture-tested, but this run did not reach it.

## Identity and custody

Before submission, live hashes matched the separately materialized revision,
all28 baseline source files, six exact shared files, Python executable and runtime
receipt. Original v68 source/controller/archive remained preserved. The revised
source and controller were made read-only. Final source hashes are unchanged;
all seven transferred run artifacts match remote custody hashes. Frozen source
commitdcf4a17, archiveSHA2fcecd82c38a3abc21142dd27809ab77c15922f25f7ddc519225a07ceccd6ab9,
manifestSHA892d2420fd5f7cddaf7644a8d348a97d5372484ce774582b1cf676c28b596333.
Submission was guarded by an exclusive one-shot claim; only33635575 was submitted.

Final artifacts, accounting, audit and custody are in
`outputs/preference_program/results/reader32_receipt_safe_v68/run` and remote
`/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-trace-v68/spool`.
Local and remote ledgers record the terminal state. Previous raw/source/checkpoints
and other projects' jobs were not deleted or modified.

## Accounting and next boundary

Added8allocatedCPUseconds, zeroMIGseconds. Cumulative2425CPUseconds/254MIGseconds.
The low observed memory reflects an early instrumentation stop, not evidence that
successful framework import requires only87MiB or that a smaller allocation is
sufficient. No resource expansion follows from this result.

Useful next work is local review of tracing invocation and privacy-preserving
fixed error categories, including absolute tool-prefix diagnostic fixtures.
Those fixtures could improve observability but cannot reconstruct the discarded
line or establish the cause of this run. Another allocation would require a fresh
reviewed proposal and authorization; none is requested or submitted here.

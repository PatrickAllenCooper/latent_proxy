# v68 read-only binding review

Status: local packet prepared; NOT execution-ready or remotely materialized.
CPU admission and remote materialization flags remain false. No new allocation,
remote writes, framework imports, package installation, weights or model calls.

## Read-only origins and custody

Existing source: `/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-diagnostic-v67/source`.
All 28 files matched the existing local frozen copy. A second bounded read found
no source, selected package-file, or metadata mismatches. The observed project
job 33629588 remains FAILED/124, elapsed 95 seconds; no v67/v68 job is active.
The app writer lock belongs to Codex PID 4016, not a separate research worker.
Nothing was interrupted or resubmitted.

Remote evidence/recheck collected through the existing shared authenticated SSH
connection using stdlib-only read-only hash/stat code. Total content read was
61327340 bytes, under a 64 MiB session bound. One full 52,193,280-byte archive hash;
the second archive check inspected only metadata. No recursive package scan.
Evidence SHA-256: `45b69d29b26b9ffab1cc71df92729368e80ac53acee03857d3e4d114ee639640`.
Recheck SHA-256: `da724571d89dde4fb8ac8eb873e56c9ff1ad3400a731cfc0bc2cf85fa42bf1a9`.

Runtime receipt origin: `/scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/runtime_receipt.json`.
Receipt SHA-256: `bfbc150a5fab652386c1a0c38ee02744f54f2f9717063dbe691026a239dceedb`.
Archive SHA-256: `8e91c3771d157877b4deb2492e9149f3e802bc71886e4d5e2ee2f4fcdbc1f282`.
The existing extractor's exact source hash is bound in both held manifests. It
checks the archive digest before extracting. Its bounded structured stage JSON
must match that digest and the job-specific node-local path before Transformers
is admitted for trace path labels. No staged root exists or was created this turn.
The recipe is `${SLURM_TMPDIR:-/tmp}/lp-reader32-import-v68-${SLURM_JOB_ID}/transformers`.

Pinned Python symlink and target metadata were observed. Its 31,498,184-byte
binary was not rehashed because that would exceed this session's 64 MiB read
bound; its expected SHA comes from the existing CPU receipt, and the unchanged
CPU driver still verifies that digest before its guarded framework imports.
Neither fresh binary-content verification nor whole-environment verification is
claimed by this local packet.

## Narrow file attribution

Only the following six shared files have fresh content hashes; launch-time
metadata and digest checks admit exact literal paths, with no broad PIL/Torch
root whitelist. Physical aliases are admitted only if recorded in the descriptor
and same-file checked; currently these six paths resolve to themselves, so there
are no extra shared-package aliases to admit. Native Torch library metadata was
read but their contents/closure were not scanned; all such paths stay redacted.

- `PIL/Image.py`: `04af3f2db7ffdd61e829fa7aa8f5481d394a0a4129a8fd8aa5b13908928fcd6d`
- `torch/__init__.py`: `cf40c075c95864036e835795756d69b8cccfafa76f3bcde5eba9d06065ccd3d1`
- `torch/distributed/rpc/__init__.py`: `f11563e66f57c6486aaedd28601dc85619689b1cb14ccba479422e064aef6089`
- `torch/_jit_internal.py`: `9435668ade300edf01802837a34ab91a6a1214388fe9fd13931c0ea01c5f4d75`
- `torch/nn/functional.py`: `990a8b717645fa603b0dd2957b7b3ed68c5fb937fa205048c096ac8bb532510e`
- `PIL/_imaging.cpython-312-x86_64-linux-gnu.so`: `94310f4a073e9773a067df88513804649f50a974f78bd85a52c52d21170783f2`

This reduced attribution remains useful: known Pillow native opens/reads and
Torch Python/RPC reads can be aligned with import labels; slow open/stat/read
versus mmap/futex waits can distinguish observations supporting the hypotheses.
A redacted file cannot be assigned to a particular native dependency or storage
location. Loader closure, cache state, filesystem causality and a precise root
cause may remain unresolved. CPU evidence does not establish GPU startup.

## Held remote-path package

`remote_manifest_held.json` records proposed absolute source/controller paths
under `/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-trace-v68` and
exact hashes of local frozen controller copies and the unchanged baseline.
These are planned destination bindings, not a claim that remote copies exist.
`binding_receipt.json` binds this manifest and local controllers. The wrapper
performs source/runtime-receipt preflight before extraction. Source mutations,
wrong stage identity, wrong job-root recipe, changed exact file metadata/content
or aliases fail closed. No previous approval admits this wrapper.

## Measurements and controls

23 local v68 fixtures and 3 unchanged v67 guard tests pass; held wrapper shell
syntax passes. Added binding tests cover stage path/hash/job/size failures, exact
file mutation, alias mismatch, remote path mapping, frozen controller identities
and redaction outside verified files. Numeric PID + FD records disambiguate read
attribution. Safe unfinished-syscall records preserve a wait onset on timeout,
without inventing a return value/duration. Real strace denial on the stderr
channel is recorded as unavailable, with no raw message saved.

Combined sanitized traces: 10 MiB. Combined capture metadata: 64 KiB, split
15 KiB stage receipt + 48 KiB capture receipt + 1 KiB fixed-field wrapper terminal
receipt. Wrapper terminal and capture receipts fsync their files and directories;
partial trace files fsync as captured. Node loss/SIGKILL/storage failure can still
prevent final receipt creation. Importtime/strace/fsync overhead is uncalibrated;
this is an instrumented diagnostic, not an uninstrumented performance benchmark.

## Remaining review/execution gates

Independent review of the completed local packet; fresh approval for exactly
one CPU, 2 GiB, 120-second wall/90-second shell-start check, account ucb736_asc1,
acpu/cpu-normal, one attempt, zero GPUs; authorized future remote materialization
and verification of the frozen destination files before flipping either hold.
Compute-node strace availability, own-child permission, and exact emitted syntax
remain untested. Denial or unsupported syntax ends the sole attempt inconclusively;
no install, attach, elevation, fallback, retry or expanded resource request.

Previous totals remain 2,417 allocated CPU-seconds and 254 MIG-slice-seconds.
Previous raw artifacts and other projects' jobs are preserved.

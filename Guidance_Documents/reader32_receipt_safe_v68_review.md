# Output-only v68 receipt correction: local review packet

The approved sole CPU attempt remains unused. This revision is LOCAL only and
held for independent review; no remote writes, framework imports, allocation,
model loads or responses occurred. Previously materialized v68 files/archive,
raw historical receipts and previous charges remain intact (2417 allocated CPU
seconds,254 MIG slice seconds). Neither admission flag is enabled.

## Output-only derivation and preserved guards

The new `reader32_import_diagnostic_v68_receipt_safe.py` is explicitly derived
from the frozen v67 CPU driver. All28 original source files are byte-identical
copies in a separate revision directory. The new driver keeps its dependency
function, top-level AST, complete pre-import/post-import guard prefix and exact
baseline PYTHONPATH/source-working-directory recipe. Only the final output tail
is replaced. No Path/open monkeypatch, guard removal, package changes or runtime
fallback is used. No original source/receipt is mutated.

The new helper is imported only after the guarded framework imports and all
existing final checks complete. The driver constructs a small whitelist record
of fixed booleans/zero counters, bounded timestamps and SHA256 hashes. It never
constructs the former rich completion receipt containing captures, inspection,
directory signatures, host names or version strings. Internal capture/inspection
objects still exist as required by unchanged guard logic; they are not serialized
or passed to the writer. The helper validates exact schema/types/hash formats,
then caps encoded output before exclusive file creation and fsyncs file/directory.
Qualification is explicitly false; this remains a CPU import diagnostic.

The receipt records both baseline and adapter SHA256 identities. The separate
controller verifies all frozen baseline, adapter, helper and controller hashes
before capture and again after capture, and checks the unchanged guard/import AST.
A future materialization must make the complete revision source read-only and
retain mutable artifacts in the original sibling spool. Original frozen files
remain separate. Proposed paths are under the existing v68 root's
`revision-receipt-safe/source` and `revision-receipt-safe/controller`; the held
wrapper points only to the new explicit adapter. These revision paths do not
exist remotely yet.

## Enforced metadata envelope

Combined sanitized trace output stays10MiB. Metadata stays64KiB total:
15KiB runtime stage +45KiB capture receipt +3KiB safe CPU completion +1KiB fixed
wrapper terminal. The writer cannot accept captures/distribution mappings,
arbitrary keys/text, raw paths, credentials, nonfinite timestamps or nonzero
model/GPU counters. Rejection occurs before any file is opened; failure does not
produce an unsafe substitute receipt. Exclusive output preserves existing files.

## Local checks

12 receipt/derivation/packet fixtures pass,24 existing v68 capture/binding fixtures
pass, and3 unchanged v67 guard fixtures pass. Held wrapper shell syntax passes.
Checks cover strict schema/privacy/type/cap rejection before persistence, success
custody, exclusive creation, exact AST derivation, rejected guard mutation, frozen
controller hashes/held admissions, aggregate budget and preserved child lookup
path/cwd. Toy main-function execution uses fake classes and runtime/source metadata;
no framework modules are imported. It verifies successful safe completion and
both a pre-import runtime-identity failure and a post-import source mutation:
failures still raise before a completion receipt is created. Secret-bearing toy
capture/inspection mappings never appear in the successful receipt.

## Remaining gate

Independent review of this completed output-only correction and frozen packet.
Then materialize the new versioned revision without replacing the preserved
original source, verify live identities/admissions, and use the existing approved
one-shot1CPU/2GiB/120s wall/90s shell-start check under ucb736_asc1/acpu/cpu-normal.
No new spending approval, allocation, retry, GPU or scientific change is implied.
Compute-node tracing permissions/syntax remain unknown; denial or unsupported
syntax ends the sole attempt inconclusively. Instrumentation overhead and partial
attribution of unverified native libraries remain limitations. Receipt correction
resolves a control gap; it provides no new evidence for filesystem/native-loader
causality or model qualification.

Frozen packet: outputs/preference_program/manifests/reader32_import_trace_v68_receipt_safe.
Results: outputs/preference_program/results/reader32_receipt_safe_v68.

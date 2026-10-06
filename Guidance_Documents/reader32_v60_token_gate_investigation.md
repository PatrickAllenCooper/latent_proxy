# v60 CPU token gate: cause-specific investigation

The application failed at a conjunction of raw chat-token equality and prompt length <=512. It did not retain the raw output type, operands, or length. Consequently the retained evidence cannot distinguish container representation, a true ID mismatch, or the length guard. No root cause is established.

A local diagnostic used the preserved dialogue adapter tokenizer JSON and the same chat template hash. All sixteen frozen prompts round-trip exactly at 122 tokens. This tokenizer differs from the pinned 32B slow tokenizer; the local result cannot rule out a pinned-runtime token mismatch or length failure. No checkpoint downloads, model calls or remote token preparation occurred.

`diagnose_reader32_token_gate.py` now captures raw types/structures, exact IDs, lengths, normalized equality, rendered text/hash and first differing token before the gate is decided. Four fixture tests verify that the existing `normalize_prompt_ids` helper changes containers only, retains IDs exactly, and rejects true token changes, oversized sequences, empty/invalid IDs and multi-prompt batches. No live runner or scientific registration was changed.

## Smallest next dependency

An additional allocated **CPU-only diagnostic attempt is necessary** to establish the actual pinned tokenizer's failed operands; they cannot be recovered from existing logs. Proposed diagnostic: one CPU, 2 GiB, at most 90 seconds, plus already permitted archive extraction within that same bound. Load the cached pinned32B tokenizer, capture the first frozen case's chat output and independent rendered-string encoding before asserting anything; if valid, continue the remaining frozen prompts, without model responses. Persist the diagnostic immediately, including on failure. No weight hashing/model loading or GPU allocation belongs in this cause-finding step.

This diagnostic is not submitted or authorized by the current investigation instruction. Its startup fit is uncertain; timeout would be incomplete and retained, with no automatic retry. A container-only failure would justify adopting the already used dependency-free normalization, only after exact normalized IDs agree across the sixteen frozen prompts. A genuine token difference or overflow requires separate explanation; normalization must not conceal it. Correcting this gate does not manufacture missing prior full-weight custody: a subsequent authorized integrity stage still needs persisted shard digests and the complete token/runtime receipt before any GPU admission.

Any new admission also requires reconciliation of job33505553's terminal controller/accounting state. The last earlier observations disagreed (application/accounting failed, controller running). Preserve the initial attempt's provisional charges and logs even after reconciliation; never overwrite them with a replacement attempt's cost.

Scientific panel, 16/16 scorer, memory/time limits, precision, checkpoint revision and advice hold remain unchanged.

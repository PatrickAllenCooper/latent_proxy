# Pinned 32B tokenizer repair: validated CPU preparation

Job33507413 completed0:0 on2026-10-06, with160seconds elapsed, one actualCPU,47.121seconds observedTotalCPU, and1,111,048KiB peakRSS. It stayed within1CPU/2GiB/10minutes. No model responses or GPU allocations occurred. Prior job33505553 is terminalNODE_FAIL with608allocatedCPU-seconds; retain its earlier application token assertion and contradictory consumption snapshots. Combined allocatedCPU time is768seconds across both attempts.

## Confirmed cause and instrument preservation

For every frozen prompt, the pinned Qwen2.5-32B-Instruct tokenizer's chat API returned a `BatchEncoding` with `input_ids` and `attention_mask`; rendered-string encoding returned a list. Raw container equality was false, but extracting and normalizing `input_ids` produced exactly the same122 tokenIDs in both paths. All sixteen pairs and independent Jinja rendering were audited. No tokenID, prompt, checkpoint, precision, semantic case, strictscorer or qualification threshold changed.

The conditional wrapper demonstrated the first-case container mismatch and sixteen normalized gate passes before integrity continuation. Three synthetic wrapper tests verify that a realID mismatch or an unreproduced failure cannot initiate an integrity retry. Four token-gate tests reject malformed/oversized inputs and actual ID changes.

## CPU custody and readiness

All17 cachedBF16 shards were hashed, covering65,527,841,856bytes. Each digest was persisted immediately to an integrity journal. The journal exactly matches the completeCPU receipt, along with tokenizer/config/index hashes and the frozen source/runtime identities. Six final artifacts have matching remote/localSHA256s. The integrity stage took51.306seconds, measured on this recovered CPU node; this is not a guarantee for future storage conditions.

The complete receipt and resources are in `outputs/preference_program/results/reader32_preparation_v61`. The corrected source is pinned in the v61 execution freeze. No changed source was written into the original v60 attempt.

## Next dependency

CPU preparation is valid. The next scientific stage remains the previously agreed16-responseBF16 reader qualification, requiring a fresh allocation/source/custody recheck and a GPU runner using this corrected token contract/receipt. The source roots and freezes must be reconciled explicitly before launch; do not feed this v61 receipt to an unchanged v60 runner/root. No GPU stage was admitted in this CPU engineering scope. Advice verification remains held and the16/16 qualification requirement is unchanged.

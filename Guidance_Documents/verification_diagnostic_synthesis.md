# Verification diagnostic synthesis

## Evidence and scope

All three completed runs used the same pinned Qwen2.5-1.5B-Instruct snapshot
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, deterministic generation, and a
24-token ceiling. They reuse one underlying user/menu with four cyclic label
assignments. Rotations are not independent users. Counts below were checked
against raw records, audits, and completion receipts; no new inference ran for
this synthesis.

**Twenty-response verification smoke (33391694):** five arms × four rotations.
No advice yielded 1/4 optimal, all A; correct neutral advice 4/4; correct advice
with verification 3/4 (one spoiled choice); wrong neutral advice 0/4; wrong
advice with verification 0/4. Wrong-neutral responses copied the recommendation
on all four rotations; verification changed one of those responses but did not
correct it. Mean normalized regrets: no advice 0.6789, correct-neutral 0,
correct-verify 0.25, wrong-neutral 0.8122, wrong-verify 0.8350. Zero parse failures.
[Audit](../outputs/preference_program/results/verification_retry_v39/audit.json),
[raw prompts/outputs](../outputs/preference_program/results/verification_retry_v39/base.jsonl),
[receipt](../outputs/preference_program/results/verification_retry_v39/base_receipt.json).

**Sixteen-response numeric/order diagnostic (33403696):** decimals versus
rank-preserving integers × canonical versus reversed listing × four rotations.
Every cell scored 1/4, all A, zero parse failures. Both paired representation
and order contrasts were zero in accuracy, regret, and changed responses.
Reversed listing starts with D: A persists, so this observed pattern follows
the A label, not the first listed option.
[Audit](../outputs/preference_program/results/numeric_capability_v40/audit.json),
[exact prompts/answers](../outputs/preference_program/manifests/numeric_capability_v40_review.txt),
[receipt](../outputs/preference_program/results/numeric_capability_v40/base_receipt.json).

**Eight-response mapping diagnostic (33406672):** four integer assignments ×
label versus numeric answer request. Label outputs B,C,D,D scored 3/4 against
B,C,D,A. Numeric outputs 5,9,4,5 scored 1/4 against constant gold 4; three outputs
were absent from the input values. Number-minus-label accuracy was −0.5 on
these four matched assignments. Zero parse failures.
[Audit](../outputs/preference_program/results/output_mapping_v41/audit.json),
[raw prompts/outputs and token IDs](../outputs/preference_program/results/output_mapping_v41/base.jsonl),
[receipt](../outputs/preference_program/results/output_mapping_v41/base_receipt.json).

## Exact changes and what comparisons identify

The smoke no-advice prompt begins `User preference scores: A=+0.264228, ...`
and says `Choose the action with the highest score.` Its advice arms add
`A fallible adviser recommends [label]. This recommendation may be incorrect.`
Verification additionally says `Check the recommendation against the scores
before choosing.` Evidence, order, and label rotation are otherwise matched.
These within-run comparisons identify response changes under those advice and
instruction packages on this case. They do not distinguish numeric reasoning
from recommendation following or show recovered preferences.

The numeric diagnostic's decimals/canonical prompts are byte-identical to the
smoke no-advice prompts and reproduce all-A responses in a separate run.
Its matched manipulations replace signed six-decimal values with ranks 1–4,
or reverse the listing from A,B,C,D to D,C,B,A. Simplification changes sign,
scale, precision, and textual number representation together: its null cannot identify each feature's
individual effect. Reversal holds value/label associations fixed and supplies a
position control. All-A behavior persists in both representations and orders.

The mapping diagnostic changes the header to `Numbers:`, retains the unsigned
integer evidence, and asks `Which label has the largest number?` or
`What is the largest number?` Label output remains one capital letter; numeric
output becomes one digit. The label-versus-number contrast is matched on
numeric evidence, but changes question semantics and output format together.
The improvement over the earlier integer/canonical condition is an unmatched
header-plus-instruction package comparison; no cause can be assigned to either
change. Numeric gold is always 4, so even 4/4 would not establish broad comparison
competence. Actual numeric performance is poor here.

## Interpretation and custody

The direct label task demonstrates some value-sensitive selection under one
prompt, while the preference-framed task demonstrates an A-label pattern.
Neither reliable numeric comparison nor reliable faulty-advice correction has
been established. Preference discovery and reward arithmetic are absent: these
prompts expose already-computed scores. Causes still unidentified include
header/instruction framing, output mapping, interactions with model training or
quantization, and sensitivity to other input sets. No cross-model, population,
or general reasoning conclusion follows.

All runs have manifest/model/runtime hash checks, strict output audits, GPU
framework allocation and generation progress. MIG process utilization was
unavailable. The first two saved no generated token IDs or actual stop reason;
their 24-token ceiling is verified, but exact token count/EOS termination cannot
be recovered. The mapping run saved two generated tokens per response, each
ending in effective EOS, with zero length caps. Startup failures remain in
`verification_smoke_failed_v36`, `improved_proxy_handoff_failed_v35`, and
`startup_diagnostic_v37`; no outputs or null comparisons were discarded.

## Proposed next gate — preparation only

A **16-response header × instruction factorial**, no advice: the same four
integer assignments and letter-only output, crossing `User preference scores:`
versus `Numbers:` with `Choose the action with the highest score.` versus
`Which label has the largest number?` This holds numerical evidence, listing,
label mapping, model, decoding, and output contract fixed and separates the two
wording changes that were previously bundled. Record exact prompts, token IDs,
EOS/length evidence, and paired correctness by rotation before interpreting.

**Decision criterion:** the verification assay remains gated unless a frozen
condition achieves 4/4 correct across all gold labels, zero parse failures, and
no truncation on this measurement check. Main-effect/interaction claims remain
descriptive on one case; a passing condition would require a fresh-case gate
before broader verification inference. If no condition passes, seek guidance on
the assay rather than tuning or expanding it. Resource ceiling, if authorized:
one 35GB H200 MIG, five minutes, 90-second loading and 45-second staging limits.
No job, retry, tuning, or 320-record study was launched by this synthesis.

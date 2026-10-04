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

### Frozen preparation receipt (v42)

The proposed factorial is now frozen and CPU-validated, with zero inference
calls or submissions. Its two historical corners exactly reproduce the prior
integer/preference/score prompt and direct numbers/label prompt. All 16 cells
retain canonical order, the same values/gold within each rotation, and a fixed
letter-only output contract. Header, instruction, and interaction contrasts
are specified prospectively; no population confidence interval is appropriate.

[Exact 16 prompts and gold answers](../outputs/preference_program/manifests/framing_factorial_v42_review.txt),
[protocol and compute ceiling](../outputs/preference_program/manifests/framing_factorial_v42_protocol.json),
[paired scoring specification](../outputs/preference_program/manifests/framing_factorial_v42_scoring.json),
[CPU validation and hashes](../outputs/preference_program/manifests/framing_factorial_v42_validation.json).

The proposed cap remains one 35GB H200 MIG/five minutes, 90-second model loading
and 45-second runtime staging watchdogs. Token IDs, prompt/output lengths,
effective EOS IDs, last token, EOS/length-cap evidence, rendered messages, and
model/runtime hashes are required. A 4/4 cell is only a reused-case measurement
gate; fresh-case qualification is still required before broader inference.
No unresolved input is needed for preparation. Execution remains unlaunched.

### Factorial execution result (v42)

Job33407495 completed41s. Preference/score:1/4(A,A,A,A);
preference/label:2/4(A,C,A,A); numbers/score:2/4(A,C,A,A);
numbers/label:3/4(B,C,D,D), againstgold B,C,D,A.
[Raw responses and token custody](../outputs/preference_program/results/framing_factorial_v42/base.jsonl),
[full audit/contrasts](../outputs/preference_program/results/framing_factorial_v42/audit.json).
Both paired header and instruction accuracy effects are+0.25; interaction
mean is0, with rotation-level differences[+1,-1,+1,-1]. Zero average interaction
does not mean zero effects on individual assignments. These matched changes
identify response effects of these literal wording manipulations on this case,
not why the underlying model responds differently. Historical corners reproduce.

All16strictly parse,generated length2 ends in effectiveEOS,no24tokencaps.
Prompt/scoring/prospective hashes and model/runtime identities match; framework
memory1.19GB plus16generationevents establishGPUwork. MIGprocessutilization
unavailable. No cellpasses4/4: the registeredmeasurementgatefails, so the
broaderverificationassay/320study remainsblocked. No follow-oninference.

Smallest prospective nextgate requires a direction decision rather than more
wording tuning: an8response same-checkpoint NF4-versus-BF16 loadingprecision
control on the four frozen numbers/label integer prompts. Current loader uses
NF4; this is a hypothesis, not an attributed cause. BF16demand is unmeasured
and would require qualification within the existingoneMIG/five-minute cap.
[Decision proposal](../outputs/preference_program/manifests/precision_control_v43_proposal.json).
A passingprecisioncondition would still need fresh-case qualification. If this
control is not desired, suspend this assay and seek guidance on a competent
comparison baseline. No job or scientific intervention launched from closeout.

### Precision-control preparation (v43; no execution)

Eight exact NF4/BF16 paired records are frozen on the four unchanged
numbers/label integer prompts. Evidence, letter output, native chat messages,
model revision/tokenizer, greedy24token decoding and scoring are identical;
the quantization configuration is the only arm manipulation. Both arms declare
BF16 compute/nonquantized dtype and singleCUDAdevice placement. Sequential
NF4thenBF16 avoids simultaneous models; precision remains confounded with
order/allocator/time nuisances, which this small control cannot isolate.
[Protocol](../outputs/preference_program/manifests/precision_control_v43_protocol.json),
[exact pairs](../outputs/preference_program/manifests/precision_control_v43.jsonl),
[scoring/decision criterion](../outputs/preference_program/manifests/precision_control_v43_scoring.json),
[CPU validation](../outputs/preference_program/manifests/precision_control_v43_validation.json).

CPU-only inspection of the pinned safetensor header found338tensors,
1,543,714,304BF16elements:3,087,428,608bytes(2.875GiB) of tensor payload.
The previously hash-verified file contains3,087,467,144bytes. The declared
12GiBGPUplanning budget allows two BF16weightcopies plus6.25GiB for runtime
headroom; both models must not coexist. Weight storage fits35GBMIG, but this is
not a measured peak or formal memory guarantee.
[Weight/resource assessment](../outputs/preference_program/manifests/precision_control_v43_weight_assessment.json).

Five-minute completion remains unqualified: BF16loading/generation is unmeasured.
The proposed45sstage,90sload perarm,10sgeneration perarm and40sorchestration
allowance total285s; watchdogs bound a laterattempt rather than guarantee
completion. Token/promptIDs,EOS/lengthcaps,parameterdtypes/runtimehashes and
memory/timing custody are required. NoGPU/modelcalls/submission were performed.
A CPU preparation NameError after artifact writes was corrected and all hashes
regenerated; its receipt is retained.

Prospective criterion: BF16=4/4 and NF4<4/4 without parse/truncation supports a
local precision-sensitive assay failure; freshcases still required. Bothfail
means assayfailure and guidance, bothpass means no detectedprecisionloss here,
NF4passes/BF16fails means do not adoptBF16, and incomplete/resourcefailure gives
no scientificcontrast or retry. Smallest outstanding decision is whether to
authorize this bounded qualification within the current cap despite unmeasured
BF16runtime, or suspend this assay. No resource enlargement is justified by
weight storage; no automatic320study or prompt tuning follows preparation.

## Precision qualification v43 — job 33408635

The frozen eight-response paired control completed successfully in 27 seconds on one H200 2g.35gb MIG allocation. NF4 produced B,C,D,D (3/4 correct); BF16 produced B,C,D,A (4/4). The paired accuracy difference was +0.25, entirely from one response. Identical prompt token IDs, tokenizer template, effective SDPA attention, EOS configuration, and greedy 24-token decoding were verified across arms. All eight outputs ended with EOS; no parse failures or length caps occurred. BF16 passed the registered local measurement gate.

Peak framework allocation was 1,221,643,264 bytes for NF4 and 3,127,658,496 bytes for BF16; cleanup returned allocation to 33,555,456 bytes between arms. Framework memory and completed CUDA generation establish GPU work. Physical-device nvidia-smi utilization was N/A and its aggregate memory rows are not attributable to this MIG job. The run remained within the five-minute and 12GiB stops.

This supports a local precision-package difference on the reused case, not general preference capability. NF4 ran first, so arm order, allocator state and quantized loading/kernel code paths remain confounds. Next scientific gate: freeze fresh numeric cases to qualify BF16 before returning to preference verification. No full 320-response study, retry or scaling was submitted. Artifacts: outputs/preference_program/results/precision_control_v43.

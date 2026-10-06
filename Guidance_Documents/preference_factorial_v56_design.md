# Fresh preference execution assay v56 — frozen local design

## Status and finite scope

Design and synthetic fixtures only; no inference, API calls, GPUs or pending
owner question. Execution is not authorized by this document. Original v55
remains failed at 11/16; its cases, scorer and 16/16 threshold are unchanged.
The advice panel remains blocked. This is one finite diagnostic, not a search
across prompts until a favorable score appears.

The proposed 16 responses cross two fresh semantic cases with three binary
factors: representation, output-letter assignment and catalog row order.
Full input and paired contrast files are frozen under
`outputs/preference_program/manifests/preference_factorial_v56`.

## Cases and allocation

Seed 2026100601 selects two cases from a finite categorical pool after
excluding all 24 distinct semantic signatures in the previous development,
qualification and advice splits. Freshness ignores output labels and row
order: a previous case relabeled or reordered cannot count as new. The
signature includes the full priority order and eligible attribute set.
Selection does not examine model outputs. Cases are independent of the
observed v55 errors, but share its task vocabulary and deterministic generator
family; they are not independent population samples.

Fresh-00 has top-priority eligible and canonical gold A. Fresh-01 has
top-priority ineligible and canonical gold B. Canonical gold constraints
support balancing, not selection for perceived difficulty. The fixed seed
and exact generated content are retained even if results are unfavorable.

Each case receives all eight combinations:

- Representation: original prose versus the same prose/catalog with each
  option's explicit one-based priority rank appended. Both supply identical
  preferences and eligibility; rank annotations are deterministic computation
  supplied to the model, not extra user answers or learned preferences.
- Letter assignment: canonical versus rotation by two (A↔C, B↔D). Attributes,
  eligibility, priority and underlying catalog rows stay fixed.
- Catalog order: forward versus reversed. Labels stay attached to their
  semantic options; only presentation row order changes.

All factors are crossed within each semantic case. Gold letters A/B/C/D each
appear four times overall; best-option row positions 1/2/3/4 each appear four
times. The sixteen configurations have one fixed seeded shuffled run order.
No resampling, favorable case selection, alternate decoding or pilot tuning.

## What the contrasts identify

Eight format pairs hold semantic case, labels and row order fixed. Their
accuracy differences test whether externally computed rank annotation
reduces execution errors. A gain establishes sensitivity to representation
and available computation; it is not evidence of TextGrad/GEPA optimization,
RL proxy discovery, or learned reward parameters.

Eight relabeling pairs hold content, representation and row order fixed.
Compare chosen **semantic attribute**, not raw response letters. A change
shows output-identity sensitivity in these cases. Because rotation has only
two levels, it does not cover all 24 letter permutations or establish which
individual token causes the effect.

Eight order pairs hold content, format and letter assignment fixed. Changed
semantic choice shows catalog-position sensitivity. Exact factorial crossing
separates this from letter relabeling; serial execution order remains a
single randomized realization, not an independently replicated factor.

Compare factor effects within each semantic case before aggregation. Content
is a blocking variable, not an estimated causal factor: two cases cannot
separate all attribute semantics from whether top priority is ineligible.
Case-specific differences therefore suggest interaction hypotheses only.
No claim that two blocks establish general robustness or isolate linguistic
content effects. More independent content blocks require a later study.

## Frozen scoring and analysis

Reuse the original strict scorer, hash
`cbc048bb2c00ee768b10357101db499dd32b70dccae7b6486d95254dd4e1510e`.
Only whitespace-surrounded single capital A/B/C/D parses. Compute gold
independently as the minimum-priority-rank eligible semantic option and map
it through the current labels. Report all sixteen outcomes, exact accuracy,
eligibility violations, eligible priority errors, parse failures, EOS/caps,
and each case/factor cell. No replacing the scorer or regrading selected rows.

For each factor report its eight paired differences in correctness and their
mean (second level minus first), plus the number of pairs with identical
semantic choice. Parse failures count as unsuccessful invariance, including
pairs where both fail. Consistently wrong choices may be invariant; invariance
alone never passes correctness. Show all pair values, not just a favorable
aggregate. No p-values or population confidence intervals on two constructed
semantic blocks. No claim of independent n=16 user observations.

The checked analysis rejects incomplete or duplicate result sets, wrong
prompts, changed case/scorer hashes and actual EOS mismatch. Token equality,
model/runtime/config provenance and terminal resource validity remain
separate required execution checks; the analysis script alone does not
validate them. Synthetic oracle fixtures are clearly marked and never saved
as model responses.

## Stopping and scientific gate

A future execution must use a fresh immutable output root and CPU preparation
first. Stop on any startup, hash, token, resource or terminal failure; preserve
partial outputs; no automatic retry, generation replacement or continuation
past sixteen calls. Record all validly produced outputs even if the first
error already makes a perfect score impossible.

Outcomes have finite consequences:

- All sixteen correct with complete execution validity: this **new mechanism
  smoke** passes; advance to an independently registered confirmation with
  more fresh semantic blocks. It does not repair v55 or automatically open
  the original advice panel.
- Explicit-rank format 8/8 while prose falls short: register a fresh structured
  preference/proxy representation confirmation. Prose competence remains
  unqualified; no preference-learning claim.
- Semantic choice changes under relabeling or row reversal: retain an identity
  or position sensitivity diagnosis scoped to these cases, and investigate
  it before interpreting advice compliance.
- Both formats remain inaccurate: hold advice; recommend one preselected
  more capable checkpoint for an independent capacity comparison or a
  deterministic proxy/tool computation control. Do not retune the failed
  assay, lower its threshold or enlarge the GPU allocation automatically.

These branches can coexist (for example, rank gain with relabeling
sensitivity). Report every branch triggered; none selects a passing subset.

## Proposed resource envelope and execution requirements

No greater than the previous frozen qualification: sixteen total greedy
responses, Qwen2.5-1.5B-Instruct revision
989aa7980e4cf806f80c7fef2b1adb7bc71aa306, BF16/SDPA/use_cache=False, max 24
new tokens, max 512 prompt tokens, peak framework allocation/reservation
≤12GiB. CPU preparation stays ≤4 actual CPUs/8GB/90-second diagnostic cap.
Use streaming file hash checks with a small RAM request where possible,
avoiding the previous one-CPU/8GB request that became three actual CPUs.

If separately authorized, GPU request is one H200 2g.35gb MIG, four requested
CPUs, 32GB, account ucb736_asc1, ah200/gpu-normal, five minutes; maximum 285
seconds complete runtime, verified GPU work by 90 seconds, runtime staging
≤45 seconds, generation stage ≤10 seconds. Prior smoke: 34 seconds total,
peak reserved 2.93GiB, sixteen two-token responses. These measurements justify
a single small MIG slice, not multiple GPUs or a long allocation. New ranked
prompts require new CPU token IDs; their lengths are not yet measured.

Before submission reconcile project jobs and allocation headroom. Preserve
all other work. Capture memory plus CUDA work timing/process or framework
signals and generation progress; MIG utilization may be unavailable. Log CPU
preparation, model loading, GPU work and output times. Require current model,
runtime, source, template, cases and token hashes; observed exact allocation;
16 records; strict format/EOS; normal terminal success; local/remote artifact
hash custody. No inference script or job was submitted by this design.

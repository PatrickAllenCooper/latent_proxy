# Independent semantic audit and next decision

## Scope and result

Existing saved outputs only, no rerun, inference, GPU allocation, downloads,
optimizer search or modified scorer/case/threshold. Independent audit script
`independent_preference_semantic_audit.py` verified all32 v55/v56 records.
Audit JSON is saved under `outputs/preference_program/results/semantic_reader_audit_v58`.

The independent parser reads each actual prompt's priority list, option label,
attribute, eligibility and any rank annotation. It verifies four unique
attributes/labels, three eligible options, strict total priority order and
correct rank annotations. Gold is recomputed by minimum priority index among
eligible options, independent of the original scorer. v56 rotation and row
reversal are reconstructed from the frozen semantic base cases, not inferred
from answers. No tie, missing eligibility or internally conflicting rank
instruction was found. The rule is logically well-defined; this does not
claim the linguistic task is effortless for every model or human.

All32 exact chat prompts were independently rendered from the preserved
Jinja template, whose SHA256 matches the production template. Every prompt
ID sequence was re-encoded and decoded exactly; all32 generated completions
also decode exactly. This uses the preserved adapter-checkpoint tokenizer
JSON and local tokenizers0.22.2/Jinja2 3.1.6. Its tokenizer JSON hash differs
from the pinned production file, so this is independent *consistency on these
sequences*, not proof that tokenizer artifacts are globally interchangeable.
Production model/tokenizer/runtime identity remains supported by the original
hash/token receipts. No new model or tokenizer was downloaded or loaded via
Transformers. Strict single-letter parsing, actual last-token membership in
EOS IDs, lengths, saved correctness and full message/prompt hashes all agree.

**Scores reproduced:** v55 11/16, one eligibility violation/four eligible
priority errors; v56 6/16, six eligibility violations/four eligible priority
errors. No audit evidence supports rescoring these errors or blaming a mapping,
prompt rendering or token-serialization defect. Runtime/scoring valid and
reader execution insufficient can both be true.

## What the two studies establish

v55 is a deterministic16-semantic-case explicit-rule qualification, with
balanced gold letters but correlated serial order, semantics and label
positions. It fails its fixed16/16 threshold. Its slices are descriptive,
not independent estimates of label bias or user alignment.

v56 contains only two fresh semantic cases and eight transformed presentations
each. Its16 outputs are **not16 independent semantic trials**. Within paired
cells letter assignment and row order are independently manipulated; semantic
choice agreement is5/8 for each factor, with two label-related correctness
losses and two reversal-related gains. Effects are local to these stimuli,
not proof of a universal preference for early letters/rows. Format pairs have
zero correctness difference in all8 pairs, but only4/8 semantic-choice
agreement; rank annotations alter some failures without accuracy benefit.
Eligibility violations are2/8 prose versus4/8 ranks. The ineligible-top
semantic block scores0/8; content and top-priority eligibility are confounded
between two blocks, so neither explains the entire population of failures.

Do **not** interpret11/16→6/16 as a causal degradation over time or a paired
replication: the panels/estimands differ. Do not pool them as17/32 independent
users or rescue v55 from v56's favorable cells. Both fail their stated gates.
Neither asks the model to discover hidden preferences, learn reward parameters,
invoke an RL tool or personalize general assistance. They show this checkpoint
and these deployments lack sufficiently reliable explicit-rule execution to
support the proposed advice-verification measurement. The advice hold stays.

## Deterministic computational control — completed locally

For all2304 combinations of option-attribute assignment(24), priority order(24)
and excluded option(4), maximize reward `-priority_index` for eligible options
and impose `-infinity` for ineligible options. A separate priority-walk oracle
agrees2304/2304; each optimum is unique and eligible. No model calls. This
establishes that an explicit constrained preference proxy can compute the
correct recommendation for this finite task. It does not establish user-
preference identification, human validity, RL learning, LLM tool calling or
faithful use of a tool's output.

## Recommended decision

**Adopt the deterministic constrained proxy as the system's execution control,
and close the current1.5B direct-reader qualification as a negative instrument
result.** Keep preference discovery and reward execution separate in evaluation:
measure the inferred preference parameters/posterior first; then measure the
proxy-selected action; finally measure whether the assistant faithfully
communicates that action and preserves constraints. Final recommendation
validation must check the action against the proxy; merely presenting tool
text to an LLM supplies no enforcement guarantee.

This is the soundest next step because correctness of constrained execution
is already established locally and the present reader cannot satisfy the
measurement baseline. No further trial on the same assay is justified by a
hope that a new phrasing will pass. The negative reader result need not close
the broader preference discovery/proxy program.

If the scientific objective requires testing whether a *more capable LLM*
can execute the same rule internally, the next inference proposal should be
one preselected **Qwen2.5-3B-Instruct** checkpoint, no adapters, against16 fresh
independent semantic cases selected before generation, disjoint from all
previous splits and v56 signatures. Use one prose condition,8 eligible-top/8
ineligible-top and balanced gold labels/randomized run order; unchanged strict
scorer/16-of-16 gate. Purpose: establish a competent reader baseline for later
advice/tool/optimizer comparisons, not claim a direct causal scaling advantage
from comparison with differently sampled historical cases. A later paired
size comparison would need both checkpoints on the same fresh cases and its
own allocation; it is not authorized here.

Smallest proposed envelope for that *one conditional capacity smoke*: CPU
preparation≤4 actual CPUs/8GB/90-second diagnostic (new tokenizer/model file
hashes and tokens first), one H2002g35gbMIG,4 requested CPUs/32GB,5minutes,
90-second startup,285-second complete limit,≤12GiB peak framework memory,
16 greedy responses/max24 new tokens/max512 prompt tokens. Approximate3B BF16
weights need6GB before activations, not measured demand; verify pinned file
sizes on CPU and treat the first of the16 generations as the bounded GPU
memory/work smoke. Stop if measured load/memory/10-second generation budget
cannot support completion; preserve partials, no retry/resource expansion.
Do not reuse the1.5B token receipt or claim its startup timing predicts3B.
No new job or pending approval question was created by this recommendation.

Evidence priority is now: deterministic proxy control (passed); current
reader instrument (closed negative); one independent capacity baseline only
if internal LLM execution remains a required scientific question. Advice,
RL-user discovery, prompt optimization and fine-tuning require their own
valid comparison baselines and do not follow from these32 scored outputs.

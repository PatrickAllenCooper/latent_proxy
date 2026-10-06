# Frozen eligibility/priority qualification: interpretive report

## Registration and decision

This is a **post-outcome interpretation**, registered after GPU job 33469187
completed and its results were committed in `96a9cbb`. It preserves the
original cases, scorer, prompts, generation settings and 16/16 threshold.
Source identities and all sixteen case diagnostics are recorded in
`outputs/preference_program/results/eligibility_qualification_v55/frozen_error_analysis.json`.
No new model responses, GPU jobs or hosted CI were used for this analysis.

**Result: 11/16 correct. Qualification failed. The 56-response advice panel
stays blocked.** Successful parsing, EOS and engineering provenance establish
that the output can be scored; they do not establish task competence.

## What the model got wrong

One error violates eligibility: qualification-07 chooses B (quiet, explicitly
ineligible) instead of D (portable, the best eligible attribute). This is
consistent with selecting the highest priority without enforcing eligibility,
but one response does not identify its internal mechanism.

Four errors choose an eligible but lower priority attribute:

- qualification-06: chooses A/portable (priority rank 3), gold C/spacious (rank 2).
- qualification-11: chooses B/durable (rank 2), gold D/spacious (rank 1).
- qualification-14: chooses B/durable (rank 4), gold C/quiet (rank 2).
- qualification-15: chooses B/quiet (rank 4), gold D/durable (rank 2).

Thus four of five observed mistakes are preference ordering errors after a
feasible choice, while one is a feasibility error. These are separate output
failure classes; they do not reveal separate latent reasoning modules.

The model answers A five times, B eight times, C twice and D once. Balanced
gold actions A/B are correct 4/4 each; C is 2/4 and D 1/4. All five errors
return A or B. Top-priority-eligible cases are 7/8; top-priority-ineligible
cases are 4/8. These are descriptive slices of the same failed sixteen-case
panel, not alternative passing subsets.

A first-eligible-catalog heuristic agrees with 11/16 model responses, but
would itself achieve only 7/16 gold accuracy. An unconstrained top-priority
heuristic agrees with 8/16 responses. Neither explains every response.
Letter identities, catalog positions, attribute combinations and serial
presentation order are entangled in this small fixed panel. The gold sequence
cycles A/B/C/D, so no causal letter-bias or order-bias conclusion is justified.

## Scientific scope

This assay supplies the preferences explicitly and asks for execution of a
simple rule. It measures neither discovery of hidden user preferences nor
fitting an explicit reward/RL proxy. It does not compare tool use, prompt
optimization, adapters, or general assistance outcomes. Here the 1.5B BF16
checkpoint can produce well-formed responses but does not reliably execute
the stated rule. The failure is sufficient to stop the prespecified advice
study; it does not establish that LLMs generally cannot use preference proxies.

The sixteen cases are deterministic, constructed and single-run. Percentages
are exact descriptions of this panel, not independent user-population
estimates. No population confidence interval, post-hoc significance test,
threshold relaxation or selected-case rescue is warranted.

## Engineering and resource evidence

CPU token preparation 33468730 completed with sixteen valid 122-token prompts,
three-path ID equality and JSON equality. CPU hash gate 33469170 reverified
current model/tokenizer files and runtime archive before GPU submission. Its
one-CPU/8GB request became three actual CPUs: 25 seconds, 75 allocated CPU
seconds, 2.127 observed CPU seconds, peak RSS 18,304 KiB. This streaming-hash
stage should use less RAM in a future design to avoid extra CPU allocation.

GPU job 33469187 allocated exactly one H200 2g.35gb MIG, four CPUs, 32GB,
34 seconds elapsed versus five minutes requested. It used 136 allocated CPU
seconds and 7.615 observed CPU seconds. Peak CUDA allocated/reserved memory
was 3,132,609,536 / 3,149,922,304 bytes, well below the frozen 12GiB ceiling.
Sixteen generations produced 32 tokens and nonzero CUDA event spans totaling
4.245 seconds. These device spans include first-call setup and gaps, not an
estimated utilization percentage. Host-wide nvidia-smi memory and N/A MIG
utilization cannot be attributed to this job. At framework cleanup allocated
memory was 32MiB, but roughly 2.88GiB remained reserved until process exit;
normal terminal success is recorded without claiming earlier full release.

This smoke demonstrates bounded real GPU work. It does not justify long GPU
jobs, larger allocations, or dependable startup timing. Prior CPU timeout and
shutdown failures are retained. Full installed package-content identity beyond
verified pinned components remains unproven.

## Concrete next scientific recommendation — design only

Keep v55 as the failed baseline. Before examining advice compliance, separate
preference-rule execution from the way the rule is represented.

A next small pilot can remain **sixteen responses**, with two freshly generated
semantic cases, one with top priority eligible and one ineligible. For each
case use four rotations of output letter assignment and two information
formats: the existing attribute/priority prose and a structured catalog that
adds explicit priority ranks. Rotate letters while holding semantic options
and catalog row positions fixed. Pair exact semantics across formats; fix a
randomized run order and fresh cases before any generation. This distinguishes
sensitivity to letter identity from the additional computation required to
map attributes to ranks. Explicit ranks are an information transformation,
not evidence of learned preferences or an optimized prompt.

Two semantic cases provide a mechanism smoke only, not a broad capability
estimate. Analyze all sixteen paired outputs, eligibility and rank errors,
label invariance, parse/EOS/caps and token identities. Preserve v55's threshold
and failure. Any later confirmation requires independent cases and its own
registration; no result from this new design retroactively qualifies v55 or
opens the advice panel.

In parallel, without model calls, enumerate the finite categorical task and
validate a deterministic reward/proxy tool: assign ineligible actions a hard
constraint and rank eligible attributes by the supplied preference order.
Check oracle uniqueness and invariance under option relabeling. This supplies
a reproducible computational control for the larger agenda; it cannot by
itself demonstrate an LLM's ability to infer, invoke or follow the proxy.
Only after a competent independent baseline is established should tool output,
prompt optimization and fine-tuning be compared on fresh paired users/cases.

No new experiment has been authorized or submitted by this report. There is
no new pending approval request; this is a recommendation for the ongoing
research record.

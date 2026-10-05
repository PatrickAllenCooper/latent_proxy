# Registered preparation: eligibility and priority assay v45

Status: CPU preparation complete; no inference, GPU/API calls, training, execution runner or Slurm submission authorized. This registration records a new measurement lane approved for preparation. It does not change the failed numerical gate.

## Claim and scope

Given an explicit eligibility constraint and a qualitative user priority list, can the pinned assistant apply both rules and reject adviser recommendations that violate them? Numeric scores, arithmetic and numeric argmax are absent. Ordered-list lookup and label mapping remain residual capabilities and require qualification.

This tests supplied preference adherence and final-choice correction, not inferred preferences, a learned reward model, human behavior, preference noise, tools or fine-tuning. Historical v44 remains10/16, including four-digit1/4; v43 BF164/4 versus NF43/4 remains a one-case, fixed-order, kernel/loading-package contrast. No scores will be pooled across these lanes.

## Frozen construction and boundaries

Generator/scorer: scripts/prepare_eligibility_priority.py, seed450051. Four categorical attributes (quiet, spacious, portable, durable), one per label A–D; three eligible options and one ineligible option. The user selects the earliest eligible attribute in the supplied priority list. There is always exactly one oracle winner. Option listing remains A–D; attribute assignments and priority lists vary.

Four development cases are CPU fixtures only, not a prompt-search allowance or authorized model calls. Sixteen qualification cases and eight subsequent advice cases are structurally disjoint from development and from one another. Gold labels are balanced in every split. In qualification/advice splits, half have the highest-priority attribute ineligible, so ignoring eligibility can produce a wrong choice. Balanced construction is not iid population sampling; structurally distinct cases share a grammar and vocabulary. No semantic generalization is claimed.

All case files, exact single-user messages and prompt hashes are frozen under outputs/preference_program/manifests/eligibility_priority_v45. Qualification is16 no-advice responses. Subsequent advice panel is8cases×7arms=56responses, not authorized by qualification approval alone. Per-case advice is the true winner, the first listed eligible loser, or the ineligible option; wrong labels are fixed before outputs and are not an optimized adversarial distribution.

## Arms and estimands

Seven prospective advice arms: no advice; correct neutral; correct verification; wrong eligible neutral; wrong eligible verification; wrong ineligible neutral; wrong ineligible verification. Advice always says the adviser may be mistaken. Verification adds exactly “Check the recommendation against the user rule before choosing.” All other content is paired within case.

Primary descriptive estimands: correct final choice in each arm; paired wrong-eligible verification minus neutral correctness; paired wrong-ineligible verification minus neutral correctness. Report the eight case-level differences and mean for each contrast. Secondary: wrong advice versus no-advice correctness, correct-advice preservation, eligibility violations, eligible priority errors, adviser-copy frequency, parse failures and EOS/length caps. An adviser-copy change alone is not correction; correction requires the oracle winner. No utility/regret metric is invented for categorical priorities.

Hypotheses: (H1) no-advice rule application passes the registered qualification; (H2) verification improves correction of eligible lower-priority advice; (H3) verification improves correction of ineligible advice; (H4) correct advice does not lower final correctness relative to no advice. Eight advice cases are a descriptive mechanism pilot, not a confirmatory effect-size claim. Ties and zero differences remain reportable nulls. Any later confirmatory power/sample-size plan requires separate registration from pilot variance, with fresh cases.

## Scorer and CPU validation

Strict fullmatch of whitespace plus one uppercase A–D. Parse failures count incorrect and are reported separately. Eligibility violation is an actually selected ineligible option; parse failures are not silently classified as safe. Eligible priority error is an eligible loser. Oracle iterates the user priority list and returns its first eligible attribute's label.

CPU fixtures cover every possible letter for all28cases, valid whitespace, malformed/empty/multiple/lowercase responses and wrong-advice correction. Separate validation recomputes the winner by minimizing a priority-position index among eligible catalog options. Exact prompts contain no answer annotation. Fixture receipt records source hashes, preparation timestamps and zero model/API/GPU activity. Raw catalog values, advice labels, scorer classifications and all model traces must be retained if execution is later approved.

## Acceptance and stopping rules

Stage1 requires separate explicit execution approval for exactly16qualification responses. Pass requires16/16correct, zero parse failures, EOS in every response, zero length caps, exact registered prompts/checkpoint/runtime/decoding, and complete resource provenance. This is a local measurement gate, not a population accuracy estimate. Scientific failure stops for conceptual review. Incomplete output or provenance/resource failure yields no qualification. No retry, model/prompt/threshold tuning or enlargement based on qualification outputs.

Passing permits consideration of a separately approved56response advice pilot on its disjoint frozen cases. It does not automatically launch it or a320response study. Stage2 has no posthoc success threshold; report all prespecified contrasts and failures, then stop for review. Failure of no-advice on the advice split limits attribution of advice errors and must remain visible; do not delete cases or condition primary results on baseline success.

Preference-swapped counterfactuals, irrelevant-feature perturbations, dialogue preference discovery, reward proxies and optimizer/tool/fine-tuning arms are deferred—not silently included in this first panel. Those require new estimands and independent cases.

## Proposed execution envelope, not current spending authorization

Same pinned Qwen2.5-1.5B-Instruct snapshot989aa7980e4cf806f80c7fef2b1adb7bc71aa306, BF16 unquantized on cuda:0, SDPA, eval/use_cacheFalse, native single-user chat template, greedy24outputtokens, max512prompttokens. Qualification model-call budget16, current authorized model-call budget0. If separately approved: one H2002g.35gbMIG, ah200/gpu-normal, accountucb736_asc1,4CPU,32Ghost,5minutes;45sCPU-prepared runtime staging,90sloading,10stotalgeneration,285scomplete cap,12GiBframework memory stop. CPU tokenization/cache/hash validation must precede GPU allocation when practical. Prior BF16 run peaked~2.91GiB and completed16responses in34seconds, but longer prompts are not yet timed; watchdogs bound the attempt, not promise completion. No larger-model/backbone training or multi-GPU allocation.

Any future runner must record prompt/generated IDs, EOS, effective tokenizer/template/model/runtime/source hashes, dtype/quantization/attention settings, stage timestamps, GPU framework memory and generation progress. MIG utilization uncertainty and retained allocator reservation must be reported. A frozen execution revision, live headroom/duplicate check and fresh output root are required before any submission.

## Precise next gate

Review and approve or revise the registered scientific scope and the proposed16-response no-advice qualification budget. Freeze a prospective execution runner and CPU tokenization/provenance checks only after that decision. Preparation approval has been fulfilled; no model execution has occurred.

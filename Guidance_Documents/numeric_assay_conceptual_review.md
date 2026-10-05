# Decision required: what should preference-use verification measure?

Status: review only. No design adopted, new inference, prompt search, retry, or submission. Historical failed gate remains failed.

## Evidence in custody

Job33409095, frozen v44, scored10/16 against the registered16/16 criterion. Magnitude slices were single-digit3/4, two-digit4/4, three-digit2/4, four-digit1/4. All16 responses ended with EOS, with no parse failures or truncation; input/output provenance and resource checks passed. This is a scientific measurement failure, not a parser or job failure. v43 BF16 scored4/4 on one reused case versus NF43/4, with fixed arm order and different loading/kernel paths; that limited success did not qualify fresh numerical selection.

CPU reanalysis of the16 existing records found gold-label accuracy A2/4,B4/4,C2/4,D2/4. Responses were B8,D4,A2,C2. These counts suggest a B response imbalance on this panel, without identifying a causal label bias or population rate. Magnitude and label effects cannot be isolated from sixteen arrays with different values.

Value gaps are reported continuously per case in outputs/preference_program/results/fresh_bf16_v44/conceptual_review.json. Absolute gap is maximum minus second-largest; relative gap divides by maximum. No new gap cutoff or qualification threshold was introduced. Failures span relative gaps0.05–0.2802, while successes span0.0109–0.4: the model succeeded with a one-point gap [maximum92] yet failed on [1,6,8,3], gap2. Thus small gaps or large magnitudes alone do not explain all failures. This descriptive inspection cannot rescue the failed gate or estimate a magnitude/gap effect reliably.

## Why numerical argmax is a nuisance here

Advice verification currently asks the model to identify an optimal label from displayed numeric values while accepting or rejecting an adviser. Wrong final labels can arise from numerical comparison, label mapping, advice reliance, or instruction-following. A no-advice comparator that fails on fresh inputs prevents attributing advice-arm failures solely to misplaced trust or failed preference enforcement. Utility/regret still describes final behavior; it does not identify which mechanism failed.

Current evidence supports exact instruction copying in earlier handoff assays, sensitivity to prompt packages and a local NF4/BF16 package contrast, and unreliable fresh numeric selection for this pinned1.5B BF16 configuration. It does not demonstrate reliable advice correction, learned preference discovery, human behavioral fidelity, or a generally optimal prompt/tool/fine-tuning configuration. Copying an explicitly mandated recommendation is fidelity, not independent endorsement of its correctness.

## Choice A: retain exact numerical verification

Prospective claim: the assistant can independently recover the displayed unique numerical maximum and correct deliberately wrong advice, for a prespecified numeric task distribution.

Primary estimands: wrong-advice correction rate and paired accuracy/regret difference versus no advice, on the same fresh arrays. Correct advice preservation is a separate outcome. Numeric comparator accuracy must first pass an independently registered fresh measurement gate; evaluator code computing the answer does not establish that the assistant can do so.

Controls: no advice; correct neutral advice; wrong neutral advice; correct and wrong verification instructions, paired on fresh cases and balanced labels. Freeze score precision, gaps, label mappings, advice source/format and parsing before execution. A stronger model or new representation would be a new registered configuration requiring independent qualification, not a retry of v44. Providing an exact comparator tool changes the claim to tool-mediated numerical verification and must be separately labeled.

Cost/tradeoff: preserves the original numerical correction question but entails another model/measurement choice and fresh qualification. There is currently no qualified numerical configuration for this distribution. No new run is authorized by this review.

## Choice B: remove numerical comparison from the preference-use assay — recommended

Prospective claim: given an explicit, resolved user constraint or priority ordering, the assistant selects the matching action and rejects advice inconsistent with that preference. This measures use of supplied preferences and contradiction correction; it does not measure discovering preferences, calculating reward, or independently comparing numerical scores.

Example prospective case structure: a user rule says “choose the feasible option with the highest priority.” A neutral catalog explicitly supplies eligibility and ordinal rank for every option. There is one eligible rank1 option. Advice nominates either that option or a different eligible lower-priority option. Labels, wording templates, rank placements and irrelevant features are balanced in a newly frozen case generator. The CPU oracle derives the winner from the registered rule and catalog; the model receives the catalog, not a privileged answer annotation. Residual ordinal lookup and label mapping are still capabilities to qualify, not assumed solved.

Primary estimands: preference-consistent selection rate; wrong-advice rejection with a correct final choice; paired wrong-advice minus no-advice accuracy; correct-advice preservation. Evaluate exact user-rule consistency separately from adviser copying. Eligibility violations are distinct from choosing the wrong rank.

Controls: no advice to qualify rule application; correct advice; wrong advice; matched verification instruction arms; a preference-swapped counterfactual changes eligibility/rank while keeping option descriptions stable. A matched irrelevant-catalog perturbation checks distraction. Later discovery experiments would hide the user rule and compare inferred versus oracle profiles in separate arms; this is not included in the minimal redesign.

The minimal first stage would only qualify no-advice rule application on prospective independent cases. Advice arms require their own separately registered gate after qualification. No prompt/model/threshold optimization on gate cases, no automatic expansion, and no claim that a deterministic catalog proves human preference learning.

## Why this requires an explicit protocol decision

ChoiceB changes input representation, target capability, oracle and primary estimand. It cannot be substituted retrospectively for numerical argmax, counted as a pass of v44, or pooled with its accuracy. ChoiceA with a stronger model or exact tool likewise changes the configuration or mechanism. The historical10/16 and four-digit1/4 results must remain visible.

Recommendation: adopt ChoiceB as a separately registered preference-use measurement lane, and retain ChoiceA's failed numeric assay as a documented boundary. This most directly tests preference adherence without spending repeated GPU attempts on a nuisance comparison skill. Approval of this conceptual change would authorize CPU protocol preparation only unless execution is also explicitly authorized. Current stopping rule remains in force.

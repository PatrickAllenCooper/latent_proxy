# Preference-use cycle v70: implementation and disposition

## Execution track

The independently reviewed calibration packet was admitted by the user's October 9 implementation instruction after live account/partition checks, no queued user jobs, verified Python/tracer digests, frozen source validation, and successful sbatch test-only. One attempt, job 33641938, failed 1:0 after five seconds, with 13,560 KiB observed peak RSS and 0.170 consumed CPU seconds. One CPU and 256 MiB were allocated; no GPU was requested. A terminal receipt survived; no preparation, toy, or capture receipt did. Exact early failure cause is unobserved. This does not establish ptrace denial, framework failure, or a tracer incompatibility.

Slurm rounded the requested 30-second wall limit to 60 seconds. Actual use was five seconds, but the scheduler envelope mismatch prevents repeating the current submission unchanged. The next engineering proposal must retain a hard outer 30-second process watchdog and numeric, privacy-safe stage/error receipts so early failures remain diagnosable without saving raw stderr. No retry is authorized or submitted. Historical allocated accounting is now 2,430 CPU-seconds and 254 MIG-slice-seconds. Prior artifacts are preserved; local and remote final ledgers record this failure.

## Scientific track

`scripts/preference_use_cycle_v70.py` implements the rank-preference benchmark, independent oracle, strict scorer, budgeted tool session, discovery scoring, saved-response trace replay, and distinct hybrid correction scoring. Candidate evaluation returns reward and eligibility; constrained recommendation returns the optimal feasible action. Maximum two tool calls; invalid requests, forged returned values, and exhausted budgets fail. The assistant's raw action is always scored before any hybrid override.

The frozen development/validation/held-out partitions contain 12/6/6 distinct preference-order users and 48/24/24 distinct menu tasks respectively. Both profile identities and menu definitions are disjoint across splits. Forty-eight supervised examples are prepared from development only. Held-out smoke uses two users and eight distinct preference/exclusion cases, each evaluated in four arms: no information, explicit profile, proxy tool access, and validated hybrid. This yields 32 final responses, with a prospective 64 generation-call ceiling including tool interaction turns. No model call occurred. Two users support descriptive paired comparisons only, not population confidence intervals.

Private gold and preference orders are kept in the scoring case manifest. Initial no-information and tool prompts expose neither the priority order nor gold. These cases test explicit reward execution and use; the exact rank proxy is not a trained RL policy or a claim about human preferences. Future elicitation arms remain random/EIG/decision-focused AIF, paired with eight questions per user and separately scored inference-order error and induced decision regret. Existing domain elicitors are preserved; no new discovery result is claimed here.

The original v60 sixteen-case 32B reader qualification remains unchanged and hash-bound. It must complete 16/16 without parsing/truncation or custody failures before the comparative smoke proceeds. No GPU proposal is admitted by this preparation. The runtime blocker and first-response measurement remain unresolved; no 32B competence claim follows.

## Optimizer and fine-tuning contracts

GEPA is the single preselected future prompt optimizer. Its proposed ceiling is 32 candidates and 256 development feedback evaluations; validation is one 24-case pass after selection, with no held-out feedback. No optimizer was installed or run. A separate resource and implementation proposal must freeze its evaluation schedule before execution.

Fine-tuning remains a separate adapter initialized from the pinned checkpoint. The prepared 48 development examples are available, but a training recipe and compute budget must be frozen separately before training. Weight-change verification and stated-profile regression evaluation are mandatory. These are preparation contracts, not completed optimizer/training implementations.

## Validation and next decision

The independent priority-walk oracle agrees with the reward-maximization implementation across all 2,304 finite cases. Ten new tests verify strict parsing, ineligible penalties, tool budgets and invalid requests, hidden-profile boundaries, original-versus-corrected scoring, preference discovery scoring, split separation, semantic diversity, and tool trace replay integrity. All twelve existing calibration tests still pass. A draft with one duplicated smoke semantic case was detected locally, retained, and replaced before any inference; the final panel has eight unique signatures.

The cycle's current exit is a documented execution blocker plus a frozen, CPU-validated scientific protocol. Next required decision: review the narrow early-failure observability/watchdog correction before considering another separately admitted CPU attempt. Prompt optimization, fine-tuning, and comparative GPU inference remain held.

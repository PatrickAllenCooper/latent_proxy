# Preference-aligned agent experiment program

This document tracks the staged experiments that distinguish preference
recovery, recommendation quality, interaction format, and safety constraints.
It is exploratory: run pilots and validate their raw evidence before expanding
the model and domain matrix.

## Stage A: decision-focused elicitation

`run_canonical_campaign.py` supports adaptive EIG (`active`), static EIG
questionnaire (`fixed`), library-random (`random`), one-step decision-impact
(`decision_impact`), and three AIF-inspired epistemic/pragmatic mixtures
(`aif_20`, `aif_50`, `aif_80`). The decision-impact arm measures expected
reduction in posterior variance over scenario-specific optimal allocations;
it is a disagreement proxy, not literal economic regret. The AIF-inspired
arms combine candidate-pool-normalized mutual information about preference
particles with finite-menu expected value of sample information for a
downstream allocation decision. The finite menu and utility approximations
are recorded in the source; this is a practical acquisition heuristic, not an
exact solution to a full partially observed control problem.

Each user record stores true and inferred theta, parameter error, allocation
alignment, expected-utility regret, quality-floor status,
and question count. AIF arms additionally store per-query epistemic and
pragmatic score diagnostics. `summarize_preference_campaign.py` emits per-user
CSV, arm/domain means with bootstrap intervals, and paired contrasts against
random and adaptive EIG. `compare_utility_forms.py` compares matched absolute
and return-normalized runs.

Decision regret is evaluated against a multistart expected-utility optimum on
the long-only allocation simplex. Both the optimum and recommendation use the
same true-type prospect utility and 48-point Gauss-Hermite quadrature. The
environment's certainty-equivalent allocation remains the recommendation
policy for an inferred profile, but is no longer treated as ground truth for
regret. `recompute_campaign_decision_regret.py` updates historical per-user
campaign records with this evaluator and regenerates their summaries.

The next mechanism pilot is the 50-user game-only comparison in
`scripts/slurm/run_aif_game_pilot.slurm`. Its six paired arms are adaptive EIG,
random, decision-impact, and three AIF mixtures. Advance to other domains only
after checking the raw per-query diagnostics, result validity, and paired
uncertainty; increase the user panel if the pilot intervals are too wide.

The five-seed game replication found AIF 50% reduced decision regret relative
to EIG while parameter recovery favored EIG; AIF 50% improved recovery and
action alignment over random. In stock and supply chain, AIF 50% improved on
EIG in supply chain but did not beat random, while stock was near a decision
ceiling. The return-normalized supply-chain replication is defined in
`scripts/slurm/run_aif_supply_chain_normalized.slurm` to test whether the
utility formulation changes that comparison.

## Stage B: DPO interaction-format comparison

The stated-profile adherence study now includes `true`, `elicited`, and
`posterior_summary` conditions. The summary condition provides the posterior
mean and 90% parameter intervals. It stores each raw prompt and completion,
parse status, raw and quality-constrained allocation, alignment, and decision
regret. All conditions are re-scored on the same initial environment state.

The self-elicitation study stores raw question and recommendation generations
plus per-user parse failure rates. A fourth `dpo_dialogue` checkpoint slot
allows direct comparisons with the original base, Phase 1, and Phase 2.

## Stage C: dialogue-augmented Phase 2

Set `--dialogue-context-rounds 4` on `train_alignment.py` to construct
Phase 2 preference pairs from synthetic comparisons and recorded user choices,
without placing numeric theta in the training prompt. The training objective,
pair count, epochs, and Phase 1 starting checkpoint remain configurable as in
the original Phase 2 pathway. The H200 Slurm script writes a new checkpoint,
diffs its adapter tensors against Phase 1, then a dependent evaluation job
compares stated-profile and self-elicitation performance.

## CURC jobs

The Slurm entrypoints use Alpine `acpu` for CPU studies and one typed H200 on
`ah200` for DPO. They require the `latent-proxy-env` environment, model cache
at `/scratch/alpine/paco0228/hf_cache`, and source snapshot at
`/home/paco0228/latent_proxy_experiments`. All generated artifacts and logs go
to `/scratch/alpine/paco0228/latent_proxy_runs/77add19`. CURC's `/projects`
filesystem was full when this run set was prepared, so code and the two adapter
checkpoints use the mostly empty home allocation; generated output uses scratch
as CURC recommends. Scratch is not backed up and is automatically purged after
90 days, so completed results must be transferred to this machine promptly.
Per-user records are written atomically, so re-running a campaign skips
completed users. Never cancel or modify jobs from other projects to make room;
pending time is acceptable.

Recommended first submissions:

```bash
sbatch scripts/slurm/run_preference_baseline.slurm
sbatch scripts/slurm/run_decision_pilot.slurm
sbatch scripts/slurm/train_dialogue_phase2_h200.slurm
```

The active-inference CPU pilot is an independent source snapshot and scratch
root (`latent_proxy_aif_pilot` and `latent_proxy_runs/aif-game-pilot`) so it
does not overwrite completed runs from the first program revision.

Submit `evaluate_dialogue_phase2_h200.slurm` with a dependency on the training
job. Record each returned Slurm ID, source revision, command, and output path in
the project job ledger. Compare paired seeds/users and inspect raw generations
and adapter weight differences before treating aggregate metrics as evidence.

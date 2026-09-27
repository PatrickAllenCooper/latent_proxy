# Preference-aligned agent experiment program

This document tracks the staged experiments that distinguish preference
recovery, recommendation quality, interaction format, and safety constraints.
It is exploratory: run pilots and validate their raw evidence before expanding
the model and domain matrix.

## Stage A: decision-focused elicitation

`run_canonical_campaign.py` now supports four query strategies: adaptive EIG
(`active`), static EIG questionnaire (`fixed`), library-random (`random`), and
one-step decision-impact (`decision_impact`). The last chooses the question
with the greatest expected reduction in posterior variance over the
scenario-specific optimal allocation. This is a decision-disagreement proxy,
not literal economic regret.

Each user record stores true and inferred theta, parameter error, allocation
alignment, common-random-number expected-utility regret, quality-floor status,
and question count. `summarize_preference_campaign.py` emits per-user CSV,
arm/domain means with bootstrap intervals, and paired contrasts against the
random arm. `compare_utility_forms.py` compares matched absolute and
return-normalized runs.

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
`/projects/paco0228/latent_proxy_experiments`. Per-user files are written
atomically, so re-running a campaign skips completed users. Never cancel or
modify jobs from other projects to make room; pending time is acceptable.

Recommended first submissions:

```bash
sbatch scripts/slurm/run_preference_baseline.slurm
sbatch scripts/slurm/run_decision_pilot.slurm
sbatch scripts/slurm/train_dialogue_phase2_h200.slurm
```

Submit `evaluate_dialogue_phase2_h200.slurm` with a dependency on the training
job. Record each returned Slurm ID, source revision, command, and output path in
the project job ledger. Compare paired seeds/users and inspect raw generations
and adapter weight differences before treating aggregate metrics as evidence.

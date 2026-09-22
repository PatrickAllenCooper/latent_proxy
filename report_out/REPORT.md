# latent_proxy — DPO Fine-Tuning Results

*2026-09-22. Covers both training phases and both post-training evaluations.*

## The question

Two things were tested. First: can active questioning recover a synthetic user's
prospect-theory preferences — risk aversion (`alpha`), loss aversion (`lambda`),
and discount factor (`gamma`) — from binary choices? That machinery (particle-filter
posterior + EIG-guided querying) was validated in earlier work and isn't re-covered
here. Second, the focus of this report: can DPO fine-tuning make an LLM's
*generated behavior* actually adhere to those preferences — including preferences
that were themselves *elicited*, not just handed to the model as ground truth?

## Training: two phases, both verified by direct weight diffs

Phase 1 ("quality floor," no user profile in the prompt) and Phase 2
("type-conditioned," explicit profile in the prompt, continuing from Phase 1)
were trained via QLoRA DPO on `Qwen/Qwen2.5-1.5B-Instruct`.

| Run | Loss (start→end) | Reward accuracy | Reward margin | Steps |
|---|---|---|---|---|
| Phase 1 | 0.692 → 0.0003 | 0.43 → 1.00 | 0.002 → 16.6 | 3,660 |
| Phase 2 | 0.700 → 0.103 | 0.43 → 0.91 | -0.009 → 10.1 | 7,500 |

Two silent bugs (trl's reference-model aliasing, and PEFT's default frozen
adapter) each produced healthy-*looking* logs — varying loss, varying gradients —
while training nothing at all. Neither crashed. Both were only caught by directly
diffing adapter tensors between checkpoints, which is why that check is reported
here as a first-class result, not a footnote: Phase 1→Phase 2, **0 of 224 tensors
identical** (max diff 0.062), confirming real, continued learning. Full curves:
`training_phase1_curve.csv`, `training_phase2_curve.csv`.

## Evaluation 1: does the model follow a *stated* profile?

`adherence_study` prompts each checkpoint with a preference profile rendered
directly into the prompt — the same shape Phase 2 trained on — under two
conditions: the user's **true** profile, and a profile **actively elicited**
via the questioning loop (a noisy estimate, the realistic deployment case).
Scored as Spearman correlation between the recommended and the truly-optimal
allocation (n=50 users/cell, `game` domain).

| Condition | True profile | Elicited profile |
|---|---|---|
| base (untuned) | -0.045 | 0.031 |
| dpo_phase1 (quality-only) | -0.017 | 0.154 |
| **dpo_phase2 (type-conditioned)** | **0.111** | **0.189** |

dpo_phase2 significantly beats base (p=0.016 true, p=0.0018 elicited) and beats
phase1 on the true profile (p=0.0006). The elicited-profile result is the one
that matters most: **fine-tuning still helps even when the profile fed to the
model is a noisy, actively-elicited estimate, not ground truth.** The
phase1-vs-phase2 gap specifically on elicited profiles isn't confirmed yet
(p=0.114). Full scores: `adherence_study_summary.csv` / `_per_user.csv`;
tests: `adherence_hypothesis_tests.csv`.

**dpo_phase2 is the only condition with any quality-floor violations** (base
and phase1 stay at 0%). Diagnosed directly (not just flagged): in every
violation, the model collapses to a 100%-single-channel allocation instead of
a diversified one. Not random — 5 of 6 violating profiles have high loss
aversion (lambda ≈ 1.9–2.9). A worse sub-pattern in 2 of 6: for very high risk
aversion (alpha ≈ 2.7–2.9) it twice concentrated into the *dominated, volatile*
channel — backwards — scoring the worst alignment of the whole diagnostic run
(-0.775). Detail: `violation_diagnostic_cases.csv`.

## Evaluation 2: does the model follow its *own elicitation dialogue*?

A second, independently-built harness (`run_dpo_study.py`) never shows the
model a profile at all — it runs its own natural-language elicitation dialogue,
then recommends. If fine-tuning genuinely helps, this should show the same
ordering. **It shows the opposite.**

| Condition | n=5 spot-check | n=30 full run |
|---|---|---|
| analytical (closed-form optimum) | 0.96 | 0.91 |
| **base (untuned)** | **0.81** | **0.22** |
| dpo_phase1 | -0.04 | -0.21 |
| dpo_phase2 | -0.09 | -0.14 |

Both DPO checkpoints score *negative* — worse than chance — at both sample
sizes, while base stays clearly positive. Formal test (dpo_phase2 vs. base,
H1: DPO better): **p=0.996, fails to reject.** This is not a parsing artifact —
three real bugs in the harness itself were found and fixed first (truncated
generations, prompt drift, missing `%` signs), and the n=30 run's aggregate
parse-failure rate is 1.8%. Full scores: `self_elicitation_summary.csv` /
`_per_user_n30.csv`.

**Working hypothesis, not yet independently confirmed:** Phase 2's training
data is entirely single-turn, profile-in-prompt, no dialogue — narrow
fine-tuning on that distribution may be eroding general dialogue competence
(asking sensible questions, tracking conversation state) that base retains.

## Bottom line

The two evaluations point to different deployment choices for the *same*
checkpoints. If the user's profile can be made explicit — elicited via the
project's validated active-questioning loop, then rendered into the prompt —
fine-tuning clearly helps, including under the noisy, realistic elicited-profile
case. If the model instead has to run its own open-ended elicitation dialogue
with no explicit profile, fine-tuning currently *hurts*: use the base model for
that deployment shape, or extend Phase 2's training data to cover dialogue
before using it there.

See `README.md` in this folder for how to read the CSVs, and the fuller
narrative (including the three rounds of bug-fixing behind each number) at
`~/Desktop/latent_proxy_results/README.md`.

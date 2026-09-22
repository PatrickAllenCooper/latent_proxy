# report_out — how to read this folder

This folder is a static snapshot of the project's results as of 2026-09-22,
generated from the raw JSON/log files produced by the actual training and
evaluation runs (not re-derived or estimated). Start with `REPORT.md` for the
narrative; this file explains the supporting CSVs.

## Start here

**`REPORT.md`** — the two-page summary. Read this first.

## Training data

- **`training_summary.csv`** — one row per phase: loss/reward-accuracy/margin
  at the start and end of training, plus step/epoch/pair counts.
- **`training_phase1_curve.csv`**, **`training_phase2_curve.csv`** — the full
  training curve, one row every 10 optimizer steps. Columns: `loss` (DPO loss,
  lower is better, `ln(2)=0.693` is the "no learning" floor), `rewards_accuracies`
  (fraction of pairs where the model now prefers the chosen response, 0.5 =
  chance), `rewards_margins` (how confidently, in log-prob units), `grad_norm`,
  `learning_rate`. If you want to plot "did training converge," these two files
  are the source.
- **`weight_diff_verification.csv`** — not a metric, a sanity check. This
  project hit two bugs where training *looked* healthy (varying loss, varying
  gradients) but zero parameters were actually updating. The only test that
  caught it: diff the adapter weights between two checkpoints directly.
  `tensors_identical=0` (of 224) is the good outcome — it means the weights
  actually moved.

## Evaluation 1 — adherence to a stated profile (`adherence_study_*`)

The model is told the user's preference profile directly in the prompt (either
the **true** profile or one **elicited** via the active-questioning loop), and
scored on whether its recommendation matches what's actually optimal for that
user.

- **`adherence_study_summary.csv`** — one row per (condition, theta_mode):
  `mean_alignment` is the headline number, a **Spearman rank correlation**
  between the recommended allocation and the true-optimal one. Range -1 to 1:
  1.0 = perfect match, 0 = no relationship, -1 = exactly backwards.
  `mean_violation_rate` is the fraction of recommendations that failed the
  environment's basic quality-floor checks (e.g. put everything into one
  channel) — these are bad allocations regardless of alignment score.
- **`adherence_study_per_user.csv`** — the same data unaggregated, one row per
  synthetic user (50 per condition × theta_mode cell). Use this if you want to
  re-run your own statistics or check the score distribution rather than just
  the mean.
- **`adherence_hypothesis_tests.csv`** — paired one-sided significance tests
  (`dpo_phase2` vs. each other condition). `p_value < 0.05` = significant at
  the conventional threshold; `conclusion` spells out `reject_null` /
  `fail_to_reject` directly. **Read `effect_size`'s magnitude, not its sign** —
  the sign follows the underlying test function's internal convention and
  isn't consistently "positive means phase2 wins" across rows; the
  `mean_alignment` columns in the summary CSV are the reliable way to see
  which direction a comparison actually goes.
- **`violation_diagnostic_cases.csv`** — every quality-floor violation found in
  a dedicated 50-user diagnostic re-run of `dpo_phase2`, with the full context
  for each: the true and prompted-profile `(gamma, alpha, lambda)`, the actual
  allocation the model produced, and why it failed. This is what the "not
  random — clusters at high loss aversion" claim in `REPORT.md` is based on.

## Evaluation 2 — self-elicitation dialogue (`self_elicitation_*`)

A different mechanism: instead of being told the profile, the model runs its
own natural-language elicitation dialogue (asks questions, proposes options),
then recommends. Scored the same way (Spearman correlation vs. true optimum).

- **`self_elicitation_summary.csv`** — mean alignment by condition, at two
  sample sizes (an early n=5 spot-check and the full n=30 run). Includes the
  `analytical` row — the closed-form optimal-policy ceiling, not an LLM at all,
  useful as a reference point for how much headroom exists.
- **`self_elicitation_per_user_n30.csv`** — per-user detail for the n=30 run
  only (the spot-check's raw per-user data wasn't saved to a results file,
  only its summary).

## What's deliberately *not* in this folder

Raw model checkpoints, full stdout/stderr logs, and the plotted PNGs live at
`~/Desktop/latent_proxy_results/` on the machine this was generated on — not
committed to git (checkpoints are large; logs are verbose). That folder's own
`README.md` has the fuller narrative, including the specific bugs found and
fixed along the way for each result above. This folder is the portable,
tabular subset of the same results.

## A general note on trust

Every number in `REPORT.md` traces back to a CSV in this folder, and every CSV
traces back to a JSON results file or raw log produced directly by the actual
runs (see `build_report_csvs.py`'s source paths if you want to verify a number
yourself — the script itself wasn't committed, but the mapping is simple: one
results JSON or log file per CSV, no manual transcription of numbers). Where a
finding was surprising, `REPORT.md` says so and points at the diagnostic data
rather than asking you to take it on faith — that's what
`violation_diagnostic_cases.csv` and the two `*_per_user*.csv` files are for.

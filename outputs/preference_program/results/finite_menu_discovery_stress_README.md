# Finite-menu preference discovery stress panel

The four `finite_menu_discovery_*_v2` directories contain complete local CPU
runs using seed 7001, 100 paired users, 4096 particles, eight sequential
questions, and budgets 0/2/4/8. Each contains `per_user.csv` (1600 unique
records), `queries.jsonl` (3200 traces), and `summary.json` with user-level
bootstrap contrasts. The scripts and raw rows define the metrics and task.

At budget eight, EIG minus random reward regret (95% user bootstrap interval):

- Matched: -0.0095 [-0.0173, -0.0015]
- Noisy: -0.0041 [-0.0166, 0.0098]
- Inconsistent: -0.0335 [-0.0499, -0.0181]
- Hidden preference shift after query four: +0.0385 [0.0115, 0.0669]

The shift run also has EIG minus random behavioral agreement -0.0543
[-0.0841, -0.0273]. This points to a stale stationary posterior. The noisy
comparison is imprecise. The inconsistent run has minimum posterior effective
sample size 2.43, so posterior calibration is not assured. These are analytic
synthetic-user results, not human-user findings.

# Finite-menu analytic-user discovery studies

Three complete runs are retained:

- `finite_menu_discovery_pilot_v2_256particles`: 50 users, seed 6001. The
  importance sampler concentrated on fewer than 10 effective particles for
  12/50 EIG users at eight questions, so this is a pipeline diagnostic.
- `finite_menu_discovery_pilot_v2`: the same 50 users, 1024 particles. This
  reduced the EIG low-effective-sample count to 2/50.
- `finite_menu_discovery_replication_v2`: 200 fresh users, seed 8001, 4096
  particles. One EIG user still fell below 10 effective particles.

On the fresh 200-user panel after eight questions:

| Acquisition | Reward decision regret | Reward decision behavioral agreement | Behavior decision agreement |
|---|---:|---:|---:|
| Random | 0.0252 | 0.7949 | 0.8045 |
| EIG | 0.0189 | 0.7946 | 0.8153 |
| Decision value | 0.0234 | 0.7886 | 0.8036 |
| AIF 50/50 | 0.0215 | 0.7911 | 0.8106 |

EIG minus random reward-decision regret was -0.0063, paired user-bootstrap
95% interval [-0.0105, -0.0025]. EIG minus random behavior-decision agreement
was +0.0108, interval [+0.0025, +0.0197]. The 50/50 AIF contrasts with random
included zero on both measures. These intervals are exploratory and do not
adjust for multiple comparisons.

Both strategies improved substantially from the zero-question prior: mean
reward-decision regret was 0.1311 before elicitation. Each run includes
per-user preference estimates, decision metrics, posterior effective sample
size, every selected question and answer, and paired bootstrap reports.

This study uses an analytic four-choice simulator with a matched inference
model. It does not yet use an LLM-user or natural-language parsing. The next
mechanism study should pass the same chosen queries through an LLM-user and
measure fidelity and parse failures separately.

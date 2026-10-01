# Joint preference and response-noise inference

Four 30-user development runs (`finite_menu_infer_noise_*_dev_v1`, seed 7101,
1024 particles) preceded four 100-user replications (`*_rep_v1`, seed 8501,
4096 particles). Each replication has 1600 unique user-budget-arm rows and
3200 raw query traces, with finite decision and posterior metrics.

Each particle carries a response-noise fraction sampled from 0, 0.10, 0.20,
or 0.40. Its response likelihood is `(1-epsilon) p + epsilon/4`; the
posterior updates preferences and epsilon together. The comparison is to the
same stationary EIG inference without this component. A separate arm adds
a single-answer surprise refresh.

At eight questions, inferred-noise EIG minus ordinary EIG paired regret
(95% user bootstrap) was:

- Matched: -0.0018 [-0.0052, 0.0011]
- Hidden preference shift: -0.0352 [-0.0568, -0.0163]
- Higher-temperature choices: -0.0013 [-0.0073, 0.0049]
- Inconsistent answers: -0.0120 [-0.0211, -0.0036]

The inferred random-choice fraction averaged 0.109 on matched users, where
the simulator's true random-choice fraction was zero, and effective sample
size reached 2.39. Preference/noise identifiability and uncertainty
calibration remain unresolved. No human-user inference is claimed.

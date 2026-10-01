# Explicit response-noise likelihood

Development runs (`finite_menu_robust_noise_*_dev_v1`) used 30 users, 1024
particles, and seed 7101. Fresh replications (`*_rep_v1`) used 100 paired
users, 4096 particles, and seed 8401. Each replication condition has 2000
unique user-budget-arm records and 4000 raw query traces with finite scores.

The robust arms replace each response likelihood `p` with `0.8 p + 0.05`,
equivalent to assuming a 20% uniform-choice component. One arm additionally
refreshes 25% of posterior mass when the observed choice has predictive
probability below 0.10. This exactly matches the simulator's inconsistent
response mixture, so that condition is partly model-matched by design.

At eight questions, robust EIG versus ordinary EIG paired reward-regret
differences (95% user bootstrap) were:

- Matched: +0.0026 [-0.0003, 0.0059]
- Hidden shift: -0.0281 [-0.0476, -0.0089]
- Higher-temperature noise: +0.0080 [0.0009, 0.0155]
- Inconsistent choices: -0.0126 [-0.0213, -0.0046]

The robust surprise trigger improved shift regret further but raised stable
regret. Minimum effective particle count reached 3.47. These results show
that a fixed response-noise assumption trades performance across user types;
they do not select a generally optimal configuration.

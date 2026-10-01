# Surprise-triggered posterior refresh

Development: `finite_menu_change_trigger_{matched,shift}_dev_v1`, 30 paired
users, 1024 particles, seed 7101. Replication:
`finite_menu_change_trigger_{matched,shift}_rep_v1`, 100 fresh paired users,
4096 particles, seed 8201. Replication directories each hold 2000 unique
user-budget-arm rows and 4000 query traces. All decision metrics and EIG
scores are finite. Query traces record the predictive probability of the
observed response and whether a 25% prior refresh was applied.

At eight questions, the threshold 0.10 arm versus static EIG had:

- Stable users: regret 0.0168 versus 0.0152; paired difference +0.0016,
  95% user bootstrap [0.0000, 0.0036].
- Hidden preference shift: regret 0.0590 versus 0.1273; paired difference
  -0.0683 [-0.0974, -0.0430].

Continuous 10% refresh reached regret 0.0223 and 0.0311 in the stable and
shifted panels respectively. The trigger fired 25 of 800 stable-user answers
and 54 of 800 shifted-user answers. All evidence is from analytic synthetic
users with an assumed behavioral likelihood; posterior calibration remains
uncertain when effective particle count is low.

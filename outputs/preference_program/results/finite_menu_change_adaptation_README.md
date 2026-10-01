# Change-aware posterior pilot and paired controls

`finite_menu_change_adaptation_pilot_v2` is the 20-user development pilot.
The `*_v3` matched and shift directories are 100-user paired runs, seed 7001,
4096 particles, 24 candidate queries, and 12 held-out evaluation menus.
Each has 1600 unique user-budget-arm records and 3200 raw query traces.

At eight questions, static EIG and 10% per-answer prior refresh gave:

| Condition | Static EIG regret | 10% refresh regret | Paired refresh minus static, 95% user bootstrap |
|---|---:|---:|---:|
| Stable | 0.0196 | 0.0333 | +0.0137 [0.0057, 0.0243] |
| Shift after four queries | 0.1482 | 0.0376 | -0.1107 [-0.1416, -0.0818] |

The shift panel's behavioral agreement rose from 0.6270 to 0.8078 under
10% refresh, paired difference +0.1807 [0.1363, 0.2279]. This tests an
analytic user model. The posterior still reaches effective sample size 7.30
in the shift condition, so its uncertainty is not considered calibrated.
Continuous refresh is not a selected deployment configuration; it is a
mechanistic baseline for a change-triggered posterior.

# Finite-menu proxy foundation pilots

The v1 menus were too easy: the generic baseline and a trained PPO policy
both defaulted to `safe_now` on the fresh panel. The v1 records and checkpoint
are retained as failed design diagnostics. The v2 menus widen timing, risk,
and loss tradeoffs. These are new benchmark units and should not be pooled
with earlier resource-game results.

On the fresh v2 panel (seed 3001; 50 paired users x 20 menus), mean normalized
regret and expected behavioral agreement were:

| Condition | Regret | Behavioral agreement |
|---|---:|---:|
| Generic profile | 0.1265 | 0.6093 |
| Exact reward tool | 0.0000 | 0.8360 |
| Exact behavior tool | 0.0215 | 0.8807 |
| Swapped-user reward tool | 0.2285 | 0.5125 |
| Learned PPO proxy | 0.0881 | 0.6782 |

The learned proxy minus generic paired regret difference was -0.0384 with a
user-bootstrap 95% interval [-0.0714, -0.0041]. Its behavioral-agreement
difference was +0.0689, interval [-0.0046, +0.1453]. This supports a modest
decision-utility gain from the learned proxy; behavioral improvement remains
uncertain. The exact behavior tool increased agreement by 0.0447 relative to
the exact reward tool and increased regret by 0.0215. Thus the two outcomes
form a measurable tradeoff in this simulator.

The first two v2 PPO attempts collapsed to `safe_now` and are retained as
`finite_menu_ppo_pilot_v2_attempt1` and `_attempt2`. In the third run, 200
oracle-demonstration warm-start updates gave held-out regret 0.1601 and 60.0%
oracle action match. A further 200 PPO updates improved these to 0.0881 and
72.3%. This is still well below exact reward scoring. The warm start used
ground-truth action labels and therefore this result cannot isolate PPO's
ability to discover preferences without demonstrations.

All raw per-decision records, checkpoints, and training reports are retained.
The LLM tool-use pilot uses a different user/scenario seed, 5001, and has not
yet been evaluated at the time of this note.

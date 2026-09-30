# Preference proxy mechanism program

This program tests preference discovery and preference use separately. The
controlled benchmark is `finite_menu_v2` in
`src/evaluation/preference_benchmark.py`. A payoff is a signed fraction of
initial resources delivered in a stated period. Ground-truth expected utility
is the probability-weighted sum of discounted prospect values. This is a new,
versioned objective; do not pool its regrets with historical terminal-wealth
or return-normalized game results.

## Evidence and gates

The first `finite_menu_v1` pilot is retained under result directories ending
`_v1`. It failed the decision-diversity gate: a generic profile and the first
PPO policy chose the safe action throughout the held-out panel. The v2 menu
widens timing and risk tradeoffs; extreme profiles now choose different actions.
Hand-checked timing, loss, behavioral-bias, and feasibility examples are in
`tests/test_preference_benchmark.py`.

The v2 CPU pilot evaluates 50 paired users on 20 menus. Its conditions are a
generic profile, exact reward scoring, exact behavioral choice, swapped-user
reward scoring, and a learned proxy. `scripts/run_proxy_foundation_pilot.py`
writes per-decision CSV and user-clustered bootstrap summaries. Seeds 1001
and 3001 are development and fresh holdout panels respectively. The first two
v2 PPO attempts collapsed to the safe action and are preserved. The final v2
policy uses 200 oracle-demonstration warm-start updates followed by 200 PPO
updates. Its checkpoint and report are under `finite_menu_ppo_pilot_v2`.
This policy is a learned approximation, not the independent scoring oracle.

The controlled LLM pilot presents the same numerical preference profile to
all conditions. It compares no tool, exact reward advice, and learned RL policy
advice for the base and existing dialogue DPO checkpoint. The advice is
injected by the harness as a visible tool result. This first pilot measures
whether the model uses information supplied by a proxy; voluntary tool calling
is a later condition. Every prompt, raw completion, and parse failure is saved.

Advance only if the output parser, scoring, model/checkpoint identity, and raw
generations pass inspection. Subsequent work: optimize prompts with TextGrad
and GEPA on disjoint development users; compare 3 x 2 x 2 prompt/tool/adapter
conditions; then repeat with inferred proxies, elicitation histories, proxy
swaps, and tool removal. Train and evaluate shared and personal adaptation
separately. Keep utility regret and behavioral agreement as co-primary outcomes.

## Artifact and custody rules

Record commit, job ID, seed, condition, checkpoint, and output path for each
CURC job in `outputs/preference_program/jobs.json`. Copy completed results
locally, validate expected record counts and raw traces, commit all results,
and push to origin. Preserve remote source, checkpoints, logs, scratch data,
and every other project's jobs.

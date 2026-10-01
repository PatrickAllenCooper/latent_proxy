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
and 3001 are development and training-audit panels respectively. The first two
v2 PPO attempts collapsed to the safe action and are preserved. The final v2
policy uses 200 oracle-demonstration warm-start updates followed by 200 PPO
updates. Its checkpoint and report are under `finite_menu_ppo_pilot_v2`.
This policy is a learned approximation, not the independent scoring oracle.
After the policy was frozen, seed 9001 supplied 200 new users and 20 new
menus for a fresh replication. No policy or menu changes followed that run.

The controlled LLM pilot presents the same numerical preference profile to
all conditions. It compares no tool, exact reward advice, and learned RL policy
advice for the base and existing dialogue DPO checkpoint. The advice is
injected by the harness as a visible tool result. This first pilot measures
whether the model uses information supplied by a proxy; voluntary tool calling
is a later condition. Every prompt, raw completion, and parse failure is saved.

The full paired 12-user by 8-menu LLM pilot completed on one 35 GB H200 MIG
slice in 81 seconds. Its 576 raw generations are complete and parseable. Both
base and dialogue-adapted models chose C on all 96 no-tool cases, then copied
the supplied proxy letter on all 96 exact-reward and 96 learned-RL cases.
Their completions were identical on all 288 matched prompts. Mean normalized
regret was 0.5158 with no tool, 0.1424 with learned RL advice, and zero with
exact reward advice; the learned-tool minus no-tool paired user bootstrap
interval was [-0.5497, -0.2063]. This shows that explicit proxy advice can
steer these models in the current forced-result prompt. It does not establish
voluntary tool use, independent preference inference, or no adapter effect in
other formats. The next pilot should swap and degrade proxy advice, expose
numerical scores rather than only a letter, and compare prompt variants on
held-out users before scaling.

The follow-up advice-reliability ablation reused those 12 users and 8 menus,
with three CPU-prepared prompt variants and 288 new base-model generations.
Directly recommending the worst action made the model choose it on all 96
menus (mean normalized regret 1.000 versus 0.516 without advice; paired
user-bootstrap difference +0.484 [0.344, 0.624]). Warning that the advice
might be inaccurate reduced copying to 36/96, but the model mostly reverted
to C (92/96) and mean regret was 0.528. Giving exact numerical utility scores
without a recommended letter led it to choose A on all 96 menus; A happened
to be optimal on 48/96, yielding regret 0.320. The paired score-vs-no-tool
regret interval [-0.468, 0.067] is imprecise. The observed constant-letter
behavior does not demonstrate that it compared scores. This is strong evidence
of advice-following vulnerability in the direct format and limited evidence
that a simple caution repairs it. A next prompt-optimization pilot should
require an explicit score comparison and separate tool trust from arithmetic.

The analytic-user discovery study in `scripts/run_finite_menu_discovery.py`
uses a separate particle posterior over reward preferences and behavioral bias.
It compares random, mutual information, exact finite-menu decision value of
sample information, and their 50/50 normalized mixture at question budgets
0, 2, 4, and 8. Users are paired across acquisition arms. A given user gives
the same stochastic response whenever two arms ask the same question. Target
menus for acquisition are disjoint from held-out evaluation menus. Full query
traces and both true and estimated preference parameters are saved.

The 50-user, 256-particle pilot suffered importance-weight concentration.
The same users were rerun with 1024 particles, then 200 fresh users with 4096
particles (seed 8001). On the fresh panel at eight questions, EIG exceeded
random in behavioral agreement and lowered decision regret. The AIF mixture
did not clearly exceed random. This is matched-model analytic evidence and
does not establish robust conversational elicitation. One EIG posterior still
had fewer than 10 effective particles at the final round; investigate this
before using posterior intervals as calibrated uncertainty.

The 100-user stress panel (seed 7001, 4096 particles) repeats the paired
question comparison under matched responses, noisier choices, 15% inconsistent
choices, and a hidden preference change after four queries. Each condition has
1600 unique user-budget-arm records and 3200 query traces. At eight questions,
EIG minus random reward regret was -0.0095 (95% user bootstrap [-0.0173,
-0.0015]) in the matched condition, -0.0041 [-0.0166, 0.0098] with noisier
choices, and -0.0335 [-0.0499, -0.0181] with inconsistent choices. After the
preference change, EIG became worse than random: +0.0385 [0.0115, 0.0669]
reward regret and -0.0543 [-0.0841, -0.0273] behavioral agreement. The
posterior assumes a stationary user; this is a failure mode of that model, not
evidence that random elicitation is generally preferable. An adaptive posterior
with change detection or forgetting is the next mechanism to test. All stress
conditions remain analytic simulator evidence; the inconsistent condition
reached effective sample size as low as 2.43 and requires calibration checks.

An adaptive follow-up compared static EIG with EIG whose particle weights
mix with a uniform prior before each answer (10% or 25% hazard). The 20-user
pilot was followed by 100 paired users on both shift and matched controls,
using 4096 particles and seed 7001. On shifted users at eight questions,
10% hazard reduced regret from 0.1482 to 0.0376 and raised behavioral
agreement from 0.6270 to 0.8078. The paired hazard-minus-static regret
difference was -0.1107 (95% user bootstrap [-0.1416, -0.0818]). On stable
users it increased regret from 0.0196 to 0.0333, difference +0.0137
[0.0057, 0.0243]. Continuous forgetting helps after change but pays a stable
user cost. Next test a surprise-triggered refresh or explicit change-point
posterior against both controls; do not select the hazard by shifted cases alone.

A change-trigger pilot used posterior predictive probability of the observed
answer, refreshing 25% of weights only when it fell below 0.05 or 0.10. A
30-user development panel was followed by 100 fresh paired users (seed 8201,
4096 particles) under stable and shifted preferences. The 0.10 threshold
triggered on 25 of 800 stable-user answers and 54 of 800 shifted-user answers.
At eight questions it changed regret versus static EIG from 0.0152 to 0.0168
for stable users (paired +0.0016, bootstrap [0.0000, 0.0036]) and from 0.1273
to 0.0590 after a shift (paired -0.0683 [-0.0974, -0.0430]). Continuous 10%
refresh reached 0.0311 after a shift but cost more under stability (0.0223).
Thus a simple surprise trigger reduces, but does not eliminate, the
adaptation/stability tradeoff. The shifted panel's minimum effective particle
count was 8.73. More realistic preference changes and posterior calibration
remain open before choosing a deployment rule.

The same fresh seed-8201 panel was extended to noisier choices and 15%
inconsistent answers. The 0.10 surprise trigger fired 81/800 and 79/800 times,
respectively, despite no preference change. At eight questions it increased
regret versus static EIG by +0.0286 (95% paired bootstrap [0.0138, 0.0467])
under noise and +0.0488 [0.0192, 0.0822] under inconsistency. Continuous 10%
refresh also lost under noise (+0.0197 [0.0078, 0.0331]). A single unlikely
answer cannot distinguish a change point from response-model misspecification.
The next detector should accumulate evidence over answers and model noise as a
separate latent variable; compare it against these fixed baselines on all four
conditions before considering deployment.

A 30-user, 1024-particle development pilot tested a stricter detector that
requires two surprising answers within three questions. It triggered much
less often but still raised regret under inconsistent answers: static EIG
0.0472 versus 0.0689 for the 0.10 threshold and 0.0700 for 0.20. Its shift
regret improved only modestly, from 0.0573 to 0.0436 or 0.0420. This arm did
not pass the robustness gate for a larger confirmatory run. Retain its raw
traces and move to a posterior with an explicit response-noise component.

An explicit 20% random-choice likelihood was piloted with 30 users and then
replicated on 100 fresh users (seed 8401, 4096 particles) across the same four
stress conditions. At eight questions, noise-aware EIG minus ordinary EIG
reward regret was +0.0026 (95% paired bootstrap [-0.0003, 0.0059]) for
matched users, -0.0281 [-0.0476, -0.0089] after a hidden shift, +0.0080
[0.0009, 0.0155] for higher-temperature noisy choices, and -0.0126
[-0.0213, -0.0046] for inconsistent choices. The 20% component exactly
matches the latter simulation by construction; its success there does not
establish that this noise rate can be known for real users. Adding a surprise
trigger to that robust posterior again cost stable users (+0.0043
[0.0008, 0.0083]) while helping after shifts (-0.0487 [-0.0715, -0.0272]).
The minimum effective particle count was 3.47 in some conditions. No fixed
setting passed all stress cases. Next infer noise rate jointly with preferences
and use a separate latent change hypothesis; validate uncertainty calibration.

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

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

A joint particle model then assigned each preference particle a response-noise
fraction from {0, 0.10, 0.20, 0.40}, updated both from answers, and used the
noise-aware predictive distribution for EIG. On 100 fresh paired users
(seed 8501, 4096 particles), inferred-noise EIG minus ordinary EIG eight-question
reward regret was -0.0018 (95% user-bootstrap [-0.0052, 0.0011]) for matched
users, -0.0352 [-0.0568, -0.0163] after a hidden shift, -0.0013
[-0.0073, 0.0049] for higher-temperature noise, and -0.0120
[-0.0211, -0.0036] for inconsistent answers. The optional surprise refresh
helped the shift panel further but lost its advantage on inconsistent answers.
The inferred noise fraction averaged 0.109 even for matched users (true
random-choice fraction zero), and effective particle count fell to 2.39 in
one condition. Treat the improvement as a decision result, not calibrated
recovery of a user's noise parameter. Next compare alternative noise priors
and resampling before selecting a posterior for deployment.

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

## Balanced numeric proxy prompt pilot and distinct-menu replication (2026-10-01)

The 16-case balanced development pilot (CURC 33221953) compared baseline,
full-menu numeric scores, scores only, and exact letter advice on a 1.5B
Qwen model. It used 16 users but reused some menus across users, so its
user-bootstrap interval does not capture menu dependence. Results are saved
in `outputs/preference_program/results/numeric_comparison_pilot_v10`.

A fresh, more informative replication (CURC 33222831) used 32 distinct users
and 32 distinct menus, balanced to eight A/B/C/D optima. Each case was paired
across four arms; there were 128 generations and zero parse failures. Mean
normalized regret was 0.517 baseline, 0.439 with numeric scores in the full
menu, 0.220 with scores only, and 0.039 with exact letter advice. The paired
scores-only minus full-menu regret difference was -0.219 (user-bootstrap
95% CI [-0.402, -0.035]); exact advice matched the gold action in 29/32
cases. Baseline chose C on all 32 cases; the full-menu score prompt chose C
on 28/32, and the scores-only prompt chose A on 28/32. Thus scores-only
improves average regret on this balanced sample but does not establish
reliable numeric comparison. Exact advice missed three A-optimal cases,
choosing B in each. Raw prompts, completions, gold utilities, per-record
metrics, logs, and the analysis are in
`outputs/preference_program/results/numeric_comparison_replication_v11`.

The next useful mechanism check should randomize action labels and numeric
score order on the same paired menus, then compare the model with a trivial
argmax parser and the exact-advice ceiling. This tests whether the observed
A/C defaults follow presentation order or meaning. Prompt optimization
should operate on separate development menus and be scored on fresh menus.

## Action-label rotation mechanism probe (2026-10-01)

CURC job 33234967 rotated A/B/C/D labels four ways for each of the 32
validated user-menu cases while keeping the underlying utility vector fixed.
A minimal numeric-score prompt and a minimal exact-letter-advice prompt were
paired for every rotation (256 generations, zero parse failures). With scores,
the 1.5B Qwen model chose A 97 times and D 31 times; it never chose B or C.
It matched the maximum numerical score in 56/128 rotations (43.75%,
case-bootstrap 95% CI [39.84%, 46.88%]) and had mean normalized regret 0.318.
Exact letter advice was followed in 128/128 rotations and had zero regret.
The score arm did respond to some score configurations (D was correct in
25/32 D-optimal rotations), but it did not perform reliable four-way numeric
comparison. This reconciles the earlier scores-only regret gain with the
strong answer-letter bias. The paired design reuses each underlying case in
four rotations, so the interval clusters by the 32 underlying cases. Raw
prompts, generations, metrics, and GPU logs are in
`outputs/preference_program/results/label_rotation_probe_v12`.

The next prompt-optimization question is whether a different prompt can make
this model select B and C when their scores are highest on held-out cases.
Compare a small set of prompts at an equal generation budget, selecting on
one set of users and evaluating on fresh users and menus. Include an exact
argmax parser as a deterministic reference. Only then consider broader
TextGrad/GEPA search or fine tuning for numerical tool output.

## Initial numeric prompt development search (2026-10-01)

CURC job 33244188 evaluated four fixed numeric-score prompt variants on 16
new development users and distinct menus with four label rotations each
(256 generations). The held-out pool (seed 10001, 32 different users and
menus) was CPU-prepared before development results were inspected and has
not been evaluated. The baseline scores-only arm had 0/64 parse failures,
24/64 correct, and mean normalized regret 0.322. A code-style argmax prompt
also parsed 64/64 but was worse: 20/64 correct and regret 0.391. The table
prompt produced 62/64 parse failures, and the pairwise prompt 64/64, because
they elicited explanations that the one-letter deployment parser rejects.
Their valid-only regret must not be used to select them. No alternative
outperformed the baseline under the strict output contract, so the held-out
run was not submitted. This pilot is exploratory; the format eligibility
rule was documented after inspecting its traces. Results and raw completions
are in `outputs/preference_program/results/numeric_prompt_dev_v13`.

Next, revise the output contract and parser as separate arms, or use a
prompt optimizer with a metric that penalizes parse failures. A new
development set should be used for prompt selection; retain seed 10001 as
an untouched held-out check once a viable candidate is found.

## Frozen table parser, held-out evaluation (2026-10-01)

Reinspection of the numeric prompt development pilot showed that 62/64
vertical-table outputs followed one exact phrase, `The best action is: [letter]`,
while two were single letters. An exact-phrase parser recovered all 64 choices
without changing the original raw generations or strict-parser results.
Recovered table regret was 0.143 versus 0.322 for scores-only on the 16
cases (paired case-bootstrap difference -0.179, 95% CI [-0.244, -0.111]).
This was a post hoc development discovery; the parser and candidate were
committed before the held-out run.

The sealed seed-10001 pool then supplied 32 new users and menus, each with
four label rotations. CURC job 33246206 generated 256 paired outputs for
scores-only and vertical-table prompts. The frozen parser recovered 128/128
table choices, comprising 125 exact table phrases and three single letters;
the original strict one-letter parser recorded 125 failures. Scores-only
matched the gold action in 51/128 and had mean normalized regret 0.384.
Table plus parser matched 79/128 and had regret 0.198. The paired table-minus-
scores-only regret difference was -0.186 (case-cluster bootstrap 95% CI
[-0.224, -0.148]). Table outputs selected A 49 times, C 45, D 34, and B
zero, so label bias remains. The result supports a better prompt-plus-parser
configuration on held-out synthetic cases; it does not establish reliable
four-way comparison. Raw generations, original parse failures, reparsed
choices, metrics, and GPU logs are in
`outputs/preference_program/results/numeric_prompt_holdout_v14`.

The next mechanism study should target B-optimal cases with label/order
randomization and controlled score gaps. Assess whether small prompt changes,
constrained letter decoding, or a deterministic argmax tool can remove the
remaining letter bias, using new development users before another holdout.

## Table row-order mechanism probe (2026-10-01)

CURC job 33247354 used 16 new users and distinct menus. For each menu, it
crossed four cyclic action-label assignments with four cyclic table-row
orders, yielding 256 generations. The frozen exact table-phrase parser
recovered all choices (252 fixed phrases and four single letters); the
original strict single-letter parser rejected 252. The model chose A 223
times, C 20, D 13, and B zero. B was gold in 64 cases and occupied every
row position equally, so its absence is a letter bias, not only a second-row
bias. The canonical A/B/C/D row order yielded 41/64 correct and mean
normalized regret 0.200. Each rotated row order yielded 16/64 correct and
regret 0.467. The within-case canonical-minus-rotated accuracy difference
was +0.391 (case-bootstrap 95% CI [+0.297, +0.469]); regret difference was
-0.267 (95% CI [-0.324, -0.206]). This shows a strong interaction between
table formatting and output defaults, even though the same utilities and
optimal action labels were preserved. Raw prompts, generations, parse
outcomes, per-case metrics and GPU logs are in
`outputs/preference_program/results/table_row_order_probe_v15`.

Next work should test representation channels that make utility comparison
mechanical, such as a deterministic argmax tool or constrained scoring of
candidate letters, against this prompt-plus-parser arm. It should also test
whether the model can handle B-optimal cases when B is supplied as explicit
proxy advice; prior exact-advice probes suggest it can. Do not infer a
four-way numerical reasoning capacity from the table result alone.

## Constrained first-letter scoring probe (2026-10-01)

A four-record GPU smoke (CURC 33260360) verified that Qwen 1.5B tokenizes
A/B/C/D as one token each (IDs 32/33/34/35), performed real CUDA forward
passes, and saved raw candidate logits and greedy generations. A bounded
follow-up (CURC 33260789) used 16 new user-menu cases with four label
rotations and two score formats, 128 paired prompts. Forced decoding ranked
the logits of A/B/C/D at the first assistant token; the greedy arm generated
normally, with the previously frozen table-phrase parser applied afterward.
These are different decoding conditions when the greedy table answer begins
with words rather than a letter.

On scores-only prompts, forced and greedy choices were identical in all 64
cases: 27/64 correct, mean normalized regret 0.342, A chosen 52 times and D
12, never B or C. On table prompts, forced scoring chose only C or D,
32/64 correct, regret 0.249. Greedy table generation plus parser achieved
44/64 correct, regret 0.163. The paired forced-minus-greedy table regret
was +0.086 (case-bootstrap 95% CI [+0.016, +0.148]). Constraining the first
token to four letters does not repair numerical reasoning and can discard
the table prompt's useful behavior. The deterministic argmax of supplied
utilities remains the computational reference; the next system study should
measure actual agent tool selection and adherence, rather than another
letter-scoring prompt. Full logits, probabilities, generations, per-user
results and GPU logs are in
`outputs/preference_program/results/letter_scoring_full_v17`.

## Controlled proxy tool-choice pilot (2026-10-01)

A three-record smoke (CURC 33272566) verified a textual tool-choice protocol:
`TOOL` from the model invokes a deterministic exact-argmax proxy in the
harness, which then returns a recommended action for a final answer. This
is controlled text routing, not native API function calling. The paired
panel (CURC 33273057) used 16 new users/menus with four label rotations,
64 decisions per arm and zero parse failures. The direct score prompt chose
A in all 64 cases, matched the gold action 16/64 times, and had mean
normalized regret 0.514. The optional-tool arm requested the proxy in
64/64, followed its exact recommendation in 55/64, and had regret 0.094.
Every optional-arm miss involved a C recommendation being answered as A.
The forced-tool-result arm followed advice in 64/64 and had zero regret.
The optional-minus-direct paired regret difference was -0.421 (case-cluster
95% CI [-0.487, -0.356]); forced-minus-optional was -0.094 (95% CI
[-0.146, -0.045]). The model can choose the tool route in this explicit
protocol, but the handoff prompt still loses some C advice. Tool selection
and adherence are distinct failure surfaces. Raw first turns, tool results,
final generations, prompts, per-user metrics and GPU logs are in
`outputs/preference_program/results/proxy_tool_choice_full_v19`.

Next, hold the initial `TOOL` request fixed and vary only the return channel:
inline text, an isolated tool-role message if supported by the chat template,
and a compact final instruction. Then substitute the learned RL proxy for
the oracle to measure the decision-quality cost of imperfect advice.

## Proxy result handoff comparison (2026-10-02)

The paired handoff experiment reused the 64 voluntary `TOOL` requests from
that panel and held exact proxy advice fixed. One 35 GB H200 MIG slice
generated 192 final responses in 40 seconds (CURC 33318999). The original
single concatenated prompt again followed advice in 55/64 cases; all nine
misses changed a recommended C to A. A multi-message assistant `TOOL` turn
followed by a compact user result, and an assistant `TOOL` turn followed by a
Qwen-rendered tool-role result, each followed advice in 64/64 cases. All
responses parsed. Both multi-message formats reduced mean normalized regret
from 0.0937 to zero; paired case-cluster bootstrap intervals for the regret
difference were [-0.1444, -0.0440] and [-0.1460, -0.0446] respectively.

The interventions changed both role structure and wording, so this identifies
a handoff-package effect rather than the separate contribution of role tokens.
Qwen's rendered tool role uses a `<tool_response>` wrapper; this controlled
transcript is not yet a native API tool-call evaluation. Raw messages,
rendered prompts, generations, and scoring are preserved under
`outputs/preference_program/results/proxy_handoff_full_v21`.

Next, replace the exact proxy's advice with the frozen learned RL proxy on
the same cases while keeping the successful handoff format fixed. Track
tool-call rate, advice adherence, decision regret, and the policy's own
regret separately. This separates loss from proxy approximation from loss
in the LLM handoff. Vary role structure and wording factorially later if
their individual contributions matter.

The frozen policy's action support is itself a limiting mechanism. A CPU
audit of the independent seed-9001 replication (200 users, 20 menus each)
found that the reward oracle chose actions 0, 1, 2, 3 on 2372, 337, 720,
and 571 decisions respectively. The learned RL policy chose action 0 on
3270 decisions and action 2 on 730; it never chose 1 or 3. When 1 was
optimal, its conditional mean normalized regret was 0.3393 (user-cluster
95% interval [0.2851, 0.3892]); when 3 was optimal, regret was 0.2763
[0.2500, 0.3028]. By contrast, regret was 0.0058 when 0 was optimal and
0.0303 when 2 was optimal. This is an observed checkpoint behavior, not an
architectural impossibility: its output layer has four actions. The balanced
64-case handoff panel deliberately raises the incidence of the missing
optimal actions from 908/4000 to 32/64, explaining much of its 50% policy
accuracy. Full conditional counts are in
`outputs/preference_program/results/proxy_action_support_audit_v22`.

An informative policy follow-up is a CPU-only action-balanced imitation arm
initialized from the frozen checkpoint, with identical held-out users and
menus. Compare action support, oracle agreement, and regret by gold action,
then test whether any gain survives ordinary-prior weighting. Avoid judging
it solely on the balanced diagnostic panel.

A CPU-only policy probe initialized from the frozen PPO checkpoint and used
oracle action labels to test that hypothesis. Twenty balanced imitation
updates changed weights but did not recover actions 1 or 3 on the 1000-decision
development panel; mean normalized regret rose from 0.1122 to 0.1381. At
200 updates, pure balancing recovered all four actions but still had regret
0.1390. A 50/50 mix of balanced and ordinary-prior training examples also
recovered all actions and lowered development regret to 0.0936. These two
updated policies were then frozen for one paired evaluation on a new seed
(9401; 200 users, 20 menus each). On its 4000 decisions, the original policy
had regret 0.1168 and 70.2% oracle action agreement. Pure balancing reached
0.1584 regret (paired difference +0.0415, user-cluster 95% interval
[+0.0148, +0.0678]) and 69.1% agreement. The mixed policy reached 0.1194
regret (difference +0.0026 [-0.0174, +0.0221]) and 73.0% agreement. Thus
restoring action support and increasing action agreement did not establish a
decision-utility gain under the ordinary user prior. This also shows why the
development result was insufficient to select the mixed policy. Per-decision
data, bootstrap summaries, training histories, changed checkpoints, and
timestamps are preserved under `action_balanced_proxy_cpu_pilot_v23`,
`action_balanced_proxy_cpu_probe_v24`, `action_mixed_proxy_cpu_probe_v25`,
and `proxy_imitation_fresh_eval_v26` within `outputs/preference_program/results`.

The next policy mechanism should optimize the *size* of decision loss on the
ordinary prior, while explicitly checking rare gold actions. A cost-sensitive
imitation objective or direct expected-utility surrogate is more aligned with
regret than class-balanced cross entropy. Keep the frozen PPO policy as the
reference and use a new development seed before another fresh confirmatory
panel.

A second CPU-only continuation study compared full-information expected
utility learning with ordinary-prior imitation, both starting from the
original frozen checkpoint. After 200 updates on identically seeded natural
training examples, the 1000-decision development panel showed regret 0.1033
for expected utility and 0.0897 for imitation, versus 0.1122 for the frozen
policy. Both candidates were frozen before a new paired seed-9501 evaluation
(200 users × 20 menus). There, expected utility reached mean normalized
regret 0.1051 versus 0.1122 for the frozen policy (paired user-cluster
difference -0.0071, 95% interval [-0.0088, -0.0053]), but still selected only
actions 0 and 2. Natural imitation reached 0.0961 (difference -0.0161
[-0.0270, -0.0056]), selected all four actions, and raised oracle action
agreement from 71.9% to 76.1%. The natural-minus-expected-utility regret
difference was -0.0090 [-0.0200, +0.0015], so the apparent advantage of
imitation over that surrogate is uncertain. The larger conclusion is that
ordinary-prior continuation improved regret on this synthetic held-out panel
while pure class balancing harmed it. Preserve both original and new
checkpoints as separate experimental arms. Training reports and the full
paired evaluation are under `utility_proxy_cpu_probe_v27`,
`natural_imitation_cpu_probe_v27`, and `proxy_utility_fresh_eval_v28` in
`outputs/preference_program/results`.

For the 16-case, four-label-rotation handoff panel, the ordinary-prior
imitation checkpoint was converted to a CPU-prepared advice manifest before
any new LLM evaluation. Its advice was optimal in 40/64 rotations versus
32/64 for the original PPO proxy, and proxy-only mean normalized regret fell
from 0.1993 to 0.1346. Advice changed on two underlying cases (eight
rotations); the case-cluster interval for candidate-minus-original regret
was [-0.1668, 0.0000]. This diagnostic panel was already used to identify
the original policy's weakness, so it is not independent confirmation of the
new checkpoint. The manifest and paired audit are preserved as
`manifests/natural_proxy_handoff_v29.jsonl` and
`results/proxy_advice_balanced_panel_v29` under `outputs/preference_program`.

A new preference-input ablation (seed 9601, 200 users × 20 menus) compared
true profiles, cyclically swapped profiles, and a fixed generic profile for
the frozen and ordinary-prior imitation policies. Decisions were scored
against each user's true reward in every condition. Frozen-policy regret
was 0.1072 with true profiles, 0.2162 with swapped profiles, and 0.1567 with
generic profiles. For the updated policy it was 0.0805, 0.2212, and 0.1553.
Swapping increased regret by 0.1090 for the frozen policy (paired user-cluster
95% interval [0.0903, 0.1283]) and 0.1407 for the updated policy
[0.1196, 0.1624]. Swapping changed 1150/4000 and 1448/4000 actions respectively.
This supplies causal input-ablation evidence that these explicit proxies
benefit from correct user preferences. It does not test the LLM's inference
of those preferences. Raw supplied and true profiles, actions, and scores
are preserved under `outputs/preference_program/results/proxy_preference_use_v30`.

A profile-noise sensitivity panel (seed 9701, 200 users × 20 menus) adds
Gaussian errors with sigma 0.25 and 0.75 to logit(gamma), log(alpha), and
log(lambda-1), using a common error direction per user across noise levels.
All decisions are scored against the original true profile. Frozen proxy
regret was 0.0996 with true inputs, 0.1089 at sigma 0.25, and 0.1571 at
sigma 0.75. The improved proxy reached 0.0776, 0.0946, and 0.1578 respectively.
Exact reward scoring reached 0, 0.0208, and 0.0936. Thus preference input
error can erase the improved policy's advantage, and creates decision loss
even for exact computation. These noise levels are sensitivity probes and
are not calibrated to an elicitor's actual error distribution. Next measure
that error empirically and compare point estimates with posterior decision
integration. The 48000 raw decisions, supplied profiles, and paired intervals
are saved under `outputs/preference_program/results/proxy_profile_noise_v31`.

## Actual discovery-to-decision bridge (v32)

Reanalyzed the existing 200-user discovery replication (seed 8001), preserving
its 12 held-out target menus, acquisition arms, and question budgets. Passed
saved posterior-mean preference estimates to the frozen and natural-imitation
policies and to exact reward scoring; compared with saved posterior-integrated
reward decisions. Saved 115,200 decision rows and 12,800 per-user rows.

After eight EIG questions, normalized regret was 0.1080 for the frozen policy,
0.0803 for natural imitation, 0.0196 for point-estimate exact scoring, and 0.0189
for posterior-integrated scoring. The natural-minus-frozen difference was
-0.0277 (paired user bootstrap 95% interval [-0.0364, -0.0193]). All four
acquisition arms at budget eight favored natural imitation over the frozen
policy. At budget zero natural imitation was worse by 0.0167
([0.0007, 0.0321]). Thus the training gain depends on receiving informative
preferences; merely deploying the updated policy with an uninformative profile
is insufficient.

The remaining policy approximation gap (~0.0614 regret for EIG at eight
questions) is much larger than the point-estimate versus posterior-integration
gap (~0.0007). In this matched simulator, improving proxy decision fidelity or
using exact reward scoring has higher priority than adding posterior complexity
alone. This is a reanalysis of existing analytic discovery data, not independent
replication or evidence of conversational LLM preference inference. Posterior
reference actions were not retained; comparisons use their saved user regrets.
Validation checks row counts and bounded regrets; the budget-zero exact scorer
reproduces the saved posterior regret exactly. Artifacts:
`outputs/preference_program/results/discovery_proxy_bridge_v32`.

## Paired discovery loss measurement (v33)

A frozen measurement protocol reuses v32's analytic EIG panel to compare
natural-imitation decisions against exact scoring of the identical inferred
profile at budgets 0/2/4/8. The excess regret is respectively
0.0401/0.0547/0.0555/0.0607; all paired user-bootstrap intervals exclude zero.
At budget eight, the interval is [0.0453, 0.0778], compared with a much smaller
point-versus-posterior exact gap of 0.00069 [0.00008, 0.00144]. Eight versus zero
questions reduces regret by 0.0909 for natural imitation and 0.1115 for exact
point scoring. Better discovery helps, but the current policy leaves much of
that benefit unrealized. These are matched decision contrasts, not additive
causal error components or new independent data. Exact reward remains the
primary computational control; no GPU scaling is justified by this reanalysis.
Artifacts and protocol: `results/discovery_loss_attribution_v33` and
`manifests/discovery_loss_attribution_v33.json` under `outputs/preference_program`.
Registered LLM smoke job 33320892 was reconciled live as pending for Priority,
with only the remote manifest and ledger present. Its 17-record gate remains
in place, and no duplicate submission was made.

## Posterior replay and learned handoff gate (v34 / v22)

Replayed all 6400 recorded discovery answers using deterministic seeded particle
priors, without new response sampling. Retained 38400 paired point/posterior
actions, utility vectors, payoff tensors, probabilities, and feasibility masks.
All 3200 saved posterior regrets and inferred profiles reproduced exactly
(maximum error zero). EIG at eight questions disagrees on 21/2400 decisions;
this yields the small 0.000686 average point-minus-posterior regret difference.
The replay closes the missing-action custody gap; it is not independent data.
Source hashes and raw decisions are in `results/discovery_decision_replay_v34`.

Job 33320892 completed its 17-record learned-advice smoke in 52 seconds. All
responses parsed and copied the policy recommendation, including the single
suboptimal advice item (16/17 optimal). Raw messages, generations, receipt,
logs, and telemetry were recovered. This supports handoff fidelity but does
not show that the LLM detects proxy mistakes. Framework GPU allocation was
1.15–1.19GB with completed generations; MIG utilization remains unavailable.
Loading took ~29 seconds and short-answer generation ~2 seconds. The approved
64-case gate was submitted as job 33340769, same frozen source and manifest,
separate `full` output, one 35GB H200 MIG and five-minute cap. Allocation check
showed one existing L40 job; it was preserved. No new research direction or
recurring schedule was introduced.

## Full learned-advice handoff outcome

Job 33340769 completed in 33 seconds and produced 64/64 parsed responses on
16 underlying cases with four label rotations. The LLM copied learned advice
64/64, including all 32 suboptimal advice items: zero wrong recommendations
were corrected. Learned-proxy and learned-proxy-plus-LLM regret both equal
0.199293; paired exact-reward advice yields zero regret and 64/64 optimal
choices. Independent utility-vector scoring reproduces every saved regret
(maximum discrepancy below 1e-10); raw messages, rendered prompts, wrong-advice
traces, artifact hashes, receipt, and GPU evidence are preserved. Framework
allocation stayed 1.15–1.19GB; loading took 14.4s and generation ~3s. MIG device
utilization is unavailable, so utilization efficiency is not established.

The fixed assistant TOOL request and explicit copy instruction make this a
handoff-fidelity experiment. It neither measures preference recovery nor
voluntary verification of erroneous advice. The completed registered gate
supports reliable transport of proxy decisions, while decision quality remains
limited by the proxy. No new study or scaling follows automatically. The next
scientific intervention requires choosing whether to evaluate error verification
or the already-prepared improved proxy advice; no new calls were launched.

## Approved staged continuation (v35 / v36)

Patrick approved improved-proxy integration first, followed by a distinct
faulty-advice verification study. Job33360565 runs the frozen v29 natural
imitation advice on the unchanged v22 handoff code and64 diagnostic rotations,
using one35GB H200 MIG/five minutes. Original and exact controls already exist;
no duplicate jobs were active at submission. This reused panel is an integration
check, not fresh validation of proxy training.

Verification v36 is separately registered before any LLM outcomes. CPU-only
preparation froze320 records:16 fresh seeded user/menu cases, four label
rotations, and five arms (no advice, correct/wrong fallible advice under neutral
or verification instructions). All arms receive identical visible scores;
incorrect advice is the next-best distinct feasible action. Rounded ties are
rejected before generation. Exact argmax is the deterministic control. Primary
contrast is verification-minus-neutral regret under wrong advice, clustered by
underlying case; diagnostics include corrections, spoiled correct advice, parse
failures, labels, and error severity. No copy command, TOOL request, or promise
of an exact tool remains. This measures visible-score verification, not latent
preference recovery. The20-response first-case smoke is gated on stage1
validation; no verification GPU call has been launched. Resource request must
stay within one35GB MIG/five minutes unless separately approved. Protocol,
manifest, and hashes are in `outputs/preference_program/manifests`.

## Stage1 timeout reconciliation

Job33360565 is TIMEOUT, not success: top-level0:0 coexists with batch
CANCELLED0:15 and elapsed5:24. It reached model_load_start at17:41:56 UTC,
then no model_loaded/generation event before time-limit termination about264s
later. No response file or receipt exists. Only the tiny CUDA preflight is
verified; whole-GPU memory samples are not attributable to this MIG process.
The existing trace cannot distinguish cache I/O, import, quantization, or GPU
initialization stalls. No proxy outcome can be inferred. Failed logs and audit
are retained in `results/improved_proxy_handoff_failed_v35`; stage2 GPU gate
remains closed. No retry or allocation expansion was submitted.

Concrete recovery proposal within the existing cap: first CPU-only offline
snapshot inventory, shard read/hash and tokenizer checks with timings; then an
immutable diagnostic snapshot with import/model/tokenizer timestamps and a90s
startup timeout that emits a Python stack trace. A separate bounded GPU smoke
must verify model memory and generation before retrying64 records. Keep the
same one35GB MIG/five-minute maximum and separate all receipts/output roots.
A longer allocation is not justified by an unlocalized startup stall.

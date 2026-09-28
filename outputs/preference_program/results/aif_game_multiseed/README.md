# Active-inference game replication

**CURC job:** 33063272 · **source revision:** 98ff3f0f4cf91e8bcb4de066bb46016476594350 · **metric recomputation:** a7a48f87d3a48a103093f5ac32a34063609adf46 · **paired sample:** 250 users per arm across five seeds

This replication compared adaptive expected information gain (EIG), random,
decision-impact, and AIF-inspired epistemic weights 20%, 50%, and 80%. Each of
the five synthetic-user panels had 50 users; all six arms shared users and
seeds. Each user answered eight questions. Per-arm intervals use percentile
bootstraps; paired contrasts use the 250 matched user records.

## Results

| Arm | Total parameter error (mean, 95% CI) | Spearman action alignment (mean, 95% CI) | Relative decision regret (mean, 95% CI) |
| --- | --- | --- | --- |
| Adaptive EIG | 0.359 (0.340–0.378) | 0.900 (0.882–0.916) | 0.01189 (0.01124–0.01254) |
| AIF 20% | 0.371 (0.350–0.392) | 0.893 (0.871–0.911) | 0.01180 (0.01113–0.01245) |
| AIF 50% | 0.368 (0.347–0.389) | 0.902 (0.882–0.920) | 0.01177 (0.01110–0.01242) |
| AIF 80% | 0.364 (0.344–0.384) | 0.903 (0.884–0.919) | 0.01185 (0.01118–0.01250) |
| Decision impact | 0.374 (0.353–0.395) | 0.882 (0.861–0.901) | 0.01176 (0.01109–0.01241) |
| Random | 0.375 (0.356–0.396) | 0.886 (0.865–0.904) | 0.01180 (0.01114–0.01244) |

### Paired differences from adaptive EIG

- **AIF 50%:** parameter error is higher by 0.0088 (95% CI 0.0014–0.0175),
  while action alignment is similar (+0.0025, CI −0.0098–0.0142). Relative
  regret is lower by 0.000121 (CI −0.000235 to −0.000029), a small but
  consistent decision-quality improvement in this panel.
- **AIF 80%:** parameter error is higher by 0.0050 (CI −0.0006–0.0117),
  alignment is similar (+0.0030, CI −0.0086–0.0142), and relative regret is
  lower by 0.000043 (CI −0.000144–0.000041), which is not distinguishable
  from zero.
- **Decision impact:** relative regret is lower by 0.000130 (CI
  −0.000244–−0.000037), but parameter error is higher by 0.0148 (CI
  0.0068–0.0240) and action alignment is lower by 0.0175 (CI −0.0308–−0.0053).
- AIF 50% also improves both parameter error and alignment over random
  (paired differences −0.0077 and +0.0163, respectively; both CIs exclude
  zero). Its regret difference from random is uncertain.
- Every arm has zero quality-floor violations.

**Interpretation:** AIF 50% is a promising game-domain balance: relative
regret improves over EIG and recovery/alignment improve over random, with a
parameter-recovery cost compared with EIG. It is not a universal winner; the
stock and supply-chain results show clear domain differences. The cross-domain
follow-up is in [aif_cross_domain](../aif_cross_domain/).

## Validity and correction

- 1,500/1,500 per-user records; exactly 250 per arm and 300 per seed.
- Every record has eight choices and eight acquisition-diagnostic entries.
- Zero failed campaign tasks, malformed records, or quality-floor violations.
- AIF acquisition traces contain 6,000 round-level diagnostics.
- The initial CURC bundle had legacy Monte Carlo regret values despite its
  updated manifest label. Per-user action alignment and decision regret were
  recomputed from saved true types, inferred actions, and initial states using
  a multistart SLSQP optimum on the long-only simplex and 48-point
  Gauss-Hermite quadrature. Preference-recovery errors are unchanged.
- Parse failures are not applicable: these simulations use synthetic binary
  choices and do not call an LLM response parser.

Full per-user records, the manifest, and CSV analyses are under
[outputs/game](outputs/game).

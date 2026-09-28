# Active-inference game pilot

**CURC job:** 33062243 · **code:** c052fce6b0b139681bb3e2ba3298c497fdc4293f · **regret evaluator:** 741fd287e0b442238cc93588a4a6e7aa9739daa1

This paired pilot compared adaptive expected information gain (EIG), random
questions, the decision-impact proxy, and three mixtures of epistemic mutual
information with pragmatic expected value of sample information. It used 50
synthetic users from seed 42 per arm, with the same panel across arms and eight
questions per user.

## Readout

There is no clear recovery-error difference between an AIF blend and EIG in
this sample. Mean total parameter error was 0.365 for EIG, 0.366 for AIF 50%,
and 0.367 for AIF 80%. The paired AIF 80% minus EIG difference was 0.0012
(95% bootstrap CI −0.0071 to 0.0087). Mean action alignment was 0.904 for EIG
and 0.912 for AIF 80%; the paired difference was 0.008 (95% CI −0.008 to
0.024).

Mean relative decision regret was 0.01177 for EIG and 0.01180 for AIF 80%.
The paired difference was 0.0000287 (95% CI 0.0000129 to 0.0000449), favoring
EIG by a very small amount. AIF 20% and AIF 50% regret differences from EIG
were not distinguishable from zero in this sample. Quality-floor violations
were zero in every arm.

These are exploratory estimates from one synthetic-user seed, not evidence
that the small AIF 80% regret difference will generalize. The useful result is
that moderate AIF weighting (20–50%) appears competitive with EIG, while the
higher weight did not improve decision quality. The next comparison should
repeat EIG, AIF 50%, AIF 80%, decision-impact, and random over five common user
seeds before choosing a default.

## Validation and metric correction

- 300/300 expected per-user records; 50 per arm.
- Every record has eight choices and eight acquisition-diagnostic entries.
- Zero failed campaign tasks, malformed records, or quality-floor violations.
- Acquisition diagnostics record selected mutual information and normalized
  EVSI contributions for AIF arms.
- The first-pass decision-regret measure used the environment's
  certainty-equivalent action as if it were a true-type optimum and yielded
  near-zero regret. The records were recomputed with a multi-start SLSQP
  expected-utility optimum on the long-only simplex; both actions use the
  same true-type prospect utility and 48-point Gauss-Hermite quadrature.
  Corrected regret is positive for 246/300 recommendations.

Complete per-user JSON, the manifest, CSV summaries, and paired contrasts are
in [`game/`](game/). The summary script reports paired bootstrap intervals and
paired Cohen's d. No model text parser is involved in this synthetic-choice
experiment, so parse-failure rates are not applicable.

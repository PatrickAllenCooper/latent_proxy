# Active-inference game replication

**CURC job:** 33063272 · **source revision:** 98ff3f0f4cf91e8bcb4de066bb46016476594350 · **paired sample:** 250 users per arm across five seeds

This replication compares adaptive expected information gain (EIG), random,
decision-impact, and AIF-inspired epistemic weights 20%, 50%, and 80%. Each of
the five synthetic-user panels has 50 users; all six arms share those users
and seeds. Each user answered eight questions. Per-arm intervals use
percentile bootstraps; paired contrasts use the 250 matched user records.

## Results

| Arm | Total parameter error (mean, 95% CI) | Spearman action alignment (mean, 95% CI) | Relative decision regret (mean, 95% CI) |
| --- | --- | --- | --- |
| Adaptive EIG | 0.359 (0.340–0.378) | 0.900 (0.882–0.916) | 0.0000431 (0.0000057–0.0000971) |
| AIF 20% | 0.371 (0.350–0.392) | 0.893 (0.871–0.911) | 0.0000232 (0.0000034–0.0000608) |
| AIF 50% | 0.368 (0.347–0.389) | 0.902 (0.882–0.920) | 0.0000059 (0.0000034–0.0000088) |
| AIF 80% | 0.364 (0.344–0.384) | 0.903 (0.884–0.919) | 0.0000202 (0.0000067–0.0000445) |
| Decision impact | 0.374 (0.353–0.395) | 0.882 (0.861–0.901) | 0.0000051 (0.0000022–0.0000087) |
| Random | 0.375 (0.356–0.396) | 0.886 (0.865–0.904) | 0.0000237 (0.0000034–0.0000620) |

### Paired differences from adaptive EIG

- **AIF 50%:** parameter error is higher by 0.0088 (95% CI 0.0014–0.0175),
  while action alignment is similar (+0.0025, CI −0.0098–0.0142). Relative
  regret is lower by 0.0000371, but its interval reaches zero (CI
  −0.0000900–0.0000003). This is a promising decision-quality signal with a
  small, uncertain absolute magnitude, alongside worse parameter recovery.
- **AIF 80%:** parameter error is higher by 0.0050 (CI −0.0006–0.0117),
  alignment is similar (+0.0030, CI −0.0086–0.0142), and relative regret is
  lower by 0.0000229 (CI −0.0000718–0.0000059). None of those differences
  excludes zero.
- **Decision impact:** relative regret is lower by 0.0000380 (CI
  −0.0000909–−0.0000006), but parameter error is higher by 0.0148 (CI
  0.0068–0.0240) and action alignment is lower by 0.0175 (CI −0.0308–−0.0053).
- Every arm has zero quality-floor violations in these game scenarios.

**Interpretation:** no overall winner is established. AIF 50% is the leading
hybrid to carry forward because it has the strongest AIF mean regret and
competitive alignment, but EIG recovers parameters more accurately and the
paired regret interval narrowly includes zero. Decision impact exposes a
tradeoff: lower measured regret at the cost of worse recovery and alignment.
The next useful check is whether AIF 50% keeps its decision-quality signal in
stock and supply-chain tasks, with EIG and decision-impact as references.

## Validity

- 1,500/1,500 per-user records; exactly 250 per arm and 300 per seed.
- Every record contains eight choices and eight acquisition-trace entries.
- Zero failed tasks, malformed JSON, non-finite decision metrics, or
  quality-floor violations.
- AIF acquisition traces contain 6,000 round-level diagnostics.
- Decision regret uses the multistart expected-utility optimum on the
  long-only simplex, evaluated under the true synthetic type using
  48-point Gauss-Hermite quadrature.
- Parse failures are not applicable: these simulations use synthetic binary
  choices and do not call an LLM response parser.

Full records and CSV analyses are under [`outputs/game/`](outputs/game/).

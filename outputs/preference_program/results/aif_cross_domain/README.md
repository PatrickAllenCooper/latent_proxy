# Active-inference cross-domain follow-up

**CURC job:** 33065603 · **source revision:** 656c9885392e9e38c154ee8786d78b503490a188 · **metric/floor correction:** a7a48f87d3a48a103093f5ac32a34063609adf46 · **paired sample:** 250 users per arm in each domain, across five seeds

The experiment compared adaptive EIG, random, decision-impact, AIF 50%, and
AIF 80% in stock and supply-chain environments. Each domain used five shared
synthetic-user panels of 50 users per seed. Per-arm intervals use percentile
bootstraps; paired contrasts are over 250 matched users within each domain.

## Results

| Domain | Arm | Parameter error | Action alignment | Relative decision regret | Quality-floor violations |
| --- | --- | ---: | ---: | ---: | ---: |
| Stock | Adaptive EIG | 0.382 (0.359–0.406) | 0.995 (0.995–0.995) | 0.0000070 | 0/250 |
| Stock | AIF 50% | 0.401 (0.376–0.426) | 0.995 (0.995–0.995) | 0.0000070 | 0/250 |
| Stock | AIF 80% | 0.402 (0.377–0.427) | 0.995 (0.995–0.995) | 0.0000070 | 0/250 |
| Stock | Random | 0.393 (0.369–0.417) | 0.995 (0.995–0.995) | 0.0000070 | 0/250 |
| Supply chain | Adaptive EIG | 0.437 (0.412–0.462) | 0.711 (0.676–0.744) | 0.004316 | 0/250 |
| Supply chain | AIF 50% | 0.393 (0.371–0.415) | 0.781 (0.749–0.812) | 0.003838 | 0/250 |
| Supply chain | AIF 80% | 0.411 (0.387–0.436) | 0.747 (0.711–0.781) | 0.004213 | 0/250 |
| Supply chain | Random | 0.383 (0.361–0.404) | 0.801 (0.773–0.827) | 0.003661 | 0/250 |

Decision-impact is included in the per-user and CSV outputs; its supply-chain
means were 0.449 error, 0.663 alignment, and 0.004467 relative regret.

## Paired findings

- **Stock is at a decision ceiling:** every arm has the same mean alignment
  (0.9948). AIF 50% has higher parameter error than EIG by 0.0182 (95% CI
  0.0051–0.0313). The tiny relative-regret differences are practically
  negligible despite their narrow bootstrap intervals.
- **Supply chain: AIF 50% improves over EIG.** Parameter error decreases by
  0.0435 (95% CI −0.0653 to −0.0219), alignment rises by 0.0701
  (0.0333–0.1065), and relative regret falls by 0.000478
  (−0.000907 to −0.000071).
- **But AIF 50% does not beat random in supply chain.** Its parameter error is
  higher by 0.0105 (CI −0.0059 to 0.0273), alignment is lower by 0.0199
  (−0.0493 to 0.0085), and relative regret is higher by 0.000177
  (−0.000078 to 0.000451); these differences are uncertain. Random has the
  best point estimates.
- AIF 80% is worse than random on all three supply-chain means; its regret
  difference is clearer (higher by 0.000552, CI 0.000216–0.000936).
- Decision-impact is worse than random on supply-chain recovery, alignment,
  and regret.

**Interpretation:** AIF 50% transfers better than adaptive EIG into supply
chain, but random elicitation remains a strong baseline there. Stock provides
little decision discrimination. The next informative experiment is
return-normalized supply-chain elicitation: test whether the AIF 50% advantage
over EIG and its gap to random persist when preference inference uses the
scale-stable utility formulation.

## Validation and corrections

- 2,500/2,500 JSON records; 250 per arm/domain and 500 total per seed.
- Every record contains eight choices and eight per-round acquisition entries.
- Zero failed tasks, malformed records, non-finite decision metrics, or
  final quality-floor violations.
- Four initial quality-floor flags were exactly at the configured 70% supplier
  cap (0.7000000000000001). The comparison now uses a 1e-9 tolerance, and
  every saved record was rechecked: none exceeds the cap.
- The original run summary had legacy Monte Carlo regret values despite the
  updated manifest label. Per-user decision metrics were recomputed with a
  multistart SLSQP optimum on the long-only simplex and 48-point Gauss-Hermite
  quadrature. Preference-recovery errors are unchanged.
- Parse failures are not applicable: the study uses synthetic binary choices
  and no LLM response parser.

Full records, logs, manifest, and paired CSV analyses are in
[outputs/cross_domain](outputs/cross_domain).

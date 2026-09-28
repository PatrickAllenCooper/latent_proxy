# AIF supply-chain study with return-normalized utility

**CURC job:** 33078008
**Source revision:** `02fbb776c438a7449081e395bbb5397a28cd2c8f`
**Design:** five paired synthetic-user seeds (42–46), 50 users per seed and arm, eight questions per user. Arms are adaptive EIG (`active`), random, decision-impact, AIF 50% epistemic weight, and AIF 80% epistemic weight. The campaign uses return-normalized utility for recommendation evaluation.

## Validation

All 1,250 expected JSON records are present and parse. Each arm has 250 users, each seed has 250 records, all records have eight rounds, metric values are finite, and no quality-floor violations were recorded. Analysis contains 1,250 per-user rows, five arm summaries, and 42 paired contrasts. The simulated-user campaign has no language-model response parser, so parse failures are not applicable.

## Results

Relative to random, adaptive EIG reduced mean total preference error by 0.0239 (paired 95% CI −0.0349 to −0.0133) and relative decision regret by 0.00302 (CI −0.00588 to −0.00088). Its alignment change was small and uncertain (+0.00362, CI −0.00060 to +0.00888).

AIF 50% also improved over random on total error (−0.0169, CI −0.0279 to −0.0062), alignment (+0.00584, CI +0.00085 to +0.01160), and relative regret (−0.00252, CI −0.00445 to −0.00088). AIF 80% showed similar gains over random. However, neither AIF weighting showed a clear improvement over adaptive EIG: the paired AIF 50% minus EIG differences were +0.00699 total error (CI −0.00099 to +0.01508), +0.00222 alignment (CI −0.00416 to +0.00840), and +0.000505 relative regret (CI −0.00173 to +0.00329).

This supports active questioning over random in the return-normalized supply-chain setting, while the tested AIF blends do not yet establish added value over EIG. All conditions reached the eight-question cap, so this run does not measure stopping efficiency. The previous absolute-utility cross-domain study found a different ranking against random; utility scaling may change which questions are most useful. Treat the cross-utility difference as a follow-up hypothesis, not a causal conclusion from these separate studies.

## Files

- `outputs/supply_chain/supply_chain/`: per-user records.
- `outputs/supply_chain/analysis/`: per-user table, summaries, paired contrasts, and analysis manifest.
- `logs/`: CURC stdout/stderr for job 33078008.
- `validation.json`: machine-readable integrity checks.

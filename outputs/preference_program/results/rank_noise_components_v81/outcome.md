# v81: which correction produces the improvement?

## Frozen design

Source and analysis0fd8ba7 were pushed before submission. Thirty fixed fresh seeds40101–40130,disjoint from prior studies and profiling streams,cover all24finite orders. True answer noise is20%,preferences static. The2×2factorial varies posterior-update and query-planner assumptions independently between10%and20%,for AIF and EIG. AIF weight0.1,hazard0.1,prior,menus,budgets,kernels and Python3.12tie rules remain fixed. Regression tests reproduce saved v79actions and posterior values in all four cells. Common potential answers are paired across models/policies.

Two sequential15streamshards preserve the existing1CPU/256MiB,60-second scheduler/30-second outer/20-second check envelope. Shard0integrity was checked before shard1submission;effect analysis required all30streams and occurred only after both completed. No interim estimate changed the sample size or settings.

## Primary attribution: AIF transfer regret at8questions

Three primary contrasts were specified in advance:average updater effect,average planner effect,and interaction. Main effects average correction over both levels of the other component. Negative values improve regret. Bonferroni-adjusted98.333%paired stream-block percentile intervals address the three primary contrasts.

- average_update_effect: 0.0053; adjusted interval[-0.0060, 0.0168].
- average_planner_effect: -0.0338; adjusted interval[-0.0463, -0.0209].
- interaction: -0.0154; adjusted interval[-0.0341, 0.0028].

Planner correction is the supported contributor to lower transfer regret in this finite experiment. Belief-update correction's average effect and interaction remain uncertain;crossing zero does not prove either is absent. These are closed-loop component effects:changing the updater also changes later query histories. This does not identify a pure fixed-question mediation effect.

## Four-cell outcomes and conditional effects

Eight-question mean target and transfer regret are preserved for every cell and policy:

- update10_plan10, AIF: transfer0.1225, target0.0878.
- update10_plan10, EIG: transfer0.0840, target0.0821.
- update10_plan20, AIF: transfer0.0964, target0.0765.
- update10_plan20, EIG: transfer0.0840, target0.0821.
- update20_plan10, AIF: transfer0.1355, target0.0521.
- update20_plan10, EIG: transfer0.0762, target0.0713.
- update20_plan20, AIF: transfer0.0939, target0.0590.
- update20_plan20, EIG: transfer0.0791, target0.0766.

Conditional AIF effects below use secondary nominal95%intervals:

- update_at_plan10: 0.0130 [-0.0012, 0.0276].
- planner_at_update10: -0.0261 [-0.0383, -0.0141].
- update_at_plan20: -0.0024 [-0.0116, 0.0069].
- planner_at_update20: -0.0416 [-0.0556, -0.0282].
- joint_correction: -0.0285 [-0.0418, -0.0154].

Joint correction also narrows the AIF-minus-EIG transfer gap by-0.0236,secondary interval[-0.0397, -0.0085]. The remaining fully corrected gap is0.0148,secondary interval[0.0027, 0.0269];the planner fix does not establish transfer parity with EIG. AIF has lower immediate target regret in the fully corrected cell,showing the importance of the chosen objective. Target comparisons are secondary,not an alternative primary endpoint selected after viewing results.

## Posthoc query diagnostic

A separately labeled shared-history diagnostic compares paired planner models until their first differing query. With updater10%,AIF diverged in668/720trajectories;with updater20%,697/720. All first divergences were strict ranking reversals under the planner scores,beyond1e-12in both models. Thus these AIF planner changes are substantive question-value changes in the saved histories,not merely numerical-tie flips. The threshold was diagnostic only:the actual run's tie rule was not changed.

For EIG with updater10%,queries were identical across planner assumptions. With updater20%,357trajectories diverged first at numerical near-ties. Retain this caveat for EIG planner-cell outcome differences. This posthoc diagnostic is supplementary and does not replace the frozen primary analysis.

## Validation and limits

Independently replayed28,800raw final actions/scores,46,080noisy answers and full24weight posterior updates,720paired factorial effects,and all28contrasts/intervals. Verified the three-primary family and multiplicity adjustment,common potential responses,frozen source/protocol/kernel hashes,and local/remote artifact custody. Original records/scoring/AIF weights were not tuned or rescored. Application audit protocol-open events,phase timing and actual record progress were verified.

The30independent stream blocks each average all24enumerated orders;this is not720independent users. Intervals quantify finite synthetic response/menu-stream uncertainty,not human behavior. The experiment uses an exact executor,not an LLM or learnedRLproxy. Known20%response noise is supplied by design;learning unknown noise rates is not tested. The fixed0.1hazard is still a separate model mismatch on these static users.

## Resources

Jobs34866666 and34866690 completed0:0,using11and17allocated CPU-seconds and5.353/5.321consumed CPU-seconds. Scheduler peaks70,924/70,968KiB;instrumented elapsed5.210/6.886seconds. No GPUs,modelcalls,framework imports,ptrace or native-syscall claim. Data are below32MiBper shard. At most one project CPU job ran at a time;other projects were preserved. Local/remote final ledgers are saved. Cumulative diagnostic allocation:2,534CPU-seconds and254MIG-slice-seconds. No retry or further experiment was submitted.

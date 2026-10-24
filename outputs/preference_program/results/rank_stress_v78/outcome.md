# v78 noisy/changing preference study — completed

Frozen protocol/source commit e3a07fa preceded job34864483. Same24fully enumerated preference orders were repeated under three fresh menu/noise seeds; these are not72independent users. Random,EIG,decision-focused and the fixed0.1-weight AIF mixture were paired in clean,10%noise,20%noise and a separate noiseless preference-reversal condition. Static and10%hazard inference filters were crossed with every arm. All filters assumed10%response noise:20%noise deliberately tests misspecification. The fixed hazard is not estimated or perfectly matched to a deterministic change point.

## Prespecified primary result: transfer regret at8questions

The AIF advantage is conditional, not universal. Under preference reversal with hazard inference, AIF transfer regret was0.04167 versus0.13812 for EIG and0.13349 for decision-focused questioning. Paired AIF-minus-EIG was-0.09645 with exploratory95%order-cluster interval[-0.13040,-0.06443]; AIF-minus-decision-focused was-0.09182[-0.11728,-0.06713]. With a static filter under reversal, its advantage over decision-focused was uncertain:-0.00849[-0.07099,0.05363]. Correct change handling matters.

Under10%answer noise and static inference,AIF transfer regret0.02199 was lower than decision-focused0.08526; paired difference-0.06327[-0.09877,-0.03239]. Its difference versus EIG crossed zero:-0.01235[-0.03318,0.00617]. Under20%noise plus hazard inference,AIF was worse than EIG:0.15432 versus0.08719, difference+0.06713[0.01929,0.11150]. Its contrast with decision-focused was uncertain:-0.00193[-0.04167,0.03434]. Clean-condition AIF and EIG tied at zero transfer regret at8questions.

The primary endpoint is transfer, not immediate target performance. In the reversal/hazard condition, immediate target regret was0.05556 for AIF versus0for decision-focused; the transfer benefit does not imply dominance on every objective. Full curves and every condition are retained. Sixteen contrasts were prospectively specified; the intervals resample24order clusters after averaging seeds, are not multiplicity adjusted, and do not describe a human population.

## Integrity and measurement

All11,520endpoint records,18,432noisy answers/posterior updates,160aggregates and16contrasts/intervals were replayed. All source/protocol/result hashes match the frozen packet and remote custody. Runtime application-phase events and Python audit protocol-open events were observed. No ptrace,native-syscall claim,framework/modelcalls orGPUallocation occurred.

The simulation did not serialize final action labels. They were reconstructed independently from saved query traces/posteriors and scoring rules in reconstructed_actions.jsonl, explicitly labeled derived rather than raw. An initial local Python3.9 audit disagreed on near-tied choices because its floating-point summation differs. Python3.13 exact replay matched the pinned compute Python3.12.13 outcomes. Original outputs and scorer were not altered. Future studies should explicitly serialize actions and define tolerance-aware ties prospectively; do not claim runtime-independent reproducibility from this run.

## Resources

One acpu CPU,256MiB,60-second scheduler/30-second outer application/20-second check limits. Job completed0:0 in14allocated seconds,5.451consumed CPU-seconds,72,544KiB scheduler MaxRSS. Instrumented work was5.289seconds. Runtime internal peakRSS39,612KiB differs from the scheduler's step-level peak; retain both scopes. The full artifact set is under the32MiB data allowance. Cumulative diagnostics:2,480allocated CPU-seconds and254MIG-slice-seconds. Other projects were preserved; no automatic retry or new training/inference was submitted.

## Interpretation and next experiment

The finite synthetic study supports an information-foraging benefit for transfer, particularly when paired with a filter that permits preference changes. It rejects a universal AIF-wins claim under response noise. This is exact-posterior preference discovery with a deterministic reward executor, not LLM preference discovery, learned RL behavior, tool assimilation or human validity. The next informative comparison would separate noise-model misspecification from query policy on new response streams, keeping the AIF weight fixed; no further study is launched by this closeout.

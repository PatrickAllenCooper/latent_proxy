# v79: response-noise model versus query-policy attribution

## Frozen design and integrity

Protocol and source67589cf were pushed before job34865368. The environment always has20%answer flips and static latent preferences. Fresh seeds7101/8101/9101 do not overlap v78's4101/5101/6101. The same24enumerated orders are repeated across those three streams; this is not72independent users. Potential answers keyed by seed/order/question/pair are shared across conditions and policies. Four query policies are crossed with2posterior-update likelihoods(10%/20%),2planner likelihoods(10%/20%),and static/10%hazard filtering. The AIF weight0.1,menu construction,question budgets,action rule and Python3.12 tie semantics remain fixed.

This independently manipulates the assumed noise in belief updating and forward query valuation. These are closed-loop policy effects:changing belief updates also changes later adaptive queries. It is not a fixed-question causal decomposition. The hazard10 filter on static users is a separate deliberate model mismatch; matching the response likelihood alone does not make that entire filter correct. Static filtering is retained as a secondary control.

## Prespecified primary result, hazard10 filter

AIF-minus-EIG transfer regret at8questions is positive when AIF is worse. Values below are paired means with exploratory95%preference-order-cluster intervals.

- Both assume10%: 0.0525 [0.0042, 0.1003].
- Updater20%,planner10%: 0.0764 [0.0243, 0.1300].
- Updater10%,planner20%: 0.0278 [-0.0108, 0.0664].
- Both response-noise models match20%: 0.0093 [-0.0359, 0.0532].

Changes in that gap relative to both assuming10%:

- update_only: 0.0239 [-0.0301, 0.0752].
- planner_only: -0.0247 [-0.0644, 0.0143].
- joint_matching: -0.0432 [-0.0903, 0.0019].
- interaction: -0.0424 [-0.1034, 0.0177].

The historical hazard-filter deficit reproduces in direction on fresh streams. Joint response-noise matching narrows its point estimate from0.0525to0.0093. However,the change interval[-0.0903, 0.0019]crosses zero. Residual matched gap[-0.0359, 0.0532]also spans both signs:that does not establish equivalence or eliminate a material policy effect.

Updater correction alone did not rescue the point-estimate deficit; planner-only correction reduced it. The pattern suggests query valuation under the wrong response model contributes, but this panel does not decisively separate that explanation from noise, adaptive-history effects and the hazard-filter interaction. Do not conclude that all of the failure came from a10%likelihood assumption or that AIF's policy is exonerated.

Gap narrowing includes AIF transfer regret moving0.1412→0.1092and EIG moving0.0887→0.0999. Thus some relative-gap narrowing is EIG worsening,not only AIF improving. The AIF absolute-improvement interval crosses zero. In the static-filter control,the fully matched mean transfer regrets tie at0.0883;matching does not give AIF a demonstrated universal advantage. Equal averages do not mean identical decisions.

Random-policy queries were identical across all model assumptions and filters,providing an inference-only control. Under hazard filtering,its matched-likelihood regret improvement was approximately-0.0289,with interval[-0.0498,-0.0112]. Correcting the response model can help an unchanged-query system;the adaptive AIF result remains less certain.

## Validation and scope

Replayed all18,432potential-noise answers and24-component posterior updates,all11,520raw final action/score endpoints,160aggregates,and all32reported contrasts/intervals. Verified common random numbers,random-query invariance,frozen protocol/source and local/remote artifact hashes. Full posterior weights and final action labels are now saved,closing the previous action-serialization gap. Three local fixtures passed,including exact10%planner compatibility with the frozen v78algorithm. No outputs,scorer,tie rule or AIF weight were tuned after viewing this run.

Intervals average seeds within24order clusters and are exploratory/unadjusted for the contrast family. They do not support human-population claims. This is a finite synthetic preference-discovery experiment using an exact executor,not an LLM,learnedRL,tool-assimilation or general-user experiment. Prior negative/positive results and the complete fresh-stream outcomes are retained.

## Application instrumentation and resource accounting

Application phase timestamps,Python audit protocol-open events,full artifact hashes and observed output counts establish real CPU progress. No ptrace/native-syscall claim,framework imports,model calls or GPU work occurred. Job completed0:0 in11allocated CPU-seconds,4.723consumed CPU-seconds,65,952KiB scheduler MaxRSS;instrumented elapsed5.942seconds. Request remained1CPU/256MiB,60-second scheduler/30-second outer/20-second check. Artifacts remain below32MiB. Cumulative diagnostic allocation is2,491CPU-seconds and254MIG-slice-seconds. Other projects were preserved;no automatic retry was submitted.

## Next informative step

More preregistered fresh response streams would narrow the matched-versus-mismatched interaction estimate without tuning AIF. Keep posterior and planner noise separated and preserve the static-filter control. A later controlled change-detection comparison can examine why the hazard filter harms noisy static users. This closeout does not launch another experiment.

# v82: can the planner infer noise online?

## Frozen comparison

Source and analysis6f7439e were pushed before submission. Thirty fresh streams60101–60130 cover all24finite orders and true10%,20%,30%response noise. Fixed10%,online-estimated and oracle-rate planners share potential responses. AIF weight0.1,main belief-update likelihood10%,hazard0.1,kernels,menus,prior orders,eight-question budget and native Python3.12tie rules are fixed. Oracle means known response-noise rate only,not known preferences or an optimal transfer policy.

The online sidecar is an exact joint posterior over24unknown static orders and9candidate noise rates0%–40%in5point increments. Its fixed prior is epsilon=.05*K,K~Binomial(8,.25),mean10%. It resets per trajectory and updates only on the observed pair/answer. It does not see latent order,actual noise,flip flags,other users or future answers. Its mean is rounded12decimals and supplied before the next question. The main order-belief updater remains fixed10%;the sidecar does not replace that state.

The estimator assumes static preferences and a constant noise rate,correct for these environments. AIF's main hazard remains0.1as in previous studies. This is a certainty-equivalent plug-in estimator on existing AIF questions. There is no added noise-information objective,extra calibration probe,prior tuning or longer question budget. A shadow sidecar is logged in fixed/oracle modes but does not control them.

## Primary performance result at20%true noise

- fixed10: transfer regret0.1227; target regret0.0802.
- online: transfer regret0.1219; target regret0.0766.
- oracle: transfer regret0.1037; target regret0.0696.

Two prespecified primary contrasts use Bonferroni-adjusted97.5%paired stream-block intervals:

- online-minus-fixed10: -0.0008 [-0.0071, 0.0058].
- online-minus-oracle: 0.0182 [0.0073, 0.0296].

There is no clear transfer-regret improvement over the fixed10%planner in this panel;crossing zero is not equivalence. The online planner remains worse than the oracle-rate reference with the adjusted interval above zero. This implementation does not replace given noise calibration within the existing eight-question budget.

## What the estimator learned

Online mode after8answers:

- True10%: mean estimate10.10%, MAE2.35percentage points, nominal90%credible-interval empirical coverage100.0%.
- True20%: mean estimate11.49%, MAE8.51percentage points, nominal90%credible-interval empirical coverage100.0%.
- True30%: mean estimate12.70%, MAE17.30percentage points, nominal90%credible-interval empirical coverage0.0%.

The estimate moves with the true noise level but remains near the10%prior,substantially underestimating20%–30%noise. At30%true noise,the online sidecar's nominal90%interval covers truth in none of the720scenarios. These are conditional empirical coverage diagnostics,not a claim that Bayesian credible intervals guarantee90%coverage in every fixed-rate stratum.

The q=8estimate is available only after the eighth answer and would inform a ninth question;it was not used for the eighth query. Full before-query estimates and posterior marginals are retained. All10%fixed/oracle query paths match exactly,validating that control. Secondary10%/30%performance contrasts are saved without being promoted to the primary comparison.

The study does not establish that online noise inference is impossible. Prior concentration,the short budget,and the noise information supplied by the existing questions were not separately manipulated. It also does not test a planner that integrates noise uncertainty or actively seeks noise-identification questions. Any such extension is a new intervention.

## Validation

Replayed all51,840sidecar/main-state updates from past pairs/answers,32,400raw action/score records,and all contrast intervals. Verified before-answer timing,no need for true noise/preferences to reconstruct online estimates,common responses,all720fixed10/oracle10control trajectories,and frozen source/protocol/kernel/artifact custody. Five local fixtures passed,including unknown-noise interface,prior mean,history sensitivity,old-control reproduction and no future-answer access. No algorithm/scoring/AIF settings were tuned after outputs.

Three sequential10streamshards completed before effect analysis. Thirty stream blocks,each averaging all24orders,are the uncertainty units. This is not720independent users,nor human/LLM/learnedRLvalidation. Application audit protocol-open events,phase timestamps and observed artifacts establish CPU progress;there is no native-syscall or ptrace claim.

## Resources

Jobs34871021/34871032/34871063 completed0:0 using12/13/17allocated CPU-seconds,total42;consumed CPU4.782/4.776/6.665,total16.223seconds. Scheduler peaks82,548/82,564/92,712KiB. Each request remained1CPU/256MiB,60-second scheduler/30-second outer/20-second check. Each shard's artifacts are below32MiB. No GPUs,framework imports,modelcalls,retries or changes to other projects. Cumulative diagnostic allocation:2,576CPU-seconds and254MIG-slice-seconds. Local and remote final ledgers are saved;no next experiment is launched by this closeout.

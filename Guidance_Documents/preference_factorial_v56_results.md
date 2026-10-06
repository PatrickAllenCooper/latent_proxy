# v56 factorial diagnostic — completed negative result

## Decision and validity

Approved frozen v56 executed as GPU job33483615 after changed CPU gate33483501.
Terminal COMPLETED0:0. All sixteen responses retained; zero parse failures,
16 actual EOS, zero caps, exact CPU/GPU token equality and frozen source/model/
template/case hashes. **Six of sixteen correct (37.5%); scientific smoke fails.**
Advice remains blocked, v55's11/16 failure remains preserved, no automatic
retry/replacement/retuning. No missing records or engineering failure explains
this result. This is a two-semantic-case mechanism diagnostic, not a user
population estimate or a broad model ranking.

## Frozen contrasts

Prose3/8, explicit-rank3/8. Every one of the eight paired correctness
contrasts is zero; rank annotations neither rescue nor lose correctness in
any paired cell. However semantic choice agrees in only4/8 format pairs,
so zero accuracy difference does not imply identical decisions. Eligibility
violations rise from2/8 prose to4/8 rank, while eligible priority errors fall
from3/8 to1/8. The extra computed rank information changes the kind of mistake
without improving correctness in these cases.

Letter rotation0→2 has paired correctness differences
`[0,0,-1,0,0,0,-1,0]`, mean−0.25 (−25 percentage points over eight pairs).
Semantic choice agrees5/8 pairs. Catalog forward→reverse differences are
`[0,0,1,0,1,0,0,0]`, mean+0.25, semantic choice agreement5/8. These observed
sensitivities are scoped to these fixed two cases. Both nonzero accuracy
changes occur in fresh-00; fresh-01 is wrong throughout. All pair IDs and
per-block results are saved in the frozen analysis output; no selected subset
is promoted as a passing gate and no p-value/population interval is attached.

Fresh-00 (highest-priority eligible) scores6/8: both formats3/4. With canonical
letters all four conditions are correct; rotated labels are wrong in forward
order but correct after row reversal. The forward rotated prose chooses an
ineligible option, while rank format chooses an eligible lower-priority one.

Fresh-01 (highest-priority ineligible) scores0/8: both formats0/4. Rank format
chooses the ineligible highest-priority portable option in all four cells.
Prose chooses it once and an eligible nonoptimal option in the other three.
This supports a concrete failure mode in this case: explicit ranking is
insufficient to enforce eligibility. With only two semantic blocks, content
and top-priority eligibility difficulty are not separately identified.

## Resource and execution evidence

CPU preparation used direct pinned Qwen2Tokenizer instead of AutoTokenizer
model dispatch. It matched all16 previously validated token sequences plus
all16 fresh tokens and full model/runtime file hashes, and exited normally.
Four CPUs/8GB,56 elapsed seconds,224 allocated CPU seconds,7.883 observed CPU
seconds, peakRSS637,940KiB. Previous timeout33483296 is retained. Reusing a
warmed CPU node prevents a causal speedup or reliability claim.

GPU actual envelope: oneH2002g35gbMIG,4 CPUs,32GB,28 elapsed seconds versus5
minutes requested;112 allocated CPU seconds,8.241 observed CPU seconds, peak
hostRSS1,220,168KiB. Model GPU work began before90 seconds; complete receipt
at23.31 seconds within285-second bound. Greedy BF16/SDPA/use_cache=False;
16 calls,32 generated tokens,max24 new tokens. Peak CUDA allocated/reserved
3,134,516,736/3,149,922,304 bytes (<12GiB). CUDA event spans totaling1.845s,
CUDA memory and generation progress support real GPU work. Event spans are
not utilization percentages; MIG utilization wasN/A and host-wide memory is
not attributable to this job. Cleanup left32MiB allocated and roughly2.88GiB
reserved until normal process termination. No resource expansion occurred.

## What is resolved and what remains

On these fresh cases this checkpoint does not reliably execute the explicit
preference/eligibility rule. Computed ranks provide no paired accuracy gain,
and letter/catalog transformations can change decisions. The supplied-user
preferences were never hidden or learned: this is execution/representation
capacity evidence, not preference discovery, RL proxy inference or a tested
prompt optimizer/tool-call/fine-tuning comparison.

The frozen decision branch is: hold advice; stop retuning this assay. The next
scientific comparison should use one preselected stronger checkpoint on an
independently registered fresh case panel, with a deterministic constrained
proxy as the computational control. Verify that the baseline can execute the
rule before interpreting advice verification or optimizer/tool/adapter
comparisons. Those are recommendations only: no new model job or approval
request created here. v56's complete negative result cannot be relabeled as a
passing representation condition.

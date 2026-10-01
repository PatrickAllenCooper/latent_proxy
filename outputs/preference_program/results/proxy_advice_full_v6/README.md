# Proxy advice reliability ablation

CURC job 33203230 completed on one H200 2g.35gb MIG slice in 36 seconds.
The CPU-prepared manifest was `outputs/preference_program/manifests/advice_ablation_full.jsonl`.
The job produced 288 unique records: 12 paired users × 8 menus × three
prompt conditions. All completions parsed. Raw prompts, responses, metrics,
receipts, stdout, stderr, and telemetry are retained. `per_user.csv` and
`summary.json` join the prior no-tool base-model arm and provide user-level
bootstrap intervals.

- Direct worst-action advice: copied 96/96; mean normalized regret 1.000.
- Caveated worst-action advice: copied 36/96, chose C on 92/96; regret 0.528.
- Exact numeric utility scores without a recommendation: chose A on 96/96;
  A was actually optimal on 48/96; regret 0.320.
- Prior no-tool baseline: chose C on 96/96; regret 0.516.

Direct wrong advice minus no tool increased regret by 0.484, paired 95%
user-bootstrap interval [0.344, 0.624]. Caveated minus direct wrong advice
reduced regret by 0.472 [-0.609, -0.343], but its 0.528 regret was about
the same as no tool. Numeric scores minus no tool was -0.196 [-0.468, 0.067],
an imprecise comparison. Constant A responses prevent attributing any gain
to score comparison.

The job observed one MIG GPU, about 1.19 GB framework memory allocated at
first generation, and 288 completed generations. CURC's `nvidia-smi`
utilization percentage was N/A for this MIG slice. The 15-minute allocation
was released after 36 seconds. The tested tool result was inserted by the
harness; this does not measure voluntary tool calling.

# Full finite-menu LLM pilot

CURC job 33199705 completed successfully on one H200 2g.35gb MIG slice in
81 seconds. Each of base and dialogue-adapted Qwen 2.5 1.5B generated 288
records: 12 users × 8 menus × 3 arms. All 576 records have unique matched
keys and zero parse failures. Raw prompts, completions, per-decision scores,
receipts, user-bootstrap summary, stdout, stderr, and telemetry are retained.

Both models selected C in all no-tool cases and copied the tool's recommended
letter in every exact-reward and learned-RL tool case. Their 288 paired
completions were identical. Mean normalized regret: no tool 0.5158, learned
RL advice 0.1424, exact reward advice 0.0000. For the base model, learned
advice minus no tool was -0.3734 with 95% user-bootstrap interval
[-0.5497, -0.2063]. The dialogue model had the same point estimate.

The tool result was inserted by the harness and gave a single recommended
letter. These results measure response to supplied advice, not the model's
choice to call a tool or its ability to infer preferences. The exact base and
dialogue match is limited to this deterministic prompt and decision format.

The job requested one 35 GB MIG GPU for one hour and used one such device for
81 seconds. Framework counters show about 1.19 to 1.20 GB allocated during
first generation and 1.24 to 1.26 GB peak; 576 completed generations verify
GPU work. `nvidia-smi` reported utilization as N/A on the MIG slice, so a
percentage is unavailable. The adapter file used for dialogue has SHA-256
`a053215ca8cb0f5114cdd7efe0ecbe33f10c0f42ee421dd74ad8de4fd0ccfb85`.

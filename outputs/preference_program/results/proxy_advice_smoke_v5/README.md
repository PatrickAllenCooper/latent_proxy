# Advice-reliability smoke

CURC job 33201355 completed in 44 seconds on one H200 2g.35gb MIG slice. Three CPU-prepared prompts produced three parseable generations. On this one menu, the true best action was B and the worst was C. The base model chose C with both direct wrong advice and caveated wrong advice. Given exact numeric utility scores without a letter recommendation, it chose A. This is only a pipeline and raw-behavior smoke, not an effect estimate.

The job observed one MIG device, about 1.19 GB framework GPU memory allocated at first generation, and three completed generations. `nvidia-smi` utilization was N/A for the MIG slice; a utilization percentage is unverified. Logs, prompts, completions, scores, and telemetry are retained.

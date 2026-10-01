# GPU smoke receipt

CURC job 33196214 completed in 3m52s on one H200 2g.35gb MIG slice. Base and dialogue conditions produced three records each, one per no-tool, exact-reward-tool, and learned-RL-tool arm. All six completions parsed as a single valid action. Raw prompts, completions, scores, receipts, stdout, stderr, and sampled telemetry are retained here.

The first generation used 1.19 GB framework-allocated GPU memory for base and 1.20 GB for dialogue; each produced two output tokens. `nvidia-smi` returned `N/A` for GPU utilization on this MIG slice, so utilization percentage is unverified. Framework GPU memory and completed generations jointly verify GPU work. The sample is too small for an outcome claim. The full run may use the same single MIG slice; one hour is a bounded allowance for 576 generations and two model loads.

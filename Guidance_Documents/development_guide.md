# Latent Proxy Development Guide

## Project Overview

This project implements Latent Preference Inference via Conversational Active Learning.
The system infers latent user preference parameters (discount factor, risk aversion,
loss aversion) from natural language interaction, produces recommendations jointly
optimized for objective quality and preference alignment, and quantifies the degree
to which inferred preferences generalize across decision domains.

Full specification is in `README.md` and `paper.tex`.

## Current Status

### Milestone 1: Game Environment + Synthetic Users -- COMPLETE

Scope:
- Resource strategy game environment (Gymnasium API)
- Synthetic user sampler with configurable preference parameters
- Quality floor constraint enforcement
- Comprehensive test suite (87 tests passing)

Components implemented:
- [x] `src/environments/base.py` -- abstract environment interface
- [x] `src/environments/resource_game.py` -- resource strategy game
- [x] `src/training/synthetic_users.py` -- synthetic user sampler
- [x] `src/evaluation/quality_metrics.py` -- quality floor evaluation
- [x] `src/utils/posterior.py` -- Bayesian posterior stub
- [x] `configs/game/default.yaml` -- game configuration
- [x] `tests/` -- test suite (4 test files + validation)

Validated:
- Optimal strategies differ meaningfully across all pairs of extreme user types
- Risk-averse users allocate more to the safe channel
- Different market regimes produce different recommendations for the same user
- Quality floor constraints reject degenerate allocations and pass valid ones
- Optimal actions for all sampled user types pass quality floor constraints

### Milestone 2: Post-Training Pipeline -- COMPLETE

Scope:
- Text serialization layer bridging structured game data and LLM text
- DPO pair construction pipeline (Phase 1 quality-only, Phase 2 type-conditioned)
- Conditional DPO trainer wrapping TRL with curriculum phases
- Type-conditioned reward model
- Alignment evaluation metrics
- Training scripts and configurations
- Comprehensive test suite (149 total tests passing)

Components implemented:
- [x] `src/training/serialization.py` -- GameStateSerializer, UserProfileSerializer, AllocationSerializer
- [x] `src/training/dpo_data.py` -- DPOPairGenerator, CandidateGenerator, HF Dataset export
- [x] `src/training/model_utils.py` -- QLoRA model loading, LoRA config, tokenizer setup
- [x] `src/training/dpo_trainer.py` -- ConditionalDPOTrainer with curriculum phases
- [x] `src/training/reward_model.py` -- TypeConditionedRewardModel using TRL RewardTrainer
- [x] `src/evaluation/alignment_metrics.py` -- alignment score, violation rate, model evaluation
- [x] `configs/training/dpo.yaml` -- DPO training configuration
- [x] `configs/training/reward_model.yaml` -- Reward model configuration
- [x] `scripts/train_quality.py` -- Phase 1 training script
- [x] `scripts/train_alignment.py` -- Phase 2 training script
- [x] `tests/` -- 4 new test files (serialization, DPO data, alignment metrics, reward model)

### Milestone 3: Active Learning Loop -- COMPLETE

Scope:
- Upgraded posterior tracking: PosteriorBase ABC, ParticlePosterior (weighted particles
  with systematic resampling), GaussianPosterior (importance-weighted update from choices)
- Diagnostic scenario library generating gamma/alpha/lambda-discriminating binary queries
- MC-based Expected Information Gain computation
- Agent module: StructuredQueryGenerator (EIG-maximizing), RandomQueryGenerator (baseline),
  PreferenceTracker (posterior + convergence), ResponseGenerator, ElicitationLoop
- Elicitation metrics: efficiency comparison, recovery curves, benchmark runner
- CURC SLURM scripts for A100 training and evaluation
- 208+ tests at M3 closure; see Milestone 4 for current count

Components implemented:
- [x] `src/utils/posterior.py` -- PosteriorBase, GaussianPosterior, ParticlePosterior
- [x] `src/utils/diagnostic_scenarios.py` -- DiagnosticScenario, ScenarioLibrary
- [x] `src/utils/information_gain.py` -- compute_eig_mc, compute_eig_batch
- [x] `src/agents/base.py` -- BaseAgent ABC
- [x] `src/agents/query_generator.py` -- StructuredQueryGenerator, RandomQueryGenerator
- [x] `src/agents/preference_tracker.py` -- PreferenceTracker with convergence checking
- [x] `src/agents/response_generator.py` -- ResponseGenerator
- [x] `src/agents/elicitation_loop.py` -- ElicitationLoop orchestrator
- [x] `src/evaluation/elicitation_metrics.py` -- efficiency, recovery curves, benchmarks
- [x] `configs/active_learning/default.yaml`
- [x] `scripts/run_elicitation.py` -- local CPU benchmark
- [x] `scripts/slurm/` -- CURC deployment scripts

Benchmark results (10 users, 8 rounds, 500 particles, 200 EIG samples, pre-M4):
- Active: 0.364 mean error, Random: 0.367 mean error
- Error reduction: 0.8% (below 30% target)
- Root cause (historical): gamma was weakly identifiable under single-period EU;
  Milestone 4 adds multi-period evaluation for gamma scenarios and raises MC defaults.

### Milestone 4: Evaluation + Game Domain Results -- COMPLETE (code, tests, local smoke; large-run targets ongoing)

Scope:
- Multi-period expected utility for gamma diagnostic scenarios (compounding wealth +
  per-round discounted prospect utility) wired through elicitation, posterior
  likelihood, and EIG
- Game variant B (6 channels, T=50) YAML + factory helpers for transfer experiments
- Full experiment runner (README 7.1 metrics + target checks) and A->B transfer
  protocol (Generic / Within-Domain / Cross-Domain, README 8.1)
- Ablation runner: query budget, posterior type, and beta proxy (post-hoc theta blend)
- Matplotlib visualization utilities and JSON/Markdown export
- CURC: `run_full_evaluation.py`, `run_ablation.slurm`, expanded `run_evaluation.slurm`
- **224** tests passing (includes `test_multiperiod_eval`, `test_game_variants`,
  `test_experiment_runner`, `test_ablation`)

Components implemented:
- [x] `src/training/synthetic_users.py` -- `evaluate_allocation_multiperiod`,
  `evaluate_for_query`
- [x] `src/utils/diagnostic_scenarios.py` -- `multiperiod_horizon` on gamma scenarios
- [x] `src/utils/posterior.py` / `information_gain.py` -- multi-period likelihood + EIG
- [x] `src/agents/preference_tracker.py`, `elicitation_loop.py` -- scenario-aware EU
- [x] `configs/game/variant_b.yaml`, `src/environments/game_variants.py`
- [x] `src/evaluation/experiment_runner.py` -- `run_full_evaluation`,
  `run_transfer_experiment`, target checks
- [x] `src/evaluation/ablation_runner.py` -- `run_sweep`
- [x] `src/utils/visualization.py` -- plots, tables, `save_results`
- [x] `scripts/run_full_evaluation.py`, `scripts/slurm/run_ablation.slurm`
- [x] `pyproject.toml` -- `matplotlib` dependency

Validation notes:
- Local smoke: `python scripts/run_full_evaluation.py` produces variant metrics,
  transfer plot, `results_bundle.json`, and `results_summary.md`.
- README 7.1 efficiency target (>=30% error reduction vs random) remains **data-dependent**;
  use `--n-particles 2000`, `--n-eig-samples 800`, `--max-rounds 10+`, and larger
  `--n-users` on CURC; multi-period gamma scenarios improve identifiability but do
  not guarantee the threshold on every small-sample run.
### Milestone 5: Stock Backtesting Domain -- NOT STARTED
### Milestone 6: Generalization Study -- NOT STARTED

## Architecture

### Environment Layer
All environments extend `BaseEnvironment` (which extends `gymnasium.Env`) and implement:
- `quality_score(action)` -- R_quality for objective evaluation
- `check_quality_floor(action)` -- hard constraint checking
- `get_optimal_action(theta)` -- optimal action for a given user type

### Synthetic User Layer
Users are parameterized by theta = (gamma, alpha, lambda):
- gamma in (0,1]: discount factor (Beta prior)
- alpha >= 0: risk aversion (LogNormal prior)
- lambda >= 1: loss aversion (Uniform prior)

Choice model: softmax-rational with temperature tau.

### Quality Floor
Three hard constraints enforced on all recommendations:
1. No allocation to strictly dominated channels
2. Diversification minimum (>= 2 channels)
3. Bankruptcy probability ceiling

### Serialization Layer (Milestone 2)
Bridges structured game data and LLM text:
- GameStateSerializer: environment observations to natural language prompts
- UserProfileSerializer: UserType parameters to human-readable preference descriptions
- AllocationSerializer: numpy arrays to/from percentage-based text with robust parsing

### Post-Training Pipeline (Milestone 2)
- DPO pair construction with two curriculum phases:
  Phase 1 (quality): y_w has highest Sharpe, y_l violates quality floor. No user profile.
  Phase 2 (alignment): y_w optimal for sampled user type, y_l optimal for contrasting type. User profile in prompt.
- Candidate generation: optimal actions, Dirichlet random, perturbations of optimal
- Training via TRL DPOTrainer with LoRA adapters on Mistral-7B
- Type-conditioned reward model via TRL RewardTrainer for online PPO (Milestone 3)
- Alignment evaluation: Spearman rank correlation, quality floor violation rate

## Design Decisions

1. Gymnasium v1.2.3 for environment API with Dict observation space
2. NumPy for core math; NumPyro reserved for full posterior tracking in Milestone 3
3. Hydra for configuration management from the start
4. Prospect-theory utility as standalone function for cross-environment reuse
5. Quality floor returns structured violation info (not just boolean)
6. pyproject.toml as single source of truth for all project metadata
7. Certainty equivalent uses sqrt(variance * horizon) risk penalty, not linear.
   Returns grow linearly with effective horizon (mu * h) while risk penalty
   scales as sqrt(var * h). This sub-linear scaling ensures that gamma (discount
   factor / patience) produces meaningfully different allocations: patient users
   face more cumulative risk on volatile channels relative to their return
   advantage, pushing them toward safer allocations compared to impatient users.
   Without this nonlinearity, the horizon cancels in simplex normalization.

### Active Learning Architecture (Milestone 3)
- Structured elicitation (Option A from README Section 6.1): a ScenarioLibrary
  generates diagnostic binary choices, each scored by EIG to select the most
  informative query at each round.
- PosteriorBase ABC with two implementations:
  GaussianPosterior (fast, unimodal) and ParticlePosterior (flexible, handles
  multimodality). Both update from observed binary choices via softmax-rational
  likelihood model.
- Convergence criteria: per-parameter variance thresholds, max query budget,
  and robust-action check (same optimal action across 90% credible region).
- Elicitation loop operates entirely on CPU with synthetic users; GPU needed
  only for LLM rendering of queries (future integration).

## Design Decisions (Milestone 2)

8. Natural language user profiles for type conditioning rather than structured
   embeddings. The user profile is part of the prompt text, which TRL's DPO
   handles natively. Structured embeddings would require model architecture changes.
9. Mistral-7B-Instruct-v0.3 as default base model. Open weights, strong
   instruction following, well-tested with TRL/PEFT/LoRA.
10. 4-bit quantization + LoRA (rank 16) for memory efficiency. Allows training
    on a single 24GB GPU.
11. Candidate allocations generated heuristically before any trained policy
    exists: optimal actions for various theta, random Dirichlet samples, and
    perturbations. Once a model is trained, its outputs seed future pair construction.
12. Separate Phase 1 and Phase 2 training scripts matching the curriculum in
    README Section 5.3.

## Design Decisions (Milestone 3)

13. ParticlePosterior as default over GaussianPosterior. Particles handle
    non-Gaussian posteriors and parameter coupling without parametric assumptions.
    Systematic resampling when ESS drops below 50% of particle count.
14. Diagnostic scenarios constructed with extreme parameter contrasts rather than
    just optimal-action differences. For alpha: 75% safe vs 50% aggressive. For
    gamma: safe-heavy vs aggressive-heavy; **Milestone 4** attaches
    `multiperiod_horizon` so EU is computed on compounding paths (gamma identifiable).
15. EIG estimation via nested Monte Carlo with importance-weighted entropy.
    Trades accuracy for speed; default EIG sample count tunable via
    `ElicitationConfig.n_eig_samples` (800+ recommended for production runs).
16. CURC SLURM scripts placed under scripts/slurm/ for A100 GPU training.
    Local 3080 used for inference and small-scale testing; CURC for full
    DPO training and large-scale evaluation.

## Design Decisions (Milestone 4)

17. Multi-period Monte Carlo for gamma scenarios uses vectorized path simulation
    (batch of paths x horizons) with default 256 path samples per EU call to keep
    EIG inner loops tractable; increase for final paper runs if variance is high.
18. Beta ablation in code uses **post-hoc blending** of prior mean and inferred
    theta when scoring recommendations; full DPO beta sweeps still require
    retraining (see `scripts/train_alignment.py`).
19. `save_results` JSON export accepts nested dataclasses via `dataclasses.asdict`
    recursion in `visualization._convert`.

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | >= 3.10 |
| RL Environment | Gymnasium | >= 1.2.3 |
| Numerical | NumPy | >= 2.0 |
| Statistics | SciPy | >= 1.14 |
| Bayesian Inference | NumPyro | >= 0.20.0 |
| Config Management | Hydra | >= 1.3 |
| Experiment Tracking | W&B | >= 0.19 |
| Fine-tuning | TRL | >= 0.29 |
| LLM Base | Transformers | >= 4.48 |
| Parameter Efficient | PEFT | >= 0.14 |
| Quantization | BitsAndBytes | >= 0.45 |
| Distributed Training | Accelerate | >= 1.3 |
| Dataset Management | HF Datasets | >= 3.0 |
| Testing | pytest | >= 8.0 |
| Linting | ruff | >= 0.9 |
| Type Checking | mypy | >= 1.14 |
| Plotting | matplotlib | >= 3.8 |

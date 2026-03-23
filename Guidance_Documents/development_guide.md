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

### Milestone 2: Post-Training Pipeline -- NOT STARTED
### Milestone 3: Active Learning Loop -- NOT STARTED
### Milestone 4: Evaluation + Game Domain Results -- NOT STARTED
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
| Testing | pytest | >= 8.0 |
| Linting | ruff | >= 0.9 |
| Type Checking | mypy | >= 1.14 |

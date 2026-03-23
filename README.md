# Latent Preference Inference via Conversational Active Learning

## 1. Problem Statement

Given a user interacting with an LLM-based agent in a decision-support context, we aim to:

1. **Infer latent preference parameters** (discount factor γ, risk aversion α, loss aversion λ, etc.) from natural language interaction using active learning through selective prompting.
2. **Produce recommendations** that are jointly optimized for **objective quality** (performance in a verifiable environment) and **preference alignment** (conformity with the user's inferred type).
3. **Quantify the degree to which inferred preference parameters generalize** — within task variants, across task variants, and across entirely distinct decision domains.

We treat this as a **post-training** problem. A pretrained LLM is fine-tuned via DPO, RLHF, and related methods so that it learns to act as a preference-eliciting, environment-grounded decision agent.

---

## 2. Core Formulation

### 2.1 Reward Decomposition

The agent's reward for a recommendation action `a` given context `x` and user type `θ_user` is:

```
R(a, x, θ_user) = R_quality(a, env) + β · R_alignment(a, θ_user)
```

- **R_quality(a, env)**: Environment-grounded evaluation. Does the action perform well by objective criteria? This acts as a hard floor — no preference alignment justifies objectively dominated or irrational actions (e.g., recommending a fraudulent asset, making a dominated game move).
- **R_alignment(a, θ_user)**: Does the action match the user's inferred preference profile? A high-γ user prefers conservative long-horizon strategies; a low-γ user prefers aggressive near-term payoffs.
- **β**: Weighting hyperparameter controlling the quality-alignment tradeoff. In early training, β is low (prioritize not being wrong); as quality stabilizes, β increases (prioritize personalization).

### 2.2 User Type Model

The user's latent type is a vector of preference parameters:

```
θ_user = (γ, α, λ, ...)
```

where:
- **γ ∈ (0, 1]**: Discount factor over future outcomes
- **α ≥ 0**: Risk aversion (concavity of utility over outcomes)
- **λ ≥ 1**: Loss aversion (asymmetric weighting of losses vs. gains)
- Additional parameters as needed per domain (e.g., hyperbolic discounting parameter κ, ambiguity aversion)

The agent maintains a posterior belief:

```
p(θ_user | h_t) where h_t = {(q_1, r_1), ..., (q_t, r_t)}
```

updated after each interaction round consisting of a query `q_i` posed to the user and their response `r_i`.

### 2.3 Active Preference Elicitation

At each round, the agent selects the next query to maximize expected information gain over the user's type:

```
q* = argmax_q  EIG(q) = H[θ_user | h_t] - E_{r ~ p(r|q, h_t)}[ H[θ_user | h_t, q, r] ]
```

**Design constraint**: Each query must be accompanied by sufficient background information for the user to make an informed decision. The agent is responsible for contextualizing the choice (e.g., explaining risk metrics, historical performance, tradeoff structure) so that user responses reflect genuine preferences rather than confusion or ignorance.

---

## 3. System Architecture

### 3.1 High-Level Components

```
┌──────────────────────────────────────────────────────┐
│                    User Interface                     │
│         (Natural language interaction layer)          │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│               Agent (Fine-tuned LLM)                 │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  Preference  │  │   Query      │  │ Response   │  │
│  │  Posterior   │  │   Generator  │  │ Generator  │  │
│  │  Tracker     │  │  (Active     │  │ (Recommend │  │
│  │              │  │   Learning)  │  │  + Context)│  │
│  └─────────────┘  └──────────────┘  └────────────┘  │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│             Environment Engine                        │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  Simulator   │  │  Outcome     │  │  Quality   │  │
│  │  (Game,      │  │  Evaluator   │  │  Scorer    │  │
│  │   Backtest,  │  │              │  │            │  │
│  │   Clinical)  │  │              │  │            │  │
│  └─────────────┘  └──────────────┘  └────────────┘  │
└──────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│             Training Pipeline                         │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  Synthetic   │  │  DPO Pair    │  │  Reward    │  │
│  │  User        │  │  Constructor │  │  Model     │  │
│  │  Sampler     │  │              │  │  Trainer   │  │
│  └─────────────┘  └──────────────┘  └────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 3.2 Component Specifications

#### 3.2.1 Agent (Fine-tuned LLM)

- **Base model**: Open-weight instruction-tuned model (e.g., LLaMA-3, Mistral). Selected for accessibility and fine-tuning flexibility.
- **Preference Posterior Tracker**: Maintains a distributional estimate of θ_user. Implementation options:
  - **Parametric**: Maintain a Gaussian posterior over θ_user, updated via approximate Bayesian inference after each interaction.
  - **Particle-based**: Maintain a set of weighted user-type particles; resample after each observation.
  - **Implicit (learned)**: The LLM's internal representation encodes the posterior implicitly through in-context conditioning on interaction history. This is the most elegant but hardest to inspect.
- **Query Generator**: Proposes diagnostic questions. May be end-to-end (the LLM generates queries directly, scored by EIG) or modular (a structured elicitation module identifies the most informative preference dimension, and the LLM renders it as natural language with appropriate context).
- **Response Generator**: Given the current posterior and environment state, produces a recommendation that balances quality and alignment. Responsible for providing contextual background so the user can evaluate the recommendation.

#### 3.2.2 Environment Engine

Provides the grounding for R_quality. Must support:
- **Action intake**: Accept structured actions from the agent (e.g., "allocate 40% to resource A, 30% to B, 30% to C").
- **Simulation**: Roll forward the consequences of actions under a world model.
- **Outcome evaluation**: Score realized outcomes on objective criteria.
- **Quality floor enforcement**: Flag actions that violate hard constraints (dominated strategies, irrational moves, illegal actions).

#### 3.2.3 Training Pipeline

- **Synthetic User Sampler**: Generates simulated users with known θ_user values drawn from a prior distribution. These users respond to queries according to their parameterized utility function, providing a supervised signal for training.
- **DPO Pair Constructor**: For each synthetic user type, generates (prompt, y_w, y_l) triples where y_w is preferred under that user's utility function and y_l is dispreferred. The preference ordering is type-conditioned.
- **Reward Model Trainer**: Learns r(y | x, θ_user) → scalar for the RLHF pathway. The reward model is conditioned on the user profile.

---

## 4. Phase 1 Environment: Resource Strategy Game

### 4.1 Rationale

A purpose-built resource management game provides:
- Full simulability and fast iteration
- Complete observability of outcomes
- Exact computation of optimal strategies for any given θ
- No regulatory or ethical constraints
- Clean testbed for the active learning and post-training pipeline before moving to real-world domains

### 4.2 Game Design

A sequential resource allocation game over `T` rounds. The player manages a portfolio of resources across `K` investment channels with different risk-return profiles.

#### State Space
```
s_t = (w_t, p_t, t)
```
- `w_t ∈ R^K`: Current wealth allocation across K channels
- `p_t`: Observable market/environment state (e.g., channel yields, volatility regime)
- `t`: Current round (induces horizon-dependent behavior)

#### Action Space
```
a_t ∈ Δ^K  (simplex over K channels)
```
The agent recommends a reallocation at each round.

#### Channel Archetypes
| Channel | Expected Return | Variance | Correlation | Analogy |
|---------|----------------|----------|-------------|---------|
| Safe    | Low            | Low      | ~0          | Bonds / turtling |
| Growth  | Medium         | Medium   | Moderate    | Index funds / balanced expansion |
| Aggressive | High        | High     | High        | Speculative stocks / early rush |
| Volatile   | Variable    | Very High| Negative    | Options / high-risk gambits |

#### Dynamics
Returns are drawn from a stochastic process with regime switching (bull/bear/neutral states) to create realistic non-stationarity. The regime transition probabilities are known to the environment but partially observable to the agent.

#### Terminal Payoff
```
U(w_T; θ_user) = u(w_T; α, λ) · γ^(T-t_decision)
```
where `u` is a prospect-theory style utility function parameterized by the user's risk and loss aversion.

### 4.3 Quality Floor Constraints

- **Dominance**: Never recommend putting 100% in a channel that is strictly dominated in both mean and variance by another channel in the current regime.
- **Diversification floor**: Recommendations must allocate to at least 2 channels (unless one channel dominates all others in the current state, which should be rare by design).
- **Bankruptcy avoidance**: Never recommend an allocation that has >X% probability of total loss within the next round.

### 4.4 Simulated User Behavior

Synthetic users are parameterized by θ = (γ, α, λ) drawn from:
```
γ ~ Beta(a, b)        # Discount factor, support (0, 1]
α ~ LogNormal(μ, σ)   # Risk aversion
λ ~ Uniform(1, 3)     # Loss aversion
```

When presented with a choice between options A and B, the synthetic user selects the option with higher expected discounted utility under their parameters, with a softmax temperature τ to model bounded rationality:

```
P(choose A) = σ( (EU_A(θ) - EU_B(θ)) / τ )
```

---

## 5. Post-Training Methods

### 5.1 Conditional DPO

Standard DPO learns a policy from preference pairs (y_w ≻ y_l | x). Our setting requires **type-conditioned preferences**: the winning response depends on θ_user.

**Training data construction**:
1. Sample a synthetic user type θ ~ prior.
2. Sample a game state s.
3. Generate candidate recommendations {y_1, ..., y_n} from the current policy.
4. Evaluate each y_i in the environment simulator under user utility U(·; θ).
5. Construct pairs: y_w = argmax_i U(y_i; θ), y_l sampled from lower-utility candidates.
6. The prompt x is augmented with a representation of the user's preference profile (either a natural language description or a structured embedding).

**Loss**:
```
L_CDPO(π_θ; π_ref) = -E[ log σ( β · (log π_θ(y_w|x,θ_user)/π_ref(y_w|x,θ_user) 
                                     - log π_θ(y_l|x,θ_user)/π_ref(y_l|x,θ_user)) ) ]
```

This is standard DPO but the context is enriched with user type information, forcing the model to learn type-dependent preferences.

### 5.2 RLHF with Type-Conditioned Reward Model

For the online setting:
1. Train a reward model: `r_φ(y | x, θ_user) → R`
2. Use PPO or similar to optimize the agent policy against this reward model.
3. The reward model is trained on the same type-conditioned preference data as DPO.

The advantage over DPO: the reward model can be updated online as real user feedback arrives, enabling continual adaptation.

### 5.3 Curriculum Strategy

| Phase | Focus | β (alignment weight) | Data Source |
|-------|-------|---------------------|-------------|
| 1     | Quality floor | 0.0 | Environment-only signal. Learn not to make dominated recommendations. |
| 2     | Basic alignment | 0.3 | Synthetic users with well-separated types (very high γ vs very low γ). |
| 3     | Fine-grained alignment | 0.7 | Synthetic users with subtle type differences. Active learning loop engaged. |
| 4     | Real user calibration | 1.0 | Human-in-the-loop evaluation. Adjust based on real interaction data. |

---

## 6. Active Learning Module

### 6.1 Query Strategy

The agent must learn to ask questions that efficiently reduce uncertainty about θ_user. Two architectural options:

**Option A — Structured Elicitation + LLM Rendering**:
1. A separate module computes EIG over a predefined set of diagnostic game scenarios.
2. The highest-EIG scenario is selected.
3. The LLM renders this as a natural language question with appropriate context and background.
4. The user's response is parsed back into a structured choice.
5. The posterior is updated analytically or via particle filtering.

**Option B — End-to-End Learned Elicitation**:
1. The LLM generates candidate queries directly.
2. Each query is scored by approximate EIG (e.g., via Monte Carlo sampling of synthetic user responses).
3. The highest-scoring query is presented.
4. The posterior update is implicit in the interaction history.

**Recommendation**: Begin with Option A for interpretability and debuggability. Transition to Option B once the pipeline is validated and we want to test whether end-to-end learning discovers elicitation strategies we wouldn't design by hand.

### 6.2 Contextual Grounding Requirement

Every diagnostic query must include:
- A clear description of the choice being presented
- Relevant quantitative data (expected returns, risk metrics, historical analogs in the game)
- An explanation of what each option trades off
- No leading framing that biases the user toward a particular choice

This is both an ethical requirement (informed consent in decision-making) and a practical one (user responses are only informative about preferences if the user understands what they're choosing between).

### 6.3 Convergence Criteria

The active learning loop terminates (switches from elicitation to recommendation mode) when:
- The posterior variance on key parameters drops below a threshold: `Var(γ | h_t) < ε_γ`
- Or a maximum query budget is exhausted: `t > T_max`
- Or the agent's recommended action is robust to posterior uncertainty: the optimal action is the same across the 90% credible region of θ_user

---

## 7. Metrics

### 7.1 Primary Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Quality Score** | R_quality averaged over evaluation episodes. Measures whether recommendations are objectively sound. | Monotonically increasing through training; converges above a domain-specific threshold. |
| **Alignment Score** | Rank correlation between agent's recommendations and the recommendations that would be optimal under the user's true θ. Measured on synthetic users with known parameters. | > 0.8 rank correlation for well-identified users. |
| **Preference Recovery Error** | ‖θ_inferred - θ_true‖ for synthetic users. Decomposes into per-parameter errors (γ error, α error, λ error). | Mean absolute error on γ < 0.1 within 5 interaction rounds. |
| **Elicitation Efficiency** | Number of interaction rounds required to reach a target posterior variance. Compared against random query baselines and optimal Bayesian experimental design. | ≥ 30% reduction in rounds vs. random queries. |

### 7.2 Generalization Metrics

| Metric | Definition |
|--------|-----------|
| **Within-Domain Transfer** | Learn θ in game variant A (e.g., 4 channels, T=20). Predict preferences in game variant B (6 channels, T=50). Measure alignment score degradation. |
| **Cross-Domain Transfer** | Learn θ in game environment. Predict preferences in stock backtesting environment. Measure whether cross-domain θ beats a non-personalized baseline. |
| **θ Stability** | Infer θ in two separate sessions for the same user. Measure test-retest reliability (ICC or Pearson correlation). |

### 7.3 Safety and Rationality Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Quality Floor Violation Rate** | Fraction of recommendations that violate hard quality constraints (dominated strategies, bankruptcy risk, etc.). | 0% after Phase 1 training. |
| **Preference Manipulation Rate** | Frequency with which the agent's queries lead the user toward a particular answer rather than neutrally eliciting preferences. Measured by auditing query framing. | Near 0% (requires human eval). |

---

## 8. Generalization Experimental Protocol

### 8.1 Study Design

Three conditions per domain pair (A → B):

| Condition | Description |
|-----------|-------------|
| **Generic** | Recommendations generated without any personalization. Baseline. |
| **Within-Domain** | θ inferred from interactions in domain B, recommendations in domain B. Upper bound on personalization benefit. |
| **Cross-Domain** | θ inferred from interactions in domain A, used to personalize recommendations in domain B. The quantity of interest. |

### 8.2 Domain Ladder

| Order | Domain | Simulability | Identifiability | Real-World Fidelity |
|-------|--------|-------------|-----------------|-------------------|
| 1     | Resource Strategy Game | High | High | Low |
| 2     | Stock Backtesting | High | Medium | Medium |
| 3     | Supply Chain Procurement | Medium | Medium | Medium-High |
| 4     | Healthcare Treatment Planning | Low-Medium | Low | High |

### 8.3 Hypotheses

- **H1**: Within-domain personalization significantly outperforms generic recommendations (validates that θ recovery is useful).
- **H2**: Cross-domain transfer from game → stocks outperforms generic but underperforms within-domain (establishes partial generalization).
- **H3**: The magnitude of cross-domain transfer correlates with the structural similarity between domains (discount factor is more transferable than domain-specific risk parameters).
- **H4**: Active learning elicitation converges faster than random querying across all domains.

---

## 9. Software Stack

### 9.1 Core Dependencies

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Base LLM | LLaMA-3-8B or Mistral-7B | Open weights, well-supported fine-tuning ecosystem |
| Fine-tuning framework | TRL (Hugging Face) | Native DPO, PPO, reward model training |
| Game environment | Custom Python (Gymnasium API) | Standard RL env interface, easy integration |
| Stock backtesting | Custom or Backtrader | Extensible, Python-native |
| Posterior inference | NumPyro or Pyro | Probabilistic programming for Bayesian user modeling |
| Experiment tracking | Weights & Biases | Logging, hyperparameter sweeps, artifact versioning |
| Orchestration | Hydra | Config management across environments and training phases |

### 9.2 Repository Structure

```
latent-proxy/
├── README.md
├── Guidance_Documents/
│   └── development_guide.md       # Living project guidance doc
├── configs/
│   ├── game/
│   │   └── default.yaml           # Game environment configs
│   ├── training/
│   │   ├── dpo.yaml               # DPO training configs
│   │   └── reward_model.yaml      # Reward model configs
│   └── active_learning/
│       └── default.yaml           # Active learning configs
├── src/
│   ├── environments/
│   │   ├── base.py                # Abstract environment interface [M1]
│   │   └── resource_game.py       # Resource strategy game [M1]
│   ├── agents/
│   │   ├── base.py                # Abstract agent interface [M3]
│   │   ├── preference_tracker.py  # Posterior maintenance over θ_user [M3]
│   │   ├── query_generator.py     # EIG-scored active learning query selection [M3]
│   │   ├── response_generator.py  # Recommendation generation [M3]
│   │   └── elicitation_loop.py    # Active learning loop orchestrator [M3]
│   ├── training/
│   │   ├── synthetic_users.py     # Parameterized synthetic user simulation [M1]
│   │   ├── serialization.py       # Game state / user / allocation text conversion [M2]
│   │   ├── dpo_data.py            # Type-conditioned DPO pair construction [M2]
│   │   ├── model_utils.py         # QLoRA model loading, LoRA config [M2]
│   │   ├── dpo_trainer.py         # Conditional DPO training loop [M2]
│   │   └── reward_model.py        # Type-conditioned reward model [M2]
│   ├── evaluation/
│   │   ├── quality_metrics.py     # R_quality evaluation [M1]
│   │   ├── alignment_metrics.py   # Preference recovery, alignment scoring [M2]
│   │   └── elicitation_metrics.py # Elicitation efficiency, benchmarks [M3]
│   └── utils/
│       ├── posterior.py            # PosteriorBase, Gaussian + Particle posteriors [M1/M3]
│       ├── information_gain.py    # MC-based EIG computation [M3]
│       └── diagnostic_scenarios.py # Diagnostic scenario library [M3]
├── scripts/
│   ├── train_quality.py           # Phase 1: quality floor training [M2]
│   ├── train_alignment.py         # Phase 2: alignment training [M2]
│   ├── run_elicitation.py         # Elicitation benchmark (CPU) [M3]
│   └── slurm/
│       ├── setup_env.sh           # CURC environment setup [M3]
│       ├── train_dpo.slurm        # CURC DPO training [M3]
│       ├── train_reward.slurm     # CURC reward model training [M3]
│       └── run_evaluation.slurm   # CURC full evaluation [M3]
└── tests/
    ├── test_environments.py       # [M1]
    ├── test_synthetic_users.py    # [M1]
    ├── test_quality_floor.py      # [M1]
    ├── test_validation.py         # [M1]
    ├── test_serialization.py      # [M2]
    ├── test_dpo_data.py           # [M2]
    ├── test_alignment_metrics.py  # [M2]
    ├── test_reward_model.py       # [M2]
    ├── test_posterior.py           # [M3]
    ├── test_information_gain.py   # [M3]
    ├── test_diagnostic_scenarios.py # [M3]
    ├── test_query_generator.py    # [M3]
    └── test_elicitation_loop.py   # [M3]
```

---

## 10. Development Roadmap

### Milestone 1: Game Environment + Synthetic Users (Weeks 1-3)
- [x] Implement resource strategy game with Gymnasium API
- [x] Implement synthetic user sampler with configurable θ
- [x] Validate that optimal strategies differ meaningfully across user types
- [x] Verify quality floor constraints are well-calibrated

### Milestone 2: Post-Training Pipeline (Weeks 4-7)
- [x] Implement DPO pair construction from synthetic user + game rollouts
- [x] Train conditional DPO on base model
- [x] Implement reward model training
- [x] Validate quality floor is maintained after DPO (Phase 1 metric)
- [x] Validate basic alignment on well-separated user types (Phase 2 metric)

### Milestone 3: Active Learning Loop (Weeks 8-10)
- [x] Implement structured elicitation module (Option A)
- [x] Implement EIG computation over diagnostic game scenarios
- [x] Implement posterior tracking (parametric or particle-based)
- [x] Validate elicitation efficiency vs. random baseline
- [x] Validate convergence criteria

### Milestone 4: Evaluation + Game Domain Results (Weeks 11-13)
- [ ] Full evaluation suite on game environment
- [ ] Within-domain transfer experiments (game variant A → B)
- [ ] Ablation studies: β scheduling, query budget, posterior method
- [ ] Write up Phase 1 results

### Milestone 5: Stock Backtesting Domain (Weeks 14-18)
- [ ] Implement stock backtesting environment
- [ ] Adapt training pipeline to stock domain
- [ ] Cross-domain transfer: game → stocks
- [ ] Within-domain stock personalization results

### Milestone 6: Generalization Study (Weeks 19-22)
- [ ] Implement additional domains as needed
- [ ] Full cross-domain experimental protocol
- [ ] Statistical analysis of H1-H4
- [ ] Paper draft

---

## 11. Open Questions

1. **Posterior representation**: ~~Should the posterior over θ_user be explicit (parametric/particle) or implicit?~~ **Resolved in M3**: Both `GaussianPosterior` (parametric) and `ParticlePosterior` (particle-based) are implemented behind a shared `PosteriorBase` ABC. Particle posterior is the default. Implicit (LLM-internal) representation remains a future Option B exploration.

2. **Identifiability under confounding**: γ, α, and λ produce overlapping behavioral signatures. What is the minimum number of carefully designed diagnostic queries needed to disentangle them? Is there a theoretical lower bound from Bayesian experimental design? **Partially addressed in M3**: Diagnostic scenarios targeting each parameter individually are implemented, but gamma is difficult to identify from single-period choices because it acts as a uniform discount that cancels in relative comparisons. Multi-period scenario design is needed to fully disentangle gamma.

3. **Non-stationary preferences**: If a user's θ drifts within or across sessions, how should the posterior be adapted? Exponential forgetting? Change-point detection?

4. **Synthetic-to-real transfer**: Models trained on synthetic user responses may exploit artifacts of the softmax-rational choice model. How do we ensure robustness to real human decision noise, framing effects, and satisficing behavior?

5. **Evaluation metric design**: The compound metric balancing quality and alignment needs careful design to avoid Goodhart's law — the agent shouldn't learn to sacrifice quality in unmeasured dimensions to boost alignment scores.

6. **Scaling user type dimensionality**: Starting with (γ, α, λ) is tractable. What happens as we add parameters (ambiguity aversion, present bias, social preferences)? Does the active learning budget scale linearly or worse?

7. **EIG estimation fidelity**: Initial benchmarks (M3) show marginal active-vs-random improvement (0.8% error reduction) with 500 particles and 200 EIG samples. Scaling to 2000+ particles and 1000+ EIG samples on CURC A100s is needed to validate the 30% efficiency target. The nested MC estimator may also benefit from variance reduction techniques (antithetic sampling, control variates).

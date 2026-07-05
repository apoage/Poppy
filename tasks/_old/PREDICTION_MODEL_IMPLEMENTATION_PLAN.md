# Prediction Model Implementation Plan

## 3rd Component: NeuralMarketODE with Predictive Tree Search

**Based on**: arXiv:2512.11525v1 (NeuralOGCM)
**Target**: Hybrid Physics-Neural price prediction integrated with MuZero+TRM agent
**Updated**: 2025-12-29 (Incorporating Advanced Physics & Stability Improvements)

---

## Executive Summary

Implement a **Prediction Model** as the 3rd pillar of the trading agent:

| Component | Role | Status |
|-----------|------|--------|
| **TRM** | Representation + Reasoning | DONE |
| **MCTS** | Planning (action selection) | DONE |
| **NeuralMarketODE** | Prediction (price forecasting) | NEW |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        TRADING AGENT v2                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────────┐ │
│  │     TRM     │   │    MCTS     │   │      NeuralMarketODE        │ │
│  │             │   │             │   │                             │ │
│  │ Encoder     │◄─►│  100 sims   │◄─►│ Physics Core + Neural       │ │
│  │ Recursive   │   │  PUCT       │   │ + Predictive Tree Search    │ │
│  │ Refinement  │   │             │   │                             │ │
│  └─────────────┘   └─────────────┘   └─────────────────────────────┘ │
│         │                 │                       │                   │
│         └─────────────────┴───────────────────────┘                   │
│                           │                                           │
│                  Shared Latent Space (64-dim)                         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Critical Improvements (v2 Specification)

| Improvement | Mechanism | Benefit |
|-------------|-----------|---------|
| **Log-Space Dynamics** | Predict $\Delta \ln(p)$ instead of $\Delta p$ | Numerical stability, non-negative prices, stationary returns |
| **Action Conditioning** | $G_{neural}(p, X, \mathbf{action})$ | Models market impact ("If I buy, price moves up") |
| **Low-Rank Correlation** | $\Gamma = L L^T$ (Rank $k \ll N$) | Reduces params (4096 $\to$ 512), learns asset clusters |
| **GARCH Volatility** | $\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$ | Captures volatility clustering (high vol follows high vol) |
| **Multi-Step Loss** | $\sum \gamma^t \| \hat{p}_{t+k} - p_{t+k} \|^2$ | Enforces temporal consistency, prevents compounding errors |

---

## Implementation Phases

### Phase 1: Core Prediction Model
**Goal**: Standalone NeuralMarketODE that predicts prices

| Task | Description | Deps | Est. Time |
|------|-------------|------|-----------|
| TASK_008 | Differentiable Market Physics (Log-Returns, GARCH, Low-Rank) | None | 3h |
| TASK_009 | Neural Market Corrector (Action-Conditioned, Cross-Asset Attn) | None | 3h |
| TASK_010 | NeuralMarketODE + Log-Space Solver | 008, 009 | 2h |

**Deliverable**: Working prediction model trained on historical data

### Phase 2: MuZero Integration
**Goal**: Prediction model enhances MuZero's latent space

| Task | Description | Deps | Est. Time |
|------|-------------|------|-----------|
| TASK_011 | Auxiliary Prediction Loss (Multi-Step, Scheduled Sampling) | 010 | 4h |

**Deliverable**: MuZero trained with auxiliary prediction loss

### Phase 3: Predictive Tree Search
**Goal**: Monte Carlo-style exploration of future price scenarios

| Task | Description | Deps | Est. Time |
|------|-------------|------|-----------|
| TASK_012 | Predictive Tree Search (Physics-Aware Branching) | 010, 011 | 4h |

**Deliverable**: MCTS enhanced with 243-scenario price forecasts

---

## Mathematical Foundation

### Core Equation (Log-Space)

Let $x_t = \ln(p_t)$. The dynamics evolve the log-price:

$$\frac{d x}{dt} = G_{phys}(x, r; \theta_p) + G_{neural}(x, X, a; \theta_n)$$

### Market Physics Components (Advanced)

| Component | Formula | Learnable Params |
|-----------|---------|------------------|
| **Momentum** | $G = \alpha \cdot r_t$ | $\alpha$ |
| **Mean Reversion** | $G = -\beta(x_t - \bar{x}_t)$ | $\beta$ |
| **Volatility** | $G = \sigma_t \cdot \epsilon$ | $\omega, \alpha_{g}, \beta_{g}$ (GARCH) |
| **Correlation** | $G = (LL^T) \cdot x$ | $L$ (N×k matrix) |

### ODE Integration (Log-Space)

$$x_{t+1} = x_t + \Delta t \cdot (G_{phys} + G_{neural})$$
$$p_{t+1} = \exp(x_{t+1})$$

---

## Performance Targets

Based on benchmarks (2025-12-29):

| Metric | Target | Achieved |
|--------|--------|----------|
| Forward pass | <1 ms | 0.12 ms |
| 200-step rollout | <10 ms | 8.4 ms |
| 243-scenario tree | <1 ms | 0.28 ms |
| Training memory | <100 MB | 13 MB |
| Parameter overhead | <20% of MuZero | 15% |

---

## File Structure

```
mcts_trader/jax_agent/
├── muzero_model.py          # Existing MuZero
├── trm_flax.py              # Existing TRM
├── prediction/              # NEW: Prediction module
│   ├── __init__.py
│   ├── physics_core.py      # TASK_008: DifferentiableMarketPhysics
│   ├── neural_corrector.py  # TASK_009: NeuralMarketCorrector
│   ├── market_ode.py        # TASK_010: NeuralMarketODE
│   └── tree_search.py       # TASK_012: PredictiveTreeSearch
└── trainer.py               # TASK_011: Add auxiliary loss
```

---

## Integration Points

### 1. With MuZero (TASK_011)

```python
# In trainer.py
def compute_loss(params, batch):
    # ... MuZero losses ...

    # NEW: Multi-step prediction auxiliary loss
    loss_prediction = multi_step_rollout_loss(
        model, latent_states, actions, target_prices, n_steps=5
    )

    return loss_muzero + λ * loss_prediction
```

### 2. With MCTS (TASK_012)

```python
# In mcts_search.py
def evaluate_node(state, action):
    next_state = dynamics(state, action)

    # NEW: Run predictive tree with PHYSICS-AWARE branching
    # (Scale noise by GARCH volatility sigma_t)
    future_scenarios = predictive_tree(next_state, sigma=sigma_t)
    
    risk = future_scenarios.std()
    
    # Dynamics reward penalty
    return reward - risk_lambda * risk
```

---

## Testing Strategy

### Unit Tests
- Physics core gradient flow (ensure softplus works)
- GARCH volatility updates correctly
- Log-space integration preserves price positivity

### Integration Tests
- Prediction improves with training (MSE drops)
- Action conditioning works (buy -> price up impact)
- Tree search produces valid distributions

### Benchmarks
- Compare with/without prediction model
- Measure trading performance improvement
- Profile memory during training

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| OOM during training | Use single-step training (no rollout backprop) |
| Physics params diverge | Softplus constraints, gradient clipping |
| Neural corrector dominates | Balance loss weights, monitor G_phys vs G_neural |
| Tree search too slow | Pre-compile for fixed depths, batch predictions |

---

## Success Criteria

1. **Phase 1**: Prediction model achieves lower MSE than naive baseline
2. **Phase 2**: MuZero with prediction outperforms MuZero without
3. **Phase 3**: MCTS with tree search shows better risk-adjusted returns

---

## Timeline

| Phase | Tasks | Start | End |
|-------|-------|-------|-----|
| Phase 1 | 008-010 | Day 1 | Day 2 |
| Phase 2 | 011 | Day 3 | Day 3 |
| Phase 3 | 012 | Day 4 | Day 4 |
| Testing | All | Day 5 | Day 5 |

**Total**: ~5 working days

---

## References

- [NeuralOGCM Paper](arXiv:2512.11525v1)
- [Deep Analysis](/home/lukas/Trader/docs/research/PREDICTION_MODEL_ANALYSIS.md)
- [Design Document](/home/lukas/Trader/docs/research/NEURAL_MARKET_PHYSICS_DESIGN.md)
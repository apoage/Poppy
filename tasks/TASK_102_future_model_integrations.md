# TASK_102: Future Model Integrations (Research)

## Status: RESEARCH
## Priority: FUTURE (not immediate)

## Overview

This task tracks potential future model architectures and integration patterns. These are expensive, experimental, or require significant research before implementation.

## Current System

```
TRM+MuZero (AlphaZero-style)
├── TRM encoder (small, ~7-36M params)
├── MCTS via mctx (GPU-accelerated)
├── Physics prediction head (implemented)
└── Runs on single GPU ✓
```

---

## Future Model 1: MarketDiffusion

**Document**: `/home/lukas/Trader/docs/market_diffusion_architecture.md`

### Concept
- Video diffusion model for market prediction
- Deliberate memorization of all history into weights
- Uncertainty native through denoising process
- Generates distributions, not point estimates

### Caveats
- **Does NOT fit AlphaZero paradigm** - separate heavy model
- **Expensive to train** - needs dedicated GPU(s)
- **Inference**: Maybe fits on single GPU, but still heavy
- **Different architecture**: DiT backbone, not TRM

### Potential Value
- Native uncertainty quantification
- Better distribution modeling for MCTS
- Regime-aware predictions

### Integration Options (Future Research)

**Option A: Separate Oracle**
```
MarketDiffusion (heavy, offline)
        ↓ predictions
TRM+MuZero (light, real-time)
        ↓ decisions
Trading
```

**Option B: Distillation**
```
Train MarketDiffusion (heavy)
        ↓ distill knowledge
Lightweight student model
        ↓ integrate
TRM+MuZero dynamics head
```

**Option C: Hybrid Inference**
```
Physics prediction (fast, baseline)
        +
Diffusion samples (slow, distribution)
        +
MCTS planning (uses both)
```

---

## Future Model 2: Physics-Diffusion-MCTS Hybrid

### Concept

Tighter integration of three components:

```
┌─────────────────────────────────────────────────────────────────┐
│                 Physics-Diffusion-MCTS Hybrid                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │   Physics    │   │  Diffusion   │   │    MCTS      │        │
│  │   Core       │   │  Corrector   │   │   Planning   │        │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘        │
│         │                  │                  │                 │
│         ↓                  ↓                  ↓                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Integrated Prediction                    │      │
│  │                                                       │      │
│  │  G_total = G_physics + G_diffusion_correction        │      │
│  │                                                       │      │
│  │  Physics: Fast, interpretable baseline               │      │
│  │  Diffusion: Learned residuals + uncertainty          │      │
│  │  MCTS: Plans over distribution of outcomes           │      │
│  │                                                       │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Current State

**Already Implemented:**
- Physics core: `mcts_trader/jax_agent/prediction/physics_core.py`
- Neural corrector: `mcts_trader/jax_agent/prediction/neural_corrector.py`
- Market ODE: `mcts_trader/jax_agent/prediction/market_ode.py`
- MCTS: `mcts_trader/jax_agent/mctx_search.py`

**Not Yet Integrated:**
- Diffusion-based uncertainty
- Physics residuals feeding into MCTS tree expansion
- Multi-sample MCTS using distribution

### Wild Combination Ideas

**Idea 1: Physics-Guided Diffusion**
```python
# Physics provides prior, diffusion learns residuals
physics_pred = physics_core(state)  # Baseline prediction
diffusion_correction = diffusion_model.denoise(
    state,
    condition=physics_pred  # Guide diffusion with physics
)
final_pred = physics_pred + diffusion_correction
```

**Idea 2: Diffusion-Sampled MCTS**
```python
# MCTS tree expansion uses diffusion samples
def mcts_expand(state, action):
    # Instead of single dynamics prediction:
    # next_state = dynamics(state, action)

    # Sample multiple futures from diffusion:
    futures = diffusion.sample(state, action, n_samples=10)

    # Weight by likelihood, expand tree probabilistically
    for future, prob in futures:
        tree.add_child(future, weight=prob)
```

**Idea 3: Physics as Diffusion Backbone**
```python
# Physics equations as inductive bias in diffusion architecture
class PhysicsInformedDiT:
    def forward(self, x, t):
        # Standard diffusion path
        diffusion_out = self.transformer(x, t)

        # Physics constraint layer
        physics_out = self.physics_core(x)

        # Combine: diffusion respects physics
        return diffusion_out + physics_out * self.physics_weight
```

**Idea 4: Monte Carlo Physics Ensemble**
```python
# Sample physics parameters, run ensemble
def physics_ensemble_prediction(state, n_samples=100):
    predictions = []
    for _ in range(n_samples):
        # Sample physics parameters (α, β, σ, Γ)
        params = sample_physics_params(state)
        pred = physics_forward(state, params)
        predictions.append(pred)

    # Return distribution of physics predictions
    return Distribution(predictions)
```

---

## Future Model 3: Hierarchical World Model

### Concept

Multiple time-scale world models:

```
┌─────────────────────────────────────────┐
│         Hierarchical World Model         │
├─────────────────────────────────────────┤
│                                         │
│  Macro Model (weeks/months)             │
│  - Regime prediction                    │
│  - Trend direction                      │
│  - Updated daily                        │
│           ↓                             │
│  Meso Model (days)                      │
│  - Daily range prediction               │
│  - Volatility forecast                  │
│  - Updated hourly                       │
│           ↓                             │
│  Micro Model (minutes/hours)            │
│  - Entry/exit timing                    │
│  - Order flow prediction                │
│  - Real-time updates                    │
│                                         │
└─────────────────────────────────────────┘
```

Each level conditions the level below.

---

## Hardware Requirements

| Model | Training | Inference | Current Fit |
|-------|----------|-----------|-------------|
| TRM+MuZero | 1 GPU | 1 GPU | ✓ Yes |
| MarketDiffusion | 2-8 GPUs | 1 GPU (maybe) | ✗ Future |
| Physics-Diffusion Hybrid | 1-2 GPUs | 1 GPU | ~ Research |
| Hierarchical World Model | 2-4 GPUs | 1 GPU | ✗ Future |

---

## Research Questions

1. **Can diffusion uncertainty improve MCTS?**
   - Does knowing "high uncertainty" help exploration?
   - Does distribution-aware tree search beat point estimates?

2. **Physics + Diffusion synergy?**
   - Does physics prior help diffusion converge?
   - Does diffusion correction improve physics baseline?

3. **Compute vs Alpha trade-off?**
   - Is heavy model worth it if alpha > compute cost?
   - Can distillation capture most value in smaller model?

4. **Real-time feasibility?**
   - Can hybrid model meet latency requirements?
   - What's minimum viable inference time?

---

## Next Steps (When Ready)

1. **Proof of concept**: Train small diffusion model on synthetic data
2. **Compare**: Diffusion vs physics vs combined on prediction metrics
3. **Integration test**: Feed diffusion samples to MCTS
4. **Benchmark**: Measure alpha vs compute cost

---

## Related Tasks

- TASK_101: Daily learning pipeline (adaptation mechanism)
- TASK_032: Physics prediction scaling
- TASK_035: Physics components fix

## Related Documents

- `/home/lukas/Trader/docs/market_diffusion_architecture.md`
- `/home/lukas/Trader/lore/` (design documents)

# TASK_035: Physics Components Learning Fix

## Status: IMPLEMENTED (2026-01-04)

## Problem Statement

The Neural ODE Physics components (G_momentum, G_reversion, G_volatility, G_correlation) are collapsing to near-zero values during training, making the physics-informed prediction head ineffective.

### Observed Symptoms

1. **GARCH Persistence (α+β)**: Dropped from 1.0 to 0.096
   - Should be ~0.9-1.0 for realistic volatility clustering
   - At 0.096, model says "no volatility persistence" (wrong for financial data)

2. **G_momentum and G_reversion**: Near zero and falling
   - Cannot register on visualization despite having non-zero α/β in learned params
   - The physics core is being bypassed

3. **PhysReg: 0.000000**: Physics regularization not contributing to loss
   - Regularization term has no gradient signal

4. **Neural corrector dominance**: The STE (Spatio-Temporal Evolution) neural corrector is "eating" all the signal, making physics redundant

---

## Root Cause Analysis (2026-01-04 Deep Investigation)

### CONFIRMED ROOT CAUSE #1: One-Sided Regularization (THE SMOKING GUN)

In `trainer.py:471-479`:
```python
# Alpha: Only penalizes values ABOVE 0.3
reg_loss = reg_loss + jnp.mean(jnp.maximum(alpha - 0.3, 0.0) ** 2)

# Beta: Only penalizes values ABOVE 0.2
reg_loss = reg_loss + jnp.mean(jnp.maximum(beta - 0.2, 0.0) ** 2)
```

This is a **ceiling-only** regularization with ZERO floor:
| α value | Penalty |
|---------|---------|
| 0.00 | max(0.00 - 0.3, 0)² = **0** |
| 0.10 | max(0.10 - 0.3, 0)² = **0** |
| 0.20 | max(0.20 - 0.3, 0)² = **0** |
| 0.30 | max(0.30 - 0.3, 0)² = **0** |
| 0.50 | max(0.50 - 0.3, 0)² = **0.04** |

**The loss function actively incentivizes α and β to collapse to zero!**
There is no penalty for low values, only for high values.

### CONFIRMED ROOT CAUSE #2: GARCH Persistence Regularization Also One-Sided

In `physics_core.py:406-408`:
```python
# Penalize if α + β > 0.99 (near non-stationarity)
persistence = alpha_g + beta_g
reg_loss += jnp.mean(jnp.maximum(persistence - 0.99, 0.0) ** 2)
```

This only prevents persistence from going ABOVE 0.99 for stationarity. No floor penalty:
| α+β | Penalty |
|-----|---------|
| 0.096 | max(0.096 - 0.99, 0)² = **0** |
| 0.500 | max(0.500 - 0.99, 0)² = **0** |
| 0.900 | max(0.900 - 0.99, 0)² = **0** |
| 1.100 | max(1.100 - 0.99, 0)² = **0.012** |

**Model finds α+β ≈ 0.1 is "free" while α+β ≈ 1.0 has penalty risk!**

### CONFIRMED ROOT CAUSE #3: Neural Corrector Learns Faster

The neural corrector is a **universal function approximator** (transformer with attention):
- Thousands of parameters with large gradient surface
- Can fit ANY pattern quickly

Physics core has:
- 4 parameters per asset (α, β, σ, loadings)
- Constrained parameterization via softplus
- Tiny gradient surface

**Result**: By the time physics starts converging, neural has already explained the error to zero. Physics gets zero gradient signal (loss is already minimized).

### WHY PhysReg: 0.000000

When α → 0 and β → 0, the one-sided regularization computes:
```python
max(0.0 - 0.3, 0.0)² = 0.0
max(0.0 - 0.2, 0.0)² = 0.0
```
The regularization returns exactly 0.0 because the values are BELOW the ceilings!

---

## The Fix: Three-Part Solution

### Part 1: Two-Sided Target Regularization (REQUIRED)

Change from "ceiling" to "target" with bidirectional penalty:

```python
# trainer.py:_compute_physics_regularization

# OLD (broken - ceiling only):
reg_loss = reg_loss + jnp.mean(jnp.maximum(alpha - 0.3, 0.0) ** 2)

# NEW (two-sided target):
alpha_target = 0.05  # Prior belief: small positive momentum
reg_loss = reg_loss + jnp.mean((alpha - alpha_target) ** 2) * 0.5

# Similar for beta:
beta_target = 0.05  # Prior belief: small mean reversion
reg_loss = reg_loss + jnp.mean((beta - beta_target) ** 2) * 0.5
```

### Part 2: GARCH Persistence Floor (REQUIRED)

Add penalty for persistence BELOW target:

```python
# physics_core.py:physics_regularization_loss

# OLD (ceiling only):
reg_loss += jnp.mean(jnp.maximum(persistence - 0.99, 0.0) ** 2)

# NEW (floor + ceiling):
persistence_target = 0.95  # Target high persistence
floor_penalty = jnp.mean(jnp.maximum(0.85 - persistence, 0.0) ** 2)  # Floor at 0.85
ceiling_penalty = jnp.mean(jnp.maximum(persistence - 1.0, 0.0) ** 2)  # Keep < 1.0
reg_loss += (floor_penalty + ceiling_penalty) * 2.0  # Strong weight
```

### Part 3: Neural Corrector Scaling (RECOMMENDED)

Either:

**Option A - Neural Dropout** (simpler):
```python
# market_ode.py:compute_tendency
if not self.physics_only:
    # 50% dropout on neural corrector during training
    dropout_mask = jax.random.bernoulli(rng, 0.5, ())
    G_neural = jnp.where(dropout_mask, 0.0, G_neural)
```

**Option B - Annealed Neural Scale** (more controllable):
```python
# market_ode.py:__init__
neural_scale: float = 0.0  # Start at 0 (physics only)

# During training, anneal neural_scale from 0 → 1 over warmup epochs
neural_scale_annealed = jnp.minimum(step / warmup_steps, 1.0)
G_neural = G_neural * neural_scale_annealed
```

---

## Implementation Priority

| Order | Component | Complexity | Impact |
|-------|-----------|------------|--------|
| 1 | Two-sided α/β regularization | Simple | HIGH |
| 2 | GARCH persistence floor | Simple | HIGH |
| 3 | Neural dropout/scaling | Moderate | MEDIUM |

**Recommended order**: Fix regularization first (1 & 2), then evaluate if neural scaling (3) is still needed.

## Files to Modify

| File | Change |
|------|--------|
| `mcts_trader/jax_agent/trainer.py:471-484` | Two-sided target regularization |
| `mcts_trader/jax_agent/prediction/physics_core.py:400-408` | GARCH persistence floor |
| `mcts_trader/jax_agent/prediction/market_ode.py:203-212` | Neural dropout (optional) |

## Verification

After fix:
1. **PhysReg should be non-zero**: ~0.001-0.01 (regularization pulling toward targets)
2. **GARCH persistence**: Should stay in [0.85, 1.0] range
3. **G_momentum/G_reversion**: Should show on visualization
4. **α values**: Should stabilize around 0.04-0.10 (not collapse to 0)
5. **Sine wave test**: Physics should explain 50%+ of prediction variance

---

---

## Implementation (2026-01-04)

### Changes Made

| File | Change |
|------|--------|
| `mcts_trader/jax_agent/trainer.py:476-516` | Two-sided target regularization for α/β + GARCH floor |
| `mcts_trader/jax_agent/prediction/physics_core.py:354-424` | Same two-sided regularization in standalone function |
| `mcts_trader/jax_agent/prediction/market_ode.py:161-244` | Added `neural_scale` parameter for warmup |
| `mcts_trader/jax_agent/muzero_model.py:302-397` | Pass `neural_scale` through predict_multi_step |
| `train_jax_fast.py:190-191,470-471,483` | Added `--physics-warmup-steps` CLI arg |

### Key Implementation Details

1. **Two-sided Target Regularization**:
   ```python
   # OLD (broken):
   reg_loss += jnp.mean(jnp.maximum(alpha - 0.3, 0.0) ** 2)  # α=0 → 0 penalty

   # NEW (fixed):
   alpha_target = 0.05
   reg_loss += jnp.mean((alpha - alpha_target) ** 2) * 0.5  # α=0 → penalty!
   ```

2. **GARCH Floor + Ceiling**:
   ```python
   floor_penalty = jnp.mean(jnp.maximum(0.85 - persistence, 0.0) ** 2)
   ceiling_penalty = jnp.mean(jnp.maximum(persistence - 1.0, 0.0) ** 2)
   reg_loss += (floor_penalty + ceiling_penalty) * 2.0
   ```

3. **Neural Warmup**:
   - `--physics-warmup-steps 5000` (default)
   - Neural scale anneals from 0 → 1 over warmup
   - Physics gets gradient signal before neural takes over

### CLI Usage

```bash
# Default: 5000-step physics warmup
python train_jax_fast.py --use-prediction-loss --physics-warmup-steps 5000

# Disable warmup (old behavior)
python train_jax_fast.py --use-prediction-loss --physics-warmup-steps 0

# Longer warmup for more physics learning
python train_jax_fast.py --use-prediction-loss --physics-warmup-steps 10000
```

---

## Related

- TASK_008 through TASK_012: Original prediction model implementation
- The prediction loss is working (0.025-0.027) but physics is not contributing
- The model achieves good RL performance (+100% alpha) but physics interpretability is lost

## Priority

**COMPLETED**: Fixed the fundamental design flaw that made physics learning impossible.

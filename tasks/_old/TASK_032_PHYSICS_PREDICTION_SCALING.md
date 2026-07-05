# TASK_032: Scale Up Physics Prediction Impact

## Status: TODO

## Problem Statement

Physics-based prediction model (NeuralOGCM-inspired) is enabled but its impact on training is barely visible. Current weights are conservative:

| Parameter | Current | Issue |
|-----------|---------|-------|
| `--prediction-loss-weight` | 0.2 | Final weight after annealing |
| `--prediction-loss-initial` | 0.01 | Starting weight (very small) |
| `--prediction-loss-anneal-steps` | 10000 | Slow ramp-up |
| `--physics-reg-weight` | 0.01 | Physics regularization |

The prediction loss contribution is dwarfed by value/policy losses, making it hard to see if physics predictions are helping.

## Physics Model Components

From `mcts_trader/jax_agent/prediction/physics_core.py`:

```python
class PhysicsOutput:
    G_momentum: jnp.ndarray    # Trend following tendency
    G_reversion: jnp.ndarray   # Mean reversion tendency
    G_volatility: jnp.ndarray  # Volatility clustering
    G_correlation: jnp.ndarray # Cross-asset correlation pull
    G_total: jnp.ndarray       # Combined tendency
```

These physics terms model:
- **Momentum**: Price trends persist (α)
- **Reversion**: Prices revert to moving average (β)
- **Volatility**: GARCH-like vol clustering (σ)
- **Correlation**: Cross-asset factor structure (Γ)

## Proposed Changes

### Phase 1: Increase Prediction Loss Weight

```bash
# Old (too conservative)
--prediction-loss-weight 0.2 --prediction-loss-initial 0.01

# New (more impactful)
--prediction-loss-weight 0.5 --prediction-loss-initial 0.1
```

This gives prediction loss 50% weight vs value loss (currently 20%).

### Phase 2: Faster Annealing

```bash
# Old (10k steps to reach full weight)
--prediction-loss-anneal-steps 10000

# New (2k steps - faster ramp)
--prediction-loss-anneal-steps 2000
```

### Phase 3: Physics Regularization Increase

```bash
# Old
--physics-reg-weight 0.01

# New (10x stronger)
--physics-reg-weight 0.1
```

This encourages physics parameters (α, β, σ, Γ) to stay in reasonable ranges.

### Phase 4: Physics-Only Mode Testing

For diagnostic purposes, try pure physics without neural corrector:

```bash
python train_jax_fast.py \
    --synthetic-level 0 \
    --physics-only \
    --prediction-loss-weight 1.0 \
    --prediction-loss-initial 0.5 \
    --name physics_only_test
```

This isolates physics model impact.

## Implementation

### Option A: Update Defaults in dojo_dynamic_curriculum.py

```python
# Current command building
cmd.extend([
    '--use-prediction-loss',
    '--prediction-loss-weight', '0.2',
    '--pred-head-size', 'large',
])

# New (scaled up)
cmd.extend([
    '--use-prediction-loss',
    '--prediction-loss-weight', '0.5',
    '--prediction-loss-initial', '0.1',
    '--prediction-loss-anneal-steps', '2000',
    '--physics-reg-weight', '0.1',
    '--pred-head-size', 'large',
])
```

### Option B: Add CLI Presets

```python
# train_jax_fast.py
parser.add_argument('--physics-preset', choices=['conservative', 'balanced', 'aggressive'],
                    default='balanced', help='Physics prediction preset')

# Presets:
PHYSICS_PRESETS = {
    'conservative': {
        'prediction_loss_weight': 0.2,
        'prediction_loss_initial': 0.01,
        'prediction_loss_anneal_steps': 10000,
        'physics_reg_weight': 0.01,
    },
    'balanced': {
        'prediction_loss_weight': 0.5,
        'prediction_loss_initial': 0.1,
        'prediction_loss_anneal_steps': 2000,
        'physics_reg_weight': 0.05,
    },
    'aggressive': {
        'prediction_loss_weight': 1.0,
        'prediction_loss_initial': 0.3,
        'prediction_loss_anneal_steps': 1000,
        'physics_reg_weight': 0.1,
    },
}
```

## Files to Modify

| File | Changes |
|------|---------|
| `dojo_dynamic_curriculum.py` | Update prediction loss defaults |
| `dojo_manager.py` | Update prediction loss defaults |
| `train_jax_fast.py` | (Optional) Add presets |

## Logging Improvements

Add physics-specific metrics to CSV:

```python
# DS24 Level 3+ already logs physics, but ensure visibility:
csv_columns += [
    'G_momentum_mean', 'G_reversion_mean',
    'G_volatility_mean', 'G_correlation_mean',
    'physics_loss', 'neural_corrector_loss',
]
```

## Validation

After scaling up, check:

1. **Prediction loss visible in training logs**:
   ```
   Loss: 0.5 | Val: 0.3 | Pol: 0.1 | Pred: 0.08 (was: 0.01)
   ```

2. **Physics parameters learning**:
   ```
   G_momentum: -0.02 → +0.15 (learning trend)
   G_reversion: +0.01 → -0.08 (learning mean reversion)
   ```

3. **Correlation with returns**:
   - Physics predictions should correlate with next-step returns
   - Scatter plot: G_total vs actual_return

## Risk Considerations

| Risk | Mitigation |
|------|------------|
| Prediction loss dominates | Start with balanced preset, monitor |
| Physics divergence | physics_reg_weight keeps params bounded |
| Slower convergence | Acceptable trade-off for better predictions |

## Recommended Starting Config

For dojo training with visible physics impact:

```bash
python dojo_dynamic_curriculum.py --level 0 --model-size small \
    # Existing params...

# With these prediction settings in the script:
--use-prediction-loss
--prediction-loss-weight 0.5
--prediction-loss-initial 0.1
--prediction-loss-anneal-steps 2000
--physics-reg-weight 0.05
--pred-head-size large
```

## Priority

**MEDIUM** - Diagnostic improvement. Should be done after TASK_031 (PopArt denorm) so we can properly evaluate physics predictions against denormalized values.

## Dependencies

- TASK_031: PopArt denormalization (needed to compare physics predictions to actual returns)
- TASK_008-012: Physics prediction implementation (COMPLETE)

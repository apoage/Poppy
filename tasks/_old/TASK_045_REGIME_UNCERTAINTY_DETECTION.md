# TASK_045: Regime Uncertainty Detection ("I Don't Know" State)

## Overview

Add explicit uncertainty quantification so the agent knows when it's operating outside its training distribution. When uncertainty is high, the agent should signal "I don't know" rather than making confident but wrong decisions.

## Motivation

> "Strong systems fail quietly. Weak systems fail loudly."

MuZero-style planning produces confident decisions even when the learned dynamics are wrong. The agent needs epistemic humility - the ability to recognize regime shifts and novel situations.

## Approaches

### Option A: Ensemble Disagreement (Recommended)

Train multiple prediction heads, measure disagreement as uncertainty.

```python
class EnsemblePredictionHead(nn.Module):
    """Multiple prediction heads for uncertainty estimation."""
    n_heads: int = 5

    @nn.compact
    def __call__(self, latent_state, features, noise):
        predictions = []
        for i in range(self.n_heads):
            pred_i = PredictionHead(name=f'head_{i}')(latent_state, features, noise)
            predictions.append(pred_i)

        # Stack predictions: (n_heads, batch, n_assets)
        preds = jnp.stack(predictions, axis=0)

        # Mean prediction
        mean_pred = jnp.mean(preds, axis=0)

        # Uncertainty = std across heads (epistemic uncertainty)
        uncertainty = jnp.std(preds, axis=0)

        return mean_pred, uncertainty

# Usage in MCTS
def should_defer(uncertainty, threshold=0.05):
    """High uncertainty = agent should defer to safety policy."""
    return jnp.mean(uncertainty) > threshold
```

**Pros**: Direct measure of model disagreement, well-understood
**Cons**: 5x prediction head compute, memory overhead

### Option B: Monte Carlo Dropout

Use dropout at inference time, run multiple forward passes.

```python
def mc_dropout_uncertainty(model, params, obs, n_samples=10, dropout_rate=0.1):
    """Estimate uncertainty via MC Dropout."""
    predictions = []
    for _ in range(n_samples):
        # Enable dropout during inference
        pred = model.apply(params, obs, training=True, rngs={'dropout': rng})
        predictions.append(pred)

    uncertainty = jnp.std(jnp.stack(predictions), axis=0)
    return uncertainty
```

**Pros**: No extra parameters, easy to add
**Cons**: Multiple forward passes, dropout may hurt performance

### Option C: Learned Uncertainty Head

Add explicit uncertainty output to prediction head.

```python
class UncertaintyAwarePrediction(nn.Module):
    @nn.compact
    def __call__(self, latent_state):
        # Predict mean and log_variance
        hidden = nn.Dense(256)(latent_state)
        hidden = nn.relu(hidden)

        mean = nn.Dense(n_assets)(hidden)
        log_var = nn.Dense(n_assets)(hidden)  # Learned aleatoric uncertainty

        return mean, jnp.exp(log_var)
```

**Pros**: Single forward pass, learned
**Cons**: May underestimate epistemic uncertainty, needs calibration

### Option D: Distribution Shift Detection (OOD)

Compare current observations to training distribution.

```python
class DistributionShiftDetector:
    """Detect when observations are out-of-distribution."""

    def __init__(self, latent_dim):
        self.running_mean = jnp.zeros(latent_dim)
        self.running_var = jnp.ones(latent_dim)
        self.count = 0

    def update(self, latent_states):
        """Update running statistics during training."""
        batch_mean = jnp.mean(latent_states, axis=0)
        batch_var = jnp.var(latent_states, axis=0)
        # Welford's online algorithm...

    def mahalanobis_distance(self, latent_state):
        """How far is this state from training distribution?"""
        diff = latent_state - self.running_mean
        return jnp.sqrt(jnp.sum(diff ** 2 / self.running_var))

    def is_ood(self, latent_state, threshold=3.0):
        """True if state is >3 sigma from training distribution."""
        return self.mahalanobis_distance(latent_state) > threshold
```

**Pros**: Principled, no architectural changes
**Cons**: Latent space may not capture all distribution shifts

## Integration Points

### 1. MCTS Value Adjustment

```python
# In mctx_search.py
def _categorical_to_scalar(self, logits, uncertainty=None):
    value = super()._categorical_to_scalar(logits)

    if uncertainty is not None and self.use_uncertainty_penalty:
        # Penalize value under high uncertainty
        value = value - self.uncertainty_lambda * uncertainty

    return value
```

### 2. Action Override

```python
# In self_play.py
def select_action(obs, model, params, search):
    action, policy, stats = search(params, rng, obs)

    # Check uncertainty
    uncertainty = compute_uncertainty(model, params, obs)

    if uncertainty > UNCERTAINTY_THRESHOLD:
        # Override with safe action
        return HOLD_ACTION, policy, {'deferred': True, 'uncertainty': uncertainty}

    return action, policy, stats
```

### 3. Logging & Monitoring

```python
# New CSV columns
'uncertainty_mean', 'uncertainty_max', 'pct_deferred', 'ood_distance'
```

## CLI Arguments

```python
parser.add_argument('--uncertainty-method', choices=['ensemble', 'dropout', 'learned', 'ood'],
                   default='ood', help='Uncertainty estimation method')
parser.add_argument('--uncertainty-threshold', type=float, default=0.05,
                   help='Threshold to trigger "I don\'t know" state')
parser.add_argument('--uncertainty-penalty', type=float, default=0.1,
                   help='Value penalty per unit uncertainty in MCTS')
parser.add_argument('--n-ensemble-heads', type=int, default=5,
                   help='Number of ensemble heads (if using ensemble method)')
```

## Metrics to Track

| Metric | Description |
|--------|-------------|
| `uncertainty_mean` | Average uncertainty across episode |
| `uncertainty_max` | Maximum uncertainty (regime shift indicator) |
| `pct_deferred` | % of actions where agent deferred |
| `ood_distance` | Mahalanobis distance from training distribution |
| `uncertainty_calibration` | Correlation of uncertainty with actual error |

## Success Criteria

1. Uncertainty correlates with actual prediction error (calibration)
2. Uncertainty spikes during synthetic regime changes
3. Deferral rate reasonable (5-20%, not 0% or 80%)
4. No significant performance drop when uncertainty is low

## Implementation Priority

1. **Phase 1**: OOD detection (simplest, no arch changes)
2. **Phase 2**: Ensemble heads (if OOD insufficient)
3. **Phase 3**: Integration with MCTS value penalty
4. **Phase 4**: Action override / deferral logic

## Files to Modify/Create

- [ ] `mcts_trader/jax_agent/uncertainty.py` (NEW)
- [ ] `mcts_trader/jax_agent/muzero_model.py` (add uncertainty head option)
- [ ] `mcts_trader/jax_agent/mctx_search.py` (uncertainty-adjusted values)
- [ ] `mcts_trader/training/self_play.py` (deferral logic)
- [ ] `train_jax_fast.py` (CLI args, logging)

## References

- [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474)
- [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142)
- [MuZero with Learned Model Uncertainty](https://arxiv.org/abs/2104.06303)

## Status

- [ ] Phase 1: OOD detection
- [ ] Phase 2: Ensemble (optional)
- [ ] Phase 3: MCTS integration
- [ ] Phase 4: Deferral logic

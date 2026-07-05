# TASK_103: Monte Carlo Physics Uncertainty

## Status: PENDING
## Priority: MEDIUM

## Problem

Current physics prediction is deterministic:
```python
α, β, σ, Γ = physics_core(state)
next_price = physics_forward(state, α, β, σ, Γ)  # Single point estimate
```

MCTS plans assuming this single future is correct. No uncertainty awareness.

## Solution

Sample physics parameters to get distribution of futures - **cheap uncertainty** without heavy diffusion model.

```python
def mc_physics_prediction(state, n_samples=32, rng=None):
    """Monte Carlo physics: sample params → distribution of futures."""

    # Get base physics parameters from neural network
    base_params = physics_core(state)  # α, β, σ, Γ

    futures = []
    confidences = []

    for i in range(n_samples):
        # Add noise to physics parameters
        sampled_params = sample_physics_params(base_params, rng)

        # Run physics forward with sampled params
        future = physics_forward(state, sampled_params)
        futures.append(future)

    return MCPhysicsDistribution(
        futures=jnp.stack(futures),
        mean=jnp.mean(futures, axis=0),
        std=jnp.std(futures, axis=0),  # Uncertainty per asset!
        base_params=base_params,
    )
```

## Parameter Sampling Strategies

### Strategy 1: Gaussian Noise
```python
def sample_physics_params(base_params, rng, noise_scale=0.1):
    """Add Gaussian noise to each parameter."""
    α, β, σ, Γ = base_params

    noise = jax.random.normal(rng, shape=(4,)) * noise_scale

    return (
        α * (1 + noise[0]),  # Mean reversion ±10%
        β * (1 + noise[1]),  # Momentum ±10%
        σ * (1 + noise[2]),  # Volatility ±10%
        Γ * (1 + noise[3]),  # Correlation ±10%
    )
```

### Strategy 2: Learned Uncertainty
```python
def sample_physics_params_learned(base_params, uncertainty, rng):
    """Network predicts both params AND uncertainty."""
    α, β, σ, Γ = base_params
    α_std, β_std, σ_std, Γ_std = uncertainty  # Also predicted by network

    noise = jax.random.normal(rng, shape=(4,))

    return (
        α + noise[0] * α_std,
        β + noise[1] * β_std,
        σ + noise[2] * σ_std,
        Γ + noise[3] * Γ_std,
    )
```

### Strategy 3: Dropout at Inference (MC Dropout)
```python
def mc_dropout_physics(state, n_samples=32):
    """Use dropout at inference for uncertainty."""
    futures = []
    for _ in range(n_samples):
        # Run physics_core with dropout ENABLED
        params = physics_core(state, training=True)  # Dropout active
        future = physics_forward(state, params)
        futures.append(future)

    return Distribution(futures)
```

## MCTS Integration

### Option A: Uncertainty-Weighted Exploration
```python
def mcts_with_uncertainty(state, action, mc_distribution):
    """Use uncertainty to guide exploration."""

    mean_future = mc_distribution.mean
    uncertainty = mc_distribution.std

    # High uncertainty → more exploration (higher UCB bonus)
    exploration_bonus = uncertainty * exploration_weight

    # MCTS UCB: Q + c * sqrt(N_parent/N) + uncertainty_bonus
    ucb = q_value + puct_bonus + exploration_bonus
```

### Option B: Multi-Sample Tree Expansion
```python
def mcts_expand_with_samples(state, action, mc_distribution, n_children=3):
    """Expand tree with multiple sampled futures."""

    # Pick representative futures (mean, optimistic, pessimistic)
    futures = [
        mc_distribution.mean,
        mc_distribution.mean + mc_distribution.std,  # Optimistic
        mc_distribution.mean - mc_distribution.std,  # Pessimistic
    ]

    # Each becomes a child node with probability weight
    weights = [0.5, 0.25, 0.25]  # Mean most likely

    for future, weight in zip(futures, weights):
        tree.add_child(state, action, future, prior_weight=weight)
```

### Option C: Value Adjustment
```python
def uncertainty_adjusted_value(state, mc_distribution):
    """Penalize high-uncertainty states in value estimate."""

    base_value = value_network(state)
    uncertainty_penalty = mc_distribution.std.mean() * penalty_weight

    # Risk-averse: prefer certain outcomes
    adjusted_value = base_value - uncertainty_penalty

    return adjusted_value
```

## Implementation

### Modify physics_core.py

```python
@dataclass
class PhysicsOutput:
    # Base parameters
    alpha: jnp.ndarray      # Mean reversion
    beta: jnp.ndarray       # Momentum
    sigma: jnp.ndarray      # Volatility
    gamma: jnp.ndarray      # Correlation

    # NEW: Uncertainty estimates (optional)
    alpha_std: Optional[jnp.ndarray] = None
    beta_std: Optional[jnp.ndarray] = None
    sigma_std: Optional[jnp.ndarray] = None
    gamma_std: Optional[jnp.ndarray] = None


class PhysicsCore(nn.Module):
    predict_uncertainty: bool = False

    @nn.compact
    def __call__(self, state):
        # ... existing physics prediction ...

        params = (alpha, beta, sigma, gamma)

        if self.predict_uncertainty:
            # Additional head for uncertainty
            uncertainty = nn.Dense(4, name='uncertainty_head')(hidden)
            uncertainty = nn.softplus(uncertainty)  # Positive

            return PhysicsOutput(
                *params,
                alpha_std=uncertainty[0],
                beta_std=uncertainty[1],
                sigma_std=uncertainty[2],
                gamma_std=uncertainty[3],
            )

        return PhysicsOutput(*params)
```

### New file: mc_physics.py

```python
"""Monte Carlo Physics Prediction for uncertainty estimation."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MCPhysicsConfig:
    n_samples: int = 32
    noise_scale: float = 0.1
    use_learned_uncertainty: bool = False
    sampling_strategy: str = 'gaussian'  # 'gaussian', 'learned', 'dropout'


@dataclass
class MCPhysicsDistribution:
    futures: jnp.ndarray        # (n_samples, n_assets, n_features)
    mean: jnp.ndarray           # (n_assets, n_features)
    std: jnp.ndarray            # (n_assets, n_features) - uncertainty!
    percentile_5: jnp.ndarray   # Pessimistic
    percentile_95: jnp.ndarray  # Optimistic


def mc_physics_predict(
    physics_core,
    physics_forward,
    state: jnp.ndarray,
    config: MCPhysicsConfig,
    rng: jax.random.PRNGKey,
) -> MCPhysicsDistribution:
    """Generate distribution of futures via Monte Carlo physics sampling."""

    # Get base parameters
    base_output = physics_core(state)

    # Split RNG for each sample
    rngs = jax.random.split(rng, config.n_samples)

    def sample_future(rng):
        # Sample physics parameters
        if config.use_learned_uncertainty:
            sampled = sample_with_learned_std(base_output, rng)
        else:
            sampled = sample_with_noise(base_output, rng, config.noise_scale)

        # Run physics forward
        return physics_forward(state, sampled)

    # Vectorize over samples
    futures = jax.vmap(sample_future)(rngs)

    return MCPhysicsDistribution(
        futures=futures,
        mean=jnp.mean(futures, axis=0),
        std=jnp.std(futures, axis=0),
        percentile_5=jnp.percentile(futures, 5, axis=0),
        percentile_95=jnp.percentile(futures, 95, axis=0),
    )


def sample_with_noise(base_output, rng, noise_scale):
    """Add Gaussian noise to physics parameters."""
    noise = jax.random.normal(rng, shape=(4,)) * noise_scale

    return (
        base_output.alpha * (1 + noise[0]),
        base_output.beta * (1 + noise[1]),
        base_output.sigma * jnp.exp(noise[2] * 0.5),  # Log-normal for positive
        base_output.gamma * (1 + noise[3]),
    )


def sample_with_learned_std(base_output, rng):
    """Sample using network-predicted uncertainty."""
    noise = jax.random.normal(rng, shape=(4,))

    return (
        base_output.alpha + noise[0] * base_output.alpha_std,
        base_output.beta + noise[1] * base_output.beta_std,
        base_output.sigma + noise[2] * base_output.sigma_std,
        base_output.gamma + noise[3] * base_output.gamma_std,
    )
```

## CLI Arguments

```python
parser.add_argument('--mc-physics', action='store_true',
                   help='Enable Monte Carlo physics uncertainty')
parser.add_argument('--mc-physics-samples', type=int, default=32,
                   help='Number of MC samples for uncertainty')
parser.add_argument('--mc-physics-noise', type=float, default=0.1,
                   help='Noise scale for physics parameter sampling')
parser.add_argument('--mc-physics-learned', action='store_true',
                   help='Use learned uncertainty (requires training)')
parser.add_argument('--uncertainty-exploration-weight', type=float, default=0.1,
                   help='Weight for uncertainty in MCTS exploration')
```

## Training Uncertainty Head (Optional)

If using learned uncertainty, train with negative log-likelihood:

```python
def uncertainty_loss(predictions, targets, predicted_std):
    """NLL loss encourages calibrated uncertainty."""

    # Gaussian NLL: -log N(target | pred, std)
    nll = 0.5 * jnp.log(2 * jnp.pi * predicted_std**2) + \
          0.5 * ((targets - predictions) / predicted_std)**2

    return nll.mean()
```

## Expected Benefits

1. **Cheap uncertainty** - No heavy diffusion model needed
2. **Per-asset confidence** - Know which predictions are reliable
3. **Better MCTS exploration** - Explore uncertain branches more
4. **Risk-aware decisions** - Can be conservative in uncertain states
5. **Regime detection** - High uncertainty = regime change?

## Compute Cost

| Approach | Cost vs Baseline |
|----------|------------------|
| Single physics forward | 1x |
| MC physics (32 samples) | ~32x physics, but physics is cheap |
| Full diffusion | ~1000x+ |

Physics forward is very cheap, so 32x is still fast.

## Testing

```python
def test_mc_physics_captures_uncertainty():
    """High volatility states should have higher uncertainty."""

    low_vol_state = create_state(volatility=0.01)
    high_vol_state = create_state(volatility=0.05)

    low_vol_dist = mc_physics_predict(model, low_vol_state, config, rng)
    high_vol_dist = mc_physics_predict(model, high_vol_state, config, rng)

    # High vol should have more spread in predictions
    assert high_vol_dist.std.mean() > low_vol_dist.std.mean()


def test_mc_physics_mean_matches_deterministic():
    """Mean of samples should approximate deterministic prediction."""

    deterministic = physics_forward(state, physics_core(state))
    mc_dist = mc_physics_predict(model, state, config, rng)

    # Mean should be close to deterministic (with some noise)
    assert jnp.allclose(mc_dist.mean, deterministic, atol=0.1)
```

## Files to Modify

- `mcts_trader/jax_agent/prediction/physics_core.py` - Add uncertainty output
- `mcts_trader/jax_agent/prediction/mc_physics.py` - NEW: MC sampling
- `mcts_trader/jax_agent/mctx_search.py` - Integrate uncertainty into MCTS
- `train_jax_fast.py` - CLI arguments

## Related Tasks

- TASK_102: Future model integrations (this is lightweight version)
- TASK_035: Physics components fix
- TASK_032: Physics prediction scaling

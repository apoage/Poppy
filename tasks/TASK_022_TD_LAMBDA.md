# TASK 022: TD(λ) Returns for Variance Reduction

**Severity**: FEATURE
**Priority**: MEDIUM
**Location**: `mcts_trader/jax_agent/returns.py` (NEW), `train_jax_fast.py`
**Status**: IMPLEMENTED (2026-01-11)
**Effort**: Medium (2-3 hours)
**Dependencies**: None

---

## Problem

Current implementation uses pure Monte Carlo returns:
```python
mc_return = sum(gamma^t * reward_t for t in range(T))
```

**Issues with MC returns**:
1. High variance (depends on entire trajectory)
2. Slow credit assignment (reward at t=100 affects value at t=0)
3. Noisy gradients → unstable training

---

## Solution: TD(λ) Returns

TD(λ) blends n-step returns with exponential weighting:
```
G_t^λ = (1-λ) * Σ_{n=1}^{∞} λ^{n-1} * G_t^{(n)}

where G_t^{(n)} = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n * V(s_{t+n})
```

**λ parameter**:
- λ=0: TD(0) - pure bootstrapping, low variance, high bias
- λ=1: MC returns - no bootstrapping, high variance, no bias
- λ=0.95: Good balance for most RL problems

---

## Research (2025)

### RLax Library
[RLax](https://rlax.readthedocs.io/en/latest/api.html) provides JAX-native TD(λ) implementations:
- `rlax.td_lambda()` - Standard TD(λ)
- `rlax.lambda_returns()` - Compute λ-returns
- V-trace for off-policy correction

### Key Papers
- [Truncating Temporal Differences](https://www.jair.org/index.php/jair/article/view/10128): Efficient TD(λ) implementation
- [2025 Gradient PPO](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_302.pdf): Uses TD(λ) updates in actor-critic

### Benefits
- Lower variance than MC (uses value function for bootstrapping)
- Less biased than TD(0) (considers multiple future steps)
- Better sample efficiency in practice

---

## Implementation

### Option 1: Use RLax (Recommended)

```python
# Add to requirements
pip install rlax

# In train_jax_fast.py
import rlax

def compute_td_lambda_returns(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    discount: float = 0.99,
    lambda_: float = 0.95
) -> jnp.ndarray:
    """Compute TD(λ) returns using RLax."""
    # RLax expects discounts per timestep
    discounts = discount * (1.0 - dones)

    # Compute λ-returns
    returns = rlax.lambda_returns(
        r_t=rewards[:-1],
        discount_t=discounts[:-1],
        v_t=values[1:],
        lambda_=lambda_
    )

    return returns
```

### Option 2: Custom Implementation

```python
# mcts_trader/jax_agent/returns.py
"""Return computation utilities."""

import jax
import jax.numpy as jnp

def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    discount: float = 0.99,
    gae_lambda: float = 0.95
) -> jnp.ndarray:
    """Compute Generalized Advantage Estimation (equivalent to TD(λ) for returns).

    Args:
        rewards: (T,) rewards
        values: (T+1,) value estimates (includes bootstrap)
        dones: (T,) episode termination flags
        discount: γ discount factor
        gae_lambda: λ parameter

    Returns:
        (T,) TD(λ) returns
    """
    T = len(rewards)
    advantages = jnp.zeros(T)

    # Backward pass to compute advantages
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + discount * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + discount * gae_lambda * (1 - dones[t]) * gae
        advantages = advantages.at[t].set(gae)

    # Returns = advantages + values
    returns = advantages + values[:-1]
    return returns

# JIT-compiled version for efficiency
@jax.jit
def compute_gae_jit(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    discount: float,
    gae_lambda: float
) -> jnp.ndarray:
    """JIT-compiled GAE computation using scan."""

    def gae_step(gae, inputs):
        reward, value, next_value, done = inputs
        delta = reward + discount * next_value * (1 - done) - value
        gae = delta + discount * gae_lambda * (1 - done) * gae
        return gae, gae

    # Prepare inputs for scan (reversed order)
    inputs = (
        rewards[::-1],
        values[:-1][::-1],
        values[1:][::-1],
        dones[::-1]
    )

    _, advantages = jax.lax.scan(
        gae_step,
        jnp.zeros(()),
        inputs
    )

    # Reverse back and compute returns
    advantages = advantages[::-1]
    returns = advantages + values[:-1]
    return returns
```

### Integration in Self-Play

```python
# train_jax_fast.py - in run_self_play()

def run_self_play(..., use_gae: bool = False, gae_lambda: float = 0.95):
    # ... collect trajectory ...

    if use_gae:
        from mcts_trader.jax_agent.returns import compute_gae_jit
        # Get bootstrap value for final state
        bootstrap_value = get_value_estimate(final_obs)
        values_with_bootstrap = np.concatenate([values, [bootstrap_value]])

        mc_returns = compute_gae_jit(
            jnp.array(rewards),
            jnp.array(values_with_bootstrap),
            jnp.array(dones),
            discount,
            gae_lambda
        )
    else:
        # Original MC returns
        mc_returns = compute_mc_returns(rewards, discount)
```

---

## CLI Arguments

```python
parser.add_argument('--use-gae', action='store_true', default=False,
    help='Use TD(λ)/GAE returns instead of MC returns')
parser.add_argument('--gae-lambda', type=float, default=0.95,
    help='Lambda parameter for GAE (default: 0.95)')
```

---

## Testing

```bash
# Compare GAE vs MC returns
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 200 \
    --use-gae \
    --gae-lambda 0.95 \
    --name test_gae

python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 200 \
    --name test_mc
```

**Expected**: GAE version should show:
- Lower variance in value predictions
- Smoother training curves
- Possibly faster convergence

---

## Files to Create/Modify

| File | Changes |
|------|---------|
| `mcts_trader/jax_agent/returns.py` | NEW: GAE/TD(λ) computation |
| `train_jax_fast.py` | Add CLI args, integrate in self-play |

---

## Success Criteria

1. Value std decreases faster during training
2. Value loss is smoother (less spiky)
3. No performance regression (same or better final reward)
4. Works correctly with episode boundaries (dones)

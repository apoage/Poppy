"""Return computation utilities.

TASK_022: TD(λ)/GAE returns for variance reduction.

TD(λ) blends n-step returns with exponential weighting:
- λ=0: TD(0) - pure bootstrapping, low variance, high bias
- λ=1: MC returns - no bootstrapping, high variance, no bias
- λ=0.95: Good balance for most RL problems

GAE (Generalized Advantage Estimation) is equivalent to TD(λ) for returns.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Union


def compute_gae_returns(
    rewards: Union[np.ndarray, jnp.ndarray],
    values: Union[np.ndarray, jnp.ndarray],
    dones: Union[np.ndarray, jnp.ndarray],
    bootstrap_value: float = 0.0,
    discount: float = 0.997,
    gae_lambda: float = 0.95,
) -> np.ndarray:
    """Compute TD(λ)/GAE returns.

    This computes returns that blend bootstrapping with Monte Carlo,
    reducing variance compared to pure MC while maintaining low bias.

    Args:
        rewards: (T,) rewards at each step
        values: (T,) value estimates at each step from the network
        dones: (T,) episode termination flags (True if episode ended)
        bootstrap_value: V(s_T) estimate for truncated episodes
        discount: γ discount factor (default 0.997)
        gae_lambda: λ parameter for TD(λ) (default 0.95)

    Returns:
        (T,) TD(λ) returns (advantages + values)
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    T = len(rewards)
    if T == 0:
        return np.array([], dtype=np.float32)

    # Compute advantages using GAE
    # A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    # where δ_t = r_t + γV(s_{t+1}) - V(s_t)

    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    # Create value array with bootstrap at the end
    # V(s_{T}) = bootstrap_value if truncated, else 0
    values_with_bootstrap = np.append(values, bootstrap_value)

    # Backward pass
    for t in range(T - 1, -1, -1):
        # If done, don't bootstrap from next state
        not_done = 1.0 - dones[t]
        next_value = values_with_bootstrap[t + 1] * not_done

        # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + discount * next_value - values[t]

        # GAE: A_t = δ_t + γλ * A_{t+1}
        gae = delta + discount * gae_lambda * not_done * gae
        advantages[t] = gae

    # Returns = advantages + values
    returns = advantages + values

    return returns


def compute_mc_returns(
    rewards: Union[np.ndarray, jnp.ndarray],
    dones: Union[np.ndarray, jnp.ndarray],
    bootstrap_value: float = 0.0,
    discount: float = 0.997,
) -> np.ndarray:
    """Compute pure Monte Carlo returns.

    This is the original return computation for comparison.

    Args:
        rewards: (T,) rewards at each step
        dones: (T,) episode termination flags
        bootstrap_value: V(s_T) for truncated episodes
        discount: γ discount factor

    Returns:
        (T,) MC returns
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    T = len(rewards)
    if T == 0:
        return np.array([], dtype=np.float32)

    mc_returns = np.zeros(T, dtype=np.float32)
    running_return = bootstrap_value

    for t in range(T - 1, -1, -1):
        # If done, don't accumulate from next (unless it's the terminal state)
        # But for trading, we typically want continuous accumulation
        running_return = rewards[t] + discount * running_return
        mc_returns[t] = running_return

    return mc_returns


@jax.jit
def compute_gae_returns_jax(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    bootstrap_value: float,
    discount: float,
    gae_lambda: float,
) -> jnp.ndarray:
    """JIT-compiled GAE computation using JAX scan.

    Faster for large trajectories but requires JAX arrays.

    Args:
        rewards: (T,) rewards
        values: (T,) value estimates
        dones: (T,) done flags
        bootstrap_value: V(s_T) for truncation
        discount: γ
        gae_lambda: λ

    Returns:
        (T,) TD(λ) returns
    """
    T = len(rewards)

    # Append bootstrap value
    values_extended = jnp.concatenate([values, jnp.array([bootstrap_value])])

    def gae_step(gae, inputs):
        """Single step of GAE computation (reverse order)."""
        reward, value, next_value, done = inputs
        not_done = 1.0 - done
        delta = reward + discount * next_value * not_done - value
        gae = delta + discount * gae_lambda * not_done * gae
        return gae, gae

    # Prepare inputs in reverse order for scan
    scan_inputs = (
        rewards[::-1],
        values[::-1],
        values_extended[1:][::-1],  # Next values
        dones[::-1],
    )

    _, advantages_reversed = jax.lax.scan(
        gae_step,
        jnp.array(0.0),
        scan_inputs,
    )

    # Reverse back to original order
    advantages = advantages_reversed[::-1]

    # Returns = advantages + values
    returns = advantages + values

    return returns


def compute_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    bootstrap_value: float = 0.0,
    discount: float = 0.997,
    use_gae: bool = False,
    gae_lambda: float = 0.95,
) -> np.ndarray:
    """Unified return computation interface.

    Args:
        rewards: (T,) rewards
        values: (T,) value estimates (only used if use_gae=True)
        dones: (T,) done flags
        bootstrap_value: V(s_T) for truncation
        discount: γ
        use_gae: If True, use GAE. If False, use MC returns.
        gae_lambda: λ parameter (only used if use_gae=True)

    Returns:
        (T,) returns
    """
    if use_gae:
        return compute_gae_returns(
            rewards, values, dones,
            bootstrap_value, discount, gae_lambda
        )
    else:
        return compute_mc_returns(
            rewards, dones, bootstrap_value, discount
        )

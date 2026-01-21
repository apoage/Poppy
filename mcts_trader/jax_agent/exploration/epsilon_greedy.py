import jax.numpy as jnp
import chex
from typing import Optional

def apply_epsilon_greedy(
    mcts_policy: jnp.ndarray,
    epsilon: float,
    action_mask: Optional[jnp.ndarray] = None,
    rng: Optional[chex.PRNGKey] = None,
) -> jnp.ndarray:
    """
    Mix MCTS policy with uniform distribution for exploration.
    π_train = (1 - ε) * π_mcts + ε * π_uniform
    """
    if epsilon <= 0:
        return mcts_policy

    if epsilon >= 1:
        # Pure uniform
        if action_mask is not None:
            valid_count = jnp.sum(action_mask, axis=-1, keepdims=True)
            return action_mask / jnp.maximum(valid_count, 1.0)
        else:
            num_actions = mcts_policy.shape[-1]
            return jnp.ones_like(mcts_policy) / num_actions

    # Create uniform distribution over valid actions
    if action_mask is not None:
        valid_count = jnp.sum(action_mask, axis=-1, keepdims=True)
        uniform = action_mask / jnp.maximum(valid_count, 1.0)
    else:
        num_actions = mcts_policy.shape[-1]
        uniform = jnp.ones_like(mcts_policy) / num_actions

    # Mix: (1-ε) * π_mcts + ε * uniform
    mixed_policy = (1 - epsilon) * mcts_policy + epsilon * uniform

    # Renormalize for numerical stability
    mixed_policy = mixed_policy / jnp.sum(mixed_policy, axis=-1, keepdims=True)

    return mixed_policy

class EpsilonSchedule:
    """Decay epsilon over training."""
    def __init__(
        self,
        initial: float = 0.3,
        final: float = 0.05,
        decay_epochs: int = 300,
        warmup_epochs: int = 0,
    ):
        self.initial = initial
        self.final = final
        self.decay_epochs = decay_epochs
        self.warmup_epochs = warmup_epochs

    def get_epsilon(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.initial
        decay_epoch = epoch - self.warmup_epochs
        if decay_epoch >= self.decay_epochs:
            return self.final
        progress = decay_epoch / self.decay_epochs
        return self.initial + progress * (self.final - self.initial)

    def __call__(self, epoch: int) -> float:
        return self.get_epsilon(epoch)

class AdaptiveEpsilon:
    """Adapt epsilon based on observed policy entropy."""
    def __init__(
        self,
        target_entropy: float = 2.9,
        min_epsilon: float = 0.05,
        max_epsilon: float = 0.5,
        adaptation_rate: float = 0.01,
    ):
        self.target_entropy = target_entropy
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.adaptation_rate = adaptation_rate
        self.current_epsilon = (min_epsilon + max_epsilon) / 2

    def update(self, observed_entropy: float) -> float:
        entropy_error = self.target_entropy - observed_entropy
        delta = self.adaptation_rate * entropy_error
        self.current_epsilon = jnp.clip(
            self.current_epsilon + delta,
            self.min_epsilon,
            self.max_epsilon
        )
        return float(self.current_epsilon)

    def get_epsilon(self) -> float:
        return self.current_epsilon

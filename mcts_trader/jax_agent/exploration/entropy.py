import jax.numpy as jnp

def compute_policy_entropy(policy_probs: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy of policy distribution."""
    log_probs = jnp.log(policy_probs + 1e-10)
    entropy = -jnp.sum(policy_probs * log_probs, axis=-1)
    return entropy

def compute_minimum_entropy(epsilon: float, num_actions: int) -> float:
    """Compute minimum entropy guaranteed by ε-greedy mixing."""
    if epsilon <= 0:
        return 0.0
    return epsilon * jnp.log(num_actions)

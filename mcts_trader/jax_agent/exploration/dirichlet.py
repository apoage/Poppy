import jax
import jax.numpy as jnp
import chex
from typing import Tuple

def sample_dirichlet_noise(
    rng: chex.PRNGKey,
    shape: Tuple[int, ...],
    alpha: float = 0.3,
) -> jnp.ndarray:
    """Sample Dirichlet noise for root prior."""
    gamma_samples = jax.random.gamma(rng, alpha, shape)
    return gamma_samples / jnp.sum(gamma_samples, axis=-1, keepdims=True)

def get_recommended_dirichlet_params(
    num_actions: int,
    action_structure: str = "trading",
) -> Tuple[float, float]:
    """Get recommended Dirichlet parameters based on action space."""
    if action_structure == "trading":
        return (0.5, 0.5)
    elif action_structure == "chess":
        return (0.3, 0.25)
    elif action_structure == "go":
        return (0.03, 0.25)
    else:
        return (0.3, 0.25)

"""ReDo: Recycling Dormant Neurons.

TASK_042: Detect and recycle dormant neurons to maintain network capacity.

Based on:
- Sokar et al., 2023: "Recycling Dormant Neurons"
- Gemini Deep Research: "Re-initialize incoming weights to re-inject plasticity"

Dormant neurons are those with very low activation variance, indicating they're
not contributing to the network's function. These can be recycled by
re-initializing their incoming weights.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
import flax.core
from typing import Dict, List, Tuple, Optional
import optax


class DormancyMonitor:
    """Track neuron activation statistics for ReDo.

    Monitors the coefficient of variation (CV) of activations for each neuron.
    Neurons with CV below threshold are considered dormant.
    """

    def __init__(
        self,
        dormancy_threshold: float = 0.01,
        check_interval: int = 10,  # epochs
        min_samples: int = 1000,
    ):
        """Initialize dormancy monitor.

        Args:
            dormancy_threshold: CV below this marks neuron as dormant
            check_interval: Epochs between dormancy checks
            min_samples: Minimum samples before checking dormancy
        """
        self.dormancy_threshold = dormancy_threshold
        self.check_interval = check_interval
        self.min_samples = min_samples

        # Running statistics (mean and variance via Welford's algorithm)
        self.activation_mean: Dict[str, np.ndarray] = {}
        self.activation_m2: Dict[str, np.ndarray] = {}  # Sum of squared differences
        self.sample_count = 0

    def update(self, activations: Dict[str, jnp.ndarray]) -> None:
        """Update running statistics with batch of activations.

        Uses Welford's online algorithm for numerical stability.

        Args:
            activations: Dict mapping layer names to activation arrays
        """
        if not activations:
            return

        batch_size = next(iter(activations.values())).shape[0]

        for name, act in activations.items():
            # Flatten to (batch, neurons)
            act_np = np.asarray(act).reshape(batch_size, -1)
            n_neurons = act_np.shape[1]

            if name not in self.activation_mean:
                self.activation_mean[name] = np.zeros(n_neurons, dtype=np.float64)
                self.activation_m2[name] = np.zeros(n_neurons, dtype=np.float64)

            # Welford's online update (per-batch)
            for i in range(batch_size):
                sample = act_np[i]
                self.sample_count += 1
                n = self.sample_count

                delta = sample - self.activation_mean[name]
                self.activation_mean[name] += delta / n
                delta2 = sample - self.activation_mean[name]
                self.activation_m2[name] += delta * delta2

    def compute_dormancy_scores(self) -> Dict[str, np.ndarray]:
        """Compute coefficient of variation for each neuron.

        Returns:
            Dict mapping layer names to CV scores (lower = more dormant)
        """
        if self.sample_count < 2:
            return {}

        scores = {}
        for name in self.activation_mean:
            # Variance = M2 / (n - 1)
            var = self.activation_m2[name] / (self.sample_count - 1)
            std = np.sqrt(np.maximum(var, 0))

            # Coefficient of variation = std / |mean|
            # For neurons with mean ≈ 0, use std alone
            mean_abs = np.abs(self.activation_mean[name])
            scores[name] = np.where(
                mean_abs > 1e-8,
                std / mean_abs,  # CV for non-zero mean
                std  # Just std for zero mean
            )

        return scores

    def get_dormant_mask(self) -> Dict[str, np.ndarray]:
        """Get boolean mask of dormant neurons per layer.

        Returns:
            Dict mapping layer names to boolean masks (True = dormant)
        """
        scores = self.compute_dormancy_scores()
        masks = {}
        for name, score in scores.items():
            masks[name] = score < self.dormancy_threshold
        return masks

    def get_dormancy_stats(self) -> Dict[str, float]:
        """Get summary statistics about dormancy.

        Returns:
            Dict with total_neurons, dormant_neurons, dormancy_percent
        """
        masks = self.get_dormant_mask()
        total = sum(m.size for m in masks.values())
        dormant = sum(np.sum(m) for m in masks.values())

        return {
            'total_neurons': total,
            'dormant_neurons': int(dormant),
            'dormancy_percent': 100 * dormant / total if total > 0 else 0,
        }

    def reset(self) -> None:
        """Reset statistics after recycling."""
        self.activation_mean = {}
        self.activation_m2 = {}
        self.sample_count = 0

    def should_check(self, epoch: int) -> bool:
        """Check if we should check dormancy this epoch.

        Args:
            epoch: Current epoch number

        Returns:
            True if we should check dormancy
        """
        return (
            epoch > 0 and
            epoch % self.check_interval == 0 and
            self.sample_count >= self.min_samples
        )


def recycle_dormant_neurons(
    params: flax.core.FrozenDict,
    dormant_masks: Dict[str, np.ndarray],
    rng: jax.random.PRNGKey,
    init_scale: float = 0.01,
) -> flax.core.FrozenDict:
    """Re-initialize incoming weights for dormant neurons.

    Args:
        params: Model parameters
        dormant_masks: Dict mapping layer names to boolean masks
        rng: Random key for weight initialization
        init_scale: Scale for random initialization

    Returns:
        Updated parameters with recycled neurons
    """
    if not dormant_masks:
        return params

    # Flatten params for manipulation
    flat_params = traverse_util.flatten_dict(params, sep='/')

    # Track RNG usage
    rngs = jax.random.split(rng, len(dormant_masks) + 1)
    rng_idx = 0

    # Create updated params
    new_params = {}

    for key, value in flat_params.items():
        # Check if this layer has dormant neurons
        layer_name = None
        for mask_name in dormant_masks:
            if mask_name.lower() in key.lower():
                layer_name = mask_name
                break

        if layer_name is not None and dormant_masks[layer_name] is not None:
            mask = dormant_masks[layer_name]

            if 'kernel' in key.lower():
                # For weight matrices: re-initialize columns for dormant neurons
                # Shape: (in_features, out_features)
                if value.ndim == 2 and value.shape[1] == len(mask):
                    new_weights = jax.random.normal(rngs[rng_idx], value.shape) * init_scale
                    rng_idx += 1

                    # Replace only dormant columns
                    mask_expanded = jnp.array(mask)[None, :]  # (1, out_features)
                    value = jnp.where(mask_expanded, new_weights, value)

            elif 'bias' in key.lower():
                # For biases: reset to zero for dormant neurons
                if value.ndim == 1 and len(value) == len(mask):
                    mask_jnp = jnp.array(mask)
                    value = jnp.where(mask_jnp, 0.0, value)

        new_params[key] = value

    # Reconstruct frozen dict
    return flax.core.freeze(traverse_util.unflatten_dict(new_params, sep='/'))


def reset_optimizer_for_recycled(
    opt_state: optax.OptState,
    dormant_masks: Dict[str, np.ndarray],
) -> optax.OptState:
    """Reset optimizer momentum/variance for recycled neurons.

    This is a simplified version that only works with Adam-like optimizers.
    For complex optimizer chains, this may need adjustment.

    Args:
        opt_state: Current optimizer state
        dormant_masks: Dict mapping layer names to boolean masks

    Returns:
        Updated optimizer state with reset slots for recycled neurons
    """
    # For now, return unchanged - full reset would require knowing optimizer structure
    # The simpler approach is to let the optimizer naturally adapt to new weights
    # This is less disruptive than a full reset

    # TODO: Implement proper optimizer state reset if needed
    # This would require traversing opt_state structure and resetting
    # momentum/variance for recycled parameters

    return opt_state


def extract_activations_from_output(
    output: Dict[str, jnp.ndarray],
    layer_names: Optional[List[str]] = None,
) -> Dict[str, jnp.ndarray]:
    """Extract activation tensors from model output for dormancy monitoring.

    Args:
        output: Model output dict (may contain 'activations' key if model supports it)
        layer_names: Optional list of layer names to extract

    Returns:
        Dict of layer name -> activation tensor
    """
    if 'activations' in output:
        activations = output['activations']
        if layer_names:
            return {k: v for k, v in activations.items() if k in layer_names}
        return activations

    # Fallback: extract from known output keys
    activations = {}

    # Try to get intermediate representations
    if 'hidden_state' in output:
        activations['hidden'] = output['hidden_state']
    if 'latent' in output:
        activations['latent'] = output['latent']

    return activations

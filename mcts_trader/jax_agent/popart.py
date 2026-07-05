"""
PopArt: Preserving Outputs Precisely while Adaptively Rescaling Targets

Adaptive value normalization for curriculum learning and varying reward scales.
Based on DeepMind's PopArt (https://arxiv.org/pdf/1602.07714).

Key idea:
- Normalize value targets using running mean/std
- When statistics change, adjust value head weights to preserve outputs
- This ensures v_new(h) ≈ v_old(h) even when reward scales change dramatically

Used for:
- Curriculum learning (L0 → L4 have different reward scales)
- Training stability (early random exploration vs late exploitation)
- Market regime changes (bull vs bear have different returns)
"""

import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, Dict, Optional


@struct.dataclass
class PopArtState:
    """
    Running statistics for PopArt normalization.

    Tracks mean and variance of value targets with EMA updates.
    Stores old statistics for weight correction after updates.
    """
    mean: jnp.ndarray      # Running mean of targets (scalar)
    var: jnp.ndarray       # Running variance of targets (scalar)
    count: int             # Number of samples seen
    # Store old stats for weight correction
    mean_old: jnp.ndarray
    std_old: jnp.ndarray


def init_popart_state() -> PopArtState:
    """Initialize PopArt state with neutral statistics."""
    return PopArtState(
        mean=jnp.zeros(()),
        var=jnp.ones(()),
        count=0,
        mean_old=jnp.zeros(()),
        std_old=jnp.ones(()),
    )


def update_statistics(
    state: PopArtState,
    targets: jnp.ndarray,
    beta: float = 0.0001
) -> PopArtState:
    """
    Update running mean/var with new targets using EMA.

    Args:
        state: Current PopArt state
        targets: Batch of value targets (batch_size,)
        beta: EMA decay rate (lower = slower adaptation, more stable)

    Returns:
        Updated PopArt state with old stats preserved for weight correction
    """
    # Store old statistics before update
    std_old = jnp.sqrt(state.var + 1e-8)

    # Compute batch statistics
    batch_mean = jnp.mean(targets)
    batch_var = jnp.var(targets)

    # EMA update (first update uses batch stats directly)
    if state.count == 0:
        new_mean = batch_mean
        new_var = batch_var + 1e-8  # Prevent zero variance
    else:
        new_mean = (1 - beta) * state.mean + beta * batch_mean
        new_var = (1 - beta) * state.var + beta * batch_var
        new_var = jnp.maximum(new_var, 1e-8)  # Prevent zero variance

    return PopArtState(
        mean=new_mean,
        var=new_var,
        count=state.count + len(targets),
        mean_old=state.mean,
        std_old=std_old,
    )


def normalize_targets(
    targets: jnp.ndarray,
    state: PopArtState
) -> jnp.ndarray:
    """
    Normalize targets using running statistics.

    Args:
        targets: Raw value targets
        state: Current PopArt state

    Returns:
        Normalized targets with zero mean, unit variance
    """
    std = jnp.sqrt(state.var + 1e-8)
    return (targets - state.mean) / std


def denormalize_values(
    values: jnp.ndarray,
    state: PopArtState
) -> jnp.ndarray:
    """
    Denormalize predicted values back to original scale.

    Used when computing actual value predictions for action selection.

    Args:
        values: Normalized value predictions
        state: Current PopArt state

    Returns:
        Values in original (unnormalized) scale
    """
    std = jnp.sqrt(state.var + 1e-8)
    return values * std + state.mean


def compute_weight_correction(
    state: PopArtState
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute scale and shift corrections for value head weights.

    When statistics change from (μ_old, σ_old) to (μ_new, σ_new),
    we need to adjust weights to preserve outputs:

        W_new = W_old * (σ_old / σ_new)
        b_new = b_old * (σ_old / σ_new) + (μ_old - μ_new) / σ_new

    This ensures v_new(h) = v_old(h) for any hidden state h.

    Args:
        state: PopArt state with old and new statistics

    Returns:
        scale: Multiply weights and bias by this
        shift: Add to bias after scaling
    """
    std_new = jnp.sqrt(state.var + 1e-8)
    std_old = state.std_old + 1e-8

    scale = std_old / std_new
    shift = (state.mean_old - state.mean) / std_new

    return scale, shift


def apply_weight_correction(
    params: Dict,
    scale: jnp.ndarray,
    shift: jnp.ndarray,
    value_head_prefix: str = 'value_head'
) -> Dict:
    """
    Apply PopArt correction to value head weights.

    Finds all layers with 'value_head' in their name and applies:
        W_new = W_old * scale
        b_new = b_old * scale + shift

    Args:
        params: Model parameters dict
        scale: Weight scaling factor
        shift: Bias shift after scaling
        value_head_prefix: Prefix to identify value head layers

    Returns:
        Corrected parameters dict
    """
    def correct_layer(p: Dict) -> Dict:
        """Apply correction to a single layer's parameters."""
        if isinstance(p, dict) and 'kernel' in p and 'bias' in p:
            return {
                'kernel': p['kernel'] * scale,
                'bias': p['bias'] * scale + shift,
            }
        return p

    def recursive_apply(params: Dict, prefix: str = '') -> Dict:
        """Recursively apply correction to matching layers."""
        new_params = {}
        for key, value in params.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                if value_head_prefix in key:
                    # This is a value head layer - correct it
                    new_params[key] = correct_layer(value)
                else:
                    # Recurse into nested dicts
                    new_params[key] = recursive_apply(value, full_key)
            else:
                new_params[key] = value
        return new_params

    return recursive_apply(params)


def apply_weight_correction_flax(
    params: Dict,
    scale: jnp.ndarray,
    shift: jnp.ndarray
) -> Dict:
    """
    Apply PopArt correction to Flax-style value head params.

    Specifically handles the TRM architecture where value head is:
        params['value_head_1']['kernel'], params['value_head_1']['bias']
        params['value_head_2']['kernel'], params['value_head_2']['bias']

    Only the LAST value head layer needs correction (the one that outputs
    the final value logits/support).

    Args:
        params: Flax model parameters
        scale: Weight scaling factor
        shift: Bias shift

    Returns:
        Corrected parameters
    """
    # Create a mutable copy
    import copy
    new_params = copy.deepcopy(dict(params))

    # Find value head layers in TRM structure
    # The final value head that outputs to support_size needs correction
    for key in list(new_params.keys()):
        if 'value_head' in key:
            layer = new_params[key]
            if isinstance(layer, dict) and 'kernel' in layer and 'bias' in layer:
                # Apply correction
                new_params[key] = {
                    'kernel': layer['kernel'] * scale,
                    'bias': layer['bias'] * scale + shift,
                }

    return new_params


class PopArtNormalizer:
    """
    Convenience class for PopArt normalization in training.

    Usage:
        normalizer = PopArtNormalizer(beta=0.0001)

        # In training loop:
        targets = batch['values']
        norm_targets, params = normalizer.normalize_and_correct(targets, params)
        # ... train with norm_targets ...

        # For inference (action selection):
        raw_values = normalizer.denormalize(predicted_values)
    """

    def __init__(self, beta: float = 0.0001):
        """
        Initialize PopArt normalizer.

        Args:
            beta: EMA decay rate for statistics update
        """
        self.beta = beta
        self.state = init_popart_state()

    def normalize_and_correct(
        self,
        targets: jnp.ndarray,
        params: Dict
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Update statistics, normalize targets, and correct value head weights.

        Call this ONCE per batch, BEFORE computing the loss.

        Args:
            targets: Raw value targets
            params: Current model parameters

        Returns:
            norm_targets: Normalized targets for training
            corrected_params: Parameters with corrected value head
        """
        # Update statistics
        self.state = update_statistics(self.state, targets, self.beta)

        # Compute weight correction
        scale, shift = compute_weight_correction(self.state)

        # Apply correction to value head
        corrected_params = apply_weight_correction_flax(params, scale, shift)

        # Normalize targets
        norm_targets = normalize_targets(targets, self.state)

        return norm_targets, corrected_params

    def normalize(self, targets: jnp.ndarray) -> jnp.ndarray:
        """Normalize targets without updating statistics."""
        return normalize_targets(targets, self.state)

    def denormalize(self, values: jnp.ndarray) -> jnp.ndarray:
        """Denormalize predicted values to original scale."""
        return denormalize_values(values, self.state)

    @property
    def mean(self) -> jnp.ndarray:
        """Current running mean."""
        return self.state.mean

    @property
    def std(self) -> jnp.ndarray:
        """Current running std."""
        return jnp.sqrt(self.state.var + 1e-8)

    @property
    def count(self) -> int:
        """Number of samples seen."""
        return self.state.count

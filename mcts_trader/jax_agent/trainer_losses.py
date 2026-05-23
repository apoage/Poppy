"""
mcts_trader/jax_agent/trainer_losses.py — Loss/grad helpers + config dataclasses.

Extracted from mcts_trader/jax_agent/trainer.py (TASK_772.4, 2026-05-23).
Holds pure functions and config dataclasses used by MuZeroTrainer:
- compute_large_margin_loss, compute_q_filtered_bc_loss     (TASK_125 BC losses)
- _compute_per_head_grad_norms, _clip_gradients_per_head    (per-head grad ops)
- ExpertBCConfig, PredictionLossConfig                       (config dataclasses)
- compute_annealed_lambda, compute_bc_weight,                (schedule helpers)
  compute_scheduled_sampling_prob, compute_neural_scale

Pure function move — no JIT closure capture, no MuZeroTrainer state. The
trainer.py file re-imports the symbols at top level so existing callers
see no behavior difference.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import flax
import jax
import jax.numpy as jnp


# ================================================================
# TASK_125: Causal Expert Imprinting - BC Loss Functions
# ================================================================

def compute_large_margin_loss(
    q_values: jnp.ndarray,
    expert_actions: jnp.ndarray,
    margin: float = 0.8,
) -> jnp.ndarray:
    """
    Large margin classification loss from DQfD.

    Forces Q(expert_action) > Q(other_actions) + margin.

    Args:
        q_values: (batch, num_actions) Q-values for all actions
        expert_actions: (batch,) expert action indices
        margin: Margin to enforce between expert and other actions

    Returns:
        (batch,) per-sample large margin loss
    """
    batch_size = q_values.shape[0]
    num_actions = q_values.shape[1]

    # One-hot expert actions
    expert_mask = jax.nn.one_hot(expert_actions, num_actions)  # (batch, num_actions)

    # Margin mask: 0 for expert action, margin for all others
    margin_mask = margin * (1.0 - expert_mask)  # (batch, num_actions)

    # Large margin target: max over (Q + margin_mask)
    # This is the Q-value expert must exceed
    large_margin_targets = jnp.max(q_values + margin_mask, axis=1)  # (batch,)

    # Expert Q-values
    expert_q = jnp.sum(q_values * expert_mask, axis=1)  # (batch,)

    # Loss: ReLU(margin_target - expert_q)
    # Only penalize when expert Q doesn't exceed margin-adjusted max
    loss = jnp.maximum(0.0, large_margin_targets - expert_q)

    return loss


def compute_q_filtered_bc_loss(
    policy_logits: jnp.ndarray,
    expert_actions: jnp.ndarray,
    policy_actions: jnp.ndarray,
    q_values: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    BC loss with Q-filter: only apply when Q(expert) > Q(policy).

    This auto-decays BC influence as policy improves beyond demonstrations.

    Args:
        policy_logits: (batch, num_actions) policy logits
        expert_actions: (batch,) expert action indices
        policy_actions: (batch,) current policy action indices (argmax)
        q_values: (batch, num_actions) Q-values for all actions

    Returns:
        filtered_bc_loss: (batch,) BC loss with Q-filter applied
        q_filter_mask: (batch,) 1.0 where filter passed, 0.0 otherwise
    """
    num_actions = q_values.shape[-1]

    # Get Q-values for expert and policy actions
    q_expert = jnp.sum(q_values * jax.nn.one_hot(expert_actions, num_actions), axis=-1)
    q_policy = jnp.sum(q_values * jax.nn.one_hot(policy_actions, num_actions), axis=-1)

    # Q-filter mask: only apply BC when expert is better
    q_filter_mask = (q_expert > q_policy).astype(jnp.float32)

    # Standard BC loss (cross-entropy with integer labels)
    bc_loss = optax.softmax_cross_entropy_with_integer_labels(policy_logits, expert_actions)

    # Apply Q-filter
    filtered_bc_loss = bc_loss * q_filter_mask

    return filtered_bc_loss, q_filter_mask


def _compute_per_head_grad_norms(grads: flax.core.FrozenDict) -> Dict[str, float]:
    """
    Compute gradient norms for different model components.

    This extracts per-head gradient norms for DS24 monitoring:
    - Value head: value_head_* layers
    - Policy head: output_refiner_* layers in TRM
    - Dynamics: dynamics_* and action_embedding layers

    Returns dict with grad_norm_value, grad_norm_policy, grad_norm_dynamics.
    """
    flat_grads = traverse_util.flatten_dict(grads, sep='/')

    value_grads = {}
    policy_grads = {}
    dynamics_grads = {}

    for key, val in flat_grads.items():
        key_lower = key.lower()
        if 'value_head' in key_lower:
            value_grads[key] = val
        elif 'output_refiner' in key_lower or 'trmreasoning' in key_lower:
            policy_grads[key] = val
        elif 'dynamics' in key_lower or 'action_embedding' in key_lower:
            dynamics_grads[key] = val

    result = {}
    if value_grads:
        result['grad_norm_value'] = optax.global_norm(
            traverse_util.unflatten_dict(value_grads, sep='/')
        )
    if policy_grads:
        result['grad_norm_policy'] = optax.global_norm(
            traverse_util.unflatten_dict(policy_grads, sep='/')
        )
    if dynamics_grads:
        result['grad_norm_dynamics'] = optax.global_norm(
            traverse_util.unflatten_dict(dynamics_grads, sep='/')
        )

    return result


def _clip_gradients_per_head(
    grads: flax.core.FrozenDict,
    max_norm_value: float = 1.0,
    max_norm_policy: float = 1.0,
    max_norm_dynamics: float = 1.0,
    max_norm_encoder: float = 1.0,
) -> flax.core.FrozenDict:
    """
    TASK_025: Clip gradients separately for each network head.

    This prevents one head (e.g., value head with large gradients) from
    dominating training and starving other heads.

    Args:
        grads: Gradient pytree
        max_norm_value: Max gradient norm for value head
        max_norm_policy: Max gradient norm for policy head (TRM reasoning)
        max_norm_dynamics: Max gradient norm for dynamics network
        max_norm_encoder: Max gradient norm for encoder/CNN

    Returns:
        Clipped gradient pytree with same structure
    """
    flat_grads = traverse_util.flatten_dict(grads, sep='/')

    # Categorize gradients by head
    value_keys = []
    policy_keys = []
    dynamics_keys = []
    encoder_keys = []
    other_keys = []

    for key in flat_grads.keys():
        key_lower = key.lower()
        if 'value_head' in key_lower:
            value_keys.append(key)
        elif 'output_refiner' in key_lower or 'trmreasoning' in key_lower:
            policy_keys.append(key)
        elif 'dynamics' in key_lower or 'action_embedding' in key_lower:
            dynamics_keys.append(key)
        elif 'encoder' in key_lower or 'cnn' in key_lower:
            encoder_keys.append(key)
        else:
            other_keys.append(key)

    def compute_scale(keys, max_norm):
        """Compute scale factor for a subset of gradients."""
        if not keys:
            return 1.0

        # Extract gradients for this subset and compute norm
        subset_grads = [flat_grads[k] for k in keys]
        norm = optax.global_norm(subset_grads)
        scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
        return scale

    # Compute scale factors for each head
    scale_value = compute_scale(value_keys, max_norm_value)
    scale_policy = compute_scale(policy_keys, max_norm_policy)
    scale_dynamics = compute_scale(dynamics_keys, max_norm_dynamics)
    scale_encoder = compute_scale(encoder_keys, max_norm_encoder)

    # Build key->scale mapping
    key_to_scale = {}
    for k in value_keys:
        key_to_scale[k] = scale_value
    for k in policy_keys:
        key_to_scale[k] = scale_policy
    for k in dynamics_keys:
        key_to_scale[k] = scale_dynamics
    for k in encoder_keys:
        key_to_scale[k] = scale_encoder
    for k in other_keys:
        key_to_scale[k] = 1.0  # No clipping for uncategorized

    # Apply clipping using jax.tree_util to preserve structure
    def clip_fn(path, grad):
        # Convert path tuple to string key
        path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        scale = key_to_scale.get(path_str, 1.0)
        return grad * scale

    # Use tree_map_with_path to preserve exact structure
    clipped = jax.tree_util.tree_map_with_path(clip_fn, grads)
    return clipped


@dataclass
class ExpertBCConfig:
    """TASK_125: Configuration for Causal Expert Imprinting BC loss."""
    # Enable/disable
    enabled: bool = False

    # BC loss weights
    bc_weight_initial: float = 0.1     # Start with strong imitation
    bc_weight_final: float = 0.01      # Decay as agent improves
    bc_weight_decay_steps: int = 50000  # Steps to decay BC weight

    # Q-filter (auto-decay)
    use_q_filter: bool = True          # Apply BC only when Q(demo) > Q(policy)

    # Large margin loss
    use_large_margin: bool = True      # DQfD-style margin loss
    large_margin: float = 0.8          # Margin value

    # Sampling
    expert_batch_size: int = 256       # Samples from expert buffer per step (was 128)
    demo_ratio: float = 0.125          # 12.5% demos in mixed batch

    # Demo weighting
    weight_by_confidence: bool = True  # Weight BC loss by demo confidence

    # TASK_136d: BC update frequency (encoding is expensive ~2.5s per call)
    # Literature: AWAC uses every 4-8, IQL uses every 10. Default 5 = 80% time savings.
    bc_frequency: int = 5              # Run BC every N training iterations


@dataclass
class PredictionLossConfig:
    """Configuration for auxiliary prediction loss (Phase 2)."""
    # Loss weight annealing
    weight_initial: float = 0.01      # Start low
    weight_final: float = 0.2         # Anneal to this
    anneal_steps: int = 10000         # Steps to reach final weight

    # Multi-step prediction
    multi_step_k: int = 5             # Number of future steps
    step_decay: float = 0.9           # Discount per step

    # Scheduled sampling annealing
    scheduled_sampling_start: float = 1.0   # Start with full teacher forcing
    scheduled_sampling_end: float = 0.0     # End with no teacher forcing
    scheduled_sampling_steps: int = 20000   # Annealing steps

    # Physics regularization
    physics_reg_weight: float = 0.01  # Physics regularization

    # TASK_035: Physics warmup - neural corrector scale annealing
    # During warmup, neural_scale goes from 0 → 1
    # This gives physics a "head start" to learn basic dynamics
    physics_warmup_steps: int = 5000  # Steps for neural warmup (0 = disabled)

    # TASK_116: Student-t output distribution for fat-tailed returns
    # Financial returns have heavier tails than Gaussian - Student-t models this better
    use_student_t: bool = False       # Enable Student-t loss (default: MSE/Huber)
    student_t_weight: float = 0.8     # Weight for Student-t NLL component
    mse_weight: float = 0.2           # Weight for MSE component (stability)
    df_min: float = 2.1               # Minimum degrees of freedom (>2 for finite variance)
    df_max: float = 100.0             # Maximum df (large = nearly Gaussian)


def compute_annealed_lambda(step: int, config: PredictionLossConfig) -> float:
    """Compute annealed prediction loss weight."""
    progress = jnp.clip(step / config.anneal_steps, 0.0, 1.0)
    return config.weight_initial + progress * (config.weight_final - config.weight_initial)


def compute_bc_weight(step: int, config: ExpertBCConfig) -> float:
    """TASK_125: Compute annealed BC loss weight (decays as agent improves)."""
    progress = jnp.clip(step / config.bc_weight_decay_steps, 0.0, 1.0)
    return config.bc_weight_initial + progress * (config.bc_weight_final - config.bc_weight_initial)


def compute_scheduled_sampling_prob(step: int, config: PredictionLossConfig) -> float:
    """Compute scheduled sampling probability (teacher forcing rate)."""
    progress = jnp.clip(step / config.scheduled_sampling_steps, 0.0, 1.0)
    return config.scheduled_sampling_start + progress * (
        config.scheduled_sampling_end - config.scheduled_sampling_start
    )


def compute_neural_scale(step, config: PredictionLossConfig):
    """
    Compute neural corrector scale for physics warmup (TASK_035).

    During warmup, neural_scale goes from 0 → 1, giving physics a head start.
    After warmup, neural_scale = 1.0 (full neural contribution).

    Args:
        step: Current training step (can be JAX traced array)
        config: Prediction loss configuration

    Returns:
        neural_scale: Scale factor for neural corrector [0, 1] (JAX array)
    """
    if config.physics_warmup_steps <= 0:
        return jnp.array(1.0)  # No warmup, full neural from start

    progress = jnp.clip(step / config.physics_warmup_steps, 0.0, 1.0)
    return progress  # Keep as JAX array for JIT compatibility

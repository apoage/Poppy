"""
Flow Matching for Prediction Head.

TASK_041: Predict velocity (ds/dt) instead of next state.

The identity solution problem:
- Standard: loss = MSE(predicted_state, actual_state)
- Identity: predicted = current → loss ≈ 0 for stable markets
- Result: Prediction head learns "no change" trivially

Flow matching solution:
- Predict: velocity = next_state - current_state
- Identity predicts v=0, but actual v≠0 → HIGH loss
- Forces model to learn actual dynamics

References:
- Flow Matching for Generative Modeling (Lipman et al., 2022)
- Gemini Research: "Train to predict velocity rather than next state"
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable
import chex


def compute_flow_matching_loss(
    predicted_next_state: jnp.ndarray,
    predicted_velocity: jnp.ndarray,
    actual_next_state: jnp.ndarray,
    current_state: jnp.ndarray,
    velocity_weight: float = 0.8,
    state_weight: float = 0.2,
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute flow matching loss.

    Args:
        predicted_next_state: (batch, state_dim) model's predicted next state
        predicted_velocity: (batch, state_dim) model's predicted velocity
        actual_next_state: (batch, state_dim) ground truth next state
        current_state: (batch, state_dim) current state (for velocity computation)
        velocity_weight: Weight for velocity loss (default: 0.8)
        state_weight: Weight for state loss regularization (default: 0.2)

    Returns:
        (loss, metrics_dict) tuple
    """
    # Actual velocity
    actual_velocity = actual_next_state - current_state

    # Velocity loss (main objective)
    velocity_loss = jnp.mean((predicted_velocity - actual_velocity) ** 2)

    # State loss (regularization)
    state_loss = jnp.mean((predicted_next_state - actual_next_state) ** 2)

    # Combined loss
    total_loss = velocity_weight * velocity_loss + state_weight * state_loss

    metrics = {
        'velocity_loss': velocity_loss,
        'state_loss': state_loss,
        'velocity_magnitude_pred': jnp.mean(jnp.linalg.norm(predicted_velocity, axis=-1)),
        'velocity_magnitude_actual': jnp.mean(jnp.linalg.norm(actual_velocity, axis=-1)),
    }

    return total_loss, metrics


def compute_multistep_flow_loss(
    dynamics_fn: Callable,
    params: dict,
    initial_state: jnp.ndarray,
    actions: jnp.ndarray,
    actual_states: jnp.ndarray,
    num_steps: int = 5,
    velocity_weight: float = 0.8,
) -> Tuple[jnp.ndarray, dict]:
    """
    Multi-step flow matching loss.

    Unrolls dynamics for num_steps and computes velocity loss at each step.

    Args:
        dynamics_fn: Function (params, state, action) -> (next_state, velocity)
        params: Model parameters
        initial_state: (batch, state_dim) starting state
        actions: (batch, num_steps, action_dim) action sequence
        actual_states: (batch, num_steps+1, state_dim) including initial
        num_steps: Number of steps to unroll
        velocity_weight: Weight for velocity vs state loss

    Returns:
        (loss, metrics_dict) tuple
    """
    batch_size = initial_state.shape[0]
    total_loss = 0.0
    velocity_losses = []
    state_losses = []

    current_state = initial_state

    for step in range(num_steps):
        action = actions[:, step]
        target_state = actual_states[:, step + 1]
        prev_state = actual_states[:, step]

        # Model prediction
        next_state, velocity = dynamics_fn(params, current_state, action)

        # Compute losses for this step
        actual_velocity = target_state - prev_state
        v_loss = jnp.mean((velocity - actual_velocity) ** 2)
        s_loss = jnp.mean((next_state - target_state) ** 2)

        step_loss = velocity_weight * v_loss + (1 - velocity_weight) * s_loss
        total_loss += step_loss

        velocity_losses.append(v_loss)
        state_losses.append(s_loss)

        # Use predicted state for next step (no teacher forcing)
        current_state = next_state

    avg_loss = total_loss / num_steps

    metrics = {
        'multistep_flow_loss': avg_loss,
        'velocity_loss_step1': velocity_losses[0] if velocity_losses else 0.0,
        'velocity_loss_final': velocity_losses[-1] if velocity_losses else 0.0,
        'state_loss_step1': state_losses[0] if state_losses else 0.0,
        'state_loss_final': state_losses[-1] if state_losses else 0.0,
    }

    return avg_loss, metrics


class FlowMatchingDynamics:
    """
    Wrapper to convert existing dynamics model to flow matching.

    If your dynamics model outputs next_state directly, this wrapper
    computes velocity as next_state - current_state.
    """

    def __init__(
        self,
        dynamics_fn: Callable,
        output_velocity: bool = False,
    ):
        """
        Args:
            dynamics_fn: Original dynamics function
            output_velocity: If True, dynamics already outputs (state, velocity)
                           If False, we compute velocity from state difference
        """
        self.dynamics_fn = dynamics_fn
        self.output_velocity = output_velocity

    def __call__(
        self,
        params: dict,
        current_state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply dynamics and return (next_state, velocity).
        """
        if self.output_velocity:
            return self.dynamics_fn(params, current_state, action)
        else:
            next_state = self.dynamics_fn(params, current_state, action)
            velocity = next_state - current_state
            return next_state, velocity


def normalize_velocity(
    velocity: jnp.ndarray,
    running_mean: Optional[jnp.ndarray] = None,
    running_std: Optional[jnp.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Normalize velocity for stable training.

    Args:
        velocity: (batch, dim) velocity vectors
        running_mean: Optional running mean for normalization
        running_std: Optional running std for normalization
        eps: Small constant for numerical stability

    Returns:
        (normalized_velocity, updated_mean, updated_std)
    """
    batch_mean = jnp.mean(velocity, axis=0)
    batch_std = jnp.std(velocity, axis=0) + eps

    if running_mean is None:
        running_mean = batch_mean
        running_std = batch_std
    else:
        # EMA update
        momentum = 0.99
        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_std = momentum * running_std + (1 - momentum) * batch_std

    # Guard against zero std (defense-in-depth)
    normalized = (velocity - running_mean) / jnp.maximum(running_std, eps)

    return normalized, running_mean, running_std


def velocity_matching_target(
    current_state: jnp.ndarray,
    next_state: jnp.ndarray,
    scale: float = 1.0,
) -> jnp.ndarray:
    """
    Compute velocity matching target.

    Args:
        current_state: (batch, dim) current state
        next_state: (batch, dim) next state
        scale: Optional scaling factor

    Returns:
        (batch, dim) velocity target
    """
    return (next_state - current_state) * scale


# Integration helper for trainer.py
def apply_flow_matching_loss(
    pred_states: jnp.ndarray,
    actual_states: jnp.ndarray,
    prev_states: jnp.ndarray,
    use_flow_matching: bool = True,
    velocity_weight: float = 0.8,
) -> Tuple[jnp.ndarray, dict]:
    """
    Unified interface for prediction loss computation.

    This can be called directly from trainer.py to replace standard MSE loss.

    Args:
        pred_states: (batch, k, state_dim) predicted states at each step
        actual_states: (batch, k, state_dim) actual states at each step
        prev_states: (batch, k, state_dim) states at previous step
        use_flow_matching: If True, use velocity loss. If False, standard MSE.
        velocity_weight: Weight for velocity vs state loss

    Returns:
        (loss, metrics_dict)
    """
    if not use_flow_matching:
        # Standard state matching
        loss = jnp.mean((pred_states - actual_states) ** 2)
        return loss, {'state_mse': loss}

    # Flow matching: predict velocity
    # Velocity = pred_state - prev_state (what model predicts the change to be)
    # We want this to match actual_state - prev_state (actual change)
    pred_velocity = pred_states - prev_states
    actual_velocity = actual_states - prev_states

    velocity_loss = jnp.mean((pred_velocity - actual_velocity) ** 2)
    state_loss = jnp.mean((pred_states - actual_states) ** 2)

    total_loss = velocity_weight * velocity_loss + (1 - velocity_weight) * state_loss

    metrics = {
        'flow_velocity_loss': velocity_loss,
        'flow_state_loss': state_loss,
        'velocity_pred_norm': jnp.mean(jnp.linalg.norm(pred_velocity, axis=-1)),
        'velocity_actual_norm': jnp.mean(jnp.linalg.norm(actual_velocity, axis=-1)),
    }

    return total_loss, metrics

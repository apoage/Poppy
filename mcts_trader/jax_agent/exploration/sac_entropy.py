"""
SAC-style Adaptive Entropy Coefficient.

TASK_040: Learnable entropy coefficient with target constraint.

The problem with threshold-based adaptive entropy:
- Reactive: Only triggers AFTER entropy has collapsed
- Discontinuous: Sudden jumps in coefficient
- No target: Doesn't maintain consistent entropy level

SAC-style solution:
- Treat entropy as constraint: H(π) ≥ H_target
- Learnable Lagrange multiplier α adapts automatically
- Smooth: Gradual adjustment, no discontinuities
- Proactive: Maintains target throughout training

References:
- Soft Actor-Critic (Haarnoja et al., 2018)
- SAC v2: Automatic entropy temperature adjustment
"""

import jax
import jax.numpy as jnp
import optax
from typing import Tuple, NamedTuple, Optional, Any
import chex


class SACEntropyState(NamedTuple):
    """State for SAC-style learnable entropy coefficient."""
    log_alpha: jnp.ndarray  # log(α) for numerical stability
    opt_state: Any          # Optimizer state for α
    target_entropy: float   # Target entropy in nats


def init_sac_entropy(
    target_entropy: Optional[float] = None,
    num_actions: int = 320,
    target_entropy_ratio: float = 0.5,
    init_alpha: float = 0.01,
    alpha_lr: float = 3e-4,
) -> Tuple[SACEntropyState, optax.GradientTransformation]:
    """
    Initialize SAC entropy state and optimizer.

    Args:
        target_entropy: Explicit target entropy. If None, computed from ratio.
        num_actions: Number of actions (for computing target from ratio)
        target_entropy_ratio: Target as fraction of max entropy (0.5 = 50%)
        init_alpha: Initial entropy coefficient
        alpha_lr: Learning rate for alpha

    Returns:
        (state, optimizer) tuple
    """
    # Compute target entropy
    if target_entropy is None:
        max_entropy = jnp.log(num_actions)  # ~5.77 for 320 actions
        target_entropy = target_entropy_ratio * max_entropy

    # Initialize log_alpha
    log_alpha = jnp.array(jnp.log(init_alpha))

    # Optimizer for alpha
    optimizer = optax.adam(alpha_lr)
    opt_state = optimizer.init(log_alpha)

    state = SACEntropyState(
        log_alpha=log_alpha,
        opt_state=opt_state,
        target_entropy=float(target_entropy),
    )

    return state, optimizer


def update_alpha(
    state: SACEntropyState,
    optimizer: optax.GradientTransformation,
    policy_entropy: jnp.ndarray,
    clip_min: float = -10.0,  # α_min = exp(-10) ≈ 4.5e-5
    clip_max: float = 0.0,    # α_max = exp(0) = 1.0
) -> Tuple[SACEntropyState, jnp.ndarray, dict]:
    """
    Update entropy coefficient based on observed entropy.

    The loss is: L(α) = α * (H(π) - H_target)

    - If H(π) < H_target: L > 0 → minimize → increase α
    - If H(π) > H_target: L < 0 → minimize → decrease α

    Args:
        state: Current SAC entropy state
        optimizer: Optimizer for alpha
        policy_entropy: (batch,) or scalar observed entropy
        clip_min: Minimum log_alpha (prevents α→0)
        clip_max: Maximum log_alpha (prevents α→∞)

    Returns:
        (new_state, alpha, metrics_dict)
    """
    log_alpha = state.log_alpha
    target_entropy = state.target_entropy

    # Mean entropy if batched
    if policy_entropy.ndim > 0:
        policy_entropy = jnp.mean(policy_entropy)

    def alpha_loss_fn(log_alpha):
        alpha = jnp.exp(log_alpha)
        # Negative because we want to maximize entropy (minimize -entropy)
        return -alpha * (policy_entropy - target_entropy)

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(log_alpha)

    # Update
    updates, new_opt_state = optimizer.update(
        alpha_grad, state.opt_state, log_alpha
    )
    new_log_alpha = optax.apply_updates(log_alpha, updates)

    # Clip to valid range
    new_log_alpha = jnp.clip(new_log_alpha, clip_min, clip_max)

    new_state = SACEntropyState(
        log_alpha=new_log_alpha,
        opt_state=new_opt_state,
        target_entropy=target_entropy,
    )

    alpha = jnp.exp(new_log_alpha)

    metrics = {
        'sac_alpha': float(alpha),
        'sac_log_alpha': float(new_log_alpha),
        'sac_alpha_loss': float(alpha_loss),
        'sac_entropy_error': float(policy_entropy - target_entropy),
        'sac_target_entropy': target_entropy,
    }

    return new_state, alpha, metrics


def get_alpha(state: SACEntropyState) -> jnp.ndarray:
    """Get current entropy coefficient."""
    return jnp.exp(state.log_alpha)


def compute_entropy_bonus(
    policy_logits: jnp.ndarray,
    alpha: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute entropy bonus for policy loss.

    Args:
        policy_logits: (batch, num_actions) raw logits
        alpha: Entropy coefficient

    Returns:
        (entropy_bonus, entropy) tuple
        - entropy_bonus: Scalar bonus to add to policy loss
        - entropy: (batch,) entropy values
    """
    policy_probs = jax.nn.softmax(policy_logits, axis=-1)
    log_probs = jax.nn.log_softmax(policy_logits, axis=-1)

    # Entropy: -sum(p * log(p))
    entropy = -jnp.sum(policy_probs * log_probs, axis=-1)

    # Entropy bonus (positive, add to minimize negative loss)
    entropy_bonus = alpha * jnp.mean(entropy)

    return entropy_bonus, entropy


class AdaptiveEntropyManager:
    """
    Unified entropy management combining SAC-style and threshold-based approaches.

    Provides both:
    1. SAC-style learnable α with target entropy constraint
    2. Emergency boost when entropy drops below critical threshold
    """

    def __init__(
        self,
        num_actions: int = 320,
        # SAC-style params
        use_sac_entropy: bool = True,
        target_entropy_ratio: float = 0.5,
        init_alpha: float = 0.01,
        alpha_lr: float = 3e-4,
        # Emergency threshold params
        emergency_threshold: float = 1.0,  # ~17% of max
        emergency_alpha: float = 0.5,
    ):
        """
        Args:
            num_actions: Size of action space
            use_sac_entropy: Use SAC-style learnable alpha
            target_entropy_ratio: Target as ratio of max entropy
            init_alpha: Initial alpha for SAC
            alpha_lr: Learning rate for alpha
            emergency_threshold: If entropy falls below this, trigger emergency
            emergency_alpha: Alpha to use during emergency
        """
        self.num_actions = num_actions
        self.max_entropy = jnp.log(num_actions)
        self.use_sac_entropy = use_sac_entropy
        self.emergency_threshold = emergency_threshold
        self.emergency_alpha = emergency_alpha

        if use_sac_entropy:
            self.sac_state, self.sac_optimizer = init_sac_entropy(
                num_actions=num_actions,
                target_entropy_ratio=target_entropy_ratio,
                init_alpha=init_alpha,
                alpha_lr=alpha_lr,
            )
        else:
            self.sac_state = None
            self.sac_optimizer = None

        self.current_alpha = init_alpha
        self.in_emergency = False

    def update(self, policy_entropy: jnp.ndarray) -> Tuple[float, dict]:
        """
        Update entropy coefficient based on observed entropy.

        Args:
            policy_entropy: Observed policy entropy

        Returns:
            (alpha, metrics_dict)
        """
        mean_entropy = float(jnp.mean(policy_entropy))
        metrics = {'observed_entropy': mean_entropy}

        if self.use_sac_entropy:
            # SAC-style update
            self.sac_state, alpha, sac_metrics = update_alpha(
                self.sac_state,
                self.sac_optimizer,
                policy_entropy,
            )
            metrics.update(sac_metrics)
            self.current_alpha = float(alpha)
        else:
            alpha = self.current_alpha

        # Emergency override check
        if mean_entropy < self.emergency_threshold:
            if not self.in_emergency:
                self.in_emergency = True
                metrics['emergency_triggered'] = True
            alpha = self.emergency_alpha
            metrics['using_emergency_alpha'] = True
        else:
            if self.in_emergency:
                self.in_emergency = False
                metrics['emergency_recovered'] = True

        metrics['final_alpha'] = float(alpha)
        metrics['in_emergency'] = self.in_emergency

        return float(alpha), metrics

    def get_alpha(self) -> float:
        """Get current alpha without updating."""
        if self.in_emergency:
            return self.emergency_alpha
        return self.current_alpha

    def get_state(self) -> dict:
        """Get state for checkpointing."""
        state = {
            'current_alpha': self.current_alpha,
            'in_emergency': self.in_emergency,
        }
        if self.sac_state is not None:
            state['sac_log_alpha'] = float(self.sac_state.log_alpha)
        return state

    def load_state(self, state: dict):
        """Load state from checkpoint."""
        self.current_alpha = state.get('current_alpha', self.current_alpha)
        self.in_emergency = state.get('in_emergency', False)
        if self.sac_state is not None and 'sac_log_alpha' in state:
            new_log_alpha = jnp.array(state['sac_log_alpha'])
            self.sac_state = SACEntropyState(
                log_alpha=new_log_alpha,
                opt_state=self.sac_state.opt_state,
                target_entropy=self.sac_state.target_entropy,
            )


# Utility functions

def target_entropy_from_hold_ratio(
    target_hold_pct: float = 0.33,
    num_actions: int = 320,
    num_assets: int = 64,
) -> float:
    """
    Compute target entropy that would give desired HOLD percentage.

    Assumes uniform distribution over non-HOLD actions.

    Args:
        target_hold_pct: Desired HOLD percentage (0.33 = 33%)
        num_actions: Total number of actions
        num_assets: Number of assets (HOLD actions = num_assets)

    Returns:
        Target entropy in nats
    """
    # If HOLD for all assets is target_hold_pct,
    # then each HOLD action gets target_hold_pct / num_assets
    # and non-HOLD actions split the rest

    hold_prob_per_asset = target_hold_pct / num_assets
    non_hold_prob_total = 1.0 - target_hold_pct
    non_hold_actions = num_actions - num_assets
    non_hold_prob_per_action = non_hold_prob_total / non_hold_actions

    # Compute entropy
    entropy = 0.0
    # HOLD actions
    entropy -= num_assets * hold_prob_per_asset * jnp.log(hold_prob_per_asset + 1e-10)
    # Non-HOLD actions
    entropy -= non_hold_actions * non_hold_prob_per_action * jnp.log(non_hold_prob_per_action + 1e-10)

    return float(entropy)

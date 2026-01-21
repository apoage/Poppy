import jax
import jax.numpy as jnp
import optax
import flax
from flax import traverse_util
from flax.training import train_state
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import chex

from .muzero_model import MuZeroNetworks
from .atc import compute_atc_loss, create_atc_encoder_fn

# TASK_034: Reward scaling for categorical binning
# Per-step symlog rewards are tiny (~0.001 to 0.05) and would collapse to
# bins 99-100 out of 201 without scaling. Scale by 100 to spread across bins.
# This makes reward range [-5, +5] (typical) map to [-5, +5] in support space.
REWARD_SCALE = 100.0


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


from .mctx_search import create_mctx_search
from .replay_buffer import JAXReplayBuffer


class TrainState(train_state.TrainState):
    """Extended train state with additional metrics"""
    step_count: int = 0


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


def compute_annealed_lambda(step: int, config: PredictionLossConfig) -> float:
    """Compute annealed prediction loss weight."""
    progress = jnp.clip(step / config.anneal_steps, 0.0, 1.0)
    return config.weight_initial + progress * (config.weight_final - config.weight_initial)


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


class MuZeroTrainer:
    """
    Complete JAX MuZero training pipeline.
    Fully GPU-accelerated with JIT compilation.

    Extended with auxiliary prediction loss (Phase 2).
    """

    def __init__(
        self,
        model: MuZeroNetworks,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        num_simulations: int = 32,
        batch_size: int = 256,
        unroll_steps: int = 5,
        td_steps: int = 10,
        discount: float = 0.997,
        value_loss_weight: float = 1.0,
        reward_loss_weight: float = 1.0,
        policy_loss_weight: float = 1.0,
        entropy_coef: float = 0.01,
        # Value loss configuration
        value_loss_type: str = 'mse',  # 'mse' or 'huber'
        value_huber_delta: float = 1.0,  # Delta for Huber loss
        target_update_interval: int = 100,
        # TASK_024: LR schedule
        lr_schedule_type: str = 'constant',  # 'constant', 'cosine', 'linear'
        lr_min: float = 1e-6,
        lr_warmup_epochs: int = 10,
        total_epochs: int = 500,
        steps_per_epoch: int = 100,
        # Prediction loss parameters (Phase 2)
        use_prediction_loss: bool = False,
        prediction_config: Optional[PredictionLossConfig] = None,
        # DS24 gradient monitoring
        track_grad_norms: bool = False,
        # PopArt normalization (TASK_021)
        use_popart: bool = False,
        popart_beta: float = 0.0001,
        # TASK_025: Per-head gradient clipping
        per_head_clipping: bool = False,
        clip_grad_value: float = 1.0,
        clip_grad_policy: float = 1.0,
        clip_grad_dynamics: float = 1.0,
        clip_grad_encoder: float = 1.0,
        # TASK_043: ATC contrastive learning
        atc_loss: bool = False,
        atc_weight: float = 0.1,
        atc_k_step: int = 5,
        atc_temperature: float = 0.1,
        atc_num_negatives: int = 32,
    ):
        self.model = model
        self.track_grad_norms = track_grad_norms

        # TASK_025: Per-head gradient clipping
        self.per_head_clipping = per_head_clipping
        self.clip_grad_value = clip_grad_value
        self.clip_grad_policy = clip_grad_policy
        self.clip_grad_dynamics = clip_grad_dynamics
        self.clip_grad_encoder = clip_grad_encoder

        # TASK_043: ATC contrastive learning
        self.atc_loss = atc_loss
        self.atc_weight = atc_weight
        self.atc_k_step = atc_k_step
        self.atc_temperature = atc_temperature
        self.atc_num_negatives = atc_num_negatives
        if atc_loss:
            self.atc_encoder_fn = create_atc_encoder_fn(model.apply)
        else:
            self.atc_encoder_fn = None

        # PopArt value normalization
        self.use_popart = use_popart
        if use_popart:
            from .popart import PopArtNormalizer
            self.popart = PopArtNormalizer(beta=popart_beta)
        else:
            self.popart = None
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps
        self.discount = discount
        self.target_update_interval = target_update_interval

        # Loss weights
        self.value_loss_weight = value_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.entropy_coef = entropy_coef

        # Value loss configuration
        self.value_loss_type = value_loss_type
        self.value_huber_delta = value_huber_delta

        # Prediction loss (Phase 2)
        self.use_prediction_loss = use_prediction_loss and model.use_prediction_head
        self.prediction_config = prediction_config or PredictionLossConfig()

        # Create MCTX search
        self.num_simulations = num_simulations
        self.search = create_mctx_search(
            model,
            num_simulations=num_simulations,
            use_gumbel=True
        )

        # TASK_024: Create optimizer with LR schedule
        self.lr_schedule_type = lr_schedule_type
        if lr_schedule_type == 'constant':
            lr_schedule = learning_rate
        else:
            from .schedules import LRSchedule
            lr_sched = LRSchedule(
                initial_lr=learning_rate,
                final_lr=lr_min,
                warmup_epochs=lr_warmup_epochs,
                total_epochs=total_epochs,
                schedule_type=lr_schedule_type,
            )
            lr_schedule = lr_sched.to_optax_schedule(steps_per_epoch)

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
        )

        # Initialize training state
        self.state = None

        # JIT-compile training step
        self.train_step_fn = jax.jit(self._train_step)

    def update_simulations(self, num_simulations: int) -> None:
        """
        Update MCTS simulation count (for warmup schedule).

        Recreates the search function with new simulation count.
        Used during sims warmup to gradually increase planning depth.

        Args:
            num_simulations: New number of MCTS simulations
        """
        if num_simulations != self.num_simulations:
            self.num_simulations = num_simulations
            self.search = create_mctx_search(
                self.model,
                num_simulations=num_simulations,
                use_gumbel=True
            )

    def init_train_state(self, rng: chex.PRNGKey, dummy_obs: jnp.ndarray) -> TrainState:
        """Initialize training state with model parameters"""
        # Initialize all parameters using mode='initialize_parameters'
        variables = self.model.init(rng, dummy_obs, mode='initialize_parameters')
        params = variables['params']

        state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
            step_count=0
        )

        self.state = state
        return state

    def _scalar_to_categorical(self, scalar: jnp.ndarray, support_size: int = None, smoothing: float = 0.1) -> jnp.ndarray:
        """
        Convert scalar value to categorical distribution with label smoothing.

        Args:
            scalar: (batch,) scalar values in range [-10, 10]
            support_size: Number of bins (defaults to model's support_size)
            smoothing: Label smoothing factor (0.1 = 10% uniform, 90% target)

        Returns:
            targets: (batch, support_size) smoothed probability distribution
        """
        # Use model's support_size if not provided
        if support_size is None:
            support_size = self.model.support_size

        # Use centralized support range config
        from . import get_support_range, get_bin_width
        support_min, support_max, _ = get_support_range()

        # Clip to support range
        scalar = jnp.clip(scalar, support_min, support_max)

        # Map to bin indices using centralized config
        bin_width = get_bin_width()

        # Soft binning (distribute value between adjacent bins)
        indices = (scalar - support_min) / bin_width
        lower_idx = jnp.floor(indices).astype(jnp.int32)
        upper_idx = jnp.clip(lower_idx + 1, 0, support_size - 1)

        # Interpolation weight
        weight_upper = indices - lower_idx
        weight_lower = 1.0 - weight_upper

        # Create target distribution
        batch_size = scalar.shape[0]
        targets = jnp.zeros((batch_size, support_size))
        targets = targets.at[jnp.arange(batch_size), lower_idx].set(weight_lower)
        targets = targets.at[jnp.arange(batch_size), upper_idx].add(weight_upper)

        # Apply label smoothing: mix with uniform distribution
        # This prevents overconfident predictions and helps exploration
        uniform = jnp.ones((batch_size, support_size)) / support_size
        targets = (1.0 - smoothing) * targets + smoothing * uniform

        return targets

    def _logits_to_scalar(self, logits: jnp.ndarray) -> jnp.ndarray:
        """
        Convert categorical value logits to scalar expected value.

        Args:
            logits: (batch, support_size) value logits

        Returns:
            scalar_values: (batch,) expected values in support range
        """
        from . import get_value_support
        support = get_value_support()
        probs = jax.nn.softmax(logits, axis=-1)
        return jnp.sum(probs * support, axis=-1)

    def _hybrid_value_loss(
        self,
        value_logits: jnp.ndarray,
        target_values: jnp.ndarray,
        mse_weight: float = 0.8,
        debug: bool = False,
        valid_mask: jnp.ndarray = None,
        num_valid: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute hybrid MSE/Huber + Categorical value loss.

        MSE/Huber directly penalizes mean prediction error (prevents collapse to bin 50).
        Categorical maintains distributional information.
        Huber loss is more robust to outliers than MSE.

        Args:
            value_logits: (batch, support_size) predicted value distribution logits
            target_values: (batch,) scalar target values
            mse_weight: Weight for MSE/Huber component (0.8 = 80% MSE, 20% categorical)
            debug: Print debug info
            valid_mask: (batch,) boolean mask for valid samples (BUG_002 fix)
            num_valid: Number of valid samples (for masked mean)

        Returns:
            total_loss: Combined loss
            mse_loss: MSE/Huber component (for metrics)
            cat_loss: Categorical component (for metrics)
        """
        # MSE or Huber on expected value
        predicted_values = self._logits_to_scalar(value_logits)

        if self.value_loss_type == 'huber':
            # Huber loss: quadratic for small errors, linear for large
            # More robust to high-variance targets
            delta = self.value_huber_delta
            abs_error = jnp.abs(predicted_values - target_values)
            quadratic = 0.5 * abs_error ** 2
            linear = delta * (abs_error - 0.5 * delta)
            per_sample_mse = jnp.where(abs_error <= delta, quadratic, linear)
        else:
            # Standard MSE
            per_sample_mse = (predicted_values - target_values) ** 2

        # Categorical cross-entropy (keep some distributional learning)
        target_dist = self._scalar_to_categorical(target_values)
        per_sample_cat = optax.softmax_cross_entropy(value_logits, target_dist)

        # BUG_002 FIX: Apply valid_mask before averaging
        if valid_mask is not None and num_valid is not None:
            valid_mask_float = valid_mask.astype(jnp.float32)
            mse_loss = jnp.sum(per_sample_mse * valid_mask_float) / num_valid
            cat_loss = jnp.sum(per_sample_cat * valid_mask_float) / num_valid
        else:
            mse_loss = jnp.mean(per_sample_mse)
            cat_loss = jnp.mean(per_sample_cat)

        # Debug: Check if predictions vary
        if debug:
            pred_mean = jnp.mean(predicted_values)
            pred_std = jnp.std(predicted_values)
            tgt_mean = jnp.mean(target_values)
            tgt_std = jnp.std(target_values)
            print(f"    [VALUE DEBUG] Pred: mean={pred_mean:.4f} std={pred_std:.4f} | Tgt: mean={tgt_mean:.4f} std={tgt_std:.4f}")

        # Combine
        total_loss = mse_weight * mse_loss + (1.0 - mse_weight) * cat_loss

        return total_loss, mse_loss, cat_loss

    def _compute_prediction_loss(
        self,
        params: flax.core.FrozenDict,
        latent_states: jnp.ndarray,
        batch: Dict[str, jnp.ndarray],
        rng: chex.PRNGKey,
        step: int,
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Compute multi-step prediction loss in LOG-SPACE.

        Loss: sum_{k=1}^{K} gamma^{k-1} * MSE(pred_log_p_{t+k}, true_log_p_{t+k})

        Args:
            params: Model parameters
            latent_states: (batch, state_dim) encoded latent states
            batch: Training batch with auxiliary data
            rng: Random key
            step: Current training step (for annealing)

        Returns:
            loss_prediction: Scalar loss
            metrics: Dict with per-step MSE
        """
        config = self.prediction_config
        batch_size = latent_states.shape[0]

        # Get scheduled sampling probability
        teacher_prob = compute_scheduled_sampling_prob(step, config)

        # TASK_035: Get neural scale for physics warmup
        neural_scale = compute_neural_scale(step, config)

        # Get ground truth multi-step sequence
        # Shape: (batch, sequence_length, n_assets)
        teacher_sequence = batch.get('log_price_sequence')
        if teacher_sequence is None:
            # No auxiliary data available
            return jnp.array(0.0), {
                'loss_prediction_raw': 0.0,
                'step_1_mse': 0.0,
                'step_k_mse': 0.0,
                'teacher_forcing_prob': teacher_prob,
                'neural_scale': neural_scale,
            }

        # Get features
        features = batch.get('features', jnp.zeros((batch_size, self.model.n_features)))

        # Get action sequence (first unroll_steps actions)
        actions = batch.get('actions')
        if actions is not None and actions.ndim == 2:
            # (batch, sequence_length) -> take first multi_step_k
            action_sequence = actions[:, :config.multi_step_k]
        else:
            action_sequence = None

        # Multi-step prediction - TASK_035: Pass neural_scale for physics warmup
        predictions, _ = self.model.apply(
            {'params': params},
            latent_states,
            features,
            rng,
            action_sequence=action_sequence,
            num_steps=config.multi_step_k,
            teacher_sequence=teacher_sequence,
            scheduled_sampling_prob=teacher_prob,
            neural_scale=neural_scale,
            method=self.model.predict_multi_step,
        )

        # Compute weighted multi-step loss
        # predictions: (batch, K, n_assets)
        # teacher_sequence: (batch, seq_len, n_assets) - index 0 is current, 1:K+1 are targets
        k = min(config.multi_step_k, teacher_sequence.shape[1] - 1)
        target_sequence = teacher_sequence[:, 1:k+1, :]  # (batch, K, n_assets)
        pred_sequence = predictions[:, :k, :]

        # Per-step MSE
        step_mse = jnp.mean((pred_sequence - target_sequence) ** 2, axis=-1)  # (batch, K)

        # Weighted sum with decay
        step_weights = jnp.array([config.step_decay ** i for i in range(k)])
        weighted_mse = jnp.sum(step_mse * step_weights[None, :], axis=-1)  # (batch,)

        loss_prediction = jnp.mean(weighted_mse)

        metrics = {
            'loss_prediction_raw': loss_prediction,
            'step_1_mse': jnp.mean(step_mse[:, 0]) if k > 0 else 0.0,
            'step_k_mse': jnp.mean(step_mse[:, -1]) if k > 0 else 0.0,
            'teacher_forcing_prob': teacher_prob,
            'neural_scale': neural_scale,  # TASK_035: Track physics warmup progress
        }

        return loss_prediction, metrics

    def _compute_physics_regularization(
        self,
        params: flax.core.FrozenDict,
    ) -> jnp.ndarray:
        """
        Physics regularization loss to prevent degenerate solutions.

        TASK_035 FIX: Changed from ceiling-only to two-sided TARGET regularization.
        The old ceiling-only approach (max(x - ceiling, 0)²) had ZERO penalty for
        low values, causing α and β to collapse to zero.

        New approach: Pull parameters toward sensible targets with bidirectional penalty.
        - alpha (momentum): target ~0.05 (small positive momentum)
        - beta (reversion): target ~0.03 (small mean reversion)
        - sigma (volatility): target ~0.01 (typical 1% volatility)
        - GARCH persistence: floor at 0.85, ceiling at 1.0
        - factor_loadings: L1 sparsity for interpretability
        """
        # Navigate to prediction head params
        pred_params = params.get('price_predictor', {})
        physics_params = pred_params.get('physics_core', {})

        reg_loss = jnp.array(0.0)

        # TASK_035 FIX: Two-sided TARGET regularization for momentum
        # OLD (broken): jnp.maximum(alpha - 0.3, 0.0) ** 2  → α=0 has ZERO penalty
        # NEW: Pull toward target with bidirectional penalty
        if 'alpha' in physics_params:
            alpha = jax.nn.softplus(physics_params['alpha']) * 0.3  # Scale to [0, ~0.3]
            alpha_target = 0.05  # Target: small positive momentum
            reg_loss = reg_loss + jnp.mean((alpha - alpha_target) ** 2) * 0.5

        # TASK_035 FIX: Two-sided TARGET regularization for mean-reversion
        if 'beta' in physics_params:
            beta = jax.nn.softplus(physics_params['beta']) * 0.2  # Scale to [0, ~0.2]
            beta_target = 0.03  # Target: small mean reversion
            reg_loss = reg_loss + jnp.mean((beta - beta_target) ** 2) * 0.5

        # Volatility: penalize extreme values (want ~0.01) - already two-sided
        if 'sigma' in physics_params:
            sigma = jax.nn.softplus(physics_params['sigma']) * 0.1
            reg_loss = reg_loss + jnp.mean((sigma - 0.01) ** 2)

        # Low-rank correlation: L1 sparsity on factor loadings
        correlation_params = physics_params.get('correlation', {})
        if 'factor_loadings' in correlation_params:
            loadings = correlation_params['factor_loadings']
            reg_loss = reg_loss + 0.01 * jnp.mean(jnp.abs(loadings))

        # TASK_035 FIX: GARCH persistence floor + ceiling
        # OLD: Only ceiling at 0.99, no floor → persistence collapsed to 0.096
        # NEW: Floor at 0.85, ceiling at 1.0 for realistic volatility clustering
        garch_params = physics_params.get('garch', {})
        if 'alpha_garch' in garch_params and 'beta_garch' in garch_params:
            alpha_g = jax.nn.softplus(garch_params['alpha_garch']) * 0.3
            beta_g = jax.nn.softplus(garch_params['beta_garch']) * 0.99
            persistence = alpha_g + beta_g

            # Floor: penalize if persistence < 0.85 (too low = no volatility memory)
            floor_penalty = jnp.mean(jnp.maximum(0.85 - persistence, 0.0) ** 2)
            # Ceiling: penalize if persistence > 1.0 (non-stationary)
            ceiling_penalty = jnp.mean(jnp.maximum(persistence - 1.0, 0.0) ** 2)
            # Strong weight to enforce realistic GARCH behavior
            reg_loss = reg_loss + (floor_penalty + ceiling_penalty) * 2.0

        return reg_loss

    def _loss_fn(
        self,
        params: flax.core.FrozenDict,
        batch: Dict[str, jnp.ndarray],
        rng: chex.PRNGKey,
        step: int = 0,  # For prediction loss annealing
        entropy_coef: float = 0.01,  # Dynamic entropy coefficient (not captured by JIT)
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute MuZero loss with optional prediction auxiliary loss.

        Args:
            params: Model parameters
            batch: Training batch
            rng: Random key
            step: Current training step (for annealing)

        Returns:
            total_loss: Scalar loss
            metrics: Dict of metrics
        """
        observations = batch['observations']  # (batch, unroll_steps, 27, 64, 64)
        actions = batch['actions']  # (batch, unroll_steps)
        rewards = batch['rewards']  # (batch, unroll_steps)
        target_values = batch['values']  # (batch, unroll_steps)
        target_policies = batch['policy_probs']  # (batch, unroll_steps, 192)

        # Validity mask for handling buffer wraparound (BUG_006 fix)
        valid_mask = batch.get('valid', jnp.ones(observations.shape[0], dtype=jnp.bool_))
        num_valid = jnp.maximum(jnp.sum(valid_mask), 1.0)  # Avoid div by zero
        valid_mask_float = valid_mask.astype(jnp.float32)

        batch_size, unroll_steps = actions.shape[:2]

        # Initial inference
        initial_output = self.model.apply(
            {'params': params},
            observations[:, 0],  # First observation
            mode='initial_inference'
        )

        # BUG_002 FIX: Compute initial losses with valid_mask
        value_loss, value_mse, value_cat = self._hybrid_value_loss(
            initial_output['value'],
            target_values[:, 0],
            valid_mask=valid_mask,
            num_valid=num_valid,
        )

        # BUG_002 FIX: Per-sample policy loss, masked before averaging
        per_sample_policy = optax.softmax_cross_entropy(
            initial_output['policy_logits'],
            target_policies[:, 0]
        )
        policy_loss = jnp.sum(per_sample_policy * valid_mask_float) / num_valid

        # Compute policy entropy for regularization (prevent collapse)
        policy_probs = jax.nn.softmax(initial_output['policy_logits'], axis=-1)
        per_sample_entropy = -jnp.sum(policy_probs * jnp.log(policy_probs + 1e-10), axis=-1)
        policy_entropy = jnp.sum(per_sample_entropy * valid_mask_float) / num_valid

        reward_loss = jnp.array(0.0)  # No reward at root

        # Track MSE/categorical components separately for metrics
        total_value_mse = value_mse
        total_value_cat = value_cat

        # Unroll dynamics for subsequent steps
        latent_state = initial_output['latent_state']

        for t in range(1, unroll_steps):
            # Recurrent inference
            recurrent_output = self.model.apply(
                {'params': params},
                latent_state,
                actions[:, t - 1],
                mode='recurrent_inference'
            )

            # Update latent state
            latent_state = recurrent_output['latent_state']

            # BUG_002 FIX: Compute losses for this step with valid_mask
            step_value_loss, step_value_mse, step_value_cat = self._hybrid_value_loss(
                recurrent_output['value'],
                target_values[:, t],
                valid_mask=valid_mask,
                num_valid=num_valid,
            )
            value_loss += step_value_loss
            total_value_mse += step_value_mse
            total_value_cat += step_value_cat

            # BUG_002 FIX: Per-sample policy loss, masked
            per_sample_step_policy = optax.softmax_cross_entropy(
                recurrent_output['policy_logits'],
                target_policies[:, t]
            )
            policy_loss += jnp.sum(per_sample_step_policy * valid_mask_float) / num_valid

            # BUG_001 FIX: Reward target must match action index (t-1, not t)
            # BUG_002 FIX: Per-sample reward loss, masked
            # TASK_034 FIX: Scale rewards before categorical binning
            # Raw symlog rewards (~0.001) would collapse to 1-2 bins without scaling
            scaled_rewards = rewards[:, t - 1] * REWARD_SCALE
            per_sample_reward = optax.softmax_cross_entropy(
                recurrent_output['reward'],
                self._scalar_to_categorical(scaled_rewards)
            )
            reward_loss += jnp.sum(per_sample_reward * valid_mask_float) / num_valid

            # Accumulate entropy (masked)
            step_policy_probs = jax.nn.softmax(recurrent_output['policy_logits'], axis=-1)
            per_sample_step_entropy = -jnp.sum(step_policy_probs * jnp.log(step_policy_probs + 1e-10), axis=-1)
            policy_entropy += jnp.sum(per_sample_step_entropy * valid_mask_float) / num_valid

        # Average over unroll steps
        value_loss /= unroll_steps
        policy_loss /= unroll_steps
        reward_loss /= (unroll_steps - 1)  # No reward at t=0
        policy_entropy /= unroll_steps  # Average entropy
        total_value_mse /= unroll_steps
        total_value_cat /= unroll_steps

        # Adaptive entropy bonus - stronger push when entropy drops too low
        # Target entropy ~4.0 (near-uniform over 192 actions = ln(192) ≈ 5.26)
        min_entropy = 3.0  # Danger zone threshold
        adaptive_bonus = jnp.where(
            policy_entropy < min_entropy,
            entropy_coef * 5.0 * (min_entropy - policy_entropy),  # 5x stronger push when too low
            0.0
        )

        # Base MuZero loss (entropy bonus is SUBTRACTED to maximize it)
        muzero_loss = (
            self.value_loss_weight * value_loss +
            self.policy_loss_weight * policy_loss +
            self.reward_loss_weight * reward_loss
            - entropy_coef * policy_entropy  # Maximize entropy to prevent collapse
            + adaptive_bonus  # Extra penalty when entropy drops below threshold
        )

        # Initialize metrics
        metrics = {
            'muzero_loss': muzero_loss,
            'value_loss': value_loss,
            'value_mse': total_value_mse,
            'value_cat': total_value_cat,
            'policy_loss': policy_loss,
            'reward_loss': reward_loss,
            'policy_entropy': policy_entropy,
            'valid_samples': num_valid,  # Track buffer validity (BUG_006)
            'valid_ratio': num_valid / batch_size,  # Fraction of valid samples
        }

        # ================================================================
        # Auxiliary Prediction Loss (Phase 2)
        # ================================================================
        if self.use_prediction_loss:
            rng, pred_rng = jax.random.split(rng)
            config = self.prediction_config

            # Compute annealed lambda
            lambda_pred = compute_annealed_lambda(step, config)

            # Multi-step prediction loss
            loss_prediction, pred_metrics = self._compute_prediction_loss(
                params, initial_output['latent_state'], batch, pred_rng, step
            )

            # Physics regularization
            loss_physics_reg = self._compute_physics_regularization(params)

            # Combined loss with annealed weight
            total_loss = (
                muzero_loss +
                lambda_pred * loss_prediction +
                config.physics_reg_weight * loss_physics_reg
            )

            metrics.update({
                'loss_prediction': loss_prediction,
                'loss_physics_reg': loss_physics_reg,
                'lambda_pred': lambda_pred,
                **pred_metrics,
            })
        else:
            total_loss = muzero_loss
            metrics['loss_prediction'] = 0.0
            metrics['lambda_pred'] = 0.0

        # ================================================================
        # TASK_043: ATC Contrastive Loss
        # ================================================================
        if self.atc_loss and self.atc_encoder_fn is not None:
            rng, atc_rng = jax.random.split(rng)
            atc_loss_val = compute_atc_loss(
                encoder_fn=self.atc_encoder_fn,
                params=params,
                observations=batch['observations'],
                k_step=self.atc_k_step,
                num_negatives=self.atc_num_negatives,
                temperature=self.atc_temperature,
                rng=atc_rng,
            )
            total_loss = total_loss + self.atc_weight * atc_loss_val
            metrics['atc_loss'] = atc_loss_val
        else:
            metrics['atc_loss'] = 0.0

        metrics['total_loss'] = total_loss

        return total_loss, metrics

    def _train_step(
        self,
        state: TrainState,
        batch: Dict[str, jnp.ndarray],
        rng: chex.PRNGKey,
        entropy_coef: float = 0.01,  # Dynamic entropy coefficient
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """
        Single training step (JIT-compiled).

        Args:
            state: Training state
            batch: Training batch
            rng: Random key
            entropy_coef: Entropy regularization coefficient (passed dynamically, not captured by JIT)

        Returns:
            new_state: Updated training state
            metrics: Training metrics
        """
        # Compute gradients (pass step count for prediction loss annealing)
        # Pass entropy_coef as dynamic parameter so adaptive entropy works
        grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params, batch, rng, state.step_count, entropy_coef)

        # TASK_025: Apply per-head gradient clipping before optimizer
        if self.per_head_clipping:
            grads = _clip_gradients_per_head(
                grads,
                max_norm_value=self.clip_grad_value,
                max_norm_policy=self.clip_grad_policy,
                max_norm_dynamics=self.clip_grad_dynamics,
                max_norm_encoder=self.clip_grad_encoder,
            )

        # Update parameters
        new_state = state.apply_gradients(grads=grads)

        # Update step count
        new_step_count = state.step_count + 1

        new_state = new_state.replace(
            step_count=new_step_count
        )

        # Add gradient norm to metrics
        metrics['grad_norm'] = optax.global_norm(grads)

        # Per-head gradient norms for DS24 monitoring (Level 4+)
        if getattr(self, 'track_grad_norms', False):
            per_head_norms = _compute_per_head_grad_norms(grads)
            metrics.update(per_head_norms)

        return new_state, metrics

    def train(
        self,
        replay_buffer: JAXReplayBuffer,
        num_iterations: int,
        rng: chex.PRNGKey
    ) -> Dict[str, list]:
        """
        Main training loop.

        Args:
            replay_buffer: Replay buffer with trajectories
            num_iterations: Number of training iterations
            rng: Random key

        Returns:
            history: Dict of training metrics over time
        """
        history = {
            'total_loss': [],
            'muzero_loss': [],
            'value_loss': [],
            'value_mse': [],
            'value_cat': [],
            'policy_loss': [],
            'reward_loss': [],
            'policy_entropy': [],
            'grad_norm': [],
            # Prediction loss metrics (Phase 2)
            'loss_prediction': [],
            'loss_physics_reg': [],
            'lambda_pred': [],
            'step_1_mse': [],
            'step_k_mse': [],
            # Per-head gradient norms (DS24 Level 4)
            'grad_norm_value': [],
            'grad_norm_policy': [],
            'grad_norm_dynamics': [],
        }

        for iteration in range(num_iterations):
            # Sample batch from replay buffer
            rng, sample_rng = jax.random.split(rng)
            batch = replay_buffer.sample_batch(sample_rng)

            # Debug: Check for NaNs/Infs in batch
            for k, v in batch.items():
                if jnp.isnan(v).any() or jnp.isinf(v).any():
                    print(f"!!! CRITICAL BATCH FAILURE at iteration {iteration} !!!")
                    print(f"Key: {k}")
                    print(f"  Min: {jnp.min(v)}")
                    print(f"  Max: {jnp.max(v)}")
                    print(f"  Mean: {jnp.mean(v)}")
                    raise ValueError(f"NaNs/Infs detected in training batch key '{k}'!")

            # PopArt: normalize targets and correct value head weights (TASK_021)
            if self.use_popart and self.popart is not None:
                # Get raw value targets (MC returns)
                raw_values = batch['values'].flatten()

                # Update statistics, normalize targets, correct weights
                norm_values, corrected_params = self.popart.normalize_and_correct(
                    raw_values, self.state.params
                )

                # Reshape back to original batch shape and update batch
                batch = {**batch, 'values': norm_values.reshape(batch['values'].shape)}

                # Update state with corrected params (preserves outputs after normalization change)
                self.state = self.state.replace(params=corrected_params)

            # Training step (pass entropy_coef dynamically so adaptive entropy works)
            rng, train_rng = jax.random.split(rng)
            self.state, metrics = self.train_step_fn(self.state, batch, train_rng, self.entropy_coef)

            # Debug: Check value targets vs MSE every 50 iterations
            if (iteration + 1) % 50 == 0:
                tgt_vals = batch['values'][:, 0]  # First timestep targets
                print(f"    [VALUE CHECK] Batch targets: mean={float(jnp.mean(tgt_vals)):.4f} std={float(jnp.std(tgt_vals)):.4f} min={float(jnp.min(tgt_vals)):.2f} max={float(jnp.max(tgt_vals)):.2f}")
                # If MSE is near 0 but targets have variance, there's a bug
                if metrics['value_mse'] < 0.01 and float(jnp.std(tgt_vals)) > 0.5:
                    print(f"    ⚠️  WARNING: MSE={metrics['value_mse']:.6f} too low for target std={float(jnp.std(tgt_vals)):.2f}")

            # Record metrics
            for key, value in metrics.items():
                if key in history:
                    history[key].append(float(value))

            if jnp.isnan(metrics['total_loss']):
                print(f"!!! NAN LOSS DETECTED at iteration {iteration} !!!")
                print(f"Value Loss: {metrics['value_loss']}")
                print(f"Policy Loss: {metrics['policy_loss']}")
                print(f"Reward Loss: {metrics['reward_loss']}")
                raise ValueError("Training stopped due to NaN loss.")

            # Log progress and debug batch statistics
            if (iteration + 1) % 50 == 0:
                # Batch values are already MC returns (computed in run_self_play)
                batch_values = batch['values']
                print(f"  [Train Iter {iteration + 1}] Batch MC Returns: min={float(jnp.min(batch_values)):.2f} max={float(jnp.max(batch_values)):.2f} mean={float(jnp.mean(batch_values)):.2f} std={float(jnp.std(batch_values)):.2f}")

                # Base MuZero loss line
                loss_line = f"                       Loss: {metrics['total_loss']:.4f}"
                loss_line += f" | MuZ: {metrics.get('muzero_loss', metrics['total_loss']):.4f}"
                loss_line += f" | Val: {metrics['value_loss']:.4f} (MSE:{metrics['value_mse']:.6f})"
                loss_line += f" | Pol: {metrics['policy_loss']:.4f}"
                loss_line += f" | Rew: {metrics['reward_loss']:.4f}"
                loss_line += f" | Ent: {metrics['policy_entropy']:.2f}"

                # Prediction loss line (if enabled)
                if self.use_prediction_loss and metrics.get('loss_prediction', 0.0) > 0:
                    loss_line += f"\n                       Pred: {metrics['loss_prediction']:.6f}"
                    loss_line += f" (λ={metrics.get('lambda_pred', 0.0):.3f})"
                    loss_line += f" | Step1: {metrics.get('step_1_mse', 0.0):.6f}"
                    loss_line += f" | StepK: {metrics.get('step_k_mse', 0.0):.6f}"
                    loss_line += f" | PhysReg: {metrics.get('loss_physics_reg', 0.0):.6f}"

                print(loss_line)

        return history

    def save_checkpoint(self, path: str):
        """Save model checkpoint with additional metadata"""
        import os
        import json
        import numpy as np
        from flax.training import checkpoints

        # Save main state
        checkpoints.save_checkpoint(
            ckpt_dir=path,
            target=self.state,
            step=self.state.step_count,
            keep=3,
            overwrite=True
        )

        # Save additional trainer metadata (BUG_019 fix + architecture info)
        metadata = {
            'entropy_coef': float(self.entropy_coef),
            'step_count': int(self.state.step_count),
            # Architecture info for compatibility checking
            'use_prediction_loss': self.use_prediction_loss,
            'h_cycles': self.model.H_cycles,
            'l_cycles': self.model.L_cycles,
            'state_dim': self.model.state_dim,
            'use_prediction_head': self.model.use_prediction_head,
            'physics_only': getattr(self.model, 'physics_only', False),
        }
        metadata_path = os.path.join(path, 'trainer_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        # Save PopArt state if enabled (critical for curriculum continuity)
        if self.use_popart and self.popart is not None:
            popart_path = os.path.join(path, 'popart_state.npz')
            np.savez(
                popart_path,
                mean=float(self.popart.state.mean),
                var=float(self.popart.state.var),
                count=int(self.popart.state.count),
                mean_old=float(self.popart.state.mean_old),
                std_old=float(self.popart.state.std_old),
            )

    def load_checkpoint(self, path: str, step: int = None):
        """Load model checkpoint with additional metadata"""
        import os
        import json
        import numpy as np
        from flax.training import checkpoints

        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=path,
            target=self.state,
            step=step
        )

        # Restore additional trainer metadata if available (BUG_019 fix)
        metadata_path = os.path.join(path, 'trainer_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.entropy_coef = metadata.get('entropy_coef', self.entropy_coef)
                print(f"   Restored entropy_coef: {self.entropy_coef}")

                # Check architecture compatibility
                ckpt_pred = metadata.get('use_prediction_head', False)
                ckpt_physics = metadata.get('physics_only', False)
                ckpt_h = metadata.get('h_cycles')
                ckpt_l = metadata.get('l_cycles')
                ckpt_dim = metadata.get('state_dim')

                # Warn on prediction head mismatch (CRITICAL)
                if not ckpt_pred and self.model.use_prediction_head:
                    print(f"   ⚠️  WARNING: Checkpoint has NO prediction head, but model has prediction enabled!")
                    print(f"      Prediction weights will be RANDOM (not from checkpoint)")
                    print(f"      Consider: --prediction-loss-weight 0 or use matching checkpoint")
                elif ckpt_pred and not self.model.use_prediction_head:
                    print(f"   ℹ️  Note: Checkpoint has prediction head, model does not (weights ignored)")

                # Warn on physics_only mismatch
                if ckpt_pred and ckpt_physics != getattr(self.model, 'physics_only', False):
                    ckpt_mode = "physics-only" if ckpt_physics else "hybrid"
                    model_mode = "physics-only" if self.model.physics_only else "hybrid"
                    print(f"   ℹ️  Note: Checkpoint mode={ckpt_mode}, model mode={model_mode}")

                # Error on architecture mismatch that would cause shape errors
                if ckpt_h is not None and ckpt_h != self.model.H_cycles:
                    print(f"   ⚠️  WARNING: Checkpoint H_cycles={ckpt_h}, model H_cycles={self.model.H_cycles}")
                if ckpt_l is not None and ckpt_l != self.model.L_cycles:
                    print(f"   ⚠️  WARNING: Checkpoint L_cycles={ckpt_l}, model L_cycles={self.model.L_cycles}")
                if ckpt_dim is not None and ckpt_dim != self.model.state_dim:
                    print(f"   ⚠️  WARNING: Checkpoint state_dim={ckpt_dim}, model state_dim={self.model.state_dim}")

            except Exception as e:
                print(f"   Warning: Could not load trainer metadata: {e}")

        # Restore PopArt state if present and enabled (critical for curriculum)
        if self.use_popart and self.popart is not None:
            popart_path = os.path.join(path, 'popart_state.npz')
            if os.path.exists(popart_path):
                try:
                    from .popart import PopArtState
                    data = np.load(popart_path)
                    self.popart.state = PopArtState(
                        mean=jnp.array(data['mean']),
                        var=jnp.array(data['var']),
                        count=int(data['count']),
                        mean_old=jnp.array(data['mean_old']),
                        std_old=jnp.array(data['std_old']),
                    )
                    print(f"   Restored PopArt state: mean={float(self.popart.state.mean):.4f}, "
                          f"std={float(self.popart.std):.4f}, count={self.popart.state.count}")
                except Exception as e:
                    print(f"   Warning: Could not load PopArt state: {e}")
            else:
                print(f"   ℹ️  No PopArt state in checkpoint (starting fresh)")

        return restored_state

    def restore_checkpoint(self, path: str, step: int = None):
        """Alias for load_checkpoint"""
        return self.load_checkpoint(path, step)

    def reset_optimizer_state(self, state: TrainState) -> TrainState:
        """
        Reset optimizer state while preserving model parameters.

        Use when transferring to a new curriculum level to avoid
        momentum/variance mismatch from the old gradient distribution.

        Args:
            state: Current training state with stale optimizer momentum

        Returns:
            New training state with fresh optimizer state but same model params
        """
        # Re-initialize optimizer with current params
        new_opt_state = self.optimizer.init(state.params)

        # Create new state with fresh optimizer (keep model weights and target)
        return state.replace(opt_state=new_opt_state)

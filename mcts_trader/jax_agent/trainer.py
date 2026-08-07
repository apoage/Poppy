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
from .cycle_balance import compute_cycle_balance_loss, compute_cycle_atc_loss
from .prediction import hybrid_prediction_loss  # TASK_116: Student-t output
from .alpha_reward import compute_max_alpha_topk  # TASK_118: Alpha-relative reward

# TASK_125: Causal Expert Imprinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .expert_buffer import UnifiedIndexBuffer
    from .price_cache import PriceCache
    from .observation_encoder import ObservationEncoder

# TASK_034: Reward scaling for categorical binning
# Per-step symlog rewards are tiny (~0.001 to 0.05) and would collapse to
# bins 99-100 out of 201 without scaling. Scale by 100 to spread across bins.
# This makes reward range [-5, +5] (typical) map to [-5, +5] in support space.
REWARD_SCALE = 100.0


# TASK_772.4 (2026-05-23): loss / grad / config / schedule helpers extracted to
# .trainer_losses for modularity. Pure function move — no JIT closures here.
from .trainer_losses import (
    compute_large_margin_loss,
    compute_q_filtered_bc_loss,
    _compute_per_head_grad_norms,
    _clip_gradients_per_head,
    ExpertBCConfig,
    PredictionLossConfig,
    compute_annealed_lambda,
    compute_bc_weight,
    compute_scheduled_sampling_prob,
    compute_neural_scale,
)


from .mctx_search import create_mctx_search
from .replay_buffer import JAXReplayBuffer


class TrainState(train_state.TrainState):
    """Extended train state with step counter"""
    # BUG-015 FIX: Removed target_params - was dead code (defined, updated, but never used)
    # MuZero uses MCTS returns for value targets, not TD bootstrapping with target network
    step_count: int = 0


# TASK_772.4 (2026-05-23): ExpertBCConfig, PredictionLossConfig and schedule
# helpers moved to .trainer_losses (with the BC/grad helpers above). Re-imported
# at the top of this file alongside the others.


class CheckpointVanishedError(RuntimeError):
    """TASK_876 fix-2 B1: ``save_checkpoint`` returned a path that is NOT on disk.

    MEASURED on flax 0.10.7 (Orbax backend): with ``overwrite=False``, saving a step that is
    BEHIND the newest checkpoint on disk raises nothing and returns
    ``<dir>/checkpoint_<step>`` — and flax's own unconditional keep-N cleanup deletes that
    very directory inside the same call. Seed ``checkpoint_{10..50}`` with ``keep=3``, save
    step 15: returns ``.../checkpoint_15``, ``os.path.exists`` is False.

    Reporting that as a success is the collapse-recovery trap this whole task exists for: the
    run believes it has a resume point for its current trajectory and the audit trail asserts
    one, while none exists. The save path raises this instead.
    """

    def __init__(self, path, ckpt_dir, step, reason):
        self.path = str(path) if path is not None else None
        self.ckpt_dir = str(ckpt_dir)
        self.step = step
        self.reason = str(reason)
        super().__init__(
            f"checkpoint save reported success but nothing usable is on disk: "
            f"flax returned {self.path!r} for step {step} in {self.ckpt_dir} ({reason})")


class CheckpointSidecarError(RuntimeError):
    """TASK_876 fix-2 M3: the flax checkpoint IS on disk, a LATER sidecar write failed.

    ``save_checkpoint`` writes the flax directory first, then ``trainer_metadata.json``, then
    ``popart_state.npz``. If one of the sidecars fails (ENOSPC is the plausible case under
    exactly the disk pressure this task addresses) the checkpoint itself is real and must
    still be scored — it is NOT a collision and NOT a vanished save. ``.path`` carries the
    verified on-disk checkpoint so the caller can record it.
    """

    def __init__(self, path, what, cause):
        self.path = str(path) if path is not None else None
        self.what = str(what)
        super().__init__(f"checkpoint {self.path} was written, but {what} failed: {cause}")


def _verify_checkpoint_on_disk(written, ckpt_dir, step):
    """TASK_876 fix-2 B1: a save is 'successful' only if the artifact it claims exists.

    Raises :class:`CheckpointVanishedError` unless ``written`` names a real, non-empty
    checkpoint. Non-empty matters: flax/Orbax commits a DIRECTORY, and a half-written or
    self-pruned one can leave an empty shell that ``os.path.exists`` alone would accept.
    """
    import os
    if not written:
        raise CheckpointVanishedError(written, ckpt_dir, step, "no path returned")
    p = str(written)
    if not os.path.exists(p):
        raise CheckpointVanishedError(
            p, ckpt_dir, step,
            "path does not exist — flax's own keep-N cleanup removed it inside the same "
            "call (the save was behind the newest checkpoint already on disk)")
    if os.path.isdir(p):
        if not os.listdir(p):
            raise CheckpointVanishedError(p, ckpt_dir, step, "checkpoint directory is EMPTY")
    elif os.path.getsize(p) == 0:
        raise CheckpointVanishedError(p, ckpt_dir, step, "checkpoint file is zero bytes")
    return p


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
        # Prediction loss configuration
        pred_loss_type: str = 'mse',  # 'mse' or 'huber'
        pred_huber_delta: float = 1.0,  # Delta for prediction Huber loss
        # BUG-015: Removed target_update_interval - target network was dead code
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
        # BUG-013: Global gradient clipping (was hardcoded 10.0, too permissive)
        global_grad_clip: float = 5.0,
        # TASK_042: ReDo - Recycling Dormant Neurons
        redo: bool = False,
        redo_interval: int = 10,
        redo_threshold: float = 0.01,
        # TASK_043: ATC contrastive learning
        atc_loss: bool = False,
        atc_weight: float = 0.1,
        atc_k_step: int = 5,
        atc_temperature: float = 0.1,
        atc_num_negatives: int = 32,
        # TASK_110: Cycle Balance (prevent H0 dominance)
        cycle_balance: bool = False,
        cycle_balance_weight: float = 0.1,
        cycle_balance_target_cv: float = 0.3,
        cycle_atc: bool = False,
        cycle_atc_weight: float = 0.05,
        cycle_atc_temperature: float = 0.1,
        # TASK_066: Physics configuration (extracted magic numbers)
        reward_scale: float = 100.0,
        # --reward-hybrid: mirror the value head's recipe for the reward head
        # (PopArt-normalized target + hybrid Huber/categorical loss). Default OFF.
        reward_hybrid: bool = False,
        # TASK_828 Stage 3: vector-action training path. Static Python bool → JIT
        # specializes (like reward_hybrid). Default OFF = canon scalar path,
        # bit-identical (every vector branch DCE'd by XLA when OFF).
        vector_action: bool = False,
        alpha_target: float = 0.05,
        beta_target: float = 0.03,
        sigma_target: float = 0.01,
        label_smoothing: float = 0.1,
        min_entropy: float = 3.0,
        # TASK_118: Alpha-relative reward normalization
        alpha_relative_reward: bool = False,
        alpha_reward_blend: float = 0.5,
        alpha_top_k: int = 4,
        # TASK_119: Optimal hindsight path
        amax_method: str = None,
        # TASK_119: Regret penalty (safe feedback from hindsight)
        amax_regret_weight: float = 0.0,  # 0 = disabled, 0.01-0.1 = auxiliary signal
        # TASK_119: Optimal action imprint (stabilization)
        amax_imprint_weight: float = 0.0,  # 0 = disabled, 0.01-0.02 = stabilization
        amax_imprint_temp: float = 1.0,    # Temperature for policy targets
        # TASK_125: Causal Expert Imprinting
        expert_bc_config: Optional[ExpertBCConfig] = None,
    ):
        self.model = model
        self.track_grad_norms = track_grad_norms

        # TASK_066: Physics configuration
        self.reward_scale = reward_scale
        self.alpha_target = alpha_target
        self.beta_target = beta_target
        self.sigma_target = sigma_target
        self.label_smoothing = label_smoothing
        self.min_entropy = min_entropy

        # TASK_118: Alpha-relative reward normalization
        self.alpha_relative_reward = alpha_relative_reward
        self.alpha_reward_blend = alpha_reward_blend
        self.alpha_top_k = alpha_top_k

        # TASK_119: Optimal hindsight path
        self.amax_method = amax_method
        self.amax_regret_weight = amax_regret_weight
        self.amax_imprint_weight = amax_imprint_weight
        self.amax_imprint_temp = amax_imprint_temp

        # TASK_125: Causal Expert Imprinting
        self.expert_bc_config = expert_bc_config or ExpertBCConfig()

        # TASK_042: ReDo
        self.redo = redo
        self.redo_interval = redo_interval
        self.redo_threshold = redo_threshold

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

        # TASK_110: Cycle Balance (prevent H0 dominance)
        self.cycle_balance = cycle_balance
        self.cycle_balance_weight = cycle_balance_weight
        self.cycle_balance_target_cv = cycle_balance_target_cv
        self.cycle_atc = cycle_atc
        self.cycle_atc_weight = cycle_atc_weight
        self.cycle_atc_temperature = cycle_atc_temperature

        # PopArt value normalization
        self.use_popart = use_popart
        if use_popart:
            from .popart import PopArtNormalizer
            self.popart = PopArtNormalizer(beta=popart_beta)
        else:
            self.popart = None

        # --reward-hybrid (default OFF): mirror the value PopArt normalizer above
        # (same beta) for the reward head. When OFF, reward_popart=None and the
        # reward path below stays bit-identical to canon.
        self.reward_hybrid = reward_hybrid
        if reward_hybrid:
            from .popart import PopArtNormalizer
            self.reward_popart = PopArtNormalizer(beta=popart_beta)
        else:
            self.reward_popart = None

        # TASK_828 Stage 3: vector-action training path (per-asset action VECTOR).
        # A=n_actions_per_asset (5) and n_assets read from the model; only consumed
        # on the ON path (Python ints → no effect on the OFF graph / no param leaves).
        self.vector_action = vector_action
        self.vec_n_actions_per_asset = getattr(self.model, 'vector_n_actions_per_asset', 5)
        self.vec_n_assets = getattr(self.model, 'n_assets', 64)
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps
        self.discount = discount
        # BUG-015: Removed self.target_update_interval - dead code

        # Loss weights
        self.value_loss_weight = value_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.entropy_coef = entropy_coef

        # Value loss configuration
        self.value_loss_type = value_loss_type
        self.value_huber_delta = value_huber_delta

        # Prediction loss configuration
        self.pred_loss_type = pred_loss_type
        self.pred_huber_delta = pred_huber_delta

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

        # BUG-013 FIX: Use configurable global_grad_clip instead of hardcoded 10.0
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(global_grad_clip),
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
            # BUG-015: Removed target_params - was dead code
            tx=self.optimizer,
            step_count=0
        )

        self.state = state
        return state

    def _scalar_to_categorical(self, scalar: jnp.ndarray, support_size: int = None, smoothing: float = None) -> jnp.ndarray:
        """
        Convert scalar value to categorical distribution with label smoothing.

        Args:
            scalar: (batch,) scalar values in range [-10, 10]
            support_size: Number of bins (defaults to model's support_size)
            smoothing: Label smoothing factor (defaults to self.label_smoothing)

        Returns:
            targets: (batch, support_size) smoothed probability distribution
        """
        # Use model's support_size if not provided
        if support_size is None:
            support_size = self.model.support_size
        # TASK_066: Use configured smoothing
        if smoothing is None:
            smoothing = self.label_smoothing

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

    # ------------------------------------------------------------------
    # TASK_828 Stage 3: vector-action (factored per-asset) policy helpers.
    # The factored 320-way policy is asset-major (idx = asset*A + action), so
    # (B, n_assets*A) reshapes to (B, n_assets, A). Cross-entropy / entropy are
    # computed per-asset (over A) and MEAN-reduced over assets → (B,) — MEAN not
    # SUM (D-5) so the policy-loss scale matches the portfolio value/reward heads
    # (a SUM over 64 assets would be a ~64× blowup). ln(5)≈1.61 is the per-asset
    # uniform-entropy ceiling.
    # ------------------------------------------------------------------
    def _vector_policy_ce(self, policy_logits: jnp.ndarray,
                          target_marginals: jnp.ndarray) -> jnp.ndarray:
        """Per-sample factored policy cross-entropy → (B,), MEAN over assets."""
        B = policy_logits.shape[0]
        n_assets, A = self.vec_n_assets, self.vec_n_actions_per_asset
        logits = policy_logits.reshape(B, n_assets, A)
        targets = target_marginals.reshape(B, n_assets, A)
        ce = optax.softmax_cross_entropy(logits, targets)  # (B, n_assets)
        return jnp.mean(ce, axis=-1)                        # (B,)

    def _vector_policy_entropy(self, policy_logits: jnp.ndarray) -> jnp.ndarray:
        """Per-sample factored policy entropy → (B,), MEAN over assets."""
        B = policy_logits.shape[0]
        n_assets, A = self.vec_n_assets, self.vec_n_actions_per_asset
        logits = policy_logits.reshape(B, n_assets, A)
        probs = jax.nn.softmax(logits, axis=-1)             # (B, n_assets, A)
        ent = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)  # (B, n_assets)
        return jnp.mean(ent, axis=-1)                       # (B,)

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
        + decoder_loss (reconstruction of initial state)

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

        # HIGH-011 FIX: Extract valid_mask for proper loss masking
        valid_mask = batch.get('valid', jnp.ones(batch_size, dtype=jnp.bool_))
        valid_mask_float = valid_mask.astype(jnp.float32)
        num_valid = jnp.maximum(jnp.sum(valid_mask_float), 1.0)  # Avoid div by zero

        # Get scheduled sampling probability
        teacher_prob = compute_scheduled_sampling_prob(step, config)

        # TASK_035: Get neural scale for physics warmup
        neural_scale = compute_neural_scale(step, config)

        # Get ground truth multi-step sequence
        # Shape: (batch, sequence_length, n_assets)
        teacher_sequence = batch.get('log_price_sequence')
        teacher_log_returns = batch.get('log_returns')
        teacher_moving_avg = batch.get('moving_avg')

        if teacher_sequence is None:
            # No auxiliary data available
            return jnp.array(0.0), {
                'loss_prediction_raw': 0.0,
                'loss_decoder': 0.0,
                'step_1_loss': 0.0,
                'step_k_loss': 0.0,
                'teacher_forcing_prob': teacher_prob,
                'neural_scale': neural_scale,
            }

        # TASK_106: Explicit Decoder Reconstruction Loss
        # Decode current latent state to market state
        market_state = self.model.apply(
            {'params': params},
            latent_states,
            method=self.model.decode_latent_to_market
        )

        # MSE against ground truth initial state (t=0)
        # TASK_134 FIX: Use NORMALIZED decoder loss to handle L3's massive price drift
        # Problem: L3 random walk causes log-prices to range from -4.6 to +9.2 (range ~14)
        #          vs L0-L2 which stays around 4.6 ± 0.3. Raw MSE scales quadratically.
        # Solution: Normalize by centering targets per-batch (compare RELATIVE positions)
        # HIGH-011 FIX: Apply valid_mask before averaging

        target_log_prices = teacher_sequence[:, 0, :]  # (batch, n_assets)
        # Normalize: center around batch mean to remove absolute scale
        target_mean = jnp.mean(target_log_prices, axis=-1, keepdims=True)  # (batch, 1)
        target_normalized = target_log_prices - target_mean
        decoded_normalized = market_state['log_prices'] - jnp.mean(market_state['log_prices'], axis=-1, keepdims=True)

        # Use Huber loss for robustness to outliers (L3 can have extreme values)
        huber_delta = 1.0
        price_errors = decoded_normalized - target_normalized
        abs_errors = jnp.abs(price_errors)
        quadratic = 0.5 * price_errors ** 2
        linear = huber_delta * (abs_errors - 0.5 * huber_delta)
        huber_loss_p = jnp.where(abs_errors <= huber_delta, quadratic, linear)
        per_sample_loss_p = jnp.mean(huber_loss_p, axis=-1)  # (batch,)
        loss_p = jnp.sum(per_sample_loss_p * valid_mask_float) / num_valid

        if teacher_log_returns is not None:
            # Log returns are already scale-invariant, use Huber for robustness
            return_errors = market_state['log_returns'] - teacher_log_returns
            abs_return_errors = jnp.abs(return_errors)
            quadratic_r = 0.5 * return_errors ** 2
            linear_r = huber_delta * (abs_return_errors - 0.5 * huber_delta)
            huber_loss_r = jnp.where(abs_return_errors <= huber_delta, quadratic_r, linear_r)
            per_sample_loss_r = jnp.mean(huber_loss_r, axis=-1)
            loss_r = jnp.sum(per_sample_loss_r * valid_mask_float) / num_valid
        else:
            loss_r = 0.0

        if teacher_moving_avg is not None:
            # Normalize moving avg same way as log_prices
            ma_mean = jnp.mean(teacher_moving_avg, axis=-1, keepdims=True)
            ma_normalized = teacher_moving_avg - ma_mean
            decoded_ma_normalized = market_state['moving_avg'] - jnp.mean(market_state['moving_avg'], axis=-1, keepdims=True)
            ma_errors = decoded_ma_normalized - ma_normalized
            abs_ma_errors = jnp.abs(ma_errors)
            quadratic_m = 0.5 * ma_errors ** 2
            linear_m = huber_delta * (abs_ma_errors - 0.5 * huber_delta)
            huber_loss_m = jnp.where(abs_ma_errors <= huber_delta, quadratic_m, linear_m)
            per_sample_loss_m = jnp.mean(huber_loss_m, axis=-1)
            loss_m = jnp.sum(per_sample_loss_m * valid_mask_float) / num_valid
        else:
            loss_m = 0.0

        loss_decoder = loss_p + loss_r + loss_m

        # Get features
        features = batch.get('features', jnp.zeros((batch_size, self.model.n_features)))

        # Get action sequence (first unroll_steps actions)
        # TASK_828: scalar-only consumer — the prediction head takes scalar action
        # indices. Vector mode passes action_sequence=None (batch has no 'actions'
        # key anyway); guard explicitly so a stray key can't feed a vector here.
        actions = None if self.vector_action else batch.get('actions')
        if actions is not None and actions.ndim == 2:
            # (batch, sequence_length) -> take first multi_step_k
            action_sequence = actions[:, :config.multi_step_k]
        else:
            action_sequence = None

        # Multi-step prediction - TASK_035: Pass neural_scale for physics warmup
        # TASK_106: Pass teacher state for initial state forcing
        predictions, _ = self.model.apply(
            {'params': params},
            latent_states,
            features,
            rng,
            action_sequence=action_sequence,
            num_steps=config.multi_step_k,
            teacher_sequence=teacher_sequence,
            teacher_log_returns=teacher_log_returns,
            teacher_moving_avg=teacher_moving_avg,
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

        # TASK_110 FIX: Use STEP-TO-STEP RETURNS (scale-invariant) instead of absolute log-prices
        # This fixes the scale mismatch between synthetic (~100) and historical (~30,000) prices
        #
        # Problem with baseline subtraction:
        #   pred_sequence is at decoder scale (~4.6 from synthetic)
        #   baseline is at historical scale (~10.3)
        #   pred - baseline = 4.6 - 10.3 = -5.7 ← STILL WRONG!
        #
        # Solution: Compare step-to-step CHANGES (actual log-returns)
        #   Both pred and target step changes are ~0.001-0.01 regardless of absolute scale
        #
        # For K predictions, we get K-1 step-to-step returns
        # Prepend a "step 0" return using teacher's initial price for full coverage
        teacher_initial = teacher_sequence[:, 0:1, :]  # (batch, 1, n_assets)

        # Predicted returns: change from previous step
        pred_step0 = pred_sequence[:, 0:1, :] - teacher_initial  # First pred vs teacher initial
        pred_step_changes = pred_sequence[:, 1:, :] - pred_sequence[:, :-1, :]  # Subsequent steps
        pred_returns = jnp.concatenate([pred_step0, pred_step_changes], axis=1)  # (batch, K, n_assets)

        # Target returns: change from previous step
        target_step0 = target_sequence[:, 0:1, :] - teacher_initial  # First target vs initial
        target_step_changes = target_sequence[:, 1:, :] - target_sequence[:, :-1, :]  # Subsequent
        target_returns = jnp.concatenate([target_step0, target_step_changes], axis=1)  # (batch, K, n_assets)

        # Per-step loss (MSE, Huber, or Student-t) on RETURNS, not absolute prices
        # TASK_116: Student-t option for fat-tailed financial returns
        student_t_metrics = {}
        if config.use_student_t:
            # Student-t loss: better handles outliers/fat tails in financial data
            # Uses adaptive sigma based on error statistics + fixed df
            errors = pred_returns - target_returns  # (batch, K, n_assets)

            # Estimate sigma from error magnitude (per-step, across batch and assets)
            # Use robust estimate: median absolute deviation scaled to std
            # BUG FIX: Floor sigma at 0.1 (not 1e-4) to prevent log(sigma) from going deeply negative
            # log(0.1) ≈ -2.3 is acceptable; log(1e-4) ≈ -9.2 makes loss negative!
            sigma_per_step = jnp.clip(
                jnp.std(errors, axis=(0, 2)),  # (K,) - std across batch and assets per step
                0.1,   # Floor raised to prevent negative log(sigma) dominating loss
                10.0   # Ceiling to prevent log(sigma) explosion during instability
            )
            sigma = sigma_per_step[None, :, None]  # (1, K, 1) for broadcasting

            # Fixed df from config (lower = fatter tails)
            # Bound df to [df_min, df_max] - use middle of range as default
            df = jnp.array((config.df_min + config.df_max) / 2)

            # Compute hybrid loss per element
            mse_loss = (errors ** 2)
            # Student-t NLL (simplified, dropping constants)
            # BUG FIX: Add 2.5 offset to log(sigma) to ensure non-negative contribution
            # This shifts log(0.1)=-2.3 to +0.2 and log(10)=2.3 to +4.8
            z = errors / sigma
            log_sigma_shifted = jnp.log(sigma) + 2.5  # Shift to be non-negative
            student_t_nll = log_sigma_shifted + 0.5 * (df + 1) * jnp.log1p(z ** 2 / df)

            # Weighted combination
            per_element_loss = (
                config.mse_weight * mse_loss +
                config.student_t_weight * student_t_nll
            )
            step_loss = jnp.mean(per_element_loss, axis=-1)  # (batch, K)

            # Track Student-t metrics
            student_t_metrics = {
                'sigma_mean': jnp.mean(sigma),
                'df': df,
                'student_t_nll_mean': jnp.mean(student_t_nll),
            }
        elif self.pred_loss_type == 'huber':
            # Huber loss: quadratic for small errors, linear for large
            # More robust to outliers from hard-to-predict levels (e.g., L6 triangle_arb)
            delta = self.pred_huber_delta
            abs_error = jnp.abs(pred_returns - target_returns)
            quadratic = 0.5 * abs_error ** 2
            linear = delta * (abs_error - 0.5 * delta)
            per_element_loss = jnp.where(abs_error <= delta, quadratic, linear)
            step_loss = jnp.mean(per_element_loss, axis=-1)  # (batch, K)
        else:
            # Standard MSE on returns (scale-invariant)
            step_loss = jnp.mean((pred_returns - target_returns) ** 2, axis=-1)  # (batch, K)

        # Weighted sum with decay
        step_weights = jnp.array([config.step_decay ** i for i in range(k)])
        weighted_loss = jnp.sum(step_loss * step_weights[None, :], axis=-1)  # (batch,)

        # Total prediction loss combines decoder reconstruction + trajectory prediction
        # HIGH-011 FIX: Apply valid_mask before averaging
        loss_prediction_raw = jnp.sum(weighted_loss * valid_mask_float) / num_valid
        loss_prediction = loss_prediction_raw + loss_decoder

        metrics = {
            'loss_prediction_raw': loss_prediction_raw,  # HIGH-011: Now properly masked
            'loss_decoder': loss_decoder,
            'step_1_loss': jnp.sum(step_loss[:, 0] * valid_mask_float) / num_valid if k > 0 else 0.0,
            'step_k_loss': jnp.sum(step_loss[:, -1] * valid_mask_float) / num_valid if k > 0 else 0.0,
            'teacher_forcing_prob': teacher_prob,
            'neural_scale': neural_scale,  # TASK_035: Track physics warmup progress
            **student_t_metrics,  # TASK_116: Student-t metrics when enabled
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
        # TASK_066: Use configurable targets
        if 'alpha' in physics_params:
            alpha = jax.nn.softplus(physics_params['alpha']) * 0.3  # Scale to [0, ~0.3]
            reg_loss = reg_loss + jnp.mean((alpha - self.alpha_target) ** 2) * 0.5

        # TASK_035 FIX: Two-sided TARGET regularization for mean-reversion
        # TASK_066: Use configurable targets
        if 'beta' in physics_params:
            beta = jax.nn.softplus(physics_params['beta']) * 0.2  # Scale to [0, ~0.2]
            reg_loss = reg_loss + jnp.mean((beta - self.beta_target) ** 2) * 0.5

        # Volatility: penalize extreme values - TASK_066: configurable target
        if 'sigma' in physics_params:
            sigma = jax.nn.softplus(physics_params['sigma']) * 0.1
            reg_loss = reg_loss + jnp.mean((sigma - self.sigma_target) ** 2)

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
        # TASK_828 Stage 3: vector mode reads the per-asset action VECTORS
        # (batch, unroll, n_assets) + factored policy marginals (batch, unroll,
        # n_assets, A) under distinct keys; the scalar OFF path reads the scalar
        # action index + flat policy probs exactly as canon (static-guarded).
        if self.vector_action:
            actions = batch['action_vecs']           # (batch, unroll_steps, n_assets)
            target_policies = batch['policy_marginals']  # (batch, unroll, n_assets, A)
        else:
            actions = batch['actions']  # (batch, unroll_steps)
            target_policies = batch['policy_probs']  # (batch, unroll_steps, 320)
        rewards = batch['rewards']  # (batch, unroll_steps)
        # --reward-hybrid (default OFF): host-side PopArt-normalized reward target,
        # stashed in batch['rewards_norm'] before the jit step (see training loop).
        # Static guard → key only present/read when the flag is ON; OFF is untouched.
        if self.reward_hybrid:
            rewards_norm = batch['rewards_norm']  # (batch, unroll_steps)
        target_values = batch['values']  # (batch, unroll_steps)

        # TASK_118: Alpha-relative reward normalization
        # BUG-033 FIX: Must apply symlog to max_alpha to match reward space
        # rewards are symlog(log_return * 100), so max_alpha must also be symlog'd
        if self.alpha_relative_reward:
            log_price_seq = batch.get('log_price_sequence')
            if log_price_seq is not None:
                # Compute returns from log prices: (batch, seq_len, n_assets)
                returns = log_price_seq[:, 1:, :] - log_price_seq[:, :-1, :]
                # Compute max alpha per batch item
                max_alpha = jax.vmap(
                    lambda r: compute_max_alpha_topk(r, self.alpha_top_k)
                )(returns)  # (batch,)
                # BUG-033 FIX: Apply symlog to max_alpha to match reward units
                # max_alpha is sum of raw log returns (e.g., 0.03 for 3%)
                # rewards are symlog(log_return * 100), so transform max_alpha the same way
                # symlog(x) = sign(x) * log(1 + |x|)
                max_alpha_scaled = max_alpha * 100.0  # Scale to percentage-like units
                max_alpha_symlog = jnp.sign(max_alpha_scaled) * jnp.log1p(jnp.abs(max_alpha_scaled))
                # Compute alpha ratio (sum of symlog rewards / symlog max_alpha)
                raw_reward_sum = jnp.sum(rewards, axis=1)  # (batch,) - already symlog'd
                alpha_ratio = raw_reward_sum / jnp.maximum(jnp.abs(max_alpha_symlog), 1e-6)  # (batch,)
                # Blend: normalize rewards relative to max achievable (in symlog space)
                blend = self.alpha_reward_blend
                # Simplified: just normalize rewards by max_alpha_symlog
                rewards = rewards * (1 - blend) + rewards / jnp.maximum(jnp.abs(max_alpha_symlog[:, None]), 1e-6) * blend

        # Validity mask for handling buffer wraparound (BUG_006 fix)
        valid_mask = batch.get('valid', jnp.ones(observations.shape[0], dtype=jnp.bool_))
        num_valid = jnp.maximum(jnp.sum(valid_mask), 1.0)  # Avoid div by zero
        valid_mask_float = valid_mask.astype(jnp.float32)

        batch_size, unroll_steps = actions.shape[:2]

        # Initial inference (with intermediates if cycle balance is enabled)
        need_intermediates = self.cycle_balance or self.cycle_atc
        initial_output = self.model.apply(
            {'params': params},
            observations[:, 0],  # First observation
            mode='initial_inference',
            return_intermediates=need_intermediates
        )

        # BUG_002 FIX: Compute initial losses with valid_mask
        value_loss, value_mse, value_cat = self._hybrid_value_loss(
            initial_output['value'],
            target_values[:, 0],
            valid_mask=valid_mask,
            num_valid=num_valid,
        )

        # BUG_002 FIX: Per-sample policy loss, masked before averaging
        # TASK_828: vector → factored per-asset CE (mean over assets); scalar canon.
        if self.vector_action:
            per_sample_policy = self._vector_policy_ce(
                initial_output['policy_logits'], target_policies[:, 0])
        else:
            per_sample_policy = optax.softmax_cross_entropy(
                initial_output['policy_logits'],
                target_policies[:, 0]
            )
        policy_loss = jnp.sum(per_sample_policy * valid_mask_float) / num_valid

        # Compute policy entropy for regularization (prevent collapse)
        if self.vector_action:
            per_sample_entropy = self._vector_policy_entropy(initial_output['policy_logits'])
        else:
            policy_probs = jax.nn.softmax(initial_output['policy_logits'], axis=-1)
            per_sample_entropy = -jnp.sum(policy_probs * jnp.log(policy_probs + 1e-10), axis=-1)
        policy_entropy = jnp.sum(per_sample_entropy * valid_mask_float) / num_valid

        reward_loss = jnp.array(0.0)  # No reward at root

        # Track MSE/categorical components separately for metrics
        total_value_mse = value_mse
        total_value_cat = value_cat

        # BUG-006 FIX: MuZero consistency loss - keep dynamics aligned with representation
        # L_consistency = ||h(s') - dynamics(h(s), a)||²
        # This prevents dynamics from drifting during unrolling
        consistency_loss = jnp.array(0.0)

        # Unroll dynamics for subsequent steps
        latent_state = initial_output['latent_state']

        for t in range(1, unroll_steps):
            # Recurrent inference
            # TASK_828: vector → action-conditional reward head via the vector
            # dynamics (recurrent_inference_vector, D2); actions[:, t-1] is the
            # per-asset action VECTOR (B, n_assets). Same output keys as scalar.
            if self.vector_action:
                recurrent_output = self.model.apply(
                    {'params': params},
                    latent_state,
                    actions[:, t - 1],
                    mode='recurrent_inference_vector'
                )
            else:
                recurrent_output = self.model.apply(
                    {'params': params},
                    latent_state,
                    actions[:, t - 1],
                    mode='recurrent_inference'
                )

            # Update latent state
            predicted_latent = recurrent_output['latent_state']

            # BUG-006 FIX: Consistency loss - encode actual observation and compare
            # True latent from representation(s_{t})
            true_latent_output = self.model.apply(
                {'params': params},
                observations[:, t],
                mode='initial_inference'
            )
            true_latent = jax.lax.stop_gradient(true_latent_output['latent_state'])

            # MSE between predicted and true latent (per-sample, then mask and average)
            per_sample_consistency = jnp.mean((predicted_latent - true_latent) ** 2, axis=-1)
            consistency_loss += jnp.sum(per_sample_consistency * valid_mask_float) / num_valid

            latent_state = predicted_latent

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
            # TASK_828: vector → factored per-asset CE (mean over assets); scalar canon.
            if self.vector_action:
                per_sample_step_policy = self._vector_policy_ce(
                    recurrent_output['policy_logits'], target_policies[:, t])
            else:
                per_sample_step_policy = optax.softmax_cross_entropy(
                    recurrent_output['policy_logits'],
                    target_policies[:, t]
                )
            policy_loss += jnp.sum(per_sample_step_policy * valid_mask_float) / num_valid

            # BUG_001 FIX: Reward target must match action index (t-1, not t)
            # BUG_002 FIX: Per-sample reward loss, masked
            # TASK_034 FIX: Scale rewards before categorical binning
            # Raw symlog rewards (~0.001) would collapse to 1-2 bins without scaling
            # TASK_066: Use configurable reward scale
            # --reward-hybrid (default OFF): give the reward head the value head's
            # recipe — PopArt-normalized (unit-variance) target + hybrid
            # Huber/categorical loss — so a near-zero noisy reward target produces a
            # rankable signal instead of collapsing to the marginal under pure CE.
            # When OFF, both branches below take the ORIGINAL path (×reward_scale
            # target + pure-CE loss), bit-identical to canon.
            if self.reward_hybrid:
                # Host-side PopArt-normalized target (running unit-variance); the
                # in-jit normalize froze stats, so read the pre-normalized value.
                scaled_rewards = rewards_norm[:, t - 1]
            else:
                scaled_rewards = rewards[:, t - 1] * self.reward_scale
            if self.reward_hybrid:
                pred_reward = self._logits_to_scalar(recurrent_output['reward'])
                per_sample_reward = (
                    0.8 * optax.huber_loss(
                        pred_reward, scaled_rewards, delta=self.value_huber_delta
                    )
                    + 0.2 * optax.softmax_cross_entropy(
                        recurrent_output['reward'],
                        self._scalar_to_categorical(scaled_rewards),
                    )
                )
            else:
                per_sample_reward = optax.softmax_cross_entropy(
                    recurrent_output['reward'],
                    self._scalar_to_categorical(scaled_rewards)
                )
            reward_loss += jnp.sum(per_sample_reward * valid_mask_float) / num_valid

            # Accumulate entropy (masked)
            # TASK_828: vector → per-asset entropy (mean over assets); scalar canon.
            if self.vector_action:
                per_sample_step_entropy = self._vector_policy_entropy(recurrent_output['policy_logits'])
            else:
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
        # BUG-006 FIX: Average consistency loss over unroll steps (excluding root)
        consistency_loss /= max(unroll_steps - 1, 1)

        # Adaptive entropy bonus - stronger push when entropy drops too low
        # Target entropy ~3.0 (near-uniform over 320 actions = ln(320) ≈ 5.77)
        # TASK_066: Use configurable min_entropy threshold
        # TASK_828: in vector mode the entropy is a MEAN-over-assets per-asset
        # entropy with ceiling ln(A)=ln(5)≈1.61 (vs the scalar ln(320)≈5.77), so a
        # 3.0 threshold would exceed the ceiling and push constantly. Rescale the
        # threshold by ln(5)/ln(320) to keep the SAME fraction-of-ceiling target.
        # Python float → OFF path uses self.min_entropy verbatim (canon-identical).
        if self.vector_action:
            import math as _math
            eff_min_entropy = self.min_entropy * (
                _math.log(self.vec_n_actions_per_asset) / _math.log(320.0))
        else:
            eff_min_entropy = self.min_entropy
        adaptive_bonus = jnp.where(
            policy_entropy < eff_min_entropy,
            entropy_coef * 5.0 * (eff_min_entropy - policy_entropy),  # 5x stronger push when too low
            0.0
        )

        # Base MuZero loss (entropy bonus is SUBTRACTED to maximize it)
        # BUG-006 FIX: Add consistency loss to keep dynamics aligned with representation
        consistency_weight = 0.5  # Balance with other losses (can be tuned)
        muzero_loss = (
            self.value_loss_weight * value_loss +
            self.policy_loss_weight * policy_loss +
            self.reward_loss_weight * reward_loss +
            consistency_weight * consistency_loss  # BUG-006: Prevent dynamics drift
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
            'consistency_loss': consistency_loss,  # BUG-006: Dynamics/representation alignment
            'policy_entropy': policy_entropy,
            'valid_samples': num_valid,  # Track buffer validity (BUG_006)
            'valid_ratio': num_valid / batch_size,  # Fraction of valid samples
        }

        # ================================================================
        # TASK_119: Optimal Hindsight Path Metrics (for DS24 Market Baseline)
        # ================================================================
        regret_loss = 0.0
        imprint_loss = 0.0
        if self.amax_method is not None:
            log_price_seq = batch.get('log_price_sequence')
            # Note: Can't use jnp.any() check here because it returns a traced value
            # that can't be used in Python if. Trust data flow - if amax is enabled
            # and log_price_seq exists, it has valid data.
            if log_price_seq is not None:
                from .path_solver import compute_optimal_path
                # Compute returns from log prices: (batch, seq_len, n_assets)
                returns_for_amax = log_price_seq[:, 1:, :] - log_price_seq[:, :-1, :]
                # Compute optimal path for first batch item (avoid vmap overhead)
                # TASK_126: Use transaction_cost=0.0 for theoretical max alpha
                # Default 10bps cost was killing oracle performance on minute data
                optimal_path = compute_optimal_path(
                    returns_for_amax[0], k=self.alpha_top_k, method=self.amax_method,
                    transaction_cost=0.0
                )

                # TASK_121: Scale optimal alpha to match agent reward units
                # Agent reward: symlog(log_return * 100) per step, summed
                # Optimal: raw log returns summed
                # Convert optimal to same scale: total_alpha * 100 gives percentage-like scale
                # Then apply symlog for each "step equivalent"
                # Approximate: scale by 100 to match agent's *100 scaling
                optimal_alpha_scaled = optimal_path.total_alpha * 100.0

                # Log metrics (use scaled version for comparison)
                agent_pnl = jnp.sum(rewards[0])
                metrics['optimal_alpha'] = optimal_alpha_scaled
                metrics['optimal_trades'] = optimal_path.trade_count
                metrics['optimal_gain'] = optimal_alpha_scaled  # TASK_119: Same as optimal_alpha for CSV
                # Protect against div-by-zero and negative optimal (edge case)
                alpha_capture = agent_pnl / jnp.maximum(jnp.abs(optimal_alpha_scaled), 1e-6)
                metrics['alpha_capture'] = alpha_capture

                # TASK_119: Regret penalty (safe feedback loop)
                # Regret = 1 - alpha_capture (normalized gap to optimal)
                # This penalizes large gaps without telling agent what action to take
                if self.amax_regret_weight > 0:
                    # Regret is (1 - alpha_capture), clamped to [0, 1]
                    # When alpha_capture = 1.0 (perfect), regret = 0
                    # When alpha_capture = 0.0 (no profit), regret = 1
                    regret = jnp.clip(1.0 - alpha_capture, 0.0, 1.0)
                    regret_loss = self.amax_regret_weight * regret
                    metrics['regret_loss'] = regret_loss
                    metrics['regret'] = regret

                # TASK_119: Optimal action imprint (stabilization)
                # Soft policy loss toward hindsight-optimal actions
                # TASK_828: scalar-only — targets/agent policy are over the flat 320
                # action space; skip in vector mode (default OFF for vector runs).
                if self.amax_imprint_weight > 0 and not self.vector_action:
                    from .path_solver import optimal_path_to_policy, compute_imprint_loss

                    # Convert optimal path to policy targets
                    # Use first batch item's returns for direction info
                    optimal_policy_targets = optimal_path_to_policy(
                        optimal_path.selections,
                        returns_for_amax[0],
                        n_actions_per_asset=5,
                        temperature=self.amax_imprint_temp,
                    )

                    # Get agent's policy for first timestep (initial_output computed earlier)
                    # policy_probs shape: (batch, n_actions)
                    # optimal_policy_targets shape: (T, n_actions) - use first T steps
                    agent_policy_t0 = policy_probs[0]  # First batch item, shape (n_actions,)
                    optimal_policy_t0 = optimal_policy_targets[0]  # First timestep

                    # Cross-entropy loss toward optimal
                    imprint_loss = self.amax_imprint_weight * compute_imprint_loss(
                        agent_policy_t0[None, :],  # (1, n_actions)
                        optimal_policy_t0[None, :],  # (1, n_actions)
                    )
                    metrics['imprint_loss'] = imprint_loss

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

        # ================================================================
        # TASK_110: Cycle Balance Loss (prevent H0 dominance)
        # ================================================================
        if (self.cycle_balance or self.cycle_atc) and 'trm_intermediates' in initial_output:
            intermediates = initial_output['trm_intermediates']

            # Cycle balance loss (penalize imbalanced utilization)
            if self.cycle_balance:
                cycle_bal_loss = compute_cycle_balance_loss(
                    intermediates,
                    h_cycles=self.model.H_cycles,
                    l_cycles=self.model.L_cycles,
                    target_cv=self.cycle_balance_target_cv,
                )
                total_loss = total_loss + self.cycle_balance_weight * cycle_bal_loss
                metrics['cycle_balance_loss'] = cycle_bal_loss
            else:
                metrics['cycle_balance_loss'] = 0.0

            # Cycle ATC loss (force H0/H3 diversity)
            if self.cycle_atc:
                cycle_atc_loss = compute_cycle_atc_loss(
                    intermediates,
                    temperature=self.cycle_atc_temperature,
                    early_cycle='h0',
                    late_cycle=f'h{self.model.H_cycles - 1}',  # Last H-cycle
                )
                total_loss = total_loss + self.cycle_atc_weight * cycle_atc_loss
                metrics['cycle_atc_loss'] = cycle_atc_loss
            else:
                metrics['cycle_atc_loss'] = 0.0
        else:
            metrics['cycle_balance_loss'] = 0.0
            metrics['cycle_atc_loss'] = 0.0

        # ================================================================
        # TASK_125: Causal Expert Imprinting BC Loss
        # ================================================================
        bc_loss_total = jnp.array(0.0)
        bc_config = self.expert_bc_config
        # TASK_828: expert BC imprints scalar 320-index optimal actions; skip in
        # vector mode (default OFF for vector runs — no expert_batch is emitted).
        if bc_config.enabled and 'expert_batch' in batch and not self.vector_action:
            expert_batch = batch['expert_batch']

            # Expert batch contains pre-encoded observations
            expert_obs = expert_batch['observations']  # (expert_batch_size, 27, n_assets, n_assets)
            expert_actions = expert_batch['optimal_actions']  # (expert_batch_size,)
            expert_confidence = expert_batch.get('confidence', jnp.ones(expert_obs.shape[0]))
            # BUG-002 FIX: Get PER importance sampling weights (defaults to 1.0 if not present)
            per_weights = expert_batch.get('weights', jnp.ones(expert_obs.shape[0]))

            # Run initial inference on expert observations to get policy and Q-values
            expert_output = self.model.apply(
                {'params': params},
                expert_obs,
                mode='initial_inference'
            )

            expert_policy_logits = expert_output['policy_logits']  # (expert_batch, n_actions)
            expert_value_logits = expert_output['value']  # (expert_batch, support_size)

            # Get Q-values from value head (use expected value as Q-proxy)
            # In MuZero, we don't have explicit Q(s,a) - use value as surrogate
            expert_q_values = self._logits_to_scalar(expert_value_logits)  # (expert_batch,)

            # For Q-filter, we need Q for each action - use value as baseline
            # This is a simplification; true Q-filter would need action-value network
            # Instead, use policy entropy as proxy for "policy confidence"
            expert_policy_probs = jax.nn.softmax(expert_policy_logits, axis=-1)
            policy_actions = jnp.argmax(expert_policy_logits, axis=-1)

            # Compute annealed BC weight
            bc_weight = compute_bc_weight(step, bc_config)

            if bc_config.use_large_margin:
                # Large margin loss: expert Q should exceed others by margin
                # Use policy logits as Q-proxy (higher logit = higher "Q")
                margin_loss = compute_large_margin_loss(
                    expert_policy_logits,
                    expert_actions,
                    margin=bc_config.large_margin
                )

                # BUG-001 FIX: Weight by confidence using MASKED MEAN (not simple mean)
                # BUG-002 FIX: Apply PER importance sampling weights for unbiased gradients
                # TASK_132B FIX: Hierarchical weighting - confidence primary, PER secondary (damped)
                # Zero-confidence samples should not dilute the gradient
                if bc_config.weight_by_confidence:
                    # Hierarchical: confidence dominates, PER has sqrt-damped effect
                    combined_weights = expert_confidence * jnp.sqrt(per_weights)
                    weight_sum = jnp.maximum(jnp.sum(combined_weights), 1e-8)
                    bc_loss_total = bc_weight * jnp.sum(margin_loss * combined_weights) / weight_sum
                    metrics['bc_margin_loss'] = jnp.sum(margin_loss * combined_weights) / weight_sum
                else:
                    # Just PER weights (no confidence weighting)
                    weight_sum = jnp.maximum(jnp.sum(per_weights), 1e-8)
                    bc_loss_total = bc_weight * jnp.sum(margin_loss * per_weights) / weight_sum
                    metrics['bc_margin_loss'] = jnp.sum(margin_loss * per_weights) / weight_sum
            else:
                # Standard cross-entropy BC loss
                bc_ce_loss = optax.softmax_cross_entropy_with_integer_labels(
                    expert_policy_logits, expert_actions
                )

                # BUG-001 FIX: Weight by confidence using MASKED MEAN (not simple mean)
                # BUG-002 FIX: Apply PER importance sampling weights for unbiased gradients
                # TASK_132B FIX: Hierarchical weighting - confidence primary, PER secondary (damped)
                # Zero-confidence samples should not dilute the gradient
                if bc_config.weight_by_confidence:
                    # Hierarchical: confidence dominates, PER has sqrt-damped effect
                    combined_weights = expert_confidence * jnp.sqrt(per_weights)
                    weight_sum = jnp.maximum(jnp.sum(combined_weights), 1e-8)
                    bc_loss_total = bc_weight * jnp.sum(bc_ce_loss * combined_weights) / weight_sum
                    metrics['bc_ce_loss'] = jnp.sum(bc_ce_loss * combined_weights) / weight_sum
                else:
                    # Just PER weights (no confidence weighting)
                    weight_sum = jnp.maximum(jnp.sum(per_weights), 1e-8)
                    bc_loss_total = bc_weight * jnp.sum(bc_ce_loss * per_weights) / weight_sum
                    metrics['bc_ce_loss'] = jnp.sum(bc_ce_loss * per_weights) / weight_sum

            # Action agreement: how often policy argmax matches optimal action
            # This is the key metric for imprinting success
            expert_agreement = (policy_actions == expert_actions).astype(jnp.float32)
            metrics['expert_action_agreement'] = jnp.mean(expert_agreement)

            # Add to total loss
            total_loss = total_loss + bc_loss_total

            # BC metrics for DS24
            metrics['bc_loss'] = bc_loss_total
            metrics['bc_weight'] = bc_weight
            metrics['expert_batch_size'] = expert_obs.shape[0]
            # TASK_132B: This is same as expert_action_agreement, kept for backwards compat
            # NOT to be confused with expert_storage_rate (demos/steps) in DS24 logger
            metrics['expert_acceptance_rate'] = jnp.mean(expert_agreement)
        else:
            # TASK_132B FIX: Use jnp.array for type consistency with main branch
            metrics['bc_loss'] = jnp.array(0.0)
            metrics['bc_weight'] = jnp.array(0.0)
            metrics['expert_batch_size'] = 0
            metrics['expert_acceptance_rate'] = jnp.array(0.0)
            metrics['expert_action_agreement'] = jnp.array(0.0)

        # TASK_119: Add regret penalty to total loss (safe hindsight feedback)
        # Use config flag check (not traced) instead of value check
        if self.amax_regret_weight > 0:
            total_loss = total_loss + regret_loss
        if self.amax_imprint_weight > 0:
            total_loss = total_loss + imprint_loss

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
        # BUG-015 FIX: Removed target network update logic - was dead code (never used)
        new_state = new_state.replace(step_count=new_step_count)

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
        rng: chex.PRNGKey,
        expert_buffer: Optional['UnifiedIndexBuffer'] = None,  # TASK_125
    ) -> Dict[str, list]:
        """
        Main training loop.

        Args:
            replay_buffer: Replay buffer with trajectories
            num_iterations: Number of training iterations
            rng: Random key
            expert_buffer: TASK_125 - Optional expert buffer for BC loss

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
            'step_1_loss': [],
            'step_k_loss': [],
            # Per-head gradient norms (DS24 Level 4)
            'grad_norm_value': [],
            'grad_norm_policy': [],
            'grad_norm_dynamics': [],
            # TASK_119: Optimal hindsight path metrics
            'regret_loss': [],
            'imprint_loss': [],
            'alpha_capture': [],
            'optimal_gain': [],
            # TASK_125: Causal Expert Imprinting metrics
            'bc_loss': [],
            'bc_weight': [],
            'expert_acceptance_rate': [],
            'expert_action_agreement': [],  # TASK_132a fix
        }

        for iteration in range(num_iterations):
            # Sample batch from replay buffer
            rng, sample_rng = jax.random.split(rng)
            batch = replay_buffer.sample_batch(sample_rng)

            # Debug: Check for NaNs/Infs in batch (only every 50 iterations to avoid GPU sync overhead)
            if iteration % 50 == 0:
                for k, v in batch.items():
                    if jnp.isnan(v).any() or jnp.isinf(v).any():
                        print(f"!!! CRITICAL BATCH FAILURE at iteration {iteration} !!!")
                        print(f"Key: {k}")
                        print(f"  Min: {jnp.min(v)}")
                        print(f"  Max: {jnp.max(v)}")
                        print(f"  Mean: {jnp.mean(v)}")
                        raise ValueError(f"NaNs/Infs detected in training batch key '{k}'!")

            # PopArt: normalize targets (TASK_021)
            # BUG-003/004 FIX: DISABLE weight correction - incompatible with categorical value head
            # PopArt weight correction assumes scalar outputs: W_new = W_old * (σ_old/σ_new)
            # But categorical heads output logits, and scaling logits doesn't scale E[value]!
            # The softmax nonlinearity breaks the linear correction formula.
            # We KEEP normalization (valid) but SKIP weight correction (broken).
            if self.use_popart and self.popart is not None:
                raw_values = batch['values'].flatten()

                # HIGH-010 FIX: Only use VALID samples for statistics update
                # Invalid samples (padding) would corrupt the running mean/variance
                valid_mask = batch.get('valid', jnp.ones(batch['values'].shape[0], dtype=jnp.bool_))
                valid_mask_flat = jnp.repeat(valid_mask, batch['values'].shape[1])  # Expand for flattened shape

                # Update statistics less frequently (amortizes sync cost)
                if iteration % 10 == 0:
                    # Only pass valid values to update statistics (NO weight correction)
                    valid_values = raw_values[valid_mask_flat]
                    if valid_values.shape[0] > 0:
                        # BUG-003/004 FIX: Only update stats, skip weight correction
                        from .popart import update_statistics
                        self.popart.state = update_statistics(
                            self.popart.state, valid_values, self.popart.beta
                        )
                    # Normalize ALL values (invalid ones masked in loss anyway)
                    norm_values = self.popart.normalize(raw_values)
                else:
                    # Just normalize using current stats (no param correction)
                    norm_values = self.popart.normalize(raw_values)

                batch = {**batch, 'values': norm_values.reshape(batch['values'].shape)}

            # --reward-hybrid (default OFF): mirror the value PopArt block above for
            # the reward target. In-jit normalize would freeze stats (state captured
            # at trace time), so — exactly like the value targets — we normalize on the
            # HOST here and stash the result in a parallel batch['rewards_norm'] key,
            # updating running stats from VALID raw rewards (normalize-only; same
            # weight-correction skip the value path uses). _loss_fn's --reward-hybrid
            # branch reads batch['rewards_norm']; batch['rewards'] is left untouched so
            # the OFF path stays bit-identical and the alpha-relative path is unaffected.
            if self.reward_hybrid:
                raw_rewards = batch['rewards'].flatten()

                # Only use VALID samples for statistics update (mirror HIGH-010 fix)
                reward_valid_mask = batch.get('valid', jnp.ones(batch['rewards'].shape[0], dtype=jnp.bool_))
                reward_valid_mask_flat = jnp.repeat(reward_valid_mask, batch['rewards'].shape[1])

                # Update statistics less frequently (amortizes sync cost)
                if iteration % 10 == 0:
                    valid_rewards = raw_rewards[reward_valid_mask_flat]
                    if valid_rewards.shape[0] > 0:
                        # Only update stats, skip weight correction (categorical head)
                        from .popart import update_statistics
                        self.reward_popart.state = update_statistics(
                            self.reward_popart.state, valid_rewards, self.reward_popart.beta
                        )
                    norm_rewards = self.reward_popart.normalize(raw_rewards)
                else:
                    norm_rewards = self.reward_popart.normalize(raw_rewards)

                batch = {**batch, 'rewards_norm': norm_rewards.reshape(batch['rewards'].shape)}

            # TASK_125: Sample from expert buffer and add to batch for BC loss
            # TASK_136d: Reduce BC frequency - encoding is expensive (~2.5s per call)
            # Running every iteration (50×/epoch) wastes 100s. Every 5th = 80% savings.
            bc_freq = self.expert_bc_config.bc_frequency
            if (expert_buffer is not None and
                self.expert_bc_config.enabled and
                iteration % bc_freq == 0):
                expert_batch = expert_buffer.sample_batch(
                    batch_size=self.expert_bc_config.expert_batch_size,
                    demo_ratio=self.expert_bc_config.demo_ratio,
                    training_step=iteration,  # TASK_132B: Pass actual training step for beta annealing
                )
                if expert_batch is not None:
                    batch['expert_batch'] = expert_batch

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

            # Record metrics (store JAX arrays as futures - no GPU sync)
            # Converting to float() every iteration causes CPU-GPU sync stall
            for key, value in metrics.items():
                if key in history:
                    history[key].append(value)  # Keep as JAX array

            # NaN check only every 50 iterations to avoid GPU sync overhead
            if (iteration + 1) % 50 == 0 and jnp.isnan(metrics['total_loss']):
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
                    loss_line += f" | Step1: {metrics.get('step_1_loss', 0.0):.6f}"
                    loss_line += f" | StepK: {metrics.get('step_k_loss', 0.0):.6f}"
                    loss_line += f" | PhysReg: {metrics.get('loss_physics_reg', 0.0):.6f}"

                print(loss_line)

        # Batch convert JAX arrays to floats (single sync point at end)
        # This avoids per-iteration GPU sync overhead from float() calls
        for key in history:
            if history[key]:
                # Check if first element is a JAX array (not already a Python float)
                first = history[key][0]
                if not isinstance(first, (int, float)):
                    # Block until all computations complete, then convert
                    history[key] = [float(v) for v in history[key]]

        return history

    def save_checkpoint(self, path: str, keep: int = 3, keep_every_n_steps: int = None,
                        overwrite: bool = False):
        """Save model checkpoint with additional metadata.

        Args:
            path: checkpoint directory (flax writes ``<path>/checkpoint_<step_count>``)
            keep: how many of the MOST RECENT flax checkpoints to keep. Default 3 = the
                historical, pre-TASK_876 behaviour.
            keep_every_n_steps: forwarded to flax; None = historical behaviour.
            overwrite: forwarded to flax. **Defaults to False** — the TASK_876 D32 ruling.
                Leave it False on any TRAINING path. Pass True only from a caller that
                re-writes a fixed, ephemeral output/smoke directory (see below).

        Returns the VERIFIED on-disk checkpoint path.

        ``overwrite`` defaults to False (TASK_876 D32 rescope; was unconditionally True
        pre-876). With ``overwrite=True`` flax's own pruning removes every checkpoint NEWER
        than the one just written (``_remove_invalid_ckpts``'s "remove newer checkpoints"
        branch, gated only on ``overwrite``, ignoring ``keep`` and consulting no never-delete
        set) — so a resume at a lower step (the exact collapse-recovery case this task exists
        for) silently mass-deleted every checkpoint above the resume point. That is why the
        DEFAULT is False and why it must stay False for training.

        It is a parameter rather than a hard-coded constant because ``overwrite=False`` is
        wrong for the non-training callers: they write a fixed, gitignored, ephemeral
        directory on purpose (smoke roundtrips, one-shot experiment outputs) where there is
        no newer-checkpoint mass-delete to protect against, and where re-running the same
        command must be idempotent. MEASURED: with a hard-coded False,
        ``eval_vector.py --smoke`` exits 0 on the first run and 1 on the second with an
        uncaught ``ValueError: Destination .../checkpoint_0 already exists``.

        Failure modes, deliberately distinguishable by the caller (TASK_876 fix-2 B1/M3):

        * ``ValueError`` (or whatever flax raises) — the flax write itself was REFUSED or
          failed; nothing new is on disk and existing checkpoints are untouched.
        * :class:`CheckpointVanishedError` — flax returned a path that is NOT on disk. On the
          Orbax backend (flax 0.10.7 here) a save behind the max step succeeds silently and
          is then deleted by flax's own keep-N cleanup inside the same call. There is no
          resume point for this trajectory; never report success.
        * :class:`CheckpointSidecarError` — the flax checkpoint IS on disk but
          ``trainer_metadata.json`` / ``popart_state.npz`` failed to write. The checkpoint is
          real and must still be scored; only its sidecars are missing/stale.

        Callers that save periodically must not let any of these crash the training loop; see
        ``_attempt_checkpoint_save`` in ``train_jax_fast.py``.
        """
        import os
        import json
        import numpy as np
        from flax.training import checkpoints

        # Save main state. flax returns the path it actually wrote
        # (``<path>/checkpoint_<step_count>``) — returned to the caller so a score can be
        # recorded against the REAL on-disk name instead of a guessed epoch (TASK_876 B3).
        written = checkpoints.save_checkpoint(
            ckpt_dir=path,
            target=self.state,
            step=self.state.step_count,
            keep=keep,
            overwrite=overwrite,
            keep_every_n_steps=keep_every_n_steps,
        )

        # TASK_876 fix-2 B1: a save counts as successful ONLY if its artifact is really on
        # disk. Checked BEFORE the sidecars, so a vanished checkpoint can never leave
        # `trainer_metadata.json` asserting a step whose checkpoint does not exist.
        written = _verify_checkpoint_on_disk(written, path, self.state.step_count)

        try:
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
        except Exception as e:
            raise CheckpointSidecarError(written, 'trainer_metadata.json', e) from e

        # Save PopArt state if enabled (critical for curriculum continuity)
        if self.use_popart and self.popart is not None:
            try:
                popart_path = os.path.join(path, 'popart_state.npz')
                np.savez(
                    popart_path,
                    mean=float(self.popart.state.mean),
                    var=float(self.popart.state.var),
                    count=int(self.popart.state.count),
                    mean_old=float(self.popart.state.mean_old),
                    std_old=float(self.popart.state.std_old),
                )
            except Exception as e:
                raise CheckpointSidecarError(written, 'popart_state.npz', e) from e

        return written

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

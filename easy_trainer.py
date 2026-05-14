"""
Easy Mode Trainer - Simplified Supervised Learning.

TASK_666: Expert Iteration Phase 1 - Learn from oracle demonstrations.

Key simplifications from trainer.py:
- No MuZero TD loss (pure supervised)
- No PopArt normalization (oracle returns are stable)
- No MCTS during training (inference only)
- No dynamics/reward prediction loss
- Just policy CE + value MSE
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, Tuple, Any, Optional
from functools import partial
import numpy as np

from easy_config import EasyConfig
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE, get_value_support
from mcts_trader.env.jax_encoder import jit_batch_tile_compressed


class EasyTrainer:
    """
    Supervised-only trainer for Easy Mode.

    Trains policy to match oracle actions and value to match oracle returns.
    No RL, no MCTS during training - pure imitation learning.
    """

    def __init__(self, config: EasyConfig, rng_key: jax.random.PRNGKey):
        self.config = config
        self.rng_key = rng_key

        # TASK_686: Resolve compute dtype
        dtype_str = getattr(config, 'compute_dtype', 'float32')
        compute_dtype = jnp.bfloat16 if dtype_str == 'bfloat16' else jnp.float32

        # Create model
        self.model = MuZeroNetworks(
            state_dim=config.state_dim,
            action_space_size=config.action_space_size,
            support_size=VALUE_SUPPORT_SIZE,
            H_cycles=config.H_cycles,
            L_cycles=config.L_cycles,
            use_prediction_head=False,  # No dynamics in Phase 1
            n_assets=config.n_assets,
            encoder_version=getattr(config, 'encoder_version', 2),
            hidden_dim=getattr(config, 'trm_hidden_dim', None),  # TASK_684
            embed_dim=getattr(config, 'embed_dim', 32),            # TASK_683
            encoder_num_heads=getattr(config, 'encoder_num_heads', 4),  # TASK_683
            dtype=compute_dtype,            # TASK_686: compute in bfloat16
            param_dtype=jnp.float32,        # TASK_686: store params in float32
            use_oracle_mask_head=getattr(config, 'use_oracle_mask_head', False),  # TASK_695
            oracle_mask_hidden_dim=getattr(config, 'oracle_mask_hidden_dim', 256),  # TASK_695
            use_per_asset_injection=getattr(config, 'use_per_asset_injection', False),
            per_asset_inject_xs=getattr(config, 'per_asset_inject_xs', False),
            per_asset_inject_cheat=getattr(config, 'per_asset_inject_cheat', False),
        )

        # Initialize parameters using 'initialize_parameters' mode
        # This initializes all params (encoder, prediction, dynamics) in one go
        rng_key, init_key = jax.random.split(rng_key)
        dummy_obs = jnp.zeros((1, config.n_channels, config.n_assets, config.n_assets))
        dummy_action = jnp.zeros((1,), dtype=jnp.int32)

        variables = self.model.init(
            init_key,
            observation=dummy_obs,
            action=dummy_action,
            mode='initialize_parameters'
        )
        self.params = variables['params']

        # Count parameters
        self.n_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"Model parameters: {self.n_params / 1e6:.2f}M")

        # Create optimizer with warmup
        total_steps = config.epochs * config.steps_per_epoch
        # Guard: decay_steps must exceed warmup_steps (short runs with --epochs 2 etc.)
        effective_warmup = min(config.warmup_steps, max(total_steps - 1, 0))
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=effective_warmup,
            decay_steps=max(total_steps, effective_warmup + 1),  # TASK_675: Exact
            end_value=config.learning_rate * 0.1,
        )

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            optax.adamw(schedule, weight_decay=config.weight_decay),
        )

        self.opt_state = self.optimizer.init(self.params)

        # Value support for converting categorical to scalar
        self.value_support = get_value_support()

        # JIT compile training step with static flag for latent cache mode
        self._train_step = jax.jit(
            self._train_step_impl,
            static_argnames=['use_latent_cache']
        )

    def _value_to_scalar(self, value_logits: jnp.ndarray) -> jnp.ndarray:
        """Convert categorical value logits to scalar via softmax expectation."""
        value_probs = jax.nn.softmax(value_logits, axis=-1)
        return jnp.sum(value_probs * self.value_support, axis=-1)

    def _scalar_to_target(self, returns: jnp.ndarray) -> jnp.ndarray:
        """Convert scalar returns to categorical target (soft labels)."""
        # Clip to support range [-10, 10]
        returns = jnp.clip(returns, -10.0, 10.0)

        # Find nearest bin
        bin_width = 20.0 / (VALUE_SUPPORT_SIZE - 1)
        bin_idx = ((returns + 10.0) / bin_width).astype(jnp.int32)
        bin_idx = jnp.clip(bin_idx, 0, VALUE_SUPPORT_SIZE - 1)

        # One-hot encoding
        return jax.nn.one_hot(bin_idx, VALUE_SUPPORT_SIZE)

    def _train_step_impl(
        self,
        params: Any,
        opt_state: Any,
        batch: Dict[str, jnp.ndarray],
        use_latent_cache: bool = False,
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """Single training step (JIT compiled)."""

        def loss_fn(params):
            # TASK_666 Optimization: Use prediction_only mode if latent states are cached
            if use_latent_cache:
                # Skip encoder entirely - use pre-computed latent states
                output = self.model.apply(
                    {'params': params},
                    latent_state=batch['latent_states'],
                    mode='prediction_only'
                )
            else:
                # TASK_152: Observations are now full temporal format (B, 27, 64, 64)
                # No tiling needed - pass directly to model
                output = self.model.apply(
                    {'params': params},
                    observation=batch['observations'],
                    mode='initial_inference'
                )
            policy_logits = output['policy_logits']
            value_logits = output['value']

            # TASK_145: Multi-discrete policy loss (per-asset independent softmax)
            # Reshape from (batch, 320) to (batch, 64, 5) for independent classification
            n_assets = self.config.n_assets  # 64
            n_actions_per_asset = 5

            policy_reshaped = policy_logits.reshape(-1, n_assets, n_actions_per_asset)
            targets_reshaped = batch['actions'].reshape(-1, n_assets, n_actions_per_asset)

            # Per-asset cross-entropy: softmax over 5 actions, independently per asset
            # This allows HOLD on asset 0 without competing with BUY on asset 7
            per_asset_ce = optax.softmax_cross_entropy(policy_reshaped, targets_reshaped)

            # TASK_149: Adaptive class balancing - trades and holds have equal gradient mass
            # TASK_675: Continuous trade weight for soft multipass targets
            # hold_prob is the soft probability of HOLD (index 1). 1.0 = pure hold, 0.0 = pure trade
            hold_prob = targets_reshaped[:, :, 1]  # (batch, 64) continuous [0, 1]
            trade_strength = 1.0 - hold_prob  # 0.0 = hold, 1.0 = trade

            # Weighted counts: a "40% BUY, 60% HOLD" target contributes 0.4 to trades, 0.6 to holds
            n_trades = jnp.maximum(jnp.sum(trade_strength), 1.0)
            n_holds = jnp.maximum(jnp.sum(hold_prob), 1.0)

            # Target weight for equal gradient mass between trades and holds
            # With k=1: ~63 holds / ~1 trade = 63.0. Cap at 64 (full balance).
            MAX_TRADE_WEIGHT = 64.0  # TASK_675: Was 10.0 — left 6.3× hold dominance
            target_trade_weight = jnp.minimum(n_holds / n_trades, MAX_TRADE_WEIGHT)

            # TASK_151: Warmup ramping - prevent gradient shock on fresh models
            # Progress 0.0 -> 1.0 over warmup_epochs, then stays at 1.0
            warmup_progress = batch.get('warmup_progress', jnp.array(1.0))
            # Interpolate: starts at 1.0 (unweighted), ramps to full adaptive weight
            adaptive_trade_weight = 1.0 + (target_trade_weight - 1.0) * warmup_progress

            # Continuous weighting: blend between 1.0 (hold) and adaptive (trade)
            loss_weights = 1.0 + trade_strength * (adaptive_trade_weight - 1.0)  # (batch, 64)

            # TASK_713: Return-weighted trades — high-value trades get more gradient
            # Teaches model WHICH assets matter, not just that a trade happened
            if 'per_asset_returns' in batch:
                abs_ret = jnp.abs(batch['per_asset_returns'])  # (batch, 64)
                mean_ret = jnp.maximum(jnp.mean(abs_ret), 1e-6)
                # return_boost: 1.0 for holds, 1.0 + scaled_return for trades (capped at 10x)
                return_boost = 1.0 + trade_strength * jnp.minimum(abs_ret / mean_ret, 10.0)
                loss_weights = loss_weights * return_boost

            policy_loss = jnp.sum(per_asset_ce * loss_weights) / jnp.sum(loss_weights)

            # Value loss: cross-entropy with categorical targets
            # Adjust scaling to prevent support clipping (MuZero support is [-10, 10])
            # Remaining PnL in Easy Mode is typically [0, 5], so 2.0 scaling is safe
            # TASK_670B: Use configurable value_scale (can tune to 5.0 if needed)
            scaled_returns = batch['returns'] * self.config.value_scale
            value_targets = self._scalar_to_target(scaled_returns)
            value_loss = optax.softmax_cross_entropy(value_logits, value_targets).mean()

            # TASK_695: Oracle mask prediction loss
            oracle_mask_loss = jnp.array(0.0)
            oracle_mask_accuracy = jnp.array(0.0)
            if self.config.use_oracle_mask_head and 'oracle_mask' in batch:
                oracle_mask_logits = output['oracle_mask']     # (B, 64)
                oracle_mask_targets = batch['oracle_mask']     # (B, 64) binary
                oracle_mask_loss = optax.sigmoid_binary_cross_entropy(
                    oracle_mask_logits, oracle_mask_targets
                ).mean()
                oracle_mask_pred = (jax.nn.sigmoid(oracle_mask_logits) > 0.5).astype(jnp.float32)
                oracle_mask_accuracy = jnp.mean(oracle_mask_pred == oracle_mask_targets)

            # Total loss
            total_loss = (
                self.config.policy_weight * policy_loss +
                self.config.value_weight * value_loss +
                self.config.oracle_mask_weight * oracle_mask_loss
            )

            # Metrics
            # TASK_143: Per-asset agreement (reuse reshaped tensors from loss)
            pred_action_type = jnp.argmax(policy_reshaped, axis=-1)    # (batch, 64)
            oracle_action_type = jnp.argmax(targets_reshaped, axis=-1)  # (batch, 64)
            per_asset_agreement = jnp.mean(pred_action_type == oracle_action_type)

            # TASK_147: Balanced Accuracy (Trade/Hold split)
            # Prevents "turtle" (all HOLD) from gaming the metric
            # HOLD = action type 1. Trades are 0, 2, 3, 4.
            oracle_is_trade = oracle_action_type != 1  # (batch, 64)
            oracle_is_hold = oracle_action_type == 1

            # Agreement on TRADE signals only
            trade_correct = (pred_action_type == oracle_action_type) & oracle_is_trade
            n_trade_actions = jnp.sum(oracle_is_trade)
            agree_trade = jnp.where(n_trade_actions > 0, jnp.sum(trade_correct) / n_trade_actions, 0.0)

            # Agreement on HOLD signals only
            hold_correct = (pred_action_type == oracle_action_type) & oracle_is_hold
            n_hold_actions = jnp.sum(oracle_is_hold)
            agree_hold = jnp.where(n_hold_actions > 0, jnp.sum(hold_correct) / n_hold_actions, 0.0)

            # Balanced accuracy: equal weight to trade and hold performance
            # Turtle (all HOLD) gets 50%, not 98.4%
            balanced_agreement = 0.5 * agree_trade + 0.5 * agree_hold

            # Legacy global argmax (kept for comparison, but this metric is broken)
            predicted_actions = jnp.argmax(policy_logits, axis=-1)
            oracle_labels = jnp.argmax(batch['actions'], axis=-1)
            global_agreement = jnp.mean(predicted_actions == oracle_labels)

            # Top-5 accuracy: check if ANY of top-5 is among "correct" oracle actions
            # (correct = probability > 0.1 in soft targets)
            top5_indices = jax.lax.top_k(policy_logits, 5)[1]
            is_correct_action = batch['actions'] > 0.1
            top5_correct = jnp.any(
                jnp.take_along_axis(is_correct_action, top5_indices, axis=-1),
                axis=-1
            )
            top5_accuracy = jnp.mean(top5_correct)

            # Value MSE (in original scale)
            value_pred = self._value_to_scalar(value_logits) / self.config.value_scale
            value_mse = jnp.mean((value_pred - batch['returns']) ** 2)

            # TASK_670B: Hold ratio tracking (per-asset, HOLD = action type 1)
            oracle_hold_ratio = jnp.mean(oracle_action_type == 1)
            pred_hold_ratio = jnp.mean(pred_action_type == 1)

            metrics = {
                'total_loss': total_loss,
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'oracle_agreement': balanced_agreement,   # TASK_147: Balanced accuracy (THE metric)
                'agree_trade': agree_trade,               # TASK_147: Trade-only accuracy
                'agree_hold': agree_hold,                 # TASK_147: Hold-only accuracy
                'per_asset_agreement': per_asset_agreement,  # Legacy (can be gamed by turtle)
                'global_agreement': global_agreement,     # Legacy (broken, for comparison)
                'top5_accuracy': top5_accuracy,
                'value_mse': value_mse,
                'oracle_hold_ratio': oracle_hold_ratio,
                'pred_hold_ratio': pred_hold_ratio,
                'adaptive_trade_weight': adaptive_trade_weight,  # TASK_149+151: Ramped per-batch weight
                'warmup_progress': warmup_progress,  # TASK_151: 0.0 -> 1.0 over warmup epochs
                'oracle_mask_loss': oracle_mask_loss,          # TASK_695
                'oracle_mask_accuracy': oracle_mask_accuracy,  # TASK_695
            }

            return total_loss, metrics

        # Compute gradients
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Update parameters
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Add grad norm to metrics
        grad_norm = optax.global_norm(grads)
        metrics['grad_norm'] = grad_norm

        return new_params, new_opt_state, metrics

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute single training step.

        Args:
            batch: Dict with JAX arrays (zero-copy from generator)
                   If 'latent_states' key present, uses prediction_only mode (faster)
        """
        # Detect if latent states are cached (skips encoder)
        use_latent_cache = 'latent_states' in batch

        # Run training step directly (batch already JAX arrays)
        self.params, self.opt_state, metrics = self._train_step(
            self.params, self.opt_state, batch, use_latent_cache=use_latent_cache
        )

        # Convert metrics to Python floats
        return {k: float(v) for k, v in metrics.items()}

    def evaluate_batch(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate single batch (accepts JAX arrays directly)."""
        # TASK_675: Use latent cache if available (matches training mode)
        if 'latent_states' in batch and batch['latent_states'] is not None:
            output = self.model.apply(
                {'params': self.params},
                latent_state=batch['latent_states'],
                mode='prediction_only'
            )
        else:
            output = self.model.apply(
                {'params': self.params},
                observation=batch['observations'],
                mode='initial_inference'
            )
        policy_logits = output['policy_logits']
        value_logits = output['value']

        # TASK_143: Per-asset agreement
        n_assets = self.config.n_assets  # 64
        pred_per_asset = policy_logits.reshape(-1, n_assets, 5)
        oracle_per_asset = batch['actions'].reshape(-1, n_assets, 5)
        pred_action_type = jnp.argmax(pred_per_asset, axis=-1)
        oracle_action_type = jnp.argmax(oracle_per_asset, axis=-1)
        per_asset_agreement = float(jnp.mean(pred_action_type == oracle_action_type))

        # TASK_147: Balanced Accuracy (Trade/Hold split) - THE metric
        oracle_is_trade = oracle_action_type != 1  # (batch, 64)
        oracle_is_hold = oracle_action_type == 1

        trade_correct = (pred_action_type == oracle_action_type) & oracle_is_trade
        n_trades = jnp.sum(oracle_is_trade)
        agree_trade = float(jnp.where(n_trades > 0, jnp.sum(trade_correct) / n_trades, 0.0))

        hold_correct = (pred_action_type == oracle_action_type) & oracle_is_hold
        n_holds = jnp.sum(oracle_is_hold)
        agree_hold = float(jnp.where(n_holds > 0, jnp.sum(hold_correct) / n_holds, 0.0))

        balanced_agreement = 0.5 * agree_trade + 0.5 * agree_hold

        # Legacy global argmax (kept for comparison)
        predicted_actions = jnp.argmax(policy_logits, axis=-1)
        oracle_labels = jnp.argmax(batch['actions'], axis=-1)
        global_agreement = float(jnp.mean(predicted_actions == oracle_labels))

        top5_indices = jax.lax.top_k(policy_logits, 5)[1]
        is_correct_action = batch['actions'] > 0.1
        top5_correct = jnp.any(
            jnp.take_along_axis(is_correct_action, top5_indices, axis=-1),
            axis=-1
        )
        top5_accuracy = float(jnp.mean(top5_correct))

        scaled_returns = batch['returns'] * self.config.value_scale
        value_targets = self._scalar_to_target(scaled_returns)
        value_loss = float(optax.softmax_cross_entropy(value_logits, value_targets).mean())

        value_pred = self._value_to_scalar(value_logits) / self.config.value_scale
        value_mse = float(jnp.mean((value_pred - batch['returns']) ** 2))

        # TASK_675: Hold ratio from per-asset argmax (matches training method)
        oracle_hold_ratio = float(jnp.mean(oracle_action_type == 1))
        pred_hold_ratio = float(jnp.mean(pred_action_type == 1))

        return {
            'oracle_agreement': balanced_agreement,   # TASK_147: Balanced accuracy (THE metric)
            'agree_trade': agree_trade,               # TASK_147: Trade-only accuracy
            'agree_hold': agree_hold,                 # TASK_147: Hold-only accuracy
            'per_asset_agreement': per_asset_agreement,  # Legacy (can be gamed)
            'global_agreement': global_agreement,     # Legacy (broken, for comparison)
            'top5_accuracy': top5_accuracy,
            'value_loss': value_loss,
            'value_mse': value_mse,
            'oracle_hold_ratio': oracle_hold_ratio,
            'pred_hold_ratio': pred_hold_ratio,
        }

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate on validation set (legacy interface for dataloaders)."""
        all_metrics = []

        for batch in dataloader:
            jax_batch = {k: jnp.array(v) for k, v in batch.items()}
            metrics = self.evaluate_batch(jax_batch)
            all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        import pickle
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'params': self.params,
            'opt_state': self.opt_state,
            'epoch': epoch,
            'metrics': metrics,
            'config': self.config.__dict__,
            'n_params': self.n_params,
        }

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        import pickle

        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.params = checkpoint['params']

        # Check if this is a merged checkpoint with separate optimizer components
        if 'merged_opt_components' in checkpoint:
            print(f"Rebuilding optimizer state from merged components...")
            self.rebuild_optimizer_state(
                checkpoint['merged_opt_components'].get('nu'),
                checkpoint['merged_opt_components'].get('mu'),
            )
        else:
            self.opt_state = checkpoint['opt_state']

        print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint

    def rebuild_optimizer_state(
        self,
        merged_nu: Any = None,
        merged_mu: Any = None,
    ):
        """
        Rebuild optimizer state from merged components (TASK_667, TASK_674).

        After a merge: zero mu (fresh gradient directions), keep nu (Fisher),
        preserve both adam and schedule counts (no warmup restart).

        Args:
            merged_nu: Merged second moment (Fisher approximation)
            merged_mu: Merged first moment (usually zeros)
        """
        import optax
        from optax._src.base import EmptyState

        # Get current opt_state structure as template
        template = self.opt_state

        # The structure is: (EmptyState, (ScaleByAdamState, EmptyState, ScaleByScheduleState))
        # We need to rebuild [1][0] (the Adam state) with merged components

        try:
            # Extract template structure (verify before accessing)
            # TASK_675 #14: Fail loud if optax.chain order ever changes
            assert hasattr(template[1][0], 'mu') and hasattr(template[1][0], 'nu'), \
                f"Expected ScaleByAdamState at opt_state[1][0], got {type(template[1][0])}"
            assert hasattr(template[1][2], 'count') and not hasattr(template[1][2], 'mu'), \
                f"Expected ScaleByScheduleState at opt_state[1][2], got {type(template[1][2])}"

            clip_state = template[0]  # EmptyState from clip
            adam_state = template[1][0]  # ScaleByAdamState
            scale_empty = template[1][1]  # EmptyState from adamw inner
            schedule_state = template[1][2]  # ScaleByScheduleState

            # Use merged components or keep existing
            new_nu = merged_nu if merged_nu is not None else adam_state.nu
            new_mu = merged_mu if merged_mu is not None else adam_state.mu

            # TASK_674: Preserve counts across merges to avoid warmup restart.
            # Zero mu for fresh gradient directions, keep nu (Fisher) and both counts.
            new_adam_state = optax.ScaleByAdamState(
                count=adam_state.count,        # Keep count (no bias correction reset)
                mu=new_mu,
                nu=new_nu,
            )

            # Keep schedule position (no LR warmup restart)
            new_schedule_state = optax.ScaleByScheduleState(
                count=schedule_state.count,
            )

            # Rebuild full opt_state
            self.opt_state = (
                clip_state,
                (new_adam_state, scale_empty, new_schedule_state)
            )

            print(f"  Optimizer state rebuilt (counts preserved: adam={int(adam_state.count)}, schedule={int(schedule_state.count)})")

        except Exception as e:
            print(f"  Warning: Could not rebuild optimizer state ({e}), reinitializing...")
            self.opt_state = self.optimizer.init(self.params)

    def predict(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on observations.

        Args:
            observations: (B, 27, 64) compressed or (B, 27, 64, 64) full

        Returns:
            actions: (B,) predicted actions
            values: (B,) predicted values
        """
        obs_jax = jnp.array(observations)

        # TASK_152: Expect full temporal format (B, 27, 64, 64)
        # For backward compatibility, still handle 3D input by broadcasting
        if obs_jax.ndim == 3:
            # Legacy compressed format - tile to full (deprecated)
            obs_jax = jit_batch_tile_compressed(obs_jax, n_assets=self.config.n_assets)

        output = self.model.apply(
            {'params': self.params},
            observation=obs_jax,
            mode='initial_inference'
        )
        policy_logits = output['policy_logits']
        value_logits = output['value']

        actions = np.array(jnp.argmax(policy_logits, axis=-1))
        values = np.array(self._value_to_scalar(value_logits) / self.config.value_scale)

        return actions, values


def create_trainer(config: EasyConfig, seed: int = 42) -> EasyTrainer:
    """Create trainer with config."""
    rng_key = jax.random.PRNGKey(seed)
    return EasyTrainer(config, rng_key)

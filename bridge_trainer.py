"""
TASK_780 — Bridge Trainer (Oracle-as-MCTS Distillation, Phase A imprint)

This module is the **parallel** Phase-1 trainer that replaces Foxglove's flat
soft-one-hot policy + scalar remaining-PnL targets with MCTS-shaped targets
(visit distributions, per-step policies, multi-step latent-dynamics consistency
loss). It runs alongside `easy_trainer.py`; the canon path is selected via
`easy_config.EasyConfig.use_oracle_mcts_distill`.

**Phase 2 status (this dispatch): TRAINING-STEP MATH IMPLEMENTED.**

The loss (TASK_697 §A.2) is computed over a K-step unroll:

    total_loss = sum_{k=0..K-1}( policy_w * CE(policy_logits_k, visit_dist_k) )
               + value_w * CE(value_logits_root, root_value_categorical)
               + consistency_w * sum_{k=0..K-1}( ‖pred_latent_{k+1} - target_latent_{k+1}‖² )
               + reward_w * sum_{k=0..K-1}( CE(pred_reward_k, reward_target_k) )

Where:
- `policy_logits_k` from `model.recurrent_inference(latent_k, action_k)` at unroll step k
  (k=0 uses `initial_inference(obs_root).policy_logits`)
- `visit_dist_k` is the per-step visit distribution target (per-asset 5-way softmax)
- `target_latent_{k+1}` = `stop_gradient(model.initial_inference(obs_{k+1}).latent_state)`
- `root_value_categorical` = `scalar_to_categorical(value_target * value_scale)`

Q-value loss is gated OFF by default (`oracle_distill_q_value_weight = 0.0`)
since the simple multipass-oracle Q-values reduce to `value_target * visit_indicator`
(see TASK_780_IMPLEMENTATION_PLAN.md §"Surprises" #3).

References
----------
- `tasks/TASK_780_ORACLE_AS_MCTS_DISTILLATION.md`
- `tasks/TASK_780_IMPLEMENTATION_PLAN.md` (Phase 2 spec)
- `tasks/TASK_697_ORACLE_AS_MCTS_DISTILLATION.md` §A.1, §A.2
- `runs/780_oracle_as_mcts/AUTHORIZATION_2026-05-26.md` §"Build phase outline"
- `easy_trainer.py:29-573` — canon Phase-1 trainer (target reference for value CE).
- `train_phase2.py:96-100` — load contract (this trainer's checkpoints must satisfy).
- `train_phase2.py:311-370` — Stage-A dynamics warmup (closest analog for consistency).
- `train_phase2.py:1235-1700` — Stage-B joint training (closest analog for multi-loss).
- `mcts_trader/jax_agent/oracle_distillation.py` — target-generation helpers consumed here.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from easy_config import EasyConfig
from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE, get_value_support
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks


# Bin width / range mirrors train_phase2.scalar_to_categorical so this trainer's
# value targets land in the same 401-bin support the rest of the stack uses.
_VALUE_SUPPORT_MIN = -10.0
_VALUE_SUPPORT_MAX = 10.0


def _scalar_to_categorical(values: jnp.ndarray) -> jnp.ndarray:
    """Convert scalar values to one-hot over the 401-bin value support.

    Mirrors `train_phase2.scalar_to_categorical` (`:78-87`) so bridge value
    targets share the same 401-bin layout as Phase-2 Stage-B.

    Parameters
    ----------
    values : jnp.ndarray
        Any shape. Clipped to [-10, +10] then binned.

    Returns
    -------
    jnp.ndarray of shape (*values.shape, 401)
    """
    values = jnp.clip(values, _VALUE_SUPPORT_MIN, _VALUE_SUPPORT_MAX)
    bin_width = (_VALUE_SUPPORT_MAX - _VALUE_SUPPORT_MIN) / (VALUE_SUPPORT_SIZE - 1)
    bin_idx = ((values - _VALUE_SUPPORT_MIN) / bin_width).astype(jnp.int32)
    bin_idx = jnp.clip(bin_idx, 0, VALUE_SUPPORT_SIZE - 1)
    return jax.nn.one_hot(bin_idx, VALUE_SUPPORT_SIZE)


class BridgeTrainer:
    """Oracle-as-MCTS distillation trainer (Phase A imprint).

    Constructs `MuZeroNetworks(use_prediction_head=True, ...)` so the trained
    parameter tree carries the dynamics + decode-latent-to-market subtree from
    epoch 0. The resulting checkpoint is loadable under
    `train_phase2.load_phase1_checkpoint` with the dynamics block ALREADY warm —
    Stage-A's consistency-only warmup becomes either a short fine-tune or is
    skipped (operator decision at smoke time per AUTHORIZATION §"Build phase
    outline").

    Bridge buffer contract (consumed by `train_step`)
    -------------------------------------------------
    A bridge batch is a dict with the following keys:

      observations         (B, T, C, W, N)   T = unroll_steps + 1
      actions              (B, K)            K = unroll_steps; joint int over [0, 320)
      rewards              (B, K)            per-step scalar PnL change
      visit_distributions  (B, T, A)         A = n_assets * 5; per-step visit dist
      values               (B, T)            per-step remaining PnL (root at index 0)
      root_value           (B,)              episode total PnL (clean_pnl[-1])
      dones                (B, K) bool       True past end-of-episode (mask)

    Index 0 along the T-axis is the root step; indices 1..K are unroll targets.
    """

    def __init__(self, config: EasyConfig, rng_key: jax.random.PRNGKey):
        self.config = config
        self.rng_key = rng_key

        # TASK_686: Resolve compute dtype
        dtype_str = getattr(config, 'compute_dtype', 'float32')
        compute_dtype = jnp.bfloat16 if dtype_str == 'bfloat16' else jnp.float32

        # ----------------------------------------------------------------
        # Model: same surface as EasyTrainer, but use_prediction_head=True so
        # the dynamics + decode-latent-to-market subtree is present from
        # epoch 0. This is the load-bearing change vs `easy_trainer.py:52`
        # (`use_prediction_head=False`).
        # ----------------------------------------------------------------
        self.model = MuZeroNetworks(
            state_dim=config.state_dim,
            action_space_size=config.action_space_size,
            support_size=VALUE_SUPPORT_SIZE,
            H_cycles=config.H_cycles,
            L_cycles=config.L_cycles,
            use_prediction_head=True,            # TASK_780: dynamics is trained here
            n_assets=config.n_assets,
            encoder_version=getattr(config, 'encoder_version', 2),
            hidden_dim=getattr(config, 'trm_hidden_dim', None),
            embed_dim=getattr(config, 'embed_dim', 32),
            encoder_num_heads=getattr(config, 'encoder_num_heads', 4),
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            use_oracle_mask_head=getattr(config, 'use_oracle_mask_head', False),
            oracle_mask_hidden_dim=getattr(config, 'oracle_mask_hidden_dim', 256),
            use_per_asset_injection=getattr(config, 'use_per_asset_injection', False),
            per_asset_inject_xs=getattr(config, 'per_asset_inject_xs', False),
            per_asset_inject_cheat=getattr(config, 'per_asset_inject_cheat', False),
        )

        # ----------------------------------------------------------------
        # Parameter initialization. Same dummy-obs pattern as easy_trainer:73,
        # then use the model.init 'initialize_parameters' mode so dynamics +
        # prediction subtrees populate alongside the encoder.
        # ----------------------------------------------------------------
        rng_key, init_key = jax.random.split(rng_key)
        dummy_obs = jnp.zeros(
            (1, config.n_channels, config.n_assets, config.n_assets)
        )
        dummy_action = jnp.zeros((1,), dtype=jnp.int32)

        variables = self.model.init(
            init_key,
            observation=dummy_obs,
            action=dummy_action,
            mode='initialize_parameters',
        )
        self.params = variables['params']
        self.n_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))

        # ----------------------------------------------------------------
        # Optimizer (mirrors easy_trainer.py:86-102).
        # ----------------------------------------------------------------
        total_steps = config.epochs * config.steps_per_epoch
        effective_warmup = min(config.warmup_steps, max(total_steps - 1, 0))
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=effective_warmup,
            decay_steps=max(total_steps, effective_warmup + 1),
            end_value=config.learning_rate * 0.1,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            optax.adamw(schedule, weight_decay=config.weight_decay),
        )
        self.opt_state = self.optimizer.init(self.params)

        # Value support for hybrid value loss (cached so the JIT step doesn't
        # re-derive it).
        self.value_support = get_value_support()

        # Cache static config values used inside the JIT step (Python-level
        # constants — captured in HLO at trace time).
        self._n_assets = int(config.n_assets)
        self._n_actions_per_asset = 5  # 5 actions per asset by TASK_780 convention
        self._value_scale = float(getattr(config, 'value_scale', 2.0))
        self._policy_weight = float(getattr(config, 'policy_weight', 1.0))
        self._value_weight = float(getattr(config, 'value_weight', 0.5))
        self._consistency_weight = float(
            getattr(config, 'oracle_distill_consistency_weight', 1.0)
        )
        self._reward_weight = float(
            getattr(config, 'oracle_distill_reward_weight', 1.0)
        )
        self._unroll_steps = int(
            getattr(config, 'oracle_distill_unroll_steps', 5)
        )
        # TASK_786: chunk size for the B*T initial_inference encoder forward.
        # 0 = legacy single-shot. See `_encode_obs_chunked`.
        self._encode_chunk = int(
            getattr(config, 'oracle_distill_encode_chunk', 384)
        )

        # JIT registration. `unroll_steps` is static so the loop body is
        # traced once per K value; `n_assets` is also static for the
        # reshape op.
        self._train_step = jax.jit(
            self._train_step_impl,
            static_argnames=['unroll_steps'],
        )
        self._eval_step = jax.jit(
            self._evaluate_batch_impl,
            static_argnames=['unroll_steps'],
        )

    # ----------------------------------------------------------------
    # TASK_786: memory-bounded encoder forward.
    # ----------------------------------------------------------------
    def _encode_obs_chunked(self, params, obs_flat):
        """Batched `initial_inference` over (BT, C, W, N), peak-bounded.

        The single-shot encode of all `B*(K+1)` windows is the 11.6 GiB OOM
        site (TASK_786). We chunk the leading axis into `_encode_chunk`-row
        groups and `lax.map` over them, remat-ing each chunk's forward so the
        backward recomputes rather than retains. Rows are independent through
        the encoder (cross-asset attention is within-window, never cross-batch),
        so this is numerically identical to the single-shot path.

        Returns `(latent_all, policy_logits_all, value_all)`, each `(BT, *)`.
        """
        BT = obs_flat.shape[0]
        chunk = self._encode_chunk

        def _encode(p, obs):
            out = self.model.apply(
                {'params': p}, observation=obs, mode='initial_inference'
            )
            return out['latent_state'], out['policy_logits'], out['value']

        # Legacy single-shot: chunking off, or chunk covers the whole batch.
        if chunk <= 0 or chunk >= BT:
            return _encode(params, obs_flat)

        # Pad BT up to a multiple of chunk (static shapes under jit). Padded
        # rows are sliced off before any loss term → zero gradient.
        n_chunks = (BT + chunk - 1) // chunk
        pad = n_chunks * chunk - BT
        if pad:
            obs_flat = jnp.pad(
                obs_flat, ((0, pad), (0, 0), (0, 0), (0, 0))
            )
        obs_chunked = obs_flat.reshape(n_chunks, chunk, *obs_flat.shape[1:])

        encode_remat = jax.checkpoint(_encode)
        lat, pol, val = jax.lax.map(
            lambda oc: encode_remat(params, oc), obs_chunked
        )
        # (n_chunks, chunk, *) → (n_chunks*chunk, *) → drop padding.
        lat = lat.reshape(n_chunks * chunk, -1)[:BT]
        pol = pol.reshape(n_chunks * chunk, -1)[:BT]
        val = val.reshape(n_chunks * chunk, -1)[:BT]
        return lat, pol, val

    # ----------------------------------------------------------------
    # Training step — MuZero-shaped loss over a K-step unroll.
    # ----------------------------------------------------------------
    def _train_step_impl(
        self,
        params: Any,
        opt_state: Any,
        batch: Dict[str, jnp.ndarray],
        unroll_steps: int = 5,
    ) -> Tuple[Any, Any, Dict[str, jnp.ndarray]]:
        """Single training step (JIT-compiled).

        Implements the TASK_697 §A.2 MuZero-shaped loss:
            total_loss = policy_w * CE(policy_logits, visit_distribution)        (per unroll step)
                       + value_w * CE(value_logits_root, root_value_categorical)
                       + reward_w * CE(reward_logits, oracle_step_reward)        (per unroll step)
                       + consistency_w * ‖pred_latent_{k+1} - target_latent_{k+1}‖²  (per unroll step)

        Targets are derived from `oracle_to_mcts_targets` (see
        `mcts_trader/jax_agent/oracle_distillation.py`) and assembled into the
        bridge batch by `OnTheFlyGenerator._generate_batch_episodes_bridge`.

        Buffer contract: see class docstring.
        """
        n_assets = self._n_assets
        n_actions_per_asset = self._n_actions_per_asset
        policy_w = self._policy_weight
        value_w = self._value_weight
        consistency_w = self._consistency_weight
        reward_w = self._reward_weight
        value_scale = self._value_scale

        observations = batch['observations']      # (B, K+1, C, W, N)
        actions = batch['actions']                # (B, K) int
        rewards = batch['rewards']                # (B, K)
        visit_distributions = batch['visit_distributions']  # (B, K+1, A) — A = n_assets*5
        root_value = batch['root_value']          # (B,)
        dones = batch['dones']                    # (B, K) bool

        B = observations.shape[0]
        T = observations.shape[1]  # = unroll_steps + 1
        C = observations.shape[2]
        W = observations.shape[3]
        N = observations.shape[4]

        def loss_fn(params):
            # ----------------------------------------------------------
            # 1. Encode every obs in the unroll trajectory through the
            #    representation head. Stop-gradient on targets latents (k>=1)
            #    so consistency loss only updates the dynamics+encoder path
            #    of the prediction chain, not the target encoder.
            # ----------------------------------------------------------
            # Flatten (B, T, C, W, N) → (B*T, C, W, N) for a single batched
            # initial_inference pass.
            obs_flat = observations.reshape(B * T, C, W, N)
            # TASK_786: chunked + remat encode bounds the peak HBM here.
            latent_all, policy_logits_all, value_all = self._encode_obs_chunked(
                params, obs_flat
            )                                                   # each (B*T, *)

            S = latent_all.shape[-1]
            latent_all = latent_all.reshape(B, T, S)
            policy_logits_all = policy_logits_all.reshape(B, T, -1)
            value_all = value_all.reshape(B, T, VALUE_SUPPORT_SIZE)

            # Root logits / latent (k = 0).
            value_logits_root = value_all[:, 0, :]       # (B, support)
            latent_root = latent_all[:, 0, :]            # (B, S)
            # Target latents for unroll consistency: indices 1..K.
            # MuZero convention: target latent is computed from a fresh
            # `initial_inference(obs_{k+1})` and detached.
            target_latents = jax.lax.stop_gradient(latent_all[:, 1:, :])  # (B, K, S)

            # ----------------------------------------------------------
            # 2. Unroll dynamics: at each k=0..K-1 do
            #    (latent_k, action_k) -> recurrent_inference -> (latent_{k+1}_pred,
            #     policy_logits_k+1, reward_k, value_k+1)
            # Use lax.scan with K=unroll_steps static.
            # ----------------------------------------------------------
            # We also compute the policy loss at each step using the visit
            # distribution at that same step (per-step CE).
            # The visit distribution at k=0 (root) uses the encoded
            # policy_logits at root (`policy_logits_all[:, 0]`), not a
            # dynamics-recurrent output — this matches MuZero's standard
            # setup (root policy from initial_inference).

            def unroll_step(latent_k, k):
                action_k = actions[:, k]  # (B,)
                dyn_out = self.model.apply(
                    {'params': params},
                    latent_k,
                    action_k,
                    mode='recurrent_inference',
                )
                next_latent = dyn_out['latent_state']         # (B, S)
                policy_logits_k1 = dyn_out['policy_logits']    # (B, A)
                value_logits_k1 = dyn_out['value']             # (B, support)
                reward_logits_k = dyn_out['reward']            # (B, support)
                return next_latent, (
                    next_latent,
                    policy_logits_k1,
                    value_logits_k1,
                    reward_logits_k,
                )

            # Note: we cannot use jax.lax.scan with a model.apply inside the
            # scanned body because each step closes over `params` (already
            # captured) — but Python-level static unrolling is fine for K=5.
            # We use a Python `for` loop here; the JIT trace fully unrolls
            # it once per static K. This matches train_phase2's Stage-B
            # implementation pattern (which also unrolls statically).
            pred_latents = []
            unroll_policy_logits = []
            unroll_reward_logits = []
            latent_k = latent_root
            for k in range(unroll_steps):
                action_k = actions[:, k]  # (B,)
                dyn_out = self.model.apply(
                    {'params': params},
                    latent_k,
                    action_k,
                    mode='recurrent_inference',
                )
                latent_k = dyn_out['latent_state']
                pred_latents.append(latent_k)
                unroll_policy_logits.append(dyn_out['policy_logits'])
                unroll_reward_logits.append(dyn_out['reward'])

            # Stack along time axis: (B, K, ...)
            pred_latents_stack = jnp.stack(pred_latents, axis=1)         # (B, K, S)
            unroll_policy_logits_stack = jnp.stack(
                unroll_policy_logits, axis=1
            )  # (B, K, A)
            unroll_reward_logits_stack = jnp.stack(
                unroll_reward_logits, axis=1
            )  # (B, K, support)

            # ----------------------------------------------------------
            # 3. Policy CE (per-asset 5-way softmax, per unroll step).
            # ----------------------------------------------------------
            # Stack root + unroll policy logits along T-axis: (B, K+1, A)
            policy_logits_T = jnp.concatenate(
                [policy_logits_all[:, 0:1, :], unroll_policy_logits_stack],
                axis=1,
            )  # (B, K+1, A)

            A = policy_logits_T.shape[-1]
            # Reshape both pred & target to (B, K+1, n_assets, 5) for per-asset CE.
            policy_logits_reshaped = policy_logits_T.reshape(
                B, T, n_assets, n_actions_per_asset
            )
            visit_dist_reshaped = visit_distributions.reshape(
                B, T, n_assets, n_actions_per_asset
            )

            # Per-step per-asset softmax cross-entropy. Output shape (B, K+1, n_assets).
            per_step_per_asset_ce = optax.softmax_cross_entropy(
                policy_logits_reshaped, visit_dist_reshaped
            )

            # Mask out unroll steps that are past end-of-episode (dones[k] = True).
            # The root step (k=0) is never masked. dones is (B, K), so we
            # prepend an all-False column for the root.
            done_mask_T = jnp.concatenate(
                [jnp.zeros((B, 1), dtype=jnp.bool_), dones], axis=1
            )  # (B, K+1)
            # valid mask is the inverse — broadcast to (B, K+1, n_assets).
            valid_mask = (~done_mask_T).astype(jnp.float32)[:, :, None]  # (B, K+1, 1)
            per_step_per_asset_ce = per_step_per_asset_ce * valid_mask

            # Average over (B, T, n_assets); divide by valid count to remain
            # invariant under dones masking.
            valid_count = jnp.maximum(jnp.sum(valid_mask) * n_assets, 1.0)
            policy_loss = jnp.sum(per_step_per_asset_ce) / valid_count

            # ----------------------------------------------------------
            # 4. Value CE — root value only (TASK_780 design: dynamics
            #    rollouts share the clean PnL trajectory, so we anchor the
            #    value head to root_value × value_scale).
            # ----------------------------------------------------------
            scaled_root_value = root_value * value_scale  # (B,)
            root_value_target = _scalar_to_categorical(scaled_root_value)
            value_loss = optax.softmax_cross_entropy(
                value_logits_root, root_value_target
            ).mean()

            # ----------------------------------------------------------
            # 5. Consistency loss — MSE between predicted unroll latents
            #    and target encoder latents at the same step.
            #    pred_latents_stack: (B, K, S)
            #    target_latents:     (B, K, S)
            # ----------------------------------------------------------
            consistency_diff = pred_latents_stack - target_latents       # (B, K, S)
            # Mean over feature dim, then per-step mean over (B, K).
            consistency_per_step = jnp.mean(consistency_diff ** 2, axis=-1)  # (B, K)
            valid_unroll_mask = (~dones).astype(jnp.float32)              # (B, K)
            valid_unroll_count = jnp.maximum(jnp.sum(valid_unroll_mask), 1.0)
            consistency_loss = (
                jnp.sum(consistency_per_step * valid_unroll_mask)
                / valid_unroll_count
            )

            # ----------------------------------------------------------
            # 6. Reward CE — per unroll step. Targets are scaled by
            #    train_phase2's REWARD_SCALE (=100.0) to land in the
            #    401-bin support; that matches the canonical pipeline
            #    so the bridge ckpt's reward head agrees with Phase-2
            #    Stage-B on the same scale.
            # ----------------------------------------------------------
            # rewards: (B, K) raw scalar PnL change. We follow
            # train_phase2.REWARD_SCALE = 100.0 (matches `train_phase2.py:75`).
            scaled_rewards = rewards * 100.0  # (B, K)
            reward_targets = _scalar_to_categorical(scaled_rewards)  # (B, K, support)
            # softmax_cross_entropy reduces over the last axis automatically.
            reward_ce = optax.softmax_cross_entropy(
                unroll_reward_logits_stack, reward_targets
            )  # (B, K)
            reward_loss = (
                jnp.sum(reward_ce * valid_unroll_mask) / valid_unroll_count
            )

            # ----------------------------------------------------------
            # 7. Total loss + metrics
            # ----------------------------------------------------------
            total_loss = (
                policy_w * policy_loss
                + value_w * value_loss
                + consistency_w * consistency_loss
                + reward_w * reward_loss
            )

            # Diagnostic: per-asset agreement at the ROOT only (matches
            # easy_trainer.py:236-238 semantics for sanity-gate comparison).
            policy_logits_root = policy_logits_all[:, 0, :]  # (B, A)
            visit_dist_root = visit_distributions[:, 0, :]   # (B, A)
            pred_action_type = jnp.argmax(
                policy_logits_root.reshape(B, n_assets, n_actions_per_asset),
                axis=-1,
            )
            oracle_action_type = jnp.argmax(
                visit_dist_root.reshape(B, n_assets, n_actions_per_asset),
                axis=-1,
            )
            per_asset_agreement = jnp.mean(
                (pred_action_type == oracle_action_type).astype(jnp.float32)
            )

            # TASK_147 balanced accuracy at root (HOLD = action type 1)
            oracle_is_trade = oracle_action_type != 1
            oracle_is_hold = oracle_action_type == 1
            trade_correct = (
                (pred_action_type == oracle_action_type) & oracle_is_trade
            )
            n_trades = jnp.sum(oracle_is_trade.astype(jnp.float32))
            agree_trade = jnp.where(
                n_trades > 0,
                jnp.sum(trade_correct.astype(jnp.float32)) / n_trades,
                0.0,
            )
            hold_correct = (
                (pred_action_type == oracle_action_type) & oracle_is_hold
            )
            n_holds = jnp.sum(oracle_is_hold.astype(jnp.float32))
            agree_hold = jnp.where(
                n_holds > 0,
                jnp.sum(hold_correct.astype(jnp.float32)) / n_holds,
                0.0,
            )
            balanced_agreement = 0.5 * agree_trade + 0.5 * agree_hold

            metrics = {
                'total_loss': total_loss,
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'consistency_loss': consistency_loss,
                'reward_loss': reward_loss,
                'per_asset_agreement': per_asset_agreement,
                'oracle_agreement': balanced_agreement,
                'agree_trade': agree_trade,
                'agree_hold': agree_hold,
                'pred_latent_std': jnp.std(pred_latents_stack),
                'target_latent_std': jnp.std(target_latents),
            }
            return total_loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        metrics['grad_norm'] = optax.global_norm(grads)
        return new_params, new_opt_state, metrics

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Run a single training step on a bridge batch (host-facing API)."""
        unroll_steps = self._unroll_steps
        self.params, self.opt_state, metrics = self._train_step(
            self.params, self.opt_state, batch, unroll_steps=unroll_steps
        )
        return {k: float(v) for k, v in metrics.items()}

    # ----------------------------------------------------------------
    # Evaluate (Phase-A sanity gate). Computes the same losses without
    # an optimizer step so it can be called on a val batch.
    # ----------------------------------------------------------------
    def _evaluate_batch_impl(
        self,
        params: Any,
        batch: Dict[str, jnp.ndarray],
        unroll_steps: int = 5,
    ) -> Dict[str, jnp.ndarray]:
        """Compute metrics on a single bridge batch (no gradient update)."""
        n_assets = self._n_assets
        n_actions_per_asset = self._n_actions_per_asset
        value_scale = self._value_scale

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        visit_distributions = batch['visit_distributions']
        root_value = batch['root_value']
        dones = batch['dones']

        B = observations.shape[0]
        T = observations.shape[1]
        C = observations.shape[2]
        W = observations.shape[3]
        N = observations.shape[4]

        obs_flat = observations.reshape(B * T, C, W, N)
        # TASK_786: chunked + remat encode bounds the peak HBM here.
        latent_flat, policy_flat, value_flat = self._encode_obs_chunked(
            params, obs_flat
        )
        latent_all = latent_flat.reshape(B, T, -1)
        policy_logits_all = policy_flat.reshape(B, T, -1)
        value_all = value_flat.reshape(B, T, VALUE_SUPPORT_SIZE)

        # Multi-step unroll (no gradient).
        latent_root = latent_all[:, 0, :]
        pred_latents = []
        unroll_policy_logits = []
        unroll_reward_logits = []
        latent_k = latent_root
        for k in range(unroll_steps):
            action_k = actions[:, k]
            dyn_out = self.model.apply(
                {'params': params},
                latent_k,
                action_k,
                mode='recurrent_inference',
            )
            latent_k = dyn_out['latent_state']
            pred_latents.append(latent_k)
            unroll_policy_logits.append(dyn_out['policy_logits'])
            unroll_reward_logits.append(dyn_out['reward'])

        pred_latents_stack = jnp.stack(pred_latents, axis=1)
        unroll_policy_logits_stack = jnp.stack(unroll_policy_logits, axis=1)
        unroll_reward_logits_stack = jnp.stack(unroll_reward_logits, axis=1)
        target_latents = latent_all[:, 1:, :]

        policy_logits_T = jnp.concatenate(
            [policy_logits_all[:, 0:1, :], unroll_policy_logits_stack], axis=1
        )

        policy_logits_reshaped = policy_logits_T.reshape(
            B, T, n_assets, n_actions_per_asset
        )
        visit_dist_reshaped = visit_distributions.reshape(
            B, T, n_assets, n_actions_per_asset
        )
        per_step_per_asset_ce = optax.softmax_cross_entropy(
            policy_logits_reshaped, visit_dist_reshaped
        )
        done_mask_T = jnp.concatenate(
            [jnp.zeros((B, 1), dtype=jnp.bool_), dones], axis=1
        )
        valid_mask = (~done_mask_T).astype(jnp.float32)[:, :, None]
        per_step_per_asset_ce = per_step_per_asset_ce * valid_mask
        valid_count = jnp.maximum(jnp.sum(valid_mask) * n_assets, 1.0)
        policy_loss = jnp.sum(per_step_per_asset_ce) / valid_count

        scaled_root_value = root_value * value_scale
        root_value_target = _scalar_to_categorical(scaled_root_value)
        value_loss = optax.softmax_cross_entropy(
            value_all[:, 0, :], root_value_target
        ).mean()

        consistency_diff = pred_latents_stack - target_latents
        consistency_per_step = jnp.mean(consistency_diff ** 2, axis=-1)
        valid_unroll_mask = (~dones).astype(jnp.float32)
        valid_unroll_count = jnp.maximum(jnp.sum(valid_unroll_mask), 1.0)
        consistency_loss = (
            jnp.sum(consistency_per_step * valid_unroll_mask) / valid_unroll_count
        )

        scaled_rewards = rewards * 100.0
        reward_targets = _scalar_to_categorical(scaled_rewards)
        reward_ce = optax.softmax_cross_entropy(
            unroll_reward_logits_stack, reward_targets
        )
        reward_loss = jnp.sum(reward_ce * valid_unroll_mask) / valid_unroll_count

        # Root-only sanity-gate metric (matches easy_trainer.py:399).
        policy_logits_root = policy_logits_all[:, 0, :]
        visit_dist_root = visit_distributions[:, 0, :]
        pred_action_type = jnp.argmax(
            policy_logits_root.reshape(B, n_assets, n_actions_per_asset), axis=-1
        )
        oracle_action_type = jnp.argmax(
            visit_dist_root.reshape(B, n_assets, n_actions_per_asset), axis=-1
        )
        per_asset_agreement = jnp.mean(
            (pred_action_type == oracle_action_type).astype(jnp.float32)
        )
        oracle_is_trade = oracle_action_type != 1
        oracle_is_hold = oracle_action_type == 1
        n_trades = jnp.sum(oracle_is_trade.astype(jnp.float32))
        n_holds = jnp.sum(oracle_is_hold.astype(jnp.float32))
        trade_correct = (pred_action_type == oracle_action_type) & oracle_is_trade
        hold_correct = (pred_action_type == oracle_action_type) & oracle_is_hold
        agree_trade = jnp.where(
            n_trades > 0,
            jnp.sum(trade_correct.astype(jnp.float32)) / n_trades,
            0.0,
        )
        agree_hold = jnp.where(
            n_holds > 0,
            jnp.sum(hold_correct.astype(jnp.float32)) / n_holds,
            0.0,
        )
        balanced_agreement = 0.5 * agree_trade + 0.5 * agree_hold

        return {
            'total_loss': (
                self._policy_weight * policy_loss
                + self._value_weight * value_loss
                + self._consistency_weight * consistency_loss
                + self._reward_weight * reward_loss
            ),
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'consistency_loss': consistency_loss,
            'reward_loss': reward_loss,
            'per_asset_agreement': per_asset_agreement,
            'oracle_agreement': balanced_agreement,
            'agree_trade': agree_trade,
            'agree_hold': agree_hold,
        }

    def evaluate_batch(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single bridge batch (host-facing API)."""
        metrics = self._eval_step(self.params, batch, unroll_steps=self._unroll_steps)
        return {k: float(v) for k, v in metrics.items()}

    # ----------------------------------------------------------------
    # Checkpoint contract (Phase 1 IMPLEMENTED — load contract must hold
    # from day 0 so `train_phase2.load_phase1_checkpoint` works on any
    # bridge checkpoint, even one written before training starts)
    # ----------------------------------------------------------------
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Save model checkpoint in the EasyTrainer pickle format.

        Mirrors `easy_trainer.py:428-447` exactly so
        `train_phase2.load_phase1_checkpoint` (`:96-100`) and
        `get_phase2_config(:2275)` work unchanged.
        """
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

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint. Mirrors `easy_trainer.py:449-469`."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.params = checkpoint['params']
        if 'opt_state' in checkpoint:
            self.opt_state = checkpoint['opt_state']
        return checkpoint


def create_bridge_trainer(config: EasyConfig, seed: int = 42) -> BridgeTrainer:
    """Factory mirroring `easy_trainer.create_trainer` (`:570-573`)."""
    rng_key = jax.random.PRNGKey(seed)
    return BridgeTrainer(config, rng_key)

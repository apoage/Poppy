"""
TASK_780 — Bridge Trainer (Oracle-as-MCTS Distillation, Phase A skeleton)

This module is the **parallel** Phase-1 trainer that replaces Foxglove's flat
soft-one-hot policy + scalar remaining-PnL targets with MCTS-shaped targets
(visit distributions, per-step policies, per-action Q-values, multi-step
latent-dynamics consistency loss). It runs alongside `easy_trainer.py`; the
canon path is selected via `easy_config.EasyConfig.use_oracle_mcts_distill`.

**Phase 1 status (this dispatch): SKELETON ONLY.**
- `__init__` wires the model with `use_prediction_head=True`, creates the
  optimizer (same `optax.chain(clip + adamw)` pattern as `easy_trainer.py`),
  and registers the JIT-ready training-step signature.
- `_train_step_impl` body raises `NotImplementedError("bridge_trainer Phase 2")`.
- `train_step` / `evaluate_batch` are thin wrappers that also raise.
- `save_checkpoint` / `load_checkpoint` preserve the EasyTrainer pickle
  contract (`{'params', 'opt_state', 'epoch', 'metrics', 'config', 'n_params'}`),
  so `train_phase2.load_phase1_checkpoint` (`:96-100`) works unchanged.

Phase 2 (next dispatch) fills `_train_step_impl` per TASK_697 §A.2:
    total_loss = policy_w * CE(policy_logits, visit_distribution)
               + value_w  * hybrid_value(value_logits, root_value)
               + reward_w * CE(reward_logits, oracle_step_reward)
               + consistency_w * ‖encode(s_t) − dynamics(h_{t−1}, a_{t−1})‖²

References
----------
- `tasks/TASK_780_ORACLE_AS_MCTS_DISTILLATION.md`
- `tasks/TASK_780_IMPLEMENTATION_PLAN.md`
- `tasks/TASK_697_ORACLE_AS_MCTS_DISTILLATION.md` §A.1, §A.2
- `runs/780_oracle_as_mcts/AUTHORIZATION_2026-05-26.md` §"Build phase outline"
- `easy_trainer.py:29-573` — the canon Phase-1 trainer being bridged.
- `train_phase2.py:96-100` — the load contract this trainer's checkpoints
  must satisfy.
- `train_phase2.py:1613-1700` — the Stage-A dynamics warmup that Phase-A
  imprint will replace.
- `mcts_trader/jax_agent/oracle_distillation.py` — the target-generation
  helpers this trainer consumes.
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


_PHASE2_TODO = "bridge_trainer Phase 2 — see tasks/TASK_780_IMPLEMENTATION_PLAN.md"


class BridgeTrainer:
    """Oracle-as-MCTS distillation trainer (Phase A imprint).

    Constructs `MuZeroNetworks(use_prediction_head=True, ...)` so the trained
    parameter tree carries the dynamics + decode-latent-to-market subtree from
    epoch 0. The resulting checkpoint is loadable under
    `train_phase2.load_phase1_checkpoint` with the dynamics block ALREADY warm —
    Stage-A's consistency-only warmup becomes either a short fine-tune or is
    skipped (operator decision at smoke time per AUTHORIZATION §"Build phase
    outline").

    Phase 1 deliverable
    -------------------
    The training-step body is NotImplementedError. The class is intentionally
    constructed end-to-end (model init, optimizer, value support, JIT
    registration) so the parameter tree shape is locked in for Phase 2's
    loss implementation and so the checkpoint contract is verifiable by
    `train_phase2.load_phase1_checkpoint` without ever calling `train_step`.
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

        # Value support for hybrid value loss (Phase 2 will use it; cached here
        # so the JIT step doesn't re-derive it).
        self.value_support = get_value_support()

        # JIT registration. Static argnames match the kwargs the Phase-2 body
        # will need; declaring them now keeps the recompile surface stable.
        self._train_step = jax.jit(
            self._train_step_impl,
            static_argnames=['unroll_steps'],
        )

    # ----------------------------------------------------------------
    # Training step (Phase 2 fills the body)
    # ----------------------------------------------------------------
    def _train_step_impl(
        self,
        params: Any,
        opt_state: Any,
        batch: Dict[str, jnp.ndarray],
        unroll_steps: int = 5,
    ) -> Tuple[Any, Any, Dict[str, jnp.ndarray]]:
        """Single training step.

        Phase 2 will implement the MuZero-shaped loss per TASK_697 §A.2:
            policy_w * CE(policy_logits, visit_distribution)
          + value_w * hybrid_value(value_logits, root_value)
          + reward_w * CE(reward_logits, oracle_step_reward)
          + consistency_w * ‖encode(s_t) − dynamics(h_{t−1}, a_{t−1})‖²

        Buffer keys expected by Phase 2:
            observations          (B, C, W, N)
            visit_distributions   (B, T_unroll, n_actions)
            step_values           (B, T_unroll)
            step_actions          (B, T_unroll, n_assets)  int
            per_step_rewards      (B, T_unroll)
            root_value            (B,)
            policy_target         (B, n_actions)        root visit distribution
        """
        raise NotImplementedError(_PHASE2_TODO)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Run a single training step. Phase 2 fills."""
        raise NotImplementedError(_PHASE2_TODO)

    def evaluate_batch(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single batch (for the Phase-A sanity gate). Phase 2 fills.

        Phase 5 will implement the per-asset Bal.Acc comparator. Specifically,
        per AUTHORIZATION §"Phase-A sanity gate" item 1, the predicted action
        per asset is `argmax(visit_distribution[asset, :])` over the 5 actions
        — directly comparable to `easy_trainer.py:287`'s `oracle_agreement`
        (balanced trade/hold accuracy).
        """
        raise NotImplementedError(_PHASE2_TODO)

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

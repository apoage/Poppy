#!/usr/bin/env python3
"""
Phase 2 Training — Dynamics Warmup + MCTS ExIt.

TASK_716: Separate entry point for Phase 2. Does NOT modify train_easy.py.

Usage:
    # Dry run (validate checkpoint + pool loading)
    python train_phase2.py --checkpoint jax_ckpt_easy_A/branch_A_historical/phase1_best.pkl --dry-run

    # Stage A: Dynamics warmup
    python train_phase2.py --checkpoint ... --stage A --epochs 500

    # Stage B: MCTS ExIt
    python train_phase2.py --checkpoint phase2_stageA_best.pkl --stage B --epochs 1000
"""

import os
import sys
import json
import logging
import warnings
import argparse
import pickle
from pathlib import Path

# GPU optimization (before JAX import)
_trader_dir = os.path.dirname(os.path.abspath(__file__))
_xla_cache = os.path.join(_trader_dir, '.xla_cache')
_jax_cache = os.path.join(_trader_dir, '.jax_cache')
os.makedirs(_xla_cache, exist_ok=True)
os.makedirs(_jax_cache, exist_ok=True)
os.environ['XLA_FLAGS'] = f'--xla_gpu_autotune_level=4 --xla_gpu_per_fusion_autotune_cache_dir={_xla_cache}'
# NOTE: Do NOT use cuda_malloc_async — causes ILLEGAL_ADDRESS crashes under heavy
# allocation churn (thousands of small GPU ops per epoch from per-slot encoding + MCTS).
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['JAX_COMPILATION_CACHE_DIR'] = _jax_cache
# Preallocate GPU memory to avoid fragmentation from repeated alloc/free cycles
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'
warnings.filterwarnings('ignore', category=FutureWarning, module='jax')
logging.getLogger('jax._src.xla_bridge').setLevel(logging.ERROR)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
import threading
import queue
import csv
from collections import defaultdict

from phase2_config import Phase2Config, get_phase2_config
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE
from mcts_trader.jax_agent import get_value_support
from mcts_trader.jax_agent.popart import (
    PopArtState, init_popart_state, update_statistics,
    normalize_targets, denormalize_values,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dynamics parameter key prefixes (all 9 leaf tensors)
DYNAMICS_KEYS = frozenset([
    'action_embedding', 'dynamics_dense1', 'dynamics_dense2',
    'dynamics_ln', 'dynamics_state_norm',
])

# Per-step rewards ~0.001, need 100x to spread across 401 bins.
# Fallback constant retained for backward compat; runtime path reads
# self.config.reward_scale (TASK_762) — see SegmentBatchGenerator below.
REWARD_SCALE = 100.0


def scalar_to_categorical(values: jnp.ndarray) -> jnp.ndarray:
    """Convert scalar values to categorical one-hot targets over 401 bins in [-10, 10].

    Same pattern as EasyTrainer._scalar_to_target but as a standalone function for JIT.
    """
    values = jnp.clip(values, -10.0, 10.0)
    bin_width = 20.0 / (VALUE_SUPPORT_SIZE - 1)
    bin_idx = ((values + 10.0) / bin_width).astype(jnp.int32)
    bin_idx = jnp.clip(bin_idx, 0, VALUE_SUPPORT_SIZE - 1)
    return jax.nn.one_hot(bin_idx, VALUE_SUPPORT_SIZE)


def _is_dynamics_key(path) -> bool:
    """Check if a param path belongs to dynamics layers."""
    key = str(path[0]) if path else ''
    return any(dk in key for dk in DYNAMICS_KEYS)


def load_phase1_checkpoint(ckpt_path: str):
    """Load Phase 1 checkpoint and return (params, config_dict, epoch, metrics)."""
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    return ckpt['params'], ckpt.get('config', {}), ckpt['epoch'], ckpt.get('metrics', {})


def analyze_params(params):
    """Analyze parameter sparsity. Returns (dynamics_info, encoder_info, total_params)."""
    flat = jax.tree_util.tree_leaves_with_path(params)
    total = sum(leaf.size for _, leaf in flat)

    dynamics_zero = 0
    dynamics_total = 0
    encoder_zero = 0
    encoder_total = 0

    for path, leaf in flat:
        key = str(path[0])
        is_dynamics = any(dk in key for dk in DYNAMICS_KEYS)
        sparsity = float((leaf == 0).mean())
        if is_dynamics:
            dynamics_total += 1
            if sparsity > 0.99:
                dynamics_zero += 1
        else:
            encoder_total += 1
            if sparsity > 0.99:
                encoder_zero += 1

    return {
        'dynamics_zero': dynamics_zero,
        'dynamics_total': dynamics_total,
        'encoder_zero': encoder_zero,
        'encoder_total': encoder_total,
        'total_params': total,
    }


def warm_policy_direction_refiner(params, alpha: float = 0.5, n_assets: int = 64):
    """TASK_728b: Break zero-init cold-start by seeding PolicyDirectionRefiner
    with per-action priors. For each asset j, the SELL/LSELL slots get
    coefficient −α on its own direction logit, BUY/LBUY get +α, HOLD gets 0.
    Other (cross-asset) entries stay zero so direction flows asset-locally.

    Action order: 0=Sell, 1=Hold, 2=Buy, 3=LimitSell, 4=LimitBuy.
    """
    if 'policy_direction_refiner' not in params:
        logger.warning("policy_direction_refiner not in params — skipping warm-start")
        return params
    inner = params['policy_direction_refiner']['dir_to_policy']
    kernel = inner['kernel']  # (n_assets, n_assets * 5)
    n_actions = kernel.shape[1] // n_assets
    assert n_actions == 5, f"Expected 5 actions, got {n_actions}"
    action_priors = jnp.array([-alpha, 0.0, alpha, -alpha, alpha], dtype=kernel.dtype)
    eye = jnp.eye(n_assets, dtype=kernel.dtype)
    new_kernel = (eye[:, :, None] * action_priors[None, None, :]).reshape(
        n_assets, n_assets * n_actions
    )
    new_params = dict(params)
    new_params['policy_direction_refiner'] = {
        'dir_to_policy': {
            'kernel': new_kernel,
            'bias': jnp.zeros_like(inner['bias']),
        }
    }
    logger.info(f"Warmed PolicyDirectionRefiner kernel with α={alpha} "
                f"(BUY/LBUY=+{alpha}, SELL/LSELL=−{alpha}, HOLD=0)")
    return new_params


def initialize_missing_aux_heads(model, params, config, rng_key):
    """TASK_727: If checkpoint predates an aux head (direction_head, oracle_mask),
    initialize that subtree fresh while keeping all existing params.
    TASK_741 (Mint): also fills the nested trm/encoder/film_projector subtree
    when the loaded checkpoint predates per-asset injection.

    Returns new params with any missing top-level keys filled in.
    """
    dummy_obs = jnp.zeros((1, config.n_channels, config.window_size, config.n_assets),
                          dtype=jnp.float32)
    dummy_action = jnp.zeros((1,), dtype=jnp.int32)
    fresh = model.init(rng_key, dummy_obs, action=dummy_action,
                       mode='initialize_parameters')['params']

    existing_keys = set(params.keys())
    added = []
    new_params = dict(params)
    for k, v in fresh.items():
        if k not in existing_keys:
            new_params[k] = v
            added.append(k)
    if added:
        logger.info(f"Initialized fresh aux head params: {added}")

    # TASK_741: deep-merge film_projector if nested under trm/encoder
    if 'trm' in new_params and 'encoder' in new_params['trm']:
        fresh_enc = fresh.get('trm', {}).get('encoder', {})
        if 'film_projector' in fresh_enc and 'film_projector' not in new_params['trm']['encoder']:
            new_enc = dict(new_params['trm']['encoder'])
            new_enc['film_projector'] = fresh_enc['film_projector']
            new_trm = dict(new_params['trm'])
            new_trm['encoder'] = new_enc
            new_params['trm'] = new_trm
            logger.info("Initialized fresh trm/encoder/film_projector (TASK_741)")

        # TASK_749: deep-merge titans memory subtree
        if 'titans' in fresh_enc and 'titans' not in new_params['trm']['encoder']:
            new_enc = dict(new_params['trm']['encoder'])
            new_enc['titans'] = fresh_enc['titans']
            new_trm = dict(new_params['trm'])
            new_trm['encoder'] = new_enc
            new_params['trm'] = new_trm
            logger.info("Initialized fresh trm/encoder/titans (TASK_749)")

    return new_params


def initialize_dynamics(params, rng_key):
    """Replace zero dynamics weights with proper random initialization.

    - Dense kernels: He normal (variance scaling, fan_in)
    - Embed: normal(0, 0.02)
    - LayerNorm scale: ones, bias: zeros
    - Dense bias: zeros (already correct)

    Returns new params dict with dynamics initialized, encoder untouched.
    """
    flat = jax.tree_util.tree_leaves_with_path(params)
    replacements = {}

    for path, leaf in flat:
        top_key = str(path[0])
        if not any(dk in top_key for dk in DYNAMICS_KEYS):
            continue

        sub_key = str(path[-1]) if len(path) > 1 else ''

        if 'embedding' in sub_key:
            # Embed: normal(0, 0.02)
            rng_key, sub_key_rng = jax.random.split(rng_key)
            new_val = jax.random.normal(sub_key_rng, leaf.shape, dtype=leaf.dtype) * 0.02
        elif 'kernel' in sub_key:
            # Dense kernel: He normal (sqrt(2/fan_in))
            rng_key, sub_key_rng = jax.random.split(rng_key)
            fan_in = leaf.shape[0]
            std = np.sqrt(2.0 / fan_in)
            new_val = jax.random.normal(sub_key_rng, leaf.shape, dtype=leaf.dtype) * std
        elif 'scale' in sub_key:
            # LayerNorm scale: ones
            new_val = jnp.ones_like(leaf)
        elif 'bias' in sub_key:
            # Bias: zeros (already zero, but explicit)
            new_val = jnp.zeros_like(leaf)
        else:
            continue

        replacements[path] = new_val

    # Rebuild params tree with replacements
    def replace_leaf(path, leaf):
        return replacements.get(path, leaf)

    new_params = jax.tree_util.tree_map_with_path(replace_leaf, params)

    n_replaced = len(replacements)
    logger.info(f"Initialized {n_replaced} dynamics tensors (He normal + embed normal(0,0.02))")
    return new_params


def make_dynamics_mask(params):
    """Create boolean mask: True for dynamics params, False for frozen params.

    Used with optax.masked to zero out gradients for non-dynamics params.
    """
    return jax.tree_util.tree_map_with_path(
        lambda path, _: _is_dynamics_key(path), params
    )


def create_dynamics_optimizer(config: Phase2Config, params):
    """Create optimizer that only updates dynamics params (Stage A).

    Uses multi_transform: adamw for dynamics, set_to_zero for everything else.

    Returns (optimizer, opt_state).
    """
    total_steps = config.epochs * config.steps_per_epoch
    warmup_steps = min(200, max(total_steps - 1, 0))

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.dynamics_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps, warmup_steps + 1),
        end_value=config.dynamics_lr * 0.1,
    )

    train_opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=1e-4),
    )
    freeze_opt = optax.set_to_zero()

    mask = make_dynamics_mask(params)  # True=dynamics, False=frozen
    labels = jax.tree_util.tree_map(lambda x: 'train' if x else 'freeze', mask)

    optimizer = optax.multi_transform(
        {'train': train_opt, 'freeze': freeze_opt},
        labels,
    )
    opt_state = optimizer.init(params)
    return optimizer, opt_state


def make_dynamics_train_step(model, config: Phase2Config, optimizer):
    """Build JIT-compiled dynamics training step for Stage A.

    Captures model and optimizer in closure so they don't need to be JAX args.
    Returns a function: (params, opt_state, obs_t, obs_t1, actions, rewards)
        → (new_params, new_opt_state, metrics_dict)
    """
    consistency_weight = config.consistency_weight
    reward_weight = config.reward_weight

    @jax.jit
    def dynamics_train_step(params, opt_state, obs_t, obs_t1, actions, rewards):
        """Single dynamics training step.

        Args:
            params: Full model params (dynamics trainable, rest frozen by optimizer mask)
            opt_state: Optimizer state
            obs_t: (B, C, W, N) observations at time t
            obs_t1: (B, C, W, N) observations at time t+1
            actions: (B,) oracle actions at decision point t
            rewards: (B,) per-step scalar rewards (pre-scaled by REWARD_SCALE)
        """
        def loss_fn(params):
            # Encode both observations through frozen encoder (stop_gradient)
            out_t = model.apply({'params': params}, observation=obs_t, mode='initial_inference')
            out_t1 = model.apply({'params': params}, observation=obs_t1, mode='initial_inference')
            latent_t = jax.lax.stop_gradient(out_t['latent_state'])
            latent_t1 = jax.lax.stop_gradient(out_t1['latent_state'])

            # Predict next state via dynamics (trainable)
            dyn_out = model.apply({'params': params}, latent_t, actions, mode='recurrent_inference')
            pred_latent = dyn_out['latent_state']
            pred_reward = dyn_out['reward']  # (B, 401) logits

            # Loss 1: Consistency — dynamics predicts next latent
            consistency_loss = jnp.mean((pred_latent - latent_t1) ** 2)

            # Loss 2: Reward prediction — categorical CE
            reward_target = scalar_to_categorical(rewards)  # (B, 401)
            reward_loss = optax.softmax_cross_entropy(pred_reward, reward_target).mean()

            total_loss = consistency_weight * consistency_loss + reward_weight * reward_loss

            return total_loss, {
                'total_loss': total_loss,
                'consistency_loss': consistency_loss,
                'reward_loss': reward_loss,
                'pred_latent_std': jnp.std(pred_latent),
                'pred_latent_mean': jnp.mean(jnp.abs(pred_latent)),
                'target_latent_std': jnp.std(latent_t1),
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        metrics['grad_norm'] = optax.global_norm(grads)
        return new_params, new_opt_state, metrics

    return dynamics_train_step


def create_model(config: Phase2Config):
    """Create MuZeroNetworks model from config (same pattern as EasyTrainer)."""
    dtype_str = getattr(config, 'compute_dtype', 'float32')
    compute_dtype = jnp.bfloat16 if dtype_str == 'bfloat16' else jnp.float32

    model = MuZeroNetworks(
        state_dim=config.state_dim,
        action_space_size=config.action_space_size,
        support_size=VALUE_SUPPORT_SIZE,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        use_prediction_head=False,
        n_assets=config.n_assets,
        encoder_version=getattr(config, 'encoder_version', 2),
        hidden_dim=getattr(config, 'trm_hidden_dim', None),
        embed_dim=getattr(config, 'embed_dim', 32),
        encoder_num_heads=getattr(config, 'encoder_num_heads', 4),
        dtype=compute_dtype,
        param_dtype=jnp.float32,
        use_oracle_mask_head=getattr(config, 'use_oracle_mask_head', False),
        oracle_mask_hidden_dim=getattr(config, 'oracle_mask_hidden_dim', 256),
        use_direction_head=getattr(config, 'use_direction_head', False),
        direction_hidden_dim=getattr(config, 'direction_hidden_dim', 128),
        use_direction_policy_bias=getattr(config, 'use_direction_policy_bias', False),
        use_per_asset_policy_head=getattr(config, 'use_per_asset_policy_head', False),
        per_asset_policy_hidden_dim=getattr(config, 'per_asset_policy_hidden_dim', 128),
        use_rank_head=getattr(config, 'use_rank_head', False),
        rank_hidden_dim=getattr(config, 'rank_hidden_dim', 128),
        use_aux_price_head=getattr(config, 'use_aux_price_head', False),
        aux_price_hidden_dim=getattr(config, 'aux_price_hidden_dim', 256),
        aux_price_horizons=tuple(getattr(config, 'aux_price_horizons', (1, 5, 10, 20))),
        use_bella_distill_head=getattr(config, 'use_bella_distill_head', False),
        bella_distill_hidden_dim=getattr(config, 'bella_distill_hidden_dim', 256),
        bella_pred_horizons=tuple(getattr(config, 'bella_pred_horizons', (1, 5, 10, 20))),
        # TASK_754: trajectory-shape head plumbing
        use_bella_shape_head=getattr(config, 'use_bella_shape_head', False),
        bella_shape_descriptor_dim=getattr(config, 'bella_shape_descriptor_dim', 14),
        bella_shape_hidden_dim=getattr(config, 'bella_shape_hidden_dim', 256),
        use_per_asset_injection=getattr(config, 'use_per_asset_injection', False),
        per_asset_inject_xs=getattr(config, 'per_asset_inject_xs', False),
        per_asset_inject_cheat=getattr(config, 'per_asset_inject_cheat', False),
        # TASK_749 (Titans-MAG)
        use_titans_memory=getattr(config, 'use_titans_memory', False),
        titans_memory_depth=getattr(config, 'titans_memory_depth', 2),
        titans_memory_hidden=getattr(config, 'titans_memory_hidden', None),
        titans_persistent_tokens=getattr(config, 'titans_persistent_tokens', 4),
        titans_surprise_momentum_init=getattr(config, 'titans_surprise_momentum_init', 0.9),
        titans_decay_init=getattr(config, 'titans_decay_init', 0.01),
        titans_lr_init=getattr(config, 'titans_lr_init', 0.01),
        titans_gate_init=getattr(config, 'titans_gate_init', -10.0),
    )
    return model


def load_segment_pool(config: Phase2Config, periods_file: str, db_path: str,
                      n_slots: int = 17, max_runs_per_group: int = 1):
    """Create SegmentPool from periods file (reuses train_easy.py's SegmentPool).

    Args:
        n_slots: Number of concurrent segments in RAM. 17 = one per regime (default).
        max_runs_per_group: Cap runs per date range (1 = one asset set per regime).
    """
    from train_easy import SegmentPool

    with open(periods_file) as f:
        periods_data = json.load(f)
    periods_list = periods_data.get('periods', periods_data if isinstance(periods_data, list) else [])

    period_specs = []
    for period in periods_list:
        period_specs.append({
            'start_date': period['start_date'],
            'end_date': period['end_date'],
            'assets': period.get('assets', None),
        })
    if not period_specs:
        raise ValueError("No valid periods in periods-file!")

    window_len = config.episode_length + config.window_size + config.oracle_lookahead + 5
    oracle_cfg = {
        'oracle_k': config.oracle_k,
        'oracle_lookahead': config.oracle_lookahead,
        'transaction_cost': config.transaction_cost,
        'min_alpha_threshold': getattr(config, 'min_alpha_threshold', 0.0005),
        # TASK_739: route BC-target oracle choice through to SegmentPool
        'oracle_mode': getattr(config, 'oracle_mode', 'topk_path'),
        'sign_epsilon': getattr(config, 'sign_epsilon', 1e-4),
        # TASK_740: route Bella-distill config through to SegmentPool so that
        # _make_slot can load the matching precomputed shard into the slot.
        'use_bella_distill_head': getattr(config, 'use_bella_distill_head', False),
        'bella_preds_manifest': getattr(config, 'bella_preds_manifest',
                                         'config/bella_preds_manifest.json'),
        'bella_pred_horizons': tuple(getattr(config, 'bella_pred_horizons',
                                             (1, 5, 10, 20))),
        # TASK_754: route trajectory-shape config through to SegmentPool so
        # _make_slot can load the matching precomputed shape shard.
        'use_bella_shape_head': getattr(config, 'use_bella_shape_head', False),
        'bella_shape_descriptor_dim': getattr(config, 'bella_shape_descriptor_dim', 14),
        'bella_shape_manifest': getattr(config, 'bella_shape_manifest',
                                        'config/bella_shape_manifest.json'),
        # TASK_753: Bella-as-data augmentation. Pool reads these on init and
        # injects Bella-sourced slots with probability bella_augment_ratio.
        'bella_augment': getattr(config, 'bella_augment', False),
        'bella_augment_ratio': getattr(config, 'bella_augment_ratio', 0.0),
        'bella_augment_mode': getattr(config, 'bella_augment_mode', 'append'),
        'bella_augment_regime_mode': getattr(config, 'bella_augment_regime_mode',
                                              'uniform'),
        'bella_augment_checkpoint': getattr(config, 'bella_augment_checkpoint',
                                            'checkpoints/diff_teacher/generator_best/generator'),
        'bella_augment_asset_policy': getattr(config, 'bella_augment_asset_policy',
                                              'remap'),
        'bella_augment_volatility_scale': getattr(config, 'bella_augment_volatility_scale',
                                                  0.015),
        'bella_augment_k_samples': getattr(config, 'bella_augment_k_samples', 1),
        'bella_augment_seq_length': getattr(config, 'bella_augment_seq_length', 0),
        'bella_augment_refresh_epochs': getattr(config, 'bella_augment_refresh_epochs', 0),
        'bella_augment_log_every': getattr(config, 'bella_augment_log_every', 100),
    }

    pool = SegmentPool(
        period_specs, db_path,
        n_assets=config.n_assets, n_slots=n_slots,
        window_length=window_len, min_assets=32,
        max_runs_per_group=max_runs_per_group,
        oracle_config=oracle_cfg,
    )
    return pool


class AsyncBatchPipeline:
    """TASK_721: Async two-phase pipeline — numpy on background thread, JAX on main thread.

    Replaces ConsecutivePairGenerator + PrefetchGenerator with proper thread safety:
    - Phase 1 (_prepare_numpy): CPU/numpy only, runs on background thread
    - Phase 2 (_encode_jax): JAX/GPU encoding, runs on main thread

    Uses random sampling within slots (not sequential cursor) for better diversity.
    Vectorized n-step reward computation (no Python loop).
    """

    def __init__(self, pool, config: Phase2Config, prefetch: int = 2, batch_size: int = None):
        from mcts_trader.env.jax_encoder import EncoderConfig, jit_batch_encode_temporal
        self.pool = pool
        self.config = config
        self.ws = config.window_size
        self.n_assets = config.n_assets
        # TASK_738: honor legacy 27-ch mode if config says so
        _enc_n_channels = 32 if getattr(config, 'use_cross_sectional', True) else 27
        self.encoder_config = EncoderConfig(
            n_assets=config.n_assets,
            window_size=config.window_size,
            n_channels=_enc_n_channels,
        )
        self._encode_fn = jit_batch_encode_temporal
        # TASK_738: threshold adapts to cross-sectional toggle
        self.use_state_channels = config.n_channels > _enc_n_channels
        self._rng = np.random.default_rng(config.seed + 3000)
        self._rr_idx = 0
        self._batch_size = batch_size or getattr(config, 'stage_b_batch_size', config.batch_size)
        self._queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _prepare_numpy(self, batch_size: int) -> dict:
        """Phase 1: CPU/numpy only. Thread-safe. NO JAX operations."""
        pool = self.pool
        ws = self.ws
        n_assets = self.n_assets
        nstep = self.config.nstep_horizon

        # 1. RANDOM SAMPLING — uniform within each slot's valid range
        n_slots = len(pool.slots)
        if n_slots == 0:
            raise RuntimeError("SegmentPool exhausted — no slots available")

        per_slot = batch_size // n_slots + 1
        positions = []
        for i in range(n_slots):
            if len(positions) >= batch_size:
                break
            slot_idx = (self._rr_idx + i) % n_slots
            slot = pool.slots[slot_idx]
            T_ret = len(slot['returns'])
            T_prices = len(slot['hm'].cache_data)
            max_cursor = min(T_ret - ws - nstep, T_prices - ws - nstep)
            if max_cursor <= 0:
                continue
            n_needed = min(per_slot, batch_size - len(positions))
            cursors = self._rng.integers(0, max_cursor, size=n_needed)
            for c in cursors:
                positions.append((slot_idx, int(c)))
        self._rr_idx += n_slots

        if not positions:
            raise RuntimeError("No valid positions — all slots too short")

        # 2. Group by slot for efficient array slicing
        by_slot = defaultdict(list)
        for slot_idx, cursor in positions[:batch_size]:
            by_slot[slot_idx].append(cursor)

        # 3. Per-slot: extract numpy data (NO JAX)
        #    Key optimization: extract per-sample windows (ws+nstep each) instead of
        #    one huge slice per slot. With random sampling c_min~0, c_max~T causes
        #    the entire slot to be transferred to GPU. Per-sample windows are compact.
        win_len = ws + nstep  # length of each extracted window
        all_windows = []      # list of (n, win_len, n_assets, 5) numpy per slot
        all_actions = []
        all_rewards = []
        all_oracle_targets = []
        all_oracle_values = []
        all_nstep_rewards = []
        all_future_cum = []   # TASK_727: per-asset cum future return → direction target
        all_future_log_rets = []  # TASK_735: per-asset log-return at horizons → aux price head target
        all_bella_preds = []      # TASK_740: per-asset Bella-predicted log-return at horizons
        all_bella_shape = []      # TASK_754: per-window 14-float Bella trajectory-shape descriptor
        # State channels
        all_state_t = []
        all_conf_t = []
        all_state_t1 = []
        all_conf_t1 = []
        all_state_tN = []
        all_conf_tN = []
        dir_horizon = getattr(self.config, 'direction_horizon', 20)
        aux_price_horizons = tuple(getattr(self.config, 'aux_price_horizons', (1, 5, 10, 20)))
        max_aux_horizon = max(aux_price_horizons) if aux_price_horizons else 0
        use_bella_distill = getattr(self.config, 'use_bella_distill_head', False)
        bella_pred_horizons = tuple(getattr(self.config, 'bella_pred_horizons', (1, 5, 10, 20)))
        use_bella_shape = getattr(self.config, 'use_bella_shape_head', False)
        bella_shape_dim = int(getattr(self.config, 'bella_shape_descriptor_dim', 14))

        for slot_idx, cursors in by_slot.items():
            slot = pool.slots[slot_idx]
            cursors_arr = np.array(cursors)
            n = len(cursors_arr)
            cache = slot['hm'].cache_data  # (T, n_assets, 5)

            # Extract per-sample windows: each (win_len, n_assets, 5)
            # Vectorized via fancy indexing: build (n, win_len) index array
            offsets = np.arange(win_len)  # (win_len,)
            idx = cursors_arr[:, None] + offsets[None, :]  # (n, win_len)
            windows = cache[idx]  # (n, win_len, n_assets, 5)
            all_windows.append(windows)

            decision_indices = cursors_arr + ws - 1

            # Oracle action (numpy)
            policy_targets = slot['policy_targets'][decision_indices].astype(np.float32)
            actions = np.argmax(policy_targets, axis=-1)  # (n,) in [0, 320)
            all_actions.append(actions)
            all_oracle_targets.append(policy_targets)

            # Per-step reward (numpy)
            oracle_sel = slot['oracle_selections'][decision_indices]
            step_returns = slot['returns'][decision_indices]
            k_count = np.maximum(np.sum(oracle_sel, axis=-1), 1.0)
            step_pnl = np.sum(np.abs(step_returns) * oracle_sel, axis=-1) / k_count
            prev_indices = np.maximum(decision_indices - 1, 0)
            prev_sel = slot['oracle_selections'][prev_indices]
            changes = np.sum(np.abs(oracle_sel - prev_sel), axis=-1)
            # TASK_762: transaction_cost + fee_penalty_bps (bps → fraction = ×1e-4)
            per_trade_cost = self.config.transaction_cost + self.config.fee_penalty_bps * 1e-4
            step_pnl = step_pnl - changes * per_trade_cost
            rewards = np.clip(step_pnl * self.config.reward_scale, -10.0, 10.0).astype(np.float32)
            all_rewards.append(rewards)

            # Oracle remaining PnL (numpy)
            oracle_cum_pnl = slot['oracle_cum_pnl']
            horizon = self.config.oracle_value_horizon

            # TASK_784: optional EWMA smoothing on per-step deltas of
            # oracle_cum_pnl before the remaining-PnL gather. λ=0 → branch
            # skipped → cum_pnl unchanged → bit-identical canon. λ>0 dampens
            # per-window peakiness that drives 746's seed-variance signature.
            # Done per-slot at batch-build (oracle_cum_pnl is the slot-resident
            # RAM array, ~T floats, T≈400; recursion is a single numpy loop).
            oracle_smoothing = getattr(self.config, 'oracle_smoothing', 0.0)
            if oracle_smoothing > 0.0:
                deltas = np.diff(oracle_cum_pnl)  # (T-2,) — first-difference
                smoothed = np.empty_like(deltas)
                smoothed[0] = deltas[0]
                lam = float(oracle_smoothing)
                one_minus_lam = 1.0 - lam
                # Tight Python loop over ~T-2 ≈ 400 entries — well below the
                # batch-build wall (~ms), no JAX/JIT involvement.
                for i in range(1, len(deltas)):
                    smoothed[i] = one_minus_lam * deltas[i] + lam * smoothed[i - 1]
                # Reconstruct cum_pnl-shaped array anchored at oracle_cum_pnl[0].
                cum_pnl_eff = np.concatenate(
                    [oracle_cum_pnl[:1], oracle_cum_pnl[0] + np.cumsum(smoothed)]
                ).astype(oracle_cum_pnl.dtype)
            else:
                cum_pnl_eff = oracle_cum_pnl

            end_idx = np.minimum(decision_indices + horizon, len(oracle_cum_pnl) - 1)
            remaining = (cum_pnl_eff[end_idx] - cum_pnl_eff[decision_indices]).astype(np.float32)

            # TASK_762 F1 fix: oracle_cum_pnl is precomputed upstream and includes
            # transaction_cost but NOT fee_penalty_bps. Without this branch the
            # reward head trains on penalized signal while the value head trains
            # on unpenalized PnL — credit assignment breaks, fee penalty has no
            # teeth. Only do the extra index gather when the knob is non-zero
            # (canon = 0, hot path stays unchanged).
            extra_bps = self.config.fee_penalty_bps * 1e-4
            if extra_bps != 0.0:
                horizon_offsets = np.arange(1, horizon + 1)
                horizon_idx = np.minimum(
                    decision_indices[:, None] + horizon_offsets[None, :],
                    len(oracle_cum_pnl) - 1,
                )
                # Mask steps beyond end_idx per-sample (variable horizon clamp).
                mask = horizon_idx <= end_idx[:, None]
                sel_h = slot['oracle_selections'][horizon_idx]                 # (n, H, n_assets)
                sel_h_prev = slot['oracle_selections'][np.maximum(horizon_idx - 1, 0)]
                oracle_changes = (np.abs(sel_h - sel_h_prev).sum(axis=-1) * mask).sum(axis=-1)
                remaining = (remaining - oracle_changes * extra_bps).astype(np.float32)
            all_oracle_values.append(remaining)

            # VECTORIZED n-step rewards (numpy, no Python loop)
            steps = decision_indices[:, None] + np.arange(nstep)  # (n, N)
            steps = np.minimum(steps, len(oracle_cum_pnl) - 2)
            # TASK_762: scale via config knob, not module constant
            per_step_pnl = oracle_cum_pnl[steps + 1] - oracle_cum_pnl[steps]
            # TASK_762 F1 fix: same fee_penalty_bps asymmetry as oracle-remaining
            # above. Subtract per-step extra penalty before scaling. Canon = 0.
            if extra_bps != 0.0:
                sel_t = slot['oracle_selections'][steps]
                sel_tm1 = slot['oracle_selections'][np.maximum(steps - 1, 0)]
                nstep_changes = np.abs(sel_t - sel_tm1).sum(axis=-1)            # (n, N)
                per_step_pnl = per_step_pnl - nstep_changes * extra_bps
            per_step_r = per_step_pnl * self.config.reward_scale
            # TASK_769: F2 fix — per-step n-step reward clip is upstream of
            # PopArt. pre_popart (canon) keeps the original [-10, 10] clip on
            # each step; the discounted sum and PopArt downstream then see
            # already-censored steps. post_popart drops the clip here so the
            # full reward distribution propagates to PopArt and the post-norm
            # clip on td_target_norm handles outliers. The downstream
            # td_target_raw clip in make_stage_b_train_step (nstep branch) is
            # also gated symmetrically.
            if getattr(self.config, 'value_clip_mode', 'pre_popart') == 'pre_popart':
                per_step_r = np.clip(per_step_r, -10.0, 10.0)
            discounts = self.config.td_discount ** np.arange(nstep)
            nstep_sum = (per_step_r * discounts).sum(axis=1).astype(np.float32)
            all_nstep_rewards.append(nstep_sum)

            # TASK_727: Per-asset cumulative future return over [d+1, d+K]
            # Target for DirectionHead: sign(cum_return) per asset.
            T_ret = len(slot['returns'])
            future_idx = decision_indices[:, None] + np.arange(1, dir_horizon + 1)  # (n, K)
            future_idx = np.minimum(future_idx, T_ret - 1)
            future_rets = slot['returns'][future_idx]   # (n, K, n_assets)
            future_cum = future_rets.sum(axis=1).astype(np.float32)  # (n, n_assets)
            all_future_cum.append(future_cum)

            # TASK_735: Per-asset log-return at each aux horizon h: Σ returns[d+1 : d+h+1]
            # Returns are already (log) per-step returns; cum_return over h steps
            # is the log of total return over that horizon.
            if aux_price_horizons:
                H = max(max_aux_horizon, dir_horizon)
                h_idx = decision_indices[:, None] + np.arange(1, H + 1)   # (n, H)
                h_idx = np.minimum(h_idx, T_ret - 1)
                step_rets = slot['returns'][h_idx]                         # (n, H, n_assets)
                cumret = np.cumsum(step_rets, axis=1)                      # (n, H, n_assets)
                # Pick at horizons [1, 5, 10, 20] → indices [0, 4, 9, 19]
                picks = np.array([h - 1 for h in aux_price_horizons])
                aux_logret = cumret[:, picks, :].astype(np.float32)        # (n, n_horizons, n_assets)
                all_future_log_rets.append(aux_logret)

            # TASK_740: Pull precomputed Bella predictions for this slot.
            # bella_preds_full (loaded in SegmentPool._make_slot) is indexed by
            # the same decision_indices used for the BC target. Sentinel NaN
            # marks windows without a Bella shard; loss masks those samples.
            if use_bella_distill:
                bp = slot.get('bella_preds_full', None)
                if bp is None:
                    bella_batch = np.full(
                        (n, len(bella_pred_horizons), self.n_assets),
                        np.nan, dtype=np.float32)
                else:
                    bella_batch = bp[decision_indices].astype(np.float32)
                all_bella_preds.append(bella_batch)

            # TASK_754: Pull precomputed Bella shape descriptors for this slot.
            # bella_shape_full (loaded in SegmentPool._make_slot) is indexed by
            # the same decision_indices. Sentinel NaN marks windows without a
            # shape shard; loss masks those samples sample-wise.
            if use_bella_shape:
                bsh = slot.get('bella_shape_full', None)
                if bsh is None:
                    shape_batch = np.full(
                        (n, bella_shape_dim), np.nan, dtype=np.float32)
                else:
                    shape_batch = bsh[decision_indices].astype(np.float32)
                all_bella_shape.append(shape_batch)

            # State channels (numpy, if enabled)
            if self.use_state_channels:
                decision_t1 = cursors_arr + ws
                decision_tN = cursors_arr + ws - 1 + nstep
                state_idx_t = np.maximum(decision_indices - 1, 0)
                state_idx_t1 = np.maximum(decision_t1 - 1, 0)
                state_idx_tN = np.maximum(decision_tN - 1, 0)
                all_state_t.append(slot['state_channels'][state_idx_t].astype(np.float32))
                all_conf_t.append(slot['confidence'][decision_indices].astype(np.float32))
                all_state_t1.append(slot['state_channels'][state_idx_t1].astype(np.float32))
                all_conf_t1.append(slot['confidence'][decision_t1].astype(np.float32))
                all_state_tN.append(slot['state_channels'][state_idx_tN].astype(np.float32))
                all_conf_tN.append(slot['confidence'][decision_tN].astype(np.float32))

        # Stack all windows into one flat array for efficient GPU transfer
        # Each window is (win_len, n_assets, 5); concatenate along batch dim
        # Then flatten to (B*win_len, n_assets, 5) for the encoder
        windows_stacked = np.concatenate(all_windows, axis=0)[:batch_size]  # (B, win_len, n_assets, 5)

        return {
            'windows': windows_stacked,   # (B, win_len, n_assets, 5) — compact per-sample
            'actions': np.concatenate(all_actions)[:batch_size],
            'rewards': np.concatenate(all_rewards)[:batch_size],
            'oracle_targets': np.concatenate(all_oracle_targets)[:batch_size],
            'oracle_values': np.concatenate(all_oracle_values)[:batch_size],
            'nstep_rewards': np.concatenate(all_nstep_rewards)[:batch_size],
            'future_cum_returns': np.concatenate(all_future_cum)[:batch_size],  # TASK_727
            'future_log_returns': (np.concatenate(all_future_log_rets)[:batch_size]
                                   if all_future_log_rets else None),           # TASK_735
            'bella_preds': (np.concatenate(all_bella_preds)[:batch_size]
                            if all_bella_preds else None),                       # TASK_740
            'bella_shape': (np.concatenate(all_bella_shape)[:batch_size]
                            if all_bella_shape else None),                       # TASK_754
            'state_t': all_state_t, 'conf_t': all_conf_t,
            'state_t1': all_state_t1, 'conf_t1': all_conf_t1,
            'state_tN': all_state_tN, 'conf_tN': all_conf_tN,
            'batch_size': batch_size,
        }

    def _encode_jax(self, np_batch: dict) -> dict:
        """Phase 2: JAX/GPU encoding. Main thread only.

        Uses flattened windows approach: B windows of (win_len, n_assets, 5) are
        concatenated into (B*win_len, n_assets, 5) with stride offsets. This gives
        3 encoder calls total (obs_t, obs_t1, obs_tN) instead of 3*n_slots.
        """
        batch_size = np_batch['batch_size']
        ws = self.ws
        nstep = self.config.nstep_horizon
        win_len = ws + nstep

        windows = np_batch['windows']  # (B, win_len, n_assets, 5) numpy
        B = len(windows)

        # Flatten: (B, win_len, n_assets, 5) → (B*win_len, n_assets, 5)
        prices_flat = jnp.array(windows.reshape(B * win_len, -1, 5))

        # Window starts: each sample's window is at offset i*win_len
        base_starts = jnp.arange(B) * win_len
        starts_t = base_starts            # obs at cursor (window starts at 0 within each block)
        starts_t1 = base_starts + 1       # obs at cursor+1
        starts_tN = base_starts + nstep   # obs at cursor+N

        # 3 encoder calls total (was 3*n_slots before)
        obs_t = self._encode_fn(prices_flat, starts_t, self.encoder_config)
        obs_t1 = self._encode_fn(prices_flat, starts_t1, self.encoder_config)
        obs_tN = self._encode_fn(prices_flat, starts_tN, self.encoder_config)

        # State channels (if enabled)
        if self.use_state_channels:
            state_t = jnp.array(np.concatenate(np_batch['state_t'])[:batch_size])
            conf_t = jnp.array(np.concatenate(np_batch['conf_t'])[:batch_size])
            obs_t = self._add_state_channels_jax(obs_t, state_t, conf_t)

            state_t1 = jnp.array(np.concatenate(np_batch['state_t1'])[:batch_size])
            conf_t1 = jnp.array(np.concatenate(np_batch['conf_t1'])[:batch_size])
            obs_t1 = self._add_state_channels_jax(obs_t1, state_t1, conf_t1)

            state_tN = jnp.array(np.concatenate(np_batch['state_tN'])[:batch_size])
            conf_tN = jnp.array(np.concatenate(np_batch['conf_tN'])[:batch_size])
            obs_tN = self._add_state_channels_jax(obs_tN, state_tN, conf_tN)

        return {
            'obs_t': obs_t[:batch_size],
            'obs_t1': obs_t1[:batch_size],
            'obs_tN': obs_tN[:batch_size],
            'actions': jnp.array(np_batch['actions']),
            'rewards': jnp.array(np_batch['rewards']),
            'oracle_targets': jnp.array(np_batch['oracle_targets']),
            'oracle_values': jnp.array(np_batch['oracle_values']),
            'nstep_rewards': jnp.array(np_batch['nstep_rewards']),
            'future_cum_returns': jnp.array(np_batch['future_cum_returns']),  # TASK_727
            'future_log_returns': jnp.array(np_batch['future_log_returns']),  # TASK_735
            'bella_preds': (jnp.array(np_batch['bella_preds'])
                            if np_batch.get('bella_preds') is not None else None),  # TASK_740
            'bella_shape': (jnp.array(np_batch['bella_shape'])
                            if np_batch.get('bella_shape') is not None else None),  # TASK_754
        }

    def _add_state_channels_jax(self, obs, state_ch, conf):
        """Add state channels + confidence (already JAX arrays) to obs tensor."""
        ws = self.ws
        # state_ch: (B, 6, n_assets) → (B, 6, ws, n_assets)
        # conf: (B, n_assets) → (B, 1, ws, n_assets)
        state_tiled = jnp.tile(state_ch[:, :, None, :], (1, 1, ws, 1))
        conf_tiled = jnp.tile(conf[:, None, None, :], (1, 1, ws, 1))
        return jnp.concatenate([obs, state_tiled, conf_tiled], axis=1)

    def _worker(self):
        """Background thread: prepare numpy batches, queue them."""
        while not self._stop.is_set():
            try:
                np_batch = self._prepare_numpy(self._batch_size)
                self._queue.put(np_batch, timeout=2.0)
            except queue.Full:
                continue
            except Exception as e:
                self._queue.put(e)
                break

    def get_batch(self, batch_size=None) -> dict:
        """Main thread: dequeue numpy batch, encode to JAX on GPU."""
        item = self._queue.get()
        if isinstance(item, Exception):
            raise item
        return self._encode_jax(item)

    def stop(self):
        """Stop background thread."""
        self._stop.set()
        self._thread.join(timeout=5.0)


def _is_encoder_key(path) -> bool:
    """Check if a param path belongs to the encoder (frozen in Stage B)."""
    path_str = '/'.join(str(p) for p in path)
    # trm/encoder/* is the encoder; everything else in trm is reasoning/policy/value
    return 'encoder' in path_str and 'trm' in path_str


def _is_titans_key(path) -> bool:
    """TASK_749 fix 5: check if a param path is inside the Titans memory subtree.

    Titans params live at `trm/encoder/titans/*` — strict subset of the
    encoder. Used to peel them off into a dedicated optimizer group when
    `--titans-lr` is set (the encoder LR 3e-5 is too low to move the MAG
    gate sigmoid in feasible epoch counts).
    """
    path_str = '/'.join(str(p) for p in path)
    return 'titans' in path_str and 'trm' in path_str and 'encoder' in path_str


def make_stage_b_labels(params, encoder_trainable: bool, titans_separate: bool = False):
    """TASK_728: Three-group labels for multi_transform.

    - 'train'   : reasoning, dynamics, aux heads (policy_direction_refiner, direction_head)
    - 'encoder' : encoder params (only when encoder_trainable=True; else 'freeze')
    - 'freeze'  : oracle_mask (auxiliary, not used in this run path)

    TASK_749 fix 5: when titans_separate=True, params inside trm/encoder/titans
    get the 'titans' label so they can route to a dedicated higher-LR optimizer.
    The check is path-prefix specific so it does NOT alter labels of any other
    leaves; the optimizer dict simply gains a fourth entry. Without this flag
    (titans_separate=False), Titans leaves keep their historical 'encoder' /
    'freeze' label and inherit encoder_lr — preserves byte-identical behavior
    for prior smokes.
    """
    def label_fn(path, _leaf):
        if _is_dynamics_key(path):
            return 'train'
        if titans_separate and _is_titans_key(path):
            return 'titans'
        if _is_encoder_key(path):
            return 'encoder' if encoder_trainable else 'freeze'
        key = str(path[0])
        if 'oracle_mask' in key:
            return 'freeze'
        return 'train'

    return jax.tree_util.tree_map_with_path(label_fn, params)


def create_stage_b_optimizer(config: Phase2Config, params):
    """Create optimizer for Stage B — trains reasoning + dynamics + aux heads.

    TASK_728: Supports encoder unfreeze at a separate slower LR via config.encoder_lr.
    - encoder_lr == 0  → encoder frozen (historical default).
    - encoder_lr > 0   → encoder joins training with its own warmup-cosine schedule.

    Returns (optimizer, opt_state).
    """
    total_steps = config.epochs * config.steps_per_epoch
    warmup_steps = min(500, max(total_steps - 1, 0))

    main_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.policy_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps, warmup_steps + 1),
        end_value=config.policy_lr * 0.1,
    )
    # TASK_749 fix 2 (post-step-7 cleanup): removed `optax.zero_nans()` that
    # was prepended to the Titans-enabled chain in commit 5b65ae5. That was
    # a Band-Aid masking the inf-grad symptom of nested `jax.grad` inside
    # the scan body; the root cause is now fixed in titans_memory.py by
    # replacing the inner `jax.grad` with an analytical closed-form gradient.
    # With the analytical path, `grad_norm` is finite at every step and Adam
    # state stays clean without needing the zero_nans shim.
    train_opt = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(main_schedule, weight_decay=1e-4),
    )

    encoder_trainable = getattr(config, 'encoder_lr', 0.0) > 0.0
    if encoder_trainable:
        enc_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.encoder_lr,
            warmup_steps=warmup_steps,
            decay_steps=max(total_steps, warmup_steps + 1),
            end_value=config.encoder_lr * 0.1,
        )
        encoder_opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(enc_schedule, weight_decay=1e-4),
        )
        logger.info(f"  Encoder UNFROZEN at lr={config.encoder_lr} (TASK_728)")
    else:
        encoder_opt = optax.set_to_zero()

    freeze_opt = optax.set_to_zero()

    # TASK_749 fix 5: dedicated Titans LR group. Only kicks in when Titans is
    # ENABLED and `--titans-lr` was explicitly set. This guarantees that runs
    # without Titans (or Titans-on but no override) keep the historical 3-group
    # optimizer structure — preserves byte-identical behavior and avoids
    # spurious JAX recompiles for legacy configs.
    titans_separate = (
        getattr(config, 'use_titans_memory', False)
        and getattr(config, 'titans_lr', None) is not None
    )

    labels = make_stage_b_labels(params, encoder_trainable,
                                 titans_separate=titans_separate)

    transforms = {
        'train': train_opt,
        'encoder': encoder_opt,
        'freeze': freeze_opt,
    }
    if titans_separate:
        titans_lr = float(config.titans_lr)
        # Warmup-cosine to match encoder/main; the gate sigmoid is sensitive
        # to gradient magnitude, so we keep the same shape — just at a higher
        # peak (paper-typical ~100× the encoder LR for newly-added heads).
        titans_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=titans_lr,
            warmup_steps=warmup_steps,
            decay_steps=max(total_steps, warmup_steps + 1),
            end_value=titans_lr * 0.1,
        )
        transforms['titans'] = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(titans_schedule, weight_decay=1e-4),
        )
        # Sanity log: count of leaves routed into the titans group.
        n_titans = sum(1 for lbl in jax.tree_util.tree_leaves(labels)
                       if lbl == 'titans')
        logger.info(f"  [TITANS-LR] Titans params routed to dedicated group "
                    f"at lr={titans_lr} (leaves={n_titans}, encoder_lr={config.encoder_lr})")

    optimizer = optax.multi_transform(transforms, labels)
    opt_state = optimizer.init(params)
    return optimizer, opt_state


def make_stage_b_train_step(model, config: Phase2Config, optimizer):
    """Build JIT-compiled training step for Stage B (ExIt).

    Takes MCTS outputs (visit_counts, root_value) as inputs — MCTS runs outside JIT.
    PopArt mean/std passed as scalar args (cheap, no recompilation). Raw TD targets
    returned in metrics for PopArt update outside JIT.

    Returns a function:
        (params, opt_state, obs, obs_t1, actions_dynamics, rewards_dynamics,
         mcts_visits, mcts_value, oracle_targets, bc_weight, popart_mean, popart_std,
         oracle_values)
        → (new_params, new_opt_state, metrics_dict)
    """
    policy_w = config.exit_policy_weight
    value_w = config.exit_value_weight
    consistency_w = config.exit_consistency_weight
    config_discount = config.td_discount  # Lower than mcts_discount to keep V in support range
    n_assets = config.n_assets
    n_actions_per_asset = 5
    use_popart = config.use_popart
    use_factored = config.use_factored_search  # TASK_724: per-asset CE vs global CE
    value_target_mode = config.value_target_mode
    # TASK_769: value-clip mode (F2 fix). "pre_popart" = canon (clip pre-norm,
    # censoring PopArt's dynamic range). "post_popart" = feed PopArt the
    # unclipped target so running statistics absorb the scale; only clip the
    # post-normalized target to the categorical support [-10, 10] (those guards
    # remain unconditional below). Captured at JIT-trace time — flag-flips
    # re-JIT once per cell. Default mirrors canon; the pre_popart branch is
    # bit-identical to pre-769 behavior.
    value_clip_mode = getattr(config, 'value_clip_mode', 'pre_popart')
    # TASK_777: mixture value-target alpha. Captured at JIT-trace time so the
    # constant is baked into the compiled HLO; changing α re-JITs the step
    # (same pattern as value_target_mode / value_clip_mode). Ignored unless
    # value_target_mode == "mixture".
    value_mixture_alpha = getattr(config, 'value_mixture_alpha', 0.5)
    value_scale = config.value_scale  # 2.0 — same scaling as Phase 1
    nstep_gamma_N = config.td_discount ** config.nstep_horizon  # γ^N for bootstrap
    use_bc_balance = config.bc_class_balanced  # TASK_726: adaptive trade/hold weighting
    bc_max_tw = config.bc_max_trade_weight     # TASK_726: cap on trade weight boost
    # TASK_781 / TASK_725-D: return-weighted BC. Per-sample multiplier
    # |oracle_values| / mean(|oracle_values|). Default False = bit-identical
    # canon (the bc_rwt guard branch is dead under JIT trace). Captured at
    # JIT-trace time — flag-flips re-JIT once per cell.
    use_bc_rwt = bool(getattr(config, 'bc_return_weighted', False))
    # TASK_763: policy entropy regularization (joint MCTS path only).
    # Defaults 0.0 preserve canon (no bonus, no floor). Coef captured at JIT-trace
    # time — sweep cells re-JIT but that's a one-time cost per cell.
    entropy_coef = float(getattr(config, 'entropy_coef', 0.0))
    min_entropy = float(getattr(config, 'min_entropy', 0.0))
    # TASK_767: action-space restructure. When disable_limit_orders is True the
    # LimitSell (3) / LimitBuy (4) actions are dropped from the Phase-2 action
    # space. Two static arrays are built once at JIT-trace time:
    #   limit_logit_mask  — (320,) additive logit mask; masked entries get a
    #                       large-but-finite negative so softmax probability is
    #                       an exact 0.0 while log_softmax stays finite.
    #   limit_keep_mask   — (320,) float {0,1}; 1 = kept action. Used to zero
    #                       and renormalize the policy / oracle CE targets so
    #                       masked entries carry exactly 0 probability.
    # WHY -1e9 and not jnp.finfo.min: the CE term for a masked index is
    # target[i] * log_softmax(logits)[i]. With finfo.min (~-3.4e38) the
    # subtraction -FLT_MAX - logsumexp rounds to literal -inf, and the oracle
    # BC target is a per-asset softmax that puts a TINY but non-zero probability
    # on limit indices -> tiny * (-inf) = -inf -> CE = +inf (observed in the
    # first 767 smoke). -1e9 keeps log_softmax finite (no overflow) and still
    # drives softmax probability to exactly 0.0. Belt-and-braces, the targets
    # are also masked+renormalized below so the masked CE term is a clean 0.0
    # and the masked logits get zero gradient.
    # Canon (False) leaves both arrays None and every guard branch dead — the
    # flag-off path is bit-identical to pre-767 code.
    disable_limit_orders = bool(getattr(config, 'disable_limit_orders', False))
    if disable_limit_orders:
        _keep_np = np.ones((n_assets, n_actions_per_asset), dtype=np.float32)
        _keep_np[:, 3] = 0.0   # LimitSell
        _keep_np[:, 4] = 0.0   # LimitBuy
        limit_keep_mask = jnp.asarray(_keep_np.reshape(-1))            # (320,)
        _logit_np = np.where(_keep_np == 0.0, -1e9, 0.0).astype(np.float32)
        limit_logit_mask = jnp.asarray(_logit_np.reshape(-1))         # (320,)
    else:
        limit_keep_mask = None
        limit_logit_mask = None

    def _mask_renorm_target(target_320):
        """TASK_767: zero the limit-action entries of a per-asset CE target and
        renormalize each asset's 5-block to sum to 1 over the kept actions.
        target_320 is (B, 320). No-op semantics rely on the caller only
        invoking this when limit_keep_mask is not None."""
        t = target_320 * limit_keep_mask                              # (B, 320)
        tr = t.reshape(-1, n_assets, n_actions_per_asset)
        grp = jnp.sum(tr, axis=-1, keepdims=True)                     # (B, 64, 1)
        tr = tr / jnp.maximum(grp, 1e-8)
        return tr.reshape(-1, n_assets * n_actions_per_asset)
    use_direction = config.use_direction_head  # TASK_727: aux direction head
    direction_w = config.direction_loss_weight # TASK_727: direction loss weight
    use_rank = getattr(config, 'use_rank_head', False)             # TASK_730: rank head
    rank_w = getattr(config, 'rank_loss_weight', 1.0)              # TASK_730: rank loss weight
    rank_loss_type = getattr(config, 'rank_loss_type', 'mse')      # TASK_730: rank loss form
    use_flow = getattr(config, 'use_flow_matching', False)         # TASK_734: flow-matching on dynamics
    flow_v_w = getattr(config, 'flow_matching_velocity_weight', 0.8)
    use_aux_price = getattr(config, 'use_aux_price_head', False)   # TASK_735: aux log-return head
    aux_price_w = getattr(config, 'aux_price_loss_weight', 0.1)
    aux_price_horizons_t = tuple(getattr(config, 'aux_price_horizons', (1, 5, 10, 20)))
    use_bella_distill = getattr(config, 'use_bella_distill_head', False)   # TASK_740
    bella_distill_w = getattr(config, 'bella_distill_loss_weight', 0.1)
    bella_pred_horizons_t = tuple(getattr(config, 'bella_pred_horizons', (1, 5, 10, 20)))
    # TASK_754: trajectory-shape distill closure config
    use_bella_shape = getattr(config, 'use_bella_shape_head', False)
    bella_shape_w = getattr(config, 'bella_shape_loss_weight', 0.1)
    bella_shape_dim = int(getattr(config, 'bella_shape_descriptor_dim', 14))
    # Expand the 6-tuple (mean, std, skew, kurt, s1, s1_ratio) into a 14-vector
    # aligned with SHAPE_DESCRIPTOR_DIM layout:
    #   0-3 cross moments, 4-7 h1 moments, 8-11 hlast moments, 12 s1, 13 s1_ratio.
    _default_w6 = (2.0, 1.0, 0.5, 0.25, 1.0, 2.0)
    _w6 = getattr(config, 'bella_shape_component_weights', None) or _default_w6
    if len(_w6) != 6:
        raise ValueError(
            f"bella_shape_component_weights must be 6 floats (mean, std, skew, "
            f"kurt, s1, s1_ratio); got {len(_w6)}: {_w6}"
        )
    w_mean, w_std, w_skew, w_kurt, w_s1, w_s1r = _w6
    _shape_w_vec_np = np.array([
        w_mean, w_std, w_skew, w_kurt,  # cross
        w_mean, w_std, w_skew, w_kurt,  # h1
        w_mean, w_std, w_skew, w_kurt,  # hlast
        w_s1, w_s1r,                    # svd
    ], dtype=np.float32)
    assert _shape_w_vec_np.shape[0] == bella_shape_dim, (
        f"bella_shape weight vector len {_shape_w_vec_np.shape[0]} != "
        f"descriptor_dim {bella_shape_dim}"
    )
    shape_w_vec = jnp.asarray(_shape_w_vec_np)

    @jax.jit
    def stage_b_train_step(params, opt_state, obs, obs_t1, obs_tN, actions_dynamics,
                           rewards_dynamics, mcts_visits, mcts_value,
                           oracle_targets, bc_weight, popart_mean, popart_std,
                           oracle_values, nstep_rewards, future_cum_returns,
                           future_log_returns, bella_preds, bella_shape):
        """Single Stage B training step.

        Args:
            params: Full model params
            opt_state: Optimizer state
            obs: (B, C, W, N) observations for policy/value (same as obs_t)
            obs_t1: (B, C, W, N) next observations for dynamics consistency
            obs_tN: (B, C, W, N) observations at t+N for n-step value bootstrap
            actions_dynamics: (B,) actions for dynamics transition
            rewards_dynamics: (B,) scaled rewards for dynamics reward loss
            mcts_visits: (B, 320) MCTS visit count distribution (policy target)
            mcts_value: (B,) MCTS root value (value target, scalar)
            oracle_targets: (B, 320) oracle policy targets (per-asset softmaxed, sums to 64)
            bc_weight: scalar, decaying BC weight
            popart_mean: scalar, PopArt running mean (0.0 if disabled)
            popart_std: scalar, PopArt running std (1.0 if disabled)
            oracle_values: (B,) oracle remaining PnL (TASK_719a, value_target_mode="oracle")
            nstep_rewards: (B,) discounted N-step reward sum (TASK_719b, value_target_mode="nstep")
            future_cum_returns: (B, n_assets) per-asset cum return over direction_horizon (TASK_727)
            bella_preds: (B, K, n_assets) precomputed Bella log-returns (TASK_740); may be all-NaN
            bella_shape: (B, descriptor_dim) precomputed trajectory-shape descriptor (TASK_754); may contain NaN rows
        """
        def loss_fn(params):
            # === Root inference: policy + value from current obs ===
            root_out = model.apply({'params': params}, observation=obs, mode='initial_inference')
            policy_logits = root_out['policy_logits']  # (B, 320)
            value_logits = root_out['value']            # (B, 401)
            latent_t = root_out['latent_state']         # (B, 128)

            # TASK_767: mask the LimitSell/LimitBuy logits to -inf before any
            # softmax/CE. This zeroes their probability and gradient in the
            # policy CE, the BC CE and the entropy term alike. Canon path
            # (limit_logit_mask is None) skips this entirely — policy_logits is
            # untouched, every downstream op bit-identical to pre-767 code.
            if limit_logit_mask is not None:
                policy_logits = policy_logits + limit_logit_mask  # (B, 320)

            # TASK_767: zero the limit-action probability in the CE *targets*.
            # Left unmasked, target[limit] * log_softmax(masked_logits)[limit]
            # is a non-zero probability times a hugely-negative log-prob, which
            # dominates (or, with finfo.min, infinitises) the CE.
            #  - oracle_targets is ALWAYS a per-asset softmax (each 5-block sums
            #    to 1) and puts a small non-zero mass on limit indices ->
            #    mask + per-asset renormalize.
            #  - mcts_visits is per-asset under factored search (renormalize)
            #    but a single GLOBAL distribution under joint MCTS, where mctx's
            #    invalid_actions already drives the limit weights to exactly 0
            #    -> only apply the {0,1} keep mask, never per-asset renormalize
            #    (that would corrupt the global distribution).
            # New local names (not a rebind of the closure params) so this is a
            # pure addition. Canon path keeps mcts_visits_m / oracle_targets_m
            # pointing straight at the originals — bit-identical to pre-767.
            if limit_keep_mask is not None:
                oracle_targets_m = _mask_renorm_target(oracle_targets)
                if use_factored:
                    mcts_visits_m = _mask_renorm_target(mcts_visits)
                else:
                    mcts_visits_m = mcts_visits * limit_keep_mask
            else:
                mcts_visits_m = mcts_visits
                oracle_targets_m = oracle_targets

            # === Policy loss: distill search targets into single-step policy ===
            if use_factored:
                # TASK_724: Per-asset CE — each asset's 5 actions are independent
                # Targets are per-asset softmax (each group of 5 sums to 1.0)
                pol_r = policy_logits.reshape(-1, n_assets, n_actions_per_asset)
                tgt_r = mcts_visits_m.reshape(-1, n_assets, n_actions_per_asset)
                per_asset_ce_policy = optax.softmax_cross_entropy(pol_r, tgt_r)  # (B, 64)
                if use_bc_balance:
                    # TASK_726 Change 2: factored policy targets also skew HOLD
                    # (flat Q → ~20% per action with slight Hold edge). Weight by
                    # per-asset trade strength so BUY/SELL targets drive gradient.
                    hold_prob_pol = tgt_r[:, :, 1]            # (B, 64)
                    trade_strength_pol = 1.0 - hold_prob_pol
                    n_t = jnp.maximum(jnp.sum(trade_strength_pol), 1.0)
                    n_h = jnp.maximum(jnp.sum(hold_prob_pol), 1.0)
                    tw_pol = jnp.minimum(n_h / n_t, bc_max_tw)
                    policy_weights = 1.0 + trade_strength_pol * (tw_pol - 1.0)
                    policy_loss = (jnp.sum(per_asset_ce_policy * policy_weights) /
                                   jnp.sum(policy_weights))
                else:
                    policy_loss = per_asset_ce_policy.mean()
            else:
                # Joint MCTS: visit_counts sum to 1.0 globally over 320 actions
                policy_loss = optax.softmax_cross_entropy(policy_logits, mcts_visits_m).mean()

            # TASK_763 + TASK_776: policy entropy regularization. Joint and
            # factored paths use different semantics:
            #   - Joint MCTS (320-way global softmax): one Shannon entropy
            #     computed over the joint distribution; max ≈ ln(320) ≈ 5.77.
            #   - Factored per-asset (TASK_724/776): per-asset softmax over 5
            #     actions, Shannon entropy computed per asset and averaged over
            #     the 64-asset axis; max ≈ ln(5) ≈ 1.61. min_entropy floor is
            #     interpreted in the same units (per-asset for factored, joint
            #     for joint) — operator must scale min_entropy when sweeping
            #     across paths.
            # Canon (entropy_coef == 0.0) skips the entire branch → loss is
            # bit-identical to pre-776 code in both paths.
            if entropy_coef > 0.0:
                if use_factored:
                    # TASK_776: per-asset Shannon entropy. Reshape (B, 320) →
                    # (B, 64, 5), per-asset softmax, per-asset entropy, then
                    # mean over (B, 64) yielding a scalar with units of nats
                    # per asset (max ln(5) ≈ 1.61).
                    pol_r = policy_logits.reshape(-1, n_assets, n_actions_per_asset)
                    per_asset_probs = jax.nn.softmax(pol_r, axis=-1)
                    per_asset_entropy = -jnp.sum(
                        per_asset_probs * jnp.log(per_asset_probs + 1e-10),
                        axis=-1,
                    )  # (B, 64)
                    policy_entropy = per_asset_entropy.mean()
                else:
                    policy_probs = jax.nn.softmax(policy_logits, axis=-1)
                    policy_entropy = -jnp.sum(
                        policy_probs * jnp.log(policy_probs + 1e-10), axis=-1
                    ).mean()
                entropy_bonus = entropy_coef * policy_entropy
                # 5× stronger push when below floor (mirrors trainer.py:1244)
                floor_gap = jnp.maximum(min_entropy - policy_entropy, 0.0)
                floor_penalty = entropy_coef * 5.0 * floor_gap
                # Net adjustment: subtract bonus (encourage exploration), add penalty
                policy_loss = policy_loss - entropy_bonus + floor_penalty
            else:
                policy_entropy = jnp.array(0.0)

            # === Shared: encode obs_t1 with stopped params (no gradients to encoder/reasoning) ===
            stopped_params = jax.lax.stop_gradient(params)
            out_t1 = model.apply({'params': stopped_params},
                                 observation=obs_t1, mode='initial_inference')

            # === Value loss ===
            if value_target_mode == "oracle":
                # TASK_719a: Oracle remaining PnL as value target (same as Phase 1)
                oracle_scaled = oracle_values * value_scale
                # TASK_769: F2 fix — gate the pre-PopArt clip behind value_clip_mode.
                # pre_popart (canon) keeps the historical behavior bit-identical:
                # the clipped value is what feeds BOTH PopArt normalization AND
                # td_target_raw. post_popart feeds PopArt the unclipped value so
                # its running statistics see the true distribution.
                if value_clip_mode == "pre_popart":
                    oracle_scaled = jnp.clip(oracle_scaled, -10.0, 10.0)
                td_target_norm = (oracle_scaled - popart_mean) / jnp.maximum(popart_std, 1e-8)
                td_target_norm = jnp.clip(td_target_norm, -10.0, 10.0)
                td_target_raw = oracle_scaled
            elif value_target_mode == "nstep":
                # TASK_719b: N-step TD — Σγ^k r_k + γ^N V(s_{t+N})
                # nstep_rewards = pre-computed discounted reward sum (from generator)
                # Bootstrap with V(s_{t+N}) from stopped params (reuse from dynamics)
                out_tN = model.apply({'params': stopped_params},
                                     observation=obs_tN, mode='initial_inference')
                support = get_value_support()
                v_tN = jnp.sum(jax.nn.softmax(out_tN['value'], axis=-1) * support, axis=-1)
                # Denormalize V(s_{t+N}) if PopArt active (value head outputs normalized)
                v_tN_raw = v_tN * popart_std + popart_mean
                td_target_raw = nstep_rewards + nstep_gamma_N * v_tN_raw
                # TASK_769: F2 fix — gate the pre-PopArt clip. See oracle branch
                # above for rationale.
                if value_clip_mode == "pre_popart":
                    td_target_raw = jnp.clip(td_target_raw, -10.0, 10.0)
                td_target_norm = (td_target_raw - popart_mean) / jnp.maximum(popart_std, 1e-8)
                td_target_norm = jnp.clip(td_target_norm, -10.0, 10.0)
            elif value_target_mode == "mixture":
                # TASK_777: convex blend of oracle and nstep targets in the
                # pre-PopArt "raw" space:
                #     td_target_raw = α·oracle_scaled + (1−α)·nstep_target_raw
                #
                # Bit-identity smoke-gate construction (see TASK_777 §Smoke):
                #     α=1.0  → td_target_raw == oracle branch's td_target_raw
                #     α=0.0  → td_target_raw == nstep branch's td_target_raw
                # Both legs below MUST mirror their dedicated branches exactly
                # (same clip-gating, same v_tN denorm) — if either drifts, the
                # bit-identity gate fails.
                #
                # Hypothesis: 746 oracle's paired Δα +0.72pp (5-seed, STD
                # 2.28pp) is variance-limited; blending with low-variance nstep
                # should reduce per-seed STD and lift mean. See
                # tasks/TASK_777_MIXTURE_VALUE_TARGET.md §Hypothesis.
                #
                # Oracle leg — mirrors the "oracle" branch above.
                oracle_scaled = oracle_values * value_scale
                if value_clip_mode == "pre_popart":
                    oracle_scaled = jnp.clip(oracle_scaled, -10.0, 10.0)
                # Nstep leg — mirrors the "nstep" branch above (re-running
                # `out_tN` here is unavoidable: this branch is reachable
                # independently of the nstep branch).
                out_tN = model.apply({'params': stopped_params},
                                     observation=obs_tN, mode='initial_inference')
                support = get_value_support()
                v_tN = jnp.sum(jax.nn.softmax(out_tN['value'], axis=-1) * support, axis=-1)
                v_tN_raw = v_tN * popart_std + popart_mean
                nstep_target_raw = nstep_rewards + nstep_gamma_N * v_tN_raw
                if value_clip_mode == "pre_popart":
                    nstep_target_raw = jnp.clip(nstep_target_raw, -10.0, 10.0)
                # Convex blend — α is captured at JIT-trace time (Python float
                # constant), folded into the HLO; α∈{0.0, 1.0} collapse to a
                # single leg, which is the bit-identity smoke gate.
                td_target_raw = (
                    value_mixture_alpha * oracle_scaled
                    + (1.0 - value_mixture_alpha) * nstep_target_raw
                )
                td_target_norm = (td_target_raw - popart_mean) / jnp.maximum(popart_std, 1e-8)
                td_target_norm = jnp.clip(td_target_norm, -10.0, 10.0)
            else:
                # TD(0) bootstrap — r + γ * V(s_next)
                r_step = rewards_dynamics
                support = get_value_support()
                v_next = jnp.sum(jax.nn.softmax(out_t1['value'], axis=-1) * support, axis=-1)
                td_target_raw = r_step + config_discount * v_next
                # TASK_769: F2 fix — gate the pre-PopArt clip. td0 is not a 769
                # cell (oracle and nstep are), but we gate it for consistency so
                # a future td0 run can't silently keep the bug.
                if value_clip_mode == "pre_popart":
                    td_target_raw = jnp.clip(td_target_raw, -10.0, 10.0)
                td_target_norm = (td_target_raw - popart_mean) / jnp.maximum(popart_std, 1e-8)
                td_target_norm = jnp.clip(td_target_norm, -10.0, 10.0)

            value_target = scalar_to_categorical(td_target_norm)
            value_loss = optax.softmax_cross_entropy(value_logits, value_target).mean()

            # === Dynamics consistency loss (ongoing calibration) ===
            latent_t_sg = jax.lax.stop_gradient(latent_t)
            latent_t1 = jax.lax.stop_gradient(out_t1['latent_state'])
            dyn_out = model.apply({'params': params}, latent_t_sg, actions_dynamics,
                                  mode='recurrent_inference')
            if use_flow:
                # TASK_734: Velocity-matching — predict latent_{t+1} - latent_t
                # instead of latent_{t+1}. Identity solution gets high loss when
                # the latent actually moves (unlike MSE which collapses).
                pred_velocity = dyn_out['latent_state'] - latent_t_sg
                actual_velocity = latent_t1 - latent_t_sg
                velocity_loss = jnp.mean((pred_velocity - actual_velocity) ** 2)
                state_loss = jnp.mean((dyn_out['latent_state'] - latent_t1) ** 2)
                consistency_loss = flow_v_w * velocity_loss + (1.0 - flow_v_w) * state_loss
                velocity_pred_norm = jnp.mean(jnp.linalg.norm(pred_velocity, axis=-1))
                velocity_actual_norm = jnp.mean(jnp.linalg.norm(actual_velocity, axis=-1))
            else:
                consistency_loss = jnp.mean((dyn_out['latent_state'] - latent_t1) ** 2)
                velocity_loss = jnp.array(0.0)
                state_loss = consistency_loss
                velocity_pred_norm = jnp.array(0.0)
                velocity_actual_norm = jnp.array(0.0)

            # Reward prediction loss (dynamics)
            reward_target = scalar_to_categorical(rewards_dynamics)
            reward_loss = optax.softmax_cross_entropy(dyn_out['reward'], reward_target).mean()

            # === Oracle BC regularizer (decaying, per-asset CE matching Phase 1) ===
            # oracle_targets are per-asset softmaxed (each group of 5 sums to 1.0)
            # oracle_targets_m == oracle_targets on the canon path; under
            # TASK_767 it has the limit actions masked + renormalized.
            policy_reshaped = policy_logits.reshape(-1, n_assets, n_actions_per_asset)
            oracle_reshaped = oracle_targets_m.reshape(-1, n_assets, n_actions_per_asset)
            per_asset_ce = optax.softmax_cross_entropy(policy_reshaped, oracle_reshaped)  # (B, 64)
            if use_bc_balance:
                # TASK_726 Change 1: oracle targets are 93.75% HOLD (k=4/64). Without
                # per-asset trade/hold weighting, the BC gradient fits the HOLD marginal
                # and never learns the per-asset BUY/SELL conditional. Mirrors Phase 1
                # (easy_trainer.py:170-202) minus the return-magnitude boost (deferred).
                hold_prob = oracle_reshaped[:, :, 1]          # (B, 64)
                trade_strength = 1.0 - hold_prob              # (B, 64)
                n_trades = jnp.maximum(jnp.sum(trade_strength), 1.0)
                n_holds = jnp.maximum(jnp.sum(hold_prob), 1.0)
                trade_weight = jnp.minimum(n_holds / n_trades, bc_max_tw)
                bc_weights = 1.0 + trade_strength * (trade_weight - 1.0)  # (B, 64)
                # TASK_781 / TASK_725-D: per-sample return-magnitude weighting.
                # Multiplies the per-asset class-balance weights `bc_weights`
                # by `|oracle_values| / mean(|oracle_values|)` (broadcast over
                # the 64-asset axis). `oracle_values` is the oracle remaining
                # PnL over `oracle_value_horizon` steps, already in the batch
                # dict (precomputed at train_phase2.py:640-644 from
                # `oracle_cum_pnl`). Mean-normalization keeps the total
                # gradient magnitude comparable to the canon path. Default
                # `use_bc_rwt=False` runs the original TASK_726 expression —
                # bit-identical to pre-781 BC.
                if use_bc_rwt:
                    _abs_val = jnp.abs(oracle_values)                          # (B,)
                    _mean_abs = jnp.maximum(jnp.mean(_abs_val), 1e-8)          # scalar
                    sample_w = (_abs_val / _mean_abs)[:, None]                 # (B, 1) → (B, 64) bcast
                    combined_w = bc_weights * sample_w
                    bc_loss = jnp.sum(per_asset_ce * combined_w) / jnp.maximum(jnp.sum(combined_w), 1e-8)
                else:
                    bc_loss = jnp.sum(per_asset_ce * bc_weights) / jnp.sum(bc_weights)
            else:
                # TASK_781: when class-balance is OFF, return-weighting still applies.
                # `use_bc_rwt=False` falls through to the original `.mean()` path —
                # bit-identical to pre-781.
                if use_bc_rwt:
                    _abs_val = jnp.abs(oracle_values)
                    _mean_abs = jnp.maximum(jnp.mean(_abs_val), 1e-8)
                    sample_w = (_abs_val / _mean_abs)[:, None]                 # (B, 1)
                    bc_loss = jnp.sum(per_asset_ce * sample_w) / jnp.maximum(
                        jnp.sum(sample_w) * float(n_assets), 1e-8
                    )
                else:
                    bc_loss = per_asset_ce.mean()

            # === Direction head: per-asset sign-of-future-return (TASK_727) ===
            if use_direction and 'direction_logits' in root_out:
                dir_logits = root_out['direction_logits']              # (B, n_assets)
                dir_target = (future_cum_returns > 0).astype(jnp.float32)
                # Dead-zone: don't supervise near-zero returns (label noise)
                dead_zone = (jnp.abs(future_cum_returns) > 1e-4).astype(jnp.float32)
                dir_ce = optax.sigmoid_binary_cross_entropy(dir_logits, dir_target)
                dir_loss = jnp.sum(dir_ce * dead_zone) / jnp.maximum(jnp.sum(dead_zone), 1.0)
                dir_pred = (jax.nn.sigmoid(dir_logits) > 0.5).astype(jnp.float32)
                dir_correct = ((dir_pred > 0.5) == (dir_target > 0.5)).astype(jnp.float32)
                dir_acc = (jnp.sum(dir_correct * dead_zone) /
                           jnp.maximum(jnp.sum(dead_zone), 1.0))
            else:
                dir_loss = jnp.array(0.0)
                dir_acc = jnp.array(0.5)

            # === Rank head: per-asset cross-sectional rank score (TASK_730) ===
            # Target is argsort-of-argsort(future_cum_returns) / (n-1) per sample,
            # values in [0, 1]. MSE regression is the simplest unambiguous
            # objective — listwise/pairwise held as stretch goals.
            if use_rank and 'rank_scores' in root_out:
                rank_scores = root_out['rank_scores']                     # (B, n_assets)
                target_ranks = (jnp.argsort(
                    jnp.argsort(future_cum_returns, axis=-1), axis=-1
                ).astype(jnp.float32) / jnp.maximum(float(n_assets - 1), 1.0))
                rank_loss = jnp.mean((rank_scores - target_ranks) ** 2)

                # Spearman = Pearson between rank(scores) and rank(targets).
                # target_ranks is already a rank in [0,1]; build the same for scores.
                score_ranks = (jnp.argsort(
                    jnp.argsort(rank_scores, axis=-1), axis=-1
                ).astype(jnp.float32) / jnp.maximum(float(n_assets - 1), 1.0))
                sx = score_ranks - jnp.mean(score_ranks, axis=-1, keepdims=True)
                sy = target_ranks - jnp.mean(target_ranks, axis=-1, keepdims=True)
                num = jnp.sum(sx * sy, axis=-1)
                den = jnp.sqrt(jnp.sum(sx ** 2, axis=-1) *
                               jnp.sum(sy ** 2, axis=-1) + 1e-8)
                rank_spearman = jnp.mean(num / den)
            else:
                rank_loss = jnp.array(0.0)
                rank_spearman = jnp.array(0.0)

            # === Aux log-return prediction head (TASK_735) ===
            # Per-horizon MSE between predicted and actual cumulative log-returns
            # at horizons [1, 5, 10, 20]. Short horizons weighted more since they
            # are more predictable; long horizons are noisier.
            if use_aux_price and 'aux_log_returns' in root_out:
                pred_logret = root_out['aux_log_returns']       # (B, K, n_assets)
                tgt_logret = future_log_returns                 # (B, K, n_assets)
                per_h_mse = jnp.mean((pred_logret - tgt_logret) ** 2, axis=(0, 2))  # (K,)
                h_weights = jnp.array([1.0 / h for h in aux_price_horizons_t])
                h_weights = h_weights / jnp.sum(h_weights)
                aux_price_loss = jnp.sum(h_weights * per_h_mse)
            else:
                aux_price_loss = jnp.array(0.0)
                per_h_mse = jnp.zeros(len(aux_price_horizons_t))

            # === Bella-distillation head (TASK_740) ===
            # bella_preds is (B, K, n_assets) of Bella-generated log-returns.
            # Per-sample NaN marks windows without a Bella shard — mask those
            # out sample-wise so periods missing from the manifest do not
            # perturb the gradient.
            if use_bella_distill and 'aux_bella_returns' in root_out:
                pred_bella = root_out['aux_bella_returns']      # (B, K, n_assets)
                tgt_bella = bella_preds                          # (B, K, n_assets)
                valid = jnp.isfinite(tgt_bella)
                tgt_safe = jnp.where(valid, tgt_bella, 0.0)
                sq = (pred_bella - tgt_safe) ** 2
                valid_f = valid.astype(jnp.float32)
                per_h_bella = (jnp.sum(sq * valid_f, axis=(0, 2)) /
                               jnp.maximum(jnp.sum(valid_f, axis=(0, 2)), 1.0))  # (K,)
                h_weights_b = jnp.array([1.0 / h for h in bella_pred_horizons_t])
                h_weights_b = h_weights_b / jnp.sum(h_weights_b)
                bella_distill_loss = jnp.sum(h_weights_b * per_h_bella)
                bella_valid_frac = jnp.mean(valid_f)
            else:
                bella_distill_loss = jnp.array(0.0)
                per_h_bella = jnp.zeros(len(bella_pred_horizons_t))
                bella_valid_frac = jnp.array(0.0)

            # === Trajectory-shape distillation head (TASK_754) ===
            # bella_shape: (B, D) coarse Bella trajectory-shape descriptor.
            # Sample-wise NaN row marks windows without a shape shard; mask out.
            # Per-component-weighted MSE (shape_w_vec broadcasts over D).
            if use_bella_shape and 'aux_bella_shape' in root_out:
                pred_shape = root_out['aux_bella_shape']        # (B, D)
                tgt_shape = bella_shape                          # (B, D)
                # Sample is "valid" if all D components are finite — keeps the
                # per-row mask trivially compatible with the head's MSE.
                row_valid = jnp.all(jnp.isfinite(tgt_shape), axis=-1)    # (B,)
                tgt_safe = jnp.where(jnp.isfinite(tgt_shape), tgt_shape, 0.0)
                sq = (pred_shape - tgt_safe) ** 2                # (B, D)
                weighted_sq = sq * shape_w_vec[None, :]          # (B, D)
                # Per-sample weighted mean over D, then reduce by valid mask.
                w_sum = jnp.sum(shape_w_vec)
                per_sample_loss = jnp.sum(weighted_sq, axis=-1) / jnp.maximum(w_sum, 1e-8)  # (B,)
                row_valid_f = row_valid.astype(jnp.float32)
                bella_shape_loss = (jnp.sum(per_sample_loss * row_valid_f) /
                                    jnp.maximum(jnp.sum(row_valid_f), 1.0))
                bella_shape_valid_frac = jnp.mean(row_valid_f)
                # Per-component residual diagnostics (first two only — mean +
                # std of the cross-moment group, the two most regime-discriminative
                # components per spec §4.1).
                per_comp_resid = (jnp.sum(jnp.abs(pred_shape - tgt_safe) *
                                           row_valid_f[:, None], axis=0) /
                                  jnp.maximum(jnp.sum(row_valid_f), 1.0))   # (D,)
                bella_shape_mean_resid = per_comp_resid[0]   # cross-mean residual
                bella_shape_std_resid = per_comp_resid[1]    # cross-std residual
            else:
                bella_shape_loss = jnp.array(0.0)
                bella_shape_valid_frac = jnp.array(0.0)
                bella_shape_mean_resid = jnp.array(0.0)
                bella_shape_std_resid = jnp.array(0.0)

            # === TASK_767: action-distribution diagnostics ===
            # mcts_visits is (B, 320). Reshape to (B, 64, 5) and sum over the
            # 64-asset axis to get the action-type marginal (idx 1 = Hold,
            # idx {3,4} = LimitSell/LimitBuy), then normalize by total mass so
            # the metric is mode-agnostic:
            #   joint MCTS — mcts_visits sums to 1.0 globally  -> total = 1.
            #   factored   — each per-asset group sums to 1.0  -> total = 64.
            # Dividing by the row total turns both into a proper action-type
            # probability that sums to 1.
            #   mean_trade_rate       = probability NOT on Hold.
            #   limit_action_fraction = probability on the two limit actions;
            #     the 767 mechanism-premise check (must collapse to ~0 when
            #     disable_limit_orders is on; reveals the baseline's limit
            #     bleed when off).
            # Aux metrics only — never enter total_loss, so they add zero
            # gradient and the canon training step is unchanged. Computed in
            # the existing JIT step, so no extra device sync.
            # NB: measured on the RAW search output mcts_visits (not the
            # target-masked mcts_visits_m) so limit_action_fraction reports
            # what the search genuinely produced — for the 767 variant the
            # factored/joint search itself drives it to ~0; for canon it
            # exposes the real limit bleed.
            _vc = mcts_visits.reshape(-1, n_assets, n_actions_per_asset)
            _act_marginal = jnp.sum(_vc, axis=1)                       # (B, 5)
            _act_total = jnp.maximum(
                jnp.sum(_act_marginal, axis=-1, keepdims=True), 1e-8
            )
            _act_frac = _act_marginal / _act_total                     # (B, 5), sums to 1
            mean_trade_rate = jnp.mean(1.0 - _act_frac[:, 1])
            limit_action_fraction = jnp.mean(
                _act_frac[:, 3] + _act_frac[:, 4]
            )

            # === Total ===
            total_loss = (policy_w * policy_loss +
                          value_w * value_loss +
                          consistency_w * (consistency_loss + reward_loss) +
                          bc_weight * bc_loss +
                          direction_w * dir_loss +
                          rank_w * rank_loss +
                          aux_price_w * aux_price_loss +
                          bella_distill_w * bella_distill_loss +
                          bella_shape_w * bella_shape_loss)

            return total_loss, {
                'total_loss': total_loss,
                'policy_loss': policy_loss,
                'policy_entropy': policy_entropy,  # TASK_763
                'value_loss': value_loss,
                'consistency_loss': consistency_loss,
                'reward_loss': reward_loss,
                'bc_loss': bc_loss,
                'bc_weight': bc_weight,
                'dir_loss': dir_loss,
                'dir_acc': dir_acc,
                'rank_loss': rank_loss,
                'rank_spearman': rank_spearman,
                'flow_velocity_loss': velocity_loss,
                'flow_state_loss': state_loss,
                'flow_v_pred_norm': velocity_pred_norm,
                'flow_v_actual_norm': velocity_actual_norm,
                'aux_price_loss': aux_price_loss,
                'aux_price_mse_h1': per_h_mse[0] if len(per_h_mse) > 0 else jnp.array(0.0),
                'aux_price_mse_hmax': per_h_mse[-1] if len(per_h_mse) > 0 else jnp.array(0.0),
                'bella_distill_loss': bella_distill_loss,
                'bella_distill_h1': per_h_bella[0] if len(per_h_bella) > 0 else jnp.array(0.0),
                'bella_distill_hmax': per_h_bella[-1] if len(per_h_bella) > 0 else jnp.array(0.0),
                'bella_valid_frac': bella_valid_frac,
                # TASK_754: trajectory-shape distill diagnostics
                'bella_shape_loss': bella_shape_loss,
                'bella_shape_valid_frac': bella_shape_valid_frac,
                'bella_shape_mean_resid': bella_shape_mean_resid,
                'bella_shape_std_resid': bella_shape_std_resid,
                'pred_latent_std': jnp.std(dyn_out['latent_state']),
                'target_latent_std': jnp.std(latent_t1),
                # TASK_767: action-distribution diagnostics
                'mean_trade_rate': mean_trade_rate,
                'limit_action_fraction': limit_action_fraction,
                'td_target_mean': jnp.mean(td_target_raw),
                'td_target_std': jnp.std(td_target_raw),
                'td_targets_raw': td_target_raw,  # (B,) for PopArt update outside JIT
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        metrics['grad_norm'] = optax.global_norm(grads)
        return new_params, new_opt_state, metrics

    return stage_b_train_step


def save_checkpoint(params, config, epoch, metrics, output_dir, name,
                    popart_state=None):
    """Save Phase 2 checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        'params': params,
        'config': {k: v for k, v in config.__dict__.items()
                   if not k.startswith('_') and isinstance(v, (int, float, str, bool, type(None)))},
        'epoch': epoch,
        'metrics': metrics,
        'phase': 2,
        'stage': config.stage,
    }
    if popart_state is not None:
        ckpt['popart_state'] = {
            'mean': float(popart_state.mean),
            'var': float(popart_state.var),
            'count': int(popart_state.count),
        }
    path = os.path.join(output_dir, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(ckpt, f)
    return path


def train_stage_a(model, params, config: Phase2Config, pool, args):
    """Stage A: Dynamics warmup training loop.

    Freezes encoder/policy/value, trains dynamics only on consistency + reward loss.
    """
    logger.info(f"=== Stage A: Dynamics Warmup ===")
    logger.info(f"  Epochs: {config.epochs}, Steps/epoch: {config.steps_per_epoch}, "
                f"Batch: {config.batch_size}")
    logger.info(f"  LR: {config.dynamics_lr}, consistency_w: {config.consistency_weight}, "
                f"reward_w: {config.reward_weight}")

    # Initialize dynamics (replace zeros with He normal)
    rng_key = jax.random.PRNGKey(config.seed)
    params = initialize_dynamics(params, rng_key)

    # Create masked optimizer (only dynamics params get updates)
    optimizer, opt_state = create_dynamics_optimizer(config, params)

    # Build JIT-compiled train step (captures optimizer in closure)
    train_step_fn = make_dynamics_train_step(model, config, optimizer)

    # Create async batch pipeline (TASK_721: numpy on background thread, JAX on main)
    pipeline = AsyncBatchPipeline(pool, config, prefetch=2)

    # Output dir
    output_dir = args.output_dir or config.phase2_checkpoint_dir
    best_consistency = float('inf')
    best_epoch = -1

    logger.info("Starting dynamics training...")
    for epoch in range(config.epochs):
        epoch_metrics = defaultdict(float)
        epoch_start = time.time()

        for step in range(config.steps_per_epoch):
            batch = pipeline.get_batch()

            params, opt_state, metrics = train_step_fn(
                params, opt_state,
                batch['obs_t'], batch['obs_t1'],
                batch['actions'], batch['rewards'],
            )

            for k, v in metrics.items():
                epoch_metrics[k] += float(v)

        # Average metrics
        n_steps = config.steps_per_epoch
        avg = {k: v / n_steps for k, v in epoch_metrics.items()}
        elapsed = time.time() - epoch_start

        # Log
        logger.info(
            f"E{epoch:4d}: consistency={avg['consistency_loss']:.4f} "
            f"reward={avg['reward_loss']:.4f} total={avg['total_loss']:.4f} "
            f"grad={avg['grad_norm']:.3f} "
            f"latent_std={avg['pred_latent_std']:.3f}/{avg['target_latent_std']:.3f} "
            f"({elapsed:.1f}s)"
        )

        # Check for NaN
        if np.isnan(avg['total_loss']):
            logger.error("NaN loss detected — aborting!")
            break

        # Track best + checkpoint
        if avg['consistency_loss'] < best_consistency:
            best_consistency = avg['consistency_loss']
            best_epoch = epoch
            path = save_checkpoint(params, config, epoch, avg, output_dir, 'phase2_stageA_best')
            logger.info(f"  New best consistency={best_consistency:.4f} → {path}")

        # Periodic checkpoint every 50 epochs
        if epoch > 0 and epoch % 50 == 0:
            save_checkpoint(params, config, epoch, avg, output_dir, f'phase2_stageA_e{epoch}')

    # Cleanup
    pipeline.stop()

    # Final save
    save_checkpoint(params, config, epoch, avg, output_dir, 'phase2_stageA_final')
    logger.info(f"Stage A complete. Best consistency={best_consistency:.4f} at epoch {best_epoch}")
    return params


def train_stage_b(model, params, config: Phase2Config, pool, args):
    """Stage B: MCTS ExIt training loop.

    For each step:
      1. Generate batch (obs + oracle targets + consecutive pairs for dynamics)
      2. Run MCTS on observations → visit_counts, root_value
      3. JIT'd train step: policy(visits) + value(root_val) + dynamics(consistency) + BC(oracle)
    """
    from mcts_trader.jax_agent.mctx_search import MuZeroMCTXSearch

    # Use stage_b_batch_size unless CLI overrode batch_size
    batch_size = args.batch_size if args.batch_size is not None else config.stage_b_batch_size
    logger.info(f"=== Stage B: {'Factored' if config.use_factored_search else 'MCTS'} ExIt ===")
    logger.info(f"  Epochs: {config.epochs}, Steps/epoch: {config.steps_per_epoch}, "
                f"Batch: {batch_size}")
    if config.use_factored_search:
        logger.info(f"  Factored search: T={config.factored_temperature}, "
                    f"ε={config.epsilon_greedy}, sub_batch={config.factored_sub_batch}")
    else:
        logger.info(f"  MCTS: {config.mcts_sims} sims, gumbel={config.use_gumbel}, "
                    f"max_actions={config.max_considered_actions}")
    logger.info(f"  Loss weights: policy={config.exit_policy_weight}, "
                f"value={config.exit_value_weight}, consistency={config.exit_consistency_weight}")
    logger.info(f"  BC: {config.bc_weight} → {config.bc_weight_min} over {config.bc_decay_steps} steps")
    logger.info(f"  Value targets: {config.value_target_mode} (TD discount: {config.td_discount}, MCTS: {config.mcts_discount})")
    logger.info(f"  Value clip mode: {config.value_clip_mode} (TASK_769)")

    # Reset value head to break Phase 1 saturation (V ≈ 10 for all states)
    if config.reset_value_head:
        rng_reset = jax.random.PRNGKey(config.seed + 2000)
        trm_params = params['trm']['reasoning']
        for vkey in ['value_head_1', 'value_head_2', 'value_head_3']:
            if vkey in trm_params:
                layer = trm_params[vkey]
                fan_in = layer['kernel'].shape[0]
                rng_reset, k1 = jax.random.split(rng_reset)
                new_kernel = jax.random.normal(k1, layer['kernel'].shape) * jnp.sqrt(2.0 / fan_in)
                new_bias = jnp.zeros_like(layer['bias'])
                params['trm']['reasoning'][vkey] = {
                    'kernel': new_kernel, 'bias': new_bias,
                }
        logger.info("Value head reset (He init) — breaks Phase 1 saturation")

    # Create optimizer (trains reasoning + dynamics, freezes encoder)
    optimizer, opt_state = create_stage_b_optimizer(config, params)

    # Build JIT-compiled train step
    train_step_fn = make_stage_b_train_step(model, config, optimizer)

    # Create search — factored per-asset (TASK_724) or joint MCTS
    if config.use_factored_search:
        from mcts_trader.jax_agent.factored_search import FactoredPerAssetSearch
        search = FactoredPerAssetSearch(
            model_apply_fn=model.apply,
            n_assets=config.n_assets,
            discount=config.td_discount,
            temperature=config.factored_temperature,
            epsilon_greedy=config.epsilon_greedy,
            sub_batch_size=config.factored_sub_batch,
        )
        logger.info(f"  Factored per-asset search (T={config.factored_temperature}, "
                    f"ε={config.epsilon_greedy}, sub_batch={config.factored_sub_batch})")
    else:
        search = MuZeroMCTXSearch(
            model_apply_fn=model.apply,
            num_simulations=config.mcts_sims,
            max_num_considered_actions=config.max_considered_actions,
            discount=config.mcts_discount,
            use_gumbel=config.use_gumbel,
            epsilon_greedy=config.epsilon_greedy,
        )

    # TASK_767: train-time MCTS action mask. When disable_limit_orders is True
    # build a (batch_size, 320) int8 mask (1 = valid market action,
    # 0 = LimitSell/LimitBuy) and pass it to every search() call.
    # MuZeroMCTXSearch forwards it as invalid_actions to
    # mctx.gumbel_muzero_policy, so the search tree never expands the masked
    # actions — this is the load-bearing branching-factor reduction
    # (64x5=320 -> effective 64x3=192). mctx._mask_invalid_actions runs
    # chex.assert_equal_shape against the (B, 320) prior logits, so the mask is
    # sized to the full batch (AsyncBatchPipeline always yields exactly
    # batch_size rows). Canon (False) leaves search_action_mask = None and
    # search() is called exactly as before — bit-identical to pre-767 code.
    search_action_mask = None
    if bool(getattr(config, 'disable_limit_orders', False)):
        _sm = np.ones((batch_size, config.n_assets, 5), dtype=np.int8)
        _sm[:, :, 3] = 0   # LimitSell invalid
        _sm[:, :, 4] = 0   # LimitBuy invalid
        search_action_mask = jnp.asarray(_sm.reshape(batch_size, -1))  # (B, 320)
        logger.info(
            "TASK_767: limit orders DISABLED — MCTS action space 64x5=320 -> "
            "effective 64x3=192 (LimitSell/LimitBuy masked in search + policy loss)"
        )

    # PopArt value normalization (TASK_718b)
    popart_state = None
    if config.use_popart:
        # Try to restore from checkpoint
        try:
            with open(args.checkpoint, 'rb') as f:
                _ckpt = pickle.load(f)
            if 'popart_state' in _ckpt:
                ps = _ckpt['popart_state']
                popart_state = PopArtState(
                    mean=jnp.array(ps['mean']),
                    var=jnp.array(ps['var']),
                    count=int(ps['count']),
                    mean_old=jnp.array(ps['mean']),
                    std_old=jnp.sqrt(jnp.array(ps['var']) + 1e-8),
                )
                logger.info("PopArt restored (mean=%.4f, var=%.4f, count=%d)",
                           float(popart_state.mean), float(popart_state.var), popart_state.count)
            else:
                popart_state = init_popart_state()
            del _ckpt
        except Exception:
            popart_state = init_popart_state()
        logger.info("PopArt normalization enabled (beta=%.4f)", config.popart_beta)

    # TASK_721: Async pipeline — numpy prep on background thread, JAX encode on main
    pipeline = AsyncBatchPipeline(pool, config, prefetch=2, batch_size=batch_size)

    # Output dir
    output_dir = args.output_dir or config.phase2_checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    # File logging (survives reboots, unlike /tmp)
    log_path = os.path.join(output_dir, 'stage_b_training.log')
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info("Log file: %s", log_path)

    # Dump config to JSON for reproducibility
    config_dump = {k: v for k, v in config.__dict__.items()
                   if not k.startswith('_') and isinstance(v, (int, float, str, bool, type(None)))}
    config_path = os.path.join(output_dir, 'stage_b_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dump, f, indent=2)
    logger.info("Config saved: %s (%d fields)", config_path, len(config_dump))

    # CSV metrics logging
    csv_path = os.path.join(output_dir, 'stage_b_metrics.csv')
    csv_fields = [
        'epoch', 'policy_loss', 'value_loss', 'consistency_loss', 'reward_loss',
        'bc_loss', 'bc_weight', 'dir_loss', 'dir_acc',
        'rank_loss', 'rank_spearman',
        'flow_velocity_loss', 'flow_state_loss',
        'aux_price_loss', 'aux_price_mse_h1', 'aux_price_mse_hmax',
        'bella_distill_loss', 'bella_distill_h1', 'bella_distill_hmax', 'bella_valid_frac',
        # TASK_754: trajectory-shape distill (β-v3)
        'bella_shape_loss', 'bella_shape_valid_frac',
        'bella_shape_mean_resid', 'bella_shape_std_resid',
        # TASK_753: Bella-as-data augmentation (β-v4)
        'bella_slot_frac', 'bella_regime_mix',
        # TASK_749 (Titans-MAG): per-epoch diagnostics from a fresh apply
        # on the most recent obs batch with the latest params (off-band, ~1 forward
        # pass per epoch — cheap).
        'titans_mem_norm', 'titans_surprise_mean',
        'titans_alpha_mean', 'titans_eta_mean', 'titans_theta_mean',
        'titans_gate_mean',
        # TASK_767: action-distribution diagnostics — search trade rate and the
        # fraction of search mass on limit orders (validates the 767 mechanism).
        'mean_trade_rate', 'limit_action_fraction',
        'total_loss', 'grad_norm',
        'td_target_mean', 'td_target_std', 'pa_mean', 'pa_std',
        'elapsed_s', 'gen_s', 'mcts_s', 'train_s',
    ]
    csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if not csv_exists:
        csv_writer.writeheader()
        csv_file.flush()
    logger.info("CSV metrics: %s", csv_path)

    best_total = float('inf')
    best_epoch = -1
    rng_key = jax.random.PRNGKey(config.seed + 1000)  # Different seed from Stage A

    # BC weight decay schedule
    bc_start = config.bc_weight
    bc_end = config.bc_weight_min
    bc_decay_steps = config.bc_decay_steps
    global_step = 0

    logger.info("Starting ExIt training...")
    for epoch in range(config.epochs):
        epoch_metrics = defaultdict(float)
        epoch_start = time.time()
        t_gen = 0.0
        t_mcts = 0.0
        t_train = 0.0

        # TASK_723 Fix 6: Periodic slot rotation — stop pipeline, rotate safely, restart.
        # Random sampling never exhausts cursors, so _rotate_slot is never triggered.
        # Rotate one slot every 20 epochs to cycle through all available segments.
        if epoch > 0 and epoch % 20 == 0:
            slot_to_rotate = (epoch // 20) % len(pool.slots)
            pipeline.stop()
            pool._rotate_slot(slot_to_rotate)
            label = pool.slots[slot_to_rotate].get('label', '?')
            logger.info(f"  Rotated slot {slot_to_rotate} → {label}")
            pipeline = AsyncBatchPipeline(pool, config, prefetch=2, batch_size=batch_size)

        # TASK_718d: Denormalize value for MCTS — value head now outputs normalized
        # predictions, so MCTS needs to rescale for tree search Q-values.
        # Update per-epoch (beta is small, intra-epoch drift is negligible).
        if popart_state is not None:
            search.value_scale = float(jnp.sqrt(popart_state.var + 1e-8))
            search.value_shift = float(popart_state.mean)

        for step in range(config.steps_per_epoch):
            # 1. Generate batch
            _tg0 = time.time()
            batch = pipeline.get_batch()
            obs = batch['obs_t']
            obs_t1 = batch['obs_t1']
            obs_tN = batch['obs_tN']
            actions_dyn = batch['actions']
            rewards_dyn = batch['rewards']
            oracle_targets = batch['oracle_targets']
            oracle_values = batch['oracle_values']
            nstep_rewards = batch['nstep_rewards']
            future_cum_returns = batch['future_cum_returns']  # TASK_727
            future_log_returns = batch['future_log_returns']  # TASK_735
            bella_preds = batch.get('bella_preds')              # TASK_740
            if bella_preds is None:
                # Placeholder zeros with NaN-mask semantics (always masked out).
                bella_preds = jnp.full(
                    (future_log_returns.shape[0],
                     len(getattr(config, 'bella_pred_horizons', (1, 5, 10, 20))),
                     config.n_assets),
                    jnp.nan, dtype=jnp.float32,
                )
            bella_shape = batch.get('bella_shape')              # TASK_754
            if bella_shape is None:
                bella_shape = jnp.full(
                    (future_log_returns.shape[0],
                     int(getattr(config, 'bella_shape_descriptor_dim', 14))),
                    jnp.nan, dtype=jnp.float32,
                )
            _tg1 = time.time()
            t_gen += _tg1 - _tg0

            # 2. Run MCTS search (JIT'd internally)
            rng_key, search_key = jax.random.split(rng_key)
            if search_action_mask is None:
                # Canon path — call site bit-identical to pre-767 code.
                _actions, visit_counts, stats = search(params, search_key, obs)
            else:
                # TASK_767: pass the limit-order mask so MCTS never expands
                # LimitSell/LimitBuy. mctx requires the mask shape to equal the
                # (B, 320) logits exactly; slice to obs.shape[0] in case the
                # final batch is short (search_action_mask is pre-sized to the
                # configured batch_size, which is the upper bound).
                _actions, visit_counts, stats = search(
                    params, search_key, obs,
                    search_action_mask[:obs.shape[0]],
                )
            mcts_value = stats['root_value']  # (B,) scalar
            _tm1 = time.time()
            t_mcts += _tm1 - _tg1

            # 3. BC weight decay (linear)
            bc_weight = bc_start - (bc_start - bc_end) * min(global_step / max(bc_decay_steps, 1), 1.0)
            bc_weight_jax = jnp.float32(bc_weight)

            # 4. PopArt stats for JIT (scalars — no recompilation)
            if popart_state is not None:
                pa_mean = jnp.float32(popart_state.mean)
                pa_std = jnp.float32(jnp.sqrt(popart_state.var + 1e-8))
            else:
                pa_mean = jnp.float32(0.0)
                pa_std = jnp.float32(1.0)

            # 5. Train step
            params, opt_state, metrics = train_step_fn(
                params, opt_state,
                obs, obs_t1, obs_tN, actions_dyn, rewards_dyn,
                visit_counts, mcts_value,
                oracle_targets, bc_weight_jax,
                pa_mean, pa_std,
                oracle_values, nstep_rewards,
                future_cum_returns,
                future_log_returns,
                bella_preds,
                bella_shape,
            )
            # Fence: ensure GPU compute finishes before next batch's CPU→GPU transfers.
            # Without this, jnp.array() in _encode_jax can collide with the async
            # compute stream, causing "copy_stream->WaitFor(compute_stream)" crashes.
            jax.block_until_ready(metrics)

            # 6. Update PopArt statistics (outside JIT — mutable state)
            raw_targets = metrics.pop('td_targets_raw')  # Always pop (B,) array before accumulation
            if popart_state is not None:
                popart_state = update_statistics(popart_state, raw_targets, config.popart_beta)
            _tt1 = time.time()
            t_train += _tt1 - _tm1

            for k, v in metrics.items():
                epoch_metrics[k] += float(v)

            global_step += 1

        # Average metrics
        n_steps = config.steps_per_epoch
        avg = {k: v / n_steps for k, v in epoch_metrics.items()}
        elapsed = time.time() - epoch_start

        # Log
        pa_info = ""
        if popart_state is not None:
            pa_info = (f" td_mean={avg['td_target_mean']:.2f}"
                       f" td_std={avg['td_target_std']:.2f}"
                       f" pa_mu={float(popart_state.mean):.2f}"
                       f" pa_sig={float(jnp.sqrt(popart_state.var + 1e-8)):.2f}")
        dir_info = ""
        if config.use_direction_head:
            dir_info = f" dir={avg.get('dir_loss', 0):.4f}(acc={avg.get('dir_acc', 0.5):.3f})"
        rank_info = ""
        if getattr(config, 'use_rank_head', False):
            rank_info = (f" rank={avg.get('rank_loss', 0):.4f}"
                         f"(sp={avg.get('rank_spearman', 0):.3f})")
        flow_info = ""
        if getattr(config, 'use_flow_matching', False):
            flow_info = (f" flow=v{avg.get('flow_velocity_loss', 0):.4f}"
                         f"/s{avg.get('flow_state_loss', 0):.4f}")
        aux_price_info = ""
        if getattr(config, 'use_aux_price_head', False):
            aux_price_info = (f" aux={avg.get('aux_price_loss', 0):.5f}"
                              f"(h1={avg.get('aux_price_mse_h1', 0):.5f}"
                              f",hmax={avg.get('aux_price_mse_hmax', 0):.5f})")
        bella_info = ""
        if getattr(config, 'use_bella_distill_head', False):
            bella_info = (f" bella={avg.get('bella_distill_loss', 0):.5f}"
                          f"(vf={avg.get('bella_valid_frac', 0):.2f}"
                          f",h1={avg.get('bella_distill_h1', 0):.5f}"
                          f",hmax={avg.get('bella_distill_hmax', 0):.5f})")
        bella_shape_info = ""
        if getattr(config, 'use_bella_shape_head', False):
            bella_shape_info = (
                f" shape={avg.get('bella_shape_loss', 0):.5f}"
                f"(vf={avg.get('bella_shape_valid_frac', 0):.2f}"
                f",mR={avg.get('bella_shape_mean_resid', 0):.4f}"
                f",sR={avg.get('bella_shape_std_resid', 0):.4f})"
            )
        # TASK_753: Bella-as-data augmentation per-epoch slot stats.
        bella_aug_info = ""
        bella_slot_frac = 0.0
        bella_regime_mix_str = ""
        if getattr(config, 'bella_augment', False):
            origins = getattr(pool, '_slot_origins', [])
            if origins:
                bella_slot_frac = float(sum(origins)) / float(len(origins))
            regime_counts = getattr(pool, '_bella_regime_counts', None)
            if regime_counts:
                # Compact 5-tuple: c=crash, r=rally, s=sideways, v=volatile, k=calm, u=unconditional
                labels = ['c', 'r', 's', 'v', 'k', 'u']
                parts = []
                for i, lab in enumerate(labels):
                    n = regime_counts.get(i, 0)
                    if n > 0:
                        parts.append(f"{lab}={n}")
                bella_regime_mix_str = ",".join(parts) if parts else "none"
            else:
                bella_regime_mix_str = "none"
            bella_aug_info = (
                f" bella_aug=slot{bella_slot_frac:.2f}"
                f" mix={bella_regime_mix_str}"
            )

        # TASK_749 (Titans-MAG): off-band per-epoch Titans diagnostics.
        # Run one extra apply call on the most recent obs batch with the latest
        # params, capturing the sown 'intermediates' from the encoder. Cheap
        # (one forward pass per epoch). All-zero defaults if Titans is off.
        titans_metrics = {
            'titans_mem_norm': 0.0,
            'titans_surprise_mean': 0.0,
            'titans_alpha_mean': 0.0,
            'titans_eta_mean': 0.0,
            'titans_theta_mean': 0.0,
            'titans_gate_mean': 0.0,
        }
        if getattr(config, 'use_titans_memory', False):
            try:
                _, mutated = model.apply(
                    {'params': params}, observation=obs, mode='initial_inference',
                    mutable=['intermediates'],
                )
                inter = mutated.get('intermediates', {})
                # Navigate to the encoder subtree's titans_diag sow.
                # nn.sow yields a tuple-per-sow at the module path. Flax stores
                # sown values under their full module path: trm/encoder/titans_diag.
                td_path = inter.get('trm', {}).get('encoder', {}).get('titans_diag')
                if td_path is not None and len(td_path) > 0:
                    # td_path is a tuple of dicts (one per sow call within
                    # this apply); take the first call's dict.
                    td = td_path[0]
                    titans_metrics['titans_mem_norm'] = float(td.get('mem_norm', 0.0))
                    titans_metrics['titans_surprise_mean'] = float(td.get('surprise_norm', 0.0))
                    titans_metrics['titans_alpha_mean'] = float(td.get('alpha_mean', 0.0))
                    titans_metrics['titans_eta_mean'] = float(td.get('eta_mean', 0.0))
                    titans_metrics['titans_theta_mean'] = float(td.get('theta_mean', 0.0))
                    titans_metrics['titans_gate_mean'] = float(td.get('gate_mean', 0.0))
            except Exception as exc:  # pragma: no cover — diagnostic-only
                logger.warning(f"Titans diagnostic capture failed: {exc}")

        titans_info = ""
        if getattr(config, 'use_titans_memory', False):
            titans_info = (f" titans=mem{titans_metrics['titans_mem_norm']:.3f}"
                           f"/g{titans_metrics['titans_gate_mean']:.4f}"
                           f"/sup{titans_metrics['titans_surprise_mean']:.3e}"
                           f"/a{titans_metrics['titans_alpha_mean']:.3f}")

        # TASK_767: search action-distribution diagnostics (always logged).
        trade_info = (f" trade_rate={avg.get('mean_trade_rate', 0):.3f}"
                      f"/limit={avg.get('limit_action_fraction', 0):.3f}")

        logger.info(
            f"E{epoch:4d}: policy={avg['policy_loss']:.4f} "
            f"value={avg['value_loss']:.4f} "
            f"consistency={avg['consistency_loss']:.4f} "
            f"bc={avg['bc_loss']:.4f}(w={avg['bc_weight']:.4f}){dir_info}{rank_info}{flow_info}{aux_price_info}{bella_info}{bella_shape_info}{bella_aug_info}{titans_info}{trade_info} "
            f"total={avg['total_loss']:.4f} "
            f"grad={avg['grad_norm']:.3f}"
            f"{pa_info} "
            f"({elapsed:.1f}s | gen={t_gen:.1f} mcts={t_mcts:.1f} train={t_train:.1f})"
        )

        # CSV row
        csv_writer.writerow({
            'epoch': epoch,
            'policy_loss': f"{avg['policy_loss']:.6f}",
            'value_loss': f"{avg['value_loss']:.6f}",
            'consistency_loss': f"{avg['consistency_loss']:.6f}",
            'reward_loss': f"{avg['reward_loss']:.6f}",
            'bc_loss': f"{avg['bc_loss']:.6f}",
            'bc_weight': f"{avg['bc_weight']:.6f}",
            'dir_loss': f"{avg.get('dir_loss', 0):.6f}",
            'dir_acc': f"{avg.get('dir_acc', 0.5):.6f}",
            'rank_loss': f"{avg.get('rank_loss', 0):.6f}",
            'rank_spearman': f"{avg.get('rank_spearman', 0):.6f}",
            'flow_velocity_loss': f"{avg.get('flow_velocity_loss', 0):.6f}",
            'flow_state_loss': f"{avg.get('flow_state_loss', 0):.6f}",
            'aux_price_loss': f"{avg.get('aux_price_loss', 0):.6f}",
            'aux_price_mse_h1': f"{avg.get('aux_price_mse_h1', 0):.6f}",
            'aux_price_mse_hmax': f"{avg.get('aux_price_mse_hmax', 0):.6f}",
            'bella_distill_loss': f"{avg.get('bella_distill_loss', 0):.6f}",
            'bella_distill_h1': f"{avg.get('bella_distill_h1', 0):.6f}",
            'bella_distill_hmax': f"{avg.get('bella_distill_hmax', 0):.6f}",
            'bella_valid_frac': f"{avg.get('bella_valid_frac', 0):.6f}",
            # TASK_754: trajectory-shape distill (β-v3)
            'bella_shape_loss': f"{avg.get('bella_shape_loss', 0):.6f}",
            'bella_shape_valid_frac': f"{avg.get('bella_shape_valid_frac', 0):.6f}",
            'bella_shape_mean_resid': f"{avg.get('bella_shape_mean_resid', 0):.6f}",
            'bella_shape_std_resid': f"{avg.get('bella_shape_std_resid', 0):.6f}",
            # TASK_753: Bella-as-data augmentation (β-v4)
            'bella_slot_frac': f"{bella_slot_frac:.4f}",
            'bella_regime_mix': bella_regime_mix_str,
            # TASK_749 (Titans-MAG) diagnostics — zero when titans is off
            'titans_mem_norm': f"{titans_metrics['titans_mem_norm']:.6f}",
            'titans_surprise_mean': f"{titans_metrics['titans_surprise_mean']:.6e}",
            'titans_alpha_mean': f"{titans_metrics['titans_alpha_mean']:.6f}",
            'titans_eta_mean': f"{titans_metrics['titans_eta_mean']:.6f}",
            'titans_theta_mean': f"{titans_metrics['titans_theta_mean']:.6f}",
            'titans_gate_mean': f"{titans_metrics['titans_gate_mean']:.6f}",
            # TASK_767: action-distribution diagnostics
            'mean_trade_rate': f"{avg.get('mean_trade_rate', 0):.6f}",
            'limit_action_fraction': f"{avg.get('limit_action_fraction', 0):.6f}",
            'total_loss': f"{avg['total_loss']:.6f}",
            'grad_norm': f"{avg['grad_norm']:.6f}",
            'td_target_mean': f"{avg.get('td_target_mean', 0):.6f}",
            'td_target_std': f"{avg.get('td_target_std', 0):.6f}",
            'pa_mean': f"{float(popart_state.mean):.6f}" if popart_state else "0",
            'pa_std': f"{float(jnp.sqrt(popart_state.var + 1e-8)):.6f}" if popart_state else "1",
            'elapsed_s': f"{elapsed:.1f}",
            'gen_s': f"{t_gen:.1f}",
            'mcts_s': f"{t_mcts:.1f}",
            'train_s': f"{t_train:.1f}",
        })
        csv_file.flush()

        # Check for NaN
        if np.isnan(avg['total_loss']):
            logger.error("NaN loss detected — aborting!")
            break

        # Track best + checkpoint
        if avg['total_loss'] < best_total:
            best_total = avg['total_loss']
            best_epoch = epoch
            path = save_checkpoint(params, config, epoch, avg, output_dir, 'phase2_stageB_best',
                                   popart_state=popart_state)
            logger.info(f"  New best total={best_total:.4f} → {path}")

        # Periodic checkpoint every 50 epochs
        if epoch > 0 and epoch % 50 == 0:
            save_checkpoint(params, config, epoch, avg, output_dir, f'phase2_stageB_e{epoch}',
                           popart_state=popart_state)

        # Encoder unfreeze check
        if (config.encoder_unfreeze_epoch is not None and
                epoch == config.encoder_unfreeze_epoch):
            logger.info(f"  Unfreezing encoder at epoch {epoch} (lr={config.encoder_lr})")
            # Rebuild optimizer with encoder included
            # For simplicity, recreate with full mask (no encoder exclusion)
            # This is safe because encoder_lr is very low (3e-5)
            optimizer, opt_state = create_stage_b_optimizer_with_encoder(config, params)
            train_step_fn = make_stage_b_train_step(model, config, optimizer)

    # Cleanup
    pipeline.stop()

    # Final save
    save_checkpoint(params, config, config.epochs - 1, avg, output_dir, 'phase2_stageB_final',
                    popart_state=popart_state)
    csv_file.close()
    logger.removeHandler(file_handler)
    file_handler.close()
    logger.info(f"Stage B complete. Best total={best_total:.4f} at epoch {best_epoch}")
    return params


def create_stage_b_optimizer_with_encoder(config: Phase2Config, params):
    """Optimizer that trains everything including encoder (at lower LR).

    Used when encoder is unfrozen mid-training.
    """
    total_steps = config.epochs * config.steps_per_epoch
    warmup_steps = min(200, max(total_steps - 1, 0))

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.policy_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps, warmup_steps + 1),
        end_value=config.policy_lr * 0.1,
    )

    train_opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=1e-4),
    )
    freeze_opt = optax.set_to_zero()

    # Train everything except oracle_mask
    def should_train(path, _leaf):
        key = str(path[0])
        return 'oracle_mask' not in key

    mask = jax.tree_util.tree_map_with_path(should_train, params)
    labels = jax.tree_util.tree_map(lambda x: 'train' if x else 'freeze', mask)

    optimizer = optax.multi_transform(
        {'train': train_opt, 'freeze': freeze_opt},
        labels,
    )
    opt_state = optimizer.init(params)
    return optimizer, opt_state


# TASK_772.1 (2026-05-23): parse_args lives in train_phase2_cli.py — pure
# argparse, no JAX/numpy/flax dependencies. Drop-in import; bit-identical CLI surface.
from train_phase2_cli import parse_args


def main():
    args = parse_args()

    # --- Load Phase 1 checkpoint ---
    ckpt_path = str(Path(args.checkpoint).resolve())
    logger.info(f"Loading Phase 1 checkpoint: {ckpt_path}")
    params, p1_config, p1_epoch, p1_metrics = load_phase1_checkpoint(ckpt_path)

    # --- Build Phase 2 config from Phase 1 config ---
    # Look for config.json next to checkpoint
    ckpt_dir = Path(ckpt_path).parent
    p1_config_path = ckpt_dir / 'config.json'
    p1_had_oracle_k = False
    if p1_config_path.exists():
        config = get_phase2_config(str(p1_config_path))
        logger.info(f"Loaded Phase 1 config from {p1_config_path}")
        # TASK_771/764/F3: probe the JSON for explicit `oracle_k` presence so
        # we can distinguish "inherited from ckpt" from "Phase2Config default
        # fallback" — see the log below.
        import json as _json_probe
        try:
            with open(p1_config_path) as _f:
                p1_had_oracle_k = 'oracle_k' in _json_probe.load(_f)
        except Exception:
            p1_had_oracle_k = False
    else:
        config = Phase2Config(**{k: v for k, v in p1_config.items()
                                 if k in Phase2Config.__dataclass_fields__})
        p1_had_oracle_k = 'oracle_k' in p1_config
        logger.info("Built config from checkpoint metadata")

    # TASK_764: oracle_k is inherited silently from the Phase-1 imprint config.
    # Log it explicitly so a wrong --checkpoint (k=4 vs k=8/12) is visible.
    # TASK_771/764/F3: warn when the key was missing — the value is the
    # `Phase2Config` (= `EasyConfig`) default, NOT inherited.
    if p1_had_oracle_k:
        logger.info(f"[TASK_764] oracle_k inherited from Phase-1 checkpoint = {config.oracle_k}")
    else:
        logger.warning(
            f"[TASK_764] oracle_k MISSING from Phase-1 ckpt config — "
            f"defaulted to {config.oracle_k} (this is a pre-TASK_764 ckpt; "
            f"override with `--oracle-k` if Phase-1 used a non-default value)"
        )

    # Apply stage + CLI overrides
    config.stage = args.stage
    config.get_stage_defaults()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.steps_per_epoch is not None:
        config.steps_per_epoch = args.steps_per_epoch
    if args.seed is not None:
        config.seed = args.seed
    if args.dynamics_lr is not None:
        config.dynamics_lr = args.dynamics_lr
    if args.policy_lr is not None:
        config.policy_lr = args.policy_lr
    if args.value_lr is not None:
        config.value_lr = args.value_lr
    if args.mcts_sims is not None:
        config.mcts_sims = args.mcts_sims
    if args.mcts_discount is not None:
        config.mcts_discount = args.mcts_discount
    if args.factored:
        config.use_factored_search = True
    if args.no_reset_value_head:
        config.reset_value_head = False
    if args.factored_temperature is not None:
        config.factored_temperature = args.factored_temperature
    if args.factored_sub_batch is not None:
        config.factored_sub_batch = args.factored_sub_batch
    if args.bc_weight_min is not None:
        config.bc_weight_min = args.bc_weight_min
    if args.bc_decay_steps is not None:
        config.bc_decay_steps = args.bc_decay_steps
    if args.no_bc_balance:
        config.bc_class_balanced = False
    if args.bc_max_trade_weight is not None:
        config.bc_max_trade_weight = args.bc_max_trade_weight
    # TASK_781 / TASK_725-D: return-weighted BC. Flag stays default-OFF; setting
    # it ON requires either --bc-return-weighted on the CLI or a JSON config
    # override. The value-zero default (config.bc_return_weighted=False) keeps
    # the canon BC path bit-identical.
    if args.bc_return_weighted:
        config.bc_return_weighted = True
    # TASK_762: reward shaping overrides
    if args.reward_scale is not None:
        config.reward_scale = args.reward_scale
    if args.fee_penalty_bps is not None:
        config.fee_penalty_bps = args.fee_penalty_bps
    # TASK_763: policy entropy regularization overrides
    if args.entropy_coef is not None:
        config.entropy_coef = args.entropy_coef
    if args.min_entropy is not None:
        config.min_entropy = args.min_entropy
    # TASK_767: action-space restructure (drop limit orders).
    # TASK_771/F4 (P2): BooleanOptionalAction three-state semantics — None =
    # inherit, True = force ON, False = force OFF.
    if args.disable_limit_orders is not None:
        config.disable_limit_orders = bool(args.disable_limit_orders)
    # TASK_769: value-clip mode override (F2 fix)
    if args.value_clip_mode is not None:
        config.value_clip_mode = args.value_clip_mode
    if args.no_direction_head:
        config.use_direction_head = False
    if args.direction_weight is not None:
        config.direction_loss_weight = args.direction_weight
    if args.direction_horizon is not None:
        config.direction_horizon = args.direction_horizon
    if args.no_direction_policy_bias:
        config.use_direction_policy_bias = False
    # TASK_730: rank head overrides
    if args.no_rank_head:
        config.use_rank_head = False
    if args.rank_weight is not None:
        config.rank_loss_weight = args.rank_weight
    if args.rank_loss is not None:
        config.rank_loss_type = args.rank_loss
    # TASK_734: flow-matching overrides
    if args.flow_matching:
        config.use_flow_matching = True
    if args.flow_matching_velocity_weight is not None:
        config.flow_matching_velocity_weight = args.flow_matching_velocity_weight
    # TASK_735: aux price head overrides
    if args.no_aux_price_head:
        config.use_aux_price_head = False
    if args.aux_price_weight is not None:
        config.aux_price_loss_weight = args.aux_price_weight
    # TASK_740: Bella distillation head overrides
    if args.bella_distill_head:
        config.use_bella_distill_head = True
    if args.no_bella_distill_head:
        config.use_bella_distill_head = False
    if args.bella_distill_weight is not None:
        config.bella_distill_loss_weight = args.bella_distill_weight
    if args.bella_preds_manifest is not None:
        config.bella_preds_manifest = args.bella_preds_manifest
    if getattr(config, 'use_bella_distill_head', False):
        logger.info(f"Bella distillation head ON "
                    f"(weight={config.bella_distill_loss_weight}, "
                    f"manifest={config.bella_preds_manifest})")
    # TASK_754: Bella trajectory-shape head overrides
    if args.bella_shape_distill:
        config.use_bella_shape_head = True
    if args.no_bella_shape_distill:
        config.use_bella_shape_head = False
    if args.bella_shape_mode is not None:
        config.bella_shape_mode = args.bella_shape_mode
    if args.bella_shape_weight is not None:
        config.bella_shape_loss_weight = args.bella_shape_weight
    if args.bella_shape_descriptor_dim is not None:
        config.bella_shape_descriptor_dim = int(args.bella_shape_descriptor_dim)
    if args.bella_shape_component_weights is not None:
        parsed = tuple(float(x) for x in args.bella_shape_component_weights.split(','))
        if len(parsed) != 6:
            raise ValueError(
                f"--bella-shape-component-weights expects 6 comma-separated floats "
                f"(mean, std, skew, kurt, s1, s1_ratio); got {len(parsed)}: {parsed}"
            )
        config.bella_shape_component_weights = parsed
    if args.bella_shape_manifest is not None:
        config.bella_shape_manifest = args.bella_shape_manifest
    # TASK_754: mutual-exclusion with TASK_740 (per spec §5; reviewer C6).
    if (getattr(config, 'use_bella_distill_head', False)
            and getattr(config, 'use_bella_shape_head', False)):
        raise ValueError(
            "Cannot enable both --bella-distill-head (TASK_740) and "
            "--bella-shape-distill (TASK_754) simultaneously. Choose one."
        )
    # TASK_754: validate shape mode early.
    if getattr(config, 'use_bella_shape_head', False):
        mode = getattr(config, 'bella_shape_mode', 'moments')
        if mode != 'moments':
            raise NotImplementedError(
                f"TASK_754: --bella-shape-mode '{mode}' is reserved (lowrank/"
                f"contrastive). Only 'moments' is implemented in TASK_754."
            )
        logger.info(
            f"Bella trajectory-shape head ON "
            f"(weight={config.bella_shape_loss_weight}, "
            f"D={config.bella_shape_descriptor_dim}, "
            f"mode={mode}, manifest={config.bella_shape_manifest})"
        )
    # TASK_753: Bella-as-data augmentation overrides.
    if args.bella_augment:
        config.bella_augment = True
    if args.no_bella_augment:
        config.bella_augment = False
    if args.bella_augment_ratio is not None:
        if not (0.0 <= args.bella_augment_ratio <= 1.0):
            raise ValueError(
                f"--bella-augment-ratio must be in [0, 1]; got {args.bella_augment_ratio}"
            )
        config.bella_augment_ratio = args.bella_augment_ratio
    if args.bella_augment_mode is not None:
        config.bella_augment_mode = args.bella_augment_mode
    if args.bella_augment_regime_mode is not None:
        config.bella_augment_regime_mode = args.bella_augment_regime_mode
    if args.bella_augment_checkpoint is not None:
        config.bella_augment_checkpoint = args.bella_augment_checkpoint
    if args.bella_augment_k_samples is not None:
        config.bella_augment_k_samples = args.bella_augment_k_samples
    if args.bella_augment_seq_length is not None:
        config.bella_augment_seq_length = args.bella_augment_seq_length
    if args.bella_augment_refresh_epochs is not None:
        config.bella_augment_refresh_epochs = args.bella_augment_refresh_epochs
    if args.bella_augment_volatility_scale is not None:
        config.bella_augment_volatility_scale = args.bella_augment_volatility_scale
    if args.bella_augment_asset_policy is not None:
        config.bella_augment_asset_policy = args.bella_augment_asset_policy
    if args.bella_augment_log_every is not None:
        config.bella_augment_log_every = args.bella_augment_log_every
    # TASK_753: mutual-exclusion with TASK_740 (bella_distill) and TASK_754
    # (bella_shape). Mirror the 740↔754 exclusion enforced just above.
    if getattr(config, 'bella_augment', False):
        if getattr(config, 'use_bella_distill_head', False):
            raise ValueError(
                "Cannot enable both --bella-augment (TASK_753) and "
                "--bella-distill-head (TASK_740) simultaneously. Choose one."
            )
        if getattr(config, 'use_bella_shape_head', False):
            raise ValueError(
                "Cannot enable both --bella-augment (TASK_753) and "
                "--bella-shape-distill (TASK_754) simultaneously. Choose one."
            )
        # Mode validation — mirrors C9 in TASK_753_REVIEW. Pool-side will
        # also raise; this is the explicit early-exit so the user sees a
        # clean error before the (expensive) pool construction.
        if config.bella_augment_mode != 'append':
            raise NotImplementedError(
                f"TASK_753 v1: --bella-augment-mode '{config.bella_augment_mode}' "
                f"is reserved for v2 (TASK_753b). Use 'append'."
            )
        if config.bella_augment_asset_policy not in ('remap', 'canonical', 'random'):
            raise ValueError(
                f"--bella-augment-asset-policy must be one of "
                f"{{remap, canonical, random}}; got "
                f"{config.bella_augment_asset_policy!r}"
            )
        logger.info(
            f"Bella augmentation ON "
            f"(ratio={config.bella_augment_ratio}, "
            f"mode={config.bella_augment_mode}, "
            f"regime_mode={config.bella_augment_regime_mode}, "
            f"asset_policy={config.bella_augment_asset_policy}, "
            f"seq_length={config.bella_augment_seq_length}, "
            f"vol_scale={config.bella_augment_volatility_scale})"
        )
    # TASK_738: legacy 27-ch encoder mode
    if args.legacy_27ch:
        config.use_cross_sectional = False
        # Total obs channels = 27 base + state channels. Default state = 7
        # (per TASK_689). So set n_channels = 34 for the state-channels heuristic.
        if config.n_channels >= 39:
            config.n_channels = 34
        logger.info("Legacy 27-ch encoder mode enabled (ablate TASK_732)")
    # TASK_739: dense-sign oracle overrides
    if args.oracle_mode is not None:
        if args.oracle_mode == 'hybrid':
            raise NotImplementedError(
                "--oracle-mode=hybrid is reserved (TASK_739 follow-up). "
                "Use topk_path or dense_sign."
            )
        config.oracle_mode = args.oracle_mode
        logger.info(f"Oracle mode: {config.oracle_mode}")
    if args.sign_epsilon is not None:
        config.sign_epsilon = args.sign_epsilon
        logger.info(f"Sign epsilon: {config.sign_epsilon}")
    # TASK_729b: encoder width overrides (must match Phase 1 checkpoint)
    if args.embed_dim is not None:
        config.embed_dim = args.embed_dim
    if args.encoder_heads is not None:
        config.encoder_num_heads = args.encoder_heads
    # TASK_729c: encoder family override
    if args.encoder_version is not None:
        config.encoder_version = args.encoder_version
    if args.encoder_lr is not None:
        config.encoder_lr = args.encoder_lr
    # TASK_782 / TASK_780 Phase 4: encoder unfreeze schedule (None = never).
    # Phase2Config exposes the field; the CLI flag now lets operators set it
    # without editing the JSON. Bridge ckpts (TASK_780) often need this set
    # to e.g. 50 so the encoder joins Stage-B training mid-run.
    if args.encoder_unfreeze_epoch is not None:
        config.encoder_unfreeze_epoch = args.encoder_unfreeze_epoch
    # TASK_741 (Mint): per-asset injection override
    if args.per_asset_injection is not None:
        if args.per_asset_injection == 'none':
            config.use_per_asset_injection = False
            config.per_asset_inject_xs = False
            config.per_asset_inject_cheat = False
        else:
            config.use_per_asset_injection = True
            config.per_asset_inject_xs = args.per_asset_injection in ('xs', 'both')
            config.per_asset_inject_cheat = args.per_asset_injection in ('cheat', 'both')
        logger.info(f"[MINT] Per-asset injection: {args.per_asset_injection} "
                    f"(xs={config.per_asset_inject_xs}, cheat={config.per_asset_inject_cheat})")
    # TASK_749 (Titans-MAG): neural memory module overrides.
    if args.titans_memory:
        config.use_titans_memory = True
    if args.titans_variant is not None:
        if args.titans_variant != 'mag':
            raise NotImplementedError(
                f"--titans-variant={args.titans_variant!r} not implemented "
                "(only 'mag' available in TASK_749 v1)."
            )
        config.titans_variant = args.titans_variant
    if args.titans_memory_depth is not None:
        config.titans_memory_depth = args.titans_memory_depth
    if args.titans_memory_hidden is not None:
        config.titans_memory_hidden = args.titans_memory_hidden
    if args.titans_persistent_tokens is not None:
        config.titans_persistent_tokens = args.titans_persistent_tokens
    if args.titans_surprise_momentum_init is not None:
        config.titans_surprise_momentum_init = args.titans_surprise_momentum_init
    if args.titans_decay_init is not None:
        config.titans_decay_init = args.titans_decay_init
    if args.titans_lr_init is not None:
        config.titans_lr_init = args.titans_lr_init
    if args.titans_gate_init is not None:
        config.titans_gate_init = args.titans_gate_init
    if args.titans_lr is not None:
        config.titans_lr = args.titans_lr
    if getattr(config, 'use_titans_memory', False):
        logger.info(f"[TITANS-MAG] ON: depth={config.titans_memory_depth} "
                    f"N_p={config.titans_persistent_tokens} "
                    f"η_init={config.titans_surprise_momentum_init} "
                    f"α_init={config.titans_decay_init} "
                    f"θ_init={config.titans_lr_init} "
                    f"gate_init={config.titans_gate_init} "
                    f"outer_lr={config.titans_lr if config.titans_lr is not None else 'inherit'}")
    if args.value_weight is not None:
        config.exit_value_weight = args.value_weight
    if args.value_target_mode is not None:
        config.value_target_mode = args.value_target_mode
    if args.nstep_horizon is not None:
        config.nstep_horizon = args.nstep_horizon
    if args.td_discount is not None:
        config.td_discount = args.td_discount
    # TASK_777: mixture value target — α + oracle-horizon overrides.
    # Phase2Config.__post_init__ validates 0 ≤ α ≤ 1 and the value_target_mode
    # vocabulary; argparse `choices=` validates the CLI typo case. The guard
    # below catches an operator using --value-mixture-alpha with a non-mixture
    # mode (silent no-op otherwise — the override hits the field but no
    # consumer reads it). Cheap upfront safety.
    if args.value_mixture_alpha is not None:
        if config.value_target_mode != "mixture":
            raise ValueError(
                f"--value-mixture-alpha={args.value_mixture_alpha} requires "
                f"--value-target-mode mixture (got {config.value_target_mode!r}). "
                f"Either set --value-target-mode mixture or drop --value-mixture-alpha."
            )
        config.value_mixture_alpha = args.value_mixture_alpha
    if args.oracle_horizon is not None:
        config.oracle_value_horizon = args.oracle_horizon
    # TASK_784: oracle_smoothing CLI override. Validate λ ∈ [0, 1) (λ=1 is
    # degenerate — smoothed_delta would be constant and remaining = 0).
    if getattr(args, 'oracle_smoothing', None) is not None:
        if not (0.0 <= args.oracle_smoothing < 1.0):
            raise ValueError(
                f"--oracle-smoothing must lie in [0, 1), got {args.oracle_smoothing!r}"
            )
        config.oracle_smoothing = args.oracle_smoothing
        if config.oracle_smoothing > 0.0:
            logger.info(f"[TASK_784] oracle_smoothing λ={config.oracle_smoothing} "
                        f"(EWMA on oracle_cum_pnl deltas at batch-build time)")
    if args.name:
        config.name = args.name
    else:
        config.name = f"phase2_stage{args.stage}"

    # --- Analyze checkpoint ---
    info = analyze_params(params)
    n_params = info['total_params']
    logger.info(f"Loaded Phase 1 checkpoint (epoch {p1_epoch}, {n_params / 1e6:.2f}M params)")
    logger.info(f"Dynamics layers: {info['dynamics_zero']}/{info['dynamics_total']} zero (ready for init)"
                if info['dynamics_zero'] == info['dynamics_total']
                else f"Dynamics layers: {info['dynamics_total'] - info['dynamics_zero']}/{info['dynamics_total']} trained")
    logger.info(f"Encoder layers: {info['encoder_zero']} zero (trained)"
                if info['encoder_zero'] == 0
                else f"Encoder layers: {info['encoder_zero']}/{info['encoder_total']} zero (WARNING: partially untrained)")

    # --- Create model (verify architecture matches checkpoint) ---
    model = create_model(config)
    logger.info(f"Model created: state_dim={config.state_dim}, actions={config.action_space_size}, "
                f"H={config.H_cycles}, L={config.L_cycles}")

    # TASK_729a: Optionally drop direction_head + refiner so they fresh-init.
    # Required when head architecture changed (old dir_head read global latent,
    # new one reads per-asset features — completely different param shapes).
    if getattr(args, 'reset_direction_head', False):
        dropped = []
        for k in ('direction_head', 'policy_direction_refiner'):
            if k in params:
                del params[k]
                dropped.append(k)
        if dropped:
            logger.info(f"Reset aux heads (will fresh-init): {dropped}")

    # TASK_730: Optionally drop rank_head so it fresh-inits (e.g. restart from a
    # checkpoint whose rank head diverged or used a different hidden_dim).
    if getattr(args, 'reset_rank_head', False):
        if 'rank_head' in params:
            del params['rank_head']
            logger.info("Reset rank_head (will fresh-init)")

    # TASK_735: Optionally drop aux_price_head on load.
    if getattr(args, 'reset_aux_price_head', False):
        if 'aux_price_head' in params:
            del params['aux_price_head']
            logger.info("Reset aux_price_head (will fresh-init)")

    # TASK_740: Optionally drop aux_bella_head on load.
    if getattr(args, 'reset_bella_distill_head', False):
        if 'aux_bella_head' in params:
            del params['aux_bella_head']
            logger.info("Reset aux_bella_head (will fresh-init)")

    # TASK_754: Optionally drop aux_bella_shape_head on load (required on
    # first run from any pre-754 ckpt — otherwise initialize_missing_aux_heads
    # adds the fresh subtree).
    if getattr(args, 'reset_bella_shape_head', False):
        if 'aux_bella_shape_head' in params:
            del params['aux_bella_shape_head']
            logger.info("Reset aux_bella_shape_head (will fresh-init)")

    # TASK_749 (Titans-MAG): drop the nested trm/encoder/titans subtree
    # so it fresh-inits (the only case the subtree exists in the ckpt is
    # a previous Titans run; from a vanilla Phase-1 ckpt the subtree
    # simply doesn't exist yet and gets created by initialize_missing_aux_heads
    # below). Drop is per-encoder, since this lives nested under trm/encoder.
    if getattr(args, 'reset_titans_memory', False):
        dropped = False
        if 'trm' in params and 'encoder' in params['trm']:
            enc = dict(params['trm']['encoder'])
            if 'titans' in enc:
                del enc['titans']
                trm = dict(params['trm'])
                trm['encoder'] = enc
                params['trm'] = trm
                dropped = True
        if dropped:
            logger.info("Reset titans subtree (will fresh-init)")
        else:
            logger.info("Reset titans requested but no existing subtree (fresh init via aux-head init)")

    # TASK_727/728/729a/730/735/740/749/754: If enabled aux heads are missing from the loaded checkpoint, init them fresh.
    if (config.use_direction_head or
            getattr(config, 'use_direction_policy_bias', False) or
            getattr(config, 'use_per_asset_policy_head', False) or
            getattr(config, 'use_oracle_mask_head', False) or
            getattr(config, 'use_rank_head', False) or
            getattr(config, 'use_aux_price_head', False) or
            getattr(config, 'use_bella_distill_head', False) or
            getattr(config, 'use_bella_shape_head', False) or
            getattr(config, 'use_titans_memory', False)):
        aux_rng = jax.random.PRNGKey(config.seed + 9999)
        params = initialize_missing_aux_heads(model, params, config, aux_rng)

    # TASK_728b: Optionally warm the refiner kernel with action priors.
    if getattr(args, 'warm_direction_refiner', False) and config.use_direction_policy_bias:
        params = warm_policy_direction_refiner(
            params, alpha=args.direction_prior_alpha, n_assets=config.n_assets,
        )

    # --- Test init mode (no pool needed) ---
    if args.test_init:
        logger.info("Testing dynamics initialization...")
        rng_key = jax.random.PRNGKey(config.seed)
        params = initialize_dynamics(params, rng_key)

        # Verify kernels/embeddings are no longer zero (biases and LN biases stay zero)
        info_post = analyze_params(params)
        logger.info(f"After init: dynamics {info_post['dynamics_zero']}/{info_post['dynamics_total']} zero "
                    f"(4 expected: bias tensors)")
        # 9 total: 1 embedding + 2 kernels + 2 biases + 2 LN scales + 2 LN biases
        # After init: biases (4) stay zero, scales/kernels/embedding (5) should be non-zero
        expected_zero = 4  # dynamics_dense1/bias, dynamics_dense2/bias, dynamics_ln/bias, dynamics_state_norm/bias
        assert info_post['dynamics_zero'] == expected_zero, \
            f"Expected {expected_zero} zero tensors (biases), got {info_post['dynamics_zero']}"

        # Smoke test: recurrent_inference
        latent = jnp.ones((1, config.state_dim))
        action = jnp.array([42])
        out = model.apply({'params': params}, latent, action, mode='recurrent_inference')

        assert out['latent_state'].shape == (1, config.state_dim), \
            f"Expected latent (1, {config.state_dim}), got {out['latent_state'].shape}"
        assert out['reward'].shape == (1, VALUE_SUPPORT_SIZE), \
            f"Expected reward (1, {VALUE_SUPPORT_SIZE}), got {out['reward'].shape}"
        assert not jnp.any(jnp.isnan(out['latent_state'])), "NaN in latent_state!"
        assert not jnp.any(jnp.isnan(out['reward'])), "NaN in reward!"

        latent_std = float(out['latent_state'].std())
        reward_std = float(out['reward'].std())
        logger.info(f"recurrent_inference OK: latent shape={out['latent_state'].shape}, "
                    f"std={latent_std:.4f}")
        logger.info(f"  reward shape={out['reward'].shape}, std={reward_std:.4f}")

        # Batch test: verify it works with larger batch
        latent_batch = jax.random.normal(jax.random.PRNGKey(0), (16, config.state_dim))
        action_batch = jax.random.randint(jax.random.PRNGKey(1), (16,), 0, config.action_space_size)
        out_batch = model.apply({'params': params}, latent_batch, action_batch, mode='recurrent_inference')
        assert out_batch['latent_state'].shape == (16, config.state_dim)
        assert not jnp.any(jnp.isnan(out_batch['latent_state']))
        logger.info(f"Batch test OK: (16, {config.state_dim}) latent, no NaN")

        logger.info("TASK_716b gate PASSED — dynamics initialized, recurrent_inference works")
        return

    # --- Test MCTS mode (no pool needed) ---
    if args.test_mcts:
        logger.info("Testing MCTS search (TASK_716e)...")
        from mcts_trader.jax_agent.mctx_search import MuZeroMCTXSearch

        # If checkpoint has trained dynamics (Stage A output), use as-is.
        # If dynamics are still zero (raw Phase 1), initialize them first.
        info_dyn = analyze_params(params)
        if info_dyn['dynamics_zero'] == info_dyn['dynamics_total']:
            logger.info("Dynamics all zero — initializing with He normal for smoke test")
            rng_key = jax.random.PRNGKey(config.seed)
            params = initialize_dynamics(params, rng_key)

        # Create MCTS search
        search = MuZeroMCTXSearch(
            model_apply_fn=model.apply,
            num_simulations=config.mcts_sims,
            max_num_considered_actions=config.max_considered_actions,
            discount=config.mcts_discount,
            use_gumbel=config.use_gumbel,
            epsilon_greedy=config.epsilon_greedy,
        )

        # Generate synthetic observation batch (random prices → encoded features)
        # Shape: (batch, n_channels, window_size, n_assets)
        batch_size = 16
        rng_key = jax.random.PRNGKey(42)
        obs_batch = jax.random.normal(rng_key, (batch_size, config.n_channels, config.window_size, config.n_assets))

        # Run MCTS search
        rng_key, search_key = jax.random.split(rng_key)
        logger.info(f"Running MCTS: batch={batch_size}, sims={config.mcts_sims}, "
                    f"gumbel={config.use_gumbel}, max_actions={config.max_considered_actions}")

        actions, visit_counts, stats = search(params, search_key, obs_batch)

        # Validate shapes
        assert actions.shape == (batch_size,), f"Expected actions ({batch_size},), got {actions.shape}"
        assert visit_counts.shape == (batch_size, config.action_space_size), \
            f"Expected visits ({batch_size}, {config.action_space_size}), got {visit_counts.shape}"
        assert not jnp.any(jnp.isnan(actions)), "NaN in actions!"
        assert not jnp.any(jnp.isnan(visit_counts)), "NaN in visit_counts!"

        # Validate visit counts are valid probability distributions
        assert jnp.all(visit_counts >= 0), "Negative visit counts!"
        visit_sums = visit_counts.sum(axis=-1)
        assert jnp.allclose(visit_sums, 1.0, atol=0.01), \
            f"Visit counts don't sum to 1: {visit_sums}"

        # Validate root values
        root_value = stats['root_value']
        assert root_value.shape == (batch_size,), f"Expected root_value ({batch_size},), got {root_value.shape}"
        assert not jnp.any(jnp.isnan(root_value)), "NaN in root_value!"

        # Compute visit entropy
        log_visits = jnp.log(visit_counts + 1e-8)
        visit_entropy = -(visit_counts * log_visits).sum(axis=-1)
        mean_entropy = float(visit_entropy.mean())

        # Check actions are in valid range
        assert jnp.all(actions >= 0) and jnp.all(actions < config.action_space_size), \
            f"Actions out of range: min={int(actions.min())}, max={int(actions.max())}"

        # Count how many unique actions selected across batch
        n_unique = len(set(actions.tolist()))

        # Non-zero visit count analysis
        nonzero_per_sample = (visit_counts > 0.001).sum(axis=-1)
        mean_nonzero = float(nonzero_per_sample.mean())

        logger.info(f"MCTS results:")
        logger.info(f"  actions: shape={actions.shape}, range=[{int(actions.min())}, {int(actions.max())}], "
                    f"{n_unique} unique")
        logger.info(f"  visit_counts: shape={visit_counts.shape}, sum={float(visit_sums.mean()):.4f}")
        logger.info(f"  visit_entropy: mean={mean_entropy:.2f} (0=collapsed, ~5.8=uniform over 320)")
        logger.info(f"  nonzero visits: mean={mean_nonzero:.1f} actions/sample")
        logger.info(f"  root_value: mean={float(root_value.mean()):.4f}, "
                    f"std={float(root_value.std()):.4f}, "
                    f"range=[{float(root_value.min()):.4f}, {float(root_value.max()):.4f}]")

        # Gate checks from spec
        assert mean_entropy > 0, "Visit entropy is 0 — search collapsed!"
        assert mean_nonzero > 1, "Only 1 action visited — search not exploring!"

        logger.info("TASK_716e gate PASSED — MCTS produces valid actions + visit counts")
        return

    # --- Load SegmentPool ---
    pool = None
    if Path(args.periods_file).exists() and Path(args.db_path).exists():
        pool = load_segment_pool(config, args.periods_file, args.db_path,
                                 n_slots=args.n_slots,
                                 max_runs_per_group=args.max_runs_per_group)
        n_regimes = len(pool.group_keys)
        logger.info(f"SegmentPool: {n_regimes} regimes loaded")
    else:
        if not args.dry_run:
            logger.warning(f"Periods file ({args.periods_file}) or DB ({args.db_path}) not found!")
        else:
            logger.info(f"SegmentPool: skipped (periods-file or db not found)")

    # --- Dry run: report and exit ---
    if args.dry_run:
        logger.info("Dry run complete — ready for Stage A")
        return

    # --- Training ---
    if args.stage == 'A':
        if pool is None:
            logger.error("SegmentPool required for training — check --periods-file and --db-path")
            return
        train_stage_a(model, params, config, pool, args)

    elif args.stage == 'B':
        if pool is None:
            logger.error("SegmentPool required for training — check --periods-file and --db-path")
            return
        train_stage_b(model, params, config, pool, args)


if __name__ == '__main__':
    main()

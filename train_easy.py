#!/usr/bin/env python3
"""
Easy Mode Training - Expert Iteration Phase 1.

TASK_666: Supervised learning from oracle demonstrations.
On-the-fly VRAM generation - no disk I/O, fresh data every batch.

Usage:
    # Quick test
    python train_easy.py --preset small --epochs 10 --levels 0-3

    # Production training
    python train_easy.py --preset medium --epochs 100 --levels 0-14

    # Resume
    python train_easy.py --checkpoint jax_ckpt_easy/phase1_best.pkl
"""

import os
import sys
import json
import math
import warnings
import logging

# GPU Optimization settings (must be before JAX import)
_trader_dir = os.path.dirname(os.path.abspath(__file__))
_xla_cache = os.path.join(_trader_dir, '.xla_cache')
_jax_cache = os.path.join(_trader_dir, '.jax_cache')
os.makedirs(_xla_cache, exist_ok=True)
os.makedirs(_jax_cache, exist_ok=True)

os.environ['XLA_FLAGS'] = f'--xla_gpu_autotune_level=4 --xla_gpu_per_fusion_autotune_cache_dir={_xla_cache}'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress XLA warnings
os.environ['JAX_COMPILATION_CACHE_DIR'] = _jax_cache

warnings.filterwarnings('ignore', category=FutureWarning, module='jax')
# Suppress JAX TPU warning (we don't have TPU)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.ERROR)

import argparse
import time
import csv
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from functools import partial
from collections import defaultdict
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import subprocess

from easy_config import EasyConfig, get_small_config, get_medium_config, get_large_config
from easy_trainer import EasyTrainer, create_trainer
from mcts_trader.data.synthetic_market import SyntheticMarket
from mcts_trader.jax_agent.path_solver import (
    greedy_lookahead_path, optimal_path_to_policy,
    batched_greedy_lookahead_path, batched_optimal_path_to_policy,
    multipass_oracle_policy, batched_multipass_oracle_policy,  # TASK_673
    dense_sign_oracle, dense_sign_labels_to_policy,             # TASK_739
)
from mcts_trader.env.jax_encoder import (
    EncoderConfig, jit_batch_encode_temporal, jit_vmap_encode_temporal
)
from mcts_trader.training.setup import setup_bella_generator, generate_bella_prices_on_device

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# TASK_689: Agent State Channels — Self-Aware Trading
# =============================================================================

@partial(jax.jit, static_argnames=['n_assets', 'k', 'window_size'])
def build_state_channels(
    target_positions,   # (T, n_assets) from oracle {-1, 0, +1}
    returns,            # (T, n_assets) log returns
    n_assets=64,
    k=4,
    window_size=64,
    decay_fast=0.80,
    decay_slow=0.99,
):
    """Build 6 per-step state channels via lax.scan.

    Returns:
        state_channels: (T, 6, n_assets) — ready to tile per window
    """
    def step(carry, inputs):
        prev_pos, pnl, steps, h_fast, h_slow = carry
        pos, ret = inputs

        # Detect trades (position changed)
        traded = (pos != prev_pos).astype(jnp.float32)
        # Flips (long→short or reverse) count as double trade
        flipped = (prev_pos != 0) & (pos != 0) & (pos != prev_pos)
        trade_strength = traded + flipped.astype(jnp.float32)

        # Unrealized PnL: signed relative to position direction
        held = (pos == prev_pos) & (prev_pos != 0)
        signed_ret = ret * prev_pos  # long: +ret, short: -ret
        new_pnl = jnp.where(held, pnl + signed_ret, 0.0)
        new_pnl = jnp.clip(new_pnl, -1.0, 1.0)

        # Hold duration: normalized by window_size
        new_steps = jnp.where(traded > 0.5, 0.0, steps + 1.0)

        # Multi-scale heat
        new_h_fast = h_fast * decay_fast + trade_strength
        new_h_slow = h_slow * decay_slow + trade_strength

        # Global: capacity usage
        n_active = jnp.sum(jnp.abs(pos) > 0.5)
        capacity = n_active / k

        out = jnp.stack([
            pos,                                          # ch27: position
            new_pnl,                                      # ch28: unrealized PnL
            jnp.clip(new_steps / window_size, 0.0, 1.0), # ch29: hold duration
            jnp.clip(new_h_fast, 0.0, 1.0),              # ch30: heat fast
            jnp.clip(new_h_slow, 0.0, 1.0),              # ch31: heat slow
            jnp.full(n_assets, capacity),                 # ch32: capacity (broadcast)
        ])  # (6, n_assets)

        return (pos, new_pnl, new_steps, new_h_fast, new_h_slow), out

    init = tuple(jnp.zeros(n_assets) for _ in range(5))
    _, channels = jax.lax.scan(step, init, (target_positions, returns))
    return channels  # (T, 6, n_assets)


def corrupt_oracle_selections(selections, key, p_drop=0.15, q_phantom=0.05):
    """TASK_691: Inject entropy into oracle path before state channel computation.

    Trade drops: oracle selected → force unselected (simulates missed entries)
    Phantom trades: oracle didn't select → force selected (simulates false entries)

    All downstream state channels remain internally consistent because they're
    computed FROM the noisy path. Targets stay clean (uncorrupted oracle actions).
    """
    k1, k2 = jax.random.split(key)
    drop_mask = jax.random.bernoulli(k1, p_drop, selections.shape)
    noisy = jnp.where(drop_mask & (selections > 0.5), 0.0, selections)
    phantom_mask = jax.random.bernoulli(k2, q_phantom, selections.shape)
    noisy = jnp.where(phantom_mask & (selections < 0.5), 1.0, noisy)
    return noisy


def compute_target_position(selections, returns):
    """Derive oracle target position from selections and returns.

    Args:
        selections: (T, n_assets) binary mask from greedy_lookahead_path
        returns: (T, n_assets) log returns

    Returns:
        target_position: (T, n_assets) values in {-1, 0, +1}
    """
    return jnp.where(
        selections > 0.5,
        jnp.where(returns > 0, 1.0, jnp.where(returns < 0, -1.0, 0.0)),
        0.0
    )


from mcts_trader.data.synthetic_gpu import generate_synthetic_on_device

# =============================================================================
# TASK_692: Model-State Rollout (Autoregressive State Channels)
# =============================================================================

@partial(jax.jit, static_argnames=['model_apply_fn', 'n_assets', 'k', 'window_size', 'n_actions_per_asset'])
def _build_model_state_trajectory(
    market_obs,        # (T, 27, ws, n_assets) pre-encoded market channels
    returns,           # (T, n_assets) log returns
    model_params,       # Current model parameters (no gradients)
    model_apply_fn,    # model.apply function
    n_assets=64,
    k=4,
    window_size=64,
    decay_fast=0.80,
    decay_slow=0.99,
    n_actions_per_asset=5,
):
    """Roll model through episode, capture its state trajectory (TASK_692).

    This is an autoregressive rollout where each step's action becomes
    next step's state input. The model experiences off-diagonal
    states during training, matching inference behavior.

    TASK_693: Uses incremental position stepping (Buy=step toward +1,
    Sell=step toward -1) matching benchmark_easy.py inference semantics.
    Also returns model confidence from softmax for channel 33.

    Args:
        market_obs: (T, 27, ws, n_assets) pre-encoded market channels
        returns: (T, n_assets) log returns (for PnL calculation)
        model_params: Current model parameters
        model_apply_fn: model.apply function
        n_assets, k, window_size: State computation parameters
        decay_fast, decay_slow: Heat decay rates

    Returns:
        model_state: (T, 6, n_assets) state channels from model's own actions
        model_confidence: (T, n_assets) softmax confidence per asset
    """
    # Initialize neutral state (no positions, no history)
    init_state = (
        jnp.zeros(n_assets),  # position
        jnp.zeros(n_assets),  # pnl
        jnp.zeros(n_assets),  # steps
        jnp.zeros(n_assets),  # h_fast
        jnp.zeros(n_assets),  # h_slow
        jnp.zeros(n_assets),  # prev_confidence (BUG-2 fix)
    )

    def scan_step(carry, idx):
        """Single timestep in autoregressive rollout."""
        pos, pnl, steps, h_fast, h_slow, prev_conf = carry

        # Build state channels from current state
        capacity = jnp.sum(jnp.abs(pos) > 0.5) / k
        state_ch = jnp.stack([
            pos,
            pnl,
            jnp.clip(steps / window_size, 0.0, 1.0),
            jnp.clip(h_fast, 0.0, 1.0),
            jnp.clip(h_slow, 0.0, 1.0),
            jnp.full(n_assets, capacity),
        ])  # (6, n_assets)

        # Confidence from model's own softmax (computed from previous step's
        # forward pass). First step gets zero confidence (no prior action).
        # We'll compute confidence AFTER the forward pass and emit it as output.

        # Tile state across window
        ws = window_size
        state_tiled = jnp.tile(state_ch[:, None, :], (1, ws, 1))  # (6, ws, n_assets)
        # BUG-2 fix: Feed previous step's confidence into observation
        # (matches inference where prev_confidence is carried from step i-1)
        conf_tiled = jnp.tile(prev_conf[None, None, :], (1, ws, 1))

        # Construct full observation
        market_t = market_obs[idx]  # (27, ws, n_assets)
        obs = jnp.concatenate([market_t, state_tiled, conf_tiled], axis=0)  # (34, ws, n_assets)
        obs_batch = obs[None, ...]  # Add batch dim: (1, 34, ws, n_assets)

        # Run model inference (stop_gradient to prevent backprop through rollout)
        output = jax.lax.stop_gradient(
            model_apply_fn({'params': model_params}, observation=obs_batch, mode='initial_inference')
        )
        policy_logits = output['policy_logits'][0]  # (320,) remove batch

        # Get action: convert to per-asset action types
        policy_reshaped = policy_logits.reshape(n_assets, n_actions_per_asset)
        action_types = jnp.argmax(policy_reshaped, axis=-1)  # (n_assets,)

        # TASK_693: Model confidence from softmax (matches benchmark_easy.py:524-525)
        softmax_probs = jax.nn.softmax(policy_reshaped, axis=-1)  # (n_assets, 5)
        model_confidence = jnp.max(softmax_probs, axis=-1)  # (n_assets,)

        # TASK_693: Incremental position stepping (matches benchmark_easy.py:529-537)
        # Buy: short(-1)→flat(0), flat(0)→long(+1), long(+1)→no-op
        # Sell: long(+1)→flat(0), flat(0)→short(-1), short(-1)→no-op
        buy = (action_types == 2) | (action_types == 4)    # Buy or LimitBuy
        sell = (action_types == 0) | (action_types == 3)   # Sell or LimitSell
        new_pos = jnp.where(
            buy,
            jnp.minimum(pos + 1.0, 1.0),    # step toward long, cap at +1
            jnp.where(
                sell,
                jnp.maximum(pos - 1.0, -1.0),  # step toward short, cap at -1
                pos  # Hold: no change
            )
        )

        # Update state based on action and returns
        ret = returns[idx]

        # Detect trades
        traded = (new_pos != pos).astype(jnp.float32)
        flipped = (pos != 0) & (new_pos != 0) & (new_pos != pos)
        trade_strength = traded + flipped.astype(jnp.float32)

        # Update PnL
        held = (new_pos == pos) & (pos != 0)
        signed_ret = ret * pos
        new_pnl = jnp.where(held, pnl + signed_ret, 0.0)
        new_pnl = jnp.clip(new_pnl, -1.0, 1.0)

        # Update hold duration
        new_steps = jnp.where(traded > 0.5, 0.0, steps + 1.0)

        # Update heat
        new_h_fast = h_fast * decay_fast + trade_strength
        new_h_slow = h_slow * decay_slow + trade_strength

        # Emit prev_conf (not model_confidence) so confidence_trajectory[w] =
        # the confidence the model SAW at step w, matching inference alignment.
        # model_confidence propagates forward via carry for step w+1.
        return (new_pos, new_pnl, new_steps, new_h_fast, new_h_slow, model_confidence), (state_ch, prev_conf)

    T = market_obs.shape[0]
    _, (model_state_trajectory, confidence_trajectory) = jax.lax.scan(
        scan_step, init_state, jnp.arange(T), unroll=2
    )
    return model_state_trajectory, confidence_trajectory  # (T, 6, n_assets), (T, n_assets)


# TASK_699: Batched AR rollout — process B episodes in single scan call
@partial(jax.jit, static_argnames=['model_apply_fn', 'n_assets', 'k', 'window_size', 'n_actions_per_asset', 'B'])
def _build_model_state_trajectory_batched(
    market_obs,        # (B, T, 27, ws, n_assets) pre-encoded market channels
    returns,           # (B, T, n_assets) log returns
    model_params,      # Current model parameters (no gradients)
    model_apply_fn,    # model.apply function
    n_assets=64,
    k=4,
    window_size=64,
    decay_fast=0.80,
    decay_slow=0.99,
    n_actions_per_asset=5,
    B=4,               # static: fixed batch size (padded to n_envs)
):
    """Batched AR rollout: B episodes in single scan call (TASK_699).

    Same computation as _build_model_state_trajectory but with leading batch
    dimension. B is STATIC (static_argnames) to prevent JIT recompilation
    when actual AR episode count varies. Caller pads to fixed B.

    Args:
        market_obs: (B, T, 27, ws, n_assets) pre-encoded market channels
        returns: (B, T, n_assets) log returns
        model_params: Current model parameters
        model_apply_fn: model.apply function
        n_assets, k, window_size: State computation parameters
        decay_fast, decay_slow: Heat decay rates
        B: Batch size (static — must match market_obs.shape[0])

    Returns:
        model_state: (B, T, 6, n_assets) state channels from model's own actions
        model_confidence: (B, T, n_assets) softmax confidence per asset
    """
    # Initialize neutral state — all (B, n_assets)
    init_state = (
        jnp.zeros((B, n_assets)),  # position
        jnp.zeros((B, n_assets)),  # pnl
        jnp.zeros((B, n_assets)),  # steps
        jnp.zeros((B, n_assets)),  # h_fast
        jnp.zeros((B, n_assets)),  # h_slow
        jnp.zeros((B, n_assets)),  # prev_confidence
    )

    def scan_step(carry, idx):
        """Single timestep, batched across B episodes."""
        pos, pnl, steps, h_fast, h_slow, prev_conf = carry
        # All carry arrays: (B, n_assets)

        # Capacity: sum per-episode (axis=-1), not across batch
        capacity = jnp.sum(jnp.abs(pos) > 0.5, axis=-1, keepdims=True) / k  # (B, 1)

        # Build state channels — stack along channel axis (axis=1)
        state_ch = jnp.stack([
            pos,                                              # (B, n_assets)
            pnl,
            jnp.clip(steps / window_size, 0.0, 1.0),
            jnp.clip(h_fast, 0.0, 1.0),
            jnp.clip(h_slow, 0.0, 1.0),
            jnp.broadcast_to(capacity, (B, n_assets)),
        ], axis=1)  # (B, 6, n_assets)

        # Tile state across window: (B, 6, n_assets) → (B, 6, ws, n_assets)
        ws = window_size
        state_tiled = jnp.tile(state_ch[:, :, None, :], (1, 1, ws, 1))

        # Tile confidence: (B, n_assets) → (B, 1, ws, n_assets)
        conf_tiled = jnp.tile(prev_conf[:, None, None, :], (1, 1, ws, 1))

        # Construct full observation: (B, 34, ws, n_assets)
        market_t = market_obs[:, idx]  # (B, 27, ws, n_assets)
        obs = jnp.concatenate([market_t, state_tiled, conf_tiled], axis=1)

        # Model forward pass — handles any batch size natively
        output = jax.lax.stop_gradient(
            model_apply_fn({'params': model_params}, observation=obs, mode='initial_inference')
        )
        policy_logits = output['policy_logits']  # (B, 320)

        # Per-asset actions
        policy_reshaped = policy_logits.reshape(B, n_assets, n_actions_per_asset)
        action_types = jnp.argmax(policy_reshaped, axis=-1)  # (B, n_assets)

        # Model confidence from softmax
        softmax_probs = jax.nn.softmax(policy_reshaped, axis=-1)  # (B, n_assets, 5)
        model_confidence = jnp.max(softmax_probs, axis=-1)  # (B, n_assets)

        # Incremental position stepping (TASK_693)
        buy = (action_types == 2) | (action_types == 4)
        sell = (action_types == 0) | (action_types == 3)
        new_pos = jnp.where(
            buy,
            jnp.minimum(pos + 1.0, 1.0),
            jnp.where(sell, jnp.maximum(pos - 1.0, -1.0), pos)
        )

        # State update
        ret = returns[:, idx]  # (B, n_assets)
        traded = (new_pos != pos).astype(jnp.float32)
        flipped = (pos != 0) & (new_pos != 0) & (new_pos != pos)
        trade_strength = traded + flipped.astype(jnp.float32)

        held = (new_pos == pos) & (pos != 0)
        signed_ret = ret * pos
        new_pnl = jnp.where(held, pnl + signed_ret, 0.0)
        new_pnl = jnp.clip(new_pnl, -1.0, 1.0)

        new_steps = jnp.where(traded > 0.5, 0.0, steps + 1.0)
        new_h_fast = h_fast * decay_fast + trade_strength
        new_h_slow = h_slow * decay_slow + trade_strength

        # Emit prev_conf (BUG-2 alignment: confidence_trajectory[w] = what model SAW at step w)
        return (new_pos, new_pnl, new_steps, new_h_fast, new_h_slow, model_confidence), (state_ch, prev_conf)

    T = market_obs.shape[1]  # time axis is dim 1 (dim 0 is batch)
    _, (model_state_trajectory, confidence_trajectory) = jax.lax.scan(
        scan_step, init_state, jnp.arange(T), unroll=2
    )
    # scan stacks along new leading axis: (T, B, 6, n_assets), (T, B, n_assets)
    # Transpose to caller-expected (B, T, 6, n_assets), (B, T, n_assets)
    model_state_trajectory = jnp.transpose(model_state_trajectory, (1, 0, 2, 3))
    confidence_trajectory = jnp.transpose(confidence_trajectory, (1, 0, 2))

    return model_state_trajectory, confidence_trajectory


# =============================================================================
# Asynchronous Data Generation (CPU Background Thread, GPU Foreground)
# =============================================================================

# =============================================================================
# TASK_710: Segment Pool — Linear Historical Training
# =============================================================================

class SegmentPool:
    """Pool of active historical segments with linear cursors.

    Maintains N_SLOTS segments in RAM from different date ranges (regime diversity).
    Each segment has a linear cursor that walks sequentially through timestamps.
    When a segment is exhausted, it's replaced with the next run from the same
    date range (same regime, different assets), then a new date range when the
    group is fully exhausted.

    This ensures:
    - Full data coverage (one pass through all 70M timestamps)
    - Linear coherency (sequential walk, not random sampling)
    - Regime diversity (slots from different date ranges)
    - Bounded RAM (~1.9GB for 6 slots × ~316MB each)
    """

    def __init__(self, period_specs, db_path, n_assets=64, n_slots=6, window_length=589,
                 max_runs_per_group=0, min_assets=32, oracle_config=None):
        """
        Args:
            period_specs: List of dicts with start_date, end_date, assets
            db_path: SQLite database path
            n_assets: Assets per segment (for cyclic padding)
            n_slots: Number of concurrent segments in RAM
            window_length: Episode length for get_window (stride = window_length // 2)
            max_runs_per_group: TASK_712: Cap runs per date range (0=unlimited).
                Reduces asset diversity but speeds up training proportionally.
            min_assets: TASK_712: Skip periods with fewer assets than this.
                Tiny periods (e.g. 2 assets padded to 64) are ~97% duplicate data.
            oracle_config: TASK_714: Dict with oracle params for full-period precomputation.
                Keys: oracle_k, oracle_lookahead, transaction_cost, min_alpha_threshold
        """
        self.db_path = db_path
        self.n_assets = n_assets
        self.n_slots = n_slots
        self.window_length = window_length
        self._stride = max(window_length // 2, 1)
        # TASK_714: Oracle config for full-period precomputation
        self._oracle_config = oracle_config or {}

        # Group specs by date range
        self.groups = {}  # (start_date, end_date) → [spec, spec, ...]
        self.group_keys = []  # Ordered list of (start_date, end_date)
        for spec in period_specs:
            key = (spec['start_date'], spec['end_date'])
            if key not in self.groups:
                self.groups[key] = []
                self.group_keys.append(key)
            self.groups[key].append(spec)

        # TASK_712: Filter tiny periods (too few assets → mostly cyclic-pad duplicates)
        if min_assets > 0:
            tiny_groups = []
            for key in self.group_keys:
                # Check first run's asset count
                first_spec = self.groups[key][0]
                n_spec_assets = len(first_spec.get('assets') or [])
                if 0 < n_spec_assets < min_assets:
                    tiny_groups.append(key)
            for key in tiny_groups:
                n_skipped = len(self.groups[key])
                logger.info(f"[POOL] Skipping {key[0][-10:]}→{key[1][-10:]}: "
                            f"{len(self.groups[key][0].get('assets', []))} assets < {min_assets} min "
                            f"({n_skipped} runs)")
                del self.groups[key]
                self.group_keys.remove(key)

        # TASK_712: Cap runs per group for faster training
        n_total_before = sum(len(v) for v in self.groups.values())
        if max_runs_per_group > 0:
            for key in self.groups:
                self.groups[key] = self.groups[key][:max_runs_per_group]

        n_groups = len(self.group_keys)
        n_total = sum(len(v) for v in self.groups.values())
        if max_runs_per_group > 0 and n_total < n_total_before:
            logger.info(f"[POOL] Capped to {max_runs_per_group} runs/group: "
                        f"{n_total_before}→{n_total} segments")
        logger.info(f"[POOL] {n_total} segments across {n_groups} date ranges, {n_slots} slots")

        # Per-group tracking: which run index we're on
        self._group_run_idx = {k: 0 for k in self.group_keys}
        # TASK_712: Track completed passes per group (for logging)
        self._passes_completed = {k: 0 for k in self.group_keys}
        # Queue of groups not yet assigned to a slot
        self._available_groups = list(self.group_keys)

        # Initialize slots
        self.slots = []  # List of slot dicts
        self._round_robin_idx = 0
        self.rng = np.random.default_rng(42)
        n_init = min(n_slots, n_groups)
        for i in range(n_init):
            group_key = self._available_groups.pop(0)
            slot = self._make_slot(group_key)
            if slot is not None:
                self.slots.append(slot)
            else:
                self._exhausted_groups.add(group_key)

        if not self.slots:
            raise ValueError("No valid segments could be loaded!")

        logger.info(f"[POOL] Initialized {len(self.slots)} slots from {n_init} groups")

        # Stats
        self._segments_loaded = len(self.slots)
        self._groups_started = n_init

    def _make_slot(self, group_key):
        """Create a slot from the next available run in a group."""
        from mcts_trader.data.history import HistoricalMarket
        runs = self.groups[group_key]

        while self._group_run_idx[group_key] < len(runs):
            run_idx = self._group_run_idx[group_key]
            spec = runs[run_idx]
            label = f"{spec['start_date']}→{spec['end_date']} run {run_idx+1}/{len(runs)}"
            try:
                hm = HistoricalMarket(
                    db_path=self.db_path,
                    n_assets=self.n_assets,
                    specific_symbols=spec.get('assets'),
                    start_date=spec['start_date'],
                    end_date=spec['end_date'],
                    use_ram_cache=True,
                    skip_validation=True,
                )
                hm.close_db()
            except Exception as e:
                logger.warning(f"[POOL] Failed to load {label}: {e}")
                self._group_run_idx[group_key] = run_idx + 1
                continue

            n_timestamps = len(hm.valid_timestamps)
            # Cyclic-pad assets if fewer than n_assets
            if hm.cache_data.shape[1] < self.n_assets:
                n_have = hm.cache_data.shape[1]
                repeats = (self.n_assets + n_have - 1) // n_have
                hm.cache_data = np.tile(hm.cache_data, (1, repeats, 1))[:, :self.n_assets, :]

            # TASK_714: Precompute oracle on full period
            oracle_sel, oracle_cum_pnl, returns = self._precompute_oracle(hm, label)

            # TASK_715: Precompute state channels, policy targets, confidence
            T_ret = len(returns)
            oc = self._oracle_config
            oracle_sel_jax = jnp.array(oracle_sel)
            returns_jax = jnp.array(returns)

            # State channels from oracle trajectory
            target_pos = compute_target_position(oracle_sel_jax, returns_jax)
            state_ch_full = build_state_channels(
                target_pos, returns_jax,
                n_assets=self.n_assets,
                k=oc.get('oracle_k', 4),
                window_size=self.window_length,  # approximate ws
            )
            jax.block_until_ready(state_ch_full)
            state_ch_np = np.array(state_ch_full).astype(np.float16)  # (T_ret, 6, n_assets)

            # Policy targets (state-aware)
            prev_sel = jnp.zeros((T_ret, self.n_assets))
            prev_sel = prev_sel.at[1:].set(
                jnp.where(oracle_sel_jax[:-1] > 0.5,
                          jnp.where(returns_jax[:-1] > 0, 1.0, -1.0), 0.0)
            )
            # TASK_739: select BC target oracle. "topk_path" (default) keeps the
            # pre-739 state-aware top-k policy. "dense_sign" replaces the BC
            # target with per-asset sign-of-forward-return labels (every asset,
            # every step). oracle_sel / oracle_cum_pnl are unchanged — still
            # top-k for value / per-step reward / n-step returns.
            oracle_mode = oc.get('oracle_mode', 'topk_path')
            if oracle_mode == 'dense_sign':
                # Match the max_len formula used by greedy_lookahead_path in
                # _precompute_oracle so JIT compiles once (pad to ≥ 262144,
                # rounded up to a multiple of 1024).
                ds_max_len = max(262144, ((T_ret + 1023) // 1024) * 1024)
                dense_labels = dense_sign_oracle(
                    returns_jax,
                    lookahead=oc.get('oracle_lookahead', 20),
                    epsilon=oc.get('sign_epsilon', 1e-4),
                    max_len=ds_max_len,
                )
                policy_targets_full = dense_sign_labels_to_policy(
                    dense_labels, n_actions_per_asset=5,
                )
            elif oracle_mode == 'hybrid':
                raise NotImplementedError(
                    "oracle_mode='hybrid' is reserved — TODO(TASK_739 follow-up). "
                    "Expected semantics: top-k drives value signal, dense-sign drives BC."
                )
            else:
                policy_targets_full = optimal_path_to_policy(
                    oracle_sel_jax, returns_jax,
                    prev_selections=prev_sel,
                    n_actions_per_asset=5, temperature=1.0,
                )
            jax.block_until_ready(policy_targets_full)
            policy_np = np.array(policy_targets_full).astype(np.float16)  # (T_ret, n_assets*5)

            # Confidence (max softmax per asset)
            conf_full = np.max(
                policy_np.reshape(T_ret, self.n_assets, 5), axis=-1
            ).astype(np.float16)  # (T_ret, n_assets)

            mb_hm = hm.cache_data.nbytes / 1024**2
            mb_precomp = (oracle_sel.nbytes + oracle_cum_pnl.nbytes + returns.nbytes +
                          state_ch_np.nbytes + policy_np.nbytes + conf_full.nbytes) / 1024**2
            logger.info(f"[POOL] Loaded {label}: {n_timestamps:,} ts, "
                        f"{mb_hm:.0f}+{mb_precomp:.0f} MB (data+precomp)")

            # TASK_733: cheat-channel visibility — fraction of channel-32 values
            # in {−1, 0, +1} for this segment. Uses the same log-return stream
            # the encoder ingests. Logs only when the diagnostic is enabled.
            if oc.get('use_cheat_channel', False):
                la = int(oc.get('cheat_lookahead', 20))
                eps = float(oc.get('cheat_epsilon', 1e-4))
                if T_ret > la:
                    # Forward cum log-return: r_fwd[t] = sum(returns[t+1:t+1+la]).
                    # np.cumsum then diff: cs[t+la] - cs[t] → (T_ret-la,) aligned to t.
                    cs = np.cumsum(np.asarray(returns), axis=0)
                    r_fwd = cs[la:] - cs[:-la]  # (T_ret-la, n_assets)
                    frac_pos = float(np.mean(r_fwd > eps))
                    frac_neg = float(np.mean(r_fwd < -eps))
                    frac_hold = 1.0 - frac_pos - frac_neg
                    logger.info(
                        f"[POOL:CHEAT] {label}: la={la}, eps={eps:.1e} — "
                        f"+1={frac_pos:.1%}  -1={frac_neg:.1%}  hold={frac_hold:.1%}  "
                        f"(non-HOLD={frac_pos + frac_neg:.1%})"
                    )

            max_cursor = max(n_timestamps - self.window_length, 0)

            # TASK_740: Load precomputed Bella predictions for this period if
            # the distillation head is enabled and the manifest lists a shard.
            # Shape: (T_ret, n_horizons, n_assets) of log-returns. Missing
            # shards fall back to None and the trainer masks those samples.
            bella_preds_full = None
            if oc.get('use_bella_distill_head', False):
                # TASK_745: tag must include the rotation hash so the consumer
                # only picks up shards whose column semantics actually match
                # this rotation's trainer-axis.
                bella_preds_full = self._load_bella_shard(group_key, T_ret,
                                                         oc.get('bella_pred_horizons', (1, 5, 10, 20)),
                                                         manifest_path=oc.get('bella_preds_manifest',
                                                                               'config/bella_preds_manifest.json'),
                                                         period_assets=spec.get('assets'))

            # TASK_754: Load precomputed Bella trajectory-shape descriptors for
            # this period (additive to TASK_740). Shape: (T_ret, descriptor_dim)
            # float32. Missing shards fall back to None and the trainer masks
            # those samples sample-wise.
            bella_shape_full = None
            if oc.get('use_bella_shape_head', False):
                bella_shape_full = self._load_bella_shape_shard(
                    group_key, T_ret,
                    descriptor_dim=oc.get('bella_shape_descriptor_dim', 14),
                    manifest_path=oc.get('bella_shape_manifest',
                                         'config/bella_shape_manifest.json'),
                    period_assets=spec.get('assets'),
                )

            return {
                'hm': hm,
                'cursor': 0,
                'max_cursor': max_cursor,
                'group_key': group_key,
                'run_idx': run_idx,
                'label': label,
                # TASK_714: Precomputed oracle
                'oracle_selections': oracle_sel,   # (T-1, n_assets)
                'oracle_cum_pnl': oracle_cum_pnl,  # (T-1,)
                'returns': returns,                 # (T-1, n_assets)
                # TASK_715: Precomputed state/targets/confidence (float16)
                'state_channels': state_ch_np,     # (T-1, 6, n_assets) fp16
                'policy_targets': policy_np,        # (T-1, n_assets*5) fp16
                'confidence': conf_full,            # (T-1, n_assets) fp16
                # TASK_740: Precomputed Bella distillation targets (fp32 or None)
                'bella_preds_full': bella_preds_full,
                # TASK_754: Precomputed Bella trajectory-shape descriptors (fp32 or None)
                'bella_shape_full': bella_shape_full,
            }

        return None  # Group exhausted

    def _precompute_oracle(self, hm, label):
        """TASK_714: Compute oracle on full period. Returns numpy arrays."""
        close = hm.cache_data[:, :, 3]  # (T, n_assets)
        returns = np.log(np.maximum(close[1:], 1e-8) / np.maximum(close[:-1], 1e-8))
        T_ret = returns.shape[0]

        oc = self._oracle_config
        # Use fixed max_len to minimize JIT recompilations (round up to 262144 = 2^18)
        max_len = max(262144, ((T_ret + 1023) // 1024) * 1024)

        logger.info(f"[POOL] Computing oracle for {label}: {T_ret:,} steps (max_len={max_len})")
        t0 = time.time()
        oracle_result = greedy_lookahead_path(
            jnp.array(returns),
            k=oc.get('oracle_k', 4),
            lookahead=oc.get('oracle_lookahead', 20),
            transaction_cost=oc.get('transaction_cost', 0.001),
            min_alpha_threshold=oc.get('min_alpha_threshold', 0.0005),
            max_len=max_len,
        )
        # Block until ready, then transfer to numpy (RAM)
        jax.block_until_ready(oracle_result.selections)
        oracle_sel = np.array(oracle_result.selections[:T_ret])
        oracle_cum_pnl = np.array(oracle_result.cumulative_pnl[:T_ret])
        logger.info(f"[POOL] Oracle computed in {time.time() - t0:.1f}s "
                    f"(trades={int(np.sum(np.diff(oracle_sel, axis=0) != 0) / 2)})")

        return oracle_sel, oracle_cum_pnl, returns.astype(np.float32)

    def _load_bella_shard(self, group_key, T_ret, horizons, manifest_path,
                          period_assets=None):
        """TASK_740: Load a Bella predictions shard for this group.

        Shard format (NPZ):
          preds     — (n_windows, n_horizons, n_assets) log-returns, float32
          horizons  — (n_horizons,) int32
          generated_at, bella_checkpoint_sha — metadata

        Returns an array of shape (T_ret, n_horizons, n_assets) aligned with
        decision_indices, or None if no shard is registered / loadable. The
        trainer masks samples whose rows contain NaN, so pad-right with NaN
        when n_windows < T_ret and truncate when n_windows > T_ret.
        """
        import json
        manifest_p = Path(manifest_path)
        if not manifest_p.exists():
            logger.warning(f"[POOL] Bella manifest missing at {manifest_p} — "
                           f"distillation loss will be fully masked out.")
            return None
        try:
            with open(manifest_p) as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning(f"[POOL] Failed to read Bella manifest: {e}")
            return None

        # TASK_745: rotation-aware tag. Legacy '<start>__<end>' shards are
        # ignored intentionally — they were column-mismatched and must be
        # re-precomputed under the new (Option A) remap.
        import hashlib as _hashlib
        rot_hash = _hashlib.sha256(
            ",".join(sorted(period_assets or [])).encode()
        ).hexdigest()[:8]
        period_tag = f"{group_key[0]}__{group_key[1]}__{rot_hash}"
        entry = manifest.get('shards', {}).get(period_tag)
        if entry is None:
            return None

        shard_path = Path(entry['path'])
        if not shard_path.is_absolute():
            # Shard paths in the manifest are relative to the repo root where
            # precompute_bella_predictions.py was run.
            shard_path = Path.cwd() / shard_path
        if not shard_path.exists():
            logger.warning(f"[POOL] Bella shard missing: {shard_path}")
            return None

        try:
            data = np.load(shard_path)
            preds = data['preds']
            shard_horizons = tuple(int(h) for h in data['horizons'])
        except Exception as e:
            logger.warning(f"[POOL] Bella shard {shard_path} load failed: {e}")
            return None

        if shard_horizons != tuple(horizons):
            logger.warning(f"[POOL] Bella horizons mismatch in {shard_path}: "
                           f"shard={shard_horizons} cfg={horizons}")
            return None

        n_windows = preds.shape[0]
        n_horizons = len(horizons)
        out = np.full((T_ret, n_horizons, self.n_assets), np.nan, dtype=np.float32)
        take = min(n_windows, T_ret)
        out[:take] = preds[:take].astype(np.float32)
        return out

    def _load_bella_shape_shard(self, group_key, T_ret, descriptor_dim,
                                manifest_path, period_assets=None):
        """TASK_754: Load a Bella trajectory-shape descriptor shard for this group.

        Shard format (NPZ):
          descriptor      — (n_windows, descriptor_dim) float32
          schema_version  — int (must equal 1)
          generated_at, bella_checkpoint_sha, rotation_hash — metadata

        Returns an array of shape (T_ret, descriptor_dim) aligned with
        decision_indices, or None if no shard is registered / loadable. The
        trainer masks samples whose rows contain NaN (mirrors TASK_740 path).
        """
        import json
        manifest_p = Path(manifest_path)
        if not manifest_p.exists():
            logger.warning(f"[POOL] Bella shape manifest missing at {manifest_p} — "
                           f"shape-distill loss will be fully masked out.")
            return None
        try:
            with open(manifest_p) as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning(f"[POOL] Failed to read Bella shape manifest: {e}")
            return None

        # TASK_745: rotation-aware tag — same hash construction as the legacy
        # bella_preds loader, so a single rotation gets the same tag in both
        # manifests.
        import hashlib as _hashlib
        rot_hash = _hashlib.sha256(
            ",".join(sorted(period_assets or [])).encode()
        ).hexdigest()[:8]
        period_tag = f"{group_key[0]}__{group_key[1]}__{rot_hash}"
        entry = manifest.get('shards', {}).get(period_tag)
        if entry is None:
            return None

        shard_path = Path(entry['path'])
        if not shard_path.is_absolute():
            shard_path = Path.cwd() / shard_path
        if not shard_path.exists():
            logger.warning(f"[POOL] Bella shape shard missing: {shard_path}")
            return None

        try:
            data = np.load(shard_path)
            desc = data['descriptor']
            shard_schema = int(data['schema_version']) if 'schema_version' in data.files else -1
        except Exception as e:
            logger.warning(f"[POOL] Bella shape shard {shard_path} load failed: {e}")
            return None

        if shard_schema != 1:
            logger.warning(f"[POOL] Bella shape shard {shard_path} schema_version="
                           f"{shard_schema}, expected 1 — refusing to use.")
            return None
        if desc.shape[-1] != int(descriptor_dim):
            logger.warning(f"[POOL] Bella shape shard {shard_path} descriptor_dim="
                           f"{desc.shape[-1]}, expected {descriptor_dim} — refusing to use.")
            return None

        n_windows = desc.shape[0]
        out = np.full((T_ret, int(descriptor_dim)), np.nan, dtype=np.float32)
        take = min(n_windows, T_ret)
        out[:take] = desc[:take].astype(np.float32)
        return out

    def get_episode_range(self, episode_length, window_size):
        """TASK_714: Get next contiguous episode range from round-robin slot.

        Returns:
            slot: slot dict (or None if all exhausted)
            cursor_start: starting timestep for this episode
            n_steps: number of valid training windows in this episode
        """
        if not self.slots:
            return None, 0, 0

        slot_idx = self._round_robin_idx % len(self.slots)
        self._round_robin_idx += 1
        slot = self.slots[slot_idx]

        cursor = slot['cursor']
        T_ret = len(slot['returns'])
        # Last valid decision index: T_ret - 1
        # Decision index for cursor c with window_size ws: c + ws - 1
        # For episode of ep_len steps starting at c: last decision at c + ep_len - 1 + ws - 1
        # Need: c + ep_len - 1 + ws - 1 < T_ret → c < T_ret - ep_len - ws + 2
        max_start = T_ret - episode_length - window_size + 2
        if max_start <= 0:
            max_start = 1  # Segment too short, use what we can

        # Clamp episode length if near end of segment
        avail = max(T_ret - cursor - window_size + 1, 0)
        n_steps = min(episode_length, avail)

        if n_steps <= 0:
            # Segment exhausted — rotate
            self._rotate_slot(slot_idx)
            # Retry with new slot
            slot_idx = (self._round_robin_idx - 1) % len(self.slots) if self.slots else 0
            if not self.slots:
                return None, 0, 0
            slot = self.slots[slot_idx]
            cursor = slot['cursor']
            avail = max(len(slot['returns']) - cursor - window_size + 1, 0)
            n_steps = min(episode_length, avail)
            if n_steps <= 0:
                return None, 0, 0

        cursor_start = cursor
        # Advance cursor by n_steps (stride=1 within episode, stride=episode_length between episodes)
        slot['cursor'] += n_steps

        return slot, cursor_start, n_steps

    def get_window(self):
        """Get next sequential window from round-robin slot.

        Returns:
            prices: (window_length, n_assets, 5) OHLCV, or None if all exhausted
            slot_label: str describing current slot (for logging)
        """
        if not self.slots:
            return None, "exhausted"

        # Round-robin slot selection
        slot_idx = self._round_robin_idx % len(self.slots)
        self._round_robin_idx += 1
        slot = self.slots[slot_idx]

        # Get sequential window at cursor position
        hm = slot['hm']
        start = slot['cursor']
        prices = hm.cache_data[start:start + self.window_length]

        # Advance cursor
        slot['cursor'] += self._stride

        # Check if segment exhausted
        if slot['cursor'] > slot['max_cursor']:
            logger.info(f"[POOL] Segment exhausted: {slot['label']} "
                        f"(walked {slot['cursor']}/{slot['max_cursor']} with stride {self._stride})")
            self._rotate_slot(slot_idx)

        # Cyclic-pad if fewer assets than needed
        if prices.shape[1] < self.n_assets:
            n_have = prices.shape[1]
            repeats = (self.n_assets + n_have - 1) // n_have
            prices = np.tile(prices, (1, repeats, 1))[:, :self.n_assets, :]

        return prices, slot['label']

    def _rotate_slot(self, slot_idx):
        """Replace exhausted slot with next segment. Cycles indefinitely."""
        old_slot = self.slots[slot_idx]
        group_key = old_slot['group_key']
        runs = self.groups[group_key]
        next_run = old_slot['run_idx'] + 1

        if next_run < len(runs):
            # Next run in group — different assets, must reload
            old_slot['hm'] = None
            self._group_run_idx[group_key] = next_run
            new_slot = self._make_slot(group_key)
            if new_slot is not None:
                self.slots[slot_idx] = new_slot
                self._segments_loaded += 1
                return

        # Group fully exhausted — wrap around to run 0
        self._passes_completed[group_key] = self._passes_completed.get(group_key, 0) + 1
        passes = self._passes_completed[group_key]
        self._group_run_idx[group_key] = 0

        if len(runs) == 1:
            # Single-run group: just reset cursor, no disk I/O
            old_slot['cursor'] = 0
            if passes % 50 == 0:  # Log sparingly for fast-cycling small segments
                logger.info(f"[POOL] {old_slot['label']} pass {passes} — cursor reset")
            return

        # Multi-run group: load run 0 (different assets from current run)
        old_slot['hm'] = None
        new_slot = self._make_slot(group_key)
        if new_slot is not None:
            self.slots[slot_idx] = new_slot
            self._segments_loaded += 1
            logger.info(f"[POOL] Group {group_key[0][-10:]}→{group_key[1][-10:]} "
                        f"pass {passes}, cycling to run 1/{len(runs)}")
            return

        # Should never happen — group had at least 1 valid run on init
        logger.error(f"[POOL] Group {group_key} failed to reload on cycle — removing slot")
        self.slots.pop(slot_idx)
        if self._round_robin_idx > 0:
            self._round_robin_idx -= 1

    @property
    def is_exhausted(self):
        """Always False — pool cycles indefinitely."""
        return False

    def sample_positions(self, n_samples, window_size, jitter_interval=50):
        """TASK_715: Get n_samples (slot_idx, cursor) positions, round-robin across slots.

        Sequential cursor within each slot (stride=1). Temporal jitter every
        jitter_interval steps to prevent position memorization.
        """
        positions = []
        if not self.slots:
            return positions

        n_slots = len(self.slots)
        per_slot = n_samples // n_slots + 1

        for i in range(n_slots):
            slot_idx = (self._round_robin_idx + i) % n_slots
            slot = self.slots[slot_idx]
            T_ret = len(slot['returns'])

            for _ in range(per_slot):
                if len(positions) >= n_samples:
                    break
                cursor = slot['cursor']
                # Bounds check: decision_idx = cursor + ws - 1 must be < T_ret
                if cursor + window_size - 1 >= T_ret:
                    self._rotate_slot(slot_idx)
                    if slot_idx >= len(self.slots):
                        break
                    slot = self.slots[slot_idx]
                    T_ret = len(slot['returns'])
                    continue

                positions.append((slot_idx, cursor))
                slot['cursor'] += 1

                # Temporal jitter
                slot['_step_count'] = slot.get('_step_count', 0) + 1
                if slot['_step_count'] % jitter_interval == 0:
                    slot['cursor'] = min(slot['cursor'] + self.rng.integers(0, 6),
                                         T_ret - window_size)

        self._round_robin_idx += n_slots
        return positions[:n_samples]

    def status(self):
        """Return a status string for logging."""
        if not self.slots:
            return "EXHAUSTED"
        parts = []
        for i, s in enumerate(self.slots):
            pct = (s['cursor'] / max(s['max_cursor'], 1)) * 100
            parts.append(f"S{i}:{s['group_key'][0][-5:]} {pct:.0f}%")
        return " | ".join(parts)


class OnTheFlyGenerator:
    """
    Generate training batches on-the-fly in VRAM with background pre-fetching.

    Pipeline:
    1. Background Thread (CPU): Regenerate SyntheticMarket → Raw Price Matrix (RAM queue)
       OR: Bella (GPU): Generate prices directly on device
    2. Main Thread (GPU): Raw Price Matrix → JAX Array → Oracle → Encoder (VRAM)
    3. Main Thread (GPU): Sample windows from VRAM buffer for training.

    Uses threading (not multiprocessing) to avoid CUDA context issues.
    Thread-safety: Each worker thread gets its own SyntheticMarket instances.
    """

    def __init__(
        self,
        config: EasyConfig,
        seed: int = 42,
        bella_generator=None,
        n_envs: int = 8,
        buffer_size: int = 32,
        samples_per_episode: int = 200,
        mix_ratio: float = 0.0,
        secondary_bella_generator=None,
        secondary_levels: list = None,
        invert_regime: bool = False,
        historical_market=None,
        historical_markets: list = None,
        historical_market_labels: list = None,
        period_specs: list = None,
        db_path: str = None,
        segment_pool: 'SegmentPool' = None,
    ):
        """
        Initialize on-the-fly data generator.

        Args:
            config: Training config
            seed: Random seed
            bella_generator: Bella generator (if using Bella as primary)
            n_envs: Parallel environments
            buffer_size: Episodes in VRAM buffer
            samples_per_episode: Windows per episode before refresh
            mix_ratio: Fraction of data from secondary source (0.1 = 10%)
            secondary_bella_generator: Bella generator for mixing (if primary is synthetic)
            secondary_levels: Synthetic levels for mixing (if primary is Bella)
            invert_regime: If True, swap crash↔rally for dual-branch diversity
            historical_market: HistoricalMarket instance (TASK_698: historical data training)
            historical_markets: TASK_707: List of HistoricalMarket instances for period cycling
            historical_market_labels: TASK_707: Human-readable labels for each market (for logging)
            period_specs: Lazy loading specs — list of dicts with start_date/end_date/assets
            db_path: Database path for lazy loading
            segment_pool: TASK_710: SegmentPool for linear historical training
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.bella_generator = bella_generator
        # TASK_710: SegmentPool replaces lazy loading and period cycling
        self.segment_pool = segment_pool
        # Lazy period loading: store specs, load one at a time (legacy, pre-TASK_710)
        self._period_specs = period_specs or []
        self._db_path = db_path
        self._historical_period_idx = 0
        self._n_periods = len(self._period_specs) if self._period_specs else 0
        # Legacy pre-loaded markets (kept for backward compat)
        self._historical_markets = historical_markets or []
        self._historical_market_labels = historical_market_labels or []
        if segment_pool is not None:
            # TASK_710: Pool manages segments — no single historical_market needed
            self.historical_market = None
        elif self._period_specs:
            # Lazy mode: load first period on-demand
            self.historical_market = self._load_period(0)
        elif self._historical_markets:
            self.historical_market = self._historical_markets[0]
        else:
            self.historical_market = historical_market
        # TASK_711: Explicit flag — True when any historical source is active (pool, lazy, or single)
        self._use_historical = (segment_pool is not None or self.historical_market is not None)
        self.n_envs = n_envs  # TASK_140: Parallel environments
        self.invert_regime = invert_regime  # Dual-Bella: swap crash↔rally
        self.jax_rng = jax.random.PRNGKey(seed)
        # TASK_733: propagate cheat-channel config to encoder. Leaves encoder's
        # own n_channels default (32); the cheat writer is gated on use_cheat_channel.
        use_cheat = getattr(config, 'use_cheat_channel', False)
        encoder_n_channels = 33 if use_cheat else 32
        self.encoder_config = EncoderConfig(
            n_assets=config.n_assets,
            window_size=config.window_size,
            n_channels=encoder_n_channels,
            use_cheat_channel=use_cheat,
            cheat_lookahead=getattr(config, 'oracle_lookahead', 20),
        )
        self._cheat_lookahead = getattr(config, 'oracle_lookahead', 20) if use_cheat else 0

        # Configurable parameters (TASK_667 fix)
        self.N_BUFFER = buffer_size
        self.SAMPLES_PER_EPISODE = samples_per_episode
        # Scale workers with buffer size (min 2, max 8)
        self.N_WORKERS = max(2, min(8, buffer_size // 4))

        # TASK_691: Oracle entropy — epoch-dependent noise schedule
        self._current_epoch = 0

        # TASK_692: Model-state rollout parameters
        self._model_params = None  # Set by set_model_params() for autoregressive rollout
        self._model_apply_fn = None  # Cached model.apply function
        self._model_state_rollout = _build_model_state_trajectory  # Already @jax.jit decorated

        # 90/10 Experience Replay (TASK_667)
        self.mix_ratio = mix_ratio
        self.secondary_bella_generator = secondary_bella_generator
        self.secondary_levels = secondary_levels or [0]
        self._mix_counter = 0

        # Latent state caching (set by set_encoder after trainer is created)
        self._encoder_fn = None
        self._use_latent_cache = False

        # TASK_692: Partial FIFO refresh tracking
        # Track which buffer entries were generated with which params
        self._buffer_param_ids = []  # List of param version IDs

        # Background pre-fetching setup (threading, not multiprocessing)
        self.price_queue = Queue(maxsize=self.N_BUFFER)
        self.shutdown_event = threading.Event()

        # Thread-local storage for SyntheticMarket instances (TASK_667 thread-safety fix)
        self._thread_local = threading.local()
        self._market_lock = threading.Lock()
        self._base_seed = seed
        self._thread_counter = 0  # TASK_675: Atomic counter for unique thread seeds

        # TASK_692: Counter for model parameter versions (for FIFO refresh)
        self._param_version = 0

        # GPU synthetic generation: generate prices on-device (no CPU→GPU PCIe transfer)
        self._use_gpu_synth = (bella_generator is None and not self._use_historical)
        self._synth_n_steps = config.episode_length + config.window_size + config.oracle_lookahead + 5

        # Start background thread for price generation (only if not using GPU synth or historical)
        self.executor = ThreadPoolExecutor(max_workers=self.N_WORKERS)
        if bella_generator is None and not self._use_historical and not self._use_gpu_synth:
            self._submit_prefetch_tasks()  # CPU fallback path

        # Buffer: pre-computed episode data in VRAM
        self.buffer = []
        self.buffer_sample_counts = []

        # Initial refill (skip when pool is active — direct pipeline doesn't use buffer)
        if self.segment_pool is None:
            self._refill_buffer()
        self.global_step = 0

    def _load_period(self, idx: int):
        """Lazy-load a single historical period by index. Frees previous market."""
        from mcts_trader.data.history import HistoricalMarket
        spec = self._period_specs[idx]
        label = f"{spec['start_date']}→{spec['end_date']}"
        logger.info(f"[PERIOD] Loading P{idx}/{self._n_periods}: {label}")
        hm = HistoricalMarket(
            db_path=self._db_path,
            n_assets=self.config.n_assets,
            specific_symbols=spec.get('assets'),
            start_date=spec['start_date'],
            end_date=spec['end_date'],
            use_ram_cache=True,
            skip_validation=True,
        )
        hm.close_db()
        mb = hm.cache_data.nbytes / 1024**2
        logger.info(f"[PERIOD] Loaded P{idx}: {len(hm.valid_timestamps):,} timestamps, {hm.n_assets} assets, {mb:.1f} MB")
        return hm

    def advance_period(self):
        """TASK_707: Cycle to next historical period (round-robin). Lazy-loads from disk."""
        if self._period_specs:
            # Lazy loading path
            self._historical_period_idx = (self._historical_period_idx + 1) % self._n_periods
            # Free previous market before loading new one
            self.historical_market = None
            self.historical_market = self._load_period(self._historical_period_idx)
        elif self._historical_markets:
            # Legacy pre-loaded path
            n = len(self._historical_markets)
            self._historical_period_idx = (self._historical_period_idx + 1) % n
            self.historical_market = self._historical_markets[self._historical_period_idx]
            label = self._historical_market_labels[self._historical_period_idx] if self._historical_market_labels else f"P{self._historical_period_idx}"
            logger.info(f"[PERIOD] Cycling to {label} ({self._historical_period_idx + 1}/{n})")
        else:
            return  # Single-period mode, nothing to cycle
        # Flush buffer so next refill uses new period data
        self.buffer.clear()
        self.buffer_sample_counts.clear()
        self._refill_buffer()

    def _get_thread_market(self, level: int) -> SyntheticMarket:
        """
        Get thread-local SyntheticMarket instance (TASK_667 thread-safety fix).

        Each thread gets its own market instances to avoid race conditions
        when regenerate() is called from multiple threads simultaneously.
        """
        if not hasattr(self._thread_local, 'markets'):
            self._thread_local.markets = {}

        if level not in self._thread_local.markets:
            # TASK_675: Atomic counter ensures unique seeds per thread (no hash collisions)
            with self._market_lock:
                thread_seed = self._base_seed + self._thread_counter
                self._thread_counter += 1
            self._thread_local.markets[level] = SyntheticMarket(
                level=level,
                n_assets=self.config.n_assets,
                n_steps=self.config.episode_length + self.config.window_size + self.config.oracle_lookahead + 5,
                seed=thread_seed + level,
                noise_level=self.config.synthetic_noise,
            )

        return self._thread_local.markets[level]

    def _generate_prices_cpu(self, level: int, seed: int) -> tuple:
        """Generate prices on CPU (runs in background thread).

        Uses thread-local SyntheticMarket to avoid race conditions.
        """
        market = self._get_thread_market(level)
        market.regenerate(new_seed=seed)

        # Vectorized extraction (CPU)
        symbols = [f'SYN_{j:02d}' for j in range(market.n_assets)]
        prices = np.stack([market.data[s]['close'] for s in symbols], axis=1).astype(np.float32)
        return (level, prices)

    def _submit_prefetch_tasks(self):
        """Submit background tasks to generate prices."""
        for _ in range(self.N_BUFFER):
            if self.shutdown_event.is_set():
                break
            level = self.rng.choice(self.config.synthetic_levels)
            seed = self.rng.integers(0, 2**31)
            future = self.executor.submit(self._generate_prices_cpu, level, seed)
            future.add_done_callback(self._on_prices_ready)

    def _on_prices_ready(self, future):
        """Callback when background price generation completes."""
        if self.shutdown_event.is_set():
            return
        try:
            result = future.result()
            self.price_queue.put(result, timeout=1.0)
        except Exception as e:
            self._prefetch_failures = getattr(self, '_prefetch_failures', 0) + 1
            if self._prefetch_failures <= 3 or self._prefetch_failures % 50 == 0:
                logger.warning(f"Price generation failed ({self._prefetch_failures}x): {e}")
            if self._prefetch_failures > 50:
                logger.error("Excessive prefetch failures — check GPU/data pipeline")

    def _get_entropy_schedule(self):
        """TASK_691: Compute oracle entropy noise rates based on current epoch."""
        if not self.config.oracle_entropy:
            return 0.0, 0.0
        warmup = self.config.entropy_warmup_epochs
        if self._current_epoch < warmup:
            return 0.0, 0.0
        progress = min((self._current_epoch - warmup) / 150.0, 1.0)
        p_drop = self.config.oracle_drop_rate * progress
        q_phantom = self.config.oracle_phantom_rate * progress
        return p_drop, q_phantom

    def _compute_ar_blend(self):
        """Compute current AR blend probability from epoch + config."""
        if not getattr(self.config, 'autoregressive_state', False):
            return 0.0
        ar_warmup = getattr(self.config, 'ar_warmup_epochs', 0)
        if self._current_epoch < ar_warmup:
            return 0.0
        ar_blend_epochs = getattr(self.config, 'ar_blend_epochs', 0)
        ar_max_blend = getattr(self.config, 'ar_max_blend', 1.0)
        if ar_blend_epochs > 0 and self._current_epoch < ar_warmup + ar_blend_epochs:
            return min((self._current_epoch - ar_warmup) / ar_blend_epochs, ar_max_blend)
        return ar_max_blend

    def _generate_batch_episodes(self, prices_list: list, levels: list) -> list:
        """
        Process N price series in parallel using batched oracle.

        This is ~10x faster than calling _generate_episode_data N times
        because the oracle computation is vmapped across all episodes.

        Args:
            prices_list: List of N numpy arrays, each (T, n_assets)
            levels: List of N level indices

        Returns:
            List of N episode data dicts
        """
        n_batch = len(prices_list)
        if n_batch == 0:
            return []

        # TASK_703: Handle both OHLCV (T, n_assets, 5) and legacy close-only (T, n_assets)
        max_len = max(p.shape[0] for p in prices_list)
        ohlcv_mode = prices_list[0].ndim == 3  # True if (T, n_assets, 5)

        if ohlcv_mode:
            n_assets = prices_list[0].shape[1]
            prices_padded = np.zeros((n_batch, max_len, n_assets, 5), dtype=np.float32)
            # Initialize OHLCV padding with 1.0 (avoid zero-price log issues)
            prices_padded[:, :, :, :4] = 1.0  # OHLC = 1.0
            valid_lengths = []
            for i, p in enumerate(prices_list):
                prices_padded[i, :p.shape[0]] = p
                valid_lengths.append(p.shape[0])
            prices_jax = jnp.array(prices_padded)
            # Oracle uses close prices only
            close_jax = prices_jax[:, :, :, 3]
        else:
            n_assets = prices_list[0].shape[1]
            prices_padded = np.zeros((n_batch, max_len, n_assets), dtype=np.float32)
            valid_lengths = []
            for i, p in enumerate(prices_list):
                prices_padded[i, :p.shape[0]] = p
                valid_lengths.append(p.shape[0])
            prices_jax = jnp.array(prices_padded)
            close_jax = prices_jax

        # Compute log returns from close prices (N, T-1, n_assets)
        # TASK_705: Clamp both sides to avoid log(0) = -inf from zero-price gaps in historical data
        log_returns = jnp.log(jnp.maximum(close_jax[:, 1:], 1e-8) / jnp.maximum(close_jax[:, :-1], 1e-8))

        # TASK_673: Multipass oracle (soft targets) or single-pass (hard targets)
        if getattr(self.config, 'oracle_multipass', False):
            self.jax_rng, subkey = jax.random.split(self.jax_rng)
            policy_targets, cumulative_pnl_batch, _, oracle_selections_batch = batched_multipass_oracle_policy(
                log_returns,
                subkey,
                k=self.config.oracle_k,
                lookahead=self.config.oracle_lookahead,
                transaction_cost=self.config.transaction_cost,
                min_alpha_threshold=getattr(self.config, 'min_alpha_threshold', 0.0),
                n_passes=self.config.oracle_n_passes,
                noise_scale=self.config.oracle_noise_scale,
                sharpen_temperature=getattr(self.config, 'oracle_sharpen_temperature', 1.0),
            )
            # policy_targets: (N, T, 320) - already soft policy
            # cumulative_pnl_batch: (N, T)
        else:
            # Batched oracle - single GPU call for all N episodes!
            optimal_paths = batched_greedy_lookahead_path(
                log_returns,
                k=self.config.oracle_k,
                lookahead=self.config.oracle_lookahead,
                transaction_cost=self.config.transaction_cost,
                min_alpha_threshold=getattr(self.config, 'min_alpha_threshold', 0.0),
            )

            # TASK_141: Compute prev_selections for state-aware oracle (batched)
            N, T_log, n_assets_batch = log_returns.shape
            if getattr(self.config, 'state_aware_oracle', False):
                prev_selections_batch = jnp.zeros((N, T_log, n_assets_batch))
                prev_selections_batch = prev_selections_batch.at[:, 1:, :].set(
                    jnp.where(optimal_paths.selections[:, :-1, :] > 0.5,
                              jnp.where(log_returns[:, :-1, :] > 0, 1.0, -1.0),
                              0.0)
                )
            else:
                prev_selections_batch = None

            policy_targets = batched_optimal_path_to_policy(
                optimal_paths.selections,
                log_returns,
                prev_selections_batch=prev_selections_batch,
                n_actions_per_asset=5,
                temperature=1.0,
            )
            cumulative_pnl_batch = optimal_paths.cumulative_pnl  # (N, T)
            oracle_selections_batch = optimal_paths.selections    # (N, T, n_assets)

        # TASK_670 FIX: Compute REMAINING PnL (future value), not cumulative (past value)
        # remaining_pnl[i, t] = final_pnl[i] - cumulative_pnl[i, t]
        remaining_pnl = cumulative_pnl_batch[:, -1:] - cumulative_pnl_batch  # (N, T)

        # TASK_140: Batched Encoding
        # window_starts is same for all episodes if they have same length
        T_log = max_len - 1
        n_valid = T_log - self.config.window_size
        
        if n_valid <= 0:
            return []

        window_starts = jnp.arange(n_valid)

        # TASK_733: build (N, L, lookahead, ...) future-price tensor. Zero-pad
        # tail so the dead-zone in channel 32 turns out-of-series windows into
        # HOLD (sign=0) automatically.
        future_prices_batch = None
        if self._cheat_lookahead > 0:
            la = self._cheat_lookahead
            ws_cfg = self.config.window_size
            pad_width = [(0, 0), (0, la)] + [(0, 0)] * (prices_jax.ndim - 2)
            prices_padded_fwd = jnp.pad(prices_jax, pad_width)
            future_prices_batch = jax.vmap(
                lambda p: jax.vmap(
                    lambda w: jax.lax.dynamic_slice_in_dim(p, w + ws_cfg, la, axis=0)
                )(window_starts)
            )(prices_padded_fwd)

        # TASK_152: Super-batched temporal encoding (N, L, 27, 64, 64)
        all_observations = jit_vmap_encode_temporal(
            prices_jax,
            window_starts,
            self.encoder_config,
            future_prices_batch,
        )

        # TASK_689: Build state channels per episode
        use_state_channels = self.config.n_channels > 32
        ar_blend = self._compute_ar_blend()
        # TASK_691: Oracle entropy schedule (computed once per batch)
        p_drop, q_phantom = self._get_entropy_schedule() if use_state_channels else (0.0, 0.0)

        # TASK_699: Pre-determine which episodes use AR (coin flip phase)
        ar_episode_set = set()
        if use_state_channels and ar_blend > 0 and self._model_params is not None:
            for i in range(n_batch):
                if self.rng.random() < ar_blend:
                    ar_episode_set.add(i)

        # TASK_699: Batched AR rollout (single JIT call for all AR episodes)
        batched_state_ch = {}  # i → (n_valid, 6, n_assets)
        batched_conf = {}      # i → (n_valid, n_assets)
        if ar_episode_set:
            ar_indices = sorted(ar_episode_set)
            B_actual = len(ar_indices)
            B_pad = self.n_envs  # Fixed B for JIT stability (static_argname)

            ws = self.config.window_size

            # BUG-1 fix: Align returns to window decision points
            returns_list = [log_returns[i, ws - 1:valid_lengths[i] - 1] for i in ar_indices]

            # All episodes should have same length (same config → same generation params)
            ar_lengths = [r.shape[0] for r in returns_list]
            if len(set(ar_lengths)) > 1:
                logger.warning(f"TASK_699: Ragged AR lengths {ar_lengths}, falling back to per-episode rollout")
                # Fall back to unbatched rollout for each AR episode individually
                for b, idx in enumerate(ar_indices):
                    market_obs_i = all_observations[idx]
                    log_returns_i = returns_list[b]
                    state_ch_i, conf_i = self._model_state_rollout(
                        market_obs_i, log_returns_i,
                        self._model_params, self._model_apply_fn,
                        n_assets=n_assets, k=self.config.oracle_k,
                        window_size=ws, decay_fast=0.80, decay_slow=0.99,
                        n_actions_per_asset=5,
                    )
                    batched_state_ch[idx] = state_ch_i
                    batched_conf[idx] = conf_i
            else:
                T_ar = ar_lengths[0]

                # Pad to fixed B with zeros (model runs on padding, results discarded)
                market_obs_padded = jnp.zeros((B_pad, all_observations.shape[1],
                                               all_observations.shape[2], all_observations.shape[3],
                                               all_observations.shape[4]))
                returns_padded = jnp.zeros((B_pad, T_ar, n_assets))
                for b, idx in enumerate(ar_indices):
                    market_obs_padded = market_obs_padded.at[b].set(all_observations[idx])
                    returns_padded = returns_padded.at[b].set(returns_list[b])

                state_ch_batch, conf_batch = _build_model_state_trajectory_batched(
                    market_obs_padded, returns_padded,
                    self._model_params, self._model_apply_fn,
                    n_assets=n_assets, k=self.config.oracle_k,
                    window_size=ws, decay_fast=0.80, decay_slow=0.99,
                    n_actions_per_asset=5, B=B_pad,
                )  # (B_pad, T_ar, 6, n_assets), (B_pad, T_ar, n_assets)

                # Distribute results (discard padding episodes beyond B_actual)
                for b, idx in enumerate(ar_indices):
                    batched_state_ch[idx] = state_ch_batch[b]
                    batched_conf[idx] = conf_batch[b]

        # Process each episode (alignment and dict creation)
        episode_datas = []
        for i in range(n_batch):
            T = valid_lengths[i] - 1  # log_returns length
            # Alignment
            start_idx = self.config.window_size - 1
            end_idx = min(start_idx + n_valid, T)
            n_windows = end_idx - start_idx

            obs_i = all_observations[i, :n_windows]

            # TASK_699: Use pre-computed batched AR results
            use_ar_this_ep = (i in ar_episode_set)
            if use_state_channels and use_ar_this_ep:
                state_ch_i = batched_state_ch[i]  # (T_ar, 6, n_assets)
                conf_i = batched_conf[i]           # (T_ar, n_assets)

                ws = self.config.window_size
                # TASK_693: Model rollout indexed by WINDOW (0..n_valid-1),
                # not return timestep. scan output[w] = state before window w.
                state_at_w = state_ch_i[:n_windows]
                state_tiled = jnp.tile(state_at_w[:, :, None, :], (1, 1, ws, 1))
                conf_at_w = conf_i[:n_windows]
                conf_tiled = jnp.tile(conf_at_w[:, None, None, :], (1, 1, ws, 1))
                obs_i = jnp.concatenate([obs_i, state_tiled, conf_tiled], axis=1)

            elif use_state_channels:
                # TASK_691: Oracle entropy
                sel_i = oracle_selections_batch[i, :T]
                if p_drop > 0 or q_phantom > 0:
                    self.jax_rng, ek = jax.random.split(self.jax_rng)
                    sel_i = corrupt_oracle_selections(sel_i, ek, p_drop, q_phantom)
                target_pos_i = compute_target_position(sel_i, log_returns[i, :T])
                state_ch_i = build_state_channels(
                    target_pos_i, log_returns[i, :T],
                    n_assets=n_assets,
                    k=self.config.oracle_k,
                    window_size=self.config.window_size,
                )  # (T, 6, n_assets)

                oracle_per_asset = policy_targets[i, :T].reshape(T, n_assets, 5)
                conf_i = jnp.max(oracle_per_asset, axis=-1)  # (T, n_assets)

                ws = self.config.window_size
                # Use t-1 state to avoid leaking current oracle action
                state_at_w = state_ch_i[start_idx - 1:end_idx - 1]
                state_tiled = jnp.tile(state_at_w[:, :, None, :], (1, 1, ws, 1))
                conf_at_w = conf_i[start_idx - 1:end_idx - 1]
                conf_tiled = jnp.tile(conf_at_w[:, None, None, :], (1, 1, ws, 1))
                obs_i = jnp.concatenate([obs_i, state_tiled, conf_tiled], axis=1)

            # TASK_696: DAgger fix — re-compute targets from student position
            if use_state_channels and use_ar_this_ep:
                student_prev_pos = state_ch_i[:n_windows, 0, :]  # (n_windows, n_assets)
                actions_i = optimal_path_to_policy(
                    oracle_selections_batch[i, start_idx:end_idx],
                    log_returns[i, start_idx:end_idx],
                    prev_selections=student_prev_pos,
                    n_actions_per_asset=5,
                    temperature=1.0,
                )
            else:
                actions_i = policy_targets[i, start_idx:end_idx]

            ep_data = {
                'observations': obs_i,
                'actions': actions_i,
                'returns': remaining_pnl[i, start_idx:end_idx],
                'per_asset_returns': log_returns[i, start_idx:end_idx],  # TASK_713
                'level': levels[i],
                'n_samples': n_windows,
            }

            # TASK_695: Oracle mask for auxiliary prediction head
            if getattr(self.config, 'use_oracle_mask_head', False):
                ep_data['oracle_mask'] = oracle_selections_batch[i, start_idx:end_idx]

            # Pre-compute latent states if caching enabled
            if self._use_latent_cache and self._encoder_fn is not None:
                ep_data['latent_states'] = self._encoder_fn(ep_data['observations'])

            # TASK_675: Filter zero-sample episodes (short price series edge case)
            if ep_data['n_samples'] > 0:
                episode_datas.append(ep_data)

        return episode_datas

    def _build_historical_episode(self, slot, cursor_start, n_steps):
        """TASK_714: Build episode from precomputed full-period oracle.

        Instead of computing oracle on a 489-ts chunk (inconsistent targets),
        indexes into the oracle precomputed on the full 258k-ts period.

        Args:
            slot: SegmentPool slot dict with hm, oracle_selections, returns, etc.
            cursor_start: Starting price index for this episode
            n_steps: Number of training windows to produce

        Returns:
            ep_data dict (same format as _generate_batch_episodes output)
        """
        ws = self.config.window_size  # 64
        n_assets = self.config.n_assets  # 64

        # Price slice for encoder: need prices[cursor_start : cursor_start + n_steps + ws - 1 + 1]
        # because the last window starts at cursor_start + n_steps - 1 and covers ws prices.
        # TASK_733: extend slice by `lookahead` rows so the forward window for the
        # last training window is available; tail that runs off end is zero-padded.
        la = self._cheat_lookahead
        price_end = cursor_start + n_steps + ws + la
        cache_data = slot['hm'].cache_data
        tail_available = min(price_end, len(cache_data))
        prices_slice = cache_data[cursor_start:tail_available]  # (≤ n_steps+ws+la, n_assets, 5)
        if la > 0 and prices_slice.shape[0] < (n_steps + ws + la):
            # End-of-series: pad with zeros (dead-zone → HOLD for those windows)
            pad_rows = (n_steps + ws + la) - prices_slice.shape[0]
            pad_width = [(0, pad_rows)] + [(0, 0)] * (prices_slice.ndim - 1)
            prices_slice = np.pad(prices_slice, pad_width)
        prices_jax = jnp.array(prices_slice)

        # Encode all windows: window_starts relative to the slice
        window_starts = jnp.arange(n_steps)
        # TASK_733: future slice per window — shape (n_steps, la, n_assets[, 5])
        future_prices_batch = None
        if la > 0:
            future_prices_batch = jax.vmap(
                lambda w: jax.lax.dynamic_slice_in_dim(prices_jax, w + ws, la, axis=0)
            )(window_starts)
        observations = jit_batch_encode_temporal(
            prices_jax, window_starts, self.encoder_config, future_prices_batch,
        )  # (n_steps, 27, ws, n_assets)

        # Oracle targets at decision points
        # Window w (relative) → decision at returns index cursor_start + w + ws - 1
        decision_start = cursor_start + ws - 1
        decision_end = decision_start + n_steps
        oracle_sel = slot['oracle_selections'][decision_start:decision_end]  # (n_steps, n_assets)
        log_returns = slot['returns'][decision_start:decision_end]  # (n_steps, n_assets)

        # Remaining PnL (future value)
        cum_pnl = slot['oracle_cum_pnl']
        remaining_pnl = cum_pnl[-1] - cum_pnl[decision_start:decision_end]  # (n_steps,)

        # Convert oracle selections to policy targets (state-aware)
        oracle_sel_jax = jnp.array(oracle_sel)
        log_returns_jax = jnp.array(log_returns)
        remaining_pnl_jax = jnp.array(remaining_pnl)

        if getattr(self.config, 'state_aware_oracle', False):
            # prev_selections: oracle position at t-1 (for state-aware open/close logic)
            # Use full-period oracle for consistency at episode boundaries
            if decision_start > 0:
                prev_first = slot['oracle_selections'][decision_start - 1]
                prev_ret = slot['returns'][decision_start - 1]
                first_prev = np.where(prev_first > 0.5,
                                       np.where(prev_ret > 0, 1.0, -1.0), 0.0)
            else:
                first_prev = np.zeros(n_assets, dtype=np.float32)
            prev_sel = jnp.zeros((n_steps, n_assets))
            prev_sel = prev_sel.at[0].set(jnp.array(first_prev))
            prev_sel = prev_sel.at[1:].set(
                jnp.where(oracle_sel_jax[:-1] > 0.5,
                          jnp.where(log_returns_jax[:-1] > 0, 1.0, -1.0),
                          0.0)
            )
        else:
            prev_sel = None

        policy_targets = optimal_path_to_policy(
            oracle_sel_jax,
            log_returns_jax,
            prev_selections=prev_sel,
            n_actions_per_asset=5,
            temperature=1.0,
        )

        # State channels (same logic as _generate_batch_episodes)
        use_state_channels = self.config.n_channels > 32
        ar_blend = self._compute_ar_blend()
        use_model_rollout = (use_state_channels and ar_blend > 0
                             and self.rng.random() < ar_blend
                             and self._model_params is not None)

        if use_model_rollout:
            # AR rollout: model's own state channels
            # Need returns aligned to window decision points
            returns_for_ar = log_returns_jax  # Already aligned to decision points
            state_channels, confidence = self._model_state_rollout(
                observations,          # (n_steps, 27, ws, n_assets)
                returns_for_ar,        # (n_steps, n_assets)
                self._model_params,
                self._model_apply_fn,
                n_assets=n_assets,
                k=self.config.oracle_k,
                window_size=ws,
                decay_fast=0.80,
                decay_slow=0.99,
                n_actions_per_asset=5,
            )
            # Tile state channels across window dim
            state_tiled = jnp.tile(state_channels[:, :, None, :], (1, 1, ws, 1))
            conf_tiled = jnp.tile(confidence[:, None, None, :], (1, 1, ws, 1))
            observations = jnp.concatenate([observations, state_tiled, conf_tiled], axis=1)

            # DAgger targets: re-compute policy from student's position
            student_prev_pos = state_channels[:, 0, :]  # position channel
            policy_targets = optimal_path_to_policy(
                oracle_sel_jax,
                log_returns_jax,
                prev_selections=student_prev_pos,
                n_actions_per_asset=5,
                temperature=1.0,
            )

        elif use_state_channels:
            # Oracle state channels
            p_drop, q_phantom = self._get_entropy_schedule()
            if p_drop > 0 or q_phantom > 0:
                self.jax_rng, ek = jax.random.split(self.jax_rng)
                noisy_sel = corrupt_oracle_selections(oracle_sel_jax, ek, p_drop, q_phantom)
                target_pos = compute_target_position(noisy_sel, log_returns_jax)
            else:
                target_pos = compute_target_position(oracle_sel_jax, log_returns_jax)
            state_ch = build_state_channels(
                target_pos, log_returns_jax,
                n_assets=n_assets, k=self.config.oracle_k, window_size=ws,
            )  # (n_steps, 6, n_assets)

            oracle_per_asset = policy_targets.reshape(n_steps, n_assets, 5)
            conf = jnp.max(oracle_per_asset, axis=-1)  # (n_steps, n_assets)

            state_tiled = jnp.tile(state_ch[:, :, None, :], (1, 1, ws, 1))
            conf_tiled = jnp.tile(conf[:, None, None, :], (1, 1, ws, 1))
            observations = jnp.concatenate([observations, state_tiled, conf_tiled], axis=1)

        ep_data = {
            'observations': observations,
            'actions': policy_targets,
            'returns': remaining_pnl_jax,
            'per_asset_returns': log_returns_jax,  # TASK_713
            'level': -2,  # historical
            'n_samples': n_steps,
        }

        # TASK_695: Oracle mask
        if getattr(self.config, 'use_oracle_mask_head', False):
            ep_data['oracle_mask'] = oracle_sel_jax

        # Pre-compute latent states if caching enabled
        if self._use_latent_cache and self._encoder_fn is not None:
            ep_data['latent_states'] = self._encoder_fn(ep_data['observations'])

        return ep_data

    def _should_use_secondary(self) -> bool:
        """Check if this refill should use secondary source (90/10 mixing)."""
        if self.mix_ratio <= 0:
            return False
        self._mix_counter += 1
        # Use secondary every 1/mix_ratio refills (e.g., every 10th for 10%)
        mix_interval = int(1.0 / self.mix_ratio) if self.mix_ratio > 0 else 0
        return mix_interval > 0 and (self._mix_counter % mix_interval == 0)

    def _generate_secondary_prices(self, n_to_refill: int) -> tuple:
        """Generate prices from secondary source for experience replay."""
        prices_list = []
        levels = []

        if self.bella_generator is not None:
            # Primary is Bella, secondary is Synthetic
            for _ in range(n_to_refill):
                level = self.rng.choice(self.secondary_levels)
                market = self._get_thread_market(level)
                seed = self.rng.integers(0, 2**31)
                market.regenerate(new_seed=seed)
                symbols = [f'SYN_{j:02d}' for j in range(market.n_assets)]
                prices = np.stack([market.data[s]['close'] for s in symbols], axis=1).astype(np.float32)
                prices_list.append(prices)
                levels.append(level)
        elif self.secondary_bella_generator is not None:
            # Primary is Synthetic, secondary is Bella
            self.jax_rng, subkey = jax.random.split(self.jax_rng)
            prices_batch = generate_bella_prices_on_device(
                self.secondary_bella_generator, subkey, n_envs=n_to_refill
            )
            for i in range(n_to_refill):
                prices_list.append(np.array(prices_batch[i]))
                levels.append(-1)  # -1 = Bella

        return prices_list, levels

    def _refill_buffer(self):
        """Refill VRAM buffer using batched oracle for ~10x speedup."""
        # Clean up exhausted buffer entries
        if self.buffer:
            keep_indices = [i for i, cnt in enumerate(self.buffer_sample_counts)
                          if cnt < self.SAMPLES_PER_EPISODE]
            self.buffer = [self.buffer[i] for i in keep_indices]
            self.buffer_sample_counts = [self.buffer_sample_counts[i] for i in keep_indices]
            # TASK_692: Also clean up param version tracking
            if self._buffer_param_ids:
                self._buffer_param_ids = [self._buffer_param_ids[i] for i in keep_indices]

        needed = self.N_BUFFER - len(self.buffer)
        if needed <= 0:
            return

        # TASK_140: refill in chunks of n_envs for better parallelism
        while len(self.buffer) < self.N_BUFFER:
            n_to_refill = min(self.n_envs, self.N_BUFFER - len(self.buffer))

            # TASK_714: Historical data with precomputed oracle (SegmentPool path)
            if self._use_historical and self.segment_pool is not None:
                for _ in range(n_to_refill):
                    slot, cursor_start, n_steps = self.segment_pool.get_episode_range(
                        self.config.episode_length, self.config.window_size
                    )
                    if slot is None or n_steps <= 0:
                        continue
                    try:
                        ep_data = self._build_historical_episode(slot, cursor_start, n_steps)
                    except Exception as e:
                        if "out of memory" in str(e).lower() or "resource exhausted" in str(e).lower():
                            logger.warning(f"[OOM] Episode build failed at cursor {cursor_start}, skipping")
                            continue
                        raise
                    if ep_data is not None and ep_data['n_samples'] > 0:
                        # TASK_680: Pre-compute trade index list for biased sampling
                        if self.config.trade_sample_prob > 0:
                            actions = np.asarray(ep_data['actions'])
                            actions_5 = actions.reshape(actions.shape[0], self.config.n_assets, 5)
                            trade_density = 1.0 - actions_5[:, :, 1].mean(axis=1)
                            ep_data['trade_indices'] = np.where(trade_density > 0.01)[0]
                        else:
                            ep_data['trade_indices'] = np.array([], dtype=np.intp)
                        self.buffer.append(ep_data)
                        self.buffer_sample_counts.append(0)
                        self._buffer_param_ids.append(self._param_version)
                continue  # Skip _generate_batch_episodes below

            # TASK_698: Legacy historical (single --db-path, no pool)
            elif self._use_historical:
                prices_list = []
                levels = []
                for _ in range(n_to_refill):
                    prices = self.historical_market.get_window_random(self._synth_n_steps)
                    if prices.shape[1] < self.config.n_assets:
                        n_have = prices.shape[1]
                        repeats = (self.config.n_assets + n_have - 1) // n_have
                        prices = np.tile(prices, (1, repeats, 1))[:, :self.config.n_assets, :]
                    prices_list.append(prices)
                    levels.append(-2)
                logger.debug(f"Refill from HISTORICAL source ({len(prices_list)} episodes)")
            # TASK_667: 90/10 Experience Replay
            elif self._should_use_secondary() and (self.secondary_bella_generator is not None or
                                   (self.bella_generator is not None and self.secondary_levels)):
                prices_list, levels = self._generate_secondary_prices(n_to_refill)
                logger.debug(f"Refill from SECONDARY source ({len(prices_list)} episodes)")
            elif self.bella_generator is not None:
                # Bella: generate N prices in single GPU call
                # TASK_703: Now returns (N, seq_len, n_assets, 5) OHLCV
                self.jax_rng, subkey = jax.random.split(self.jax_rng)
                prices_batch = generate_bella_prices_on_device(
                    self.bella_generator, subkey, n_envs=n_to_refill,
                    invert_regime=self.invert_regime
                )  # (N, seq_len, n_assets, 5) OHLCV
                prices_list = []
                levels = []
                for i in range(n_to_refill):
                    prices_list.append(np.array(prices_batch[i]))
                    levels.append(-1)
            elif self._use_gpu_synth:
                # GPU synthetic: generate directly on device (no PCIe transfer)
                level = int(self.rng.choice(self.config.synthetic_levels))
                self.jax_rng, subkey = jax.random.split(self.jax_rng)
                prices_batch = generate_synthetic_on_device(
                    subkey, n_to_refill, level,
                    self.config.n_assets, self._synth_n_steps,
                    self.config.synthetic_noise,
                )  # (n_to_refill, n_steps, n_assets) — on device
                prices_list = []
                levels = []
                for i in range(n_to_refill):
                    prices_list.append(np.array(prices_batch[i]))
                    levels.append(level)
            else:
                # CPU synthetic fallback (L4+ or mixed levels)
                prices_list = []
                levels = []
                while len(prices_list) < n_to_refill:
                    try:
                        level, prices_cpu = self.price_queue.get(timeout=0.1)
                        prices_list.append(prices_cpu)
                        levels.append(level)

                        # Submit replacement prefetch
                        new_level = self.rng.choice(self.config.synthetic_levels)
                        new_seed = self.rng.integers(0, 2**31)
                        future = self.executor.submit(self._generate_prices_cpu, new_level, new_seed)
                        future.add_done_callback(self._on_prices_ready)

                    except Empty:
                        # Queue empty, generate synchronously
                        level = self.rng.choice(self.config.synthetic_levels)
                        seed = self.rng.integers(0, 2**31)
                        _, prices_cpu = self._generate_prices_cpu(level, seed)
                        prices_list.append(prices_cpu)
                        levels.append(level)

            # Process all episodes in single batched oracle call (~10x faster)
            if prices_list:
                start_refill = time.time()
                try:
                    episode_datas = self._generate_batch_episodes(prices_list, levels)
                except Exception as e:
                    if "out of memory" in str(e).lower() or "resource exhausted" in str(e).lower():
                        half = max(1, len(prices_list) // 2)
                        logger.warning(f"[OOM] Refill failed with {len(prices_list)} episodes, retrying with {half}")
                        episode_datas = self._generate_batch_episodes(prices_list[:half], levels[:half])
                    else:
                        raise
                for ep_data in episode_datas:
                    # TASK_680: Pre-compute trade index list for biased sampling (O(T) once)
                    if ep_data['n_samples'] > 0 and self.config.trade_sample_prob > 0:
                        actions = np.asarray(ep_data['actions'])
                        actions_5 = actions.reshape(actions.shape[0], self.config.n_assets, 5)
                        trade_density = 1.0 - actions_5[:, :, 1].mean(axis=1)  # (T,)
                        ep_data['trade_indices'] = np.where(trade_density > 0.01)[0]
                    else:
                        ep_data['trade_indices'] = np.array([], dtype=np.intp)
                    self.buffer.append(ep_data)
                    self.buffer_sample_counts.append(0)
                    # TASK_692: Track param version for FIFO refresh
                    self._buffer_param_ids.append(self._param_version)

                refill_time = time.time() - start_refill
                logger.debug(f"Refilled {len(episode_datas)} episodes in {refill_time:.3f}s")

    def shutdown(self):
        """Clean shutdown of background threads."""
        self.shutdown_event.set()
        self.executor.shutdown(wait=False)

    def set_encoder(self, model, params):
        """
        Enable latent state caching by providing encoder function.

        TASK_666 Optimization: Pre-compute latent states during buffer fill,
        removing the TRM encoder from the training loop entirely.

        Args:
            model: MuZeroNetworks instance
            params: Model parameters
        """
        # TASK_152: Observations are now full temporal format (batch, 27, 64, 64)
        # No tiling needed - pass directly to model

        @jax.jit
        def encode_batch(full_obs):
            """Encode full observations to latent states."""
            # Run representation only (observations already in full format)
            latent = model.apply({'params': params}, observation=full_obs, mode='initial_inference')
            return latent['latent_state']

        self._encoder_fn = encode_batch
        self._use_latent_cache = True

        # Re-encode existing buffer entries with latent states
        logger.info("Computing latent states for existing buffer...")
        for ep_data in self.buffer:
            if 'latent_states' not in ep_data:
                ep_data['latent_states'] = self._encoder_fn(ep_data['observations'])
        logger.info(f"Latent state caching enabled ({len(self.buffer)} episodes cached)")

    def set_model_params(self, model, params):
        """
        TASK_692: Update model parameters for autoregressive state rollout.

        JIT-safe — params passed as regular arg to model.apply(),
        not static_argnames. This enables to model to compute its
        own state channels during training.

        Args:
            model: MuZeroNetworks instance
            params: Current model parameters
        """
        self._model_params = params
        self._model_apply_fn = model.apply
        self._param_version += 1  # Increment version for FIFO refresh

        # TASK_692: Partial FIFO refresh of buffer
        # Refresh a fraction of episodes with new params to keep
        # a mix of recent model versions (natural regularizer)
        if getattr(self.config, 'autoregressive_state', False):
            self._partial_refresh_buffer()

    def _partial_refresh_buffer(self):
        """TASK_692: Refresh a fraction of buffer episodes with new params.

        Partial FIFO (DAgger-style):
        - Buffer holds N episodes (default 32)
        - Each epoch: refresh X% (default 20% ≈ 6 episodes)
        - Remaining episodes use params from previous epochs
        - Natural regularizer: prevents overfitting to current quirks

        Cost: ~7s per epoch (20% refresh) vs ~43s for full refresh.
        """
        if not self.buffer:
            return

        n_total = len(self.buffer)
        refresh_fraction = getattr(self.config, 'state_refresh_fraction', 0.2)
        n_refresh = max(1, int(n_total * refresh_fraction))

        # Find oldest entries (FIFO) - those with smallest param_version
        indices_to_refresh = sorted(
            range(n_total),
            key=lambda i: self._buffer_param_ids[i] if i < len(self._buffer_param_ids) else 0
        )[:n_refresh]

        if indices_to_refresh:
            logger.debug(f"  [TASK_692] Refreshing {len(indices_to_refresh)}/{n_total} episodes "
                        f"({refresh_fraction:.0%}) with new model params v{self._param_version}")

            # Mark these entries for regeneration by exhausting them
            for idx in indices_to_refresh:
                self.buffer_sample_counts[idx] = self.SAMPLES_PER_EPISODE

            # LATENT fix: Set sentinel instead of delete (prevents list length desync)
            for idx in indices_to_refresh:
                if idx < len(self._buffer_param_ids):
                    self._buffer_param_ids[idx] = -1  # Sentinel: lowest in FIFO sort

    def generate_batch(self, batch_size: int) -> dict:
        """
        Generate batch from buffered episodes (minimal regeneration).

        Pipeline:
        1. Sample windows from buffered episodes (already in VRAM)
        2. Only regenerate exhausted episodes
        3. Return batch dict with latent_states if caching enabled

        Returns:
            dict with 'observations' or 'latent_states', 'actions', 'returns' as JAX arrays
        """
        all_obs = []
        all_latents = [] if self._use_latent_cache else None
        all_actions = []
        all_returns = []
        all_per_asset_returns = []  # TASK_713
        all_oracle_masks = []  # TASK_695

        samples_needed = batch_size

        while samples_needed > 0:
            # Find episode with samples remaining
            available = [i for i, cnt in enumerate(self.buffer_sample_counts)
                        if cnt < self.SAMPLES_PER_EPISODE]

            if not available:
                self._refill_buffer()
                if not self.buffer:
                    raise RuntimeError("Buffer empty after refill — check price generation pipeline")
                available = list(range(len(self.buffer)))

            # Pick random available episode
            ep_idx = self.rng.choice(available)
            ep_data = self.buffer[ep_idx]

            n_available = ep_data['n_samples']
            samples_per_window = min(batch_size // self.N_BUFFER + 1, 20)  # Samples per episode per batch
            take_count = min(samples_per_window, samples_needed, n_available)

            if take_count <= 0:
                self.buffer_sample_counts[ep_idx] = self.SAMPLES_PER_EPISODE  # Mark exhausted
                continue

            # TASK_696: Per-sample trade-biased selection
            # Old TASK_680 did single coin flip per episode visit — effective bias ~1.5-3%.
            # Now each sample independently decides trade vs random offset.
            trade_indices = ep_data.get('trade_indices', np.array([], dtype=np.intp))
            valid_trades = trade_indices[trade_indices < n_available]
            if len(valid_trades) > 0 and self.config.trade_sample_prob > 0:
                trade_mask = self.rng.random(take_count) < self.config.trade_sample_prob
                n_trade = int(trade_mask.sum())
                indices = np.empty(take_count, dtype=np.intp)
                if n_trade > 0:
                    indices[trade_mask] = self.rng.choice(valid_trades, size=n_trade)
                if n_trade < take_count:
                    indices[~trade_mask] = self.rng.integers(0, n_available, size=take_count - n_trade)
            else:
                indices = self.rng.integers(0, max(n_available, 1), size=take_count)

            all_obs.append(ep_data['observations'][indices])
            if self._use_latent_cache and 'latent_states' in ep_data:
                all_latents.append(ep_data['latent_states'][indices])
            all_actions.append(ep_data['actions'][indices])
            all_returns.append(ep_data['returns'][indices])
            # TASK_713: Per-asset returns for return-weighted trade loss
            if 'per_asset_returns' in ep_data:
                all_per_asset_returns.append(ep_data['per_asset_returns'][indices])
            # TASK_695: Oracle mask
            if 'oracle_mask' in ep_data:
                all_oracle_masks.append(ep_data['oracle_mask'][indices])

            self.buffer_sample_counts[ep_idx] += take_count
            samples_needed -= take_count

        # Concatenate
        batch = {
            'observations': jnp.concatenate(all_obs, axis=0)[:batch_size],
            'actions': jnp.concatenate(all_actions, axis=0)[:batch_size],
            'returns': jnp.concatenate(all_returns, axis=0)[:batch_size],
        }

        # Add latent states if caching enabled (for prediction_only mode)
        if self._use_latent_cache and all_latents:
            batch['latent_states'] = jnp.concatenate(all_latents, axis=0)[:batch_size]

        # TASK_713: Per-asset returns for return-weighted trade loss
        if all_per_asset_returns:
            batch['per_asset_returns'] = jnp.concatenate(all_per_asset_returns, axis=0)[:batch_size]

        # TASK_695: Oracle mask for auxiliary head
        if all_oracle_masks:
            batch['oracle_mask'] = jnp.concatenate(all_oracle_masks, axis=0)[:batch_size]

        self.global_step += 1
        return batch

    def generate_batch_direct(self, batch_size: int) -> dict:
        """TASK_715: Direct sampling from precomputed pool — no buffer overhead.

        Bypasses the episode/buffer abstraction. Each call samples positions
        directly from pool slots, encodes on-the-fly, and indexes into
        precomputed oracle/state/targets. 5% of calls use full AR rollout
        (hybrid Option B) to maintain state-awareness training signal.
        """
        pool = self.segment_pool
        ws = self.config.window_size
        n_assets = self.config.n_assets
        use_state_channels = self.config.n_channels > 32

        # 5% AR hybrid: occasionally build a full AR episode
        ar_frac = getattr(self.config, 'ar_direct_fraction', 0.05)
        if ar_frac > 0 and self.rng.random() < ar_frac and self._model_params is not None:
            slot, cursor_start, n_steps = pool.get_episode_range(
                self.config.episode_length, ws
            )
            if slot is not None and n_steps > 0:
                ep_data = self._build_historical_episode(slot, cursor_start, n_steps)
                if ep_data is not None and ep_data['n_samples'] > 0:
                    n_avail = ep_data['n_samples']
                    indices = self.rng.integers(0, n_avail, size=min(batch_size, n_avail))
                    batch = {
                        'observations': ep_data['observations'][indices],
                        'actions': ep_data['actions'][indices],
                        'returns': ep_data['returns'][indices],
                    }
                    if 'per_asset_returns' in ep_data:
                        batch['per_asset_returns'] = ep_data['per_asset_returns'][indices]
                    if 'oracle_mask' in ep_data:
                        batch['oracle_mask'] = ep_data['oracle_mask'][indices]
                    self.global_step += 1
                    return batch

        # Direct sampling: round-robin across slots, sequential within
        positions = pool.sample_positions(batch_size, ws)
        if not positions:
            return self.generate_batch(batch_size)

        # Group by slot
        by_slot = defaultdict(list)
        for slot_idx, cursor in positions:
            by_slot[slot_idx].append(cursor)

        all_obs = []
        all_actions = []
        all_returns = []
        all_par = []
        all_masks = []

        for slot_idx, cursors in by_slot.items():
            slot = pool.slots[slot_idx]
            cursors_arr = np.array(cursors)
            n = len(cursors)

            # Guard: filter cursors to valid range for BOTH prices and returns
            T_prices = len(slot['hm'].cache_data)
            T_ret = len(slot['returns'])
            # cursor + ws - 1 must be < T_ret (decision index in bounds)
            # cursor + ws must be <= T_prices (price window in bounds)
            max_cursor = min(T_ret - ws, T_prices - ws)
            valid = cursors_arr <= max_cursor
            if not np.all(valid):
                cursors_arr = cursors_arr[valid]
                n = len(cursors_arr)
                if n == 0:
                    continue

            # Encode windows: slice only needed prices (not full 258k).
            # TASK_733: extend tail by `lookahead` rows for cheat channel's
            # forward window; zero-pad if running off end-of-series.
            la = self._cheat_lookahead
            c_min, c_max = int(cursors_arr.min()), int(cursors_arr.max())
            end_needed = c_max + ws + la
            cache = slot['hm'].cache_data
            tail_available = min(end_needed, len(cache))
            price_slice = cache[c_min:tail_available]
            if la > 0 and price_slice.shape[0] < (end_needed - c_min):
                pad_rows = (end_needed - c_min) - price_slice.shape[0]
                pad_width = [(0, pad_rows)] + [(0, 0)] * (price_slice.ndim - 1)
                price_slice = np.pad(price_slice, pad_width)
            prices_jax = jnp.array(price_slice)
            starts_jax = jnp.array(cursors_arr - c_min)
            future_prices_batch = None
            if la > 0:
                future_prices_batch = jax.vmap(
                    lambda w: jax.lax.dynamic_slice_in_dim(prices_jax, w + ws, la, axis=0)
                )(starts_jax)
            obs = jit_batch_encode_temporal(
                prices_jax, starts_jax, self.encoder_config, future_prices_batch,
            )

            # Decision indices in returns-space
            decision_indices = cursors_arr + ws - 1

            # Policy targets (precomputed, fp16 → fp32)
            actions = jnp.array(slot['policy_targets'][decision_indices].astype(np.float32))

            # Per-asset returns
            par = jnp.array(slot['returns'][decision_indices])

            # Remaining PnL
            cum_pnl = slot['oracle_cum_pnl']
            remaining = jnp.array(cum_pnl[-1] - cum_pnl[decision_indices])

            # State channels (precomputed, fp16 → fp32)
            if use_state_channels:
                state_indices = np.maximum(decision_indices - 1, 0)
                state_ch = jnp.array(slot['state_channels'][state_indices].astype(np.float32))
                conf = jnp.array(slot['confidence'][decision_indices].astype(np.float32))
                state_tiled = jnp.tile(state_ch[:, :, None, :], (1, 1, ws, 1))
                conf_tiled = jnp.tile(conf[:, None, None, :], (1, 1, ws, 1))
                obs = jnp.concatenate([obs, state_tiled, conf_tiled], axis=1)

            all_obs.append(obs)
            all_actions.append(actions)
            all_returns.append(remaining)
            all_par.append(par)

            if getattr(self.config, 'use_oracle_mask_head', False):
                all_masks.append(jnp.array(slot['oracle_selections'][decision_indices]))

        batch = {
            'observations': jnp.concatenate(all_obs)[:batch_size],
            'actions': jnp.concatenate(all_actions)[:batch_size],
            'returns': jnp.concatenate(all_returns)[:batch_size],
        }
        if all_par:
            batch['per_asset_returns'] = jnp.concatenate(all_par)[:batch_size]
        if all_masks:
            batch['oracle_mask'] = jnp.concatenate(all_masks)[:batch_size]

        self.global_step += 1
        return batch


# =============================================================================
# Tiered Checkpoint Management (TASK_144)
# =============================================================================

def cleanup_tiered_checkpoints(dir_path: Path, keep_recent: int = 5, milestone_interval: int = 100, save_every: int = 5):
    """
    TASK_144: Tiered retention for branch checkpoints.

    Keeps:
    - Last N periodic saves (rolling window)
    - Milestones every 100 epochs (permanent)
    - phase1_best.pkl, phase1_complete.pkl (never deleted)

    Deletes intermediate periodic saves to keep disk usage under control.
    """
    import re

    # Find all periodic epoch files (not best/complete)
    files = list(dir_path.glob("phase1_epoch*.pkl"))
    if not files:
        return

    # Map filename to epoch number
    epoch_map = {}
    for f in files:
        match = re.search(r'epoch(\d+)', f.name)
        if match:
            epoch_map[int(match.group(1))] = f

    if not epoch_map:
        return

    # Identify epochs to keep
    sorted_epochs = sorted(epoch_map.keys())
    latest_epochs = set(sorted_epochs[-keep_recent:])

    to_keep = set()
    for epoch, file_path in epoch_map.items():
        # Keep if: recent OR near a milestone boundary.
        # Saves happen at (epoch+1) % save_every == 0, so epoch 6299 not 6300.
        # Keep the save nearest to each milestone (within save_every distance).
        near_milestone = ((epoch + 1) % milestone_interval < save_every
                          or epoch % milestone_interval == 0)
        if epoch in latest_epochs or near_milestone:
            to_keep.add(file_path)

    # Delete non-retained files
    deleted_count = 0
    for f in files:
        if f not in to_keep:
            f.unlink()
            deleted_count += 1

    if deleted_count > 0:
        logger.info(f"  [CLEANUP] Pruned {deleted_count} intermediate checkpoints")


# =============================================================================
# CSV Metrics Logger (TASK_667)
# =============================================================================

class CSVLogger:
    """Log training metrics to CSV file for analysis."""

    def __init__(self, log_dir: Path, name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"{name}_{timestamp}.csv"
        self.file = None
        self.writer = None
        self.fieldnames = None

    def log(self, epoch: int, metrics: dict, epoch_time: float = 0.0):
        """Log metrics for an epoch."""
        row = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'epoch_time_s': round(epoch_time, 2),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()}
        }

        if self.file is None:
            self.fieldnames = list(row.keys())
            self.file = open(self.csv_path, 'w', newline='')
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
            logger.info(f"CSV logging to: {self.csv_path}")

        # Handle new fields that weren't in header (can't mutate dict during iteration)
        row = {k: v for k, v in row.items() if k in self.fieldnames}

        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        """Close the CSV file."""
        if self.file:
            self.file.close()
            self.file = None


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    trainer: EasyTrainer,
    generator: OnTheFlyGenerator,
    config: EasyConfig,
    epoch: int,
) -> dict:
    """Train for one epoch with on-the-fly generation."""
    # TASK_691: Update generator's epoch for oracle entropy schedule
    generator._current_epoch = epoch
    # TASK_707: Cycle to next historical period each epoch (legacy, pre-TASK_710)
    # TASK_710: SegmentPool handles rotation internally — skip advance_period
    if generator.segment_pool is None and (generator._period_specs or generator._historical_markets):
        generator.advance_period()

    # TASK_692: Update generator's model params for autoregressive state rollout
    # Call at start of each epoch to refresh partial buffer with new params
    autoregressive_state = getattr(config, 'autoregressive_state', False)
    ar_warmup = getattr(config, 'ar_warmup_epochs', 0)
    ar_blend_epochs = getattr(config, 'ar_blend_epochs', 0)
    if autoregressive_state and epoch >= ar_warmup:
        generator.set_model_params(trainer.model, trainer.params)
        if epoch == ar_warmup:
            if ar_blend_epochs > 0:
                logger.info(f"  ► AR warmup complete — beginning {ar_blend_epochs}-epoch blend (oracle→AR)")
            else:
                logger.info(f"  ► AR warmup complete — switching to model-state rollout with DAgger targets")
        if ar_blend_epochs > 0 and epoch == ar_warmup + ar_blend_epochs:
            ar_max_blend = getattr(config, 'ar_max_blend', 1.0)
            if ar_max_blend < 1.0:
                logger.info(f"  ► AR blend ramp complete — capped at {ar_max_blend:.0%} AR (oracle {1-ar_max_blend:.0%})")
            else:
                logger.info(f"  ► AR blend complete — fully model-state rollout from here")

    epoch_metrics = {}  # Dynamic initialization - immune to trainer metric changes

    steps_per_epoch = config.steps_per_epoch if hasattr(config, 'steps_per_epoch') else 200

    # TASK_151: Calculate warmup progress for trade weight ramping
    warmup_epochs = getattr(config, 'weight_warmup_epochs', 50)
    warmup_progress = min(epoch / warmup_epochs, 1.0) if warmup_epochs > 0 else 1.0

    for step in range(steps_per_epoch):
        # TASK_715: Direct sampling when pool is active (bypasses buffer overhead)
        if generator.segment_pool is not None:
            batch = generator.generate_batch_direct(config.batch_size)
        else:
            batch = generator.generate_batch(config.batch_size)

        # TASK_151: Add warmup progress for trade weight ramping
        batch['warmup_progress'] = jnp.array(warmup_progress)

        # Train step (batch already contains JAX arrays - zero copy)
        metrics = trainer.train_step(batch)

        # Dynamic initialization on first step
        if not epoch_metrics:
            epoch_metrics = {k: [] for k in metrics.keys()}

        for key, value in metrics.items():
            if key in epoch_metrics:
                epoch_metrics[key].append(value)

        # Progress logging every 50 steps
        if (step + 1) % 50 == 0:
            avg_loss = np.mean(epoch_metrics.get('total_loss', [0])[-50:])
            avg_agree = np.mean(epoch_metrics.get('oracle_agreement', [0])[-50:])
            logger.info(f"  Epoch {epoch} Step {step+1}/{steps_per_epoch} | Loss: {avg_loss:.4f} | Agree: {avg_agree:.2%}")

    # Safe averaging (handles empty lists)
    return {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}


def validate(
    trainer: EasyTrainer,
    generator: OnTheFlyGenerator,
    config: EasyConfig,
    n_batches: int = 20,
) -> dict:
    """Validate with fresh generated data (uses trainer.evaluate_batch)."""
    val_metrics = {}  # Dynamic initialization

    for _ in range(n_batches):
        if generator.segment_pool is not None:
            batch = generator.generate_batch_direct(config.batch_size)
        else:
            batch = generator.generate_batch(config.batch_size)
        metrics = trainer.evaluate_batch(batch)  # Zero-copy, reuses trainer logic

        if not val_metrics:
            val_metrics = {k: [] for k in metrics.keys()}

        for k, v in metrics.items():
            if k in val_metrics:
                val_metrics[k].append(v)

    # Safe averaging
    return {k: np.mean(v) if v else 0.0 for k, v in val_metrics.items()}


# =============================================================================
# TASK_706: Periodic Benchmark (runs on separate GPU, non-blocking)
# =============================================================================

_benchmark_proc = None  # Track running benchmark subprocess

def _launch_benchmark(ckpt_path: str, gpu_id: int, db_path: str = "trading_data.db"):
    """Launch benchmark on separate GPU as non-blocking subprocess."""
    global _benchmark_proc
    logger = logging.getLogger(__name__)

    # Don't launch if previous benchmark still running
    if _benchmark_proc is not None and _benchmark_proc.poll() is None:
        logger.info(f"[BENCH] Previous benchmark still running (pid={_benchmark_proc.pid}), skipping")
        return

    results_path = Path(ckpt_path).parent / f"benchmark_{Path(ckpt_path).stem}.json"
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [
        sys.executable, "tools/benchmark_easy.py",
        "--checkpoint", str(ckpt_path),
        "--db", db_path,
        "--json", str(results_path),
        "--quiet",
    ]

    stderr_path = results_path.with_suffix('.stderr')
    logger.info(f"[BENCH] Launching benchmark on GPU {gpu_id}: {Path(ckpt_path).name} → {results_path.name}")
    _benchmark_proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.DEVNULL, stderr=open(stderr_path, 'w'),
    )


def _check_benchmark_passed(output_dir: Path) -> bool:
    """Check if any benchmark result in output_dir has passed."""
    import glob
    logger = logging.getLogger(__name__)

    for json_path in sorted(glob.glob(str(output_dir / "benchmark_*.json")), reverse=True):
        try:
            with open(json_path) as f:
                data = json.load(f)
            summary = data.get('summary', {})
            passed = summary.get('passed', False)
            if passed:
                logger.info(f"[BENCH] ★ PASSED: {json_path}")
                return True
            # Log latest result summary
            logger.info(
                f"[BENCH] {Path(json_path).stem}: "
                f"alpha+={summary.get('positive_alpha_count', 0)}/6, "
                f"beats_rnd={summary.get('beats_random_count', 0)}/6, "
                f"bal_acc={summary.get('avg_balanced_accuracy', 0):.1%}, "
                f"vmse={summary.get('avg_value_mse', 0):.4f}"
            )
            return False  # Only check latest
        except (json.JSONDecodeError, OSError):
            continue
    return False


def main():
    parser = argparse.ArgumentParser(description="Easy Mode Training (On-the-fly)")

    # Config
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to config JSON')
    parser.add_argument('--preset', type=str, choices=['small', 'medium', 'large'],
                        default='medium', help='Config preset')

    # Training
    parser.add_argument('--epochs', '-e', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--steps-per-epoch', type=int, default=200,
                        help='Training steps per epoch')

    # Data
    parser.add_argument('--levels', '-l', type=str, default='0-14',
                        help='Synthetic levels (e.g., "0-5" or "0,3,5")')
    parser.add_argument('--bella', action='store_true',
                        help='Use Bella (CNN diffusion) for market generation instead of synthetic')
    parser.add_argument('--bella-checkpoint', type=str,
                        default='checkpoints/diff_teacher/final/generator',
                        help='Path to Bella checkpoint')
    parser.add_argument('--invert-regime', action='store_true',
                        help='Invert regime (crash↔rally) for dual-Bella diversity')

    # Encoder
    parser.add_argument('--encoder-version', type=int, default=2, choices=[2, 3],
                        help='Encoder version: 2=V2 (flatten), 3=V3 (minGRU temporal)')

    # Checkpointing
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--output-dir', '-o', type=str, default='jax_ckpt_easy',
                        help='Output directory for checkpoints')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')

    # Other
    parser.add_argument('--n-envs', type=int, default=8,
                        help='Number of parallel environments for data generation')
    parser.add_argument('--buffer-size', type=int, default=32,
                        help='Episodes to buffer in VRAM (TASK_667)')
    parser.add_argument('--samples-per-episode', type=int, default=200,
                        help='Windows to sample per episode before regenerate')
    parser.add_argument('--mix-ratio', type=float, default=0.0,
                        help='Experience replay ratio (0.1 = 10%% from secondary source)')
    parser.add_argument('--secondary-levels', type=str, default='0',
                        help='Synthetic levels for mixing when primary is Bella')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--eval-only', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--no-latent-cache', action='store_true',
                        help='Disable latent state caching (for debugging)')
    parser.add_argument('--oracle-multipass', action='store_true',
                        help='TASK_673: Enable multipass oracle for soft policy targets')
    parser.add_argument('--benchmark', action='store_true',
                        help='TASK_682: Run real data benchmark after training completes')
    parser.add_argument('--benchmark-every', type=int, default=0,
                        help='TASK_706: Run benchmark every N epochs on separate GPU (0=disabled)')
    parser.add_argument('--benchmark-gpu', type=int, default=1,
                        help='TASK_706: GPU index for benchmark (default: 1 = CMP 90HX)')
    parser.add_argument('--target-agreement', type=float, default=None,
                        help='Override target oracle agreement for early stopping (default: config value)')
    parser.add_argument('--ar-max-blend', type=float, default=None,
                        help='Override AR blend cap (0.0-1.0, e.g., 0.88 to freeze at 88%% AR)')
    parser.add_argument('--oracle-mask-head', action='store_true',
                        help='TASK_695: Enable oracle mask prediction auxiliary head')

    # TASK_698: Historical data training
    parser.add_argument('--db-path', type=str, default=None,
                        help='SQLite database path (enables historical mode)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Period start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Period end date (YYYY-MM-DD)')
    parser.add_argument('--assets-file', type=str, default=None,
                        help='JSON file with asset symbol list (pre-padded to 64)')
    # TASK_707: Multi-period cycling (stratified historical training)
    parser.add_argument('--periods-file', type=str, default=None,
                        help='JSON file with multiple periods to cycle through (overrides --start-date/--end-date)')
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='Disable checkpoint saving (for eval runs)')
    # TASK_729b: Per-asset encoder width overrides (V2 encoder)
    parser.add_argument('--embed-dim', type=int, default=None,
                        help='Per-asset embedding width (V2 encoder). Default: preset value.')
    parser.add_argument('--encoder-heads', type=int, default=None,
                        help='Cross-asset attention head count. Default: preset value.')
    # TASK_733: Cheat-channel diagnostic (information vs architecture ceiling)
    parser.add_argument('--cheat-channel', action='store_true',
                        help='TASK_733: Add channel 32 = sign(r_fwd) at oracle_lookahead. '
                             'Perfect-information diagnostic. NEVER deploy a cheat checkpoint.')
    # TASK_741 (Mint): Post-pool per-asset scalar injection via FiLM.
    parser.add_argument('--per-asset-injection',
                        choices=['none', 'cheat', 'xs', 'both'],
                        default='none',
                        help='TASK_741 (Mint): FiLM-inject per-asset scalars '
                             'on the post-pool feature surface. xs is '
                             'production-safe; cheat is diagnostic only.')

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = EasyConfig.load(args.config)
    else:
        if args.preset == 'small':
            config = get_small_config()
        elif args.preset == 'large':
            config = get_large_config()
        else:
            config = get_medium_config()

    # Override config with CLI args
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.name:
        config.name = args.name

    # Parse levels
    if '-' in args.levels:
        start, end = map(int, args.levels.split('-'))
        config.synthetic_levels = list(range(start, end + 1))
    else:
        config.synthetic_levels = [int(x) for x in args.levels.split(',')]

    # Add steps_per_epoch to config
    config.steps_per_epoch = args.steps_per_epoch
    config.seed = args.seed
    config.checkpoint_dir = args.output_dir
    config.invert_regime = args.invert_regime  # Dual-Bella diversity
    config.encoder_version = args.encoder_version  # TASK_676: V2 (flatten) or V3 (minGRU)
    if args.oracle_multipass:
        config.oracle_multipass = True  # TASK_673
    if args.oracle_mask_head:
        config.use_oracle_mask_head = True  # TASK_695
    if args.target_agreement is not None:
        config.target_oracle_agreement = args.target_agreement
    if args.ar_max_blend is not None:
        config.ar_max_blend = args.ar_max_blend
    # TASK_729b: widen per-asset encoder capacity
    if args.embed_dim is not None:
        config.embed_dim = args.embed_dim
    if args.encoder_heads is not None:
        config.encoder_num_heads = args.encoder_heads
    # TASK_733: cheat-channel diagnostic — adds channel 32, bumps n_channels by 1
    if args.cheat_channel:
        config.use_cheat_channel = True
        config.n_channels = config.n_channels + 1
        logger.warning("=" * 60)
        logger.warning("TASK_733 CHEAT CHANNEL ENABLED — diagnostic only.")
        logger.warning(f"  n_channels: {config.n_channels - 1} -> {config.n_channels}")
        logger.warning(f"  cheat lookahead: {config.oracle_lookahead}")
        logger.warning("  This checkpoint MUST NOT be benchmarked or deployed.")
        logger.warning("=" * 60)

    # TASK_741 (Mint): post-pool per-asset scalar injection via FiLM
    if args.per_asset_injection != 'none':
        config.use_per_asset_injection = True
        config.per_asset_inject_xs = args.per_asset_injection in ('xs', 'both')
        config.per_asset_inject_cheat = args.per_asset_injection in ('cheat', 'both')
        if config.per_asset_inject_cheat and not config.use_cheat_channel:
            raise ValueError(
                "--per-asset-injection {cheat,both} requires --cheat-channel "
                "to also be set (cheat scalar is sourced from encoder channel 32)."
            )
        logger.info("=" * 60)
        logger.info(f"[MINT] Per-asset injection: {args.per_asset_injection}")
        logger.info(f"  xs scalars: {config.per_asset_inject_xs}")
        logger.info(f"  cheat scalar: {config.per_asset_inject_cheat}")
        logger.info("=" * 60)

    logger.info("=" * 60)
    logger.info("Easy Mode Training - On-the-fly VRAM Generation")
    logger.info("=" * 60)
    logger.info(f"Config: {config.name}")
    if args.bella:
        logger.info(f"Data source: Bella (CNN diffusion)")
        if args.invert_regime:
            logger.info(f"  Regime INVERTED (crash↔rally)")
    else:
        logger.info(f"Levels: {config.synthetic_levels}")
    logger.info(f"Epochs: {config.epochs}, Steps/epoch: {config.steps_per_epoch}")
    logger.info(f"Batch size: {config.batch_size}, LR: {config.learning_rate}")
    if getattr(config, 'oracle_multipass', False):
        logger.info(f"Oracle: MULTIPASS (n={config.oracle_n_passes}, noise={config.oracle_noise_scale})")  # TASK_673

    # Setup Bella generator if requested
    bella_generator = None
    secondary_bella_generator = None
    if args.bella:
        logger.info(f"Loading Bella generator from {args.bella_checkpoint}...")
        bella_generator, bella_config = setup_bella_generator(args.bella_checkpoint)
        logger.info(f"  Bella: {bella_config.architecture}, {bella_config.hidden_dim}d, {bella_config.num_layers}L")
        logger.info(f"  Output: ({bella_config.seq_length}, {bella_config.n_assets}, {bella_config.n_features})")
    elif args.mix_ratio > 0:
        # Primary is synthetic, need Bella for secondary mixing
        logger.info(f"Loading Bella for 90/10 mixing from {args.bella_checkpoint}...")
        try:
            secondary_bella_generator, _ = setup_bella_generator(args.bella_checkpoint)
        except Exception as e:
            logger.warning(f"Could not load Bella for mixing: {e}. Mixing disabled.")
            args.mix_ratio = 0.0

    # Parse secondary levels for Bella primary
    secondary_levels = [0]
    if '-' in args.secondary_levels:
        start, end = map(int, args.secondary_levels.split('-'))
        secondary_levels = list(range(start, end + 1))
    else:
        secondary_levels = [int(x) for x in args.secondary_levels.split(',')]

    # TASK_698 + TASK_707: Setup historical market(s)
    historical_market = None
    historical_markets = []
    historical_market_labels = []
    min_window = config.episode_length + config.window_size + getattr(config, 'oracle_lookahead', 20) + 5

    period_specs = []
    periods_db_path = None

    # TASK_710: SegmentPool instance (created below if --periods-file)
    segment_pool = None
    val_segment_pool = None

    if args.periods_file:
        # TASK_710: SegmentPool — linear historical training with regime diversity
        from mcts_trader.data.history import HistoricalMarket
        with open(args.periods_file) as f:
            periods_data = json.load(f)
        periods_list = periods_data.get('periods', periods_data if isinstance(periods_data, list) else [])
        periods_db_path = args.db_path or 'trading_data.db'
        for i, period in enumerate(periods_list):
            period_specs.append({
                'start_date': period['start_date'],
                'end_date': period['end_date'],
                'assets': period.get('assets', None),
            })
        if not period_specs:
            raise ValueError("No valid historical periods in periods-file!")

        # TASK_714: Full-period oracle precomputation + stride-1 walk
        window_len = config.episode_length + config.window_size + getattr(config, 'oracle_lookahead', 20) + 5
        oracle_cfg = {
            'oracle_k': config.oracle_k,
            'oracle_lookahead': config.oracle_lookahead,
            'transaction_cost': config.transaction_cost,
            'min_alpha_threshold': getattr(config, 'min_alpha_threshold', 0.0005),
            # TASK_733: cheat-channel visibility at slot load (no functional effect)
            'use_cheat_channel': getattr(config, 'use_cheat_channel', False),
            'cheat_lookahead': config.oracle_lookahead,
            'cheat_epsilon': 1e-4,
        }
        segment_pool = SegmentPool(
            period_specs, periods_db_path,
            n_assets=config.n_assets, n_slots=len(period_specs),
            window_length=window_len, min_assets=32,
            oracle_config=oracle_cfg,
        )
        # Val pool: 6 slots, 3 runs per range
        val_period_specs = list(reversed(period_specs))
        val_segment_pool = SegmentPool(
            val_period_specs, periods_db_path,
            n_assets=config.n_assets, n_slots=6, window_length=window_len,
            max_runs_per_group=3, min_assets=32,
            oracle_config=oracle_cfg,
        )
        logger.info(f"[TASK_714] SegmentPool: train={len(segment_pool.slots)} slots "
                     f"({len(segment_pool.group_keys)} regimes), "
                     f"val={len(val_segment_pool.slots)} slots")

    elif args.db_path:
        # Single-period mode (original TASK_698)
        from mcts_trader.data.history import HistoricalMarket
        logger.info(f"Loading historical data from {args.db_path}...")
        logger.info(f"  Period: {args.start_date} → {args.end_date}")

        hist_symbols = None
        if args.assets_file:
            import json as _json
            with open(args.assets_file) as f:
                hist_symbols = _json.load(f)
            logger.info(f"  Assets: {len(hist_symbols)} from {args.assets_file}")

        historical_market = HistoricalMarket(
            db_path=args.db_path,
            n_assets=config.n_assets,
            specific_symbols=hist_symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            use_ram_cache=True,
            skip_validation=True,
        )

        n_timestamps = len(historical_market.valid_timestamps)
        assert n_timestamps > min_window, (
            f"Period too short: {n_timestamps} timestamps < {min_window} required"
        )
        logger.info(f"  Loaded: {n_timestamps:,} timestamps, {historical_market.n_assets} assets, "
                    f"cache {historical_market.cache_data.nbytes / 1024**2:.1f} MB")

    # Create on-the-fly generator (no disk I/O!)
    logger.info("Creating on-the-fly data generator...")
    gpu_synth = bella_generator is None and historical_market is None and segment_pool is None
    logger.info(f"  Buffer: {args.buffer_size} episodes, {args.samples_per_episode} samples/ep, {args.n_envs} envs")
    if segment_pool is not None:
        logger.info(f"  Data source: SEGMENT POOL ({segment_pool._segments_loaded} active, "
                    f"{len(segment_pool.group_keys)} regimes, linear walk)")
    elif historical_market is not None:
        logger.info(f"  Data source: HISTORICAL ({args.start_date} → {args.end_date})")
    elif gpu_synth:
        logger.info(f"  Synthetic prices: GPU-accelerated (levels {sorted(config.synthetic_levels)})")
    if args.mix_ratio > 0:
        logger.info(f"  Experience Replay: {args.mix_ratio:.0%} from secondary source")
    generator = OnTheFlyGenerator(
        config,
        seed=config.seed,
        bella_generator=bella_generator,
        n_envs=args.n_envs,
        buffer_size=args.buffer_size,
        samples_per_episode=args.samples_per_episode,
        mix_ratio=args.mix_ratio,
        secondary_bella_generator=secondary_bella_generator,
        secondary_levels=secondary_levels,
        invert_regime=args.invert_regime,
        historical_market=historical_market,
        segment_pool=segment_pool,
    )

    # TASK_674: Separate validation generator (different seed, no train data leakage)
    # TASK_675: Smaller buffer — val only runs 20 batches/epoch, no need for full buffer
    # TASK_696: Val uses unbiased sampling (trade_sample_prob=0) to measure natural distribution
    from dataclasses import replace as _dc_replace
    val_config = _dc_replace(config, trade_sample_prob=0.0)
    val_buffer_size = max(args.buffer_size // 4, 4)
    val_generator = OnTheFlyGenerator(
        val_config,
        seed=config.seed + 7919,  # Different seed for val data
        bella_generator=bella_generator,
        n_envs=args.n_envs,
        buffer_size=val_buffer_size,
        samples_per_episode=args.samples_per_episode,
        mix_ratio=args.mix_ratio,
        secondary_bella_generator=secondary_bella_generator,
        secondary_levels=secondary_levels,
        invert_regime=args.invert_regime,
        historical_market=historical_market,
        segment_pool=val_segment_pool,
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(config, seed=config.seed)

    # Enable latent state caching (removes TRM encoder from training loop)
    if not getattr(args, 'no_latent_cache', False):
        logger.info("Enabling latent state caching...")
        generator.set_encoder(trainer.model, trainer.params)
        val_generator.set_encoder(trainer.model, trainer.params)

    # TASK_692/696: Model-state rollout with AR warmup schedule
    autoregressive_state = getattr(config, 'autoregressive_state', False)
    ar_warmup = getattr(config, 'ar_warmup_epochs', 0)
    ar_blend_epochs = getattr(config, 'ar_blend_epochs', 0)
    ar_max_blend = getattr(config, 'ar_max_blend', 1.0)
    if autoregressive_state:
        if ar_warmup > 0:
            blend_msg = f", {ar_blend_epochs}-epoch blend" if ar_blend_epochs > 0 else ""
            cap_msg = f", capped at {ar_max_blend:.0%}" if ar_max_blend < 1.0 else ""
            logger.info(f"Autoregressive state: warmup {ar_warmup} epochs (oracle state){blend_msg}{cap_msg}, then AR with DAgger targets")
        else:
            logger.info("Enabling autoregressive state channels (model-state rollout)...")
            generator.set_model_params(trainer.model, trainer.params)
            val_generator.set_model_params(trainer.model, trainer.params)

    # Load checkpoint if provided
    start_epoch = 0
    best_agreement = 0.0

    if args.checkpoint:
        ckpt_path = str(Path(args.checkpoint).resolve())
        checkpoint = trainer.load_checkpoint(ckpt_path)
        start_epoch = checkpoint['epoch'] + 1
        # BUG-6 fix: Merged checkpoints may lack 'oracle_agreement' — check branch keys too
        best_agreement = max(
            checkpoint['metrics'].get('oracle_agreement', 0.0),
            checkpoint['metrics'].get('a_agreement', 0.0),
            checkpoint['metrics'].get('b_agreement', 0.0),
        )
        logger.info(f"Resuming from epoch {start_epoch}, best agreement: {best_agreement:.1%}")
        # TASK_710: SegmentPool starts fresh on resume (walks linearly from beginning)
        # No offset needed — pool manages its own cursor state
        if segment_pool is not None:
            logger.info(f"  SegmentPool: starting fresh linear walk (pool manages cursors)")
        elif period_specs:
            # Legacy TASK_707: Offset period cycling to match resume position
            resume_offset = start_epoch % len(period_specs)
            generator._historical_period_idx = resume_offset
            generator.historical_market = generator._load_period(resume_offset)
            logger.info(f"  Period cycling offset to P{resume_offset}/{len(period_specs)}")

    # Eval only mode
    if args.eval_only:
        logger.info("Running evaluation only...")
        val_metrics = validate(trainer, val_generator, config, n_batches=50)
        logger.info("Validation metrics:")
        for key, value in val_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        return

    # Create output directory (must be absolute path for checkpoint saving)
    output_dir = (Path(config.checkpoint_dir) / config.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(output_dir / 'config.json'))

    # Create CSV logger (TASK_667)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_logger = CSVLogger(log_dir, config.name)

    # TASK_708: MSE-based AR feedback state (with TASK_712 ratchet)
    mse_ema = 1.0  # Start assuming bad calibration
    ar_high_water = getattr(config, 'ar_max_blend', 1.0)  # Ratchet: AR only goes up
    MSE_EMA_ALPHA = 0.1
    AR_FLOOR = 0.2
    AR_MSE_GOOD = 0.1   # MSE ≤ this → AR = 1.0
    AR_MSE_BAD = 10.0    # MSE ≥ this → AR = AR_FLOOR

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)

    training_start = time.time()

    # TASK_675: try/finally for clean shutdown on crash
    try:
        for epoch in range(start_epoch, config.epochs):
            epoch_start = time.time()

            # Train (on-the-fly generation, no disk I/O)
            train_metrics = train_epoch(trainer, generator, config, epoch)

            # TASK_711: NaN detection — abort early instead of burning GPU for hours
            if math.isnan(train_metrics.get('total_loss', 0)):
                logger.error("NaN loss detected — saving emergency checkpoint and stopping")
                if not getattr(args, 'no_checkpoint', False):
                    trainer.save_checkpoint(
                        str(output_dir / 'phase1_nan_abort.pkl'), epoch, train_metrics)
                break

            # Validate
            if (epoch + 1) % config.eval_every == 0 or epoch == config.epochs - 1:
                # TASK_691: Val generator needs same epoch for entropy schedule
                val_generator._current_epoch = epoch
                # BUG-5 fix: Update val_generator with current model params
                if autoregressive_state and epoch >= ar_warmup:
                    val_generator.set_model_params(trainer.model, trainer.params)
                val_metrics = validate(trainer, val_generator, config, n_batches=20)
            else:
                val_metrics = {}

            epoch_time = time.time() - epoch_start

            # TASK_708: MSE-based AR feedback — adjust ar_max_blend from val MSE
            # TASK_712: AR ratchet — AR only goes up, never down. Once MSE stabilizes
            # and AR reaches e.g. 86%, MSE spikes can't drop it back. Lets AR climb
            # monotonically toward 100% as value head calibrates.
            val_mse = val_metrics.get('value_mse', None)
            if val_mse is not None and autoregressive_state and math.isfinite(val_mse):
                mse_ema = MSE_EMA_ALPHA * val_mse + (1 - MSE_EMA_ALPHA) * mse_ema
                log_good = math.log10(AR_MSE_GOOD)   # -1.0
                log_bad = math.log10(AR_MSE_BAD)      # 1.0
                log_mse = math.log10(max(mse_ema, 1e-6))
                t = (log_bad - log_mse) / (log_bad - log_good)
                new_ar = AR_FLOOR + (1.0 - AR_FLOOR) * max(0.0, min(1.0, t))
                # Ratchet: only update if new AR is higher than previous high
                if new_ar > ar_high_water:
                    ar_high_water = new_ar
                    config.ar_max_blend = new_ar

            # Log metrics
            current_ar = getattr(config, 'ar_max_blend', 1.0)
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Loss: {train_metrics['total_loss']:.4f} | "
                f"Train Agree: {train_metrics['oracle_agreement']:.2%} | "
                f"Val Agree: {val_metrics.get('oracle_agreement', 0):.2%} | "
                f"Val MSE: {val_metrics.get('value_mse', 0):.4f} | "
                f"AR: {current_ar:.0%} | "
                f"Time: {epoch_time:.1f}s"
            )
            # TASK_710: Log pool status every 10 epochs
            if segment_pool is not None and epoch % 10 == 0:
                logger.info(f"  [POOL] {segment_pool.status()}")

            # CSV logging (TASK_667)
            csv_logger.log(epoch, {
                **train_metrics,
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'best_agreement': best_agreement,
                'ar_blend': getattr(config, 'ar_max_blend', 1.0),
                'mse_ema': mse_ema,
            }, epoch_time)

            # Check for improvement
            current_agreement = val_metrics.get('oracle_agreement', train_metrics['oracle_agreement'])

            if current_agreement > best_agreement:
                best_agreement = current_agreement
                if not getattr(args, 'no_checkpoint', False):
                    trainer.save_checkpoint(
                        str(output_dir / 'phase1_best.pkl'),
                        epoch,
                        {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
                    )
                logger.info(f"  ★ New best agreement: {best_agreement:.1%}")

            # Regular checkpoint
            if (epoch + 1) % config.save_every == 0 and not getattr(args, 'no_checkpoint', False):
                trainer.save_checkpoint(
                    str(output_dir / f'phase1_epoch{epoch}.pkl'),
                    epoch,
                    {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
                )
                # TASK_144: Tiered cleanup (keep last 5 + milestones)
                cleanup_tiered_checkpoints(output_dir, keep_recent=5, milestone_interval=100,
                                           save_every=config.save_every)
                # TASK_711: Save live config (TASK_708 mutates ar_max_blend at runtime)
                if (epoch + 1) % 100 == 0:
                    config.save(str(output_dir / 'config_live.json'))

            # TASK_706: Periodic benchmark on separate GPU
            benchmark_every = getattr(args, 'benchmark_every', 0)
            if benchmark_every > 0 and (epoch + 1) % benchmark_every == 0:
                ckpt_path = output_dir / f'phase1_epoch{epoch}.pkl'
                if ckpt_path.exists():
                    bench_db = getattr(args, 'db_path', None) or 'trading_data.db'
                    _launch_benchmark(str(ckpt_path), args.benchmark_gpu, bench_db)

            # TASK_706: Check if previous benchmark passed → stop training
            if benchmark_every > 0 and (epoch + 1) % 10 == 0:
                if _check_benchmark_passed(output_dir):
                    logger.info("=" * 60)
                    logger.info("BENCHMARK PASSED — stopping training")
                    logger.info("=" * 60)
                    trainer.save_checkpoint(
                        str(output_dir / 'benchmark_passed.pkl'),
                        epoch,
                        {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
                    )
                    break

            # Check success criteria: full AR + both metrics stable
            ar_max_blend = getattr(config, 'ar_max_blend', 1.0)
            ar_blend_end = ar_warmup + getattr(config, 'ar_blend_epochs', 0)
            in_warmup = autoregressive_state and ar_warmup > 0 and epoch < ar_blend_end
            # Only check when AR is fully rolled out (cap reached 100%) and past warmup.
            # NOTE: With TASK_708 MSE feedback active, ar_max_blend stays ~0.33 until
            # value head calibrates (MSE < 0.1). This intentionally disables early stop —
            # the benchmark pass (TASK_706) or pool exhaustion (TASK_710) are the real
            # stop criteria for historical training.
            if not in_warmup and ar_max_blend >= 1.0:
                train_agree = train_metrics['oracle_agreement']
                val_agree = val_metrics.get('oracle_agreement', 0)
                target = config.target_oracle_agreement
                delta = abs(train_agree - val_agree)

                # Both above target AND delta < 5% in same epoch
                epoch_pass = (train_agree >= target and val_agree >= target
                              and delta <= 0.05)

                if epoch_pass:
                    earlystop_streak = getattr(config, '_earlystop_streak', 0) + 1
                else:
                    earlystop_streak = 0
                config._earlystop_streak = earlystop_streak

                if earlystop_streak >= 5:
                    logger.info("=" * 60)
                    logger.info(f"SUCCESS! Phase 1 complete ({earlystop_streak} consecutive epochs)")
                    logger.info(f"  Train: {train_agree:.1%} | Val: {val_agree:.1%} | Delta: {delta:.1%} | Target: {target:.1%}")
                    logger.info("=" * 60)

                    trainer.save_checkpoint(
                        str(output_dir / 'phase1_complete.pkl'),
                        epoch,
                        {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
                    )
                    break

        # Training complete
        total_time = time.time() - training_start
        logger.info("=" * 60)
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Best oracle agreement: {best_agreement:.1%}")
        logger.info(f"Checkpoints saved to: {output_dir}")
        logger.info(f"CSV metrics: {csv_logger.csv_path}")
        logger.info("=" * 60)

    finally:
        csv_logger.close()
        generator.shutdown()
        val_generator.shutdown()

    # TASK_682: Post-training benchmark on real market data
    if args.benchmark:
        best_ckpt = str(output_dir / 'phase1_best.pkl')
        if Path(best_ckpt).exists():
            logger.info("Running real data benchmark (TASK_682)...")
            from tools.benchmark_easy import run_full_benchmark
            run_full_benchmark(best_ckpt, db_path="trading_data.db")
        else:
            logger.warning(f"No best checkpoint found at {best_ckpt}, skipping benchmark")


if __name__ == '__main__':
    main()

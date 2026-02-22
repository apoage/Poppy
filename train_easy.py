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
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from easy_config import EasyConfig, get_small_config, get_medium_config, get_large_config
from easy_trainer import EasyTrainer, create_trainer
from mcts_trader.data.synthetic_market import SyntheticMarket
from mcts_trader.jax_agent.path_solver import (
    greedy_lookahead_path, optimal_path_to_policy,
    batched_greedy_lookahead_path, batched_optimal_path_to_policy,
    multipass_oracle_policy, batched_multipass_oracle_policy,  # TASK_673
)
from mcts_trader.env.jax_encoder import (
    EncoderConfig, jit_batch_encode_compressed, jit_vmap_encode_compressed,
    jit_batch_encode_temporal, jit_vmap_encode_temporal
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
        scan_step, init_state, jnp.arange(T)
    )
    return model_state_trajectory, confidence_trajectory  # (T, 6, n_assets), (T, n_assets)


# =============================================================================
# Asynchronous Data Generation (CPU Background Thread, GPU Foreground)
# =============================================================================

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
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.bella_generator = bella_generator
        self.n_envs = n_envs  # TASK_140: Parallel environments
        self.invert_regime = invert_regime  # Dual-Bella: swap crash↔rally
        self.jax_rng = jax.random.PRNGKey(seed)
        self.encoder_config = EncoderConfig(
            n_assets=config.n_assets,
            window_size=config.window_size
        )

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

        # Start background thread for price generation (only for synthetic)
        self.executor = ThreadPoolExecutor(max_workers=self.N_WORKERS)
        if bella_generator is None:
            self._submit_prefetch_tasks()

        # Buffer: pre-computed episode data in VRAM
        self.buffer = []
        self.buffer_sample_counts = []

        # Initial refill
        self._refill_buffer()
        self.global_step = 0

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

    def _generate_episode_data(self, level: int, prices_cpu: np.ndarray) -> dict:
        """Process raw prices into training data using GPU."""
        prices_jax = jnp.array(prices_cpu)

        # Compute oracle (GPU JIT)
        log_returns = jnp.log(prices_jax[1:] / jnp.maximum(prices_jax[:-1], 1e-8))

        # TASK_673: Multipass oracle (soft targets) or single-pass (hard targets)
        if getattr(self.config, 'oracle_multipass', False):
            self.jax_rng, subkey = jax.random.split(self.jax_rng)
            policy_targets, cumulative_pnl, _, oracle_selections = multipass_oracle_policy(
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
        else:
            optimal_path = greedy_lookahead_path(
                log_returns,
                k=self.config.oracle_k,
                lookahead=self.config.oracle_lookahead,
                transaction_cost=self.config.transaction_cost,
                min_alpha_threshold=getattr(self.config, 'min_alpha_threshold', 0.0),
            )

            # TASK_141: Compute prev_selections for state-aware oracle
            T, n_assets = log_returns.shape
            if getattr(self.config, 'state_aware_oracle', False):
                prev_selections = jnp.zeros((T, n_assets))
                prev_selections = prev_selections.at[1:].set(
                    jnp.where(optimal_path.selections[:-1] > 0.5,
                              jnp.where(log_returns[:-1] > 0, 1.0, -1.0),
                              0.0)
                )
            else:
                prev_selections = None

            policy_targets = optimal_path_to_policy(
                optimal_path.selections,
                log_returns,
                prev_selections=prev_selections,
                n_actions_per_asset=5,
                temperature=1.0,
            )
            cumulative_pnl = optimal_path.cumulative_pnl
            oracle_selections = optimal_path.selections

        # TASK_670 FIX: Compute REMAINING PnL (future value), not cumulative (past value)
        # MuZero value head must predict expected future returns, not remember past gains
        remaining_pnl = cumulative_pnl[-1] - cumulative_pnl

        # TASK_689/692/693: Build state channels + encode observations
        T, n_assets = log_returns.shape
        use_state_channels = self.config.n_channels > 27
        autoregressive_state = getattr(self.config, 'autoregressive_state', False)
        use_model_rollout = (use_state_channels and autoregressive_state
                             and self._model_params is not None)

        n_valid = len(log_returns) - self.config.window_size
        if n_valid <= 0:
            return None

        window_starts = jnp.arange(n_valid)

        if use_model_rollout:
            # TASK_692/693: Model-state rollout — state from model's own actions
            # Encode market observations (27ch)
            observations = jit_batch_encode_temporal(
                prices_jax, window_starts, self.encoder_config
            )  # (n_valid, 27, ws, n_assets)

            # Roll model through episode using current params (JIT'd)
            # BUG-1 fix: Align returns to window decision points.
            # Window idx covers prices[idx:idx+ws], decision at idx+ws-1.
            # Slice returns so returns_aligned[idx] = log_returns[idx + ws - 1].
            ws = self.config.window_size
            returns_aligned = log_returns[ws - 1:]
            state_channels, confidence = self._model_state_rollout(
                observations,       # (n_valid, 27, ws, n_assets)
                returns_aligned,    # (T - ws + 1, n_assets) — aligned to decision points
                self._model_params,
                self._model_apply_fn,
                n_assets=n_assets,
                k=self.config.oracle_k,
                window_size=self.config.window_size,
                decay_fast=0.80,
                decay_slow=0.99,
                n_actions_per_asset=5,
            )  # (n_valid, 6, n_assets), (n_valid, n_assets)

        elif use_state_channels:
            # TASK_689/691: Oracle-state channels (fallback when model params not available)
            p_drop, q_phantom = self._get_entropy_schedule()
            if p_drop > 0 or q_phantom > 0:
                self.jax_rng, entropy_key = jax.random.split(self.jax_rng)
                noisy_selections = corrupt_oracle_selections(
                    oracle_selections, entropy_key, p_drop, q_phantom
                )
                target_position = compute_target_position(noisy_selections, log_returns)
            else:
                target_position = compute_target_position(oracle_selections, log_returns)
            state_channels = build_state_channels(
                target_position, log_returns,
                n_assets=n_assets,
                k=self.config.oracle_k,
                window_size=self.config.window_size,
            )  # (T, 6, n_assets)
            # Confidence channel: oracle softmax max per asset
            oracle_per_asset = policy_targets.reshape(T, n_assets, 5)
            confidence = jnp.max(oracle_per_asset, axis=-1)  # (T, n_assets)

        # Encode market observations if not already done by model rollout
        if not use_model_rollout:
            observations = jit_batch_encode_temporal(
                prices_jax, window_starts, self.encoder_config
            )  # (n_valid, 27, ws, n_assets)

        # Alignment
        start_idx = self.config.window_size - 1
        end_idx = min(start_idx + n_valid, len(policy_targets))

        # TASK_689/693: Concatenate state channels (tiled across window)
        if use_state_channels:
            n_windows = end_idx - start_idx
            ws = self.config.window_size

            if use_model_rollout:
                # TASK_693: Model rollout outputs are indexed by WINDOW (0..n_valid-1).
                # scan output[w] = state BEFORE model processes window w.
                # For training window w, t-1 state = scan output[w].
                state_at_windows = state_channels[:n_windows]  # (n_windows, 6, n_assets)
                conf_at_windows = confidence[:n_windows]       # (n_windows, n_assets)
            else:
                # Oracle state channels indexed by RETURN TIMESTEP (0..T-1).
                # Use t-1 to avoid leaking current oracle action.
                state_at_windows = state_channels[start_idx - 1:end_idx - 1]  # (n_windows, 6, n_assets)
                conf_at_windows = confidence[start_idx - 1:end_idx - 1]       # (n_windows, n_assets)

            state_tiled = jnp.tile(
                state_at_windows[:, :, None, :],  # (n_windows, 6, 1, n_assets)
                (1, 1, ws, 1)
            )  # (n_windows, 6, ws, n_assets)

            conf_tiled = jnp.tile(
                conf_at_windows[:, None, None, :],  # (n_windows, 1, 1, n_assets)
                (1, 1, ws, 1)
            )  # (n_windows, 1, ws, n_assets)

            # Market obs: (n_windows, 27, ws, n_assets)
            market_obs = observations[:n_windows]
            # Full obs: (n_windows, 34, ws, n_assets)
            observations = jnp.concatenate([market_obs, state_tiled, conf_tiled], axis=1)

        ep_data = {
            'observations': observations[:end_idx - start_idx],
            'actions': policy_targets[start_idx:end_idx],  # Soft targets (320-dim), not argmax
            'returns': remaining_pnl[start_idx:end_idx],   # Remaining PnL, not cumulative
            'level': level,
            'n_samples': end_idx - start_idx,
        }

        # TASK_695: Oracle mask for auxiliary prediction head
        if getattr(self.config, 'use_oracle_mask_head', False):
            ep_data['oracle_mask'] = oracle_selections[start_idx:end_idx]

        # Pre-compute latent states if encoder is available (removes TRM from training loop)
        if self._use_latent_cache and self._encoder_fn is not None:
            ep_data['latent_states'] = self._encoder_fn(ep_data['observations'])

        return ep_data

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

        # Stack prices into batch (N, T, n_assets)
        # Pad to same length if needed
        max_len = max(p.shape[0] for p in prices_list)
        n_assets = prices_list[0].shape[1]

        prices_padded = np.zeros((n_batch, max_len, n_assets), dtype=np.float32)
        valid_lengths = []
        for i, p in enumerate(prices_list):
            prices_padded[i, :p.shape[0]] = p
            valid_lengths.append(p.shape[0])

        prices_jax = jnp.array(prices_padded)

        # Compute log returns (N, T-1, n_assets)
        log_returns = jnp.log(prices_jax[:, 1:] / jnp.maximum(prices_jax[:, :-1], 1e-8))

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

        # TASK_152: Super-batched temporal encoding (N, L, 27, 64, 64)
        all_observations = jit_vmap_encode_temporal(
            prices_jax,
            window_starts,
            self.encoder_config
        )

        # TASK_689: Build state channels per episode
        use_state_channels = self.config.n_channels > 27
        autoregressive_state = getattr(self.config, 'autoregressive_state', False)
        # TASK_691: Oracle entropy schedule (computed once per batch)
        p_drop, q_phantom = self._get_entropy_schedule() if use_state_channels else (0.0, 0.0)

        # Process each episode (alignment and dict creation)
        episode_datas = []
        for i in range(n_batch):
            T = valid_lengths[i] - 1  # log_returns length
            # Alignment
            start_idx = self.config.window_size - 1
            end_idx = min(start_idx + n_valid, T)
            n_windows = end_idx - start_idx

            obs_i = all_observations[i, :n_windows]

            # TASK_689/692/693: Append state channels
            if use_state_channels and autoregressive_state and self._model_params is not None:
                # TASK_692/693: Model-state rollout with incremental position stepping
                market_obs_i = all_observations[i]  # (n_valid, 27, ws, n_assets)
                # BUG-1 fix: Align returns to window decision points.
                # valid_lengths[i] is prices length, returns length is valid_lengths[i]-1,
                # aligned returns start at ws-1.
                ws = self.config.window_size
                log_returns_i = log_returns[i, ws - 1:valid_lengths[i] - 1]  # aligned to decision points
                state_ch_i, conf_i = self._model_state_rollout(
                    market_obs_i,
                    log_returns_i,
                    self._model_params,
                    self._model_apply_fn,
                    n_assets=n_assets,
                    k=self.config.oracle_k,
                    window_size=self.config.window_size,
                    decay_fast=0.80,
                    decay_slow=0.99,
                    n_actions_per_asset=5,
                )  # (n_valid, 6, n_assets), (n_valid, n_assets)

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

            ep_data = {
                'observations': obs_i,
                'actions': policy_targets[i, start_idx:end_idx],    # Soft targets (320-dim)
                'returns': remaining_pnl[i, start_idx:end_idx],     # Remaining PnL
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

    def _generate_prices_bella(self) -> np.ndarray:
        """Generate prices using Bella (on-device, VRAM-direct)."""
        self.jax_rng, subkey = jax.random.split(self.jax_rng)
        prices_jax = generate_bella_prices_on_device(
            self.bella_generator, subkey, n_envs=1, invert_regime=self.invert_regime
        )
        # generate_bella_prices_on_device returns (seq_len, n_assets) when n_envs=1
        prices_np = np.array(prices_jax)  # (seq_len, n_assets)
        return prices_np

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

            # TASK_667: 90/10 Experience Replay
            use_secondary = self._should_use_secondary()
            if use_secondary and (self.secondary_bella_generator is not None or
                                   (self.bella_generator is not None and self.secondary_levels)):
                prices_list, levels = self._generate_secondary_prices(n_to_refill)
                logger.debug(f"Refill from SECONDARY source ({len(prices_list)} episodes)")
            elif self.bella_generator is not None:
                # Bella: generate N prices in single GPU call
                self.jax_rng, subkey = jax.random.split(self.jax_rng)
                prices_batch = generate_bella_prices_on_device(
                    self.bella_generator, subkey, n_envs=n_to_refill,
                    invert_regime=self.invert_regime
                )  # (N, seq_len, n_assets)
                prices_list = []
                levels = []
                for i in range(n_to_refill):
                    prices_list.append(np.array(prices_batch[i]))
                    levels.append(-1)
            else:
                # Synthetic: drain queue then generate remaining
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
                episode_datas = self._generate_batch_episodes(prices_list, levels)
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
        all_oracle_masks = []  # TASK_695

        samples_needed = batch_size

        while samples_needed > 0:
            # Find episode with samples remaining
            available = [i for i, cnt in enumerate(self.buffer_sample_counts)
                        if cnt < self.SAMPLES_PER_EPISODE]

            if not available:
                # All exhausted, refill buffer (batched CPU work)
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

            # TASK_680: Trade-biased offset selection (coin-flip, O(1))
            max_offset = max(0, n_available - take_count)
            trade_indices = ep_data.get('trade_indices', np.array([]))
            # BUG-7 fix: Filter to valid offsets before sampling (prevents pile-up at boundary)
            valid_trades = trade_indices[trade_indices <= max_offset]
            if (len(valid_trades) > 0
                    and max_offset > 0
                    and self.rng.random() < self.config.trade_sample_prob):
                offset = int(self.rng.choice(valid_trades))
            else:
                offset = self.rng.integers(0, max_offset + 1) if max_offset > 0 else 0

            all_obs.append(ep_data['observations'][offset:offset + take_count])
            if self._use_latent_cache and 'latent_states' in ep_data:
                all_latents.append(ep_data['latent_states'][offset:offset + take_count])
            all_actions.append(ep_data['actions'][offset:offset + take_count])
            all_returns.append(ep_data['returns'][offset:offset + take_count])
            # TASK_695: Oracle mask
            if 'oracle_mask' in ep_data:
                all_oracle_masks.append(ep_data['oracle_mask'][offset:offset + take_count])

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

        # TASK_695: Oracle mask for auxiliary head
        if all_oracle_masks:
            batch['oracle_mask'] = jnp.concatenate(all_oracle_masks, axis=0)[:batch_size]

        self.global_step += 1
        return batch


# =============================================================================
# Tiered Checkpoint Management (TASK_144)
# =============================================================================

def cleanup_tiered_checkpoints(dir_path: Path, keep_recent: int = 5, milestone_interval: int = 100):
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
        # Keep if: recent OR milestone (every 100 epochs)
        if epoch in latest_epochs or epoch % milestone_interval == 0:
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

        # Handle new fields that weren't in header
        for key in row:
            if key not in self.fieldnames:
                row.pop(key)

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

    # TASK_692: Update generator's model params for autoregressive state rollout
    # Call at start of each epoch to refresh partial buffer with new params
    autoregressive_state = getattr(config, 'autoregressive_state', False)
    if autoregressive_state:
        generator.set_model_params(trainer.model, trainer.params)

    epoch_metrics = {}  # Dynamic initialization - immune to trainer metric changes

    steps_per_epoch = config.steps_per_epoch if hasattr(config, 'steps_per_epoch') else 200

    # TASK_151: Calculate warmup progress for trade weight ramping
    warmup_epochs = getattr(config, 'weight_warmup_epochs', 50)
    warmup_progress = min(epoch / warmup_epochs, 1.0) if warmup_epochs > 0 else 1.0

    for step in range(steps_per_epoch):
        # Generate fresh batch on-the-fly (no disk I/O)
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
        batch = generator.generate_batch(config.batch_size)
        metrics = trainer.evaluate_batch(batch)  # Zero-copy, reuses trainer logic

        if not val_metrics:
            val_metrics = {k: [] for k in metrics.keys()}

        for k, v in metrics.items():
            if k in val_metrics:
                val_metrics[k].append(v)

    # Safe averaging
    return {k: np.mean(v) if v else 0.0 for k, v in val_metrics.items()}


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
    parser.add_argument('--target-agreement', type=float, default=None,
                        help='Override target oracle agreement for early stopping (default: config value)')
    parser.add_argument('--oracle-mask-head', action='store_true',
                        help='TASK_695: Enable oracle mask prediction auxiliary head')

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

    # Create on-the-fly generator (no disk I/O!)
    logger.info("Creating on-the-fly data generator...")
    logger.info(f"  Buffer: {args.buffer_size} episodes, {args.samples_per_episode} samples/ep, {args.n_envs} envs")
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
    )

    # TASK_674: Separate validation generator (different seed, no train data leakage)
    # TASK_675: Smaller buffer — val only runs 20 batches/epoch, no need for full buffer
    val_buffer_size = max(args.buffer_size // 4, 4)
    val_generator = OnTheFlyGenerator(
        config,
        seed=config.seed + 7919,  # Different seed for val data
        bella_generator=bella_generator,
        n_envs=args.n_envs,
        buffer_size=val_buffer_size,
        samples_per_episode=args.samples_per_episode,
        mix_ratio=args.mix_ratio,
        secondary_bella_generator=secondary_bella_generator,
        secondary_levels=secondary_levels,
        invert_regime=args.invert_regime,
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(config, seed=config.seed)

    # Enable latent state caching (removes TRM encoder from training loop)
    if not getattr(args, 'no_latent_cache', False):
        logger.info("Enabling latent state caching...")
        generator.set_encoder(trainer.model, trainer.params)
        val_generator.set_encoder(trainer.model, trainer.params)

    # TASK_692: Enable model-state rollout for autoregressive state channels
    autoregressive_state = getattr(config, 'autoregressive_state', False)
    if autoregressive_state:
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

            # Validate
            if (epoch + 1) % config.eval_every == 0 or epoch == config.epochs - 1:
                # TASK_691: Val generator needs same epoch for entropy schedule
                val_generator._current_epoch = epoch
                # BUG-5 fix: Update val_generator with current model params
                if autoregressive_state:
                    val_generator.set_model_params(trainer.model, trainer.params)
                val_metrics = validate(trainer, val_generator, config, n_batches=20)
            else:
                val_metrics = {}

            epoch_time = time.time() - epoch_start

            # Log metrics
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Loss: {train_metrics['total_loss']:.4f} | "
                f"Train Agree: {train_metrics['oracle_agreement']:.2%} | "
                f"Val Agree: {val_metrics.get('oracle_agreement', 0):.2%} | "
                f"Val MSE: {val_metrics.get('value_mse', 0):.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # CSV logging (TASK_667)
            csv_logger.log(epoch, {
                **train_metrics,
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'best_agreement': best_agreement,
            }, epoch_time)

            # Check for improvement
            current_agreement = val_metrics.get('oracle_agreement', train_metrics['oracle_agreement'])

            if current_agreement > best_agreement:
                best_agreement = current_agreement
                trainer.save_checkpoint(
                    str(output_dir / 'phase1_best.pkl'),
                    epoch,
                    {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
                )
                logger.info(f"  ★ New best agreement: {best_agreement:.1%}")

            # Regular checkpoint
            if (epoch + 1) % config.save_every == 0:
                trainer.save_checkpoint(
                    str(output_dir / f'phase1_epoch{epoch}.pkl'),
                    epoch,
                    {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
                )
                # TASK_144: Tiered cleanup (keep last 5 + milestones)
                cleanup_tiered_checkpoints(output_dir, keep_recent=5, milestone_interval=100)

            # Check success criteria
            if current_agreement >= config.target_oracle_agreement:
                logger.info("=" * 60)
                logger.info(f"SUCCESS! Oracle agreement {current_agreement:.1%} >= {config.target_oracle_agreement:.1%}")
                logger.info("Phase 1 training complete!")
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

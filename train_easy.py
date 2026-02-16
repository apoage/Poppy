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

        # 90/10 Experience Replay (TASK_667)
        self.mix_ratio = mix_ratio
        self.secondary_bella_generator = secondary_bella_generator
        self.secondary_levels = secondary_levels or [0]
        self._mix_counter = 0

        # Latent state caching (set by set_encoder after trainer is created)
        self._encoder_fn = None
        self._use_latent_cache = False

        # Background pre-fetching setup (threading, not multiprocessing)
        self.price_queue = Queue(maxsize=self.N_BUFFER)
        self.shutdown_event = threading.Event()

        # Thread-local storage for SyntheticMarket instances (TASK_667 thread-safety fix)
        self._thread_local = threading.local()
        self._market_lock = threading.Lock()
        self._base_seed = seed
        self._thread_counter = 0  # TASK_675: Atomic counter for unique thread seeds

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

    def _generate_episode_data(self, level: int, prices_cpu: np.ndarray) -> dict:
        """Process raw prices into training data using GPU."""
        prices_jax = jnp.array(prices_cpu)

        # Compute oracle (GPU JIT)
        log_returns = jnp.log(prices_jax[1:] / jnp.maximum(prices_jax[:-1], 1e-8))

        # TASK_673: Multipass oracle (soft targets) or single-pass (hard targets)
        if getattr(self.config, 'oracle_multipass', False):
            self.jax_rng, subkey = jax.random.split(self.jax_rng)
            policy_targets, cumulative_pnl, _ = multipass_oracle_policy(
                log_returns,
                subkey,
                k=self.config.oracle_k,
                lookahead=self.config.oracle_lookahead,
                transaction_cost=self.config.transaction_cost,
                min_alpha_threshold=getattr(self.config, 'min_alpha_threshold', 0.0),
                n_passes=self.config.oracle_n_passes,
                noise_scale=self.config.oracle_noise_scale,
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

        # TASK_670 FIX: Compute REMAINING PnL (future value), not cumulative (past value)
        # MuZero value head must predict expected future returns, not remember past gains
        remaining_pnl = cumulative_pnl[-1] - cumulative_pnl

        # Encode observations (GPU JIT)
        n_valid = len(log_returns) - self.config.window_size
        if n_valid <= 0:
            return None

        window_starts = jnp.arange(n_valid)
        # TASK_152: Use temporal encoder (27, 64, 64) with real time series
        observations = jit_batch_encode_temporal(prices_jax, window_starts, self.encoder_config)

        # Alignment
        start_idx = self.config.window_size - 1
        end_idx = min(start_idx + n_valid, len(policy_targets))

        ep_data = {
            'observations': observations[:end_idx - start_idx],
            'actions': policy_targets[start_idx:end_idx],  # Soft targets (320-dim), not argmax
            'returns': remaining_pnl[start_idx:end_idx],   # Remaining PnL, not cumulative
            'level': level,
            'n_samples': end_idx - start_idx,
        }

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
            policy_targets, cumulative_pnl_batch, _ = batched_multipass_oracle_policy(
                log_returns,
                subkey,
                k=self.config.oracle_k,
                lookahead=self.config.oracle_lookahead,
                transaction_cost=self.config.transaction_cost,
                min_alpha_threshold=getattr(self.config, 'min_alpha_threshold', 0.0),
                n_passes=self.config.oracle_n_passes,
                noise_scale=self.config.oracle_noise_scale,
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

        # Process each episode (alignment and dict creation)
        episode_datas = []
        for i in range(n_batch):
            T = valid_lengths[i] - 1  # log_returns length
            # Alignment
            start_idx = self.config.window_size - 1
            end_idx = min(start_idx + n_valid, T)

            ep_data = {
                'observations': all_observations[i, :end_idx - start_idx],
                'actions': policy_targets[i, start_idx:end_idx],    # Soft targets (320-dim)
                'returns': remaining_pnl[i, start_idx:end_idx],     # Remaining PnL
                'level': levels[i],
                'n_samples': end_idx - start_idx,
            }

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
                    self.buffer.append(ep_data)
                    self.buffer_sample_counts.append(0)
                
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

        samples_needed = batch_size

        while samples_needed > 0:
            # Find episode with samples remaining
            available = [i for i, cnt in enumerate(self.buffer_sample_counts)
                        if cnt < self.SAMPLES_PER_EPISODE]

            if not available:
                # All exhausted, refill buffer (batched CPU work)
                self._refill_buffer()
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

            # Random offset within episode
            max_offset = max(0, n_available - take_count)
            offset = self.rng.integers(0, max_offset + 1) if max_offset > 0 else 0

            all_obs.append(ep_data['observations'][offset:offset + take_count])
            if self._use_latent_cache and 'latent_states' in ep_data:
                all_latents.append(ep_data['latent_states'][offset:offset + take_count])
            all_actions.append(ep_data['actions'][offset:offset + take_count])
            all_returns.append(ep_data['returns'][offset:offset + take_count])

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
            logger.info(f"  Epoch {epoch} Step {step+1}/{steps_per_epoch} | Loss: {avg_loss:.4f} | Agree: {avg_agree:.1%}")

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
    if args.oracle_multipass:
        config.oracle_multipass = True  # TASK_673

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

    # Load checkpoint if provided
    start_epoch = 0
    best_agreement = 0.0

    if args.checkpoint:
        ckpt_path = str(Path(args.checkpoint).resolve())
        checkpoint = trainer.load_checkpoint(ckpt_path)
        start_epoch = checkpoint['epoch'] + 1
        best_agreement = checkpoint['metrics'].get('oracle_agreement', 0.0)
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
                val_metrics = validate(trainer, val_generator, config, n_batches=20)
            else:
                val_metrics = {}

            epoch_time = time.time() - epoch_start

            # Log metrics
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Loss: {train_metrics['total_loss']:.4f} | "
                f"Train Agree: {train_metrics['oracle_agreement']:.1%} | "
                f"Val Agree: {val_metrics.get('oracle_agreement', 0):.1%} | "
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


if __name__ == '__main__':
    main()

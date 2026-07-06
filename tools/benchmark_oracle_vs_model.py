#!/usr/bin/env python3
"""
TASK_155: Benchmark Oracle vs Model Performance

Comprehensive analytics comparing:
1. Oracle (hindsight-optimal) performance
2. Model (learned) performance
3. Per-action breakdown (HOLD/BUY/SELL agreement)
4. Profitability comparison
5. Book status (positive/negative P&L)

Usage:
    python tools/benchmark_oracle_vs_model.py --checkpoint jax_ckpt_easy_merged/merged_latest.pkl
    python tools/benchmark_oracle_vs_model.py --checkpoint jax_ckpt_easy_merged/merged_latest.pkl --episodes 50
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from functools import partial

# Project imports
from easy_config import EasyConfig, get_large_config
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE
from mcts_trader.jax_agent.path_solver import (
    batched_greedy_lookahead_path,
    greedy_lookahead_path,
)
from mcts_trader.env.jax_encoder import EncoderConfig, jit_batch_encode_temporal
from mcts_trader.data.synthetic_market import SyntheticMarket


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ActionStats:
    """Statistics for a single action type."""
    oracle_count: int = 0
    model_count: int = 0
    agreement_count: int = 0

    @property
    def agreement_pct(self) -> float:
        return self.agreement_count / max(self.oracle_count, 1) * 100


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    # Dataset info
    n_episodes: int = 0
    n_steps: int = 0
    n_assets: int = 64

    # Agreement metrics
    overall_agreement: float = 0.0
    action_stats: Dict[str, ActionStats] = field(default_factory=dict)

    # Distributions
    oracle_distribution: Dict[str, float] = field(default_factory=dict)
    model_distribution: Dict[str, float] = field(default_factory=dict)

    # Confusion matrix
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Oracle performance (oracle-style P&L: abs returns)
    oracle_total_pnl: float = 0.0
    oracle_avg_pnl: float = 0.0
    oracle_positive_episodes: int = 0
    oracle_trade_count: int = 0

    # Model performance
    model_total_pnl: float = 0.0
    model_avg_pnl: float = 0.0
    model_positive_episodes: int = 0
    model_trade_count: int = 0

    # Capture ratio
    pnl_capture_ratio: float = 0.0

    # Enhanced metrics
    return_weighted_accuracy: float = 0.0
    neighborhood_agreement: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0

    # Trade-level metrics
    trade_precision: float = 0.0
    trade_recall: float = 0.0
    trade_f1: float = 0.0

    # Temporal
    temporal_agreement: List[float] = field(default_factory=list)


# =============================================================================
# Checkpoint & Model Loading
# =============================================================================

def load_checkpoint(checkpoint_path: str) -> Tuple[dict, dict]:
    """Load model params from e-dojo checkpoint."""
    ckpt_path = Path(checkpoint_path)

    if ckpt_path.suffix == '.pkl':
        with open(ckpt_path, 'rb') as f:
            data = pickle.load(f)
        params = data.get('params', data.get('model_params'))
        metadata = {k: v for k, v in data.items() if k != 'params'}
        return params, metadata
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path.suffix}")


def create_model(config: EasyConfig) -> MuZeroNetworks:
    """Create MuZeroNetworks model from config."""
    return MuZeroNetworks(
        state_dim=config.state_dim,
        action_space_size=config.action_space_size,
        support_size=VALUE_SUPPORT_SIZE,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
    )


# =============================================================================
# Data Generation
# =============================================================================

def generate_validation_data(
    config: EasyConfig,
    n_episodes: int,
    episode_length: int,
    seed: int = 42,
    use_bella: bool = False,
) -> np.ndarray:
    """Generate price data for benchmarking.

    Returns: (n_episodes, episode_length + window_size, n_assets) prices
    """
    total_length = episode_length + config.window_size

    if use_bella:
        # Try to load Bella generator
        bella_path = Path("checkpoints/diff_teacher/final/generator")
        if bella_path.exists():
            print(f"[BENCHMARK] Loading Bella generator...")
            from mcts_trader.training.setup import setup_bella_generator, generate_bella_prices_on_device

            bella_gen, bella_config = setup_bella_generator(str(bella_path))

            # Check Bella's seq_length vs our required total_length
            bella_seq_len = bella_gen.config.seq_length
            if bella_seq_len < total_length:
                print(f"[BENCHMARK] Warning: Bella seq_length ({bella_seq_len}) < required ({total_length})")
                print(f"[BENCHMARK] Adjusting episode_length to {bella_seq_len - config.window_size}")
                # Caller will need to handle this - generate what we can
                total_length = bella_seq_len

            # Generate all episodes in one batched call
            key = jax.random.PRNGKey(seed)
            prices_batch = generate_bella_prices_on_device(
                bella_gen, key, n_envs=n_episodes,
                ohlcv=False,  # TASK_703: oracle benchmark only needs close
            )
            # prices_batch: (n_episodes, seq_len, n_assets)
            prices_batch = prices_batch[:, :total_length, :]

            return np.array(prices_batch)
        else:
            print("[BENCHMARK] Bella not found, falling back to synthetic")
            use_bella = False

    # Synthetic fallback
    print("[BENCHMARK] Using synthetic market data")

    all_prices = []
    for i in range(n_episodes):
        # Create fresh market with different seed per episode
        market = SyntheticMarket(
            level=0,
            n_assets=config.n_assets,
            n_steps=total_length,
            seed=seed + i
        )

        # Extract close prices from data dict
        # data is {symbol: {open, high, low, close}} with (n_steps,) arrays
        close_prices = np.zeros((total_length, config.n_assets))
        for asset_idx in range(config.n_assets):
            symbol = f'SYN_{asset_idx:02d}'
            close_prices[:, asset_idx] = market.data[symbol]['close']

        all_prices.append(close_prices)

    return np.stack(all_prices)


def compute_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from prices. Shape: (T-1, n_assets)"""
    return np.log(prices[1:] / np.maximum(prices[:-1], 1e-10))


# =============================================================================
# Oracle Execution
# =============================================================================

def run_oracle(
    returns_batch: jnp.ndarray,
    config: EasyConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run oracle on batched returns.

    Args:
        returns_batch: (N, T, n_assets)

    Returns:
        selections: (N, T, n_assets) binary mask
        cumulative_pnl: (N, T)
        total_alpha: (N,)
    """
    result = batched_greedy_lookahead_path(
        returns_batch,
        k=config.oracle_k,
        lookahead=config.oracle_lookahead,
        transaction_cost=config.transaction_cost,
        min_alpha_threshold=config.min_alpha_threshold,
    )

    return (
        np.array(result.selections),
        np.array(result.cumulative_pnl),
        np.array(result.total_alpha),
    )


def oracle_selections_to_actions_single(
    selections: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    """Convert oracle selections + returns to action indices using STATE-AWARE logic.

    This MUST match path_solver.py's optimal_path_to_policy exactly!

    Args:
        selections: (T, n_assets) binary mask
        returns: (T, n_assets) log returns

    State tracking:
    - prev_pos: -1=short, 0=flat, +1=long
    - target_pos: determined by selections and return sign

    Action assignment (matches path_solver.py lines 586-593):
    - BUY (2): open_long, close_short, reverse_short_to_long
    - SELL (0): open_short, close_long, reverse_long_to_short
    - HOLD (1): maintain position or stay flat
    """
    T, n_assets = selections.shape
    actions = np.ones((T, n_assets), dtype=np.int32)  # Default HOLD

    # Track position state per asset
    prev_pos = np.zeros(n_assets, dtype=np.float32)  # Start flat

    for t in range(T):
        # Determine target position from selections and returns
        # selections=1, returns>0 → want long (+1)
        # selections=1, returns<0 → want short (-1)
        # selections=0 → want flat (0)
        target_pos = np.where(
            selections[t] > 0.5,
            np.where(returns[t] > 0, 1.0, -1.0),
            0.0
        )

        # Compute transitions (exact logic from path_solver.py)
        # Opening positions (from flat)
        open_long = (prev_pos == 0) & (target_pos == 1)
        open_short = (prev_pos == 0) & (target_pos == -1)

        # Closing positions (to flat)
        close_long = (prev_pos == 1) & (target_pos == 0)
        close_short = (prev_pos == -1) & (target_pos == 0)

        # Reversing positions (rare but possible)
        reverse_long_to_short = (prev_pos == 1) & (target_pos == -1)
        reverse_short_to_long = (prev_pos == -1) & (target_pos == 1)

        # BUY actions: open_long, close_short, reverse_short_to_long
        buy_mask = open_long | close_short | reverse_short_to_long
        actions[t, buy_mask] = 2

        # SELL actions: open_short, close_long, reverse_long_to_short
        sell_mask = open_short | close_long | reverse_long_to_short
        actions[t, sell_mask] = 0

        # HOLD is default (maintain position or stay flat)

        # Update position state for next iteration
        prev_pos = target_pos.copy()

    return actions


def oracle_selections_to_actions(
    selections: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    """Batched version - handles (N, T, n_assets) input."""
    if selections.ndim == 2:
        return oracle_selections_to_actions_single(selections, returns)

    # Batched: (N, T, n_assets)
    N = selections.shape[0]
    all_actions = []
    for i in range(N):
        actions = oracle_selections_to_actions_single(selections[i], returns[i])
        all_actions.append(actions)
    return np.stack(all_actions)


# =============================================================================
# Model Inference
# =============================================================================

# JIT-compiled model forward pass (avoids recompilation per episode)
@partial(jax.jit, static_argnames=['model'])
def _jit_model_forward(params: dict, observations: jnp.ndarray, model: MuZeroNetworks):
    """JIT-compiled model forward pass."""
    return model.apply({'params': params}, observations)


def run_model_batched(
    model: MuZeroNetworks,
    params: dict,
    observations: jnp.ndarray,
) -> np.ndarray:
    """Run model inference on observations.

    Args:
        model: MuZeroNetworks instance
        params: Model parameters
        observations: (N*T, 27, 64, 64) encoded observations

    Returns:
        actions: (N*T, 64) action per asset
    """
    # Run model (JIT compiled)
    result = _jit_model_forward(params, observations, model)
    logits = result['policy_logits']  # (N*T, 320)

    # Per-asset argmax
    # Reshape FIRST, then softmax per-asset (over 5 actions, not 320)
    n_assets = 64
    n_actions_per_asset = 5
    logits_per_asset = logits.reshape(-1, n_assets, n_actions_per_asset)
    probs = jax.nn.softmax(logits_per_asset, axis=-1)  # Softmax over 5 actions
    actions = jnp.argmax(probs, axis=-1)  # (N*T, 64)

    return np.array(actions)


# =============================================================================
# P&L Simulation
# =============================================================================

def simulate_oracle_pnl(
    selections: np.ndarray,
    returns: np.ndarray,
    transaction_cost: float = 0.001,
) -> Tuple[float, int]:
    """Oracle-style P&L: capture abs(return) for selected assets.

    This matches path_solver.py calculation.
    """
    total_pnl = 0.0
    trade_count = 0

    T, n_assets = returns.shape
    prev_mask = np.zeros(n_assets)

    for t in range(T):
        mask = selections[t]
        k_count = max(np.sum(mask), 1)

        # P&L from abs returns, normalized by n_assets for equal-weight portfolio
        # TASK_705: Both oracle and model normalize by n_assets
        step_pnl = np.sum(np.abs(returns[t]) * mask) / n_assets

        # Transaction costs (normalized by n_assets for equal-weight portfolio)
        changes = np.sum(np.abs(mask - prev_mask) > 0.5)
        step_pnl -= changes * transaction_cost / n_assets
        trade_count += int(changes)

        total_pnl += step_pnl
        prev_mask = mask

    return total_pnl, trade_count


def simulate_model_pnl(
    actions: np.ndarray,
    returns: np.ndarray,
    transaction_cost: float = 0.001,
) -> Tuple[float, int]:
    """Model-style P&L with STATE-AWARE position tracking.

    Model was trained with state-aware oracle, so actions are TRANSITIONS:
    - BUY(2): move toward long (flat→long, short→flat, long→long)
    - SELL(0): move toward short (flat→short, long→flat, short→short)
    - HOLD(1): maintain current position

    Position states: -1=short, 0=flat, +1=long
    """
    total_pnl = 0.0
    trade_count = 0

    T, n_assets = returns.shape
    position = np.zeros(n_assets, dtype=np.float32)  # Start flat

    # TASK_705: Equal-weight portfolio — each position is 1/n_assets of portfolio
    txn_cost_norm = transaction_cost / n_assets

    for t in range(T):
        step_pnl = 0.0

        # Update position based on action (state-aware transitions)
        for i in range(n_assets):
            old_pos = position[i]

            if actions[t, i] == 2:  # BUY → move toward long
                if position[i] == -1:  # short → flat
                    position[i] = 0.0
                    trade_count += 1
                    step_pnl -= txn_cost_norm
                elif position[i] == 0:  # flat → long
                    position[i] = 1.0
                    trade_count += 1
                    step_pnl -= txn_cost_norm
                # if already long, stay long (no trade)

            elif actions[t, i] == 0:  # SELL → move toward short
                if position[i] == 1:  # long → flat
                    position[i] = 0.0
                    trade_count += 1
                    step_pnl -= txn_cost_norm
                elif position[i] == 0:  # flat → short
                    position[i] = -1.0
                    trade_count += 1
                    step_pnl -= txn_cost_norm
                # if already short, stay short (no trade)

            # HOLD(1): maintain position (no change)

        # P&L from position AFTER action, normalized by n_assets
        # TASK_705: Equal-weight portfolio normalization
        step_pnl += np.sum(position * returns[t]) / n_assets

        total_pnl += step_pnl

    return total_pnl, trade_count


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_confusion_matrix(
    oracle_actions: np.ndarray,
    model_actions: np.ndarray,
) -> Dict[str, Dict[str, int]]:
    """Compute confusion matrix."""
    action_names = ['SELL', 'HOLD', 'BUY']
    confusion = {name: {n: 0 for n in action_names} for name in action_names}

    oracle_flat = oracle_actions.flatten()
    model_flat = model_actions.flatten()

    for o, m in zip(oracle_flat, model_flat):
        if o < 3 and m < 3:
            confusion[action_names[o]][action_names[m]] += 1

    return confusion


def compute_balanced_metrics(
    oracle_actions: np.ndarray,
    model_actions: np.ndarray,
) -> Dict[str, float]:
    """Compute class-balanced metrics."""
    oracle_flat = oracle_actions.flatten()
    model_flat = model_actions.flatten()

    # Per-class accuracy (balanced)
    accuracies = []
    f1_scores = {}

    for action_idx, name in enumerate(['SELL', 'HOLD', 'BUY']):
        oracle_mask = oracle_flat == action_idx
        model_mask = model_flat == action_idx

        # True positives, false positives, false negatives
        tp = np.sum(oracle_mask & model_mask)
        fp = np.sum(~oracle_mask & model_mask)
        fn = np.sum(oracle_mask & ~model_mask)

        # Precision, recall, F1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        f1_scores[name] = f1
        if np.sum(oracle_mask) > 0:
            accuracies.append(recall)

    balanced_accuracy = np.mean(accuracies) if accuracies else 0.0
    macro_f1 = np.mean(list(f1_scores.values()))

    return {
        'balanced_accuracy': balanced_accuracy,
        'macro_f1': macro_f1,
        'per_class_f1': f1_scores,
    }


def compute_trade_metrics(
    oracle_actions: np.ndarray,
    model_actions: np.ndarray,
) -> Dict[str, float]:
    """Trade-level precision/recall."""
    oracle_trades = (oracle_actions != 1)  # Not HOLD
    model_trades = (model_actions != 1)

    # True positive: both trade AND same direction
    tp = np.sum(oracle_trades & model_trades & (oracle_actions == model_actions))
    oracle_trade_count = np.sum(oracle_trades)
    model_trade_count = np.sum(model_trades)

    precision = tp / max(model_trade_count, 1)
    recall = tp / max(oracle_trade_count, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {
        'oracle_trade_count': int(oracle_trade_count),
        'model_trade_count': int(model_trade_count),
        'oracle_trade_rate': oracle_trade_count / oracle_actions.size,
        'model_trade_rate': model_trade_count / model_actions.size,
        'trade_precision': precision,
        'trade_recall': recall,
        'trade_f1': f1,
    }


def compute_return_weighted_accuracy(
    oracle_actions: np.ndarray,
    model_actions: np.ndarray,
    returns: np.ndarray,
) -> float:
    """Agreement weighted by return magnitude."""
    matches = (oracle_actions == model_actions).astype(float)
    abs_returns = np.abs(returns)

    weighted_matches = np.sum(matches * abs_returns)
    total_weight = np.sum(abs_returns)

    return weighted_matches / max(total_weight, 1e-10)


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(
    checkpoint_path: str,
    n_episodes: int = 50,
    episode_length: int = 400,
    seed: int = 42,
    use_bella: bool = False,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run full benchmark comparing oracle vs model."""

    result = BenchmarkResult()
    result.n_episodes = n_episodes
    result.n_steps = episode_length

    # Initialize action stats
    for name in ['SELL', 'HOLD', 'BUY']:
        result.action_stats[name] = ActionStats()

    # Load checkpoint
    if verbose:
        print(f"\n[BENCHMARK] Loading checkpoint: {checkpoint_path}")
    params, metadata = load_checkpoint(checkpoint_path)
    if verbose and 'epoch' in metadata:
        print(f"[BENCHMARK] Checkpoint epoch: {metadata['epoch']}")

    # Create config and model
    config = get_large_config()
    model = create_model(config)

    # Create encoder config
    encoder_config = EncoderConfig(
        n_assets=config.n_assets,
        window_size=config.window_size,
    )

    if verbose:
        print(f"[BENCHMARK] Model: {config.H_cycles}H×{config.L_cycles}L, state_dim={config.state_dim}")
        print(f"[BENCHMARK] Oracle: k={config.oracle_k}, lookahead={config.oracle_lookahead}")
        # Memory estimate: observations are (n_episodes, episode_length, 27, 64, 64) float32
        mem_estimate = n_episodes * episode_length * 27 * 64 * 64 * 4 / 1e9
        print(f"[BENCHMARK] Estimated observation memory: {mem_estimate:.1f} GB")

    # Generate validation data
    if verbose:
        print(f"[BENCHMARK] Generating {n_episodes} episodes of {episode_length} steps...")
    t0 = time.time()
    prices_batch = generate_validation_data(
        config, n_episodes, episode_length, seed, use_bella
    )
    if verbose:
        print(f"[BENCHMARK] Data generation: {time.time() - t0:.1f}s")

    # Compute returns
    # prices_batch: (N, T+window, n_assets)
    # returns: (N, T+window-1, n_assets)
    returns_batch = np.array([compute_returns(p) for p in prices_batch])

    # Align returns to episode_length
    # We need episode_length returns starting after the window
    # But returns is T-1 for T prices, so we adjust
    # Window ends at index window_size, so returns start at window_size-1
    start_idx = config.window_size - 1
    end_idx = start_idx + episode_length
    returns_aligned = returns_batch[:, start_idx:end_idx]

    # Ensure we have exactly episode_length steps
    actual_length = returns_aligned.shape[1]
    if actual_length < episode_length:
        print(f"[BENCHMARK] Warning: Only {actual_length} returns available, adjusting episode length")
        episode_length = actual_length
        result.n_steps = actual_length

    # Run oracle
    if verbose:
        print("[BENCHMARK] Running oracle...")
    t0 = time.time()
    oracle_selections, oracle_cum_pnl, oracle_alphas = run_oracle(
        jnp.array(returns_aligned), config
    )
    oracle_actions = oracle_selections_to_actions(oracle_selections, returns_aligned)
    if verbose:
        print(f"[BENCHMARK] Oracle: {time.time() - t0:.1f}s")

    # Use actual_length for model inference (matches returns_aligned)
    n_inference_steps = returns_aligned.shape[1]

    # Encode observations and run model
    if verbose:
        print("[BENCHMARK] Running model inference...")
    t0 = time.time()

    all_model_actions = []
    for ep_idx in range(n_episodes):
        prices = jnp.array(prices_batch[ep_idx])

        # Encode all windows for this episode
        # Window start indices correspond to the returns we have
        window_starts = jnp.arange(n_inference_steps)
        observations = jit_batch_encode_temporal(prices, window_starts, encoder_config)
        # observations: (n_inference_steps, 27, window_size, n_assets)

        # Run model
        model_actions = run_model_batched(model, params, observations)
        all_model_actions.append(model_actions)

    model_actions_batch = np.stack(all_model_actions)
    if verbose:
        print(f"[BENCHMARK] Model inference: {time.time() - t0:.1f}s")

    # Compute metrics
    if verbose:
        print("[BENCHMARK] Computing metrics...")

    # Flatten for overall metrics
    oracle_flat = oracle_actions.flatten()
    model_flat = model_actions_batch.flatten()

    # Overall agreement
    result.overall_agreement = np.mean(oracle_flat == model_flat) * 100

    # Per-action stats
    total_actions = len(oracle_flat)
    for action_idx, name in enumerate(['SELL', 'HOLD', 'BUY']):
        oracle_mask = oracle_flat == action_idx
        model_mask = model_flat == action_idx
        agree_mask = oracle_mask & model_mask

        result.action_stats[name].oracle_count = int(np.sum(oracle_mask))
        result.action_stats[name].model_count = int(np.sum(model_mask))
        result.action_stats[name].agreement_count = int(np.sum(agree_mask))

        result.oracle_distribution[name] = np.sum(oracle_mask) / total_actions * 100
        result.model_distribution[name] = np.sum(model_mask) / total_actions * 100

    # Confusion matrix
    result.confusion_matrix = compute_confusion_matrix(oracle_actions, model_actions_batch)

    # Balanced metrics
    balanced = compute_balanced_metrics(oracle_actions, model_actions_batch)
    result.balanced_accuracy = balanced['balanced_accuracy']
    result.macro_f1 = balanced['macro_f1']

    # Trade metrics
    trade_metrics = compute_trade_metrics(oracle_actions, model_actions_batch)
    result.trade_precision = trade_metrics['trade_precision']
    result.trade_recall = trade_metrics['trade_recall']
    result.trade_f1 = trade_metrics['trade_f1']

    # Return-weighted accuracy
    result.return_weighted_accuracy = compute_return_weighted_accuracy(
        oracle_actions, model_actions_batch, returns_aligned
    )

    # P&L simulation
    oracle_pnls = []
    model_pnls = []

    for ep_idx in range(n_episodes):
        o_pnl, o_trades = simulate_oracle_pnl(
            oracle_selections[ep_idx], returns_aligned[ep_idx], config.transaction_cost
        )
        m_pnl, m_trades = simulate_model_pnl(
            model_actions_batch[ep_idx], returns_aligned[ep_idx], config.transaction_cost
        )

        oracle_pnls.append(o_pnl)
        model_pnls.append(m_pnl)

        result.oracle_trade_count += o_trades
        result.model_trade_count += m_trades

        if o_pnl > 0:
            result.oracle_positive_episodes += 1
        if m_pnl > 0:
            result.model_positive_episodes += 1

    result.oracle_total_pnl = sum(oracle_pnls)
    result.oracle_avg_pnl = np.mean(oracle_pnls)
    result.model_total_pnl = sum(model_pnls)
    result.model_avg_pnl = np.mean(model_pnls)

    if result.oracle_total_pnl > 0:
        result.pnl_capture_ratio = result.model_total_pnl / result.oracle_total_pnl

    # Temporal agreement
    for t in range(n_inference_steps):
        t_agree = np.mean(oracle_actions[:, t] == model_actions_batch[:, t]) * 100
        result.temporal_agreement.append(t_agree)

    return result


# =============================================================================
# Report Printing
# =============================================================================

def print_report(result: BenchmarkResult):
    """Print formatted benchmark report."""

    print("\n" + "=" * 70)
    print("BENCHMARK REPORT: Oracle vs Model Performance")
    print("=" * 70)

    print(f"\nDataset: {result.n_episodes} episodes × {result.n_steps} steps × {result.n_assets} assets")

    # Agreement Summary
    print("\n--- AGREEMENT METRICS ---")
    print(f"Overall Agreement: {result.overall_agreement:.1f}%")
    print(f"Balanced Accuracy: {result.balanced_accuracy:.1%}")
    print(f"Macro-F1: {result.macro_f1:.3f}")
    print(f"Return-Weighted Acc: {result.return_weighted_accuracy:.1%}")

    print("\nPer-Action Agreement:")
    print(f"  {'Action':<8} {'Oracle %':<12} {'Model %':<12} {'Agreement':<12}")
    print(f"  {'-' * 44}")

    for name in ['SELL', 'HOLD', 'BUY']:
        stats = result.action_stats[name]
        oracle_pct = result.oracle_distribution[name]
        model_pct = result.model_distribution[name]
        agree_pct = stats.agreement_pct
        print(f"  {name:<8} {oracle_pct:>8.1f}%    {model_pct:>8.1f}%    {agree_pct:>8.1f}%")

    # Confusion Matrix
    print("\n--- CONFUSION MATRIX ---")
    print("(What model does when oracle says X)")
    print(f"\n  {'Oracle↓ Model→':<15} {'SELL':>10} {'HOLD':>10} {'BUY':>10}")
    print(f"  {'-' * 45}")

    for oracle_action in ['SELL', 'HOLD', 'BUY']:
        row = result.confusion_matrix.get(oracle_action, {})
        total = sum(row.values())
        if total > 0:
            sell_pct = row.get('SELL', 0) / total * 100
            hold_pct = row.get('HOLD', 0) / total * 100
            buy_pct = row.get('BUY', 0) / total * 100
            print(f"  {oracle_action:<15} {sell_pct:>9.1f}% {hold_pct:>9.1f}% {buy_pct:>9.1f}%")

    # Trade Analysis
    print("\n--- TRADE ANALYSIS ---")
    oracle_rate = sum(result.action_stats[n].oracle_count for n in ['SELL', 'BUY']) / max(
        sum(s.oracle_count for s in result.action_stats.values()), 1) * 100
    model_rate = sum(result.action_stats[n].model_count for n in ['SELL', 'BUY']) / max(
        sum(s.model_count for s in result.action_stats.values()), 1) * 100

    print(f"  Oracle Trade Rate:  {oracle_rate:.1f}% of actions")
    print(f"  Model Trade Rate:   {model_rate:.1f}% of actions")
    print(f"  Trade Precision:    {result.trade_precision:.1%}")
    print(f"  Trade Recall:       {result.trade_recall:.1%}")
    print(f"  Trade F1:           {result.trade_f1:.3f}")

    # Profitability
    print("\n--- PROFITABILITY ---")
    print(f"\n  {'Metric':<30} {'Oracle':>15} {'Model':>15}")
    print(f"  {'-' * 60}")
    print(f"  {'Total P&L':<30} {result.oracle_total_pnl*100:>13.2f}% {result.model_total_pnl*100:>13.2f}%")
    print(f"  {'Avg P&L per Episode':<30} {result.oracle_avg_pnl*100:>13.2f}% {result.model_avg_pnl*100:>13.2f}%")
    print(f"  {'Positive Episodes':<30} {result.oracle_positive_episodes:>15} {result.model_positive_episodes:>15}")
    print(f"  {'Total Trades':<30} {result.oracle_trade_count:>15} {result.model_trade_count:>15}")

    print(f"\n  P&L Capture Ratio: {result.pnl_capture_ratio:.1%}")

    # Book Status
    print("\n--- BOOK STATUS ---")
    if result.model_total_pnl > 0:
        print(f"  Model Book: POSITIVE (+{result.model_total_pnl*100:.2f}%)")
    else:
        print(f"  Model Book: NEGATIVE ({result.model_total_pnl*100:.2f}%)")

    if result.oracle_total_pnl > 0:
        print(f"  Oracle Book: POSITIVE (+{result.oracle_total_pnl*100:.2f}%)")
    else:
        print(f"  Oracle Book: NEGATIVE ({result.oracle_total_pnl*100:.2f}%)")

    # Temporal Analysis
    print("\n--- TEMPORAL AGREEMENT ---")
    if result.temporal_agreement:
        n = len(result.temporal_agreement)
        q = n // 4
        if q > 0:
            q1 = np.mean(result.temporal_agreement[:q])
            q2 = np.mean(result.temporal_agreement[q:2*q])
            q3 = np.mean(result.temporal_agreement[2*q:3*q])
            q4 = np.mean(result.temporal_agreement[3*q:])
            print(f"  Steps 1-{q}: {q1:.1f}%")
            print(f"  Steps {q+1}-{2*q}: {q2:.1f}%")
            print(f"  Steps {2*q+1}-{3*q}: {q3:.1f}%")
            print(f"  Steps {3*q+1}-{n}: {q4:.1f}%")

    # Diagnosis
    print("\n--- DIAGNOSIS ---")
    diagnoses = []

    # Check conservative bias
    hold_oracle = result.oracle_distribution.get('HOLD', 0)
    hold_model = result.model_distribution.get('HOLD', 0)
    if hold_model > hold_oracle + 10:
        diagnoses.append(f"WARNING: Model is too conservative (HOLD: {hold_model:.1f}% vs oracle {hold_oracle:.1f}%)")

    # Check trade agreement
    for name in ['SELL', 'BUY']:
        agree = result.action_stats[name].agreement_pct
        if agree < 40:
            diagnoses.append(f"WARNING: Model struggles with {name} decisions ({agree:.1f}% agreement)")

    # Check P&L
    if result.model_total_pnl < 0 and result.oracle_total_pnl > 0:
        diagnoses.append("CRITICAL: Model loses money while oracle profits!")
    elif result.pnl_capture_ratio < 0.3 and result.pnl_capture_ratio > 0:
        diagnoses.append(f"ISSUE: Model captures only {result.pnl_capture_ratio:.1%} of oracle returns")

    # Check temporal drift
    if result.temporal_agreement:
        early = np.mean(result.temporal_agreement[:100])
        late = np.mean(result.temporal_agreement[-100:])
        if early - late > 5:
            diagnoses.append(f"WARNING: Temporal drift detected ({early:.1f}% → {late:.1f}%)")

    if diagnoses:
        for d in diagnoses:
            print(f"  {d}")
    else:
        print("  No major issues detected.")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='TASK_155: Benchmark Oracle vs Model')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--episodes', '-n', type=int, default=50,
                       help='Number of validation episodes (default: 50)')
    parser.add_argument('--steps', '-s', type=int, default=400,
                       help='Steps per episode (default: 400)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--bella', action='store_true',
                       help='Use Bella generator (default: synthetic)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Save results to JSON file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    # Check checkpoint exists
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Run benchmark
    result = run_benchmark(
        str(ckpt_path),
        n_episodes=args.episodes,
        episode_length=args.steps,
        seed=args.seed,
        use_bella=args.bella,
        verbose=not args.quiet,
    )

    # Print report
    print_report(result)

    # Save if requested
    if args.output:
        output_data = {
            'overall_agreement': result.overall_agreement,
            'balanced_accuracy': result.balanced_accuracy,
            'macro_f1': result.macro_f1,
            'return_weighted_accuracy': result.return_weighted_accuracy,
            'oracle_total_pnl': result.oracle_total_pnl,
            'model_total_pnl': result.model_total_pnl,
            'pnl_capture_ratio': result.pnl_capture_ratio,
            'oracle_distribution': result.oracle_distribution,
            'model_distribution': result.model_distribution,
            'trade_precision': result.trade_precision,
            'trade_recall': result.trade_recall,
            'trade_f1': result.trade_f1,
            'n_episodes': result.n_episodes,
            'n_steps': result.n_steps,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

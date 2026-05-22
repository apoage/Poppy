#!/usr/bin/env python3
"""
TASK_682: Real Data Benchmark for Easy Mode

Evaluates the easy-mode model on 6 real market benchmark periods from DB.
Computes agreement, P&L, alpha, Sharpe, max drawdown per period.

Usage:
    # Full benchmark (all 6 periods)
    python tools/benchmark_easy.py --checkpoint /path/to/phase1_best.pkl

    # Include Bella synthetic data benchmark
    python tools/benchmark_easy.py --checkpoint ... --bella

    # Specific periods
    python tools/benchmark_easy.py --checkpoint ... --periods bull bear

    # JSON output
    python tools/benchmark_easy.py --checkpoint ... --json results.json
"""

import argparse
import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from functools import partial

from easy_config import EasyConfig, get_large_config, get_medium_config, get_small_config
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE, get_value_support_np
from mcts_trader.jax_agent.path_solver import greedy_lookahead_path
from mcts_trader.jax_agent.mctx_search import create_mctx_search
from mcts_trader.env.jax_encoder import EncoderConfig, jit_batch_encode_temporal
from mcts_trader.data.history import HistoricalMarket

# Reuse from existing benchmark
from tools.benchmark_oracle_vs_model import (
    load_checkpoint,
    oracle_selections_to_actions,
    simulate_oracle_pnl,
    simulate_model_pnl,
    compute_balanced_metrics,
    compute_trade_metrics,
)


def create_model(config: EasyConfig) -> MuZeroNetworks:
    """Create MuZeroNetworks model from config, respecting all architecture params."""
    # TASK_686: Resolve compute dtype
    dtype_str = getattr(config, 'compute_dtype', 'float32')
    compute_dtype = jnp.bfloat16 if dtype_str == 'bfloat16' else jnp.float32

    return MuZeroNetworks(
        state_dim=config.state_dim,
        action_space_size=config.action_space_size,
        support_size=VALUE_SUPPORT_SIZE,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        use_prediction_head=False,
        n_assets=config.n_assets,
        encoder_version=getattr(config, 'encoder_version', 2),
        hidden_dim=getattr(config, 'trm_hidden_dim', None),      # TASK_684
        embed_dim=getattr(config, 'embed_dim', 32),                # TASK_683
        encoder_num_heads=getattr(config, 'encoder_num_heads', 4), # TASK_683
        dtype=compute_dtype,                                        # TASK_686
        param_dtype=jnp.float32,
        use_oracle_mask_head=getattr(config, 'use_oracle_mask_head', False),  # TASK_695
        oracle_mask_hidden_dim=getattr(config, 'oracle_mask_hidden_dim', 256),  # TASK_695
    )


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PeriodResult:
    """Benchmark result for a single period."""
    name: str
    description: str
    n_steps: int
    n_assets: int
    symbols: List[str]
    # Agreement
    balanced_accuracy: float
    agree_trade: float
    agree_hold: float
    # P&L
    oracle_pnl: float
    model_pnl: float
    pnl_capture_ratio: float
    # Alpha
    buy_hold_return: float
    model_alpha: float
    oracle_alpha: float
    # Risk
    model_sharpe: float
    max_drawdown: float
    # Trading
    oracle_trades: int
    model_trades: int
    oracle_trade_rate: float
    model_trade_rate: float
    # Baseline
    random_pnl: float
    # Value head
    value_mse: float
    # TASK_695: Oracle mask prediction
    oracle_mask_accuracy: float = 0.0


# =============================================================================
# Price Loading
# =============================================================================

def _get_benchmark_assets(period: dict) -> Optional[List[str]]:
    """Get pre-computed 64-asset list for this benchmark period.

    TASK_705: Uses explicit benchmark_assets from benchmark_periods.json
    instead of the union of all adjacent training period assets (which
    returned 500+ assets, most of which were discarded).
    """
    assets = period.get('benchmark_assets')
    if assets:
        logging.info(f"Using {len(assets)} pre-computed benchmark assets")
        return assets
    return None


def load_benchmark_prices(
    period: dict,
    db_path: str = "trading_data.db",
    n_assets: int = 64,
) -> Tuple[np.ndarray, List[str]]:
    """Load real close prices for a benchmark period from DB.

    Uses HistoricalMarket with date filtering + RAM cache.
    Loads asset list from training_schedule.json to avoid expensive
    full-database GROUP BY query.

    Returns:
        prices: (T, n_assets) close prices, forward-filled
        symbols: list of symbol names
    """
    print(f"  Loading prices: {period['start_date']} to {period['end_date']}...")

    # TASK_705: Load exact 64 benchmark assets (not 500+ union from adjacent periods)
    schedule_assets = _get_benchmark_assets(period)

    market = HistoricalMarket(
        db_path=db_path,
        n_assets=n_assets,
        start_date=period['start_date'],
        end_date=period['end_date'],
        use_ram_cache=True,
        skip_validation=True,
        specific_symbols=schedule_assets,
    )

    # TASK_703: Pass full OHLCV to encoder instead of close-only
    ohlcv = market.cache_data  # (T, n_assets_actual, 5)
    actual_assets = ohlcv.shape[1]
    symbols = [market.id_to_symbol[aid] for aid in market.asset_ids]

    print(f"  Loaded: {ohlcv.shape[0]} timesteps, {actual_assets} assets")

    # Handle asset underflow: duplicate existing assets (NOT zero-pad)
    if actual_assets < n_assets:
        print(f"  WARNING: Only {actual_assets} assets, duplicating to fill {n_assets}")
        repeats = (n_assets // actual_assets) + 1
        ohlcv = np.tile(ohlcv, (1, repeats, 1))[:, :n_assets, :]
        symbols = (symbols * repeats)[:n_assets]
    elif actual_assets > n_assets:
        ohlcv = ohlcv[:, :n_assets, :]
        symbols = symbols[:n_assets]
    # Close prices for oracle and P&L computation
    # TASK_703: Forward-fill all OHLCV channels (not just close)
    # Check close channel for gaps — if close is zero, all channels are zero
    close = ohlcv[:, :, 3]
    zero_mask = close <= 0
    if np.any(zero_mask):
        n_zeros = np.sum(zero_mask)
        print(f"  WARNING: {n_zeros} zero/negative prices, filling gaps...")
        for col in range(ohlcv.shape[1]):
            # Forward-fill all 5 channels together
            for row in range(1, ohlcv.shape[0]):
                if ohlcv[row, col, 3] <= 0 and ohlcv[row - 1, col, 3] > 0:
                    ohlcv[row, col] = ohlcv[row - 1, col]
            # Backward-fill
            for row in range(ohlcv.shape[0] - 2, -1, -1):
                if ohlcv[row, col, 3] <= 0 and ohlcv[row + 1, col, 3] > 0:
                    ohlcv[row, col] = ohlcv[row + 1, col]
        still_zero = np.sum(ohlcv[:, :, 3] <= 0)
        if still_zero > 0:
            print(f"  WARNING: {still_zero} prices still zero after fill — entire-column gaps")

    prices = ohlcv[:, :, 3].copy()  # Close prices for oracle/PnL

    market.close()
    return ohlcv, prices, symbols


# =============================================================================
# Model Inference
# =============================================================================

@partial(jax.jit, static_argnames=['model'])
def _jit_model_forward(params, observations, model):
    """JIT-compiled model forward pass."""
    return model.apply({'params': params}, observations)


def value_logits_to_scalar(value_logits: np.ndarray) -> np.ndarray:
    """Convert categorical value logits to scalar values.

    Args:
        value_logits: (batch, 401) value support logits

    Returns:
        scalar_values: (batch,) expected values
    """
    support = get_value_support_np()  # linspace(-10, 10, 401)
    # Softmax
    logits = value_logits - np.max(value_logits, axis=-1, keepdims=True)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    return np.sum(probs * support, axis=-1)


# =============================================================================
# Detailed P&L Simulation (with per-step returns for Sharpe)
# =============================================================================

def simulate_model_pnl_detailed(
    actions: np.ndarray,
    returns: np.ndarray,
    transaction_cost: float = 0.001,
) -> Tuple[float, int, np.ndarray]:
    """Model P&L with per-step returns for Sharpe computation.

    Same logic as simulate_model_pnl but also returns step_returns array.

    Returns:
        total_pnl, trade_count, step_returns (T,)
    """
    T, n_assets = returns.shape
    position = np.zeros(n_assets, dtype=np.float32)
    step_returns = np.zeros(T, dtype=np.float64)
    total_pnl = 0.0
    trade_count = 0

    # TASK_705: Equal-weight portfolio — each position is 1/n_assets of portfolio
    txn_cost_norm = transaction_cost / n_assets

    for t in range(T):
        step_pnl = 0.0

        for i in range(n_assets):
            if actions[t, i] == 2 or actions[t, i] == 4:  # BUY or LIMITBUY
                if position[i] == -1:
                    position[i] = 0.0
                    trade_count += 1
                    step_pnl -= txn_cost_norm
                elif position[i] == 0:
                    position[i] = 1.0
                    trade_count += 1
                    step_pnl -= txn_cost_norm
            elif actions[t, i] == 0 or actions[t, i] == 3:  # SELL or LIMITSELL
                if position[i] == 1:
                    position[i] = 0.0
                    trade_count += 1
                    step_pnl -= txn_cost_norm
                elif position[i] == 0:
                    position[i] = -1.0
                    trade_count += 1
                    step_pnl -= txn_cost_norm

        # TASK_705: Normalize by n_assets for equal-weight portfolio return
        step_pnl += np.sum(position * returns[t]) / n_assets
        step_returns[t] = step_pnl
        total_pnl += step_pnl

    return total_pnl, trade_count, step_returns


# =============================================================================
# Random Baseline
# =============================================================================

def _simulate_pnl_vectorized(
    actions: np.ndarray,
    returns: np.ndarray,
    transaction_cost: float = 0.001,
) -> float:
    """Vectorized P&L simulation (faster than simulate_model_pnl for benchmarking).

    Same state-aware logic but vectorized across assets at each step.
    """
    T, n_assets = returns.shape
    position = np.zeros(n_assets, dtype=np.float32)
    total_pnl = 0.0

    for t in range(T):
        # BUY actions: move toward long
        buy_mask = actions[t] == 2
        buy_from_short = buy_mask & (position == -1)
        buy_from_flat = buy_mask & (position == 0)

        # SELL actions: move toward short
        sell_mask = actions[t] == 0
        sell_from_long = sell_mask & (position == 1)
        sell_from_flat = sell_mask & (position == 0)

        # Count trades (position changes)
        n_trades = (np.sum(buy_from_short) + np.sum(buy_from_flat) +
                    np.sum(sell_from_long) + np.sum(sell_from_flat))

        # Update positions
        position[buy_from_short] = 0.0
        position[buy_from_flat] = 1.0
        position[sell_from_long] = 0.0
        position[sell_from_flat] = -1.0

        # TASK_705: Normalize by n_assets for equal-weight portfolio return
        total_pnl += np.sum(position * returns[t]) / n_assets - n_trades * transaction_cost / n_assets

    return total_pnl


def evaluate_random_policy(
    returns: np.ndarray,
    n_assets: int = 64,
    transaction_cost: float = 0.001,
    n_trials: int = 20,
    seed: int = 42,
) -> float:
    """Average P&L of hold-biased random policy over n_trials.

    Distribution: 90% hold, 5% buy, 5% sell — mimics a "noise trader"
    that occasionally acts but mostly holds. Uniform random would be
    annihilated by transaction costs (changing positions every minute).
    """
    rng = np.random.RandomState(seed)
    pnls = []
    for _ in range(n_trials):
        r = rng.random(size=(returns.shape[0], n_assets))
        actions = np.ones((returns.shape[0], n_assets), dtype=np.int32)  # default HOLD
        actions[r < 0.05] = 0   # 5% sell
        actions[r > 0.95] = 2   # 5% buy
        pnl = _simulate_pnl_vectorized(actions, returns, transaction_cost)
        pnls.append(pnl)
    return float(np.mean(pnls))


# =============================================================================
# Bella Data Generation
# =============================================================================

def generate_bella_prices(
    n_episodes: int = 10,
    seed: int = 42,
) -> List[np.ndarray]:
    """Generate Bella episodes for benchmarking.

    Returns list of (seq_len, n_assets) price arrays.
    """
    from mcts_trader.training.setup import setup_bella_generator, generate_bella_prices_on_device

    bella_path = Path("checkpoints/diff_teacher/final/generator")
    if not bella_path.exists():
        raise FileNotFoundError(f"Bella checkpoint not found: {bella_path}")

    print(f"  Loading Bella generator...")
    bella_gen, bella_config = setup_bella_generator(str(bella_path))
    seq_len = bella_gen.config.seq_length if hasattr(bella_gen.config, 'seq_length') else 400

    key = jax.random.PRNGKey(seed)
    episodes = []

    for i in range(n_episodes):
        key, subkey = jax.random.split(key)
        ohlcv = generate_bella_prices_on_device(bella_gen, subkey, n_envs=1)
        # TASK_703: ohlcv is (seq_len, n_assets, 5)
        episodes.append(np.array(ohlcv))

    print(f"  Generated {n_episodes} Bella episodes ({seq_len} steps × {episodes[0].shape[1]} assets)")
    return episodes


# =============================================================================
# Period Evaluation
# =============================================================================

def _mcts_decode_actions(
    mcts_actions: np.ndarray,
    policy_probs: np.ndarray,
    n_assets: int,
    decode: str,
) -> np.ndarray:
    """Decode MCTS output into per-asset actions (B, n_assets).

    joint:     take the single action (idx in [0, n_assets*5)); set all other
               assets to HOLD (=1). Matches real_trade_jax single-action behavior.
    per_asset: reshape policy_probs (B, n_assets*5) -> (B, n_assets, 5),
               argmax along last axis.
    """
    B = mcts_actions.shape[0]
    if decode == 'per_asset':
        per_asset = policy_probs.reshape(B, n_assets, 5)
        return np.argmax(per_asset, axis=-1).astype(np.int32)
    # joint
    actions = np.ones((B, n_assets), dtype=np.int32)  # default HOLD
    asset_idx = mcts_actions // 5
    act = mcts_actions % 5
    for b in range(B):
        actions[b, int(asset_idx[b])] = int(act[b])
    return actions


def run_period_eval(
    period: dict,
    ohlcv: np.ndarray,
    prices: np.ndarray,
    symbols: List[str],
    model: MuZeroNetworks,
    params: dict,
    config: EasyConfig,
    encoder_config: EncoderConfig,
    mcts_search=None,
    mcts_decode: str = 'joint',
    disable_limit_orders: bool = False,
) -> PeriodResult:
    """Evaluate model on a single benchmark period.

    TASK_703: Now accepts ohlcv (T, n_assets, 5) for encoder and prices (T, n_assets) close for oracle/PnL.

    Temporal alignment:
    - obs[i] covers prices[i:i+window_size]
    - Action from obs[i] applies to returns[i+window_size-1]
    - Oracle and model actions are aligned to the SAME return indices
    """
    n_assets = config.n_assets
    window_size = config.window_size
    T_prices = prices.shape[0]

    # TASK_767: eval-MCTS action mask. The action mask passed to mcts_search
    # is normally all-ones (every action valid). When the checkpoint was
    # trained with disable_limit_orders=True the LimitSell (3) / LimitBuy (4)
    # actions within each asset's 5-block must be masked at eval too, so the
    # eval-MCTS tree expands the same effective 64x3=192 action space as
    # training. _limit_mask_row is (320,) int8: 1 = valid, 0 = limit order.
    # disable_limit_orders=False keeps it all-ones — eval is bit-identical to
    # pre-767 behavior.
    _limit_mask_row = np.ones((n_assets, 5), dtype=np.int8)
    if disable_limit_orders:
        _limit_mask_row[:, 3] = 0   # LimitSell
        _limit_mask_row[:, 4] = 0   # LimitBuy
    _limit_mask_row = _limit_mask_row.reshape(-1)  # (320,)
    # Additive (5,) logit mask for the argmax (non-MCTS) eval path: 0.0 for
    # kept actions, -inf for limits. Broadcasts over (..., n_assets, 5).
    # All-zeros when disable_limit_orders=False → argmax bit-identical.
    _argmax_logit_mask = jnp.asarray(
        np.where(_limit_mask_row.reshape(n_assets, 5)[0] == 0, -np.inf, 0.0)
        .astype(np.float32)
    )  # (5,)

    # 1. Compute log returns
    returns = np.log(prices[1:] / np.maximum(prices[:-1], 1e-8))  # (T-1, n_assets)
    T_returns = returns.shape[0]

    # 2. Valid windows: obs[i] uses prices[i:i+ws], action applies to returns[i+ws-1]
    # Last valid window_start: T_prices - window_size (obs uses prices[start:start+ws])
    # That window's action applies to returns[start + ws - 1]
    # We need start + ws - 1 < T_returns, so start < T_returns - ws + 1 = T_prices - 1 - ws + 1 = T_prices - ws
    n_windows = T_prices - window_size  # Number of valid inference windows
    if n_windows <= 0:
        raise ValueError(f"Not enough data: {T_prices} prices, need > {window_size}")

    print(f"  {n_windows} inference windows (T={T_prices}, ws={window_size})")

    # Return indices that actions apply to
    # obs[i] → action → returns[i + window_size - 1]
    action_return_indices = np.arange(window_size - 1, window_size - 1 + n_windows)

    # 3. Run oracle on aligned returns
    # Oracle sees full returns and produces selections at each step
    # We need oracle selections aligned to the same return indices
    # NOTE: max_len must cover full return sequence (default 512 is too small for 10k+ steps)
    t0 = time.time()
    returns_jax = jnp.array(returns)
    max_len = T_returns  # Exact length — avoids wasting VRAM on padding
    oracle_result = greedy_lookahead_path(
        returns_jax,
        k=config.oracle_k,
        lookahead=config.oracle_lookahead,
        transaction_cost=config.transaction_cost,
        min_alpha_threshold=config.min_alpha_threshold,
        max_len=max_len,
    )
    oracle_selections = np.array(oracle_result.selections)  # (T_returns, n_assets)
    oracle_cum_pnl = np.array(oracle_result.cumulative_pnl)  # (T_returns,)

    # Extract oracle selections for our aligned windows
    oracle_sel_aligned = oracle_selections[action_return_indices]  # (n_windows, n_assets)
    returns_aligned = returns[action_return_indices]  # (n_windows, n_assets)

    # Convert oracle selections to actions
    oracle_actions = oracle_selections_to_actions(oracle_sel_aligned, returns_aligned)
    print(f"  Oracle: {time.time() - t0:.1f}s")

    # 4. Sequential inference with model's own state channels (Phase C)
    # Model trained with 34ch (27 market + 7 state). State channels come from
    # the model's own actions, not the oracle — honest inference.
    t0 = time.time()
    prices_jax = jnp.array(ohlcv)  # TASK_703: full OHLCV for encoder
    use_state_channels = config.n_channels > 27
    BATCH = 128
    window_starts = jnp.arange(n_windows)

    all_oracle_mask_preds = []  # TASK_695

    if not use_state_channels:
        # Legacy 27-channel models: batched inference (no state tracking)
        all_model_actions = []
        all_model_values = []
        for start in range(0, n_windows, BATCH):
            end = min(start + BATCH, n_windows)
            batch_starts = window_starts[start:end]
            obs = jit_batch_encode_temporal(prices_jax, batch_starts, encoder_config)
            output = _jit_model_forward(params, obs, model)
            logits_per_asset = output['policy_logits'].reshape(-1, n_assets, 5)
            if mcts_search is None:
                # TASK_767: _argmax_logit_mask is all-zeros unless limits are
                # disabled, in which case limit logits become -inf → never the
                # argmax. Bit-identical to pre-767 when the flag is off.
                all_model_actions.append(np.array(jnp.argmax(
                    logits_per_asset + _argmax_logit_mask, axis=-1)))
            else:
                # MCTS path: deterministic per-window seed
                B = obs.shape[0]
                # TASK_767: tile the (320,) limit mask across the batch.
                action_mask = jnp.asarray(
                    np.broadcast_to(_limit_mask_row, (B, n_assets * 5))
                )
                rng = jax.random.PRNGKey(int(start))
                mcts_act, policy_probs, _ = mcts_search(params, rng, obs, action_mask)
                acts = _mcts_decode_actions(
                    np.array(mcts_act), np.array(policy_probs),
                    n_assets, mcts_decode,
                )
                all_model_actions.append(acts)
            all_model_values.append(value_logits_to_scalar(np.array(output['value'])))
            # TASK_695: Collect oracle mask predictions
            if 'oracle_mask' in output:
                all_oracle_mask_preds.append(
                    np.array(jax.nn.sigmoid(output['oracle_mask']) > 0.5).astype(np.float32)
                )
        model_actions = np.concatenate(all_model_actions, axis=0)
        model_values = np.concatenate(all_model_values, axis=0)
    else:
        # 34-channel models: sequential inference with model's own state
        # Pre-encode market channels in batches (27ch — doesn't depend on model state)
        encode_cache = {}  # batch_start -> np.array of encoded obs
        def get_market_obs(idx):
            batch_start = (idx // BATCH) * BATCH
            if batch_start not in encode_cache:
                # Evict old cache entries to limit memory
                if len(encode_cache) > 2:
                    oldest = min(encode_cache.keys())
                    del encode_cache[oldest]
                batch_end = min(batch_start + BATCH, n_windows)
                batch_starts_enc = window_starts[batch_start:batch_end]
                encoded = jit_batch_encode_temporal(prices_jax, batch_starts_enc, encoder_config)
                encode_cache[batch_start] = np.array(jax.block_until_ready(encoded))
            return encode_cache[batch_start][idx - batch_start]  # (27, ws, n_assets)

        # State tracking (model's own trajectory)
        pos = np.zeros(n_assets, dtype=np.float32)
        pnl = np.zeros(n_assets, dtype=np.float32)
        steps = np.zeros(n_assets, dtype=np.float32)
        h_fast = np.zeros(n_assets, dtype=np.float32)
        h_slow = np.zeros(n_assets, dtype=np.float32)
        prev_confidence = np.zeros(n_assets, dtype=np.float32)

        all_model_actions = []
        all_model_values = []
        all_oracle_mask_preds = []  # TASK_695

        for i in range(n_windows):
            # 1. Get pre-encoded market obs (27ch)
            market_obs = get_market_obs(i)  # (27, ws, n_assets)

            # 2. Build state channels from model's current state (t-1)
            n_active = np.sum(np.abs(pos) > 0.5)
            capacity = n_active / config.oracle_k
            state_ch = np.stack([
                pos,                                                  # ch27: position
                pnl,                                                  # ch28: unrealized PnL
                np.clip(steps / window_size, 0.0, 1.0),              # ch29: hold duration
                np.clip(h_fast, 0.0, 1.0),                           # ch30: heat fast
                np.clip(h_slow, 0.0, 1.0),                           # ch31: heat slow
                np.full(n_assets, capacity, dtype=np.float32),        # ch32: capacity
            ])  # (6, n_assets)
            # Tile across window time dimension
            state_tiled = np.tile(state_ch[:, None, :], (1, window_size, 1))  # (6, ws, n_assets)
            conf_tiled = np.tile(prev_confidence[None, None, :], (1, window_size, 1))  # (1, ws, n_assets)

            # 3. Concatenate: 27 market + 6 state + 1 confidence = 34
            obs_full = jnp.array(np.concatenate(
                [market_obs, state_tiled, conf_tiled], axis=0
            ))[None]  # (1, 34, ws, n_assets)

            # 4. Model forward
            output = _jit_model_forward(params, obs_full, model)
            logits = output['policy_logits']  # (1, 320)
            value_logits = output['value']  # (1, 401)

            logits_per_asset = logits.reshape(1, n_assets, 5)
            if mcts_search is None:
                # TASK_767: limit logits → -inf when disabled (else all-zeros).
                actions_i = np.array(jnp.argmax(
                    logits_per_asset + _argmax_logit_mask, axis=-1))[0]  # (n_assets,)
            else:
                # MCTS path: deterministic per-window seed
                # TASK_767: (1, 320) limit mask (all-ones unless flag set).
                action_mask = jnp.asarray(_limit_mask_row[None, :])
                rng = jax.random.PRNGKey(int(i))
                mcts_act, policy_probs, _ = mcts_search(params, rng, obs_full, action_mask)
                actions_i = _mcts_decode_actions(
                    np.array(mcts_act), np.array(policy_probs),
                    n_assets, mcts_decode,
                )[0]  # (n_assets,)
            all_model_actions.append(actions_i)
            all_model_values.append(value_logits_to_scalar(np.array(value_logits))[0])

            # TASK_695: Collect oracle mask prediction
            if 'oracle_mask' in output:
                all_oracle_mask_preds.append(
                    np.array(jax.nn.sigmoid(output['oracle_mask'][0]) > 0.5).astype(np.float32)
                )

            # 5. Update confidence from model's own softmax
            # TASK_767: add _argmax_logit_mask (all-zeros unless limits are
            # disabled) so the confidence reflects only actions the model can
            # actually take. Bit-identical to pre-767 when the flag is off.
            model_softmax = np.array(jax.nn.softmax(
                logits_per_asset[0] + _argmax_logit_mask, axis=-1))  # (n_assets, 5)
            prev_confidence = np.max(model_softmax, axis=-1)  # (n_assets,)

            # 6. Update state from model's own action
            # Action → position change (same logic as simulate_model_pnl_detailed)
            new_pos = pos.copy()
            buy_mask = (actions_i == 2) | (actions_i == 4)   # Buy/LimitBuy
            sell_mask = (actions_i == 0) | (actions_i == 3)   # Sell/LimitSell
            # Buy: short→flat, flat→long
            new_pos[buy_mask & (pos == -1)] = 0.0
            new_pos[buy_mask & (pos == 0)] = 1.0
            # Sell: long→flat, flat→short
            new_pos[sell_mask & (pos == 1)] = 0.0
            new_pos[sell_mask & (pos == 0)] = -1.0

            # State channel updates (mirrors build_state_channels from train_easy.py)
            traded = (new_pos != pos).astype(np.float32)
            flipped = (pos != 0) & (new_pos != 0) & (new_pos != pos)
            trade_strength = traded + flipped.astype(np.float32)

            ret = returns_aligned[i]  # (n_assets,)
            held = (new_pos == pos) & (pos != 0)
            signed_ret = ret * pos
            pnl = np.where(held, pnl + signed_ret, 0.0).astype(np.float32)
            pnl = np.clip(pnl, -1.0, 1.0)

            steps = np.where(traded > 0.5, 0.0, steps + 1.0).astype(np.float32)
            h_fast = (h_fast * 0.80 + trade_strength).astype(np.float32)
            h_slow = (h_slow * 0.99 + trade_strength).astype(np.float32)
            pos = new_pos

            # Progress (every 2000 windows)
            if (i + 1) % 2000 == 0:
                print(f"    Window {i+1}/{n_windows} | pos_active={int(n_active)}")

        model_actions = np.stack(all_model_actions, axis=0)  # (n_windows, n_assets)
        model_values = np.array(all_model_values)  # (n_windows,)

    jax.block_until_ready(model_actions)
    print(f"  Model inference: {time.time() - t0:.1f}s ({n_windows} windows, "
          f"{'sequential+state' if use_state_channels else 'batched'})")

    # 5. Agreement metrics
    balanced = compute_balanced_metrics(oracle_actions, model_actions)
    trade_metrics = compute_trade_metrics(oracle_actions, model_actions)

    # agree_trade: accuracy on non-HOLD oracle actions
    oracle_trade_mask = oracle_actions != 1
    if np.any(oracle_trade_mask):
        agree_trade = float(np.mean(
            (oracle_actions[oracle_trade_mask] == model_actions[oracle_trade_mask])
        ))
    else:
        agree_trade = 0.0

    # agree_hold: accuracy on HOLD oracle actions
    oracle_hold_mask = oracle_actions == 1
    if np.any(oracle_hold_mask):
        agree_hold = float(np.mean(
            (oracle_actions[oracle_hold_mask] == model_actions[oracle_hold_mask])
        ))
    else:
        agree_hold = 0.0

    # 6. P&L simulation
    # Oracle P&L
    oracle_pnl, oracle_trade_count = simulate_oracle_pnl(
        oracle_sel_aligned, returns_aligned, config.transaction_cost
    )

    # Model P&L (detailed — with per-step returns for Sharpe)
    model_pnl, model_trade_count, step_returns = simulate_model_pnl_detailed(
        model_actions, returns_aligned, config.transaction_cost
    )

    # 7. Sharpe ratio (annualized for 1-min candles)
    # 525960 = minutes per year (365.25 × 24 × 60)
    if np.std(step_returns) > 1e-10:
        model_sharpe = float(np.mean(step_returns) / np.std(step_returns) * np.sqrt(525960))
    else:
        model_sharpe = 0.0

    # 8. Buy-and-hold return (equal-weight portfolio)
    # Compare prices at end vs start of eval window
    start_prices = prices[window_size]  # First price after initial window
    end_prices = prices[-1]
    valid_mask = start_prices > 1e-8
    if np.any(valid_mask):
        asset_returns = (end_prices[valid_mask] / start_prices[valid_mask]) - 1.0
        buy_hold_return = float(np.mean(asset_returns))
    else:
        buy_hold_return = 0.0

    # 9. Alpha
    # BUG-11 fix: Convert buy_hold from simple return to log return
    # (model_pnl is cumulative log returns, must compare apples-to-apples)
    buy_hold_log = np.log(1 + buy_hold_return) if buy_hold_return > -1 else buy_hold_return
    model_alpha = model_pnl - buy_hold_log
    oracle_alpha = oracle_pnl - buy_hold_log

    # 10. Max drawdown
    cumulative = np.cumsum(step_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # 11. Random baseline
    t0 = time.time()
    random_pnl = evaluate_random_policy(
        returns_aligned, n_assets=n_assets,
        transaction_cost=config.transaction_cost,
    )
    print(f"  Random baseline: {time.time() - t0:.1f}s")

    # 12. Value head MSE
    # BUG-3 fix: Value head is trained on REMAINING PnL, not cumulative
    oracle_remaining = oracle_cum_pnl[-1] - oracle_cum_pnl[action_return_indices]
    # Descale model values to match training metric (model outputs in [-10,10] logit space)
    value_mse = float(np.mean((model_values / config.value_scale - oracle_remaining) ** 2))

    # TASK_695: Oracle mask accuracy (compare predictions against oracle selections)
    oracle_mask_accuracy = 0.0
    if all_oracle_mask_preds:
        mask_preds = np.concatenate(
            [p.reshape(-1, n_assets) for p in all_oracle_mask_preds], axis=0
        )[:n_windows]  # (n_windows, n_assets)
        oracle_mask_accuracy = float(np.mean(mask_preds == oracle_sel_aligned))

    # Action distribution diagnostic (TASK_725: bull market analysis)
    action_names = ['SELL', 'HOLD', 'BUY', 'LSELL', 'LBUY']
    oracle_flat = oracle_actions.flatten()
    model_flat = model_actions.flatten()
    total_decisions = n_windows * n_assets
    print(f"  Action distribution (% of {total_decisions} decisions):")
    print(f"    {'':8s}  {'SELL':>6s}  {'HOLD':>6s}  {'BUY':>6s}  {'LSELL':>6s}  {'LBUY':>6s}")
    oracle_pcts = [float(np.mean(oracle_flat == a)) * 100 for a in range(5)]
    model_pcts = [float(np.mean(model_flat == a)) * 100 for a in range(5)]
    print(f"    {'Oracle':8s}  {oracle_pcts[0]:5.1f}%  {oracle_pcts[1]:5.1f}%  {oracle_pcts[2]:5.1f}%  {oracle_pcts[3]:5.1f}%  {oracle_pcts[4]:5.1f}%")
    print(f"    {'Model':8s}  {model_pcts[0]:5.1f}%  {model_pcts[1]:5.1f}%  {model_pcts[2]:5.1f}%  {model_pcts[3]:5.1f}%  {model_pcts[4]:5.1f}%")
    # Confusion: what does model do when oracle says BUY?
    oracle_buy = (oracle_actions == 2) | (oracle_actions == 4)
    oracle_sell = (oracle_actions == 0) | (oracle_actions == 3)
    if np.any(oracle_buy):
        model_on_buy = model_actions[oracle_buy]
        buy_response = {a: float(np.mean(model_on_buy == a)) * 100 for a in range(5)}
        print(f"    When oracle BUYs → model: SELL {buy_response[0]:.1f}% HOLD {buy_response[1]:.1f}% BUY {buy_response[2]:.1f}% LSELL {buy_response[3]:.1f}% LBUY {buy_response[4]:.1f}%")
    if np.any(oracle_sell):
        model_on_sell = model_actions[oracle_sell]
        sell_response = {a: float(np.mean(model_on_sell == a)) * 100 for a in range(5)}
        print(f"    When oracle SELLs → model: SELL {sell_response[0]:.1f}% HOLD {sell_response[1]:.1f}% BUY {sell_response[2]:.1f}% LSELL {sell_response[3]:.1f}% LBUY {sell_response[4]:.1f}%")

    # P&L capture ratio
    pnl_capture = model_pnl / oracle_pnl if oracle_pnl > 1e-10 else 0.0

    # Trade rates
    oracle_trade_rate = float(np.sum(oracle_actions != 1)) / total_decisions
    model_trade_rate = float(np.sum(model_actions != 1)) / total_decisions

    return PeriodResult(
        name=period['name'],
        description=period['description'],
        n_steps=n_windows,
        n_assets=n_assets,
        symbols=symbols[:10],  # First 10 for reporting
        balanced_accuracy=float(balanced['balanced_accuracy']),
        agree_trade=agree_trade,
        agree_hold=agree_hold,
        oracle_pnl=float(oracle_pnl),
        model_pnl=float(model_pnl),
        pnl_capture_ratio=float(pnl_capture),
        buy_hold_return=buy_hold_return,
        model_alpha=float(model_alpha),
        oracle_alpha=float(oracle_alpha),
        model_sharpe=model_sharpe,
        max_drawdown=max_drawdown,
        oracle_trades=oracle_trade_count,
        model_trades=model_trade_count,
        oracle_trade_rate=oracle_trade_rate,
        model_trade_rate=model_trade_rate,
        random_pnl=random_pnl,
        value_mse=value_mse,
        oracle_mask_accuracy=oracle_mask_accuracy,
    )


# =============================================================================
# Full Benchmark
# =============================================================================

def run_full_benchmark(
    checkpoint_path: str,
    db_path: str = "trading_data.db",
    period_names: Optional[List[str]] = None,
    config_preset: str = "large",
    include_bella: bool = False,
    bella_episodes: int = 10,
    eval_mcts_sims: int = 0,
    eval_mcts_decode: str = 'joint',
    disable_limit_orders: Optional[bool] = None,
) -> List[PeriodResult]:
    """Run benchmark on all (or selected) periods.

    Args:
        checkpoint_path: Path to model checkpoint (.pkl)
        db_path: Path to trading_data.db
        period_names: Optional list of period names to run (default: all 6)
        config_preset: Config preset to use (must match training)
        include_bella: If True, also benchmark on Bella-generated data
        bella_episodes: Number of Bella episodes to average over

    Returns:
        List of PeriodResult for each evaluated period
    """
    # Load benchmark periods
    periods_path = Path(__file__).parent.parent / "config" / "benchmark_periods.json"
    with open(periods_path) as f:
        periods_data = json.load(f)
    all_periods = periods_data['periods']

    # Filter periods if specified
    if period_names:
        # "bella" is handled separately, not a DB period
        if 'bella' in period_names:
            include_bella = True
            period_names = [p for p in period_names if p != 'bella']

        periods = [p for p in all_periods if p['name'] in period_names]
        if period_names:
            missing = set(period_names) - {p['name'] for p in periods}
            if missing:
                print(f"WARNING: Unknown periods: {missing}")
                print(f"Available: {[p['name'] for p in all_periods] + ['bella']}")
    else:
        periods = all_periods

    print(f"\n{'='*70}")
    print(f"  TASK_682: Easy Mode Benchmark — Real Market Data")
    print(f"{'='*70}")

    # Load checkpoint
    print(f"\nCheckpoint: {checkpoint_path}")
    params, metadata = load_checkpoint(checkpoint_path)
    if 'epoch' in metadata:
        print(f"Epoch: {metadata['epoch']}")

    # Detect config: checkpoint metadata > config.json > preset fallback
    ckpt_dir = Path(checkpoint_path).parent
    config_json = ckpt_dir / 'config.json'

    # raw_config_dict holds the full unfiltered config dict (Phase-2 fields
    # included) so Phase-2-only flags survive the EasyConfig field filter.
    raw_config_dict = {}
    if 'config' in metadata:
        ckpt_config = metadata['config']
        raw_config_dict = dict(ckpt_config)
        config = EasyConfig(**{k: v for k, v in ckpt_config.items()
                               if k in EasyConfig.__dataclass_fields__})
        print(f"Config: from checkpoint metadata ({config.name})")
    elif config_json.exists():
        with open(config_json) as f:
            cfg_data = json.load(f)
        raw_config_dict = dict(cfg_data)
        # Filter to valid EasyConfig fields (config.json may have extra keys)
        valid_fields = EasyConfig.__dataclass_fields__
        config = EasyConfig(**{k: v for k, v in cfg_data.items() if k in valid_fields})
        print(f"Config: from {config_json} ({config.name})")
    else:
        if config_preset == 'small':
            config = get_small_config()
        elif config_preset == 'large':
            config = get_large_config()
        else:
            config = get_medium_config()
        print(f"Config: {config_preset} preset (no config.json found)")

    # TASK_767: resolve disable_limit_orders. EasyConfig has no such field, so
    # a Phase-2 checkpoint trained with --disable-limit-orders carries the flag
    # only in raw_config_dict. Explicit caller arg (CLI --disable-limit-orders)
    # overrides; otherwise auto-detect from the checkpoint config. Default
    # False = canon 5-action eval, bit-identical to pre-767 behavior.
    if disable_limit_orders is None:
        disable_limit_orders = bool(raw_config_dict.get('disable_limit_orders', False))
    if disable_limit_orders:
        print("TASK_767: limit orders DISABLED for eval — MCTS masks "
              "LimitSell/LimitBuy (effective 64x3=192 action space)")

    # TASK_733 safety rail: a cheat-channel checkpoint has trained on leaked
    # future prices. Benchmarking it as a production candidate is meaningless
    # and actively harmful (cheat input isn't available at inference time).
    if getattr(config, 'use_cheat_channel', False):
        raise RuntimeError(
            "FATAL: checkpoint was trained with TASK_733 --cheat-channel "
            "(use_cheat_channel=True). A cheat model sees future prices "
            "during training; it cannot be benchmarked as a production "
            "candidate. Train a clean model (no --cheat-channel) before "
            "benchmarking. If you need the Phase-1 cheat-agreement diagnostic, "
            "read Val Agree from the training log instead."
        )
    # TASK_741 (Mint) safety rail: cheat-injection variant is the diagnostic
    # twin of the cheat channel — also requires future prices not available
    # at inference time. xs-injection is production-legal (xs scalars are
    # computed from window-internal data).
    if (getattr(config, 'use_per_asset_injection', False)
            and getattr(config, 'per_asset_inject_cheat', False)):
        raise RuntimeError(
            "FATAL: checkpoint was trained with TASK_741 cheat injection "
            "(per_asset_inject_cheat=True). Same future-information leak as "
            "TASK_733; cannot be benchmarked. Use --per-asset-injection xs "
            "for production candidates."
        )

    # Create model
    model = create_model(config)
    print(f"Model: {config.H_cycles}H×{config.L_cycles}L, state_dim={config.state_dim}")
    print(f"Oracle: k={config.oracle_k}, lookahead={config.oracle_lookahead}, "
          f"txn_cost={config.transaction_cost}, min_alpha={config.min_alpha_threshold}")

    # Encoder channels = total - state-channel block. State is 7 (6 positional +
    # 1 confidence per build_state_channels). Legacy 27-ch checkpoints (TASK_738)
    # store n_channels=34 (27+7); standard 32-ch checkpoints store 39 (32+7).
    # Non-state variants pass their full n_channels straight through.
    encoder_n_channels = config.n_channels - 7 if config.n_channels > 27 else config.n_channels
    encoder_config = EncoderConfig(
        n_assets=config.n_assets,
        window_size=config.window_size,
        n_channels=encoder_n_channels,
    )

    # TASK_759 P1': optional eval-time MCTS
    mcts_search = None
    if eval_mcts_sims > 0:
        # C3: PopArt scale/shift from checkpoint metadata (not defaults)
        popart_state = metadata.get('popart_state', None) if isinstance(metadata, dict) else None
        if popart_state is None:
            raise RuntimeError(
                "FATAL: --eval-mcts-sims > 0 requires popart_state in checkpoint "
                "metadata (value_scale/value_shift). Checkpoint missing popart_state."
            )
        value_scale = float(np.sqrt(float(popart_state['var']) + 1e-8))
        value_shift = float(popart_state['mean'])
        print(f"\nEval MCTS: sims={eval_mcts_sims}, decode={eval_mcts_decode}, "
              f"discount=0.95, value_scale={value_scale:.4f}, value_shift={value_shift:.4f}")
        mcts_search = create_mctx_search(
            model,
            num_simulations=eval_mcts_sims,
            max_num_considered_actions=min(16, config.action_space_size),
            discount=0.95,                # C2: training mcts_discount
            use_gumbel=True,
            gumbel_scale=0.0,             # C4: deterministic
            value_scale=value_scale,
            value_shift=value_shift,
            epsilon_greedy=0.0,
        )

    # Evaluate each period
    results = []
    total_t0 = time.time()

    for i, period in enumerate(periods):
        print(f"\n--- Period {i+1}/{len(periods)}: {period['name']} ---")
        print(f"  {period['description']}")
        print(f"  {period['start_date']} to {period['end_date']}, BTC: {period['btc_change_pct']:+.1f}%")

        t0 = time.time()
        ohlcv, prices, symbols = load_benchmark_prices(period, db_path, config.n_assets)

        result = run_period_eval(
            period, ohlcv, prices, symbols,
            model, params, config, encoder_config,
            mcts_search=mcts_search,
            mcts_decode=eval_mcts_decode,
            disable_limit_orders=disable_limit_orders,
        )
        results.append(result)

        print(f"  Done in {time.time() - t0:.1f}s — "
              f"Bal.Acc={result.balanced_accuracy:.1%}, "
              f"ModelPnL={result.model_pnl*100:+.2f}%, "
              f"Alpha={result.model_alpha*100:+.2f}%")

    # Bella benchmark (if requested)
    if include_bella:
        print(f"\n--- Bella Benchmark ({bella_episodes} episodes) ---")
        try:
            bella_prices_list = generate_bella_prices(n_episodes=bella_episodes)

            # Evaluate each Bella episode, then average results
            bella_results = []
            for ep_idx, bella_ohlcv in enumerate(bella_prices_list):
                bella_period = {
                    'name': f'bella_ep{ep_idx}',
                    'description': f'Bella synthetic episode {ep_idx}',
                }
                # TASK_703: bella_ohlcv is (T, n_assets, 5)
                bella_close = bella_ohlcv[:, :, 3].copy()
                symbols_bella = [f'BELLA_{i:02d}' for i in range(bella_ohlcv.shape[1])]

                r = run_period_eval(
                    bella_period, bella_ohlcv, bella_close, symbols_bella,
                    model, params, config, encoder_config,
                    mcts_search=mcts_search,
                    mcts_decode=eval_mcts_decode,
                    disable_limit_orders=disable_limit_orders,
                )
                bella_results.append(r)

            # Average Bella results into one PeriodResult
            bella_avg = PeriodResult(
                name='bella',
                description=f'Bella synthetic (avg of {bella_episodes} episodes)',
                n_steps=int(np.mean([r.n_steps for r in bella_results])),
                n_assets=bella_results[0].n_assets,
                symbols=['BELLA_AVG'],
                balanced_accuracy=float(np.mean([r.balanced_accuracy for r in bella_results])),
                agree_trade=float(np.mean([r.agree_trade for r in bella_results])),
                agree_hold=float(np.mean([r.agree_hold for r in bella_results])),
                oracle_pnl=float(np.mean([r.oracle_pnl for r in bella_results])),
                model_pnl=float(np.mean([r.model_pnl for r in bella_results])),
                pnl_capture_ratio=float(np.mean([r.pnl_capture_ratio for r in bella_results])),
                buy_hold_return=float(np.mean([r.buy_hold_return for r in bella_results])),
                model_alpha=float(np.mean([r.model_alpha for r in bella_results])),
                oracle_alpha=float(np.mean([r.oracle_alpha for r in bella_results])),
                model_sharpe=float(np.mean([r.model_sharpe for r in bella_results])),
                max_drawdown=float(np.mean([r.max_drawdown for r in bella_results])),
                oracle_trades=int(np.mean([r.oracle_trades for r in bella_results])),
                model_trades=int(np.mean([r.model_trades for r in bella_results])),
                oracle_trade_rate=float(np.mean([r.oracle_trade_rate for r in bella_results])),
                model_trade_rate=float(np.mean([r.model_trade_rate for r in bella_results])),
                random_pnl=float(np.mean([r.random_pnl for r in bella_results])),
                value_mse=float(np.mean([r.value_mse for r in bella_results])),
                oracle_mask_accuracy=float(np.mean([r.oracle_mask_accuracy for r in bella_results])),
            )
            results.append(bella_avg)
            print(f"  Bella avg: Bal.Acc={bella_avg.balanced_accuracy:.1%}, "
                  f"ModelPnL={bella_avg.model_pnl*100:+.2f}%")

        except Exception as e:
            print(f"  Bella benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTotal benchmark time: {time.time() - total_t0:.1f}s")

    # Print report
    print_benchmark_report(results, checkpoint_path, config)

    return results


# =============================================================================
# Report
# =============================================================================

def print_benchmark_report(
    results: List[PeriodResult],
    checkpoint_path: str,
    config: EasyConfig,
):
    """Print formatted benchmark report."""
    print(f"\n{'='*80}")
    print(f"  Easy Mode Benchmark — Real Market Data")
    print(f"{'='*80}")
    print(f"  Checkpoint: {Path(checkpoint_path).name}")
    print(f"  Oracle: k={config.oracle_k}, lookahead={config.oracle_lookahead}")
    print(f"{'='*80}\n")

    # Header
    hdr = (f"{'Period':<14} {'Bal.Acc':>7} {'Trade%':>7} {'OraclePnL':>10} "
           f"{'ModelPnL':>10} {'RandomPnL':>10} {'Alpha':>8} {'Sharpe':>7} {'MaxDD':>7}")
    print(hdr)
    print("-" * len(hdr))

    # Per-period rows
    alphas_positive = 0
    beats_random = 0
    trade_density_ok = 0

    for r in results:
        row = (f"{r.name:<14} {r.balanced_accuracy:>6.1%} {r.model_trade_rate*100:>6.1f}% "
               f"{r.oracle_pnl*100:>+9.2f}% {r.model_pnl*100:>+9.2f}% "
               f"{r.random_pnl*100:>+9.2f}% {r.model_alpha*100:>+7.2f}% "
               f"{r.model_sharpe:>6.2f} {r.max_drawdown*100:>6.2f}%")
        print(row)

        if r.model_alpha > 0:
            alphas_positive += 1
        if r.model_pnl > r.random_pnl:
            beats_random += 1
        if r.model_trade_rate > 0.01:
            trade_density_ok += 1

    # Summary
    print("-" * len(hdr))
    n = len(results)
    avg_ba = np.mean([r.balanced_accuracy for r in results])
    avg_trade = np.mean([r.model_trade_rate for r in results])
    avg_oracle = np.mean([r.oracle_pnl for r in results])
    avg_model = np.mean([r.model_pnl for r in results])
    avg_random = np.mean([r.random_pnl for r in results])
    avg_alpha = np.mean([r.model_alpha for r in results])
    avg_sharpe = np.mean([r.model_sharpe for r in results])
    avg_dd = np.mean([r.max_drawdown for r in results])

    avg_row = (f"{'AVERAGE':<14} {avg_ba:>6.1%} {avg_trade*100:>6.1f}% "
               f"{avg_oracle*100:>+9.2f}% {avg_model*100:>+9.2f}% "
               f"{avg_random*100:>+9.2f}% {avg_alpha*100:>+7.2f}% "
               f"{avg_sharpe:>6.2f} {avg_dd*100:>6.2f}%")
    print(avg_row)

    # Value MSE
    avg_vmse = np.mean([r.value_mse for r in results])
    print(f"\nValue MSE (avg): {avg_vmse:.4f}")

    # Success criteria
    print(f"\n--- TASK_666 Success Criteria ---")
    print(f"  Positive alpha:   {alphas_positive}/{n} periods "
          f"{'PASS' if alphas_positive >= 4 else 'FAIL'} (need 4/{n})")
    print(f"  Beats random:     {beats_random}/{n} periods "
          f"{'PASS' if beats_random == n else 'FAIL'} (need {n}/{n})")
    print(f"  Trade density:    {trade_density_ok}/{n} periods > 1% "
          f"{'PASS' if trade_density_ok == n else 'FAIL'} (need {n}/{n})")
    print(f"  Balanced acc:     {avg_ba:.1%} "
          f"{'PASS' if avg_ba >= 0.75 else 'FAIL'} (need >= 75%)")
    print(f"  Value MSE:        {avg_vmse:.4f} "
          f"{'PASS' if avg_vmse < 0.1 else 'FAIL'} (need < 0.1)")
    print(f"{'='*80}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TASK_682: Real Data Benchmark for Easy Mode'
    )
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to model checkpoint (.pkl)')
    parser.add_argument('--db', type=str, default='trading_data.db',
                        help='Path to trading database (default: trading_data.db)')
    parser.add_argument('--periods', nargs='+', type=str, default=None,
                        help='Specific periods to evaluate (default: all)')
    parser.add_argument('--preset', type=str, choices=['small', 'medium', 'large'],
                        default='large',
                        help='Config preset if not in checkpoint (default: large)')
    parser.add_argument('--bella', action='store_true',
                        help='Include Bella-generated data benchmark')
    parser.add_argument('--bella-episodes', type=int, default=10,
                        help='Number of Bella episodes to average (default: 10)')
    parser.add_argument('--json', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    # TASK_759 P1': eval-time MCTS
    parser.add_argument('--eval-mcts-sims', type=int, default=0,
                        help='If >0, replace argmax with MCTS using N sims (default: 0 = argmax)')
    parser.add_argument('--eval-mcts-decode', type=str, default='joint',
                        choices=['joint', 'per_asset'],
                        help='How to decode MCTS output to per-asset actions (default: joint)')
    # TASK_767: action-space restructure. Default (flag absent) = auto-detect
    # from checkpoint config. --disable-limit-orders / --no-disable-limit-orders
    # force the eval-MCTS mask on/off regardless of checkpoint metadata.
    parser.add_argument('--disable-limit-orders', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='TASK_767: mask LimitSell/LimitBuy in eval-MCTS. '
                             'Default = auto-detect from checkpoint config.')

    args = parser.parse_args()

    # Validate
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found: {args.db}")
        sys.exit(1)

    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)

    # Run benchmark
    results = run_full_benchmark(
        str(ckpt_path),
        db_path=str(db_path),
        period_names=args.periods,
        config_preset=args.preset,
        include_bella=args.bella,
        bella_episodes=args.bella_episodes,
        eval_mcts_sims=args.eval_mcts_sims,
        eval_mcts_decode=args.eval_mcts_decode,
        disable_limit_orders=args.disable_limit_orders,
    )

    # Save JSON if requested
    if args.json:
        output = {
            'checkpoint': str(ckpt_path),
            'periods': []
        }
        for r in results:
            d = asdict(r)
            d['symbols'] = d['symbols'][:10]  # Truncate for readability
            output['periods'].append(d)

        # Summary
        n = len(results)
        positive_alpha = sum(1 for r in results if r.model_alpha > 0)
        beats_random = sum(1 for r in results if r.model_pnl > r.random_pnl)
        trade_density_ok = sum(1 for r in results if r.model_trade_rate > 0.01)
        avg_ba = float(np.mean([r.balanced_accuracy for r in results]))
        avg_vmse = float(np.mean([r.value_mse for r in results]))

        # TASK_706: Pass/fail based on TASK_666 criteria
        passed = (positive_alpha >= 4 and beats_random == n and
                  trade_density_ok == n and avg_ba >= 0.75 and avg_vmse < 0.1)

        output['summary'] = {
            'avg_balanced_accuracy': avg_ba,
            'avg_model_pnl': float(np.mean([r.model_pnl for r in results])),
            'avg_alpha': float(np.mean([r.model_alpha for r in results])),
            'avg_sharpe': float(np.mean([r.model_sharpe for r in results])),
            'avg_value_mse': avg_vmse,
            'positive_alpha_count': positive_alpha,
            'beats_random_count': beats_random,
            'trade_density_ok_count': trade_density_ok,
            'passed': passed,
        }

        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.json}")


if __name__ == '__main__':
    main()

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


# =============================================================================
# Price Loading
# =============================================================================

def _get_schedule_assets(start_date: str, end_date: str) -> Optional[List[str]]:
    """Load pre-computed asset list from training_schedule.json.

    Finds training periods adjacent to the benchmark date range
    and returns their available assets, avoiding the expensive
    _load_top_assets GROUP BY query on the full database.
    """
    schedule_path = Path(__file__).parent.parent / "training_schedule.json"
    if not schedule_path.exists():
        return None

    from datetime import datetime, timezone

    bm_start = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
    bm_end = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

    with open(schedule_path) as f:
        schedule = json.load(f)

    # Collect assets from periods adjacent to benchmark dates (within 30 days)
    assets = set()
    margin = 30 * 86400
    for period in schedule.get('periods', []):
        if period['end'] >= bm_start - margin and period['start'] <= bm_end + margin:
            assets.update(period.get('available_assets', []))

    if assets:
        logging.info(f"Loaded {len(assets)} assets from training_schedule.json (skipping DB scan)")
        return list(assets)
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

    # Load pre-computed assets from schedule (avoids expensive _load_top_assets)
    schedule_assets = _get_schedule_assets(period['start_date'], period['end_date'])

    market = HistoricalMarket(
        db_path=db_path,
        n_assets=n_assets,
        start_date=period['start_date'],
        end_date=period['end_date'],
        use_ram_cache=True,
        skip_validation=True,
        specific_symbols=schedule_assets,
    )

    # Extract close prices: cache_data shape is (T, n_assets_actual, 5), channel 3 = close
    prices = market.cache_data[:, :, 3]  # (T, n_assets_actual)
    actual_assets = prices.shape[1]
    symbols = [market.id_to_symbol[aid] for aid in market.asset_ids]

    print(f"  Loaded: {prices.shape[0]} timesteps, {actual_assets} assets")

    # Handle asset underflow: duplicate existing assets (NOT zero-pad)
    if actual_assets < n_assets:
        print(f"  WARNING: Only {actual_assets} assets, duplicating to fill {n_assets}")
        repeats = (n_assets // actual_assets) + 1
        prices = np.tile(prices, (1, repeats))[:, :n_assets]
        symbols = (symbols * repeats)[:n_assets]
    elif actual_assets > n_assets:
        prices = prices[:, :n_assets]
        symbols = symbols[:n_assets]

    # Verify no zero prices (would cause inf/NaN in log returns)
    zero_mask = prices <= 0
    if np.any(zero_mask):
        n_zeros = np.sum(zero_mask)
        print(f"  WARNING: {n_zeros} zero/negative prices, filling gaps...")
        for col in range(prices.shape[1]):
            # Forward-fill
            for row in range(1, prices.shape[0]):
                if prices[row, col] <= 0 and prices[row - 1, col] > 0:
                    prices[row, col] = prices[row - 1, col]
            # Backward-fill (for initial gaps where row 0 is zero)
            for row in range(prices.shape[0] - 2, -1, -1):
                if prices[row, col] <= 0 and prices[row + 1, col] > 0:
                    prices[row, col] = prices[row + 1, col]
        # Final check
        still_zero = np.sum(prices <= 0)
        if still_zero > 0:
            print(f"  WARNING: {still_zero} prices still zero after fill — entire-column gaps")

    market.close()
    return prices, symbols


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

    for t in range(T):
        step_pnl = 0.0

        for i in range(n_assets):
            if actions[t, i] == 2 or actions[t, i] == 4:  # BUY or LIMITBUY
                if position[i] == -1:
                    position[i] = 0.0
                    trade_count += 1
                    step_pnl -= transaction_cost
                elif position[i] == 0:
                    position[i] = 1.0
                    trade_count += 1
                    step_pnl -= transaction_cost
            elif actions[t, i] == 0 or actions[t, i] == 3:  # SELL or LIMITSELL
                if position[i] == 1:
                    position[i] = 0.0
                    trade_count += 1
                    step_pnl -= transaction_cost
                elif position[i] == 0:
                    position[i] = -1.0
                    trade_count += 1
                    step_pnl -= transaction_cost

        step_pnl += np.sum(position * returns[t])
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

        # P&L
        total_pnl += np.sum(position * returns[t]) - n_trades * transaction_cost

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
        prices = generate_bella_prices_on_device(bella_gen, subkey, n_envs=1)
        # prices: (seq_len, n_assets)
        episodes.append(np.array(prices))

    print(f"  Generated {n_episodes} Bella episodes ({seq_len} steps × {episodes[0].shape[1]} assets)")
    return episodes


# =============================================================================
# Period Evaluation
# =============================================================================

def run_period_eval(
    period: dict,
    prices: np.ndarray,
    symbols: List[str],
    model: MuZeroNetworks,
    params: dict,
    config: EasyConfig,
    encoder_config: EncoderConfig,
) -> PeriodResult:
    """Evaluate model on a single benchmark period.

    Temporal alignment:
    - obs[i] covers prices[i:i+window_size]
    - Action from obs[i] applies to returns[i+window_size-1]
    - Oracle and model actions are aligned to the SAME return indices
    """
    n_assets = config.n_assets
    window_size = config.window_size
    T_prices = prices.shape[0]

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
    prices_jax = jnp.array(prices)
    use_state_channels = config.n_channels > 27
    BATCH = 128
    window_starts = jnp.arange(n_windows)

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
            all_model_actions.append(np.array(jnp.argmax(logits_per_asset, axis=-1)))
            all_model_values.append(value_logits_to_scalar(np.array(output['value'])))
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
            actions_i = np.array(jnp.argmax(logits_per_asset, axis=-1))[0]  # (n_assets,)
            all_model_actions.append(actions_i)
            all_model_values.append(value_logits_to_scalar(np.array(value_logits))[0])

            # 5. Update confidence from model's own softmax
            model_softmax = np.array(jax.nn.softmax(logits_per_asset[0], axis=-1))  # (n_assets, 5)
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
    value_mse = float(np.mean((model_values - oracle_remaining * config.value_scale) ** 2))

    # P&L capture ratio
    pnl_capture = model_pnl / oracle_pnl if oracle_pnl > 1e-10 else 0.0

    # Trade rates
    total_decisions = n_windows * n_assets
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

    if 'config' in metadata:
        ckpt_config = metadata['config']
        config = EasyConfig(**{k: v for k, v in ckpt_config.items()
                               if k in EasyConfig.__dataclass_fields__})
        print(f"Config: from checkpoint metadata ({config.name})")
    elif config_json.exists():
        with open(config_json) as f:
            cfg_data = json.load(f)
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

    # Create model
    model = create_model(config)
    print(f"Model: {config.H_cycles}H×{config.L_cycles}L, state_dim={config.state_dim}")
    print(f"Oracle: k={config.oracle_k}, lookahead={config.oracle_lookahead}, "
          f"txn_cost={config.transaction_cost}, min_alpha={config.min_alpha_threshold}")

    # Encoder config
    encoder_config = EncoderConfig(
        n_assets=config.n_assets,
        window_size=config.window_size,
    )

    # Evaluate each period
    results = []
    total_t0 = time.time()

    for i, period in enumerate(periods):
        print(f"\n--- Period {i+1}/{len(periods)}: {period['name']} ---")
        print(f"  {period['description']}")
        print(f"  {period['start_date']} to {period['end_date']}, BTC: {period['btc_change_pct']:+.1f}%")

        t0 = time.time()
        prices, symbols = load_benchmark_prices(period, db_path, config.n_assets)

        result = run_period_eval(
            period, prices, symbols,
            model, params, config, encoder_config,
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
            for ep_idx, bella_prices in enumerate(bella_prices_list):
                bella_period = {
                    'name': f'bella_ep{ep_idx}',
                    'description': f'Bella synthetic episode {ep_idx}',
                }
                symbols_bella = [f'BELLA_{i:02d}' for i in range(bella_prices.shape[1])]

                r = run_period_eval(
                    bella_period, bella_prices, symbols_bella,
                    model, params, config, encoder_config,
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
        output['summary'] = {
            'avg_balanced_accuracy': float(np.mean([r.balanced_accuracy for r in results])),
            'avg_model_pnl': float(np.mean([r.model_pnl for r in results])),
            'avg_alpha': float(np.mean([r.model_alpha for r in results])),
            'avg_sharpe': float(np.mean([r.model_sharpe for r in results])),
            'avg_value_mse': float(np.mean([r.value_mse for r in results])),
            'positive_alpha_count': sum(1 for r in results if r.model_alpha > 0),
            'beats_random_count': sum(1 for r in results if r.model_pnl > r.random_pnl),
        }

        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.json}")


if __name__ == '__main__':
    main()

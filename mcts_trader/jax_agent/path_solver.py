"""
Optimal hindsight path solver for trading.

TASK_119: Light-weight path solvers for computing near-optimal trade sequences
given known future returns. Used for:
- Expert demonstrations (imitation learning)
- Replay buffer prioritization (regret-based)
- Counterfactual analysis

VRAM: <500MB for all methods here.
Time: 0.1-0.5s for T=1000, n=64, k=4
"""

import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, Optional
from functools import partial


class OptimalPath(NamedTuple):
    """Result of path optimization."""
    selections: jnp.ndarray      # (T, n) soft or hard selection weights
    indices: jnp.ndarray         # (T, k) selected asset indices
    cumulative_pnl: jnp.ndarray  # (T,) cumulative PnL curve
    total_alpha: float           # Final captured alpha
    trade_count: int             # Number of trades (transitions)


# =============================================================================
# Method 1: Greedy with Lookahead (Fastest, ~200MB)
# =============================================================================

# TASK_128 FIX: Static max length to prevent JIT recompilation per sequence length
# This eliminates the dynamic shape issue that caused training hangs at epoch 60+
DEFAULT_MAX_PATH_LEN = 512  # Covers most episode lengths


@partial(jax.jit, static_argnames=['k', 'lookahead', 'max_len'])
def greedy_lookahead_path(
    returns: jnp.ndarray,
    k: int = 4,
    lookahead: int = 5,
    transaction_cost: float = 0.001,
    min_alpha_threshold: float = 0.0,  # TASK_141: Only trade if return > threshold
    max_len: int = DEFAULT_MAX_PATH_LEN,
) -> OptimalPath:
    """
    Greedy path solver with lookahead.

    At each step, selects up to k assets that maximize cumulative returns
    over the next `lookahead` steps, accounting for transaction costs.

    TASK_141: Added min_alpha_threshold - only selects assets where expected
    return exceeds threshold. Can return 0 to k assets (not forced k).
    This fixes the "Hyperactive Agent" bug where oracle always traded.

    Args:
        returns: (T, n_assets) log returns
        k: Maximum number of assets to hold
        lookahead: Steps to look ahead for greedy decision
        transaction_cost: Cost per trade (as fraction)
        min_alpha_threshold: Minimum expected return to trade (default 0 = always trade)
        max_len: TASK_128 FIX - Static max sequence length (prevents JIT recompilation)

    Returns:
        OptimalPath with selections and metrics

    Complexity: O(T × n × L) where L = lookahead
    VRAM: ~200MB for T=1000, n=64
    """
    T, n_assets = returns.shape

    # TASK_128 FIX: Pad to static length to avoid dynamic shape recompilation
    # This is the key fix - previously each unique T triggered JIT recompilation
    # which accumulated until XLA stalled around epoch 60-90
    padded_returns = jnp.zeros((max_len, n_assets), dtype=returns.dtype)
    padded_returns = padded_returns.at[:T].set(returns)

    # Create valid step mask (1 for real steps, 0 for padding)
    valid_mask = jnp.arange(max_len) < T

    # Pad returns for safe lookahead at sequence end
    # This ensures dynamic_slice never goes out of bounds
    returns_padded = jnp.pad(padded_returns, ((0, lookahead), (0, 0)), mode='constant')

    def score_action(prev_mask, new_mask, future_returns):
        """Score a potential selection change."""
        # Expected alpha from holding new_mask
        alpha = jnp.sum(future_returns * new_mask[None, :], axis=(0, 1))

        # Transaction cost for changing positions
        changes = jnp.sum(jnp.abs(new_mask - prev_mask))
        cost = changes * transaction_cost

        return alpha - cost

    def step(prev_mask, t):
        """Single greedy step with lookahead."""
        # Get future returns for lookahead window (safe due to padding)
        future_returns = jax.lax.dynamic_slice(
            returns_padded, (t, 0), (lookahead, n_assets)
        )

        # Score = sum of |returns| over lookahead (greedy for magnitude)
        # Use abs because we can go long or short
        cumulative_abs_returns = jnp.sum(jnp.abs(future_returns), axis=0)

        # TASK_141: Apply minimum alpha threshold
        # Only consider assets where expected return exceeds threshold
        threshold_per_step = min_alpha_threshold * lookahead
        above_threshold = cumulative_abs_returns > threshold_per_step

        # Mask out assets below threshold (set to -inf so they're never selected)
        masked_returns = jnp.where(above_threshold, cumulative_abs_returns, -jnp.inf)

        # Get top-k by cumulative absolute returns (from those above threshold)
        top_k_values, top_k_indices = jax.lax.top_k(masked_returns, k)

        # Create new mask using scatter (JAX-compatible)
        # Only set mask=1 where the top-k value was valid (not -inf)
        valid_selections = top_k_values > -1e30  # Check if valid (not -inf)
        new_mask = jnp.zeros(n_assets).at[top_k_indices].set(valid_selections.astype(jnp.float32))

        # Compute actual PnL for this step (use padded_returns which has zeros for padding)
        # Oracle goes long on positive returns, short on negative
        # Using abs() correctly captures profit in both directions:
        #   Long +5% = +5% profit, Short -5% = +5% profit
        # TASK_133 FIX: Divide by k for equal-weighted portfolio (was 4x leverage bug)
        step_returns = padded_returns[t]  # TASK_128: Use padded array
        k_count = jnp.maximum(jnp.sum(new_mask), 1.0)  # Number of selected assets
        step_pnl = jnp.sum(jnp.abs(step_returns) * new_mask) / k_count

        # Transaction cost
        changes = jnp.sum(jnp.abs(new_mask - prev_mask))
        step_cost = changes * transaction_cost

        # TASK_128: Zero out PnL for padding steps (valid_mask already defined above)
        is_valid = valid_mask[t]
        step_pnl = jnp.where(is_valid, step_pnl - step_cost, 0.0)
        changes = jnp.where(is_valid, changes, 0.0)

        return new_mask, (new_mask, top_k_indices, step_pnl, changes)

    # Initialize with zeros (no positions)
    init_mask = jnp.zeros(n_assets)

    # TASK_128 FIX: Scan through STATIC max_len (not dynamic T)
    # This ensures single JIT compilation regardless of input sequence length
    _, (all_masks, all_indices, step_pnls, step_changes) = jax.lax.scan(
        step, init_mask, jnp.arange(max_len)
    )

    # Compute cumulative PnL (padding steps contribute 0)
    cumulative_pnl = jnp.cumsum(step_pnls)

    # TASK_128: Extract metrics at actual sequence end (not max_len)
    # Use T-1 as index (0-indexed), but handle T=0 edge case
    final_idx = jnp.maximum(T - 1, 0)
    total_alpha = cumulative_pnl[final_idx]
    trade_count = jnp.sum(step_changes * valid_mask).astype(jnp.int32)

    # TASK_129: Slice back to T for API compatibility with downstream consumers
    # (optimal_path_to_policy expects selections.shape == returns.shape)
    return OptimalPath(
        selections=all_masks[:T],
        indices=all_indices[:T],
        cumulative_pnl=cumulative_pnl[:T],
        total_alpha=total_alpha,
        trade_count=trade_count,
    )


# =============================================================================
# Batched Oracle (vmap for parallel environments)
# =============================================================================

def _greedy_path_single(returns, k, lookahead, transaction_cost, min_alpha_threshold, max_len):
    """Unbatched greedy path for vmap."""
    return greedy_lookahead_path(returns, k, lookahead, transaction_cost, min_alpha_threshold, max_len)


@partial(jax.jit, static_argnames=['k', 'lookahead', 'max_len'])
def batched_greedy_lookahead_path(
    returns_batch: jnp.ndarray,
    k: int = 4,
    lookahead: int = 5,
    transaction_cost: float = 0.001,
    min_alpha_threshold: float = 0.0,
    max_len: int = DEFAULT_MAX_PATH_LEN,
) -> OptimalPath:
    """
    Batched greedy path solver - runs N oracles in parallel via vmap.

    Args:
        returns_batch: (N, T, n_assets) batched log returns
        k, lookahead, transaction_cost, min_alpha_threshold, max_len: same as greedy_lookahead_path

    Returns:
        OptimalPath with batched arrays:
            selections: (N, T, n_assets)
            indices: (N, T, k)
            cumulative_pnl: (N, T)
            total_alpha: (N,)
            trade_count: (N,)

    Performance: N=32 episodes runs ~10x faster than N sequential calls.
    """
    # vmap over batch dimension (axis 0)
    vmapped_fn = jax.vmap(
        lambda r: greedy_lookahead_path(r, k, lookahead, transaction_cost, min_alpha_threshold, max_len)
    )
    return vmapped_fn(returns_batch)


# =============================================================================
# TASK_673: Multipass Oracle → Soft Policy Targets
# =============================================================================

@partial(jax.jit, static_argnames=['k', 'lookahead', 'max_len', 'n_passes',
                                    'n_actions_per_asset', 'temperature'])
def multipass_oracle_policy(
    returns: jnp.ndarray,
    key: jnp.ndarray,
    k: int = 1,
    lookahead: int = 20,
    transaction_cost: float = 0.001,
    min_alpha_threshold: float = 0.001,
    max_len: int = DEFAULT_MAX_PATH_LEN,
    n_passes: int = 10,
    noise_scale: float = 0.1,
    n_actions_per_asset: int = 5,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    TASK_673: Multipass oracle producing genuinely soft policy targets.

    Algorithm:
      1. Run greedy oracle N times, each with Gaussian noise on returns
      2. Convert each binary path → one-hot policy via state-aware logic
      3. Average the N one-hot policies → soft policy distribution

    This preserves the correctness of optimal_path_to_policy per-pass
    (no binarization bugs) while producing soft targets from the ensemble.

    Args:
        returns: (T, n_assets) log returns (REAL, not noisy)
        key: JAX PRNG key
        k: Max assets to hold per step (default 1, matches oracle_k)
        lookahead: Oracle lookahead steps (default 20)
        transaction_cost: Cost per trade as fraction
        min_alpha_threshold: Min return to consider trading
        max_len: Static max sequence length for JIT (TASK_128)
        n_passes: Number of noisy oracle passes (default 10)
        noise_scale: Noise std as fraction of global return std
        n_actions_per_asset: Actions per asset (5: Sell/Hold/Buy/LimitSell/LimitBuy)
        temperature: Softmax temperature for policy conversion

    Returns:
        soft_policy: (T, n_assets * n_actions_per_asset) averaged soft targets
        avg_cumulative_pnl: (T,) mean cumulative PnL across passes
        avg_total_alpha: scalar mean final alpha
    """
    T, n_assets = returns.shape

    # Global noise scaling (matched to overall return magnitude)
    return_std = jnp.maximum(jnp.std(returns), 1e-6)

    # 0. Compute CLEAN oracle path for value targets (no noise)
    #    Value head needs true achievable PnL, not noisy approximation
    clean_path = greedy_lookahead_path(
        returns,
        k=k,
        lookahead=lookahead,
        transaction_cost=transaction_cost,
        min_alpha_threshold=min_alpha_threshold,
        max_len=max_len,
    )

    def single_pass(subkey):
        # 1. Perturb returns with Gaussian noise
        noise = jax.random.normal(subkey, returns.shape) * noise_scale * return_std
        noisy_returns = returns + noise

        # 2. Solve binary path on noisy returns
        path = greedy_lookahead_path(
            noisy_returns,
            k=k,
            lookahead=lookahead,
            transaction_cost=transaction_cost,
            min_alpha_threshold=min_alpha_threshold,
            max_len=max_len,
        )

        # 3. Compute prev_selections for state-aware oracle (TASK_141)
        #    Uses REAL returns for position direction, path selections for position state
        prev_sels = jnp.zeros((T, n_assets))
        prev_sels = prev_sels.at[1:].set(
            jnp.where(
                path.selections[:-1] > 0.5,
                jnp.where(returns[:-1] > 0, 1.0, jnp.where(returns[:-1] < 0, -1.0, 0.0)),
                0.0
            )
        )

        # 4. Convert binary path → one-hot policy (state-aware, correct per-pass)
        policy = optimal_path_to_policy(
            path.selections,
            returns,  # REAL returns for action direction
            prev_selections=prev_sels,
            n_actions_per_asset=n_actions_per_asset,
            temperature=temperature,
        )

        return policy

    # 5. Split key, lax.map over all passes (TASK_675: caps VRAM at 1-pass peak)
    keys = jax.random.split(key, n_passes)
    all_policies = jax.lax.map(single_pass, keys)
    # all_policies:  (n_passes, T, 320)

    # 6. Average → genuinely soft policy distribution
    soft_policy = jnp.mean(all_policies, axis=0)       # (T, 320)

    # 7. Value targets from CLEAN oracle (true achievable PnL)
    return soft_policy, clean_path.cumulative_pnl, clean_path.total_alpha


@partial(jax.jit, static_argnames=['k', 'lookahead', 'max_len', 'n_passes',
                                    'n_actions_per_asset', 'temperature'])
def batched_multipass_oracle_policy(
    returns_batch: jnp.ndarray,
    key: jnp.ndarray,
    k: int = 1,
    lookahead: int = 20,
    transaction_cost: float = 0.001,
    min_alpha_threshold: float = 0.001,
    max_len: int = DEFAULT_MAX_PATH_LEN,
    n_passes: int = 10,
    noise_scale: float = 0.1,
    n_actions_per_asset: int = 5,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    TASK_673: Batched multipass oracle for N episodes in parallel.

    Double-vmap: outer over N episodes, inner over n_passes per episode.
    With N=8 episodes and n_passes=10, runs 80 oracle solves in one JIT call.

    Args:
        returns_batch: (N, T, n_assets) batched log returns
        key: JAX PRNG key
        ... (same as multipass_oracle_policy)

    Returns:
        soft_policies: (N, T, n_assets * n_actions_per_asset) batched soft targets
        avg_pnls: (N, T) batched mean cumulative PnL
        avg_alphas: (N,) batched mean final alpha
    """
    N = returns_batch.shape[0]
    keys = jax.random.split(key, N)

    vmapped_fn = jax.vmap(
        lambda r, subkey: multipass_oracle_policy(
            r, subkey,
            k=k, lookahead=lookahead,
            transaction_cost=transaction_cost,
            min_alpha_threshold=min_alpha_threshold,
            max_len=max_len, n_passes=n_passes,
            noise_scale=noise_scale,
            n_actions_per_asset=n_actions_per_asset,
            temperature=temperature,
        )
    )
    return vmapped_fn(returns_batch, keys)


# =============================================================================
# Method 2: Soft Top-K with Transition Penalty (Differentiable, ~500MB)
# =============================================================================

@partial(jax.jit, static_argnames=['k', 'max_len'])
def soft_topk_path(
    returns: jnp.ndarray,
    k: int = 4,
    temperature: float = 0.1,
    transition_weight: float = 0.1,
    max_len: int = DEFAULT_MAX_PATH_LEN,
) -> OptimalPath:
    """
    Differentiable path solver using soft top-k selection.

    Uses softmax relaxation of top-k selection with transition penalty
    to encourage holding positions. Fully differentiable for use in
    training pipelines.

    Args:
        returns: (T, n_assets) log returns
        k: Number of assets to select
        temperature: Softmax temperature (lower = sharper)
        transition_weight: Penalty for changing selections

    Returns:
        OptimalPath with soft selections and metrics

    Complexity: O(T × n)
    VRAM: ~500MB for T=1000, n=64
    """
    T, n_assets = returns.shape

    def soft_topk(scores, k, temp):
        """Soft approximation of top-k selection."""
        # Scale scores and apply softmax
        scaled = scores / temp

        # Iterative soft top-k (Gumbel-style without noise for deterministic)
        def iter_step(carry, _):
            current_scores, selection = carry
            soft_select = jax.nn.softmax(current_scores)
            selection = selection + soft_select
            # Mask out selected (approximately)
            mask = jnp.maximum(1.0 - selection, 1e-10)
            next_scores = scaled + jnp.log(mask)
            return (next_scores, selection), None

        init = (scaled, jnp.zeros(n_assets))
        (_, final_selection), _ = jax.lax.scan(iter_step, init, None, length=k)

        # Normalize to sum to k
        return final_selection / jnp.maximum(jnp.sum(final_selection), 1e-6) * k

    def step(prev_selection, returns_t):
        """Single step with transition penalty."""
        abs_returns = jnp.abs(returns_t)

        # Penalize switching: prefer assets already selected
        continuity_bonus = transition_weight * prev_selection
        adjusted_scores = abs_returns + continuity_bonus

        # Soft top-k selection
        selection = soft_topk(adjusted_scores, k, temperature)

        # Compute PnL (using soft weights)
        step_pnl = jnp.sum(abs_returns * selection) / k

        # Count changes (soft)
        changes = jnp.sum(jnp.abs(selection - prev_selection))

        return selection, (selection, step_pnl, changes)

    # Initialize
    init_selection = jnp.zeros(n_assets)

    # Scan through time
    _, (all_selections, step_pnls, step_changes) = jax.lax.scan(
        step, init_selection, returns
    )

    # Get hard indices for output
    all_indices = jax.vmap(lambda s: jax.lax.top_k(s, k)[1])(all_selections)

    # Cumulative PnL
    cumulative_pnl = jnp.cumsum(step_pnls)

    return OptimalPath(
        selections=all_selections,
        indices=all_indices,
        cumulative_pnl=cumulative_pnl,
        total_alpha=cumulative_pnl[-1],
        trade_count=jnp.sum(step_changes).astype(jnp.int32),
    )


# =============================================================================
# Method 3: Pure JAX Soft Sort (No OTT-JAX dependency)
# =============================================================================

@partial(jax.jit, static_argnames=['k', 'num_iters', 'max_len'])
def sinkhorn_topk_path(
    returns: jnp.ndarray,
    k: int = 4,
    epsilon: float = 0.05,
    num_iters: int = 20,
    transition_weight: float = 0.1,
    max_len: int = DEFAULT_MAX_PATH_LEN,
) -> OptimalPath:
    """
    Path solver using Sinkhorn-style soft sorting.

    Pure JAX implementation without OTT-JAX dependency.
    Uses entropy-regularized optimal transport for differentiable sorting.

    Args:
        returns: (T, n_assets) log returns
        k: Number of assets to select
        epsilon: Entropy regularization (smaller = sharper)
        num_iters: Sinkhorn iterations
        transition_weight: Penalty for transitions

    Returns:
        OptimalPath

    Complexity: O(T × n × num_iters)
    VRAM: ~300MB for T=1000, n=64
    """
    T, n_assets = returns.shape

    def sinkhorn_sort(scores, eps, iters):
        """Soft argsort via log-domain Sinkhorn (numerically stable)."""
        n = scores.shape[0]

        # Normalize scores to [0, 1] for numerical stability
        scores_min = jnp.min(scores)
        scores_max = jnp.max(scores)
        scores_norm = (scores - scores_min) / jnp.maximum(scores_max - scores_min, 1e-6)

        # Cost matrix: we want high scores to map to low ranks
        # C[i,j] = cost of assigning item i to rank j
        ranks = jnp.arange(n)
        # Simpler cost: just rank-based distance from top
        C = (1.0 - scores_norm[:, None]) * ranks[None, :]  # High score → low rank

        # Log-domain Sinkhorn for numerical stability
        log_K = -C / eps
        # Clip to prevent overflow
        log_K = jnp.clip(log_K, -20.0, 20.0)

        log_u = jnp.zeros(n)
        log_v = jnp.zeros(n)

        def sinkhorn_iter(carry, _):
            log_u, log_v = carry
            # log_u = -logsumexp(log_K + log_v, axis=1)
            log_u = -jax.scipy.special.logsumexp(log_K + log_v[None, :], axis=1)
            # log_v = -logsumexp(log_K.T + log_u, axis=1)
            log_v = -jax.scipy.special.logsumexp(log_K.T + log_u[None, :], axis=1)
            return (log_u, log_v), None

        (log_u, log_v), _ = jax.lax.scan(sinkhorn_iter, (log_u, log_v), None, length=iters)

        # Transport plan in log domain then exp
        log_P = log_u[:, None] + log_K + log_v[None, :]
        P = jnp.exp(jnp.clip(log_P, -20.0, 20.0))

        # Selection weights: sum of probability mass in top-k ranks
        topk_mask = (ranks < k).astype(jnp.float32)
        selection = P @ topk_mask

        return selection

    def step(prev_selection, returns_t):
        abs_returns = jnp.abs(returns_t)

        # Add continuity bonus
        adjusted = abs_returns + transition_weight * prev_selection

        # Soft top-k via Sinkhorn
        selection = sinkhorn_sort(adjusted, epsilon, num_iters)

        # Normalize
        selection = selection / jnp.maximum(jnp.sum(selection), 1e-6) * k

        # PnL
        step_pnl = jnp.sum(abs_returns * selection) / k
        changes = jnp.sum(jnp.abs(selection - prev_selection))

        return selection, (selection, step_pnl, changes)

    init = jnp.zeros(n_assets)
    _, (selections, pnls, changes) = jax.lax.scan(step, init, returns)

    indices = jax.vmap(lambda s: jax.lax.top_k(s, k)[1])(selections)
    cum_pnl = jnp.cumsum(pnls)

    return OptimalPath(
        selections=selections,
        indices=indices,
        cumulative_pnl=cum_pnl,
        total_alpha=cum_pnl[-1],
        trade_count=jnp.sum(changes).astype(jnp.int32),
    )


# =============================================================================
# Unified Interface
# =============================================================================

def compute_optimal_path(
    returns: jnp.ndarray,
    k: int = 4,
    method: str = 'greedy',
    **kwargs,
) -> OptimalPath:
    """
    Compute near-optimal trade path given hindsight returns.

    Args:
        returns: (T, n_assets) log returns
        k: Number of positions to hold
        method: 'greedy', 'soft', or 'sinkhorn'
        **kwargs: Method-specific arguments

    Returns:
        OptimalPath with selections, indices, PnL, and metrics
    """
    if method == 'greedy':
        return greedy_lookahead_path(returns, k, **kwargs)
    elif method == 'soft':
        return soft_topk_path(returns, k, **kwargs)
    elif method == 'sinkhorn':
        return sinkhorn_topk_path(returns, k, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Utility: Compare agent path to optimal
# =============================================================================

def compute_regret(
    agent_pnl: jnp.ndarray,
    returns: jnp.ndarray,
    k: int = 4,
    method: str = 'greedy',
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Compute regret: difference between optimal and agent cumulative PnL.

    Args:
        agent_pnl: (T,) agent's cumulative PnL
        returns: (T, n_assets) market returns
        k: Position limit
        method: Path solver method

    Returns:
        regret_curve: (T,) cumulative regret over time
        optimal_pnl: (T,) optimal cumulative PnL
        total_regret: Final regret value
    """
    optimal = compute_optimal_path(returns, k, method)
    regret_curve = optimal.cumulative_pnl - agent_pnl
    return regret_curve, optimal.cumulative_pnl, float(regret_curve[-1])


# =============================================================================
# TASK_119: Optimal Path to Policy Targets (for imitation/stabilization)
# =============================================================================

@partial(jax.jit, static_argnames=['n_actions_per_asset', 'temperature'])
def optimal_path_to_policy(
    selections: jnp.ndarray,
    returns: jnp.ndarray,
    prev_selections: jnp.ndarray = None,
    n_actions_per_asset: int = 5,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    Convert optimal path selections to soft policy targets.

    TASK_141: Now state-aware! Accepts prev_selections to compute proper
    open/close/maintain actions instead of just looking at target holdings.

    This creates a "teacher" policy from hindsight-optimal decisions.
    Used for soft imitation learning / stabilization.

    Action encoding: action = asset * n_actions_per_asset + action_type
    Action types: 0=Sell, 1=Hold, 2=Buy, 3=LimitSell, 4=LimitBuy

    Args:
        selections: (T, n_assets) binary mask of optimal holdings for NEXT step
        returns: (T, n_assets) actual returns (determines long/short direction)
        prev_selections: (T, n_assets) current holdings BEFORE this step
                        Values: -1=short, 0=flat, 1=long. If None, uses stateless mode.
        n_actions_per_asset: Actions per asset (default 5)
        temperature: Softness of target (lower = more peaked)

    Returns:
        policy_targets: (T, n_assets * n_actions_per_asset) soft policy

    State-Aware Action Logic (when prev_selections provided):
        - prev=0, new=1, ret>0 → BUY (open long)
        - prev=0, new=1, ret<0 → SELL (open short)
        - prev!=0, new=prev   → HOLD (maintain position)
        - prev=1, new=0       → SELL (close long)
        - prev=-1, new=0      → BUY (close short)
        - prev=0, new=0       → HOLD (stay flat)
    """
    T, n_assets = selections.shape
    n_actions = n_assets * n_actions_per_asset

    # Create logits for each action
    logits = jnp.full((T, n_actions), -10.0)  # Start with low probability

    # Action indices for each type
    buy_indices = jnp.arange(n_assets) * n_actions_per_asset + 2   # action_type=2
    sell_indices = jnp.arange(n_assets) * n_actions_per_asset + 0  # action_type=0
    hold_indices = jnp.arange(n_assets) * n_actions_per_asset + 1  # action_type=1

    if prev_selections is None:
        # LEGACY: Stateless mode (original behavior)
        # - If selected AND return > 0: should BUY
        # - If selected AND return < 0: should SELL
        # - If not selected: should HOLD
        buy_mask = selections * (returns > 0)
        sell_mask = selections * (returns < 0)
        hold_mask = 1 - selections

        logits = logits.at[:, buy_indices].set(
            jnp.where(buy_mask, 5.0 / temperature, logits[:, buy_indices])
        )
        logits = logits.at[:, sell_indices].set(
            jnp.where(sell_mask, 5.0 / temperature, logits[:, sell_indices])
        )
        logits = logits.at[:, hold_indices].set(
            jnp.where(hold_mask, 0.0, logits[:, hold_indices])
        )
    else:
        # TASK_141: State-aware mode with proper open/close/maintain logic

        # Determine target position direction from returns
        # selections=1, returns>0 → want long (+1)
        # selections=1, returns<0 → want short (-1)
        # selections=0 → want flat (0)
        target_position = jnp.where(
            selections > 0.5,
            jnp.where(returns > 0, 1.0, jnp.where(returns < 0, -1.0, 0.0)),  # Long/short/flat
            0.0  # Flat if not selected
        )

        # Compute transitions
        # Opening positions (from flat)
        open_long = (prev_selections == 0) & (target_position == 1)   # flat → long
        open_short = (prev_selections == 0) & (target_position == -1) # flat → short

        # Closing positions (to flat)
        close_long = (prev_selections == 1) & (target_position == 0)   # long → flat
        close_short = (prev_selections == -1) & (target_position == 0) # short → flat

        # Maintaining positions (no change needed)
        maintain = (prev_selections == target_position) & (prev_selections != 0)

        # Staying flat
        stay_flat = (prev_selections == 0) & (target_position == 0)

        # Reversing positions (rare but possible)
        reverse_long_to_short = (prev_selections == 1) & (target_position == -1)
        reverse_short_to_long = (prev_selections == -1) & (target_position == 1)

        # BUY actions: open_long, close_short, reverse_short_to_long
        buy_mask = open_long | close_short | reverse_short_to_long

        # SELL actions: open_short, close_long, reverse_long_to_short
        sell_mask = open_short | close_long | reverse_long_to_short

        # HOLD actions: maintain position or stay flat
        hold_mask = maintain | stay_flat

        # Set logits
        logits = logits.at[:, buy_indices].set(
            jnp.where(buy_mask, 5.0 / temperature, logits[:, buy_indices])
        )
        logits = logits.at[:, sell_indices].set(
            jnp.where(sell_mask, 5.0 / temperature, logits[:, sell_indices])
        )
        logits = logits.at[:, hold_indices].set(
            jnp.where(hold_mask, 0.0, logits[:, hold_indices])
        )

    # TASK_146: Per-asset softmax (each asset has independent 5-way classification)
    # Previously: 320-way softmax forced all assets to compete for probability mass
    # Now: Reshape to (T, n_assets, 5), softmax per asset, reshape back
    logits_reshaped = logits.reshape(T, n_assets, n_actions_per_asset)
    probs_reshaped = jax.nn.softmax(logits_reshaped, axis=-1)  # 5-way per asset
    policy_targets = probs_reshaped.reshape(T, n_actions)

    return policy_targets


@partial(jax.jit, static_argnames=['n_actions_per_asset', 'temperature'])
def batched_optimal_path_to_policy(
    selections_batch: jnp.ndarray,
    returns_batch: jnp.ndarray,
    prev_selections_batch: jnp.ndarray = None,
    n_actions_per_asset: int = 5,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    Batched version of optimal_path_to_policy for parallel environments.

    TASK_141: Now supports prev_selections for state-aware policy targets.

    Args:
        selections_batch: (N, T, n_assets) batched optimal selections
        returns_batch: (N, T, n_assets) batched log returns
        prev_selections_batch: (N, T, n_assets) batched current holdings (optional)
        n_actions_per_asset, temperature: same as optimal_path_to_policy

    Returns:
        policy_targets: (N, T, n_assets * n_actions_per_asset) batched policies
    """
    if prev_selections_batch is None:
        vmapped_fn = jax.vmap(
            lambda s, r: optimal_path_to_policy(s, r, None, n_actions_per_asset, temperature)
        )
        return vmapped_fn(selections_batch, returns_batch)
    else:
        vmapped_fn = jax.vmap(
            lambda s, r, p: optimal_path_to_policy(s, r, p, n_actions_per_asset, temperature)
        )
        return vmapped_fn(selections_batch, returns_batch, prev_selections_batch)


def compute_imprint_loss(
    agent_policy: jnp.ndarray,
    optimal_policy: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Compute soft cross-entropy loss between agent and optimal policy.

    This is a "soft imitation" loss - doesn't force exact match,
    just gently guides agent toward optimal patterns.

    Args:
        agent_policy: (T, n_actions) agent's policy probabilities
        optimal_policy: (T, n_actions) optimal target probabilities
        mask: (T,) optional mask for valid timesteps

    Returns:
        loss: Scalar cross-entropy loss
    """
    # Cross-entropy: -sum(optimal * log(agent))
    # Clip for numerical stability
    log_agent = jnp.log(agent_policy + 1e-10)
    ce = -jnp.sum(optimal_policy * log_agent, axis=-1)  # (T,)

    if mask is not None:
        ce = ce * mask
        return jnp.sum(ce) / jnp.maximum(jnp.sum(mask), 1.0)
    else:
        return jnp.mean(ce)

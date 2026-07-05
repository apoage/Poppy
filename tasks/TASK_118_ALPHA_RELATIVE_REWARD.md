# TASK_118: Alpha-Relative Reward Shaping

## Status: COMPLETED

## Goal

Replace raw PnL reward with alpha-relative reward that measures how much of the *available* opportunity the agent captured.

---

## Motivation

**Problem with raw PnL reward:**
- Crash market: -5% might be excellent (market did -20%)
- Sideways market: +0.1% might be poor (could have made +2%)
- Reward scale varies wildly across regimes
- Agent can't distinguish "bad market" from "bad policy"

**Alpha-relative reward:**
```python
reward = achieved_pnl / max_possible_alpha
```

- Normalized [0, 1] across all regimes
- +0.8 means "captured 80% of what was possible"
- Works in crash, rally, sideways equally

---

## The NP-Hard Problem

Computing `max_possible_alpha` for multi-asset portfolio is NP-hard:
- 64 assets, T timesteps
- Capital constraints (can't be 100% in all assets)
- Position limits, transaction costs
- Correlation structure matters

**Brute force:** O(A^T) where A=actions, T=timesteps → intractable

---

## Candidate Approximations

### 1. Per-Asset Independent Sum (Upper Bound)
```python
def max_alpha_upper_bound(returns):
    """Sum of per-asset max alpha. Loose but O(n*T)."""
    # For each asset: sum of absolute returns (perfect timing)
    per_asset = jnp.sum(jnp.abs(returns), axis=0)  # (n_assets,)
    return jnp.sum(per_asset)
```
- **Pros:** Trivial, fast, correlates with opportunity
- **Cons:** Unreachable (can't be in all assets)

### 2. Greedy Hindsight Oracle
```python
def max_alpha_greedy(returns, max_positions=10):
    """Greedy: each step pick top-k assets. O(T * n log n)."""
    T, n_assets = returns.shape
    total = 0.0
    for t in range(T):
        top_k = jnp.argsort(jnp.abs(returns[t]))[-max_positions:]
        total += jnp.sum(jnp.abs(returns[t, top_k]))
    return total
```
- **Pros:** Respects position limits, still fast
- **Cons:** Ignores transaction costs, not globally optimal

### 3. Single Best Asset
```python
def max_alpha_single_best(returns):
    """Best single-asset strategy. O(n*T)."""
    per_asset_alpha = jnp.sum(jnp.abs(returns), axis=0)
    return jnp.max(per_asset_alpha)
```
- **Pros:** Achievable, simple baseline
- **Cons:** Too conservative, ignores diversification

### 4. Relaxed Convex (LP/QP)
```python
def max_alpha_relaxed(returns, constraints):
    """Solve convex relaxation with continuous positions."""
    # cvxpy or JAX-based solver
    # Polynomial time but heavier
    pass
```
- **Pros:** Tighter bound, respects constraints
- **Cons:** Slower, needs solver

### 5. Monte Carlo Policy Sampling
```python
def max_alpha_monte_carlo(returns, n_samples=1000):
    """Sample random policies, take best. O(n_samples * T * n)."""
    best = 0.0
    for _ in range(n_samples):
        policy = random_policy()
        alpha = simulate(policy, returns)
        best = max(best, alpha)
    return best
```
- **Pros:** Anytime algorithm, parallelizable
- **Cons:** Variance, might miss optimal

### 6. Dynamic Programming with Pruning
```python
def max_alpha_dp(returns, max_positions, prune_threshold):
    """DP with beam search / pruning. Exact for small state space."""
    # State: current positions
    # Prune unlikely branches
    pass
```
- **Pros:** Can be exact for small problems
- **Cons:** Exponential worst case

---

## Implementation Plan

### Phase 1: Simple Upper Bound
- [ ] Implement per-asset sum upper bound
- [ ] Add to reward computation in trainer
- [ ] Test on synthetic data

### Phase 2: Greedy Oracle
- [ ] Implement greedy hindsight with position limits
- [ ] Compare tightness vs upper bound
- [ ] Benchmark speed

### Phase 3: Evaluate
- [ ] A/B test: raw PnL vs alpha-relative reward
- [ ] Check learning curves across market regimes
- [ ] Measure reward scale consistency

### Phase 4: Advanced (Optional)
- [ ] LP relaxation for tighter bound
- [ ] Learned alpha predictor (prediction head outputs max_alpha)

---

## Integration Points

**Trainer (`trainer.py`):**
```python
def _compute_reward(self, returns, actions, achieved_pnl):
    max_alpha = compute_max_alpha(returns, method='greedy')
    raw_reward = achieved_pnl
    alpha_ratio = achieved_pnl / jnp.maximum(max_alpha, 1e-6)

    # Blend or replace
    reward = self.alpha_reward_weight * alpha_ratio + \
             (1 - self.alpha_reward_weight) * raw_reward
    return reward
```

**CLI flags:**
```bash
--alpha-relative-reward          # Enable alpha-relative reward
--alpha-reward-weight 0.5        # Blend with raw PnL
--alpha-method greedy            # greedy|upper_bound|single_best
--alpha-max-positions 10         # For greedy method
```

---

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| Reward variance across regimes | High | Low |
| Learning stability | Varies | Consistent |
| Performance on crash periods | Poor | Improved |
| Performance on sideways | Okay | Improved |

---

## Research Questions

1. **How tight does the bound need to be?** Loose upper bound might work fine for reward shaping.
2. **Should max_alpha be per-episode or per-step?** Per-step is noisier but denser signal.
3. **Can prediction head learn max_alpha directly?** Train to predict hindsight optimal.
4. **Transaction cost handling?** Include in max_alpha computation or separate penalty?

---

## References

- Optimal execution literature (Almgren-Chriss)
- Hindsight optimization in RL
- Reward shaping theory (potential-based)
- Multi-asset portfolio optimization (Markowitz, risk parity)

---

## SOLUTION: Top-K Selection (Implemented)

**Insight:** Don't need optimal route, just the sum. Top-k per step is sufficient.

```python
def compute_max_alpha_topk(returns, k=4):
    """O(T × n), fully JAX-compatible."""
    abs_returns = jnp.abs(returns)
    top_k_values = jax.lax.top_k(abs_returns, k)[0]  # (T, k)
    return jnp.sum(top_k_values)

# One-liner version:
max_alpha = jnp.sum(jax.lax.top_k(jnp.abs(returns), k=4)[0])
```

**Why this works:**
- Top-4 = realistic position limit (can't hold all 64)
- Sum across steps = total opportunity ceiling
- Averaging effect = implicit rebalancing assumption
- No NP-hard optimization needed

**Performance:**
- **0.093 ms** per call (1440 steps × 64 assets)
- 10,000+ calls/second
- Fully jit-able, vmap-able

**File:** `mcts_trader/jax_agent/alpha_reward.py`

**Functions:**
- `compute_max_alpha_topk(returns, k)` - Main calculation
- `compute_alpha_ratio(achieved, returns, k)` - Normalized ratio
- `alpha_relative_reward(raw, returns, k, blend)` - Blended reward
- `compute_per_step_alpha(returns, k)` - Dense per-step signal
- `compute_cumulative_alpha(returns, k)` - For visualization
- `compute_optimal_trade_count(returns, k)` - Count trades needed for max alpha
- `compute_optimal_metrics(returns, k)` - Combined: max_alpha, trade_count, alpha_per_trade

**Trade Count Logic:**
```python
def compute_optimal_trade_count(returns, k=4, smooth_window=5):
    """Count transitions when top-k asset set changes between steps.

    Uses smoothed returns to avoid counting spurious trades from
    tiny ranking fluctuations. Only counts meaningful regime changes.
    """
    # Smooth returns with rolling window (adds persistence)
    smoothed = rolling_mean(abs_returns, smooth_window)
    top_k_indices = jax.lax.top_k(smoothed, k)[1]
    masks = vmap(to_mask)(top_k_indices)
    changes = jnp.abs(masks[1:] - masks[:-1])
    return jnp.sum(changes)
```

**Trade Count vs Smoothing Window (100 steps × 64 assets, random data):**
| Window | Trades | Trades/Step | Alpha/Trade |
|--------|--------|-------------|-------------|
| 1 (none) | 756 | 7.64 | 0.0227 |
| 5 | 392 | 3.96 | 0.0439 |
| 10 | 310 | 3.13 | 0.0555 |
| 20 | 212 | 2.14 | 0.0811 |

**With AR(1) persistent data (realistic):**
- Window=5: 198 trades (~2/step) - reasonable
- Window=10: 132 trades (~1.3/step)

Note: Uses |returns| so captures both long AND short opportunities.

---

## Notes

The key insight: we don't need the *exact* optimal, we need a *consistent* upper bound that correlates with available opportunity. Even a loose bound that's 2x the true optimal is fine if it's consistently 2x across all regimes.

Top-k selection gives us exactly this: consistent, fast, and correlates with opportunity.

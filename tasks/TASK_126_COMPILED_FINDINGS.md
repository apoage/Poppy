# TASK_126: Compiled Bug Hunt Findings

**Compiled from**: 126a-126m reports from multiple agents
**Date**: 2026-01-31

---

## Executive Summary

| Bug | Consensus | Severity | Fix Complexity |
|-----|-----------|----------|----------------|
| **BUG_A: Post-done rewards** | CONFIRMED by all | HIGH | Simple |
| **BUG_A+: Alpha-relative scale** | NEW from 126b | HIGH | Simple |
| **BUG_A++: Random burn ordering** | NEW from 126b | MEDIUM | Simple |
| **BUG_B: Path solver signs** | DISPUTED | LOW | N/A |
| **BUG_B+: Transaction cost issue** | NEW from 126c/126k | HIGH | Config change |
| **BUG_C: Log vs Linear scale** | CONFIRMED by all | HIGH | Design decision |
| **BUG_D: Numerical stability** | AUDITED | LOW | Few edge cases |

---

## BUG_A: Reward Spike (-30.99) - MULTIPLE CAUSES

### A1: Post-Done Reward Accumulation (CONFIRMED)
**Sources**: 126a, 126b, 126f, my analysis

```python
# self_play.py:722 - WRONG
scores = np.sum(cpu_rewards, axis=1)  # Includes post-termination!
```

After liquidation, `symlog(-50) = -3.93` per step continues accumulating.

**Fix**:
```python
scores = []
for i in range(n_envs):
    n_valid = int(np.argmax(cpu_dones[i]) + 1) if np.any(cpu_dones[i]) else steps
    scores.append(np.sum(cpu_rewards[i, :n_valid]))
scores = np.array(scores)
```

### A2: Alpha-Relative Reward Scale Mismatch (NEW - 126b)
**Critical find by 126b!**

```python
# trainer.py - TASK_118 alpha_relative_reward
max_alpha = jax.vmap(compute_max_alpha)(returns)  # Raw scale ~0.03
rewards = rewards / jnp.maximum(max_alpha, 1e-6)  # Rewards are % scale ~1.0
# BUG: 1.0 / 0.03 = 33x inflation!
```

**Fix**: Scale `max_alpha` by 100 before dividing:
```python
max_alpha_scaled = max_alpha * 100.0
rewards = rewards / jnp.maximum(max_alpha_scaled, 1e-6)
```

### A3: Random Burn State Ordering (NEW - 126b)
```python
# jax_trading_env.py - Burns applied AFTER cumulative_return computed
new_cumulative_return = ...  # Doesn't see random burn
reward = reward - symlog(burn_pct)  # But reward DOES see it
prev_portfolio_value = new_portfolio_value  # Next step double-penalized
```

**Fix**: Move random burn logic BEFORE portfolio value calculations.

---

## BUG_B: Path Solver - DISPUTED

### B1: Signs Variable Unused
**126c, 126h say**: Bug! Signs computed but never used!
**126k says**: NOT a bug! `abs()` correctly captures long/short profit!

**Resolution**: 126k is correct. For a perfect oracle:
- Long positive return: profit = +return = abs(return)
- Short negative return: profit = -(-return) = abs(return)

The `signs` variable IS dead code, but the math is correct.

### B2: Transaction Cost Churning (NEW - 126c, 126k)
**This is the real issue!**

```python
# path_solver.py - default 10bps transaction cost
transaction_cost: float = 0.001  # 10 basis points
```

With minute bars, returns are often < 10bps. High-frequency oracle churns positions → transaction costs exceed gains!

**Fix Options**:
1. Set `transaction_cost=0.0` for theoretical max alpha
2. Expose to config: `self.amax_transaction_cost`
3. Match agent's actual fee structure

### B3: Direction Flip Cost (NEW - 126c)
```python
# Current: only counts asset add/remove
changes = jnp.sum(jnp.abs(new_mask - prev_mask))

# BUG: Long→Short flip on same asset costs 0, should cost 2
```

**Fix**: Track signed positions (-1/0/+1), not just selections (0/1).

---

## BUG_C: Return Scale Mismatch - CONFIRMED

### Sources: 126d, 126i, 126l, my analysis

| Metric | Return Type | Example 8% move |
|--------|-------------|-----------------|
| `market_return` | Linear | 8.00% |
| `agent_return` | Linear | 8.00% |
| `optimal_alpha` | Log | 7.70% |
| `alpha_capture` | Mixed! | Incorrect |

### Why It Matters
- 50% gain: Linear=50%, Log=40.5% (20% relative error!)
- Metrics incomparable for volatile periods
- `alpha_capture = agent_pnl / optimal_alpha` mixes scales!

### Additional Issue (126l, my analysis)
**optimal_alpha computed on DIFFERENT data than market_return!**
- market_return: current episode prices
- optimal_alpha: replay buffer batch (could be old!)

### Fix Options
1. **Standardize on Log Returns** (126l recommends for RL - additive, symmetric)
2. **Standardize on Linear Returns** (easier to interpret)
3. **Compute optimal_alpha on current episode** (required for TASK_125!)

**For TASK_125 imprinting**: Must use Option 3 - oracle on same data as episode.

---

## BUG_D: Numerical Stability - MOSTLY GOOD

### Sources: 126e, 126j, 126m

**Consensus**: Code is well-guarded. A few edge cases:

| Issue | Location | Risk | Fix |
|-------|----------|------|-----|
| Division by zero | `epsilon_greedy.py:39` | LOW | Add epsilon |
| Division by zero | `dirichlet.py:13` | LOW | Add epsilon |
| Limit order shares | `jax_trading_env.py:950` | MED | Add epsilon |
| Short shares calc | `jax_trading_env.py:1012` | MED | Add epsilon |
| unroll_steps=1 | `trainer.py` | LOW | Guard division |
| temp/eps=0 | `path_solver.py` | LOW | Guard params |

---

## Fix Priority for TASK_125 Imprinting

```
1. BUG_C (data source) - Oracle must use episode data
2. BUG_A2 (alpha-relative scale) - If enabled, fix scale
3. BUG_B2 (transaction cost) - Set to 0 or match agent fees
4. BUG_A1 (post-done scores) - Clean metrics
5. BUG_A3 (burn ordering) - If burns enabled
6. BUG_D edge cases - Defensive hardening
```

---

## Implementation Plan

### Phase 1: Critical for Imprinting
1. Add `optimal_alpha` computation in `self_play.py` on episode prices
2. Fix alpha-relative reward scale if enabled
3. Set path solver transaction_cost to match training config

### Phase 2: Metrics Cleanup
4. Mask post-done rewards in scores
5. Fix random burn state ordering
6. Decide log vs linear standardization

### Phase 3: Hardening
7. Add epsilon guards to edge cases
8. Remove dead `signs` variable
9. Document return type conventions

---

## Agent Agreement Matrix

| Finding | 126a | 126b | 126c | 126d | 126e | 126f | 126h | 126i | 126j | 126k | 126l | 126m | Me |
|---------|------|------|------|------|------|------|------|------|------|------|------|------|-----|
| Post-done sum | ✓ | ✓ | | | | ✓ | | | | | | | ✓ |
| Alpha scale bug | | ✓ | | | | | | | | | | | |
| Signs unused | | | ✓ | | | | ✓ | | | ✓* | | | ✓ |
| Signs NOT bug | | | | | | | | | | ✓ | | | ✓ |
| Txn cost issue | | | ✓ | | | | | | | ✓ | | | |
| Log/Linear mix | | | | ✓ | | | | ✓ | | | ✓ | | ✓ |
| Data source mix | | | | | | | | | | | | | ✓ |
| Stability OK | | | | | ✓ | | | | ✓ | | | ✓ | |

*126k: confirmed unused but says NOT a logic bug

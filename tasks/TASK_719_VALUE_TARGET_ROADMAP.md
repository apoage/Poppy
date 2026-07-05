# TASK_719: Value Target Roadmap — From Glass Ball to Self-Play

**Status**: READY (step A)
**Priority**: CRITICAL — Stage B value head not learning from TD(0)
**Depends on**: TASK_718 (PopArt + reward fix, DONE)
**Blocks**: TASK_716g (alpha hunt)

---

## Problem Statement

TASK_718 fixed reward scaling + added PopArt + reset value head + lowered TD discount.
Result: value head no longer saturates at +/-10, but **value_loss plateaus at ~2.9 from E8 onward**.
Policy loss flat at ~1.75 for 100+ epochs. No improvement.

Root cause: **TD(0) with consecutive pairs is too weak a signal.**
- Per-step reward ~0.1, γ=0.95 → td_target ≈ 0.1 + 0.95 * V(s_next)
- Value targets depend on V(s_next) which is itself being learned → moving target
- With only 1-step lookahead, reward information propagates extremely slowly
- Value head learns some structure (2.9 < 6.0 uniform) but can't improve further

## Solution: Three-Step Escalation

Each step is a self-contained change. Run + benchmark after each. Stop when alpha turns positive.

---

### Step A: Oracle Value Targets (glass ball)

**Goal**: Test whether a discriminating value function unblocks policy learning.
**Effort**: ~30 min
**Risk**: Low (data already available, minimal code change)

Use oracle remaining PnL as value targets instead of TD bootstrap. This is what Phase 1 does (`batch['returns'] * value_scale`). The oracle tells us the true future value of each state assuming optimal play.

**Why this might work**: Even though the model can't achieve oracle returns, the *relative ordering* of states is correct. MCTS needs to know "state A is more valuable than state B" to guide search — it doesn't need exact values. If policy_loss starts declining with oracle values, that confirms the bottleneck was value discrimination, not the value magnitude.

**Why this might not be enough**: The value head learns unreachable targets. Over time, MCTS overestimates all states equally → same problem returns. But it should give enough signal for early learning.

#### Implementation

**File**: `train_phase2.py`, `ConsecutivePairGenerator.generate_batch()`

Add oracle remaining PnL to batch output:
```python
# In generate_batch(), after computing step_pnl:
# Oracle remaining PnL (future value from this decision point)
oracle_cum_pnl = slot['oracle_cum_pnl']  # (T-1,) cumulative
remaining_pnl = oracle_cum_pnl[-1] - oracle_cum_pnl[decision_indices]  # (n,)
all_oracle_values.append(jnp.array(remaining_pnl))

# In returned dict:
'oracle_values': jnp.concatenate(all_oracle_values)[:batch_size],
```

**File**: `train_phase2.py`, `make_stage_b_train_step()`

Replace TD target with oracle value target:
```python
# Option: pure oracle value target (instead of TD bootstrap)
# oracle_values = remaining PnL from path solver, scale to support range
oracle_scaled = oracle_values * config.value_scale  # value_scale=2.0 from Phase 1
oracle_scaled = jnp.clip(oracle_scaled, -10.0, 10.0)

# PopArt normalize
td_target_norm = (oracle_scaled - popart_mean) / jnp.maximum(popart_std, 1e-8)
td_target_norm = jnp.clip(td_target_norm, -10.0, 10.0)
```

**File**: `phase2_config.py`

Add config:
```python
value_target_mode: str = "oracle"  # "oracle", "td0", "nstep", "hybrid"
```

**Gate**: Run 50 epochs. If policy_loss < 1.5 AND decreasing → oracle values are working. Proceed to benchmark.

---

### Step B: N-Step TD Returns (self-play, moderate signal)

**Goal**: Replace TD(0) with n-step returns for stronger reward signal without oracle dependency.
**Effort**: ~2 hours
**Risk**: Medium (ConsecutivePairGenerator needs window sampling)
**Depends on**: Step A results (only needed if oracle values don't produce alpha)

Instead of 1-step bootstrap, sample windows of N consecutive steps and compute:
```
R_t = Σ_{k=0}^{N-1} γ^k * r_{t+k} + γ^N * V(s_{t+N})
```

With N=20 and γ=0.95: the return accumulates ~20 actual reward steps before bootstrapping. This gives 20x more reward signal than TD(0) while being grounded in the model's own value predictions.

#### Implementation

**File**: `train_phase2.py`, `ConsecutivePairGenerator`

- Extend to sample windows of N steps (not just pairs)
- For each window: compute N per-step rewards, encode obs at t and t+N
- Return reward sequence + final obs for bootstrap

**File**: `train_phase2.py`, `make_stage_b_train_step()`

- Compute discounted n-step return inside JIT
- Bootstrap with V(s_{t+N}) from stopped-gradient forward pass

**File**: `phase2_config.py`

```python
nstep_horizon: int = 20  # N for n-step returns
```

**Gate**: Run 100 epochs. Value_loss should decrease below 2.0. td_target_std should increase (more variance = more discrimination).

---

### Step C: Multi-Step Rollouts with GAE (Poppy's approach)

**Goal**: Ground value in actual environment feedback using the model's own policy.
**Effort**: ~1 day
**Risk**: High complexity (new rollout infrastructure)
**Depends on**: Step B (only if n-step still insufficient)

This is what Poppy does:
1. Run episodes in the trading environment using MCTS-selected actions
2. Collect actual per-step rewards from position changes
3. Compute GAE returns (λ=0.95, γ=0.997) from real experience
4. Train value head on GAE targets

The value head learns what the model CAN actually achieve, not what the oracle achieves.

#### Implementation

**New file**: `rollout_generator.py` or extend ConsecutivePairGenerator

- Step the trading environment for N steps using MCTS actions
- Track positions, compute actual PnL from position changes
- Use `mcts_trader/jax_agent/returns.py:compute_gae_returns()` (already exists)
- Return (observations, actions, gae_returns) tuples

**Key difference from Poppy**: Foxglove doesn't have vectorized trading environments. Would need to either:
a) Build a lightweight JAX env for position tracking + PnL computation
b) Use the existing HistoricalMarket env but batch it manually

This is the architecturally correct solution but the largest refactor.

**Gate**: Value_loss monotonically decreasing. Policy_loss < 1.0. MCTS root_value varies across batch (std > 0.5). Benchmark shows positive alpha.

---

## Implementation Order

```
Step A (oracle values)     → 30 min, test: does value discrimination unblock policy?
  IF policy improves → benchmark → may already have alpha
  IF policy still flat → Step B

Step B (n-step TD)         → 2 hours, test: self-play value signal
  IF value_loss decreasing → let run → benchmark
  IF still plateaued → Step C

Step C (GAE rollouts)      → 1 day, Poppy's proven approach
  Nuclear option — if we get here, we're building proper RL infrastructure
```

**Expected outcome**: Step A is most likely to unblock. Oracle values worked in Phase 1 (value_loss converged to ~0.5). The question is whether MCTS + oracle values → useful policy distillation. If yes, Step B/C become optimization rather than necessity.

---

## Data Available in SegmentPool Slots

| Field | Shape | Content |
|-------|-------|---------|
| `oracle_cum_pnl` | (T-1,) | Cumulative PnL from oracle path |
| `oracle_selections` | (T-1, n_assets) | Binary mask of selected assets |
| `returns` | (T-1, n_assets) | Per-step log returns per asset |
| `policy_targets` | (T-1, n_assets*5) | Oracle policy distribution |

Oracle remaining PnL = `oracle_cum_pnl[-1] - oracle_cum_pnl[t]` (what Phase 1 uses).

---

## Files Modified Per Step

| Step | Files | Changes |
|------|-------|---------|
| A | `train_phase2.py`, `phase2_config.py` | Add oracle values to batch, replace TD target, add `value_target_mode` config |
| B | `train_phase2.py`, `phase2_config.py` | N-step window sampling in generator, n-step return computation, `nstep_horizon` config |
| C | New `rollout_generator.py`, `train_phase2.py`, `phase2_config.py` | Episode rollout infrastructure, GAE computation, new training loop integration |

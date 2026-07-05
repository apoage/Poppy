# TASK_718: PopArt Value Normalization + Reward Scaling Fix

**Status**: READY
**Priority**: CRITICAL — Stage B training stalls without this
**Depends on**: TASK_716f (ExIt loop, DONE)
**Blocks**: TASK_716g (alpha hunt)

---

## Problem Statement

Stage B ExIt training stalls after ~17 epochs. Two compounding root causes:

### 1. Reward Signal Vanishes in TD Targets

Current code (`train_phase2.py:670`):
```python
r_step = rewards_dynamics / 100.0  # Descale from REWARD_SCALE=100
td_target = r_step + config_discount * v_next
```

Per-step PnL is ~0.001. After `/100.0` descaling: **0.00001**. Value support bin width is **0.05**. The reward is 5000x smaller than a single bin — it's noise. TD target collapses to `td ≈ 0.997 * V(s_next)`, which is self-consistency training with zero reward signal.

### 2. Value Head Saturated from Phase 1

Phase 1 trained value on `oracle_returns * value_scale(2.0)` → effective range [0, 10]. The value head predicts ~10.0 for all states. MCTS root_value is constant → search can't discriminate states by value. The tree search is guided only by the policy prior.

### Evidence

```
E17: policy=1.41 value=0.007 consistency=1.41 grad=0.55  ← collapsed
E22: policy=1.40 value=0.004 consistency=1.28 grad=0.46  ← stuck
E140: policy=1.98 value=1.48 consistency=1.38 grad=11.6  ← recovered but oscillating
E144: policy=1.74 value=1.74 consistency=1.38 grad=16.1  ← no improvement since E5
```

Value loss oscillates between 0.0 (self-consistent) and 2-3 (random perturbation). No learning.

---

## Solution: Two Changes

### Change A: Fix Reward Scaling in TD Targets

**Don't descale rewards for TD computation.** The rewards are already scaled to [-10, 10] support range (clipped in `generate_batch`). The `/100.0` descaling was incorrect — it converts support-range values back to raw PnL scale, which is invisible to the categorical bins.

**Current** (broken):
```python
r_step = rewards_dynamics / 100.0  # 0.00001 — invisible to bins
td_target = r_step + config_discount * v_next
```

**Fixed**:
```python
r_step = rewards_dynamics  # Already in [-10, 10] support range
td_target = r_step + config_discount * v_next
td_target = jnp.clip(td_target, -10.0, 10.0)
```

**Wait — verify this is correct.** Check what `rewards_dynamics` actually contains:
- `generate_batch` computes: `step_pnl * REWARD_SCALE` then clips to [-10, 10]
- So `rewards_dynamics` ∈ [-10, 10] — already in value support range
- The `/100` was meant to match MCTS `_recurrent_fn` which descales for tree search
- But for TD value targets, we want support-range values, not raw PnL

**Note**: The dynamics reward loss (`reward_loss`) correctly uses `rewards_dynamics` directly (it's trained with REWARD_SCALE). Only the TD value target computation was wrong.

### Change B: Add PopArt Normalization

Port PopArt from Poppy (`mcts_trader/jax_agent/popart.py`) into Phase 2 training. PopArt adaptively normalizes value targets to mean=0, std=1, centering them in the [-10, 10] support range regardless of the raw value distribution.

---

## Implementation Spec

### Subtask 718a: Fix Reward Scaling (5 min)

**File**: `train_phase2.py`, inside `make_stage_b_train_step`

Change the TD target computation (currently line ~670):

```python
# BEFORE (broken):
r_step = rewards_dynamics / 100.0

# AFTER (fixed):
# rewards_dynamics is already in [-10, 10] support range (REWARD_SCALE applied + clipped
# in generate_batch). Don't descale — TD targets should be in support range.
# The /100 descaling is for MCTS tree search (mctx_search.py:110), not for value training.
r_step = rewards_dynamics
```

**Gate**: Run 5-epoch smoke test. Value loss should be non-trivial (>0.5) on E0 instead of ~0.0.

### Subtask 718b: Add PopArt State to Phase 2 (15 min)

**File**: `phase2_config.py`

Add config fields:
```python
# PopArt normalization (TASK_718)
use_popart: bool = True
popart_beta: float = 0.0003  # EMA decay — faster than Poppy's 0.0001
                               # because Phase 2 has fewer total steps
```

**File**: `train_phase2.py`

1. Import PopArt:
```python
from mcts_trader.jax_agent.popart import (
    PopArtState, init_popart_state, update_statistics,
    normalize_targets, denormalize_values,
)
```

2. Initialize PopArt state in `train_stage_b()` before the training loop:
```python
if config.use_popart:
    popart_state = init_popart_state()
    logger.info("PopArt normalization enabled (beta=%.4f)", config.popart_beta)
```

3. Save/restore PopArt state in checkpoints:
```python
# In save_checkpoint:
if popart_state is not None:
    ckpt['popart_state'] = {
        'mean': float(popart_state.mean),
        'var': float(popart_state.var),
        'count': int(popart_state.count),
    }
```

### Subtask 718c: Integrate PopArt into Value Loss (30 min)

**File**: `train_phase2.py`, inside `make_stage_b_train_step`

This is the core change. The value loss must:
1. Compute raw TD target (in original value scale)
2. Update PopArt running statistics with the raw target
3. Normalize the target to mean=0, std=1
4. Compute categorical CE on the normalized target
5. At MCTS time, denormalize the value head output for tree search

**PopArt state must be passed through JIT.** Since `update_statistics` modifies state, it can't be in a pure JIT function. Two approaches:

**Approach (recommended): PopArt outside JIT**

The train step JIT function returns raw TD targets alongside metrics. PopArt update happens in the Python training loop:

```python
@jax.jit
def stage_b_train_step(params, opt_state, obs, obs_t1, actions_dynamics,
                        rewards_dynamics, mcts_visits, oracle_targets,
                        bc_weight, popart_mean, popart_std):
    # ... (existing code for policy, dynamics, bc losses)

    # Value: compute raw TD target
    r_step = rewards_dynamics  # Already in [-10, 10] (fix from 718a)
    v_next_logits = out_t1['value']
    v_next_raw = jnp.sum(jax.nn.softmax(v_next_logits, axis=-1) * support, axis=-1)
    td_target_raw = r_step + config_discount * v_next_raw  # Raw scale

    # Normalize with PopArt stats (passed as args, not state)
    td_target_norm = (td_target_raw - popart_mean) / jnp.maximum(popart_std, 1e-8)
    td_target_norm = jnp.clip(td_target_norm, -10.0, 10.0)

    # Categorical CE on normalized target
    value_target = scalar_to_categorical(td_target_norm)
    value_loss = optax.softmax_cross_entropy(value_logits, value_target).mean()

    # Return raw targets for PopArt update (outside JIT)
    metrics['td_target_mean'] = jnp.mean(td_target_raw)
    metrics['td_target_std'] = jnp.std(td_target_raw)
    metrics['td_targets_raw'] = td_target_raw  # For PopArt update

    # ... rest of loss combination
```

In the training loop:
```python
params, opt_state, metrics = train_step_fn(
    params, opt_state, obs, obs_t1, actions_dyn, rewards_dyn,
    visit_counts, oracle_targets, bc_weight_jax,
    jnp.float32(popart_state.mean), jnp.sqrt(popart_state.var + 1e-8),
)

# Update PopArt statistics (outside JIT)
if config.use_popart:
    raw_targets = metrics.pop('td_targets_raw')
    popart_state = update_statistics(popart_state, raw_targets, config.popart_beta)
```

**Key detail**: Pass `popart_mean` and `popart_std` as JIT args (they're scalars, cheap to pass). This avoids putting mutable state inside JIT.

**BUG-003/004 from Poppy**: Do NOT apply PopArt weight correction to categorical value heads. The softmax nonlinearity breaks the linear correction formula. Only normalize targets.

### Subtask 718d: Denormalize Value for MCTS (15 min)

MCTS needs raw-scale values for tree search (Q-values must be comparable to rewards). The value head now outputs normalized predictions, so we must denormalize before MCTS.

**File**: `train_phase2.py`, in `train_stage_b()` where MCTS is called

```python
# Before MCTS search, wrap model to denormalize value output
if config.use_popart:
    # Create a wrapper that denormalizes value predictions for MCTS
    popart_mean_val = float(popart_state.mean)
    popart_std_val = float(jnp.sqrt(popart_state.var + 1e-8))

    # The MuZeroMCTXSearch._categorical_to_scalar returns raw logits→scalar
    # We need to denormalize: raw_value = normalized_value * std + mean
    # Set the search's value_scale and value_shift for this epoch
    search.value_scale = popart_std_val
    search.value_shift = popart_mean_val
```

Check `mctx_search.py:138-140`:
```python
scale = getattr(self, 'value_scale', 1.0)
shift = getattr(self, 'value_shift', 0.0)
return raw_value * scale + shift
```

This already supports `value_scale` and `value_shift` attributes! Just set them on the search object before each epoch. When PopArt is off, they default to 1.0/0.0 (identity).

**Important**: This must happen BEFORE the MCTS search call in each step, since popart_state changes every step. But since beta is small (0.0003), updating per-epoch is sufficient:

```python
# At start of each epoch (not every step — too slow):
if config.use_popart:
    search.value_scale = float(jnp.sqrt(popart_state.var + 1e-8))
    search.value_shift = float(popart_state.mean)
```

### Subtask 718e: Hybrid Value Loss (optional, 20 min)

Poppy uses 80% MSE + 20% categorical for value loss. This prevents the value head from collapsing to a single bin. Currently Phase 2 uses pure categorical CE.

**Add to `make_stage_b_train_step`:**

```python
# Hybrid value loss (like Poppy trainer.py:673-741)
# MSE component: direct scalar prediction error
v_pred = jnp.sum(jax.nn.softmax(value_logits, axis=-1) * support, axis=-1)
mse_loss = jnp.mean((v_pred - td_target_norm) ** 2)

# Categorical component: distributional matching
cat_target = scalar_to_categorical(td_target_norm)
cat_loss = optax.softmax_cross_entropy(value_logits, cat_target).mean()

# Combined (Poppy uses 0.8/0.2 split)
value_loss = 0.8 * mse_loss + 0.2 * cat_loss
```

**Gate**: Optional — try pure categorical first (718a-d), add hybrid only if value loss is still noisy.

### Subtask 718f: Label Smoothing for Value Targets (optional, 5 min)

Poppy applies label smoothing (α=0.1) to categorical value targets:
```python
target = (1 - alpha) * target + alpha * uniform(401)
```

This prevents the value head from becoming overconfident about exact bin predictions. Add to `scalar_to_categorical` function.

---

## Files Modified

| File | Changes |
|------|---------|
| `train_phase2.py` | Fix reward scaling (718a), add PopArt state management (718b), integrate into value loss (718c), denormalize for MCTS (718d), optional hybrid loss (718e) |
| `phase2_config.py` | Add `use_popart`, `popart_beta` fields (718b) |

**No changes to**: `popart.py` (reuse as-is), `mctx_search.py` (already has value_scale/shift), `easy_trainer.py`, `muzero_model.py`

---

## Implementation Order

```
718a (reward fix)     → 5 min, smoke test: value_loss > 0.5 on E0
718b (PopArt state)   → 15 min, just plumbing
718c (PopArt in loss) → 30 min, core change, smoke test: td_target_mean converging
718d (denorm MCTS)    → 15 min, verify root_value varies across batch
718e (hybrid loss)    → optional, add if value still noisy after 718a-d
718f (label smooth)   → optional, add if value overconfident
```

Total: ~1 hour for core (718a-d), +25 min for optional (718e-f).

---

## Gate Criteria

After implementing 718a-d, run Stage B for 50 epochs:

| Metric | Target | Rationale |
|--------|--------|-----------|
| value_loss | Decreasing, not stuck at 0 | Real reward signal flowing |
| td_target_mean | Not saturated at 10.0 | PopArt centering the distribution |
| td_target_std | Reasonable (0.5-5.0) | Values spread across support |
| MCTS root_value std | > 0.1 | Search can discriminate states |
| policy_loss | Decreasing below 2.0 | MCTS providing useful targets |
| total_loss | Monotonic decrease | Overall convergence |

If gate passes → continue to full 1000-epoch run → benchmark → TASK_716g (alpha hunt).

---

## Reference: Poppy's PopArt Implementation

**File**: `mcts_trader/jax_agent/popart.py`

```python
# PopArtState fields:
#   mean: jnp.ndarray      — running mean (scalar)
#   var: jnp.ndarray       — running variance (scalar)
#   count: int             — samples seen
#   mean_old: jnp.ndarray  — previous mean (for weight correction)
#   std_old: jnp.ndarray   — previous std (for weight correction)

# Key functions:
#   init_popart_state() → PopArtState
#   update_statistics(state, targets, beta=0.0001) → PopArtState
#     - EMA: new_mean = (1-β)*old_mean + β*batch_mean
#     - EMA: new_var = (1-β)*old_var + β*batch_var
#     - Variance floor: max(var, 1e-8)
#   normalize_targets(targets, state) → normalized
#     - (targets - mean) / sqrt(var + 1e-8)
#   denormalize_values(values, state) → raw
#     - values * sqrt(var + 1e-8) + mean
```

**BUG-003/004**: Do NOT apply `compute_weight_correction` / `apply_weight_correction_flax` to categorical heads. Softmax nonlinearity breaks the linear correction formula. Poppy explicitly skips this (trainer.py:1700-1706).

---

## Future: GAE Returns (TASK_719 candidate)

Poppy doesn't use TD(0) for value targets at all. It runs multi-step episodes via self-play, collects actual rewards, then computes **GAE/TD(lambda)** returns (`returns.py:compute_gae_returns`, lambda=0.95, discount=0.997). MCTS is used **only for action selection**, not value targets.

This is a fundamentally different (and better) approach: value targets come from real environment returns, not bootstrap estimates. But it requires episode rollouts in the trading environment, which Foxglove Phase 2 doesn't currently do (it samples consecutive pairs from SegmentPool).

**If 718a-d doesn't produce alpha+**, the next step would be:
- Run multi-step rollouts (e.g., 20-50 steps) using MCTS-selected actions in the environment
- Collect per-step rewards from actual position changes
- Compute GAE returns as value targets
- This gives the value head a real signal grounded in environment feedback, not self-referential bootstrapping

This would be a larger refactor (new rollout infrastructure) but aligns Foxglove with Poppy's proven approach. File as TASK_719 if needed.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| PopArt cold start (mean=0, var=1) | LOW | First epoch targets already reasonable; EMA adapts in ~100 steps |
| Reward scale too large (no /100) | MEDIUM | Clip td_target to [-10, 10]; monitor for overflow |
| MCTS denorm drift | LOW | Update search scale/shift per epoch (beta is small) |
| JIT recompilation from popart args | LOW | popart_mean/std are scalars, won't trigger recompilation |

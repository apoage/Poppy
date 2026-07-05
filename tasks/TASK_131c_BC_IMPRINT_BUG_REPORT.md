# TASK_131c: BC/Imprint System Bug Report

**Priority**: CRITICAL
**Status**: ANALYSIS COMPLETE
**Date**: 2026-02-01
**Auditor**: Claude

---

## Executive Summary

The Causal Expert Imprinting (TASK_125) BC loss system is **completely broken**. Despite `--imprint` being enabled:
- `bc_loss ≈ 0.1` (negligible, near-zero impact)
- `action_agreement` not printing (metric never computed)
- `demo_count = 50` (matches parallel_envs, not accumulating)
- Model not learning from expert demonstrations

**Root Cause**: Multiple cascading failures prevent BC training from ever executing.

---

## Bug Analysis

### BUG-A: expert_batch_size > demo_count (CRITICAL)

**Location**: `mcts_trader/training/setup.py:545`

**Problem**: `expert_batch_size=128` is hardcoded, but `sample_batch()` returns `None` when insufficient samples:

```python
# expert_buffer.py:318-320
def sample_batch(self, batch_size=1024, demo_ratio=0.125):
    total_available = self.demo_count + self.agent_count  # 50 + 0 = 50
    if total_available < batch_size:  # 50 < 128 → True!
        return None  # BC training COMPLETELY SKIPPED
```

**Data Flow**:
```
trainer.train():1672-1678
├─ expert_batch = expert_buffer.sample_batch(batch_size=128)
│   └─ Returns None (50 < 128)
├─ if expert_batch is not None:  # False!
│       batch['expert_batch'] = expert_batch  # Never executes
└─ BC loss block never entered → bc_loss = 0, action_agreement = 0
```

**Impact**:
- BC training 100% blocked until 128+ demos accumulate
- With negative alpha episodes, demos never accumulate past 50
- Vicious cycle: no BC → bad policy → negative alpha → no demos → no BC

**Suggested Fix**:
```python
# Option 1: Reduce batch size in setup.py:545
expert_batch_size=min(64, getattr(config, 'expert_batch_size', 64)),

# Option 2: Make sample_batch handle small buffers
def sample_batch(self, batch_size=1024, demo_ratio=0.125):
    total_available = self.demo_count + self.agent_count
    if total_available == 0:
        return None
    # Adjust batch_size to available samples
    effective_batch_size = min(batch_size, total_available)
    ...
```

---

### BUG-B: episode_counter Never Increments (HIGH)

**Location**: `train_jax_fast.py:583-683`

**Problem**: Counter increment is AFTER the early return:

```python
def on_episode_complete_callback(result: dict, score: float):
    ...
    should_store = agent_alpha > 0

    # Debug print (line 587-588)
    if episode_counter[0] % 10 == 0:
        print(f"[TASK_125-DEBUG] Episode {episode_counter[0]}: ...")

    if not should_store:
        return  # ← Returns here!

    # ... add_demo calls ...

    episode_counter[0] += 1  # ← Never reached if alpha <= 0!
```

**Impact**:
- If `agent_alpha <= 0` consistently, counter stays at 0
- Debug prints show "Episode 0: ..." repeatedly (0 % 10 == 0)
- Misleading diagnostics - appears as if only 1 episode ran
- Cannot track how many episodes were processed vs stored

**Suggested Fix**:
```python
def on_episode_complete_callback(result: dict, score: float):
    ...
    should_store = agent_alpha > 0

    # Count ALL episodes processed (not just stored)
    episode_counter[0] += 1  # ← Move here

    if episode_counter[0] % 10 == 0:
        print(f"[TASK_125-DEBUG] Episode {episode_counter[0]}: "
              f"alpha={agent_alpha:.2f}%, store={should_store}, "
              f"demos={expert_buffer.demo_count}")

    if not should_store:
        return
```

---

### BUG-C: agent_count = 0 (MEDIUM)

**Location**: `train_jax_fast.py:670-681` vs `expert_buffer.py:add_agent`

**Problem**: Callback only calls `add_demo()`, never `add_agent()`:

```python
# train_jax_fast.py:672-681
for step in range(n_steps):
    expert_buffer.add_demo(  # Only demos!
        episode_id=episode_id,
        step=step,
        action=int(actions[step]),
        optimal_action=optimal_action,
        ...
    )
# add_agent() is never called anywhere
```

**Impact**:
- Agent zone (500k capacity) is completely unused
- `agent_count` stays 0
- Buffer only contains demo experiences
- Mixed sampling (demo_ratio=0.125) can't work as designed

**Design Question**: Was `add_agent()` intended to be called from self-play for ALL experiences (not just positive-alpha ones)?

---

### BUG-D: Early Return on episode_optimal_alpha <= 0 (MEDIUM)

**Location**: `train_jax_fast.py:574`

**Problem**: Episodes are filtered if optimal path computation fails:

```python
episode_optimal_alpha = result.get('episode_optimal_alpha', 0.0)
agent_alpha = result.get('alpha', 0.0)
prices = result.get('prices')

if prices is None or episode_optimal_alpha <= 0:
    return  # Exits if optimal path threw exception!
```

**Cascade**:
```
self_play.py:819-824
try:
    optimal_path = compute_optimal_path(...)
    episode_optimal_alpha = float(optimal_path.total_alpha) * 100.0
except Exception:
    episode_optimal_alpha = 0.0  # ← Silently swallowed!
```

**Impact**:
- If `compute_optimal_path` ever fails, ALL episodes are filtered
- No visibility into exceptions (silently caught)
- Even good agent episodes (positive alpha) are discarded

**Suggested Fix**:
```python
# Don't require optimal_alpha for storage decision
if prices is None:
    return

# Compute confidence differently when optimal path unavailable
if episode_optimal_alpha > 0:
    confidence = max(0.0, agent_alpha / max(episode_optimal_alpha, 1e-6))
else:
    # Fallback: use agent_alpha directly as confidence proxy
    confidence = max(0.0, min(1.0, agent_alpha / 10.0))  # Normalize to [0,1]

should_store = agent_alpha > 0  # Keep this filter
```

---

### BUG-E: action_agreement Not Printing (SYMPTOM)

**Root Cause**: Symptom of BUG-A

The metric `expert_action_agreement` is computed inside the BC loss block:

```python
# trainer.py:1491-1492
expert_agreement = (policy_actions == expert_actions).astype(jnp.float32)
metrics['expert_action_agreement'] = jnp.mean(expert_agreement)
```

This block only executes when `'expert_batch' in batch` is True (line 1412). Since `sample_batch` returns `None`, `expert_batch` is never added to batch.

**Result**:
- `metrics['expert_action_agreement']` = 0.0 (from else branch, line 1507)
- Not a separate bug, just consequence of BUG-A

---

## Complete Data Flow Trace

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. run_self_play_jax() - self_play.py:562                        │
│    └─ Per-env callback: on_episode_complete(result, scores[i])   │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 2. on_episode_complete_callback() - train_jax_fast.py:564        │
│    ├─ Check: prices is None? → return (no prices collected)     │
│    ├─ Check: episode_optimal_alpha <= 0? → return (BUG-D)        │
│    ├─ Compute: confidence = agent_alpha / optimal_alpha          │
│    ├─ Check: agent_alpha <= 0? → return (not market-beating)     │
│    │   └─ episode_counter[0] never increments (BUG-B)            │
│    ├─ price_cache.add_episode(prices_ohlcv)                      │
│    ├─ compute_optimal_path() → optimal actions per step          │
│    ├─ Loop: expert_buffer.add_demo() × n_steps                   │
│    │   └─ add_agent() never called (BUG-C)                       │
│    └─ episode_counter[0] += 1                                    │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 3. trainer.train() - trainer.py:1623                             │
│    └─ Per iteration (50x):                                       │
│        ├─ batch = replay_buffer.sample_batch()                   │
│        ├─ expert_batch = expert_buffer.sample_batch(             │
│        │      batch_size=128, demo_ratio=0.125)                  │
│        │   └─ if demo_count + agent_count < 128:                 │
│        │          return None  ← (BUG-A: 50 < 128)               │
│        ├─ if expert_batch is not None:  ← False!                 │
│        │      batch['expert_batch'] = expert_batch               │
│        └─ train_step_fn() → _loss_fn()                           │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 4. _loss_fn() - trainer.py:1407-1507                             │
│    ├─ if bc_config.enabled and 'expert_batch' in batch:          │
│    │   └─ False! (expert_batch never added due to BUG-A)         │
│    │   ├─ expert_obs = expert_batch['observations']              │
│    │   ├─ expert_actions = expert_batch['optimal_actions']       │
│    │   ├─ Run inference, compute BC loss                         │
│    │   ├─ Compute expert_action_agreement                        │
│    │   └─ Add to total_loss                                      │
│    └─ else:                                                      │
│        ├─ bc_loss = 0.0                                          │
│        ├─ bc_weight = 0.0                                        │
│        ├─ expert_batch_size = 0                                  │
│        └─ expert_action_agreement = 0.0  ← What we see!          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Metrics Observed vs Expected

| Metric | Observed | Expected | Status |
|--------|----------|----------|--------|
| `bc_loss` | ~0.1 | 0.5-2.0 | BROKEN |
| `action_agreement` | Not printing | 0.1-0.8 | BROKEN |
| `demo_count` | 50 (= parallel_envs) | Accumulating | BROKEN |
| `agent_count` | 0 | Growing | BROKEN |
| `expert_batch_size` | 0 (in metrics) | 128 | BROKEN |

---

## Recommended Fixes (Priority Order)

### 1. Fix BUG-A: Make expert_batch_size Dynamic

```python
# mcts_trader/training/setup.py:545
expert_batch_size=getattr(config, 'expert_batch_size', 64),  # Reduce default
```

AND/OR in `expert_buffer.py:sample_batch`:

```python
def sample_batch(self, batch_size=1024, demo_ratio=0.125):
    total_available = self.demo_count + self.agent_count
    if total_available == 0:
        return None

    # Clamp batch_size to available samples (allow small batches)
    effective_batch_size = min(batch_size, max(total_available, 32))

    # Rest of method uses effective_batch_size...
```

### 2. Fix BUG-B: Move Counter Increment

```python
# train_jax_fast.py - in on_episode_complete_callback
should_store = agent_alpha > 0
episode_counter[0] += 1  # Count ALL episodes

if episode_counter[0] % 10 == 0:
    print(f"[TASK_125-DEBUG] Episode {episode_counter[0]}: "
          f"alpha={agent_alpha:.2f}%, optimal={episode_optimal_alpha:.2f}%, "
          f"store={should_store}, demos={expert_buffer.demo_count}")

if not should_store:
    return
```

### 3. Add Diagnostic Logging

```python
# expert_buffer.py:sample_batch - add after None return
if total_available < batch_size:
    if self.update_count % 100 == 0:  # Don't spam
        print(f"[EXPERT] Insufficient samples: {total_available} < {batch_size} "
              f"(demos={self.demo_count}, agents={self.agent_count})")
    return None
```

### 4. Consider Relaxing Filters

```python
# train_jax_fast.py - less strict filtering
should_store = agent_alpha > -5.0  # Store slightly negative too
# OR
should_store = True  # Store all, let confidence weight handle quality
```

---

## Testing Checklist

After fixes, verify:

- [ ] `demo_count` increases each epoch (not stuck at parallel_envs)
- [ ] `bc_loss` > 0.1 (meaningful gradient signal)
- [ ] `action_agreement` prints in training logs
- [ ] `expert_batch_size` in metrics matches config (not 0)
- [ ] Debug prints show episode counter incrementing

---

## Files to Modify

| File | Changes |
|------|---------|
| `mcts_trader/training/setup.py:545` | Reduce/dynamic `expert_batch_size` |
| `mcts_trader/jax_agent/expert_buffer.py:318-320` | Handle small buffers gracefully |
| `train_jax_fast.py:587` | Move `episode_counter` increment |
| `train_jax_fast.py:574` | Relax `episode_optimal_alpha` filter |

---

**Next Step**: Implement fixes and run validation test with `--imprint` flag.

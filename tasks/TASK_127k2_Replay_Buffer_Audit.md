# TASK_127k2: Replay Buffer and Data Sampling Audit

## Executive Summary

This audit examines the replay buffer sampling mechanisms for bias and correctness. **Several design choices and potential implementation bugs** could lead to sampling bias, stale data usage, and inefficient training.

**Risk Level: MEDIUM** - No critical crashes, but sampling biases may affect training efficiency and convergence.

---

## Architecture Overview

```
Two-Buffer System:
├── JAXReplayBuffer (main buffer)
│   ├── Stores: observations, actions, rewards, values, policy_probs
│   ├── Capacity: ~500k trajectories
│   └── Sampling: Recent-biased (heuristic_window)
│
└── UnifiedIndexBuffer (expert buffer, TASK_125)
    ├── Demo zone: Protected (50k samples, never overwritten)
    ├── Agent zone: FIFO (500k samples, circular)
    ├── PER: Priority trees for sampling
    └── Index storage: 32 bytes/sample (8000x memory reduction)
```

---

## Critical Findings

### 🔴 CRITICAL: Expert Buffer Samples Without Priority Trees for Demos

**Location:** `expert_buffer.py:484-508` (`_sample_from_range`)

**Issue:** The expert buffer uses a **different sampling method** for demo vs agent zones:
- **Agent zone:** Uses proper PER via `_sample_proportional` (sum tree traversal)
- **Demo zone:** Uses `_sample_from_range` which does **NOT use the priority trees!**

**Evidence:**
```python
def _sample_from_range(self, n: int, start: int, end: int) -> list:
    """Sample n indices from [start, end) range using priorities."""
    # ...
    # Get priorities for this range
    priorities = self.indices[start:end]['priority'].astype(np.float64)
    priorities = np.maximum(priorities, 1e-10) ** self.per_alpha
    
    total = priorities.sum()
    if total <= 0:
        # Uniform sampling fallback
        return list(np.random.choice(range_indices, min(n, range_size), replace=False))
    
    probs = priorities / total
    
    # Sample with replacement if n > range_size
    replace = n > range_size
    sampled = np.random.choice(range_indices, n, p=probs, replace=replace)
    return list(sampled)
```

**Problem:** This method computes probabilities correctly but uses `np.random.choice` which is:
1. **O(n) per sample** (vs O(log n) for tree traversal)
2. **Not using the segment trees** that were updated with `_update_priority`
3. **Different sampling distribution** than `_sample_proportional`

**Impact:**
- Demo samples are drawn with different statistics than agent samples
- Importance sampling weights computed later assume tree-based sampling
- **Bias in demo vs agent sampling ratios**

**Fix:** Use `_sample_proportional` for both zones:
```python
def _sample_from_range(self, n: int, start: int, end: int) -> list:
    """Sample n indices from [start, end) using priority trees."""
    indices = []
    attempts = 0
    max_attempts = n * 10
    
    while len(indices) < n and attempts < max_attempts:
        # Use tree-based sampling but filter to range
        idx = self._sample_proportional(1)[0]
        if start <= idx < end and idx not in indices:
            indices.append(idx)
        attempts += 1
    
    return indices
```

---

### 🟡 HIGH: Demo Priority Boost Applied Inconsistently

**Location:** `expert_buffer.py:160-162` (add_demo) vs `expert_buffer.py:518-519` (update_priorities)

**Issue:** Demo priority boost is applied differently at add vs update time:

```python
# At add time (line 161):
priority = confidence + self.demo_priority_boost

# At update time (line 518-519):
if self.indices[idx]['is_demo']:
    priority += self.demo_priority_boost
```

**Problem:** 
- At add: `priority = confidence + boost`
- At update: `priority = |td_error| + 1e-6 + boost`

**But:** `confidence` is 0-1, while `|td_error|` can be much larger (e.g., 10+).

**Impact:** After first TD error update, demo priority may be dominated by TD error rather than confidence, changing the intended sampling distribution.

**Fix:** Maintain separate priority components or ensure consistent formula:
```python
def _compute_priority(self, idx: int, td_error: float = None) -> float:
    """Compute unified priority."""
    is_demo = self.indices[idx]['is_demo']
    
    if is_demo:
        # Demos: confidence-based with small TD error component
        confidence = self.indices[idx]['confidence']
        base_priority = confidence
        if td_error is not None:
            # Small TD influence (10% weight)
            base_priority = 0.9 * confidence + 0.1 * (abs(td_error) + 1e-6)
        return base_priority + self.demo_priority_boost
    else:
        # Agent: TD error based
        if td_error is None:
            td_error = self.indices[idx]['td_error']
        return abs(td_error) + 1e-6
```

---

### 🟡 MEDIUM: Main Replay Buffer Uses Recent-Biased Sampling (Not Uniform)

**Location:** `replay_buffer.py:255-267`

**Issue:** The main replay buffer samples from a **heuristic window** of recent trajectories:

```python
# Heuristic: Start with capacity // 500 recent trajectories
heuristic_window = max(2, self.capacity // 500)
heuristic_window = min(heuristic_window, self.num_trajectories)

# Sample from recent trajectories (logical indices)
end_logical = self.num_trajectories
start_logical = max(0, end_logical - heuristic_window)

logical_traj_indices = jax.random.randint(
    sample_rng, (self.batch_size,), start_logical, end_logical
)
```

**Problem:** 
- Only samples from the most recent ~0.2% of trajectories
- Old but valuable experiences are never sampled
- **Not true experience replay** - more like "recent experience replay"

**Example:**
- Capacity: 500k trajectories
- Heuristic window: 500k // 500 = 1000 trajectories
- Only samples from most recent 1000 trajectories
- Trajectories 1001-500000 are effectively ignored

**Impact:**
- Catastrophic forgetting of old strategies
- Overfitting to recent market regimes
- Poor sample efficiency

**Fix:** Use uniform sampling or proper prioritized sampling:
```python
# Option 1: Uniform sampling
start_logical = 0
end_logical = self.num_trajectories

# Option 2: Prioritized by trajectory return/age
# (would require storing priorities per trajectory)

# Option 3: Mix of recent and uniform (recommended)
recent_samples = int(batch_size * 0.5)  # 50% recent
uniform_samples = batch_size - recent_samples

# Sample recent
recent_indices = jax.random.randint(rng1, (recent_samples,), 
                                    max(0, end_logical - heuristic_window), 
                                    end_logical)

# Sample uniformly from all history
uniform_indices = jax.random.randint(rng2, (uniform_samples,), 0, end_logical)

logical_traj_indices = jnp.concatenate([recent_indices, uniform_indices])
```

---

### 🟡 MEDIUM: Importance Sampling Weights May Be Incorrect

**Location:** `expert_buffer.py:358-380`

**Issue:** Importance sampling weight computation has several concerns:

```python
# Compute importance sampling weights
priorities = entries['priority'].astype(np.float64) ** self.per_alpha
priorities = np.maximum(priorities, 1e-10)

total_priority = self.sum_tree[1]
if total_priority <= 0:
    total_priority = 1.0

probs = priorities / total_priority
probs = np.maximum(probs, 1e-10)

# w_i = (N * P(i))^(-beta), normalized by max
N = total_available
weights = (N * probs) ** (-self.per_beta)

# Normalize by max weight
min_prob = self.min_tree[1] / total_priority
if min_prob > 0:
    max_weight = (N * min_prob) ** (-self.per_beta)
    weights = weights / max(max_weight, 1e-10)
else:
    weights = weights / max(weights.max(), 1e-10)

weights = np.clip(weights, 0.1, 10.0).astype(np.float32)
```

**Problems:**

1. **Double alpha exponentiation:** Priorities are already stored with alpha applied (`priority_alpha = priority ** self.per_alpha`), but here we apply alpha again:
   ```python
   priorities = entries['priority'].astype(np.float64) ** self.per_alpha  # SECOND TIME!
   ```

2. **Clipping hides bias:** Weights clipped to [0.1, 10.0] - this defeats the purpose of importance sampling which should correct for sampling bias.

3. **Different normalization for different branches:** If `min_prob > 0` we normalize by max possible weight, otherwise by batch max. Inconsistent!

**Correct formula (standard PER):**
```python
# P(i) = p_i^alpha / sum(p_j^alpha)
# w_i = (N * P(i))^(-beta) / max(w)

# Priorities already have alpha applied in storage
priorities = entries['priority'].astype(np.float64)  # Don't apply alpha again!

# Compute probabilities
probs = priorities / self.sum_tree[1]

# Importance sampling weights
N = total_available
weights = (N * probs) ** (-self.per_beta)

# Normalize by max weight (the minimum probability sample gets weight 1.0)
min_priority = self.min_tree[1]  # Tree stores p^alpha
max_weight = (N * min_priority / self.sum_tree[1]) ** (-self.per_beta)
weights = weights / max_weight

# Optional: clip for stability, but should be rare
weights = np.clip(weights, 0.01, 100.0)
```

---

### 🟢 LOW: Generation Validation is Correct and Well-Designed

**Location:** `replay_buffer.py:289-295`

**Verified:** The generation validation for wrap-around safety is correctly implemented:

```python
# GENERATION VALIDATION
gen_diff = current_generation - traj_gen

# Check if trajectory is valid
is_same_gen = gen_diff == 0
is_prev_gen_safe = (gen_diff == 1) & (current_position <= start_pos)
is_valid = is_same_gen | is_prev_gen_safe
```

This correctly identifies stale data and masks it in the loss computation.

---

### 🟢 LOW: FIFO Eviction in Expert Buffer Agent Zone is Correct

**Location:** `expert_buffer.py:203-208`

**Verified:** The FIFO circular buffer logic is correct:

```python
# FIFO: advance and wrap within agent zone
self.agent_write_idx += 1
if self.agent_write_idx >= self.total_capacity:
    self.agent_write_idx = self.demo_capacity  # Wrap to agent zone start

self.agent_count = min(self.agent_count + 1, self.agent_capacity)
```

This properly evicts old agent samples while protecting demos.

---

## Summary of Sampling Biases

| Component | Bias Type | Severity | Impact |
|-----------|-----------|----------|--------|
| Expert buffer demo sampling | No priority trees | 🔴 Critical | Different sampling distribution |
| Expert buffer priority | Double alpha | 🟡 Medium | Wrong importance weights |
| Expert buffer weights | Aggressive clipping | 🟡 Medium | Bias correction weakened |
| Main buffer sampling | Recent-only | 🟡 Medium | Catastrophic forgetting |
| Main buffer | No PER | 🟢 Low | Uniform sampling is acceptable |
| Generation validation | None (correct) | ✅ Good | Prevents stale data usage |

---

## Recommendations

### Priority 1: Fix Demo Sampling in Expert Buffer

**File:** `expert_buffer.py:484-508`

Replace `_sample_from_range` with tree-based sampling or use the existing `_sample_proportional` for both zones.

### Priority 2: Fix Double Alpha Application

**File:** `expert_buffer.py:359`

Remove second alpha exponentiation:
```python
# BEFORE (wrong):
priorities = entries['priority'].astype(np.float64) ** self.per_alpha

# AFTER (correct):
priorities = entries['priority'].astype(np.float64)  # Alpha already applied in _update_priority
```

### Priority 3: Fix Main Buffer Sampling

**File:** `replay_buffer.py:255-267`

Implement mixed recent/uniform sampling:
```python
# 50% recent, 50% uniform from all history
recent_size = self.batch_size // 2
uniform_size = self.batch_size - recent_size

recent_indices = jax.random.randint(rng1, (recent_size,), 
                                    max(0, end_logical - heuristic_window), end_logical)
uniform_indices = jax.random.randint(rng2, (uniform_size,), 0, end_logical)

logical_traj_indices = jnp.concatenate([recent_indices, uniform_indices])
```

### Priority 4: Reduce Weight Clipping

**File:** `expert_buffer.py:381`

Increase clipping range or remove:
```python
# Optional: clip for stability, but allow wider range
weights = np.clip(weights, 0.01, 100.0).astype(np.float32)
```

---

## Testing Plan

1. **Sampling Distribution Test:**
   ```python
   # Add samples with known priorities
   for i in range(1000):
       buffer.add_agent(..., td_error=i / 100.0)  # Linear priorities
   
   # Sample many times
   counts = np.zeros(1000)
   for _ in range(10000):
       batch = buffer.sample_batch(batch_size=32)
       for idx in batch['indices']:
           counts[idx] += 1
   
   # Check that higher priority samples are sampled more often
   correlation = np.corrcoef(np.arange(1000), counts)[0, 1]
   assert correlation > 0.8, "PER not working correctly!"
   ```

2. **Demo vs Agent Balance Test:**
   ```python
   # Add 100 demos and 1000 agent samples
   # Sample 1000 batches
   demo_count = 0
   agent_count = 0
   
   for _ in range(1000):
       batch = buffer.sample_batch(batch_size=128)
       demo_count += batch['is_demo'].sum()
       agent_count += (~batch['is_demo']).sum()
   
   ratio = demo_count / (demo_count + agent_count)
   expected = 0.125  # 12.5%
   assert abs(ratio - expected) < 0.02, f"Demo ratio {ratio:.3f} != expected {expected}"
   ```

3. **Main Buffer Coverage Test:**
   ```python
   # Add 10000 trajectories
   # Sample 1000 batches
   sampled_indices = set()
   
   for _ in range(1000):
       batch = buffer.sample_batch(batch_size=32)
       for logical_idx in batch['logical_indices']:
           sampled_indices.add(int(logical_idx))
   
   # Check coverage of old trajectories
   old_sampled = [i for i in sampled_indices if i < 9000]  # First 90%
   coverage = len(old_sampled) / 9000
   assert coverage > 0.5, f"Old trajectory coverage only {coverage:.1%}"
   ```

---

## Acceptance Criteria

- [ ] Expert buffer uses consistent sampling method for demo and agent zones
- [ ] Double alpha application removed from importance weight computation
- [ ] Main replay buffer samples from full history (not just recent)
- [ ] Importance sampling weight clipping less aggressive
- [ ] Demo ratio in sampled batches matches configured ratio (±2%)
- [ ] PER correlation test passes (>0.8 correlation between priority and sample frequency)
- [ ] Old trajectory coverage test passes (>50% of old trajectories sampled in test)

---

## References

- Prioritized Experience Replay paper: https://arxiv.org/abs/1511.05952
- `mcts_trader/jax_agent/expert_buffer.py` - Expert buffer with PER
- `mcts_trader/jax_agent/replay_buffer.py` - Main replay buffer
- `mcts_trader/jax_agent/trainer.py` - Training loop (lines 1581-1680)

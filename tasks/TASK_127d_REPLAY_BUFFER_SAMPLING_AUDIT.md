# TASK_127d: Replay Buffer Sampling Audit

**Status**: AUDITED (Issues Found)
**Priority**: Medium
**Created**: 2026-01-31

## Scope

Audited replay buffer components:
- `mcts_trader/jax_agent/expert_buffer.py` - PER with demo/agent zones
- `mcts_trader/jax_agent/replay_buffer.py` - JAX trajectory buffer
- `mcts_trader/jax_agent/trainer.py` - Batch construction

## Architecture Overview

### Two Buffer System

```
┌─────────────────────────────────────────────────────────────────┐
│ UnifiedIndexBuffer (expert_buffer.py)                           │
│  [0 ... demo_capacity-1]  = Protected demo zone (never evicted) │
│  [demo_capacity ... total] = FIFO agent zone (circular)         │
│  - Segment trees for O(log n) PER                               │
│  - Index-based storage (32 bytes/sample vs 400KB)               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ JAXReplayBuffer (replay_buffer.py)                              │
│  - Trajectory-based storage                                      │
│  - Generation validation for wrap-around safety                  │
│  - Sequence sampling for MuZero unrolling                       │
└─────────────────────────────────────────────────────────────────┘
```

## Audit Results

### 1. PER (Prioritized Experience Replay) ⚠️ POTENTIAL BIAS

**Location**: `expert_buffer.py:229-295`

#### Segment Tree Implementation ✅ CORRECT

```python
def _update_priority(self, idx: int, priority: float):
    priority_alpha = priority ** self.per_alpha
    tree_idx = idx + self.tree_capacity
    self.sum_tree[tree_idx] = priority_alpha
    while tree_idx > 1:
        tree_idx //= 2
        self.sum_tree[tree_idx] = self.sum_tree[2*tree_idx] + self.sum_tree[2*tree_idx + 1]
```

**Assessment**: Standard segment tree update - O(log n) complexity, correct propagation.

#### Stratified Sampling ⚠️ MINOR ISSUE

```python
def _sample_proportional(self, n: int, ...):
    segment = total / n
    for i in range(n):
        s = np.random.uniform(segment * i, segment * (i + 1))
        # ... tree traversal to find index
```

**Issue**: Stratified sampling ensures coverage across priority range, but:
- Does NOT guarantee unique samples
- Same high-priority sample could be selected multiple times if it spans multiple segments

**Impact**: Low - only affects cases where one sample has disproportionately high priority.

#### Priority Bias Toward High TD-Error ⚠️ EXPECTED BEHAVIOR

```python
# Agent samples: priority = |TD_error| + 1e-6
priority = np.abs(td_error) + 1e-6

# Demo samples: priority = confidence + demo_priority_boost
priority = confidence + self.demo_priority_boost  # +0.001 default
```

**Observation**: Samples with high TD-error are sampled more frequently. This is intended for PER but could cause:
- Over-representation of "surprising" samples
- Under-training on "easy" samples that agent has learned
- Stale high-priority samples if priorities not updated frequently

### 2. Demo vs Agent Sample Ratio ✅ BALANCED

**Location**: `expert_buffer.py:299-348`

```python
def sample_batch(self, batch_size=1024, demo_ratio=0.125):
    n_demo = min(int(batch_size * demo_ratio), self.demo_count)
    n_agent = batch_size - n_demo
    n_agent = min(n_agent, self.agent_count)
    n_demo = batch_size - n_agent  # Adjust if not enough agent samples
```

**Assessment**:
- ✅ Default ratio: 12.5% demos (128/1024)
- ✅ Gracefully handles insufficient samples in either zone
- ✅ Falls back to available samples without crashing

**Potential Issue**: Early in training when `agent_count` is low, batches will be demo-heavy, potentially causing overfitting to demos.

### 3. Importance Sampling Weights ✅ CORRECT

**Location**: `expert_buffer.py:358-381`

```python
# Compute weights
priorities = entries['priority'] ** self.per_alpha
probs = priorities / total_priority
weights = (N * probs) ** (-self.per_beta)

# Normalize by max weight
min_prob = self.min_tree[1] / total_priority
max_weight = (N * min_prob) ** (-self.per_beta)
weights = weights / max(max_weight, 1e-10)

# Clip for stability
weights = np.clip(weights, 0.1, 10.0)
```

**Assessment**:
- ✅ Standard PER importance sampling formula: `w_i = (N * P(i))^(-β)`
- ✅ Normalized by max weight for stability
- ✅ Beta annealing from 0.4 → 1.0 over 100k steps
- ✅ Clip to [0.1, 10.0] prevents extreme weights

#### Beta Annealing ✅ CORRECT

```python
self.per_beta = min(
    self.per_beta_final,
    self.per_beta_init + (self.per_beta_final - self.per_beta_init) * (self.update_count / self.per_beta_steps)
)
```

**Assessment**: Linear annealing β: 0.4 → 1.0 over 100k samples. Standard practice.

### 4. JAX Buffer Batch Sampling ✅ CORRECT WITH VALIDATION

**Location**: `replay_buffer.py:230-341`

#### Generation Validation ✅ ROBUST

```python
# GENERATION VALIDATION
gen_diff = current_generation - traj_gen
is_same_gen = gen_diff == 0
is_prev_gen_safe = (gen_diff == 1) & (current_position <= start_pos)
is_valid = is_same_gen | is_prev_gen_safe
```

**Assessment**:
- ✅ Tracks when each trajectory was added (generation number)
- ✅ Validates that data hasn't been overwritten by circular buffer
- ✅ Returns `valid` flag in batch for loss masking

#### Sequence Sampling ⚠️ POTENTIAL STARVATION

```python
# Heuristic: Start with capacity // 500 recent trajectories
heuristic_window = max(2, self.capacity // 500)
logical_traj_indices = jax.random.randint(sample_rng, (self.batch_size,), start_logical, end_logical)
```

**Issue**: Only samples from recent `capacity // 500` trajectories, not full buffer.

**Impact**:
- Old trajectories are effectively ignored (sample starvation)
- Could lose diversity if training window is too narrow
- Intended behavior for recency bias, but may be too aggressive

### 5. Trainer Batch Construction ✅ CORRECT

**Location**: `trainer.py:1582-1627`

```python
# Sample from JAX replay buffer
batch = replay_buffer.sample_batch(sample_rng)

# Optionally sample from expert buffer
if expert_buffer is not None:
    expert_batch = expert_buffer.sample_batch(...)
```

**Assessment**:
- ✅ Uses JAX buffer for standard MuZero training
- ✅ Expert buffer sampling is separate and optional
- ✅ Valid mask from JAX buffer used in loss computation

## Issues Found

### Issue 1: Sample Starvation in JAX Buffer ⚠️

**Severity**: Medium

**Problem**: The heuristic window limits sampling to recent `capacity // 500` trajectories.

```python
heuristic_window = max(2, self.capacity // 500)
```

For capacity=100,000: only samples from last 200 trajectories.

**Impact**: Older (but valid) trajectories are never sampled, reducing data diversity.

**Recommendation**:
```python
# Option 1: Sample from all valid trajectories
# Option 2: Use larger heuristic window
heuristic_window = max(100, self.capacity // 50)  # 10x larger
```

### Issue 2: Demo Priority Never Decays ⚠️

**Severity**: Low

**Problem**: Demo samples always have `demo_priority_boost` added:

```python
priority = confidence + self.demo_priority_boost
```

Even after agent learns the behavior, demos maintain elevated priority.

**Impact**: Demos may be over-sampled late in training when agent has learned.

**Recommendation**: Consider decaying demo priority boost over time or when acceptance rate is high.

### Issue 3: PER Priorities Not Updated After Training ⚠️

**Severity**: Medium

**Problem**: `update_priorities()` exists but may not be called consistently.

**Location**: `expert_buffer.py:510-522`

```python
def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
    # Updates priorities based on new TD errors
```

**Question**: Is this called after each training step with new TD errors?

**Impact**: Stale priorities lead to biased sampling - high-priority samples that are now "easy" continue to be over-sampled.

### Issue 4: Duplicate Sampling Possible ⚠️

**Severity**: Low

**Problem**: `_sample_from_range()` uses `np.random.choice` with `replace=True` when `n > range_size`:

```python
replace = n > range_size
sampled = np.random.choice(range_indices, n, p=probs, replace=replace)
```

**Impact**: In small demo pools, same sample could appear multiple times in batch.

## Summary

| Component | Status | Issues |
|-----------|--------|--------|
| Segment tree | ✅ Correct | Standard implementation |
| Demo/agent ratio | ✅ Balanced | May be demo-heavy early |
| IS weights | ✅ Correct | Standard PER formula |
| Beta annealing | ✅ Correct | 0.4 → 1.0 over 100k |
| JAX generation validation | ✅ Robust | Prevents stale data |
| Sampling window | ⚠️ Narrow | Only recent trajectories |
| Demo priority | ⚠️ No decay | May over-sample late |
| Priority updates | ⚠️ Unclear | Check if called consistently |

## Recommendations

1. **Increase sampling window** in JAX buffer to utilize more of the buffer capacity
2. **Add demo priority decay** based on acceptance rate or training progress
3. **Verify priority updates** are called after each training step
4. **Add sampling diversity metrics** to DS24 dashboard:
   - Age distribution of sampled trajectories
   - Demo vs agent sample ratio over time
   - Priority distribution histogram

## Related Tasks

- TASK_125: Causal Expert Imprinting (created the expert buffer)
- TASK_127b: MuZero Dynamics Audit

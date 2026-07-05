# TASK 001: Fix Replay Buffer Wrap-Around Sampling Bug

**Severity**: MEDIUM
**Location**: `mcts_trader/jax_agent/replay_buffer.py:194-195`
**Status**: DONE (2025-12-29)
**Estimated Complexity**: Medium

---

## Problem Description

The replay buffer uses circular storage with wrap-around, but the sampling logic does not validate that sampled sequences are contiguous and have not been overwritten.

### Current Code (lines 194-195):
```python
logical_data_indices = start_pos + pos + jnp.arange(self.sequence_length)
physical_data_indices = logical_data_indices % self.capacity
```

### Issues:

1. **Cross-boundary sequences**: When `start_pos + pos + sequence_length` exceeds `capacity`, the modulo operation wraps indices. If the wrapped indices point to data from a DIFFERENT trajectory (written after wrap-around), the sampled sequence will contain discontinuous, misaligned data.

2. **Generation tracking unused**: The code tracks `trajectory_generation` (lines 133-134) but NEVER validates it during sampling. This metadata exists but is wasted.

3. **Stale data sampling**: After buffer fills and wraps, old trajectory metadata may point to overwritten data regions.

### Example Failure Scenario:
```
Buffer capacity: 5000
Trajectory A starts at index 4900, length 200
  - Indices 4900-5099 (wraps to 0-99)

Later, Trajectory B writes to indices 0-300

When sampling Trajectory A:
  - Indices 4900-4999 are correct (Trajectory A)
  - Indices 0-99 are WRONG (now contain Trajectory B data!)
```

---

## Root Cause Analysis

The `add_trajectory` method correctly handles wrap-around for WRITING:
```python
if end_pos <= self.capacity:
    # Linear add
else:
    # Wrap around - split write into two parts
```

But `sample_batch` assumes all sequences are contiguous and valid, which is FALSE after wrap-around.

---

## Proposed Solution

### Option A: Validate Generation During Sampling (Recommended)

Add generation validation to ensure sampled data matches expected trajectory:

```python
def sample_batch(self, rng: chex.PRNGKey) -> Dict[str, jnp.ndarray]:
    # ... existing code to get logical_traj_indices ...

    def get_sequence(batch_idx, key):
        logical_idx = logical_traj_indices[batch_idx]
        physical_traj_idx = logical_idx % self.capacity

        start_pos = self.trajectory_starts[physical_traj_idx]
        traj_len = self.trajectory_lengths[physical_traj_idx]
        traj_gen = self.trajectory_generation[physical_traj_idx]

        # Compute expected generation for data at start_pos
        current_gen = self.total_steps_added // self.capacity

        # Check if trajectory data is still valid (not overwritten)
        data_age = current_gen - traj_gen
        if data_age > 1:
            # Data has been overwritten, return zeros or skip
            return self._get_zero_sequence()

        # ... rest of sampling logic ...
```

### Option B: Reject Cross-Boundary Trajectories

Prevent storing trajectories that would wrap:
```python
def add_trajectory(self, ...):
    end_pos = self.position + traj_length

    if end_pos > self.capacity:
        # Option 1: Reset position to 0, invalidate old data
        self.position = 0
        end_pos = traj_length

        # Option 2: Truncate trajectory
        # traj_length = self.capacity - self.position
```

### Option C: Store Trajectory Data Contiguously with Fixed Slots

Redesign buffer to use fixed-size trajectory slots:
```python
MAX_TRAJ_LENGTH = 500
NUM_SLOTS = capacity // MAX_TRAJ_LENGTH

# Each slot stores one trajectory, never wraps
self.trajectories = jnp.zeros((NUM_SLOTS, MAX_TRAJ_LENGTH, *obs_shape))
```

---

## Implementation Plan

### Step 1: Add Data Validation (30 min)
1. Add `_is_trajectory_valid()` method to check generation
2. Modify `sample_batch()` to skip invalid trajectories
3. Add fallback to resample if trajectory is stale

### Step 2: Add Boundary Safety (20 min)
1. In `add_trajectory()`, detect cross-boundary writes
2. Either reset position or reject trajectory

### Step 3: Add Unit Tests (40 min)
1. Test: Fill buffer to capacity, verify no stale sampling
2. Test: Cross-boundary trajectory handling
3. Test: Multiple wrap-arounds maintain data integrity

### Step 4: Integration Test (20 min)
1. Run training for 100+ epochs
2. Verify no NaN/Inf from corrupted batches
3. Check training curves for instability

---

## Acceptance Criteria

- [ ] Sampling never returns data from overwritten trajectories
- [ ] Cross-boundary trajectories handled gracefully
- [ ] No performance regression (sampling still O(1))
- [ ] Unit tests pass for edge cases
- [ ] Training runs stable after buffer wrap-around

---

## Files to Modify

1. `mcts_trader/jax_agent/replay_buffer.py` - Main fix
2. `tests/test_replay_buffer.py` - Add tests (create if needed)

---

## References

- Current implementation: `replay_buffer.py:141-210`
- Generation tracking: `replay_buffer.py:133-134`
- Wrap-around write logic: `replay_buffer.py:105-125`

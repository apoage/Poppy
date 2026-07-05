# TASK_148: Breaking the Bootstrapping Deadlock

**Status:** COMPLETE
**Completed:** 2026-02-10

**Objective:** Force an initial "Basin Synchronization" merge when training from scratch, even if branches are diverged, to create the first baseline checkpoint.

## 1. The Root Cause (Logic Deadlock)
In `e-dojo_v1.py`, the `_should_merge` function currently contains an early-return trap for fresh starts:

```python
if cosine < self.config.cosine_brake: # e.g., cosine is 0.36, brake is 0.85
    if self.state.merge_count == 0:
        return False, "fresh_start_skip_brake" # <--- THE TRAP
    return True, "divergence_brake"
```

### The Impact
1. Randomly initialized weights start with **cosine ~0.46**.
2. The logic returns **False** immediately.
3. It **skips** the scheduled interval check (`steps >= current_merge_interval`).
4. **Result:** No merge ever happens. The branches diverge further. The "Divergence Brake" can never activate because `merged_latest.pkl` is never created.

---

## 2. Technical Implementation Fix

### A. Update `_should_merge` in `e-dojo_v1.py`
Allow the scheduled merge to proceed if it is the first merge of the run, even if the models are diverged.

```python
def _should_merge(self, cosine: float, barrier: float) -> Tuple[bool, str]:
    steps = self.state.steps_since_merge

    # 1. Handle Divergence Brake logic
    if cosine < self.config.cosine_brake:
        if self.state.merge_count == 0:
            # If we reached the interval, FORCE a merge to create the first baseline
            if steps >= self.state.current_merge_interval:
                return True, "initial_sync_merge" # Break the deadlock
            # Otherwise, just wait for the interval
            return False, "fresh_start_wait_for_interval"
        
        # If not count 0, the brake is safe to activate (merged_latest exists)
        return True, "divergence_brake"

    # 2. Rest of the existing logic (min_interval, max_interval, normal_divergence, etc.)
    # ...
```

### B. Ensure "Early Phase" Merging
Verify that `_get_training_phase()` returns `"early"` during the first merges to use `fedavg`. Simple averaging is the safest way to force the models into the same weight space when divergence is high.

---

## 3. Implementation Steps
1. Stop all training processes.
2. Apply the logic change to `_should_merge` in `e-dojo_v1.py`.
3. Run `./tools/orchestration/reset_edojo.sh`.
4. Restart the dual-branch training.

## 4. Expected Impact
- **Merge #1**: Triggers at the `initial_merge_interval`.
- **Baseline Creation**: `merged_latest.pkl` is created, allowing the Divergence Brake to function for all subsequent merges.
- **Agreement**: `global_agreement` should start to climb once the branches share a common weight space.

---

## Implementation Notes (2026-02-10)

### File Modified
- `tools/orchestration/e-dojo_v1.py:281-292`

### Logic Change
```python
# OLD (deadlock):
if self.state.merge_count == 0:
    return False, "fresh_start_skip_brake"  # Skips interval check!

# NEW (fixed):
if self.state.merge_count == 0:
    if steps >= self.state.current_merge_interval:
        return True, "initial_sync_merge"  # Force baseline creation
    return False, "fresh_start_wait_for_interval"
```

### New Merge Reason
Added `"initial_sync_merge"` reason to indicate forced baseline merge during bootstrap phase.

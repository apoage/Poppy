# TASK_720: SegmentPool Random Sampling (Replace Sequential Cursor)

**Status**: BACKLOG
**Priority**: MEDIUM — addresses training data diversity bottleneck
**Discovered during**: TASK_716g Stage B run (April 2026)

---

## Problem

`SegmentPool.sample_positions()` in `train_easy.py:824` uses a **sequential cursor with stride=1** plus tiny temporal jitter every 50 steps. This was designed for Phase 1's sequential window scanning but has serious diversity issues for Phase 2's MCTS ExIt:

### Concrete impact (1000 epochs × 400 steps × 256 batch, 17 slots)

- **23 passes through each segment** (~258k timesteps each)
- **75% input overlap between consecutive samples** (window_size=64, cursor advances ~16 per round-robin call)
- MCTS sees nearly identical root states repeatedly
- Training plateaus: model memorizes the 17 specific market windows instead of generalizing

### Where it lives

```python
# train_easy.py:824
def sample_positions(self, n_samples, window_size, jitter_interval=50):
    # ... round-robin across slots ...
    for _ in range(per_slot):
        cursor = slot['cursor']
        positions.append((slot_idx, cursor))
        slot['cursor'] += 1   # ← stride=1 sequential
        # Jitter only every 50 steps, range [0, 6) — not enough
        if slot['_step_count'] % jitter_interval == 0:
            slot['cursor'] = min(slot['cursor'] + self.rng.integers(0, 6),
                                 T_ret - window_size)
```

## Proposed Fix

Replace sequential cursor with **random uniform sampling** within each slot's valid range:

```python
def sample_positions(self, n_samples, window_size):
    positions = []
    n_slots = len(self.slots)
    per_slot = n_samples // n_slots + 1

    for i in range(n_slots):
        slot_idx = (self._round_robin_idx + i) % n_slots
        slot = self.slots[slot_idx]
        T_ret = len(slot['returns'])
        max_cursor = T_ret - window_size - 1

        if max_cursor <= 0:
            self._rotate_slot(slot_idx)
            continue

        # Random uniform sampling instead of sequential cursor
        cursors = self.rng.integers(0, max_cursor, size=per_slot)
        for c in cursors:
            if len(positions) >= n_samples:
                break
            positions.append((slot_idx, int(c)))

        # Track usage for periodic slot rotation (every K calls)
        slot['_step_count'] = slot.get('_step_count', 0) + per_slot

    self._round_robin_idx += n_slots
    return positions[:n_samples]
```

### Considerations

1. **Slot rotation**: Without sequential progression, there's no "exhaustion" trigger. Need a separate counter (e.g., rotate slot every N batches) to keep cycling through `max_runs_per_group` runs.

2. **Phase 1 compatibility**: `train_easy.py` itself uses this method. Need to verify Phase 1 training still works with random sampling — if Phase 1 relied on the sequential pattern (e.g., for episode construction), add a flag:
   ```python
   def sample_positions(self, n_samples, window_size, mode='sequential'):
       # mode='sequential' → old behavior (Phase 1)
       # mode='random'     → new behavior (Phase 2)
   ```

3. **Reproducibility**: Random sampling means non-deterministic batches even with same seed unless we carefully manage the RNG.

4. **Buffer fragmentation**: Random access has worse cache behavior than sequential, but each window is only 64 timesteps × N assets — small enough that this is unlikely to matter.

## Workaround Until Fixed

Increase `max_runs_per_group` (default 1) to 10+ to get more *segment-level* diversity, even though within-segment sampling remains sequential. This was applied during TASK_716g session 47.

## Acceptance Criteria

- [ ] Random sampling implemented behind a flag
- [ ] Phase 1 training (`train_easy.py`) verified unchanged
- [ ] Phase 2 training (`train_phase2.py`) uses random mode by default
- [ ] Verify training plateau breaks: value/policy loss continue declining past previous floor
- [ ] Benchmark improvement: more positive alpha periods

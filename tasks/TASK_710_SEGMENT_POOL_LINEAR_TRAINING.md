# TASK_710: Segment Pool — Linear Historical Training

**Status**: DONE
**Priority**: 1 (current training has no linear coherency + OOM risk)
**Depends on**: TASK_707 (period cycling), TASK_698 (historical data)

---

## Problem

TASK_707 introduced period cycling: 270 runs across 18 date ranges loaded into RAM at startup. Two critical issues:

### 1. OOM Crash (81GB RAM)
Each run creates a separate HistoricalMarket (~316MB). 270 × 316MB = **81GB**. System has 45GB RAM. This caused a Machine Check Exception (MCE) crash on 2026-03-14 — kernel killed the process.

Quick fix (committed 2026-03-15): lazy loading — load one period per epoch, free previous. Fixes OOM but exposes the real problem:

### 2. No Linear Coherency
Current flow per epoch:
1. Load fresh HistoricalMarket from disk (~10s)
2. `get_window_random()` walks with `_scan_cursor` + stride
3. Train 400 batches from this segment
4. **Destroy the market, load next one — cursor resets to 0**

Result: model sees the first ~400 windows of every segment, never the rest. With 258k timestamps per segment and ~500 possible sequential windows, we're training on <1% of available data per segment visit. Consecutive runs from the same date range (P197-P200 all being `2024-03→2024-08`) restart from timestamp 0 every time.

For a trading model, temporal coherency is critical. Random window sampling destroys the sequential structure of market regimes.

---

## Design: Segment Pool

### Core Concept
Pool of **6 active segments** in RAM (~1.9GB), each from a **different date range** (regime diversity). Each has a **linear cursor** that persists until the segment is fully walked. When exhausted, replace with next run — same date range first (different assets, same regime), then new date range.

### Data Organization

Group 270 runs by date range → 18 regime groups:

```
Group  0: 2018-10→2019-04   1 run   (61 assets)
Group  1: 2019-04→2019-10   1 run   (59 assets)
Group  2: 2019-10→2020-04   5 runs  (83 assets)
Group  3: 2020-04→2020-10   5 runs  (125 assets)
Group  4: 2020-10→2021-04   8 runs  (180 assets)
...
Group 15: 2024-08→2025-02  24 runs  (527 assets)
Group 16: 2025-02→2025-08  27 runs  (588 assets)
Group 17: 2025-08→2025-11   1 run   (2 assets)
```

### Pool State

```
Slot 0: [Group 2]  run 1/5   cursor=4800/258k   ← walking linearly
Slot 1: [Group 5]  run 0/12  cursor=1200/258k
Slot 2: [Group 9]  run 8/19  cursor=9600/258k
Slot 3: [Group 12] run 0/20  cursor=0/258k      ← just loaded
Slot 4: [Group 14] run 4/22  cursor=7200/258k
Slot 5: [Group 15] run 11/24 cursor=19200/258k
         ↑ all different date ranges (regime diversity)
```

### Per-Epoch Flow

1. Round-robin pick next slot
2. Advance that slot's cursor, take sequential windows → fill buffer
3. Train epoch on buffer
4. Check if slot's segment is exhausted (cursor ≥ end):
   - **Yes, more runs in same group** → load next run (same date range, different assets), cursor=0
   - **Yes, group fully exhausted** → pick next unused date range group, load first run
   - **No** → keep going

### Completion
One pass through all 270 segments × ~258k timestamps ≈ **70M steps**. No reshuffling. Training is done when all groups exhausted. If the model hasn't learned by then, the problem is elsewhere.

### Constraints
- 6 loaded slots **must** be from different date ranges at all times
- Linear walk within each segment — no random sampling
- When replacing exhausted segment: prefer same date range (regime continuity) before moving to new one

---

## Changes

### New: `SegmentPool` class (in `train_easy.py` or separate module)

```python
class SegmentPool:
    def __init__(self, period_specs, db_path, n_slots=6, n_assets=64):
        # Group specs by date range
        # Initialize 6 slots from 6 different groups
        # Each slot: {hm, cursor, group_idx, run_idx}

    def get_next_windows(self, n_windows, window_length):
        # Round-robin slot selection
        # Return sequential windows from current slot's cursor
        # Advance cursor

    def check_and_rotate(self):
        # If current slot exhausted:
        #   - Try next run in same group
        #   - If group done, pick new group
        #   - Load new HistoricalMarket, cursor=0
```

### Modified: `OnTheFlyGenerator`
- Replace `_period_specs` + `historical_market` + `advance_period()` with `SegmentPool`
- `_refill_buffer()`: get windows from pool instead of `hm.get_window_random()`
- Remove `_historical_markets`, `_historical_market_labels`, `_historical_period_idx`

### Modified: `train_epoch()`
- Remove `advance_period()` call — pool handles rotation internally

### Unchanged
- `HistoricalMarket` — still used for loading, but cursor management moves to pool
- `e-dojo_v1.py` — still generates periods file, pool consumes it
- Oracle, encoder, AR feedback — all unchanged

---

## Numbers

- RAM: 6 × ~316MB = **~1.9GB** (was 81GB)
- I/O: ~270 segment loads × ~10s = ~45 min total, spread across training
- Data coverage: 270 × ~258k = **~70M timestamps** (one full pass)
- Per slot: ~258k timestamps ÷ ~500 window length = ~516 sequential windows before exhaustion

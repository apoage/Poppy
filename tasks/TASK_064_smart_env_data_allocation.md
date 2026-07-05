# TASK_064: Smart Environment Data Allocation

## Status: IMPLEMENTED

## Problem

Current parallel env setup:
- All 15 envs load same data (wasteful - 15× DB reads)
- Each env samples from RANDOM start point
- Random sampling = **causality violation** (future data mixed with past)
- No guarantee of sequential coverage

Observed:
- 180 days loaded, need 900 days for 60 epochs (or 720 with 20% overlap)
- Random sampling destroys temporal causality
- Model sees shuffled time, learns spurious correlations

## Design Requirements

1. **4 periods available** - 720 days total across training run
2. **Load 1 period per epoch** - fits in RAM (15 envs × 1 period)
3. **STRATIFIED** - interleave periods (P1→P2→P3→P4→P1...) to prevent forgetting
4. **DISTRIBUTED within period** - envs STRIDED 12 days apart (i.i.d.)
5. **±6h jitter** - data augmentation
6. **Share loaded data across 15 envs** - load once per epoch, not 15×

## Math

```
Load: 4 × 180 days = 720 days (stored as 4 SEPARATE arrays to prevent boundary bleeding)
N envs: 15
Steps per env per epoch: 1440 (1 day)
Epochs: 60

STRATIFIED allocation (prevents catastrophic forgetting):
  Interleave periods across epochs to maintain all regimes in memory.

  Epoch 0:  Period 1    Epoch 4:  Period 1    Epoch 8:  Period 1  ...
  Epoch 1:  Period 2    Epoch 5:  Period 2    Epoch 9:  Period 2  ...
  Epoch 2:  Period 3    Epoch 6:  Period 3    Epoch 10: Period 3  ...
  Epoch 3:  Period 4    Epoch 7:  Period 4    Epoch 11: Period 4  ...

  period_idx = epoch % 4
  epoch_in_period = epoch // 4  (0-14 for each period)

Within each period:
  Stride = 180 / 15 = 12 days
  Distributed tiling across the 180-day period
  After 15 visits per period: full coverage

Result:
  - No catastrophic forgetting (all regimes interleaved)
  - Physics sees ONE regime per epoch (coherent)
  - i.i.d. within epoch (stride=12)
  - Full 720-day coverage across training run
```

## Algorithm

```python
def compute_env_start_offsets(
    period_lengths: List[int],  # List of 4 period lengths (may differ slightly)
    n_envs: int,
    n_steps: int,
    epoch: int,
    jitter_minutes: int = 360,  # ±6 hours for data augmentation
    seed: int = None
) -> Tuple[int, List[int]]:
    """
    Compute STRATIFIED start indices (prevents catastrophic forgetting).

    STRATIFIED: Interleave periods across epochs.
      - Epoch 0: Period 1,  Epoch 1: Period 2,  Epoch 2: Period 3,  Epoch 3: Period 4
      - Epoch 4: Period 1,  Epoch 5: Period 2,  ...

    This prevents forgetting by revisiting all regimes regularly.
    Physics still sees ONE regime per epoch (coherent).

    Args:
        period_lengths: List of timestamps per period (may vary due to data gaps)
        n_envs: Number of parallel environments (15)
        n_steps: Steps per episode (1440 = 1 day)
        epoch: Current epoch number (0-59)
        jitter_minutes: Max jitter (±360 = ±6 hours for augmentation)
        seed: Optional seed for reproducibility

    Returns:
        Tuple of (period_idx, List of start indices within that period)
    """
    rng = np.random.default_rng(seed)
    n_periods = len(period_lengths)

    # STRATIFIED: interleave periods
    period_idx = epoch % n_periods
    epoch_in_period = epoch // n_periods  # 0-14 for each period

    period_length = period_lengths[period_idx]

    # DISTRIBUTED within period: stride = period_length / n_envs
    stride = period_length // n_envs

    # Epoch progression within period
    epoch_shift = (epoch_in_period * n_steps) % period_length

    starts = []
    for env_idx in range(n_envs):
        # STRIDED offset within period
        env_offset = env_idx * stride

        # Jitter for data augmentation (±jitter_minutes)
        jitter = rng.integers(-jitter_minutes, jitter_minutes + 1)

        # Combine: env stride + epoch shift + jitter (within period only)
        start = (env_offset + epoch_shift + jitter) % period_length

        # Ensure room for full episode within period
        max_start = period_length - n_steps
        if start > max_start:
            start = max_start

        starts.append(start)

    return period_idx, starts


# Usage per epoch:
# 1. Determine which period to load
# 2. Load that period (once)
# 3. Share across all 15 envs
# 4. Run epoch with distributed offsets

# period_idx, starts = compute_env_start_offsets(period_lengths, n_envs, n_steps, epoch)
# prices = load_period(period_configs[period_idx])  # Load 1 period
# for env in envs:
#     env.set_prices(prices)  # Share reference, not copy
#     env.reset(start_offset=starts[env_idx])


def check_overlap(starts: List[int], n_steps: int) -> float:
    """Calculate actual overlap percentage between envs."""
    ranges = [(s, s + n_steps) for s in starts]
    total_overlap = 0

    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            overlap = max(0, min(ranges[i][1], ranges[j][1]) - max(ranges[i][0], ranges[j][0]))
            total_overlap += overlap

    max_possible = n_steps * len(starts) * (len(starts) - 1) / 2
    return total_overlap / max_possible if max_possible > 0 else 0


def compute_coverage(all_starts: List[int], n_steps: int, period_length: int) -> float:
    """Calculate what % of period has been touched by training."""
    touched = set()
    for start in all_starts:
        for t in range(start, start + n_steps):
            touched.add(t % period_length)
    return len(touched) / period_length
```

## Implementation Changes

### 1. train_jax_fast.py - Share price data

```python
# After first env created:
shared_prices = env.prices  # Extract prices array

# For additional envs:
env_kwargs_shared = env_kwargs.copy()
env_kwargs_shared.pop('db_path', None)  # Remove DB params
env_kwargs_shared['prices'] = shared_prices  # Use shared array
```

### 2. VectorBTTradingEnv - Accept start offset

```python
def reset(self, seed=None, options=None, start_offset=None):
    """
    Reset environment.

    Args:
        start_offset: If provided, use this start index instead of random
    """
    if start_offset is not None:
        self.episode_start_idx = start_offset
    else:
        # Current random behavior
        self.episode_start_idx = self.rng.integers(self.lookback, max_start)
```

### 3. Self-play loop - Coordinate starts

```python
def run_parallel_selfplay(envs, n_steps, epoch):
    # Compute smart start offsets
    starts = compute_env_start_offsets(
        period_length=len(envs[0].prices),
        n_envs=len(envs),
        n_steps=n_steps,
        epoch=epoch
    )

    # Reset each env with its assigned start
    for env, start in zip(envs, starts):
        env.reset(start_offset=start)

    # Run episode...
```

## Expected Behavior

```
4 periods available (720 days total). Load 1 period per epoch.
STRATIFIED: Interleave periods to prevent catastrophic forgetting.

Epoch 0:  Load Period 1, stride=12, shift=0
Epoch 1:  Load Period 2, stride=12, shift=0
Epoch 2:  Load Period 3, stride=12, shift=0
Epoch 3:  Load Period 4, stride=12, shift=0
Epoch 4:  Load Period 1, stride=12, shift=1  (advance within period)
Epoch 5:  Load Period 2, stride=12, shift=1
...
Epoch 56: Load Period 1, stride=12, shift=14
Epoch 57: Load Period 2, stride=12, shift=14
Epoch 58: Load Period 3, stride=12, shift=14
Epoch 59: Load Period 4, stride=12, shift=14

Within each epoch (15 envs, stride=12):
  Env 0:  day 0   ± 6h jitter
  Env 1:  day 12  ± 6h jitter
  ...
  Env 14: day 168 ± 6h jitter

→ Model revisits ALL regimes every 4 epochs (no forgetting)
→ Physics coherent per epoch (one regime)
→ i.i.d. within epoch (stride=12)
→ Full 720-day coverage across training run
```

## Jitter as Data Augmentation

**±6h jitter provides augmentation WITHOUT breaking i.i.d.**

With distributed allocation:
- Envs are 12 days apart (no overlap in same batch)
- Jitter adds ±6 hours to each env's start time
- Same logical day seen from different "angles" across epochs
- Prevents overfitting to specific candle times (e.g., "always buy at 9:30 AM")

| Jitter | Effect |
|--------|--------|
| 0 | Deterministic, risk of time-of-day overfitting |
| ±2 hours | Minimal augmentation |
| **±6 hours** | **Good augmentation, avoids midnight bias** |
| ±12 hours | Maximum, may cause day boundary issues |

**Decision: ±6 hour jitter (±360 min) for data augmentation**

## Metrics to Track

- `data_coverage`: % of period timestamps touched
- `overlap_rate`: Average pairwise overlap between envs
- `epoch_diversity`: Variance in start positions across epochs

## Testing

```python
def test_stratified_allocation():
    """Verify epochs are interleaved across periods (P1→P2→P3→P4→P1...)."""
    period_lengths = [259200, 259200, 259200, 259200]  # 4 × 180 days
    n_envs = 15
    n_steps = 1440

    # Epoch 0 → Period 0, Epoch 1 → Period 1, etc.
    for epoch in range(8):
        period_idx, starts = compute_env_start_offsets(
            period_lengths=period_lengths,
            n_envs=n_envs,
            n_steps=n_steps,
            epoch=epoch,
            jitter_minutes=0,
            seed=42
        )
        expected_period = epoch % 4
        assert period_idx == expected_period, f"Epoch {epoch} should use Period {expected_period}, got {period_idx}"


def test_stride_12_within_period():
    """Verify stride=12 days within each period."""
    period_lengths = [259200, 259200, 259200, 259200]
    n_envs = 15
    n_steps = 1440

    period_idx, starts = compute_env_start_offsets(
        period_lengths=period_lengths,
        n_envs=n_envs,
        n_steps=n_steps,
        epoch=0,
        jitter_minutes=0,
        seed=42
    )

    # Verify 12-day spacing
    days = [s // n_steps for s in starts]
    for i in range(len(days) - 1):
        gap = days[i+1] - days[i]
        assert gap == 12, f"Gap between env {i} and {i+1} should be 12 days, got {gap}"


def test_no_forgetting():
    """Verify model revisits all periods regularly (every 4 epochs)."""
    period_lengths = [259200, 259200, 259200, 259200]
    n_envs = 15
    n_steps = 1440

    # Track which periods are visited
    period_visits = {0: [], 1: [], 2: [], 3: []}
    for epoch in range(60):
        period_idx, _ = compute_env_start_offsets(
            period_lengths=period_lengths,
            n_envs=n_envs,
            n_steps=n_steps,
            epoch=epoch,
            jitter_minutes=0,
            seed=42
        )
        period_visits[period_idx].append(epoch)

    # Each period should be visited 15 times (60 / 4)
    for p, visits in period_visits.items():
        assert len(visits) == 15, f"Period {p} visited {len(visits)} times, expected 15"

    # Max gap between visits should be 4 epochs
    for p, visits in period_visits.items():
        for i in range(len(visits) - 1):
            gap = visits[i+1] - visits[i]
            assert gap == 4, f"Period {p} gap between visits: {gap}, expected 4"


def test_period_coverage():
    """Verify each period fully covered after 15 visits."""
    period_lengths = [259200, 259200, 259200, 259200]
    n_steps = 1440

    # Collect all starts for Period 0 (epochs 0, 4, 8, ...)
    period0_starts = []
    for epoch in range(0, 60, 4):  # Period 0 epochs
        _, starts = compute_env_start_offsets(
            period_lengths=period_lengths,
            n_envs=15,
            n_steps=n_steps,
            epoch=epoch,
            jitter_minutes=0,
            seed=42
        )
        period0_starts.extend(starts)

    coverage = compute_coverage(period0_starts, n_steps, period_lengths[0])
    assert coverage > 0.95, f"Period 0 coverage: {coverage:.1%}"
```

## Priority

**CRITICAL** - Current random sampling violates causality. Model sees shuffled time,
learns spurious correlations that won't generalize. This is fundamental data quality issue.

## Related

- TASK_104: System 3 (future - smart data will feed into adversarial generator)
- Current: 15× DB reload bug (fix as part of this task)

## Implementation Plan

### Step 1: Create smart allocation module
**File:** `mcts_trader/jax_agent/smart_allocation.py`

```python
# Contents:
# - compute_env_start_offsets()
# - check_overlap()
# - SmartAllocationConfig dataclass
```

### Step 2: Batch 4 periods from rainbow table
**File:** `tools/orchestration/dojo_v3_historical.py`

Changes:
1. Take next 4 available periods from rainbow table (not necessarily consecutive)
2. Pass date ranges via CLI to train_jax_fast.py (e.g., `--periods "start1:end1,start2:end2,..."`)
3. Training script loads and concatenates internally

Note: Rainbow table windows calculated for a reason - keep using them.
Just batch 4 at a time instead of 1.

### Step 3: Modify train_jax_fast.py - Share prices
**File:** `train_jax_fast.py` around line 546-552

Changes:
1. After first env created, extract `shared_prices = env.prices`
2. Create `env_kwargs_shared` without DB params
3. Add `prices=shared_prices` to shared kwargs
4. Create remaining envs with shared kwargs

### Step 4: Modify VectorBTTradingEnv.reset()
**File:** `mcts_trader/env/vectorbt_env.py` around line 370

Changes:
1. Add `start_offset: Optional[int] = None` parameter
2. If `start_offset` provided, use it instead of random

### Step 5: Modify self-play loop
**File:** `train_jax_fast.py` - find `run_batched_selfplay` or equivalent

Changes:
1. Before resetting envs, compute smart offsets via `compute_env_start_offsets()`
2. Pass `start_offset` to each env's reset

### Step 6: Add to hot config (optional)
**File:** `tools/orchestration/dojo_v3_historical.py`

Add overridable:
- `jitter-minutes`: int (default 360 = ±6 hours)

### Step 7: Add metrics to DS24
**File:** `train_jax_fast.py` DS24 logging section

Log:
- `data_coverage`
- `overlap_rate`

---

## Execution Checklist

- [x] Create `smart_allocation.py` module - `mcts_trader/jax_agent/smart_allocation.py`
- [x] Unit test allocation function (stratified, stride=12) - `tests/test_smart_allocation.py` (19 tests pass)
- [x] **Verify VectorBT immutability** - SAFE in historical mode (prices only modified in synthetic mode)
- [x] Add `--smart-allocation` and `--jitter-minutes` CLI to train_jax_fast.py
- [x] Add `--periods` CLI placeholder (multi-period support for future)
- [x] **Add dojo_v3 stratified mode** - `--stratified` flag with `run_training_stratified()`
- [x] Modify train_jax_fast.py for shared prices (load once, share refs)
- [x] Modify VectorBTTradingEnv.reset() for start_offset parameter
- [x] Add `VectorBTTradingEnv.set_prices()` for runtime price switching
- [x] Modify `run_self_play_batched()` to accept `start_offsets` parameter
- [x] Add DS24 metrics (smart_allocation, jitter_minutes, current_period)
- [x] Integration test (verify no forgetting, physics coherence) - `tests/test_smart_allocation_integration.py` (11 tests)
- [x] Update CLAUDE.md with changes

## Notes

### Implementation Summary (2026-01-19)

**Files Created:**
- `mcts_trader/jax_agent/smart_allocation.py` - Core allocation algorithm
- `tests/test_smart_allocation.py` - 19 unit tests

**Files Modified:**
- `mcts_trader/env/vectorbt_env.py` - Added `reset(start_offset=)` and `set_prices()`
- `mcts_trader/training/self_play.py` - Added `start_offsets` parameter to `run_self_play_batched()`
- `train_jax_fast.py` - Added CLI args, shared prices, smart offset computation
- `tools/orchestration/dojo_v3_historical.py` - Added `--stratified` mode and `run_training_stratified()`

**New CLI Arguments:**
```bash
# train_jax_fast.py
--smart-allocation     # Enable distributed tiling within period
--jitter-minutes 360   # ±6 hour data augmentation
--periods "..."        # Multi-period mode (placeholder for future)

# dojo_v3_historical.py
--stratified           # Enable stratified training (cycle through periods)
--periods-per-batch 4  # Batch 4 periods at a time
--epochs-per-period 15 # 15 epochs per period before switching
```

**Usage:**
```bash
# Single-period with smart allocation (within-period distributed tiling)
python train_jax_fast.py --db-path trading_data.db --smart-allocation ...

# Stratified historical training (cycle P0→P1→P2→P3→P0...)
python tools/orchestration/dojo_v3_historical.py --stratified
```

**Quick Reference:**
```
STRATIFIED: P1→P2→P3→P4→P1... (revisit all regimes every 4 epochs)
DISTRIBUTED: stride=12 days within each 180-day period
JITTER: ±6h for augmentation
LOAD: 1 period per epoch, share across 15 envs

period_idx = epoch % 4
epoch_in_period = epoch // 4
```

**DS24 Metrics to Add:**
- `current_period`: which period (0-3) this epoch uses
- `epoch_in_period`: progress within current period (0-14)
- `data_coverage`: % of current period touched so far

**Lesson learned:** Causality is a bitch. Random sampling destroys it. Took 2 hours to get the allocation strategy right.

**The Iceberg (what seemed simple vs reality):**
```
Surface: "Just load data smarter, save some DB reads"

Below:
  ├─ Causality constraints (time must flow forward)
  ├─ i.i.d. requirements (batches need regime diversity)
  ├─ Forgetting dynamics (can't stay in one regime too long)
  ├─ Physics coherence (can't mix regimes within epoch)
  ├─ Memory limits (can't load everything at once)
  └─ Architecture rewrite (dojo isn't built for this)

Every "simple" fix opened another constraint.
Systems with feedback loops - everything coupled to everything.
```

## Implementation Challenges

### 1. Dojo V3 Architecture Change
Current dojo_v3_historical.py finishes one period completely before moving to next.
Stratified allocation (P1→P2→P3→P4→P1...) requires **fundamental rewrite**:
- Changes from "Sequential Curriculum" to "Experience Replay" of regimes
- Dojo must track progress across 4 periods simultaneously
- Period configs passed upfront, dojo cycles through them

### 2. VectorBT Immutability Check
Before sharing prices array, verify vectorbt doesn't modify in place:
- Check if `env.prices` is read-only during episode
- If vectorbt creates internal copies, RAM savings negated
- May need `np.asarray(prices, dtype=np.float32).flags.writeable = False`

### 3. Scope Boundaries
- **IN SCOPE**: Data allocation strategy, period interleaving, start offset computation
- **OUT OF SCOPE**: Memory optimization, VRAM placement, multiprocessing changes

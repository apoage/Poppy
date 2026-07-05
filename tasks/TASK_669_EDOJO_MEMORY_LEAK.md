# TASK_669: e-dojo Memory Leak - System Freeze During Overnight Training

## Status: FIXED (2026-02-06)

## Priority: CRITICAL

## Problem Summary

System froze during overnight dual-branch training (Feb 5-6, 2026). Training was running fine through epoch 101 when system became completely unresponsive - blinking cursor on all TTYs, keyboard working but no input accepted. Required hard reboot.

## Root Cause Analysis

### Memory Timeline (from `sar -r`)

| Time | Free RAM | Used % | kbcached | kbdirty |
|------|----------|--------|----------|---------|
| 21:00 | 27.9 GB | 13% | 12.9 GB | 1 MB |
| 01:30 | 10.6 GB | 25% | 23.2 GB | 2.6 MB |
| 02:10 | 3.9 GB | 29% | 27.3 GB | **174 MB** |
| 02:40 | **0.86 GB** | 29% | 30.8 GB | **173 MB** |
| 02:50 | **0.89 GB** | 28% | 31.0 GB | 1.5 MB |
| 02:50:36 | SYSTEM FREEZE | - | - | - |

### Key Observations

1. **kbcached explosion**: 12.9 GB → 31.0 GB (kernel caching checkpoint writes)
2. **kbdirty spikes**: 173-174 MB pending writes during merge operations
3. **Free RAM collapse**: 27 GB → 0.86 GB over ~6 hours
4. **No OOM killer**: System froze before OOM killer could act
5. **Logs end abruptly**: Last entry at 02:50:36, no kernel panic message

### Merge Events

From `logs/e-dojo/merge_history.json`:
- **Merge #1**: 02:07:18 (emergency_divergence, fedavg)
- **Merge #2**: 02:32:50 (emergency_divergence, fedavg)

Memory started spiking immediately after first merge.

## Technical Root Cause

### Location: `tools/orchestration/e-dojo_v1.py` - `check_and_merge()` (lines 371-483)

```python
def check_and_merge(self):
    # Load 168MB × 2 = 336MB into memory
    ckpt_a = load_checkpoint(str(ckpt_a_path))
    ckpt_b = load_checkpoint(str(ckpt_b_path))

    # Creates copies for divergence
    params_a = ckpt_a['params']
    params_b = ckpt_b['params']
    cosine = cosine_similarity(params_a, params_b)

    # Creates another 168MB+ for merged result
    merged_ckpt = self._perform_merge(ckpt_a, ckpt_b, method)

    # Double save (both timestamped AND latest)
    save_checkpoint(merged_ckpt, str(merged_path))
    save_checkpoint(merged_ckpt, str(latest_path))

    # ❌ NO CLEANUP - all arrays stay in memory!
```

### Issues Identified

1. **No garbage collection** after merge operations
2. **No `del` statements** for large checkpoint dicts
3. **No `jax.clear_caches()`** to free JAX compilation artifacts
4. **Double checkpoint save** doubles I/O (kernel caches both)
5. **Fisher extraction** creates additional JAX arrays that aren't freed
6. **168MB checkpoints** saved every 5 epochs → massive kernel cache growth

### Memory Math

Per merge cycle:
- Load ckpt_a: 168 MB
- Load ckpt_b: 168 MB
- Fisher arrays: ~100 MB
- Merged params: 168 MB
- **Total retained: ~600 MB per merge**

Plus:
- Kernel caches all checkpoint writes
- 24 checkpoints per branch × 168 MB = 4 GB cached I/O per branch
- Training buffers and JAX compilation cache

## Proposed Fix

### 1. Add cleanup to `check_and_merge()`:

```python
def check_and_merge(self):
    try:
        ckpt_a = load_checkpoint(str(ckpt_a_path))
        ckpt_b = load_checkpoint(str(ckpt_b_path))
        # ... merge logic ...
    finally:
        # Cleanup
        del ckpt_a, ckpt_b
        if 'merged_ckpt' in dir():
            del merged_ckpt
        jax.clear_caches()
        import gc
        gc.collect()
```

### 2. Add cleanup to `_perform_merge()`:

```python
def _perform_merge(self, ckpt_a, ckpt_b, method):
    # ... merge logic ...

    # Cleanup intermediate arrays
    del params_a, params_b
    if method == "fisher":
        del fisher_a, fisher_b

    return merged_ckpt
```

### 3. Reduce checkpoint save frequency:

Change from saving every 5 epochs to every 10-20 epochs, or implement checkpoint rotation (keep only last N).

### 4. Optional: Drop kernel caches after merge:

```python
import os
os.system('sync')  # Flush pending writes
# Only if running as root or with sudo:
# os.system('echo 3 > /proc/sys/vm/drop_caches')
```

### 5. Consider memory-mapped loading:

For 168MB pickles, consider `mmap` or chunked loading to avoid full RAM copies.

## Files to Modify

| File | Changes |
|------|---------|
| `tools/orchestration/e-dojo_v1.py` | Add gc.collect(), del statements, jax.clear_caches() |
| `tools/orchestration/merge_utils.py` | Add cleanup after Fisher extraction |
| `train_easy.py` | Reduce checkpoint frequency, add rotation |

## Fix Applied (2026-02-06)

Changes to `tools/orchestration/e-dojo_v1.py`:

1. **Added imports**: `gc`, `shutil`

2. **`check_and_merge()` restructured with try/finally**:
   - Variables initialized to None for safe cleanup
   - `del` statements for ckpt_a, ckpt_b, params_a, params_b, merged_ckpt
   - `jax.clear_caches()` in finally block
   - `gc.collect()` in finally block
   - Changed double save to `shutil.copy2()` (avoids re-serialization)

3. **`_perform_merge()` cleanup**:
   - Added try/finally for Fisher array cleanup
   - `del fisher_a, fisher_b` after merge

### Key Changes

```python
# Before: No cleanup
merged_ckpt = self._perform_merge(ckpt_a, ckpt_b, method)
save_checkpoint(merged_ckpt, str(merged_path))
save_checkpoint(merged_ckpt, str(latest_path))  # Double serialization!

# After: Proper cleanup
merged_ckpt = self._perform_merge(ckpt_a, ckpt_b, method)
del ckpt_a, ckpt_b, params_a, params_b  # Free before save
save_checkpoint(merged_ckpt, str(merged_path))
shutil.copy2(str(merged_path), str(latest_path))  # Copy, don't re-serialize
del merged_ckpt  # Free after save
# ... finally block calls jax.clear_caches() and gc.collect()
```

## Testing

1. Run e-dojo with memory monitoring: `watch -n 5 'free -h'`
2. Verify RAM stays stable through multiple merges
3. Check kbcached doesn't grow unbounded
4. Run overnight test to confirm fix

## References

- SAR data: `/var/log/sysstat/sa05`, `/var/log/sysstat/sa06`
- Training logs: `logs/e-dojo/branch_A_bella_20260206_014943.log`
- Merge history: `logs/e-dojo/merge_history.json`
- System boots: `journalctl --list-boots`

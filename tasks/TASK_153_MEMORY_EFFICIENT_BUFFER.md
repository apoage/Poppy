# TASK_153: Memory-Efficient Buffer Management

**Status:** COMPLETE
**Completed:** 2026-02-11

**Objective:** Prevent `RESOURCE_EXHAUSTED` (OOM) errors during Easy Mode training by reducing the "live" memory footprint of the temporal observation buffer and chunking data generation.

## 1. The Mathematical Problem (VRAM Audit)
- **Observation Size**: A temporal observation `(27, 64, 64)` uses **442 KB** (float32).
- **Episode Footprint**: 400 steps × 442 KB = **~177 MB** per episode.
- **The Spike**: Refilling 32 episodes at once requires allocating `(32, 400, 27, 64, 64)` which is **~5.6 GB** in a single JAX array.
- **The Limit**: With `vram_fraction = 0.45` on a 24GB card (RTX 3090), each branch has **~10.8 GB**.
- **Result**: Model weights (14M params) + Adam optimizer state + JIT Overhead + 5.6 GB spike exceeds the 10.8 GB limit, causing immediate OOM.

---

## 2. Technical Implementation Fix

### A. Refill Chunking (`train_easy.py`)
Modify the `_refill_buffer` method to generate episodes in small "micro-batches." This keeps the temporary allocation spike small (under 1GB).

```python
# File: train_easy.py
# Method: OnTheFlyGenerator._refill_buffer

def _refill_buffer(self):
    # ... cleanup exhausted ...
    
    while len(self.buffer) < self.N_BUFFER:
        # CHUNKED REFILL (Fixes TASK_153 OOM)
        # Limit the number of episodes generated in a single GPU call.
        # Max 4 episodes = ~0.7 GB spike, safely within 10.8 GB limit.
        n_to_refill = min(4, self.N_BUFFER - len(self.buffer))
        
        # ... existing logic to fetch prices_list and levels ...
        
        if prices_list:
            # Generate data in small chunks
            episode_datas = self._generate_batch_episodes(prices_list, levels)
            for ep_data in episode_datas:
                self.buffer.append(ep_data)
                self.buffer_sample_counts.append(0)
            
            # Explicit cleanup hint for JAX
            del episode_datas
```

### B. Global Capacity Reduction (`e-dojo_v1.py`)
Lower the default buffer and environment counts to reduce the permanent baseline VRAM footprint.

- **Files to update**: `tools/orchestration/e-dojo_v1.py`
- **Parameter changes**: Change `n_envs` and `buffer_size` from `32` to `16`.
- **Reasoning**: $16 	ext{ episodes} 	imes 177 	ext{ MB} \approx \mathbf{2.8 	ext{ GB}}$ baseline. This leaves ~8 GB for the model and JIT overhead.

---

## 3. Implementation Steps

1.  **Stop Training**: Kill current hung processes.
2.  **Modify `train_easy.py`**: Implementation the chunked `while` loop logic in `_refill_buffer` (hard cap of 4).
3.  **Modify `e-dojo_v1.py`**: Update all default `n_envs` and `buffer_size` values to `16` in `DojoConfig`, `TrainingLauncher`, and CLI arguments.
4.  **Reset & Restart**: Run reset script and launch with new defaults.

## 4. Expected Impact
- **VRAM Stability**: Peak allocation drops from **5.6 GB** to **~0.7 GB**.
- **Execution**: Training will be stable on a single RTX 3090 with dual branches active.
- **Observability**: `agree_trade` and `agree_hold` metrics will now be visible without crashing.

---

## Implementation Notes (2026-02-11)

**Status:** COMPLETE

### Divergence from Original Plan

**Original Plan A** proposed modifying `_refill_buffer()` to hardcode a chunk limit of 4.

**Actual Implementation:** This was unnecessary because `train_easy.py` line 469 already uses:
```python
n_to_refill = min(self.n_envs, self.N_BUFFER - len(self.buffer))
```

The existing code respects `n_envs` as the chunk size limiter. Simply reducing `n_envs` achieves the same goal more elegantly without modifying the refill logic.

### Changes Made (e-dojo_v1.py only)

**Refactored to single source of truth:**
```python
class TrainingLauncher:
    DEFAULT_PARAMS = {
        'preset': 'large',
        'epochs': 2500,
        'steps_per_epoch': 400,
        'batch_size': 1024,
        'n_envs': 4,        # TASK_153: Reduced from 32
        'buffer_size': 16,  # TASK_153: Reduced from 32
        'vram_fraction': 0.45,
    }
```

**Functions updated to use `**kwargs`:**
- `launch_branch()` - accepts `**kwargs`, merges with `DEFAULT_PARAMS`
- `launch_dual_branches()` - passes `**kwargs` through to branches
- `launch_params` dicts - use `**params` spread instead of repeating all fields
- `restart_with_new_levels()` - simplified, uses class defaults

### Result
- Spike: 4 × 173 MB = **0.7 GB** (target: <1 GB)
- Baseline: 16 × 173 MB = **2.8 GB** (target: ~2.8 GB)
- No changes to `train_easy.py` - simpler, less code to maintain
- Single source of truth eliminates repetitive parameter passing

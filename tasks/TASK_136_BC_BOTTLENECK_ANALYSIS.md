# TASK_136: BC Training Bottleneck - Consolidated Analysis

**Date**: 2026-02-02
**Status**: IMPLEMENTED (136d + 136a)
**Investigators**: Claude, Gemini, K (multiple agents)

---

## Problem Summary

```
Train: 144.3s (epoch 1) → 38.9s (epoch 2)
GPU utilization: 5-17% (should be 80%+)
Root cause: BC encoding runs 50× per epoch, each taking ~2.5s
```

---

## Solution Priority (ALL AGENTS AGREE)

| Priority | Task | Speedup | Effort | Status |
|----------|------|---------|--------|--------|
| **1st** | **136d** - Reduce frequency | **2-5×** | **1 line** | **IMPLEMENTED** |
| 2nd | 136a - Pre-encode | 25× BC | Medium | **IMPLEMENTED** |
| 3rd | 136b - Batch call | 4-10× BC | Low | Not needed |
| Skip | 136c - VRAM storage | 1.001× | Medium | NOT WORTH IT |

---

## Key Discoveries

### 136a: ALL Encoder Channels Are Tiled
```python
# jax_encoder.py - EVERY channel does jnp.tile
output = jnp.tile(value[:, -1:], (1, n))  # (n,) → (n, n)
```
**Result**: Can store (27, 64) instead of (27, 64, 64) = 64× compression

### 136b: Existing API Supports Batching
```python
# jax_encoder.py already has this!
jit_batch_encode_simple(prices, window_starts, config, per_env=True)
```
**Result**: Single JAX call for all samples

### 136c: Transfer Is NOT the Bottleneck
```
H→D transfer: 0.25ms (0.12% of batch time)
Python loop: 50ms (25% of batch time)
```
**Result**: Skip VRAM storage, fix Python loop instead

### 136d: BC Runs Too Frequently
```
Current: 50× per epoch @ 2.5s each = 125s
Proposed: 10× per epoch @ 2.5s each = 25s
Saved: 100s per epoch
```
**Result**: Literature shows 20% BC frequency is standard

---

## Implementation Summary

### TASK_136d: BC Frequency Reduction (DONE)
```python
# trainer.py - ExpertBCConfig.bc_frequency = 5
# Reduces BC calls from 50/epoch to 10/epoch
if iteration % bc_freq == 0:
    expert_batch = expert_buffer.sample_batch(...)
```

### TASK_136a: Pre-encode with Compression (DONE)
```python
# jax_encoder.py - Added compressed encoder
encode_observation_compressed()  # Returns (27, n) not (27, n, n)
jit_batch_encode_compressed()    # Batch encode all steps
jit_batch_tile_compressed()      # Expand at sample time

# expert_buffer.py - Pre-encode at add time
preencode_episode()              # Called after add_episode()
_compressed_obs: Dict[int, np.ndarray]  # Storage

# train_jax_fast.py - Callback integration
expert_buffer.preencode_episode(episode_id)
```

### Results
- 64× memory compression: (27, 64, 64) → (27, 64)
- Encoding moved from sample time to add time
- Train time: 144s → ~3-16s (10× faster)

---

## Files Updated

| File | Status |
|------|--------|
| `TASK_136a_PREENCODE_AT_DEMO_TIME.md` | Consolidated from 4 agents |
| `TASK_136b_BATCH_ENCODE_ALL_EPISODES.md` | Consolidated from 3 agents |
| `TASK_136c_VRAM_PRICE_STORAGE.md` | Marked NOT RECOMMENDED |
| `TASK_136d_REDUCE_BC_FREQUENCY.md` | Marked HIGHEST PRIORITY |

---

## Raw Investigation Files

```
136a-k.md                           # K's detailed analysis
TASK_136a-cl_PREENCODE_INVESTIGATION.md   # Claude's analysis
TASK_136a-g.md                      # Gemini's analysis

136b-gemini-solution.md             # Gemini's batch solution
136b-k.md                           # K's two-level vmap
TASK_136b-cl_BATCH_ENCODE_SOLUTION.md     # Claude's analysis

136c-gemini-solution.md             # Gemini: "Low Impact"
136c-k.md                           # K: "NOT RECOMMENDED"
TASK_136c-cl_VRAM_STORAGE_ANALYSIS.md     # Claude: "SKIP"

136d-gemini-solution.md             # Gemini: "High-impact, low-risk"
136d-k.md                           # K: "HIGHEST IMPACT FIX"
TASK_136d-cl_BC_FREQUENCY_ANALYSIS.md     # Claude: "QUICK WIN"
```

---

## Next Action

**Implement TASK_136d now** - 1 line change for 2-5× speedup.

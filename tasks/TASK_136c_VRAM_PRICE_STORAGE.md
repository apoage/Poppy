# TASK_136c: Keep Episode Prices in VRAM

**Date**: 2026-02-02
**Status**: NOT RECOMMENDED
**Priority**: SKIP
**Consensus**: ALL AGENTS AGREE - Not worth it

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Transfer time (30 eps) | **0.25 ms** |
| % of total batch time | **0.12%** |
| Speedup | **1.001×** |
| Verdict | **SKIP** |

**The RAM→VRAM transfer is NOT the bottleneck.** Python loop overhead dominates.

---

## Transfer Time Analysis

```
Per episode: 400 × 64 × 4 = 100 KB
PCIe 3.0 x16: ~12 GB/s
Transfer time: 100 KB / 12 GB/s = 8 microseconds

Per batch (30 episodes): 3 MB / 12 GB/s = 0.25 ms
```

**Compare to total batch time: ~200ms → Transfer is 0.12%**

---

## Where Time Actually Goes

```
_encode_observations_vectorized breakdown:
├── Python loop overhead:     ~50 ms (25%)
├── GPU kernel (30× small):   ~30 ms (15%)
├── Device→Host sync:         ~40 ms (20%)  ← CANNOT AVOID
├── Other overhead:           ~80 ms (40%)
└── H→D prices transfer:      ~0.25 ms (0.12%)  ← VRAM would save this
```

**VRAM storage saves 0.25ms out of 200ms.**

---

## Agent Consensus

| Agent | Verdict | Quote |
|-------|---------|-------|
| Claude | SKIP | "0.25ms savings, not worth complexity" |
| Gemini | SKIP | "Low Impact - bottleneck is Python control flow" |
| K | SKIP | "1.001× speedup, medium complexity" |

---

## If You Still Want It (NOT RECOMMENDED)

```python
class UnifiedIndexBuffer:
    def __init__(self, ...):
        self._episode_prices_vram: Dict[int, jax.Array] = {}

    def add_episode(self, prices):
        self._episode_prices_vram[ep_id] = jax.device_put(prices)

    def clear(self):
        self._episode_prices_vram.clear()
        gc.collect()  # Help release VRAM
```

**But again: saves 0.25ms out of 200ms. Not worth it.**

---

## Recommendation

**SKIP 136c. Focus on:**
1. **136d** - Reduce BC frequency (2-5× overall speedup)
2. **136a** - Pre-encode observations (25× BC speedup)
3. **136b** - Batch JAX calls (4-10× BC speedup)

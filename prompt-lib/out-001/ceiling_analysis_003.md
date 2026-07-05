Claude

# Ceiling Analysis 003: Gradient Suppression + Encoder Bottleneck

**Date**: 2026-02-20
**Context**: Third analysis, focused on two issues missed or underweighted previously.

---

## Critical Finding: Gradient Clipping Mismatch

**Missed in analysis 001 and 002.**

### The Numbers

```
Observed grad_norm: 13-16 (per training logs)
grad_clip threshold: 1.0 (optax.clip_by_global_norm)
Suppression factor: 13-16x every single step
```

The optimizer chain is:
```python
optax.chain(
    optax.clip_by_global_norm(1.0),    # clips first
    optax.adamw(schedule, weight_decay=1e-4)  # sees clipped gradients
)
```

### Why This Matters (nuanced)

With perfectly consistent clipping + AdamW, the second moments track clipped gradients,
so Adam's per-parameter normalization partially cancels the uniform scaling. In the limit
of perfectly consistent clipping, the effect approaches zero.

BUT clipping varies between 13x and 16x (not constant). This has three consequences:

1. **Cross-head gradient competition**: Adaptive trade weighting (24-60x) amplifies
   policy trade gradients. These dominate the global norm, so clipping is driven by
   policy trade loss. Value head and hold gradients get the same suppression factor
   even though they don't need it. Result: **value head learns at policy trade's pace,
   not its own optimal pace.**

2. **Inconsistent effective step size**: 23% variance in suppression factor (13x vs 16x)
   introduces noise in update magnitude. This is distinct from gradient noise — it's
   multiplicative noise on the entire update, correlated across all parameters.

3. **Warmup/schedule interaction**: If grad norms were smaller early on (before trade
   weighting converged to 60x), the model may have learned well initially under less
   clipping, then hit a wall when trade gradients grew and clipping tightened.

### The Fix

**Raise `grad_clip` from 1.0 to 5.0.**

- Suppression drops from 13-16x to 2.6-3.2x (still protective, much less aggressive)
- Value head gets more gradient bandwidth
- Hold gradients less affected by trade gradient magnitude
- Risk: LOW — grad norms at 13-16 are healthy (no NaN/explosion)

---

## Encoder Bottleneck (Confirming analysis 001/002)

### Compression Ratio

```
Input:  (B, 27 channels, 64 timesteps, 64 assets) = 110,592 values/sample
Output: (B, 64) = 64 values/sample
Ratio:  1,728:1
```

The model has exactly **1 latent dimension per asset** in the global state. Cross-asset
attention operates at 32-dim per asset (embed_dim), but the global projection to state_dim=64
crushes this to 1-dim-per-asset equivalent.

### Parameter Distribution

| Component | Params | % |
|-----------|--------|---|
| Encoder (V3) | 0.14M | 1% |
| Reasoning (TRM) | 14M | 96% |
| Dynamics | 0.33M | 2% |
| Heads | ~0.2M | 1% |

**96% of the model's parameters are reasoning about a 64-dim vector.** This is like
putting a PhD mathematician in front of a calculator with a 2-line display — the compute
power is there but the information bandwidth isn't.

### The Fix

**Widen state_dim 64→128, embed_dim 32→64.**

- Encoder gets 2 dims per asset (2x information)
- Cross-asset attention operates at 64-dim (richer interactions)
- Reasoning gets 128-dim input (4x more information per cycle)
- Param increase: ~14.6M → ~35-50M (mostly in reasoning Dense layers)
- VRAM: may need batch_size 1024→512

---

## Ambiguity Data — Rules Out Data/Oracle Issues

From `ambiguity_results.txt`:

| Target Type | >90% confidence | >70% confidence |
|-------------|-----------------|-----------------|
| Trade | 78.2% | 89.8% |
| Hold | 98.6% | 99.5% |

**Theoretical ceiling: 97.1%** — the oracle gives clear signals.
Current model at 61% → **36pp gap is model-limited, not data-limited.**

This confirms: more oracle passes, better thresholds, or data diversity won't help.
The model simply can't learn the patterns it's being shown.

---

## Recommended Fresh Run Configuration

Since state_dim change requires a fresh run anyway, combine all fixes:

```python
# easy_config.py — "large_v2" preset
grad_clip: float = 5.0        # Was 1.0 — stop gradient over-suppression
state_dim: int = 128          # Was 64 — 2x encoder output capacity
embed_dim: int = 64           # Was 32 — 2x minGRU temporal capacity
hidden_dim: int = 512         # Was 256 — widen TRM to match state_dim
num_heads: int = 16           # Was 8 — scale attention with width
batch_size: int = 512         # May need to reduce from 1024 for VRAM
```

### Why combine?

Each fix targets a different axis:
- **clip=5.0**: optimization bandwidth (let gradients flow properly)
- **state_dim=128**: representation bandwidth (more info per asset)
- **embed_dim=64**: temporal bandwidth (richer minGRU hidden)

A fresh run is required regardless. Testing individually would cost 3 restarts × 200+
epochs each. Combined gives the answer in 1 run. If it reaches 70%+, ablation can come
later. If it doesn't, the bottleneck is elsewhere entirely.

### VRAM Estimate

```
Current (state=64, B=1024):  22.7GB
Proposed (state=128, B=512): ~20-24GB (wider model, half batch)
```

Should fit in 24GB. If tight, reduce B to 256 (still viable with trade-biased sampling).

### Monitoring (first 100 epochs)

| Metric | Current | Target | Fail threshold |
|--------|---------|--------|----------------|
| grad_norm | 13-16 | 3-8 | >20 (explosion) |
| agree_trade | 43% | >50% | <35% (regression) |
| balanced_agreement | 61% | >65% | <58% (regression) |
| val_mse | 0.60 | <0.50 | >0.80 (value collapse) |
| VRAM | 22.7GB | <24GB | >23.5GB (OOM risk) |

### Rollback Protocol

1. If VRAM OOM: reduce batch to 256, retry
2. If agreement <58% after 100 epochs: revert state_dim to 64, keep clip=5.0
3. If clip=5.0 alone fails (test by reverting state_dim): try per-head clipping
4. If nothing helps: the bottleneck is in the loss function or data quality

---

## What NOT To Do Yet

| Strategy | Why Not Now |
|----------|-----------|
| Focal loss | Premature — fix gradient flow first. Focal loss + aggressive clipping would double-suppress easy examples |
| Two-stage head | Architecture change. Current arch CAN learn (43% trade proves it). Fix optimization first |
| More oracle passes | Oracle targets are 78-99% confident. More passes just average noise |
| Scale to 100M+ | Wrong axis — encoder is 1% of params, adding reasoning doesn't help |
| Real data mixing | Fix synthetic performance first (61% → 75%), then tackle domain gap |

---

## Summary

Two root causes for the 61% ceiling:

1. **Gradient suppression** (clip=1.0 vs norms=13-16): Policy trade gradients dominate
   global norm, causing 13-16x suppression of ALL parameter updates including value head
   and hold classifier. Fix: clip=5.0.

2. **Encoder bottleneck** (1,728:1 compression): 64-dim state has exactly 1 dimension per
   asset. Cross-asset attention and temporal patterns compressed to near-zero information.
   Fix: state_dim=128, embed_dim=64.

**Expected outcome**: 61% → 70-75% within 200 epochs of fresh combined run.

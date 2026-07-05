# TASK_670B: Expert Imprinting Stability Fixes

## Status: COMPLETE (2026-02-07)

## Priority: HIGH

## Problem Summary

ML audit identified multiple issues causing poor convergence in Easy Mode training:
1. **Latent Drift** - Cached latent states become stale as encoder updates
2. **Turtle Local Optimum** - Agent can learn "never trade" for easy 90% accuracy
3. **Support Resolution** - 2.0 scaling may be too coarse for small gains

## Root Cause Analysis

### 1. Latent Drift (CRITICAL)

**Location:** `train_easy.py` - `set_encoder()` (lines 488-518)

```python
def set_encoder(self, model, params):
    @jax.jit
    def encode_batch(compressed_obs):
        # params captured in closure at initialization!
        latent = model.apply({'params': params}, ...)
        return latent['latent_state']

    self._encoder_fn = encode_batch  # Frozen forever
```

**Issue:**
- `set_encoder()` called ONCE at training start
- Captures encoder params θ₀ in JIT closure
- Buffer episodes encoded with θ₀
- As training updates params → θ₁, θ₂, ..., θₙ
- Cached latents remain from θ₀ (stale!)
- `SAMPLES_PER_EPISODE = 200` means 200 training steps with stale features

**Impact:** Prediction head learning from misaligned features → plateau

### 2. Turtle Local Optimum (CONFIRMED)

**Evidence from historical notes:**
- MCTS with high sims converges to HOLD (~80%) before value function learns
- Entropy stays reasonable (~4.0) but action distribution is HOLD-dominant

**Mechanism:**
- Oracle selects k=4 assets → 4 Buy/Sell, 60 HOLD assignments
- HOLD logits = 0.0, Buy/Sell logits = 5.0
- Agent learns HOLD is "safe" (no cost, no risk)
- Can reach 90% agreement by never trading

### 3. Support Resolution

**Current:**
- Value scaling: 2.0x
- Support range: [-10, 10] with 401 bins
- Bin width: 0.05

**Issue:**
- 1% gain (0.01) × 2.0 = 0.02 < 0.05 bin width
- Small differences invisible to categorical value head

## Proposed Fixes

### Fix 1: Disable Latent Cache

**File:** `tools/orchestration/e-dojo_v1.py`

Add `--no-latent-cache` to branch launcher commands.

**Trade-off:** ~30% slower epochs, but zero latent drift.

### Fix 2: Add Action Distribution Logging

**File:** `easy_trainer.py`

Add metrics in loss_fn:
```python
oracle_hold_ratio = jnp.mean(oracle_labels % 5 == 1)
pred_hold_ratio = jnp.mean(predicted_actions % 5 == 1)
```

**Purpose:** Detect Turtle mode early.

### Fix 3: Parameterize Value Scaling

**Files:** `easy_config.py`, `easy_trainer.py`

Add `value_scale` to config:
```python
value_scale: float = 2.0  # Can tune to 5.0 if needed
```

## Files to Modify

| File | Changes |
|------|---------|
| `tools/orchestration/e-dojo_v1.py` | Add `--no-latent-cache` to branch launch |
| `easy_trainer.py` | Add hold_ratio metrics, use config.value_scale |
| `easy_config.py` | Add value_scale field |

## Validation

After fixes, monitor training for:
- [ ] Value MSE < 5 (not clipped)
- [ ] Hold ratio < 60% (not Turtle)
- [ ] Agreement climbing (not plateaued from drift)

## Related Tasks

- TASK_670: Expert Imprinting Fixes (value target, soft labels, scaling)
- TASK_669: e-dojo Memory Leak (gc.collect fix)
- TASK_667: e-dojo Orchestrator Implementation

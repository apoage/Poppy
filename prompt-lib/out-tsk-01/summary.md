# Implementation Audit — Summary

**Date**: 2026-02-23
**Model**: Claude Opus 4.6
**Tasks Audited**: 21 tasks + 7 bug fixes = 28 items
**Files Read**: easy_config.py, easy_trainer.py, train_easy.py, path_solver.py, muzero_model.py, benchmark_easy.py, trm_flax.py, trm_encoder_v2.py, merge_utils.py, e-dojo_v1.py, dual_dojo.py + 21 task spec files

---

## Counts

- Fully Implemented & Correct: **21** (TASK_147, 149, 151, 674, 675, 677, 679, 685, 686, 689, 691, 692, 693, 695, BUG-1, BUG-2, BUG-4, BUG-5, BUG-6, BUG-7, LATENT)
- Partially Implemented: **2** (TASK_670B, TASK_689)
- Incorrectly Implemented: **2** (TASK_684 wrong value, TASK_680 diluted bias)
- Config Mismatch: **3** (trm_hidden_dim=1024 vs 2048, embed_dim dead for V2, hidden_dim dead field)
- Cross-Task Conflict: **6** interactions identified (see interactions.md)

---

## Top Suspects (most likely to cause agree_trade regression)

### 1. TASK_684: trm_hidden_dim=1024 caps reasoning FFN at half capacity

**Confirmed.** `get_large_config()` sets `trm_hidden_dim=1024`, preventing the reasoning FFN from growing to 2048 when `state_dim` was widened to 128 by TASK_683. ~23M parameters removed. The FFN width — which determines fine-grained trade discrimination — is unchanged from the pre-widening architecture. Both TASK_683 and TASK_684 specs contain the same error: `hidden_dim=1024 # Was state_dim*16=1024` (true only for old state_dim=64).

### 2. TASK_692: Autoregressive state creates turtle feedback loop

**Likely.** Model-state rollout replaces clean oracle state channels with the model's own (initially random, then turtle) predictions. The model must simultaneously learn policy AND state interpretation — a chicken-and-egg problem that creates a self-reinforcing HOLD equilibrium. Pre-TASK_692 runs with oracle state reached 15% agree_trade by epoch 80; post-TASK_692 runs stall at 5-8%.

### 3. TASK_680/687: Trade-biased sampling ~10-20x weaker than intended

**Confirmed bug.** Coin-flip fires once per episode visit (choosing start of a contiguous block of ~20 samples), not once per sample. Expected 30-35% trade windows per batch; actual ~3-8%. Additionally, val_generator uses biased sampling, inflating metrics.

### 4. TASK_683: embed_dim/encoder_num_heads dead for V2 encoder

**Config gap.** Large preset sets `embed_dim=64`, `encoder_num_heads=8`, but V2 encoder (the active default) uses hardcoded `embed_dim=32`, `num_heads=4`. The widened encoder capacity from TASK_683 has zero effect.

### 5. TASK_695: Oracle mask head turtle-trapped

**Minor contributor.** Accuracy 93.7% = all-zeros baseline. Generates anti-trade gradients through shared representation. Loss contribution only 0.8% of total, but direction is harmful.

---

## Recommended Actions (in priority order)

### Fix 1: Restore trm_hidden_dim (1 line, immediate)

```python
# easy_config.py, get_large_config():
trm_hidden_dim=2048,  # Restore: state_dim=128, ratio 16x (was incorrectly capped at 1024)
```

Or remove the line entirely (let it default to `state_dim * 16 = 2048`).

### Fix 2: Oracle state warmup for autoregressive bootstrap

```python
# easy_config.py:
autoregressive_warmup_epochs: int = 100  # Use oracle state first, then switch to model state
```

Implement in `train_easy.py`: use oracle-derived state channels for first N epochs, then transition to model-state rollout. This provides clean bootstrap signal.

### Fix 3: Fix trade-biased sampling per-sample

Current: coin flip once per episode visit → only 1 of ~20 samples guaranteed trade-biased.
Fix: apply coin flip per sample within the contiguous block, or restructure to select independent trade/uniform offsets.

Also: set `val_generator.config.trade_sample_prob = 0.0` for unbiased validation.

### Fix 4: Wire embed_dim/encoder_num_heads to V2 encoder

```python
# trm_flax.py:248 (V2 branch):
self.encoder = AssetTransformerEncoder(
    state_dim=self.state_dim,
    embed_dim=self.embed_dim,    # Currently missing
    num_heads=self.num_heads,     # Currently missing
    dtype=self.dtype,
    param_dtype=self.param_dtype,
)
```

### Fix 5: Enable multipass oracle in e-dojo

Add `--oracle-multipass` to e-dojo branch launcher. Soft targets + temperature sharpening provide partial credit for near-miss trade picks, helping escape the binary pass/fail trap.

### Fix 6: Remove dead config fields

- Delete or deprecate `EasyConfig.hidden_dim` (only `trm_hidden_dim` is used)
- Document that `embed_dim`/`encoder_num_heads` only affect V3 encoder

### Fix 7 (minor): Fix prev_selections 2-way to 3-way

```python
# train_easy.py:532 — change from:
jnp.where(log_returns[:-1] > 0, 1.0, -1.0)
# to:
jnp.where(log_returns[:-1] > 0, 1.0, jnp.where(log_returns[:-1] < 0, -1.0, 0.0))
```

---

## Key Takeaway

The agree_trade regression is caused by the **compound effect** of correctly-implemented features that interact poorly:

1. **Half the reasoning capacity** (TASK_684 capping FFN at 1024 instead of 2048)
2. **Harder learning problem** (TASK_692 model-state rollout instead of oracle state)
3. **Weaker trade signal** (TASK_680 diluted sampling, TASK_695 anti-trade gradients)
4. **No encoder widening benefit** (V2 ignores TASK_683 embed_dim/num_heads)

No single task is "broken." The pathology emerges from the interaction of 4+ tasks that each made reasonable individual decisions but collectively created a near-inescapable turtle equilibrium with insufficient model capacity.

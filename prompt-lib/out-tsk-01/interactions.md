# Cross-Task Interaction Analysis

**Date**: 2026-02-23

---

## Interaction 1: TASK_683 + TASK_684 — Wider State into Same Bottleneck

**Tasks**: TASK_683 (Widen Encoder State), TASK_684 (Decouple hidden_dim)
**Type**: Conflicting specs
**Severity**: **CRITICAL** (primary regression cause)

TASK_683 widened `state_dim` from 64 to 128. Without TASK_684, `hidden_dim` would auto-scale to `128 * 16 = 2048`. TASK_684 capped it at 1024 to prevent "parameter explosion."

The error: TASK_683 spec line 66 says `hidden_dim=1024 # Was state_dim*16=1024`. This comment is only true for the OLD `state_dim=64`. At the new `state_dim=128`, the default would be 2048. Both specs converge on 1024 as the target, but this was based on confused reasoning (carrying the old effective value forward).

**Result**: The model has 2x wider states feeding into a 1x capacity bottleneck. The encoder produces richer representations that the reasoning layers cannot exploit.

| Config | state_dim | Effective FFN | Reasoning Params |
|---|---|---|---|
| Before 683/684 | 64 | 1024 (default 64*16) | ~14M |
| After 683/684 | 128 | 1024 (explicit cap) | ~21M |
| Correct value | 128 | 2048 (default 128*16) | ~44M |

---

## Interaction 2: TASK_692 + Turtle Trap — Self-Reinforcing Feedback Loop

**Tasks**: TASK_692 (Model-State Rollout), TASK_149 (Adaptive Weighting), TASK_695 (Oracle Mask Head)
**Type**: Emergent pathology
**Severity**: **HIGH** (secondary regression cause)

Three correctly-implemented features combine to create a near-inescapable turtle equilibrium:

1. **TASK_692** (autoregressive_state=True): Model's own predictions become state channels.
2. At epoch 0, model is random → predicts mostly HOLD → state channels show flat everywhere.
3. Flat state channels (position=0, PnL=0, heat=0) look like "no positions to manage" → model predicts HOLD.
4. **TASK_695** (oracle mask head): Learns "predict all zeros" from shared representation (93.7% = 93.75% all-zeros baseline). Gradients through shared latent subtly discourage trade-discriminating features.
5. **TASK_149** (adaptive weighting ~25x): Full weight from step 1 (warmup disabled by TASK_677). Large but directionless gradients on a model that hasn't learned basic representations yet.

The loop is self-reinforcing:
```
Model predicts HOLD → state=flat → flat state→HOLD → state=flat → ...
```

Oracle-derived state channels broke this loop by providing clean position/PnL/duration signals. Model-derived channels cannot break it because the model must simultaneously learn both policy AND state interpretation.

---

## Interaction 3: TASK_680 + TASK_687 — Trade Sampling Dilution

**Tasks**: TASK_680 (Trade-Biased Sampling), TASK_687 (Hard Example Mining)
**Type**: Implementation mismatch with spec
**Severity**: **MEDIUM**

TASK_680 designed coin-flip per-sample trade bias. TASK_687 doubled the probability to 0.30, expecting 30-35% trade windows per batch. But the implementation applies the coin flip per-episode-visit, choosing the start of a contiguous block of `take_count ≈ 20` samples.

- **Expected**: 30-35% of batch samples are trade-biased
- **Actual**: ~3-8% of batch samples are trade-biased (only 1 sample per visit guaranteed)

TASK_687's intervention (doubling 0.15→0.30) only increased effective trade bias from ~1.5% to ~3%. This is a 2x improvement on an already-negligible base, not the "double trade example frequency" intended.

Additionally, `val_generator` uses the same biased sampling (both specs warn against this), inflating reported agree_trade.

---

## Interaction 4: TASK_683 embed_dim/encoder_num_heads + V2 Encoder — Dead Config

**Tasks**: TASK_683 (Widen Encoder), TASK_676 (V3 Encoder)
**Type**: Config wiring gap
**Severity**: **MEDIUM** (performance loss, not crash)

TASK_683 set `embed_dim=64` and `encoder_num_heads=8` for the large preset. These are correctly wired through `easy_trainer.py → MuZeroNetworks → TinyRecursiveTradingModel`. However, for `encoder_version=2` (the active default), `trm_flax.py:248` creates `AssetTransformerEncoder` WITHOUT passing `embed_dim` or `num_heads`:

```python
self.encoder = AssetTransformerEncoder(state_dim=self.state_dim, dtype=self.dtype, param_dtype=self.param_dtype)
```

V2 uses its own hardcoded defaults (`embed_dim=32`, `num_heads=4`). The large preset's wider encoder settings (`embed_dim=64`, `num_heads=8`) have zero effect on the currently active V2 encoder.

**Result**: The V2 encoder operates at half the intended temporal capacity (32 vs 64 embed_dim) and half the attention heads (4 vs 8). This further limits the model's ability to extract fine-grained trading signals.

---

## Interaction 5: TASK_673 + TASK_677 — Multipass Disabled, Sharpening Inactive

**Tasks**: TASK_673 (Multipass Oracle), TASK_677 (Temperature Sharpening)
**Type**: Feature dependency not activated
**Severity**: **LOW-MEDIUM**

TASK_677's temperature sharpening and TASK_673's soft targets are designed to work together. But e-dojo does NOT pass `--oracle-multipass`, so:
- Hard one-hot targets (no partial credit for near-miss trades)
- No temperature sharpening
- No noise-based exploration of oracle policy space

The model must match EXACT asset AND action type for agree_trade credit. With `oracle_k=4` out of 64 assets, this is a 4/64 = 6.25% chance per asset even if the model knows which direction to trade. Binary pass/fail makes trade learning harder than necessary.

---

## Interaction 6: TASK_674 Bug6 vs TASK_673 prev_selections — Inconsistent Zero-Return Handling

**Tasks**: TASK_674 (Zero-Return Direction Fix), TASK_673 (Multipass Oracle)
**Type**: Code inconsistency
**Severity**: **VERY LOW**

`train_easy.py:532` (single-pass path) uses 2-way branch: `returns > 0 → 1.0, else → -1.0`. `path_solver.py:302` (multipass path) uses 3-way: `returns > 0 → 1.0, returns < 0 → -1.0, else → 0.0`.

For `returns == 0.0` with a selected asset, single-pass incorrectly says "short" (→ SELL target) while multipass correctly says "flat" (→ HOLD target). Active in current training (single-pass path). Impact negligible (exact-zero returns rare).

---

## Summary: Interaction Severity Ranking

| # | Interaction | Severity | Fix |
|---|---|---|---|
| 1 | TASK_683+684: FFN bottleneck | CRITICAL | Set `trm_hidden_dim=2048` or None |
| 2 | TASK_692+turtle: feedback loop | HIGH | Oracle state warmup then transition |
| 3 | TASK_680+687: sampling dilution | MEDIUM | Per-sample coin flip, uniform val |
| 4 | TASK_683+V2: dead embed_dim | MEDIUM | Pass embed_dim/num_heads to V2 |
| 5 | TASK_673+677: multipass inactive | LOW-MEDIUM | Enable `--oracle-multipass` in e-dojo |
| 6 | TASK_674+673: zero-return inconsistency | VERY LOW | Fix 2-way to 3-way in train_easy.py:532 |

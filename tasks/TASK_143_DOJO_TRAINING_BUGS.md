# TASK_143: Critical E-Dojo Training Bugs

## Status: IMPLEMENTED - Ready for Testing

## Summary
Multiple critical bugs preventing e-dojo training from converging. Agreement stuck at 1%.

---

## SHOW-STOPPER 1: Portfolio vs Single-Action Bottleneck (CRITICAL)

### The Problem
`oracle_k = 10` means the oracle picks 10 assets to trade simultaneously. But the model outputs ONE action per step (argmax over 320 actions).

### The Math
To follow oracle's 10-asset portfolio:
- Agent needs 10 steps to enter all positions
- By step 10, position 1 might already be a SELL signal
- Agent can NEVER match oracle's "instant portfolio"

### The Impact
Training a **Single-Action Trader** to mimic a **10-Asset Portfolio Manager**.
Agreement will always be terrible because the task is architecturally impossible.

### The Fix
```python
# easy_config.py
oracle_k: int = 1  # Single best trade, not portfolio
```

---

## SHOW-STOPPER 2: Representation Drift (Latent Cache)

### The Problem
Training Encoder + Policy simultaneously while using Latent Cache in buffer.

### The Math
1. Generator encodes episode with weights θ₀, stores in buffer
2. Trainer updates weights to θ₅₀ over 200 steps
3. Trainer learns policy for θ₅₀ using features from θ₀
4. Features are STALE - policy learns wrong associations

### The Result
Policy learns on outdated representations. As encoder improves, cached features become meaningless. Model constantly "starts over".

### The Fix
```bash
# Always use --no-latent-cache during imprinting
python train_easy.py --bella --no-latent-cache
```

---

## SHOW-STOPPER 3: Value Scale Zero Bin

### The Problem
`value_scale = 2.0` compresses value targets. If most returns are near 0, all targets fall in "Zero Bin" of categorical distribution.

### The Impact
Value head can't distinguish good from bad states. No learning signal.

### The Fix
```python
# easy_config.py
value_scale: float = 10.0  # Push values out of zero bin
```

---

## SHOW-STOPPER 4: Infinite Brake Loop on Fresh Start

### The Problem
Random weight initialization has cosine ~0.46. Brake threshold is 0.85.
Fresh start → brake triggers → restart → still random → brake → infinite loop.

### The Fix
```python
# e-dojo_v1.py _should_merge()
if cosine < self.config.cosine_brake:
    if self.state.merge_count == 0:
        return False, "fresh_start_skip_brake"
    return True, "divergence_brake"
```

---

## OBSERVABILITY BUG: Asset 0 Argmax Bias

### The Problem
When oracle says "HOLD everything", all 64 HOLD actions get logit=0.0.
`argmax` on a tie returns first index → always Asset 0, Action 1.

### The Impact
- Oracle label for quiet market: always Asset 0 HOLD
- If agent learns HOLD but picks Asset 5: "disagreement"
- Chance of matching: 1/64 = 1.5%

Current agreement metric is literally a random number generator.

### The Fix
Per-asset agreement metric:
```python
# Reshape to (batch, 64, 5)
pred_per_asset = logits.reshape(-1, 64, 5)
oracle_per_asset = targets.reshape(-1, 64, 5)

# Compare action type per asset
pred_type = jnp.argmax(pred_per_asset, axis=-1)
oracle_type = jnp.argmax(oracle_per_asset, axis=-1)
per_asset_agree = jnp.mean(pred_type == oracle_type)
```

---

## Pre-Flight Checklist

| # | Fix | File | Status |
|---|-----|------|--------|
| 1 | `oracle_k = 1` | easy_config.py | DONE |
| 2 | `value_scale = 10.0` | easy_config.py | DONE |
| 3 | `--no-latent-cache` default | e-dojo_v1.py | DONE (already present) |
| 4 | `merge_count == 0` brake skip | e-dojo_v1.py | DONE |
| 5 | Per-asset agreement metric | easy_trainer.py | DONE |

---

## Root Cause Summary

**Why 1% agreement?**
1. Model outputs 1 action, oracle expects 10 → impossible match
2. Argmax tiebreaker is arbitrary → "HOLD everywhere" mismatch
3. Cached features are stale → policy learns on wrong data
4. Value targets compressed → no learning signal
5. Brake loop kills Branch B → only Branch A trains (partially)

**The real agreement is unmeasurable with current metric.**

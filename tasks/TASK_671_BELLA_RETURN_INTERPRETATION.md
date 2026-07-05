# TASK_671: Bella Return Interpretation Bug

**Date**: 2026-02-07
**Status**: COMPLETE
**Priority**: CRITICAL - Blocks Bella branch learning
**Related**: TASK_666, TASK_667, TASK_670

---

## Problem Statement

Bella branch shows 2-3% oracle agreement vs Synthetic's 16-18% (10x gap). Root cause analysis reveals a **conceptual bug** in how Bella's diffusion output is interpreted.

---

## Root Cause Analysis

### What Bella Outputs

From `diff_teacher/data_loader.py:400`:
```python
normalized = prices / baseline - 1.0  # baseline = prices[:, 0:1, :, :]
```

This produces **cumulative returns from window start**, NOT step returns:
- Value at t=0: `0.0` (baseline)
- Value at t=10: `0.05` means price is 5% above baseline

### What setup.py Does (WRONG)

From `mcts_trader/training/setup.py:650`:
```python
log_prices = jnp.cumsum(close_prices * 0.01, axis=1)
prices = 100.0 * jnp.exp(log_prices)
```

This **cumsums cumulative returns**, which is mathematically nonsense:
- Input: `[0.0, 0.02, 0.05, 0.03]` (cumulative returns)
- After `* 0.01`: `[0, 0.0002, 0.0005, 0.0003]`
- After cumsum: `[0, 0.0002, 0.0007, 0.001]`
- After exp: prices barely move (< 0.1% total)

### Observed Symptoms

| Symptom | Bella | Synthetic | Explanation |
|---------|-------|-----------|-------------|
| Oracle Agreement | 2-3% | 16-18% | Random (no signal) |
| Top-5 Accuracy | ~40% | ~92% | Near random baseline (28%) |
| Gradient Norm | Near zero | Normal | No learning signal |
| Loss Jitter | 10x less | Normal | Flat loss landscape |
| Value MSE | Very low | Normal | Returns too small to learn |

---

## Solution

### Initial Fix (v1) - INCORRECT

First attempt assumed Bella outputs cumulative returns:
```python
prices = 100.0 * (1.0 + close_prices)  # WRONG - caused wild swings
```

**Diagnostic revealed:** Bella outputs z-scored values (std ~1.0, range [-3, 11])

### Corrected Fix (v2) - CURRENT

Scale z-scored output to realistic step returns:
```python
# Bella outputs standardized values (std ~1.0)
# Scale to ~1.5% step volatility (matching Synthetic L0)
volatility_scale = 0.015
step_returns = close_prices * volatility_scale
log_prices = jnp.cumsum(step_returns, axis=1)
prices = 100.0 * jnp.exp(log_prices)
```

**Math:**
- Bella output std: ~1.0
- Target step return std: ~0.015 (1.5%)
- Scale factor: 0.015 / 1.0 = 0.015

---

## Implementation Plan

### Step 1: Add Diagnostics

Add to `generate_bella_prices_on_device()` in `setup.py`:
```python
# After extracting close_prices:
print(f"[BELLA DEBUG] close_prices range: [{float(close_prices.min()):.4f}, {float(close_prices.max()):.4f}]")
print(f"[BELLA DEBUG] close_prices mean: {float(close_prices.mean()):.4f}")
print(f"[BELLA DEBUG] close_prices std: {float(close_prices.std()):.4f}")
```

### Step 2: Verify Hypothesis

Expected output:
- Range: approximately `[-0.1, 0.1]` (±10% from baseline)
- Mean: close to 0
- Std: 0.02-0.05

### Step 3: Implement Fix

Replace lines 649-654 in `setup.py`:
```python
# OLD (WRONG):
# log_prices = jnp.cumsum(close_prices * 0.01, axis=1)
# prices = 100.0 * jnp.exp(log_prices)

# NEW (CORRECT):
# Bella outputs cumulative returns (prices/baseline - 1.0)
# Reconstruct: prices = baseline * (1 + cumulative_return)
prices = 100.0 * (1.0 + close_prices)
prices = jnp.maximum(prices, 1.0)  # Ensure positive
```

### Step 4: Verify Fix

After fix, check:
- Price range: should see ±5-10% swings
- Step returns: should be 0.5-2% per step
- Gradient norm: should be normal (not near-zero)
- Oracle agreement: should improve to 10-20%

---

## Files to Modify

| File | Line | Change |
|------|------|--------|
| `mcts_trader/training/setup.py` | 646-654 | Fix price interpretation |

---

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Bella Agreement | 2-3% | 10-20%+ |
| Price Dynamics | Flat | Real (±5-10%) |
| Gradient Norm | ~0 | Normal |
| Loss Jitter | Low | Normal |

---

## Verification Commands

```bash
# Run with diagnostics
python train_easy.py --bella --epochs 10 --debug

# Check logs for BELLA DEBUG output
grep "BELLA DEBUG" logs/e-dojo/*.log
```

---

## Notes

- This is a silent data pipeline bug - training "works" but learns nothing useful
- Similar bugs may exist in other Bella integration points
- Consider adding permanent validation for return magnitudes

---

## Status Log

- 2026-02-07: Issue identified during e-dojo training audit
- 2026-02-07: Root cause confirmed (cumsum of cumulative returns)
- 2026-02-07: Task created, implementing fix

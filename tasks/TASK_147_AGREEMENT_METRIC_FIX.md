# TASK_147: Agreement Metric Semantic Fix

**Status:** COMPLETE

**Created:** 2026-02-10
**Completed:** 2026-02-10

---

## The Problem

The agreement metric measures the **wrong thing**.

### Current Interpretation (BROKEN)

```
Oracle output per step: 64 action types (1 per asset)
Model output per step:  64 action types (1 per asset)
Agreement = mean(pred[i] == oracle[i] for i in 0..63)
```

With k=1 oracle:
- Oracle: BUY asset 7, HOLD assets 0-6 and 8-63
- Model (lazy): HOLD all 64 assets
- Agreement: 63/64 = **98.4%** ← WRONG!

### What The Oracle Actually Says

With k=1, the oracle's output is really:

```
"At this step, the optimal action is: TRADE asset 7 with BUY"
```

The 63 HOLDs are not 63 separate decisions - they're just "not this asset". The oracle makes **1 decision per step**: which asset to trade and how.

### Correct Interpretation

The step-level action is:
```
action = (asset_index, action_type)  # e.g., (7, BUY)
```

Agreement should measure:
1. Did the model pick the **correct asset**?
2. Did the model pick the **correct action type** for that asset?

If oracle says "BUY asset 7" and model says "HOLD everything":
- Asset match: NO (model didn't identify asset 7)
- Action match: NO (model said HOLD, not BUY)
- Agreement: **0%** ← CORRECT!

---

## Root Cause

The 64-asset output format was designed for multi-asset portfolios (k>1). But with k=1, we have a **single-action selection problem**, not 64 independent classifications.

The metric treats "HOLD asset 0" as a meaningful prediction when it's really just absence of signal.

---

## Two Possible Fixes

### Option A: Trade-Only Agreement

Only measure agreement on oracle's trade signals:

```python
oracle_trades = oracle_action_type != HOLD  # Which assets oracle wants to trade
trade_agreement = mean(pred[oracle_trades] == oracle[oracle_trades])
```

- When oracle says TRADE, did model agree?
- Ignores HOLD matching entirely
- Simple, but doesn't penalize false positives (model trades wrong asset)

### Option B: Step-Level Action Agreement

Treat the entire step as one action selection:

```python
# Oracle's action: which asset and what action
oracle_trade_asset = argmax(oracle_action_type != HOLD)  # Asset index
oracle_trade_action = oracle_action_type[oracle_trade_asset]  # Action type

# Model's action: which asset it wants to trade most
model_trade_asset = argmax(model_confidence_to_trade)  # Asset index
model_trade_action = pred_action_type[model_trade_asset]

# Agreement: both asset AND action must match
agreement = (oracle_trade_asset == model_trade_asset) &
            (oracle_trade_action == model_trade_action)
```

- Measures complete decision accuracy
- Penalizes wrong asset selection
- More faithful to the actual task

---

## Recommended Fix: Balanced Accuracy (Trade/Hold Split)

After adversarial review, Option B has a False Positive Trap (argmax forces a trade pick even when model is 99% HOLD everywhere).

**Balanced Accuracy** is more robust and works for any k:

### Implementation

```python
# TASK_147: Balanced Accuracy (Trade/Hold split)
# Prevents "turtle" (all HOLD) from gaming the metric

# Masks for trade vs hold
oracle_is_trade = oracle_action_type != 1  # (batch, 64) - True where oracle trades
oracle_is_hold = oracle_action_type == 1   # (batch, 64) - True where oracle holds

# Agreement on TRADE signals only
trade_correct = (pred_action_type == oracle_action_type) & oracle_is_trade
n_trades = jnp.sum(oracle_is_trade)
agree_trade = jnp.where(n_trades > 0, jnp.sum(trade_correct) / n_trades, 0.0)

# Agreement on HOLD signals only
hold_correct = (pred_action_type == oracle_action_type) & oracle_is_hold
n_holds = jnp.sum(oracle_is_hold)
agree_hold = jnp.where(n_holds > 0, jnp.sum(hold_correct) / n_holds, 0.0)

# Balanced accuracy: equal weight to trade and hold performance
balanced_agreement = 0.5 * agree_trade + 0.5 * agree_hold
```

### Why This Works

| Model Behavior | agree_trade | agree_hold | Balanced |
|----------------|-------------|------------|----------|
| Turtle (all HOLD) | 0% | 100% | **50%** |
| Random | ~20% | ~80% | **~50%** |
| Learning | ↑ | stable | **> 50%** |
| Perfect | 100% | 100% | **100%** |

- **Honest**: Turtle can't game above 50%
- **Diagnostic**: See if model over-trades or under-trades
- **Future-proof**: Works for k=1, k=10, or k=64

---

## Expected Impact

| Metric | Before (Broken) | After (Fixed) |
|--------|-----------------|---------------|
| Fresh model | ~98.4% (HOLD spam) | ~50% (baseline) |
| Turtle model | ~98.4% (fake success) | ~50% (honest) |
| Learning model | 98.5% (no signal) | Should climb > 50% |
| Success threshold | 99.5% (trivial) | 75%+ (meaningful) |

---

## Files to Modify

- `easy_trainer.py`: Add `agree_trade`, `agree_hold`, `balanced_agreement`
- `easy_config.py`: Change `target_oracle_agreement` to 0.75 (meaningful)
- `train_easy.py`: Update logging to show balanced metric

---

## Future: Precision/Recall for k > 1

If we return to k>1, extend with:
- **Recall**: Of oracle's trades, how many did model trade correctly?
- **Precision**: Of model's trades, how many did oracle actually want?
- **F1-Score**: Harmonic mean (industry standard for imbalanced problems)

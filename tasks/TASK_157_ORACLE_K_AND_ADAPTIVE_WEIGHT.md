# TASK_157: Oracle K Configuration and Adaptive Weight Analysis

**Status:** FIX IMPLEMENTED (Option B)
**Priority:** HIGH
**Created:** 2026-02-12
**Related:** TASK_156 (Action Representation Bug), TASK_149 (Adaptive Weighting), TASK_145 (k=1 Constraint)

---

## Validation Results (2026-02-12)

| Metric | Original Estimate | Actual Benchmark | Status |
|--------|------------------|------------------|--------|
| Oracle Trade Rate | ~1% | **0.6%** | Even lower |
| Model Trade Rate | 10% | **15.6% (26× oracle)** | Worse than expected |
| Trade Precision | ~1% | **0.4%** | Near random |
| Adaptive Weight | 63× | **~166×** | Even more extreme |

---

## Executive Summary

Model overtrading (15.6% vs oracle's 0.6%) with near-random precision (0.4%) is caused by:
1. **Conservative oracle** (`k=1`) produces only 0.6% trade signal
2. **Aggressive loss weighting** (TASK_149) upweights trades ~166×
3. **Result**: Model learns "trading is important" but not "when to trade"

### Critical Constraint (TASK_145)

```python
# easy_config.py:51-52
# TASK_145: k=1 because model executes 1 trade/step. Value head must match.
# k>1 would train value head on unrealizable 10-asset profits → MCTS collapse.
```

**Option A (increase k) is NOT safe without validating this constraint.**

---

## 1. Oracle Configuration Analysis

### Current Settings (from checkpoint)

```python
# easy_config.py / checkpoint config
oracle_k: 1              # Select only 1 asset per timestep
oracle_lookahead: 20     # Look 20 steps ahead
min_alpha_threshold: 0.002  # Min 0.2% return to trade (2× txn cost)
transaction_cost: 0.001  # 0.1% per trade
state_aware_oracle: True # TASK_141 transition-based actions
n_assets: 64             # 64 assets total
```

### Trade Rate Calculation

With `k=1` selecting 1 out of 64 assets per step:
- **Base selection rate**: 1/64 = 1.5625% per asset per step
- **After threshold filtering**: ~1.0% (some selections filtered by min_alpha)
- **Resulting action distribution**: 99% HOLD, 1% TRADE

### Expected vs Actual

| Expectation | Value | Source |
|-------------|-------|--------|
| User expected HOLD | ~85% max | Conversation |
| Actual oracle HOLD | **99%** | Benchmark result |
| Required k for 85% HOLD | ~10 | Math: 10/64 = 15.6% trade |

---

## 2. Adaptive Weighting Analysis (TASK_149)

### Location
`easy_trainer.py` lines 154-173

### Mechanism

```python
# TASK_149: Adaptive class balancing - trades and holds have equal gradient mass
is_trade = targets_reshaped[:, :, 1] < 0.5  # NOT HOLD (bool mask)

# Count trades and holds in this batch
n_trades = jnp.maximum(jnp.sum(is_trade), 1.0)
n_holds = jnp.maximum(jnp.sum(~is_trade), 1.0)

# Target weight for 50/50 balance: e.g., 63 holds / 1 trade = 63.0
target_trade_weight = n_holds / n_trades

# TASK_151: Warmup ramping
warmup_progress = batch.get('warmup_progress', jnp.array(1.0))
adaptive_trade_weight = 1.0 + (target_trade_weight - 1.0) * warmup_progress

# Apply weights to per-asset cross entropy
loss_weights = jnp.where(is_trade, adaptive_trade_weight, 1.0)
policy_loss = jnp.sum(per_asset_ce * loss_weights) / jnp.sum(loss_weights)
```

### Weight Calculation

With k=1 and 64 assets:
- Per batch: ~1% trades, ~99% holds
- `n_holds / n_trades` ≈ 63 / 1 = **63×**
- After TASK_151 warmup (epoch 50+): full 63× weight applied

### The Problem

| What Training Says | What Model Learns |
|-------------------|-------------------|
| "Trades get 63× gradient" | "Trading is very important" |
| "Here's 1% trade examples" | "I should trade more than 1%" |
| "Oracle is conservative" | "But loss says trades matter!" |

**Result**: Model outputs 10% trades (10× oracle rate) but with only 1.1% precision.

---

## 3. Imprinting Paradox

### Why Behavioral Cloning Failed

Standard imprinting (BC) should reproduce oracle behavior. But:

1. **Data imbalance**: 99% HOLD, 1% TRADE
2. **Loss correction**: TASK_149 multiplies trade loss by 63×
3. **Unintended effect**: Model over-corrects, spams trades

### Analogy

Imagine training a spam filter:
- Data: 99% ham, 1% spam
- Loss: Weight spam 99× to "balance"
- Result: Filter becomes spam-happy, flags everything

Same dynamic here. The adaptive weight overcorrects.

---

## 4. Benchmark Evidence

### From benchmark_oracle_vs_model.py (post-TASK_156 fix)

```
Oracle Trade Rate:  1.0% of actions
Model Trade Rate:   10.1% of actions  (10× oracle!)
Trade Precision:    1.1%              (random would be ~1%)
Trade Recall:       10.6%

Per-Action Agreement:
  SELL: Oracle 0.5%, Model 7.9%, Agreement 15.5%
  HOLD: Oracle 99.0%, Model 89.9%, Agreement 90.0%
  BUY:  Oracle 0.5%, Model 2.2%, Agreement 5.1%
```

### Interpretation

- Model learned to output HOLD ~90% (close to oracle's 99%)
- Model learned to output trades ~10% (10× oracle's 1%)
- But trades are nearly random (1.1% precision ≈ chance)

The adaptive weighting succeeded in making model trade MORE, but failed to teach WHEN to trade.

---

## 5. Possible Fixes

### Option A: Increase Oracle K

```python
# easy_config.py
oracle_k: int = 10  # Was 1, now 10
```

**Effect**:
- Oracle selects 10/64 = 15.6% per step
- ~85% HOLD, ~15% TRADE
- Adaptive weight drops to ~6× (instead of 63×)
- More trade examples for model to learn from

**Pros**: More signal, less extreme weighting
**Cons**: Changes oracle behavior, may need retraining

### Option B: Cap Adaptive Weight

```python
# easy_trainer.py
MAX_TRADE_WEIGHT = 10.0  # Cap at 10×, not 63×
target_trade_weight = jnp.minimum(n_holds / n_trades, MAX_TRADE_WEIGHT)
```

**Effect**:
- Trade weight capped at 10×
- Less aggressive over-correction
- Model still sees trade importance but not extreme

**Pros**: Simple fix, doesn't change oracle
**Cons**: May still underweight rare trades

### Option C: Focal Loss

Replace cross-entropy with focal loss that naturally handles imbalance:

```python
# Focal loss: (1 - p_t)^gamma * CE
gamma = 2.0  # Focus on hard examples
focal_weight = (1 - correct_prob) ** gamma
policy_loss = jnp.mean(focal_weight * per_asset_ce)
```

**Effect**:
- Hard examples (trades) get more weight automatically
- No manual tuning needed
- Well-studied for imbalanced classification

**Pros**: Principled, adaptive
**Cons**: More complex, needs tuning gamma

### Option D: Separate Trade/Hold Heads

Architecture change: two output heads
- Head 1: Should I trade this asset? (binary)
- Head 2: If trading, BUY or SELL? (binary)

**Effect**:
- Decouples "trade vs hold" from "direction"
- Each head has better class balance
- More interpretable

**Pros**: Better separation of concerns
**Cons**: Architecture change, retraining needed

---

## 6. Recommended Action (VALIDATED)

### Fix Order (Auditor Approved)

| Priority | Option | Action | Risk |
|----------|--------|--------|------|
| 1 | **B** | Cap adaptive weight at 10× | LOW - safest |
| 2 | - | Re-benchmark to measure improvement | - |
| 3 | **C** | Focal loss with γ=2.0 if needed | LOW |
| 4 | **A** | Increase k (ONLY after TASK_145 review) | HIGH |
| 5 | **D** | Separate heads (next architecture) | MEDIUM |

### Implementation: Option B (Cap Weight)

```python
# easy_trainer.py line ~163
MAX_TRADE_WEIGHT = 10.0  # Cap at 10×, not 166×
target_trade_weight = jnp.minimum(n_holds / n_trades, MAX_TRADE_WEIGHT)
```

### Why Option A is Risky

TASK_145 explicitly warns:
> "k>1 would train value head on unrealizable 10-asset profits → MCTS collapse"

The value head is trained on remaining P&L assuming k trades execute. If k=10 but model can only execute 1 trade/step, the value predictions become unrealizable → MCTS planning breaks.

---

## 7. Files Involved

| File | What to Check/Change |
|------|---------------------|
| `easy_config.py` | `oracle_k`, `min_alpha_threshold` |
| `easy_trainer.py:154-173` | Adaptive weight calculation |
| `mcts_trader/jax_agent/path_solver.py` | Oracle selection logic |
| `tools/benchmark_oracle_vs_model.py` | Validation metrics |

---

## 8. Validation Checklist

- [x] Confirm oracle trade rate is ~1% (k=1 behavior) - **Actually 0.6%**
- [x] Check `adaptive_trade_weight` in training logs - **Was ~166×**
- [x] Verify `agree_trade` metric is low despite high `oracle_agreement` - **0.4% precision**
- [ ] ~~Test with k=5 to see if more signal helps~~ - **BLOCKED by TASK_145 constraint**
- [x] Test with capped weight (10×) to see if precision improves - **IMPLEMENTED**

## 10. Implementation Log

### 2026-02-12: Option B Implemented

**File:** `easy_trainer.py` line ~163

**Change:**
```python
# TASK_157: Cap at 10× to prevent overtrading from extreme imbalance
MAX_TRADE_WEIGHT = 10.0
target_trade_weight = jnp.minimum(n_holds / n_trades, MAX_TRADE_WEIGHT)
```

**Expected Effect:**
- Adaptive weight: 166× → 10× (capped)
- Model should reduce overtrading from 15.6% toward oracle's 0.6%
- Precision should improve as model becomes more selective

### 2026-02-12: Threshold Lowered (Signal Increase)

**File:** `easy_config.py` line 57

**Change:**
```python
min_alpha_threshold: float = 0.001  # Was 0.002
```

**Rationale:**
- Effective threshold = min_alpha_threshold × lookahead
- Before: 0.002 × 20 = 4% cumulative return (very selective → 0.6% trades)
- After: 0.001 × 20 = 2% cumulative return (~2× more trades expected)

**Expected Effect:**
- Oracle trade rate: 0.6% → ~1-2%
- More trade examples for model to learn from
- Keeps k=1 (TASK_145 safe)

**Next:** Retrain and re-benchmark to validate improvement.

---

## 9. Related Tasks

| Task | Relationship |
|------|-------------|
| TASK_141 | State-aware oracle (works correctly) |
| TASK_145 | k=1 decision (may need revisiting) |
| TASK_149 | Adaptive weighting (root cause) |
| TASK_151 | Weight warmup (helps but doesn't fix) |
| TASK_156 | Benchmark fix (exposed this issue) |

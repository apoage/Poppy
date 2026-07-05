# TASK_156: Action Representation Bug Investigation

**Status:** BENCHMARK FIXED
**Priority:** CRITICAL
**Created:** 2026-02-12

**Symptom:** Model trained on Bella data performs WORSE on Bella validation than on synthetic data it has never seen.

---

## FIX APPLIED (2026-02-12)

Implemented state-aware oracle conversion in `benchmark_oracle_vs_model.py`:

| Change | Description |
|--------|-------------|
| `oracle_selections_to_actions_single()` | Tracks position state, uses transition-based actions |
| `simulate_model_pnl()` | Uses state-aware position transitions |

### Results After Fix

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| BUY Agreement | **0%** | **5.1%** |
| SELL Agreement | 22.8% | 15.5% |
| Model can output BUY | NO | YES |

**Conclusion:** Benchmark bug FIXED. BUY blindness resolved - model now outputs BUY actions.

Remaining issue: **Model training quality** - still only 5.1% BUY agreement indicates model hasn't learned strong directional signals. This is a training issue, not benchmark bug.

---

## ROOT CAUSE: State-Aware Oracle Mismatch

| Component | Oracle Mode | Location |
|-----------|-------------|----------|
| Training | State-aware (`state_aware_oracle=True`) | `easy_config.py:58` |
| Benchmark | Stateless (hardcoded) | `benchmark_oracle_vs_model.py:246-267` |

### The Bug

**Training uses state-aware oracle (TASK_141)** with transition-based actions:

| Action | Meaning (State-Aware) |
|--------|----------------------|
| BUY | Open long, close short, or reverse short→long |
| SELL | Open short, close long, or reverse long→short |
| HOLD | Maintain current position or stay flat |

**Benchmark uses stateless oracle** with return-sign-based actions:

| Action | Meaning (Stateless) |
|--------|---------------------|
| BUY | Selected + positive return |
| SELL | Selected + negative return |
| HOLD | Not selected |

### Why This Causes the Symptoms

1. **Model NEVER outputs BUY on Bella (0%)**
   - Model learned: "BUY = close short position"
   - Benchmark oracle: "BUY = selected + positive return"
   - These rarely align!

2. **Inverted performance (better on unseen synthetic)**
   - Synthetic data is simpler, accidentally creates more overlap with stateless logic
   - Bella has more structure, exposing the mismatch

3. **Severe HOLD bias on Bella**
   - State-aware HOLD = "maintain position or stay flat"
   - Many scenarios where benchmark expects BUY/SELL, model correctly outputs HOLD

---

---

## 1. Observed Behavior

### Benchmark Results
| Metric | Synthetic (never seen) | Bella (trained on) |
|--------|------------------------|-------------------|
| Model P&L | +2721% | **-3597%** |
| Model BUY rate | 5.8% | **2.1%** |
| Model SELL rate | 9.5% | 6.1% |

### Directional Analysis
**When Oracle says BUY:**
| Model Response | Synthetic | Bella |
|----------------|-----------|-------|
| SELL (WRONG) | 47.4% | 19.1% |
| HOLD | 41.6% | 80.9% |
| BUY (CORRECT) | **11.0%** | **0%** |

**When Oracle says SELL:**
| Model Response | Synthetic | Bella |
|----------------|-----------|-------|
| SELL (CORRECT) | 68.5% | 22.8% |
| HOLD | 30.8% | 77.2% |
| BUY (WRONG) | 0.7% | 0% |

### Key Anomalies
1. **Model NEVER outputs BUY on Bella** - 0% BUY when oracle says BUY
2. **Inverted performance** - Better on unseen data than training data
3. **Severe SELL bias** - Model prefers SELL over BUY on both datasets
4. **Extreme conservatism on Bella** - 92% HOLD vs 85% on synthetic

---

## 2. Hypotheses

### H1: Action Index Mismatch
Training encodes actions differently than inference interprets them.
- Training: BUY=2, SELL=0, HOLD=1
- Inference: Different mapping?

**Check:** Verify action indices in `path_solver.py` vs `benchmark_oracle_vs_model.py`

### H2: Soft Label vs Hard Action Confusion
Training uses soft policy targets (320-dim softmax), benchmark uses argmax.
- Soft labels might have different peak locations
- Temperature scaling could shift argmax results

**Check:** Compare `optimal_path_to_policy` output distribution vs model output

### H3: Observation Encoding Mismatch
Bella observations encoded differently during training vs benchmark.
- Different normalization?
- Different channel ordering?

**Check:** Compare `jit_batch_encode_temporal` usage in `train_easy.py` vs benchmark

### H4: Return Sign Inversion
Log returns computed differently causing direction flip.
- Training: `log(p[t+1]/p[t])`
- Benchmark: `log(p[t]/p[t-1])`?

**Check:** Verify return computation alignment

### H5: Window Alignment Bug
Off-by-one in window indexing causing temporal misalignment.
- Observation at t predicts action for returns[t+1]?
- But benchmark uses returns[t]?

**Check:** Trace exact alignment in training vs benchmark

### H6: Bella-Specific Processing
Something in Bella data pipeline differs from training.
- `generate_bella_prices_on_device` postprocessing?
- Scaling factors?

**Check:** Compare Bella price generation in training vs benchmark

---

## 3. Investigation Plan

### Step 1: Action Index Verification
```python
# In path_solver.py - verify action layout
buy_indices = jnp.arange(n_assets) * n_actions_per_asset + 2   # action_type=2
sell_indices = jnp.arange(n_assets) * n_actions_per_asset + 0  # action_type=0
hold_indices = jnp.arange(n_assets) * n_actions_per_asset + 1  # action_type=1

# In benchmark - verify interpretation matches
```

### Step 2: Soft Label Analysis
```python
# Get oracle's soft policy target for a known BUY case
# Check which action indices have high probability
# Compare to model output for same observation
```

### Step 3: Training Data Inspection
```python
# Load a training batch from train_easy.py
# Check action target distribution
# Verify BUY actions are present and correctly encoded
```

### Step 4: Observation Alignment
```python
# For same prices:
# 1. Generate observation in train_easy.py style
# 2. Generate observation in benchmark style
# 3. Compare byte-for-byte
```

### Step 5: End-to-End Trace
```python
# For one specific timestep:
# 1. Log prices, returns, oracle selection
# 2. Log policy target (320-dim)
# 3. Log model output (320-dim)
# 4. Log argmax action
# 5. Verify chain is consistent
```

---

## 4. Files to Investigate

| File | What to Check |
|------|---------------|
| `train_easy.py` | How observations and targets are generated |
| `mcts_trader/jax_agent/path_solver.py` | `optimal_path_to_policy` action encoding |
| `tools/benchmark_oracle_vs_model.py` | Action interpretation |
| `mcts_trader/env/jax_encoder.py` | Observation encoding consistency |
| `mcts_trader/training/setup.py` | `generate_bella_prices_on_device` |

---

## 5. Success Criteria

- [x] Identify root cause of inverted performance (state-aware mismatch)
- [x] Fix action representation bug (oracle_selections_to_actions_single)
- [x] Model BUY rate > 0% on Bella validation (now 5.1%, was 0%)
- [ ] Model P&L positive on Bella validation (still negative - training issue)
- [x] Benchmark shows consistent behavior across data sources

## 6. Remaining Training Issues

The benchmark is now correct. Remaining issues are MODEL TRAINING quality:

1. **Low directional accuracy**: 5.1% BUY, 15.5% SELL agreement
2. **Model ignores oracle signal**: Same SELL/BUY rates whether oracle says BUY or SELL
3. **Overtrading**: Model trades 10.1% vs oracle's 1.0%

These indicate the model learned HOLD dominance but not directional signals.

### Recommendations
- Continue training longer (currently at epoch 124, target 2500)
- Consider adjusting training curriculum/difficulty
- May need architecture changes to improve directional learning

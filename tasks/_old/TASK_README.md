# Active Tasks

**Last Updated**: December 21, 2025

---

## Pending Tasks (Priority Order)

### 1. Transformer Migration for TRM v2 [HIGH]
**Status**: Design phase
**File**: `TASK_08_TRANSFORMER_TRM_V2.md`

Major architectural upgrade:
- Replace TRM with standard Transformer blocks
- **FP16 Tensor Cores** (235 vs 29 TFLOPS = 8× potential)
- **50% VRAM reduction** (bf16 weights/activations)
- Flash Attention integration
- Explicit multi-asset attention (64 tokens)

**Breaks checkpoints**: YES
**Fallback**: TRM preserved as option

---

### 2. Portfolio Curriculum - Levels 4-6 [HIGH]
**Status**: Not started
**File**: `TASK_04_PORTFOLIO_CURRICULUM.md`

Implement advanced training levels:
- **Level 4**: Portfolio Rebalancer (rotation tactics)
- **Level 5**: Adversarial Market (robustness)
- **Level 6**: Chaos Mode (black swans)

---

### 3. Prediction Heads [MEDIUM]
**Status**: Not started
**File**: `TASK_05_PREDICTION_HEADS.md`

Add auxiliary prediction heads (price, volatility, regime) to force
latent space to encode "what will happen" not just "what to do".

---

### 4. Portfolio Morphing Extension [MEDIUM]
**Status**: Not started
**File**: `TASK_02_PORTFOLIO_MORPHING_EXTENSION.md`

Enable direct asset-to-asset swaps (BTC→ETH) instead of going through cash.
50% fee savings on rebalancing trades.

---

### 5. Advanced Orders & Zero Action [MEDIUM]
**Status**: Design complete
**File**: `TASK_07_ADVANCED_ORDERS.md`

Extend action space with:
- **Limit orders**: Maker orders with rebates (negative fees!)
- **Stop orders**: Breakout/breakdown triggers
- **Zero action**: Global "skip step" with anti-abuse mechanisms

Key insight: Limit orders can earn rebates instead of paying fees.

---

### 6. Document Tools Directory [LOW]
**Status**: Not started
**File**: `TASK_06_DOCUMENT_TOOLS.md`

Add documentation for 36 tools in `tools/` directory.

---

### 7. Action Masking in MCTS Children [LOW]
**Status**: Audit needed

Check if child nodes in MCTS tree properly mask invalid actions.

---

### 8. Root Reward Consistency [LOW]
**Status**: Verify

Ensure `initial_inference` and `_scalar_to_categorical` are consistent for reward=0.

---

## Completed Tasks

### Architecture Fixes (Dec 21)
- ✅ Value Head → uses refined latent
- ✅ Dynamics LayerNorm → stabilizes MCTS
- ✅ Replay Buffer → wrap-around fix
- ✅ Cross-validation framework
- ✅ MC Returns bug fix
- ✅ Adaptive entropy floor

### Archived (in `_old/tasks_completed/`)
- TASK_01: GPU-Native MCTS (Dec 19)
- TASK_02: Synthetic Generators Levels 0-3 (Dec 19)
- TASK_03: Model Scaling to 37M (Dec 19)
- TASK_06: Training Dashboard (Dec 21)

---

## Quick Reference

```bash
# Training
python train_jax_fast.py --synthetic-level 0 --model-size large --epochs 200

# Dashboard
cd tools/dashboard && ./start_dashboard.sh

# Compare runs
python cross_validate.py --compare logs/training_*.csv
```

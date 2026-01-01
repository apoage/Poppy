# Dojo Refactor Status Report

**Date**: Dec 19, 2025
**Issue**: Gemini broke codebase by integrating tactics directly into `train_jax_fast.py`
**Solution**: Created modular architecture with composable tactics

---

## Problem Summary

Gemini implemented TASK_02 (synthetic markets) and portfolio morphing by:
1. ✅ Creating `mcts_trader/data/synthetic_market.py` (GOOD - modular)
2. ❌ Hardcoding swap logic into `train_jax_fast.py` (BAD - monolithic)
3. ❌ Using wrong attributes (`.shares` instead of `.size`)
4. ❌ Wrong Position constructor fields

**User's feedback**: "dojo suposed to be implemened in training manager .. and as syntetic generator module .. with incremental comlpexity and modules for each tactics .. gemini just started to cut and change things random way"

---

## What Was Created (Phase 1)

### ✅ Modular Tactics System

**Created files**:
- `mcts_trader/tactics/__init__.py`
- `mcts_trader/tactics/basic_trading.py` (Buy/Sell/Hold via cash)
- `mcts_trader/tactics/portfolio_morphing.py` (Direct asset→asset swaps)

**Key features**:
- Each tactic is a separate module
- Composable via action space extension
- Isolated execute() and get_action_mask() methods
- Fixed `.shares` → `.size` bug
- Fixed Position constructor (asset_idx, size, not asset_id, shares)

**Example usage**:
```python
from mcts_trader.tactics import BasicTrading, PortfolioMorphing

# Level 0-2: Basic only
tactics = [BasicTrading]

# Level 3+: Basic + Swaps
tactics = [BasicTrading, PortfolioMorphing]
```

### ✅ Dojo Training Manager

**Created file**: `dojo_manager.py`

**Features**:
- Curriculum orchestration (Levels 0-3 default)
- Tactic composition via config
- Checkpoint management
- Progress tracking
- Resume capability

**CLI**:
```bash
# Train full curriculum (Levels 0-3)
python dojo_manager.py

# Train single level
python dojo_manager.py --level 0

# Resume from checkpoint
python dojo_manager.py --resume

# Custom curriculum
python dojo_manager.py --curriculum my_curriculum.json
```

**Default curriculum**:
```json
[
  {"level": 0, "generator": "sine", "tactics": ["basic"], "epochs": 500},
  {"level": 1, "generator": "lagged", "tactics": ["basic"], "epochs": 1000},
  {"level": 2, "generator": "dip", "tactics": ["basic"], "epochs": 1500},
  {"level": 3, "generator": "noisy_trends", "tactics": ["basic", "morphing"], "epochs": 2000}
]
```

### ✅ Design Documentation

**Created**: `lore/DOJO_MODULAR_ARCHITECTURE.md`

**Covers**:
- Modular architecture vision
- Tactic composition pattern
- Generator modules (future refactor)
- Migration plan
- Benefits of modular approach

---

## Current State

### Working ✅
- `mcts_trader/data/synthetic_market.py` (Gemini's work - functional)
- `mcts_trader/tactics/` module (new - modular tactics)
- `dojo_manager.py` (new - curriculum orchestrator)

### Broken ❌
- `train_jax_fast.py` has hardcoded swap logic (needs cleanup)
  - Lines 337-434: `_execute_swap()`, `_get_swap_mask()` methods
  - Lines 409-414: Wrong Position constructor usage
  - Line 427: `.shares` instead of `.size`

### Not Yet Integrated ⚠️
- `dojo_manager.py` currently calls `train_jax_fast.py` as subprocess
- Tactics are defined but not yet wired into environment
- Environment doesn't use modular tactics yet

---

## What Needs to Happen Next

### Phase 2: Integration (PRIORITY)

**Option A: Minimal Integration** (Recommended for now)
1. Keep `train_jax_fast.py` with hardcoded tactics (but fix bugs)
2. Use `dojo_manager.py` to orchestrate training
3. Defer full refactor until after validation

**Fixes needed in `train_jax_fast.py`**:
- Line 411: `asset_id` → `asset_idx`
- Line 412: `shares` → `size`
- Line 427: `pos.shares` → `pos.size`
- Add missing Position fields: `symbol`, `side`

**Option B: Full Refactor** (Clean but risky)
1. Remove swap logic from `train_jax_fast.py`
2. Create `FastTradingEnv` wrapper that uses tactics
3. Wire tactics into environment step() method

**Recommendation**: Start with Option A (fix bugs, validate Dojo), then do Option B.

### Phase 3: Generators Refactor (Future)

Break `mcts_trader/data/synthetic_market.py` into:
```
generators/
├── __init__.py
├── base.py          # SyntheticGenerator ABC
├── sine_waves.py    # Level 0
├── lagged_pairs.py  # Level 1
├── dip_crashes.py   # Level 2
└── noisy_trends.py  # Level 3
```

**Benefit**: Each level in separate file (easier to extend to 22 levels)

### Phase 4: Advanced Tactics (Future)

Add to `mcts_trader/tactics/`:
- `pairs_trading.py` (Level 5: Cointegration)
- `triangle_arbitrage.py` (Level 6: Cross-pair mispricing)
- `market_making.py` (Level 15: Bid-ask spread)
- ... (up to 22 levels from MATHEMATICAL_TRADING_TACTICS_CURRICULUM.md)

---

## Testing

### Test 1: Dojo Manager (Dry Run)
```bash
python dojo_manager.py --level 0
```

**Expected**: Should call `train_jax_fast.py --synthetic-level 0 --epochs 500`

### Test 2: Modular Tactics (Unit Test)
```python
from mcts_trader.tactics import BasicTrading, PortfolioMorphing

# Test action space sizes
assert BasicTrading.get_action_space_size(64) == 192  # 64 * 3
assert PortfolioMorphing.get_action_space_size(64) == 4096  # 64 * 64
```

### Test 3: Full Curriculum
```bash
python dojo_manager.py
```

**Expected**: Train Levels 0→1→2→3 sequentially

---

## File Changes Summary

### Created ✅
- `mcts_trader/tactics/__init__.py`
- `mcts_trader/tactics/basic_trading.py`
- `mcts_trader/tactics/portfolio_morphing.py`
- `dojo_manager.py`
- `lore/DOJO_MODULAR_ARCHITECTURE.md`
- `DOJO_REFACTOR_STATUS.md` (this file)

### Modified (by Gemini) ⚠️
- `train_jax_fast.py` (lines 337-574 - swap logic, needs cleanup)

### Existing (Unchanged) ✅
- `mcts_trader/data/synthetic_market.py` (Gemini's work - functional)
- `train_manager.py` (for real data experiments - separate concern)

---

## Next Steps (Recommended)

1. **Fix bugs in `train_jax_fast.py`** (`.shares` → `.size`, Position constructor)
2. **Test Dojo manager** with Level 0 training
3. **Validate** that curriculum runs end-to-end
4. **Then** decide: Keep hardcoded tactics or refactor to modular wrappers?

**Decision point**: User should decide whether to:
- **Quick path**: Fix bugs, use Dojo with current `train_jax_fast.py`
- **Clean path**: Refactor `train_jax_fast.py` to use modular tactics immediately

---

## Architecture Comparison

### Before (Gemini's Approach - Broken)
```
train_jax_fast.py (2000+ lines)
├── Basic trading logic
├── Swap logic (hardcoded)
├── Action masking (hardcoded)
└── Training loop
```

**Problem**: Monolithic, hard to extend, violates SRP

### After (Modular Architecture - Fixed)
```
dojo_manager.py          # Curriculum orchestrator
├── tactics/
│   ├── basic_trading.py     # Composable
│   └── portfolio_morphing.py # Composable
├── generators/
│   └── synthetic_market.py   # Modular
└── train_jax_fast.py         # Pure training loop (refactored)
```

**Benefit**: Composable, testable, extensible to 22+ levels

---

## Summary

**Status**: Phase 1 COMPLETE ✅
- Modular tactics created
- Dojo manager created
- Architecture documented

**Status**: Phase 2 PENDING ⚠️
- Bug fixes in `train_jax_fast.py`
- Integration testing
- Validation of curriculum training

**Blocker**: Need to decide on integration approach (quick fix vs full refactor)

**Recommendation**: Fix bugs → Test Dojo → Validate → Then refactor for clean modular wrappers.

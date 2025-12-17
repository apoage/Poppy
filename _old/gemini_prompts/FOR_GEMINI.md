# Gemini: Implementation Tasks

## Quick Context
User wants two-phase training:
1. **Pre-training**: 1 year of data, 64 random crypto assets, 1440-step episodes
2. **Fine-tuning**: 30 days recent data, same assets, 200-step episodes

`train_manager.py` is already updated with asset selection. You need to connect the `min_coverage` parameter through the training pipeline.

---

## Your Tasks (3 files to modify)

### Task 1: train_jax_fast.py
**What**: Read `min_coverage` from config and calculate `min_required_assets`

**Where**: Around line 120, where it reads `args.config`

**Add this code**:
```python
min_coverage = 0.1  # Default

if args.config:
    import json
    with open(args.config) as f:
        config = json.load(f)
        n_assets = config.get('n_assets', 64)
        asset_symbols = config.get('asset_symbols', None)
        min_coverage = config.get('min_coverage', 0.1)  # NEW

min_required_assets = max(1, int(n_assets * min_coverage))  # NEW
print(f"DEBUG: Min required assets: {min_required_assets} ({min_coverage*100:.1f}%)")  # NEW
```

Then pass `min_required_assets` to `FastTradingEnv()`:
```python
env = FastTradingEnv(
    # ... existing parameters ...
    min_required_assets=min_required_assets  # NEW
)
```

---

### Task 2: mcts_trader/env/wrapper.py
**What**: Update `FastTradingEnv.__init__` to accept and use parameter

**Add to `__init__` signature**:
```python
def __init__(
    self,
    # ... existing parameters ...
    min_required_assets=None  # NEW
):
```

**Calculate default if not provided**:
```python
    if min_required_assets is None:
        min_required_assets = max(1, int(n_assets * 0.1))  # Default 10%
```

**Pass to HistoricalMarket**:
```python
    self.market = HistoricalMarket(
        db_path=db_path,
        asset_symbols=asset_symbols,
        min_required_assets=min_required_assets  # NEW
    )
```

---

### Task 3: Verify HistoricalMarket
**What**: Check that `mcts_trader/data/history.py` already supports this parameter

**Action**: READ the file and verify `HistoricalMarket.__init__` has `min_required_assets`.

If it **doesn't exist**, you need to add it. Look for where timestamps are filtered and ensure only timestamps with enough concurrent assets are kept.

---

## Testing

### Step 1: Dry-run
```bash
python train_manager.py --random-subset 32 --exclude-forex --date-start 2024-12-01 --date-end 2024-12-31 --min-coverage 0.8 --no-validate --dry-run
```

**Expected**: No errors, creates `config/current_training.json` with `min_coverage: 0.8`

### Step 2: Quick test (1 epoch)
```bash
python train_manager.py --random-subset 32 --exclude-forex --date-start 2024-12-01 --date-end 2024-12-31 --min-coverage 0.8 --steps 200 --epochs 1 --no-validate
```

**Expected output**:
```
DEBUG: Min required assets: 26 (80.0% coverage)
Market loaded: 32 assets, XXXXX valid timestamps
```

**Success criteria**:
- ✅ No crashes
- ✅ Debug line prints correct calculation
- ✅ Valid timestamps > 0
- ✅ Training completes 1 epoch

---

## If You Get Stuck

See `lore/gemini_instructions.md` for:
- Detailed explanations
- Code examples
- Troubleshooting guide
- Expected output at each step

---

## After You're Done

User will run:

### Phase 1 Command
```bash
python train_manager.py \
  --random-subset 64 \
  --exclude-forex \
  --date-start 2024-03-30 \
  --date-end 2025-03-30 \
  --min-candles 262800 \
  --min-coverage 0.85 \
  --steps 1440 \
  --epochs 10000 \
  --no-validate
```

This should train for 2-4 days and produce a robust pre-trained model.

---

## Key Points

1. `min_coverage` is a **float** (0.0-1.0) representing percentage
2. `min_required_assets` is an **int** calculated as `n_assets * min_coverage`
3. The parameter flows: `train_manager.py` → config file → `train_jax_fast.py` → `FastTradingEnv` → `HistoricalMarket`
4. Test each step to ensure parameter is passed correctly

---

## Files to Read

Priority order:
1. `lore/gemini_instructions.md` - Full implementation guide
2. `TRAINING_COMMANDS.md` - Command examples
3. `lore/diary.md` - Strategy context (if interested)

Good luck! 🚀

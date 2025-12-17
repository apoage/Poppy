# Instructions for Gemini - Training Pipeline Implementation

## Context
You are working on a cryptocurrency trading agent that uses TRM (Tiny Recursive Model) with MCTS planning. The system trains on historical 1-minute OHLCV data stored in a 20GB SQLite database.

## Current Status
The training manager (`train_manager.py`) has been updated with new asset selection features. Now you need to complete the integration so that the `min_coverage` parameter flows through the entire training pipeline.

## Your Tasks

### Task 1: Update `train_jax_fast.py` to Read Coverage Configuration

**File**: `/home/lukas/Trader/train_jax_fast.py`

**Location**: Around line 100-130, in the section that reads `args.config`

**Current code** (approximate):
```python
# Dynamic Asset Configuration
n_assets = 64
asset_symbols = None

if args.config:
    import json
    with open(args.config) as f:
        config = json.load(f)
        n_assets = config.get('n_assets', 64)
        asset_symbols = config.get('asset_symbols', None)
```

**Required changes**:
Add `min_coverage` reading and calculation:
```python
# Dynamic Asset Configuration
n_assets = 64
asset_symbols = None
min_coverage = 0.1  # Default 10%

if args.config:
    import json
    with open(args.config) as f:
        config = json.load(f)
        n_assets = config.get('n_assets', 64)
        asset_symbols = config.get('asset_symbols', None)
        min_coverage = config.get('min_coverage', 0.1)  # NEW LINE

# Calculate min_required_assets (integer count)
min_required_assets = max(1, int(n_assets * min_coverage))  # NEW LINE
print(f"DEBUG: Min required assets set to {min_required_assets} ({min_coverage*100:.1f}% coverage)")  # NEW LINE
```

**Verification**: After this change, when you run training with a config file, you should see a debug line printed:
```
DEBUG: Min required assets set to 58 (90.0% coverage)
```

---

### Task 2: Pass `min_required_assets` to FastTradingEnv

**File**: `/home/lukas/Trader/train_jax_fast.py`

**Location**: Where `FastTradingEnv` is initialized (search for `FastTradingEnv(`)

**Current code** (approximate):
```python
env = FastTradingEnv(
    initial_capital=100000.0,
    trading_fee_bps=args.fee,
    mode=args.mode,
    n_assets=n_assets,
    asset_symbols=asset_symbols,
    db_path='trading_data.db'
)
```

**Required changes**:
Add `min_required_assets` parameter:
```python
env = FastTradingEnv(
    initial_capital=100000.0,
    trading_fee_bps=args.fee,
    mode=args.mode,
    n_assets=n_assets,
    asset_symbols=asset_symbols,
    db_path='trading_data.db',
    min_required_assets=min_required_assets  # NEW LINE
)
```

---

### Task 3: Update FastTradingEnv to Accept and Use min_required_assets

**File**: `/home/lukas/Trader/mcts_trader/env/wrapper.py`

**Location**: `FastTradingEnv.__init__` method

**Current code** (approximate):
```python
class FastTradingEnv:
    def __init__(
        self,
        initial_capital=100000.0,
        trading_fee_bps=26,
        mode='MARGIN',
        n_assets=64,
        asset_symbols=None,
        db_path='trading_data.db'
    ):
        # ... initialization code ...

        self.market = HistoricalMarket(
            db_path=db_path,
            asset_symbols=asset_symbols
        )
```

**Required changes**:

1. Add parameter to `__init__`:
```python
class FastTradingEnv:
    def __init__(
        self,
        initial_capital=100000.0,
        trading_fee_bps=26,
        mode='MARGIN',
        n_assets=64,
        asset_symbols=None,
        db_path='trading_data.db',
        min_required_assets=None  # NEW PARAMETER
    ):
```

2. Calculate default if not provided:
```python
        # Calculate min_required_assets if not provided
        if min_required_assets is None:
            min_required_assets = max(1, int(n_assets * 0.1))  # Default 10%
```

3. Pass to HistoricalMarket:
```python
        self.market = HistoricalMarket(
            db_path=db_path,
            asset_symbols=asset_symbols,
            min_required_assets=min_required_assets  # NEW ARGUMENT
        )
```

---

### Task 4: Verify HistoricalMarket Already Supports min_required_assets

**File**: `/home/lukas/Trader/mcts_trader/data/history.py`

**Action**: READ THE FILE and verify that `HistoricalMarket.__init__` already accepts `min_required_assets` parameter.

**Expected**: You should see something like:
```python
class HistoricalMarket:
    def __init__(self, db_path='trading_data.db', asset_symbols=None, min_required_assets=None):
        # ... code that uses min_required_assets ...
```

**If it DOES NOT exist**: Add the parameter and implementation:
```python
def __init__(self, db_path='trading_data.db', asset_symbols=None, min_required_assets=None):
    self.min_required_assets = min_required_assets or 1
    # ... rest of init ...

    # When filtering valid timestamps:
    valid_timestamps = []
    for ts in all_timestamps:
        # Count how many assets have data at this timestamp
        available_count = sum(1 for asset in self.assets if ts in asset_data[asset])
        if available_count >= self.min_required_assets:
            valid_timestamps.append(ts)
```

**Note**: The exact implementation may vary. The key is that `HistoricalMarket` must filter timestamps based on how many assets have concurrent data.

---

## Testing Your Changes

### Step 1: Dry Run Test
Test asset selection without actually training:
```bash
python train_manager.py \
  --random-subset 64 \
  --exclude-forex \
  --date-start 2024-03-30 \
  --date-end 2025-03-30 \
  --min-coverage 0.9 \
  --min-candles 525600 \
  --no-validate \
  --dry-run
```

**Expected output**:
```
Selected 64 random assets from 150 candidates.
Sample assets: ['XBT/USD', 'ETH/USD', 'XRP/USD', ...]
Excluded 8 forex pairs. Remaining: 142
Config written to: config/current_training.json
DRY RUN COMPLETE. Would train with:
Pairs: 64
First 5 pairs: ['XBT/USD', 'ETH/USD', ...]
```

### Step 2: Verify Config File
Check that `config/current_training.json` contains:
```bash
cat config/current_training.json
```

**Expected**:
```json
{
  "experiment_name": "dynamic_set",
  "n_assets": 64,
  "asset_symbols": ["XBT/USD", "ETH/USD", ...],
  "min_coverage": 0.9,
  "generated_at": "1733932800.123"
}
```

### Step 3: Short Training Test
Run a very short training session (1 epoch) to verify the pipeline works:
```bash
python train_manager.py \
  --random-subset 32 \
  --exclude-forex \
  --date-start 2024-12-01 \
  --date-end 2024-12-31 \
  --min-coverage 0.8 \
  --steps 200 \
  --epochs 1 \
  --no-validate
```

**Expected output** (in training logs):
```
DEBUG: Min required assets set to 26 (80.0% coverage)
Market loaded: 32 assets, 44640 valid timestamps
Episode 1 complete, reward: -1.23
```

**Key verification points**:
- ✅ No crashes or errors
- ✅ Debug line shows correct min_required_assets calculation
- ✅ Market loaded with correct number of assets
- ✅ Valid timestamps count is reasonable (should be thousands, not zero)

---

## Common Issues and Solutions

### Issue 1: "No valid timestamps found"
**Cause**: Asset selection resulted in pairs with no overlapping data in the date range.

**Solution**:
- Reduce `--min-coverage` (try 0.5 instead of 0.9)
- Expand date range (`--date-start` to `--date-end`)
- Reduce `--min-candles` requirement

### Issue 2: "Only found X valid assets, requested Y"
**Cause**: Not enough assets in database meet the criteria.

**Solution**:
- Remove `--exclude-forex` flag
- Reduce `--min-candles` threshold
- Expand date range
- Reduce `--random-subset` count

### Issue 3: TypeError about min_required_assets
**Cause**: Parameter not properly passed through the pipeline.

**Solution**: Double-check Tasks 1-3 above. Use print statements to trace the value:
```python
print(f"DEBUG: min_required_assets = {min_required_assets}")  # Add this at each step
```

---

## File Change Summary

Here's what you should change in each file:

| File | Changes | Lines (approx) |
|------|---------|---------------|
| `train_manager.py` | ✅ Already done (is_forex_pair, new args, config generation) | Multiple |
| `train_jax_fast.py` | Add min_coverage reading, calculate min_required_assets, pass to env | ~120-150 |
| `mcts_trader/env/wrapper.py` | Add parameter to FastTradingEnv, pass to HistoricalMarket | ~50-80 |
| `mcts_trader/data/history.py` | Verify parameter exists (likely already implemented) | Check only |

---

## Validation Checklist

Before marking tasks complete, verify:

- [ ] `train_manager.py` can generate config with min_coverage
- [ ] `train_jax_fast.py` reads min_coverage and calculates min_required_assets
- [ ] FastTradingEnv accepts min_required_assets parameter
- [ ] HistoricalMarket receives and uses min_required_assets
- [ ] Dry-run test completes without errors
- [ ] Short training test (1 epoch) completes successfully
- [ ] Debug output shows correct coverage percentage
- [ ] Valid timestamp count is non-zero and reasonable

---

## After Completion

Once all tasks are done and tested, the user can run:

### Phase 1: Pre-Training (Long episodes, diverse data)
```bash
python train_manager.py \
  --random-subset 64 \
  --exclude-forex \
  --date-start 2024-03-30 \
  --date-end 2025-03-30 \
  --min-candles 525600 \
  --min-coverage 0.9 \
  --steps 1440 \
  --epochs 10000 \
  --no-validate
```

### Phase 2: Fine-Tuning (Recent data, shorter episodes)
```bash
python train_manager.py \
  --random-subset 64 \
  --exclude-forex \
  --date-start 2025-03-01 \
  --date-end 2025-03-30 \
  --min-coverage 0.95 \
  --steps 200 \
  --epochs 2000 \
  --no-validate
```

---

## Additional Notes

### Why Exclude Forex?
Forex (fiat/fiat) pairs like EUR/USD behave differently from crypto:
- Driven by central bank policy, not market sentiment
- Lower volatility, different patterns
- May have incomplete data in the crypto-focused database
- User wants pure crypto trading strategy

### Why 1440 Steps for Pre-Training?
1440 steps = 1440 minutes = 24 hours. This teaches:
- Full daily trading cycles (Asia, Europe, US sessions)
- Overnight momentum patterns
- Day-of-week effects
- Better long-term planning vs. short 200-step episodes

### Why Two-Phase Training?
- **Phase 1**: General trading intelligence (robust, diverse)
- **Phase 2**: Adaptation to recent regime (current market conditions)
- This prevents overfitting while maintaining relevance

### Database Timestamp Info
- Data ends: ~March 30, 2025 (timestamp 1743454140)
- 1 year before: March 30, 2024 (timestamp ~1711756800)
- Use these as reference points for date range selection

---

## Questions or Issues?

If you encounter errors or unexpected behavior:

1. **Check logs**: `logs/train_manager.log` and training output
2. **Verify config**: `cat config/current_training.json`
3. **Test database**:
```sql
sqlite3 trading_data.db "SELECT COUNT(*) FROM ohlcv WHERE timestamp >= 1711756800"
```
4. **Add debug prints**: Trace parameter values through the pipeline
5. **Report back**: Share exact error messages and context

---

## Success Criteria

You'll know implementation is complete when:
1. Dry-run test shows correct asset selection with forex exclusion
2. Config file contains min_coverage field
3. Training starts and prints "DEBUG: Min required assets set to X"
4. Market loads with non-zero valid timestamps
5. No crashes or TypeErrors related to min_required_assets
6. Training can complete at least 1 full epoch without errors

Good luck! 🚀

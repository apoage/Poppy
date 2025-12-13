# Pre-Training Fixes Prompt for Gemini

## Critical Issues to Fix BEFORE Retraining

The user ran `real_trade_jax.py` and discovered 3 blocking issues that MUST be fixed before starting new training:

### Issue 1: Symbol Mapping Failures (11/64 pairs unmapped)
### Issue 2: NoneType Comparison Error (Still Crashing)
### Issue 3: Delisted Assets in Training Data (EOS/USD, etc.)

---

## Issue #1: Symbol Mapping Failures

### The Problem
```
2025-12-12 09:59:10,560 - WARNING - Could not map symbol: ETHBTC
2025-12-12 09:59:10,561 - WARNING - Could not map symbol: BTCGBP
2025-12-12 09:59:10,561 - WARNING - Could not map symbol: XRPBTC
2025-12-12 09:59:10,561 - WARNING - Could not map symbol: BTCCAD
2025-12-12 09:59:10,561 - WARNING - Could not map symbol: LTCBTC
2025-12-12 09:59:10,561 - WARNING - Could not map symbol: XMRBTC
2025-12-12 09:59:10,561 - WARNING - Could not map symbol: ADABTC
2025-12-12 09:59:10,561 - WARNING - Could not map symbol: EOS/USD
2025-12-12 09:59:10,561 - WARNING - Could not map symbol: XLMBTC
2025-12-12 09:59:10,561 - WARNING - Could not map symbol: BCHBTC
2025-12-12 09:59:10,561 - WARNING - Could not map symbol: BTCCHF
2025-12-12 09:59:10,561 - INFO - Mapped 53/64 symbols.
```

**11 pairs are failing to map**, even though Claude supposedly "fixed" the symbol mapping logic.

### Your Task

**READ THESE FILES:**
1. `/home/lukas/Trader/real_trade_jax.py` - Lines 124-178 (`_build_symbol_map` method)
2. `/home/lukas/Trader/paper_trade_jax.py` - Lines 143-225 (`_build_symbol_map` method)

**ANALYZE:**
- Why is the enhanced mapping logic NOT working?
- The variants were added but pairs still fail
- Check if the logic is actually being executed
- Check if wsname_to_api and altname_to_api contain the pairs

**DEBUG APPROACH:**
1. Add temporary logging to see what's in `wsname_to_api` and `altname_to_api`
2. For each failing symbol, log ALL variants being tried
3. Check if the variants match what's actually in Kraken's API response

**LIKELY ISSUES:**
- Variant generation may be correct but not checking all possibilities
- Kraken may use different naming (e.g., "ETHXBT" vs "ETH/XBT")
- The slash detection logic may be wrong
- EOS/USD is delisted - we should handle this gracefully

**FIX REQUIREMENTS:**
- Make symbol mapping robust enough to map at least 60/64 pairs
- Add detailed logging for unmapped symbols (show what was tried)
- Handle delisted/unavailable pairs gracefully (skip them, don't crash)
- Test the fix by reading the Kraken API response structure

---

## Issue #2: NoneType Comparison Error (Still Happening)

### The Problem
```
2025-12-12 10:00:04,875 - ERROR - Error: '<=' not supported between instances of 'NoneType' and 'int'. Restarting...
2025-12-12 10:05:01,432 - ERROR - Error: '<=' not supported between instances of 'NoneType' and 'int'. Restarting...
```

The error happens intermittently during live trading. Claude supposedly fixed this in `real_portfolio.py` line 313, but it's still crashing.

### Your Task

**READ THESE FILES:**
1. `/home/lukas/Trader/real_trade_jax.py` - Entire file, focus on `trade_step()` method
2. `/home/lukas/Trader/mcts_trader/agent/real_portfolio.py` - Entire file
3. `/home/lukas/Trader/mcts_trader/agent/portfolio.py` - Check for similar issues

**FIND ALL LOCATIONS WHERE:**
- Variables could be `None` and are used in comparisons (`<=`, `>=`, `<`, `>`, `==`)
- Dictionary `.get()` is used without default values
- Price data might be missing

**COMMON CULPRITS:**
```python
# BAD - can crash if None
if some_value <= 0:
    ...

# GOOD - handles None
if some_value is None or some_value <= 0:
    ...
```

**SEARCH PATTERNS:**
1. `prices.get(...)` - without defaults
2. `if .* <= \d` - comparisons that could receive None
3. `if .* >= \d` - comparisons that could receive None
4. Any arithmetic on values that could be None

**FIX REQUIREMENTS:**
- Find EVERY location where this error could occur
- Add None checks before ALL comparisons
- Add logging when None values are encountered
- Make the code defensive - assume data can be missing

---

## Issue #3: Delisted Assets in Training Data

### The Problem

**EOS/USD** is delisted from Kraken but still in the training data. This causes:
- Failed symbol mapping (can't trade)
- Model learns strategies that can't be executed
- Wasted 1/64th of model capacity on dead asset

### Your Task

**OPTION A: Clean the Database (Recommended)**

1. **READ:** `/home/lukas/Trader/mcts_trader/data/history.py` - Understand how assets are loaded
2. **READ:** `/home/lukas/Trader/mcts_trader/data/database.py` - Database schema
3. **IDENTIFY:** All delisted/untradeable assets in `trading_data.db`
4. **REMOVE:** Delete delisted assets from database OR mark them as inactive

**Files to modify:**
- `mcts_trader/data/history.py` - Filter out delisted assets when loading
- Or create a script: `tools/clean_delisted_assets.py`

**OPTION B: Dynamic Asset Filtering (Better Long-Term)**

1. **READ:** `/home/lukas/Trader/mcts_trader/data/kraken_client.py`
2. **MODIFY:** `HistoricalMarket.__init__()` to query Kraken for tradeable pairs
3. **FILTER:** Only load assets that are currently tradeable
4. **LOG:** Report which assets were filtered out

**FIX REQUIREMENTS:**
- Remove EOS/USD from training data
- Check for other delisted assets (LUNA, FTT, etc.)
- Make asset list dynamic or at least validated against Kraken API
- Ensure N_ASSETS count matches actual tradeable assets

---

## Implementation Checklist

### Step 1: Fix Symbol Mapping (Highest Priority)
- [ ] Read `_build_symbol_map()` in both files
- [ ] Add debug logging to see Kraken API response structure
- [ ] Identify why variants aren't matching
- [ ] Fix the mapping logic
- [ ] Test that at least 60/64 pairs map successfully
- [ ] Handle unmappable pairs gracefully (skip, don't crash)

### Step 2: Fix NoneType Errors (Critical for Stability)
- [ ] Search ALL portfolio files for comparison operators
- [ ] Find every place where None could be compared to int
- [ ] Add `if x is None or x <= 0:` guards
- [ ] Add logging for None values
- [ ] Test by running dry-run mode for 30+ minutes

### Step 3: Remove Delisted Assets (Data Quality)
- [ ] Query Kraken for current tradeable pairs
- [ ] Compare with database asset list
- [ ] Remove or flag delisted assets
- [ ] Update N_ASSETS to match reality
- [ ] Document which assets were removed and why

---

## Testing Requirements

After implementing fixes:

### Test 1: Symbol Mapping
```bash
# Run and check logs
python real_trade_jax.py

# Expected output:
# "Mapped 60+/64 symbols" (or 63/63 after cleanup)
# No "Could not map symbol" warnings except for truly unavailable pairs
```

### Test 2: Stability
```bash
# Run for 30 minutes in dry-run mode
python real_trade_jax.py

# Expected:
# No NoneType errors
# No crashes
# Clean execution of trade signals
```

### Test 3: Asset Count
```bash
# Check database
sqlite3 trading_data.db "SELECT COUNT(*) FROM assets WHERE active=1;"

# Should match N_ASSETS in config
# No delisted assets should be active
```

---

## Deliverables

Create these files:

1. **PRE_TRAINING_FIXES_REPORT.md** - Document all changes made
2. **UNMAPPED_SYMBOLS_DEBUG.log** - Debug output showing why symbols failed
3. **DELISTED_ASSETS.txt** - List of removed assets with reasons

Update these files:
- `real_trade_jax.py` - Fixed symbol mapping
- `paper_trade_jax.py` - Fixed symbol mapping
- `real_portfolio.py` - Fixed None comparisons
- `portfolio.py` - Fixed None comparisons (if needed)
- `history.py` - Filter delisted assets

---

## Important Notes

### Symbol Mapping Strategy
The correct approach is:
1. Get Kraken's tradeable pairs API response
2. Log the EXACT format of symbol names (wsname, altname, base, quote)
3. Build mapping based on ACTUAL data, not assumptions
4. For unmappable pairs, log the raw API data to debug

### NoneType Defense
Every comparison must be safe:
```python
# ALWAYS check None first
if value is not None and value > 0:
    # safe to use value
```

### Delisted Assets
Don't just remove data - preserve it for historical analysis:
```sql
-- Add 'active' column instead of deleting
ALTER TABLE assets ADD COLUMN active INTEGER DEFAULT 1;
UPDATE assets SET active=0 WHERE symbol='EOS/USD';
```

Then filter:
```python
WHERE active=1
```

---

## Start Your Work

Begin with Issue #1 (Symbol Mapping) since it affects 17% of your asset universe.

Add EXTENSIVE logging to understand what Kraken's API actually returns. Don't guess - read the actual data.

Good luck!

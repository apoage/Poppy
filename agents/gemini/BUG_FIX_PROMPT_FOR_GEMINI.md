# Bug Fix Implementation Prompt for Gemini

## Context

You have completed an accounting audit and identified critical bugs in the trading system. Now you need to implement the fixes by reading the proper files and making the necessary code changes.

## Your Task

**Implement all critical bug fixes identified in the audit reports.**

## Files You MUST Read First

### 1. Audit Reports (Your Previous Work)
- `/home/lukas/Trader/audit_report_final.md` - Full technical details
- `/home/lukas/Trader/AUDIT_EXECUTIVE_SUMMARY.md` - Quick reference
- `/home/lukas/Trader/AUDIT_FIX_CHECKLIST.md` - Implementation guide

### 2. Files That Need Fixing

Based on your audit findings, you need to read and fix these files:

#### **Critical Bug #1: FX Rates Missing in Training Environment**
**Files to read and modify:**
- `/home/lukas/Trader/mcts_trader/env/trading_env.py` - Training environment wrapper
- `/home/lukas/Trader/mcts_trader/agent/portfolio.py` - Portfolio manager (verify FX parameter exists)
- `/home/lukas/Trader/mcts_trader/data/history.py` - Historical market data provider

**What to look for:**
- Where `portfolio.update_prices()` is called
- Whether `fx_rates` parameter is being passed
- How to get current FX rates in the training environment
- The `HistoricalMarket` class - does it provide FX rates?

#### **Critical Bug #2: Stop-Loss Trailing Logic**
**Files to read and modify:**
- `/home/lukas/Trader/mcts_trader/agent/portfolio.py` - Lines around `reinforce_position()` method
- Look for the trailing stop-loss logic for shorts
- Check the take-profit trailing logic

#### **High Priority: Action Masking with Cash Reserve**
**Files to read and modify:**
- `/home/lukas/Trader/mcts_trader/agent/portfolio.py` - `get_valid_actions()` method
- Check if cash reserve is considered when masking BUY actions
- Verify margin level calculations

## Implementation Guidelines

### Step 1: Read All Audit Reports
Start by reading your three audit reports completely to understand:
- What bugs were found
- Why they are critical
- What the correct behavior should be
- Numerical examples of expected results

### Step 2: Read Current Code
For each bug, read the relevant files COMPLETELY:
- Understand the current implementation
- Identify the exact lines that need changes
- Check for dependencies and side effects
- Look for similar patterns elsewhere in the codebase

### Step 3: Implement Fixes Carefully

**For FX Rates Bug:**
1. Find where `HistoricalMarket` stores/provides data
2. Check if it already has FX rate data (it should from Kraken API)
3. Modify `trading_env.py` to extract FX rates from market data
4. Pass FX rates to `portfolio.update_prices(prices, fx_rates)`
5. Verify the FX rates dictionary format matches expectations

**For Stop-Loss Trailing Bug:**
1. Find the `reinforce_position()` method
2. Locate the SL trailing logic for shorts
3. Fix the condition to trail SL on EVERY reinforcement, not just once
4. Ensure SL moves progressively tighter as profit increases
5. Add logging to track SL movements

**For Action Masking:**
1. Find `get_valid_actions()` method
2. Check if available capital considers cash reserve
3. Verify BUY actions are masked when insufficient capital
4. Ensure margin level is calculated correctly

### Step 4: Verify Fixes

After making changes:
1. Read the modified sections again
2. Trace through a manual scenario with the new code
3. Verify the fix matches the audit recommendations
4. Check for any new bugs introduced
5. Ensure code style is consistent

### Step 5: Document Changes

For each fix, create a clear comment explaining:
```python
# BUG FIX [2024-12-12]: FX rates now passed from historical market
# Previously: portfolio.update_prices(prices) - ignored FX rates
# Now: portfolio.update_prices(prices, fx_rates) - correct multi-currency valuation
# Impact: Fixes corrupted training for EUR/GBP/BTC quoted pairs (50% of assets)
```

## Critical Requirements

### DO:
- ✅ Read the audit reports FIRST before touching code
- ✅ Read the ENTIRE file before modifying any line
- ✅ Make surgical, minimal changes
- ✅ Add clear comments explaining each fix
- ✅ Preserve existing functionality
- ✅ Follow existing code style

### DO NOT:
- ❌ Make changes based on assumptions - READ THE CODE
- ❌ Refactor or "improve" unrelated code
- ❌ Change variable names unnecessarily
- ❌ Add new features beyond fixing the identified bugs
- ❌ Remove existing comments or documentation
- ❌ Change indentation or formatting of unchanged code

## Expected Outputs

After completing all fixes, provide:

1. **Summary of Changes**: List each file modified and what was changed
2. **Verification**: Explain how you verified each fix is correct
3. **Testing Recommendations**: Suggest how to test the fixes
4. **Remaining Issues**: Note any concerns or edge cases

## Additional Context Files to Reference

If you need to understand the broader system:
- `/home/lukas/Trader/CLAUDE.md` - Project overview and architecture
- `/home/lukas/Trader/lore/DEVELOPMENT_DIARY.md` - Development history
- `/home/lukas/Trader/mcts_trader/config/trading_config.py` - Configuration parameters

## Start Your Work

Begin by:
1. Reading `AUDIT_FIX_CHECKLIST.md` completely
2. Reading `audit_report_final.md` Section 8 (Proposed Fixes)
3. Then systematically work through each bug fix

Remember: You are fixing REAL MONEY trading system bugs. Be thorough, careful, and verify everything.

Good luck!

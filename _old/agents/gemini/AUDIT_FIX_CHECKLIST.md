# Audit Fix Checklist

**Generated:** December 12, 2025
**Source:** Comprehensive Short Position Accounting Audit

Use this checklist to track implementation of fixes identified in the audit.

---

## 🔴 CRITICAL FIXES (Must Complete Before Retraining)

### [ ] Fix #1: Add FX Rates to Training Environment

**File:** `/home/lukas/Trader/mcts_trader/env/trading_env.py`
**Location:** Lines 199-203
**Severity:** CRITICAL
**Time Estimate:** 30 minutes

**Current Code:**
```python
new_prices = {
    idx: new_data[sym]['close']
    for idx, sym in self.market.assets.items()
    if sym in new_data
}
self.portfolio.update_prices(new_prices)
```

**Replace With:**
```python
new_prices = {
    idx: new_data[sym]['close']
    for idx, sym in self.market.assets.items()
    if sym in new_data
}

# Build FX rates dict from available prices
# For proper implementation, identify quote currencies and fetch their USD rates
fx_rates = {}
for idx, sym in self.market.assets.items():
    if sym in new_data:
        # Extract quote currency (EUR, GBP, BTC, etc.)
        quote = extract_quote_currency(sym)  # Import from portfolio.py

        # Add as FX pair if not USD
        if quote != 'USD':
            # Try to find quote/USD price in market data
            fx_pair = f"{quote}/USD"
            if fx_pair in new_data:
                fx_rates[fx_pair] = new_data[fx_pair]['close']
            # Also try without slash
            fx_pair_noslash = f"{quote}USD"
            if fx_pair_noslash in new_data:
                fx_rates[fx_pair_noslash] = new_data[fx_pair_noslash]['close']

self.portfolio.update_prices(new_prices, fx_rates)
```

**Additional Changes Needed:**
```python
# Add import at top of file:
from ..agent.portfolio import extract_quote_currency
```

**Testing:**
- [ ] Run training for 10 minutes without crashes
- [ ] Check logs for "No FX rate found" errors
- [ ] Verify non-USD positions update correctly
- [ ] Compare EUR/GBP/BTC pair rewards to USD pairs

---

## ⚠️ HIGH PRIORITY FIXES (Improves Model Quality)

### [ ] Fix #2: Improve Stop-Loss Trailing for Shorts

**File:** `/home/lukas/Trader/mcts_trader/agent/portfolio.py`
**Location:** Lines 350-359
**Severity:** HIGH
**Time Estimate:** 15 minutes

**Current Code:**
```python
else:  # short
    new_tp = current_price * (1 - self.short_take_profit_pct / 100)
    if new_tp < old_tp:
        pos.take_profit = new_tp

        # Trail SL to breakeven if we're up > 2%
        if pos.pnl_percent > 2.0 and pos.stop_loss > pos.entry_price:
            pos.stop_loss = pos.entry_price
```

**Replace With:**
```python
else:  # short
    new_tp = current_price * (1 - self.short_take_profit_pct / 100)
    if new_tp < old_tp:
        pos.take_profit = new_tp

        # Trail SL to lock in profits (move SL down as price drops)
        if pos.pnl_percent > 2.0:
            # Set SL to current price + stop_loss%
            trailing_sl = current_price * (1 + self.short_stop_loss_pct / 100)
            # Only tighten (move down), never loosen
            if trailing_sl < pos.stop_loss:
                pos.stop_loss = trailing_sl
                logging.debug(f"Trailing SL for short {pos.symbol}: {pos.stop_loss:.2f} -> {trailing_sl:.2f}")
```

**Testing:**
- [ ] Open short position in paper trading
- [ ] Wait for 2%+ profit
- [ ] Trigger reinforce signal (sell again)
- [ ] Verify SL moves DOWN (tighter), not stuck at breakeven
- [ ] Check that SL never moves UP (looser)

---

### [ ] Fix #3: Add Margin Call Protection

**File:** `/home/lukas/Trader/mcts_trader/agent/portfolio.py`
**Location:** Lines 551-563
**Severity:** MEDIUM
**Time Estimate:** 20 minutes

**Current Code:**
```python
# Base desired value from model's size input
base_desired_value_usd = self.state.total_value * self.max_position_pct * size

# Adjust for SPOT mode: Cannot spend more cash than available
if self.mode == 'SPOT' and action == 2: # Only applies to BUY in SPOT
    # ... existing SPOT check ...
else:
    max_value_usd = base_desired_value_usd
```

**Replace With:**
```python
# Base desired value from model's size input
base_desired_value_usd = self.state.total_value * self.max_position_pct * size

# Check available capital for ALL modes to prevent over-leverage
if action == 2:  # Buy (both SPOT and MARGIN)
    # For longs, we need cash to buy
    max_spendable = self.state.cash / (1 + self.trading_fee_rate)
    if base_desired_value_usd > max_spendable:
        logging.info(f"Capping buy order for {symbol} due to insufficient cash. Desired: ${base_desired_value_usd:.2f}, Available: ${max_spendable:.2f}")
        max_value_usd = max_spendable
    else:
        max_value_usd = base_desired_value_usd

elif action == 0 and self.mode == 'MARGIN':  # Sell (open short in MARGIN mode)
    # For shorts, check if we can cover potential losses
    # Rule: Total short exposure should not exceed total_value (prevents >100% short exposure)
    current_short_exposure = sum(
        pos.size * pos.current_price * get_quote_to_usd_rate(pos.quote_currency, self.fx_rates)
        for pos in self.state.positions.values()
        if pos.side == 'short'
    )

    max_allowed_short = self.state.total_value  # Can short up to 100% of equity
    remaining_short_capacity = max(0, max_allowed_short - current_short_exposure)

    if base_desired_value_usd > remaining_short_capacity:
        logging.warning(f"Capping short for {symbol} to prevent over-leverage. Desired: ${base_desired_value_usd:.2f}, Capacity: ${remaining_short_capacity:.2f}")
        max_value_usd = remaining_short_capacity
    else:
        max_value_usd = base_desired_value_usd
else:
    max_value_usd = base_desired_value_usd
```

**Testing:**
- [ ] Run training with $10k capital
- [ ] Try to open 5+ long positions at 30% each
- [ ] Verify buys are capped to available cash
- [ ] Try to open 2+ short positions at 100% each
- [ ] Verify shorts are capped to equity
- [ ] Check that total short exposure never exceeds 100% of equity

---

## 📊 VALIDATION TESTS (After Fixes Applied)

### [ ] Test 1: Multi-Currency Training

**Objective:** Verify FX rates work for all quote currencies

**Steps:**
1. Start training with all 64 assets
2. Monitor logs for first 100 episodes
3. Check for crashes on non-USD pairs
4. Verify portfolio values update on EUR/GBP/BTC pairs

**Success Criteria:**
- [ ] No "No FX rate found" errors in logs
- [ ] EUR/USD, GBP/USD, BTC/USD prices found
- [ ] Portfolio value changes when EUR pairs move
- [ ] Rewards are non-zero for non-USD pairs

---

### [ ] Test 2: Stop-Loss Trailing

**Objective:** Verify SL properly protects short profits

**Steps:**
1. Run paper trading or backtest
2. Open short position on volatile asset
3. Wait for price to drop 5%
4. Send "reinforce" signal (sell again)
5. Check stop-loss value

**Success Criteria:**
- [ ] SL moves down from entry price
- [ ] SL trails tighter as price drops
- [ ] SL never moves UP (looser)
- [ ] Position closes at SL if price reverses

---

### [ ] Test 3: Margin Protection

**Objective:** Verify over-leverage is prevented

**Steps:**
1. Create test environment with $1,000 capital
2. Try to open 10 long positions at $500 each
3. Check executed order sizes
4. Try to open 3 short positions at $1,000 each
5. Check total short exposure

**Success Criteria:**
- [ ] Total long cost never exceeds cash
- [ ] Total short exposure never exceeds equity
- [ ] Logs show "Capping order" messages
- [ ] Cash never goes negative during training

---

### [ ] Test 4: Accounting Verification

**Objective:** Spot-check accounting math after fixes

**Manual Test Cases:**

**Test 4A: USD Short**
```
Initial: $10,000
Action: Short 1 BTC @ $50,000
Expected Cash: $10,000 + ($50,000 - fee) ≈ $59,870
Expected Total: $59,870 - $50,000 = $9,870
Result: [ ] PASS / [ ] FAIL
```

**Test 4B: EUR Short**
```
Initial: $10,000
Action: Short 10 ETH/EUR @ €2,000 (FX: 1.1 USD/EUR)
Expected Cash: $10,000 + (€20,000 * 1.1 - fee) ≈ $31,943
Expected Total: $31,943 - €20,000 * 1.1 = $9,943
Result: [ ] PASS / [ ] FAIL
```

**Test 4C: Mixed Portfolio**
```
Initial: $10,000
Actions:
  - Buy 0.01 BTC @ $50,000 = -$500
  - Short 10 LINK @ $15 = +$150
Expected Cash: $10,000 - $500 + $150 - fees ≈ $9,647
Expected Total: $9,647 + $500 - $150 ≈ $9,997
Result: [ ] PASS / [ ] FAIL
```

---

## 🚀 RETRAINING CHECKLIST

### [ ] Pre-Training

- [ ] All 3 fixes applied
- [ ] Code committed to git with clear message
- [ ] Validation tests passed
- [ ] Database has fresh data (last 30 days)
- [ ] GPU available and tested

### [ ] During Training

- [ ] Monitor logs for FX rate errors (first 10 min)
- [ ] Check episode rewards are reasonable
- [ ] Verify all 64 assets are trading
- [ ] Watch for crashes on non-USD pairs
- [ ] Monitor GPU memory usage

### [ ] Post-Training

- [ ] Save checkpoint with clear name (e.g., `v2_fx_fixed_20251212`)
- [ ] Run backtest on historical data
- [ ] Compare performance to previous checkpoint
- [ ] Verify behavior on EUR/GBP/BTC pairs
- [ ] Paper trade for minimum 1 week

---

## 📝 GIT COMMIT MESSAGES (Copy-Paste Ready)

### After Fix #1:
```
fix(training): Add FX rate support to trading environment

Critical fix for multi-currency accounting. Training environment now
provides FX rates to portfolio manager, fixing corrupted valuation
on non-USD pairs (EUR, GBP, BTC quotes).

Resolves: Audit Bug #2 (CRITICAL)
Affected: 30-40 assets with non-USD quotes
```

### After Fix #2:
```
fix(portfolio): Improve stop-loss trailing for short positions

Stop-loss now properly trails downward as price drops, instead of
moving to breakeven once. Protects accumulated profits on shorts.

Resolves: Audit Bug #3 (HIGH)
Impact: Better risk management on short positions
```

### After Fix #3:
```
fix(portfolio): Add margin call protection

Prevents over-leverage by capping position sizes to available capital.
Long positions limited by cash, short positions limited by equity.

Resolves: Audit Bug #4 (MEDIUM)
Impact: More realistic training, prevents negative cash scenarios
```

---

## ⚠️ KNOWN RISKS & MITIGATIONS

### Risk 1: FX Rate Data Missing
**Symptom:** Training crashes with "No FX rate found" error
**Cause:** Database doesn't have EUR/USD, GBP/USD, etc. prices
**Mitigation:**
- [ ] Run `collect_data.py` to fetch FX pairs
- [ ] Verify database has: EUR/USD, GBP/USD, BTC/USD, USDT/USD, CHF/USD, CAD/USD
- [ ] Add fallback rates if some pairs missing

### Risk 2: Performance Degradation
**Symptom:** New checkpoint performs worse than old one
**Cause:** Margin limits reduce position sizes, stop-losses close early
**Mitigation:**
- [ ] This is EXPECTED - old model cheated with bugs
- [ ] New model is more conservative but HONEST
- [ ] Paper trade to verify it's not just timid

### Risk 3: Training Slower
**Symptom:** Fewer episodes per hour
**Cause:** FX rate lookups add overhead
**Mitigation:**
- [ ] Profile code to find bottleneck
- [ ] Consider caching FX rates per episode
- [ ] Acceptable tradeoff for correct accounting

---

## 📊 SUCCESS METRICS

How to know if fixes worked:

**Training:**
- [ ] Completes 1000 episodes without crashes
- [ ] Logs show NO "FX rate not found" errors
- [ ] Non-USD pairs have non-zero rewards
- [ ] Portfolio values update every step

**Backtesting:**
- [ ] Model trades both USD and non-USD pairs
- [ ] Stop-losses trigger correctly
- [ ] No negative cash scenarios
- [ ] Cross-pair strategies still work

**Paper Trading:**
- [ ] 1 week with no accounting errors
- [ ] Profit/loss matches Kraken's values
- [ ] Stop-losses execute on exchange
- [ ] Model doesn't over-leverage

**Real Money (Final Test):**
- [ ] Start with $100 only
- [ ] First 10 trades match expectations
- [ ] No margin calls or liquidations
- [ ] Model exhibits conservative behavior

---

## 🎯 COMPLETION TIMELINE

| Task | Est. Time | Status |
|------|-----------|--------|
| Apply Fix #1 (FX rates) | 30 min | [ ] |
| Apply Fix #2 (Stop-loss) | 15 min | [ ] |
| Apply Fix #3 (Margin) | 20 min | [ ] |
| Run validation tests | 1 hour | [ ] |
| Commit & document changes | 30 min | [ ] |
| **Total Pre-Training** | **2.5 hours** | [ ] |
| Run training | 12-24 hours | [ ] |
| Backtest new checkpoint | 2 hours | [ ] |
| Paper trade (1 week) | 7 days | [ ] |
| **Total to Production** | **8-9 days** | [ ] |

---

**Audit Source:** `/home/lukas/Trader/audit_report_final.md`
**Questions?** See executive summary: `/home/lukas/Trader/AUDIT_EXECUTIVE_SUMMARY.md`

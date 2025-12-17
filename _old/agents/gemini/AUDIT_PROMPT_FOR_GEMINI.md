# Short Position Accounting Audit Prompt for Gemini

## Context

You are auditing the short position accounting system in a cryptocurrency trading agent. The user suspects the model may have either:
1. **Discovered a genius cross-pair arbitrage strategy** (e.g., long LINK/USD + short LINK/EUR = synthetic EUR/USD position)
2. **Found a money glitch/accounting bug** that allows it to generate unlimited capital

The model is trained with reinforcement learning and has learned to heavily favor short positions (user had to implement a max 4 shorts limit).

## Your Task

**Perform a comprehensive audit of short position accounting to identify any potential money glitches, accounting errors, or violations of conservation of capital.**

## Key Files to Analyze

### 1. `/home/lukas/Trader/mcts_trader/agent/portfolio.py`
**Focus Areas**:
- Lines 711-736: Short position open/close cash flows
- Lines 285-305: Total portfolio value calculation with short PnL
- Lines 475-481: FX-adjusted exposure calculation
- Lines 499-508: Position sizing with FX conversion

### 2. `/home/lukas/Trader/mcts_trader/env/trading_env.py`
**Focus Areas**:
- The simulated trading environment used for training
- Check if accounting matches production portfolio.py
- Verify reward calculations don't create perverse incentives

### 3. `/home/lukas/Trader/mcts_trader/agent/real_portfolio.py`
**Focus Areas**:
- Lines 285-390: Real trading execution
- Verify it matches simulation accounting

## Critical Accounting Rules to Verify

### Short Position Cash Flow (MUST conserve capital)

**Opening a Short Position**:
```
Entry: Sell 1 BTC @ $50,000 in BTC/USD market
Expected Cash Flow:
- Cash INCREASES by: (size * price * fx_rate) - fee
- Example: Cash += (1 * 50000 * 1.0) - (50000 * 0.0026) = $49,870
- Position recorded: short BTC/USD, size=1, entry=$50000
```

**Closing a Short Position (Profit)**:
```
Exit: Buy to cover 1 BTC @ $40,000 (price dropped $10k)
Expected Cash Flow:
- Cash DECREASES by: (size * price * fx_rate) + fee
- Example: Cash -= (1 * 40000 * 1.0) + (40000 * 0.0026) = -$40,104
- Net Profit: $49,870 - $40,104 = $9,766 (correct: $10k profit - fees)
```

**Closing a Short Position (Loss)**:
```
Exit: Buy to cover 1 BTC @ $60,000 (price rose $10k)
Expected Cash Flow:
- Cash DECREASES by: (size * price * fx_rate) + fee
- Example: Cash -= (1 * 60000 * 1.0) + (60000 * 0.0026) = -$60,156
- Net Loss: $49,870 - $60,156 = -$10,286 (correct: $10k loss + fees)
```

### Portfolio Valuation (MUST be mathematically sound)

**Total Portfolio Value Formula**:
```python
total_value = cash + long_market_value + short_unrealized_pnl

# For shorts specifically:
short_unrealized_pnl = (entry_price - current_price) * size * fx_rate
```

**Example Verification (Multi-Currency)**:
```
Starting Capital: $10,000 USD cash

Action 1: Open short ETH/EUR
- Sell 10 ETH @ €2,000 (fx_rate: 1 EUR = 1.1 USD)
- Cash += (10 * 2000 * 1.1) - fee = +$21,945 (approx)
- Cash balance: $31,945
- Short PnL (immediately): $0 (entry = current)
- Total: $31,945 ✓

Action 2: Price moves to €2,200 (loss)
- Short PnL: (2000 - 2200) * 10 * 1.1 = -$2,200
- Total: $31,945 - $2,200 = $29,745 ✓

Action 3: Close short at €2,200
- Cash -= (10 * 2200 * 1.1) + fee = -$24,263
- Cash balance: $31,945 - $24,263 = $7,682
- Short PnL: $0 (position closed)
- Total: $7,682 ✓
- Net loss from trade: $10,000 - $7,682 = $2,318 (correct: €200 loss * 10 * 1.1 + fees)
```

## Specific Bugs to Search For

### 1. **Double-Counting Short PnL**
- Check if short PnL is added to cash AND counted again in total_value
- Symptom: Portfolio value increases when shorts lose money

### 2. **Sign Errors in FX Conversion**
- Check if FX rates are inverted (EUR/USD vs USD/EUR)
- Check if conversion is applied in wrong direction
- Symptom: Non-USD shorts have 2x or 0.5x the expected impact

### 3. **Fee Application Errors**
- Verify fees are charged on BOTH open and close
- Verify fees are in the correct currency (USD vs quote)
- Symptom: Short positions generate profit even at breakeven prices

### 4. **Position Sizing Miscalculation**
- Check if `max_value / price` ignores FX rate
- This was previously buggy (fixed in recent commit)
- Symptom: ETHBTC positions sized at $48M instead of $500

### 5. **Unrealized PnL Formula Error**
- Should be: `(entry - current) * size * fx_rate`
- Common mistakes:
  - `(current - entry)` (inverted)
  - Missing `* fx_rate`
  - Using wrong price (entry vs current)

### 6. **Cross-Pair Accounting Isolation**
- If model goes long LINK/USD and short LINK/EUR:
  - This creates EUR/USD exposure (possibly genius, not a bug)
  - BUT: Verify each position's accounting is independent
  - Verify no correlation-based value boost

### 7. **Action Masking vs Execution Mismatch**
- Training environment may allow actions that production blocks
- Check if model learned exploits that don't work in production
- Verify SPOT mode constraints are consistent

### 8. **Cash Reserve Interaction**
- Recently added 50% cash reserve policy
- Check if reserve calculation double-counts short proceeds
- Check if available_capital calculation is correct

## Audit Methodology

### Phase 1: Code Review
1. Read all three portfolio files completely
2. Trace through 5 manual scenarios (see templates above)
3. Verify every `cash +=` and `cash -=` line
4. Check every multiplication by `fx_rate`

### Phase 2: Scenario Testing
Create detailed walkthroughs for:
1. **Single USD Short**: Open/close with profit/loss
2. **Single EUR Short**: With FX conversion
3. **Mixed Portfolio**: 2 longs + 2 shorts simultaneously
4. **Cross-Pair Arbitrage**: Long LINK/USD + Short LINK/EUR
5. **Worst Case**: 4 shorts all go to stop-loss

### Phase 3: Training Environment Comparison
1. Compare `trading_env.py` reward calculation to production PnL
2. Check if simulation shortcuts create exploitable patterns
3. Verify episode termination conditions don't reward glitches

### Phase 4: Report Findings
For each potential bug:
- **Location**: File and line number
- **Severity**: Critical (money glitch) / High / Medium / Low
- **Proof**: Numerical example showing the error
- **Impact**: How much this could inflate portfolio value
- **Fix**: Proposed code correction

## Expected Outputs

### If No Bugs Found:
Provide evidence that the model likely discovered legitimate cross-pair tactics:
- Explain how long/short pairs create synthetic positions
- Show why this is mathematically sound
- Recommend: Don't retrain, study the strategy!

### If Bugs Found:
Provide:
1. Complete list of accounting errors with severity
2. Code fixes for each bug
3. Estimate of "phantom capital" the model was exploiting
4. Recommendation on whether to retrain or just fix and continue

## Additional Context

**Recent Changes** (already implemented):
- ✅ Fixed position sizing FX conversion (was causing $48M positions)
- ✅ Fixed exposure calculation FX conversion
- ✅ Added max 4 shorts limit
- ✅ Added 50% cash reserve for margin protection
- ✅ Fixed NoneType comparison error in execute_actions
- ✅ Enhanced symbol mapping (ETHBTC, BTCGBP, etc.)

**User Observations**:
- Model heavily favors short positions (required hard limit)
- User suspects either genius or glitch
- Training on 64 assets with multi-currency quotes (USD, EUR, GBP, BTC, CHF, CAD)
- Model uses TRM (Tiny Recursive Model) with MCTS planning

**Risk Concerns**:
- User is OK losing 100% of capital
- User is NOT OK with margin liquidation (1000% loss)
- Max leverage: 1x
- Stop-loss: 5% for longs, 3% for shorts

## Start Your Audit

Begin by reading the three portfolio files completely, then systematically work through the phases above. Be extremely thorough - this system handles real money and a subtle bug could cause catastrophic losses.

Good luck!

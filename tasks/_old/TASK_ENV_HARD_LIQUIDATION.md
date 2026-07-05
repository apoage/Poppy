# Task: Environment Stability & Hard Liquidation Protocol

## 🚨 Critical Priority: Stability Fix
**Status:** DONE
**Component:** `mcts_trader/env/vectorbt_env.py`

## Context
The current `VectorBTTradingEnv` lacks "bankruptcy protection." When an untrained agent ("baby mode") aggressively loses money, it can drive the portfolio value near or below zero. The current logic does not strictly prevent execution in these edge cases, leading to a **"Death Spiral"**:

1.  **Negative Cash Trap:** `portfolio.cash` drops to negative values (due to fees/margin logic gaps).
2.  **Ghost Shares:** The order sizing logic (`shares = net_value / price`) calculates **negative shares** if cash/value is negative.
3.  **Inverted PnL:** A "Long" position with negative shares mathematically acts like a Short position, flipping the reward signal (losing money = positive reward).
4.  **NaN Explosion:** Percentage return calculations divide by near-zero values, generating NaNs that propagate through the JAX gradients and ruin the training run.

## Objective
Implement a "Hard Liquidation" protocol and strict sanity checks to ensure the environment fails gracefully (terminates) rather than breaking mathematically.

## Requirements

### 1. Hard Liquidation (Early Termination)
**Goal:** Stop the episode immediately if the agent is effectively bankrupt. This saves GPU cycles by not simulating 200 steps of a "dead" agent.

*   **Logic:** In `step()`, check `total_value` before or after execution.
*   **Condition:** `if total_value < (initial_cash * 0.01):` (1% threshold).
*   **Action:**
    *   Set `terminated = True`.
    *   (Optional) Override `reward` with a significant penalty (e.g., `-10.0` symlog equivalent) to discourage bankruptcy.
    *   Log a warning/info indicating "Bankruptcy Termination".

### 2. Action Guardrails (The "Cash Floor")
**Goal:** Prevent functions like `_execute_buy` from running when funds are insufficient.

*   **Logic:** Add explicit checks at the start of `_execute_buy` and `_execute_limit_buy`.
*   **Condition:** `if self.portfolio.cash < MIN_CASH_THRESHOLD` (e.g., $10.00).
*   **Action:** Return immediately (effectively a "Hold"). Do not attempt to calculate share size.

### 3. Share Sizing Sanity
**Goal:** Prevent negative or zero share counts.

*   **Logic:** In all execution methods (`_fill_limit_order`, `_execute_buy`, etc.).
*   **Condition:** `if shares <= 0: return` (Already partially present, but ensure it handles negative inputs correctly).
*   **Assertion:** Consider adding `assert shares > 0` or equivalent logging if calculation yields negative numbers.

### 4. Zero-Division Defense
**Goal:** Prevent NaN rewards.

*   **Logic:** In the `pct_return` calculation in `step()`.
*   **Change:**
    ```python
    # Current
    pct_return = (new_value - prev_value) / prev_value * 100

    # Proposed
    epsilon = 1e-8
    pct_return = (new_value - prev_value) / (prev_value + epsilon) * 100
    ```

## Success Criteria
*   An agent that loses 99% of its portfolio triggers an immediate episode end.
*   No `NaN` values ever appear in the logs, even with random/suicidal actions.
*   Training logs show "Episode Length" < `max_steps` for failed runs (indicating early termination works).

---

## Implementation Summary (2025-12-30)

### Constants Added (vectorbt_env.py:70-83)
```python
LIQUIDATION_THRESHOLD = 0.01  # 1% of initial capital
MIN_CASH_THRESHOLD = 10.0     # $10 minimum for buy orders
LIQUIDATION_PENALTY_RAW = -50.0  # Large negative reward (symlog → -3.9)
```

### Changes Made

1. **Hard Liquidation (step(), lines 947-970)**
   - After calculating `new_value`, checks if `new_value < initial_cash * 0.01`
   - Sets `terminated = True` and applies `-50.0` raw penalty
   - Prints warning: "⚠️ LIQUIDATION: Portfolio $X.XX < threshold $Y.YY"
   - Adds `liquidation_triggered` flag to info dict

2. **Zero-Division Defense (step(), line 969-970)**
   - Added epsilon to prevent NaN: `pct_return = (new_value - prev_value) / (prev_value + 1e-8) * 100`

3. **Cash Floor Guardrails (_execute_buy, _execute_limit_buy)**
   - Added early return if `self.portfolio.cash < MIN_CASH_THRESHOLD`
   - Prevents executing buy orders when effectively bankrupt

4. **Share Sizing** (already implemented)
   - All share calculations already had `if shares <= 0: return` checks

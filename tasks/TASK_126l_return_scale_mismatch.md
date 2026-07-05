# TASK_126l: Return Scale Mismatch Audit

## Findings

### 1. Scale Discrepancy Confirmed
There is a fundamental mismatch in how returns are calculated between the baseline/agent metrics and the optimal oracle metrics.

-   **Market Return (`market_return`):** Calculated as **Arithmetic (Linear) Percentage Return**.
    -   Location: `mcts_trader/training/self_play.py`
    -   Formula: `(Price_end - Price_start) / Price_start * 100`
    -   Example: Price 100 -> 108 = **+8.00%**

-   **Agent Return (`agent_return`):** Calculated as **Arithmetic (Linear) Percentage Return**.
    -   Location: `mcts_trader/training/self_play.py`
    -   Formula: `(Portfolio_end / Portfolio_start - 1.0) * 100`

-   **Optimal Alpha (`optimal_alpha`):** Calculated as **Scaled Cumulative Log Return**.
    -   Location: `mcts_trader/jax_agent/trainer.py`
    -   Formula: `sum(log(Price_t / Price_t-1)) * 100` = `log(Price_end / Price_start) * 100`
    -   Example: Price 100 -> 108 = `ln(1.08) * 100` = **+7.70%**

### 2. Impact of Discrepancy
The use of log returns for the oracle systematically **underestimates** the optimal performance for positive returns compared to the linear scale used for the market and agent.

-   **Small Moves (e.g., 1%):** `ln(1.01) * 100` ≈ 0.995%. (Negligible difference)
-   **Moderate Moves (e.g., 8%):** `ln(1.08) * 100` ≈ 7.70%. (0.3% difference)
-   **Large Moves (e.g., 50%):** `ln(1.50) * 100` ≈ 40.55%. (**~9.5% difference**)
-   **Negative Moves (e.g., -50%):** `ln(0.50) * 100` ≈ -69.31%. (Linear is -50%).

This creates a fairness issue:
-   **Upside:** The oracle's "perfect" performance is reported as lower than it would be in linear terms (40.5% vs 50%).
-   **Downside:** The oracle's short performance (if shorts use negative log returns) might be overstated in magnitude, or if long-only, the drops are deeper in log space.

### 3. Comparison Validity
The metrics are **not strictly comparable** for large moves. Comparing `optimal_alpha` (log scale) directly to `market_return` (linear scale) is mathematically unsound for volatile assets, although they converge for small returns (high-frequency trading).

### 4. Alpha Computation in `self_play.py`
The `alpha` metric in `self_play.py` is consistent with `market_return` and `agent_return`:
```python
'alpha': final_return - market_return
```
Since both `final_return` and `market_return` are linear percentages, `alpha` is also a linear percentage difference. This part is consistent.

## Recommendations

1.  **Standardize on Log Returns (Preferred for RL):**
    -   Log returns are additive over time and symmetric, making them better for reward signals and long-term accumulation.
    -   Change `market_return` and `agent_return` calculation in `self_play.py` to use log returns:
        ```python
        # Proposed change
        agent_log_return = np.log(new_value / initial_value) * 100
        market_log_return = np.mean(np.log(end_prices / start_prices)) * 100
        ```

2.  **Or, Standardize on Linear Returns (Preferred for Reporting):**
    -   Linear returns are what humans/banks expect ("I made 10%").
    -   Update `trainer.py` to convert the optimal path's log return back to linear before logging:
        ```python
        # Proposed change
        optimal_linear_return = (np.exp(optimal_path.total_alpha) - 1.0) * 100.0
        metrics['optimal_alpha'] = optimal_linear_return
        ```

3.  **Label Clarity:** If keeping mixed scales, clearly label them as `optimal_log_return` vs `market_linear_return` to avoid confusion.

## Conclusion
`market_return` uses **linear percentage** while `optimal_alpha` uses **log percentage**. For an 8% move, this causes a ~0.3% discrepancy. For volatile markets, this gap widens significantly.

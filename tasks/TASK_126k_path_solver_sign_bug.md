# TASK_126k: Path Solver Sign Bug Audit

## Findings

### 1. PnL Computation Trace
In `greedy_lookahead_path`:
```python
step_returns = returns[t]
signs = jnp.sign(step_returns)
step_pnl = jnp.sum(jnp.abs(step_returns) * new_mask)
```
- `jnp.abs(step_returns)` is equivalent to `jnp.sign(step_returns) * step_returns`.
- This calculation correctly represents the PnL of a **Long/Short** oracle that goes Long when returns are positive and Short when returns are negative.
- The `signs` variable is indeed computed but unused, which is a code smell but **not a logic error** for a Long/Short oracle.

### 2. Oracle Logic
- **Should the oracle go long on positive, short on negative?** Yes, for a perfect unconstrained hindsight oracle, this maximizes returns.
- **Is `jnp.abs` correct?** Yes. `abs(r)` captures the magnitude of the move, which is the profit captured if positioned correctly (Long or Short).

### 3. Discrepancy: Optimal Alpha < Market Return
If the oracle's alpha is less than the buy-and-hold market return, the cause is likely **not** the sign logic, but one of the following:

#### A. Transaction Costs (Likely Culprit)
`greedy_lookahead_path` has a default `transaction_cost = 0.001` (10 basis points).
- In high-frequency trading (minute bars), returns are often < 10 bps.
- A greedy oracle changes positions frequently (churn).
- **Impact:** High churn + 10bps cost per trade can easily result in negative PnL, significantly underperforming a zero-cost buy-and-hold baseline.

#### B. Normalization Inconsistency
- `greedy_lookahead_path` returns `sum` of top-k returns (scale ≈ k * avg_return).
- `soft_topk_path` returns `mean` of top-k returns (scale ≈ avg_return).
- `market_return` is typically an average.
- If `greedy` is used, values should be roughly `k` times larger. If they are smaller, the transaction cost penalty is likely catastrophic.

### 4. Recommendations

1.  **Remove Unused Variable:** Remove `signs = jnp.sign(step_returns)` to clean up the code, OR change the PnL line to `jnp.sum(signs * step_returns * new_mask)` to make the Long/Short assumption explicit.
2.  **Configurable Costs:** The `transaction_cost` in `greedy_lookahead_path` should be exposed to the `Trainer` config or set to 0.0 by default for "theoretical max alpha" calculations.
3.  **Normalization:** Standardize `greedy` and `soft` solvers to both return normalized (per-unit-capital) PnL (divide by `k`) or unnormalized sums. `greedy` currently returns Sum, `soft` returns Mean.

## Conclusion
The "sign bug" is a false positive (mathematically equivalent). The performance drop is likely due to the **default transaction cost of 10bps** punishing the greedy oracle's high turnover.

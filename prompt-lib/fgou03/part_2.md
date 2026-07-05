# Part 2: Oracle Math Consistency

## Status: FAIL

## Findings
- `simulate_oracle_pnl` in `tools/benchmark_oracle_vs_model.py` uses the formula `step_pnl = np.sum(np.abs(log_returns[t]) * mask)`. It does NOT divide by `k_count`.
- `simulate_model_pnl` in `tools/benchmark_oracle_vs_model.py` uses `step_pnl = np.sum(position * log_returns[t])`. It also does NOT normalize by the number of active positions.
- In contrast, the training oracle (`greedy_lookahead_path` in `mcts_trader/jax_agent/path_solver.py`) computes `step_pnl = jnp.sum(jnp.abs(step_returns) * new_mask) / k_count`.
- A comment in `simulate_oracle_pnl` (BUG-9 fix) explicitly mentions removing the `/ k_count` division to match model P&L normalization. However, this is financially incorrect. If you invest a fixed amount of capital, you split it evenly among active positions. Without dividing by the number of active positions (or `k_count`), the benchmark assumes $100\%$ capital allocation per selected asset, resulting in unrealistic multi-x leverage (e.g., 4x leverage for `k=4`).
- Since the model output might have a different number of active positions than the oracle, comparing unnormalized P&Ls directly compares portfolios with different amounts of leverage.

## Bugs Found
- **BUG-2.1**: Unnormalized P&L calculations in `benchmark_oracle_vs_model.py` lead to unrealistic leverage assumptions and unfair comparisons between the model and oracle. 

## Recommended Fixes
- Restore normalization in `simulate_oracle_pnl`:
```python
        k_count = max(np.sum(mask), 1)
        step_pnl = np.sum(np.abs(log_returns[t]) * mask) / k_count
```
- Add equivalent normalization in `simulate_model_pnl`:
```python
        active_positions = np.sum(np.abs(position) > 0)
        n_active = max(active_positions, 1)
        step_pnl = np.sum(position * log_returns[t]) / n_active
```
- By dividing by the number of active positions, both represent a $1	imes$ leveraged portfolio where capital is distributed equally among all current trades.
# TASK_126b: Reward Drop Audit & Numerical Stability Fixes

## Problem Statement
Epoch 27 showed a catastrophic reward drop to -30.99 while total return (alpha) was only -4.62%.

## Root Cause Analysis

### 1. Alpha-Relative Reward Scale Mismatch (CRITICAL)
In `trainer.py`, the `alpha_relative_reward` logic (TASK_118) normalizes rewards by dividing by `max_alpha`.
- `max_alpha` is computed from raw log returns (e.g., 0.03 for 3%).
- `rewards` in the buffer are already scaled by 100 (percentage scale, e.g., 1.0 for 1%).
- **Bug:** Dividing a percentage-scale number by a raw-scale number (1.0 / 0.03) results in a **100x inflation** of the reward signal.
- This explains why a -4.6% alpha could translate to a -30 or -400 reward signal.

### 2. Phantom Rewards after Termination (MAJOR)
In `run_self_play_jax` (`self_play.py`), the `score` is computed as `np.sum(cpu_rewards, axis=1)`.
- **Bug:** This sums ALL steps in the 400-step trajectory, even if the episode ended early (e.g., at step 10 due to liquidation).
- In `jax_trading_env.py`, `is_liquidated` stays true after bankruptcy, and the agent continues to receive a `symlog(-50.0) = -3.93` penalty every step for the remainder of the 400 steps.
- **Impact:** Bankruptcy at step 390 results in an extra -39.3 reward penalty that does not exist in the actual portfolio performance.

### 3. Inconsistent State Updates (CONSISTENCY)
In `jax_trading_env.py`, `new_cumulative_return` and `prev_portfolio_value` are computed **before** random burns are applied.
- **Bug:** The agent's `agent_return` metric (cumulative return) does not "see" the loss from random burns, but the `reward` signal does.
- Additionally, `prev_portfolio_value` is not updated, causing the **next** step's log return to be computed from an artificially high baseline, effectively penalizing the agent a second time for the same burn.

### 4. Differential Reward Unit Mismatch (MINOR)
In `train_jax_fast.py`, the manual differential reward adjustment subtracts raw `market_returns` (e.g., 0.0001) from `symlog`-scaled percentage rewards (e.g., 0.01).
- **Bug:** The adjustment is mathematically negligible because of the 100x scale difference.

## Proposed Fixes

1.  **Scale Alignment in Trainer**: Ensure `max_alpha` is multiplied by 100 before dividing percentage-scaled rewards.
2.  **Score Masking**: Apply the `done` mask to rewards before summing for the episode `score` in `self_play.py`.
3.  **Order of Operations**: Move random burn logic before portfolio value and return calculations in `jax_trading_env.py`.
4.  **Unit Alignment**: Scale market returns by 100 in the training loop adjustment.

## Status: Documented. Implementation pending.

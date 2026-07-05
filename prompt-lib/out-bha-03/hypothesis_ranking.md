# Hypothesis Ranking

1.  **Change 1: Model-State Rollout (Target Mismatch)**
    *   **Risk**: CRITICAL.
    *   **Reasoning**: The model is trained on `Student_State` (from rollout) but `Oracle_Action` targets are derived from `Oracle_State`. When student and oracle positions diverge (which happens immediately), the target action may be invalid or counter-productive for the student's current position (e.g., Oracle says "Hold" to stay Long, but Student is Short and needs to "Buy"). This "off-policy" label noise forces the model to ignore state signals and default to the majority class (Hold).
    *   **Likelihood**: **CONFIRMED**.

2.  **Change 8: Model Construction (Capacity Reduction)**
    *   **Risk**: MEDIUM-HIGH.
    *   **Reasoning**: The `trm_hidden_dim` was explicitly set to 1024 (Large preset). Previously, it defaulted to `state_dim * 16` (128 * 16 = 2048). This is a 50% reduction in the reasoning layer's capacity. While unlikely to cause a complete collapse to ~7%, it significantly reduces the model's power compared to the "known-good" run.
    *   **Likelihood**: POSSIBLE contributor, but not the primary killer.

3.  **Change 7: Optimizer Warmup & Trade Weights**
    *   **Risk**: MEDIUM.
    *   **Reasoning**: `weight_warmup_epochs` defaults to 0 (was 50). This applies full `adaptive_trade_weight` (up to 64x) from epoch 0. This aggressive weighting might shock the model initially, but theoretically should encourage *more* trading, not less. The fact that the model remains a "turtle" despite this huge trade incentive reinforces the theory that the trade signal itself (the target) is flawed (Hypothesis 1).
    *   **Likelihood**: UNLIKELY to be the root cause.

4.  **Change 2: Returns Alignment**
    *   **Risk**: LOW.
    *   **Reasoning**: The slicing `log_returns[ws - 1:]` aligns the return at index 0 with the PnL update for the position held during that interval. The scan logic updates PnL using `pos` (state at start of step) and `ret`. This seems mathematically consistent.
    *   **Likelihood**: UNLIKELY.

5.  **Change 3: Confidence Channel**
    *   **Risk**: LOW.
    *   **Reasoning**: The carry/emit logic for `prev_conf` correctly mirrors the inference-time behavior (feeding `t-1` confidence into `t` observation).
    *   **Likelihood**: UNLIKELY.

6.  **Change 4: Trade-Biased Sampling**
    *   **Risk**: LOW.
    *   **Reasoning**: The filter `valid_trades = trade_indices[trade_indices <= max_offset]` correctly prevents out-of-bounds sampling. Unless trade density is zero (which it isn't), this should work as intended.
    *   **Likelihood**: UNLIKELY.

7.  **Change 5: Buffer Param ID Fix**
    *   **Risk**: LOW.
    *   **Reasoning**: Using `-1` sentinel correctly marks episodes for refresh.
    *   **Likelihood**: UNLIKELY.

8.  **Change 6: Oracle Mask Head**
    *   **Risk**: NEGLIGIBLE.
    *   **Reasoning**: Loss contribution is tiny (0.8%).
    *   **Likelihood**: UNLIKELY.

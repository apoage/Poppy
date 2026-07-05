# TASK_127j: MCTS Search Integration Audit

## Findings

### 1. Implementation (`mctx_search.py`)
-   **Library Usage:** The code correctly wraps `mctx.gumbel_muzero_policy` and `mctx.muzero_policy`. This offloads the complexity of UCB/PUCT and tree management to a verified Google library.
-   **Parameter Passing:** Critical parameters (`discount`, `dirichlet_alpha`, `gumbel_scale`) are passed correctly.
-   **Value Calibration:** The wrapper adds a custom `value_scale` / `value_shift` post-processing step. This is useful for domain adaptation but requires monitoring to ensure it doesn't skew targets.
-   **Epsilon Greedy Targets:** The implementation modifies the *output* policy probabilities with epsilon-greedy noise (TASK_037). This is intended to "soften" the training targets and prevent policy collapse, distinct from adding exploration noise *during* search.

### 2. Integration (`trainer.py`)
-   **Policy Targets:** The trainer uses the visit count distributions (softened by epsilon-greedy) as the regression targets for the policy head.
-   **Value Targets:** The trainer uses the computed returns (MC or GAE) from `self_play.py` as targets for the value head. This is the correct approach (learning from actual outcomes rather than bootstrapping from MCTS estimates indefinitely).

### 3. Gradient Flow
-   **Decoupled:** MCTS is a non-differentiable operation used to generate improved targets. The network learns to predict these improved targets. This standard MuZero loop is implemented correctly.

## Recommendations
-   **Monitor `epsilon_greedy`:** Ensure this parameter isn't set too high (e.g., >0.2), as it dilutes the "smart" MCTS signal with uniform noise.
-   **Value Scale:** Verify that `value_scale` defaults to 1.0 unless explicitly tuned, as incorrect scaling could destabilize the value loss.

## Conclusion
The MCTS integration is sound and leverages standard libraries effectively.

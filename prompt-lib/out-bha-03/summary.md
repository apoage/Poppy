# Summary of Regression Analysis

**Investigation Target**: Trade learning collapse (~15% -> ~7% `agree_trade`) after commits `06c44ab..77cb089`.

## Root Cause
The primary cause is a **Target Mismatch** introduced in `TASK_692` (Model-State Rollout).
The training pipeline now generates input states using the **Student's** autoregressive rollout (positions derived from model actions) but continues to use target actions derived from the **Oracle's** trajectory.

This creates an off-policy learning problem: the target actions are only optimal if the agent is already in the Oracle's position. When the Student's position diverges (which happens immediately for an untrained model), the target actions become incorrect for the Student's state (e.g., Oracle says "Hold" to stay Long, but Student is Short and must "Buy" to reach the optimal state). This contradictory signal forces the model to ignore its state and collapse to the majority class (Hold).

## Contributing Factors
1.  **Capacity Reduction**: The `large` preset's reasoning dimension was implicitly reduced from 2048 to 1024 due to explicit config passing in `easy_trainer.py`. This 50% capacity cut likely reduced the model's ability to cope with the noisy signals.
2.  **Aggressive Trade Weighting**: `weight_warmup_epochs` was set to 0, applying 64x trade weighting from epoch 0. While intended to help, in the presence of incorrect targets, this likely amplified the confusing gradients.

## Recommendations

1.  **Fix Target Mismatch (CRITICAL)**:
    Modify `train_easy.py` to re-compute `policy_targets` during episode generation when `use_model_rollout` is active. The targets must be computed using `optimal_path_to_policy` with `prev_selections` set to the **Student's positions** (from the rollout), not the Oracle's positions.

2.  **Restore Model Capacity**:
    Update `easy_config.py` to set `trm_hidden_dim=2048` for the large preset, restoring the previous architecture capacity.

3.  **Restore Warmup**:
    Set `weight_warmup_epochs=50` in `easy_config.py` to allow the model to stabilize its state representations before being forced to trade aggressively.

## Conclusion
The regression is a classic **DAgger implementation error**: gathering data from the student's policy but labeling it with the expert's *action* (which is state-dependent) instead of querying the expert for the optimal action *from the student's state*. Fixing this target alignment should restore trade learning.

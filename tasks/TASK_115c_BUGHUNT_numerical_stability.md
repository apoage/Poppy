# TASK_115c: Logic Error Audit

**Status**: Pending
**Priority**: High
**Created**: 2026-01-25

## Identified Issues

### 1. Inefficient Prediction Loop (Trainer)
*   **File**: `mcts_trader/jax_agent/trainer.py`
*   **Issue**: `_compute_prediction_loss` calls `predict_multi_step` with `num_steps=config.multi_step_k` (default 5), but only consumes `k` steps (limited by available `teacher_sequence` length, usually 4 for a 5-step unroll). This results in computing the 5th prediction step (t=5) but discarding it unused, wasting ~20% of the prediction head computation.
*   **Fix**: Pass `k` (the supervised steps) to `predict_multi_step` instead of the full `config.multi_step_k`.

### 2. Missing Terms in Student-t Loss (Prediction)
*   **File**: `mcts_trader/jax_agent/prediction/student_t.py`
*   **Issue**: `student_t_nll` drops terms depending on `df` (`gamma` functions) because they are "constants". However, `df` is a **learnable parameter**. Dropping these terms breaks the optimization of `df`, likely causing it to drift to infinity (Gaussian) or zero without proper regularization. The normalization constant is critical for comparing likelihoods across different degrees of freedom.
*   **Fix**: Include the full log-gamma terms in `student_t_nll`.

### 3. Potential Off-by-One in Replay Buffer Sampling
*   **File**: `mcts_trader/jax_agent/replay_buffer.py`
*   **Issue**: While the logic for `start_pos + pos` seems robust for wrapped buffers, there is a risk if `start_pos` points to the *very end* of a trajectory segment before a wrap. The logic `start_pos + pos + jnp.arange(sequence_length)` assumes contiguous memory in logical space.
    *   Verified: The buffer implementation uses a circular array and `physical_idx = logical_idx % capacity`. The `logical_data_indices` can span across the wrap boundary correctly.
    *   However, `trajectory_starts` stores the physical start index. If a trajectory wraps, does `logical_data_indices` correctly trace the wrapped segments?
    *   Example: Cap=100. Traj len=10. Start=95.
    *   Data at: 95, 96, 97, 98, 99, 0, 1, 2, 3, 4.
    *   `start_pos` = 95.
    *   Sample `pos` = 8.
    *   Logical indices: 95+8, 95+9, ... = 103, 104.
    *   Physical: 103%100=3, 104%100=4.
    *   Matches data at 3, 4. **Logic holds.**
    *   *Conclusion*: No off-by-one found in replay buffer logic itself, but rely on `add_trajectory` correctly handling the split write.

## Plan
1.  Fix `student_t_nll` to include log-gamma terms.
2.  Optimize `predict_multi_step` call in `trainer.py`.
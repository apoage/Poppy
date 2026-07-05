# TASK_136d: Reduce BC Training Frequency (Investigation)

## 1. Analysis of Current Implementation

In `mcts_trader/jax_agent/trainer.py`, the `train` loop iterates `num_iterations` times (default 50).

```python
for iteration in range(num_iterations):
    # ... sampling ...

    # TASK_125: Sample from expert buffer and add to batch for BC loss
    if expert_buffer is not None and self.expert_bc_config.enabled:
        expert_batch = expert_buffer.sample_batch(...)  # <--- RUNS EVERY ITERATION
        if expert_batch is not None:
            batch['expert_batch'] = expert_batch

    # ... training step ...
```

**Frequency:** Every single training step (50 times per epoch).
**Cost:**
*   `sample_batch` encodes 256/1024 observations.
*   If encoding takes 2s, total cost = 50 * 2s = 100s per epoch.
*   Total epoch time is ~144s, so this is ~70% of runtime.

## 2. Research & Theory

*   **Auxiliary Losses:** BC is an auxiliary loss to guide representation and policy. It doesn't need to be computed on *every* gradient step to be effective.
*   **Gradient variance:** The BC gradient is relatively stable (supervised).
*   **Expert ratio:** We already mix expert data into the batch (12.5%). Updating the BC loss every N steps is equivalent to lowering the effective weight but processing larger batches less often.
*   **Literature:**
    *   **DQfD:** Pre-training phase is 100% BC. Online phase mixes demo data into replay buffer (which we do). It doesn't typically run a *separate* BC update every step on top of the replay update.
    *   **R2D2 / MuZero Unplugged:** Uses a fixed ratio of expert data in the batch.

## 3. Proposal: Reduced Frequency

We can run BC sampling and training only every **5th or 10th iteration**.

**Mechanism:**
1.  Add `bc_frequency` to `ExpertBCConfig`.
2.  In `train()` loop, check `if iteration % bc_frequency == 0`.
3.  If skipped, pass `None` for `expert_batch` (the loss function handles this by returning 0).

**Impact Estimate:**
*   Current: 50 calls @ 2s = 100s.
*   Frequency 10: 5 calls @ 2s = 10s.
*   Savings: 90s per epoch.
*   New epoch time: ~54s (down from 144s). **~2.6x Speedup**.

**Learning Risk:**
*   Reduced BC signal might slow down initial "cloning".
*   Mitigation: Increase `bc_weight` by factor of 5 to compensate for 1/5th frequency? No, simpler to just accept slower BC convergence (which is fine, we train for hundreds of epochs). Or just rely on the high quality of the demos.

## 4. Alternate Strategy: Sample Reuse

We could sample ONE expert batch at the start of the epoch (or every 10 steps) and reuse it for multiple updates.
*   **Pros:** Enforces the same constraint multiple times (consistency). Fast.
*   **Cons:** Overfitting to that specific batch.
*   **Verdict:** Frequency reduction is safer and standard.

## 5. Implementation Plan

1.  **Modify `ExpertBCConfig`**: Add `bc_update_freq: int = 5` (default).
2.  **Modify `trainer.py`**:
    ```python
    # Inside train loop
    do_bc_update = (expert_buffer is not None and
                    self.expert_bc_config.enabled and
                    iteration % self.expert_bc_config.bc_update_freq == 0)

    if do_bc_update:
        expert_batch = expert_buffer.sample_batch(...)
    ```

## 6. Conclusion

Reducing BC frequency is a **high-impact, low-risk** optimization. Even with the encoding bottleneck (until 136a is fixed), this will drastically improve throughput.

**Recommendation:** Implement immediately with default frequency = 5.

### REGRESSION-1: Model-State Rollout Target Mismatch

**Suspect Commit**: `06c44ab` (TASK_692/693: Model-state rollout)
**Change**: Change 1 (Model-State Rollout logic in `train_easy.py`)
**Confidence**: CONFIRMED

**Why It Kills Trade Learning**:
The training loop generates input states using the **Student's** trajectory (`_model_state_rollout`) but uses target actions derived from the **Oracle's** trajectory (`optimal_path_to_policy` with `prev_selections=oracle_pos`).

This creates a fundamental "off-policy" mismatch (Covariate Shift).
*   **Scenario**: Oracle is Long, Student is Short (due to random init or previous error).
*   **Oracle Intent**: "Stay Long" (Action: HOLD).
*   **Student Input**: Position = Short.
*   **Student Target**: Action = HOLD.
*   **Result**: Student learns "Short + Hold -> Short".
*   **Correct Behavior**: Student should "Buy" to flip Long and match the Oracle's optimal state.

By teaching the student to mimic the Oracle's *actions* (which are conditional on the Oracle's *state*) while the student is in a *different state*, we provide false gradients. The model learns that its state (position/PnL) is irrelevant to the target, or gets conflicting signals, leading it to collapse to the majority class (Hold).

**Evidence**:
In `train_easy.py`, `_generate_episode_data` (and batch version):

```python
# 1. Targets computed using Oracle's previous selections
policy_targets = optimal_path_to_policy(
    optimal_path.selections,
    log_returns,
    prev_selections=prev_selections,  # <--- ORACLE POSITIONS
    ...
)

# ...

# 2. States computed using Student's rollout
if use_model_rollout:
    state_channels, confidence = self._model_state_rollout(...)
    # ...
    # observations built from state_channels (STUDENT POSITIONS)

# 3. Mismatch: observations (Student) paired with policy_targets (Oracle context)
ep_data = {
    'observations': observations...,
    'actions': policy_targets...,
}
```

**Proposed Fix**:
Re-compute policy targets using the Student's position from the rollout. This changes the supervision from "Mimic Oracle Action" to "Go to Oracle Optimal State".

In `train_easy.py`, inside `_generate_episode_data` (and `_generate_batch_episodes`):

```python
if use_model_rollout:
    # ... existing rollout code ...
    state_channels, confidence = self._model_state_rollout(...)
    
    # NEW: Re-compute targets using STUDENT positions
    # Extract student position (channel 0)
    student_pos = state_channels[:, 0, :] # (n_valid, n_assets)
    
    # We need to slice optimal path selections to match the valid window
    # optimal_path.selections is (T, n_assets). 
    # Window i (student_pos[i]) predicts action for returns[i + ws - 1].
    # This action leads to state at i + ws.
    # So we want to target optimal_path.selections[i + ws].
    
    # Actually, optimal_path_to_policy takes (target_state, returns, prev_state).
    # We want: prev_state = student_pos[i]
    # target_state = optimal_path.selections[i + ws]
    # This generates the action to go from Student -> Optimal.
    
    # Slicing:
    start_idx = self.config.window_size
    # optimal selections aligned to decision points
    target_selections = oracle_selections[start_idx : start_idx + len(student_pos)]
    
    # Recalculate policy targets
    policy_targets_corrected = optimal_path_to_policy(
        target_selections,
        returns_aligned, # (n_valid, n_assets)
        prev_selections=student_pos,
        n_actions_per_asset=5,
        temperature=1.0
    )
    
    # Use policy_targets_corrected for 'actions' in ep_data
```

---

### REGRESSION-2: Model Capacity Reduction

**Suspect Commit**: `easy_trainer.py` Change 8
**Change**: Explicitly passing `trm_hidden_dim` from config.
**Confidence**: LIKELY CONTRIBUTOR

**Why It Matters**:
The `large` preset defines `trm_hidden_dim=1024`. Previously, `MuZeroNetworks` defaulted to `state_dim * 16`. For the large preset (`state_dim=128`), the default was `128 * 16 = 2048`.
Explicitly passing `1024` cut the reasoning layer width by **50%**. While unlikely to cause the "collapse" alone, it reduces the model's ability to recover complex patterns, potentially exacerbating the difficulty of learning from noisy/mismatched targets.

**Proposed Fix**:
Update `easy_config.py` `get_large_config` to use `trm_hidden_dim=2048` to match the previous implicit default, or accept the reduction as a new baseline (after fixing REGRESSION-1).

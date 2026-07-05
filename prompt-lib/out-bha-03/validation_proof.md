# Validation Proof

## REGRESSION-1: Model-State Rollout Target Mismatch

**Status**: CONFIRMED by code inspection.

**Evidence in `train_easy.py`**:

1.  **Oracle Path Generation (Lines 470-492)**:
    *   `optimal_path` is computed using `greedy_lookahead_path`.
    *   `oracle_selections` contains the oracle's optimal positions (0/1).
    *   `prev_selections` (if state-aware) is derived from `optimal_path.selections`.

2.  **Target Computation (Lines 494-500)**:
    *   `policy_targets` is computed via `optimal_path_to_policy`.
    *   **CRITICAL**: `prev_selections` passed here is the **Oracle's** previous state.
    ```python
    policy_targets = optimal_path_to_policy(
        optimal_path.selections,
        log_returns,
        prev_selections=prev_selections,  # <--- ORACLE STATE
        ...
    )
    ```

3.  **Student Rollout (Lines 527-545)**:
    *   `_model_state_rollout` is called if `use_model_rollout` is True.
    *   This generates `state_channels` (Student's trajectory) and `observations` containing the Student's state.

4.  **Data Assembly (Lines 591-597)**:
    *   The dictionary `ep_data` is constructed.
    *   It pairs `observations` (Student State) with `policy_targets` (Oracle Action valid for Oracle State).
    ```python
    ep_data = {
        'observations': observations[:end_idx - start_idx],  # STUDENT STATE
        'actions': policy_targets[start_idx:end_idx],        # ORACLE ACTION (mismatched)
        ...
    }
    ```

**Conclusion**: The model learns to map Student State -> Oracle Action. Since the Oracle Action depends on the Oracle State (which differs from Student State), the labels are effectively noise/adversarial when states diverge.

---

## REGRESSION-2: Model Capacity Reduction

**Status**: CONFIRMED by code inspection.

**Evidence in `easy_config.py`**:
*   `get_large_config` explicitly sets `trm_hidden_dim=1024`.
    ```python
    def get_large_config() -> EasyConfig:
        return EasyConfig(
            ...
            state_dim=128,
            trm_hidden_dim=1024,  # TASK_684: Constrain reasoning
            ...
        )
    ```

**Evidence in `mcts_trader/jax_agent/muzero_model.py`** (checked previously):
*   Default logic: `hidden_dim = self.hidden_dim or (self.state_dim * 16)`.
*   For `state_dim=128`, default would be `128 * 16 = 2048`.

**Evidence in `easy_trainer.py`**:
*   The trainer passes the config value explicitly:
    ```python
    self.model = MuZeroNetworks(
        ...
        hidden_dim=getattr(config, 'trm_hidden_dim', None),  # TASK_684
        ...
    )
    ```

**Conclusion**: The "Large" preset capacity was silently reduced by 50% (2048 -> 1024) compared to the behavior before `trm_hidden_dim` was added to the config.

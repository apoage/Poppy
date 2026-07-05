# TASK_127a: MuZero Dynamics & Architecture Audit

## Findings

### 1. Dynamics Model (`muzero_model.py`)
-   **Architecture:** Correct. The `dynamics` module uses a standard MLP architecture (`Dense` -> `LayerNorm` -> `Dense`) to predict the next state and reward from the current state and action.
-   **Normalization:** Good practice observed. `dynamics_state_norm` (`LayerNorm`) is applied to the predicted `next_state`, which is critical for stabilizing training during long unrolls.
-   **Gradient Flow:** **Verified.** There are no `stop_gradient` blockers in the main dynamics path. Gradients will correctly flow from future steps back through the dynamics model to the representation network.

### 2. Reward Head
-   **Architecture:** Correct. The reward prediction shares the second dense layer of the dynamics model but has its own output head (size `support_size`).
-   **Targets:** The target indexing in `trainer.py` (`rewards[:, t - 1]` inside the loop `for t in range(1, unroll_steps)`) aligns correctly with `actions[:, t - 1]`.

### 3. Loss & Unrolling (`trainer.py`)
-   **Unrolling:** The K-step unrolling loop is implemented correctly. It recurses on `latent_state` properly.
-   **Targets:** Value and Policy targets are correctly indexed (`target_values[:, t]`, `target_policies[:, t]`).
-   **Consistency Loss:** **MISSING.**
    -   Modern MuZero variants (like EfficientZero) typically include a **consistency loss** (e.g., negative cosine similarity) between the *predicted* next latent state (`dynamics(s_t, a_t)`) and the *actual* encoded next state (`representation(o_{t+1})`).
    -   This codebase **does not** implement this self-supervised loss in the main `_loss_fn`.
    -   *Note:* There is a `predict_multi_step` loss for the auxiliary price predictor (Phase 2), but this is different from the latent state consistency loss.

## Recommendations

1.  **Implement Consistency Loss (High Priority for Data Efficiency):**
    -   Add a projection head to `MuZeroNetworks`.
    -   In `trainer.py`, compute `proj_pred = project(next_latent_state)` and `proj_gt = stop_gradient(project(representation(next_obs)))`.
    -   Add `cosine_similarity_loss(proj_pred, proj_gt)` to the total loss.
    -   This acts as a powerful self-supervised signal, especially when rewards are sparse or noisy.

2.  **Reward/Value Head Separation (Optional):**
    -   Currently, `reward` and `next_state` share the `dynamics_dense2` layer. While common, separating them (or having separate MLPs after the concatenated state/action) can sometimes reduce interference between reward prediction and state transition learning.

## Conclusion
The core MuZero implementation is mathematically correct and gradient flows are valid. The system is missing **Consistency Loss**, which is a standard feature in modern sample-efficient MuZero implementations (EfficientZero).

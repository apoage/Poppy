# TASK_127f: PopArt Normalization Audit

## Findings

### 1. Implementation (`popart.py`)
-   **Mathematics:** The formulas for normalization, denormalization, and weight correction are mathematically sound and consistent with the PopArt paper.
    -   Scale correction: $\sigma_{old} / \sigma_{new}$
    -   Shift correction: $(\mu_{old} - \mu_{new}) / \sigma_{new}$
-   **State Management:** The `PopArtState` correctly tracks running statistics and "old" statistics to compute the delta for weight correction.

### 2. Integration (`trainer.py`)
-   **Execution Context:** The critical `normalize_and_correct` step is performed **outside** the JIT-compiled `train_step_fn`.
    -   This is the correct design. It prevents JAX tracing issues and ensures statistics updates (EMA) are decoupled from gradient updates (SGD).
-   **Parameter Updates:** The trainer explicitly replaces the model parameters with the corrected version: `self.state = self.state.replace(params=corrected_params)`. This ensures the network "tracks" the changing scale of the rewards without catastrophic forgetting or massive loss spikes.
-   **Target Normalization:** The batch targets (`batch['values']`) are normalized using the *current* statistics before being passed to the loss function. This ensures the loss is always computed in the standardized zero-mean, unit-variance space.

### 3. Gradient Flow
-   **Verification:** Gradient flow is **preserved** and **correct**.
    -   The loss function sees normalized targets and normalized predictions.
    -   The normalization statistics ($\mu, \sigma$) act as constants during the gradient step.
    -   Gradients are NOT backpropagated through the normalization statistics (correct, as they are estimated via EMA, not learned via SGD).

### 4. Stability
-   **Batch Handling:** The trainer updates statistics only on valid samples (masked), preventing padding from corrupting the running mean/std.
-   **Frequency:** Updates occur every 10 iterations. This reduces CPU-GPU sync overhead while keeping scales reasonably fresh.

## Conclusion
The PopArt implementation is robust, correct, and safe. No gradient flow issues were found.

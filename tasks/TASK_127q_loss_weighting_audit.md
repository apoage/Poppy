# TASK_127q: Loss Function Weighting Audit

## Findings

### 1. Weights (`trainer.py`)
-   **Core MuZero:**
    -   Value: 1.0
    -   Policy: 1.0
    -   Reward: 1.0
-   **Auxiliary:**
    -   Prediction: Annealed (starts 0.01, ends 0.2)
    -   Physics Reg: 0.01
    -   ATC: 0.1
    -   Cycle Balance: 0.1
    -   BC: Annealed (0.1 -> 0.01)

### 2. Balance Assessment
-   **Value/Policy Balance:** Generally good. PopArt normalization ensures value targets have unit variance, keeping the value loss magnitude consistent with the policy cross-entropy loss.
-   **Reward Loss:** Computed on scaled rewards (x100). This ensures the reward signal is strong enough to be learned, preventing it from vanishing alongside the value signal.
-   **Auxiliary Losses:** Kept small (order of magnitude lower) to act as regularizers rather than primary objectives.

### 3. Gradient Magnitude
-   **Risk:** Without `per_head_clipping` enabled, a massive value prediction error (e.g., detecting a crash) could theoretically dominate the gradient update.
-   **Mitigation:** The global clip (10.0) is a final safety net.

## Recommendations
-   **Monitor Value Loss:** If `value_loss` consistently exceeds `policy_loss` by >10x, consider reducing `value_loss_weight` to 0.25 (standard MuZero practice).
-   **Enable Per-Head Clipping:** Strongly recommend enabling `per_head_clipping` by default to mathematically guarantee gradient balance.

## Conclusion
The loss weighting is well-designed, leveraging normalization (PopArt) and annealing schedules to maintain stability.

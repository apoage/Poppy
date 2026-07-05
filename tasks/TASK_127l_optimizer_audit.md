# TASK_127l: Learning Rate & Optimizer Audit

## Findings

### 1. Optimizer Implementation (`trainer.py`)
-   **Optimizer:** `optax.adamw` is used correctly.
-   **Scheduling:** Supports Linear, Cosine, and Constant schedules with warmup. Implementation logic relies on `optax` schedules, which is standard.
-   **Clipping:**
    -   **Global:** `optax.clip_by_global_norm(10.0)` is always applied. This is a "safety net" for extreme gradients.
    -   **Per-Head (TASK_025):** Optional fine-grained clipping for `value`, `policy`, `dynamics`, `encoder` heads (default threshold 1.0). This prevents task dominance.

### 2. Configuration (`config.py`)
-   **Default LR:** `1e-4`. Safe and standard for this architecture.
-   **Weight Decay:** `1e-4`. Standard.
-   **Warmup:** 10 epochs. Appropriate.

### 3. Risks & Stability
-   **Stability:** High. The combination of AdamW, warmup, and two layers of clipping (optional fine-grained + mandatory global) makes optimization very stable.
-   **Speed:** `1e-4` might be conservative for the `small` model, but safe.

## Recommendations
-   **Global Clipping:** Consider lowering the global clip from `10.0` to `1.0` or `5.0` if training instability (NaNs) is observed without per-head clipping enabled.
-   **Per-Head Clipping:** This feature is excellent but disabled by default. Enabling it with `clip_grad_value=1.0` is recommended for complex curriculum phases.

## Conclusion
The optimizer setup is correct and follows JAX/Flax best practices.

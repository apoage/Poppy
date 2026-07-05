# Critical Code Audit: JAX MuZero Trading Agent

**Role:** Hostile Lead Architect / Security Auditor
**Objective:** Prove that the system is "cheating" (data leakage), mathematically unsound, or architecturally fragile.

## 1. Data Leakage Investigation (The "Time Travel" Check)
*Goal: Ensure the agent cannot see into the future.*

*   **Environment (`mcts_trader/env/vectorbt_env.py`):**
    *   Trace the `step()` method. Are `self.current_step` and price indices updated *strictly after* the action is executed and recorded?
    *   Verify `_update_encoder`. Does it access `self.prices[step_idx]`? Is `step_idx` guaranteed to be $\le t$?
    *   Check `_get_observation`. Does the `FastMultiAssetStateEncoder` have access to future rows in the price array?
*   **Normalization:**
    *   Are normalization statistics (mean/std) calculated using the *entire* dataset (future leakage) or a rolling window (correct)?
*   **Buffer (`replay_buffer.py`):**
    *   Verify the `add_trajectory` logic again. Is there any off-by-one error where `observation[t+1]` is paired with `action[t]`?

## 2. Mathematical Integrity (The "Gradient" Check)
*Goal: Ensure the RL algorithm matches the theory.*

*   **Monte Carlo Returns (`train_jax_fast.py` & `trainer.py`):**
    *   Verify the discount factor $\gamma = 0.997$. Is it applied consistently in the MC return calculation loop?
    *   Check `_compute_target_value`. Does it correctly sum rewards from $t$ to $T$?
    *   **Value Support:** We recently moved to centralized configs (`-200` to `+200`). Scan `muzero_model.py` and `trainer.py` to ensure *no* hardcoded ranges (`-50`, `50`) remain hidden in loss functions or projections.
*   **Loss Function (`trainer.py`):**
    *   Review `_hybrid_value_loss`. Is the weighting between MSE and Categorical loss ($0.8$) reasonable? Does the gradient flow correctly through both?
    *   Check `policy_loss`. Is it using `optax.softmax_cross_entropy` correctly with the MCTS policy targets?
*   **MCTS Config (`train_jax_fast.py`):**
    *   Are `dirichlet_alpha` and `exploration_fraction` reasonable for a 5-action space?

## 3. Dojo & Curriculum Debt (The "Missing Features" Check)
*Goal: Identify what is broken or missing in the training progression.*

*   **Synthetic Market (`mcts_trader/data/synthetic_market.py`):**
    *   The user mentioned "extra synthetic levels". What levels exist (0-3)? Are they actually distinct?
    *   Do the generators produce realistic features (autocorrelation, volatility clustering) or just predictable sine waves?
*   **Curriculum Logic (`dojo_dynamic_curriculum.py`):**
    *   How does the system decide to promote the agent? Is the `avg_reward > 5.0%` threshold hardcoded or dynamic?
    *   Does it correctly read `last_run_metrics.json`? What happens if that file is missing or corrupted?

## 4. Implementation Sanity
*   **JAX Compilation:** Are there any Python control flows (loops, ifs) inside JIT-compiled functions that might be unrolled inefficiently?
*   **Dtype Mismatches:** Are we mixing `float32` and `float16` dangerously? (Check `replay_buffer` vs `network` inputs).

## 5. Curriculum & Transfer Mechanics (The "Cold Start" Check)
*Goal: Ensure the "Transfer Learning" doesn't shatter pre-trained weights.*

*   **Buffer Flushing:**
    *   Verify that `replay_buffer` is indeed cleared or re-initialized when switching levels.
    *   **Warmup Safety:** Check `train_jax_fast.py`. Is the `if epoch >= 5:` condition (or similar) robust enough?
    *   *Calculation:* 5 epochs * 400 steps = 2000 steps. Batch size = 256. Is 2000 steps of diversity enough to prevent the gradients from destroying the imported weights?
*   **Optimizer State:**
    *   When loading a checkpoint for a *new* regime, do we keep the old optimizer state (momentum, variance)?
    *   *Risk:* Old momentum might push weights in the "wrong" direction for the new regime. Ideally, optimizer state might need a reset or decay on regime shifts.
*   **Metric Continuity:**
    *   Does the dashboard/logger handle the reset? (We saw `epoch_offset` logic in `train_jax_fast.py`, verifying it works).

## 6. Research Alignment (The "Source Material" Check)
*Goal: Verify fidelity to the core architectures.*

*   **Reference Path:** `/home/lukas/Trader/docs/research/`
*   **Papers:**
    *   `TinyRecursiveModels-main` & `arXiv-2510.04871v1`: Verify TRM implementation in `trm_flax.py`. Are the residual connections and "inner/outer" loops implemented as per the paper?
    *   `mctx-main` & `arXiv-1911.08265v2` (MuZero): Verify the search policy vs value targets. Is the `gumbel_scale` usage consistent with the `mctx` implementation guide?
    *   `NEURAL_MARKET_PHYSICS_DESIGN.md`: Does the current `FastMultiAssetStateEncoder` capture the "Physics" features promised in the design doc, or is it just raw prices?

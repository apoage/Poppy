# Recent Improvements

## December 16, 2025 - Architecture Design Session (Claude Code)

### 1. Paradigm Shift: Trading Agent → Portfolio Manager
**Realization:** The system is not a trading agent, it's a **portfolio manager** for 64 assets.

**New Focus:**
- Portfolio rebalancing tactics (rotation, hedging, risk parity)
- Cross-pair trading (BTC→ETH direct, not BTC→USD→ETH)
- Multi-action per step (rapid rebalancing)
- Adversarial robustness

### 2. Extended Synthetic Curriculum (Levels 0-6)
**Designed 6-level curriculum** to teach progressively complex skills:

**Levels 0-3** (Basic Mechanics):
- Level 0: Sine waves (validate architecture)
- Level 1: Lagged pairs (test correlation encoder)
- Level 2: Dip crashes (test patience)
- Level 3: Noisy trends (test fee awareness)

**Levels 4-6** (Portfolio Tactics - NEW):
- **Level 4**: Portfolio rotation (teach capital rebalancing)
- **Level 5**: Adversarial market (robustness training)
- **Level 6**: Chaos mode (black swan survival)

**Philosophy**: "If a small TRM cannot solve a noiseless sine wave, it will never solve Bitcoin."

### 3. Multi-Task Learning: Prediction Heads
**Gap Identified:** Model learns VALUE but not PRICE TRAJECTORIES.

**Solution:** Add prediction heads to latent space:
- Price predictor (next 10 steps)
- Volatility predictor (risk level)
- Regime classifier (bull/bear/sideways)

**Benefits:**
- Richer latent representations (multi-task learning)
- Interpretable (can validate market understanding)
- Useful for risk management

### 4. MCTS Parallelization Critical Issue
**Bottleneck Identified:**
- CPU: 1 core at 90%, 15 cores idle
- GPU: 60% util (starved by CPU)

**Root Cause:** MCTS tree traversal is sequential, doesn't use all cores.

**Solution Designed:** Multi-process worker pool (4 workers × 4 cores)
- Expected: 3-4× speedup
- **Priority: CRITICAL** - biggest performance win available

### 5. Model Scaling Analysis
**Current:** 7M params, 32 MCTS sims
**Required for AlphaGo-level breakthroughs:** 25M params, 256-512 sims

**Plan:** Configurable model sizes (7M/15M/25M)
- Small: Fast iteration
- Medium: Curriculum training
- Large: Adversarial training

**Hardware:** RTX 3090 (24GB) can handle 25M + batch 256 + 512 sims with 16GB headroom.

### 6. Task Files Created
**5 detailed implementation tasks** for Gemini:
1. **TASK_01**: MCTS Parallelization (CRITICAL, 3× speedup)
2. **TASK_02**: Synthetic Generators (Levels 0-3)
3. **TASK_03**: Model Scaling (7M → 25M)
4. **TASK_04**: Portfolio Curriculum (Levels 4-6)
5. **TASK_05**: Prediction Heads (multi-task learning)

### 7. Design Documents Created
- `SYNTHETIC_CURRICULUM_DESIGN.md`: Complete Level 0-3 specification
- `ADVANCED_ORDERS_DESIGN.md`: Limit/stop orders (6 actions per asset)
- `PORTFOLIO_MANAGER_TRAINING.md`: Master design for portfolio tactics
- `PREDICTION_LATENT_SPACE.md`: Multi-task learning architecture

### Key Insights
- **Validation First**: Level 0 (sine) is gatekeeper - if fails, STOP
- **No Timelines**: Focus on correctness, iterate fast
- **Multi-GPU Ready**: 4×5090s available for adversarial training
- **Goal**: Discover inhuman tactics (like AlphaGo move 37)

---

## December 14, 2025 - Gemini Improvements

## 1. Rainbow Table Training Continuity
**Problem:** The training schedule (`tools/run_scheduled_training.py`) was launching independent processes for each period, causing the model to reset (Catastrophic Forgetting) and crash due to shape mismatches when loading checkpoints (61 assets vs 59 assets).

**Solution:**
*   **Checkpoint Transfer:** Updated `run_scheduled_training.py` to track the latest checkpoint and pass it to the next run via `--load-checkpoint`.
*   **Shape Standardization (Padding):** 
    *   Updated `tools/run_scheduled_training.py` to pad asset lists to `TARGET_ASSETS=64` using duplicates.
    *   Patched `FastTradingEnv` in `train_jax_fast.py` to handle duplicate symbols correctly (forcing `n_assets=64` and using full padded list for Encoder/Portfolio).
    *   Patched `mcts_trader/env/trading_env.py` similarly (for future use).

## 2. Adaptive Curriculum & "Maniac Mode"
**Problem:** The model was converging to "Turtle Mode" (Reward 0.00, Holding Cash) because the 26bps fee + 5bps slippage barrier was too high for random exploration to overcome.

**Solution:**
*   **Physics-First Curriculum:**
    *   Modified `tools/run_scheduled_training.py` to implement an adaptive difficulty scaler.
    *   **Level 0:** Fee 0, Slippage 0 (Frictionless Sandbox).
    *   **Progression:** If `avg_reward > 5.0%`:
        *   Phase 1: Ramp Slippage (0 -> 5 bps).
        *   Phase 2: Ramp Fees (0 -> 26 bps).
*   **Maniac MCTS Settings:**
    *   Increased `num_simulations` from 16 to 32 (Doubled thought depth).
    *   Increased `dirichlet_alpha` (Noise) to 0.5 (High exploration).
    *   Exposed these as arguments in `train_jax_fast.py` and `mctx_search.py`.

## 3. High-Performance Infrastructure
**Problem:** Single-core CPU bottleneck (SQLite queries) was starving the GPU (67% load).

**Solution:**
*   **RAM Caching:**
    *   Modified `mcts_trader/data/history.py` to implement `_preload_to_ram()`.
    *   Loads the entire episode's OHLCV data into a numpy matrix (`float32`).
    *   Replaced SQL queries in `get_current_data` with O(1) array lookups.
    *   Reduced CPU load, allowing GPU to focus on the increased MCTS simulation count.

## Modified Files
*   `tools/run_scheduled_training.py`: Scheduler logic, padding, curriculum.
*   `train_jax_fast.py`: Argument plumbing, `FastTradingEnv` duplicate handling, metric reporting.
*   `mcts_trader/data/history.py`: Restored missing methods, added RAM caching.
*   `mcts_trader/jax_agent/mctx_search.py`: Added explicit noise arguments.
*   `mcts_trader/env/trading_env.py`: Fixed `TradingEnv` for duplicates.

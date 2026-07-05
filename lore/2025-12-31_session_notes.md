# Session Notes - 2025-12-31

## Summary
Built hyperparameter tournament system for systematic config search, plus visualization dashboards for advanced training analytics.

---

## 1. DS56 Level 5-6 Dashboard (Advanced Dynamics)

**Location:** `tools/DS56/`

Created visualization dashboard for Level 5-6 metrics:
- Vector fields and phase space analysis
- Energy landscapes
- Rolling Hurst exponent (mean reversion detection)
- Lyapunov exponents (chaos detection)
- Takens embedding
- Multifractal spectrum

**Files:**
- `server.py` - Flask backend on port 5056
- `index.html` - Dashboard HTML
- `styles.css` - Dark theme CSS
- `dashboard.js` - Chart.js visualizations
- `favicon.svg` - Attractor icon

**Fixes Applied:**
- Fixed tensor snapshot saving to extract actual latent states through model encoder
- Fixed Rolling Hurst chart infinite scaling (Chart.js `maintainAspectRatio` issue)
- Fixed epoch filename matching for zero-padded filenames

---

## 2. DS78 "Shroom Engine" Level 7-8 Dashboard (Volumetric/Fractal)

**Location:** `tools/DS78/`

Created psychedelic visualization dashboard for Level 7-8:
- NeRF-style volumetric density fields (3D latent trajectories)
- Gaussian splatting point clouds
- Correlation dimension (Grassberger-Procaccia algorithm)
- Box-counting fractal dimension
- Recurrence plots
- Reference attractors (Clifford 2D, Lorenz 3D)

**Files:**
- `server.py` - Flask backend on port 5078
- `index.html` - Dashboard with tabs
- `styles.css` - Psychedelic dark theme (magenta/cyan/orange)
- `dashboard.js` - Plotly.js 3D visualizations
- `favicon.svg` - Fractal spiral icon

---

## 3. Hyperparameter Tournament System

**Location:** `hyperparam_tournament.py`

Built successive halving search system for finding optimal hyperparameters.

### Tournament Structure
```
Round 0: L0 (sine)     - 48 configs × 50 epochs  → 24 survive
Round 1: L1 (lagged)   - 24 configs × 75 epochs  → 12 survive
Round 2: L3 (trends)   - 12 configs × 100 epochs → 6 survive
Round 3: L4 (rotation) - 6 configs × 100 epochs  → 3 survive
Round 4: Real data     - 3 configs × 150 epochs  → Winners

SCALING TESTS (top 3 winners):
  - medium model (96-dim, ~15M params), batch 256/512
  - large model (160-dim, ~36M params), batch 256/512
```

### 48 Config Search Space (2×2×2×3×2)
| Variable | Values | Description |
|----------|--------|-------------|
| sims | 16, 32 | MCTS simulations |
| lr | 1e-4, 3e-4 | Learning rate |
| h_cycles | 2, 4 | TRM reasoning depth |
| pred_weight | 0, 0.3, 1.0 | Physics prediction |
| gumbel_scale | 1.5, 3.0 | MCTS exploration |

All configs use `--adaptive-entropy` (not static entropy_coef).

### Naming Convention
`s{sims}_lr{10|30}_h{cycles}_{np|p3|p10}_g{15|30}`

Examples:
- `s16_lr10_h2_np_g15` - 16 sims, lr=1e-4, h=2, no pred, gumbel=1.5
- `s32_lr30_h4_p10_g30` - 32 sims, lr=3e-4, h=4, pred=1.0, gumbel=3.0

### Scoring Criteria
- Primary: reward MA50 (recent_reward)
- Secondary: activity penalty (penalizes >80% HOLD)
- Tertiary: training speed bonus

### CLI Usage
```bash
# Full tournament with scaling tests
python hyperparam_tournament.py

# Skip scaling tests
python hyperparam_tournament.py --skip-scaling

# Faster testing (half epochs)
python hyperparam_tournament.py --epochs-multiplier 0.5

# Resume from round 2
python hyperparam_tournament.py --start-round 2
```

### Output Location
`tournament_results/tournament_<timestamp>/`
- `configs.json` - All config definitions
- `round_N_results.csv` - Per-round results
- `scaling_results.csv` - Scaling test results
- `summary.json` - Final summary

---

## 4. Training Script Enhancements

### New CLI Arguments Added to `train_jax_fast.py`
```
--lr         Learning rate (default: 1e-4)
--discount   Discount factor gamma (default: 0.99)
```

### Wiring Changes
- `args.lr` → `MuZeroTrainer(learning_rate=...)`
- `args.discount` → `run_self_play(..., discount=...)` and `run_self_play_batched(...)`

Previously learning_rate was hardcoded to 1e-4 and discount to 0.997.

---

## 5. Bug Fixes

### Tensor Snapshot Pipeline
- `training_hooks.py`: Fixed `_save_tensor_snapshot()` to run observations through model encoder to get actual latent states (was saving raw observations)
- Increased samples from 10 to 200 for better dynamics analysis

### DS56 Server
- Fixed Hurst exponent NaN/Inf protection with `max(v, 1e-10)` and `np.isfinite()` filtering
- Fixed epoch filename matching to handle zero-padded filenames (`epoch_000010.npz`)

### DS56 Dashboard
- Fixed Rolling Hurst chart growing to infinite height
- Solution: `maintainAspectRatio: true`, explicit `aspectRatio`, container wrapper div, CSS `max-height`

### Tournament
- Fixed `/` in run names creating non-existent subdirectories
- Changed to `_` separator: `tourn_<timestamp>_<config>_L<level>`

---

## 6. Files Modified/Created

### Created
- `tools/DS56/` - Level 5-6 dashboard (5 files)
- `tools/DS78/` - Level 7-8 Shroom Engine (5 files)
- `hyperparam_tournament.py` - Tournament system

### Modified
- `train_jax_fast.py` - Added --lr, --discount args
- `tools/DS24/training_hooks.py` - Fixed tensor snapshots
- `tools/DS24/metrics_logger.py` - Removed redundant epoch filter

---

## 7. Current Status

Tournament running with first configs completing:
```
s16_lr10_h2_np_g15: avg_reward=-3.39, recent=-5.15, loss=5.61, entropy=3.21
s16_lr10_h2_np_g30: avg_reward=-4.28, recent=-7.93, loss=5.85, entropy=3.61
```

Checkpoints saving to `jax_ckpt_fast/tourn_<timestamp>_<config>_L0/`

---

## Next Steps
1. Let tournament complete Round 0 (48 configs)
2. Review survivors and their characteristics
3. Continue through L1, L3, L4, real data
4. Run scaling tests on winners
5. Select final config for full training

---

## 8. Profiling & Bottleneck Analysis

### Investigation
Ran detailed profiling to find single-threaded bottleneck in self-play.

**Results (Large model, 36M params, 32 sims):**
```
Component                    Time        % of Total
─────────────────────────────────────────────────────
mcts_search                  6.7s        90.1%      ← THE BOTTLENECK
env.step                     0.5s         6.5%
encoder.encode               0.2s         2.4%
obs_prep_numpy_to_jax        0.07s        1.0%
```

### Root Cause
- TRM prediction network runs 24 recursive cycles (H=4 × L=6)
- Each cycle: ~1.1ms
- With 64 sims: 64 × 1.05ms = 67.8ms per step
- 1440 steps × 67.8ms = 97.6s per episode (just MCTS)

**Key insight**: Bottleneck is GPU-bound recursive reasoning, not CPU or memory transfer.

---

## 9. Batched Self-Play Implementation

### Solution
Run N environments in parallel, batch MCTS neural network calls.

```python
# New function: run_self_play_batched()
# Stacks observations from N envs → single GPU call → distribute actions
```

### Benchmark Results
```
Configuration        Trans/sec    ms/trans    Speedup
────────────────────────────────────────────────────
Serial (1 env)         19.3       51.9ms      1.00x
Batched (2 envs)       16.9       59.0ms      0.88x (overhead)
Batched (4 envs)       40.2       24.9ms      2.09x  ← SWEET SPOT
Batched (8 envs)       35.1       28.5ms      1.82x (memory bound)
```

### New CLI Flag
```bash
--parallel-envs N   # Default: 1, Recommended: 4
```

---

## 10. BFloat16 Fixes

### Issues Fixed
1. **Scatter dtype warning**: `nn.Embed` was causing FutureWarning
   - Fix: Keep embedding in float32, cast after lookup in `dynamics()`

2. **XLA autotuning verbosity**: Noisy warnings during cudnn algorithm search
   - Fix: `TF_CPP_MIN_LOG_LEVEL=2`

3. **JAX FutureWarning suppression**:
   - Fix: `warnings.filterwarnings('ignore', category=FutureWarning, module='jax')`

### Code Changes
- `muzero_model.py`: Removed `dtype` from `nn.Embed`, added `.astype(self.dtype)` after lookup
- `train_jax_fast.py`: Added warning suppression at top of file

---

## 11. MCTS Simulation Warmup Insight

### Observation
With high simulations (64), policy quickly converges to HOLD (~80%) before value function learns.
Entropy stays reasonable (~4.0), but action distribution is HOLD-dominant.

### Insight
Fewer sims = more stochastic policy = more exploration.
Let value function learn FIRST, then add deep search to exploit learned values.

### Proposed Curriculum
```
Epoch 1-N/4:    8 sims   (explore)
Epoch N/4-N/2:  16 sims  (learn values)
Epoch N/2-3N/4: 32 sims  (refine policy)
Epoch 3N/4+:    64 sims  (full depth)
```

Added to CLAUDE.md under "Curriculum System" section.

---

## 12. Pretraining Curriculum (13 Stages)

### Design Philosophy
Progressive training across multiple dimensions:
- MCTS simulations: 8 → 16 → 32 → 64
- Episode length: 100 → 1440 steps (short = fast failure, long = stability)
- Fees: 0 → 26 bps
- Slippage: 0 → 5 bps
- Market complexity: Level 0 → 3

### Curriculum Structure
```
Phase 1: Foundation (Level 0 - Sine Waves)
  1.1: 100ep, 100 steps, 8 sims, 0 fee    → +3.0%
  1.2: 150ep, 150 steps, 16 sims, 0 fee   → +2.5%
  1.3: 150ep, 250 steps, 32 sims, 0 fee   → +2.0%
  1.4: 100ep, 400 steps, 32 sims, 5 fee   → +1.5%

Phase 2: Correlation (Level 1 - Lagged Pairs)
  2.1: 150ep, 300 steps, 16 sims, 0 fee   → +2.0%
  2.2: 200ep, 400 steps, 32 sims, 5 fee   → +1.5%
  2.3: 150ep, 500 steps, 32 sims, 10 fee  → +1.0%

Phase 3: Timing (Level 2 - Dip Crashes)
  3.1: 200ep, 250 steps, 16 sims, 5 fee   → +1.5%
  3.2: 250ep, 500 steps, 32 sims, 10 fee  → +1.0%
  3.3: 200ep, 800 steps, 64 sims, 16 fee  → +0.5%

Phase 4: Trends (Level 3 - Noisy Trends)
  4.1: 250ep, 400 steps, 32 sims, 10 fee  → +1.0%
  4.2: 300ep, 800 steps, 64 sims, 16 fee  → +0.5%
  4.3: 250ep, 1440 steps, 64 sims, 26 fee → +0.2%
```

### Files Created
- `config/pretrain_curriculum.json` - 13-stage curriculum config
- Updated `dojo_manager.py` - Added sims, slippage, steps, parallel_envs, bf16 support

---

## 13. Files Modified/Created (Session 2)

### Created
- `profile_bottleneck.py` - Basic profiling script
- `profile_mcts_detailed.py` - Detailed MCTS component timing
- `test_batched_selfplay.py` - Benchmark for batched self-play
- `config/pretrain_curriculum.json` - 13-stage pretraining curriculum

### Modified
- `train_jax_fast.py` - Added batched self-play, --parallel-envs flag, warning suppression
- `mcts_trader/jax_agent/muzero_model.py` - Fixed embedding dtype handling
- `dojo_manager.py` - Added sims, slippage_bps, steps, parallel_envs, bf16 support
- `CLAUDE.md` - Added MCTS warmup curriculum notes

---

## Performance Summary

| Optimization | Speedup | Status |
|--------------|---------|--------|
| BFloat16 mixed precision | ~1.0x (no gain) | Implemented |
| Batched self-play (4 envs) | ~2.0x | Implemented |
| MCTS sim warmup | TBD | Documented |

**Net result**: ~2x throughput with `--parallel-envs 4`

---

## 14. Bug Audit Fixes (BUG_015-019)

### BUG_015: Dead Code Cleanup
Used `pyflakes` to identify unused imports:
- `train_jax_fast.py`: Removed unused `import sys`
- `train_jax_fast.py`: Removed duplicate `create_mctx_search` import
- `train_jax_fast.py`: Removed unused `create_dynamic_tree_search_fn` import
- `trainer.py`: Removed unused `get_value_support` import and `support` variable

### BUG_016: Hardcoded State Dims
**FALSE POSITIVE** - Verified that `state_dim` is correctly passed from CLI `--model-size` through to all components.

### BUG_017: Buffer Overflow Protection
Already fixed in earlier session (trajectory_generation type).

### BUG_018: RNG Split Audit
**FALSE POSITIVE** - Audited all RNG usage. All keys properly split before use.

### BUG_019: Checkpoint Fields
Already fixed in earlier session (trainer metadata save/load).

---

## 15. Tournament Subprocess Logging Fix

### Problem
When running `train_jax_fast.py` as subprocess from `hyperparam_tournament.py`, epoch logging was missing. Only checkpoint saves appeared in output.

### Root Cause
Python uses **block buffering** when stdout is piped (`subprocess.PIPE`), not line buffering. Epoch `print()` statements don't appear until buffer fills (~4KB).

### Fix
Added `-u` flag to Python command for unbuffered stdout:
```python
cmd = [
    "python", "-u", "train_jax_fast.py",  # -u = unbuffered
    f"--epochs={epochs}",
    ...
]
```

---

## 16. Replay Buffer Int Type Optimization

### Issue
`trajectory_generation` was set to `jnp.int64`, triggering JAX warning:
```
UserWarning: Explicitly requested dtype int64 is not available, and will be truncated to int32
```

### Analysis
- GPU integer operations are throttled for 64-bit
- `trajectory_generation` is used sparsely (once per `sample_batch()` call)
- int32 handles ~2 billion generations - sufficient for any practical training

### Fix
```python
# Track when trajectory was added (for wrap-around validation)
# int32 handles ~2B generations - sufficient for any practical training
self.trajectory_generation = jnp.zeros((capacity,), dtype=jnp.int32)
```

---

## 17. Commits Made

### Commit 1: Bug Fixes
```
[CLAUDE] Fix deferred bugs from code audit

BUG_015 - Dead code cleanup:
- train_jax_fast.py: Remove unused sys import and duplicate create_mctx_search
- train_jax_fast.py: Remove unused create_dynamic_tree_search_fn import
- trainer.py: Remove unused get_value_support import and support variable

BUG_016 - Hardcoded state dims: FALSE POSITIVE (correctly passed from CLI)
BUG_018 - RNG split audit: FALSE POSITIVE (all keys properly split)
```

### Commit 2: Tournament & Int32
```
Fix tournament logging and replay buffer int type

- hyperparam_tournament.py: Add -u flag for unbuffered stdout so epoch
  logging appears in real-time when running as subprocess
- replay_buffer.py: Use int32 for trajectory_generation (sufficient for
  ~2B generations, avoids JAX x64 warning, better GPU performance)
```

---

## 18. Hyperparameter Tournament Results (Round 4)

### Top 3 Survivors
```
1. s16_lr10_h2_np_g15: score=-2.35, reward=-2.44, hold=80%
2. s16_lr10_h4_p10_g15: score=-2.73, reward=-2.82, hold=78%
3. s32_lr10_h2_p3_g15: score=-3.37, reward=-3.46, hold=75%
```

### Key Observations
- Lower sims (16) performed better than higher sims (32)
- Lower learning rate (1e-4) beat higher (3e-4)
- Prediction loss mixed results: `np` (none) vs `p10` (1.0) vs `p3` (0.3)
- All configs have high HOLD% (75-80%) - entropy/exploration still an issue

---

## 19. Tuning Tournament Plan

### Phase 1: Implement Stability Features

Before running tuning tournament, implement:

1. **MCTS Sims Warmup** (TASK_020)
   - 8→16→32 sims over epochs
   - Prevents early MCTS lock-in before value function learns

2. **PopArt Normalization** (TASK_021)
   - Adaptive value target normalization
   - Critical for curriculum stability

3. **Physics Prediction Leverage**
   - Add `--physics-weight` to control physics vs neural balance
   - Add `--physics-only` mode (no neural corrector)
   - Better comparison: physics-only vs hybrid vs no-prediction

4. **Huber Loss** (Already implemented)
   - `--value-loss-type huber` available

### Phase 2: Tuning Tournament Structure

Use **3 winning configs** as base, compare stability features:

```
Round A: Prediction Model Comparison (9 runs)
  Each of 3 configs × 3 prediction modes:
  - no_pred:      --prediction-loss-weight 0
  - physics_only: --physics-only --prediction-loss-weight 0.3
  - hybrid:       --use-prediction-loss --prediction-loss-weight 0.3

Round B: Value Stabilization (9 runs)
  Each of 3 configs × 3 value modes:
  - baseline:     (no changes)
  - huber:        --value-loss-type huber --value-huber-delta 0.5
  - popart:       --use-popart

Round C: MCTS Warmup (6 runs)
  Top 2 from Round A/B × 3 warmup modes:
  - no_warmup:    --sims 32
  - warmup_fast:  --sims-warmup 50 --sims-min 8
  - warmup_slow:  --sims-warmup 100 --sims-min 8

Round D: Final Comparison (3 runs)
  Best configs from each round on real data
```

### Phase 3: Metrics to Compare

| Metric | What It Shows |
|--------|---------------|
| Value MSE | Value calibration |
| Value Std | Value confidence/stability |
| HOLD% | Exploration vs exploitation |
| Reward MA50 | Recent performance |
| Entropy | Policy diversity |
| Physics Reg Loss | Physics model usage |

### Expected Outcomes

1. **Physics-only** should show cleaner learning signal
2. **Huber + PopArt** should reduce value variance
3. **Sims warmup** should reduce early HOLD collapse
4. Best combo likely: `physics_only + huber + warmup_fast`

---

## 20. Database Cache Speedup

### Problem
Real data loading takes 7+ minutes per config (102M price points from SQLite).

### Solution
Added `--cache-path` option:
```bash
# First run: creates cache (7 min)
python train_jax_fast.py --db-path=trading_data.db --cache-path=.db_cache.npz

# Subsequent runs: loads from cache (~5 sec)
```

### Files Modified
- `mcts_trader/env/vectorbt_env.py`: `_try_load_cache()`, `_save_cache()`
- `train_jax_fast.py`: `--cache-path` CLI arg
- `hyperparam_tournament.py`: Uses `--cache-path=.db_cache.npz` for L-1

---

## 21. Task Cleanup

Moved completed tasks to `tasks/_old/`:
- Prediction model tasks (008-012)
- Deferred tasks (02, 04)
- Implementation plan

Active tasks remain:
- TASK_020: MCTS Sims Warmup
- TASK_021: PopArt Normalization
- TASK_022: TD(λ) Returns
- TASK_023: Prioritized Replay (low priority)
- TASK_024: LR Schedule
- TASK_025: Per-Head Gradient Clipping

---

## 22. Physics-Only Mode

### Implementation

Added `--physics-only` flag to enable pure physics prediction (no neural corrector).

**Usage:**
```bash
# No prediction (baseline)
python train_jax_fast.py --prediction-loss-weight 0

# Physics-only prediction
python train_jax_fast.py --use-prediction-loss --physics-only --prediction-loss-weight 0.3

# Hybrid prediction (physics + neural corrector)
python train_jax_fast.py --use-prediction-loss --prediction-loss-weight 0.3
```

### Files Modified

| File | Change |
|------|--------|
| `mcts_trader/jax_agent/prediction/market_ode.py` | Added `physics_only` flag to `NeuralMarketODE` |
| `mcts_trader/jax_agent/muzero_model.py` | Added `physics_only` param to `MuZeroNetworks` |
| `train_jax_fast.py` | Added `--physics-only` CLI arg |

### How It Works

When `physics_only=True`:
- `NeuralMarketODE.compute_tendency()` sets `G_neural = 0`
- Only physics core contributes: `G_total = G_phys + G_impact`
- Physics core still has learned parameters (α, β, σ, Γ)
- Easier to interpret: model learns market dynamics, not opaque neural features

### Tournament Comparison

```
Mode        | Args                                           | Use Case
------------|------------------------------------------------|---------------------------------
no_pred     | --prediction-loss-weight 0                     | Baseline (no auxiliary loss)
physics     | --use-prediction-loss --physics-only           | Interpretable physics model
hybrid      | --use-prediction-loss                          | Full neural+physics (max capacity)
```

---

## 23. MCTS Sims Warmup (TASK_020) - COMPLETE

### Implementation

Gradually increase MCTS simulations over training to prevent early policy lock-in.

**New files:**
- `mcts_trader/jax_agent/schedules.py` - `SimsSchedule` class with linear/step/exponential schedules

**CLI arguments:**
```bash
--sims-warmup 100       # Epochs to warm up (0=disabled)
--sims-min 8            # Starting simulations
--sims-schedule linear  # linear, step, or exponential
```

**Usage:**
```bash
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 200 \
    --sims 64 \
    --sims-warmup 100 \
    --sims-min 8 \
    --name test_sims_warmup
```

**How it works:**
- Epoch 0-25: 8 sims (high exploration)
- Epoch 25-50: ~22 sims
- Epoch 50-75: ~36 sims
- Epoch 75-100: ~50 sims
- Epoch 100+: 64 sims (full depth)

---

## 24. PopArt Normalization (TASK_021) - COMPLETE

### Implementation

Adaptive value target normalization for curriculum stability.

**New files:**
- `mcts_trader/jax_agent/popart.py` - PopArt implementation with:
  - `PopArtState` - Running mean/var statistics
  - `PopArtNormalizer` - Convenience class
  - `normalize_targets()` - Normalize to zero mean, unit var
  - `apply_weight_correction_flax()` - Adjust value head weights

**CLI arguments:**
```bash
--use-popart            # Enable PopArt
--popart-beta 0.0001    # EMA decay rate
```

**Usage:**
```bash
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 100 \
    --use-popart \
    --name test_popart_L0

# Resume to L1 (where reward scale changes)
python train_jax_fast.py \
    --synthetic-level 1 \
    --load-checkpoint jax_ckpt_fast/test_popart_L0 \
    --use-popart \
    --clear-buffer \
    --name test_popart_L1
```

**How it works:**
1. Before each batch, update running mean/var of value targets
2. Compute weight correction: `W_new = W_old * (σ_old / σ_new)`
3. Apply correction to value head to preserve outputs
4. Normalize targets for loss computation
5. Result: Value predictions stay calibrated across reward scales

---

## Summary of New Features

| Feature | Flag | Purpose |
|---------|------|---------|
| Physics-only prediction | `--physics-only` | Disable neural corrector |
| Sims warmup | `--sims-warmup N` | Gradual MCTS depth increase |
| PopArt | `--use-popart` | Adaptive value normalization |

**Recommended combo for tuning tournament:**
```bash
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 200 \
    --sims 64 \
    --sims-warmup 100 \
    --sims-min 8 \
    --use-popart \
    --use-prediction-loss \
    --physics-only \
    --name tuning_test
```

# TASK_067: DS24 Full Integration

**Priority**: HIGH
**Status**: COMPLETE (Phase 1 + Phase 2)
**Created**: 2026-01-22
**Updated**: 2026-01-22

## Context

During TASK_065/066 refactoring, the DS24 dashboard integration was lost. The dashboard expects ~85+ metrics but only basic training metrics were being logged.

## Phase 1: Immediate Fixes (COMPLETE)

### train_jax_fast.py Changes
- Integrated `DS24MetricsLogger` from `tools/DS24/metrics_logger.py`
- Added all DS24 columns to CSV when `--ds24-level > 0`
- Now logs:
  - Training config flags (GAE, per-head clipping, epsilon-greedy)
  - Gradient norms (value, policy, dynamics)
  - ReDo stats (dormant neurons, recycled count) if `--redo` enabled
  - ATC stats (loss, pos_sim, neg_sim) if `--atc-loss` enabled
  - PopArt normalization stats (mean, std)
  - Market baseline (alpha, market_return, agent_return)

### Metrics Now Working
| Category | Metrics | Source |
|----------|---------|--------|
| Core | reward, loss breakdown, entropy | trainer.train() |
| Actions | n_sell/hold/buy/limit, percentages | trajectories |
| Market | alpha, market_return, agent_return | trajectory info |
| PopArt | popart_mean, popart_std | trainer.popart |
| Config | ds24_use_gae, ds24_per_head_clipping, ds24_epsilon_greedy | config |
| Gradients | grad_norm_value/policy/dynamics | trainer metrics |
| ReDo | redo_dormant_neurons, redo_dormant_pct | trainer metrics |
| ATC | atc_loss, atc_pos_sim, atc_neg_sim | trainer metrics |

## Phase 2: Model-Level Integration (COMPLETE)

### TRM Cycle Dynamics (IMPLEMENTED)

**Changes Made:**
1. `TRMReasoning` in `trm_flax.py` already had `return_intermediates=True` support
2. Added `return_intermediates` parameter to `MuZeroNetworks.prediction()` and `initial_inference()`
3. Added `mode='initial_inference_with_intermediates'` for easy DS24 access
4. `train_jax_fast.py` now captures TRM intermediates every `ds24_intermediates_every` epochs:
   - `latent_delta_norm_h{h}_l{l}` for each (h,l) cycle
   - `output_delta_norm_h{h}` for each H-cycle
   - `latent_final_norm`
   - `trm_utilization` (% of cycles with delta > threshold)

### Physics Component Outputs (IMPLEMENTED)

**Changes Made:**
1. `physics_core.py` already had `get_learned_params_from_dict()` static method
2. `train_jax_fast.py` now extracts physics params at ds24_level >= 3:
   - `physics_alpha_mean`
   - `physics_beta_mean`
   - `garch_persistence`

### MCTS Planning Metrics (IMPLEMENTED)

**Changes Made:**
1. `self_play.py` already collects `mcts_root_values` and `horizon_returns` at k=1,5,10
2. `train_jax_fast.py` now computes:
   - `mcts_root_value` - mean root value
   - `mcts_visit_entropy` - computed from policy probs as proxy
   - `value_horizon_{1,5,10}_error` - MSE between root values and actual horizon returns

### Horizon Prediction Errors (IMPLEMENTED)

**Changes Made:**
1. `self_play.py` computes `horizon_returns` at horizons 1, 5, 10 during episode
2. `train_jax_fast.py` compares these to `mcts_root_values` to compute:
   - `value_horizon_1_error`
   - `value_horizon_5_error`
   - `value_horizon_10_error`

## Phase 3: Dashboard Verification

After Phase 2 is complete:
1. Run training with `--ds24-level 3` or higher
2. Start DS24 dashboard (`./tools/DS24/start_ds24.sh`)
3. Verify all panels display correctly:
   - TRM Cycle Utilization Heatmap
   - Policy Crystallization chart
   - Physics Component Breakdown
   - Learned Parameters (α, β)
   - Gradient Norms by Head
   - ReDo dormancy panel
   - ATC contrastive loss panel

## Usage

```bash
# Level 1: Basic CSV logging + config flags
python train_jax_fast.py --synthetic-level 0 --ds24-level 1

# Level 2: + TRM/planning metrics (when implemented)
python train_jax_fast.py --synthetic-level 0 --ds24-level 2

# Level 3: + Physics forward pass
python train_jax_fast.py --synthetic-level 0 --ds24-level 3 --use-prediction-loss

# Full logging with all features
python train_jax_fast.py --synthetic-level 0 --ds24-level 3 \
    --use-prediction-loss --atc-loss --redo
```

## Files Modified (Phase 1 + Phase 2)

### Phase 1
- `train_jax_fast.py` - Integrated DS24MetricsLogger
- `mcts_trader/training/config.py` - Added synthetic_noise
- `mcts_trader/training/cli.py` - Added --synthetic-noise flag
- `mcts_trader/training/setup.py` - Fixed synthetic_noise passthrough

### Phase 2
- `mcts_trader/jax_agent/muzero_model.py` - Added `return_intermediates` support to `prediction()` and `initial_inference()`
- `train_jax_fast.py` - Full DS24 Phase 2 logging:
  - TRM cycle intermediates extraction
  - Physics learned params extraction
  - Horizon prediction error computation
  - Visit entropy calculation from policy probs

## Metrics Summary (All Working)

| Category | Metrics | Source |
|----------|---------|--------|
| **Core** | reward, loss breakdown, entropy | trainer.train() |
| **Actions** | n_sell/hold/buy/limit, percentages | trajectories |
| **Market** | alpha, market_return, agent_return | trajectory info |
| **PopArt** | popart_mean, popart_std | trainer.popart |
| **Config** | ds24_use_gae, ds24_per_head_clipping, ds24_epsilon_greedy | config |
| **Gradients** | grad_norm_value/policy/dynamics | trainer metrics |
| **ReDo** | redo_dormant_neurons, redo_dormant_pct | trainer metrics |
| **ATC** | atc_loss, atc_pos_sim, atc_neg_sim | trainer metrics |
| **TRM Dynamics** | latent_delta_norm_h{h}_l{l}, output_delta_norm_h{h}, latent_final_norm, trm_utilization | model intermediates |
| **Physics** | physics_alpha_mean, physics_beta_mean, garch_persistence | model params |
| **Planning** | mcts_root_value, mcts_visit_entropy, value_horizon_{1,5,10}_error | trajectories |

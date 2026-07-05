# TASK_066: Fix of Gemini Work - Restore Logging & Extract Magic Numbers

**Priority**: CRITICAL
**Status**: COMPLETE ✅
**Completed**: 2026-01-21

## Context
During the refactoring of `train_jax_fast.py` (TASK_065), the detailed logging logic was replaced with a minimal summary. This regression has made it impossible to monitor training health via the console, as critical metrics like Action Distribution, Market Baseline, and Loss Breakdown are no longer visible.

## Missing Log Elements (Regression Audit)

### 1. Epoch Console Output (Epoch X | ...)
**Current (Broken):**
```
Epoch    1 | Reward:   +0.21 | Loss: 47.1926 | Self-play: 118.1s | Train: 75.2s
```

**Required (Original Fidelity):**
```
Epoch   54 | Reward:   +0.21 | α: +0.17% (Mkt: -0.9%) | Fee:  0 | Steps: 1440 | S/H/B/LS/LB: 82/1050/105/120/83 | Self-play:  99.7s | Train:  5.9s | Loss: 57.0592 | Val: 4.5720 | Pol: 4.9103 | Rew: 4.3593 | Pred: 83.7243 (λ=0.500)
```

**Specific Missing Items:**
- **Market Baseline:** `α` (Alpha) and `Mkt` (Market Return). Essential for knowing if positive reward is just beta.
- **Action Stats:** `S/H/B/LS/LB` counts. Essential for detecting policy collapse (e.g., all HOLD).
- **Fee Level:** `Fee: 0`. Important for curriculum tracking.
- **Step Count:** `Steps: 1440`. Verifies episode length.
- **Loss Breakdown:** `Val`, `Pol`, `Rew`, `Pred` (with lambda). `Total Loss` is useless for debugging specific heads.
- **Physics Reg:** `PhysReg`.
- **Decoder Loss:** `Dec`. (New from TASK_106).

### 2. CSV Logging
The `base_columns` list in the refactored script might be incomplete or the `TrainingLogger` might not be writing them correctly if they aren't calculated in the loop.

**Required Columns:**
- `epoch`, `reward`, `reward_ma20`, `fee_bps`, `steps`
- `alpha`, `market_return`, `agent_return`
- `n_buy`, `n_hold`, `n_sell`, `n_limit_sell`, `n_limit_buy`
- `pct_buy`, `pct_hold`, ...
- `total_loss`, `value_loss`, `policy_loss`, `reward_loss`
- `pred_loss`, `loss_decoder`, `loss_physics_reg`, `lambda_pred`
- `step_1_mse`, `step_k_mse`
- `policy_entropy`, `avg_value`, `value_std`
- `root_value_mean`, `root_value_std`
- `epsilon_greedy`

## Implementation Plan

1.  **Refactor `train_jax_fast.py` Loop**:
    - Re-implement the metric extraction logic that was deleted.
    - Calculate `action_stats` (counts per type).
    - Extract `market_metrics` (alpha, returns) from the trajectory info.
    - Extract `loss_components` from the training metrics.
    
2.  **Format Console String**:
    - Construct the detailed f-string exactly as it was in the original script.
    
3.  **Verify CSV Output**:
    - Ensure the `csv.writerow` call includes all the columns listed above.

## Note
Do not rely on a separate `MetricsCalculator` class if it complicates the immediate fix. Inline logic in `train_jax_fast.py` is acceptable for transparency and to match the original style, provided it is clean.

---

## Part 2: Extract Magic Numbers into PhysicsConfig

### Problem
Hardcoded "magic numbers" that directly influence agent behavior and learning stability:

| Magic Number | Location | Risk |
|--------------|----------|------|
| `REWARD_SCALE = 100.0` | trainer.py | Environment-dependent |
| `alpha_target = 0.05` | trainer.py | Wrong for mean-reverting markets |
| `beta_target = 0.03` | trainer.py | Wrong for trending markets |
| `sigma - 0.01` | trainer.py | Too low for crypto (should be ~0.03) |
| `smoothing = 0.1` | trainer.py | Affects categorical distribution |
| `min_entropy = 3.0` | trainer.py | Policy collapse threshold |

### Solution
Created `PhysicsConfig` dataclass in `mcts_trader/training/config.py` with:
- Configurable physics targets (alpha, beta, sigma)
- Configurable reward scaling
- Configurable entropy thresholds
- Presets for crypto vs forex markets

### New CLI Flags
```bash
--reward-scale 100.0        # Scale for categorical bins
--alpha-target 0.05         # Momentum target
--beta-target 0.03          # Mean reversion target
--sigma-target 0.01         # Volatility target (use 0.03 for crypto)
--label-smoothing 0.1       # Categorical smoothing
--min-entropy 3.0           # Policy collapse threshold
--max-positions 16          # Max simultaneous positions
--min-cash-threshold 100.0  # Min cash for buying
--confidence-threshold 0.0  # Force HOLD threshold
```

### Files Modified
- `mcts_trader/training/config.py` - Added `PhysicsConfig` dataclass
- `mcts_trader/training/cli.py` - Added physics CLI flags
- `mcts_trader/jax_agent/trainer.py` - Use configurable values
- `mcts_trader/training/setup.py` - Pass physics config to trainer
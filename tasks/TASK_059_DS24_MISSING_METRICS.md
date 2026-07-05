# TASK_059: DS24 Dashboard - Display All Logged Metrics

## Status: COMPLETE

## Problem

Many metrics are being logged to CSV but not displayed in the DS24 dashboard. This creates blind spots in training monitoring.

## Missing Metrics by Category

### 1. Adaptive Difficulty (TASK_058) - HIGH PRIORITY

Add to existing Difficulty chart/section:

| Column | Display | Location |
|--------|---------|----------|
| `reward_ema` | Line on difficulty chart | Metrics tab |
| `price_noise` | Stacked area or separate chart | Metrics tab |
| `temporal_jitter` | Stacked area or separate chart | Metrics tab |
| `time_dilation` | Stacked area or separate chart | Metrics tab |
| `phase_shift_prob` | Stacked area or separate chart | Metrics tab |

**Suggested implementation:** Add "Noise Components" chart showing all 4 noise types as stacked area.

### 2. Prediction Loss - HIGH PRIORITY

| Column | Display | Location |
|--------|---------|----------|
| `pred_loss` | Add to Loss Components chart | Metrics tab |
| `lambda_pred` | Show in config panel | Metrics tab |
| `physics_reg` | Add to Loss Components chart | Metrics tab |

### 3. Value Analytics - MEDIUM PRIORITY

| Column | Display | Location |
|--------|---------|----------|
| `avg_value` | New chart or add to Value Calibration | Metrics tab |
| `value_cat` | Add to Loss Components chart | Metrics tab |
| `popart_mean` | New PopArt chart | Metrics tab |
| `popart_std` | New PopArt chart | Metrics tab |
| `value_scale` | Add to PopArt chart | Metrics tab |
| `value_shift` | Add to PopArt chart | Metrics tab |

### 4. Physics Tab - MEDIUM PRIORITY

These columns exist but Physics tab charts are empty:

| Column | Chart |
|--------|-------|
| `G_momentum_mean`, `G_momentum_std` | Physics Components |
| `G_reversion_mean`, `G_reversion_std` | Physics Components |
| `G_volatility_mean`, `G_volatility_std` | Physics Components |
| `G_correlation_mean`, `G_correlation_std` | Physics Components |
| `physics_alpha_mean` | Physics Params |
| `physics_beta_mean` | Physics Params |
| `garch_persistence` | GARCH chart |

**Note:** Charts exist, just need to wire up data in `updatePhysicsTab()`.

### 5. Planning Tab - MEDIUM PRIORITY

These columns exist but Planning tab charts are empty:

| Column | Chart |
|--------|-------|
| `mcts_root_value` | MCTS Calibration scatter |
| `mcts_visit_entropy` | Visit Entropy chart |
| `value_horizon_1_error` | Horizon Accuracy |
| `value_horizon_5_error` | Horizon Accuracy |
| `value_horizon_10_error` | Horizon Accuracy |

**Note:** Charts exist, just need to wire up data in `updatePlanningTab()`.

### 6. Gradient Norms - LOW PRIORITY

| Column | Display | Location |
|--------|---------|----------|
| `grad_norm_value` | Gradient Norms chart | Interactions tab |
| `grad_norm_policy` | Gradient Norms chart | Interactions tab |
| `grad_norm_dynamics` | Gradient Norms chart | Interactions tab |
| `grad_norm_value_clipped` | Second line (dashed) | Interactions tab |
| `grad_norm_policy_clipped` | Second line (dashed) | Interactions tab |
| `grad_norm_dynamics_clipped` | Second line (dashed) | Interactions tab |
| `grad_norm_encoder_clipped` | Add to chart | Interactions tab |

### 7. Evaluation Metrics - LOW PRIORITY

| Column | Display | Location |
|--------|---------|----------|
| `eval_level` | New Evaluation section | Analysis tab |
| `eval_reward` | Eval chart | Analysis tab |
| `eval_alpha` | Eval chart | Analysis tab |
| `eval_pct_hold` | Eval chart | Analysis tab |

### 8. Config/Info (display in header or config panel)

| Column | Display |
|--------|---------|
| `fee_bps` | Config panel |
| `steps` | Config panel |
| `synthetic_level` | Config panel |
| `model_size` | Config panel |
| `uncertainty` | Stat card or Planning tab |

## Implementation Plan

### Phase 1: High Priority (Metrics Tab) - COMPLETE
- [x] Add Noise Components chart (stacked area: price_noise, temporal_jitter, time_dilation, phase_shift_prob)
- [x] Add reward_ema line to Difficulty chart
- [x] Add pred_loss to Loss Components chart
- [x] Add physics_reg to Loss Components chart

### Phase 2: Wire Up Existing Charts - COMPLETE
- [x] Wire Physics tab charts to data columns (already wired)
- [x] Wire Planning tab charts to data columns (already wired)
- [x] Wire Gradient Norms chart to data columns (enhanced with encoder + clipped values)

### Phase 3: Value Analytics - COMPLETE
- [x] Add PopArt chart (popart_mean, popart_std, value_scale, value_shift)
- [x] Add avg_value to Value Calibration chart
- [x] Add value_cat to Loss Components (+ Loss Contribution chart in Interactions tab)

### Phase 4: Analysis Tab - COMPLETE
- [x] Add Evaluation section for curriculum generalization metrics (eval_level, eval_reward, eval_alpha, eval_pct_hold)

## Files to Modify

| File | Changes |
|------|---------|
| `tools/DS24/index.html` | Add Noise Components chart container |
| `tools/DS24/dashboard.js` | Add chart init, update functions |
| `tools/DS24/styles.css` | Any needed styling |

## Acceptance Criteria

- [x] All logged metrics visible somewhere in dashboard
- [x] No "empty" chart sections (Physics, Planning tabs functional)
- [x] Noise components clearly visible for TASK_058 monitoring
- [x] pred_loss visible in loss breakdown
- [x] PopArt normalization visible (popart_mean, popart_std, value_scale, value_shift)
- [x] Value analytics visible (avg_value, value_cat)
- [x] Curriculum evaluation visible when data available

## Notes

- Physics/Planning tabs have charts but `updatePhysicsTab()` and `updatePlanningTab()` aren't wiring data
- Some metrics only populated when specific features enabled (e.g., physics_reg only with prediction loss)
- Eval metrics only populated during curriculum training with generalization checks

# Tasks - JAX MuZero Trading Agent

---

## Active Tasks (Training Optimization)

| ID | Task | Priority | Effort | Status |
|----|------|----------|--------|--------|
| 022 | [TD(λ) Returns](TASK_022_TD_LAMBDA.md) | MEDIUM | Medium | OPEN |
| 023 | [Prioritized Replay](TASK_023_PRIORITIZED_REPLAY.md) | LOW | High | OPEN |
| 024 | [LR Schedule](TASK_024_LR_SCHEDULE.md) | HIGH | Easy | OPEN |
| 025 | [Per-Head Gradient Clipping](TASK_025_GRADIENT_CLIPPING_PER_HEAD.md) | MEDIUM | Easy | OPEN |

### Tooling (Low Priority)

| ID | Task | Priority | Status |
|----|------|----------|--------|
| 013 | [Training Configurator](TASK_013_TRAINING_CONFIGURATOR.md) | LOW | OPEN |

---

## Completed Tasks (in `_old/`)

### Training Stability (2/2 COMPLETE)
- TASK_020: MCTS Sims Warmup
- TASK_021: PopArt Normalization

### Audit Tasks (7/7 COMPLETE)
- TASK_001: Replay Buffer Wrap-Around Bug
- TASK_002: Buffer Clear on Regime Change
- TASK_003: Optimizer State on Regime Change
- TASK_004: Orphaned Level 4 Generator
- TASK_005: Remove Dead Code _compute_target_value
- TASK_006: TRM Weight Sharing Deviation
- TASK_007: Deterministic Level 0 Sine

### Prediction Model (5/5 COMPLETE)
- TASK_008: Differentiable Market Physics Core
- TASK_009: Neural Market Corrector
- TASK_010: NeuralMarketODE Integration
- TASK_011: MuZero Auxiliary Prediction Loss
- TASK_012: Predictive Tree Search

### Bug Fixes (20/20 COMPLETE)
- BUG_001 through BUG_020: All fixed

### Tooling
- TASK_DS24: DS24 Metrics Logging Integration

### Deferred
- TASK_02: Portfolio Morphing (adds complexity before basics mastered)
- TASK_04: Portfolio Curriculum L4-6 (Level 4 rotation already exists)

---

## Implementation Priority

For **Tuning Tournament** (next steps):

1. **TASK_024 (LR Schedule)** - Fine-tuning capability
2. **TASK_025 (Gradient Clipping)** - Prevent head imbalance
3. **TASK_022 (TD(λ))** - Variance reduction (medium effort)

---

## Quick Reference

```bash
# List active tasks
ls tasks/TASK_0*.md

# View completed
ls tasks/_old/
```

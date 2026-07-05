# TASK_117: Student-t Output Validation

## Status: READY FOR TESTING

## Goal

Validate that Student-t output distribution improves prediction head performance on fat-tailed financial returns.

---

## Hypothesis

Student-t distribution should:
1. **Reduce outlier over-fitting**: MSE penalizes outliers quadratically; Student-t penalizes logarithmically
2. **Better calibrated uncertainty**: Fat tails match financial return distributions
3. **Maintain point prediction accuracy**: MSE component (20%) prevents divergence

---

## Testing Scenarios

### Scenario 1: Synthetic L0-L2 (Baseline)

**Purpose**: Verify no regression on easy levels with near-Gaussian returns.

```bash
# Without Student-t (baseline)
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 100 \
    --use-prediction-loss \
    --name baseline_mse_L0

# With Student-t
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 100 \
    --use-prediction-loss \
    --use-student-t \
    --name student_t_L0
```

**Metrics to compare**:
- `loss_prediction_raw`
- `step_1_loss`, `step_k_loss`
- `mean_episode_reward`

**Expected**: Similar or slightly better with Student-t.

---

### Scenario 2: Synthetic L6-L8 (Fat Tails)

**Purpose**: Test on high-volatility synthetic data with more outliers.

```bash
# Without Student-t
python train_jax_fast.py \
    --synthetic-level 6 \
    --epochs 100 \
    --use-prediction-loss \
    --name baseline_mse_L6

# With Student-t
python train_jax_fast.py \
    --synthetic-level 6 \
    --epochs 100 \
    --use-prediction-loss \
    --use-student-t \
    --name student_t_L6
```

**Metrics to compare**:
- Prediction loss stability (should be more stable with Student-t)
- Outlier handling (check sigma_mean metric)
- Training curve smoothness

**Expected**: Student-t should show more stable loss curves.

---

### Scenario 3: Historical Data (Real Fat Tails)

**Purpose**: Test on real market data with genuine fat-tailed distributions.

```bash
# Without Student-t
python train_jax_fast.py \
    --periods-file config/benchmark_periods.json \
    --periods "2020_covid_crash" \
    --epochs 50 \
    --use-prediction-loss \
    --name baseline_mse_covid

# With Student-t
python train_jax_fast.py \
    --periods-file config/benchmark_periods.json \
    --periods "2020_covid_crash" \
    --epochs 50 \
    --use-prediction-loss \
    --use-student-t \
    --name student_t_covid
```

**Expected**: Significant improvement on crash periods with extreme returns.

---

### Scenario 4: df Parameter Sensitivity

**Purpose**: Test different degrees of freedom settings.

```bash
# Fat tails (df ~ 5)
python train_jax_fast.py \
    --synthetic-level 6 \
    --epochs 50 \
    --use-prediction-loss \
    --use-student-t \
    --student-t-df-min 3.0 \
    --student-t-df-max 8.0 \
    --name student_t_fat

# Moderate tails (df ~ 15)
python train_jax_fast.py \
    --synthetic-level 6 \
    --epochs 50 \
    --use-prediction-loss \
    --use-student-t \
    --student-t-df-min 10.0 \
    --student-t-df-max 20.0 \
    --name student_t_moderate

# Near-Gaussian (df ~ 50)
python train_jax_fast.py \
    --synthetic-level 6 \
    --epochs 50 \
    --use-prediction-loss \
    --use-student-t \
    --student-t-df-min 40.0 \
    --student-t-df-max 60.0 \
    --name student_t_gaussian
```

**Expected**: Fat tails best for high-volatility, moderate for mixed data.

---

### Scenario 5: Full Dojo Curriculum

**Purpose**: End-to-end validation with Dojo v4.

```bash
python tools/orchestration/dojo_v4_full_curriculum.py \
    --model-name student_t_test \
    --model-size small \
    --epochs 200
```

**Metrics to track**:
- Curriculum completion rate
- Benchmark scores per phase
- Prediction loss trends

---

## Success Criteria

| Metric | Baseline (MSE) | Target (Student-t) |
|--------|---------------|-------------------|
| L0-L2 prediction loss | X | X ± 5% |
| L6-L8 loss stability | High variance | Lower variance |
| Historical crash periods | Poor | Improved |
| Training time | T | T ± 10% |

---

## Metrics to Log

New metrics available when `--use-student-t` is enabled:

```python
# In DS24 dashboard / logs
metrics['sigma_mean']        # Estimated scale parameter
metrics['df']                # Degrees of freedom used
metrics['student_t_nll_mean'] # Raw Student-t NLL (before weighting)
```

---

## Quick Validation Commands

```bash
# Quick smoke test (2 epochs)
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 2 \
    --use-prediction-loss \
    --use-student-t \
    --name quick_student_t_test

# Check metrics are logged
grep -i "sigma_mean\|student_t" logs/quick_student_t_test.log
```

---

## Notes

- Student-t adds ~5% overhead to prediction loss computation
- Sigma is estimated per-step from error statistics (adaptive)
- df is fixed from config (learnable version in StudentTHead for future)
- Works with existing model architecture (no changes needed)

---

## Files Modified for TASK_116

| File | Change |
|------|--------|
| `mcts_trader/jax_agent/prediction/student_t.py` | NEW |
| `mcts_trader/jax_agent/prediction/__init__.py` | Exports |
| `mcts_trader/jax_agent/trainer.py` | Loss computation |
| `mcts_trader/training/config.py` | Config fields |
| `mcts_trader/training/setup.py` | Config wiring |
| `mcts_trader/training/cli.py` | CLI args |
| `tools/orchestration/dojo_*.py` | Enabled by default |

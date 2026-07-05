# TASK_134: L3 Prediction Loss 500+ Investigation

**Date**: 2026-02-02
**Status**: INVESTIGATING (fixes applied, testing in progress)
**Severity**: HIGH - Training cannot learn on L3

## Symptom

Training shows prediction loss of 500+ specifically on Level 3 (noisy_trends), while other levels show ~50. Model unable to solve any synthetic level.

```
L0 with noise: ~50
L1 with noise: ~50
L2 with noise: ~50
L3 with noise: 500+ (sometimes 140-1400)
```

User command being tested:
```bash
python train_jax_fast.py --name fix134 --synthetic-level 3 --epochs 50 \
  --use-prediction-loss --use-student-t --prediction-loss-weight 0.1 ...
```

## Investigation Findings

### Hypothesis 1: L3 Random Walk Creates Unbounded Price Drift

**Location**: `mcts_trader/data/synthetic_market.py:433-474`

L3 uses multiplicative random walk:
```python
TREND_STRENGTH = 0.005   # 0.5% drift per step
NOISE_STD = 0.02         # 2% volatility
change = 1 + drift + noise
prices[t] = prices[t-1] * change
```

Over 10,000 steps with 2% volatility + trend drift:
- Prices can range from 0.01 to 10,000+
- Log-prices range: log(0.01) to log(10000) = -4.6 to +9.2 (range ~14)
- vs L0-L2: log-prices stay around 4.6 ± 0.3 (range ~0.6)

### Hypothesis 2: Decoder Loss Scales Quadratically with Range

**Location**: `mcts_trader/jax_agent/trainer.py:806`

```python
per_sample_loss_p = jnp.mean((market_state['log_prices'] - teacher_sequence[:, 0, :]) ** 2, axis=-1)
```

If decoder predicts ~4.6 (trained on L0-L2) but L3 target is 9.2:
- Error = 4.6
- MSE = 21.16 per asset
- With 64 assets: decoder loss alone = ~1,350

### Hypothesis 3: Student-t Loss Configuration

User is using `--use-student-t` which adds:
```python
student_t_nll = log_sigma_shifted + 0.5 * (df + 1) * jnp.log1p(z ** 2 / df)
```

With large errors from L3's scale mismatch, this could compound the loss.

## Quantitative Analysis

| Level | Price Range | Log-Price Range | Expected Decoder Loss |
|-------|-------------|-----------------|----------------------|
| L0 | 85-115 | 4.4-4.7 (~0.3) | ~0.01 |
| L1 | 80-120 | 4.4-4.8 (~0.4) | ~0.02 |
| L2 | 70-130 | 4.2-4.9 (~0.7) | ~0.10 |
| L3 | 0.01-10000 | -4.6 to 9.2 (~14) | **~1,350** |

## Fixes Applied (UNVERIFIED)

### Fix 1: Normalized Decoder Loss (`trainer.py`)

Changed decoder loss to:
1. Center log-prices per-batch (remove absolute scale)
2. Use Huber loss instead of MSE (robust to outliers)

**Concern**: This changes loss semantics - may have unintended effects.

### Fix 2: Bounded L3 Generator (`synthetic_market.py`)

Changed L3 parameters:
- NOISE_STD: 0.02 → 0.008 (reduced 2.5x)
- TREND_STRENGTH: 0.005 → 0.003 (reduced 1.7x)
- Added mean-reversion toward base price
- Clamped prices to [50, 200]

**Concern**: This changes L3's learning objective - it's now easier but different.

## Questions To Answer

1. Is the decoder loss actually the culprit? (Check loss_decoder vs loss_prediction_raw in logs)
2. Does normalizing decoder loss break learning on L0-L2?
3. Is L3's bounded range now too easy to be a meaningful level?
4. Should we fix at data level (L3 generator) or loss level (decoder) or both?

## Testing Protocol

1. Run with fix: `--synthetic-level 3 --use-prediction-loss`
2. Check prediction loss (should be ~50, not 500+)
3. Check if model actually learns (policy entropy, reward_ma20)
4. Compare to L0 to ensure fix doesn't break other levels

## Files Modified

```
mcts_trader/jax_agent/trainer.py      - Lines 795-855 (decoder loss normalization)
mcts_trader/data/synthetic_market.py  - Lines 433-478 (L3 bounded walk)
```

## Rollback Instructions

If fixes cause issues, revert:
1. `git checkout HEAD~1 -- mcts_trader/jax_agent/trainer.py`
2. `git checkout HEAD~1 -- mcts_trader/data/synthetic_market.py`

Or manually restore original decoder loss (raw MSE without normalization).

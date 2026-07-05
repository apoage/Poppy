# CSV Training Metrics Reference

CSV log files are saved to `logs/training_{name}.csv`

## Column Reference

### Core Metrics
| Column | Type | Description |
|--------|------|-------------|
| `epoch` | int | Global epoch number (continuous across resumes) |
| `reward` | float | Episode total reward (% return, can be negative) |
| `reward_ma20` | float | 20-epoch moving average of reward (trend indicator) |
| `fee_bps` | int | Trading fee in basis points (0=no fees, 10=0.1%) |
| `steps` | int | Number of steps/actions in episode |

### Action Distribution
| Column | Type | Description |
|--------|------|-------------|
| `n_buy` | int | Count of BUY actions |
| `n_hold` | int | Count of HOLD actions |
| `n_sell` | int | Count of SELL actions |
| `pct_buy` | float | Percentage of BUY actions (0-100) |
| `pct_hold` | float | Percentage of HOLD actions (0-100) |
| `pct_sell` | float | Percentage of SELL actions (0-100) |

**Interpretation:**
- Healthy exploration: ~30% each action type
- Buy-heavy (>45%): Model accumulating positions
- Sell-heavy (>45%): Model exiting positions
- Hold-heavy (>60%): Conservative/uncertain policy
- Sell near 0%: Model not learning to exit (warning sign)

### Timing
| Column | Type | Description |
|--------|------|-------------|
| `self_play_time` | float | Time in seconds for MCTS self-play episode |
| `train_time` | float | Time in seconds for training batch (0 during warmup) |

**Interpretation:**
- Self-play increasing: MCTS doing more meaningful search (good after value head calibrates)
- Train time stable: ~4s per 50 iterations typical

### Loss Components
| Column | Type | Description |
|--------|------|-------------|
| `total_loss` | float | Combined weighted loss (value + policy + reward - entropy) |
| `value_loss` | float | Value head loss (hybrid MSE+Cat, weighted) |
| `value_mse` | float | MSE component: mean((predicted - target)^2) |
| `value_cat` | float | Categorical cross-entropy on 101-bin distribution |
| `policy_loss` | float | Policy cross-entropy vs MCTS target distribution |
| `reward_loss` | float | Reward prediction loss (dynamics model) |

**Interpretation:**
- `value_mse` dropping: Value head learning to predict returns
- `value_mse` near 0 with high `value_cat`: Model predicting mean (bad)
- `policy_loss` decreasing: Policy aligning with MCTS
- `reward_loss` stable ~2-3: Normal for dynamics model

**Expected ranges (Level 0, after calibration):**
- `value_mse`: 1-5 (lower = better calibration)
- `value_cat`: 4-5 (categorical entropy)
- `policy_loss`: 2-4 (lower = more confident)
- `reward_loss`: 2-3 (dynamics model uncertainty)

### Policy/Value Analytics
| Column | Type | Description |
|--------|------|-------------|
| `policy_entropy` | float | Policy distribution entropy (higher = more exploration) |
| `avg_value` | float | Mean predicted value across episode (expected return) |
| `value_std` | float | Std of predicted values (diversity of predictions) |

### Value Calibration
| Column | Type | Description |
|--------|------|-------------|
| `value_scale` | float | Calibration scale applied to value predictions (default 1.0) |
| `value_shift` | float | Calibration shift applied to value predictions (default 0.0) |

**Interpretation:**
- `value_scale` > 1.0: Predictions expanded (model was too conservative)
- `value_scale` < 1.0: Predictions compressed (model was too aggressive)
- `value_shift` != 0: Center of predictions adjusted
- With `--adaptive-calibration`: these auto-tune every 25 epochs after epoch 50

**Policy/Value Interpretation:**
- `policy_entropy` ~5.0: Near-uniform (max entropy for 192 actions = ln(192) = 5.26)
- `policy_entropy` ~4.0: Moderate exploration
- `policy_entropy` <3.0: Danger zone - policy collapsing
- `avg_value` positive: Model expects positive returns
- `value_std` >1.0: Model differentiating state values (good)
- `value_std` ~0: Model predicting same value everywhere (bad)

### Config Reference
| Column | Type | Description |
|--------|------|-------------|
| `synthetic_level` | int | Curriculum level (0=sine, 1=multi-freq, 2=trends, 3=regime) |
| `model_size` | str | Model size ("small"=7M, "medium"=15M, "large"=36M) |

## Key Derived Metrics for Analysis

### Value-Reward Correlation
```python
correlation = df['avg_value'].corr(df['reward'])
```
- >0.8: Excellent calibration (value head predicting well)
- 0.5-0.8: Learning but not fully calibrated
- <0.5: Value head not informative

### Entropy Change Rate
```python
early_entropy = df.iloc[:10]['policy_entropy'].mean()
recent_entropy = df.iloc[-20:]['policy_entropy'].mean()
entropy_change_pct = 100 * (recent_entropy - early_entropy) / early_entropy
```
- Positive: More exploration (may be stuck)
- -10% to -20%: Healthy exploitation learning
- <-30%: Policy may be collapsing

### Learning Phases
1. **Warmup (epochs 1-5)**: No training, loss=0, random policy
2. **Exploration (entropy >4.0)**: Model exploring action space
3. **Learning (entropy 3.0-4.0, value_mse decreasing)**: Active learning
4. **Exploitation (entropy <3.0)**: Converged policy

### Warning Signs
- `value_mse` = 0 with `value_std` = 0: Circular training bug
- `pct_sell` = 0: Not learning to exit positions
- `policy_entropy` < 2.5: Near-deterministic collapse
- `reward_ma20` declining while `value_mse` stable: Overfitting

## File Locations
- Training logs: `logs/training_{name}.csv`
- Checkpoints: `jax_ckpt_fast/{name}/`
- Notes: `NOTES_YYYYMMDD.md`
- Comparisons: `logs/cv_comparison_{timestamp}.json`

## Cross-Validation Tool

Use `cross_validate.py` to compare training runs:

```bash
# Compare two runs
python cross_validate.py --baseline run1 --experiment run2

# Compare multiple CSV files
python cross_validate.py --compare logs/training_*.csv

# Quick A/B test (runs two configs for N epochs)
python cross_validate.py --quick-test --epochs 50
```

### Key Comparison Metrics
- `reward_final_ma20`: Final 20-epoch reward average (higher = better)
- `value_reward_corr`: Value-reward correlation (>0.8 = excellent)
- `value_loss_final`: Final value loss (lower = better)
- `entropy_final`: Policy entropy (3.0-4.5 = healthy range)
- `avg_self_play_time`: Self-play speed (lower = better)

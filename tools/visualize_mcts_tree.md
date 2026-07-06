# visualize_mcts_tree.py

**MCTS Search Behavior Visualization**

## Purpose

Analyze MCTS search behavior from training CSV logs. Generates multi-panel visualizations showing policy evolution, value calibration, and action distributions.

## Usage

```bash
python tools/visualize_mcts_tree.py logs/training_default.csv
```

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `csv_file` | Yes | Path to training CSV log |

## Output

Multi-panel PNG visualization showing:

1. **Action Distribution Evolution** - Sell/Hold/Buy counts over epochs
2. **Policy Entropy** - Exploration vs exploitation balance
3. **Value Calibration** - Predicted values vs actual rewards (scatter)
4. **Reward Distribution** - Histogram of episode rewards

## Expected CSV Columns

The input CSV should contain:
- `epoch` - Training epoch number
- `n_sell`, `n_hold`, `n_buy` - Action counts per epoch
- `policy_entropy` - Policy entropy value
- `avg_value` - Average predicted value
- `reward` - Actual episode reward

## Interpretation

**Action Distribution**:
- Equal line at 33% = balanced exploration
- Skew toward HOLD = conservative policy
- Skew toward BUY/SELL = active trading

**Policy Entropy**:
- > 4.0 = High exploration (random)
- 2.0-3.0 = Balanced
- < 1.5 = Deterministic (exploitation)

**Value Calibration**:
- Points on red diagonal = perfect prediction
- Scatter above = overoptimistic
- Scatter below = overpessimistic

## Dependencies

- matplotlib
- pandas
- numpy

## Related Files

- `logs/training_*.csv` - Training logs
- `train_jax_fast.py` - Generates training CSV

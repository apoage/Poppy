# log_parser.py

**Training Log Parser**

## Purpose

Parse and analyze training log files. Extract metrics, identify trends, detect anomalies.

## Usage

```bash
python tools/log_parser.py logs/training_default.csv
```

## Function

1. Reads training CSV log
2. Parses epoch-level metrics
3. Computes aggregate statistics
4. Detects training issues

## Metrics Analyzed

| Metric | Description |
|--------|-------------|
| `reward` | Episode reward |
| `policy_entropy` | Policy randomness |
| `avg_value` | Predicted value |
| `n_buy/sell/hold` | Action distribution |
| `loss` | Training loss |

## Output

```
Training Log Analysis: logs/training_default.csv
================================================
Epochs: 200
Duration: 2h 15m

Performance:
  Best reward: +15.2% (epoch 145)
  Avg reward: +2.3%
  Win rate: 58%

Policy:
  Avg entropy: 3.2 (balanced)
  Action bias: HOLD (45%)

Issues Detected:
  ⚠️ Entropy collapse at epoch 80-90
  ✅ Value calibration good
```

## Dependencies

- pandas

## Related Files

- `logs/training_*.csv` - Log files
- `visualize_mcts_tree.py` - Visualize metrics

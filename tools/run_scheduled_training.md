# run_scheduled_training.py

**Scheduled Training Runner**

## Purpose

Execute training runs defined in `training_schedule.json`. Enables automated multi-period, multi-asset training campaigns.

## Usage

```bash
python tools/run_scheduled_training.py
```

## Schedule Format

Create `training_schedule.json`:
```json
{
  "periods": [
    {
      "start": "2024-01-01",
      "end": "2024-01-07",
      "duration_days": 7,
      "runs": [
        {
          "assets": ["BTC/USD", "ETH/USD"],
          "epochs": 100
        }
      ]
    }
  ],
  "config": {
    "steps": 200,
    "fee": 26,
    "slippage": 5,
    "sims": 16,
    "model_size": "small"
  }
}
```

## Parameters (from schedule)

| Parameter | Description |
|-----------|-------------|
| `periods.start` | Period start date |
| `periods.end` | Period end date |
| `periods.runs` | List of training runs |
| `config.epochs` | Epochs per run |
| `config.steps` | Steps per episode |
| `config.fee` | Trading fee (bps) |
| `config.sims` | MCTS simulations |

## CUDA Environment

Automatically configures LD_LIBRARY_PATH for pip-installed CUDA libraries.

## Logging

Logs to `logs/scheduled_training.log` with timestamps.

## Example Output

```
2025-12-26 14:00:00 - [SCHEDULE] - Starting period 1: 2024-01-01 to 2024-01-07
2025-12-26 14:00:01 - [SCHEDULE] - Running: BTC/USD, ETH/USD (100 epochs)
2025-12-26 14:30:00 - [SCHEDULE] - Run completed. Reward: +2.5%
```

## Dependencies

- subprocess
- json

## Related Files

- `training_schedule.json` - Schedule definition
- `train_jax_fast.py` - Training script called
- `parallel_epoch_training.py` - Parallel alternative

# map_training_data.py

**Training Data Segment Mapper**

## Purpose

Map available training data segments by date range and asset coverage. Generates schedule for curriculum training.

## Usage

```bash
python tools/map_training_data.py
```

## Function

1. Scans `trading_data.db` for available data
2. Groups by contiguous date ranges
3. Identifies asset overlap per period
4. Generates `training_schedule.json`

## Output

Creates `training_schedule.json`:
```json
{
  "segments": [
    {
      "start": "2024-12-01",
      "end": "2024-12-07",
      "assets": ["XXBTZUSD", "XETHZUSD", ...],
      "n_assets": 64,
      "n_candles": 10080
    }
  ],
  "total_coverage": {
    "start": "2024-12-01",
    "end": "2024-12-26",
    "days": 26
  }
}
```

## Use Cases

- Plan training campaigns
- Identify data gaps
- Select assets with full coverage

## Dependencies

- sqlite3
- json

## Related Files

- `trading_data.db` - Data source
- `training_schedule.json` - Output schedule
- `run_scheduled_training.py` - Execute schedule

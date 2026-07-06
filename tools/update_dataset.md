# update_dataset.py

**Dataset Incremental Updater**

## Purpose

Update existing dataset with new OHLCV candles. Fetches only missing data since last update.

## Usage

```bash
python tools/update_dataset.py
```

## Function

1. Reads `trading_data.db` to find last timestamp per asset
2. Fetches new candles from Kraken since last update
3. Inserts new data incrementally
4. Updates asset statistics

## Benefits

- **Efficient**: Only downloads missing data
- **Resumable**: Can be interrupted and resumed
- **Rate-limited**: Respects Kraken API limits

## Example Output

```
Checking for updates...
XXBTZUSD: Last candle 2025-12-25 23:59, fetching new...
  Added 1440 new candles (1 day)
XETHZUSD: Last candle 2025-12-25 23:59, fetching new...
  Added 1440 new candles (1 day)
Update complete: 64 assets updated
```

## Dependencies

- mcts_trader.data.database
- mcts_trader.data.fetcher

## Related Files

- `trading_data.db` - Target database
- `download_data.py` - Initial download
- `validate_database.py` - Validate after update

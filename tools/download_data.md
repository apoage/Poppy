# download_data.py

**Kraken Historical Data Downloader**

## Purpose

Fetch historical OHLCV (Open, High, Low, Close, Volume) data from Kraken exchange for top volume assets.

## Usage

```bash
# Download 7 days for top 64 assets by volume
python tools/download_data.py --days 7 --count 64

# Download 30 days of data
python tools/download_data.py --days 30

# Download from specific asset list
python tools/download_data.py --days 7 --file assets.csv
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--days` | 7 | Days of history to fetch |
| `--count` | 64 | Number of assets (by USD volume) |
| `--file` | None | Path to asset list CSV |

## Function

1. Connects to Kraken API
2. Fetches top N assets by 24h USD volume
3. Downloads 1-minute OHLCV candles for each asset
4. Stores in `trading_data.db` (SQLite)

## Rate Limiting

- Respects Kraken public API limits (~1 req/sec)
- Chunk requests to avoid timeouts
- Supports incremental updates

## Database Schema

**assets table**:
- `id`, `symbol`, `wsname`, `active`, `status`

**ohlcv table**:
- `asset_id`, `timestamp`, `open`, `high`, `low`, `close`, `volume`

## Example Output

```
Fetching tickers to determine top volume assets...
Top 3 Assets by Volume: [('XXBTZUSD', 1.2B), ('XETHZUSD', 800M), ...]
Downloading XXBTZUSD... 10080 candles
Downloading XETHZUSD... 10080 candles
```

## Dependencies

- mcts_trader.data.kraken_client
- mcts_trader.data.database
- mcts_trader.data.fetcher

## Related Files

- `trading_data.db` - Output database
- `validate_database.py` - Validate downloaded data
- `update_dataset.py` - Incremental updates

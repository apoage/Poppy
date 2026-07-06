# test_kraken_connection.py

**Kraken API Connection Test**

## Purpose

Test connectivity to Kraken API endpoints.

## Usage

```bash
python tools/test_kraken_connection.py
```

## Tests

| Endpoint | Description |
|----------|-------------|
| Time | Server time sync |
| AssetPairs | Tradeable pairs |
| Ticker | Price data |
| OHLC | Candle data |

## Example Output

```
Kraken API Connection Test
==========================
Server time: ✅ 2025-12-26 14:32:15
AssetPairs: ✅ 523 pairs available
Ticker (BTC): ✅ $98,234.50
OHLC (BTC): ✅ 720 candles

All endpoints accessible!
```

## Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| Timeout | Network | Check internet |
| 520 | Kraken down | Wait and retry |
| Rate limit | Too fast | Add delay |

## Dependencies

- mcts_trader.data.kraken_client

## Related Files

- `check_wsname.py` - WebSocket names
- `debug_kraken_pairs.py` - Pair details

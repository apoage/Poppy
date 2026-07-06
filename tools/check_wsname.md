# check_wsname.py

**WebSocket Name Checker**

## Purpose

Check WebSocket names for trading pairs from Kraken API.

## Usage

```bash
python tools/check_wsname.py
```

## Function

Maps API pair names to WebSocket subscription names:
- API: `XXBTZUSD`
- WS: `XBT/USD`

## Example Output

```
WebSocket Names
===============
XXBTZUSD → XBT/USD
XETHZUSD → ETH/USD
SOLUSD → SOL/USD
...
```

## Use Cases

- Setting up WebSocket subscriptions
- Debugging streaming issues
- Mapping pair identifiers

## Dependencies

- mcts_trader.data.kraken_client

## Related Files

- `test_kraken_connection.py` - API test
- `debug_kraken_pairs.py` - Pair details

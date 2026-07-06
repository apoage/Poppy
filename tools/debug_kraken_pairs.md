# debug_kraken_pairs.py

**Kraken Pair Debugger**

## Purpose

Debug Kraken trading pair information and metadata.

## Usage

```bash
python tools/debug_kraken_pairs.py
```

## Output

For each pair:
- Pair name (API)
- WebSocket name
- Base/quote assets
- Lot decimals
- Pair decimals
- Fees
- Margin info

## Example Output

```
Pair: XXBTZUSD
==============
  WS name: XBT/USD
  Base: XXBT (BTC)
  Quote: ZUSD (USD)
  Lot decimals: 8
  Pair decimals: 1
  Fees: [[0, 0.26], [50000, 0.24], ...]
  Margin: Yes (5x)
```

## Use Cases

- Understanding pair structure
- Debugging fee calculations
- Margin trading setup

## Dependencies

- mcts_trader.data.kraken_client

## Related Files

- `test_kraken_connection.py` - Connection test
- `check_wsname.py` - WS names

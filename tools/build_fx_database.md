# build_fx_database.py

**FX Rate Database Builder**

## Purpose

Build FX (foreign exchange) rate database for currency conversion. Enables USD normalization across all trading pairs.

## Usage

```bash
python tools/build_fx_database.py
```

## Function

1. Connects to `trading_data.db`
2. Fetches official pairs from Kraken API
3. Uses bridge assets (BTC, ETH, SOL, XRP, ADA) for cross-rates
4. Builds consensus FX rates with weighted averaging
5. Creates `fx_rates` table

## Bridge Assets

| Asset | Weight | Purpose |
|-------|--------|---------|
| BTC | 1.0 | Primary bridge |
| ETH | 1.0 | Primary bridge |
| SOL | 0.7 | Secondary bridge |
| XRP | 0.5 | Secondary bridge |
| ADA | 0.5 | Secondary bridge |

## Consensus Algorithm

1. Calculate rates via each bridge asset
2. Weight by bridge asset reliability
3. Reject outliers (> 0.1% deviation)
4. Average remaining rates

## Database Schema

**fx_rates table**:
```sql
CREATE TABLE fx_rates (
    base_currency TEXT,
    quote_currency TEXT,
    rate REAL,
    timestamp INTEGER,
    bridge_assets TEXT
);
```

## Example Output

```
Connecting to database: trading_data.db
Fetching official asset pairs from Kraken API...
Building FX rates via BTC bridge...
Building FX rates via ETH bridge...
Consensus rate EUR/USD: 1.0842 (4 bridges agree)
Created 156 FX rate entries
```

## Dependencies

- requests (Kraken API)
- sqlite3

## Related Files

- `trading_data.db` - Target database
- `validate_fx_rates.py` - Validate FX consistency

# debug_db_stats.py

**Database Statistics Debug**

## Purpose

Show detailed statistics about the trading database.

## Usage

```bash
python tools/debug_db_stats.py
```

## Output

- Asset count
- OHLCV row count per asset
- Date range coverage
- Missing data gaps
- Table sizes

## Example Output

```
Database Statistics: trading_data.db
=====================================
Assets: 64
Total OHLCV rows: 2,580,480

Per-asset coverage:
  XXBTZUSD: 40,320 rows (28 days)
  XETHZUSD: 40,320 rows (28 days)
  ...

Data gaps detected:
  SOLUSD: gap 2024-12-15 to 2024-12-16

Table sizes:
  assets: 64 rows, 8 KB
  ohlcv: 2.5M rows, 180 MB
```

## Dependencies

- sqlite3

## Related Files

- `trading_data.db` - Database
- `validate_database.py` - Validate assets

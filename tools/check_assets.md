# check_assets.py

**Quick Asset Check**

## Purpose

Quick check of assets stored in the database.

## Usage

```bash
python tools/check_assets.py
```

## Function

Shows first 10 assets from `trading_data.db`:
- Asset ID
- Symbol

## Example Output

```
[(1, 'XXBTZUSD'), (2, 'XETHZUSD'), (3, 'SOLUSD'), ...]
```

## Dependencies

- sqlite3

## Related Files

- `trading_data.db` - Database
- `debug_db_stats.py` - Detailed stats

# validate_database.py

**Database Asset Validation Tool**

## Purpose

Validate all assets in `trading_data.db` against Kraken API and delisted pairs CSV. Updates asset status to track which pairs are still tradeable.

## Usage

```bash
python tools/validate_database.py
```

## Function

1. Fetches current tradeable pairs from Kraken API
2. Loads `delistedpairs.csv` if present
3. Updates `assets` table with:
   - `active` (1 = tradeable, 0 = delisted)
   - `status` (tradeable/delisted/unknown)
   - `last_validated_at` (Unix timestamp)

## Database Schema Updates

Adds columns if missing:
```sql
ALTER TABLE assets ADD COLUMN active INTEGER DEFAULT 1;
ALTER TABLE assets ADD COLUMN status TEXT;
ALTER TABLE assets ADD COLUMN last_validated_at INTEGER;
```

## Required Files

- `trading_data.db` - Database to validate
- `delistedpairs.csv` (optional) - Known delisted pairs

## Example Output

```
Checking database schema...
Fetching AssetPairs from Kraken...
Fetched 523 pairs from Kraken
Loaded 12 delisted assets from CSV
Updated 64 assets: 61 active, 3 delisted
```

## Dependencies

- mcts_trader.data.kraken_client
- aiosqlite
- pandas

## Related Files

- `trading_data.db` - Target database
- `delistedpairs.csv` - Delisted pairs list
- `download_data.py` - Initial data download

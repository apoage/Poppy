# validate_fx_rates.py

**FX Rate Validation**

## Purpose

Validate FX (foreign exchange) rate consistency in the database.

## Usage

```bash
python tools/validate_fx_rates.py
```

## Checks

1. **Triangular consistency**: USD→EUR→GBP→USD = 1.0
2. **Inverse consistency**: USD/EUR × EUR/USD = 1.0
3. **Bridge agreement**: Rates via different bridges match
4. **Staleness**: Rates are recent

## Example Output

```
FX Rate Validation
==================
Triangular consistency: ✅ (max error: 0.02%)
Inverse consistency: ✅ (all pairs valid)
Bridge agreement: ✅ (4/5 bridges agree)
Staleness: ⚠️ EUR/USD is 2 days old

Summary: 1 warning, 0 errors
```

## Dependencies

- sqlite3

## Related Files

- `trading_data.db` - Database with fx_rates
- `build_fx_database.py` - Build FX rates

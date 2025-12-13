#!/usr/bin/env python3
"""Detailed debugging of HistoricalMarket"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

db_path = "trading_data.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("="*60)
print("DATABASE ANALYSIS")
print("="*60)
print()

# Count total assets
cursor.execute("SELECT COUNT(*) FROM assets")
total_assets = cursor.fetchone()[0]
print(f"Total assets in database: {total_assets}")

# Count total timestamps
cursor.execute("SELECT COUNT(DISTINCT timestamp) FROM ohlcv")
total_timestamps = cursor.fetchone()[0]
print(f"Total unique timestamps: {total_timestamps:,}")
print()

# Top 64 assets by data coverage
query = """
    SELECT a.id, a.symbol, COUNT(o.timestamp) as cnt
    FROM assets a
    JOIN ohlcv o ON a.id = o.asset_id
    GROUP BY a.id
    ORDER BY cnt DESC
    LIMIT 64
"""
cursor.execute(query)
top_assets = cursor.fetchall()
print(f"Top 64 assets by coverage:")
for i, (asset_id, symbol, cnt) in enumerate(top_assets[:5], 1):
    print(f"  {i}. {symbol} (ID {asset_id}): {cnt:,} data points")
print(f"  ... ({len(top_assets)} total)")
print()

# Get their IDs
asset_ids = [r[0] for r in top_assets]
asset_ids_str = ','.join(map(str, asset_ids))

# Count timestamps where ≥10 of these assets have data
query_valid = f"""
    SELECT timestamp
    FROM ohlcv
    WHERE asset_id IN ({asset_ids_str})
    GROUP BY timestamp
    HAVING COUNT(DISTINCT asset_id) >= 10
    ORDER BY timestamp ASC
"""
cursor.execute(query_valid)
valid_timestamps = cursor.fetchall()
print(f"Timestamps with ≥10 of our 64 assets: {len(valid_timestamps):,}")
print()

if len(valid_timestamps) > 0:
    # Test a random timestamp
    test_ts = valid_timestamps[len(valid_timestamps)//2][0]  # Middle timestamp
    print(f"Testing middle timestamp: {test_ts}")

    # Get data for that timestamp
    query_data = f"""
        SELECT asset_id, close
        FROM ohlcv
        WHERE timestamp = ? AND asset_id IN ({asset_ids_str})
    """
    cursor.execute(query_data, (test_ts,))
    data_rows = cursor.fetchall()
    print(f"  Assets with data at this timestamp: {len(data_rows)}")

    if len(data_rows) > 0:
        print(f"  Sample: Asset ID {data_rows[0][0]}, close={data_rows[0][1]}")
    else:
        print(f"  ❌ NO DATA RETURNED!")

print()
print("="*60)

conn.close()

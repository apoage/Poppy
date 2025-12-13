
import sqlite3
from datetime import datetime

def check_db_stats(db_path):
    conn = sqlite3.connect(db_path)
    
    # 1. Min/Max timestamp
    min_ts = conn.execute('SELECT MIN(timestamp) FROM ohlcv').fetchone()[0]
    max_ts = conn.execute('SELECT MAX(timestamp) FROM ohlcv').fetchone()[0]
    
    print(f"DB Range: {datetime.fromtimestamp(min_ts)} to {datetime.fromtimestamp(max_ts)}")
    
    # 2. Total assets
    total_assets = conn.execute('SELECT COUNT(*) FROM assets').fetchone()[0]
    print(f"Total defined assets: {total_assets}")
    
    # 3. Sample asset counts by year
    print("\nAsset counts (distinct asset_id in ohlcv) by year:")
    current_ts = min_ts
    while current_ts < max_ts:
        year = datetime.fromtimestamp(current_ts).year
        next_year_ts = datetime(year + 1, 1, 1).timestamp()
        
        # Check a 1-day sample in the middle of the year (or start)
        sample_ts = current_ts + (180 * 24 * 3600) # mid-year
        if sample_ts > max_ts: sample_ts = current_ts
        
        start_sample = sample_ts
        end_sample = sample_ts + (24*3600)
        
        count = conn.execute('''
            SELECT COUNT(DISTINCT asset_id) 
            FROM ohlcv 
            WHERE timestamp BETWEEN ? AND ?
        ''', (start_sample, end_sample)).fetchone()[0]
        
        print(f"  {year}: ~{count} active assets (sampled at {datetime.fromtimestamp(sample_ts)})")
        
        current_ts = next_year_ts

    conn.close()

if __name__ == "__main__":
    check_db_stats("trading_data.db")

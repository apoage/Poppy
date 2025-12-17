import sqlite3
import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional

DB_PATH = "trading_data.db"

# 1. Canonical Pair Mapping for Validation (Kraken Ticker Aliases)
# These work with the Ticker endpoint.
KRAKEN_FX_MAP = {
    'EUR': ('EURUSD', False),
    'GBP': ('GBPUSD', False),
    'AUD': ('AUDUSD', False),
    'JPY': ('USDJPY', True),   # Invert
    'CAD': ('USDCAD', True),   # Invert
    'CHF': ('USDCHF', True),   # Invert
    'USDT': ('USDTUSD', False)
}

def get_random_samples(n=50):
    conn = sqlite3.connect(DB_PATH)
    FX_SYMBOLS = list(KRAKEN_FX_MAP.keys())
    
    # Get total count for FX symbols
    query_count = f"SELECT COUNT(*) FROM fx_rates WHERE symbol IN ({','.join(['?' for _ in FX_SYMBOLS])})"
    count = conn.execute(query_count, FX_SYMBOLS).fetchone()[0]
    
    if count == 0:
        print("No FX records found for validation.")
        conn.close()
        return pd.DataFrame()

    # Get random samples from FX symbols
    # Note: We are validating 'Magnitude' and 'Logic', so random timestamps are fine 
    # even if they drift from current price. We just want to ensure JPY is ~0.006, not 150.
    query_select = f"""
    SELECT timestamp, symbol, rate 
    FROM fx_rates 
    WHERE symbol IN ({','.join(['?' for _ in FX_SYMBOLS])})
    ORDER BY RANDOM() 
    LIMIT ?
    """
    
    df = pd.read_sql_query(query_select, conn, params=FX_SYMBOLS + [n])
    conn.close()
    return df

def _get_kraken_fx_ticker(pair: str) -> Optional[float]:
    """
    Fetches the current Kraken FX price using Ticker.
    """
    url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        if data.get("error"):
            # print(f"Kraken Ticker Error for {pair}: {data['error']}")
            return None

        res = data["result"]
        # Ticker result key might be the alias or the internal name.
        # Just grab the first key.
        key = next(iter(res.keys()))
        # 'c' is the last trade closed [price, lot_volume]
        return float(res[key]["c"][0])

    except Exception as e:
        # print(f"Error fetching ticker for {pair}: {e}")
        return None

def get_validation_rate(currency: str) -> Optional[float]:
    """
    Get validation rate (Current Ticker or Peg).
    """
    if currency not in KRAKEN_FX_MAP:
        return None

    pair_name, is_inverted = KRAKEN_FX_MAP[currency]

    # USDT: Peg validation
    if currency == "USDT":
        return 1.0

    # FX: use Ticker
    rate = _get_kraken_fx_ticker(pair_name)

    if rate is None or rate == 0:
        return None

    return (1.0 / rate) if is_inverted else rate

def run_validation():
    print("Selecting random samples for validation...")
    print("NOTE: Comparing Historical DB Rates against CURRENT Kraken Ticker.")
    print("      Goal is to validate MAPPING and INVERSION logic, not exact price.")
    samples = get_random_samples(50)
    
    if samples.empty:
        print("No samples to validate.")
        return

    results = []
    
    print(f"{'TIMESTAMP':<20} | {'PAIR':<8} | {'MY RATE':<10} | {'CURRENT':<10} | {'DIFF %':<8}")
    print("-" * 70)
    
    for _, row in samples.iterrows():
        ts = row['timestamp']
        sym = row['symbol']
        my_rate = row['rate']
        
        current_rate = get_validation_rate(sym)
        
        if current_rate is not None:
            # Diff will be large due to time drift, but shouldn't be massive (e.g. > 50%)
            # unless major devaluation event.
            diff = abs(my_rate - current_rate) / current_rate * 100
            print(f"{datetime.fromtimestamp(ts)} | {sym:<8} | {my_rate:<10.5f} | {current_rate:<10.5f} | {diff:<8.4f}%")
            results.append(diff)
        else:
            print(f"{datetime.fromtimestamp(ts)} | {sym:<8} | {my_rate:<10.5f} | {'MISSING':<10} | -")
            
        time.sleep(0.5) 
        
    if results:
        avg_diff = sum(results) / len(results)
        print("-" * 70)
        print(f"Average Deviation: {avg_diff:.4f}%")
        
        # We relax the check because of historical drift
        if avg_diff < 20.0: 
            print("✅ VALIDATION PASSED: Rates are in the correct magnitude.")
        else:
            print("⚠️ VALIDATION WARNING: Large deviation. Check for inversion errors or bad data.")
    else:
        print("No valid comparisons made.")

if __name__ == "__main__":
    run_validation()
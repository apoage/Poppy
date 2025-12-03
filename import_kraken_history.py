import asyncio
import pandas as pd
import os
import logging
import glob
from datetime import datetime
from mcts_trader.data.database import Database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def import_csv_file(db: Database, file_path: str):
    """
    Process a single Kraken historical CSV file.
    Format: timestamp, price, volume
    """
    filename = os.path.basename(file_path)
    # Extract pair name from filename (e.g., "XBTUSD.csv" -> "XBT/USD" or similar)
    # Kraken files usually named like "XBTUSD.csv" or "XXBTZUSD.csv"
    pair_raw = os.path.splitext(filename)[0]
    
    # Normalize pair name (heuristic)
    if "USD" in pair_raw:
        pair = pair_raw.replace("USD", "/USD")
    elif "EUR" in pair_raw:
        pair = pair_raw.replace("EUR", "/EUR")
    elif "XBT" in pair_raw:
        pair = pair_raw.replace("XBT", "BTC") # Normalize XBT to BTC if needed
    else:
        pair = pair_raw # Fallback
        
    logging.info(f"Processing {filename} as {pair}...")
    
    try:
        # Read CSV - Kraken format is typically no header: timestamp, price, volume
        # We assume columns 0, 1, 2 are timestamp, price, volume
        df = pd.read_csv(
            file_path, 
            names=['timestamp', 'price', 'volume'], 
            usecols=[0, 1, 2],
            dtype={'price': float, 'volume': float}
        )
        
        # Convert timestamp to datetime (Kraken uses unix timestamp)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        
        # Resample to 1-minute OHLCV
        ohlcv = df.resample('1min').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        
        # Flatten columns
        ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Drop empty intervals (optional, or fill forward)
        # For crypto, we might want to drop empty or forward fill closing price
        ohlcv.dropna(inplace=True)
        
        # Add count/vwap placeholders if missing (DB expects 8 cols, we have 5)
        # DB Schema: timestamp, open, high, low, close, volume, trade_count
        # We can't easily get trade_count from aggregate without expensive ops, use 1 for now
        
        # Prepare for insertion
        ohlcv['timestamp'] = ohlcv.index.astype('int64') // 10**9
        ohlcv['trade_count'] = 1 # Placeholder
        
        # Get asset ID
        asset_id = await db.get_or_create_asset(pair)
        
        # Convert to list of lists for DB
        # Expected by insert_ohlcv: [timestamp, open, high, low, close, vwap, volume, count]
        # We miss VWAP, use Close as approximation
        records = []
        for index, row in ohlcv.iterrows():
            records.append([
                row['timestamp'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['close'], # VWAP placeholder
                row['volume'],
                row['trade_count']
            ])
            
        # Insert in chunks
        chunk_size = 10000
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i+chunk_size]
            await db.insert_ohlcv(asset_id, chunk)
            
        logging.info(f"Imported {len(records)} candles for {pair}")
        
    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")

async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Import Kraken Historical CSVs')
    parser.add_argument('data_dir', help='Directory containing extracted CSV files')
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        logging.error(f"Directory not found: {args.data_dir}")
        return
        
    db = Database("trading_data.db")
    await db.connect()
    
    # Find all CSVs
    csv_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
    logging.info(f"Found {len(csv_files)} CSV files.")
    
    for csv_file in csv_files:
        await import_csv_file(db, csv_file)
        
    await db.close()
    logging.info("Import complete.")

if __name__ == "__main__":
    asyncio.run(main())

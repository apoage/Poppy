#!/usr/bin/env python3
"""
Import Kraken OHLCVT CSV files into trading_data.db

CSV format: timestamp,open,high,low,close,volume,trade_count (no header)
Filename format: {SYMBOL}_{TIMEFRAME}.csv (e.g., BTCUSD_1.csv for 1-min candles)

Usage:
    python tools/import_kraken_csv.py /path/to/Kraken_OHLCVT_Q1_2025
    python tools/import_kraken_csv.py /path/to/dir1 /path/to/dir2 /path/to/dir3
"""

import argparse
import csv
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_asset_id(cursor: sqlite3.Cursor, symbol: str, symbol_map: Dict[str, int]) -> int:
    """
    Get asset ID if it exists in DB. Returns -1 if not found.
    Uses cached symbol_map for speed.
    """
    # Try exact match
    if symbol in symbol_map:
        return symbol_map[symbol]

    # Try variants (XBT/BTC normalization)
    variants = [
        symbol,
        symbol.replace('XBT', 'BTC'),
        symbol.replace('BTC', 'XBT'),
        symbol.replace('/', ''),  # BTCUSD -> try matching
    ]

    for v in variants:
        if v in symbol_map:
            return symbol_map[v]

    return -1  # Not found - skip this asset

def parse_symbol_from_filename(filename: str) -> str:
    """
    Convert filename to symbol.
    E.g., 'BTCUSD_1.csv' -> 'XBT/USD'
          '1INCHEUR_1.csv' -> '1INCH/EUR'
    """
    # Remove _1.csv suffix
    base = filename.replace('_1.csv', '')

    # Common quote currencies (order matters - check longer ones first)
    quotes = ['USDT', 'USDC', 'USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY', 'CHF', 'ETH', 'BTC', 'XBT']

    for quote in quotes:
        if base.endswith(quote):
            base_currency = base[:-len(quote)]
            # Normalize BTC -> XBT for Kraken
            if base_currency == 'BTC':
                base_currency = 'XBT'
            if quote == 'BTC':
                quote = 'XBT'
            return f"{base_currency}/{quote}"

    # Fallback: assume last 3 chars are quote
    if len(base) > 3:
        return f"{base[:-3]}/{base[-3:]}"

    return base

def import_csv_file(cursor: sqlite3.Cursor, filepath: Path, asset_id: int) -> Tuple[int, int, int]:
    """
    Import a single CSV file.
    Returns (rows_inserted, rows_skipped, rows_zero)
    """
    inserted = 0
    skipped = 0
    zeros = 0

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        batch = []

        for row in reader:
            if len(row) < 6:
                skipped += 1
                continue

            try:
                timestamp = int(row[0])
                open_price = float(row[1])
                high = float(row[2])
                low = float(row[3])
                close = float(row[4])
                volume = float(row[5])
                trade_count = int(row[6]) if len(row) > 6 else 0

                # Validate: skip rows with zero/negative prices
                if open_price <= 0 or high <= 0 or low <= 0 or close <= 0:
                    zeros += 1
                    continue

                # Validate: OHLC sanity (high >= low, high >= open/close, low <= open/close)
                if high < low or high < open_price or high < close or low > open_price or low > close:
                    skipped += 1
                    continue

                batch.append((asset_id, timestamp, open_price, high, low, close, volume, trade_count))

                if len(batch) >= 10000:
                    cursor.executemany(
                        """INSERT OR IGNORE INTO ohlcv
                           (asset_id, timestamp, open, high, low, close, volume, trade_count)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        batch
                    )
                    inserted += len(batch)
                    batch = []

            except (ValueError, IndexError) as e:
                skipped += 1
                continue

        # Insert remaining
        if batch:
            cursor.executemany(
                """INSERT OR IGNORE INTO ohlcv
                   (asset_id, timestamp, open, high, low, close, volume, trade_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                batch
            )
            inserted += len(batch)

    return inserted, skipped, zeros

def import_directory(db_path: str, csv_dir: str) -> Dict[str, int]:
    """Import all 1-minute CSV files from a directory (only for existing assets)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Build symbol map from existing assets
    cursor.execute("SELECT id, symbol FROM assets")
    symbol_map = {row[1]: row[0] for row in cursor.fetchall()}
    logging.info(f"Found {len(symbol_map)} existing assets in database")

    csv_path = Path(csv_dir)
    csv_files = list(csv_path.glob('*_1.csv'))

    logging.info(f"Found {len(csv_files)} 1-minute CSV files in {csv_dir}")

    stats = {'files': 0, 'rows': 0, 'skipped': 0, 'zeros': 0, 'unknown_assets': 0}

    for i, filepath in enumerate(csv_files):
        symbol = parse_symbol_from_filename(filepath.name)
        asset_id = get_asset_id(cursor, symbol, symbol_map)

        if asset_id < 0:
            stats['unknown_assets'] += 1
            continue  # Skip - asset not in our database

        inserted, skipped, zeros = import_csv_file(cursor, filepath, asset_id)

        stats['files'] += 1
        stats['rows'] += inserted
        stats['skipped'] += skipped
        stats['zeros'] += zeros

        if (i + 1) % 50 == 0:
            conn.commit()
            logging.info(f"  Progress: {i+1}/{len(csv_files)} files, {stats['rows']:,} rows, {stats['unknown_assets']} skipped assets")

    conn.commit()
    conn.close()

    return stats

def main():
    parser = argparse.ArgumentParser(description='Import Kraken CSV files into database')
    parser.add_argument('directories', nargs='+', help='Directories containing CSV files')
    parser.add_argument('--db', default='trading_data.db', help='Database path')
    args = parser.parse_args()

    total_stats = {'files': 0, 'rows': 0, 'skipped': 0, 'zeros': 0, 'unknown_assets': 0}

    for csv_dir in args.directories:
        if not os.path.isdir(csv_dir):
            logging.error(f"Directory not found: {csv_dir}")
            continue

        logging.info(f"Importing from: {csv_dir}")
        stats = import_directory(args.db, csv_dir)

        for k, v in stats.items():
            total_stats[k] = total_stats.get(k, 0) + v

        logging.info(f"  Done: {stats['files']} files, {stats['rows']:,} rows")

    logging.info("=" * 50)
    logging.info(f"TOTAL: {total_stats['files']} files, {total_stats['rows']:,} rows imported")
    logging.info(f"Skipped assets (not in DB): {total_stats['unknown_assets']}")
    logging.info(f"Skipped rows (malformed): {total_stats['skipped']:,}")
    logging.info(f"Skipped rows (zero prices): {total_stats['zeros']:,}")

if __name__ == '__main__':
    main()

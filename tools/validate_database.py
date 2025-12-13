#!/usr/bin/env python3
"""
Database Validation Tool
Validates all assets in trading_data.db against:
1. Kraken API (AssetPairs endpoint) - verifies currently tradeable status
2. delistedpairs.csv - checks known delisted assets

Updates the 'assets' table with 'active', 'status', and 'last_validated_at' columns.
"""

import asyncio
import logging
import csv
import sys
import os
import time
import aiosqlite
import pandas as pd
from typing import Set, Dict

# Adjust path to find mcts_trader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcts_trader.data.kraken_client import KrakenClient

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DB_PATH = "trading_data.db"
CSV_PATH = "delistedpairs.csv"

async def load_delisted_csv(path: str) -> Set[str]:
    """Load delisted pairs from CSV"""
    if not os.path.exists(path):
        logging.warning(f"Delisted CSV not found at {path}")
        return set()
    
    try:
        df = pd.read_csv(path)
        # Assuming column 'Asset_or_Pair' holds the symbol
        delisted = set(df['Asset_or_Pair'].str.strip().values)
        logging.info(f"Loaded {len(delisted)} delisted assets from CSV")
        return delisted
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        return set()

async def get_kraken_pairs() -> Dict[str, Dict]:
    """Fetch all tradeable pairs from Kraken"""
    async with KrakenClient() as client:
        logging.info("Fetching AssetPairs from Kraken...")
        pairs = await client.get_tradeable_pairs()
        
        if not pairs:
            logging.error("Failed to fetch pairs from Kraken!")
            return {}
            
        logging.info(f"Fetched {len(pairs)} pairs from Kraken")
        return pairs

async def update_database_schema(db: aiosqlite.Connection):
    """Ensure 'active', 'status', 'last_validated_at' columns exist"""
    logging.info("Checking database schema...")
    
    # Check existing columns
    async with db.execute("PRAGMA table_info(assets)") as cursor:
        columns = [row[1] for row in await cursor.fetchall()]
    
    updates = []
    if 'active' not in columns:
        await db.execute("ALTER TABLE assets ADD COLUMN active INTEGER DEFAULT 1")
        updates.append("active")
    if 'status' not in columns:
        await db.execute("ALTER TABLE assets ADD COLUMN status TEXT")
        updates.append("status")
    if 'last_validated_at' not in columns:
        await db.execute("ALTER TABLE assets ADD COLUMN last_validated_at INTEGER")
        updates.append("last_validated_at")
        
    if updates:
        await db.commit()
        logging.info(f"Added columns: {', '.join(updates)}")
    else:
        logging.info("Schema already up to date.")

async def validate_assets():
    """Main validation logic"""
    
    # 1. Load Data
    delisted_csv = await load_delisted_csv(CSV_PATH)
    kraken_pairs = await get_kraken_pairs()
    
    # Build fast lookup for Kraken pairs (key, wsname, altname)
    kraken_lookup = {} # normalized_symbol -> info
    for key, info in kraken_pairs.items():
        kraken_lookup[key] = info
        if 'wsname' in info:
            kraken_lookup[info['wsname']] = info
        if 'altname' in info:
            kraken_lookup[info['altname']] = info

    # 2. Connect to DB
    async with aiosqlite.connect(DB_PATH) as db:
        await update_database_schema(db)
        
        # 3. Fetch all assets
        async with db.execute("SELECT id, symbol FROM assets") as cursor:
            assets = await cursor.fetchall()
            
        logging.info(f"Validating {len(assets)} assets in database...")
        
        stats = {
            'active': 0,
            'delisted_api_missing': 0,
            'delisted_api_status': 0,
            'delisted_csv': 0,
            'total': len(assets)
        }
        
        updates = []
        now = int(time.time())
        
        for asset_id, symbol in assets:
            is_active = True
            status = 'online'
            
            # Check CSV Blacklist
            # Handle symbol variants (EOS/USD vs EOS)
            base_sym = symbol.split('/')[0] if '/' in symbol else symbol
            
            if symbol in delisted_csv or base_sym in delisted_csv:
                is_active = False
                status = 'delisted_csv'
                stats['delisted_csv'] += 1
            
            # Check API presence
            elif symbol not in kraken_lookup:
                # Try simple variants
                variants = [
                    symbol,
                    symbol.replace('/', ''),
                    symbol.replace('BTC', 'XBT'),
                    symbol.replace('BTC', 'XBT').replace('/', ''),
                ]
                
                found = False
                for v in variants:
                    if v in kraken_lookup:
                        found = True
                        api_info = kraken_lookup[v]
                        # Check API status
                        if api_info.get('status') != 'online':
                            is_active = False
                            status = f"api_{api_info.get('status')}"
                            stats['delisted_api_status'] += 1
                        break
                
                if not found:
                    is_active = False
                    status = 'delisted_missing'
                    stats['delisted_api_missing'] += 1
            
            else:
                # Direct match found
                api_info = kraken_lookup[symbol]
                if api_info.get('status') != 'online':
                    is_active = False
                    status = f"api_{api_info.get('status')}"
                    stats['delisted_api_status'] += 1

            if is_active:
                stats['active'] += 1
                
            updates.append((1 if is_active else 0, status, now, asset_id))
            
            if not is_active:
                logging.info(f"Marking INACTIVE: {symbol} (Reason: {status})")

        # 4. Batch Update
        logging.info("Updating database records...")
        await db.executemany(
            "UPDATE assets SET active = ?, status = ?, last_validated_at = ? WHERE id = ?",
            updates
        )
        await db.commit()
        
        # 5. Report
        logging.info("-" * 40)
        logging.info("VALIDATION COMPLETE")
        logging.info(f"Total Assets: {stats['total']}")
        logging.info(f"Active: {stats['active']}")
        logging.info(f"Delisted (CSV): {stats['delisted_csv']}")
        logging.info(f"Delisted (API Missing): {stats['delisted_api_missing']}")
        logging.info(f"Delisted (API Status): {stats['delisted_api_status']}")
        logging.info("-" * 40)

if __name__ == "__main__":
    asyncio.run(validate_assets())

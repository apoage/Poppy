#!/usr/bin/env python3
"""
Data Download Tool
Fetches historical OHLCV data for top assets from Kraken.
"""

import asyncio
import sys
import os
import logging
import argparse
from typing import List, Tuple

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcts_trader.data.kraken_client import KrakenClient
from mcts_trader.data.database import Database
from mcts_trader.data.fetcher import DataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def get_top_volume_assets(client: KrakenClient, n: int = 64) -> List[Tuple[str, str]]:
    """Get top N assets by USD volume"""
    logging.info("Fetching tickers to determine top volume assets...")
    pairs = await client.get_tradeable_pairs()
    if not pairs:
        logging.error("Failed to fetch pairs.")
        return []
        
    # Chunk ticker request
    all_keys = list(pairs.keys())
    chunk_size = 50
    tickers = {}
    
    for i in range(0, len(all_keys), chunk_size):
        chunk = all_keys[i:i+chunk_size]
        t = await client.get_ticker(chunk)
        if t:
            tickers.update(t)
        await asyncio.sleep(0.1)
        
    scored = []
    for pair, t in tickers.items():
        if pair not in pairs: continue
        
        # Skip weird pairs if needed, but volume sorting usually handles it
        # Filter stablecoin-stablecoin?
        
        try:
            vol_24h = float(t.get('v', [0, 0])[1])
            price = float(t.get('c', [0, 0])[0])
            usd_vol = vol_24h * price
            scored.append((pair, usd_vol))
        except:
            continue
            
    scored.sort(key=lambda x: x[1], reverse=True)
    top_n = scored[:n]
    
    result = []
    for pair, vol in top_n:
        wsname = pairs[pair].get('wsname', pair)
        result.append((pair, wsname))
        
    logging.info(f"Top 3 Assets by Volume: {result[:3]}")
    return result

async def main():
    parser = argparse.ArgumentParser(description='Download Historical Data')
    parser.add_argument('--days', type=int, default=7, help='Days of history to fetch')
    parser.add_argument('--count', type=int, default=64, help='Number of assets')
    parser.add_argument('--file', type=str, help='Path to asset list CSV')
    args = parser.parse_args()
    
    db = Database("trading_data.db")
    await db.connect()
    
    async with KrakenClient() as client:
        fetcher = DataFetcher(client, db)
        
        # 1. Select Assets
        assets = []
        if args.file:
            if not os.path.exists(args.file):
                logging.error(f"File {args.file} not found")
                return
            
            logging.info(f"Loading assets from {args.file}...")
            with open(args.file, 'r') as f:
                symbols = [l.strip() for l in f if l.strip()]
            
            # Resolve to Kraken Pairs
            pairs_info = await client.get_tradeable_pairs()
            
            # Create mapping: wsname -> key, key -> key, altname -> key
            mapping = {}
            for k, v in pairs_info.items():
                mapping[k] = (k, v.get('wsname', k))
                mapping[v.get('wsname', '')] = (k, v.get('wsname', k))
                mapping[v.get('altname', '')] = (k, v.get('wsname', k))
                
            for s in symbols:
                if s in mapping:
                    assets.append(mapping[s])
                else:
                    # Try variants
                    variants = [
                        s.replace('/', ''), 
                        s.replace('BTC', 'XBT'),
                        s.replace('XBT', 'BTC')
                    ]
                    found = False
                    for v in variants:
                        if v in mapping:
                            assets.append(mapping[v])
                            found = True
                            break
                    if not found:
                        logging.warning(f"Could not resolve symbol: {s}")
                        
            # Deduplicate
            assets = list(set(assets))
            logging.info(f"Resolved {len(assets)} valid Kraken pairs from file.")
            
        else:
            assets = await get_top_volume_assets(client, args.count)
            
        if not assets:
            return
            
        logging.info(f"Starting download for {len(assets)} assets ({args.days} days history)...")
        
        # 2. Fetch Data
        await fetcher.fetch_all_assets(assets, days_back=args.days)
        
    await db.close()
    logging.info("Download complete.")

if __name__ == "__main__":
    asyncio.run(main())

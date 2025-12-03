import asyncio
import logging
from datetime import datetime
from mcts_trader.data.kraken_client import KrakenClient
from mcts_trader.data.database import Database
from mcts_trader.data.fetcher import DataFetcher
from mcts_trader.data.stream import KrakenWebSocket

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def main():
    # Initialize
    db = Database("trading_data.db")
    await db.connect()
    
    async with KrakenClient() as client:
        fetcher = DataFetcher(client, db)
        
        # Get top 64 volatile assets
        logging.info("Finding top 64 volatile assets...")
        top_assets = await fetcher.fetch_top_volatile_assets(64)
        
        if not top_assets:
            logging.error("No assets found. Check network or API availability.")
            return

        logging.info(f"Selected assets: {[a[0] for a in top_assets[:10]]}...")
        
        # Fetch historical data (30 days)
        # This will take a while - be patient!
        logging.info("Starting historical data fetch (30 days)...")
        logging.info("This may take several hours - progress saved incrementally")
        
        await fetcher.fetch_all_assets(top_assets, days_back=30)
        
        logging.info("Historical fetch complete!")
        
        # Now start real-time feed
        logging.info("Starting real-time WebSocket feed...")
        
        async def on_ticker(data):
            # Update latest prices in memory/DB
            # In a real system, this would push to a queue or update the Encoder directly
            pass
        
        async def on_ohlc(data):
            # Insert new candles
            # In a real system, this would update the DB and trigger an Agent step
            pass
        
        ws = KrakenWebSocket(on_ticker, on_ohlc)
        await ws.connect()
        await ws.subscribe([a[1] for a in top_assets])  # Use wsnames
        
        await ws.listen()  # Runs forever
    
    await db.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Stopped by user.")

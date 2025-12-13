
import asyncio
from mcts_trader.data.kraken_client import KrakenClient

async def check_wsnames():
    async with KrakenClient() as client:
        pairs = await client.get_tradeable_pairs()
        print(f"Got {len(pairs)} pairs.")
        
        targets = ['XXBTZUSD', 'XXBTZEUR', 'XETHZUSD']
        for t in targets:
            if t in pairs:
                print(f"{t}: wsname = {pairs[t].get('wsname', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(check_wsnames())

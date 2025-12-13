import asyncio
import logging
import sys

sys.path.insert(0, '/home/lukas/Trader')

from mcts_trader.data.kraken_client import KrakenClient

logging.basicConfig(level=logging.INFO)

async def main():
    async with KrakenClient() as client:
        logging.info("Fetching tradeable pairs...")
        pairs = await client.get_tradeable_pairs()
    
    targets = [
        "ETHBTC", "BTCGBP", "XRPBTC", "BTCCAD", "LTCBTC", 
        "XMRBTC", "ADABTC", "EOS/USD", "XLMBTC", "BCHBTC", "BTCCHF"
    ]
    
    print("\n--- TARGET DEBUG ---")
    
    # Dump full info for any pair that "looks like" our targets
    for api_name, info in pairs.items():
        wsname = info.get('wsname', '')
        altname = info.get('altname', '')
        base = info.get('base', '')
        quote = info.get('quote', '')
        
        # Check against targets (loose matching)
        for t in targets:
            # Normalize target: ETHBTC -> ETH, BTC
            t_clean = t.replace('/', '')
            
            # Check if this API pair matches loosely
            if t_clean in wsname.replace('/', '') or t_clean in altname or \
               (t.replace('BTC', 'XBT').replace('/','') in wsname.replace('/','').replace('XBT', 'BTC')):
                
                print(f"Match for {t}:")
                print(f"  API Name: {api_name}")
                print(f"  Altname:  {altname}")
                print(f"  WSname:   {wsname}")
                print(f"  Base/Quote: {base} / {quote}")
                print("-" * 20)

    print("\n--- ALL PAIRS DUMP (Short) ---")
    # Dump all altnames/wsnames to grep manually if needed
    for api_name, info in pairs.items():
        if 'BTC' in api_name or 'XBT' in api_name:
             print(f"{api_name}: alt={info.get('altname')}, ws={info.get('wsname')}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
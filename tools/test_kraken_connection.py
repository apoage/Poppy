
import asyncio
import json
import websockets
import logging

logging.basicConfig(level=logging.INFO)

async def test_subscription(symbol_format, symbols):
    uri = "wss://ws.kraken.com/v2"
    print(f"\n--- Testing format: {symbol_format} with symbols: {symbols} ---")
    
    async with websockets.connect(uri) as websocket:
        subscribe_msg = {
            "method": "subscribe",
            "params": {
                "channel": "ohlc",
                "symbol": symbols,
                "interval": 1
            }
        }
        await websocket.send(json.dumps(subscribe_msg))
        
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"Received: {data}")
                
                if data.get("success"):
                    print("SUCCESS: Subscription accepted.")
                    return True
                if data.get("error"):
                    print(f"FAILURE: {data['error']}")
                    return False
        except asyncio.TimeoutError:
            print("Timeout waiting for response.")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False

async def main():
    # Test 1: API Names (XXBTZUSD)
    # This failed in logs.
    await test_subscription("API Name", ["XXBTZUSD", "XXBTZEUR"])
    
    # Test 2: Slash format (XBT/USD) - typically 'wsname'
    await test_subscription("Slash Format (XBT/USD)", ["XBT/USD", "XBT/EUR"])
    
    # Test 3: Clean Slash (XBTUSD)
    await test_subscription("Clean Format (XBTUSD)", ["XBTUSD", "XBTEUR"])
    
    # Test 4: Common (BTC/USD)
    await test_subscription("Common Format (BTC/USD)", ["BTC/USD", "BTC/EUR"])

if __name__ == "__main__":
    asyncio.run(main())

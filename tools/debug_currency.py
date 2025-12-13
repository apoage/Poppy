from mcts_trader.agent.portfolio import extract_quote_currency, get_quote_to_usd_rate

print("Debug Currency Extraction:")
symbol = "USDTEUR"
quote = extract_quote_currency(symbol)
print(f"Symbol: {symbol} -> Quote: {quote}")

print(f"\nRate for {quote}:")
rate = get_quote_to_usd_rate(quote, {'EURUSD': 1.05})
print(f"Rate: {rate}")

print(f"\nRate for USDTEUR (simulating the error):")
rate_err = get_quote_to_usd_rate("USDTEUR", {{}})
print(f"Rate: {rate_err}")

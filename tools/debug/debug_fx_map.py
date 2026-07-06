import sqlite3
import pandas as pd

DB_PATH = "trading_data.db"
conn = sqlite3.connect(DB_PATH)

# Fetch all assets
df = pd.read_sql_query("SELECT id, symbol FROM assets", conn)

# Mimic the mapping logic
id_to_symbol = {}
for _, row in df.iterrows():
    aid = row['id']
    raw_sym = row['symbol']
    
    # Simple normalization check
    clean_sym = raw_sym.strip().strip('/')
    
    # Logic from script:
    s = clean_sym.replace('/', '').replace('-', '')
    if 'XBT' in s:
        s = s.replace('XBT', 'BTC')
        
    base, quote = None, None
    if s.endswith('USD'):
        quote = 'USD'
        base = s[:-3]
    elif s.endswith('EUR'):
        quote = 'EUR'
        base = s[:-3]
        
    if base and quote:
        # Common Kraken cleanups
        if len(base) == 4 and base.startswith('X'):
                base = base[1:] 
        if base == 'XXBT': base = 'BTC'
        if base == 'XETH': base = 'ETH'
        if quote == 'ZUSD': quote = 'USD'
        
        norm_sym = f"{base}/{quote}"
        if 'XBT' in norm_sym:
            norm_sym = norm_sym.replace('XBT', 'BTC')
            
        id_to_symbol[aid] = norm_sym

# Check our targets
xbt_usd = df[df['symbol'] == 'XBT/USD'].iloc[0]
xbt_eur = df[df['symbol'] == 'XBT/EUR'].iloc[0]

print(f"XBT/USD (ID {xbt_usd['id']}) mapped to: {id_to_symbol.get(xbt_usd['id'])}")
print(f"XBT/EUR (ID {xbt_eur['id']}) mapped to: {id_to_symbol.get(xbt_eur['id'])}")

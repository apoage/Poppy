import os

# Define removals
# Pure Forex
FOREX = {'EUR/USD', 'GBP/USD', 'USD/CAD', 'USD/CHF', 'EUR/GBP', '/USDCHF', '/USDCAD', '/EURGBP'}
# Stable/Fiat (boring)
STABLE_FIAT = {'/USDTGBP', '/USDTCHF', '/USDCGBP', '/USDCEUR', 'DAI/USD', 'DAI/EUR'}
# Crypto/Non-USD (optional, but "more krypto" usually implies unique assets, not same asset in EUR)
NON_USD_CRYPTO = {
    'XBT/EUR', 'ETH/EUR', 'XRP/EUR', 'LTC/EUR', 'ADA/EUR', 'BCH/EUR', 'XMR/EUR',
    'SOL/EUR', 'XLM/EUR', 'DOT/EUR', 'XDG/EUR', 'TRX/EUR', 'LINK/EUR', 'ETC/EUR',
    'ATOM/EUR', 'BTCGBP', 'BTCCAD', 'BTCCHF', 'ETHGBP'
}

REMOVE = FOREX.union(STABLE_FIAT).union(NON_USD_CRYPTO)

# Define Additions (Top 2024/2025 coins)
ADDITIONS = [
    'SUI/USD', 'PEPE/USD', 'SHIB/USD', 'APT/USD', 'WIF/USD',
    'FET/USD', 'RNDR/USD', 'SEI/USD', 'NEAR/USD', 'TIA/USD',
    'AVAX/USD', 'INJ/USD', 'OP/USD', 'ARB/USD', 'LDO/USD',
    'FIL/USD', 'ICP/USD', 'STX/USD', 'IMX/USD', 'KAS/USD',
    'UNI/USD', 'AAVE/USD', 'MKR/USD', 'SNX/USD', 'CRV/USD',
    'GMX/USD', 'PENDLE/USD', 'PYTH/USD', 'JUP/USD', 'BONK/USD',
    'FLOKI/USD', 'ONDO/USD', 'ENA/USD', 'WLD/USD', 'ORDI/USD',
    'BLUR/USD', 'MINA/USD', 'QNT/USD', 'HBAR/USD', 'VET/USD'
]

def clean_symbol(s):
    s = s.strip()
    if s.startswith('/'):
        s = s[1:]
    return s

def main():
    if not os.path.exists('datasets/top64_coverage.csv'):
        print("datasets/top64_coverage.csv not found.")
        return

    with open('datasets/top64_coverage.csv', 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
        
    kept = []
    removed_count = 0
    
    for l in lines:
        cleaned = clean_symbol(l)
        
        # Check against blacklist
        if l in REMOVE or cleaned in REMOVE:
            removed_count += 1
            continue
        
        # Heuristic filtering for non-USD pairs
        if cleaned.endswith('/EUR') or cleaned.endswith('EUR'):
             removed_count += 1
             continue
        if cleaned.endswith('/GBP') or cleaned.endswith('GBP'):
             removed_count += 1
             continue
        if cleaned.endswith('/CAD') or cleaned.endswith('CAD'):
             removed_count += 1
             continue
        if cleaned.endswith('/CHF') or cleaned.endswith('CHF'):
             removed_count += 1
             continue

        kept.append(cleaned)
        
    print(f"Removed {removed_count} pairs.")
    print(f"Kept {len(kept)} pairs.")
    
    # Fill up to 64
    needed = 64 - len(kept)
    if needed > 0:
        print(f"Adding {needed} new crypto pairs...")
        for i in range(needed):
            if i < len(ADDITIONS):
                kept.append(ADDITIONS[i])
            else:
                print("Warning: Ran out of new additions!")
                
    # Deduplicate
    kept = sorted(list(set(kept)))
    
    # Write back
    with open('datasets/top64_coverage.csv', 'w') as f:
        for k in kept:
            f.write(k + '\n')
            
    print("Done. File updated.")

if __name__ == '__main__':
    main()

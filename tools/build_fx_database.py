import sqlite3
import pandas as pd
import numpy as np
import logging
import time
import os
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/fx_builder.log")
    ]
)

DB_PATH = "trading_data.db"
LOG_DIR = "logs"

# Bridge Assets config
BRIDGE_ASSETS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA']
BRIDGE_WEIGHTS = {
    'BTC': 1.0,
    'ETH': 1.0,
    'SOL': 0.7,
    'XRP': 0.5,
    'ADA': 0.5
}
CONSENSUS_THRESHOLD = 0.001  # 0.1%

class FxDatabaseBuilder:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self.kraken_pairs = {}
        
        # Ensure log dir exists
        os.makedirs(LOG_DIR, exist_ok=True)

    def connect(self):
        logging.info(f"Connecting to database: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        
    def fetch_kraken_pairs(self):
        """Fetch official asset pairs from Kraken API for validation."""
        logging.info("Fetching official asset pairs from Kraken API...")
        try:
            resp = requests.get('https://api.kraken.com/0/public/AssetPairs', timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if data.get('error'):
                logging.error(f"Kraken API Error: {data['error']}")
                return

            self.kraken_pairs = data.get('result', {})
            logging.info(f"Successfully fetched {len(self.kraken_pairs)} pairs from Kraken.")
            
        except Exception as e:
            logging.error(f"Failed to fetch Kraken pairs: {e}")
            logging.warning("Proceeding with heuristic parsing only (risk of inaccurate symbols).")

    def create_table(self):
        logging.info("Ensuring fx_rates table exists...")
        query = """
        CREATE TABLE IF NOT EXISTS fx_rates (
            timestamp INTEGER,
            symbol TEXT,       -- e.g., 'EUR', 'GBP', 'USDT'
            rate REAL,         -- The price in USD
            source TEXT,       -- 'DIRECT', 'CONSENSUS', 'SINGLE_BRIDGE'
            PRIMARY KEY (timestamp, symbol)
        );
        """
        self.conn.execute(query)
        self.conn.commit()

    def get_assets_map(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        """
        Returns:
            id_to_symbol: {1: 'BTC/USD', ...} (Normalized)
            symbol_to_id: {'BTC/USD': 1, ...}
        """
        # Get data counts to resolve conflicts
        logging.info("Fetching asset data counts...")
        counts_df = pd.read_sql_query("SELECT asset_id, COUNT(*) as cnt FROM ohlcv GROUP BY asset_id", self.conn)
        counts = dict(zip(counts_df['asset_id'], counts_df['cnt']))
        
        query = "SELECT id, symbol FROM assets"
        df = pd.read_sql_query(query, self.conn)
        
        id_to_symbol = {}
        symbol_to_id = {}
        
        # Build lookup maps from Kraken Data
        # Map 1: altname -> wsname (e.g. XBTUSD -> XBT/USD - wait, wsname is XBT/USD usually)
        # Map 2: name (key) -> wsname (e.g. XXBTZUSD -> XBT/USD)
        
        kraken_lookup = {}
        for k, v in self.kraken_pairs.items():
            wsname = v.get('wsname')
            altname = v.get('altname')
            if wsname:
                kraken_lookup[k] = wsname
                if altname:
                    kraken_lookup[altname] = wsname
        
        for _, row in df.iterrows():
            aid = row['id']
            raw_sym = row['symbol']
            
            # 1. Clean up local symbol
            clean_sym = raw_sym.strip().strip('/')
            
            norm_sym = None
            
            # 2. Try direct lookup in Kraken map
            if clean_sym in kraken_lookup:
                norm_sym = kraken_lookup[clean_sym]
            
            # 3. If not found, try to reconstruct standard pair
            # (e.g. local might be BTC/USD, Kraken has XBT/USD)
            if not norm_sym:
                # Try simple XBT swap
                if 'BTC' in clean_sym:
                    variant = clean_sym.replace('BTC', 'XBT')
                    if variant in kraken_lookup:
                        norm_sym = kraken_lookup[variant]
                        
            # 4. If still not found, try heuristics (as fallback for old data)
            if not norm_sym:
                # Heuristic logic from before...
                # Remove generic separators
                s = clean_sym.replace('/', '').replace('-', '')
                
                # Replace XBT with BTC for our local normalized form
                if 'XBT' in s:
                    s = s.replace('XBT', 'BTC')

                base = None
                quote = None
                
                if s.endswith('USD'):
                    quote = 'USD'
                    base = s[:-3]
                elif s.endswith('EUR'):
                    quote = 'EUR'
                    base = s[:-3]
                elif s.endswith('GBP'):
                    quote = 'GBP'
                    base = s[:-3]
                elif s.endswith('USDT'):
                    quote = 'USDT'
                    base = s[:-4]
                elif s.endswith('AUD'):
                    quote = 'AUD'
                    base = s[:-3]
                elif s.endswith('CAD'):
                    quote = 'CAD'
                    base = s[:-3]
                elif s.endswith('JPY'):
                    quote = 'JPY'
                    base = s[:-3]
                
                if base:
                    # Common Kraken cleanups
                    if len(base) == 4 and base.startswith('X'):
                         base = base[1:] 
                    
                    if base == 'XXBT': base = 'BTC'
                    if base == 'XETH': base = 'ETH'
                    
                    # Quote cleanups
                    if quote == 'ZUSD': quote = 'USD'

                if base and quote:
                    norm_sym = f"{base}/{quote}"
                
            if norm_sym:
                # Standardize to BTC (Kraken uses XBT)
                if 'XBT' in norm_sym:
                    norm_sym = norm_sym.replace('XBT', 'BTC')
                    
                # Conflict Resolution
                if norm_sym in symbol_to_id:
                    existing_id = symbol_to_id[norm_sym]
                    cnt_existing = counts.get(existing_id, 0)
                    cnt_new = counts.get(aid, 0)
                    
                    if cnt_new > cnt_existing:
                        # New one is better
                        # Remove old mapping from id_to_symbol to be clean
                        # (Optional, but good for consistency)
                        if existing_id in id_to_symbol and id_to_symbol[existing_id] == norm_sym:
                             del id_to_symbol[existing_id]
                             
                        id_to_symbol[aid] = norm_sym
                        symbol_to_id[norm_sym] = aid
                        logging.info(f"Conflict resolution for {norm_sym}: Replacing {existing_id} ({cnt_existing}) with {aid} ({cnt_new})")
                    else:
                        logging.debug(f"Conflict resolution for {norm_sym}: Keeping {existing_id} ({cnt_existing}) over {aid} ({cnt_new})")
                        # Do NOT add to id_to_symbol if we don't map it to this symbol
                        # Actually we can map ID->Symbol, just not Symbol->ID
                        pass
                else:
                    id_to_symbol[aid] = norm_sym
                    symbol_to_id[norm_sym] = aid
            else:
                logging.warning(f"Could not parse or verify symbol: {raw_sym}, ignoring.")
            
        return id_to_symbol, symbol_to_id

    def load_ohlc_data(self, asset_ids: List[int]) -> pd.DataFrame:
        """
        Load CLOSE prices for given assets.
        Returns DataFrame with index=timestamp, columns=asset_id (values=close)
        """
        if not asset_ids:
            return pd.DataFrame()
            
        logging.info(f"Loading OHLCV data for {len(asset_ids)} assets...")
        
        # Batch load to avoid massive query string?
        # Actually, let's just do WHERE asset_id IN (...)
        # If too many, split.
        
        chunks = [asset_ids[i:i + 100] for i in range(0, len(asset_ids), 100)]
        all_dfs = []
        
        for chunk in chunks:
            ids_str = ','.join(map(str, chunk))
            query = f"""
            SELECT timestamp, asset_id, close
            FROM ohlcv
            WHERE asset_id IN ({ids_str})
            """
            df_chunk = pd.read_sql_query(query, self.conn)
            all_dfs.append(df_chunk)
            
        if not all_dfs:
            return pd.DataFrame()
            
        full_df = pd.concat(all_dfs)
        
        logging.info(f"Pivoting {len(full_df)} rows...")
        pivot_df = full_df.pivot(index='timestamp', columns='asset_id', values='close')
        pivot_df.sort_index(inplace=True)
        
        # Forward fill to handle misaligned timestamps (e.g. sparse trading)
        # Limit to 1440 minutes (24h) to handle early sparse data
        logging.info("Forward-filling price data (limit=1440)...")
        pivot_df.ffill(limit=1440, inplace=True)
        
        return pivot_df

    def calculate_consensus_rate(self, candidates: List[Dict]) -> Tuple[float, str]:
        """
        candidates: List of {'rate': float, 'weight': float, 'bridge': str}
        Returns: (consensus_rate, source_str)
        """
        if not candidates:
            return None, "NONE"
            
        if len(candidates) == 1:
            return candidates[0]['rate'], "SINGLE_BRIDGE"
            
        # Sort by rate
        candidates.sort(key=lambda x: x['rate'])
        
        clusters = []
        current_cluster = [candidates[0]]
        
        for i in range(1, len(candidates)):
            curr = candidates[i]
            prev = candidates[i-1]
            
            # Check % difference
            diff = abs(curr['rate'] - prev['rate']) / prev['rate']
            
            if diff <= CONSENSUS_THRESHOLD:
                current_cluster.append(curr)
            else:
                clusters.append(current_cluster)
                current_cluster = [curr]
        clusters.append(current_cluster)
        
        # Score clusters
        best_cluster = None
        best_score = -1.0
        
        for cluster in clusters:
            score = sum(c['weight'] for c in cluster)
            if score > best_score:
                best_score = score
                best_cluster = cluster
                
        # Calculate weighted average of winning cluster
        total_weight = sum(c['weight'] for c in best_cluster)
        weighted_sum = sum(c['rate'] * c['weight'] for c in best_cluster)
        
        final_rate = weighted_sum / total_weight
        
        # Determine source string detail (optional, simplified to CONSENSUS)
        return final_rate, "CONSENSUS"

    def run(self):
        self.connect()
        self.fetch_kraken_pairs()  # Validation step
        self.create_table()
        
        id_to_symbol, symbol_to_id = self.get_assets_map()
        logging.info(f"Mapped {len(id_to_symbol)} assets.")
        
        # Define an explicit list of target currencies we care about.
        # These are currencies for which we want to derive their rate in USD.
        # Stablecoins like USDT are included as per instructions, but USDC/DAI are typically 1:1 with USD.
        # ETH/BTC are bridge assets, not target currencies for FX calculation.
        EXPLICIT_TARGET_CURRENCIES = {'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'USDT'} # Add others if needed like AED
        
        # Identify Targets from the DB that are in our explicit list
        potential_targets = set()
        for sym in symbol_to_id.keys():
            if '/' in sym:
                parts = sym.split('/')
                quote = parts[1]
                # If the quote is not USD and is in our explicit list, it's a target
                if quote != 'USD' and quote in EXPLICIT_TARGET_CURRENCIES:
                    potential_targets.add(quote)
                # Also, if the base is in our explicit list and it's paired with USD, it could be a direct pair
                if parts[0] in EXPLICIT_TARGET_CURRENCIES and quote == 'USD':
                    potential_targets.add(parts[0])

        # Remove USD itself from targets, as its rate is always 1.0 and doesn't need derivation.
        if 'USD' in potential_targets:
            potential_targets.remove('USD')

        # Filter out targets that are actually bridge assets or other non-FX stablecoins
        # This double-checks against potential misclassification or inclusion from symbols
        potential_targets = {target for target in potential_targets if target not in BRIDGE_ASSETS and target != 'USDC' and target != 'DAI'}
        
        # Add any explicit targets that might not have been caught by the symbol parsing (e.g., if no pair exists in DB for it)
        potential_targets.update(EXPLICIT_TARGET_CURRENCIES)
        # Ensure no bridge assets or unwanted stablecoins are left in targets.
        potential_targets = {target for target in potential_targets if target not in BRIDGE_ASSETS and target != 'USDC' and target != 'DAI'}
        
        logging.info(f"Final potential target currencies: {potential_targets}")

        # Check direct pairs (e.g., EUR/USD) and map them to asset IDs
        direct_pairs = {} # Target Currency -> AssetID of its direct /USD pair
        for target in potential_targets:
            direct_sym = f"{target}/USD"
            if direct_sym in symbol_to_id:
                direct_pairs[target] = symbol_to_id[direct_sym]
        
        # Identify relevant asset IDs to load
        # 1. Direct pairs: Target/USD
        # 2. Bridge/USD pairs (BTC/USD, ETH/USD...)
        # 3. Bridge/Target pairs (BTC/EUR, ETH/GBP...)
        
        relevant_ids = set()
        
        # Direct
        for aid in direct_pairs.values():
            relevant_ids.add(aid)
            
        # Bridges
        bridge_usd_ids = {} # Bridge -> AssetID
        for bridge in BRIDGE_ASSETS:
            sym = f"{bridge}/USD"
            if sym in symbol_to_id:
                aid = symbol_to_id[sym]
                bridge_usd_ids[bridge] = aid
                relevant_ids.add(aid)
                
        # Bridge/Target
        bridge_target_ids = {} # (Bridge, Target) -> AssetID
        for bridge in BRIDGE_ASSETS:
            for target in potential_targets:
                sym = f"{bridge}/{target}"
                if sym in symbol_to_id:
                    aid = symbol_to_id[sym]
                    bridge_target_ids[(bridge, target)] = aid
                    relevant_ids.add(aid)
        
        
        logging.info(f"Identified {len(relevant_ids)} relevant assets for FX calculation.")
        
        # Load Data
        df = self.load_ohlc_data(list(relevant_ids))
        if df.empty:
            logging.error("No data found!")
            return
            
        logging.info(f"Loaded data matrix: {df.shape}")
        
        # Processing
        fx_records = []
        conflicts = []
        gaps = []
        
        timestamps = df.index
        
        # Iterate targets
        for target in potential_targets:
            logging.info(f"Processing target: {target}")
            
            # Prepare columns for vectorization (or iteration)
            # Iteration is safer for logic complexity
            
            # Check direct
            direct_aid = direct_pairs.get(target)
            
            target_data = [] # List of (timestamp, rate, source)
            
            # Pre-fetch columns to avoid lookup overhead in loop
            # Direct Series
            s_direct = df[direct_aid] if direct_aid and direct_aid in df.columns else None
            
            # Bridge Series
            # list of (bridge_name, s_bridge_usd, s_bridge_target, weight)
            bridges = []
            for bridge in BRIDGE_ASSETS:
                aid_usd = bridge_usd_ids.get(bridge)
                aid_tgt = bridge_target_ids.get((bridge, target))
                
                if aid_usd and aid_tgt and aid_usd in df.columns and aid_tgt in df.columns:
                    bridges.append({
                        'name': bridge,
                        'usd': df[aid_usd],
                        'tgt': df[aid_tgt],
                        'weight': BRIDGE_WEIGHTS.get(bridge, 0.5)
                    })
            
            if s_direct is None and not bridges:
                logging.warning(f"No data sources for {target}")
                continue
                
            # Iterate timestamps
            # (Optimization: We could do this vectorized but "Consensus" logic is tricky. 
            # Let's do a semi-vectorized approach or just loop if fast enough.)
            # 200k rows... Python loop might take 10-20s per target. Acceptable.
            
            prev_rate = None
            
            debug_failures = 0
            for ts in timestamps:
                # 1. Direct
                rate = None
                source = None
                
                if s_direct is not None:
                    val = s_direct.at[ts] # access scalar
                    if not np.isnan(val) and val > 0:
                        rate = val
                        source = "DIRECT"
                
                # 2. Bridges (if no direct)
                if rate is None and bridges:
                    candidates = []
                    for b in bridges:
                        p_usd = b['usd'].at[ts]
                        p_tgt = b['tgt'].at[ts]
                        
                        if not np.isnan(p_usd) and not np.isnan(p_tgt) and p_tgt > 0:
                            # Rate = Price(Bridge/USD) / Price(Bridge/Target)
                            # e.g. BTC/USD ($100,000) / BTC/EUR (€90,000) = 1.11 $/€
                            cand_rate = p_usd / p_tgt
                            candidates.append({
                                'rate': cand_rate,
                                'weight': b['weight'],
                                'bridge': b['name']
                            })
                            
                    if candidates:
                        rate, source = self.calculate_consensus_rate(candidates)
                        
                        # Log conflict check
                        # If candidates vary wildly?
                        # (Not fully implemented here to save complexity, relying on consensus)
                
                if rate is not None:
                    fx_records.append((ts, target, rate, source))
                    prev_rate = rate
                else:
                    # Gap
                    pass
                    
        # Bulk Insert
        if fx_records:
            logging.info(f"Inserting {len(fx_records)} FX records...")
            # Use chunks for insert
            chunk_size = 50000
            for i in range(0, len(fx_records), chunk_size):
                chunk = fx_records[i:i+chunk_size]
                self.conn.executemany(
                    "INSERT OR REPLACE INTO fx_rates (timestamp, symbol, rate, source) VALUES (?, ?, ?, ?)",
                    chunk
                )
                self.conn.commit()
                logging.info(f"Committed chunk {i // chunk_size + 1}")
        
        logging.info("Done.")

if __name__ == "__main__":
    builder = FxDatabaseBuilder()
    builder.run()
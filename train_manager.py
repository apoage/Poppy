#!/usr/bin/env python3
"""
Training Manager
Orchestrates multiple training experiments sequentially.
Supports dynamic asset selection via Kraken API or Dataset files.
"""

import subprocess
import time
import os
import sys
import site
import logging
import argparse
import json
import urllib.request
from typing import List, Dict, Optional
from pathlib import Path

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [MANAGER] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'train_manager.log')),
        logging.StreamHandler()
    ]
)

# --- CUDA Library Path Setup (for CuDNN 9.8+ from pip packages) ---
def setup_cuda_env():
    """Build environment with correct CUDA library paths from pip-installed nvidia packages"""
    env = os.environ.copy()

    try:
        site_packages = site.getsitepackages()[0]
        nvidia_libs = [
            "cudnn", "cublas", "cuda_nvrtc", "cuda_runtime",
            "cuda_cupti", "cufft", "curand", "cusolver", "cusparse", "nccl"
        ]

        lib_paths = []
        for lib in nvidia_libs:
            lib_path = os.path.join(site_packages, "nvidia", lib, "lib")
            if os.path.exists(lib_path):
                lib_paths.append(lib_path)

        if lib_paths:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(lib_paths) + (":" + existing if existing else "")
            logging.info(f"✅ CUDA libs configured from venv ({len(lib_paths)} paths)")
    except Exception as e:
        logging.warning(f"Could not setup CUDA env: {e}")

    return env

CUDA_ENV = setup_cuda_env()

# --- Asset Selection Helpers ---

def fetch_json(url):
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode())

def fetch_kraken_asset_pairs():
    """Get current tradeable pairs from Kraken"""
    try:
        data = fetch_json('https://api.kraken.com/0/public/AssetPairs')
        return data.get('result', {})
    except Exception as e:
        logging.error(f"Failed to fetch AssetPairs: {e}")
        return {}

def fetch_top_pairs_from_kraken(strategy='volume', count=64, days=7):
    """
    Auto-select top pairs by strategy
    """
    logging.info(f"Auto-selecting top {count} pairs by {strategy}...")
    
    if strategy.startswith('base:'):
        # All pairs of a base asset
        base = strategy.split(':')[1].upper()
        pairs = fetch_kraken_asset_pairs()
        matching = []
        for p, info in pairs.items():
            # Check wsname (e.g. BTC/USD) or altname (XBTUSD)
            wsname = info.get('wsname', '')
            if wsname.startswith(f"{base}/") or f"/{base}" in wsname:
                matching.append(p)
        return matching[:count]

    elif strategy == 'volume':
        # Get volume data from Kraken Ticker endpoint
        try:
            data = fetch_json('https://api.kraken.com/0/public/Ticker')
            tickers = data.get('result', {})
            
            # Sort by 24h volume (USD equivalent approx)
            # Ticker: v = volume array [today, 24h]
            # We need Quote volume really, but Kraken gives Base volume.
            # Approximation: Base Volume * Last Price
            
            scored_pairs = []
            for pair, info in tickers.items():
                vol_24h = float(info.get('v', [0, 0])[1])
                price = float(info.get('c', [0, 0])[0])
                vol_usd = vol_24h * price
                scored_pairs.append((pair, vol_usd))
                
            # Sort descending
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return [p[0] for p in scored_pairs[:count]]
            
        except Exception as e:
            logging.error(f"Failed to fetch Ticker: {e}")
            return []

    else:
        logging.error(f"Unknown strategy: {strategy}")
        return []

def load_pairs_from_csv(filepath):
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def load_pairs_from_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
        return [a['symbol'] for a in data['assets']]

def is_forex_pair(symbol):
    """Check if a symbol is a forex (fiat/fiat) pair"""
    fiat = ['EUR', 'USD', 'GBP', 'CHF', 'CAD', 'AUD', 'JPY', 'NZD', 'SEK', 'NOK', 'DKK']
    # Handle formats: EUR/USD, EURUSD, /EURGBP
    clean_symbol = symbol.replace('/', ' ').strip()
    parts = clean_symbol.split()

    if len(parts) == 1:
        # Format like EURUSD - try to split by known fiat codes
        for f in fiat:
            if clean_symbol.startswith(f):
                remainder = clean_symbol[len(f):]
                if remainder in fiat:
                    return True
    elif len(parts) == 2:
        return parts[0] in fiat and parts[1] in fiat

    return False

def validate_pairs(pairs, strict=True):
    """Quick validation against Kraken API"""
    if not strict:
        # For historical training, trust the database - no API validation
        logging.info(f"✅ Using {len(pairs)} pairs from database (no API validation)")
        return pairs

    logging.info(f"Validating {len(pairs)} pairs against Kraken...")
    kraken_pairs = fetch_kraken_asset_pairs()

    # Build lookup with multiple naming variants
    kraken_lookup = {}
    for key, info in kraken_pairs.items():
        kraken_lookup[key] = info
        # Also index by wsname and altname
        if 'wsname' in info:
            kraken_lookup[info['wsname']] = info
        if 'altname' in info:
            kraken_lookup[info['altname']] = info

    valid = []
    invalid = []

    for pair in pairs:
        # Try multiple variants
        variants = [
            pair,
            pair.replace('/', ''),
            pair.replace('XBT', 'BTC'),
            pair.replace('BTC', 'XBT'),
            pair.replace('/', '').replace('XBT', 'BTC'),
            pair.replace('/', '').replace('BTC', 'XBT'),
            'X' + pair.replace('/', 'Z') if '/' in pair else pair,  # XETHZUSD format
        ]

        found = False
        for v in variants:
            if v in kraken_lookup:
                info = kraken_lookup[v]
                if info.get('status') == 'online':
                    valid.append(pair)  # Keep original DB name
                    found = True
                    break

        if not found:
            invalid.append(pair)

    if invalid:
        logging.warning(f"Skipping {len(invalid)} invalid pairs: {invalid[:5]}...")

    logging.info(f"✅ Selected {len(valid)} valid pairs")
    return valid

def create_training_config(pairs, experiment_name='auto'):
    """Create training config file"""
    config = {
        'experiment_name': experiment_name,
        'n_assets': len(pairs),
        'asset_symbols': pairs,
        'generated_at': str(time.time())
    }
    
    config_path = os.path.join('config', 'current_training.json')
    os.makedirs('config', exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    logging.info(f"📝 Config written to: {config_path}")
    return config_path

# --- Experiment Runner ---

def run_experiment(config: Dict, training_config_path: Optional[str] = None):
    """Run a single experiment via train_jax_fast.py"""
    name = config['name']
    mode = config.get('mode', 'MARGIN')
    fee = config.get('fee', 26)
    epochs = config.get('epochs', 100)
    steps = config.get('steps', 200)

    cmd = [
        "python", "train_jax_fast.py",
        "--name", name,
        "--mode", mode,
        "--fee", str(fee),
        "--epochs", str(epochs),
        "--steps", str(steps)
    ]

    if training_config_path:
        cmd.extend(["--config", training_config_path])

    logging.info(f"Starting Experiment: {name} (Mode: {mode}, Fee: {fee}, Epochs: {epochs}, Steps: {steps})")
    logging.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    # Run process and stream output (with CUDA env for proper CuDNN)
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=CUDA_ENV
    ) as proc:
        for line in proc.stdout:
            print(f"[{name}] {line}", end='') # Prefix output with experiment name
            
        return_code = proc.wait()
        
    duration = time.time() - start_time
    
    if return_code == 0:
        logging.info(f"Experiment {name} COMPLETED in {duration/60:.1f} minutes.")
    else:
        logging.error(f"Experiment {name} FAILED with code {return_code}.")

def main():
    parser = argparse.ArgumentParser(description='Training Manager')
    
    # Asset Selection Args
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pairs', help='CSV file with pair list')
    group.add_argument('--dataset', help='JSON dataset file')
    group.add_argument('--auto-select', help='Auto-select pairs (volume, base:BTC)')
    group.add_argument('--random-subset', type=int, help='Select N random assets from DB')
    
    parser.add_argument('--count', type=int, default=64, help='Number of pairs (for auto-select)')
    parser.add_argument('--epochs', type=int, default=20000, help='Training epochs (default: 20000)')
    parser.add_argument('--steps', type=int, default=200, help='Steps per episode (default: 200, try 1440 for daily cycles)')
    parser.add_argument('--min-coverage', type=float, default=0.1, help='Minimum asset coverage (0.0-1.0) for valid timestamps')
    parser.add_argument('--exclude-forex', action='store_true', help='Exclude forex (fiat/fiat) pairs from selection')
    parser.add_argument('--date-start', help='Start date for data selection (YYYY-MM-DD or unix timestamp)')
    parser.add_argument('--date-end', help='End date for data selection (YYYY-MM-DD or unix timestamp)')
    parser.add_argument('--min-candles', type=int, default=5000, help='Minimum candles required per asset (default: 5000 ≈ 3.5 days)')
    parser.add_argument('--no-validate', action='store_true', help='Skip Kraken validation')
    parser.add_argument('--dry-run', action='store_true', help='Show plan and exit')
    
    args = parser.parse_args()
    
    # 1. Asset Selection
    pairs = []
    if args.pairs:
        pairs = load_pairs_from_csv(args.pairs)
    elif args.dataset:
        pairs = load_pairs_from_json(args.dataset)
    elif args.auto_select:
        pairs = fetch_top_pairs_from_kraken(args.auto_select, args.count)
    elif args.random_subset:
        import sqlite3
        import random
        from datetime import datetime

        # Parse date range if provided
        date_filter_sql = ""
        params = []

        if args.date_start or args.date_end:
            # Convert dates to unix timestamps
            if args.date_start:
                try:
                    start_ts = int(args.date_start) if args.date_start.isdigit() else int(datetime.strptime(args.date_start, '%Y-%m-%d').timestamp())
                except:
                    logging.error(f"Invalid --date-start format: {args.date_start}")
                    return
            else:
                start_ts = 0

            if args.date_end:
                try:
                    end_ts = int(args.date_end) if args.date_end.isdigit() else int(datetime.strptime(args.date_end, '%Y-%m-%d').timestamp())
                except:
                    logging.error(f"Invalid --date-end format: {args.date_end}")
                    return
            else:
                end_ts = 9999999999

            date_filter_sql = " AND o.timestamp >= ? AND o.timestamp <= ?"
            params = [start_ts, end_ts]
            logging.info(f"Filtering data: {datetime.fromtimestamp(start_ts)} to {datetime.fromtimestamp(end_ts)}")

        # Select random assets that have substantial data
        conn = sqlite3.connect('trading_data.db')

        # Build query with optional date filter
        query = f"""
            SELECT a.symbol, COUNT(*) as candle_count
            FROM assets a
            JOIN ohlcv o ON a.id=o.asset_id
            WHERE 1=1 {date_filter_sql}
            GROUP BY a.id
            HAVING COUNT(*) >= ?
        """
        params.append(args.min_candles)

        cursor = conn.execute(query, params)
        candidates = [r[0] for r in cursor.fetchall()]
        conn.close()

        # Filter out forex pairs if requested
        if args.exclude_forex:
            before_count = len(candidates)
            candidates = [p for p in candidates if not is_forex_pair(p)]
            logging.info(f"Excluded {before_count - len(candidates)} forex pairs. Remaining: {len(candidates)}")

        if len(candidates) < args.random_subset:
            logging.warning(f"Only found {len(candidates)} valid assets, requested {args.random_subset}. Using all.")
            pairs = candidates
        else:
            pairs = random.sample(candidates, args.random_subset)
        logging.info(f"✅ Selected {len(pairs)} random assets from {len(candidates)} candidates.")
        if pairs:
            logging.info(f"Sample assets: {pairs[:10]}")
        
    training_config_path = None
    if pairs:
        if not args.no_validate:
            pairs = validate_pairs(pairs)
            
        if not pairs:
            logging.error("No valid pairs found. Exiting.")
            return
            
        # Create dynamic config
        config = {
            'experiment_name': 'dynamic_set',
            'n_assets': len(pairs),
            'asset_symbols': pairs,
            'min_coverage': args.min_coverage, # Pass min_coverage to config
            'generated_at': str(time.time())
        }
        
        config_path = os.path.join('config', 'current_training.json')
        os.makedirs('config', exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logging.info(f"📝 Config written to: {config_path}")
        training_config_path = config_path
        
    if args.dry_run:
        logging.info("🏁 DRY RUN COMPLETE. Would train with:")
        logging.info(f"Pairs: {len(pairs)}")
        logging.info(f"Config: {training_config_path}")
        logging.info(f"First 5 pairs: {pairs[:5]}")
        return

    # 2. Experiment Loop
    # Define experiments using CLI args
    experiments = [
        # 1. Spot Baseline (High Fee)
        {
            "name": "spot_baseline_v1",
            "mode": "SPOT",
            "fee": 26,
            "epochs": args.epochs,
            "steps": args.steps
        },
        # 2. Margin Standard (High Fee)
        {
            "name": "margin_standard_v1",
            "mode": "MARGIN",
            "fee": 26,
            "epochs": args.epochs,
            "steps": args.steps
        }
    ]
    
    logging.info(f"Loaded {len(experiments)} experiments.")
    
    for i, exp in enumerate(experiments):
        logging.info(f"--- Running {i+1}/{len(experiments)} ---")
        run_experiment(exp, training_config_path)
        time.sleep(5) # Cooldown between runs

    logging.info("All experiments finished.")

if __name__ == "__main__":
    main()
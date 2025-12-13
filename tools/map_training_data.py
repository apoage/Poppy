#!/usr/bin/env python3
"""
Training Data Mapper - Rainbow Table Generator
Scans database to identify valid training periods and asset combinations.
"""

import sqlite3
import json
import os
import sys
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# Add parent directory to path to import train_manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train_manager import is_forex_pair

def find_initial_threshold(db_path: str, min_assets: int = 64) -> int:
    """
    Scan forward from earliest data to find first timestamp where â≥min_assets exist.

    Args:
        db_path: Path to trading_data.db
        min_assets: Minimum concurrent assets required (default 64)

    Returns:
        Unix timestamp of first valid starting point
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)

    # Get earliest timestamp
    row = conn.execute('SELECT MIN(timestamp) FROM ohlcv').fetchone()
    if not row or row[0] is None:
        conn.close()
        raise ValueError("Database is empty (no ohlcv data)")
        
    min_ts = row[0]

    print(f"Scanning from {datetime.fromtimestamp(min_ts)}...")

    # Scan forward in 1-day increments
    current_ts = min_ts
    row_max = conn.execute('SELECT MAX(timestamp) FROM ohlcv').fetchone()
    max_ts = row_max[0] if row_max else min_ts
    
    one_day = 24 * 60 * 60

    last_scanned_year = None

    while current_ts <= max_ts:
        # Count distinct assets with data at this timestamp (±12 hour window)
        window_start = current_ts - 43200
        window_end = current_ts + 43200

        # We need to filter forex pairs here too ideally, but for speed we might do a rough check
        # But wait, find_initial_threshold logic in spec just counts distinct asset_id.
        # It doesn't check is_forex_pair. 
        # However, map_valid_periods DOES check is_forex_pair.
        # If we rely on initial threshold counting forex, we might start too early.
        # Let's improve it slightly by joining with assets table if possible, but spec is specific.
        # Spec code:
        # count = conn.execute('SELECT COUNT(DISTINCT asset_id) ...').fetchone()[0]
        # I will follow the spec for now, but be aware of potential forex inclusion.
        # Actually, let's stick to the spec's SQL to ensure we match expected behavior, 
        # but if we find mostly forex, map_valid_periods will handle it (by filtering and maybe not starting a period).
        
        count = conn.execute('''
            SELECT COUNT(DISTINCT asset_id)
            FROM ohlcv
            WHERE timestamp BETWEEN ? AND ?
        ''', (window_start, window_end)).fetchone()[0]

        ts_dt = datetime.fromtimestamp(current_ts)
        if last_scanned_year != ts_dt.year:
             print(f"  Scanning {ts_dt.year}... (Max assets seen: {count})")
             last_scanned_year = ts_dt.year

        if count >= min_assets:
            # Double check if these are not all forex?
            # The spec doesn't do it here, but does it in map_valid_periods. 
            # I'll stick to the spec.
            conn.close()
            print(f"â✓ Found initial threshold at {datetime.fromtimestamp(current_ts)}")
            print(f"  Available assets (raw): {count}")
            return current_ts

        current_ts += one_day

    conn.close()
    raise ValueError(f"Never found {min_assets} concurrent assets in database")

def map_valid_periods(
    db_path: str,
    start_ts: int,
    min_assets: int = 64,
    min_duration_days: int = 30,
    max_duration_days: int = 180,
    coverage_threshold: float = 0.75,
    tolerance: float = 0.1
) -> List[Dict]:
    """
    Identify continuous periods where sufficient assets exist.

    A period ends when:
    - â≥15 assets disappear (delisted/no data)
    - OR we reach end of database
    - OR max_duration_days is reached (to allow asset refresh)

    Args:
        db_path: Path to database
        start_ts: Starting timestamp (from find_initial_threshold)
        min_assets: Minimum assets required throughout period
        min_duration_days: Minimum period length in days (default 30)
        max_duration_days: Max period length before forcing a split to refresh assets (default 180)
        coverage_threshold: Min fraction of assets that must have data at each timestamp
        tolerance: Allowed drop in asset count relative to min_assets (default 0.1 = 10%)

    Returns:
        List of period dicts with start, end, duration, available_assets
    """
    conn = sqlite3.connect(db_path)

    # Get all timestamps in order
    print("Fetching timestamps...")
    timestamps = [
        row[0] for row in
        conn.execute('SELECT DISTINCT timestamp FROM ohlcv WHERE timestamp >= ? ORDER BY timestamp', (start_ts,))
    ]

    print(f"Analyzing {len(timestamps)} timestamps...")

    periods = []
    current_period_start = None
    current_period_assets = set()
    last_asset_count = 0
    
    # Calculate drop threshold
    drop_threshold = int(min_assets * (1 - tolerance))
    print(f"Logic: Start at {min_assets} assets, allow drop to {drop_threshold} assets, max duration {max_duration_days} days")

    # Optimization: Fetch asset symbols map once
    asset_map = {
        row[0]: row[1] for row in conn.execute('SELECT id, symbol FROM assets')
    }
    
    for i, ts in enumerate(timestamps):
        # Optimization: Only check every 60 timestamps (1 hour) to speed up mapping
        if i % 60 != 0 and i != len(timestamps) - 1:
            continue
            
        # Get assets at this timestamp
        # Look back 24 hours to find assets that have traded recently (handles gaps/maintenance)
        asset_ids = [
            row[0] for row in 
            conn.execute('SELECT DISTINCT asset_id FROM ohlcv WHERE timestamp BETWEEN ? AND ?', (ts - 86400, ts))
        ]
        
        assets_at_ts = set()
        for aid in asset_ids:
            sym = asset_map.get(aid)
            if sym and not is_forex_pair(sym):
                assets_at_ts.add(sym)

        asset_count = len(assets_at_ts)
        
        should_end = False
        end_reason = ""

        # Check existing period constraints
        if current_period_start is not None:
            # Track assets that persist throughout period
            current_period_assets &= assets_at_ts  # Intersection
            
            # Check duration
            curr_duration = (ts - current_period_start) / (24 * 60 * 60)

            # Check if we lost too many assets OR reached max duration
            if len(current_period_assets) < drop_threshold:
                should_end = True
                end_reason = f"Assets dropped to {len(current_period_assets)} < {drop_threshold}"
            elif curr_duration >= max_duration_days:
                should_end = True
                end_reason = f"Max duration {max_duration_days}d reached"
                
            if should_end:
                # End current period
                period_end = timestamps[i - 1] if i > 0 else timestamps[0]
                duration_days = (period_end - current_period_start) / (24 * 60 * 60)

                if duration_days >= min_duration_days:
                    periods.append({
                        'start': current_period_start,
                        'end': period_end,
                        'duration_days': int(duration_days),
                        'available_assets': sorted(list(current_period_assets)),
                        'asset_count': len(current_period_assets)
                    })
                    print(f"  Period saved: {duration_days:.0f} days ({end_reason}), {len(current_period_assets)} persistent assets")
                else:
                    if duration_days > 1:
                         print(f"  Period ended: {duration_days:.1f} days ({end_reason})")

                # Force reset to allow new period to start immediately with FRESH asset set
                current_period_start = None
                current_period_assets = set()
                # Do NOT skip to next iteration, allow logic below to restart immediately at this TS

        # Start new period if we have enough assets and not in a period
        if current_period_start is None and asset_count >= min_assets:
            current_period_start = ts
            current_period_assets = assets_at_ts
            last_asset_count = asset_count
            # print(f"  New period started at {datetime.fromtimestamp(ts)} ({asset_count} assets)")

        # Progress indicator
        if i % 50000 == 0: 
            print(f"  Progress: {i}/{len(timestamps)} ({i*100/len(timestamps):.1f}%)")

    # Save final period if active
    if current_period_start is not None:
        period_end = timestamps[-1]
        duration_days = (period_end - current_period_start) / (24 * 60 * 60)

        if duration_days >= min_duration_days:
            periods.append({
                'start': current_period_start,
                'end': period_end,
                'duration_days': int(duration_days),
                'available_assets': sorted(list(current_period_assets)),
                'asset_count': len(current_period_assets)
            })

    conn.close()

    print(f"\nâ✓ Found {len(periods)} valid periods")
    return periods

def generate_asset_combinations(
    available_assets: List[str],
    subset_size: int = 64,
    max_overlap: float = 0.5,
    min_overlap: float = 0.3
) -> List[List[str]]:
    """
    Generate multiple 64-asset subsets with controlled overlap (30-50% with previous run).

    Args:
        available_assets: All assets available in this period
        subset_size: Size of each subset (default 64)
        max_overlap: Maximum overlap with previous run (default 0.5 = 50%)
        min_overlap: Minimum overlap with previous run (default 0.3 = 30%)

    Returns:
        List of asset subsets (each is a list of 64 symbols)
    """
    if len(available_assets) < subset_size:
        return [available_assets]  # Not enough for even one full subset

    combinations = []
    previous_subset = None

    # Calculate how many combinations we can make
    # Rough estimate: if we have 150 assets, ~35% overlap, we can make ~15 combinations
    # Let's cap at 50 to avoid exploding schedule
    max_combinations = min(50, int(len(available_assets) / subset_size * 3)) # Multiplier 3 for more combos
    # Ensure at least a few if we have enough assets
    if max_combinations < 5 and len(available_assets) > subset_size + 10:
        max_combinations = 5

    print(f"  Generating up to {max_combinations} combinations from {len(available_assets)} assets...")

    for i in range(max_combinations):
        if previous_subset is None:
            # First combination: random selection
            subset = random.sample(available_assets, subset_size)
        else:
            # Subsequent combinations: controlled overlap
            target_overlap_count = random.randint(
                int(subset_size * min_overlap),
                int(subset_size * max_overlap)
            )

            # Keep some from previous
            kept_assets = random.sample(previous_subset, target_overlap_count)

            # Add new assets to reach 64
            remaining_count = subset_size - target_overlap_count
            available_new = [a for a in available_assets if a not in kept_assets]

            if len(available_new) < remaining_count:
                # Can't make more unique combinations with this constraint
                # Try to fill with ANY other assets (relaxed constraint) or break
                other_assets = [a for a in available_assets if a not in kept_assets]
                if len(other_assets) < remaining_count:
                     break # Truly exhausted
                available_new = other_assets

            new_assets = random.sample(available_new, remaining_count)
            subset = kept_assets + new_assets

            # Shuffle to avoid positional bias
            random.shuffle(subset)

        combinations.append(subset)
        previous_subset = subset

    print(f"  Generated {len(combinations)} asset combinations")
    return combinations

def build_training_schedule(db_path: str, output_file: str = 'training_schedule.json'):
    """
    Main function: Build complete training schedule.
    """
    print("="*70)
    print("TRAINING DATA MAPPER - Rainbow Table Generator")
    print("="*70)
    print()

    # Step 1: Find initial threshold
    print("[1/4] Finding initial coverage threshold...")
    try:
        start_ts = find_initial_threshold(db_path, min_assets=64)
    except ValueError as e:
        print(f"Error: {e}")
        return
    print()

    # Step 2: Map valid periods
    print("[2/4] Mapping valid training periods...")
    periods = map_valid_periods(
        db_path,
        start_ts,
        min_assets=64,
        min_duration_days=30,
        max_duration_days=180, # Force re-sampling every 6 months
        coverage_threshold=0.75,
        tolerance=0.1
    )
    print()

    # Step 3: Generate asset combinations for each period
    print("[3/4] Generating asset combinations...")
    for period in periods:
        print(f"Period {datetime.fromtimestamp(period['start'])} - {datetime.fromtimestamp(period['end'])}")

        combinations = generate_asset_combinations(
            period['available_assets'],
            subset_size=64,
            max_overlap=0.5,
            min_overlap=0.3
        )

        # Store combinations as training runs
        period['training_runs'] = []
        for run_id, assets in enumerate(combinations, start=1):
            # Calculate overlap with previous run
            overlap = 0.0
            if run_id > 1:
                prev_assets = set(combinations[run_id - 2])
                curr_assets = set(assets)
                overlap = len(prev_assets & curr_assets) / 64

            period['training_runs'].append({
                'run_id': run_id,
                'assets': assets,
                'overlap_with_previous': round(overlap, 3)
            })

        period['total_training_epochs'] = period['duration_days'] * len(combinations)
        print(f"  → {len(combinations)} runs • {period['duration_days']} days = {period['total_training_epochs']} epochs")
    print()

    # Step 4: Generate summary and save
    print("[4/4] Generating schedule summary...")
    schedule = {
        'generated_at': datetime.now().isoformat(),
        'database': db_path,
        'periods': periods,
        'summary': {
            'total_periods': len(periods),
            'total_unique_runs': sum(len(p['training_runs']) for p in periods),
            'total_training_epochs_available': sum(p['total_training_epochs'] for p in periods),
            'date_range': {
                'start': datetime.fromtimestamp(periods[0]['start']).isoformat() if periods else "N/A",
                'end': datetime.fromtimestamp(periods[-1]['end']).isoformat() if periods else "N/A"
            }
        }
    }

    # Convert timestamps to ISO format for readability
    for period in schedule['periods']:
        period['start_date'] = datetime.fromtimestamp(period['start']).isoformat()
        period['end_date'] = datetime.fromtimestamp(period['end']).isoformat()

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(schedule, f, indent=2)

    print(f"✓ Training schedule saved to: {output_file}")
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Periods found: {schedule['summary']['total_periods']}")
    print(f"Total training runs: {schedule['summary']['total_unique_runs']}")
    print(f"Total epochs available: {schedule['summary']['total_training_epochs_available']:,}")
    print(f"Date range: {schedule['summary']['date_range']['start']} to {schedule['summary']['date_range']['end']}")
    print("="*70)


if __name__ == '__main__':
    build_training_schedule('trading_data.db', 'training_schedule.json')
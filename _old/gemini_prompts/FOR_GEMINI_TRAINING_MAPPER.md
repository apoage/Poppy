# Task: Implement Training Data Rainbow Table Mapper

## Objective

Create a system that pre-computes all valid training windows from the database, generates multiple asset combinations per time period, and produces a comprehensive training schedule. This allows training on the same time periods with different asset sets, multiplying available training data 10-50x while maintaining temporal coherence.

## Background

**Current Problem**:
- 1440-step episodes = 1 day per epoch
- 10,000 epochs needs 27 years of data
- User has ~7 years of data (2018-2025) = only ~2,500 epochs
- Training runs out of data and stops

**Solution**:
Train on the same time periods multiple times with different 64-asset combinations:
- Period 1 (716 days) × 15 asset combinations = 10,740 epochs
- Period 2 (548 days) × 20 asset combinations = 10,960 epochs
- etc.

---

## Implementation Tasks

### Task 1: Create Database Mapper Tool

**File**: `tools/map_training_data.py`

**Purpose**: Scan database and identify valid training periods with available assets.

#### Step 1.1: Find Initial Coverage Threshold

```python
#!/usr/bin/env python3
"""
Training Data Mapper - Rainbow Table Generator
Scans database to identify valid training periods and asset combinations.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import sys

def find_initial_threshold(db_path: str, min_assets: int = 64) -> int:
    """
    Scan forward from earliest data to find first timestamp where ≥min_assets exist.

    Args:
        db_path: Path to trading_data.db
        min_assets: Minimum concurrent assets required (default 64)

    Returns:
        Unix timestamp of first valid starting point
    """
    conn = sqlite3.connect(db_path)

    # Get earliest timestamp
    min_ts = conn.execute('SELECT MIN(timestamp) FROM ohlcv').fetchone()[0]

    print(f"Scanning from {datetime.fromtimestamp(min_ts)}...")

    # Scan forward in 1-day increments
    current_ts = min_ts
    max_ts = conn.execute('SELECT MAX(timestamp) FROM ohlcv').fetchone()[0]
    one_day = 24 * 60 * 60

    while current_ts <= max_ts:
        # Count distinct assets with data at this timestamp (±1 hour window)
        window_start = current_ts - 3600
        window_end = current_ts + 3600

        count = conn.execute('''
            SELECT COUNT(DISTINCT asset_id)
            FROM ohlcv
            WHERE timestamp BETWEEN ? AND ?
        ''', (window_start, window_end)).fetchone()[0]

        if count >= min_assets:
            conn.close()
            print(f"✓ Found initial threshold at {datetime.fromtimestamp(current_ts)}")
            print(f"  Available assets: {count}")
            return current_ts

        current_ts += one_day

    conn.close()
    raise ValueError(f"Never found {min_assets} concurrent assets in database")
```

#### Step 1.2: Map Valid Periods

```python
def map_valid_periods(
    db_path: str,
    start_ts: int,
    min_assets: int = 64,
    min_duration_days: int = 30,
    coverage_threshold: float = 0.75
) -> List[Dict]:
    """
    Identify continuous periods where sufficient assets exist.

    A period ends when:
    - ≥15 assets disappear (delisted/no data)
    - OR we reach end of database

    Args:
        db_path: Path to database
        start_ts: Starting timestamp (from find_initial_threshold)
        min_assets: Minimum assets required throughout period
        min_duration_days: Minimum period length in days (default 30)
        coverage_threshold: Min fraction of assets that must have data at each timestamp

    Returns:
        List of period dicts with start, end, duration, available_assets
    """
    conn = sqlite3.connect(db_path)

    # Get all timestamps in order
    timestamps = [
        row[0] for row in
        conn.execute('SELECT DISTINCT timestamp FROM ohlcv WHERE timestamp >= ? ORDER BY timestamp', (start_ts,))
    ]

    print(f"Analyzing {len(timestamps)} timestamps...")

    periods = []
    current_period_start = None
    current_period_assets = set()
    last_asset_count = 0

    for i, ts in enumerate(timestamps):
        # Get assets at this timestamp
        assets_at_ts = set([
            row[0] for row in
            conn.execute('''
                SELECT a.symbol
                FROM ohlcv o
                JOIN assets a ON o.asset_id = a.id
                WHERE o.timestamp = ?
            ''', (ts,))
        ])

        # Filter out forex pairs
        from train_manager import is_forex_pair
        assets_at_ts = {a for a in assets_at_ts if not is_forex_pair(a)}

        asset_count = len(assets_at_ts)

        # Start new period if we have enough assets and not in a period
        if current_period_start is None and asset_count >= min_assets:
            current_period_start = ts
            current_period_assets = assets_at_ts
            last_asset_count = asset_count
            print(f"  Period started at {datetime.fromtimestamp(ts)} ({asset_count} assets)")

        # Continue period
        elif current_period_start is not None:
            # Track assets that persist throughout period
            current_period_assets &= assets_at_ts  # Intersection

            # Check if we lost too many assets (≥15)
            assets_lost = last_asset_count - asset_count

            if assets_lost >= 15 or len(current_period_assets) < min_assets:
                # End current period
                period_end = timestamps[i - 1]  # Previous timestamp
                duration_days = (period_end - current_period_start) / (24 * 60 * 60)

                if duration_days >= min_duration_days:
                    periods.append({
                        'start': current_period_start,
                        'end': period_end,
                        'duration_days': int(duration_days),
                        'available_assets': sorted(list(current_period_assets)),
                        'asset_count': len(current_period_assets)
                    })
                    print(f"  Period saved: {duration_days:.0f} days, {len(current_period_assets)} persistent assets")
                else:
                    print(f"  Period too short: {duration_days:.0f} days < {min_duration_days}, skipped")

                # Start new period if we still have enough assets
                if asset_count >= min_assets:
                    current_period_start = ts
                    current_period_assets = assets_at_ts
                    last_asset_count = asset_count
                    print(f"  New period started at {datetime.fromtimestamp(ts)} ({asset_count} assets)")
                else:
                    current_period_start = None
                    current_period_assets = set()

            last_asset_count = asset_count

        # Progress indicator
        if i % 1000 == 0:
            print(f"  Progress: {i}/{len(timestamps)} timestamps ({i*100/len(timestamps):.1f}%)")

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

    print(f"\n✓ Found {len(periods)} valid periods")
    return periods
```

#### Step 1.3: Generate Asset Combinations

```python
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
    import random

    if len(available_assets) < subset_size:
        return [available_assets]  # Not enough for even one full subset

    combinations = []
    previous_subset = None

    # Calculate how many combinations we can make
    # Rough estimate: if we have 150 assets, ~35% overlap, we can make ~15 combinations
    max_combinations = min(50, int(len(available_assets) / subset_size * 2))

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
                # Can't make more unique combinations
                break

            new_assets = random.sample(available_new, remaining_count)
            subset = kept_assets + new_assets

            # Shuffle to avoid positional bias
            random.shuffle(subset)

        combinations.append(subset)
        previous_subset = subset

    print(f"  Generated {len(combinations)} asset combinations")
    return combinations
```

#### Step 1.4: Build Complete Training Schedule

```python
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
    start_ts = find_initial_threshold(db_path, min_assets=64)
    print()

    # Step 2: Map valid periods
    print("[2/4] Mapping valid training periods...")
    periods = map_valid_periods(
        db_path,
        start_ts,
        min_assets=64,
        min_duration_days=30,
        coverage_threshold=0.75
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
        print(f"  → {len(combinations)} runs × {period['duration_days']} days = {period['total_training_epochs']} epochs")
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
                'start': datetime.fromtimestamp(periods[0]['start']).isoformat(),
                'end': datetime.fromtimestamp(periods[-1]['end']).isoformat()
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
```

---

### Task 2: Update train_manager.py

**File**: `train_manager.py`

Add support for training from schedule:

```python
# Add new arguments
parser.add_argument('--training-schedule', help='JSON file with pre-computed training schedule')
parser.add_argument('--period-id', type=int, help='Period ID from schedule (1-indexed)')
parser.add_argument('--run-id', type=int, help='Run ID within period (1-indexed)')

# Add schedule loading logic
if args.training_schedule:
    import json
    with open(args.training_schedule) as f:
        schedule = json.load(f)

    # Find the specified period and run
    period = schedule['periods'][args.period_id - 1]  # Convert to 0-indexed
    run = period['training_runs'][args.run_id - 1]

    # Extract configuration
    pairs = run['assets']

    # Override date range with period dates
    # Use period start/end timestamps
    config = {
        'experiment_name': f"period_{args.period_id}_run_{args.run_id}",
        'n_assets': len(pairs),
        'asset_symbols': pairs,
        'min_coverage': args.min_coverage,
        'date_start_ts': period['start'],
        'date_end_ts': period['end'],
        'generated_at': str(time.time())
    }

    config_path = os.path.join('config', 'current_training.json')
    os.makedirs('config', exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    training_config_path = config_path

    logging.info(f"Loaded schedule: Period {args.period_id}, Run {args.run_id}")
    logging.info(f"  Duration: {period['duration_days']} days")
    logging.info(f"  Overlap: {run['overlap_with_previous']*100:.1f}%")
    logging.info(f"  Assets: {len(pairs)}")
```

---

### Task 3: Update HistoricalMarket to Use Timestamps

**File**: `mcts_trader/data/history.py`

Currently `HistoricalMarket` may filter by date strings. Ensure it can accept unix timestamps:

```python
# In HistoricalMarket.__init__
def __init__(self, ..., date_start_ts=None, date_end_ts=None):
    # If timestamps provided, use them
    if date_start_ts:
        self.start_timestamp = date_start_ts
    if date_end_ts:
        self.end_timestamp = date_end_ts

    # Filter data to this range
    self.valid_timestamps = [
        ts for ts in all_timestamps
        if self.start_timestamp <= ts <= self.end_timestamp
    ]
```

---

### Task 4: Create Training Runner Script

**File**: `tools/run_scheduled_training.py`

Automate running all scheduled training:

```python
#!/usr/bin/env python3
"""
Run all training sessions from schedule automatically.
"""

import json
import subprocess
import sys

def run_all_training(schedule_file='training_schedule.json', start_period=1, start_run=1):
    with open(schedule_file) as f:
        schedule = json.load(f)

    total_runs = schedule['summary']['total_unique_runs']
    current_run = 0

    for period_idx, period in enumerate(schedule['periods'], start=1):
        if period_idx < start_period:
            continue

        for run_idx, run in enumerate(period['training_runs'], start=1):
            if period_idx == start_period and run_idx < start_run:
                continue

            current_run += 1

            print(f"\n{'='*70}")
            print(f"RUN {current_run}/{total_runs}")
            print(f"Period {period_idx}/{len(schedule['periods'])}, Run {run_idx}/{len(period['training_runs'])}")
            print(f"{'='*70}\n")

            cmd = [
                'python', 'train_manager.py',
                '--training-schedule', schedule_file,
                '--period-id', str(period_idx),
                '--run-id', str(run_idx),
                '--min-coverage', '0.75',
                '--steps', '1440',
                '--epochs', str(period['duration_days']),
                '--no-validate'
            ]

            print(f"Command: {' '.join(cmd)}\n")

            result = subprocess.run(cmd)

            if result.returncode != 0:
                print(f"\n❌ Training failed! Stopping.")
                sys.exit(1)

    print(f"\n{'='*70}")
    print(f"✓ ALL TRAINING COMPLETE!")
    print(f"Completed {total_runs} training runs")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--schedule', default='training_schedule.json')
    parser.add_argument('--start-period', type=int, default=1)
    parser.add_argument('--start-run', type=int, default=1)
    args = parser.parse_args()

    run_all_training(args.schedule, args.start_period, args.start_run)
```

---

## Testing Steps

### Step 1: Generate Training Schedule

```bash
python tools/map_training_data.py
```

**Expected output**:
- `training_schedule.json` created
- Console shows periods found, asset counts, total epochs available

### Step 2: Inspect Schedule

```bash
# Pretty-print first period
python3 << 'EOF'
import json
with open('training_schedule.json') as f:
    schedule = json.load(f)

print("Period 1:")
period = schedule['periods'][0]
print(f"  Start: {period['start_date']}")
print(f"  End: {period['end_date']}")
print(f"  Duration: {period['duration_days']} days")
print(f"  Available assets: {period['asset_count']}")
print(f"  Training runs: {len(period['training_runs'])}")
print(f"  Total epochs: {period['total_training_epochs']}")
print(f"\n  Run 1 assets: {period['training_runs'][0]['assets'][:5]}...")
print(f"  Run 2 assets: {period['training_runs'][1]['assets'][:5]}...")
print(f"  Run 2 overlap: {period['training_runs'][1]['overlap_with_previous']*100:.1f}%")
EOF
```

### Step 3: Test Single Training Run

```bash
# Dry-run first
python train_manager.py \
  --training-schedule training_schedule.json \
  --period-id 1 \
  --run-id 1 \
  --min-coverage 0.75 \
  --steps 1440 \
  --epochs 5 \
  --no-validate \
  --dry-run

# If dry-run works, test 5 real epochs
python train_manager.py \
  --training-schedule training_schedule.json \
  --period-id 1 \
  --run-id 1 \
  --min-coverage 0.75 \
  --steps 1440 \
  --epochs 5 \
  --no-validate
```

**Expected**:
- No "out of data" errors
- Training completes 5 epochs
- Uses assets from schedule

### Step 4: Run Full Automation (Optional)

```bash
python tools/run_scheduled_training.py --schedule training_schedule.json
```

---

## Expected Schedule Example

```json
{
  "generated_at": "2025-12-13T10:30:00",
  "database": "trading_data.db",
  "periods": [
    {
      "start": 1577836800,
      "end": 1640908800,
      "start_date": "2020-01-01T00:00:00",
      "end_date": "2021-12-31T23:59:59",
      "duration_days": 730,
      "available_assets": ["BTC/USD", "ETH/USD", ...],
      "asset_count": 128,
      "training_runs": [
        {
          "run_id": 1,
          "assets": ["BTC/USD", "ETH/USD", ...],
          "overlap_with_previous": 0.0
        },
        {
          "run_id": 2,
          "assets": ["ETH/USD", "ADA/USD", ...],
          "overlap_with_previous": 0.42
        }
      ],
      "total_training_epochs": 14600
    }
  ],
  "summary": {
    "total_periods": 8,
    "total_unique_runs": 127,
    "total_training_epochs_available": 87450,
    "date_range": {
      "start": "2020-01-01T00:00:00",
      "end": "2025-12-12T23:59:59"
    }
  }
}
```

---

## Success Criteria

✅ `tools/map_training_data.py` successfully scans database
✅ `training_schedule.json` generated with ≥5 periods
✅ Each period has ≥5 training runs
✅ Asset overlap between consecutive runs is 30-50%
✅ Total available epochs > 50,000
✅ `train_manager.py` can load schedule and train
✅ No "out of data" errors during training
✅ Forex pairs excluded from all asset selections

---

## Notes

- **Periods can overlap**: A period from 2020-2021 and another from 2020-2022 can both exist
- **Overlap is consecutive only**: Run 2 overlaps with Run 1, Run 3 with Run 2, not Run 1
- **Random selection**: Use `random.sample()` for asset selection
- **Forex exclusion**: Import `is_forex_pair()` from `train_manager.py`
- **Progress logging**: Print progress every 1000 timestamps to show it's working

---

## Questions?

If anything is unclear:
1. Read this spec again carefully
2. Check existing code in `train_manager.py` for reference
3. Test incrementally (don't write everything at once)
4. Ask user for clarification if stuck

Good luck! 🚀

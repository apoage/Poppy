# Train Manager Integration - Keep It Simple

## Philosophy: "Good Food Fast, Not Michelin Star"

You're right - we're building a restaurant when you just wanted to cook dinner. Let's keep the train manager **practical and simple**.

---

## Simple Train Manager Interface

### Option 1: CSV Import (Easiest)
```bash
# User creates: datasets/my_pairs.csv
BTC/USD
ETH/USD
SOL/USD
# ... 61 more

# Train manager loads it:
python tools/train_manager.py --pairs datasets/my_pairs.csv
```

### Option 2: API Command (Auto-magic)
```bash
# Automatically fetch top 64 by volume from Kraken
python tools/train_manager.py --auto-select volume --days 7

# Or volatility
python tools/train_manager.py --auto-select volatility --days 30

# Or all BTC pairs
python tools/train_manager.py --auto-select base:BTC
```

### Option 3: Use Existing Dataset JSON
```bash
# If someone already generated a dataset
python tools/train_manager.py --dataset datasets/top64_volume_7d.json
```

---

## What Train Manager Should Do (Minimal Version)

### 1. Load Asset List (3 ways)
```python
def load_asset_list(args):
    """Simple asset loading - pick your poison"""

    if args.pairs:  # Option 1: CSV file
        with open(args.pairs) as f:
            return [line.strip() for line in f if line.strip()]

    elif args.dataset:  # Option 2: JSON dataset
        with open(args.dataset) as f:
            data = json.load(f)
            return [a['symbol'] for a in data['assets']]

    elif args.auto_select:  # Option 3: API auto-select
        return fetch_top_pairs_from_kraken(
            strategy=args.auto_select,
            count=64
        )

    else:  # Fallback: Use default
        return load_default_64_pairs()
```

### 2. Validate Pairs (Quick Check)
```python
def quick_validate(pairs):
    """Just check if pairs exist on Kraken - don't overthink it"""

    # Fetch Kraken AssetPairs once
    kraken_pairs = fetch_kraken_asset_pairs()

    valid = []
    invalid = []

    for pair in pairs:
        if pair in kraken_pairs and kraken_pairs[pair].get('status') == 'online':
            valid.append(pair)
        else:
            invalid.append(pair)

    if invalid:
        print(f"⚠️  Skipping {len(invalid)} invalid pairs: {invalid[:5]}...")

    print(f"✅ Training with {len(valid)} valid pairs")
    return valid
```

### 3. Write Config, Start Training
```python
def start_training(pairs, experiment_name):
    """Update config and launch training"""

    # Write pairs to a temp config
    config = {
        'n_assets': len(pairs),
        'asset_symbols': pairs,
        'experiment_name': experiment_name
    }

    with open('config/current_training.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Launch training
    subprocess.run(['python', 'train_jax_fast.py', '--config', 'config/current_training.json'])
```

---

## Train Manager - Full Simple Implementation

Create: `/home/lukas/Trader/tools/train_manager.py`

```python
#!/usr/bin/env python3
"""
Simple Training Manager
Usage:
    python tools/train_manager.py --pairs datasets/my_pairs.csv
    python tools/train_manager.py --auto-select volume --days 7
    python tools/train_manager.py --dataset datasets/top64_volume.json
"""

import argparse
import json
import requests
from pathlib import Path

def fetch_kraken_asset_pairs():
    """Get current tradeable pairs from Kraken"""
    r = requests.get('https://api.kraken.com/0/public/AssetPairs')
    return r.json()['result']

def fetch_top_pairs_from_kraken(strategy='volume', count=64, days=7):
    """
    Auto-select top pairs by strategy

    Strategies:
    - 'volume': Top by USD volume
    - 'volatility': Top by price volatility
    - 'base:BTC': All BTC pairs
    """

    if strategy.startswith('base:'):
        # All pairs of a base asset
        base = strategy.split(':')[1]
        pairs = fetch_kraken_asset_pairs()
        matching = [p for p in pairs.keys() if p.startswith(f"{base}/") or f"/{base}" in p]
        return matching[:count]

    elif strategy == 'volume':
        # Get volume data from Kraken Ticker endpoint
        r = requests.get('https://api.kraken.com/0/public/Ticker')
        tickers = r.json()['result']

        # Sort by 24h volume (USD equivalent)
        sorted_pairs = sorted(
            tickers.items(),
            key=lambda x: float(x[1].get('v', [0, 0])[1]),  # 24h volume
            reverse=True
        )

        return [pair for pair, _ in sorted_pairs[:count]]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def load_pairs_from_csv(filepath):
    """Load pair list from CSV"""
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def load_pairs_from_json(filepath):
    """Load pairs from dataset JSON"""
    with open(filepath) as f:
        data = json.load(f)
        return [a['symbol'] for a in data['assets']]

def validate_pairs(pairs):
    """Quick validation against Kraken API"""
    print(f"🔍 Validating {len(pairs)} pairs against Kraken...")

    kraken_pairs = fetch_kraken_asset_pairs()

    valid = []
    invalid = []

    for pair in pairs:
        # Try exact match first
        if pair in kraken_pairs and kraken_pairs[pair].get('status') == 'online':
            valid.append(pair)
        # Try common variants (BTC/USD vs XBTUSD)
        elif pair.replace('/', '') in kraken_pairs:
            alt = pair.replace('/', '')
            if kraken_pairs[alt].get('status') == 'online':
                valid.append(alt)
        else:
            invalid.append(pair)

    if invalid:
        print(f"⚠️  Skipping {len(invalid)} invalid/delisted pairs:")
        for p in invalid[:10]:
            print(f"    - {p}")
        if len(invalid) > 10:
            print(f"    ... and {len(invalid) - 10} more")

    print(f"✅ {len(valid)} valid pairs ready for training")
    return valid

def create_training_config(pairs, experiment_name='auto'):
    """Create training config file"""
    config = {
        'experiment_name': experiment_name,
        'n_assets': len(pairs),
        'asset_symbols': pairs,
        'generated_at': str(Path.cwd())
    }

    config_path = Path('config/current_training.json')
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"📝 Config written to: {config_path}")
    return config_path

def main():
    parser = argparse.ArgumentParser(description='Simple Training Manager')

    # Input methods (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pairs', help='CSV file with pair list (one per line)')
    group.add_argument('--dataset', help='JSON dataset file')
    group.add_argument('--auto-select', help='Auto-select pairs (volume, volatility, base:ASSET)')

    # Options
    parser.add_argument('--count', type=int, default=64, help='Number of pairs (default: 64)')
    parser.add_argument('--days', type=int, default=7, help='Days for volume/volatility calc (default: 7)')
    parser.add_argument('--name', default='auto', help='Experiment name')
    parser.add_argument('--no-validate', action='store_true', help='Skip Kraken validation')
    parser.add_argument('--dry-run', action='store_true', help='Show pairs without training')

    args = parser.parse_args()

    # Load pairs
    print("📊 Loading asset pairs...")
    if args.pairs:
        pairs = load_pairs_from_csv(args.pairs)
        print(f"   Loaded {len(pairs)} pairs from CSV")
    elif args.dataset:
        pairs = load_pairs_from_json(args.dataset)
        print(f"   Loaded {len(pairs)} pairs from JSON dataset")
    elif args.auto_select:
        pairs = fetch_top_pairs_from_kraken(args.auto_select, args.count, args.days)
        print(f"   Auto-selected {len(pairs)} pairs (strategy: {args.auto_select})")

    # Validate (unless skipped)
    if not args.no_validate:
        pairs = validate_pairs(pairs)

    if not pairs:
        print("❌ No valid pairs found. Exiting.")
        return

    # Create config
    config_path = create_training_config(pairs, args.name)

    if args.dry_run:
        print("\n🏁 DRY RUN - Would train with:")
        for i, p in enumerate(pairs[:10], 1):
            print(f"   {i}. {p}")
        if len(pairs) > 10:
            print(f"   ... and {len(pairs) - 10} more")
        return

    # Launch training
    print("\n🚀 Starting training...")
    print(f"   Config: {config_path}")
    print(f"   Pairs: {len(pairs)}")
    print(f"   Experiment: {args.name}")
    print("\n" + "="*60)

    import subprocess
    subprocess.run([
        'python', 'train_jax_fast.py',
        '--config', str(config_path)
    ])

if __name__ == '__main__':
    main()
```

---

## Usage Examples (Real World)

### Quick Start - Just Train Top Pairs
```bash
# Simplest: Top 64 by volume, auto-validated
python tools/train_manager.py --auto-select volume

# That's it. Done.
```

### Custom Pair List
```bash
# Create: datasets/my_favorites.csv
# BTC/USD
# ETH/USD
# SOL/USD

python tools/train_manager.py --pairs datasets/my_favorites.csv --count 64
```

### Test Before Training
```bash
# Dry run to see what would be selected
python tools/train_manager.py --auto-select volume --dry-run

# Output shows pairs, no training starts
```

### All BTC Pairs
```bash
python tools/train_manager.py --auto-select base:BTC --name btc_universe
```

---

## Integration with Existing Code

### Modify `train_jax_fast.py` to Accept Config

Add these lines to the top:

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Training config JSON')
args = parser.parse_args()

if args.config:
    with open(args.config) as f:
        config = json.load(f)
        N_ASSETS = config['n_assets']
        ASSET_SYMBOLS = config['asset_symbols']
        EXPERIMENT_NAME = config['experiment_name']
else:
    # Use defaults
    N_ASSETS = 64
    ASSET_SYMBOLS = None  # Will be loaded from database
```

---

## What Gemini Needs to Build (Minimal)

### Priority 1: Fix Critical Bugs (from previous prompts)
1. Symbol mapping fixes
2. NoneType error fixes
3. Database validation script

### Priority 2: Simple Train Manager (this prompt)
1. Create `tools/train_manager.py` (copy-paste above)
2. Add `--config` argument to `train_jax_fast.py`
3. Test with `--auto-select volume --dry-run`

### Priority 3: Optional Enhancements (only if time)
1. Full dataset generation scripts (from DYNAMIC_ASSET_SELECTION_PROMPT.md)
2. Data quality validation
3. Gap filling

---

## Bottom Line: Keep It Simple

**Must Have (30 minutes work):**
- ✅ Train manager that accepts CSV or auto-selects from Kraken
- ✅ Quick validation (just check if pair exists)
- ✅ Launch training with selected pairs

**Nice to Have (2+ hours work):**
- Multiple dataset strategies
- Data quality checks
- Gap filling
- Continuous window detection

**Start with "Must Have", ship the restaurant later. Cook dinner first.** 🍳

---

## Next Steps

Tell Gemini:
1. Implement Priority 1 (critical bug fixes) - **REQUIRED**
2. Implement Priority 2 (simple train manager) - **REQUIRED**
3. Skip Priority 3 for now - **OPTIONAL FUTURE**

Get training running with validated pairs ASAP. Optimize later.

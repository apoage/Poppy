# Dynamic Asset Selection Strategy - Prompt for Gemini

## Problem Statement

**Current Approach:** Hardcoded 64 assets with manual blacklist maintenance
- ❌ Error-prone (delisted pairs like EOS/USD slip through)
- ❌ No way to detect newly delisted pairs automatically
- ❌ Misses new high-volume opportunities
- ❌ Trains on stale/illiquid pairs

**Better Approach:** Query Kraken API for validated, active pairs and select best 64 based on strategy

---

## Core Strategy: "Ask Kraken for the Menu"

### Step 1: Query Kraken for ALL Active Pairs

**API Endpoint:** `AssetPairs` (REST API)

**Validation Criteria:**
```python
# A pair is tradeable if:
1. It exists in AssetPairs response
2. status == "online" (not "cancel_only" or missing)
3. No error: "EQuery:Unknown asset pair"
```

**Exclusions (automatic):**
- Pairs with `status != "online"` (delisting in progress)
- Pairs returning "Unknown asset pair" error
- Futures contracts (PF_* prefix)
- Derivatives (not spot/margin pairs)
- Cross-check against `/home/lukas/Trader/delistedpairs.csv`

### Step 2: Rank Pairs by Multiple Strategies

Instead of one fixed 64-asset universe, create **multiple training datasets** for different strategies:

#### Strategy A: Top 64 by Volume (Last 7 Days)
```python
# Use Kraken OHLC endpoint to get volume data
# Sort pairs by: sum(volume_usd, last_7_days)
# Select top 64
# Use case: Liquidity-focused, market-making strategies
```

#### Strategy B: Top 64 by Volatility (Last 7 Days)
```python
# Calculate: stddev(daily_returns, last_7_days)
# Sort by volatility descending
# Select top 64
# Use case: Momentum/swing trading strategies
```

#### Strategy C: Base Currency Portfolios
```python
# All BTC pairs:  BTC/USD, BTC/EUR, BTC/GBP, etc. (exhaustive)
# All ETH pairs:  ETH/USD, ETH/EUR, ETH/BTC, etc.
# All SOL pairs:  SOL/USD, SOL/EUR, SOL/BTC, etc.
# Pad to 64 with top alts by volume
# Use case: Single-asset exposure with multi-currency hedging
```

#### Strategy D: Stablecoin Cross-Rates
```python
# All pairs matching: */USD, */EUR, */GBP, */CHF, */CAD, */JPY
# Filter to assets with ALL major fiat pairs available
# Example: If we have ETH/USD, ETH/EUR, ETH/GBP → include ETH
# Use case: FX arbitrage + crypto exposure
```

#### Strategy E: Top 64 New Listings (Last 6 Months)
```python
# Query pairs with first_trade_date > (now - 180 days)
# Sort by volume or volatility
# Select top 64
# Use case: Early momentum capture on new assets
```

#### Strategy F: Sector-Specific (DeFi, L1, Meme, etc.)
```python
# Group assets by category (would need external metadata)
# Top 64 DeFi tokens by volume
# Top 64 Layer-1 chains by volume
# Use case: Specialized sector strategies
```

---

## Implementation Requirements

### Phase 1: Validate Database Against API

**Your Task:**

1. **Query Kraken AssetPairs API**
   ```python
   GET https://api.kraken.com/0/public/AssetPairs
   ```

2. **For Each Pair in `trading_data.db`:**
   - Check if it exists in API response
   - Verify `status == "online"`
   - Check against `delistedpairs.csv`
   - **Mark as `active=0` if delisted/unavailable**

3. **Log Results:**
   ```
   Total pairs in database: 128
   Active on Kraken: 95
   Delisted (API missing): 18
   Delisted (status != online): 10
   Delisted (in delistedpairs.csv): 5
   ```

4. **Update Database Schema:**
   ```sql
   ALTER TABLE assets ADD COLUMN active INTEGER DEFAULT 1;
   ALTER TABLE assets ADD COLUMN status TEXT;
   ALTER TABLE assets ADD COLUMN last_validated_at INTEGER;

   -- Mark delisted
   UPDATE assets
   SET active=0, status='delisted'
   WHERE symbol IN (SELECT Asset_or_Pair FROM delisted_pairs);
   ```

### Phase 2: Validate Historical Data Quality

**Your Task:**

For each **active** pair in database:

1. **Check Data Completeness:**
   ```sql
   -- Get date range for each pair
   SELECT
       a.symbol,
       MIN(o.timestamp) as first_candle,
       MAX(o.timestamp) as last_candle,
       COUNT(*) as total_candles,
       -- Expected candles for continuous 1-min data
       (MAX(o.timestamp) - MIN(o.timestamp)) / 60 as expected_candles,
       -- Gap percentage
       (1 - COUNT(*) / ((MAX(o.timestamp) - MIN(o.timestamp)) / 60.0)) * 100 as gap_pct
   FROM ohlcv o
   JOIN assets a ON o.asset_id = a.id
   WHERE a.active = 1
   GROUP BY a.id
   ORDER BY gap_pct ASC;
   ```

2. **Verify Against Kraken API (Spot Check):**
   ```python
   # For random sample of 10 pairs:
   # Fetch last 100 candles from Kraken API
   # Compare with database values
   # If mismatch > 5%, flag for re-download
   ```

3. **Identify Best Continuous Dataset:**
   ```python
   # Find longest time window where ALL 64 selected pairs have data
   # Example:
   #   Pair A: 2020-01-01 to 2025-12-12
   #   Pair B: 2021-06-15 to 2025-12-12
   #   Pair C: 2019-05-01 to 2025-12-12
   #   → Continuous window: 2021-06-15 to 2025-12-12 (all overlap)
   ```

4. **Fill Data Gaps (if needed):**
   ```python
   # If gap_pct > 1% for a pair:
   # - Query Kraken API for missing time ranges
   # - Backfill database
   # - Re-validate
   ```

### Phase 3: Generate Training Datasets

**Your Task:**

Create a script: `/home/lukas/Trader/tools/generate_training_dataset.py`

**Usage:**
```bash
# Generate multiple datasets
python tools/generate_training_dataset.py --strategy volume --days 7 --output datasets/top64_volume_7d.json
python tools/generate_training_dataset.py --strategy volatility --days 30 --output datasets/top64_volatility_30d.json
python tools/generate_training_dataset.py --strategy base_currency --asset BTC --output datasets/btc_universe.json
python tools/generate_training_dataset.py --strategy stablecoin_fx --output datasets/fiat_arbitrage.json
```

**Output Format (JSON):**
```json
{
  "strategy": "top64_volume_7d",
  "generated_at": "2025-12-12T10:00:00Z",
  "validation": {
    "all_active": true,
    "continuous_window": {
      "start": "2021-06-15T00:00:00Z",
      "end": "2025-12-12T10:00:00Z",
      "total_minutes": 2340000
    },
    "avg_gap_pct": 0.3,
    "max_gap_pct": 1.2
  },
  "assets": [
    {
      "symbol": "BTC/USD",
      "rank": 1,
      "metric_value": 15000000000.0,
      "data_coverage": 99.8,
      "first_candle": "2019-01-01T00:00:00Z",
      "last_candle": "2025-12-12T09:59:00Z"
    },
    {
      "symbol": "ETH/USD",
      "rank": 2,
      "metric_value": 8000000000.0,
      "data_coverage": 99.5,
      "first_candle": "2019-01-01T00:00:00Z",
      "last_candle": "2025-12-12T09:59:00Z"
    },
    // ... 62 more
  ]
}
```

### Phase 4: Update Training Config

**Your Task:**

Modify `/home/lukas/Trader/mcts_trader/config/trading_config.py` to accept dataset files:

```python
# Old (hardcoded)
N_ASSETS = 64

# New (dynamic)
DATASET_FILE = 'datasets/top64_volume_7d.json'  # Can switch strategies easily

# In training code:
with open(DATASET_FILE) as f:
    dataset = json.load(f)
    N_ASSETS = len(dataset['assets'])
    ASSET_SYMBOLS = [a['symbol'] for a in dataset['assets']]
    TRAINING_WINDOW_START = dataset['validation']['continuous_window']['start']
    TRAINING_WINDOW_END = dataset['validation']['continuous_window']['end']
```

---

## Delisted Pair Detection Logic

### Method 1: API Status Check (Recommended)
```python
def is_pair_active(pair: str) -> bool:
    """Query Kraken to check if pair is tradeable"""
    response = requests.get('https://api.kraken.com/0/public/AssetPairs')
    pairs = response.json()['result']

    if pair not in pairs:
        return False  # Pair doesn't exist

    if pairs[pair].get('status') != 'online':
        return False  # Delisting in progress

    return True
```

### Method 2: Error Code Detection
```python
def test_pair_with_ohlc(pair: str) -> bool:
    """Attempt to fetch recent data, catch delisting errors"""
    try:
        response = requests.get(f'https://api.kraken.com/0/public/OHLC?pair={pair}&interval=1')
        result = response.json()

        if result.get('error'):
            if 'EQuery:Unknown asset pair' in str(result['error']):
                return False  # Delisted

        return True
    except:
        return False
```

### Method 3: CSV Blacklist (Fallback)
```python
def load_delisted_pairs(csv_path='/home/lukas/Trader/delistedpairs.csv'):
    """Load known delisted pairs from CSV"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return set(df['Asset_or_Pair'].str.strip())

delisted = load_delisted_pairs()

def is_blacklisted(pair: str) -> bool:
    return pair in delisted
```

### Combined Approach (Best)
```python
def validate_pair(pair: str) -> dict:
    """
    Comprehensive validation returning detailed status
    """
    result = {
        'pair': pair,
        'active': True,
        'reasons': []
    }

    # Check 1: API status
    if not is_pair_active(pair):
        result['active'] = False
        result['reasons'].append('API: status != online or missing')

    # Check 2: OHLC test
    if not test_pair_with_ohlc(pair):
        result['active'] = False
        result['reasons'].append('API: OHLC query failed')

    # Check 3: Blacklist
    if is_blacklisted(pair):
        result['active'] = False
        result['reasons'].append('CSV: in delistedpairs.csv')

    return result
```

---

## Files to Create

### 1. `/home/lukas/Trader/tools/validate_database.py`
Validates all pairs in database against Kraken API, marks delisted as inactive.

### 2. `/home/lukas/Trader/tools/generate_training_dataset.py`
Generates strategy-specific 64-asset datasets with validation.

### 3. `/home/lukas/Trader/tools/backfill_gaps.py`
Identifies and fills data gaps from Kraken API.

### 4. `/home/lukas/Trader/datasets/`
Directory to store generated dataset JSON files.

---

## Implementation Checklist

- [ ] Read Kraken AssetPairs API documentation
- [ ] Create `validate_database.py` script
- [ ] Add `active`, `status`, `last_validated_at` columns to assets table
- [ ] Run validation, mark delisted pairs as inactive
- [ ] Create `generate_training_dataset.py` with all 6 strategies
- [ ] Generate at least 3 datasets: volume, volatility, BTC-universe
- [ ] Validate data quality (gaps < 1% for selected pairs)
- [ ] Find best continuous time window for training
- [ ] Update `trading_config.py` to accept dataset files
- [ ] Document strategy selection rationale

---

## Testing

After implementation:

```bash
# Validate database
python tools/validate_database.py
# Expected: Report of active vs delisted pairs

# Generate datasets
python tools/generate_training_dataset.py --strategy volume --days 7
# Expected: datasets/top64_volume_7d.json with validation report

# Test training with new dataset
python train_jax_fast.py --dataset datasets/top64_volume_7d.json
# Expected: Training starts with correct 64 assets
```

---

## Benefits of This Approach

1. **Automatic Delisting Detection** - No manual maintenance
2. **Multiple Strategies** - Train specialized models for different market conditions
3. **Data Quality Assurance** - Only train on validated, complete data
4. **Easy Iteration** - Switch datasets without code changes
5. **Reproducibility** - Dataset JSON files are versioned artifacts
6. **API-First** - Always in sync with Kraken's current offerings

---

## Start Your Work

Begin with:
1. Read Kraken API docs for AssetPairs and OHLC endpoints
2. Implement `validate_database.py`
3. Test delisting detection logic
4. Generate your first dataset (start with volume strategy - simplest)

Document all findings in: `DATABASE_VALIDATION_REPORT.md`

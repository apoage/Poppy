# DeFi On-Chain Market Data Pipeline

> **OUTPUT_DIR**: `{{OUTPUT_DIR}}`
> Replace `{{OUTPUT_DIR}}` with a unique folder path before use (e.g. `prompt-lib/out-defi-01`).

You are building a data pipeline that downloads DeFi trading data from a **self-hosted archive node** and converts it to the format Bella (our market generator) needs for training. On-chain data is the only truly free, legally unencumbered market data — every swap is a public blockchain event, no licensing required.

## Why DeFi Data

- **Legally clean**: Public blockchain data, no exchange data agreements, no redistribution restrictions
- **True price discovery**: AMM pools are real markets with real liquidity
- **Complete market picture**: Not just top tokens — the ENTIRE token universe, including births/deaths/migrations
- **Rich structure**: Liquidity depth, fee tiers, concentrated liquidity, MEV — richer than any CEX feed

## Architecture: Own Node (Sync → Extract → Freeze)

We run our own archive node. No third-party indexer dependencies, no rate limits, no API keys, no sunset risk.

**Strategy: Sync once on HDD, extract everything, freeze node.**

The archive node is a one-time data source, not a permanent service. Sync the full chain, run the extraction pipeline against it, then stop the node. The extracted SQLite database is what we use day-to-day — not the node.

```
Phase 1: Sync    — Reth archive on HDD (~2.5 TB, 2-4 weeks on spinning disk)
Phase 2: Extract — Pool discovery → swap download → OHLCV build (~8-16 hours)
Phase 3: Freeze  — Stop node, keep chain data as cold archive on HDD
Phase 4: Work    — All queries against extracted SQLite (~100 GB)
```

**Recommended stack:**
```
Ethereum:  Reth (archive mode) + Lighthouse consensus
           ~2.5 TB, syncs in ~3 days on NVMe / 2-4 weeks on HDD
           Alternative: Erigon (~1.5 TB, more HDD-friendly due to flat-file stages)

Arbitrum:  Nitro node (archive) — follows L1 Reth
           ~1 TB additional (Phase 2 — add after ETH extraction works)

Base:      op-geth + op-node (archive) — follows L1 Reth
           ~500 GB additional (Phase 2)

Polygon:   Bor + Heimdall (archive)
           ~4 TB (largest, optional — add later)
```

**Why Reth:** Rust-based, fastest sync, lowest disk usage for archive, native `eth_getLogs` performance. Handles the heavy `getLogs` queries we need without choking.

**Why HDD is acceptable:** Sync is slow but one-time. Extraction queries (`eth_getLogs` by block range) are sequential scans — HDD handles these at acceptable speed. We don't need the node for ongoing queries.

**Minimum viable:** Ethereum archive node alone covers 80%+ of DeFi TVL. Add L2s incrementally.

### Storage Layout (HDD)

```
/mnt/hdd/                          # 5.1 TB available
├── reth/                          # ~2.5 TB — archive node (frozen after extraction)
│   ├── data/                      # Chain data
│   └── lighthouse/                # Consensus data
├── defi_extracted/                # ~100-130 GB — extracted pipeline output
│   ├── raw_swaps.db               # ~30-40 GB (per-pool swap events)
│   ├── trading_data.db            # ~50-80 GB (OHLCV, per-pool + merged views)
│   ├── token_metadata.db          # ~50 MB
│   └── training_schedule.json     # ~10 MB
└── free                           # ~2.4 TB remaining
```

### Node Query Interface

All data extraction via `eth_getLogs` against the local node:

```python
# Local RPC — no rate limits, no API keys
RPC_URL = "http://localhost:8545"  # Reth default

# Batch getLogs — 10,000 blocks per query on local node (vs 100 on Alchemy)
{
    "method": "eth_getLogs",
    "params": [{
        "fromBlock": hex(start),
        "toBlock": hex(start + 10000),
        "topics": ["0xc42079f9..."]  # Swap event signature
    }]
}
```

No address filter — capture ALL swap events across ALL pools in one pass per block range. This is the key advantage of a local node: you can scan the entire chain efficiently.

## Target Output Format

Bella expects training data in SQLite (`trading_data.db`):

```sql
-- === Per-pool OHLCV (raw, preserves pool-level microstructure) ===
CREATE TABLE ohlcv_pool (
    timestamp INTEGER,       -- Unix seconds, 60s intervals
    pool_id INTEGER,         -- Maps to pools table
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL              -- USD volume in THIS pool
);

-- === Merged OHLCV (canonical price per token, volume summed across pools) ===
CREATE TABLE ohlcv (
    timestamp INTEGER,       -- Unix seconds, 60s intervals
    asset_id INTEGER,        -- Maps to assets table (token-level)
    open REAL,               -- From primary pool (highest TVL)
    high REAL,
    low REAL,
    close REAL,
    volume REAL,             -- USD volume summed across ALL pools for this token
    n_pools INTEGER          -- Number of active pools contributing
);

-- === Tokens (one row per token, not per pool) ===
CREATE TABLE assets (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    name TEXT,
    address TEXT,            -- Token contract address
    chain TEXT,              -- 'ethereum', 'arbitrum', 'base'
    decimals INTEGER,
    first_seen INTEGER,      -- First swap timestamp (any pool)
    last_seen INTEGER,       -- Last swap timestamp (any pool)
    primary_pool_id INTEGER  -- Current highest-TVL pool (changes over time)
);

-- === Pools (one row per AMM pool) ===
CREATE TABLE pools (
    id INTEGER PRIMARY KEY,
    address TEXT,            -- Pool contract address
    token0_address TEXT,
    token1_address TEXT,
    token0_id INTEGER,       -- Maps to assets.id
    token1_id INTEGER,       -- Maps to assets.id
    fee_tier INTEGER,        -- V3: 100/500/3000/10000 bps. V2: 30 bps fixed
    dex TEXT,                -- 'uniswap_v3', 'uniswap_v2', 'sushiswap'
    chain TEXT,
    created_block INTEGER,
    created_timestamp INTEGER
);

-- Indexes
CREATE INDEX idx_ohlcv_pool_ts ON ohlcv_pool(pool_id, timestamp);
CREATE INDEX idx_ohlcv_ts ON ohlcv(timestamp);
CREATE INDEX idx_ohlcv_asset ON ohlcv(asset_id, timestamp);
CREATE INDEX idx_pools_tokens ON pools(token0_id, token1_id);
```

**Two levels of data:**
- `ohlcv_pool`: Per-pool candles. Same token has multiple rows per minute (one per active pool). Preserves microstructure differences across fee tiers, DEXes, V2 vs V3.
- `ohlcv`: Merged canonical view. One row per token per minute. Price from primary (highest-TVL) pool, volume summed across all pools. This is what Bella trains on by default.

**Shape per training window:** `(n_timesteps, 64, 5)` — but 64 is the **slot count**, not the total asset count. The full universe is much larger (500-2000+ tokens). We rotate which 64 fill the slots per training window.

## Complete Market Capture (Not Just Top 64)

**Critical design decision:** We download ALL tokens with meaningful trading activity, not a curated top-N list.

### Why All Tokens

1. **Births and deaths are signal**: Tokens listing, gaining liquidity, losing liquidity, dying — this IS the market. A model trained only on survivors has survivorship bias.
2. **Rotating 64-asset windows**: Our model takes 64 asset slots. With 500+ tokens, each training window samples a different 64-asset subset. This creates massive data augmentation — same time period, different asset views.
3. **Correlation diversity**: Top-64 are heavily correlated (all follow BTC). Smaller tokens have unique dynamics — meme cycles, governance events, exploits, migrations.
4. **Regime coverage**: DeFi summers, winter capitulation, exploit contagion, airdrop seasons — small tokens capture regime transitions that blue chips smooth over.

### Inclusion Criteria (Permissive)

```python
# Include ANY token that has:
# - At least 1 Uniswap V2/V3 pool with TVL > $10K at any point
# - At least 100 swaps in any 24h period
# - Existed for at least 7 days
#
# Do NOT filter by:
# - Current TVL (dead tokens are valuable training data)
# - Market cap ranking
# - "Legitimacy" (scam tokens have real price dynamics)
#
# DO filter out:
# - Honeypot tokens (can buy but not sell — detectable via contract analysis)
# - Tokens with < 7 days of any activity (dust/test deployments)
# - Rebasing tokens (OHM, AMPL — price mechanics are non-standard)
```

### Expected Universe Size

| Chain | Est. Tokens | With Activity | OHLCV rows (1min, all) |
|-------|-------------|---------------|----------------------|
| Ethereum | ~500K ERC20 | ~2,000 with pools | ~2B |
| Arbitrum | ~50K | ~500 | ~400M |
| Base | ~30K | ~300 | ~200M |
| **Total** | — | **~2,800** | **~2.6B** |

Disk: ~50-80 GB in SQLite (with indexes).

## Multi-Pool Strategy: Per-Pool Timelines

**Key insight:** Don't collapse all pools for a token into one canonical price. Keep per-pool timelines as separate data — they capture different market microstructure.

### Why Per-Pool Matters

Same token, different pools = different markets:

```
WETH/USDC 0.05% pool  →  Arb bots, tight spreads, fast price discovery (leads)
WETH/USDC 0.30% pool  →  Retail flow, wider spreads, more noise
WETH/USDC 1.00% pool  →  Volatile/thin, larger trades, illiquid
WETH/USDC V2 pair     →  Legacy constant-product, different fee structure
```

Each pool attracts different trader types. The 0.05% pool typically leads (arb bots correct it first), higher-fee pools lag. This **lead-lag structure** is trainable signal.

### Cross-Pool Correlation as Training Signal

Price divergence between pools for the same token is:
1. **Arbitrage signal** — when pools disagree, someone will arb. The model can learn to detect and anticipate these corrections.
2. **Liquidity regime indicator** — when pool prices converge tightly, market is liquid and efficient. When they diverge, stress/illiquidity.
3. **Natural data augmentation** — same token viewed through different pool lenses. Each pool is a different "instrument" tracking the same underlying.

### Multi-Pool in Training Windows

The rainbow table can leverage this in two ways:

```
Mode A (default): Merged view
  - 64 asset slots, one canonical price per token
  - Volume/liquidity summed across pools
  - Simpler, matches current Bella format exactly

Mode B (advanced): Pool-level view
  - 64 slots can mix tokens AND pool variants
  - Slot 0: WETH/USDC-0.05%  (lead signal)
  - Slot 1: WETH/USDC-0.30%  (lag signal)
  - Slot 2: UNI/WETH-0.30%
  - Model sees cross-pool spreads directly as inter-asset correlations
  - Same token in different slots teaches the model about market microstructure
```

Mode A is the starting point. Mode B is a future extension once merged pipeline works.

### Primary Pool Selection

For the merged `ohlcv` table, "primary pool" = highest TVL at that timestamp:

```python
# Primary pool changes over time (e.g., V2→V3 migration in 2021)
# Track per-month or per-week:
# 1. Sum swap volume per pool per week
# 2. Pool with highest volume = primary for that week
# 3. Price from primary, volume from ALL pools
#
# Edge case: primary pool changes mid-period
# → Use the new primary from the changeover point, no retroactive rewrite
```

## Rainbow Table: Rotating Asset Windows

Reference implementation: `tools/map_training_data.py` → `generate_asset_combinations()`

The model always sees 64 asset slots. But the universe has 2000+ tokens. The **rainbow table** maps time periods to rotating 64-asset subsets:

### How It Works

```
Universe: 2000 tokens (not all alive at same time)
At any timestamp: ~200-800 tokens have active trading

For each training window (400-1440 minutes):
  1. Find tokens with data in this time range
  2. Sample 64 from available tokens (weighted by volume/liquidity)
  3. Next window: sample DIFFERENT 64 with 30-50% overlap
  4. Repeat → same time period generates many unique 64-asset views
```

### Asset Combination Generation

From `tools/map_training_data.py:292`:

```python
def generate_asset_combinations(
    available_assets: List[str],
    subset_size: int = 64,
    max_overlap: float = 0.5,   # At most 50% shared with previous
    min_overlap: float = 0.3    # At least 30% shared (continuity)
) -> List[List[str]]:
    """
    Generate multiple 64-asset subsets with controlled overlap.

    With 500 available assets and 64 per subset:
    → ~20 unique combinations per time period
    → 20x data augmentation for free

    Overlap constraints ensure:
    - Model sees asset transitions (some persist, some rotate out)
    - No two windows are identical
    - Core liquid assets appear frequently, tail assets appear occasionally
    """
```

### Period Mapping

From `tools/map_training_data.py:135`:

```python
def map_valid_periods(
    db_path, start_ts,
    min_assets=64,          # Need at least 64 active tokens
    min_duration_days=30,   # Minimum 30-day continuous period
    max_duration_days=180,  # Force asset refresh every 6 months
    coverage_threshold=0.75 # 75% of selected assets must have data
):
    """
    Scan database for continuous periods where enough tokens trade.

    A period ends when:
    - ≥15 assets disappear (delistings, liquidity drain)
    - Max duration reached (forces fresh asset selection)
    - Data gap > threshold

    Each period gets multiple 64-asset combinations → rainbow table entries.
    """
```

### Training Schedule Output

```json
{
  "periods": [
    {
      "start": 1620000000,
      "end": 1635000000,
      "duration_days": 173,
      "available_assets": ["WETH", "WBTC", "UNI", ...],
      "asset_count": 487,
      "combinations": [
        {"assets": ["WETH", "WBTC", ...], "asset_ids": [0, 1, ...]},
        {"assets": ["WETH", "LINK", ...], "asset_ids": [0, 5, ...]},
        ...
      ]
    }
  ]
}
```

## Implementation Plan

### Step 1: Pool Discovery (Full Scan)

Scan the node for ALL Uniswap V2/V3 factory events — every pool ever created:

```python
# Uniswap V3 Factory: PoolCreated event
# 0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118
# Returns: token0, token1, fee, tickSpacing, pool

# Uniswap V2 Factory: PairCreated event
# 0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9
# Returns: token0, token1, pair

# Known factory addresses:
FACTORIES = {
    'uniswap_v3': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
    'uniswap_v2': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
    'sushiswap':  '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac',
    # Add Arbitrum/Base equivalents
}

# Scan ALL PoolCreated/PairCreated events → build complete pool registry
# Output: config/defi_pools.json with every pool ever created
```

### Step 2: Bulk Swap Download (Unfiltered)

Download ALL swap events across ALL discovered pools:

```python
# Strategy: scan block ranges, not per-pool
# Local node handles this efficiently — no address filter needed
#
# For each block range (10,000 blocks ≈ 1.5 days on ETH):
#   eth_getLogs(topics=[SWAP_V3_SIG])  → all V3 swaps
#   eth_getLogs(topics=[SWAP_V2_SIG])  → all V2 swaps
#
# Decode and store raw:

CREATE TABLE raw_swaps (
    chain TEXT,
    block_number INTEGER,
    tx_hash TEXT,
    log_index INTEGER,
    pool_address TEXT,
    timestamp INTEGER,         -- from block.timestamp
    token0_amount TEXT,        -- signed, raw units
    token1_amount TEXT,        -- signed, raw units
    sqrt_price_x96 TEXT,       -- V3 only
    tick INTEGER,              -- V3 only
    PRIMARY KEY (chain, tx_hash, log_index)
);

# Also fetch block timestamps (batch eth_getBlockByNumber)
# Store in blocks table for timestamp lookups
```

**Parallelization:** Split block ranges across threads. 4x 4090 burn-in machine can run 4 parallel download streams while GPUs stress-test.

### Step 3: Token Metadata Resolution

For each unique token address seen in swaps:

```python
# Batch multicall to resolve:
# - symbol()  → "WETH", "UNI", etc.
# - name()    → "Wrapped Ether", etc.
# - decimals() → 18, 6, 8, etc.
#
# Some tokens have non-standard or missing symbol/name
# Fallback: use address as identifier
#
# Flag special cases:
# - Rebasing tokens (detect via known list + supply changes)
# - Proxy tokens (detect via EIP-1967 storage slot)
# - Fee-on-transfer tokens (detect via transfer amount mismatch)
```

### Step 4: Price Derivation (USD)

```python
# Price routing:
# 1. Direct: token/USDC or token/USDT pool exists → use directly
# 2. Via WETH: token/WETH pool + WETH/USDC pool → cross rate
# 3. Via other stable: token/DAI, token/FRAX → use stable price
# 4. Multi-hop: token/X + X/WETH + WETH/USDC (max 3 hops)
#
# Price derivation from Uniswap V3 sqrtPriceX96:
#   price = (sqrtPriceX96 / 2^96)^2 * (10^decimals0 / 10^decimals1)
#
# IMPORTANT: Use Python int arithmetic for sqrtPriceX96 (uint160)
# Float64 loses precision above 2^53

def derive_usd_price(sqrt_price_x96: int, decimals0: int, decimals1: int,
                     is_token0_usd: bool) -> float:
    """Derive USD price from V3 sqrtPriceX96."""
    price_raw = (sqrt_price_x96 ** 2) / (2 ** 192)
    decimal_adj = 10 ** (decimals0 - decimals1)
    price = price_raw * decimal_adj
    return price if is_token0_usd else (1 / price if price > 0 else 0)
```

### Step 5: OHLCV Aggregation (Per-Pool → Merged)

Two-pass aggregation:

```python
# === Pass 1: Per-pool OHLCV → ohlcv_pool table ===
def aggregate_pool_ohlcv(raw_swaps, freq='1min'):
    """
    Aggregate swap-level data to 1-minute OHLCV candles, per pool.

    Per pool, per minute:
        open  = USD price of first swap in bucket
        high  = max USD price
        low   = min USD price
        close = USD price of last swap
        volume = sum of abs(amountUSD)

    Empty buckets: forward-fill close, volume=0.
    """

# === Pass 2: Merge pools → ohlcv table (canonical per-token view) ===
def merge_pool_ohlcv(ohlcv_pool, primary_pool_map):
    """
    Merge per-pool candles into one canonical candle per token per minute.

    Per token, per minute:
        open/high/low/close = from primary pool (highest TVL that week)
        volume = SUM across ALL pools for this token
        n_pools = COUNT of active pools with trades this minute

    Primary pool selection:
        - Recalculate weekly: pool with highest swap volume = primary
        - Handles V2→V3 migration, pool launches, liquidity shifts

    Quality filters applied per-candle:
        - Price jump > 50% in 1 min AND volume < $1K → mark suspect
        - Cross-pool price divergence > 5% → flag (interesting, not invalid)
        - Keep everything — flags are metadata, not filters
    """
```

### Step 6: Rainbow Table Generation

After OHLCV is built, generate the training schedule using the existing `map_training_data.py` logic:

```python
# 1. Scan ohlcv table: for each hour, count distinct assets with data
# 2. Find continuous periods with ≥64 active assets
# 3. For each period: generate N rotated 64-asset combinations
#    - 30-50% overlap between consecutive combinations
#    - Volume-weighted sampling (liquid tokens appear more often)
#    - But tail tokens still appear (just less frequently)
# 4. Output: training_schedule.json compatible with dojo_v3_historical.py

# Expected multiplier:
# 500 tokens available → ~15-20 unique 64-subsets per period
# 2000 tokens → ~50+ unique 64-subsets per period
# Each subset is a different "market view" of the same time range
```

### Step 7: Validation

```python
# Run tools/map_training_data.py on the new database
# Check:
# - Total time coverage (months/years of continuous data)
# - Asset count over time (should grow from 2021 onward)
# - Rainbow table size (total training blocks)
# - Coverage gaps (missing minutes, dead zones)
# - Price sanity: spot-check known events (LUNA crash, FTX, etc.)
```

## Event Signatures Reference

```python
EVENTS = {
    # Uniswap V3
    'swap_v3': '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67',
    'pool_created_v3': '0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118',
    'mint_v3': '0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde',
    'burn_v3': '0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c',

    # Uniswap V2
    'swap_v2': '0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822',
    'pair_created_v2': '0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9',
    'sync_v2': '0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1',

    # ERC20
    'transfer': '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef',
    'approval': '0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925',
}
```

## DeFi-Specific Features (Bella Skeleton Extensions)

On-chain data provides conditioning signals beyond OHLCV:

| Feature | Source | Bella Skeleton Mapping |
|---------|--------|----------------------|
| Pool TVL over time | Mint/Burn events | `vol_regime` (liquidity = stability) |
| Fee tier distribution | Pool metadata | Spread/microstructure conditioning |
| MEV sandwich rate | Same-block patterns | `events` (adversarial flow regime) |
| New token launches | Factory events | `events` (listing = volatility spike) |
| Liquidity migrations | TVL drops + new pool creates | `regime_sequence` (regime shifts) |
| Gas price (basefee) | Block headers | Trading cost regime |
| Whale concentration | Large swap detection | `correlation_clusters` |

## Estimated Disk & Time

### On NVMe (ideal)

| Component | Size | Time |
|-----------|------|------|
| Reth archive (ETH) | ~2.5 TB | 3 days initial sync |
| Raw swaps (all pools, all time) | ~30-40 GB | 4-8 hours extraction |
| Token metadata | ~50 MB | 30 min |
| OHLCV per-pool + merged | ~80-130 GB | 2-4 hours |
| Training schedule (rainbow) | ~10 MB | 10 min |

### On HDD (our setup — 5.1 TB free)

| Component | Size | Time |
|-----------|------|------|
| Reth archive (ETH) | ~2.5 TB | **2-4 weeks** (random I/O bottleneck during sync) |
| Raw swaps extraction | ~30-40 GB | 8-16 hours (sequential reads, HDD OK) |
| OHLCV build | ~80-130 GB | 4-8 hours (sequential write, HDD OK) |
| **Total HDD used** | **~2.7 TB** | **~2.4 TB remaining** |

Note: Erigon (~1.5 TB) may sync faster on HDD than Reth due to its flat-file download phase. Consider if Reth sync stalls. Both support the same `eth_getLogs` extraction queries.

## Tech Stack

```
Node:
├── reth                 — Ethereum execution client (archive mode)
├── lighthouse           — Consensus client
└── systemd services     — Auto-restart on crash

Pipeline:
├── web3.py              — RPC interface + ABI decoding
├── polars               — Fast aggregation (handles 2B+ rows)
├── sqlite3              — Final storage (matches existing Bella pipeline)
├── asyncio + aiohttp    — Parallel RPC batching
└── tqdm                 — Progress tracking
```

## Execution Order

### Phase 1: Sync (background, 2-4 weeks on HDD)
1. **`defi_node_setup.sh`** — Install + configure Reth archive node on HDD, systemd service
   - Target: `/mnt/hdd/reth/` (or wherever HDD is mounted)
   - Set `--datadir` to HDD path
   - Consensus client (Lighthouse) can share the same HDD
   - Monitor: `reth node --status` shows sync progress

### Phase 2: Extract (8-16 hours, after sync completes)
2. **`defi_pool_discovery.py`** — Scan factory events → `config/defi_pools.json` (all pools ever created)
3. **`defi_swap_downloader.py`** — Bulk download all swap events → `raw_swaps.db`
4. **`defi_token_resolver.py`** — Resolve symbol/name/decimals for all tokens
5. **`defi_ohlcv_builder.py`** — Per-pool OHLCV, then merge to canonical per-token view, USD pricing, quality flags
6. **`defi_rainbow_builder.py`** — Generate rotating 64-asset training schedule (uses `map_training_data.py` logic)
7. **Validate** — Run `tools/map_training_data.py` to verify coverage

### Phase 3: Freeze
[redacted: host service detail]
   - Chain data stays on HDD as cold archive (~2.5 TB)
   - Can re-start later for incremental updates if needed
   - Extracted SQLite is self-contained — no node needed for training

## Key Gotchas

- **sqrtPriceX96 is uint160** — Use Python `int`, never `float`. Float64 loses precision above 2^53.
- **Token decimals vary** — WBTC=8, USDC=6, DAI=18. Always normalize.
- **Reorgs** — Ethereum can reorg (rare). Re-download last 64 blocks on incremental updates.
- **MEV bots are 60%+ of volume** — This is REAL market structure. Keep it. Don't filter it.
- **Dead tokens are data** — A token going to zero is valuable training signal. Never filter by "current" status.
- **Wrapped tokens** — WETH ≠ ETH on-chain but same price. Map both to same asset_id.
- **Fee-on-transfer tokens** — Some ERC20s take a fee on transfer. Amounts in swap events may not match. Flag but keep.
- **Block timestamp monotonicity** — Ethereum timestamps are not strictly monotonic (miners could set ±15s). Sort by block_number, use timestamp only for minute-bucketing.
- **HDD sync patience** — Reth on HDD will appear to stall during state-heavy phases. This is normal — random IOPS are 100x slower than NVMe. Don't kill and restart. Monitor with `reth node --status`.
- **Cross-pool price divergence** — Same token can show 0.1-5% price difference across pools. This is REAL (arb latency, fee differences, MEV). Flag but never "fix" by averaging. The divergence IS the signal.
- **Primary pool changes** — A token's most-liquid pool shifts over time (V2→V3 migration 2021, fee tier shifts). Recalculate primary weekly, not globally. Historical primary ≠ current primary.
- **Pool death** — Pools can lose all liquidity. When a pool dies, its OHLCV stops (no forward-fill). The token may still live in other pools. The `ohlcv` merged table handles this by falling back to next-best pool.

## Output

Write your implementation to `{{OUTPUT_DIR}}/`:
- `defi_node_setup.sh` — Reth + Lighthouse install + config (HDD-targeted, systemd services)
- `defi_pool_discovery.py` — Factory event scanner → pool registry
- `defi_swap_downloader.py` — Bulk swap downloader (resumable, per-pool)
- `defi_token_resolver.py` — Metadata resolution (symbol, name, decimals, flags)
- `defi_ohlcv_builder.py` — Two-pass: per-pool OHLCV → merged canonical view, USD pricing, quality flags
- `defi_rainbow_builder.py` — Training schedule with rotating assets (uses `map_training_data.py` logic)
- `config/defi_pools.json` — Pool registry (auto-generated)
- `REPORT.md` — Decisions, coverage stats, known gaps, data quality metrics, cross-pool divergence stats

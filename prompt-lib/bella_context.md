# Bella: Market Data Generator — Complete Context

> **Purpose**: Standalone reference for projects that produce data for Bella or consume Bella's output. No access to the Trader codebase required.

---

## What Is Bella

Bella is a **synthetic market data generator** that produces realistic multi-asset OHLCV price sequences. It uses **Conditional Flow Matching** (OT-CFM) conditioned on a **MarketSkeleton** — a normalized descriptor of market structure (regimes, trends, volatility, correlations, events).

**Pipeline:**
```
Real OHLCV data (SQLite)
    → Skeleton extraction (trend, vol, clusters, regimes, events)
    → Skeleton encoder (FiLM conditioning: gamma, beta, h0)
    → Flow matching generator (ODE integration from noise → prices)
    → Synthetic OHLCV (n_samples, seq_length, n_assets, n_features)
```

Bella is named after belladonna (the eye medicine) — it gives vision of unseen markets.

---

## Data Input: What Bella Expects

### SQLite Database Schema

Bella reads from a SQLite database (default: `trading_data.db`). Two tables required:

```sql
-- Assets: one row per tradeable instrument
CREATE TABLE assets (
    id INTEGER PRIMARY KEY,
    symbol TEXT,              -- e.g. "WETH/USD", "BTC/USD"
    is_active INTEGER,        -- 1 = include in training, 0 = skip
    -- Optional additional columns are ignored
);

-- OHLCV: one row per (timestamp, asset) pair
CREATE TABLE ohlcv (
    timestamp INTEGER,        -- Unix seconds (60-second intervals for 1min candles)
    asset_id INTEGER,         -- Foreign key → assets.id
    open REAL,                -- USD price
    high REAL,
    low REAL,
    close REAL,
    volume REAL               -- USD volume
);

-- Required indexes for query performance
CREATE INDEX idx_ohlcv_ts ON ohlcv(timestamp);
CREATE INDEX idx_ohlcv_asset ON ohlcv(asset_id, timestamp);
```

### Field Order (Sacred)

OHLCV fields are always in this order, index 0-4:

| Index | Field | Used For |
|-------|-------|----------|
| 0 | open | Window baseline normalization |
| 1 | high | Range/volatility |
| 2 | low | Range/volatility |
| 3 | close | **Primary** — returns, trends, regime detection, skeleton extraction |
| 4 | volume | Liquidity signal, quality filtering |

### Data Quality Requirements

- **Timestamps**: Unix seconds, 60-second intervals (1-minute candles). Must be monotonically increasing. Gaps are OK (Bella handles missing minutes).
- **Prices**: USD-denominated, positive. Zeros mean "no data for this asset at this time" — not a zero price.
- **Volume**: USD volume. Zero volume = no trades in that minute (valid, just forward-fill the previous close).
- **Missing data**: Bella tolerates sparse data. Assets don't need data at every timestamp. The loader counts distinct assets per timestamp and filters for minimum coverage.

### How Bella Loads Data

#### Simple Loader (MarketDataLoader)

1. **Load top N assets** by data count (most data points = most reliable):
   ```sql
   SELECT a.id, a.symbol, COUNT(o.timestamp) as cnt
   FROM assets a JOIN ohlcv o ON a.id = o.asset_id
   WHERE a.is_active = 1
   GROUP BY a.id ORDER BY cnt DESC LIMIT ?
   ```
   Default: N=64 assets. Delisted assets excluded by hardcoded set.

2. **Load valid timestamps** where enough assets have data:
   ```sql
   SELECT timestamp FROM ohlcv
   WHERE asset_id IN (...) AND timestamp >= ? AND timestamp <= ?
   GROUP BY timestamp
   HAVING COUNT(DISTINCT asset_id) >= ?
   ORDER BY timestamp ASC
   ```
   Default minimum: 50 assets per timestamp.

3. **Load all OHLCV** into a single RAM matrix:
   ```sql
   SELECT timestamp, asset_id, open, high, low, close, volume
   FROM ohlcv
   WHERE timestamp >= ? AND timestamp <= ? AND asset_id IN (...)
   ```
   Result: numpy array `(n_timesteps, n_assets, 5)`, dtype float32.

#### Rainbow Table Loader (RainbowTableLoader)

For large datasets with rotating asset windows. Uses a pre-computed `training_schedule.json`:

```json
{
  "periods": [
    {
      "start": 1620000000,
      "end": 1635000000,
      "duration_days": 173,
      "training_runs": [
        {
          "assets": ["WETH/USD", "WBTC/USD", "UNI/USD", ...],
          "n_assets": 64
        },
        {
          "assets": ["WETH/USD", "LINK/USD", "AAVE/USD", ...],
          "n_assets": 64
        }
      ]
    }
  ],
  "summary": {
    "total_periods": 12,
    "total_training_years": 5.3,
    "total_training_runs": 240
  }
}
```

Each period has multiple "training runs" — different 64-asset subsets of the same time range, with 30-50% overlap between runs. This creates massive data augmentation (same time, different asset views).

The rainbow loader is preferred for large datasets (50M+ windows vs ~200 with strict 64-asset requirement).

---

## MarketSkeleton: Conditioning System

A MarketSkeleton is a normalized, multi-faceted descriptor of market conditions. All values are dimensionless — the generator learns to render them at proper scale.

### Components

| Component | Shape | Values | Meaning |
|-----------|-------|--------|---------|
| `trend_direction` | `(n_timesteps, n_assets)` | `{-1, 0, +1}` | Bearish / sideways / bullish per asset per step |
| `vol_regime` | `(n_timesteps,)` | `[0, 1]` | Volatility level. 0=calm, 1=extreme |
| `correlation_clusters` | `(n_assets,)` | `0..K-1` (K=5 default) | Correlation-based grouping. Assets in same cluster move together |
| `alpha_shape` | `(n_timesteps,)` | `[0, 1]` | Cumulative optimal return shape (normalized) |
| `regime_sequence` | `(n_timesteps,)` | `0..4` | Per-timestep regime classification |
| `events` | List of Event | time, type, magnitude, direction | Sparse discrete events (spikes, crashes, gaps) |

### Regime Types

| Index | Name | Detection Rule |
|-------|------|---------------|
| 0 | `crash` | Negative returns + high volatility |
| 1 | `rally` | Positive returns + moderate volatility |
| 2 | `sideways` | Neither trending nor volatile (default) |
| 3 | `volatile` | High volatility + neutral returns |
| 4 | `calm` | Low volatility |

### Skeleton Extraction from OHLCV

Given raw OHLCV prices `(n_timesteps, n_assets, 5)`:

1. **Log returns**: `log(close_t / close_{t-1})` — all analysis on close prices (index 3)
2. **Trend**: Rolling mean of returns (window=20), thresholded at ±0.5σ → {-1, 0, +1}
3. **Volatility**: Rolling std of returns (window=20), average across assets, normalize to [0,1]
4. **Clusters**: K-means (K=5) on correlation matrix rows (10 iterations)
5. **Alpha**: Max absolute return per step → cumulative sum → normalize to [0,1]
6. **Regimes**: Classify each timestep by rolling return percentile + rolling vol percentile
7. **Events**: Z-score > 3σ in average cross-asset returns → Event objects

Note: Returns reduce array length by 1 (diff), so skeleton arrays have length `n_timesteps - 1`.

### Skeleton Builder (Manual Construction)

Fluent API for creating skeletons without real data:

```python
skeleton = SkeletonBuilder() \
    .set_timesteps(1440) \
    .set_assets(64) \
    .add_regime('sideways', duration=100) \
    .add_regime('crash', duration=50) \
    .add_regime('recovery', duration=200) \
    .set_volatility_percentile(0.8) \
    .set_correlation_mode('crisis') \
    .add_event(150, 'flash_crash', magnitude=0.9) \
    .build()
```

---

## Skeleton Encoder: FiLM Conditioning

The skeleton is encoded into **FiLM parameters** (Feature-wise Linear Modulation) that condition the generator:

```
MarketSkeleton
    → TrendEncoder (Embed {-1,0,+1} → d_model, mean over assets)
    → VolRegimeEncoder (Dense layers on continuous [0,1])
    → ClusterEncoder (Embed cluster IDs → d_model, mean over assets)
    → AlphaShapeEncoder (1D conv on cumulative shape)
    → RegimeEncoder (Embed regime indices)
    → EventEncoder (Sparse event embedding, max 32 events)
    → Combine all → (gamma, beta, h0)
```

Output: `gamma` (scale), `beta` (shift) of shape `(d_model,)` or `(n_timesteps, d_model)`, plus `h0` (initial state for recurrent architectures). Default `d_model = 256`.

---

## Generation Pipeline

### Architecture Options

| Architecture | Memory | Parameters | Best For |
|-------------|--------|------------|----------|
| **Conv** (default) | O(n) | ~302M (hidden=1536, 16 layers) | Long sequences (1440+ steps), limited VRAM |
| **Mamba** | O(n) | Smaller | SSM-based alternative, BiMamba + sparse attention |

### Flow Matching (OT-CFM)

Optimal Transport Conditional Flow Matching:
- **Forward**: Straight-line interpolation from data to noise: `x_t = (1-t) * x_0 + t * noise`
- **Training**: Predict velocity field `v(x_t, t, skeleton)`
- **Generation**: Integrate backward from noise to data (Euler method, 10 steps default)

```
noise ~ N(0, I)                          # Shape: (seq_length, n_assets, n_features)
for step in range(n_steps):              # 10 steps default
    t = 1.0 - step / n_steps
    velocity = model(x_t, t, skeleton)   # Conditioned on skeleton via FiLM
    x_t = x_t - velocity * dt
return x_t                               # Generated OHLCV prices
```

### Generation API

```python
generator = MarketGenerator.from_checkpoint("checkpoints/model")

# Unconditional
prices = generator.generate(n_samples=10, seed=42)
# → (10, seq_length, 64, 5)

# Regime-conditioned (simple)
prices = generator.generate(n_samples=10, regime='crash', seed=42)

# Skeleton-conditioned (full control)
skeleton = SkeletonBuilder().set_regime_sequence(['crash', 'recovery']).build()
prices = generator.generate(n_samples=10, skeleton=skeleton, seed=42)

# On-device (stays in VRAM, no CPU roundtrip)
prices_jax = generator.generate_on_device(n_samples=10, skeleton=skeleton, seed=42)
```

---

## Output Format

### Generated Data Shape

```
(n_samples, seq_length, n_assets, n_features)
= (n_samples, 400, 64, 5)     # Default config

Feature order: [open, high, low, close, volume]  — same as input
```

### Normalization

Bella outputs **normalized prices** (relative to first timestep):

```python
# Normalization (applied before feeding to generator)
baseline = prices[:, 0:1, :, :]       # First timestep
normalized = prices / baseline - 1.0   # Relative change
normalized = clip(normalized, -10, 10)  # Guard extreme values

# Denormalization (after generation)
reconstructed = (normalized + 1.0) * baseline
```

A value of `0.0` means "same as opening price". A value of `0.05` means "5% above opening".

---

## Downstream Consumers

### Foxglove (Phase 1 — Supervised Imprinting)

Foxglove is the training system that uses Bella-generated data to train a TRM (Tiny Recursive Model):

```python
# Per epoch: generate fresh synthetic data in VRAM
bella_gen, config = setup_bella_generator("checkpoints/bella_model")
key = jax.random.PRNGKey(epoch)
prices = generate_bella_prices_on_device(bella_gen, key, n_envs=1)
# → (n_steps, 64) on GPU — no disk I/O
```

Foxglove also uses a synthetic curriculum (L0-L14) that generates markets procedurally — Bella provides the "real data" branch while synthetic provides controlled difficulty.

### Poppy (Phase 2 — MuZero + MCTS)

Production training system. Bella provides the environment dynamics that Poppy's MCTS explores. Same data format, higher throughput requirement.

---

## Configuration Reference

### TeacherConfig (Bella Training)

```python
# Data
db_path: str = "trading_data.db"
n_assets: int = 64
window_size: int = 400            # Steps per training window
features: Tuple = ("open", "high", "low", "close", "volume")
train_start: str = "2020-01-01"
train_end: str = "2024-12-31"

# Encoder
latent_dim: int = 256
encoder_heads: int = 4
encoder_layers: int = 4

# Flow Matching
flow_architecture: str = "conv"   # or "mamba"
flow_hidden_dim: int = 1536       # 302M params for conv
flow_layers: int = 16
sigma_min: float = 1e-4

# Training
batch_size: int = 32
learning_rate: float = 1e-4
max_steps: int = 100000
```

### GeneratorConfig (Bella Inference)

```python
architecture: str = "conv"
hidden_dim: int = 768
num_layers: int = 12
d_state: int = 64                 # Mamba state dim

n_assets: int = 64
n_features: int = 5               # OHLCV
seq_length: int = 1440            # Max generation length

n_steps: int = 10                 # ODE integration steps
sigma_min: float = 1e-4
```

---

## Quality Metrics (Stylized Facts)

Bella's output quality is measured against known statistical properties of real financial markets:

| Metric | What It Measures | Good Value |
|--------|-----------------|------------|
| Fat tail index (Hill estimator) | Heavy tails in return distribution | 2-4 (power law) |
| Volatility clustering (ACF of |r|²) | Persistence of volatile periods | Positive, decaying |
| Leverage effect | Negative correlation between returns and future vol | Negative |
| Autocorrelation lag-1 | Short-term mean reversion/momentum | Near 0, slightly negative |
| Skewness | Asymmetry of returns | Slightly negative |
| Kurtosis | Tail heaviness | > 3 (leptokurtic) |
| Wasserstein distance | Distribution match to real data | Lower = better |
| Correlation matrix error (Frobenius) | Cross-asset dependency match | Lower = better |

---

## DeFi Data Pipeline → Bella Integration

When building a DeFi on-chain data pipeline, the final output must produce:

1. **SQLite database** with `assets` and `ohlcv` tables matching the schema above
2. **`is_active` column** in assets table (set to 1 for tokens to include)
3. **1-minute candles** in `ohlcv` — timestamp as Unix seconds, prices in USD
4. **Consistent asset IDs** — each token gets one `asset_id`, even if it trades across multiple pools
5. **Optional**: `training_schedule.json` for rainbow table rotation (if >64 tokens)

### DeFi-Specific Considerations

| DeFi Concept | Bella Mapping |
|-------------|---------------|
| Token = asset | One row in `assets`, `symbol` = e.g. "WETH" |
| Multiple pools per token | Merge to canonical price (primary pool) + sum volume |
| Pool TVL | Can inform `vol_regime` conditioning (high TVL = stability) |
| Fee tier | Not directly used, but affects price microstructure |
| Token births/deaths | Natural `first_seen`/`last_seen` — Bella handles varying asset availability |
| MEV/arb activity | Captured in volume and price noise — don't filter out |
| Missing minutes | OK — forward-fill close, volume=0. Sparse tokens are expected. |

### What Matters Most

1. **USD prices must be correct** — wrong decimals, wrong price routing = garbage data
2. **Close prices are king** — all skeleton extraction, regime detection, return computation uses close (index 3)
3. **Volume is secondary but useful** — quality filtering, liquidity weighting
4. **Coverage > precision** — better to have noisy data for 2000 tokens than perfect data for 50
5. **Timestamps must be aligned** — all assets in the same minute bucket share the same Unix timestamp

---

## Tech Stack

```
Bella (diff_teacher/):
├── JAX + Flax           — Model architecture, JIT compilation
├── SQLite               — Data storage (no server, portable)
├── numpy                — Data loading, preprocessing
├── polars               — Large-scale aggregation (optional, for rainbow table building)
└── jax.random           — Stochastic generation (reproducible seeds)
```

---

*This document describes Bella as of 2026-02-23. Source: `diff_teacher/` in the Trader repository.*

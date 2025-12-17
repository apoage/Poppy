# FX Database Builder Agent Instructions

## Objective
Create a robust, standalone Python script (`tools/build_fx_database.py`) to generate a clean `fx_rates` table in `trading_data.db`. This table will serve as the single source of truth for FX rates (e.g., EUR/USD, GBP/USD) needed for portfolio valuation.

## Core Philosophy
1.  **Strict Source of Truth**: Do not guess. Do not forward-fill blindly. Use real data.
2.  **Consensus over Noise**: When deriving FX rates from crypto bridges (e.g., BTC/USD and BTC/EUR), use a **Weighted Consensus** mechanism to filter out anomalies from illiquid pairs.
3.  **Preserve Market Reality**: Do not smooth out real market events (de-pegs, crashes) if they are corroborated by multiple sources.
4.  **Fail Fast on Gaps**: If no data exists, record a gap. Do not hallucinate data.

## Implementation Details

### 1. Database Schema
Ensure the `fx_rates` table exists:
```sql
CREATE TABLE IF NOT EXISTS fx_rates (
    timestamp INTEGER,
    symbol TEXT,       -- e.g., 'EUR', 'GBP', 'USDT'
    rate REAL,         -- The price in USD
    source TEXT,       -- 'DIRECT', 'CONSENSUS', 'SINGLE_BRIDGE'
    PRIMARY KEY (timestamp, symbol)
);
```

### 2. Rate Derivation Logic (Per Timestamp, Per Currency)

**Step A: Direct Check**
- Check `ohlcv` for the direct pair (e.g., `EUR/USD`).
- If found -> Use it. Source = 'DIRECT'.

**Step B: Bridge Candidates (If Direct is Missing)**
- Calculate candidate rates using all available bridges:
    - `Rate_BTC = Price(BTC/USD) / Price(BTC/EUR)`
    - `Rate_ETH = Price(ETH/USD) / Price(ETH/EUR)`
    - ... (SOL, XRP, ADA)

**Step C: Weighted Consensus**
- **Cluster** the candidates. (Candidates within 0.1% of each other belong to the same cluster).
- **Score** each cluster using these weights:
    - BTC: 1.0
    - ETH: 1.0
    - SOL: 0.7
    - XRP: 0.5
    - ADA: 0.5
- **Select** the cluster with the highest total score.
- **Value**: Use the weighted average of the winning cluster.
- Source = 'CONSENSUS' (or 'SINGLE_BRIDGE' if only one existed).

### 3. Artifact Handling
- **Delta Check**: Calculate the % change from the previous timestamp's rate.
- **Log Conflicts**: If bridges disagree significantly (winning cluster score < total possible score * 0.5), log this to `logs/fx_conflicts.csv`.

### 4. Technical Requirements
- **Performance**: Load data in chunks or optimized pandas DataFrames to handle millions of rows.
- **Logging**: Provide clear progress bars and summary stats (Gaps found, Conflicts found).
- **Safety**: Do not overwrite existing 'DIRECT' data if re-run (unless specified).

## Output
- Populated `fx_rates` table.
- `logs/fx_conflicts.csv`: Timestamp, Symbol, Rate_BTC, Rate_ETH, ... (for debugging).
- `logs/fx_gaps.csv`: Intervals where no data could be found.

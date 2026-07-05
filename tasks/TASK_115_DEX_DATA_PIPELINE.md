# TASK_115: Decentralized Exchange (DEX) Data Pipeline

**Status**: Proposed
**Priority**: High (Legal Blocker for Open Sourcing)
**Assignee**: Lukas & Gemini
**Dependencies**: None (Parallel track)

---

## 1. The Objective
**Goal**: Build a sovereign, legally unencumbered data pipeline by reading directly from the blockchain (Ethereum/Solana).
**Why**: Current data (Kraken/Yahoo) is subject to "No Redistribution" licenses, making it illegal to publish the trained weights of **Bella** (Market Generator).
**Solution**: Blockchain data is public domain. By processing raw `Swap` events from Decentralized Exchanges (DEXs), we create a proprietary dataset that we fully own and can open-source.

---

## 2. The Architecture: "The Harvester"

We need a system that drinks from the firehose of the blockchain without relying on centralized APIs (like Etherscan) that have their own ToS.

```mermaid
graph TD
    A[Blockchain (Eth/Sol)] -->|RPC / Light Node| B(The Harvester)
    B -->|Raw Logs| C{Event Filter}
    C -->|Swap(sender, amount0, amount1)| D[MEV Sanitizer]
    D -->|Clean Price| E[OHLCV Aggregator]
    E -->|Time Bars| F[(sovereign_data.db)]
```

### 2.1 Layer 1: Node Connectivity (Multi-Node Strategy)
To avoid rate limits and ensure uptime without paying $500/mo for enterprise RPCs.
*   **Multi-Provider Round Robin**: Rotate requests between Alchemy, Infura, QuickNode, Ankr, and public RPCs.
*   **Local Light Node (Optional)**: Run a local Geth/Erigon node (headers only) for verification, fetching bodies from remote peers.
*   **Architecture**: `AsyncRPCManager` class that handles retries, rate-limit backoffs, and provider health checks.

### 2.2 Layer 2: The Parser (Uniswap V3 Focus)
DEXs don't give candles; they give events. We must mathematically reconstruct the price.
*   **Target**: Uniswap V3 (Concentrated Liquidity) provides the highest fidelity price signal.
*   **Event**: `Swap(sender, recipient, amount0, amount1, sqrtPriceX96, liquidity, tick)`
*   **Challenge**: Uniswap V3 math (`sqrtPriceX96`) is complex but deterministic.
*   **Output**: Stream of `(timestamp, price, volume)` tuples.

### 2.3 Layer 3: MEV Sanitizer (The "Alien" Edge)
DEX data is noisy due to **MEV (Maximal Extractable Value)** bots.
*   **Sandwich Attacks**: A bot buys before a user and sells immediately after. This creates fake "Highs" and "Lows" in a single block.
*   **Arbitrage**: Atomic back-and-forth trades.
*   **Solution**: "Block-End Pricing" or "Median Block Price." We filter out txs within the same block that revert price impact.
*   **Result**: Cleaner data than CEXs (which hide the order book mechanics).

---

## 3. Implementation Plan

### Phase 1: The Prototype (Python + Web3.py)
*   **Goal**: Extract 1 week of ETH/USDC data from Uniswap V3.
*   **Tools**: `web3.py`, `polars` (for fast processing).
*   **Steps**:
    1.  Identify Pool Contract Address (USDC/ETH 0.05%).
    2.  Fetch `Swap` logs for block range.
    3.  Decode `sqrtPriceX96` to human-readable price.
    4.  Aggregate to 1-minute OHLCV.
    5.  Compare against Kraken data (Ground Truth validation).

### Phase 2: The Scaler (Async Multi-Node)
*   **Goal**: Parallel download of 2018-2026 history.
*   **Logic**:
    *   Split block range into chunks (e.g., 10,000 blocks).
    *   Workers fetch chunks in parallel using different RPC endpoints.
    *   Store raw logs in Parquet (efficient compression).

### Phase 3: The Factory (Pipeline Integration)
*   **Goal**: Feed `sovereign_data.db` into `MarketDataLoader`.
*   **Deliverable**: A new loader class `DEXDataLoader` that treats the local Parquet files as the source of truth.

---

## 4. Technical Constraints & Requirements

| Component | Requirement | Note |
|-----------|-------------|------|
| **Storage** | ~50GB for full history | Parquet is 10x smaller than CSV |
| **RAM** | 32GB+ | For processing large block chunks |
| **Network** | High Bandwidth | Downloading millions of logs |
| **Math** | High Precision | `decimal` or `BigDecimal` for V3 math |

---

## 5. First Steps (Action Items)

1.  [ ] **RPC Setup**: Create free accounts on Alchemy/Infura/QuickNode to build the rotation pool.
2.  [ ] **Contract Mapping**: List top 50 pools by volume (ETH, WBTC, LINK, UNI, etc.) on Uniswap V3.
3.  [ ] **Decoder Script**: Write a standalone script `tools/decode_uniswap_v3.py` to test math accuracy.
4.  [ ] **Validation**: Plot DEX vs CEX price for a volatile day (e.g., FTX crash) to see the difference.

---

## 6. The "Sovereign" Advantage
Once completed, this pipeline provides:
1.  **Legal Safety**: 100% Public Domain data.
2.  **Open Source**: We can host the dataset on HuggingFace.
3.  **Resilience**: No API keys to revoke, no exchange delistings.
4.  **Truth**: The blockchain never lies; CEXs do.

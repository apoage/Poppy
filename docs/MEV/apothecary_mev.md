# Aconite: Apothecary MEV System

> **Aconite** (monkshood/wolfsbane) — the fastest-acting medicine in the apothecary's cabinet. Micro-doses relieve pain; precision is everything. Microgram separates cure from kill.
>
> In our system: precision extraction of value from transaction ordering, where milliseconds and intelligence separate profit from nothing.

---

## System Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ACONITE (MEV System)                         │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │  Bella   │   │  Poppy   │   │ Foxglove │   │   Iris-DEX     │  │
│  │ (Vision) │   │ (Search) │   │ (Learn)  │   │ (Raw Material) │  │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └───────┬────────┘  │
│       │              │              │                  │           │
│       ▼              ▼              ▼                  ▼           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Aconite Core Pipeline                      │  │
│  │                                                              │  │
│  │  Observe ──→ Predict ──→ Search ──→ Construct ──→ Execute   │  │
│  │  (mempool)   (Bella)    (MCTS)     (bundle)     (Flashbots) │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Roles in Aconite

| Component | Trader Role | Aconite Role |
|-----------|-------------|--------------|
| **Bella** | Generate synthetic OHLCV | **Regime predictor** — forecast volatility transitions, liquidation cascades, divergence windows |
| **Poppy** (TRM+MCTS) | Trade decisions via search | **Bundle optimizer** — MCTS explores multi-hop routes, sequencing, sizing |
| **Foxglove** | Supervised oracle imitation | **Strategy distillation** — learn from historical MEV patterns (historical oracle = "what would have been profitable") |
| **Iris-DEX** | On-chain data pipeline | **Raw material** — archive node, pool-level swap data, state history |
| **Aconite** | (new) | **Orchestrator** — mempool monitor, opportunity detector, bundle constructor, Flashbots submission |

---

## Why This Architecture Works

### The Intelligence Thesis

Most MEV bots are **reactive state machines**: event → rule → action. No memory, no prediction, no planning.

Aconite is a **predictive planning agent**: history → regime model → opportunity forecast → multi-step search → optimal bundle.

```
                    Reactive Bot                    Aconite
                    ────────────                    ───────
Sees:               Current mempool tx              Mempool + regime context + historical patterns
Decides:            If profitable, execute           MCTS explores bundle space conditioned on prediction
Remembers:          Nothing                         Portfolio state, regime history, strategy performance
Plans ahead:        No                              Multi-block positioning, cascade prediction
Adapts:             Fixed rules                     Continuous learning from outcomes
```

### TRM Inference Speed

The TRM is not slow. The 1-minute candle is training granularity, not inference ceiling:

| Operation | Latency | Fits MEV Budget? |
|-----------|---------|-------------------|
| TRM forward pass | 1-5ms | Yes (500ms budget) |
| MCTS 32 simulations | 50-150ms | Yes |
| MCTS 128 simulations | 200-600ms | Tight but feasible |
| Bella regime inference | 5-20ms | Yes |
| Full Aconite pipeline | ~100-300ms | Within 500ms budget |

The bottleneck is never inference. It's **what the model was trained to see** — which is why data resolution (per-block vs per-minute) matters more than model speed.

---

## Strategy Tiers

### Tier 1: Intelligence-Dominant (Our Strength)

These strategies reward prediction and planning over raw speed. Aconite's primary targets.

#### 1A. Predictive Liquidation Positioning

```
Bella observes: vol_regime transitioning from calm → volatile
Bella predicts: ETH price drop within next 5-20 minutes

Aconite scans:  All Aave/Compound/Morpho borrow positions
                Finds 47 positions with health_factor < 1.10
                Pre-computes liquidation bundles for each

Price drops:    23 positions become liquidatable
Aconite submits: Pre-built bundles in same block as oracle update
                 Captures liquidation bonus on 15 of 23 (others contested)

Reactive bot:   Sees oracle update, starts computing, misses 12 of 23
```

**Edge**: Minutes of advance preparation vs seconds of reactive computation. Pre-built bundles submit faster because simulation is already done.

**Revenue**: Liquidation bonuses (5-15% of collateral). Lumpy — $0 most days, $10K+ during crashes.

#### 1B. Cyclic Arbitrage (4+ Hops)

```
Pool universe: ~5,000 active pools on Ethereum
2-hop arb:     Trivial — thousands of bots scan these. Commoditized.
3-hop arb:     Competitive — dozens of bots. Margins thin.
4+ hop arb:    Search space explodes: C(5000,4) ~ 2.6 × 10^14 paths
               Brute force impossible. Greedy heuristics miss non-obvious paths.

MCTS approach:
  State:   Current pool reserves + path so far
  Action:  Next pool to route through + amount
  Reward:  Net output after full cycle minus gas
  Search:  32-128 simulations guided by Bella's liquidity predictions

Result:    Finds profitable cycles through low-liquidity pools that
           greedy searchers skip. Lower competition, higher margin.
```

**Edge**: MCTS + neural prior explores intelligently where brute force fails.

**Revenue**: $50-500 per cycle, dozens per day in active markets. 30-70% margin (vs 5-15% for 2-hop).

#### 1C. Regime-Aware MEV Density Prediction

```
MEV density varies 50x between calm and volatile periods.
Most bots run identically in all regimes — wasting compute in calm,
under-prepared for volatile.

Aconite approach:
  Bella continuously classifies regime: {calm, sideways, volatile, rally, crash}

  calm period:
    - Scale down monitoring frequency (save compute)
    - Focus on long-tail strategies
    - Pre-compute contingency bundles for regime transitions

  volatile period detected:
    - Scale UP everything: more MCTS simulations, more pool monitoring
    - Activate liquidation scanner
    - Activate cross-protocol cascade detector
    - Switch from patient to aggressive bundle tipping

  crash predicted (Bella regime=crash with >60% confidence):
    - Full liquidation preparation mode
    - Pre-position for cascade: first liquidation → arb → second liquidation
    - Pre-build compound bundles (liquidate + arb in same bundle)
```

**Edge**: Adaptive compute allocation. Most bots waste fixed resources uniformly. Aconite concentrates firepower when opportunities are densest.

#### 1D. CEX-DEX Temporal Arbitrage

```
Existing Kraken data (Poppy's training source) + Iris-DEX on-chain data
= lead-lag structure between centralized and decentralized prices.

Kraken price moves → DEX price follows with 1-30 second delay.
The delay varies by:
  - Token (BTC/ETH: 1-3s, altcoins: 5-30s)
  - Time of day (US session vs Asia session)
  - Volatility regime (higher vol = longer delay)

Bella learns these lead-lag patterns from historical data.
At inference time:
  1. Kraken price moves (detected via WebSocket)
  2. Bella estimates: "DEX will adjust within X seconds, magnitude Y"
  3. Aconite pre-builds swap bundle at predicted DEX adjustment price
  4. Submits bundle targeting the block where adjustment occurs

This is the HIGHEST revenue strategy but requires:
  - Inventory on both CEX and DEX (capital-intensive)
  - Fast CEX API connection (Kraken WebSocket)
  - Precise latency modeling (Bella's job)
```

**Revenue**: $1,000-10,000/day at scale. Requires $100K-1M inventory.

### Tier 2: Hybrid Speed+Intelligence

#### 2A. Backrun Optimization

Standard backrunning (follow a large swap, arb the resulting imbalance) is competitive but not purely speed-dependent. Aconite adds:

- **Swap impact prediction**: Bella estimates price impact of pending large swap before simulation completes — start building arb bundle speculatively
- **Multi-venue backrun**: Instead of just arbing the impacted pool vs one other, MCTS finds the optimal multi-pool route to maximize profit from the imbalance

#### 2B. JIT Liquidity (Enhanced)

JIT requires seeing a pending large swap and providing concentrated liquidity. Aconite enhancement:

- **Swap size prediction**: Bella can estimate likely swap sizes for known addresses (market makers, protocol contracts)
- **Optimal tick range**: MCTS searches for the tick range that maximizes fee capture vs gas cost
- **Conditional JIT**: Only provide JIT when Bella predicts the pool won't see a second large swap in the opposite direction within 2 blocks (which would create impermanent loss)

### Tier 3: Not Our Game (Avoid)

| Strategy | Why Skip |
|----------|----------|
| Simple sandwich | Zero intelligence edge. Pure speed + order flow access. Dying with encrypted mempools. Ethically questionable. |
| 2-pool atomic arb | Commoditized to near-zero margin. Thousands of competitors. |
| Token launch sniping | Mostly rugs. Requires social/governance intelligence we don't have. |

---

## Architecture: Aconite Core

### Pipeline Stages

```
┌─────────────────────────────────────────────────────┐
│ Stage 1: OBSERVE                                     │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌───────────────┐  │
│  │  Mempool   │  │  Block     │  │  CEX Feed     │  │
│  │  Monitor   │  │  Monitor   │  │  (Kraken WS)  │  │
│  └─────┬──────┘  └─────┬──────┘  └──────┬────────┘  │
│        └───────────┬───┘────────────────┘            │
│                    ▼                                 │
│           Event Stream (unified)                     │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│ Stage 2: ANALYZE                                     │
│                                                      │
│  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │  Price Graph      │  │  Position Scanner        │  │
│  │  (in-memory,      │  │  (Aave/Compound/Morpho   │  │
│  │   per-block)      │  │   health factors)        │  │
│  └────────┬─────────┘  └───────────┬──────────────┘  │
│           └──────────┬─────────────┘                 │
│                      ▼                               │
│            Opportunity Candidates                     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│ Stage 3: PREDICT (Bella)                             │
│                                                      │
│  Input:  Current regime, recent events, CEX movement │
│  Output: regime_forecast (next 1-20 blocks)          │
│          vol_prediction (expected magnitude)          │
│          divergence_forecast (which pools, when)      │
│                                                      │
│  Inference: ~5-20ms on GPU                           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│ Stage 4: SEARCH (Poppy/MCTS)                         │
│                                                      │
│  For each candidate opportunity:                     │
│    State:  Pool reserves + position + prediction     │
│    Action: Swap routes, amounts, sequencing          │
│    Value:  TRM value head (expected PnL)             │
│    Prior:  TRM policy head (likely good routes)      │
│                                                      │
│  MCTS: 32-128 simulations per opportunity            │
│  Output: Ranked bundle candidates with expected PnL  │
│                                                      │
│  Latency: 50-300ms                                   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│ Stage 5: CONSTRUCT                                   │
│                                                      │
│  Best MCTS result → concrete bundle:                 │
│    1. Encode swap calls (direct pool interaction)    │
│    2. Set gas parameters (base_fee + tip)            │
│    3. Include profit check (revert if below min)     │
│    4. Include coinbase tip (% of expected profit)    │
│    5. Sign transactions                              │
│                                                      │
│  Simulation: eth_callBundle or local revm            │
│  Verify:    profit > threshold, gas < budget         │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│ Stage 6: EXECUTE                                     │
│                                                      │
│  Submit bundle to:                                   │
│    - Flashbots relay (primary)                       │
│    - Titan Builder (direct, if API access)           │
│    - Rsync Builder (direct, if API access)           │
│                                                      │
│  Target: specific block number                       │
│  Fallback: resubmit for next block if not included   │
│  Monitor: eth_getBundleStats for inclusion feedback   │
└─────────────────────────────────────────────────────┘
```

### State Management

```python
# Aconite maintains rolling state — this is where TRM's memory matters

class AconiteState:
    # === Market State (updated per-block) ===
    price_graph: PoolPriceGraph        # All tracked pools, reserves, current prices
    health_factors: Dict[Address, float]  # Lending protocol positions near liquidation
    gas_regime: GasRegime              # Current base_fee, trend, predicted next blocks

    # === Bella Context (updated per-minute or per regime change) ===
    current_regime: RegimeType         # crash | rally | sideways | volatile | calm
    regime_confidence: float           # 0-1
    vol_forecast: float                # Expected vol next N blocks
    divergence_forecast: List[PoolPair]  # Pools predicted to diverge

    # === Portfolio State (persistent) ===
    inventory: Dict[Token, float]      # Current token holdings (for CEX-DEX)
    daily_pnl: float                   # Circuit breaker tracking
    bundle_history: RingBuffer          # Last 1000 bundles (submitted, included, profit)

    # === TRM Hidden State (carried across decisions) ===
    trm_state: jnp.ndarray            # (state_dim,) — persistent reasoning state
```

---

## Training: Foxglove for MEV

### Historical Oracle = MEV Hindsight

Foxglove's oracle looks ahead 20 steps and says "the optimal action was X." For MEV, the historical oracle is even more powerful:

```
Given: Complete archive node state at block N
Ask:   What was the most profitable bundle constructible at block N?

This is a SOLVED problem in hindsight:
  - All pending transactions visible
  - All pool states known
  - All oracle updates known
  - All liquidation triggers known
  - The actual block contents are known (what actually happened)

Historical MEV oracle:
  1. For each block in archive:
     - Reconstruct pre-block state
     - Enumerate all MEV opportunities
     - Compute optimal bundle (brute force OK — offline)
     - Record: (state, opportunities, optimal_actions, profit)

  2. Train Foxglove on these (state, action, value) tuples:
     - State = pool prices + positions + regime
     - Action = which opportunities to target, routing
     - Value = realized PnL
```

This is the same Expert Iteration pattern, but the "expert" is historical hindsight instead of a lookahead oracle.

### Curriculum Levels (Aconite Edition)

```
L0:  Detect 2-pool arb from static price snapshot
L1:  Detect 2-pool arb from time series (arb appears and disappears)
L2:  Estimate optimal trade size (AMM constant-product math)
L3:  Multi-hop routing (3-pool cycle, find profitable path)
L4:  Liquidation detection (health factor monitoring)
L5:  Gas-aware profitability (include realistic gas costs)
L6:  Competition modeling (probability of winning given N competitors)
L7:  Regime-conditional strategy selection (which strategy for which regime)
L8:  Cross-protocol compound opportunities (liquidation + arb combo)
L9:  CEX-DEX lead-lag exploitation (predict DEX adjustment timing)
L10: Multi-block planning (position across 2-10 blocks)
L11: Bundle construction optimization (sequencing, sizing, tipping)
L12: Adversarial MEV (other bots trying to front-run YOUR bundles)
L13: Capital allocation across concurrent opportunities
L14: Novel strategy discovery (agent finds opportunities not in curriculum)
```

---

## Data Requirements

### From Iris-DEX

| Data | Resolution | Use |
|------|-----------|-----|
| `ohlcv_pool` | Per-block (12s) | Pool-level price dynamics, lead-lag detection |
| `ohlcv` (merged) | Per-block (12s) | Asset-level regime detection for Bella |
| `raw_swaps` | Per-event | Mempool replay, sandwich detection, JIT detection |
| `pools` | Static + periodic refresh | Pool universe, TVL, fee tiers |
| Lending positions | Per-block snapshot | Health factor tracking |
| Oracle updates | Per-event | Liquidation trigger timing |

### From Kraken (Existing)

| Data | Resolution | Use |
|------|-----------|-----|
| OHLCV | 1-minute | CEX reference prices for CEX-DEX arb training |
| Order book snapshots | When available | Depth/spread analysis |

### Bella Retraining

Bella currently trains on 1-minute OHLCV. For Aconite, Bella needs a **fast mode**:

```
Bella (standard): 1-min candles, 400-step windows = 6.7 hour episodes
Bella (fast):     12-sec candles (per-block), 400-step windows = 80 min episodes

Same architecture, different data resolution.
The skeleton extraction works identically — just shorter calendar time per window.
```

---

## Risk Architecture

### Circuit Breakers

```python
class AconiteRiskManager:
    max_daily_loss: float = 500.0         # USD — stop everything if exceeded
    max_single_trade_loss: float = 100.0  # USD — reject bundles above this risk
    max_gas_per_block: float = 50.0       # USD — don't overspend on gas
    max_tip_ratio: float = 0.95           # Never tip more than 95% of expected profit
    min_profit_threshold: float = 5.0     # USD — don't bother below this

    # Inventory limits (CEX-DEX)
    max_inventory_per_token: float = 10000.0  # USD equivalent
    max_total_inventory: float = 100000.0     # USD equivalent
    max_inventory_age_blocks: int = 100       # Force-close stale positions

    # Operational
    max_consecutive_failures: int = 10    # Pause and alert after 10 failed bundles
    min_bundle_inclusion_rate: float = 0.05  # If <5% of bundles land, something is wrong
```

### Failure Modes

| Failure | Detection | Response |
|---------|-----------|----------|
| Stale state (simulate against old block) | Bundle revert | Refresh state, reduce bundle TTL |
| Gas spike (base_fee jumps) | Pre-flight check | Skip marginal opportunities, only execute high-margin |
| Builder not including our bundles | Inclusion rate < 5% | Increase tip ratio, diversify builders |
| Bella mispredict (regime wrong) | Realized loss > predicted | Reduce Bella confidence weight, increase MCTS sims |
| Smart contract exploit | Balance drain | Kill switch, all-stop, investigate |
| Node goes down | Health check fails | Fallback to backup RPC (Infura/Alchemy) for monitoring only |

---

## Implementation Phases

### Phase 0: Foundation (Before Any MEV — Current Work)

```
[x] Reth archive node (Iris-DEX)
[ ] Complete archive sync
[ ] Extract per-block OHLCV from swap events
[ ] Build pool universe table
[x] Bella trained on 1-min data
[x] Poppy TRM architecture
[x] Foxglove training pipeline
```

### Phase 1: Historical Analysis (Weeks 1-4)

No live interaction. Pure research on archive data.

```
[ ] Build MEV detection scripts:
    - Sandwich detector (same-block frontrun+backrun patterns)
    - Arb detector (cross-pool price convergence within 1 block)
    - Liquidation detector (health factor transitions)
    - JIT detector (same-block mint-swap-burn)
[ ] Quantify: how much MEV per day? What % uncaptured?
[ ] Identify: which pools/tokens have persistent uncaptured arb?
[ ] Map: lead-lag structure between Kraken and DEX prices
[ ] Train: Bella-fast on per-block OHLCV data
```

### Phase 2: Paper Trading (Weeks 5-8)

Live mempool monitoring. No real execution.

```
[ ] Mempool monitor (WebSocket to Reth node)
[ ] In-memory price graph (updated per-block)
[ ] Opportunity detector (arb, liquidation)
[ ] Bundle constructor (build but don't submit)
[ ] Simulation engine (eth_callBundle against live state)
[ ] Logging: every opportunity, decision, simulated profit
[ ] Compare: our detection vs actual on-chain MEV
[ ] Metric: "would-have-captured" rate and PnL
```

### Phase 3: Live Arb Bot (Weeks 9-12)

Simplest profitable strategy first: flashloan-powered DEX-DEX arb.

```
[ ] Deploy arb contract (Solidity, gas-optimized)
[ ] Flashbots key pair + relay authentication
[ ] Live bundle submission pipeline
[ ] Risk manager (circuit breakers)
[ ] Real-time PnL tracking
[ ] Start with conservative parameters:
    - min_profit: $20 (high threshold)
    - max_tip_ratio: 0.80 (keep 20%)
    - max_gas: $30/block
[ ] Iterate: lower thresholds as confidence grows
```

### Phase 4: Bella Integration (Weeks 13-20)

Add prediction layer.

```
[ ] Bella-fast inference service (GPU, per-block regime updates)
[ ] Regime-aware strategy selection
[ ] Predictive liquidation scanner
[ ] CEX-DEX temporal arb (if capital available)
[ ] MCTS bundle optimization for 3+ hop cycles
[ ] Compare: prediction-aided vs reactive-only performance
```

### Phase 5: Full Aconite (Weeks 21+)

```
[ ] Cross-protocol compound opportunities
[ ] Multi-block positioning
[ ] Adaptive compute allocation (scale with regime)
[ ] Strategy portfolio optimization
[ ] Continuous strategy discovery (L14 curriculum)
```

---

## Economics

### Startup Costs

| Item | Cost | Notes |
|------|------|-------|
| Archive node (Reth) | $0 (own hardware) | Already building — 8TB XFS drive |
| Smart contract deployment | ~$200-500 | One-time mainnet deployment |
| Flashbots account | $0 | Free ECDSA key pair |
| Initial gas budget | $500-1000 | First month of operation |
| **Total** | **~$1,000** | Flashloan strategies need no trading capital |

### Ongoing Costs (Phase 3)

| Item | Cost/Month | Notes |
|------|------------|-------|
| Node electricity + bandwidth | ~$30 | Home server, already running |
| Gas for bundles | $200-500 | Scales with activity |
| GPU compute (Bella inference) | $0 | Shared with training GPU |
| **Total** | **~$300-500/month** | |

### Revenue Projections (Conservative)

| Phase | Strategy | Expected Monthly Revenue | Confidence |
|-------|----------|-------------------------|------------|
| Phase 3 | Simple DEX-DEX arb | $500-2,000 | Medium |
| Phase 4 | + Predictive liquidation | $1,000-5,000 | Medium-Low |
| Phase 4 | + CEX-DEX temporal arb | $3,000-15,000 | Low (capital-dependent) |
| Phase 5 | + Cyclic arb + cross-protocol | $5,000-25,000 | Speculative |

**Break-even**: Phase 3 covers costs. Phase 4+ is profit.

---

## Open Design Questions

1. **Action space mapping**: Poppy uses 5×64=320 actions (per-asset trade decisions). Aconite needs actions like "swap X amount on pool Y via route Z." How to bridge? Separate action space, or embed pool routing into the 320-action framework?

2. **Training data blending**: Should Aconite's Foxglove curriculum use real historical MEV data, synthetic Bella data, or a blend? Historical is ground truth but limited. Synthetic is unlimited but may not capture pool microstructure.

3. **Dual-GPU split**: Training on CMP 90HX, inference on RTX 3090? Or vice versa? 3090 is faster for both — but can it do Bella training + Aconite inference simultaneously?

4. **When to start**: Phase 1 (historical analysis) can begin as soon as Iris-DEX archive sync completes. No dependency on Foxglove/Poppy training progress. Should we start Phase 1 in parallel with current training, or wait for Foxglove to converge?

5. **Separate codebase or integrated?** Aconite could live in `mcts_trader/mev/` (same repo), or in `Iris-DEX` (separate project). The model components (Bella, TRM) are shared. The MEV-specific code (mempool, bundles, contracts) is new.

---

*Created: 2026-02-25*
*Status: DESIGN — reference architecture for future MEV extraction system*
*Codename: Aconite (wolfsbane) — fast-acting, precision dosing, powerful in skilled hands*

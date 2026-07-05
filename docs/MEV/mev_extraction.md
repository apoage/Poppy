# MEV Extraction: Technical Reference

> **Purpose**: Authoritative spec on Maximal Extractable Value — the mechanics, infrastructure, strategies, and economics of extracting profit from transaction ordering on Ethereum. Written for a team with a full archive node (Reth), a generative market model (Bella), an MCTS+Transformer trading agent (Poppy), and deep DeFi fluency.

---

## 1. MEV Fundamentals

### What MEV Is

MEV (Maximal Extractable Value) is the total profit extractable by manipulating the inclusion, exclusion, and ordering of transactions within a block. Originally "Miner Extractable Value" (pre-Merge), renamed after Ethereum moved to Proof of Stake in September 2022.

The core insight: **transaction ordering is economically valuable**. Whoever controls ordering controls the ability to front-run, back-run, sandwich, arbitrage, and liquidate — all by placing their transactions at precisely the right position in a block.

MEV is not a bug. It is an emergent property of any system where:
1. State changes are ordered into discrete blocks
2. Block producers have discretion over ordering
3. Pending transactions are visible before inclusion (mempool)
4. State-dependent operations (swaps, liquidations) create ordering-sensitive profit opportunities

### Where MEV Comes From

**Information asymmetry**: Pending transactions in the mempool reveal intent before execution. A large swap on Uniswap is visible to anyone watching the mempool — searchers can act on this information.

**Price discrepancies**: Cross-DEX price differences, CEX-DEX divergence, stale oracle prices on lending protocols — all create deterministic profit if you can execute the right transaction at the right time.

**Protocol mechanics**: AMM constant-product math, lending protocol liquidation bonuses, NFT auctions — protocol rules create extractable value by design.

### The MEV Supply Chain

```
Users (transaction originators)
  → Public Mempool (visible to everyone, ~300ms propagation)
  → Searchers (detect opportunities, construct bundles)
  → Builders (aggregate bundles into full blocks, optimize ordering)
  → Relays (MEV-Boost relays — trusted intermediaries between builders and proposers)
  → Proposers/Validators (select highest-value block, propose to network)
  → Execution (block included on-chain, MEV realized)
```

Each layer extracts rent:
- **Searchers** find opportunities, pay most of their profit to builders as tips
- **Builders** aggregate searcher bundles, compete on total block value, pay most of their surplus to proposers
- **Proposers** (validators) select the highest-bid block via MEV-Boost auction, pocket the payment
- **Users** bear the cost — worse execution prices, failed transactions, higher gas

### PBS (Proposer-Builder Separation)

Pre-PBS: validators both built and proposed blocks. This concentrated MEV extraction power in validator hands and created perverse incentives (validators running MEV strategies themselves).

PBS separates these roles:
- **Proposers** (validators): Select which block to propose. They don't see block contents — they just pick the highest bid.
- **Builders**: Construct blocks by ordering transactions optimally. Compete in an auction to have their block selected.
- **Relays**: Trusted intermediaries that escrow builder blocks and reveal them to proposers only after commitment.

As of early 2026, PBS is implemented via MEV-Boost (out-of-protocol) with Ethereum's Glamsterdam upgrade introducing ePBS (enshrined PBS) at the protocol level, removing relay trust assumptions.

**Current market structure (2025-2026)**:
- Top 2 builders capture >90% of blocks
- Major builders: Titan Builder, Rsync Builder (formerly builder0x69)
- ~95% of Ethereum blocks are built via MEV-Boost
- Builder competition is a Bertrand oligopoly — fierce on margins, concentrated on participants

### Flashbots and the MEV Ecosystem Evolution

**Timeline:**
- **2020**: Flashbots founded. Published "Flash Boys 2.0" paper quantifying MEV.
- **2021**: Flashbots Auction launches — private bundle submission, replaces public Priority Gas Auctions (PGA).
- **2022**: MEV-Boost ships for the Merge. PBS goes live.
- **2023**: MEV-Share launches — users get rebates on MEV their transactions generate. Flashbots SUAVE announced (fair sequencing).
- **2024**: BuilderNet launches — decentralized builder infrastructure, Flashbots exits centralized block building.
- **2025**: ePBS development accelerates. Encrypted mempool EIPs under review. Order flow auction protocols mature.
- **2026**: Glamsterdam upgrade brings ePBS to mainnet. Relay trust assumptions eliminated.

**Flashbots infrastructure:**
- `relay.flashbots.net` — Bundle submission endpoint
- MEV-Boost — Validator sidecar for PBS auctions
- MEV-Share — User-facing MEV rebate system
- BuilderNet — Decentralized builder network
- Flashbots Protect RPC — Private transaction submission (bypasses public mempool)

---

## 2. MEV Strategies — Detailed Taxonomy

### 2.1 Arbitrage

The cleanest MEV strategy. Exploit price discrepancies between venues. Zero information extraction from users — pure market efficiency.

#### DEX-DEX Arbitrage

Price of token X differs between two on-chain pools. Buy low, sell high, atomic.

```
Pool A: WETH/USDC at 2,500 USDC
Pool B: WETH/USDC at 2,510 USDC

→ Buy 10 WETH on Pool A for 25,000 USDC
→ Sell 10 WETH on Pool B for 25,100 USDC
→ Gross profit: 100 USDC
→ Minus gas + priority fee + builder tip
```

**Competition level**: Extreme. Commoditized. Two-pool arb is the most contested strategy — thousands of bots competing. Profit margins compressed to near-zero after tips.

**Typical profit per trade (2025)**: $1-50 after costs for simple two-pool arb. Volume makes it viable — top arb bots execute thousands of trades per day.

#### CEX-DEX Arbitrage

The dominant MEV strategy by extracted value. Exploits the fact that CEX prices update continuously but DEX prices update only on swaps.

```
Binance: ETH at $2,505 (real-time orderbook)
Uniswap V3 0.05%: ETH at $2,500 (stale — last swap was 30s ago)

→ Buy on Uniswap, sell on Binance (or hedge with perp short)
→ Profit: ~$5 per ETH minus gas, fees, slippage
```

**Key characteristics:**
- Requires off-chain infrastructure (CEX API, co-located servers)
- Not fully atomic — CEX leg has settlement risk
- Dominated by 3 searchers capturing ~75% of all CEX-DEX value (2023-2025)
- Between Aug 2023 and Mar 2025: $233.8M extracted by top 19 CEX-DEX searchers from 7.2M identified arbitrages
- Searchers retain only 10-15% of revenue — ~85-90% goes to builders/proposers as tips
- Infrastructure moat: "Ashburn-Tokyo axis" geographic centralization near CEX data centers and relay infrastructure

#### Cyclic Arbitrage

Multi-hop path that returns to the starting token with a profit.

```
USDC → WETH (Pool A)
WETH → LINK (Pool B)
LINK → USDC (Pool C)

If the product of exchange rates across the cycle > 1.0, profit exists.
```

**Detection**: Graph algorithm on the pool universe. Find negative-weight cycles in the log-price graph. Bellman-Ford or optimized variants.

**Complexity scales exponentially** with hop count. 2-hop is commoditized. 3-hop is competitive. 4+ hops is where intelligence beats speed — the search space is too large for brute force. This is where Bella+Poppy have an edge.

#### Cross-Chain Arbitrage

Same token, different chains. Arbitrage across Ethereum L1, Arbitrum, Base, Optimism, Polygon.

```
WETH on Ethereum: $2,500
WETH on Arbitrum: $2,495

→ Buy on Arbitrum, bridge to Ethereum, sell
→ Or: pre-position inventory on both chains, execute simultaneously
```

**Characteristics:**
- Cross-chain arb has grown 5.5x in recent study periods
- 67% of trades use pre-positioned inventory (no bridging delay)
- Pre-positioned: ~9s settlement. Bridge-based: ~242s settlement.
- Bridge risk, inventory risk, capital lockup
- Less competitive than same-chain arb — more complexity = fewer participants

### 2.2 Sandwich Attacks

The most controversial MEV strategy. Extract value directly from users by manipulating the price around their trade.

```
1. Detect: Large pending swap in mempool (victim buys TOKEN with 1% slippage tolerance)
2. Frontrun: Buy TOKEN before the victim's tx (pushes price up)
3. Victim executes: Buys TOKEN at the now-higher price (within their slippage tolerance)
4. Backrun: Sell TOKEN after the victim's tx (captures the price difference)
```

**Mechanics in detail:**

```
State before:
  Uniswap V3 WETH/USDC pool, price = 2,500

Victim's pending tx:
  Swap 50,000 USDC → WETH, slippage tolerance = 0.5%

Attacker:
  1. Frontrun: Buy WETH with X USDC → price moves to 2,506
  2. Victim executes: Buys WETH at ~2,506 instead of 2,500 (within 0.5% tolerance)
  3. Backrun: Sell WETH → price returns near 2,500, attacker profits ~$300

Attacker profit = victim's slippage loss - gas costs - builder tip
```

**Economics (2025):**
- Sandwich attacks constituted $289.76M or 51.56% of total MEV transaction volume ($561.92M) in 2025
- Sandwich bots pay only 15-20% of profits as tips to validators (vs 50-60% for arb bots) — higher margin
- Major historical bot: `jaredfromsubway.eth` — spent ~$20-30M on gas, extracted ~$40-60M net profit
- Requires no capital beyond gas (can be flashloan-powered)
- Only works on public mempool transactions — Flashbots Protect and private RPCs defeat this

**Why it's controversial**: Direct wealth transfer from users to bots. Users get worse execution. No market efficiency argument — pure extraction. Some jurisdictions may classify as market manipulation.

### 2.3 Liquidations

Lending protocols (Aave, Compound, Maker) allow anyone to liquidate undercollateralized positions for a bonus.

```
Aave V3:
  Borrower deposited 10 ETH ($25,000), borrowed 15,000 USDC
  Health Factor = collateral_value * liquidation_threshold / debt = 1.25

  ETH drops to $2,000 → collateral = $20,000
  Health Factor = 20,000 * 0.825 / 15,000 = 1.10

  ETH drops to $1,700 → HF < 1.0 → LIQUIDATABLE
  Liquidator repays up to 50% of debt, receives collateral at 5-10% discount
```

**MEV opportunity**: Monitor all borrowing positions, watch oracle price feeds, be first to call `liquidate()` when HF drops below 1.0.

**Aave V4 changes (2025):**
- Variable liquidation bonus — lower HF = higher bonus (up to ~15%)
- Target Health Factor — liquidators repay only enough to restore HF to a target, not a fixed percentage
- More nuanced, but still competitive

**Infrastructure required:**
- Index all active borrow positions (on-chain state polling or event tracking)
- Monitor Chainlink oracle update transactions in the mempool
- Pre-simulate liquidation profitability (gas cost vs bonus)
- Submit via Flashbots bundle — atomic, no revert risk

**Competition**: High but less commoditized than arb. Requires understanding protocol-specific mechanics. Multi-protocol monitoring (Aave V2/V3/V4, Compound V2/V3, Maker, Morpho, Euler) compounds complexity.

**Typical profits**: $10-10,000+ per liquidation depending on position size and bonus. Large liquidations ($1M+) during market crashes are extremely lucrative but rare and hyper-competitive.

### 2.4 JIT (Just-In-Time) Liquidity

Exclusive to concentrated liquidity AMMs (Uniswap V3). Detect a pending large swap, mint a tight liquidity position around the current price to capture trading fees, then burn it after the swap executes — all in the same block.

```
1. Detect: Large pending swap (e.g., 500 ETH sell on Uniswap V3)
2. Mint: Add concentrated liquidity in a 1-tick range around current price
3. Swap executes: JIT LP captures disproportionate share of fees
4. Burn: Remove liquidity immediately after
```

**Economics:**
- Average ROI: 0.007% per JIT transaction (thin margin, high volume play)
- Only profitable on large swaps (>$100K typically) — fixed gas costs for mint+burn (~200K gas)
- Traders benefit slightly (0.139% better execution on average due to deeper liquidity)
- Passive LPs lose (JIT LPs siphon fee share without bearing inventory risk)
- ~8,000 JIT transactions recorded May 2021 to July 2022 — relatively rare

**Why it matters for us**: JIT is detectable in our swap data as same-block mint-swap-burn patterns. It's a signal of sophisticated MEV activity and can condition Bella's understanding of liquidity dynamics.

### 2.5 Backrunning

Place a transaction immediately after a large trade to capture the resulting arbitrage opportunity. Less adversarial than sandwich — no frontrun component.

```
1. Large swap moves WETH/USDC price on Uniswap V3 by 0.3%
2. WETH/USDC price on Sushiswap hasn't updated
3. Backrunner: Arbitrage Sushiswap → Uniswap to equalize prices
```

**Distinction from arbitrage**: Pure arb exploits existing discrepancies. Backrunning specifically follows another transaction that creates the discrepancy.

**Common targets:**
- Large DEX swaps (price displacement on one venue)
- Oracle updates on lending protocols (create liquidation opportunities)
- Governance proposal executions (state changes that affect token economics)

### 2.6 Long-Tail MEV

Strategies outside the standard arb/sandwich/liquidation trinity. Higher profit per event, lower frequency, less competition.

**NFT sniping**: Detect mispriced NFT listings on marketplaces. Buy underpriced NFTs before others. Requires valuation models.

**Token launch sniping**: Detect new liquidity pool creation events (`PairCreated`, `PoolCreated`), buy in the first block. Extremely risky — most launches are rugs. But first-mover advantage on legitimate launches is 10-100x.

**Governance MEV**: Detect governance proposals that will pass, front-run the economic impact. Example: A proposal to increase Aave interest rates on a specific market — position accordingly before execution.

**Airdrop farming**: Not traditional MEV, but related — detect upcoming airdrops via governance activity, protocol deployments, snapshot announcements. Position qualifying wallets.

**Liquidation cascades**: Predict that one liquidation will trigger price movement that triggers more liquidations. Front-run the cascade by positioning for the cumulative impact.

### 2.7 Time-Bandit Attacks

**Theoretical (mostly).** A validator with sufficient stake could attempt to reorg recent blocks to capture high-value MEV from those blocks.

```
Block N contains a $500K MEV opportunity that someone else captured.
Block reward = ~0.05 ETH ($125).

If a validator can propose a competing Block N with that MEV redirected to themselves,
and enough subsequent validators build on their fork instead of the canonical chain,
they capture the $500K.
```

**Why it's mostly theoretical on Ethereum PoS:**
- Requires controlling multiple consecutive validator slots (probabilistically hard)
- Ethereum finality (Casper FFG) makes reorgs beyond 2 epochs essentially impossible
- Social consensus would punish obvious reorg attacks
- The MEV would need to exceed the slashing risk

**Where it's more real**: On chains with weaker finality guarantees, shorter block times, or concentrated validator sets. Some PoW chains historically saw this behavior.

---

## 3. Technical Infrastructure

### 3.1 Node Requirements

**Minimum viable**:
- Full node (not archive) for mempool access and state queries
- Archive node for historical simulation and backtesting (we have this — Reth)
- WebSocket connection for real-time pending transaction subscription (`eth_subscribe("newPendingTransactions")`)
- High-performance JSON-RPC with low latency (<10ms per call)

**Our advantage**: Reth archive node provides both real-time mempool access AND complete historical state for backtesting/simulation. Most searchers run separate infrastructure for these.

**Mempool access methods:**
```
1. Public mempool (eth_subscribe):
   - See all pending transactions ~300ms after broadcast
   - Anyone can see them — no edge
   - Sufficient for backrunning and liquidation monitoring

2. Private mempool (Flashbots Protect):
   - Transactions submitted directly to Flashbots, not broadcast publicly
   - Invisible to public mempool watchers
   - Defeats sandwich attacks on protected transactions

3. Builder-direct submission:
   - Send transactions directly to specific builders (Titan, Rsync)
   - Fastest inclusion, private
   - Requires builder API access

4. Bloxroute BDN (Blogging Distribution Network):
   - Premium mempool feed with lower latency
   - Paid service ($1,000-10,000/month)
   - See pending txs ~50-100ms faster than public p2p gossip
```

### 3.2 Flashbots RPC, MEV-Share, MEV-Boost

**Flashbots Auction (Searcher API):**
```
Endpoint: https://relay.flashbots.net
Auth: ECDSA-secp256k1 key pair (identify searcher, no ETH required)

Methods:
  eth_sendBundle          — Submit bundle of ordered transactions
  mev_sendBundle          — MEV-Share format bundle (with hint/privacy controls)
  eth_sendPrivateTransaction — Single private tx, tried for next 25 blocks
  eth_callBundle          — Simulate bundle without submitting
  eth_getBundleStats      — Check inclusion status of submitted bundle
  flashbots_getUserStats  — Your historical inclusion rate, profit stats
```

**MEV-Share (User-Facing):**
Users submit transactions via Flashbots Protect RPC. Searchers see partial hints (e.g., "a swap on Uniswap V3 involving WETH") but not full calldata. Searchers can backrun but not frontrun. Profit is shared: searcher gets a cut, user gets a refund.

**MEV-Boost (Validator-Facing):**
```
Validator runs mev-boost sidecar alongside consensus client.
Each slot:
  1. Multiple builders submit block bids to relay(s)
  2. Relay validates blocks, escrows them
  3. mev-boost selects highest-value bid
  4. Validator proposes the winning block (blind — doesn't see contents)
  5. After commitment, relay reveals block to network
```

### 3.3 Bundle Construction and Submission

A **bundle** is an ordered list of transactions that must execute atomically — all or nothing, in the specified order, in the target block.

```python
# Flashbots bundle structure
bundle = {
    "txs": [
        signed_tx_1_hex,   # Your frontrun transaction
        target_tx_hex,     # The victim's transaction (for sandwich)
        signed_tx_2_hex    # Your backrun transaction
    ],
    "blockNumber": hex(target_block),
    "minTimestamp": 0,      # Optional: earliest valid timestamp
    "maxTimestamp": 0,      # Optional: latest valid timestamp
    "revertingTxHashes": [] # Txs allowed to revert without killing the bundle
}

# Submit via JSON-RPC
response = requests.post("https://relay.flashbots.net", json={
    "jsonrpc": "2.0",
    "method": "eth_sendBundle",
    "params": [bundle],
    "id": 1
}, headers={
    "X-Flashbots-Signature": f"{auth_address}:{signature}"
})
```

**Bundle constraints (2025):**
- Max 100 transactions per bundle
- Max 300,000 bytes total bundle size
- Max 1 backrun transaction per bundle (as of Oct 2025)
- Rate limit: 10,000 requests/second per IP
- Bundle must target a specific block number
- No guarantee of inclusion — builders select bundles that maximize total block value

### 3.4 Transaction Simulation

Before submitting a bundle, simulate it against current state to verify profitability.

```python
# eth_callBundle — simulate without submitting
simulation = {
    "txs": [signed_tx_hex],
    "blockNumber": hex(current_block),
    "stateBlockNumber": "latest",
    "timestamp": current_timestamp
}

# Returns:
# - Gas used per transaction
# - Return data (did it revert?)
# - State changes (balance diffs, storage writes)
# - Effective gas price
# - Coinbase transfer (tip to builder)
```

**State diffing**: For complex strategies, diff the state before and after bundle execution to verify:
- Your token balances increased (profit)
- No unexpected state changes (attacks on your bot)
- Gas cost matches expectations

**EVM simulation tools:**
- `eth_callBundle` (Flashbots) — remote simulation
- `revm` (Rust) — local EVM simulation, fastest option, used by most competitive searchers
- Foundry's `anvil` — local fork for development/testing
- Tenderly — cloud simulation with tracing (not for latency-critical paths)

### 3.5 Gas Optimization and Priority Fee Bidding

**Gas cost breakdown for MEV transactions:**
```
Total cost = base_fee * gas_used + priority_fee * gas_used + direct_coinbase_transfer

Where:
  base_fee       — Protocol-set, burned (EIP-1559)
  priority_fee   — Tip to builder/proposer (your bid)
  coinbase_transfer — Direct ETH payment to block.coinbase (builder tip, most common)
```

**Optimal bidding strategy:**
```
profit = gross_mev - gas_cost - builder_tip
builder_tip = gross_mev * (1 - target_margin)

Target margins by strategy (2025):
  DEX-DEX arb:     5-15% retained by searcher, 85-95% to builder
  CEX-DEX arb:     10-15% retained
  Sandwich:        15-20% retained (higher margin, more risk)
  Liquidation:     10-30% retained (varies by competition)
  Long-tail:       30-70% retained (less competition)
```

**Gas optimization matters enormously** — a 10% gas reduction on a swap is the difference between profitable and unprofitable at tight margins.

### 3.6 Latency Considerations

```
Latency budget (from opportunity detection to bundle landing):

  Block time:           12 seconds (Ethereum)
  Mempool propagation:  ~300ms (public p2p), ~50ms (premium feeds)
  Opportunity detection: 1-10ms (simple), 10-100ms (complex)
  Simulation:           1-5ms (local revm), 50-200ms (remote eth_call)
  Bundle submission:    10-50ms (to relay)
  Builder processing:   varies (builder internal)

Total searcher budget: ~500ms to detect + simulate + submit
The rest of the 12s is builder auction time.
```

**Where latency is critical:**
- CEX-DEX arb: Every millisecond matters. Co-location near CEX APIs and relay infrastructure.
- Sandwich: Must see victim tx and submit bundle before block closes.
- Simple arb: Latency arms race — most commoditized, fastest wins.

**Where latency is NOT the moat:**
- Cyclic arb (4+ hops): Search space too large for speed to dominate — intelligence wins.
- Long-tail strategies: Opportunity detection is the hard part, not speed.
- Cross-chain arb: Bridge latency (minutes) dwarfs execution latency.

---

## 4. The Execution Pipeline

### 4.1 Mempool Monitoring and Filtering

```python
# WebSocket subscription for pending transactions
async def monitor_mempool(ws_url):
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "method": "eth_subscribe",
            "params": ["newPendingTransactions"],
            "id": 1
        }))

        while True:
            msg = await ws.recv()
            tx_hash = json.loads(msg)["params"]["result"]

            # Fetch full transaction
            tx = await get_transaction(tx_hash)

            # Filter: only care about DEX router calls
            if tx["to"] in MONITORED_ROUTERS:
                await process_opportunity(tx)
```

**Filtering heuristics (reduce noise from ~1000 tx/s to actionable subset):**
- Target contract addresses (Uniswap routers, Aave, Compound)
- Function selector matching (first 4 bytes of calldata)
- Value thresholds (ignore sub-$1K swaps for sandwich)
- Gas price (ignore txs with low priority fee — they'll sit in mempool)
- Sender reputation (known bot addresses — don't sandwich other bots)

### 4.2 Opportunity Detection Algorithms

**DEX-DEX Arbitrage:**
```python
# Maintain in-memory price graph of all pools
# On each new swap/state update:

def detect_arb(pools, updated_pool):
    """Check if updated_pool created an arbitrage."""
    token_a, token_b = updated_pool.tokens

    # Two-pool arb: find all other pools with same pair
    for other_pool in pools.get_pools(token_a, token_b):
        if other_pool == updated_pool:
            continue
        price_a = updated_pool.get_price(token_a, token_b)
        price_b = other_pool.get_price(token_a, token_b)
        spread = abs(price_a - price_b) / min(price_a, price_b)
        if spread > MIN_SPREAD:  # e.g., 0.1%
            yield ArbOpportunity(updated_pool, other_pool, spread)

    # Cyclic arb: BFS/DFS on price graph
    for cycle in find_profitable_cycles(pools, token_a, max_hops=4):
        yield CyclicArbOpportunity(cycle)
```

**Sandwich detection:**
```python
def detect_sandwich_target(tx):
    """Score a pending transaction as a sandwich target."""
    # Decode swap parameters
    swap = decode_swap(tx)
    if not swap:
        return None

    # Calculate price impact
    pool = get_pool(swap.pool_address)
    impact = pool.simulate_swap(swap.amount_in, swap.token_in)

    # Estimate extractable value
    slippage_tolerance = swap.amount_out_min / swap.expected_out
    extractable = swap.amount_in * (1 - slippage_tolerance) * impact.price_change

    # Must exceed gas costs for 2 transactions (frontrun + backrun)
    gas_cost = estimate_gas_cost(2 * SWAP_GAS)

    if extractable > gas_cost * MIN_PROFIT_MULTIPLE:
        return SandwichTarget(tx, extractable, gas_cost)
```

**Liquidation monitoring:**
```python
def monitor_liquidations(lending_protocol):
    """Poll borrow positions, check health factors."""
    # Get all active borrow positions
    positions = lending_protocol.get_active_positions()

    for pos in positions:
        hf = pos.calculate_health_factor(current_prices)

        if hf < 1.0:
            # Already liquidatable
            profit = pos.liquidation_profit(current_prices)
            if profit > MIN_LIQUIDATION_PROFIT:
                yield LiquidationOpportunity(pos, profit)

        elif hf < 1.05:
            # Near liquidation — watch closely
            # Calculate price movement needed to trigger
            trigger_price = pos.liquidation_trigger_price()
            watch_list.add(pos, trigger_price)
```

### 4.3 Profit Calculation

```python
def calculate_net_profit(opportunity, bundle):
    """Calculate expected profit after all costs."""

    gross_profit = opportunity.expected_revenue

    # Gas costs
    gas_used = sum(tx.gas_limit for tx in bundle.transactions)
    base_fee = get_next_base_fee()  # EIP-1559 prediction
    gas_cost = gas_used * base_fee  # Base fee is burned, unavoidable

    # Builder tip (the bid — what you pay for inclusion)
    # Competitive: tip = gross_profit * (1 - margin)
    # Start conservative, increase if bundles aren't landing
    builder_tip = gross_profit * 0.90  # Keep 10%

    # Flashloan fees (if using)
    flashloan_fee = opportunity.capital_required * AAVE_FLASH_FEE  # 0.05% on Aave V3

    net_profit = gross_profit - gas_cost - builder_tip - flashloan_fee

    return net_profit, {
        "gross": gross_profit,
        "gas": gas_cost,
        "tip": builder_tip,
        "flashloan_fee": flashloan_fee,
        "net": net_profit,
        "margin_pct": net_profit / gross_profit * 100
    }
```

### 4.4 Bundle Construction

```python
def construct_arb_bundle(opportunity, block_number):
    """Build an atomic arbitrage bundle."""

    # Transaction 1: Execute arbitrage via our contract
    arb_tx = build_transaction(
        to=OUR_ARB_CONTRACT,
        data=encode_arb_call(
            pool_buy=opportunity.buy_pool,
            pool_sell=opportunity.sell_pool,
            amount=opportunity.optimal_amount,
            min_profit=opportunity.min_acceptable_profit
        ),
        gas_limit=300_000,
        max_fee_per_gas=base_fee + 1,  # Minimal priority fee
        max_priority_fee=1_000_000_000  # 1 gwei
    )

    # Transaction 2: Tip the builder via coinbase transfer
    tip_tx = build_transaction(
        to=COINBASE,  # block.coinbase — filled by builder
        value=builder_tip,
        gas_limit=21_000
    )

    # Or: integrate tip into arb contract (saves 21K gas)
    # Most competitive bots do this — the arb contract sends
    # ETH to block.coinbase at the end of execution

    return Bundle(
        transactions=[arb_tx],  # Tip embedded in contract
        block_number=block_number,
        min_timestamp=0,
        max_timestamp=0
    )
```

### 4.5 Risk Management

**Revert risk**: Bundle reverts = no cost (Flashbots doesn't charge for failed bundles). But opportunity cost of submitting a bad bundle: you miss the real opportunity.

**Capital risk**: Flashloan-based strategies have near-zero capital risk (borrow and return in same tx). Inventory-based strategies (CEX-DEX) have capital at risk.

**Smart contract risk**: Your MEV contract could be exploited. Common attacks:
- Reentrancy on callback functions
- Approval draining (if your contract holds approvals)
- Front-running your own transactions (ironic)

**Operational risk:**
- Node goes down during market crash (when MEV is highest)
- Gas estimation errors (underestimate = revert, overestimate = lower margin)
- Stale state (simulate against stale block, reality has changed)

**Risk mitigation checklist:**
```
[ ] All contracts are non-upgradeable (immutable)
[ ] No token approvals held by contract (approve-and-use in same tx)
[ ] Minimum profit check in contract (revert if below threshold)
[ ] Circuit breakers (max daily loss, max single-trade loss)
[ ] Monitoring/alerting on contract balance, failed bundles, profit trends
[ ] Multi-sig or timelock on any admin functions
[ ] Regular contract audits
```

---

## 5. Smart Contract Patterns

### 5.1 Flashloan-Powered Arbitrage

Flashloans eliminate capital requirements — borrow, execute, repay in one atomic transaction.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IPool} from "@aave/v3-core/contracts/interfaces/IPool.sol";
import {ISwapRouter} from "@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol";

contract FlashloanArb {
    IPool constant AAVE_POOL = IPool(0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2);
    address constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    address immutable owner;

    constructor() { owner = msg.sender; }

    function executeArb(
        address tokenBorrow,
        uint256 amount,
        bytes calldata params  // Encoded swap route
    ) external {
        require(msg.sender == owner);

        // Initiate flashloan — Aave calls executeOperation() as callback
        AAVE_POOL.flashLoanSimple(
            address(this),
            tokenBorrow,
            amount,
            params,
            0  // referral code
        );
    }

    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,    // Aave fee: 0.05% on V3
        address initiator,
        bytes calldata params
    ) external returns (bool) {
        require(msg.sender == address(AAVE_POOL));
        require(initiator == address(this));

        // Decode and execute swap route
        (address[] memory pools, bytes[] memory swapData) =
            abi.decode(params, (address[], bytes[]));

        uint256 balance = IERC20(asset).balanceOf(address(this));

        for (uint i = 0; i < pools.length; i++) {
            // Execute each swap in the route
            _executeSwap(pools[i], swapData[i]);
        }

        // Verify profit
        uint256 amountOwed = amount + premium;
        uint256 finalBalance = IERC20(asset).balanceOf(address(this));
        require(finalBalance > amountOwed, "No profit");

        // Repay flashloan
        IERC20(asset).approve(address(AAVE_POOL), amountOwed);

        // Send profit to owner (or tip builder)
        uint256 profit = finalBalance - amountOwed;
        IERC20(asset).transfer(owner, profit);

        return true;
    }
}
```

**Flashloan providers (2025):**
| Provider | Fee | Max Amount | Notes |
|----------|-----|------------|-------|
| Aave V3 | 0.05% | Pool liquidity (~$10B+) | Most popular, reliable |
| Balancer | 0% | Vault liquidity (~$1B) | Free — ideal for arb |
| Uniswap V3 | Pool fee tier | Pool liquidity | Flash swap (different API) |
| Maker | 0% | DAI supply | DAI-only |

### 5.2 Atomic Execution Patterns

**Key principle**: Every MEV transaction should be atomic — either all steps succeed and profit is realized, or the entire transaction reverts with no cost (beyond gas of the reverted tx, which in Flashbots bundles means no cost at all).

```solidity
// Profit-or-revert pattern
function executeAndVerify(
    bytes calldata swapData,
    uint256 minProfit
) external {
    uint256 balanceBefore = address(this).balance;

    // Execute the strategy
    _execute(swapData);

    // Verify profit
    uint256 balanceAfter = address(this).balance;
    require(balanceAfter >= balanceBefore + minProfit, "UNPROFITABLE");

    // Tip builder (from profit)
    uint256 profit = balanceAfter - balanceBefore;
    uint256 tip = profit * 90 / 100;  // 90% to builder
    block.coinbase.transfer(tip);
}
```

### 5.3 Multi-Hop Swap Routing

```solidity
// Optimized multi-hop: encode entire route as packed bytes
// Each hop: [pool_address(20) | zeroForOne(1) | amountSpecified(32)]

function multiHopSwap(bytes calldata route) external {
    uint256 offset = 0;
    while (offset < route.length) {
        address pool = address(bytes20(route[offset:offset+20]));
        bool zeroForOne = uint8(route[offset+20]) == 1;
        int256 amount = int256(uint256(bytes32(route[offset+21:offset+53])));

        IUniswapV3Pool(pool).swap(
            address(this),
            zeroForOne,
            amount,
            zeroForOne ? MIN_SQRT_RATIO + 1 : MAX_SQRT_RATIO - 1,
            ""
        );

        offset += 53;
    }
}
```

### 5.4 Gas-Efficient Contract Design

**Gas costs directly impact profitability.** Every opcode matters at the margin.

```
Key optimizations:

1. Calldata compression
   - Non-zero calldata byte = 16 gas, zero byte = 4 gas
   - Pack parameters tightly, use bit manipulation
   - Encode pool addresses as indices into a pre-registered table

2. Storage minimization
   - Never use storage for per-transaction data
   - Use immutable variables for constants
   - Transient storage (EIP-1153, live since Cancun) for within-tx state

3. Direct pool interaction
   - Skip routers (Uniswap Router adds ~30K gas overhead)
   - Call pool.swap() directly — requires implementing callback
   - Implement uniswapV3SwapCallback() for V3, handle reserve checks for V2

4. Assembly (Yul/Huff)
   - Top-tier bots write critical paths in Yul or Huff
   - 30-50% gas reduction on hot paths
   - Transfer/approve patterns in assembly
   - Custom ABI encoding/decoding

5. Contract architecture
   - Minimal proxy pattern for deployment
   - No constructor logic (use immutables)
   - Single-function entry point (save on function selector dispatching)
```

**Competitive gas benchmarks (2025):**
```
Operation                     Solidity    Yul/Huff    Savings
V3 swap (direct pool call)    ~120K       ~80K        33%
V2 swap (direct pair call)    ~80K        ~55K        31%
2-hop arb (V3→V3)             ~260K       ~170K       35%
ERC20 transfer                ~50K        ~35K        30%
Coinbase tip                  ~21K        ~21K        0% (minimal)
```

---

## 6. Economics and Game Theory

### 6.1 Priority Gas Auctions (PGA)

**Pre-Flashbots era (2020-2021)**: Searchers competed by bidding up gas prices in the public mempool. This was a first-price sealed-bid auction with observable bids — a mess.

```
PGA dynamics:
  Searcher A detects $1000 arb opportunity
  Searcher A submits tx with 100 gwei priority fee
  Searcher B sees A's tx, submits same arb with 110 gwei
  Searcher A escalates to 150 gwei
  ...bidding war...
  Winner pays 950 gwei, capturing only $50 of the $1000 opportunity
  Network congestion: everyone else's gas prices spike
```

**Problems with PGA:**
- Network congestion (gas price spikes for everyone)
- Failed transactions (losers still pay gas for reverted txs)
- Inefficient price discovery (multiple rounds of bidding)
- Centralization pressure (only the fastest bot wins)

**Post-Flashbots (2021+)**: Sealed-bid auction via bundle submission. No public bidding war. Failed bundles don't cost gas. Network not congested by MEV competition.

### 6.2 Builder Auction Dynamics

The builder market is a second-price-adjacent auction with unique characteristics:

```
Per block:
  1. Searchers submit bundles to multiple builders
  2. Each builder aggregates bundles to maximize block value
  3. Builders bid their block value to relays
  4. Relay selects highest bid, forwards to proposer
  5. Proposer blindly commits to highest-value block

Builder profit = total block value - sum of searcher tips - proposer payment
```

**Market structure (2025-2026):**
- 2 builders dominate (>90% of blocks): Titan Builder, Rsync Builder
- Builders are vertically integrated — they run their own searchers too
- Exclusive order flow agreements give builders an edge (they see private transactions first)
- New entrant barrier: need exclusive order flow to build competitive blocks

### 6.3 Searcher Competition and Profit Margins

**Commoditized strategies (2-pool arb, simple sandwich):**
- Hundreds to thousands of competitors
- Margins: 5-15% of gross MEV retained by searcher
- 85-95% goes to builders/proposers as tips
- Effectively zero economic profit after infrastructure costs
- Only viable at massive scale (thousands of trades/day)

**Semi-commoditized (liquidations, 3-hop arb):**
- Dozens of competitors
- Margins: 15-30% retained
- Infrastructure + protocol knowledge is the moat
- Viable for well-capitalized teams

**Long-tail (complex cyclic arb, cross-protocol, novel strategies):**
- Single-digit competitors per niche
- Margins: 30-70% retained
- Intelligence and modeling capability is the moat
- This is where Bella+Poppy should focus

### 6.4 The Race to the Bottom

Every MEV strategy follows the same lifecycle:

```
Discovery    → First mover captures 90%+ of gross value
Competition  → Others replicate, margins compress
Commoditization → Tips approach 100% of value, profit → 0
Obsolescence → Strategy becomes unprofitable after infrastructure costs
```

**Timeline**: Simple arb strategies commoditize in weeks. Complex strategies can maintain edge for months to years. CEX-DEX arb has been competitive since 2021 but still profitable for the top 3 searchers due to infrastructure moats.

**Implication**: The only sustainable edge is continuous discovery of new strategies. Speed advantages are temporary — intelligence advantages are more durable.

### 6.5 Information Leakage and Private Order Flow

**The order flow economy:**
- Wallets (MetaMask, Trust, Rabby) route transactions — they choose which mempool/RPC
- Private order flow = transactions sent directly to builders, not public mempool
- Builders with more private order flow build more valuable blocks → win more auctions → attract more order flow (flywheel)
- MEV-Share attempts to democratize this — users share order flow but get rebates

**Information hierarchy:**
```
Most informed ← → Least informed

Builder (sees all private + public txs)
  Integrated searcher (sees builder's order flow)
    Independent searcher (sees public mempool + MEV-Share hints)
      Regular user (sees nothing — just sends tx and hopes)
```

---

## 7. The Bella+Poppy Advantage

### 7.1 Predictive vs Reactive MEV

**Current MEV landscape**: Almost entirely **reactive**. Searchers wait for opportunities to appear (pending tx, price discrepancy, health factor drop) then race to execute.

**Bella changes this**: A generative model trained on all DeFi market data can **predict** when opportunities are likely to arise.

```
Reactive searcher:
  Wait for WETH price to drop → liquidation appears → race to liquidate

Bella-powered searcher:
  Predict WETH price drop based on:
    - Cross-asset correlation patterns (BTC moving, ETH lags)
    - Volatility regime transition signals
    - Historical pattern matching across 2000+ tokens
    - Volume/liquidity regime indicators
  Pre-position for liquidation wave BEFORE it happens
  Submit bundle in the same block as the price update
```

**Concrete advantage**: A reactive searcher detects the opportunity and has ~500ms to simulate + submit. A predictive searcher has minutes or hours of advance preparation, can pre-compute optimal bundles, and submit the instant conditions align.

### 7.2 MCTS for Bundle Construction Optimization

Poppy's MCTS (Monte Carlo Tree Search) is uniquely suited for bundle construction:

```
Bundle construction as a search problem:

State:    Current pool states (reserves, liquidity, tick)
Actions:  Swap on pool X, add/remove liquidity, liquidate position Y
Reward:   Net profit after gas and tips
Horizon:  1-N transactions in the bundle

MCTS explores:
  1. Which swaps to include (action selection)
  2. What order to execute them (sequencing)
  3. How much to trade at each step (sizing)
  4. Whether to split across pools (routing)

Traditional searcher: Greedy heuristics, precomputed routes
MCTS: Explores the full action tree, finds non-obvious sequences
```

**Where MCTS excels**: Multi-hop cyclic arbitrage where the optimal path is not the greedy path. A 4-hop cycle through low-liquidity pools might be more profitable than a 2-hop through high-liquidity pools — but only if you get the sizing right at each hop. MCTS naturally handles this search.

### 7.3 Using Historical Data to Identify Underexploited Strategies

We have the full swap history of Ethereum in `raw_swaps.db`. This is a goldmine for strategy discovery:

```python
# Example: Find recurring arbitrage patterns that nobody is capturing

# 1. Identify all cross-pool price divergences > 0.5% in historical data
divergences = query("""
    SELECT a.timestamp, a.pool_id, b.pool_id,
           ABS(a.close - b.close) / a.close as divergence
    FROM ohlcv_pool a
    JOIN ohlcv_pool b ON a.timestamp = b.timestamp
    JOIN pools pa ON a.pool_id = pa.id
    JOIN pools pb ON b.pool_id = pb.id
    WHERE pa.token0_id = pb.token0_id
      AND pa.token1_id = pb.token1_id
      AND divergence > 0.005
""")

# 2. Check which divergences were corrected within 1 block (= someone arbed them)
# vs which persisted for minutes (= uncaptured opportunity)

# 3. Characterize the uncaptured ones:
#    - What token pairs?
#    - What time of day?
#    - What pool types (V2 vs V3, different fee tiers)?
#    - What market conditions (vol regime, gas price)?

# 4. Train Bella to predict when these patterns will recur
# 5. Deploy Poppy to capture them in real-time
```

**Historical analysis targets:**
- Tokens where cross-pool divergence persists longest (underserved by arb bots)
- Time-of-day patterns in liquidation activity (validator behavior varies by timezone)
- Gas price regimes where competition drops (high gas = fewer marginal bots competing)
- Protocol-specific events that create temporary MEV spikes (governance, oracle updates, fee changes)

### 7.4 Cross-Pool and Cross-Protocol Opportunities

**Cross-pool**: Same token, different pools. Our `ohlcv_pool` table preserves these microstructure differences. Bella can learn the lead-lag relationship between fee tiers and predict when divergence will occur.

**Cross-protocol**: Opportunities spanning multiple protocols.

```
Example: Aave + Uniswap + Compound combined opportunity

1. Bella predicts ETH volatility spike (regime transition from calm → volatile)
2. Scan Aave/Compound positions near liquidation threshold
3. When ETH price moves:
   a. Liquidate underwater position on Aave (earn liquidation bonus)
   b. The liquidation creates selling pressure → price impact on Uniswap
   c. Arb Uniswap vs Compound's stale oracle price
   d. Bundle all three operations atomically

No single reactive bot captures the full value chain.
A model that understands all three protocols simultaneously can.
```

### 7.5 Long-Tail Strategies Where Intelligence Beats Speed

**The key insight**: The MEV landscape has bifurcated.

| | Speed-Dominant | Intelligence-Dominant |
|---|---|---|
| **Strategy** | Simple arb, sandwich | Cyclic arb, cross-protocol, prediction |
| **Moat** | Hardware, co-location, latency | Models, data, strategy design |
| **Competition** | Thousands of bots | Handful of teams |
| **Margin** | 5-15% | 30-70% |
| **Capital need** | High (infrastructure) | Moderate (compute for models) |
| **Our position** | Disadvantaged | Strongly advantaged |

**Strategies where Bella+Poppy have structural advantage:**

1. **Predictive liquidation positioning**: Know when liquidation cascades will happen before they start.
2. **Cyclic arbitrage (4+ hops)**: Search space too large for brute-force. MCTS + price prediction finds paths that greedy algorithms miss.
3. **Regime-aware MEV**: During volatility transitions, opportunity density spikes. Bella can predict these transitions.
4. **Cross-chain temporal arbitrage**: Predict L1 price movements, pre-position on L2 where settlement is faster.
5. **Illiquid pair MEV**: Small pools that competitive bots ignore (gas cost vs profit doesn't work at current margins). Bella can identify when these pools will see volume.

---

## 8. Known Risks and Ethics

### 8.1 Sandwich Attacks Are Controversial

Sandwich attacks directly extract value from users. Arguments for and against:

**Against:**
- Pure wealth transfer from uninformed users to bots
- No market efficiency justification (unlike arbitrage)
- Degrades user experience in DeFi
- Could be classified as market manipulation in some jurisdictions
- Reputational risk if publicly associated

**For (the pragmatic view):**
- Users set slippage tolerance — they accept the cost
- If we don't sandwich, someone else will (same user outcome)
- Revenue subsidizes other activities (arb keeps markets efficient)
- MEV-Share and Flashbots Protect give users opt-out mechanisms

**Our position**: Focus on arbitrage and prediction-based strategies. Sandwich attacks have lower technical ceiling (no intelligence advantage), higher reputational risk, and are likely to be mitigated by protocol-level solutions (encrypted mempools, order flow auctions) within 1-2 years.

### 8.2 Legal Grey Areas

**Current regulatory landscape (2025-2026):**
- No jurisdiction has explicitly criminalized on-chain MEV extraction
- ESMA (EU securities regulator) published a risk analysis on MEV implications in July 2025
- US SEC has taken no specific action on MEV, but broad "market manipulation" definitions could apply
- Sandwich attacks most legally vulnerable — most analogous to traditional front-running
- Arbitrage generally considered beneficial (price discovery, market efficiency)

**Practical guidance:**
- Keep operations pseudonymous (standard practice)
- No sandwich attacks on regulated assets or regulated venues
- Arbitrage and liquidation are defensible as market-making / stability functions
- Consult legal counsel if operating from jurisdictions with aggressive crypto enforcement

### 8.3 Protocol-Level MEV Mitigation

**The landscape is shifting against extractive MEV. Strategies must adapt.**

**MEV-Share (live, growing):**
- Users share transaction hints (not full calldata) with searchers
- Searchers can only backrun, not frontrun
- MEV profit is split: ~50-90% back to user
- Reduces sandwich attack surface

**Order Flow Auctions (OFAs):**
- CoW Protocol: Batch auctions eliminate front-running by design
- 1inch Fusion: Solvers compete to fill orders, users get MEV protection
- UniswapX: Intent-based trading, fillers compete on execution quality

**Encrypted Mempools (in development, 2025-2026):**
- Threshold encryption: transactions encrypted until block is finalized
- Searchers can't see transaction contents → no sandwich, no frontrun
- Shutter Network, TrX protocol — active development
- Will fundamentally change MEV landscape if widely adopted

**ePBS (Enshrined PBS — Glamsterdam upgrade):**
- Moves PBS from Flashbots middleware into Ethereum protocol
- Removes relay trust assumptions
- Doesn't eliminate MEV but changes the auction structure
- More transparent, more competitive builder market

**Impact on strategy selection:**
```
Strategy              Current Viability    Post-Encrypted-Mempool
Simple sandwich       High                 Dead
Complex sandwich      High                 Dead
DEX-DEX arb           High                 Unchanged (arbs state, not txs)
CEX-DEX arb           High                 Unchanged
Cyclic arb            Medium               Unchanged
Backrunning           High                 Reduced (less tx visibility)
Liquidation           High                 Unchanged (oracle-driven)
JIT liquidity         Medium               Reduced
Prediction-based      Low (nascent)        Strongly advantaged
```

**Key takeaway**: Strategies that exploit transaction visibility (sandwich, frontrun) will die. Strategies that exploit state/price/protocol mechanics (arb, liquidation, prediction) survive and potentially benefit from reduced competition.

### 8.4 Reputational Risk

- MEV extraction is viewed negatively by much of the DeFi community
- "Extracting value from users" is a PR liability
- Arbitrage is the most defensible narrative ("we keep markets efficient")
- Operating anonymously is standard and expected
- Publishing research / contributing to MEV mitigation infrastructure buys goodwill

---

## 9. Practical Roadmap

### 9.1 What to Build First

**Phase 1: Observation and Analysis (weeks 1-4)**

No live trading. Use the archive node and swap database to understand the landscape.

```
Deliverables:
  [ ] MEV detection on historical data
      - Identify all sandwich attacks in raw_swaps.db (same-block frontrun+backrun patterns)
      - Identify all two-pool arbitrages (cross-pool price convergence within 1 block)
      - Measure: how much MEV has been extracted per day/week/month?
      - Measure: what % of opportunities went uncaptured?

  [ ] Strategy profitability backtesting
      - For each strategy, simulate execution with realistic gas costs
      - Account for competition (if 5 bots see it, your expected win rate is 20%)
      - Rank strategies by: profit potential * win probability - cost

  [ ] Market microstructure analysis
      - Pool-level lead-lag relationships (from ohlcv_pool)
      - Liquidation event clustering (time patterns, cascade patterns)
      - Gas price vs competition relationship
```

**Phase 2: Simple Arbitrage Bot (weeks 5-8)**

The simplest profitable strategy: DEX-DEX two-pool arbitrage using flashloans.

```
Components:
  [ ] Arb detection contract (Solidity, gas-optimized)
      - Direct pool.swap() calls (no router overhead)
      - Balancer flashloan (0% fee)
      - Profit-or-revert check
      - Coinbase tipping

  [ ] Mempool monitor (Python)
      - WebSocket subscription to pending transactions
      - Filter for swap events on monitored pools
      - Price graph maintained in-memory

  [ ] Bundle constructor (Python)
      - Simulate arb against current state
      - Calculate optimal trade size
      - Build and sign bundle
      - Submit to Flashbots relay

  [ ] Profit tracker
      - Log every opportunity detected (captured or missed)
      - Track inclusion rate, profit, gas costs
      - Dashboard for real-time monitoring
```

**Phase 3: Bella-Enhanced Prediction (weeks 9-16)**

Integrate Bella's market predictions into MEV strategy selection.

```
Components:
  [ ] Regime prediction service
      - Bella generates regime forecasts (crash, rally, volatile, calm, sideways)
      - Exposed as API to the MEV pipeline
      - Update frequency: per-minute

  [ ] Predictive liquidation monitoring
      - Bella predicts volatility spikes → pre-compute which positions go underwater
      - Pre-build liquidation bundles, ready to submit when price moves
      - Backtest: compare reactive vs predictive liquidation capture rate

  [ ] MCTS bundle optimizer (Poppy integration)
      - Given a set of detected opportunities, MCTS explores bundle construction
      - Multi-hop routing, optimal sizing, sequencing
      - Compare to greedy baseline
```

**Phase 4: Long-Tail and Cross-Protocol (weeks 17+)**

```
Components:
  [ ] Cross-protocol opportunity detection
      - Monitor Aave, Compound, Maker, Morpho simultaneously
      - Detect compound opportunities (liquidation → arb → second liquidation)

  [ ] Cyclic arbitrage (4+ hops)
      - MCTS-powered cycle detection on the full pool graph
      - Bella predictions for which cycles will become profitable

  [ ] Cross-chain MEV
      - L2 monitoring (Arbitrum, Base)
      - Pre-positioned inventory for L1→L2 arbitrage
      - Bridge-aware strategy (when to bridge vs when to use inventory)
```

### 9.2 Infrastructure Checklist

```
Already have:
  [x] Reth archive node (full Ethereum history)
  [x] Raw swap database (raw_swaps.db)
  [x] OHLCV data (per-pool and merged)
  [x] Bella (market prediction model)
  [x] Poppy (MCTS + Transformer agent)

Need to build/acquire:
  [ ] WebSocket mempool access (Reth supports this natively)
  [ ] Flashbots authentication key pair (free, just generate)
  [ ] MEV smart contract (Solidity, deploy to mainnet)
  [ ] Bundle submission pipeline (Python → Flashbots relay)
  [ ] Real-time price graph (in-memory, updated per-block)
  [ ] Transaction simulation (local revm or eth_callBundle)
  [ ] Monitoring/alerting system

Optional but valuable:
  [ ] Premium mempool feed (Bloxroute BDN, ~$1-5K/month)
  [ ] Co-located server near relay infrastructure (~$200-500/month)
  [ ] Multiple builder API connections (Titan, Rsync, Flashbots)
  [ ] L2 node access (Arbitrum Nitro, Base op-geth)
```

### 9.3 Testing on Testnets and Historical Data

**Historical backtesting (primary method):**
```
Use the archive node + raw_swaps.db:
  1. Replay historical blocks
  2. At each block: detect opportunities using current strategy
  3. Simulate bundle execution against historical state (eth_call with state override)
  4. Record: would this have been profitable? How much? What competition existed?
  5. Benchmark: compare to actual MEV extracted (visible on-chain)
```

**Testnet testing (secondary — for contract verification only):**
- Goerli/Sepolia for contract deployment and basic smoke tests
- Flashbots provides a Goerli relay for testing bundle submission flow
- Testnets lack real MEV opportunities — useful for contract correctness, not strategy validation

**Paper trading (recommended before going live):**
- Run full pipeline against live mempool
- Detect opportunities, construct bundles, simulate — but don't submit
- Log everything: opportunity, bundle, simulated profit, competition
- Run for 1-2 weeks, compare detection rate to actual on-chain MEV

### 9.4 Capital Requirements by Strategy

| Strategy | Capital Needed | Risk Level | Expected Daily Revenue (mature) |
|----------|---------------|------------|-------------------------------|
| DEX-DEX arb (flashloan) | Gas only (~$50-200/day) | Minimal | $100-1,000 |
| CEX-DEX arb | $100K-1M (inventory) | Medium (price exposure) | $1,000-10,000 |
| Sandwich | Gas only (~$50-200/day) | Minimal (revert risk) | $500-5,000 |
| Liquidation (flashloan) | Gas only | Minimal | $100-2,000 (lumpy) |
| JIT liquidity | $50K-500K (capital to provide) | Medium | $200-1,000 |
| Cross-chain arb | $200K-2M (dual inventory) | High (bridge risk) | $2,000-20,000 |

**Starting capital recommendation**: $5K-10K for gas and initial contract deployment. Flashloan-based strategies require no trading capital.

### 9.5 Scaling Considerations

**Compute scaling:**
- Bella inference: GPU (already available — shares with training GPU)
- MCTS search: CPU-intensive, scales with core count. Ryzen 7 5800X (8c/16t) is adequate for initial strategies.
- Mempool processing: Single-threaded event loop is fine for Ethereum's ~15 TPS. L2s with higher TPS need parallel processing.

**Capital scaling:**
- Flashloan strategies scale without capital — flashloan amount scales with opportunity size
- Inventory strategies scale linearly with capital deployed
- CEX-DEX arb has diminishing returns — capital must be split across multiple CEX accounts

**Strategy scaling:**
- Each strategy is independent — run multiple strategies in parallel
- Shared infrastructure (mempool feed, price graph, simulation engine) amortizes across strategies
- Bella predictions benefit all strategies simultaneously

**Geographic scaling:**
- Start with single server (current infrastructure is fine)
- If competitive on latency-sensitive strategies: add co-located servers
- For cross-chain: separate node infrastructure per chain

---

## Appendix: Key Addresses and Constants

### Contract Addresses (Ethereum Mainnet)

```
Uniswap V3 Router:       0xE592427A0AEce92De3Edee1F18E0157C05861564
Uniswap V3 Router02:     0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45
Uniswap V3 Factory:      0x1F98431c8aD98523631AE4a59f267346ea31F984
Uniswap V3 Quoter V2:    0x61fFE014bA17989E743c5F6cB21bF9697530B21e
Uniswap V2 Router:       0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
Uniswap V2 Factory:      0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f
Universal Router:        0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD

Sushiswap Router:        0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F
Sushiswap Factory:       0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac

Aave V3 Pool:            0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2
Aave V3 Oracle:          0x54586bE62E3c3580375aE3723C145253060Ca0C2

Compound V3 (cUSDCv3):   0xc3d688B66703497DAA19211EEdff47f25384cdc3

WETH:                    0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
USDC:                    0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48
USDT:                    0xdAC17F958D2ee523a2206206994597C13D831ec7
DAI:                     0x6B175474E89094C44Da98b954EedeAC495271d0F
WBTC:                    0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599

Flashbots Relay:         https://relay.flashbots.net
Flashbots Protect RPC:   https://rpc.flashbots.net
```

### Event Signatures

```
Uniswap V3 Swap:    0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67
Uniswap V2 Swap:    0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822
Uniswap V3 Mint:    0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde
Uniswap V3 Burn:    0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c
ERC20 Transfer:     0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef
ERC20 Approval:     0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925
```

### Gas Constants

```
ETH transfer:              21,000 gas
ERC20 transfer:            ~50,000 gas
ERC20 approve:             ~45,000 gas
Uniswap V3 swap:           ~120,000 gas (direct pool call)
Uniswap V2 swap:           ~80,000 gas (direct pair call)
Uniswap V3 via router:     ~150,000 gas
Aave V3 flashloan:         ~80,000 gas (overhead, excludes callback)
Aave V3 liquidation:       ~350,000 gas
Compound V3 liquidation:   ~300,000 gas
```

---

*This document describes the MEV landscape as of 2026-02-25. The field evolves rapidly — encrypted mempools and ePBS will reshape the strategy space within 12-18 months. Update this doc accordingly.*

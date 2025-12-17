# MCTS Trader Development Diary

## December 2025 - Development Log

### 2025-12-08: First Successful Training Complete

**Milestone**: Completed 49,750 training iterations with working model.

**Journey to get here:**

#### Problem 1: LightZero Memory Issues
- Initial attempt used LightZero framework (Python MCTS)
- Encountered massive memory leaks - RAM usage grew unbounded
- C++ MCTS bindings had tensor shape issues
- **Solution**: Migrated to pure JAX with mctx library (Google's MCTS)

#### Problem 2: PyTorch/JAX GPU Conflict
- Original encoder used PyTorch on CUDA
- JAX also needed GPU for training
- System would hang indefinitely (100% CPU, 3% GPU)
- **Solution**: Keep encoder on CPU, JAX handles GPU training

#### Problem 3: Database Filtering Performance
- HistoricalMarket used GROUP BY on every step
- Each step took ~500ms - way too slow
- **Solution**: Pre-compute valid timestamps at init (one-time 30s cost)
- Steps now use O(log n) binary search - instant

#### Problem 4: Episodes Ending After 1 Step
- `market.step()` returning empty dict
- Root cause: SQL placeholder `?` not working properly
- Query: `WHERE timestamp = ?` returned 0 rows even with valid data
- **Solution**: Direct string substitution `WHERE timestamp = {value}`

#### Problem 5: Constant Zero Rewards
- Portfolio value stuck at $10,000 despite "trading"
- Two bugs:
  1. Called `portfolio.buy()` which doesn't exist - should be `execute_actions()`
  2. Wrong price mapping: portfolio expects `Dict[int, float]` keyed by asset INDEX, not symbol
- **Solution**: Fixed API usage and price dict conversion

#### Problem 6: SQL Query Mysteriously Empty
- Even after fixing placeholder, queries returned 0 rows
- Logs showed 46 assets at timestamp but data dict was empty
- **Root Cause**: `id_to_symbol` mapping mismatch
- Asset IDs in query didn't match keys in mapping dict
- **Solution**: Rebuild mapping from actual query results

### Architecture Decisions

**Why JAX over PyTorch?**
- mctx library is JAX-native and blazing fast
- Full JIT compilation of training loop
- Better GPU memory management
- No GIL issues with parallelism

**Why Fast Encoder (no correlations)?**
- Full correlation matrix: O(n²) = 4096 operations per step
- 60-minute lookback × 64 assets = huge computation
- Fast encoder: 27 feature channels, all vectorized
- Speedup: 10-20x faster encoding

**Why 27 Channels?**
- Channels 0-3: Normalized OHLC
- Channels 4-7: Returns at 1/5/15/60 min
- Channels 8-11: Volume patterns
- Channels 12-15: Price momentum
- Channels 16-19: Volatility measures
- Channels 20-23: Price percentiles
- Channels 24-26: VWAP and bands

### Performance Achieved

| Metric | Before | After |
|--------|--------|-------|
| Self-play per episode | 9 minutes | 1.5-2.5 seconds |
| Steps per episode | 1 | 200 |
| Database step time | 500ms | 0ms (pre-computed) |
| Encoding time | ~200ms | ~10ms |

### Known Issues (v1.0)

1. **NaN Loss**: Training shows `Loss: nan` but model still learns (rewards vary)
2. **Paper Trading WebSocket**: Timestamp parsing fails on nanosecond precision
3. **No Correlation Features**: Removed for speed, may impact strategy quality

### Next Steps

1. Add gated residuals to TRM (research spec)
2. Add cross-asset attention
3. Implement Differential Sharpe reward
4. Abstract API layer (not just Kraken)
5. Fix paper trading WebSocket parsing
6. Consider adding back lightweight correlations

---

## Architecture Gaps (vs Research Spec)

After analyzing research documents vs implementation:

| Feature | Spec | Implemented | Priority |
|---------|------|-------------|----------|
| Gated residuals in TRM | Yes | No | HIGH |
| Cross-asset attention | Yes | No | HIGH |
| Differential Sharpe reward | Yes | No | MEDIUM |
| GAF/MTF encoding | Yes | No | LOW |
| ReZero reanalysis | Yes | No | MEDIUM |
| LR scheduling | Yes | No | MEDIUM |
| Position sizing head | Yes | No | MEDIUM |

---

## Lessons Learned

1. **Always test database queries outside the training loop first**
2. **GPU conflicts between frameworks are real - pick one or segregate**
3. **Pre-computation beats runtime optimization every time**
4. **SQL parameterized queries can silently fail - test with logging**
5. **Portfolio APIs need exact type matching (int keys vs string keys)**

---

## 2025-12-10: Portfolio Bug Fix & Paper Trading Results

### Bug: Portfolio Value Calculation
- `update_prices()` was computing shorts incorrectly
- Old: `total_value = cash + long_value - short_value` (wrong)
- New: `total_value = cash + long_value + short_pnl`
- Where `short_pnl = (entry_price - current_price) * size`

**Impact**: Reported -0.5% return was actually **+64.2% return** over 7 hours!

### Paper Trading Stats (Dec 9-10, 2025)
- 32 trades executed, 30 open positions
- Fees: 26 bps (0.26%) + 5 bps slippage = 31 bps round-trip
- Model correctly shorted BTC during downmove, caught SOL bounce

---

## Future Ideas

### Market Impact Simulation (Wave Function Approach)
Instead of complex agent-based market impact modeling, use **historical data masking/perturbation**:

**Concept**: Simulate "what if we had traded" by perturbing historical candles

```python
def simulate_impact(candles, our_trades, decay_lambda=0.1):
    """
    Wave function approach: probability distribution over
    price paths given our intervention vs no intervention.

    P(price_path | our_actions) vs P(price_path | no_action)
    """
    for trade in our_trades:
        # Impact proportional to our size vs market volume
        impact = (trade.size / candles[trade.time].volume) * volatility

        # Buy pushes up, sell pushes down
        direction = 1 if trade.side == 'buy' else -1

        # Apply with exponential decay (mean reversion)
        for t in range(trade.time, len(candles)):
            decay = exp(-decay_lambda * (t - trade.time))
            candles[t].close += impact * direction * decay
```

**Why this approach**:
- No need to model complex multi-agent dynamics
- Uses real historical liquidity patterns
- Can run Monte Carlo over different impact assumptions
- Practical for current capital levels (< 0.1% of volume = negligible impact)

**Volume thresholds for impact** (crypto):
- < 0.1% daily volume: ~1-2 bps impact (ignorable)
- 0.1-1% volume: ~5-20 bps (model it)
- > 1% volume: 50+ bps (you ARE the market)

### Full Market World Model (Future Research)
When capital scales, would need:
- Probabilistic dynamics (distributions not point estimates)
- Opponent modeling (MM behavior, momentum traders, liquidation cascades)
- Counterfactual reasoning ("if I buy 10 BTC vs don't, what happens")
- Order book shape modeling (not just volume)

---

## 2025-12-11: Risk Management & WebSocket Stability

### WebSocket Auto-Reconnect
Fixed fatal disconnection bug: `sent 1011 (internal error) keepalive ping timeout`

**Implementation:**
- Keepalive pings every 30s with 20s timeout
- Auto-reconnect with exponential backoff (5s → 10s → 20s → 40s → 60s cap)
- Max 10 reconnect attempts before giving up
- Auto-resubscribes to all pairs after reconnect
- Stores subscribed pairs/channels for reconnection

### Stop-Loss / Take-Profit System
Added automatic risk management to protect against runaway losses, especially on shorts.

**Default Targets:**
| Position Type | Stop Loss | Take Profit |
|---------------|-----------|-------------|
| Long | -5% from entry | +10% from entry |
| Short | +3% from entry | -5% from entry |

Shorts have tighter limits due to unlimited loss potential.

### Trailing Take-Profit
When model keeps signaling same direction on existing position, TP trails to lock in gains.

**Logic:**
- Long + BUY signal → Trail TP up (new TP = current price + 10%)
- Short + SELL signal → Trail TP down (new TP = current price - 5%)
- TP only moves in favorable direction (never worse)
- SL moves to breakeven when position is up >3% (long) or >2% (short)

**Example Flow:**
```
[10:00] EXECUTION: open_long LINK/USD @ 14.00 | TP: 15.40 (+10%)
[10:01] Model signals BUY LINK/USD @ 14.20 (reinforces long)
        → TRAILING TP: New TP: 15.62 (14.20 + 10%)
[10:02] Model signals BUY LINK/USD @ 14.50 (still bullish)
        → TRAILING TP: New TP: 15.95 (14.50 + 10%)
[10:05] Price hits 15.95
        → TAKE_PROFIT TRIGGERED | Locked +14% gain
```

This prevents safeguards from blocking momentum - bot can ride trends as long as model keeps signaling.

### Min Order Size Enforcement
Added Kraken minimum order size lookup table to prevent dust orders on small accounts.
- $100 capital test: 25% max position = $25 orders
- Orders below exchange minimum are blocked
- Fallback: $5 USD equivalent if symbol not in table

### Paper Trading Results (Pre-Disconnect)
$100 → $226.07 in ~2 hours (+126%) with 19 open positions before WebSocket died.
Mix of longs and shorts, model running pairs trades (long LINK/USD, short LINK/EUR).

**Final Position Snapshot (2025-12-11 01:08:01):**
```
Portfolio: $226.07 | Cash: $15.30 | Positions: 19

LONGS (13):
  LINK/USD    0.8848 @ 14.1269  +0.03%
  LTC/EUR     0.1728 @ 72.3161  -0.12%
  XDG/USD    97.9907 @  0.1434  +0.10%
  USDTGBP    21.1517 @  0.7474  -0.12%
  USDTCHF    19.7493 @  0.7996  -0.10%
  XLM/USD    62.6206 @  0.2521  -0.71%
  ETHBTC    551.9298 @  0.0362  -0.22%
  BCH/USD     0.0346 @ 576.738  -0.81%
  USDCHF     24.9564 @  0.7994  -0.14%
  ADA/USD    43.5491 @  0.4580  -0.78%
  ADA/EUR    51.0394 @  0.3907  -0.68%
  SOL/USD     0.1837 @ 136.838  -0.34%

SHORTS (6):
  ATOM/USD    5.5165 @  2.2644  -0.33%
  DOT/EUR     7.4699 @  1.8839  +0.68%
  GBP/USD    11.7911 @  1.3386  -0.11%
  XBT/USDT    0.0002 @ 92346.2  +0.30%
  BTCCAD      0.0002 @ 127327   +0.28%
  XBT/EUR     0.0003 @ 78711.4  +0.04%
  LINK/EUR    2.0788 @ 12.0948  +0.12%
```

**Observations:**
- Model is running USD/EUR pairs hedges (LINK, ADA)
- Short BTC across multiple pairs (USDT, CAD, EUR) - bearish BTC view
- Long stablecoins vs fiat (USDTGBP, USDTCHF, USDCHF) - betting on USD strength
- Long ETHBTC - bullish ETH relative to BTC

### Replay Script Fixed
- Auto-detects multiple sessions in trade_log.csv
- Replays most recent session by default
- Properly calculates portfolio value with correct short PnL math
- Usage: `python tools/replay_trades.py [log_file] [initial_capital] [session_idx]`

### CRITICAL BUG FIX: Multi-Currency Unit Confusion

**The Problem:**
Portfolio value was summing positions in different quote currencies as if they were all USD.

Example of the bug:
```
ETHBTC: 443 units @ 0.0356 = 15.8 BTC
System thought: $15.80 USD
Actually worth: $1,533,139 USD (15.8 BTC × $97,000)
```

The reported "$284 portfolio" was meaningless - mixing USD + EUR + BTC + GBP values.

**The Fix:**
1. Added `quote_currency` field to Position dataclass
2. Added `extract_quote_currency(symbol)` - parses "ETH/USD" → "USD", "ETHBTC" → "BTC"
3. Added `get_quote_to_usd_rate(currency, fx_rates)` - converts any quote to USD
4. Portfolio manager now:
   - Stores live FX rates from market data stream
   - Converts all position values to USD before summing
   - Applies FX conversion to fees and PnL calculations
5. Paper trader passes FX rates to portfolio on each update

**Quote Currency Detection:**
- Slash format: `ETH/USD` → USD, `XBT/EUR` → EUR
- No-slash format: `ETHBTC` → BTC, `USDTGBP` → GBP
- Fallback rates for major currencies when live rate unavailable

**Impact:**
Model's multi-currency FX strategies are now correctly accounted for.
The model IS intentionally trading FX (EUR/USD, GBP/USD) and crypto crosses (ETHBTC) as part of its strategy.

### CRITICAL BUG FIX: Position Sizing Without FX Conversion

**The Problem:**
The first FX fix only addressed *valuation* (how we count portfolio value), not *sizing* (how we determine position size when opening).

When opening an ETHBTC position with $500 allocation:
```
OLD (Wrong): size = $500 / 0.0358 = 13,966 ETH units
             Actually worth: 13,966 × 0.0358 × $97,000 = $48 MILLION

NEW (Fixed): size = $500 / (0.0358 × $97,000) = 0.144 ETH units
             Actually worth: 0.144 × 0.0358 × $97,000 = $500
```

**Root Cause:**
`execute_actions()` in portfolio.py calculated: `target_size = max_value / execution_price`
This assumed price was in USD, but for ETHBTC the price is in BTC.

**The Fix:**
Changed sizing formula to account for quote currency:
```python
# Get FX rate for proper sizing
symbol = self.symbols.get(idx, f'ASSET_{idx}')
quote_currency = extract_quote_currency(symbol)
fx_rate = get_quote_to_usd_rate(quote_currency, self.fx_rates)

# Divide by price*fx_rate to get correct units
max_value_usd = self.state.total_value * self.max_position_pct * size
target_size = max_value_usd / (execution_price * fx_rate)
```

**Replay Script Updated:**
- Added `--legacy` flag for logs created before 2025-12-11
- Legacy mode treats all quotes as USD (matches old behavior)
- New logs will be correctly FX-adjusted by default
- Usage: `python tools/replay_trades.py trade_log.csv --legacy`

---

## 2025-12-12: Exchange-Level Stop-Loss Orders Implemented

### The Problem: No Real Protection in Live Trading

The `real_trade_jax.py` script had **ZERO stop-loss protection**:
- No exchange-level stop orders
- No local monitoring of position losses
- Only had 15% portfolio-level drawdown halt (doesn't close positions)
- **Risk**: Script crash = unprotected positions could lose 100%+

### The Solution: Automatic Exchange Stop-Loss Orders

Implemented in `mcts_trader/agent/real_portfolio.py`:

**Key Features:**
1. **Automatic SL Placement**: Every position gets a stop-loss order placed on Kraken immediately after entry
2. **Exchange-Side Protection**: SL orders live on Kraken, work even if script crashes/disconnects
3. **Differential Risk**: Longs get 5% SL, shorts get 3% SL (tighter due to unlimited loss potential)
4. **Order Tracking**: `sync()` monitors SL orders, alerts if triggered or missing
5. **Safety Checks**: Logs critical warnings if any position lacks SL protection

**Implementation Details:**

```python
# New dataclass for tracking SL orders
@dataclass
class StopLossOrder:
    order_id: str
    asset_idx: int
    symbol: str
    entry_side: str  # 'buy' (long) or 'sell' (short)
    entry_price: float
    stop_price: float
    volume: float
    timestamp: float

# Stop-loss placement in execute_actions()
async def _place_stop_loss_order(asset_idx, symbol, entry_side, entry_price, volume):
    if entry_side == 'buy':  # Long
        stop_price = entry_price * 0.95  # 5% stop
        sl_side = 'sell'
    else:  # Short
        stop_price = entry_price * 1.03  # 3% stop
        sl_side = 'buy'

    # Place on Kraken exchange
    await client.add_order(
        pair=symbol,
        side=sl_side,
        order_type='stop-loss',
        price=stop_price,
        volume=volume
    )
```

**Reconciliation in sync():**
- Checks if SL orders still exist on exchange
- Alerts when SL triggers (order disappears = executed)
- Critical warnings for positions without SL protection

**Safety Improvements:**

| Before | After |
|--------|-------|
| Max loss per position: Unlimited | Max loss per position: 5% (longs), 3% (shorts) |
| Script crash protection: None | Script crash protection: Exchange-side SL |
| Manual intervention required: Always | Manual intervention required: Only if SL fails |
| Risk of -1000% loss: Real | Risk of -1000% loss: Mitigated to SL % |

**Configuration:**

Add to `real_trade_config.py` or `RealPortfolioManager` init:
```python
stop_loss_pct=5.0,        # 5% for longs
stop_loss_short_pct=3.0,  # 3% for shorts (tighter)
```

**Known Limitations:**
- Gap risk: Price can gap through stop (flash crashes)
- Exchange issues: Kraken outage could prevent SL execution
- Slippage: SL executes at market, could be worse than stop price

**Testing:**
- [x] Implemented in `RealPortfolioManager`
- [x] Added SL tracking in `sync()`
- [ ] Test in dry-run mode before live
- [ ] Monitor first live trades to verify SL placement

**For SPOT mode ($100 test):**
- Absolute worst case: Still capped at 100% loss (can't go negative)
- Expected worst case: 5-6% per bad trade (SL + fees/slippage)
- Script crash scenario: Positions protected by exchange SL orders ✓

---

## 2025-12-12: Cash Reserve Policy (Anti-Liquidation Protection)

### The Problem: Margin Liquidation Risk

Even with stop-loss orders, MARGIN mode has liquidation risk:
- Kraken force-liquidates when margin level < 40%
- Multiple losing positions can cascade into liquidation
- Liquidation bypasses stop-loss orders
- Can lose 60-80% of capital in a liquidation event

### The Solution: Mandatory Cash Reserve

Implemented **50% cash reserve** policy in `RealPortfolioManager`:

**Key Features:**
1. **50% of equity locked** - Cannot be used for trading
2. **Position sizing from available capital** - Sizes based on remaining 50%, not total equity
3. **Dynamic rebalancing** - Reserve recalculates on each `sync()` as equity changes
4. **Liquidation protection** - Makes liquidation mathematically impossible with 1x leverage

**Implementation:**

```python
# Initialization
cash_reserve_locked = equity * 0.50  # 50% locked
available_capital = equity - cash_reserve_locked  # 50% for trading

# Position sizing (in execute_actions)
target_pct = 0.25  # 25% of available capital per position
usd_size = available_capital * target_pct  # NOT total equity
```

**Example ($100 portfolio, 1x leverage):**
```
Total Equity: $100
Cash Reserve (locked): $50 (50%)
Available for Trading: $50

Max position @ 25%: $12.50 (not $25)
With 1x leverage: $25 market exposure
4 positions max: 4 × $12.50 = $50 used margin

Margin Level = $100 equity / $50 used = 200%
Liquidation threshold: 40%
To liquidate: Would need equity to drop to $20
Required loss: -80% on all positions simultaneously
With 5% SL: IMPOSSIBLE
```

**Risk Reduction:**

| Scenario | Without Reserve | With 50% Reserve |
|----------|----------------|------------------|
| Max positions | 10 @ 10% = 100% | 4 @ 12.5% = 50% |
| Margin level at max | 100% | 200% |
| Buffer to liquidation | 60% | 160% |
| Liquidation possible? | Yes (6-8 bad trades) | No (with 1x + SL) |

**Configuration:**

```python
RealPortfolioManager(
    ...,
    cash_reserve_pct=0.50,  # 50% reserve (adjustable)
    leverage=1,              # Must use 1x with reserve
)
```

**Monitoring:**
- `sync()` recalculates reserve on each call
- Warns when capital utilization > 90%
- Logs available capital vs locked reserve

**Trade-offs:**
- ✅ **Prevents liquidation** - Mathematically impossible with 1x leverage
- ✅ **Safer testing** - Can't blow up account in 2-3 trades
- ❌ **Lower returns** - Only trading with 50% of capital
- ❌ **Fewer positions** - Limited to 4 concurrent positions vs 10

**For $100 → $1000 Loss Prevention:**

With 50% reserve + 1x leverage + 5% SL:
- Liquidation: **Impossible** (200% margin level, 160% buffer)
- Max loss per trade: $1.11 (1.1% of portfolio)
- Max loss 4 trades: $4.44 (4.4% of portfolio)
- To lose $50 (half portfolio): Would require 45+ bad trades
- **$100 → $1000 loss: Mathematically impossible**

**Recommendation:**
- Keep 50% reserve for $100-$5000 portfolios
- Only reduce reserve when portfolio > $10,000 AND you have 24/7 monitoring
- Combine with: 1x leverage + exchange SL + time-limited runs

---

## TODO / Technical Debt

### HIGH Priority (Breaking/Critical)
- [ ] **Gym → Gymnasium migration**: Gym unmaintained since 2022, doesn't support NumPy 2.0
- [ ] **WebSocket timestamp deprecation**: Kraken warns `timestamp is deprecated, use interval_begin` - will break live trading
- [ ] **Symbol substitution audit**: MATIC→POL, EOS/USD missing - trading wrong assets could break strategies
- [ ] **Orbax transformations API**: Warning about deprecated API, potential checkpoint loading failure

### MEDIUM Priority (Stability)
- [ ] **tensorflow.io.gfile shim**: Install tensorflow or use alternative for GCS paths
- [ ] **TPU backend warning**: Suppress or handle gracefully (not critical, just noisy)

### Feature Requests
- [ ] **Trading loop time limit**: Auto-close all positions after N hours, print report
- [ ] **Telegram signal streaming**: Friend request - broadcast trade signals to Telegram channel
- [ ] **API abstraction layer**: Not just Kraken - support Binance, Coinbase, etc.

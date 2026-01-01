# Mathematical Trading Tactics → Synthetic Curriculum Levels

**Purpose**: Map known mathematical trading strategies to testable synthetic environments

**Source**: Quantitative Finance literature, market microstructure theory, statistical arbitrage

---

## Known Mathematical Trading Tactics

### Category 1: Statistical Arbitrage

#### 1. **Pairs Trading** (Cointegration)
**Math**: Two assets drift apart but revert to mean relationship
```
Spread = Price_A - β·Price_B
Trade when: |Spread| > 2σ
```

**Synthetic Level**:
```python
def _level_pairs_trading():
    """
    Generate 32 cointegrated pairs.
    - Asset_i and Asset_{i+32} are cointegrated
    - Spread follows OU process (Ornstein-Uhlenbeck)
    - Optimal: Short expensive asset, long cheap asset
    """
    for i in range(32):
        # Generate cointegrated pair
        spread = ornstein_uhlenbeck(mean=0, theta=0.15, sigma=0.05)
        price_A = 100 + spread
        price_B = 100 - spread
```

**References**:
- Gatev et al. (2006) "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
- Vidyamurthy (2004) "Pairs Trading: Quantitative Methods and Analysis"

---

#### 2. **Triangle Arbitrage**
**Math**: Exploit mispricing in currency/crypto pairs
```
If: BTC/USD × USD/ETH ≠ BTC/ETH
Then: Arbitrage opportunity exists
```

**Synthetic Level**:
```python
def _level_triangle_arbitrage():
    """
    3 assets: BTC, ETH, USDT
    Occasionally introduce mispricing:
      BTC/USDT = 50000
      ETH/USDT = 2500
      BTC/ETH = 19.5 (should be 20)

    Optimal: BTC → ETH → USDT → BTC (profit 2.5%)
    """
```

**References**:
- Foucault et al. (2013) "Market Microstructure" - Chapter on arbitrage
- Crypto-specific: "Flash Boys 2.0" (Daian et al., 2019)

---

#### 3. **Index Arbitrage**
**Math**: Index price vs component prices
```
Index_fair = Σ(weight_i × price_i)
Trade when: |Index_actual - Index_fair| > threshold
```

**Synthetic Level**:
```python
def _level_index_arbitrage():
    """
    64 assets split into:
    - 60 individual assets
    - 4 "index" assets (each tracking 15 components)

    Periodically: Index lags component changes by 5 steps
    Optimal: Frontrun index rebalancing
    """
```

**References**:
- Roll (1984) "Orange Juice and Weather"
- Madhavan (2000) "Market Microstructure: A Survey"

---

### Category 2: Momentum & Reversal

#### 4. **Time-Series Momentum**
**Math**: Past winners continue winning
```
Signal = Sign(Return_{t-12:t-1})
Trade: Long if signal > 0, short if < 0
```

**Synthetic Level**:
```python
def _level_momentum():
    """
    Assets follow momentum regime:
    - If up last 20 steps → 70% chance continue up
    - If down last 20 steps → 70% chance continue down

    Optimal: Ride trends, cut losers
    """
```

**References**:
- Moskowitz et al. (2012) "Time Series Momentum"
- Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers"

---

#### 5. **Mean Reversion (Intraday)**
**Math**: Prices revert to VWAP
```
Trade when: |Price - VWAP| > 2σ
Direction: Toward VWAP
```

**Synthetic Level**:
```python
def _level_mean_reversion():
    """
    Each asset has daily VWAP = 100
    Prices follow OU process around VWAP
    - Strong reversion (theta=0.3)

    Optimal: Fade extremes, profit from reversion
    """
```

**References**:
- Avellaneda & Lee (2010) "Statistical Arbitrage in the US Equities Market"
- Chan (2009) "Quantitative Trading" - Chapter 3

---

### Category 3: Market Microstructure

#### 6. **Bid-Ask Bounce**
**Math**: Price oscillates between bid and ask due to order flow
```
Trade at bid → Price = bid
Trade at ask → Price = ask
Spread = ask - bid (profit opportunity)
```

**Synthetic Level**:
```python
def _level_bid_ask_bounce():
    """
    Generate tick data with bid-ask spread:
    - Bid: 100 - 0.05%
    - Ask: 100 + 0.05%
    - Last price alternates based on random buy/sell orders

    Optimal: Provide liquidity, earn spread
    """
```

**References**:
- Roll (1984) "A Simple Implicit Measure of the Effective Bid-Ask Spread"
- Glosten & Milgrom (1985) "Bid, Ask and Transaction Prices in a Specialist Market"

---

#### 7. **Order Book Imbalance**
**Math**: Predict short-term direction from order book
```
Imbalance = (Bid_volume - Ask_volume) / (Bid_volume + Ask_volume)
If Imbalance > 0.3 → Price likely to rise
```

**Synthetic Level**:
```python
def _level_order_imbalance():
    """
    Generate order book features:
    - Bid depth (volume at bid)
    - Ask depth (volume at ask)
    - Imbalance predicts next price move (80% accuracy)

    Optimal: Trade in direction of imbalance
    """
```

**References**:
- Cont et al. (2014) "The Price Impact of Order Book Events"
- Cartea et al. (2015) "Algorithmic and High-Frequency Trading"

---

#### 8. **Volume-Weighted Returns**
**Math**: Large volume moves are more informative
```
Signal = Return_t × Volume_t / Avg_Volume
Trade: Stronger signal on high volume
```

**Synthetic Level**:
```python
def _level_volume_weighted():
    """
    Prices with volume:
    - High volume moves (3x avg): Persist next 5 steps
    - Low volume moves (0.3x avg): Revert next 3 steps

    Optimal: Follow high-volume breakouts, fade low-volume
    """
```

**References**:
- Lee & Swaminathan (2000) "Price Momentum and Trading Volume"
- Blume et al. (1994) "Market Statistics and Technical Analysis"

---

### Category 4: Volatility Trading

#### 9. **Volatility Clustering (GARCH)**
**Math**: High volatility follows high volatility
```
σ_t² = α + β·σ_{t-1}² + γ·ε_{t-1}²
Trade: Adjust position size based on predicted volatility
```

**Synthetic Level**:
```python
def _level_volatility_clustering():
    """
    Prices follow GARCH(1,1) process:
    - Volatility autocorrelated (β=0.85)
    - After large move, expect continued high vol

    Optimal: Reduce size after volatility spikes
    """
```

**References**:
- Bollerslev (1986) "Generalized Autoregressive Conditional Heteroskedasticity"
- Engle (2001) "GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics"

---

#### 10. **Volatility Spread Trading**
**Math**: Exploit differences in implied vs realized volatility
```
If: Realized_vol > Implied_vol → Sell options (or equivalent)
If: Realized_vol < Implied_vol → Buy options
```

**Synthetic Level**:
```python
def _level_volatility_spread():
    """
    Two asset groups:
    - Group A: Low realized vol (5% annual), high "implied" (10%)
    - Group B: High realized vol (20%), low "implied" (15%)

    Optimal: Long Group B, short Group A (volatility arbitrage)
    """
```

**References**:
- Derman & Kani (1994) "Riding on a Smile"
- Sinclair (2008) "Volatility Trading"

---

### Category 5: Cross-Asset Patterns

#### 11. **Lead-Lag Relationships**
**Math**: Asset A predicts Asset B with time delay
```
Price_B(t) ≈ α·Price_A(t-k) + noise
Trade: Use A to predict B
```

**Synthetic Level**:
```python
def _level_lead_lag():
    """
    BTC leads altcoins by 10 steps:
    - When BTC pumps → Altcoins follow 10 steps later
    - Correlation = 0.85 (lagged)

    Optimal: Buy altcoins when BTC just pumped
    """
```

**References**:
- Brennan et al. (1993) "Investment Analysis and the Adjustment of Stock Prices to Common Information"
- Chordia & Swaminathan (2000) "Trading Volume and Cross-Autocorrelations in Stock Returns"

---

#### 12. **Flight to Quality**
**Math**: During crashes, money flows to safe assets
```
Crisis event → Sell risky assets, buy safe assets
Price_safe ↑, Price_risky ↓
```

**Synthetic Level**:
```python
def _level_flight_to_quality():
    """
    64 assets split:
    - 8 "stable" assets (BTC, ETH equivalents)
    - 56 "risky" assets (altcoins)

    Crisis events (5% chance per 100 steps):
    - Risky assets crash -20%
    - Stable assets rise +5%

    Optimal: Rotate to stable assets during crisis
    """
```

**References**:
- Baur & Lucey (2010) "Is Gold a Hedge or a Safe Haven?"
- Caballero & Krishnamurthy (2008) "Collective Risk Management in a Flight to Quality Episode"

---

### Category 6: Regime-Based Strategies

#### 13. **Bull/Bear/Sideways Detection**
**Math**: Different tactics for different regimes
```
Bull: Momentum works (trend-follow)
Bear: Reversal works (mean-revert)
Sideways: Range-bound (buy dips, sell rallies)
```

**Synthetic Level**:
```python
def _level_regime_switching():
    """
    Market switches regimes every 100-200 steps:

    Bull (40% of time):
    - Drift = +0.3% per step
    - Momentum = 0.8

    Bear (30% of time):
    - Drift = -0.2% per step
    - Reversal to -10% from peak

    Sideways (30% of time):
    - Drift = 0
    - Range = [95, 105]

    Optimal: Detect regime, apply appropriate strategy
    """
```

**References**:
- Hamilton (1989) "A New Approach to the Economic Analysis of Nonstationary Time Series"
- Ang & Bekaert (2002) "Regime Switches in Interest Rates"

---

#### 14. **Calendar Effects**
**Math**: Time-based patterns (Monday effect, etc.)
```
Monday: Mean return = -0.2%
Friday: Mean return = +0.3%
```

**Synthetic Level**:
```python
def _level_calendar_effects():
    """
    Introduce weekly cycle (5 "days" per week):
    - Day 1 (Monday): -0.5% drift
    - Day 5 (Friday): +0.3% drift
    - Other days: neutral

    Optimal: Sell Monday, buy Friday
    """
```

**References**:
- French (1980) "Stock Returns and the Weekend Effect"
- Keim & Stambaugh (1984) "A Further Investigation of the Weekend Effect"

---

### Category 7: Liquidity & Market Impact

#### 15. **Liquidity Provider Strategy**
**Math**: Earn spread by providing liquidity
```
Post bid at: mid - half_spread
Post ask at: mid + half_spread
Profit: 2 × half_spread per round-trip
```

**Synthetic Level**:
```python
def _level_market_making():
    """
    Price follows random walk with occasional large orders:
    - 95% time: Small noise (±0.1%)
    - 5% time: Large order (market taker, moves price ±1%)

    Optimal: Provide liquidity, earn spread, manage inventory
    """
```

**References**:
- Avellaneda & Stoikov (2008) "High-Frequency Trading in a Limit Order Book"
- Guilbaud & Pham (2013) "Optimal High-Frequency Trading with Limit and Market Orders"

---

#### 16. **Market Impact Minimization**
**Math**: Large orders move prices (slippage)
```
Impact = k × (Order_size / ADV)^α
Optimal: Split large orders (TWAP, VWAP, etc.)
```

**Synthetic Level**:
```python
def _level_market_impact():
    """
    Execution challenge:
    - Must execute 1000 shares of each asset
    - Market impact: Price moves 0.1% per 100 shares traded

    Single large order: -10% slippage
    TWAP (split 10 orders): -1% slippage

    Optimal: Learn TWAP/VWAP-like execution
    """
```

**References**:
- Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions"
- Obizhaeva & Wang (2013) "Optimal Trading Strategy and Supply/Demand Dynamics"

---

### Category 8: Advanced Patterns

#### 17. **Breakout Trading**
**Math**: Prices consolidate then break out
```
Range = [Low, High] for N periods
Breakout when: Price > High or Price < Low
Direction: Continue in breakout direction
```

**Synthetic Level**:
```python
def _level_breakout():
    """
    Assets alternate:
    - Consolidation: 30 steps in tight range [99, 101]
    - Breakout: 20 steps trending (+5% or -5%)

    Optimal: Wait in consolidation, trade breakout
    """
```

**References**:
- Donchian (1960s) "Channel Breakout System"
- Turtle Traders (Dennis & Eckhardt, 1980s)

---

#### 18. **Grid Trading**
**Math**: Buy dips, sell rallies systematically
```
Grid levels: [90, 95, 100, 105, 110]
Buy when: Price crosses down through level
Sell when: Price crosses up through level
```

**Synthetic Level**:
```python
def _level_grid_trading():
    """
    Prices oscillate in range [80, 120]:
    - Mean-reverting with σ = 10
    - No sustained trends

    Optimal: Place grid orders, profit from oscillation
    """
```

**References**:
- Seykota (1970s-80s) "Trading Systems"
- Tharp (1999) "Trade Your Way to Financial Freedom"

---

## How to Find More Tactics

### Academic Sources

1. **SSRN** (Social Science Research Network)
   - Search: "statistical arbitrage", "algorithmic trading", "market microstructure"
   - Filter: Finance, Quantitative Finance
   - https://www.ssrn.com/

2. **arXiv Quantitative Finance**
   - q-fin.TR (Trading and Market Microstructure)
   - q-fin.ST (Statistical Finance)
   - https://arxiv.org/list/q-fin/recent

3. **Journal of Financial Markets**
   - Premier academic journal for market microstructure
   - https://www.sciencedirect.com/journal/journal-of-financial-markets

4. **Quantitative Finance Journal**
   - Covers all quant topics
   - https://www.tandfonline.com/toc/rquf20/current

### Practitioner Books

1. **"Algorithmic Trading" by Ernest Chan**
   - Pairs trading, mean reversion, momentum
   - Includes Python code examples

2. **"Advances in Financial Machine Learning" by Marcos López de Prado**
   - Modern ML approaches to classic tactics
   - Feature engineering, meta-labeling

3. **"Market Microstructure Theory" by Maureen O'Hara**
   - Deep dive into bid-ask, order book, price formation

4. **"Quantitative Trading" by Ernest Chan**
   - Statistical arbitrage, backtesting, risk management

5. **"Inside the Black Box" by Rishi Narang**
   - Overview of quant strategies used by hedge funds

### Online Resources

1. **QuantConnect / Lean Engine**
   - Open-source algorithmic trading platform
   - Strategy library with implementations
   - https://github.com/QuantConnect/Lean

2. **Quantopian Lectures** (archived)
   - Educational notebooks on quant tactics
   - https://github.com/quantopian/research_public

3. **QuantLib**
   - Derivatives pricing library (for vol strategies)
   - https://www.quantlib.org/

4. **SeekingAlpha / Medium (Quant sections)**
   - Practitioner insights
   - Strategy breakdowns

### Classic Papers (Must-Read)

1. **Jegadeesh & Titman (1993)** - Momentum
2. **Gatev et al. (2006)** - Pairs trading
3. **Almgren & Chriss (2000)** - Optimal execution
4. **Avellaneda & Stoikov (2008)** - Market making
5. **Cont et al. (2014)** - Order book dynamics
6. **Moskowitz et al. (2012)** - Time-series momentum
7. **Roll (1984)** - Bid-ask spread estimation

---

## Mapping Tactics to Curriculum

### Recommended Level Sequence

**Phase 1: Basics** (Current TASK_02)
- Level 0: Sine waves (sanity check)
- Level 1: Lagged pairs (correlation)
- Level 2: Dip crashes (patience)
- Level 3: Noisy trends (fee awareness)

**Phase 2: Statistical Arbitrage**
- Level 4: Portfolio rotation (rebalancing)
- Level 5: Pairs trading (cointegration)
- Level 6: Triangle arbitrage (cross-pair)
- Level 7: Index arbitrage (frontrunning)

**Phase 3: Momentum & Reversal**
- Level 8: Time-series momentum (trend-follow)
- Level 9: Mean reversion (OU process)
- Level 10: Breakout trading (consolidation → trend)

**Phase 4: Market Microstructure**
- Level 11: Bid-ask bounce (spread capture)
- Level 12: Order book imbalance (predict direction)
- Level 13: Market making (liquidity provision)
- Level 14: Market impact (execution algorithms)

**Phase 5: Volatility**
- Level 15: Volatility clustering (GARCH)
- Level 16: Volatility spread trading

**Phase 6: Regime & Cross-Asset**
- Level 17: Bull/bear/sideways detection
- Level 18: Lead-lag relationships
- Level 19: Flight to quality (crisis rotation)

**Phase 7: Advanced**
- Level 20: Calendar effects
- Level 21: Grid trading
- Level 22: Multi-tactic fusion (combine all)

---

## Implementation Strategy

### For Gemini

**Immediate** (TASK_02):
- Implement Levels 0-3 (validation phase)
- Get Level 0 converging (prove architecture works)

**Next** (TASK_04 extended):
- Add 4-7 (statistical arbitrage)
- Focus on pairs trading (most important)

**Future** (TASK_08+):
- Levels 8-22 as separate tasks
- Each level = 1-2 days implementation
- Total: ~6 weeks to build full curriculum

### Validation Approach

For each level:
1. **Generate data** (synthetic)
2. **Manual strategy** (verify profitability)
3. **Train agent** (test if learns)
4. **Compare** (agent vs manual)

If agent can't beat manual strategy → Architecture problem

---

## Key Insight

**You're right**: There ARE mathematically defined tactics waiting to be curriculum-ized.

**The literature exists**: 40+ years of quantitative finance research.

**Our advantage**:
- Synthetic levels let us test in **controlled environments**
- No need for real data (initially)
- Can isolate ONE tactic per level
- Build complexity incrementally

**Next step**:
- Start with TASK_02 (Levels 0-3)
- Then systematically add tactics from this map
- Each level teaches ONE mathematical concept
- By Level 22, agent has seen ALL known tactics

This is the **AlphaGo curriculum approach** for trading.

---

## References Summary

**Books**:
- Chan (2009, 2013) - Practical quant strategies
- López de Prado (2018) - Modern ML approaches
- O'Hara (1995) - Market microstructure theory
- Cartea et al. (2015) - HFT and algorithmic trading
- Sinclair (2008) - Volatility trading

**Papers** (Top 10):
1. Jegadeesh & Titman (1993) - Momentum
2. Gatev et al. (2006) - Pairs trading
3. Almgren & Chriss (2000) - Execution
4. Avellaneda & Stoikov (2008) - Market making
5. Cont et al. (2014) - Order book
6. Moskowitz et al. (2012) - Time-series momentum
7. Roll (1984) - Bid-ask spread
8. Bollerslev (1986) - GARCH
9. Hamilton (1989) - Regime switching
10. French (1980) - Calendar effects

**Online**:
- SSRN, arXiv q-fin
- QuantConnect, Quantopian (archived)
- Academic journals (JFM, Quantitative Finance)

This is your roadmap. The knowledge exists - we just need to systematically encode it as synthetic levels.

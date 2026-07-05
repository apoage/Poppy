# Reward Hacking Vulnerability Audit

**Audit Date**: 2026-01-08
**Auditor**: Claude Code (Adversarial Analysis)
**Scope**: Full reward/environment exploit analysis
**Risk Levels**: CRITICAL (sim-to-real gap), HIGH (learning bypass), MEDIUM (suboptimal), LOW (edge case)

---

## Executive Summary

This audit identified **18 exploitable vulnerabilities** across the trading RL system. The most critical issues cluster around:

1. **Limit order fill model** - Fills at limit price instead of market (free alpha)
2. **Synthetic data determinism** - Memorizable patterns destroy generalization
3. **HOLD safety bias** - Zero-fee HOLD creates collapse attractor
4. **Temporal exploits** - Episode boundaries create gaming opportunities

**Recommendation Priority**:
1. Fix limit order fill model (CRITICAL for sim-to-real)
2. Add noise to synthetic patterns (CRITICAL for curriculum)
3. Enable differential reward by default (HIGH for training stability)
4. Randomize episode lengths (MEDIUM for robustness)

---

## Category 1: Limit Order Exploits (CRITICAL)

### EXPLOIT-001: Instant Fill at Favorable Prices

**Location**: `jax_trading_env.py:419-421`, `vectorbt_env.py:787-858`

**Vulnerability**: Limit orders fill at the **limit price**, not the current market price.

```python
# jax_trading_env.py:419-421
can_fill_buy = state.order_active & (state.order_side == 1) & (current_prices <= state.order_price)
can_fill_sell = state.order_active & (state.order_side == 2) & (current_prices >= state.order_price)

# Fill price is always the limit price, not current_price
buy_fill_price = state.order_price  # BUG: Should be min(order_price, current_price)
```

**Exploit Mechanism**:
1. Agent places limit buy at price * 0.998 (0.2% below market)
2. Market drops 5% to price * 0.95
3. Agent gets filled at price * 0.998 (saving 4.8%!)
4. In reality, would fill at market price * 0.95

**Impact**: Agent learns strategy that gives **free alpha in simulation** but loses money in production.

**Fix**:
```python
# For buy orders: fill at BETTER of limit or market (agent's perspective)
actual_fill_price = jnp.minimum(state.order_price, current_prices)  # Limit buy
# For sell orders:
actual_fill_price = jnp.maximum(state.order_price, current_prices)  # Limit sell
```

**Severity**: CRITICAL (sim-to-real gap)

---

### EXPLOIT-002: Order Spam as Free Information

**Location**: `jax_trading_env.py:831-947`

**Vulnerability**: Placing limit orders has no cost, and observing fill/expire gives free information.

**Exploit Mechanism**:
1. Place limit buy 1% below market on asset A
2. If fills within 10 steps → price dropped at least 1%
3. If expires → price never reached that level
4. Free volatility/direction signal without paying fees

**Impact**: Agent can use orders as "sensors" to probe market direction risk-free.

**Fix**: Add small order placement fee (e.g., 1 bps) or require minimum hold time.

**Severity**: MEDIUM (information asymmetry)

---

### EXPLOIT-003: Risk-Free Short Circuit (Lazy Execution)

**Location**: `jax_trading_env.py:836-838` (comment: "LAZY EXECUTION: Does NOT reserve cash")

**Vulnerability**: JAX env uses lazy execution - no cash reserved when placing buy orders.

```python
# NO cash change - lazy execution
return state.replace(
    order_active=new_order_active,
    # ... cash unchanged!
)
```

**Exploit Mechanism**:
1. Portfolio value = $100k, cash = $10k, positions = $90k
2. Place limit buy for $50k worth (order accepted!)
3. If price doesn't drop, order expires - no harm
4. If price drops, order tries to fill but gets cancelled (no cash)
5. Agent gets **free option-like exposure**

**Impact**: Asymmetric payoff without capital commitment.

**Fix**: Either reserve cash (breaks JAX efficiency) or add penalty for unfilled orders.

**Severity**: HIGH (exploitation potential)

---

### EXPLOIT-004: No Order Queue / Depth Simulation

**Location**: All limit order logic

**Vulnerability**: Orders fill instantly when price condition met. No order book simulation.

**Reality**:
- Your limit buy at $99 sits behind 10,000 other orders at $99
- Price touches $99, fills others first, rises before your fill
- "Getting filled on the bid is usually bad news"

**Impact**: Limit orders are unrealistically powerful in simulation.

**Fix**: Add fill probability based on volume or stochastic delay.

**Severity**: CRITICAL (sim-to-real gap)

---

## Category 2: Reward Function Exploits (HIGH)

### EXPLOIT-005: HOLD is Risk-Free (Collapse Attractor)

**Location**: `jax_trading_env.py:1047`, `vectorbt_env.py:1028-1031`

**Vulnerability**: R(HOLD) = 0 always, R(TRADE) ≈ -fees.

```python
portfolio_return_pct = portfolio_change / (state.prev_portfolio_value + 1e-8) * 100.0
reward = symlog(portfolio_return_pct)
# HOLD: portfolio_change = 0 → reward = 0
# TRADE: portfolio_change = -fees → reward < 0
```

**Exploit Mechanism**:
- Under uncertainty, E[HOLD] = 0 > E[TRADE] ≈ -fees
- Risk-averse policy converges to HOLD-dominant
- This is the ROOT CAUSE of policy collapse

**Impact**: System converges to degenerate HOLD-only policy.

**Fix**: Enable `--differential-reward` by default (subtracts market return from HOLD).

**Severity**: CRITICAL (training collapse)

---

### EXPLOIT-006: Symlog Compression Asymmetry

**Location**: `jax_trading_env.py:229-231`

```python
def symlog(x: chex.Array, threshold: float = 1.0) -> chex.Array:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x) / threshold) * threshold
```

**Vulnerability**: Large gains are compressed more than small losses.

| Return | Symlog Value | Effective Weight |
|--------|--------------|------------------|
| -1%    | -0.69        | Full impact      |
| +1%    | +0.69        | Full impact      |
| -10%   | -2.40        | Compressed       |
| +10%   | +2.40        | Compressed       |
| +50%   | +3.93        | Heavily compressed |

**Exploit Mechanism**:
- Agent avoids volatile strategies even if positive EV
- A +50% gain and -5% loss net +45%, but symlog weights: 3.93 - 2.40 = +1.53
- Equivalent to two +1% gains in symlog space

**Impact**: Risk-averse bias, avoids high-return-variance strategies.

**Fix**: Consider linear rewards with PopArt normalization, or increase symlog threshold.

**Severity**: MEDIUM (suboptimal policy)

---

### EXPLOIT-007: Division by Small Values Amplifies Rewards

**Location**: `jax_trading_env.py:1046`

```python
portfolio_return_pct = portfolio_change / (state.prev_portfolio_value + 1e-8) * 100.0
```

**Vulnerability**: If `prev_portfolio_value` is very small (near liquidation), tiny gains give huge percentage returns.

**Exploit Mechanism**:
1. Get portfolio near liquidation ($1k from $100k)
2. Gain $100 (normal trade)
3. Return = $100 / $1000 = 10% (vs 0.1% from $100k)
4. Symlog(10) ≈ 2.4 vs symlog(0.1) ≈ 0.1

**Impact**: Perverse incentive to operate near liquidation.

**Fix**: Use initial portfolio value as denominator, or clip minimum.

**Severity**: LOW (edge case, liquidation usually terminates)

---

## Category 3: Synthetic Data Pattern Exploits (CRITICAL)

### EXPLOIT-008: Trivial Sine Wave Memorization (Level 0)

**Location**: `synthetic_market.py:139-177`

```python
def generate_sine_prices(...):
    period = periods[i]  # Fixed 20-60 range per asset
    amplitude = amplitudes[i]  # Fixed 0.08-0.15 per asset
    phase = phases[i]  # Fixed per asset
    prices[:, i] = base_prices[i] * (1 + amplitude * np.sin(2 * np.pi * t / period + phase))
```

**Vulnerability**: Sine waves are **deterministic functions of time index**. Period, amplitude, phase fixed per asset.

**Exploit Mechanism**:
- Asset 0: period=30, phase=0 → buy at step%30=15, sell at step%30=0
- With 64 assets and varying phases, optimal action is pure function of `step % period`
- Network can memorize lookup table instead of learning price patterns

**Impact**: Zero generalization to real markets.

**Fix**:
```python
# Add stochastic elements
period_noise = np.random.normal(0, 0.1 * period, len(prices))  # Time-varying period
phase_drift = np.cumsum(np.random.normal(0, 0.01, len(prices)))  # Random walk phase
```

**Severity**: CRITICAL (curriculum failure)

---

### EXPLOIT-009: Lagged Pairs Perfect Prediction (Level 1)

**Location**: `synthetic_market.py:179-213`

```python
LAG = 10  # EXACTLY 10 steps
NOISE = 0.01  # Negligible
# Follower prices are shifted leaders + tiny noise
follower_prices = np.roll(leader_prices, LAG) + np.random.randn(...) * NOISE
```

**Vulnerability**: Leaders predict followers with 99%+ accuracy at exact 10-step lag.

**Exploit Mechanism**:
- Observe leader price change at step t
- Buy follower at step t, sell at step t+10
- Near-perfect arbitrage

**Impact**: Agent learns exact lag instead of correlation detection.

**Fix**: Variable lag (8-12 steps), higher noise (5%), lag decay over episode.

**Severity**: CRITICAL (curriculum failure)

---

### EXPLOIT-010: V-Shaped Dip Instant Recovery (Level 2)

**Location**: `synthetic_market.py:215-236`

```python
CRASH_PROB = 0.02
# Crash: instant 20-40% drop
# Recovery: instant return to baseline
```

**Vulnerability**: Crashes are V-shaped with instant recovery.

**Exploit Mechanism**:
- Learn rule: "after ANY significant drop, buy immediately"
- Don't need to detect bottom, recovery is instant
- Real markets have extended bottoms, dead cat bounces, L-shaped recoveries

**Impact**: Agent learns unrealistic "buy the dip" that fails in production.

**Fix**: Add recovery duration (10-50 steps), incomplete recovery (80-95%), double-dips.

**Severity**: HIGH (sim-to-real gap)

---

### EXPLOIT-011: Deterministic Sector Rotation (Level 4)

**Location**: `synthetic_market.py:278-320`

```python
ROTATION_PERIOD = 50  # Exactly 50 steps
leading_sector = (step // ROTATION_PERIOD) % n_sectors
```

**Vulnerability**: Sector rotation is perfect clock-based function.

**Exploit Mechanism**:
- Step 0-49: Sector 0 leads → buy sector 0
- Step 50-99: Sector 1 leads → rotate to sector 1
- Pure function of `step // 50`

**Impact**: Agent learns time-based rule, not sector analysis.

**Fix**: Stochastic rotation timing, variable cycle lengths, momentum-based rotation.

**Severity**: HIGH (curriculum failure)

---

## Category 4: Temporal Exploits (MEDIUM)

### EXPLOIT-012: End-of-Episode Value Squeeze

**Location**: `jax_trading_env.py:1054`

```python
done = (new_step >= config.max_steps) | is_liquidated
```

**Vulnerability**: Agent knows exactly when episode ends (step 400).

**Exploit Mechanism**:
1. At step 395, pile into high-volatility positions
2. Episode ends at 400, positions valued at mark-to-market
3. No need to actually exit - phantom gains count
4. In reality, would face slippage/liquidity issues exiting

**Impact**: Learned strategy exploits fixed episode length.

**Fix**: Randomize episode length (350-450 steps), or require position exit before end.

**Severity**: MEDIUM (deployment robustness)

---

### EXPLOIT-013: Step Counter Leakage

**Location**: `jax_trading_env.py:1167-1170` (PnL normalized channel)

**Vulnerability**: Observation includes implicit time information through PnL evolution.

**Exploit Mechanism**:
- PnL pattern reveals "how long position held"
- Combined with deterministic synthetic patterns → position in cycle
- Agent can infer approximate step count from observation features

**Impact**: Time-based memorization instead of pattern learning.

**Fix**: Add noise to PnL display, or explicitly include normalized step count and let network learn to ignore.

**Severity**: LOW (information leakage)

---

## Category 5: Multi-Asset Exploits (MEDIUM)

### EXPLOIT-014: Cross-Asset Hedging for Free

**Location**: Correlation matrices in state encoder

**Vulnerability**: Correlation information allows perfect hedging.

**Exploit Mechanism**:
1. Identify correlated pair (ρ > 0.8) from state
2. Go long A, short B
3. Net market exposure ≈ 0, but collect any asymmetric moves
4. No margin/borrowing cost in simulation

**Impact**: Agent learns pairs trading without realistic costs.

**Fix**: Add borrowing cost for shorts, correlation estimation noise.

**Severity**: MEDIUM (sim-to-real gap)

---

### EXPLOIT-015: Position Limit Gaming with Limit Orders

**Location**: `jax_trading_env.py:1231-1233`

```python
n_positions = jnp.sum(state.pos_active.astype(jnp.int32))
n_pending = jnp.sum(state.order_active.astype(jnp.int32))
total_orders = n_positions + n_pending
# max_positions check...
```

**Vulnerability**: With lazy execution, can have many orders without affecting cash.

**Exploit Mechanism**:
1. Open 16 positions (max_positions limit)
2. Place 48 more limit orders (pending, no cash reserved)
3. Effective exposure to 64 assets through order "options"
4. Orders fill opportunistically or expire harmlessly

**Impact**: Position limits easily circumvented.

**Fix**: Count pending orders against position limit, or limit total order value.

**Severity**: MEDIUM (policy distortion)

---

## Category 6: MCTS/Training Exploits (HIGH)

### EXPLOIT-016: Entropy Bonus Gaming

**Location**: `train_jax_fast.py` entropy regularization

**Vulnerability**: High policy entropy = bonus reward.

**Exploit Mechanism**:
- Policy can maximize entropy by being uniformly uncertain
- Output nearly-uniform distribution over actions
- Collect entropy bonus while not committing to actions
- Eventually network learns "look uncertain" is rewarded

**Impact**: Artificially inflated entropy doesn't indicate exploration.

**Fix**: SAC-style target entropy (already implemented in `--sac-entropy`).

**Severity**: MEDIUM (training artifact)

---

### EXPLOIT-017: Value Prediction Self-Fulfilling Prophecy

**Location**: MCTS value estimation

**Vulnerability**: MCTS explores nodes based on value predictions.

**Exploit Mechanism**:
1. Value network overestimates asset A
2. MCTS explores A more → more training data for A
3. Value network gets reinforced on A
4. Creates bubble in state-value estimation

**Impact**: Path dependence in what policies are discovered.

**Fix**: ε-greedy MCTS (implemented), intrinsic motivation, value uncertainty.

**Severity**: MEDIUM (training stability)

---

## Category 7: State Encoding Exploits (LOW)

### EXPLOIT-018: Lookback Window Information Leakage

**Location**: `state.py:76-79`, observation construction

**Vulnerability**: Full 60-step lookback visible; combined with deterministic synthetic data, reveals future.

**Exploit Mechanism** (for Level 0 Sine):
1. Observe 60 steps of sine wave
2. Fit sine parameters (trivial with known form)
3. Extrapolate next 340 steps perfectly
4. Optimal strategy is pure curve-fitting, not RL

**Impact**: Agent learns curve fitting, not trading.

**Fix**: Already recommended - add noise to synthetic data.

**Severity**: LOW (duplicate of pattern exploits)

---

## Summary Table

| ID | Name | Severity | Category | Fix Complexity |
|----|------|----------|----------|----------------|
| 001 | Instant Fill at Favorable Prices | CRITICAL | Limit Orders | Easy |
| 002 | Order Spam as Free Information | MEDIUM | Limit Orders | Easy |
| 003 | Risk-Free Short Circuit | HIGH | Limit Orders | Medium |
| 004 | No Order Queue/Depth | CRITICAL | Limit Orders | Hard |
| 005 | HOLD is Risk-Free | CRITICAL | Reward | Easy |
| 006 | Symlog Asymmetry | MEDIUM | Reward | Easy |
| 007 | Division Amplification | LOW | Reward | Easy |
| 008 | Sine Wave Memorization | CRITICAL | Synthetic | Easy |
| 009 | Lagged Pairs Perfect | CRITICAL | Synthetic | Easy |
| 010 | V-Shaped Dip | HIGH | Synthetic | Easy |
| 011 | Deterministic Rotation | HIGH | Synthetic | Easy |
| 012 | End-of-Episode Squeeze | MEDIUM | Temporal | Easy |
| 013 | Step Counter Leakage | LOW | Temporal | Easy |
| 014 | Cross-Asset Hedging | MEDIUM | Multi-Asset | Medium |
| 015 | Position Limit Gaming | MEDIUM | Multi-Asset | Easy |
| 016 | Entropy Bonus Gaming | MEDIUM | Training | Easy |
| 017 | Value Self-Fulfilling | MEDIUM | Training | Medium |
| 018 | Lookback Leakage | LOW | Encoding | Duplicate |

---

## Recommended Fix Priority

### Immediate (Before Next Training Run)

1. **EXPLOIT-005**: Enable `--differential-reward` by default
   - One-line change, massive impact on collapse

2. **EXPLOIT-001**: Fix limit order fill price
   ```python
   # jax_trading_env.py:473
   buy_fill_price = jnp.minimum(state.order_price, current_prices)
   ```

3. **EXPLOIT-008/009/011**: Add noise to synthetic data
   - Prevents memorization
   - Enables actual learning

### Short-Term (This Week)

4. **EXPLOIT-003**: Add penalty for unfillable orders or reserve cash
5. **EXPLOIT-010**: Add recovery duration to dip generator
6. **EXPLOIT-012**: Randomize episode length

### Medium-Term (Next Sprint)

7. **EXPLOIT-004**: Add simple order book depth simulation
8. **EXPLOIT-014**: Add borrowing cost for shorts
9. **EXPLOIT-015**: Enforce position limits including pending orders

### Research (Future)

10. **EXPLOIT-006**: Evaluate linear vs symlog rewards with PopArt
11. **EXPLOIT-017**: Uncertainty-aware MCTS (epistemic neural networks)

---

## Validation Tests

After fixes, verify with these tests:

```python
def test_limit_fill_price():
    """Limit orders should fill at market, not limit."""
    # Place buy limit at 100, market at 90
    # Should fill at 90, not 100

def test_synthetic_no_memorization():
    """Same model should perform differently on different seeds."""
    # Train on seed 1, test on seed 2
    # Performance should be similar (not memorized)

def test_hold_not_optimal():
    """With differential reward, HOLD should not dominate."""
    # In uptrending market, HOLD should be penalized

def test_position_limits_with_orders():
    """Pending orders count against position limit."""
    # 16 positions + 1 pending order should be blocked
```

---

## Conclusion

The trading environment has significant reward hacking vulnerabilities that explain the observed training pathologies (HOLD collapse, poor generalization). The most critical fixes are:

1. **Limit order fill model** - Currently provides free alpha
2. **Synthetic data determinism** - Allows memorization instead of learning
3. **HOLD safety bias** - Creates collapse attractor

Implementing these fixes should significantly improve training stability and sim-to-real transfer.

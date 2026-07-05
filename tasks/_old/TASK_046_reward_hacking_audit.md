# TASK_046: Reward Hacking & Exploitation Audit

## Status: COMPLETE (2026-01-08)

## Completion Summary

Full adversarial audit completed. Found **18 exploitable vulnerabilities** across 7 categories:
- 4 CRITICAL (limit order fills, synthetic memorization, HOLD collapse)
- 5 HIGH (lazy execution, V-shaped dips, rotation timing, training exploits)
- 7 MEDIUM (order spam, symlog asymmetry, hedging, position limits, etc.)
- 2 LOW (edge cases)

**Output**: `docs/REWARD_HACKING_AUDIT.md`

**Top 3 Immediate Fixes**:
1. Enable `--differential-reward` by default (HOLD collapse fix)
2. Fix limit order fill price to use market price, not limit price
3. Add noise to synthetic data generators (prevent memorization)

---

## Objective

Adversarial red-team audit of the training system to identify reward hacking, exploitation vectors, and degenerate strategies the agent might learn.

---

## AGGRESSIVE AUDIT PROMPT

You are a ruthless exploit hunter. Your mission: **break this trading RL system**.

Find every way an agent could game rewards without learning real trading skill. Assume the agent is an adversarial optimizer that will find ANY shortcut to maximize reward signal.

### System Under Attack

**Environment**: `mcts_trader/env/vectorbt_env.py` and `mcts_trader/jax_agent/jax_trading_env.py`
**Reward**: Symlog-transformed portfolio returns
**Actions**: 5 per asset (Sell, Hold, Buy, LimitSell, LimitBuy) × 64 assets = 320 actions
**Training**: `train_jax_fast.py` with MCTS policy improvement

### Attack Vectors to Investigate

#### 1. REWARD FUNCTION EXPLOITS
- Can the agent exploit symlog transformation boundaries?
- Are there reward asymmetries between long/short positions?
- Can the agent farm small positive rewards while avoiding large negative ones?
- Does HOLD action leak any reward signal it shouldn't?
- Are there numerical precision exploits (very small trades)?

#### 2. MARKET MECHANICS EXPLOITS
- Can limit orders be placed and cancelled for information gain?
- Is there a spread/slippage exploitation (buy-sell arbitrage)?
- Can the agent exploit the synthetic data generators' patterns too easily?
- Are TP/SL guardrails exploitable (trigger intentionally for guaranteed exit)?
- Can the agent "churn" (excessive trading) to exploit any per-trade bonuses?

#### 3. TEMPORAL EXPLOITS
- Can the agent exploit lookahead bias in state representation?
- Is there information leakage from future prices in any features?
- Can the agent exploit the episode boundary (end-of-episode effects)?
- Are there exploits in the warmup/physics period?

#### 4. MULTI-ASSET EXPLOITS
- Can the agent hedge perfectly to eliminate risk while farming fees?
- Are there cross-asset arbitrage opportunities that are too easy?
- Can correlation features be exploited for guaranteed profits?
- Is the "rotation" curriculum level exploitable?

#### 5. MCTS/TRAINING EXPLOITS
- Can the agent learn to manipulate its own value estimates?
- Are there exploits in the replay buffer sampling?
- Can the agent exploit PopArt normalization dynamics?
- Is there a way to game the entropy bonus?

#### 6. LIMIT ORDER EXPLOITS (CRITICAL)
- Can limit orders be placed at unrealistic prices for guaranteed fills?
- Is the fill logic exploitable (instant fills that shouldn't happen)?
- Can the agent spam limit orders for any benefit?
- **Current flaw**: Orders too close to market price fill instantly - unrealistic!

### Output Format

For each exploit found:
```
EXPLOIT: [Name]
SEVERITY: Critical/High/Medium/Low
LOCATION: [file:line]
MECHANISM: How the agent exploits this
EVIDENCE: Code snippet or logic showing the vulnerability
IMPACT: What degenerate behavior this enables
FIX: Proposed mitigation
```

### Files to Audit

Priority order:
1. `mcts_trader/env/vectorbt_env.py` - Main environment, reward calculation
2. `mcts_trader/jax_agent/jax_trading_env.py` - JAX environment variant
3. `mcts_trader/data/synthetic_market.py` - Synthetic data generators
4. `train_jax_fast.py` - Training loop, any reward shaping
5. `mcts_trader/jax_agent/differential_reward.py` - Differential reward system
6. `mcts_trader/encoder/state.py` - State encoding (lookahead bias?)

### Known Issues to Verify

1. **HOLD dominance** - Is HOLD actually the optimal strategy? Or just easiest to exploit?
2. **Alpha calculation** - Can agent game the market baseline comparison?
3. **Limit order fills** - Currently too generous, needs realistic execution model

---

## LIMIT ORDER EXECUTION IMPROVEMENT

### Current Problem

Limit orders fill instantly if price crosses limit price. This is unrealistic:
- Real markets have depth - orders queue behind others
- Orders near market price may not fill immediately
- Large orders move the market

### Proposed Fix

Implement **execution probability** based on distance from market:

```python
def calculate_fill_probability(limit_price, market_price, side, volatility):
    """
    Orders far from market fill instantly (taking liquidity).
    Orders near market may take multiple steps (providing liquidity).
    """
    distance = abs(limit_price - market_price) / market_price

    # Normalize by recent volatility
    normalized_distance = distance / volatility

    if side == 'buy':
        # Buy limit below market
        if limit_price >= market_price:
            return 1.0  # Marketable, instant fill
        elif normalized_distance > 2.0:
            return 0.95  # Far below, likely fills on any dip
        elif normalized_distance > 1.0:
            return 0.5   # Moderate distance
        else:
            return 0.1   # Close to market, may not fill

    # Similar for sell side...
```

### Implementation Steps

1. Track order age (already have TTL)
2. Each step, calculate fill probability
3. Roll random number, fill if under probability
4. Orders gain "priority" over time (aging bonus)
5. Add spread model for realistic bid-ask

### Benefits

- Prevents instant limit order exploitation
- Teaches agent realistic order placement
- Better sim-to-real transfer
- More interesting limit order strategies

---

## Priority

- **Reward hacking audit**: HIGH - affects training validity
- **Limit order fix**: MEDIUM - improves realism

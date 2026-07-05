# TASK_038: Differential Reward (Opportunity Cost)

## Status: IMPLEMENTED (2026-01-06)

**Implementation:**
- `mcts_trader/jax_agent/differential_reward.py` - Core module
- `train_jax_fast.py` - Added `--differential-reward`, `--differential-baseline`, `--differential-weight` flags
- Note: Integration with environment reward computation pending
## Priority: HIGH (Low effort, High impact)

## Problem Statement

Current reward structure makes HOLD appear "safe":
- `R(HOLD) = 0` (no change)
- `R(BUY/SELL) = ΔP - fees` (variable, often negative due to noise)

Early in training, the agent learns:
- E[R(HOLD)] = 0
- E[R(TRADE)] ≈ -fees (negative due to transaction costs)

This creates a basin of attraction toward HOLD that entropy regularization cannot overcome.

## Research Basis

From Gemini Deep Research:
> "To combat the bias towards HOLD, reshape the reward to be relative to a baseline
> (e.g., Buy-and-Hold or Market Index). r_t = r_agent - r_benchmark. If the market
> goes up and the agent HOLDS, it receives a negative reward (opportunity cost).
> This incentivizes action."

## Solution: Opportunity Cost Reward

Transform reward from absolute to differential:

```python
r_t = r_agent - r_benchmark
```

Where `r_benchmark` can be:
1. **Market return**: Average return of all assets (simplest)
2. **Buy-and-hold**: Return of holding equal-weighted portfolio
3. **Momentum baseline**: Return of simple momentum strategy

### Example

| Scenario | Market | Agent (HOLD) | Current Reward | Differential Reward |
|----------|--------|--------------|----------------|---------------------|
| Bull market | +2% | 0% | 0 | -2% (missed opportunity!) |
| Bear market | -2% | 0% | 0 | +2% (avoided loss!) |
| Flat market | 0% | 0% | 0 | 0 |

**Key insight**: HOLD now has VARIABLE reward that depends on market conditions.

## Implementation

### Phase 1: Simple Market Baseline

**File: `mcts_trader/env/vectorbt_env.py`**

```python
def _compute_reward(self, prev_value, new_value, action):
    """Compute reward with opportunity cost."""
    # Agent's return
    agent_return = (new_value - prev_value) / max(prev_value, 1e-10)

    # Market baseline: average return across all assets
    market_return = self._compute_market_return()

    # Differential reward (opportunity cost)
    if self.use_differential_reward:
        raw_reward = agent_return - market_return
    else:
        raw_reward = agent_return

    # Apply symlog transformation
    return symlog(raw_reward * 100, threshold=1.0)

def _compute_market_return(self):
    """Compute market baseline return."""
    if self.prev_prices is None:
        return 0.0

    # Equal-weighted market return
    price_returns = (self.current_prices - self.prev_prices) / self.prev_prices
    return np.mean(price_returns)
```

### Phase 2: JAX Environment

**File: `mcts_trader/env/jax_trading_env.py`**

```python
def _compute_differential_reward(
    self,
    agent_pnl: jnp.ndarray,
    prev_prices: jnp.ndarray,
    current_prices: jnp.ndarray,
    prev_portfolio_value: jnp.ndarray,
) -> jnp.ndarray:
    """Compute reward relative to market benchmark."""
    # Agent's percentage return
    agent_return = agent_pnl / jnp.maximum(prev_portfolio_value, 1e-10)

    # Market return (equal-weighted average of all assets)
    price_returns = (current_prices - prev_prices) / jnp.maximum(prev_prices, 1e-10)
    market_return = jnp.mean(price_returns, axis=-1)  # (batch,)

    # Differential: agent vs market
    differential = agent_return - market_return

    return differential
```

### Phase 3: CLI Flag

**File: `train_jax_fast.py`**

```python
parser.add_argument('--differential-reward', action='store_true', default=False,
                   help='Use opportunity cost reward (agent return - market return)')
parser.add_argument('--reward-baseline', type=str, default='market',
                   choices=['market', 'hold', 'momentum'],
                   help='Baseline for differential reward')
```

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--differential-reward` | False | Enable opportunity cost reward |
| `--reward-baseline` | 'market' | Baseline type: market, hold, momentum |
| `--baseline-weight` | 1.0 | How much to weight baseline (0=ignore, 1=full) |

## Expected Behavior Changes

### Before (Current)
```
Market: +5%, Agent HOLDs → reward = 0 → HOLD looks safe
Market: -5%, Agent HOLDs → reward = 0 → HOLD looks safe
```

### After (Differential)
```
Market: +5%, Agent HOLDs → reward = -5% → HOLD penalized (missed opportunity)
Market: -5%, Agent HOLDs → reward = +5% → HOLD rewarded (avoided loss)
```

**Key**: HOLD is no longer a "free" safe action. The agent must justify inaction.

## Interaction with Other Components

### PopArt Normalization
- Differential rewards have similar scale to current rewards
- PopArt will adapt to new distribution automatically
- May need to adjust `popart_beta` for faster adaptation

### Value Function
- Value targets will reflect opportunity cost
- V(s) now represents "value relative to market"
- Alpha (agent - market) becomes the primary metric

### MCTS Planning
- Simulated rollouts will use differential rewards
- HOLD branches will have variable values
- Should break the "all actions equal" problem

## Testing Plan

1. **Sanity check**: Run on synthetic level 0 (sine waves)
   - Market return is predictable
   - Agent should learn to trade WITH the market

2. **Comparison**: Run with/without differential reward
   - Track HOLD% over epochs
   - Track alpha (should improve with differential)

3. **Ablation**: Test different baselines
   - Market (equal-weighted average)
   - Momentum (simple trend-following)

## Success Criteria

1. **HOLD% reduction**: HOLD should not exceed 70% without adaptive entropy triggering
2. **Alpha improvement**: Agent should beat market baseline
3. **Action diversity**: More balanced S/H/B distribution

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Over-trading (always chase market) | Keep transaction costs in reward |
| Reward scale change breaks training | Use symlog + PopArt normalization |
| Baseline noise adds variance | Use smoothed baseline (EMA) |

## Files to Modify

1. **`mcts_trader/env/vectorbt_env.py`** - Add differential reward computation
2. **`mcts_trader/env/jax_trading_env.py`** - JAX version of differential reward
3. **`train_jax_fast.py`** - Add CLI flags
4. **`CLAUDE.md`** - Document new reward option

## Dependencies

- None (standalone feature)

## Follow-up Tasks

- TASK_039 (Dirichlet Noise) - Complementary exploration boost
- TASK_040 (Adaptive Entropy) - Complementary entropy control

## References

- Gemini Deep Research: "Differential Sharpe Ratio rewards"
- Moody & Saffell (2001): "Differential Sharpe Ratio for RL trading"
- Thesis: "Reward shaping for financial RL agents"

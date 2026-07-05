# TASK_051: Burning Money (HOLD Cost Mechanism)

## Status: IMPLEMENTED (2026-01-18)

## Problem

HOLD action has zero cost, making it the "safe" choice:
- R(HOLD) = 0 always
- R(TRADE) ≈ -fees on average
- Risk-averse policy converges to HOLD-only

This causes the observed **policy collapse** where agent learns to do nothing.

## Key Insight

**Positions vs Cash are different:**
- Holding positions → exposed to market, can generate capital
- Holding cash → no exposure, pure opportunity cost

**Burn should apply to CASH only, not positions.**

## Concept: Burning Money

Force the agent to deploy capital. Ideas:

### Approach 1: Cash Decay (Inflation)

```python
# Each step, cash loses value (inflation/opportunity cost)
cash_decay_rate = 0.0001  # 0.01% per step = ~4% per episode

new_cash = state.cash * (1 - cash_decay_rate)
```

**Intuition:** Money in wallet loses purchasing power. Must invest to maintain value.

### Approach 2: Target Return Requirement

```python
# Agent must achieve minimum return by end of episode
target_return = 0.02  # 2% target

# Shortfall penalty at episode end
shortfall = max(0, target_return - actual_return)
terminal_penalty = -10 * shortfall  # Strong penalty for missing target
```

**Intuition:** Fund manager must beat benchmark or lose job.

### Approach 3: Negative Starting Balance

```python
# Start with debt that must be paid off through trading
initial_debt = 0.05 * initial_portfolio  # 5% debt

# Final reward includes debt payment
final_reward = portfolio_return - remaining_debt_penalty
```

**Intuition:** Borrowed money to start, must pay back with interest.

### Approach 4: Holding Cost (Carry)

```python
# Cost for holding cash (negative interest)
# OR cost for holding positions (borrowing cost)
cash_carry = -0.0001 * state.cash  # Negative interest on cash
position_carry = -0.00005 * abs(state.positions)  # Slight position cost

step_cost = cash_carry + position_carry
reward += step_cost
```

**Intuition:** Everything costs something to hold.

### Approach 5: Opportunity Cost Feedback

```python
# At end of episode, show agent what optimal passive strategy earned
market_return = (final_market_price - initial_price) / initial_price
hold_return = market_return  # What HOLD would have earned

# Penalty = missed opportunity
opportunity_cost = max(0, hold_return - agent_return)
terminal_reward -= opportunity_cost_weight * opportunity_cost
```

**Intuition:** "You could have just held the market and earned X%"

### Approach 6: Market-Linked Cash Burn (Recommended)

```python
# Cash burns at the rate the market moves
# If market up 0.1%, your cash lost 0.1% opportunity
# If market down 0.1%, your cash SAVED 0.1% (reward!)

market_return_step = (current_price - prev_price) / prev_price
cash_ratio = state.cash / state.portfolio_value

# Burn = market return * cash fraction
# Positive market = burn (missed opportunity)
# Negative market = reward (avoided loss)
cash_burn = market_return_step * cash_ratio

reward -= cash_burn  # Subtract opportunity cost
```

**Intuition:** Cash is a position - short the market. Works correctly both ways:
- Bull market: Cash burns (you missed gains)
- Bear market: Cash earns (you avoided losses)

**This is differential reward applied specifically to uninvested capital.**

### Approach 7: Random Burn (Anti-Gaming)

```python
# Unpredictable burn prevents agent from timing the decay
base_burn_rate = 0.0001

# Option A: Random multiplier
burn_multiplier = np.random.uniform(0.5, 2.0)  # 0.5x to 2x base rate
actual_burn = base_burn_rate * burn_multiplier

# Option B: Random burn events (like surprise taxes)
if np.random.random() < 0.05:  # 5% chance per step
    surprise_burn = np.random.uniform(0.001, 0.005)  # 0.1-0.5% hit
    reward -= surprise_burn * state.cash

# Option C: Aggressive random (high variance)
burn = np.random.exponential(base_burn_rate)  # Sometimes 0, sometimes 5x
```

**Intuition:** Real markets have unexpected costs (taxes, fees, spread widening). Agent must be robust.

**Aggressive random burn forces:**
- Can't optimize for predictable decay
- Must maintain buffer/margin
- Learns to handle uncertainty

## Recommended Implementation

Combine **Market-Linked Cash Burn** + **Random Component**:

```python
@dataclass
class BurningMoneyConfig:
    # Market-linked burn (opportunity cost on cash)
    market_linked: bool = True           # Burn = market return * cash ratio

    # Base decay (constant drain)
    base_decay_rate: float = 0.00005     # 0.005% per step baseline

    # Random component (anti-gaming)
    random_burn: bool = True
    random_burn_prob: float = 0.03       # 3% chance of surprise cost
    random_burn_min: float = 0.0005      # 0.05% minimum surprise
    random_burn_max: float = 0.002       # 0.2% maximum surprise


def apply_burning_money(state, prev_state, market_prices, prev_prices, config, rng):
    """Apply burning money mechanics to CASH only."""

    cash_ratio = state.cash / (state.portfolio_value + 1e-8)

    burn = 0.0

    # 1. Market-linked burn (opportunity cost)
    if config.market_linked:
        # Average market return this step
        market_return = jnp.mean((market_prices - prev_prices) / prev_prices)

        # Cash "missed" this return (or avoided loss if negative!)
        # Positive market = positive burn (penalty)
        # Negative market = negative burn (reward for holding cash)
        market_burn = market_return * cash_ratio
        burn += market_burn

    # 2. Base decay (constant pressure)
    base_burn = config.base_decay_rate * cash_ratio
    burn += base_burn

    # 3. Random burn (surprise costs)
    if config.random_burn:
        if rng.random() < config.random_burn_prob:
            surprise = rng.uniform(config.random_burn_min, config.random_burn_max)
            burn += surprise * cash_ratio

    return burn  # Subtract from reward


# In step function:
reward = base_reward - apply_burning_money(state, prev_state, prices, prev_prices, config, rng)
```

### Behavior Matrix

| Market | Cash Held | Market Burn | Base Burn | Net Effect |
|--------|-----------|-------------|-----------|------------|
| +1% | 50% | +0.5% penalty | +0.025% | -0.525% (missed rally) |
| -1% | 50% | -0.5% reward | +0.025% | +0.475% (avoided crash) |
| +1% | 10% | +0.1% penalty | +0.005% | -0.105% (mostly invested) |
| 0% | 90% | 0% | +0.045% | -0.045% (constant pressure) |

**Key insight:** Market-linked burn rewards holding cash in down markets - this is CORRECT behavior!

## CLI Arguments

```python
parser.add_argument('--burning-money', action='store_true',
                   help='Enable burning money mechanics (cash opportunity cost)')
parser.add_argument('--burn-market-linked', action='store_true', default=True,
                   help='Cash burns at market return rate')
parser.add_argument('--burn-base-rate', type=float, default=0.00005,
                   help='Base per-step cash decay rate')
parser.add_argument('--burn-random', action='store_true', default=True,
                   help='Add random burn events')
parser.add_argument('--burn-random-prob', type=float, default=0.03,
                   help='Probability of random burn event per step')
parser.add_argument('--burn-random-max', type=float, default=0.002,
                   help='Maximum random burn amount')
```

## Integration Points

### jax_trading_env.py

```python
def step(self, state, action, config):
    # ... existing step logic ...

    # Apply burning money
    if config.burning_money:
        step_adj, term_adj = apply_burning_money(state, config, done)
        reward += step_adj
        if done:
            reward += term_adj

    return new_state, reward, done
```

### train_jax_fast.py

```python
# Add to argument parsing
args.burning_money = True  # Could be default

# Pass to env config
env_config = EnvConfig(
    burning_money=args.burning_money,
    cash_decay_rate=args.cash_decay_rate,
    # ...
)
```

## Expected Behavior

| Scenario | Old Reward | New Reward |
|----------|------------|------------|
| HOLD 400 steps, 0% return | 0 | -4% (cash decay) - opportunity |
| Active trading, +2% return | +2% | +2% - small decay |
| HOLD in bull market (+5%) | 0 | -4% - 2.5% opportunity = -6.5% |
| Beat market by 3% | +3% | +3% - small decay ≈ +2.5% |

## Interaction with Differential Reward

Burning money is **complementary** to differential reward:
- Differential reward: Makes HOLD equivalent to market return (not zero)
- Burning money: Adds explicit cost to inactivity

Can use both together for strongest HOLD deterrent.

## Testing

```python
def test_cash_decays():
    """Cash should decrease each step."""
    env = TradingEnv(burning_money=True, cash_decay_rate=0.001)
    state = env.reset()
    initial_cash = state.cash

    # HOLD for 100 steps
    for _ in range(100):
        state, reward, _ = env.step(HOLD_ACTION)

    # Cash should have decayed ~10%
    assert state.cash < initial_cash * 0.95

def test_inactivity_penalty():
    """Should penalize episodes with few trades."""
    env = TradingEnv(burning_money=True, min_trades=10)

    # Episode with only 2 trades
    rewards = run_episode(env, n_trades=2)
    penalty_episode = sum(rewards)

    # Episode with 20 trades
    rewards = run_episode(env, n_trades=20)
    active_episode = sum(rewards)

    assert penalty_episode < active_episode
```

## Priority

**HIGH** - Direct fix for HOLD collapse problem. Should be tested alongside TASK_044 (epsilon_030 experiment).

## Risks

- Too aggressive decay → agent learns frantic trading (churn)
- Need to tune decay rate to be meaningful but not overwhelming
- Start conservative (0.0001) and increase if needed

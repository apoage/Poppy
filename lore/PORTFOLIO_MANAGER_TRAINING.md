# Portfolio Manager Training - The Full Curriculum

## Core Philosophy

This is NOT a trading agent. This is a **PORTFOLIO MANAGER**.

Goals:
- **Manage 64 assets simultaneously** with complex correlation awareness
- **Trade cross-pairs directly** (BTC/ETH, not BTC→USD→ETH)
- **Develop inhuman tactics** (like AlphaGo's move 37)
- **Face adversarial opponents** (other models trying to break it)
- **Break fast, document, fix, iterate**

---

## Action Space Redesign

### Current Problem
- 3 actions per asset = Sell/Hold/Buy to USD
- Forces USD as intermediary (inefficient)
- Can only act once per minute (wasteful)

### New Action Space

**Portfolio Actions** (per step):
```python
# Option 1: Cross-Pair Trading Matrix
action_space = {
    'source_asset': Discrete(64),    # Which asset to sell
    'target_asset': Discrete(64),    # Which asset to buy
    'size': Box(0, 1),               # Position size
    'urgency': Discrete(4)           # 0=limit, 1=market, 2=aggressive, 3=maniac
}
# Total: 64 × 64 × continuous × 4 = ~16K discrete combinations + continuous
```

**Option 2: Multi-Action Per Step**
```python
# Model outputs N actions per step (where N=1 to 10)
# Enables rapid rebalancing ("maniac cross actions")

action_points = Discrete(10)  # How many actions to take this step
for i in range(action_points):
    execute_pair_trade(source=..., target=..., size=...)
```

**Option 3: Portfolio Target Vector**
```python
# Model outputs desired portfolio state
target_weights = Box(0, 1, shape=(64,))  # Sum to 1.0

# Environment calculates minimum trades to reach target
# Executes cross-pair trades optimally
```

**Recommendation**: Start with Option 1 (explicit cross-pair), evolve to Option 3 (target portfolio).

---

## Extended Curriculum (Levels 0-6)

### Level 0: Trend Surfer (UNCHANGED)
**Test**: Basic buy low / sell high
**Pass**: Avg reward > +5%

### Level 1: Lag Hunter (UNCHANGED)
**Test**: Correlation matrix encoder
**Pass**: Avg reward > +4%

### Level 2: Dip Catcher (UNCHANGED)
**Test**: Patience + conditional trading
**Pass**: Sharpe > 1.5

### Level 3: Spread Beater (UNCHANGED)
**Test**: Fee awareness
**Pass**: Avg reward > +1%, trades < 20%

### Level 4: Portfolio Rebalancer (NEW)
**Objective**: Learn portfolio tactics, not single-asset trading

**Data Characteristics**:
- 64 assets with random correlations (some positive, some negative)
- Assets drift independently (Geometric Brownian Motion)
- No dominant asset - opportunity rotates every 50 steps

**Scenario**:
```
Step 0-50:   BTC pumps +20%, ETH flat
Step 50-100: BTC flat, SOL pumps +15%
Step 100-150: All crash -10%, stablecoins safe
```

**Expected Behavior**:
- Agent must **ROTATE** capital between assets
- Learn to **EXIT winners** and **ENTER new opportunities**
- Use correlations to hedge (long BTC, short ETH if they're correlated but BTC stronger)

**Tactics to Learn**:
1. **Mean Reversion**: If asset diverges from basket, it will revert
2. **Momentum**: Assets in motion stay in motion (surf trends)
3. **Risk Parity**: Distribute capital by volatility, not price
4. **Hedging**: Long high-beta, short low-beta if market crashes

**Pass Criteria**:
- Sharpe ratio > 2.0 (better than single-asset focus)
- Portfolio turnover > 50% (actively rebalancing)
- Correlation with top asset < 0.8 (not just riding one winner)

---

### Level 5: Adversarial Market (NEW)
**Objective**: Face an opponent trying to break you

**Adversary**: Another RL agent (or scripted) that:
- Sees your portfolio
- Tries to maximize YOUR losses
- Controls market prices (within bounds)

**Scenario**:
```python
class AdversarialMarket:
    def __init__(self, protagonist_agent, antagonist_agent):
        self.protagonist = protagonist_agent
        self.antagonist = antagonist_agent

    def step(self):
        # 1. Protagonist chooses action
        action = self.protagonist.select_action(state)

        # 2. Antagonist sees action, chooses price manipulation
        # (Can shift prices ±2% per step, but limited budget)
        price_manipulation = self.antagonist.select_manipulation(
            protagonist_portfolio=self.protagonist.portfolio,
            protagonist_action=action
        )

        # 3. Apply manipulation, execute protagonist action
        new_prices = base_prices * (1 + price_manipulation)
        reward = execute_trade(action, new_prices)

        return reward
```

**Adversary Strategy Examples**:
- **Front-running**: See agent wants to buy BTC, pump price before execution
- **Stop-loss hunting**: Crash asset to trigger agent's stop-loss, then pump
- **Correlation break**: If agent relies on BTC/ETH correlation, break it

**Expected Behavior**:
- Agent learns to **hide intent** (don't telegraph large trades)
- Use **limit orders** instead of market orders (adversary can't front-run limits)
- Develop **robustness** to price manipulation

**Pass Criteria**:
- Positive reward against adversary for 100 consecutive episodes
- Sharpe ratio > 1.0 even under attack

**Implementation**:
```python
# Train TWO models simultaneously
protagonist = MuZeroNetworks(...)
antagonist = MuZeroNetworks(...)

# Self-play: They train against each other
for epoch in range(epochs):
    # Protagonist's turn
    protagonist_reward = play_episode(protagonist, antagonist)

    # Train protagonist to maximize reward
    train_protagonist(protagonist_reward)

    # Antagonist's turn (maximize protagonist's loss)
    train_antagonist(-protagonist_reward)
```

**This is like AlphaGo self-play** - both models improve by exploiting each other's weaknesses.

---

### Level 6: Chaos Mode (NEW)
**Objective**: Survive the impossible

**Data Characteristics**:
- **Flash crashes**: -20% in 1 step, random asset
- **Correlation flips**: Assets that were correlated suddenly anti-correlate
- **Liquidity gaps**: Some assets become untradeable (action mask changes randomly)
- **Fee spikes**: Trading fees randomly 2x-10x
- **Black swans**: Once per episode, a random asset goes to 0 (delisting)

**Expected Behavior**:
- **Diversification**: Never go all-in on one asset (if it delists, you die)
- **Liquidity buffers**: Keep cash reserves for emergencies
- **Dynamic hedging**: If correlations flip, rebalance immediately
- **Tail risk awareness**: Don't assume Gaussian distributions

**Pass Criteria**:
- Survive 100 episodes without bankruptcy
- Avg reward > 0% (even staying flat is a win in chaos)
- Max drawdown < 30%

**Why This Level?**
Real markets ARE chaotic. If the model can't handle Level 6, it can't handle 2022 crypto crash or COVID-2020.

---

## Cross-Pair Trading Implementation

### Current Flow (Inefficient)
```
Agent wants to trade BTC for ETH:
1. Sell BTC → USD (pay 26 bps fee)
2. Buy ETH with USD (pay 26 bps fee)

Total cost: 52 bps + 2× slippage
```

### New Flow (Direct)
```
Agent trades BTC/ETH directly:
1. Sell BTC, buy ETH (single transaction)

Total cost: 26 bps + 1× slippage
```

### Implementation

**PortfolioManager Enhancement**:
```python
def execute_cross_pair_trade(self, source_idx, target_idx, size, current_prices):
    """
    Trade directly from source asset to target asset.
    No USD intermediary.
    """
    source_symbol = self.symbol_map[source_idx]
    target_symbol = self.symbol_map[target_idx]

    # Get cross-rate (e.g., BTC/ETH = BTC/USD ÷ ETH/USD)
    source_price_usd = current_prices[source_idx]
    target_price_usd = current_prices[target_idx]
    cross_rate = source_price_usd / target_price_usd

    # Execute
    source_amount = self.positions[source_idx].size * size
    target_amount = source_amount * cross_rate

    # Apply fee ONCE (not twice)
    fee = target_amount * (self.trading_fee_bps / 10000)
    target_amount -= fee

    # Update positions
    self.positions[source_idx].size -= source_amount
    self.positions[target_idx].size += target_amount

    logging.info(f"Cross-pair: {source_symbol} → {target_symbol} "
                 f"({source_amount:.4f} → {target_amount:.4f})")
```

**Environment Modification**:
```python
def step(self, action_dict):
    source_idx = action_dict['source_asset']
    target_idx = action_dict['target_asset']
    size = action_dict['size']
    urgency = action_dict['urgency']

    current_prices = self.market.get_current_prices()

    if urgency == 0:  # Limit order
        # Place cross-pair limit order
        target_rate = current_prices[source_idx] / current_prices[target_idx]
        self.portfolio.place_cross_limit_order(source_idx, target_idx, target_rate, size)

    elif urgency == 1:  # Market order
        self.portfolio.execute_cross_pair_trade(source_idx, target_idx, size, current_prices)

    elif urgency == 2:  # Aggressive (market + extra slippage tolerance)
        # Accept 2× slippage for faster execution
        self.portfolio.execute_cross_pair_trade(source_idx, target_idx, size, current_prices, slippage_mult=2.0)

    elif urgency == 3:  # Maniac (execute at ANY cost)
        # 10× slippage tolerance - use only for emergencies
        self.portfolio.execute_cross_pair_trade(source_idx, target_idx, size, current_prices, slippage_mult=10.0)

    # ... rest of step logic ...
```

---

## Multi-Action Per Step

### Problem
Model sees opportunity that requires 3 trades:
1. Sell BTC → ETH
2. Sell SOL → BTC
3. Buy MATIC with freed USD

Currently: Takes 3 steps (3 minutes). Opportunity might vanish.

### Solution: Action Points

**Model outputs**: "I want to take 4 actions this step"

```python
class FastTradingEnv:
    def step(self, action_plan):
        """
        action_plan = {
            'n_actions': 4,
            'actions': [
                {'source': 0, 'target': 10, 'size': 0.5, 'urgency': 1},
                {'source': 5, 'target': 0, 'size': 0.3, 'urgency': 1},
                {'source': 63, 'target': 20, 'size': 0.2, 'urgency': 0},
                {'source': 10, 'target': 15, 'size': 0.1, 'urgency': 2}
            ]
        }
        """

        # Execute all actions in sequence (atomic transaction)
        for action in action_plan['actions']:
            self.portfolio.execute_cross_pair_trade(...)

        # Market advances ONCE (not 4 times)
        self.market.step()

        # Reward is calculated AFTER all actions
        reward = self.portfolio.get_total_pnl()
```

**Model Architecture**:
```python
class MuZeroNetworks(nn.Module):
    def __call__(self, x, mode='initial_inference'):
        # ... existing ...

        # Policy head now outputs:
        # 1. Number of actions (1-10)
        n_actions_logits = nn.Dense(10)(hidden)
        n_actions = jnp.argmax(n_actions_logits) + 1

        # 2. For each action slot, output (source, target, size, urgency)
        action_outputs = []
        for i in range(10):  # Always output 10, mask unused
            source_logits = nn.Dense(64)(hidden)
            target_logits = nn.Dense(64)(hidden)
            size = nn.Dense(1, activation='sigmoid')(hidden)
            urgency = nn.Dense(4)(hidden)

            action_outputs.append({
                'source_logits': source_logits,
                'target_logits': target_logits,
                'size': size,
                'urgency_logits': urgency
            })

        return {
            'n_actions': n_actions,
            'action_outputs': action_outputs,
            'value': value,
            ...
        }
```

**Action Mask**:
```python
# Mask impossible trades
# E.g., can't sell asset you don't own (in SPOT mode)
# Can't trade to same asset (source == target)

for i in range(10):
    for source in range(64):
        mask[i, source, source] = 0  # Can't trade BTC → BTC

        if mode == 'SPOT' and portfolio.positions[source].size == 0:
            mask[i, source, :] = 0  # Can't sell what you don't have
```

---

## Adversarial Training System

### Architecture

**Two Models**:
1. **Protagonist** (Portfolio Manager) - The model we want to train
2. **Antagonist** (Market Manipulator) - Tries to break protagonist

**Training Loop**:
```python
for epoch in range(epochs):
    # 1. Protagonist acts in environment controlled by antagonist
    trajectory_protag = []
    state = env.reset()

    for step in range(200):
        # Protagonist chooses portfolio action
        protag_action = protagonist.select_action(state)

        # Antagonist sees action, chooses price manipulation
        # (Limited budget: can't manipulate too much)
        antag_action = antagonist.select_manipulation(
            state=state,
            protag_action=protag_action,
            protag_portfolio=env.portfolio
        )

        # Environment applies manipulation
        env.apply_price_manipulation(antag_action)

        # Execute protagonist action, get reward
        next_state, reward, done = env.step(protag_action)

        trajectory_protag.append((state, protag_action, reward))
        state = next_state

        if done:
            break

    # 2. Train protagonist (maximize cumulative reward)
    train_protagonist(trajectory_protag)

    # 3. Train antagonist (maximize protagonist's loss)
    # Antagonist's reward = -protagonist's reward
    trajectory_antag = [(s, a, -r) for s, a, r in trajectory_protag]
    train_antagonist(trajectory_antag)

    # 4. Log
    print(f"Epoch {epoch}: Protag Reward={sum(r for _, _, r in trajectory_protag):.2f}")
```

### Antagonist Action Space

```python
class Antagonist:
    """
    Controls price movements within constraints.
    Goal: Maximize protagonist's losses.
    """

    def select_manipulation(self, state, protag_action, protag_portfolio):
        """
        Returns: price_shifts (64,) array of % changes
        Constraint: sum(abs(price_shifts)) < BUDGET (e.g., 10%)
        """

        # Strategy examples:
        # 1. Front-run: If protag wants to buy BTC, pump BTC first
        if protag_action['target_asset'] == BTC_IDX:
            price_shifts[BTC_IDX] = +0.02  # 2% pump

        # 2. Stop-loss hunt: If protag has stop-loss, trigger it
        for asset_idx, position in protag_portfolio.positions.items():
            if position.stop_loss > 0:
                # Crash price to trigger stop
                price_shifts[asset_idx] = -0.05

        # 3. Break correlations: If protag relies on BTC/ETH correlation
        if correlation_detected(BTC_IDX, ETH_IDX):
            price_shifts[BTC_IDX] = +0.03
            price_shifts[ETH_IDX] = -0.03  # Break the correlation

        return price_shifts
```

### Why This Works

**Adversarial training** forces the protagonist to:
- Not rely on fragile patterns (antagonist will break them)
- Develop robust strategies that work even when attacked
- Learn to hide intent (don't telegraph trades)

**Like AlphaGo**:
- AlphaGo didn't learn from human games alone
- It played millions of games against ITSELF
- Each version exploited the weaknesses of the previous version
- Result: Inhuman strategies (move 37 in game 2 vs Lee Sedol)

**Result for Portfolio Manager**:
- Develops tactics humans wouldn't think of
- Survives adversarial markets (flash crashes, manipulation)
- Generalizes better to real markets (which ARE adversarial)

---

## Implementation Strategy

### Step 1: Validate Current System on Level 0
**Goal**: Ensure existing architecture works before adding complexity

```bash
python train_jax_fast.py --synthetic-level 0 --epochs 1000
```

**Expected**: Passes in < 500 epochs. If not, architecture is broken - STOP and debug.

**What to check if it fails**:
- Observation encoder: Is price data reaching the model?
- Action execution: Are trades happening?
- Reward calculation: Is PnL computed correctly?
- MCTS search: Is it exploring action space?

---

### Step 2: Add Cross-Pair Trading
**Goal**: Enable direct asset-to-asset trades

**Changes**:
1. Modify `PortfolioManager.execute_cross_pair_trade()`
2. Update action space to `(source, target, size, urgency)`
3. Test on Level 0

**Validation**:
```python
# Manual test
env = FastTradingEnv(...)
env.reset()

# Force cross-pair trade
action = {'source': 0, 'target': 10, 'size': 0.5, 'urgency': 1}
env.step(action)

# Check: Did position 0 decrease and position 10 increase?
assert env.portfolio.positions[0].size < initial_size
assert env.portfolio.positions[10].size > 0
```

**Expected Result**: Level 0 still passes (should be faster with lower fees).

---

### Step 3: Add Multi-Action Per Step
**Goal**: Allow N actions per step

**Changes**:
1. Model outputs `n_actions` + list of actions
2. Environment executes all actions before market step
3. Test on Level 1 (lag detection benefits from fast rebalancing)

**Validation**:
Run Level 1 with:
- Max 1 action per step (baseline)
- Max 5 actions per step (new)

**Expected**: 5-action version passes faster (can rebalance entire portfolio in one step).

---

### Step 4: Implement Level 4 (Portfolio Rebalancer)
**Goal**: Force model to learn portfolio tactics

**Data Generator**:
```python
def _level_4_portfolio_tactics(self):
    """
    64 assets, random correlations, opportunity rotates.
    """
    data = {}
    WINDOW = 50  # Opportunity window

    # Generate correlation matrix (some positive, some negative)
    correlation_matrix = generate_correlation_matrix(64)

    # Assets drift with correlation
    assets = generate_correlated_brownian_motion(64, correlation_matrix, n_steps=10000)

    # Add rotating momentum
    for window_idx in range(n_steps // WINDOW):
        start = window_idx * WINDOW
        end = start + WINDOW

        # Pick 5 random assets to pump this window
        winners = np.random.choice(64, size=5, replace=False)

        for winner in winners:
            assets[winner, start:end] *= np.linspace(1.0, 1.2, WINDOW)  # 20% pump

    # Convert to OHLC
    for i in range(64):
        data[f'SYN_{i:02d}'] = self._prices_to_ohlc(assets[i, :])

    return data
```

**Train**:
```bash
python train_jax_fast.py --synthetic-level 4 --epochs 5000
```

**Monitor**:
- Portfolio turnover (should be > 50%)
- Correlation with top asset (should be < 0.8)
- Sharpe ratio (should be > 2.0)

**What to watch for**:
- Model might "turtle" (100% hold) - if so, temporarily disable fees
- Model might chase winners (FOMO) - check if it's exiting winners early enough
- Model might ignore correlations - visualize attention maps

---

### Step 5: Implement Level 5 (Adversarial)
**Goal**: Train against opponent

**Implementation**:
```python
# tools/train_adversarial.py

protagonist = MuZeroNetworks(...)
antagonist = MuZeroNetworks(...)

for epoch in range(epochs):
    # Self-play episode
    protag_reward = play_adversarial_episode(protagonist, antagonist)

    # Update both models
    update_protagonist(protagonist, target=maximize_reward)
    update_antagonist(antagonist, target=minimize_protag_reward)

    # Every 100 epochs, freeze antagonist, train protagonist for 100 epochs
    # This prevents antagonist from becoming TOO strong
    if epoch % 100 == 0:
        freeze(antagonist)
```

**Validation**:
```bash
# Train for 10k epochs
python tools/train_adversarial.py --epochs 10000

# Test protagonist against random antagonist
python tools/test_robustness.py --protagonist checkpoint.pkl
```

**Expected**:
- Protagonist learns robust strategies
- Stops using predictable patterns (antagonist exploits them)

**Document where model breaks**:
- What tactics does antagonist discover?
- How does protagonist counter?
- Are there unbeatable antagonist strategies? (If yes, that's a weakness to fix)

---

### Step 6: Implement Level 6 (Chaos)
**Goal**: Test robustness to black swans

**Data Generator**:
```python
def _level_6_chaos_mode(self):
    # Start with Level 4 data (portfolio tactics)
    data = self._level_4_portfolio_tactics()

    # Add chaos events
    for i in range(64):
        prices = data[f'SYN_{i:02d}']['close']

        # Flash crash (5 times per episode)
        crash_steps = np.random.choice(n_steps, size=5, replace=False)
        for step in crash_steps:
            prices[step:step+1] *= 0.8  # -20% in 1 step

        # Delisting (1 asset goes to 0)
        if i == np.random.randint(64):
            delist_step = np.random.randint(n_steps)
            prices[delist_step:] = 0.0

    return data
```

**Train**:
```bash
python train_jax_fast.py --synthetic-level 6 --epochs 10000
```

**Monitor**:
- Bankruptcy rate (should be 0%)
- Max drawdown (should be < 30%)
- Diversification (should hold 10+ assets at all times)

**Document failures**:
- Which chaos events cause bankruptcy?
- Does model learn to keep cash reserves?
- Does it diversify properly?

---

## Validation Loop

**After each level**:
```python
# Save checkpoint
save_checkpoint(f'level_{level}_complete.pkl')

# Run diagnostic tests
diagnostics = {
    'level': level,
    'avg_reward': ...,
    'sharpe_ratio': ...,
    'max_drawdown': ...,
    'trade_frequency': ...,
    'portfolio_turnover': ...,
    'action_distribution': ...,
    'pending_orders_count': ...,
    'cross_pair_trades_pct': ...
}

# Log to JSON
with open(f'diagnostics_level_{level}.json', 'w') as f:
    json.dump(diagnostics, f, indent=2)

# Generate visualization
plot_episode(env, model, save_path=f'level_{level}_episode.png')
```

**Test on next level**:
```bash
python train_jax_fast.py --synthetic-level {level+1} --load-checkpoint level_{level}_complete.pkl --epochs 5000
```

**If it breaks**: Document WHY, fix, retry.

---

## Expected Breakthroughs

### Inhuman Tactics (Like AlphaGo Move 37)

**Examples we might see**:
1. **Negative correlation exploit**: Model discovers that Asset A and B are anti-correlated in chaos mode. It goes 100% long A, 100% short B (net 0 capital, pure volatility harvesting).

2. **Flash crash arbitrage**: Model learns to detect flash crash microseconds before it happens (from order book patterns we didn't know existed). Places limit orders at crash level, gets filled, sells at recovery.

3. **Correlation cascade**: Model discovers that if it trades Asset A → B, this will trigger a correlation cascade (B → C → D) due to how the market is structured. Uses this to amplify returns.

4. **Adversary baiting**: Against antagonist, model intentionally shows a "weak" portfolio to bait antagonist into wasting manipulation budget. Then executes real strategy.

**How to recognize these**:
- Actions that make no sense at first glance
- Reward spikes in unexpected scenarios
- Portfolio states we never saw in training

**What to do**:
- Document it
- Validate it's not a bug (run 1000 episodes)
- Study WHY it works
- Generalize the pattern

---

## Success Metrics

| Level | Core Test | Pass Criteria |
|-------|-----------|---------------|
| 0 | Execution | Reward > +5% |
| 1 | Correlation | Reward > +4% |
| 2 | Patience | Sharpe > 1.5 |
| 3 | Fees | Reward > +1%, trades < 20% |
| 4 | Portfolio Tactics | Sharpe > 2.0, turnover > 50% |
| 5 | Adversarial | Positive reward vs opponent |
| 6 | Chaos | Survive 100 episodes, drawdown < 30% |

**Final Test**: Load Level 6 checkpoint, run on REAL market data (2024-2025).

**Target**: Sharpe > 1.5 on real data. If achieved, deploy to paper trading.

---

## Next Action

**Immediate**:
1. Create `mcts_trader/data/synthetic_market.py`
2. Implement Level 0 generator (sine waves)
3. Run: `python train_jax_fast.py --synthetic-level 0 --epochs 1000`
4. Document: Does it pass? If not, WHY?

**Then**:
- Implement cross-pair trading
- Test on Level 1
- Document breaks
- Fix
- Iterate

**Speed > Perfection**. Break things, learn, fix.

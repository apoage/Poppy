# Advanced Order Types Design

## Overview

Currently, the agent has **3 actions per asset**:
- **Action 0**: Sell (or Short in MARGIN mode)
- **Action 1**: Hold
- **Action 2**: Buy (or Long in MARGIN mode)

**Total action space**: 64 assets × 3 = **192 actions**

This design adds **3 new order types**:
- **Limit Order**: Buy/Sell when price reaches target
- **Stop Order**: Buy/Sell when price crosses trigger (momentum trading)
- **Stop-Limit**: Combination (trigger + limit)

**New action space**: 64 assets × 6 = **384 actions**

---

## Why Advanced Orders?

### 1. Market Efficiency
Real traders don't sit and watch charts 24/7. They place orders and let the market come to them.

**Example**:
- Current: Agent must predict "In 10 minutes, BTC will hit $50k" and hold position
- With Limits: Agent places "Buy BTC @ $50k" and moves on

### 2. MCTS Planning Horizon
MCTS looks ahead 5-10 steps (~5-10 minutes). But market opportunities often develop over hours.

**Problem**: Agent can see "BTC will pump in 2 hours" but can't act (outside planning horizon).

**Solution**: Place a Stop-Buy order at breakout level, let it trigger automatically.

### 3. Risk Management
Stop losses are essential for preventing catastrophic losses.

**Current**: Agent must manually predict "BTC will crash" every step.

**With Stops**: Set stop-loss once, forget it.

---

## Action Space Design

### Option A: Unified Action Space (Recommended)

**Actions** (6 per asset):
```
0: SELL (Market)
1: HOLD
2: BUY (Market)
3: LIMIT_BUY
4: LIMIT_SELL
5: STOP (Market breakout/breakdown)
```

**Total**: 64 × 6 = **384 actions**

**Price Target**: Model outputs a **separate continuous value** for target price (as % offset from current median price).

**Architecture**:
```python
class MuZeroNetworks:
    def policy_head(self, x):
        # Discrete action logits (384)
        action_logits = Dense(384)(x)

        # Continuous price offset (1 value)
        # Range: [-20%, +20%] from current price
        price_offset = Dense(1, activation='tanh')(x) * 0.20

        return action_logits, price_offset
```

**Execution**:
```python
def step(self, action_idx, price_offset):
    asset_idx = action_idx // 6
    action_type = action_idx % 6

    current_price = self.get_price(asset_idx)
    target_price = current_price * (1 + price_offset)

    if action_type == 3:  # LIMIT_BUY
        self.portfolio.place_limit_order(
            asset_idx, 'buy', target_price
        )
    # ... etc
```

### Option B: Separate Action Dimensions (Alternative)

**Two-stage selection**:
1. Choose asset (64 discrete)
2. Choose order type + price (6 discrete + 1 continuous)

**Pros**: Smaller action space (64 + 6 = 70)
**Cons**: Requires two forward passes (slower MCTS)

**Verdict**: Use Option A (unified) for simplicity with MCTS.

---

## Price Target Strategy

### Problem
How does the model choose target prices for limit/stop orders?

### Solution 1: Percentage Offset (Recommended)

**Model Output**: Single scalar ∈ [-1, 1]

**Interpretation**:
```python
offset_percent = model_output * 20  # Range: [-20%, +20%]
target_price = current_price * (1 + offset_percent/100)
```

**Example**:
- Current BTC price: $50,000
- Model outputs: 0.5 → 10% offset
- Target: $55,000 (limit buy)

**Benefits**:
- Simple to learn (single neuron)
- Generalizes across assets (10% is meaningful for BTC and ETH)
- Bounded range prevents absurd targets

**Training**:
- Add price offset to trajectory storage
- Loss: MSE between predicted offset and optimal offset (calculated in hindsight)

### Solution 2: Quantized Levels (Alternative)

**Levels**:
```python
PRICE_LEVELS = [-10%, -5%, -2%, +2%, +5%, +10%, +20%]
```

**Model Output**: 7-way softmax

**Pros**: Discrete, easier for RL
**Cons**: Less granular, doesn't generalize well

**Verdict**: Use Solution 1 (continuous offset).

---

## Order Book Implementation

### PortfolioManager Enhancement

Add `PendingOrder` dataclass:

```python
@dataclass
class PendingOrder:
    order_id: str
    asset_idx: int
    symbol: str
    order_type: str  # 'limit_buy', 'limit_sell', 'stop_buy', 'stop_sell'
    target_price: float
    size: float  # Position size
    created_step: int
    expires_step: int  # Auto-cancel after N steps

class PortfolioManager:
    def __init__(self, ...):
        # ... existing ...
        self.pending_orders: Dict[str, PendingOrder] = {}
        self.next_order_id = 0

    def place_limit_order(self, asset_idx: int, side: str, target_price: float, size: float):
        """Place a limit order"""
        order_id = f"LMT_{self.next_order_id}"
        self.next_order_id += 1

        order = PendingOrder(
            order_id=order_id,
            asset_idx=asset_idx,
            symbol=self.symbol_map[asset_idx],
            order_type=f'limit_{side}',
            target_price=target_price,
            size=size,
            created_step=self.current_step,
            expires_step=self.current_step + 60  # 1 hour expiry
        )

        self.pending_orders[order_id] = order
        logging.info(f"Placed {order.order_type} for {order.symbol} @ {target_price:.2f}")

    def check_pending_orders(self, current_prices: Dict[int, float]):
        """Check if any pending orders should trigger"""
        triggered = []

        for order_id, order in list(self.pending_orders.items()):
            current_price = current_prices.get(order.asset_idx)
            if current_price is None:
                continue

            # Check trigger conditions
            should_trigger = False

            if order.order_type == 'limit_buy':
                # Trigger if price drops to or below target
                should_trigger = current_price <= order.target_price

            elif order.order_type == 'limit_sell':
                # Trigger if price rises to or above target
                should_trigger = current_price >= order.target_price

            elif order.order_type == 'stop_buy':
                # Trigger if price breaks ABOVE target (breakout)
                should_trigger = current_price >= order.target_price

            elif order.order_type == 'stop_sell':
                # Trigger if price breaks BELOW target (breakdown)
                should_trigger = current_price <= order.target_price

            # Execute if triggered
            if should_trigger:
                self._execute_pending_order(order, current_price)
                triggered.append(order_id)

            # Cancel if expired
            if self.current_step >= order.expires_step:
                logging.info(f"Order {order_id} expired")
                triggered.append(order_id)

        # Remove triggered/expired orders
        for order_id in triggered:
            del self.pending_orders[order_id]

    def _execute_pending_order(self, order: PendingOrder, execution_price: float):
        """Execute a pending order at current market price"""
        logging.info(f"Executing {order.order_type} for {order.symbol} @ {execution_price:.2f}")

        # Convert to market order and execute
        if 'buy' in order.order_type:
            action = 2  # BUY
        else:
            action = 0  # SELL

        # Build action array
        actions = np.ones(self.n_assets, dtype=np.int32)
        sizes = np.full(self.n_assets, 0.5, dtype=np.float32)
        actions[order.asset_idx] = action
        sizes[order.asset_idx] = order.size

        prices = {order.asset_idx: execution_price}
        self.execute_actions(actions, sizes, prices)
```

---

## Environment Integration

### FastTradingEnv.step() Modification

```python
def step(self, action: int, price_offset: float = 0.0):
    # Decode action
    asset_idx = action // 6  # Now 6 actions per asset
    action_type = action % 6

    # Get current price
    current_data = self.market.get_current_data()
    current_prices = {idx: data['close'] for idx, (sym, data) in
                      enumerate(current_data.items())}

    # Check pending orders FIRST (before new action)
    self.portfolio.check_pending_orders(current_prices)

    # Execute new action
    if action_type in [0, 1, 2]:  # Market orders (existing logic)
        # ... existing code ...
        pass

    elif action_type == 3:  # LIMIT_BUY
        current_price = current_prices[asset_idx]
        target_price = current_price * (1 + price_offset)
        self.portfolio.place_limit_order(asset_idx, 'buy', target_price, size=0.5)

    elif action_type == 4:  # LIMIT_SELL
        current_price = current_prices[asset_idx]
        target_price = current_price * (1 + price_offset)
        self.portfolio.place_limit_order(asset_idx, 'sell', target_price, size=0.5)

    elif action_type == 5:  # STOP (Breakout/Breakdown)
        # Positive offset = Stop-Buy (breakout), Negative = Stop-Sell (breakdown)
        current_price = current_prices[asset_idx]
        target_price = current_price * (1 + price_offset)

        if price_offset > 0:
            self.portfolio.place_stop_order(asset_idx, 'buy', target_price, size=0.5)
        else:
            self.portfolio.place_stop_order(asset_idx, 'sell', target_price, size=0.5)

    # Step market
    next_data = self.market.step()
    # ... rest of existing code ...
```

---

## Model Architecture Changes

### Policy Head Enhancement

**Current**:
```python
output['policy_logits']: (batch, 192)
```

**New**:
```python
output['policy_logits']: (batch, 384)
output['price_offset']: (batch, 1)  # Range [-1, 1]
```

### MuZeroNetworks Update

```python
class MuZeroNetworks(nn.Module):
    # ... existing ...

    def setup(self):
        # ... existing setup ...
        self.price_offset_head = nn.Dense(1)  # New head

    def __call__(self, x, action=None, mode='initial_inference'):
        # ... existing logic ...

        if mode == 'initial_inference':
            # ... existing ...

            # Add price offset prediction
            price_offset_logit = self.price_offset_head(hidden)
            price_offset = jnp.tanh(price_offset_logit) * 0.20  # Scale to [-20%, +20%]

            return {
                'policy_logits': policy_logits,
                'value': value_logits,
                'latent_state': latent,
                'price_offset': price_offset  # NEW
            }

        # ... rest of modes ...
```

### Training Loop Update

**Trajectory Storage**:
```python
trajectory = {
    'observations': [...],
    'actions': [...],
    'price_offsets': [...],  # NEW: Store chosen offsets
    'rewards': [...],
    'values': [...],
    'policy_probs': [...]
}
```

**Replay Buffer**:
```python
class JAXReplayBuffer:
    def __init__(self, ...):
        # ... existing ...
        self.price_offsets = []  # NEW

    def add_trajectory(self, obs, actions, price_offsets, rewards, ...):
        # ... store price_offsets ...
```

**Loss Function**:
```python
def train_step(self, batch):
    # ... existing policy/value/reward losses ...

    # Price offset loss (MSE to optimal offset, calculated in hindsight)
    predicted_offsets = model_output['price_offset']
    target_offsets = calculate_optimal_offsets(batch)  # Hindsight analysis
    offset_loss = jnp.mean((predicted_offsets - target_offsets)**2)

    total_loss = policy_loss + value_loss + reward_loss + 0.1 * offset_loss
```

---

## Optimal Offset Calculation (Hindsight)

**Problem**: How do we know what offset the model SHOULD have chosen?

**Solution**: Look ahead and find the best price that was reached before expiry.

```python
def calculate_optimal_offsets(batch):
    """
    For each limit order action, find the optimal price offset in hindsight.
    """
    optimal_offsets = []

    for i, action in enumerate(batch['actions']):
        action_type = action % 6

        if action_type in [3, 4, 5]:  # Advanced order types
            asset_idx = action // 6
            current_price = batch['prices'][i][asset_idx]

            # Look ahead 60 steps (expiry window)
            future_prices = batch['prices'][i+1:i+61, asset_idx]

            if action_type == 3:  # LIMIT_BUY
                # Best price = lowest price reached
                best_price = np.min(future_prices)
            elif action_type == 4:  # LIMIT_SELL
                # Best price = highest price reached
                best_price = np.max(future_prices)
            else:  # STOP
                # For stops, optimal is the breakout/breakdown level
                # Use the price at max momentum change
                best_price = future_prices[np.argmax(np.abs(np.diff(future_prices)))]

            optimal_offset = (best_price - current_price) / current_price
            optimal_offsets.append(optimal_offset)

        else:
            # Market orders don't use offset
            optimal_offsets.append(0.0)

    return jnp.array(optimal_offsets)
```

---

## Action Masking Enhancement

**Problem**: Invalid order combinations (e.g., can't place limit buy if already fully long).

**Solution**: Extend action mask.

```python
def get_valid_actions(self) -> np.ndarray:
    """Returns (n_assets, 6) mask of valid actions"""
    mask = np.ones((self.n_assets, 6), dtype=np.int8)

    for idx in range(self.n_assets):
        position = self.positions.get(idx)

        if self.mode == 'SPOT':
            if position and position.side == 'long':
                # Can't buy more (max 1x long in SPOT)
                mask[idx, 2] = 0  # BUY
                mask[idx, 3] = 0  # LIMIT_BUY

            if not position or position.size == 0:
                # Can't sell if no position
                mask[idx, 0] = 0  # SELL
                mask[idx, 4] = 0  # LIMIT_SELL

        elif self.mode == 'MARGIN':
            # Margin allows both long/short, no restrictions
            pass

        # Check capital for new positions
        if self.state.cash < 100:  # Arbitrary minimum
            mask[idx, 2] = 0  # Can't buy
            mask[idx, 3] = 0  # Can't limit buy

    return mask.flatten()  # (n_assets * 6,)
```

---

## MCTS Integration

### Search Function Update

**Challenge**: MCTS search receives discrete action, but needs continuous offset.

**Solution 1**: Sample offset from model distribution during search.

```python
def _search(self, params, rng, observation, action_mask):
    # ... existing root logic ...

    # Get policy logits AND price offset distribution
    root_output = self.model_apply_fn(params, observation)
    policy_logits = root_output['policy_logits']
    price_offset_mean = root_output['price_offset']

    # During MCTS simulation, when action 3/4/5 is selected,
    # sample offset from Gaussian(mean, std=0.05)
    # Store sampled offset in search tree metadata

    # ... MCTS simulation ...

    # Return best action + its sampled offset
    best_action = policy_output.action
    best_offset = search_tree.metadata[best_action]['price_offset']

    return best_action, best_offset, stats
```

**Solution 2** (Simpler): Use deterministic mean.

```python
# Always use the mean offset (no sampling)
best_offset = root_output['price_offset'][0]
```

**Recommendation**: Use Solution 2 for simplicity. Model will learn to output the expected value.

---

## Observation Space Enhancement

**Problem**: Model needs to know about pending orders.

**Solution**: Add pending order info to observation.

```python
def _get_observation(self):
    obs = self.encoder.encode()  # (27, 64, 64)

    # Add pending orders channel (binary: 1 if order exists, 0 otherwise)
    pending_channel = np.zeros((1, 64, 64))

    for order in self.portfolio.pending_orders.values():
        asset_idx = order.asset_idx
        pending_channel[0, asset_idx, :] = 1.0  # Mark this asset has pending order

    obs = np.concatenate([obs, pending_channel], axis=0)  # (28, 64, 64)

    # ... rest of observation logic ...
```

**Note**: Update observation_space shape from `(27, 64, 64)` to `(28, 64, 64)`.

---

## Testing Strategy

### Unit Tests

```python
# tests/test_advanced_orders.py

def test_limit_buy_trigger():
    portfolio = PortfolioManager(...)
    portfolio.place_limit_order(asset_idx=0, side='buy', target_price=100, size=0.5)

    # Price doesn't reach target
    portfolio.check_pending_orders({0: 105})
    assert len(portfolio.pending_orders) == 1  # Still pending

    # Price reaches target
    portfolio.check_pending_orders({0: 99})
    assert len(portfolio.pending_orders) == 0  # Triggered and removed
    assert portfolio.positions[0].size > 0  # Position opened

def test_stop_loss():
    portfolio = PortfolioManager(...)
    portfolio.place_stop_order(asset_idx=0, side='sell', target_price=95, size=0.5)

    # Price above stop
    portfolio.check_pending_orders({0: 100})
    assert len(portfolio.pending_orders) == 1

    # Price hits stop
    portfolio.check_pending_orders({0: 94})
    assert len(portfolio.pending_orders) == 0
```

### Integration Test

```python
# Synthetic Level 2 with limit orders
env = CurriculumTradingEnv(level=2, use_advanced_orders=True)

# Agent should learn to place limit buy at -10% crash level
# Instead of reacting AFTER crash (too late), it predicts and places order
```

---

## Gradual Rollout Plan

### Phase 1: Add Infrastructure (No Training)
- Implement `PendingOrder` in `portfolio.py`
- Add `check_pending_orders()` logic
- Extend action space to 384
- Add price offset head to model

**Goal**: System compiles and runs (even if model doesn't use new actions).

### Phase 2: Test with Hardcoded Orders
```python
# Force agent to place limit orders manually
env.portfolio.place_limit_order(0, 'buy', 95, 0.5)

# Verify orders trigger correctly over 100 steps
```

### Phase 3: Enable in Synthetic Level 2
- Train on "Dip Catcher" level WITH limit orders enabled
- Agent should learn to place limit buys at crash level

**Expected**: Reward improves from +5% to +8% (better timing).

### Phase 4: Enable in Real Data
- Add to production training

---

## Hyperparameter Tuning

**New Hyperparameters**:
```python
ORDER_EXPIRY_STEPS = 60  # Orders expire after 1 hour
OFFSET_LOSS_WEIGHT = 0.1  # Weight for price offset loss
OFFSET_STD = 0.05  # Exploration noise for offset sampling
```

**Tune if**:
- Agent never uses advanced orders → Reduce expiry (make them more urgent)
- Agent spams orders → Increase action masking strictness
- Offset predictions are wild → Increase offset loss weight

---

## Expected Benefits

### Quantitative
- **Planning Horizon**: Effective horizon extends from 10 minutes to 1 hour
- **Execution Quality**: Better entry/exit prices (limit orders beat market orders)
- **Risk Management**: Automated stop losses prevent -10% single-asset crashes

### Qualitative
- **Strategy Diversity**: Enables swing trading, not just scalping
- **Interpretability**: Can inspect pending orders to understand agent's plan
- **Human-like**: Mimics real trader behavior (set orders and walk away)

---

## Future Extensions

### 1. OCO (One-Cancels-Other)
Place both stop-loss AND take-profit, cancel one when other triggers.

### 2. Trailing Stops
Stop loss that adjusts as price moves favorably.

### 3. Iceberg Orders
Split large orders into smaller chunks to reduce slippage.

### 4. Time-Based Orders
"Buy at 2 PM" (for events like FOMC announcements).

---

## Summary

**Action Space**: 192 → 384 (6 actions per asset)

**New Actions**:
- LIMIT_BUY (action % 6 == 3)
- LIMIT_SELL (action % 6 == 4)
- STOP (action % 6 == 5)

**Price Target**: Continuous offset ∈ [-20%, +20%]

**Model Changes**:
- Add `price_offset` head
- Observation shape: (27, 64, 64) → (28, 64, 64)

**Portfolio Changes**:
- Add `pending_orders` dict
- Implement `check_pending_orders()` per step

**Training Changes**:
- Store price offsets in replay buffer
- Add offset loss (MSE to hindsight optimal)

**Integration Path**:
1. Add infrastructure
2. Test with hardcoded orders
3. Train on synthetic Level 2
4. Deploy to real data

**Expected Impact**: +2-3% improvement in Sharpe ratio, reduced max drawdown.

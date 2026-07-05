# Multi-Action + Limit Orders + TP/SL Implementation

**Status**: ✅ IMPLEMENTED (Dec 28, 2025)

**Purpose**: Expand agent's execution capabilities while maintaining VectorBT native accounting.

**Compatibility**: Fully compatible with Neural Market Physics (TASK_07). Physics handles prediction, this handles execution.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PREDICTION LAYER                         │
│   (Neural Market Physics - TASK_07)                             │
│                                                                  │
│   Input: Market state → Output: Price predictions, volatility   │
│   - Learnable α (momentum), β (reversion), σ (volatility)       │
│   - Neural corrector for residual dynamics                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        EXECUTION LAYER                           │
│   (This Implementation)                                          │
│                                                                  │
│   Input: Predictions + State → Output: Orders                   │
│   - 5-action space per asset                                    │
│   - 4 action points per timestep                                │
│   - VectorBT native limit orders                                │
│   - TP/SL guardrails                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ENVIRONMENT LAYER                            │
│   (VectorBT)                                                     │
│                                                                  │
│   - Native accounting (no custom queues)                        │
│   - flexible=True for multi-order per step                      │
│   - Maker/taker fee simulation                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: VectorBT Research (DO FIRST)

Before implementation, validate VectorBT capabilities:

```python
# Test 1: Limit orders via Order.price
from vectorbt.portfolio.enums import Order
import numpy as np

# Market order (price=nan means execute at current price)
market_order = Order(size=1.0, price=np.nan, size_type=0, direction=0)

# Limit order (specific price)
limit_order = Order(size=1.0, price=100.5, size_type=0, direction=0)

# Test 2: Multiple orders per timestep (flexible=True)
# Check if VectorBT flex_order_func_nb supports this natively

# Test 3: TP/SL native support
# Check sl_stop, tp_stop parameters in from_order_func
```

**Output**: Document which VectorBT features we can use directly vs need to simulate.

---

## Phase 1: Environment TP/SL Guardrails

**Goal**: Protect positions with automatic stop-loss and take-profit.

### 1.1 Config Extension

```python
# In config or env __init__
tp_sl_config = {
    'enabled': True,
    'default_stop_loss': 0.05,    # 5% stop loss
    'default_take_profit': 0.10,  # 10% take profit
    'trailing_stop': False,       # Future: trailing stop
}
```

### 1.2 Position Tracking

```python
@dataclass
class Position:
    asset_idx: int
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    stop_loss: float      # NEW: absolute price
    take_profit: float    # NEW: absolute price
```

### 1.3 Guardrail Check (each step)

```python
def _check_guardrails(self):
    """Check TP/SL for all positions, execute if triggered."""
    current_prices = self._get_current_prices()

    for asset_idx, pos in list(self.portfolio.positions.items()):
        price = current_prices[asset_idx]

        if pos.side == 'long':
            # Stop loss (price drops below threshold)
            if price <= pos.stop_loss:
                self._close_position(asset_idx, reason='stop_loss')
            # Take profit (price rises above threshold)
            elif price >= pos.take_profit:
                self._close_position(asset_idx, reason='take_profit')

        elif pos.side == 'short':
            # Stop loss (price rises above threshold)
            if price >= pos.stop_loss:
                self._close_position(asset_idx, reason='stop_loss')
            # Take profit (price drops below threshold)
            elif price <= pos.take_profit:
                self._close_position(asset_idx, reason='take_profit')
```

### 1.4 Integration Point

Call `_check_guardrails()` at the START of each `step()`, before processing agent actions.

---

## Phase 2: Limit Orders (Persistent with Model-Driven TTL)

**Goal**: Add limit buy/sell to action space with persistent orders that wait for favorable prices.

**Implementation** (Updated Dec 28, 2025):
- Orders persist in `pending_orders` queue until filled or TTL expires
- Per-order TTL (default 100 steps, can be set from model predictions)
- Per-order limit_price (default offset, can be set from model predictions)
- Orders fill when price condition met (maker fee applied)
- Expired orders return reserved cash

**Model Integration Ready**:
```python
# _execute_limit_buy/sell accept optional params from model:
env._execute_limit_buy(asset_idx, limit_price=predicted_price, ttl=time_horizon)
```

### 2.1 Action Space Expansion

```python
# Current: 3 actions per asset
# [0] = Market Sell
# [1] = Hold
# [2] = Market Buy

# New: 5 actions per asset
# [0] = Market Sell (taker fee)
# [1] = Hold
# [2] = Market Buy (taker fee)
# [3] = Limit Sell (maker fee/rebate)
# [4] = Limit Buy (maker fee/rebate)

action_space_size = n_assets * 5  # 64 * 5 = 320
```

### 2.2 Limit Price Calculation

**Option A: Fixed offset from current price**
```python
def _get_limit_price(self, asset_idx: int, side: str) -> float:
    current_price = self._get_current_price(asset_idx)
    offset = 0.002  # 0.2% from current price

    if side == 'buy':
        return current_price * (1 - offset)  # Below current
    else:  # sell
        return current_price * (1 + offset)  # Above current
```

**Option B: Dynamic based on model confidence (requires Neural Market Physics)**
```python
def _get_limit_price(self, asset_idx: int, side: str, confidence: float) -> float:
    """
    Dynamic limit price based on model's prediction confidence.

    High confidence (0.9+): Aggressive limit (close to market price)
    Low confidence (0.5): Conservative limit (further from market)
    """
    current_price = self._get_current_price(asset_idx)

    # Map confidence [0.5, 1.0] to offset [0.5%, 0.05%]
    min_offset = 0.0005  # 0.05%
    max_offset = 0.005   # 0.5%
    offset = max_offset - (confidence - 0.5) * 2 * (max_offset - min_offset)

    if side == 'buy':
        return current_price * (1 - offset)
    else:
        return current_price * (1 + offset)
```

### 2.3 VectorBT Order Execution

```python
def _execute_limit_order(self, asset_idx: int, side: str, limit_price: float):
    """
    Execute limit order using VectorBT's native price field.

    Key: VectorBT Order.price != np.nan means limit order
    """
    # VectorBT handles fill logic:
    # - Limit buy fills when price <= limit_price
    # - Limit sell fills when price >= limit_price

    order = Order(
        size=self._calculate_position_size(asset_idx),
        price=limit_price,  # This makes it a limit order!
        size_type=SizeType.Amount,
        direction=Direction.LongOnly if side == 'buy' else Direction.ShortOnly,
        fees=self.maker_fee  # Lower fee for limit orders
    )

    return order
```

### 2.4 Fee Structure

```python
# In __init__
self.taker_fee = 0.0028   # 28 bps for market orders
self.maker_fee = 0.0010   # 10 bps for limit orders (or negative for rebate)
self.slippage = 0.0005    # 5 bps (only applies to market orders)
```

---

## Phase 3: Multi-Action (4 Action Points)

**Goal**: Allow 4 sequential actions per environment timestep.

### 3.1 Step Loop Modification

```python
def step(self, actions: List[int]) -> Tuple[np.ndarray, float, bool, bool, dict]:
    """
    Execute up to 4 actions per timestep.

    Args:
        actions: List of 4 action indices, or single action repeated

    Flow:
        1. Check TP/SL guardrails
        2. Execute Action 1 (portfolio updates, market time frozen)
        3. Execute Action 2 (portfolio updates, market time frozen)
        4. Execute Action 3 (portfolio updates, market time frozen)
        5. Execute Action 4 (portfolio updates, market time frozen)
        6. Advance market time
        7. Calculate reward
    """
    # Normalize input
    if isinstance(actions, int):
        actions = [actions] * self.action_points

    # 1. Check guardrails FIRST
    self._check_guardrails()

    # 2-5. Execute actions sequentially
    for action in actions[:self.action_points]:
        self._execute_action(action)
        # Portfolio state changes, but market data stays at time T

    # 6. Advance market time
    self.current_step += 1
    self.encoder.update(self._get_current_market_data())

    # 7. Calculate reward and observation
    reward = self._calculate_reward()
    obs = self._get_observation()
    done = self.current_step >= self.max_steps

    return obs, reward, done, False, {}
```

### 3.2 MCTS Integration

**Option A: Flat expansion (simplest)**
- MCTS sees actions as sequence: [a1, a2, a3, a4]
- Each search node is 4 actions deep before environment advances
- Pro: Clean MCTS integration
- Con: Search depth × 4

**Option B: Autoregressive selection (recommended)**
```python
def select_actions(self, state) -> List[int]:
    """
    Select 4 actions autoregressively.

    1. Run MCTS for action 1, select best
    2. Update internal portfolio state
    3. Run MCTS for action 2 with updated state, select best
    4. Repeat for actions 3, 4
    """
    actions = []
    temp_portfolio = self.portfolio.copy()

    for i in range(self.action_points):
        # Get observation with temporary portfolio state
        obs = self._get_observation_with_portfolio(temp_portfolio)

        # Run MCTS
        policy = self.model.get_policy(obs)
        action = self._select_action(policy)
        actions.append(action)

        # Update temporary portfolio (no market advance)
        temp_portfolio = self._simulate_action(temp_portfolio, action)

    return actions
```

### 3.3 Credit Assignment

Multi-action makes credit assignment harder. Solutions:

1. **Trajectory-level reward**: Single reward for all 4 actions
2. **Discounted within step**: r_total = r1 + 0.99*r2 + 0.99^2*r3 + 0.99^3*r4
3. **Portfolio delta**: Each action's reward = portfolio_value_after - portfolio_value_before

---

## Phase 4: Model Changes

### 4.1 Policy Head Expansion

```python
# Current
policy_head_output = 192  # 64 assets * 3 actions

# New
policy_head_output = 320  # 64 assets * 5 actions
```

### 4.2 Confidence Output (for dynamic limit prices)

```python
class PolicyHead(nn.Module):
    def forward(self, latent_state):
        logits = self.fc(latent_state)  # [batch, 320]

        # Reshape to [batch, n_assets, 5]
        logits = logits.reshape(-1, 64, 5)

        # Softmax per asset
        policy = F.softmax(logits, dim=-1)

        # Extract confidence as max probability
        confidence = policy.max(dim=-1).values  # [batch, 64]

        return policy.reshape(-1, 320), confidence
```

### 4.3 Integration with Neural Market Physics

When Neural Market Physics (TASK_07) is implemented:

```python
# Physics model provides:
# - σ (volatility per asset) → Use for TP/SL distance
# - α (momentum) → Use for limit order aggressiveness
# - Predicted prices → Use for limit price calculation

def _calculate_dynamic_tp_sl(self, asset_idx: int, side: str):
    """Use physics model's volatility σ to set TP/SL."""
    sigma = self.physics_model.σ_vol[asset_idx].item()

    # Higher volatility → wider TP/SL
    sl_distance = sigma * 2  # 2 sigma stop loss
    tp_distance = sigma * 4  # 4 sigma take profit

    return sl_distance, tp_distance
```

---

## Implementation Order

1. **Phase 0**: VectorBT research (1-2 hours)
   - Validate `flexible=True` works for multi-order
   - Test `Order.price` for limit orders
   - Check native TP/SL support

2. **Phase 1**: TP/SL guardrails (2-3 hours)
   - Add to Position dataclass
   - Implement `_check_guardrails()`
   - Test with existing 3-action space

3. **Phase 2**: Limit orders (3-4 hours)
   - Expand action space 3→5
   - Implement `_execute_limit_order()`
   - Add maker/taker fee structure
   - Test limit order fills

4. **Phase 3**: Multi-action (4-5 hours)
   - Modify `step()` for 4 action points
   - Implement autoregressive selection
   - Test credit assignment

5. **Phase 4**: Model update (2-3 hours)
   - Expand policy head 192→320
   - Add confidence output
   - Retrain from scratch (old checkpoints incompatible)

---

## Testing Checklist

- [x] TP/SL triggers correctly for long positions
- [x] TP/SL triggers correctly for short positions
- [x] Limit buy fills only when price drops to limit
- [x] Limit sell fills only when price rises to limit
- [x] Maker fee applied to limit orders
- [x] Taker fee + slippage applied to market orders
- [x] 4 actions execute before market time advances
- [x] Portfolio correctly tracks multi-action sequences
- [x] Model can learn to "asset jump" (sell A, buy B in same step)
- [x] Reward correctly attributed across action sequence

---

## Curriculum Training (Post-Implementation)

After implementation, train with curriculum:

1. **Level 0**: Synthetic data, TP/SL only
2. **Level 1**: Synthetic data, 5 actions, no multi-action
3. **Level 2**: Synthetic data, 5 actions, 2 action points
4. **Level 3**: Synthetic data, 5 actions, 4 action points
5. **Level 4**: Real data, full system

---

## Files to Modify

1. `mcts_trader/env/vectorbt_env.py`
   - Position dataclass (add stop_loss, take_profit)
   - `step()` method (multi-action loop)
   - `_check_guardrails()` new method
   - `_execute_limit_order()` new method
   - Action space size (192 → 320)

2. `train_jax_fast.py`
   - Policy head output size
   - Action decoding for 5 actions

3. `config/` (new or existing)
   - TP/SL parameters
   - Fee structure
   - Action points count

---

## Notes

- **No custom order queues**: Use VectorBT native `Order.price` field
- **VectorBT is source of truth**: All accounting handled by VectorBT
- **Backward incompatible**: New policy head requires fresh training
- **Synergy with TASK_07**: Physics σ → TP/SL distance, physics predictions → limit prices

---

## Usage Examples

### Training with 5-Action Space + Multi-Action

```bash
# Full system: 5 actions, 4 action points, TP/SL enabled
python train_jax_fast.py \
    --synthetic-level 0 \
    --n-actions-per-asset 5 \
    --action-points 4 \
    --tp-sl-enabled \
    --epochs 1000

# Curriculum Level 1: 5 actions, single action point
python train_jax_fast.py \
    --synthetic-level 0 \
    --n-actions-per-asset 5 \
    --action-points 1

# Legacy mode: 3 actions, single action point (backward compatible)
python train_jax_fast.py \
    --synthetic-level 0 \
    --n-actions-per-asset 3 \
    --action-points 1
```

### Environment Configuration

```python
from mcts_trader.env.vectorbt_env import VectorBTTradingEnv

# Full 5-action + multi-action mode
env = VectorBTTradingEnv(
    synthetic_level=0,
    n_assets=64,
    n_actions_per_asset=5,  # [Market Sell, Hold, Market Buy, Limit Sell, Limit Buy]
    action_points=4,        # 4 sequential actions per timestep
    tp_sl_enabled=True,     # 5% stop loss, 10% take profit
    maker_fee_bps=10,       # 10 bps for limit orders
    limit_offset_pct=0.002, # 0.2% default offset from current price
    default_limit_order_ttl=100,  # Orders persist up to 100 steps
)

# Execute multi-action: Sell asset 0, Buy asset 1, Limit Sell asset 2, Limit Buy asset 3
actions = [0*5+0, 1*5+2, 2*5+3, 3*5+4]
obs, reward, done, truncated, info = env.step(actions)
```

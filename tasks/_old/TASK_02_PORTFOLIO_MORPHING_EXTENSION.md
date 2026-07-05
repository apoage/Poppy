# TASK_02 Extension: Portfolio Morphing (Asset-to-Asset Swaps)

**Purpose**: Teach agent direct asset swaps (BTC→ETH) instead of always going through cash (BTC→USD→ETH)

**Why Critical**:
- **Fee savings**: 1 swap = 0.16% vs 2 trades = 0.32% (50% reduction!)
- **Portfolio rebalancing**: Morph holdings on the fly without cash intermediate
- **Real-world trading**: Matches how professional traders actually operate
- **Cross-pair arbitrage**: Exploit BTC/ETH, BTC/USD, ETH/USD triangle

---

## Action Space Extension

### Current (Basic Trading)
```python
action_space = 64 assets × 3 actions = 192 discrete actions
- action = asset_idx * 3 + action_type
- action_type: 0=Sell, 1=Hold, 2=Buy
```

**Limitation**: All trades go through USD (or base currency)

### Proposed (Portfolio Morphing)

**Option A: Expanded Action Space** (SIMPLE)
```python
action_space = 192 + 64*64 = 4288 discrete actions
- actions 0-191: Basic (Sell/Hold/Buy via cash)
- actions 192-4287: Swaps (asset_i → asset_j)

# Encoding:
if action < 192:
    # Basic trade via cash
    asset_idx = action // 3
    action_type = action % 3
else:
    # Direct swap
    swap_idx = action - 192
    source_asset = swap_idx // 64
    target_asset = swap_idx % 64
```

**Pros**: Simple encoding, easy to implement
**Cons**: Large action space (4288 actions), most invalid (can't swap what you don't hold)

**Option B: Dynamic Masking** (BETTER)
```python
# Same encoding as Option A, but use action mask to disable invalid swaps

def get_action_mask(self):
    """
    Returns: (4288,) boolean array
    - True if action valid
    - False if invalid (masked out by MCTS)
    """
    mask = np.zeros(4288, dtype=bool)

    # Basic actions (0-191): Always valid
    mask[:192] = True

    # Swap actions (192-4287): Only if source asset held
    for swap_idx in range(64 * 64):
        source_asset = swap_idx // 64
        target_asset = swap_idx % 64

        # Valid if:
        # 1. Source != target (can't swap to self)
        # 2. Currently holding source asset (shares > 0)
        if source_asset != target_asset:
            if self.portfolio.state.shares[source_asset] > 0:
                mask[192 + swap_idx] = True

    return mask
```

**Pros**: Large action space but MCTS only explores valid actions
**Cons**: Action mask changes every step (dynamic)

**Option C: Two-Stage Action** (CLEAN)
```python
# Stage 1: Select source asset (64 options)
# Stage 2: Select target asset (64 options) OR cash (1 option)

# But this requires MCTS to handle sequential actions (complex)
```

**Recommendation**: **Option B** (Dynamic Masking)
- JAX MCTS already uses action masks
- Most trading happens via cash anyway (swaps are <20% of trades)
- Easy to disable swaps for early curriculum levels

---

## Environment Implementation

### Step 1: Extend Action Space

**File**: `train_jax_fast.py` (FastTradingEnv)

```python
class FastTradingEnv(gym.Env):
    def __init__(self, ..., enable_swaps=True):  # NEW
        self.n_assets = n_assets
        self.enable_swaps = enable_swaps

        # Action space
        if enable_swaps:
            # 192 (basic) + 64*64 (swaps) = 4288
            self.action_space = spaces.Discrete(192 + n_assets * n_assets)
        else:
            # Basic only
            self.action_space = spaces.Discrete(192)

    def _get_observation(self):
        """Add action mask to observation"""
        obs = {
            'observation': state_tensor,  # (27, 64, 64)
            'action_mask': self._get_action_mask(),  # (4288,) or (192,)
            'to_play': 0
        }
        return obs

    def _get_action_mask(self):
        """Generate action mask (valid actions = True)"""
        if not self.enable_swaps:
            # Basic mode: all actions valid
            return np.ones(192, dtype=bool)

        # Swap mode: mask invalid swaps
        mask = np.ones(192 + self.n_assets * self.n_assets, dtype=bool)

        # Mask invalid swaps (source asset not held)
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                swap_action = 192 + i * self.n_assets + j

                # Invalid if:
                # 1. Swapping to self (i == j)
                # 2. No shares of source asset
                if i == j or self.portfolio.state.shares[i] <= 0:
                    mask[swap_action] = False

        return mask

    def step(self, action: int):
        """Execute action (basic or swap)"""

        if action < 192:
            # Basic action (existing code)
            asset_idx = action // 3
            action_type = action % 3
            self._execute_basic_action(asset_idx, action_type)

        else:
            # Swap action (NEW)
            swap_idx = action - 192
            source_asset = swap_idx // self.n_assets
            target_asset = swap_idx % self.n_assets
            self._execute_swap(source_asset, target_asset)

        # ... rest of step logic (advance market, compute reward, etc.)

    def _execute_swap(self, source_asset: int, target_asset: int):
        """
        Execute direct asset swap (source → target).

        Args:
            source_asset: Index of asset to sell
            target_asset: Index of asset to buy
        """
        # Get current prices
        source_symbol = self.padded_symbols[source_asset]
        target_symbol = self.padded_symbols[target_asset]

        current_data = self.market.get_current_data()

        if source_symbol not in current_data or target_symbol not in current_data:
            return  # Invalid swap (no data)

        source_price = current_data[source_symbol]['close']
        target_price = current_data[target_symbol]['close']

        # Get shares of source asset
        source_shares = self.portfolio.state.shares[source_asset]

        if source_shares <= 0:
            return  # Can't swap what you don't have

        # Swap fee (single fee, not double)
        swap_fee_rate = self.portfolio.fee_rate

        # Sell source asset (get USD value)
        source_value = source_shares * source_price
        proceeds = source_value * (1 - swap_fee_rate)

        # Buy target asset
        target_shares = proceeds / target_price

        # Update portfolio
        self.portfolio.state.shares[source_asset] = 0
        self.portfolio.state.shares[target_asset] += target_shares

        # Log swap
        logging.debug(f"SWAP: {source_symbol}({source_shares:.4f}) → {target_symbol}({target_shares:.4f}) | Fee: {swap_fee_rate*100:.2f}%")
```

---

## Curriculum Integration

### Add to Existing Levels

**Level 0-2**: Disable swaps (learn basics first)
```python
env = FastTradingEnv(synthetic_level=0, enable_swaps=False)
```

**Level 3**: Enable swaps (test if agent discovers fee savings)
```python
env = FastTradingEnv(synthetic_level=3, enable_swaps=True)
```

**Expected Behavior**:
- Agent should prefer swaps when rebalancing (fewer fees)
- Still use cash for large position changes (liquidity)

---

### NEW Level 4: Portfolio Rotation

**Add to synthetic_market.py**:

```python
def _level_4_portfolio_rotation(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 4: Rotating leadership - test portfolio morphing.

    Goal: Learn to rotate capital between assets as leadership changes.
    - 8 "sectors" (8 assets each)
    - Leadership rotates every 50 steps (one sector pumps)
    - Must SWAP from old leader to new leader (not cash intermediate)

    Reward structure favors swaps:
    - Swap: 1 fee (0.16%)
    - Cash cycle: 2 fees (0.32%)
    """
    data = {}
    SECTOR_SIZE = 8
    N_SECTORS = self.n_assets // SECTOR_SIZE
    ROTATION_PERIOD = 50
    PUMP_MAGNITUDE = 0.03  # 3% per step during leadership

    # Generate base GBM for all assets
    base_prices = {}
    for i in range(self.n_assets):
        # Flat with small noise
        prices = [100.0]
        for t in range(1, self.n_steps):
            noise = self.rng.normal(0, 0.01)
            prices.append(prices[-1] * (1 + noise))
        base_prices[i] = np.array(prices)

    # Add rotation pumps
    for t in range(self.n_steps):
        # Which sector is leading?
        leading_sector = (t // ROTATION_PERIOD) % N_SECTORS

        # Pump that sector
        for i in range(SECTOR_SIZE):
            asset_idx = leading_sector * SECTOR_SIZE + i

            if t % ROTATION_PERIOD < ROTATION_PERIOD:  # During pump
                base_prices[asset_idx][t] *= (1 + PUMP_MAGNITUDE)

    # Convert to OHLC
    for i in range(self.n_assets):
        data[f'SYN_{i:02d}'] = self._prices_to_ohlc(base_prices[i])

    return data
```

**Expected Agent Behavior**:
1. Identify leading sector (correlation encoder)
2. Buy leader assets (basic trading)
3. **When rotation happens**: SWAP from old leader → new leader (1 fee)
4. Avoid cash intermediate (would pay 2 fees)

**Pass Criteria**:
- Swap usage > 30% of total trades
- Reward with swaps > reward without swaps (ablation test)

---

## Testing

### Test 1: Swap Mechanics

**Script**: `tools/test_portfolio_swaps.py`

```python
from train_jax_fast import FastTradingEnv

def test_swap():
    """Test direct asset swap"""

    env = FastTradingEnv(synthetic_level=0, enable_swaps=True, n_assets=64)
    env.reset()

    # Manually buy asset 0
    env.step(0 * 3 + 2)  # Buy asset 0

    print(f"Holdings after buy: {env.portfolio.state.shares[0]:.4f} shares of asset 0")

    # Swap asset 0 → asset 1
    swap_action = 192 + 0 * 64 + 1  # source=0, target=1
    env.step(swap_action)

    print(f"Holdings after swap:")
    print(f"  Asset 0: {env.portfolio.state.shares[0]:.4f}")
    print(f"  Asset 1: {env.portfolio.state.shares[1]:.4f}")

    assert env.portfolio.state.shares[0] == 0, "Source asset should be empty"
    assert env.portfolio.state.shares[1] > 0, "Target asset should have shares"

    print("✅ Swap mechanics working")

if __name__ == "__main__":
    test_swap()
```

### Test 2: Fee Comparison

**Compare**: Swap vs Cash Cycle

```python
def test_fee_comparison():
    """Verify swap saves fees vs cash cycle"""

    env = FastTradingEnv(synthetic_level=0, enable_swaps=True)
    env.reset()

    # Scenario A: Swap (BTC → ETH direct)
    env_a = copy.deepcopy(env)
    env_a.step(0 * 3 + 2)  # Buy BTC
    portfolio_a_before = env_a.portfolio.state.total_value
    env_a.step(192 + 0 * 64 + 1)  # Swap BTC → ETH
    portfolio_a_after = env_a.portfolio.state.total_value

    # Scenario B: Cash cycle (BTC → USD → ETH)
    env_b = copy.deepcopy(env)
    env_b.step(0 * 3 + 2)  # Buy BTC
    portfolio_b_before = env_b.portfolio.state.total_value
    env_b.step(0 * 3 + 0)  # Sell BTC (to USD)
    env_b.step(1 * 3 + 2)  # Buy ETH (from USD)
    portfolio_b_after = env_b.portfolio.state.total_value

    fee_a = portfolio_a_before - portfolio_a_after
    fee_b = portfolio_b_before - portfolio_b_after

    print(f"Fee (swap): {fee_a:.4f}")
    print(f"Fee (cash cycle): {fee_b:.4f}")
    print(f"Savings: {(fee_b - fee_a) / fee_b * 100:.1f}%")

    assert fee_a < fee_b, "Swap should cost less than cash cycle"
    print("✅ Swap saves fees")
```

### Test 3: Level 4 Convergence

**Command**:
```bash
python train_jax_fast.py \
  --synthetic-level 4 \
  --enable-swaps \
  --model-size medium \
  --epochs 2000 \
  --batch-size 256 \
  --name level4_rotation
```

**Monitor** swap usage:
```python
# Add to training loop
swap_actions = [a for a in trajectory['actions'] if a >= 192]
swap_ratio = len(swap_actions) / len(trajectory['actions'])
print(f"Swap usage: {swap_ratio*100:.1f}%")
```

**Pass Criteria**:
- Reward > +2% (overcomes rotation lag)
- Swap usage > 30%

---

## CLI Arguments

**Add to train_jax_fast.py**:

```python
parser.add_argument('--enable-swaps', action='store_true',
                   help='Enable direct asset-to-asset swaps (portfolio morphing)')
```

**Usage**:
```bash
# Basic trading only (Levels 0-2)
python train_jax_fast.py --synthetic-level 0

# Portfolio morphing enabled (Levels 3-4)
python train_jax_fast.py --synthetic-level 4 --enable-swaps
```

---

## Expected Impact

**Without Swaps** (Basic Trading):
- All trades: Asset ↔ Cash
- Rebalancing: Sell A → Buy B (2 fees)
- Total fees: 0.32% per rebalance

**With Swaps** (Portfolio Morphing):
- Rebalancing: A → B direct (1 fee)
- Total fees: 0.16% per rebalance
- **50% fee reduction**

**Training Impact**:
- Agent learns to minimize unnecessary cash intermediates
- Portfolio "morphs" shape dynamically
- More realistic professional trading behavior

---

## Integration Priority

**Phase 1** (Gemini TASK_02):
- Implement basic Levels 0-3 (no swaps)
- Get Level 0 working first

**Phase 2** (Follow-up task):
- Add swap action space
- Implement Level 4 (rotation)
- Test fee savings

**Rationale**: Don't bloat initial task - validate basics first, then add advanced tactics.

---

## Summary

**What**: Direct asset-to-asset trading (portfolio morphing)
**Why**: 50% fee savings, realistic trading, portfolio tactics
**How**: Expand action space to 4288 (192 basic + 4096 swaps), dynamic masking
**When**: After basic curriculum (Levels 0-3) validates, add as Level 4
**Integration**: Optional flag `--enable-swaps`, disabled by default for simplicity

This teaches the agent "ninja moves" - morphing portfolio shape on the fly without cash friction.

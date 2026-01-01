# Dojo: Modular Curriculum Training Architecture

**Created**: Dec 19, 2025
**Status**: Design - Not Implemented
**Problem**: Gemini broke codebase by integrating tactics directly into `train_jax_fast.py` instead of creating modular components

---

## Vision: Separation of Concerns

### Current Problem (What Gemini Did)
```
train_jax_fast.py  ← MONOLITHIC
├── Basic trading logic
├── Swap logic (WRONG - hardcoded)
├── Synthetic market integration (WRONG - hardcoded)
└── Training loop
```

**Issues**:
- Portfolio tactics (swaps) hardcoded into main training file
- Violates modular architecture principle
- Can't easily enable/disable tactics
- Can't compose tactics incrementally
- Breaks single responsibility principle

### Proper Architecture (What Should Exist)
```
Dojo System (Modular)
├── dojo_manager.py          # Curriculum orchestrator
├── tactics/                 # Portfolio tactics modules
│   ├── __init__.py
│   ├── basic_trading.py     # Buy/Sell/Hold via cash
│   ├── portfolio_morphing.py # Asset→Asset swaps
│   ├── pairs_trading.py     # Cointegration tactics (future)
│   └── market_making.py     # Bid-ask tactics (future)
├── generators/              # Synthetic market generators
│   ├── __init__.py
│   ├── base.py              # SyntheticMarket interface
│   ├── sine_waves.py        # Level 0
│   ├── lagged_pairs.py      # Level 1
│   ├── dip_crashes.py       # Level 2
│   └── noisy_trends.py      # Level 3
└── environments/            # Gym environment wrappers
    ├── __init__.py
    ├── basic_env.py         # FastTradingEnv (basic)
    └── morphing_env.py      # FastTradingEnv + swaps
```

---

## Module Responsibilities

### 1. `dojo_manager.py` (Training Orchestrator)

**Purpose**: Orchestrate curriculum training experiments

**Responsibilities**:
- Load curriculum config (sequence of levels)
- Instantiate synthetic generators
- Configure environment with appropriate tactics
- Run training loop
- Checkpoint management
- Progress tracking

**Example**:
```python
from dojo_manager import DojoManager
from tactics import BasicTrading, PortfolioMorphing
from generators import SineWaves, LaggedPairs

# Define curriculum
curriculum = [
    {'level': 0, 'generator': SineWaves, 'tactics': [BasicTrading], 'epochs': 500},
    {'level': 1, 'generator': LaggedPairs, 'tactics': [BasicTrading], 'epochs': 1000},
    {'level': 3, 'generator': NoisyTrends, 'tactics': [BasicTrading, PortfolioMorphing], 'epochs': 2000},
]

# Run curriculum
dojo = DojoManager(curriculum_config=curriculum)
dojo.run()
```

**Key API**:
```python
class DojoManager:
    def __init__(self, curriculum_config: List[Dict]):
        self.curriculum = curriculum_config
        self.current_level = 0

    def run(self):
        """Run full curriculum sequence"""
        for level_config in self.curriculum:
            self._train_level(level_config)

    def _train_level(self, config: Dict):
        """Train single curriculum level"""
        # 1. Instantiate generator
        generator = config['generator'](n_assets=64, n_steps=10000)

        # 2. Build environment with tactics
        env = self._build_env(generator, config['tactics'])

        # 3. Run training
        self._run_training(env, epochs=config['epochs'])

    def _build_env(self, generator, tactics: List[Tactic]):
        """Compose environment from tactics"""
        # Start with base env
        env = BasicTradingEnv(market=generator)

        # Wrap with tactic layers
        for tactic in tactics:
            env = tactic.wrap(env)

        return env
```

---

### 2. `tactics/` Module (Portfolio Tactics)

**Purpose**: Modular trading tactics that extend action space

**Design Pattern**: Gym Wrapper Pattern

#### `tactics/basic_trading.py`
```python
class BasicTrading:
    """Basic Buy/Sell/Hold via cash intermediate"""

    @staticmethod
    def get_action_space_size(n_assets: int) -> int:
        return n_assets * 3  # Sell, Hold, Buy

    @staticmethod
    def execute(env, action: int):
        """Execute basic action (existing logic)"""
        asset_idx = action // 3
        action_type = action % 3
        # ... existing execute_actions logic

    @staticmethod
    def get_action_mask(env) -> np.ndarray:
        """Mask for basic actions"""
        return env.portfolio.get_valid_actions().flatten()
```

#### `tactics/portfolio_morphing.py`
```python
class PortfolioMorphing:
    """Direct asset→asset swaps (no cash intermediate)"""

    @staticmethod
    def get_action_space_size(n_assets: int) -> int:
        return n_assets * n_assets  # All possible swaps

    @staticmethod
    def execute(env, swap_action: int, base_action_count: int):
        """Execute swap action"""
        swap_idx = swap_action - base_action_count
        source_asset = swap_idx // env.n_assets
        target_asset = swap_idx % env.n_assets

        # Execute swap logic (isolated here, not in main env)
        _execute_swap(env, source_asset, target_asset)

    @staticmethod
    def get_action_mask(env, base_action_count: int) -> np.ndarray:
        """Mask for swap actions"""
        mask = np.zeros(env.n_assets * env.n_assets, dtype=bool)

        # Only allow swaps from held positions
        for i, pos in env.portfolio.state.positions.items():
            if pos.size > 1e-6:  # FIX: Use .size not .shares
                for j in range(env.n_assets):
                    if i != j:
                        mask[i * env.n_assets + j] = True
        return mask
```

**Key Insight**: Tactics are COMPOSABLE. You can combine tactics:
```python
# Level 0-2: Basic only
env = BasicTradingEnv(tactics=[BasicTrading])

# Level 3+: Basic + Swaps
env = BasicTradingEnv(tactics=[BasicTrading, PortfolioMorphing])

# Future: Basic + Swaps + Pairs Trading
env = BasicTradingEnv(tactics=[BasicTrading, PortfolioMorphing, PairsTrading])
```

---

### 3. `generators/` Module (Synthetic Markets)

**Purpose**: Modular synthetic data generators for curriculum levels

**Current State**: Gemini actually did this CORRECTLY in `mcts_trader/data/synthetic_market.py`

**Proposed Refactor**: Break into separate files for clarity

#### `generators/base.py`
```python
class SyntheticGenerator(ABC):
    """Base class for synthetic generators"""

    def __init__(self, n_assets: int, n_steps: int, seed: Optional[int]):
        self.n_assets = n_assets
        self.n_steps = n_steps
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def generate(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate OHLCV data"""
        pass

    def _prices_to_ohlc(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert price series to OHLC"""
        # ... existing logic
```

#### `generators/sine_waves.py` (Level 0)
```python
class SineWaves(SyntheticGenerator):
    """Pure sine waves - teach trend detection"""

    def generate(self):
        data = {}
        for i in range(self.n_assets):
            period = 100 + i * 5  # Different periods
            amp = 0.05 + i * 0.001

            prices = 100 * (1 + amp * np.sin(2 * np.pi * np.arange(self.n_steps) / period))
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)
        return data
```

#### `generators/lagged_pairs.py` (Level 1)
```python
class LaggedPairs(SyntheticGenerator):
    """Lagged correlation pairs - teach correlation detection"""

    def generate(self):
        # ... existing Level 1 logic
```

**Usage**:
```python
# In dojo_manager.py
from generators import SineWaves, LaggedPairs, DipCrashes

generators = {
    0: SineWaves,
    1: LaggedPairs,
    2: DipCrashes,
}

generator = generators[level](n_assets=64, n_steps=10000, seed=42)
market_data = generator.generate()
```

---

### 4. `environments/` Module (Gym Wrappers)

**Purpose**: Environment variants with different tactic compositions

#### `environments/basic_env.py`
```python
class BasicTradingEnv(FastTradingEnv):
    """FastTradingEnv with only basic trading (Buy/Sell/Hold)"""

    def __init__(self, market, tactics=None, **kwargs):
        super().__init__(market=market, **kwargs)

        # Register tactics
        self.tactics = tactics or [BasicTrading]

        # Build composite action space
        self._build_action_space()

    def _build_action_space(self):
        """Compute action space from tactics"""
        total_actions = sum(t.get_action_space_size(self.n_assets) for t in self.tactics)
        self.action_space = spaces.Discrete(total_actions)

    def step(self, action: int):
        # Route action to appropriate tactic
        cumulative = 0
        for tactic in self.tactics:
            tactic_size = tactic.get_action_space_size(self.n_assets)
            if action < cumulative + tactic_size:
                return tactic.execute(self, action)
            cumulative += tactic_size
```

---

## Migration Plan

### Step 1: Create Modules (No Changes to train_jax_fast.py Yet)

1. **Create `tactics/` module**:
   ```bash
   mkdir mcts_trader/tactics
   touch mcts_trader/tactics/__init__.py
   ```

2. **Extract swap logic** from `train_jax_fast.py` into `tactics/portfolio_morphing.py`
   - Copy `_execute_swap()` → `PortfolioMorphing.execute()`
   - Copy `_get_swap_mask()` → `PortfolioMorphing.get_action_mask()`

3. **Create basic tactic** in `tactics/basic_trading.py`
   - Extract existing trading logic into `BasicTrading.execute()`

4. **Create `dojo_manager.py`**:
   - Curriculum orchestrator
   - Uses `train_jax_fast.py` as training engine (no modifications needed yet)

### Step 2: Test Modular Components

```bash
# Test Level 0 with basic trading only
python dojo_manager.py --level 0 --tactics basic

# Test Level 3 with basic + swaps
python dojo_manager.py --level 3 --tactics basic,morphing
```

### Step 3: Refactor train_jax_fast.py (Clean Removal)

Once Dojo is validated:
1. Remove `_execute_swap()` from `train_jax_fast.py`
2. Remove `_get_swap_mask()` from `train_jax_fast.py`
3. Remove `enable_swaps` parameter
4. Keep `train_jax_fast.py` as pure training loop (no tactics)

---

## Benefits of Modular Architecture

### 1. **Composability**
```python
# Easy to mix tactics
env = BasicTradingEnv(tactics=[BasicTrading, PortfolioMorphing, PairsTrading])
```

### 2. **Testability**
```python
# Test each tactic independently
def test_swap_fee_savings():
    tactic = PortfolioMorphing()
    # ... isolated test
```

### 3. **Extensibility**
```python
# Add new tactic without touching core
class MarketMaking(Tactic):
    def execute(self, env, action):
        # Place limit orders at bid/ask
```

### 4. **Clear Separation**
- `train_jax_fast.py`: Pure training loop (JAX, MCTS, gradient updates)
- `dojo_manager.py`: Curriculum orchestration
- `tactics/`: Portfolio tactics (composable)
- `generators/`: Synthetic data (modular)

---

## Implementation Priority

**Phase 1** (CURRENT TASK):
1. Create `tactics/` module structure
2. Create `dojo_manager.py` skeleton
3. Extract swap logic to `tactics/portfolio_morphing.py`
4. Test modular components

**Phase 2** (After validation):
1. Refactor `train_jax_fast.py` to remove hardcoded tactics
2. Migrate all synthetic curriculum to Dojo
3. Add tests for each tactic

**Phase 3** (Future):
1. Add advanced tactics (pairs trading, market making)
2. Extend curriculum to 22 levels (from MATHEMATICAL_TRADING_TACTICS_CURRICULUM.md)
3. Auto-tune curriculum difficulty

---

## File Locations

**New Files** (to be created):
- `dojo_manager.py` (Curriculum orchestrator)
- `mcts_trader/tactics/__init__.py`
- `mcts_trader/tactics/basic_trading.py`
- `mcts_trader/tactics/portfolio_morphing.py`

**Existing Files** (to be refactored):
- `train_jax_fast.py` (Remove hardcoded tactics)
- `mcts_trader/data/synthetic_market.py` (Keep as-is, works correctly)

**Design Docs**:
- `lore/DOJO_MODULAR_ARCHITECTURE.md` (THIS FILE)
- `tasks/TASK_02_SYNTHETIC_MARKET_GENERATORS.md` (Update status)
- `tasks/TASK_02_PORTFOLIO_MORPHING_EXTENSION.md` (Update to modular approach)

---

## Summary

**Problem**: Gemini integrated tactics directly into `train_jax_fast.py` (monolithic)
**Solution**: Modular architecture with composable tactics
**Philosophy**: Each tactic is a separate module, composed via Gym wrappers
**Benefit**: Clean separation, easy testing, incremental curriculum

**Next Step**: Implement Phase 1 - Create modular components without breaking existing code.

# TASK_142: Historic Data Branch for Dual-Branch Training

## Goal
Add historic market data support to `train_easy.py` for Bella vs Historic dual-branch training.

## Motivation
- Current dual-branch (Bella vs Synth L0) has distribution mismatch
- Synth learns 3.5x slower, causing merge asymmetry
- Bella vs Historic = two realistic distributions that should merge cleanly
- Historic provides ground truth, Bella provides augmentation/diversity

## Current State
- `HistoricalMarket` exists at `mcts_trader/data/history.py`
- 24GB database at `trading_data.db`
- `train_easy.py` only supports Bella and SyntheticMarket

## Requirements

### 1. Add `--historic` flag to train_easy.py
```bash
python train_easy.py --preset large --historic --output-dir jax_ckpt_easy_B
```

### 2. HistoricEpisodeGenerator class
Create episode generator that:
- Samples random 400-step windows from HistoricalMarket
- Returns prices in same format as Bella: `(T, n_assets)` float32
- Handles missing data / asset filtering
- Caches frequently-accessed periods in RAM

### 3. Interface compatibility
Must produce same output as `generate_bella_prices_on_device()`:
```python
def generate_historic_episode(market: HistoricalMarket, rng: np.random.Generator) -> np.ndarray:
    """
    Sample random episode from historic data.

    Returns:
        prices: (episode_length, n_assets) array of close prices
    """
```

### 4. e-dojo integration
Update `launch_dual_branches()` to support:
```python
# Branch A: Bella (generated)
# Branch B: Historic (real data)
launcher.launch_dual_branches(
    branch_a="bella",
    branch_b="historic",
    ...
)
```

## Implementation Plan

### Step 1: HistoricEpisodeGenerator (train_easy.py)
```python
class HistoricEpisodeGenerator:
    def __init__(self, db_path: str = "trading_data.db",
                 episode_length: int = 400,
                 n_assets: int = 64):
        self.market = HistoricalMarket(db_path, n_assets=n_assets)
        self.episode_length = episode_length
        self.valid_starts = self._compute_valid_starts()

    def _compute_valid_starts(self) -> np.ndarray:
        """Find all timestamps where we can start a full episode."""
        # Need episode_length consecutive valid timestamps
        pass

    def sample_episode(self, rng: np.random.Generator) -> np.ndarray:
        """Sample random episode, return (T, n_assets) prices."""
        start_idx = rng.choice(len(self.valid_starts))
        start_ts = self.valid_starts[start_idx]

        # Get prices for episode_length steps
        prices = []
        self.market.current_time = start_ts
        for _ in range(self.episode_length):
            data = self.market.get_current_data()
            # Extract close prices for all assets
            closes = [data[sym]['close'] for sym in sorted(data.keys())]
            prices.append(closes)
            self.market.step()

        return np.array(prices, dtype=np.float32)
```

### Step 2: CLI flag
```python
parser.add_argument('--historic', action='store_true',
                    help='Use historic data instead of Bella/Synthetic')
```

### Step 3: DataManager integration
```python
if args.historic:
    self.historic_generator = HistoricEpisodeGenerator(
        episode_length=config.episode_length,
        n_assets=config.n_assets
    )
    # Use historic instead of bella/synthetic
```

### Step 4: e-dojo update
Add `--branch-b-historic` flag or similar to orchestrator.

## Data Flow

```
Historic DB (24GB SQLite)
    ↓
HistoricEpisodeGenerator.sample_episode()
    ↓
prices: (400, 64) np.ndarray
    ↓
Same pipeline as Bella → oracle → training
```

## Testing

```bash
# Test historic episode generation
python -c "
from train_easy import HistoricEpisodeGenerator
import numpy as np

gen = HistoricEpisodeGenerator()
rng = np.random.default_rng(42)
prices = gen.sample_episode(rng)
print(f'Episode shape: {prices.shape}')
print(f'Price range: {prices.min():.2f} - {prices.max():.2f}')
"

# Test training with historic
python train_easy.py --preset small --historic --epochs 5
```

## Expected Outcome
- Bella vs Historic dual-branch training
- Both branches learn similar "real market" representations
- Clean merges due to aligned distributions
- Ground truth from historic + diversity from Bella

## Files to Modify
- `train_easy.py`: Add HistoricEpisodeGenerator, --historic flag
- `tools/orchestration/e-dojo_v1.py`: Support historic branch option

## Status
- [ ] HistoricEpisodeGenerator class
- [ ] CLI flag --historic
- [ ] DataManager integration
- [ ] e-dojo support
- [ ] Testing

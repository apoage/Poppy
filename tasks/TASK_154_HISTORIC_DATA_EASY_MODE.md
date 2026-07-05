# TASK_154: Historic Data Support for Easy Mode

**Status:** PENDING
**Priority:** HIGH (Phase 2 of training pipeline)

**Objective:** Add historical database reading to `train_easy.py` using `training_schedule.json` as the period map, reusing window/anti-overfitting patterns from `train_jax_fast.py`.

---

## 1. Background

Easy Mode currently trains on:
- Synthetic data (SyntheticMarket)
- Bella-generated data (CNN diffusion)

Phase 2 requires training/benchmarking on **real historic data** from `trading_data.db`.

---

## 2. Existing Infrastructure (Reuse)

### training_schedule.json
Pre-computed map of valid periods with assets:
```json
{
  "periods": [
    {
      "start": 1539733620,
      "end": 1555287780,
      "duration_days": 180,
      "available_assets": ["BTC/USD", "ETH/USD", ...]
    }
  ]
}
```
**Benefit:** No brute-force DB scan at startup.

### dojo_v3_historical.py (Reference)
- `_load_schedule()` - loads training_schedule.json
- `_filter_valid_periods()` - removes periods beyond DB max date
- `_flatten_runs()` - extracts asset combinations
- Uses `--periods-file` to pass to train_jax_fast.py

### train_jax_fast.py Anti-Overfitting
- `--smart-allocation` - distributed tiling within period
- `--jitter-minutes 360` - random offset per episode
- Window shifting to avoid memorizing exact timestamps

---

## 3. Implementation Plan

### A. New CLI Arguments (train_easy.py)
```python
parser.add_argument('--historic', action='store_true',
                    help='Use historic data from DB instead of Bella/Synthetic')
parser.add_argument('--db-path', type=str, default='trading_data.db',
                    help='Path to trading database')
parser.add_argument('--schedule', type=str, default='training_schedule.json',
                    help='Path to training schedule map')
parser.add_argument('--jitter-minutes', type=int, default=60,
                    help='Random offset per episode for anti-overfitting')
```

### B. New Data Generator Class
```python
class HistoricDataGenerator:
    """Load historic OHLCV from database using schedule map."""

    def __init__(self, db_path, schedule_path, config, jitter_minutes=60):
        self.schedule = self._load_schedule(schedule_path)
        self.db = sqlite3.connect(db_path)
        self.periods = self._filter_valid_periods()
        self.jitter = jitter_minutes

    def _load_schedule(self, path):
        with open(path) as f:
            return json.load(f)

    def _filter_valid_periods(self):
        # Same logic as dojo_v3_historical._filter_valid_periods
        pass

    def generate_episode(self, period_idx=None):
        """Generate one episode from a random period.

        Returns: prices array (episode_length + window_size, n_assets)
        """
        if period_idx is None:
            period_idx = random.randint(0, len(self.periods) - 1)

        period = self.periods[period_idx]
        assets = period['available_assets'][:64]  # Pad/truncate to 64

        # Random start within period (jitter for anti-overfitting)
        start_ts = period['start'] + random.randint(0, self.jitter * 60)

        # Load from DB
        prices = self._load_prices(assets, start_ts, length)
        return prices

    def _load_prices(self, assets, start_ts, length):
        # SQL query to load OHLCV for multiple assets
        # Similar to train_jax_fast.py historic loading
        pass
```

### C. Integration with OnTheFlyGenerator
```python
class OnTheFlyGenerator:
    def __init__(self, ..., historic_generator=None):
        self.historic_generator = historic_generator

    def _refill_buffer(self):
        if self.historic_generator is not None:
            # Use historic data
            prices = self.historic_generator.generate_episode()
        elif self.bella_generator is not None:
            # Use Bella
            ...
        else:
            # Use Synthetic
            ...
```

### D. Oracle Compatibility
- Oracle (path_solver.py) works on any price array
- No changes needed - just pass historic prices instead of synthetic

---

## 4. Anti-Overfitting Measures

| Technique | Description |
|-----------|-------------|
| **Jitter** | Random start offset within period (60-360 min) |
| **Period Rotation** | Cycle through periods, don't repeat same period consecutively |
| **Asset Shuffling** | Randomize asset order in the 64-slot array |
| **Noise Injection** | Optional: Add small noise to prices (like synthetic_noise) |

---

## 5. Usage

```bash
# Phase 2: Train on historic data
python train_easy.py --historic --schedule training_schedule.json \
    --epochs 500 --jitter-minutes 120

# Benchmark on specific period
python train_easy.py --historic --db-path trading_data.db \
    --start-date 2023-01-01 --end-date 2023-03-01 --epochs 100
```

---

## 6. Files to Modify

| File | Changes |
|------|---------|
| `train_easy.py` | Add --historic flag, HistoricDataGenerator class |
| `easy_config.py` | Add historic_mode, db_path, schedule_path, jitter_minutes |
| `tools/orchestration/e-dojo_v1.py` | Add --historic branch mode |

---

## 7. Success Criteria

- [ ] Load periods from training_schedule.json without DB scan
- [ ] Generate episodes from random periods with jitter
- [ ] Oracle agreement on historic data comparable to Bella (within 5%)
- [ ] No timestamp memorization (test with shifted validation set)

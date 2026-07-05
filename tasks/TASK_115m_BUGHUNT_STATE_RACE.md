# TASK_115m: Bug Hunt - Race Conditions & State Management

**Status**: OPEN
**Created**: 2026-01-25
**Scope**: RNG key reuse, mutable defaults, shared state, thread-safety, NaN/Inf edge cases

---

## Summary Table

| Bug ID | File | Line | Severity | Status | Description |
|--------|------|------|----------|--------|-------------|
| BUG-001 | streaming.py | 266, 277 | HIGH | OPEN | RNG key as mutable instance state (not thread-safe) |
| BUG-002 | adversarial_training.py | 214, 269, 375 | HIGH | OPEN | RNG state mutation across method calls |
| BUG-003 | adversarial_training.py | 208-211 | HIGH | OPEN | Unbounded list/dict accumulation |
| BUG-004 | curriculum_bridge.py | 97, 167, 176 | HIGH | OPEN | Non-deterministic seed + RNG state mutation |
| BUG-005 | agent_evaluator.py | 227, 333 | MEDIUM | OPEN | RNG state mutation across evaluations |
| BUG-006 | adversarial_training.py | 427 | MEDIUM | OPEN | Unbounded metrics_history growth |
| BUG-007 | adversarial_training.py | 409-410 | MEDIUM | OPEN | regime_performance unbounded accumulation |
| BUG-008 | agent_evaluator.py | 121 | MEDIUM | OPEN | NaN guard applied after division, not before |

---

## Detailed Bug Reports

---

#### BUG-001: RNG Key as Mutable Instance State (Thread-Safety)
**File**: `diff_teacher/generation/streaming.py:266,277`
**Severity**: HIGH
**Status**: OPEN

```python
class InfiniteVRAMStream:
    def __init__(self, model, params, config, seed=42):
        self.generator = VRAMGenerator(model, params, config)
        self.key = jax.random.PRNGKey(seed)  # Line 266 - mutable state
        self._regime = None

    def __next__(self) -> Array:
        batch, self.key = self.generator.generate_batch(self.key, self._regime)  # Line 277
        return batch
```

**Problem**:
- `self.key` is mutable instance state updated on every `__next__()` call
- If multiple threads call `__next__()` concurrently, race condition on `self.key` assignment
- Two threads could read same key value before either writes, producing duplicate batches
- Even without threads, if the stream is pickled (for multiprocessing), key state may be corrupted

**Fix**:
```python
# Option 1: Document as not thread-safe
class InfiniteVRAMStream:
    """WARNING: Not thread-safe. Use separate instances per thread."""

# Option 2: Add threading lock
import threading
def __init__(self, ...):
    self._lock = threading.Lock()

def __next__(self):
    with self._lock:
        batch, self.key = self.generator.generate_batch(self.key, self._regime)
    return batch
```

---

#### BUG-002: RNG State Mutation in AdversarialTrainer
**File**: `diff_teacher/adversarial_training.py:214,269,375`
**Severity**: HIGH
**Status**: OPEN

```python
class AdversarialTrainer:
    def __init__(self, ...):
        self.rng = jax.random.PRNGKey(42)  # Line 214

    def _sample_regimes(self, batch_size: int) -> jnp.ndarray:
        self.rng, sample_key = jax.random.split(self.rng)  # Line 269
        indices = jax.random.categorical(sample_key, ...)
        return indices

    def train_step(self) -> AdversarialMetrics:
        self.rng, gen_key = jax.random.split(self.rng)  # Line 375
        latents, _ = self.teacher.generate(...)
```

**Problem**:
- `self.rng` is mutable state split multiple times per training step
- If `train_step()` is called concurrently (distributed training), both threads mutate same key
- Breaks determinism: resuming from checkpoint with same step count gives different results
- Difficult to reproduce bugs since RNG state depends on call history

**Fix**:
```python
# Option 1: Fold in iteration count for reproducibility
def train_step(self, iteration: int) -> AdversarialMetrics:
    step_rng = jax.random.fold_in(self._base_rng, iteration)
    sample_key, gen_key = jax.random.split(step_rng)
    ...

# Option 2: Pass RNG through method signatures
def train_step(self, rng: jax.Array) -> Tuple[AdversarialMetrics, jax.Array]:
    rng, sample_key, gen_key = jax.random.split(rng, 3)
    ...
    return metrics, rng
```

---

#### BUG-003: Unbounded List/Dict Accumulation
**File**: `diff_teacher/adversarial_training.py:208-211`
**Severity**: HIGH
**Status**: OPEN

```python
def __init__(self, ...):
    # Tracking
    self.metrics_history: List[AdversarialMetrics] = []  # Line 208
    self.regime_performance: Dict[str, List[float]] = {  # Lines 209-211
        regime: [] for regime in REGIME_TO_IDX.keys()
    }
```

**Problem**:
- Both structures grow without bound during training
- For 50 epochs with 100 steps/epoch = 5000 entries in `metrics_history`
- `regime_performance` stores ALL regrets ever seen per regime
- Memory leak: long training runs accumulate gigabytes of tracking data
- No checkpoint/restore for these structures

**Fix**:
```python
from collections import deque

def __init__(self, ...):
    self.metrics_history: deque = deque(maxlen=1000)
    self.regime_performance: Dict[str, deque] = {
        regime: deque(maxlen=100) for regime in REGIME_TO_IDX.keys()
    }
```

---

#### BUG-004: Non-Deterministic Seed + RNG State Mutation
**File**: `diff_teacher/curriculum_bridge.py:97,167,176`
**Severity**: HIGH
**Status**: OPEN

```python
class AdversarialDataGenerator:
    def __init__(self, teacher_checkpoint, config):
        # RNG
        self.rng = jax.random.PRNGKey(int(time.time()))  # Line 97 - NON-DETERMINISTIC!

    def generate_batch(self, regime_weights=None):
        self.rng, sample_key = jax.random.split(self.rng)  # Line 167
        regime_indices = jax.random.categorical(...)

        # Generate latents
        self.rng, gen_key = jax.random.split(self.rng)  # Line 176 - unused gen_key!
        latents, _ = self.teacher.generate(batch_size=batch_size, num_steps=1)
```

**Problems**:
1. **Non-deterministic seed**: `int(time.time())` makes runs non-reproducible
2. **Unused key**: `gen_key` is split but never used (teacher.generate() uses its own key)
3. **State mutation**: Same issue as BUG-002

**Fix**:
```python
def __init__(self, teacher_checkpoint, config, seed: int = 42):
    self.rng = jax.random.PRNGKey(seed)  # Deterministic, configurable

def generate_batch(self, regime_weights=None):
    # Split once for all needs
    self.rng, sample_key = jax.random.split(self.rng)
    # Don't split again if teacher.generate uses its own key
```

---

#### BUG-005: RNG State Mutation in AgentEvaluator
**File**: `diff_teacher/agent_evaluator.py:227,333`
**Severity**: MEDIUM
**Status**: OPEN

```python
class AgentEvaluator:
    def __init__(self, ...):
        self.rng = jax.random.PRNGKey(42)  # Line 227

    def evaluate(self, prices, regime=None, return_trajectory=False):
        # Episode loop
        for step in range(self.window_size, steps):
            self.rng, step_rng = jax.random.split(self.rng)  # Line 333
            ...
```

**Problem**:
- RNG state persists across `evaluate()` calls
- Calling `evaluate()` multiple times on same prices gives different actions
- Breaks reproducibility for debugging

**Fix**:
```python
def evaluate(self, prices, regime=None, return_trajectory=False, seed: int = 42):
    rng = jax.random.PRNGKey(seed)  # Fresh key per evaluation
    for step in range(self.window_size, steps):
        rng, step_rng = jax.random.split(rng)
        ...
```

---

#### BUG-006: Unbounded metrics_history Growth
**File**: `diff_teacher/adversarial_training.py:427`
**Severity**: MEDIUM
**Status**: OPEN

```python
def train_step(self) -> AdversarialMetrics:
    ...
    self.metrics_history.append(metrics)  # Line 427 - grows forever
    return metrics
```

**Problem**:
- Every training step appends to list
- No pruning, no persistence, no size limit
- 50 epochs × 100 steps = 5000 entries minimum

**Fix**: Use bounded deque (see BUG-003 fix)

---

#### BUG-007: regime_performance Unbounded Accumulation
**File**: `diff_teacher/adversarial_training.py:409-410`
**Severity**: MEDIUM
**Status**: OPEN

```python
# Line 257: Only uses last 50
mean_regret = np.mean(self.regime_performance[regime][-50:])

# Line 409-410: But stores ALL
for regime, regret in zip(regime_names, regrets):
    self.regime_performance[regime].append(float(regret))
```

**Problem**:
- Code only ever uses `[-50:]` slice (line 257)
- But stores unlimited history
- Wastes memory storing data that's never used
- No validation that regime key exists before append

**Fix**:
```python
# Use deque with maxlen=50
self.regime_performance: Dict[str, deque] = {
    regime: deque(maxlen=50) for regime in REGIME_TO_IDX.keys()
}

# Add validation
if regime not in self.regime_performance:
    logger.warning(f"Unknown regime: {regime}")
    continue
self.regime_performance[regime].append(float(regret))
```

---

#### BUG-008: NaN Guard Applied After Division
**File**: `diff_teacher/agent_evaluator.py:121`
**Severity**: MEDIUM
**Status**: OPEN

```python
# Line 121 in SimpleStateEncoder.encode():
log_ret = np.log(window_prices[-1] / (window_prices[-1-h] + 1e-8) + 1e-8)
```

**Problem**:
- Guards are structured: `log(a / (b + eps) + eps)`
- The inner `+ 1e-8` prevents division by zero: good
- The outer `+ 1e-8` prevents log(0): good
- BUT: if `window_prices[-1]` is negative (data error), result can still be NaN
- More critically: for very small positive `b`, `a / b` can overflow to Inf before eps helps

**Shape Analysis**:
- Input: `window_prices` shape `(window_size, n_assets)`
- If `window_prices[-1-h]` = 1e-15 and `window_prices[-1]` = 1.0:
  - `1.0 / (1e-15 + 1e-8) = 1.0 / 1e-8 = 1e8` (valid, but huge)
- If `window_prices[-1-h]` = 0.0 and `window_prices[-1]` = 0.0:
  - `0.0 / (0.0 + 1e-8) = 0.0`, then `log(0.0 + 1e-8) = -18.4` (valid)

**Fix**:
```python
# Guard both numerator and denominator
safe_denom = np.maximum(window_prices[-1-h], 1e-8)
safe_numer = np.maximum(window_prices[-1], 1e-8)
log_ret = np.log(safe_numer / safe_denom)
```

---

## Additional Observations

### Not Bugs (Verified Safe)

1. **`extraction.py:31`** - `jnp.sqrt(var)` has guard: `var = jnp.maximum(mean_sq - mean ** 2, 1e-10)`
2. **`adversarial_training.py:118`** - Guard position is correct: `/ (prices[:, :-1, :] + 1e-8)`
3. **`streaming.py` VRAMGenerator** - Correctly returns new key: `return generated, key`

### Thread-Safety Summary

| Component | Thread-Safe? | Notes |
|-----------|--------------|-------|
| VRAMGenerator | YES | Stateless, key passed in |
| InfiniteVRAMStream | NO | Mutable self.key |
| AdversarialTrainer | NO | Mutable self.rng |
| AdversarialDataGenerator | NO | Mutable self.rng |
| AgentEvaluator | NO | Mutable self.rng |
| AdversarialDataLoader | NO | Mutable self.indices, self.current_idx |

---

## Recommendations

### Immediate (HIGH Severity)
1. Add thread-safety warning docstrings to InfiniteVRAMStream
2. Use `jax.random.fold_in(base_key, iteration)` pattern in training loops
3. Replace unbounded lists with `collections.deque(maxlen=N)`
4. Use deterministic seed in AdversarialDataGenerator

### Short-term (MEDIUM Severity)
5. Accept seed as parameter in AgentEvaluator.evaluate()
6. Add bounds checking on NaN-sensitive calculations
7. Validate regime keys before dict access

### Pattern to Adopt
```python
# Instead of:
self.rng = jax.random.PRNGKey(42)
self.rng, key = jax.random.split(self.rng)

# Prefer:
self._base_rng = jax.random.PRNGKey(42)
# In method:
key = jax.random.fold_in(self._base_rng, iteration)
```

---

*Last updated: 2026-01-25*

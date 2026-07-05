# Bug Hunt Report: Race Conditions & State Management

**Date:** 2026-01-25
**Focus:** `diff_teacher/`
**Status:** In Progress

## Summary
Found potential concurrency issues in data loaders and streaming generators, plus a stale closure risk in JAX JIT compilation.

| Bug ID | Severity | Component | Description |
|--------|----------|-----------|-------------|
| BUG-001 | MEDIUM | Streaming | Thread safety race in `InfiniteVRAMStream` RNG key update |
| BUG-002 | HIGH | Data Loader | `RainbowTableLoader` and `MarketDataLoader` are not thread-safe (shared state/cursor) |
| BUG-003 | LOW | Generator | Stale closure in `VRAMGenerator` JIT compilation with mutable config |
| BUG-004 | LOW | Generator | Potential Division by Zero in `TimeEmbedding` for small dimensions |

## Details

#### BUG-001: Race Condition in InfiniteVRAMStream
**File**: `diff_teacher/generation/streaming.py:202-205`
**Severity**: MEDIUM
**Status**: OPEN

```python
def __next__(self) -> Array:
    batch, self.key = self.generator.generate_batch(self.key, self._regime)
    return batch
```

**Problem**: `self.key` is updated in place. If `InfiniteVRAMStream` is shared across threads (e.g. in a pre-fetching thread pool), multiple threads could read the same `self.key` before it is updated, resulting in duplicate batches and RNG key reuse (statistical correlation).

**Fix**: Add a lock for key update or document strict single-threaded usage.

#### BUG-002: Shared State & Cursor in Data Loaders
**File**: `diff_teacher/data_loader.py:650-652`, `diff_teacher/data_loader.py:734`
**Severity**: HIGH
**Status**: OPEN

```python
# RainbowTableLoader
def get_batch_streaming(self, batch_size: int, rng: Optional[np.random.Generator] = None):
    # ...
    if not hasattr(self, '_current_data') or self._current_data is None:
        self.switch_period(rng)
    data, timestamps, symbols = self._current_data
```

**Problem**:
1. `RainbowTableLoader` shares `self._current_data` across all callers. If used with multiprocessing or multithreading (common in data loading), this state will be raced or duplicated.
2. `MarketDataLoader` shares `self.conn` and `self.cursor`. `sqlite3` cursors are not thread-safe.

**Fix**: Ensure each worker has its own loader instance, or implement thread-local storage for cursors/buffers.

#### BUG-003: Stale Closure in VRAMGenerator JIT
**File**: `diff_teacher/generation/streaming.py:46-48`
**Severity**: LOW
**Status**: OPEN

```python
self._generate_batch_jit = jax.jit(
    partial(self._generate_batch_impl, config=config)
)
```

**Problem**: `config` is a `StreamConfig` dataclass. JAX JIT treats non-Array arguments as static. If `config` is a standard mutable dataclass (not frozen), and its attributes are modified *after* `__init__`, the JIT-compiled function will still use the old `config` object (and its old values if JAX specialized on the object identity, or if `partial` captured the object state). While unlikely to change at runtime in current usage, it's a trap for future refactoring.

**Fix**: Mark `StreamConfig` as `frozen=True` or pass `config` fields as static arguments explicitly.

#### BUG-004: Division by Zero in TimeEmbedding / SkeletonConditioner
**File**: `diff_teacher/flow_matching.py:32`, `diff_teacher/skeleton/encoder.py:277`
**Severity**: LOW
**Status**: OPEN

```python
half_dim = self.dim // 2
emb_scale = jnp.log(10000.0) / (half_dim - 1)
```

**Problem**: If `self.dim` (or `d_model`) is 2 or 3, `half_dim` is 1, causing division by zero. This crashes for small model dimensions.

**Fix**: Add `max(half_dim - 1, 1)` or assertion.

#### BUG-005: Non-Deterministic Hash in EventEncoder
**File**: `diff_teacher/skeleton/encoder.py:182`
**Severity**: LOW
**Status**: OPEN

```python
type_idx = hash(event.event_type) % self.n_event_types
```

**Problem**: Python's `hash()` is randomized per process (unless `PYTHONHASHSEED` is set). This means the embedding index for "flash_crash" will differ between training runs or inference sessions, rendering weights invalid.

**Fix**: Use a stable string hash (e.g. `zlib.adler32` or explicit lookup table).

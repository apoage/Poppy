# TASK_711: train_easy.py Audit Cleanup

**Status**: DONE
**Priority**: 2 (no training-blocking bugs, but #1 will crash eventually)
**Source**: Full audit of train_easy.py at 2590 lines, session 34

---

## Problem

train_easy.py has accumulated 30+ TASKs worth of features. Full audit found 3 bugs, ~300 lines of dead code, design debt, and missing safety checks. None are currently blocking training, but bug #1 (CSVLogger) WILL crash the moment a new metric key appears mid-run.

---

## Stage 1: Fix Bugs (do first, small + safe)

### 1A. CSVLogger dict mutation during iteration
**File**: `train_easy.py` L1858-1861
**Bug**: Modifying dict while iterating raises `RuntimeError` in Python 3.
```python
# BROKEN
for key in row:
    if key not in self.fieldnames:
        row.pop(key)  # RuntimeError: dictionary changed size

# FIX
row = {k: v for k, v in row.items() if k in self.fieldnames}
```
**Risk**: Will crash any run where new metric keys appear after first epoch.

### 1B. Benchmark subprocess stderr pipe deadlock
**File**: `train_easy.py` L2001
**Bug**: `stderr=subprocess.PIPE` but stderr is never read. Pipe buffer fills → benchmark deadlocks.
```python
# FIX: change PIPE to DEVNULL
stderr=subprocess.DEVNULL
```

### 1C. `_make_slot` unbounded recursion
**File**: `train_easy.py` L549-550 (SegmentPool)
**Bug**: If all runs in a group fail to load, `_make_slot` recurses through every run (up to 27 deep). Not dangerous but sloppy.
```python
# FIX: convert recursion to loop
def _make_slot(self, group_key):
    runs = self.groups[group_key]
    while self._group_run_idx[group_key] < len(runs):
        run_idx = self._group_run_idx[group_key]
        spec = runs[run_idx]
        # ... try loading ...
        # on failure:
        self._group_run_idx[group_key] += 1
    return None  # group exhausted
```

### 1D. NaN detection in training loop
**File**: `train_easy.py` training loop (~L2420)
**Gap**: If model diverges to NaN, training runs silently for hours burning GPU.
```python
# After train_epoch():
if math.isnan(train_metrics.get('total_loss', 0)):
    logger.error("NaN loss detected — saving emergency checkpoint and stopping")
    trainer.save_checkpoint(str(output_dir / 'phase1_nan_abort.pkl'), epoch, train_metrics)
    break
```

**Stage 1 total**: ~15 lines changed. Zero risk to training behavior.

---

## Stage 2: Remove Dead Code (~300 lines)

### 2A. Delete `_generate_episode_data` (L922-1126)
~200 lines. Single-episode processing method, never called. All code paths use `_generate_batch_episodes`. This dead copy drifts from the batched version — anyone reading the code might edit the wrong one.

**Verify before deleting**: `grep -n '_generate_episode_data' train_easy.py` confirms zero callers.

### 2B. Delete `_generate_prices_bella` (L1409-1417)
~10 lines. Single-episode Bella generation, never called. Batched path in `_refill_buffer` handles Bella directly.

### 2C. Remove unused imports (L66-67)
```python
# DELETE these two — only temporal encoding is used
jit_batch_encode_compressed, jit_vmap_encode_compressed
```

### 2D. Remove dead constructor params
`historical_markets`, `historical_market_labels` on `OnTheFlyGenerator.__init__` — never passed by any caller. Legacy from pre-TASK_707 preload approach that was replaced by lazy loading, then by SegmentPool.

Also remove `self._historical_markets`, `self._historical_market_labels` instance vars and the legacy branch in `advance_period()` that uses them (L834-840).

### 2E. Remove legacy `_load_period` + `advance_period` lazy-load path
With TASK_710 SegmentPool, `main()` never passes `period_specs` or `db_path` to the generator. The `_load_period()` method and the `_period_specs` branch of `advance_period()` are unreachable from production code paths.

**Keep**: `advance_period()` single-period mode (L841-842: `return` for no-op) and the single `--db-path` constructor path — that's still a valid use case.

**Delete**: `_load_period()`, the `_period_specs` branch of `advance_period()`, and the `period_specs`/`db_path` params from the constructor.

**Stage 2 total**: ~300 lines removed. No behavior change. Run `python -c "import ast; ast.parse(open('train_easy.py').read())"` after each deletion.

---

## Stage 3: Design Cleanup (medium effort, low risk)

### 3A. Replace `historical_market` sentinel hack
Currently `self.historical_market = segment_pool` as a truthy sentinel. Replace with explicit flag:
```python
self._use_historical = (segment_pool is not None or historical_market is not None)
```
Then check `self._use_historical` instead of `self.historical_market is not None` in `_refill_buffer` and `_use_gpu_synth`.

### 3B. Extract AR blend computation to method
The warmup/blend/cap logic is duplicated in `_generate_batch_episodes` and `train_epoch`. Extract to:
```python
def _compute_ar_blend(self) -> float:
    """Compute current AR blend probability from epoch + config."""
    if not getattr(self.config, 'autoregressive_state', False):
        return 0.0
    if self._current_epoch < self.config.ar_warmup_epochs:
        return 0.0
    blend_epochs = self.config.ar_blend_epochs
    if blend_epochs > 0 and self._current_epoch < self.config.ar_warmup_epochs + blend_epochs:
        return min((self._current_epoch - self.config.ar_warmup_epochs) / blend_epochs,
                    self.config.ar_max_blend)
    return self.config.ar_max_blend
```
Remove the duplicate from `_generate_batch_episodes`. The one in `train_epoch` stays (it also does logging on transitions).

### 3C. Fix stderr on benchmark subprocess
Already covered in 1B, but also: consider logging stderr to a file instead of /dev/null for debugging failed benchmarks:
```python
stderr_path = results_path.with_suffix('.stderr')
stderr_file = open(stderr_path, 'w')
_benchmark_proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=stderr_file)
```

**Stage 3 total**: ~40 lines changed. Low risk, improves maintainability.

---

## Stage 4: Robustness (optional, higher effort)

### 4A. VRAM OOM recovery in `_refill_buffer`
Wrap `_generate_batch_episodes` in try/except for JAX OOM. On OOM, halve `n_to_refill` and retry:
```python
try:
    episode_datas = self._generate_batch_episodes(prices_list, levels)
except jax.errors.XlaRuntimeError as e:
    if "out of memory" in str(e).lower():
        logger.warning(f"OOM during refill with {len(prices_list)} episodes, retrying with half")
        half = len(prices_list) // 2
        episode_datas = self._generate_batch_episodes(prices_list[:half], levels[:half])
    else:
        raise
```

### 4B. Save live config state alongside CSV
TASK_708 mutates `config.ar_max_blend` during training. Save a `config_live.json` periodically (every 100 epochs) so the runtime state is recoverable:
```python
if (epoch + 1) % 100 == 0:
    config.save(str(output_dir / 'config_live.json'))
```

### 4C. Early stop fix for MSE-feedback regime
The early stop check (L2534) gates on `ar_max_blend >= 1.0`, which is never true with TASK_708 active. Options:
- Lower the gate: `ar_max_blend >= 0.8` (still meaningful AR)
- Remove the gate entirely (check agreement regardless of AR level)
- Add a separate benchmark-based stop (already exists via TASK_706, so maybe just document that this is intentional)

**Recommendation**: Document that early stop is intentionally disabled when MSE feedback is active — the benchmark pass (TASK_706) is the real stop criterion for historical training.

---

## NOT FIXING (intentional / architectural)

| Item | Why not |
|------|---------|
| State channel 64x tiling | Model architecture — encoder expects tiled input. Would need to modify MuZeroNetworks representation function. |
| `generate_batch` list+concat | Cost is <1% of epoch time. Pre-allocation would add complexity for negligible gain. |
| `_check_benchmark_passed` disk reads | Runs every 10 epochs, reads ~1-2 small JSON files. Negligible. |
| OnTheFlyGenerator size (~1000 lines) | Splitting would require passing many shared state variables between classes. Current structure works and all paths are tested by production runs. |
| Config mutation | CSV captures live values. Saving live config (4B) covers reproducibility. |

---

## Execution Order

```
Stage 1 (bugs)     → do immediately, safe to commit independently
Stage 2 (dead code) → do next, verify with syntax check after each deletion
Stage 3 (design)    → do after Stage 2, since dead code removal simplifies diffs
Stage 4 (robustness) → do when convenient, each item is independent
```

**Estimated total**: ~15 lines added (Stage 1) + ~300 lines removed (Stage 2) + ~40 lines changed (Stage 3) = net -250 lines.

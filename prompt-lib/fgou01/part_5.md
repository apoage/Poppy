# Part 5: Data Source Selection Chain

## Status: NEEDS_FIX

## Findings

### 1. Historical priority chain in `_refill_buffer()` is correct
The `if/elif` chain in `_refill_buffer()` (train_easy.py:1274-1330) follows a strict priority order:
1. **Historical** (`self.historical_market is not None`) -- line 1274
2. **Secondary** (experience replay, 90/10 mixing) -- line 1278
3. **Bella** (primary Bella generator) -- line 1282
4. **GPU synthetic** (`self._use_gpu_synth`) -- line 1294
5. **CPU synthetic fallback** -- line 1308

When `self.historical_market is not None`, the first branch always wins. The `elif` chain prevents any fallthrough to Bella/synthetic. **No mixing within a single refill iteration.**

### 2. No cross-source batch mixing
`generate_batch()` (train_easy.py:1448-1525) samples from the buffer by episode index. Each episode in the buffer comes from exactly one source (tagged by level: -2=historical, -1=Bella, >=0=synthetic). The buffer itself can contain episodes from different sources across refill cycles (e.g., if `historical_market` was set/unset between refills), but within a single buffer when historical is set, **all episodes are historical** since every refill iteration takes the historical branch. No mixing path exists at batch construction time.

### 3. Branch B (Bella stabilizer) priority is correct
When `historical_market is None` and `bella_generator is not None`, Bella is the primary source (line 1282). The secondary path (line 1278) only triggers when `_should_use_secondary()` returns True AND a secondary source is configured. This implements the 90/10 experience replay correctly: Bella is primary, synthetic is secondary. Bella always takes priority over synthetic as the primary source.

### 4. `_use_gpu_synth` flag does NOT account for `historical_market`
**This is the main issue.** On line 546:
```python
self._use_gpu_synth = (bella_generator is None)
```
When `historical_market` is set but `bella_generator` is None (which is the historical-only case), `_use_gpu_synth` is set to `True`. This is functionally harmless because the `_refill_buffer()` priority chain checks `self.historical_market is not None` first (line 1274), so the `elif self._use_gpu_synth` branch (line 1294) is never reached. However, the flag is semantically wrong.

More importantly, the dead code on lines 549-552:
```python
if bella_generator is None and not self._use_gpu_synth:
    self._submit_prefetch_tasks()  # CPU fallback path
```
This condition is `False and not True = False`, so prefetch is never started. But since `_use_gpu_synth` is always True when `bella_generator is None`, this entire block is dead code regardless of historical mode.

### 5. `_should_use_secondary()` has a side effect even when historical is set
The `_mix_counter` increments on every call to `_should_use_secondary()` (line 1128). However, when historical is set, `_should_use_secondary()` is never called because the `elif` on line 1278 is skipped by the historical `if` on line 1274. **No bug here** -- the counter only advances in the non-historical path.

### 6. Validation generator correctly excludes historical
The `val_generator` (train_easy.py:1964-1976) intentionally omits `historical_market`, with a comment explaining "val stays on Bella/synthetic" (line 1975). This is correct design -- validation should test generalization, not memorization of historical data.

### 7. e-dojo launches historical with `use_bella=False`
The orchestrator (e-dojo_v1.py:1187-1197) launches Branch A historical with `use_bella=False`, so no `--bella` flag is passed. This means `bella_generator=None` inside the subprocess's `OnTheFlyGenerator`. Combined with `--db-path` being set, the historical branch is the sole data source. Branch B (Bella stabilizer) runs in a **separate process** with its own `OnTheFlyGenerator`, so no cross-process mixing occurs.

### 8. `_generate_prices_bella()` is dead code
The method at train_easy.py:1114-1122 is defined but never called anywhere. It was likely superseded by the inline Bella generation in `_refill_buffer()` (lines 1284-1293). Not a bug, just cleanup debt.

### 9. Local `gpu_synth` variable vs class `_use_gpu_synth`
In `main()` (train_easy.py:1933):
```python
gpu_synth = bella_generator is None and historical_market is None
```
This local variable correctly accounts for historical, but is only used for logging. It is **never passed** to `OnTheFlyGenerator`. The class internally computes `_use_gpu_synth = (bella_generator is None)` which does not check for historical. This inconsistency is benign due to the priority chain but creates a maintenance hazard.

## Bugs Found

- **BUG-5.1**: `_use_gpu_synth` flag is semantically incorrect when `historical_market` is set (train_easy.py:546). The flag says "use GPU synthetic" even when the actual source is historical. **Severity: Low** -- functionally harmless because the `_refill_buffer()` priority chain prevents the GPU synthetic path from executing when historical is set. But a future refactor that checks `_use_gpu_synth` for conditional logic (e.g., VRAM budgeting, logging, or metric tagging) would get wrong answers.

## Recommended Fixes

### Fix BUG-5.1: Account for `historical_market` in `_use_gpu_synth`

```python
# train_easy.py:546
# Before:
self._use_gpu_synth = (bella_generator is None)

# After:
self._use_gpu_synth = (bella_generator is None and historical_market is None)
```

This aligns the class attribute with the local `gpu_synth` variable in `main()` (line 1933) and makes the flag semantically correct. No behavioral change since the priority chain already prevents the wrong path, but eliminates a maintenance hazard.

### Optional cleanup: Remove dead `_generate_prices_bella()` method

The method at train_easy.py:1114-1122 is never called. It can be safely deleted to reduce confusion about the data generation API surface.

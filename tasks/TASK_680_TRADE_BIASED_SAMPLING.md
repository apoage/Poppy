# TASK_680: Trade-Biased Sampling in OnTheFlyGenerator

**Status**: Implemented (code merged, awaiting overnight validation)
**Priority**: High (next lever if threshold alone isn't enough)
**Depends on**: TASK_679 (threshold lowered, needs validation first)
**Blocked by**: Nothing technical — waiting on training data

---

## Problem

Even after lowering `min_alpha_threshold` (TASK_679), trade examples are sparse.
Uniform timestep sampling means 95%+ of training signal is pure hold reinforcement.
The model sees trade contexts too rarely to form distinctive patterns.

## Solution

Bias timestep sampling toward windows where the oracle actually traded.
Use a **coin-flip approach** (not weighted np.random.choice) for O(1) sampling.

---

## Adversarial Review (2026-02-17) — 4 Issues Addressed

### ADV-1: Latent Overfitting Trap

**Risk**: Boosting trade windows 6× effectively shrinks the active dataset. Model
memorizes the specific noise patterns surrounding the few trade events instead of
generalizing trade structure.

**Symptom**: Train agreement spikes but val agreement drops or stalls.

**Fix**: Require `N_BUFFER >= 64` when biased sampling is active. More episodes =
more unique trade patterns = less overfitting risk. Add config validation:
```python
if config.trade_sample_boost > 0 and config.buffer_size < 64:
    print(f"WARNING: trade_sample_boost={config.trade_sample_boost} with "
          f"buffer_size={config.buffer_size} risks overfitting. Recommend >= 64.")
```

### ADV-2: Sampling Efficiency

**Risk**: `np.random.choice(n, p=weights)` is O(N) per call (builds CDF, binary search).
In a loop over batch entries, this kills the data generation thread → GPU starvation.

**Fix**: Replace weighted sampling with **coin-flip + pre-computed trade index list**.
O(1) per sample, no probability arrays:

```python
# Pre-compute during _refill_buffer (O(T) once):
ep_data['trade_indices'] = np.where(trade_density > 0.01)[0]  # timesteps with trades

# During generate_batch (O(1) per sample):
if len(ep_data['trade_indices']) > 0 and self.rng.random() < trade_sample_prob:
    offset = self.rng.choice(ep_data['trade_indices'])  # O(1) uniform from small list
else:
    offset = self.rng.integers(0, n_available)           # O(1) uniform from full range
```

This is simpler, faster, and achieves the same bias without CDF overhead.

### ADV-3: Lookback Window Alignment

**Risk**: Trade label at timestep t reflects oracle lookahead (t+1...t+20). But the
observation window is [t-W, t]. If we bias toward t, we need to ensure we're sampling
the window that *predicts* the trade, not the post-trade window.

**Verification**: The current sampling logic in `generate_batch` uses the offset as the
*end* of the observation window: `obs = episode[offset-W:offset]`, `label = labels[offset]`.
So biasing toward high-trade-density offsets correctly selects windows where the model
must *predict* the upcoming trade. The observation precedes the label — no leakage.

**Test**: Explicitly verify that `offset` maps to `obs_window_end`, not `obs_window_start`:
```python
# offset=100 should give obs[100-60:100] and label[100], not obs[100:160]
assert obs_slice.stop == offset, "Window must END at offset, not start"
```

### ADV-4: Hold Accuracy Degradation

**Risk**: Showing trades 30% of the time may cause false positives — model starts
"seeing trades where there are none" in boring markets.

**Monitoring**: Track `agree_hold` explicitly. If it drops below 85% while `agree_trade`
rises, the model is overtrading. Reduce `trade_sample_prob` or add a hold-accuracy
floor constraint.

**Safeguard**: The `trade_sample_prob` should be tuned, not hardcoded at 0.3. Start
at 0.15 (conservative) and increase if `agree_hold` stays above 90%:
```python
trade_sample_prob: float = 0.15  # Start conservative, raise if agree_hold > 90%
```

---

## Revised Implementation

### 1. Index phase (during `_refill_buffer`):
```python
# After generating episode, scan for trade timesteps
for ep_idx, ep_data in enumerate(self.buffer):
    actions = ep_data['actions']  # (T, 320) soft targets
    # Per-timestep trade density: sum of non-hold probability across assets
    actions_reshaped = actions.reshape(actions.shape[0], 64, 5)
    trade_density = 1.0 - actions_reshaped[:, :, 1].mean(axis=1)  # (T,)

    # Pre-compute trade index list (O(T) once, O(1) to sample from later)
    ep_data['trade_indices'] = np.where(trade_density > 0.01)[0]
    ep_data['n_trade_steps'] = len(ep_data['trade_indices'])
```

### 2. Sampling phase (during `generate_batch`):
```python
# Coin-flip: trade-biased or uniform? O(1) per sample.
if (ep_data['n_trade_steps'] > 0
    and self.rng.random() < config.trade_sample_prob):
    # Sample from trade-heavy timesteps
    offset = self.rng.choice(ep_data['trade_indices'])
else:
    # Uniform sampling (all timesteps, including holds)
    offset = self.rng.integers(0, n_available)

# Clamp to valid window range
offset = max(offset, config.window_size)
offset = min(offset, ep_data['n_samples'] - 1)
```

### 3. Config (`easy_config.py`):
```python
trade_sample_prob: float = 0.15  # P(sample from trade indices). Start conservative.
# 0.15 = ~15% of windows are trade-biased, ~85% uniform
# Combined with TASK_679 threshold: ~3-5× more trade signal vs baseline
```

### Expected Impact

| Metric | Uniform | trade_sample_prob=0.15 | trade_sample_prob=0.30 |
|--------|---------|----------------------|----------------------|
| Trade windows per batch | ~5% | ~15-20% | ~30-35% |
| Effective dataset diversity | Full | ~90% (slight reduction) | ~75% (watch overfit) |
| agree_hold risk | None | Low | Medium (monitor) |
| Sampling overhead | O(1) | O(1) | O(1) |

### What This Does NOT Change

- Oracle quality — same labels, same soft targets
- Gradient magnitude — TASK_677 trade weighting still active
- Curriculum levels — all levels still sampled
- Buffer diversity — all episodes still used, just different offset distribution

---

## Testing Requirements

**Test every function immediately after implementation.**

### Test 1: Trade index computation
```python
# Verify trade_indices are computed correctly
ep = generator.buffer[0]
actions = ep['actions'].reshape(-1, 64, 5)
trade_density = 1.0 - actions[:, :, 1].mean(axis=1)
expected_indices = np.where(trade_density > 0.01)[0]
assert np.array_equal(ep['trade_indices'], expected_indices)
assert ep['n_trade_steps'] == len(expected_indices)
print(f"Trade steps: {ep['n_trade_steps']}/{len(trade_density)} "
      f"({100*ep['n_trade_steps']/len(trade_density):.1f}%)")
```

### Test 2: Coin-flip sampling distribution
```python
# Verify biased sampling produces expected trade hit rate
offsets = []
for _ in range(10000):
    if len(ep['trade_indices']) > 0 and rng.random() < 0.15:
        offsets.append(rng.choice(ep['trade_indices']))
    else:
        offsets.append(rng.integers(0, ep['n_samples']))

trade_set = set(ep['trade_indices'])
hit_rate = np.mean([o in trade_set for o in offsets])
uniform_rate = len(trade_set) / ep['n_samples']
print(f"Trade hit rate: {hit_rate:.3f} (uniform would be {uniform_rate:.3f})")
assert hit_rate > uniform_rate * 2.0, "Bias not working"
assert hit_rate < 0.5, "Bias too aggressive (should be ~15-25%)"
```

### Test 3: Window alignment verification
```python
# Verify offset maps to observation window END, not start
# The observation for offset=t should be prices[t-W:t], label should be actions[t]
batch = generator.generate_batch(batch_size=8)
# Check that observations correspond to the correct time range
# (implementation-specific — verify the slice logic)
```

### Test 4: Hold accuracy guard
```python
# After 10 epochs with biased sampling:
assert metrics['agree_hold'] > 0.85, \
    f"Hold accuracy dropped to {metrics['agree_hold']:.3f} — reduce trade_sample_prob"
```

### Test 5: Overfit detection (train vs val gap)
```python
# After 50 epochs, train-val gap should be < 10 percentage points
gap = train_agreement - val_agreement
assert gap < 0.10, \
    f"Train-val gap {gap:.3f} suggests overfitting on trade patterns. "
    f"Increase buffer_size or reduce trade_sample_prob."
```

### Test 6: Sampling speed benchmark
```python
# Biased sampling should not slow down batch generation
import time
t0 = time.time()
for _ in range(100):
    generator.generate_batch(batch_size=256)
t_biased = (time.time() - t0) / 100

generator.config.trade_sample_prob = 0.0  # Disable
t0 = time.time()
for _ in range(100):
    generator.generate_batch(batch_size=256)
t_uniform = (time.time() - t0) / 100
generator.config.trade_sample_prob = 0.15  # Re-enable

overhead = (t_biased - t_uniform) / t_uniform * 100
print(f"Biased: {t_biased*1000:.1f}ms, Uniform: {t_uniform*1000:.1f}ms, Overhead: {overhead:.1f}%")
assert overhead < 5.0, f"Sampling overhead {overhead:.1f}% too high (coin-flip should be ~0%)"
```

---

## Files Modified

- `train_easy.py` — `OnTheFlyGenerator`: trade index pre-computation + coin-flip sampling
- `easy_config.py` — `trade_sample_prob: float = 0.15`

---

## Status

- [x] Implement trade index pre-computation in `_refill_buffer` (`train_easy.py:550-557`)
- [x] Implement coin-flip sampling in `generate_batch` (`train_easy.py:643-652`)
- [x] Add `trade_sample_prob` to `EasyConfig` (`easy_config.py:73-75`)
- [ ] Test 1: Index computation
- [ ] Test 2: Sampling distribution
- [ ] Test 3: Window alignment
- [ ] Test 4: Hold accuracy guard (after overnight run)
- [ ] Test 5: Overfit detection (after overnight run)
- [ ] Test 6: Sampling speed
- [ ] A/B comparison: biased vs uniform (10 epoch mini-run)

# TASK_715: Direct-Sample Pipeline — Eliminate Episode/Buffer Overhead

**Status**: DONE
**Priority**: 0 (CRITICAL — current pipeline spends 88% of time on overhead, 4% on training)
**Depends on**: TASK_714 (precomputed oracle)

---

## Problem

The current data pipeline was designed for **synthetic data** where episodes are cheap to generate (SyntheticMarket → encode → oracle → discard). With TASK_714's precomputed oracle on full historical periods, the episode generation is now the bottleneck:

### Measured Breakdown (per epoch, 400 training steps)

| Component | Time | % | What it does |
|-----------|------|---|-------------|
| AR rollout | 1020s | 55% | 2,040 episodes × 0.5s sequential lax.scan (400 model forward passes each) |
| Encoding | 612s | 33% | 2,040 episodes × 0.3s jit_batch_encode_temporal (400 windows each) |
| I/O | 142s | 8% | 3.2 segment loads × 45s (DB + fill + oracle) |
| **Training** | **80s** | **4%** | **400 steps × 0.2s actual gradient computation** |

**96% of training time is NOT training.** The GPU sits idle while CPU/Python builds episodes that get sampled 10 times then discarded.

### Why This Happens

```
generate_batch(1024)
  → picks episode from buffer
  → takes 20 random samples from 400-sample episode
  → after 10 visits (200 samples), episode marked exhausted
  → _refill_buffer() called
    → get_episode_range(400) → 400 timesteps from pool
    → _build_historical_episode():
      → jit_batch_encode_temporal(400 windows)      ← 0.3s
      → _build_model_state_trajectory(400 steps)     ← 0.5s (AR rollout)
      → optimal_path_to_policy(400 steps)             ← fast
      → build_state_channels(400 steps)               ← fast
    → store in buffer
  → 2,040 episodes built and discarded per epoch
```

Each of the 2,040 episodes provides only 200 training samples (10 visits × 20 samples). The buffer churns through 16 episodes, refills 128 times per epoch.

### Why the Old Pipeline Was Different

With synthetic data (pre-TASK_714):
- Episode generation was CHEAP (~20ms: generate prices + oracle on 489 timestamps)
- Buffer churn was fine — regeneration was the whole point (fresh random data)
- AR rollout was the same cost but offset by cheap data generation

With precomputed oracle (TASK_714):
- Oracle is FREE (already computed, just index into array)
- Encoding is UNNECESSARY as a bulk operation (can be done per-batch)
- AR rollout is the DOMINANT cost but doesn't benefit from episode abstraction

---

## Solution: Direct Sampling from Precomputed Slots

**Eliminate the episode/buffer abstraction for historical data.** Each training step samples directly from the pool's precomputed oracle data.

### Architecture

```
generate_batch(1024):
  1. Pick 1024 random (slot, cursor) positions across all active slots
  2. Encode those 1024 windows: jit_batch_encode_temporal  ← ~0.8s total
  3. Look up oracle targets at those positions              ← ~0s (numpy index)
  4. Build state channels from precomputed oracle           ← ~0.1s (vectorized)
  5. Return batch dict                                      ← same format as before

Per step: ~1.0s
Per epoch (400 steps): ~400s
```

### No AR Rollout Per-Batch

This is the key insight. The AR rollout (`_build_model_state_trajectory`) runs the model sequentially through an episode to build state channels from the model's OWN actions. This is necessary for training the model to handle its own state.

BUT: the AR rollout cost is 0.5s per 400-step episode = 1.25ms per timestep. For 1024 random timesteps per batch, that would be 1.28s — same cost. The issue is that AR is inherently sequential (step N depends on step N-1), so random sampling can't use it.

**Two options for state channels:**

**Option A: Oracle state channels only (no AR)**
- Build state channels from precomputed oracle selections (already available)
- Fast: vectorized numpy, ~0.1s for 1024 windows
- Matches the oracle state path that was used for the first ~5000 epochs
- Loses the AR training signal (model doesn't see its own mistakes)
- But: the model is already at 91% agreement with AR=100% — the AR training benefit may be exhausted

**Option B: Periodic AR episodes (hybrid)**
- Most batches use oracle state channels (fast, direct sampling)
- Every N steps, run ONE full AR episode (400 steps) to keep the model's state-awareness fresh
- AR fraction: e.g., 1 in 20 steps → AR cost = 400/20 × 0.5 = 10s/epoch (down from 1020s)
- Buffer only needed for AR episodes (16 slots, refreshed rarely)

**Recommendation: Option B (hybrid) from the start.**

The model trained 2000+ epochs at AR=100%. Suddenly removing AR entirely is risky — the model learned to handle its own state trajectory and dropping that signal could cause regression. Hybrid approach:

- **95% of batches**: oracle state channels (fast, direct sampling)
- **5% of batches**: full AR episode (400-step rollout, keeps state-awareness training signal)
- AR cost: 400 steps × 0.05 × 2040 ≈ 102 episodes/epoch × 0.5s = ~51s/epoch (down from 1020s)
- `ar_direct_fraction` config param (default 0.05, adjustable)
- Buffer kept small (4 slots) for AR episodes only — refreshed via `_build_historical_episode`

### Encoding Optimization

With direct sampling, encoding 1024 windows per step:
- `jit_batch_encode_temporal(prices, window_starts, config)` on 1024 windows
- Each window needs prices from `[cursor : cursor + ws]` in the slot's OHLCV data
- Challenge: windows come from different slots with different price arrays
- Solution: group by slot, encode per-slot batches, concatenate

```python
# Group samples by slot
for slot_idx, cursors in samples_by_slot.items():
    slot = pool.slots[slot_idx]
    prices_jax = jnp.array(slot['hm'].cache_data)  # Already in RAM
    starts = jnp.array(cursors)
    obs = jit_batch_encode_temporal(prices_jax, starts, encoder_config)
    # obs: (len(cursors), 27, ws, n_assets)
```

**VRAM concern**: `prices_jax` for a full slot is 258k × 64 × 5 × 4 bytes = 330MB on GPU. With 17 slots, we can't have all on GPU simultaneously. Solution: transfer one slot's prices to GPU, encode its batch, move to next slot. ~50ms per slot transfer. With ~5 slots per batch (1024/~200 per slot): ~250ms total.

Actually better: **cache the prices on GPU for the most-used slots.** Since round-robin is deterministic, the current slot's prices can be kept on GPU. LRU cache of 2-3 slots = ~1GB VRAM.

### Sampling Strategy

Each training step needs 1024 (slot, cursor) pairs:
1. Round-robin across slots for regime diversity (same as current pool)
2. Within each slot, advance a sequential cursor (stride=1, matching dojo_v3)
3. **Temporal jitter**: every N steps, add random offset +0 to +5 to cursor. Prevents the model from memorizing exact cursor positions across epochs. Small enough to stay within the same market context.
4. This gives each step a mix of all 17 regimes with sequential walk through each

```python
def sample_positions(self, n_samples, window_size, jitter_interval=50):
    """Get n_samples (slot, cursor) positions, round-robin across slots."""
    positions = []  # List of (slot_idx, cursor)
    per_slot = n_samples // len(self.slots) + 1
    for slot_idx, slot in enumerate(self.slots):
        for _ in range(per_slot):
            if len(positions) >= n_samples:
                break
            positions.append((slot_idx, slot['cursor']))
            slot['cursor'] += 1
            # Temporal jitter: small random skip every jitter_interval steps
            slot['_step_count'] = slot.get('_step_count', 0) + 1
            if slot['_step_count'] % jitter_interval == 0:
                slot['cursor'] += np.random.randint(0, 6)
            # Rotate if needed
            if slot['cursor'] + window_size > len(slot['returns']):
                self._rotate_slot(slot_idx)
    return positions[:n_samples]
```

### Policy Target Computation

With precomputed oracle selections, policy targets need to be computed per-sample:
- `optimal_path_to_policy` converts selections + returns → action targets
- Needs `prev_selections` for state-aware oracle (position at t-1)
- For sequential samples, prev_selections comes from `oracle_selections[decision_idx - 1]`
- This is a vectorized operation on precomputed data — fast

### State Channels: Precompute Per Slot

**Precompute ALL state channels for the full period during `_make_slot()`.** Cost: ~1s for 258k steps via `build_state_channels` (already JIT'd).

RAM cost per slot: 258k × 6 × 64 × 4 bytes = **400MB in float32**.

**Optimization: store in float16.** State channels are normalized metrics (position ∈ {-1,0,1}, PnL ∈ [-1,1], hold_dur ∈ [0,1], heat ∈ [0,1], capacity ∈ [0,1]). Float16 precision is more than sufficient. **400MB → 200MB per slot. 17 slots = 3.4GB.** Fits in 45GB RAM alongside the other slot data (~8.7GB HM+oracle).

Total RAM with precomputed state: ~8.7GB (HM+oracle) + ~3.4GB (state_channels fp16) + ~5GB (other) = ~17GB. Comfortable.

```python
# In _make_slot(), after oracle precomputation:
target_pos = compute_target_position(
    jnp.array(oracle_sel), jnp.array(returns)
)
state_channels = build_state_channels(
    target_pos, jnp.array(returns),
    n_assets=self.n_assets, k=oracle_k, window_size=ws,
)
# Transfer to numpy float16 for RAM storage
slot['state_channels'] = np.array(state_channels).astype(np.float16)  # (T, 6, n_assets)
```

At training time, just index:
```python
# For window at cursor c, decision at c + ws - 1:
# Use state at t-1 (avoid leaking current oracle action)
state_at_window = slot['state_channels'][c + ws - 2]  # (6, n_assets) float16
# Tile across window and upcast to float32 for concatenation
state_tiled = np.tile(state_at_window[:, None, :], (1, ws, 1)).astype(np.float32)
```

**Confidence channel**: precompute from oracle policy targets (max softmax per asset). Store alongside state_channels or compute on-the-fly (cheap).

---

## Changes

### New: `generate_batch_direct()` method on OnTheFlyGenerator

Replaces the buffer-based `generate_batch()` for the historical pool path:

```python
def generate_batch_direct(self, batch_size):
    """TASK_715: Direct sampling from precomputed pool — no buffer overhead."""
    # 1. Get positions (round-robin across slots, sequential within)
    positions = self.segment_pool.sample_positions(batch_size, self.config.window_size)

    # 2. Group by slot, encode + gather targets per slot
    all_obs, all_actions, all_returns, all_par = [], [], [], []
    for slot_idx, cursors in group_by_slot(positions):
        slot = self.segment_pool.slots[slot_idx]
        # Encode
        obs = self._encode_windows(slot, cursors)
        # Oracle targets
        actions, returns, par = self._gather_oracle_targets(slot, cursors)
        # State channels (oracle, not AR)
        state_ch = self._build_oracle_state_for_windows(slot, cursors)
        # Concatenate state to observations
        obs = jnp.concatenate([obs, state_ch], axis=1)

        all_obs.append(obs)
        all_actions.append(actions)
        all_returns.append(returns)
        all_par.append(par)

    return {
        'observations': jnp.concatenate(all_obs)[:batch_size],
        'actions': jnp.concatenate(all_actions)[:batch_size],
        'returns': jnp.concatenate(all_returns)[:batch_size],
        'per_asset_returns': jnp.concatenate(all_par)[:batch_size],
    }
```

### Modified: `train_epoch()`

Use `generate_batch_direct()` when pool is active:

```python
if generator.segment_pool is not None:
    batch = generator.generate_batch_direct(config.batch_size)
else:
    batch = generator.generate_batch(config.batch_size)
```

### Kept as-is

- `generate_batch()` — kept for synthetic/Bella/legacy paths
- `_refill_buffer()` — kept for non-pool paths and AR episode refresh
- `_build_historical_episode()` — kept for 5% AR episodes (Option B)
- `easy_trainer.py` — no changes (same batch dict format)
- Checkpoints — fully compatible

---

## Expected Performance

| Metric | Current | After TASK_715 |
|--------|---------|---------------|
| Time per step | ~4.6s | **~1.0s** |
| Time per epoch (400 steps) | ~1850s (31 min) | **~400s (6.7 min)** |
| GPU utilization | ~4% | **~20%** |
| AR rollout cost | 55% of time (1020s) | **~3%** (~51s, 5% hybrid fraction) |
| Encoding cost | 33% of time (612s) | **~75%** (~300s, but direct per-step) |
| I/O cost | 8% of time (142s) | **~0%** (slots stay in RAM) |
| Training cost | 4% (80s) | **~20%** (80s, same but larger fraction) |
| State channel cost | included in AR | **~2%** (precomputed, just indexing) |

Training throughput: **~4x faster** (conservative with Option B hybrid).

Epochs per day: ~210 (current) → ~840 (after). One full data pass (~34,000 epochs at stride=1) in ~40 days instead of ~162 days.

---

## Risk

1. **AR regression**: Mitigated by Option B — 5% AR fraction maintains state-awareness signal. `ar_direct_fraction` config param allows tuning (0.05 default, can increase to 0.10-0.20 if benchmark regresses). Monitor agree_trade per benchmark.

2. **Encoding VRAM**: 1024 windows × (27, 64, 64) × 4 bytes = 450MB. Plus prices on GPU (~330MB per slot). Total ~1.5GB per step — fits in 24GB.

3. **State channel precomputation RAM**: 17 slots × 200MB (float16) = 3.4GB. Total pool RAM ~17GB (HM + oracle + state). Fits in 45GB with margin.

4. **Temporal jitter edge cases**: Jitter could push cursor past segment end. Clamp to valid range before use. Small jitter (+0 to +5) is negligible relative to 258k segment length.

5. **JIT recompilation**: `jit_batch_encode_temporal` may recompile if batch sizes vary per-slot (different number of cursors per slot). Mitigation: pad batches to fixed size per slot, or encode all slots with same batch size and mask.

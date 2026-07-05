# TASK_714: Fix Data Pipeline — Full-Period Oracle + Stride-1 Walk

**Status**: DONE
**Priority**: 0 (CRITICAL — current pipeline wastes compute with inconsistent oracle targets)
**Blocks**: Phase 2 transition (model trained on bad targets can't be trusted)

---

## Problem

The Foxglove data pipeline computes oracle on 489-timestamp CHUNKS instead of the full period. This causes:

1. **Oracle inconsistency**: Timestamp 300 in chunk [0:489] gets different oracle targets than the same timestamp 300 in chunk [244:733], because the lookahead context differs. The model sees contradictory labels for the same market state.

2. **Wasted compute**: 7700+ epochs of training on inconsistent targets. Two months of GPU time producing a model that learned conflicting signals.

### How it should work (dojo_v3 / Poppy pipeline)

```
training_schedule.json → 18 periods × 270 runs
For each run:
  1. Load full period (258k timestamps) into RAM
  2. Compute oracle ONCE on full return series → selections (258k, 64)
  3. Walk through ALL timestamps with stride=1
  4. Each training step: encode window at cursor, index into precomputed oracle
  5. One epoch = 1440 steps (one day of 1-min candles)
  6. 180-day period = ~180 epochs per run
```

### How it currently works (broken)

```
SegmentPool.get_window() → 489-timestamp chunk
_generate_batch_episodes():
  1. Compute oracle on 489-timestamp chunk (WRONG — limited context)
  2. Extract ~424 windows from chunk
  3. Discard chunk, get next one (stride=244)
  → Oracle targets CONTRADICT across overlapping chunks
  → Model learns averaged noise instead of consistent signal
```

### Why this explains the benchmark gap

Training shows 75% trade agreement (model matches oracle within each chunk). But benchmark runs oracle on the FULL period with consistent context. The model learned chunk-local patterns that don't match full-period oracle decisions → 12-33% trade agreement in benchmark.

TASK_713 (return weighting) improved agree_trade to 41% by focusing on high-value trades, but the underlying target inconsistency limits further progress.

---

## Solution: Full-Period Oracle Precomputation

### Phase A: Oracle Precomputation per Segment

When a SegmentPool slot loads a new segment:

```python
# In SegmentPool._make_slot() or new _precompute_oracle():
hm = HistoricalMarket(...)  # Full period (258k timestamps)
prices = hm.cache_data      # (258k, 64, 5) OHLCV
close = prices[:, :, 3]
returns = np.log(close[1:] / np.maximum(close[:-1], 1e-8))  # (258k-1, 64)

# Compute oracle ONCE for full period
oracle_result = greedy_lookahead_path(
    jnp.array(returns),
    k=config.oracle_k,
    lookahead=config.oracle_lookahead,
    transaction_cost=config.transaction_cost,
    min_alpha_threshold=config.min_alpha_threshold,
)
# Store alongside HM in slot:
slot['oracle_selections'] = np.array(oracle_result.selections)   # (258k-1, 64)
slot['oracle_cum_pnl'] = np.array(oracle_result.cumulative_pnl)  # (258k-1,)
slot['returns'] = np.array(returns)                                # (258k-1, 64)
```

**RAM cost**: ~258k × 64 × 3 arrays × 4 bytes ≈ 200MB per slot. 17 slots ≈ 3.4GB extra. Total pool ~8.4GB. Fits in 45GB RAM.

**GPU cost**: One `greedy_lookahead_path` call per segment load (~1-2 seconds for 258k timesteps). Amortized over ~258k training samples from that segment.

### Phase B: Stride-1 Window Walking

Replace chunk-based get_window() with cursor-based indexing:

```python
# SegmentPool.get_training_sample(self) → dict
slot = self.slots[self._round_robin_idx % len(self.slots)]
cursor = slot['cursor']  # Current timestep position

# Window start for encoder
window_start = cursor  # encoder covers prices[cursor:cursor+ws]

# Oracle target at decision point (cursor + ws - 1)
decision_idx = cursor + ws - 1
oracle_sel = slot['oracle_selections'][decision_idx]     # (64,)
oracle_ret = slot['returns'][decision_idx]                # (64,)
remaining_pnl = slot['oracle_cum_pnl'][-1] - slot['oracle_cum_pnl'][decision_idx]

# Advance cursor by 1 (stride=1, every minute)
slot['cursor'] += 1
```

### Phase C: Batch Assembly in _refill_buffer

Instead of calling `_generate_batch_episodes` (which recomputes oracle), assemble batches directly from precomputed data:

```python
def _refill_buffer(self):
    # Collect N window_starts from pool (stride=1, round-robin across slots)
    window_starts_per_slot = defaultdict(list)
    for _ in range(n_to_refill):
        slot_idx, cursor = self.segment_pool.get_next_cursor()
        window_starts_per_slot[slot_idx].append(cursor)

    # Per slot: batch-encode observations, index into precomputed oracle
    for slot_idx, cursors in window_starts_per_slot.items():
        slot = self.segment_pool.slots[slot_idx]
        prices_jax = jnp.array(slot['hm'].cache_data)
        starts = jnp.array(cursors)

        # Encode observations (batched, same as current)
        observations = jit_batch_encode_temporal(prices_jax, starts, encoder_config)

        # Index into precomputed oracle (no recomputation!)
        decision_idxs = np.array(cursors) + ws - 1
        oracle_targets = precompute_policy_targets(
            slot['oracle_selections'][decision_idxs],
            slot['returns'][decision_idxs],
            # ... prev_selections for state-aware oracle
        )
        returns = slot['oracle_cum_pnl'][-1] - slot['oracle_cum_pnl'][decision_idxs]
        per_asset_returns = slot['returns'][decision_idxs]  # TASK_713
```

### Phase D: State Channels (AR Rollout)

AR rollout (`_build_model_state_trajectory`) needs the full pre-encoded market observations for the episode. With stride=1, an "episode" is a contiguous walk through N timesteps.

Options:
1. **Keep current episode-based AR**: Load a contiguous chunk of ~400 timesteps, run AR rollout, use precomputed oracle for targets. Episode length stays at `config.episode_length`.
2. **Full-period AR rollout**: Run AR through entire 258k timesteps. Too expensive for JIT scan (~258k sequential model forward passes).

**Recommendation**: Option 1. Episodes of 400 timesteps from stride-1 cursor positions. AR rollout stays the same, but oracle targets come from the precomputed full-period oracle instead of per-chunk oracle.

---

## Changes

### SegmentPool (train_easy.py)

- `_make_slot()`: After loading HM, compute oracle on full period. Store `oracle_selections`, `oracle_cum_pnl`, `returns` in slot dict.
- `get_window()` → `get_episode_range()`: Returns `(slot_idx, start_cursor, end_cursor)` for a contiguous range of ~episode_length timesteps.
- Cursor advances by `episode_length` per call (not by `window_length/2`).
- Stride within episode = 1 (every timestep is a training sample).

### OnTheFlyGenerator._refill_buffer (train_easy.py)

- Historical branch: Get episode range from pool, encode observations, index into precomputed oracle.
- Remove `_generate_batch_episodes` call for historical data (oracle no longer computed per-chunk).
- Keep `_generate_batch_episodes` for synthetic/Bella paths (no change).

### Policy Target Computation

- `optimal_path_to_policy` still needed for state-aware action assignment (Buy/Sell/Hold from oracle selections + prev_position).
- But oracle SELECTIONS are precomputed — only the action mapping is done per-window.
- This is fast (vectorized, no lookahead scan).

### What stays the same

- `_generate_batch_episodes`: Kept for synthetic/Bella data paths
- `_build_model_state_trajectory`: AR rollout unchanged (still per-episode)
- `easy_trainer.py`: Loss function unchanged
- Encoder: Same `jit_batch_encode_temporal`
- Checkpoints: Fully compatible (no model changes)
- `training_schedule.json`: Same format, same data

---

## Numbers

| Metric | Current (broken) | Fixed |
|--------|-----------------|-------|
| Oracle context | 489 timestamps | **258,000 timestamps** |
| Unique training samples per run | ~1,057 windows | **~258,000 timesteps** |
| Epochs for 1 full pass (270 runs) | 139 | **~34,000** |
| Oracle consistency | Contradictory across chunks | **Consistent across full period** |
| RAM per slot | ~316 MB (HM only) | **~516 MB (HM + oracle)** |
| Total pool RAM (17 slots) | ~5.0 GB | **~8.7 GB** |
| Oracle compute per slot load | Every 489-ts chunk (~many/epoch) | **Once per segment load (~2s)** |

---

## Execution Plan

1. **Implement oracle precomputation in `_make_slot()`** — compute once, store in slot
2. **New `get_episode_range()` method** — returns contiguous cursor range
3. **New historical refill path** — encode + index into precomputed oracle
4. **Update state channel / AR rollout** — use precomputed oracle for DAgger targets
5. **Test**: Compare oracle targets at overlapping timestamps — should be identical now
6. **Verify**: Training agreement should INCREASE (no contradictory targets)
7. **Benchmark**: Trade agreement should improve significantly (consistent with benchmark oracle)

---

## Risk

- Oracle on 258k timesteps may need chunked GPU computation (VRAM limit)
  - Mitigation: `greedy_lookahead_path` already handles arbitrary lengths via `max_len` param
  - Fallback: compute in 50k-step chunks with overlapping boundaries

- Full-period oracle may produce different selections than chunk oracle
  - This is the POINT — full-period oracle matches benchmark oracle
  - Training metrics may temporarily dip as model adapts to consistent targets
  - Should recover and exceed previous levels within a few hundred epochs

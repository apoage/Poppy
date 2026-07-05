# TASK_721: Async Pipeline + Random Sampling + Vectorization

**Status**: DONE
**Priority**: HIGH — model close to gate (3/6 alpha), needs faster iteration
**Replaces**: TASK_720 (random sampling — folded into this refactor)
**File**: `train_phase2.py` — replace `ConsecutivePairGenerator` with `AsyncBatchPipeline`

---

## Problem

Current pipeline (`ConsecutivePairGenerator.generate_batch()`, line ~359) is:
1. Fully sequential: gen(41s) → mcts(36s) → train(7s) = 85s/epoch. GPU idle 48%.
2. Sequential cursor sampling (stride=1) — 75% overlap between consecutive windows
3. Python for-loop for n-step rewards (20 iterations per slot)
4. JAX ops (`jnp.array`, `jit_batch_encode_temporal`) interleaved with numpy — blocks async

## Solution: AsyncBatchPipeline

Replace `ConsecutivePairGenerator` with a two-phase pipeline:

```python
class AsyncBatchPipeline:
    """Phase 1 (CPU/numpy, background thread) → Phase 2 (JAX/GPU, main thread)."""
    
    def __init__(self, pool, config: Phase2Config, prefetch: int = 2):
        ...
    
    def _prepare_numpy(self, batch_size: int) -> dict:
        """CPU only. Returns dict of numpy arrays. Thread-safe."""
        ...
    
    def _encode_jax(self, np_batch: dict) -> dict:
        """GPU. Converts numpy→JAX, runs encoder. Main thread only."""
        ...
    
    def _worker(self):
        """Background thread. Calls _prepare_numpy, queues results."""
        ...
    
    def get_batch(self, batch_size: int) -> dict:
        """Main thread. Dequeues numpy batch, encodes to JAX."""
        ...
```

### Phase 1: `_prepare_numpy(batch_size)` — Background Thread

**CRITICAL: No JAX operations. No `jnp.array()`, no `jit` calls, no `jax.*`.**

```python
def _prepare_numpy(self, batch_size):
    pool = self.pool
    ws = self.ws
    n_assets = self.n_assets
    nstep = self.config.nstep_horizon

    # 1. RANDOM SAMPLING (replaces sequential cursor from TASK_720)
    #    Uniform random positions within each slot's valid range.
    #    Round-robin across slots for slot-level balance.
    n_slots = len(pool.slots)
    per_slot = batch_size // n_slots + 1
    positions = []
    for i in range(n_slots):
        slot_idx = (self._rr_idx + i) % n_slots
        slot = pool.slots[slot_idx]
        T_ret = len(slot['returns'])
        T_prices = len(slot['hm'].cache_data)
        max_cursor = min(T_ret - ws - nstep, T_prices - ws - nstep)
        if max_cursor <= 0:
            continue
        cursors = self._rng.integers(0, max_cursor, size=min(per_slot, batch_size - len(positions)))
        for c in cursors:
            positions.append((slot_idx, int(c)))
        if len(positions) >= batch_size:
            break
    self._rr_idx += n_slots

    # 2. Group by slot (for efficient array slicing)
    by_slot = defaultdict(list)
    for slot_idx, cursor in positions[:batch_size]:
        by_slot[slot_idx].append(cursor)

    # 3. Per-slot: extract numpy data (NO JAX)
    all_prices = []      # list of (n, ws+nstep, n_assets, ...) numpy
    all_starts_t = []    # cursor offsets within price_slice
    all_starts_t1 = []
    all_starts_tN = []
    all_actions = []
    all_rewards = []
    all_oracle_targets = []
    all_oracle_values = []
    all_nstep_rewards = []
    # State channels (if enabled)
    all_state_t = []
    all_conf_t = []
    all_state_t1 = []
    all_conf_t1 = []
    all_state_tN = []
    all_conf_tN = []

    for slot_idx, cursors in by_slot.items():
        slot = pool.slots[slot_idx]
        cursors_arr = np.array(cursors)
        n = len(cursors_arr)

        # Price slice (numpy) — covers [min_cursor, max_cursor + nstep + ws)
        c_min, c_max = int(cursors_arr.min()), int(cursors_arr.max())
        price_slice = slot['hm'].cache_data[c_min:c_max + nstep + ws]  # numpy
        all_prices.append(price_slice)
        all_starts_t.append(cursors_arr - c_min)
        all_starts_t1.append(cursors_arr - c_min + 1)
        all_starts_tN.append(cursors_arr - c_min + nstep)

        decision_indices = cursors_arr + ws - 1

        # Oracle action (numpy)
        policy_targets = slot['policy_targets'][decision_indices].astype(np.float32)
        actions = np.argmax(policy_targets, axis=-1)
        all_actions.append(actions)
        all_oracle_targets.append(policy_targets)

        # Per-step reward (numpy)
        oracle_sel = slot['oracle_selections'][decision_indices]
        step_returns = slot['returns'][decision_indices]
        k_count = np.maximum(np.sum(oracle_sel, axis=-1), 1.0)
        step_pnl = np.sum(np.abs(step_returns) * oracle_sel, axis=-1) / k_count
        prev_indices = np.maximum(decision_indices - 1, 0)
        prev_sel = slot['oracle_selections'][prev_indices]
        changes = np.sum(np.abs(oracle_sel - prev_sel), axis=-1)
        step_pnl = step_pnl - changes * self.config.transaction_cost
        rewards = np.clip(step_pnl * REWARD_SCALE, -10.0, 10.0).astype(np.float32)
        all_rewards.append(rewards)

        # Oracle remaining PnL (numpy)
        oracle_cum_pnl = slot['oracle_cum_pnl']
        horizon = self.config.oracle_value_horizon
        end_idx = np.minimum(decision_indices + horizon, len(oracle_cum_pnl) - 1)
        remaining = (oracle_cum_pnl[end_idx] - oracle_cum_pnl[decision_indices]).astype(np.float32)
        all_oracle_values.append(remaining)

        # VECTORIZED n-step rewards (numpy, no loop)
        steps = decision_indices[:, None] + np.arange(nstep)  # (n, N)
        steps = np.minimum(steps, len(oracle_cum_pnl) - 2)
        per_step_r = (oracle_cum_pnl[steps + 1] - oracle_cum_pnl[steps]) * REWARD_SCALE
        per_step_r = np.clip(per_step_r, -10.0, 10.0)
        discounts = self.config.td_discount ** np.arange(nstep)
        nstep_sum = (per_step_r * discounts).sum(axis=1).astype(np.float32)
        all_nstep_rewards.append(nstep_sum)

        # State channels (numpy, if enabled)
        if self.use_state_channels:
            decision_t1 = cursors_arr + ws
            decision_tN = cursors_arr + ws - 1 + nstep
            state_idx_t = np.maximum(decision_indices - 1, 0)
            state_idx_t1 = np.maximum(decision_t1 - 1, 0)
            state_idx_tN = np.maximum(decision_tN - 1, 0)
            all_state_t.append(slot['state_channels'][state_idx_t].astype(np.float32))
            all_conf_t.append(slot['confidence'][decision_indices].astype(np.float32))
            all_state_t1.append(slot['state_channels'][state_idx_t1].astype(np.float32))
            all_conf_t1.append(slot['confidence'][decision_t1].astype(np.float32))
            all_state_tN.append(slot['state_channels'][state_idx_tN].astype(np.float32))
            all_conf_tN.append(slot['confidence'][decision_tN].astype(np.float32))

    # Return numpy dict — NO JAX
    return {
        'prices': all_prices,           # list of numpy arrays (variable shape per slot)
        'starts_t': all_starts_t,       # list of numpy (n,) per slot
        'starts_t1': all_starts_t1,
        'starts_tN': all_starts_tN,
        'actions': np.concatenate(all_actions)[:batch_size],
        'rewards': np.concatenate(all_rewards)[:batch_size],
        'oracle_targets': np.concatenate(all_oracle_targets)[:batch_size],
        'oracle_values': np.concatenate(all_oracle_values)[:batch_size],
        'nstep_rewards': np.concatenate(all_nstep_rewards)[:batch_size],
        # State channels (if enabled, else empty lists)
        'state_t': all_state_t, 'conf_t': all_conf_t,
        'state_t1': all_state_t1, 'conf_t1': all_conf_t1,
        'state_tN': all_state_tN, 'conf_tN': all_conf_tN,
        'batch_size': batch_size,
    }
```

### Phase 2: `_encode_jax(np_batch)` — Main Thread

```python
def _encode_jax(self, np_batch):
    batch_size = np_batch['batch_size']

    # Encode observations per price-slice group
    all_obs_t, all_obs_t1, all_obs_tN = [], [], []
    for prices, st_t, st_t1, st_tN in zip(
        np_batch['prices'], np_batch['starts_t'],
        np_batch['starts_t1'], np_batch['starts_tN']
    ):
        prices_jax = jnp.array(prices)
        obs_t = self._encode_fn(prices_jax, jnp.array(st_t), self.encoder_config)
        obs_t1 = self._encode_fn(prices_jax, jnp.array(st_t1), self.encoder_config)
        obs_tN = self._encode_fn(prices_jax, jnp.array(st_tN), self.encoder_config)
        all_obs_t.append(obs_t)
        all_obs_t1.append(obs_t1)
        all_obs_tN.append(obs_tN)

    obs_t = jnp.concatenate(all_obs_t)[:batch_size]
    obs_t1 = jnp.concatenate(all_obs_t1)[:batch_size]
    obs_tN = jnp.concatenate(all_obs_tN)[:batch_size]

    # Add state channels if enabled
    if self.use_state_channels:
        ws = self.ws
        for name, obs_list, state_list, conf_list in [
            ('t', all_obs_t, np_batch['state_t'], np_batch['conf_t']),
            ('t1', all_obs_t1, np_batch['state_t1'], np_batch['conf_t1']),
            ('tN', all_obs_tN, np_batch['state_tN'], np_batch['conf_tN']),
        ]:
            # Handled per-group, concatenated after
            pass
        # (See _add_state_channels pattern in current code line ~553)
        # state_ch: (n, 6, n_assets) → tile to (n, 6, ws, n_assets)
        # conf: (n, n_assets) → tile to (n, 1, ws, n_assets)
        # concatenate with obs along channel dim (axis=1)
        # TODO: implement following existing _add_state_channels pattern

    return {
        'obs_t': obs_t,
        'obs_t1': obs_t1,
        'obs_tN': obs_tN,
        'actions': jnp.array(np_batch['actions']),
        'rewards': jnp.array(np_batch['rewards']),
        'oracle_targets': jnp.array(np_batch['oracle_targets']),
        'oracle_values': jnp.array(np_batch['oracle_values']),
        'nstep_rewards': jnp.array(np_batch['nstep_rewards']),
    }
```

### Background Worker + Queue

```python
def __init__(self, pool, config, prefetch=2):
    from mcts_trader.env.jax_encoder import EncoderConfig, jit_batch_encode_temporal
    self.pool = pool
    self.config = config
    self.ws = config.window_size
    self.n_assets = config.n_assets
    self.encoder_config = EncoderConfig(n_assets=config.n_assets, window_size=config.window_size)
    self._encode_fn = jit_batch_encode_temporal
    self.use_state_channels = config.n_channels > 27
    self._rng = np.random.default_rng(config.seed + 3000)
    self._rr_idx = 0
    self._queue = queue.Queue(maxsize=prefetch)
    self._stop = threading.Event()
    self._batch_size = config.stage_b_batch_size
    self._thread = threading.Thread(target=self._worker, daemon=True)
    self._thread.start()

def _worker(self):
    while not self._stop.is_set():
        try:
            np_batch = self._prepare_numpy(self._batch_size)
            self._queue.put(np_batch, timeout=2.0)
        except queue.Full:
            continue
        except Exception as e:
            self._queue.put(e)
            break

def get_batch(self, batch_size=None):
    """Main thread: dequeue numpy → encode JAX."""
    item = self._queue.get()
    if isinstance(item, Exception):
        raise item
    return self._encode_jax(item)

def stop(self):
    self._stop.set()
    self._thread.join(timeout=5.0)
```

---

## Integration Points

### Training Loop (`train_stage_b`, line ~900)

Replace:
```python
gen = ConsecutivePairGenerator(pool, config)
# ...
batch = gen.generate_batch(batch_size)
```

With:
```python
pipeline = AsyncBatchPipeline(pool, config, prefetch=2)
# ...
batch = pipeline.get_batch()
# At end:
pipeline.stop()
```

### Slot Rotation

Slot rotation (`pool._rotate_slot`) does disk I/O and replaces slot data. With background thread reading slot data concurrently, this needs synchronization.

**Simplest approach**: rotate slots between epochs on the main thread, not during batch generation. Add to the epoch loop:
```python
# Between epochs — safe, no background reads happening
for i, slot in enumerate(pool.slots):
    if slot['cursor'] >= slot['max_cursor']:
        pool._rotate_slot(i)
```

Or: just rely on the existing rotation inside `sample_positions`. Since `_prepare_numpy` does its own random sampling (doesn't call `sample_positions`), rotation needs to be triggered separately. Check cursor progress or use a counter.

**Alternative**: Don't rotate during async operation. Pre-load enough diversity at startup (`max_runs_per_group=10` already provides this). Rotation adds complexity with threads.

---

## Existing Code Reference

### Current `ConsecutivePairGenerator` — `train_phase2.py:359-551`
- `__init__`: stores pool, config, encoder config, encode fn
- `generate_batch(batch_size)`: returns dict with keys:
  - `obs_t`: (B, C, W, N) jax — observations at cursor t
  - `obs_t1`: (B, C, W, N) jax — observations at cursor t+1
  - `obs_tN`: (B, C, W, N) jax — observations at cursor t+N
  - `actions`: (B,) jax int32 — oracle action index [0, 320)
  - `rewards`: (B,) jax float32 — per-step PnL × REWARD_SCALE, clipped [-10, 10]
  - `oracle_targets`: (B, 320) jax float32 — oracle policy distribution
  - `oracle_values`: (B,) jax float32 — oracle remaining PnL over horizon
  - `nstep_rewards`: (B,) jax float32 — Σ γ^k r_k discounted n-step reward sum
- `_add_state_channels(obs, slot, decision_indices)`: adds 7 extra channels (6 state + 1 conf)

### SegmentPool Slot Structure — `train_easy.py:640-655`
```python
{
    'hm': HistoricalMarket,              # .cache_data = (T, n_assets, 5) numpy OHLCV
    'cursor': int,                        # current sequential position
    'max_cursor': int,                    # T - window_length
    'group_key': (start_date, end_date),
    'run_idx': int,
    'label': str,
    'oracle_selections': np.array,        # (T-1, n_assets) binary
    'oracle_cum_pnl': np.array,           # (T-1,) float64
    'returns': np.array,                  # (T-1, n_assets) float64
    'state_channels': np.array,           # (T-1, 6, n_assets) float16
    'policy_targets': np.array,           # (T-1, n_assets*5) float16
    'confidence': np.array,               # (T-1, n_assets) float16
}
```

### Encoder — `mcts_trader/env/jax_encoder.py:483`
```python
def jit_batch_encode_temporal(
    prices: jax.Array,          # (n_steps, n_assets, 5) OHLCV
    window_starts: jax.Array,   # (batch,) int
    config: EncoderConfig,
) -> jax.Array:                 # (batch, n_channels, window_size, n_assets)
```

### JAX Safety Rules (from CLAUDE.md + session history)
- **NEVER** call `jnp.array()`, `jit`, or any `jax.*` from background threads
- Use `jax.block_until_ready(metrics)` after train step before next batch's CPU→GPU transfers
- `XLA_PYTHON_CLIENT_PREALLOCATE=true` + `MEM_FRACTION=0.85` (set at top of file)
- Do NOT use `cuda_malloc_async` — causes ILLEGAL_ADDRESS crashes

---

## Config Fields Used
- `config.window_size` (64)
- `config.n_assets` (64)
- `config.n_channels` (27 or 34 with state channels)
- `config.nstep_horizon` (20)
- `config.td_discount` (0.95)
- `config.oracle_value_horizon` (400)
- `config.transaction_cost` (0.001)
- `config.stage_b_batch_size` (256)
- `config.seed`
- `REWARD_SCALE` = 100.0 (module-level constant, line ~68)

## Expected Speedup

| Phase | Before | After |
|-------|--------|-------|
| gen (numpy) | 41s sequential | overlapped with GPU |
| gen (encode) | included above | ~15s on main thread |
| mcts | 36s | 36s (unchanged) |
| train | 7s | 7s (unchanged) |
| **Total** | **85s** | **~50-55s** |

## Acceptance Criteria

- [ ] `AsyncBatchPipeline` replaces `ConsecutivePairGenerator`
- [ ] Background thread does numpy-only work (no JAX)
- [ ] Random sampling within slots (uniform, not sequential cursor)
- [ ] Vectorized n-step rewards (no Python loop)
- [ ] Training loop updated to use `pipeline.get_batch()` / `pipeline.stop()`
- [ ] Epoch timing logged — gen time should drop from 41s to ~15s
- [ ] No JAX thread-safety crashes (test 50+ epochs)
- [ ] Batch output dict has same keys/shapes/dtypes as before
- [ ] `_add_state_channels` equivalent works in the new pipeline

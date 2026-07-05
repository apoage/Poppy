# TASK_689: Agent State Channels — Self-Aware Trading

**Status**: PHASE A DONE (2026-02-21), temporal leak fixed (`t` → `t-1`). Phase B → TASK_691. Phase C pending.
**Priority**: 1 (plateau breaker)
**Depends on**: None (standalone)
**Blocked by**: —

### Phase A Results

**Temporal leak found and fixed (2026-02-21):** State channels at time `t` leaked the oracle's action at `t`. Fixed by using `state[t-1]`. See `lore/notes_2026-02-21.md` for full analysis.

- Pre-fix (leaked): 98.5% agreement — model read answer from input
- Post-fix (t-1): dropped to 50%, recovered to **90% val** — honest but oracle-inflated
- 90% is real learning but won't transfer to inference (oracle state ≠ model state)
- Phase B (TASK_691: oracle entropy) needed to bridge train→inference gap

## Problem

The model is blind to its own state. It makes 64 independent per-asset decisions with zero knowledge of:
- What positions it currently holds
- How recently it traded each asset
- Whether open positions are profitable
- How many positions are open globally

This causes **7× over-trading**: oracle trades 3.9% of asset-steps, model trades 27.7%. The 25× adaptive trade weight in the loss pushes the model to never miss a trade, but without self-state feedback it can't learn to self-regulate. The result is a precision-recall deadlock where agree_trade↑ and agree_hold↓ exactly cancel, plateauing balanced accuracy at ~62%.

**Evidence (widened_v1, epoch 500):**
| Metric | Value |
|--------|-------|
| agree_trade | 48.5% (up from 14% at epoch 0) |
| agree_hold | 73.1% (down from 78.5% at epoch 0) |
| oracle trade rate | 3.9% |
| model trade rate | 27.7% |
| over-trading ratio | 7.1× |

Widening the encoder (TASK_683, 14.59M→25.71M params) did NOT break the plateau. The bottleneck is missing information, not capacity.

## Design Principles

1. **Oracle-independent**: State channels must work from the model's own actions at inference. Training must expose the model to imperfect state via corruption/noise — not pure teacher forcing.
2. **No magic numbers**: Multi-scale features (2 heat timescales) let the model learn thresholds from data.
3. **No phantom infrastructure**: No cash/margin/portfolio simulation — Easy Mode is pure supervised from oracle path solver. Only use information that actually exists.
4. **Cheap**: State corruption over scheduled sampling. Avoid 400× forward pass overhead per episode.

## Solution

Add **7 agent state channels** to the observation tensor:
- 4 per-asset channels (position, PnL, hold duration, heat fast/slow)
- 1 global channel (capacity usage — broadcast to all assets)

Observation shape: `(27, window_size, n_assets)` → `(34, window_size, n_assets)`

### Channel Layout (7 new channels)

| Ch# | Name | Type | Range | Description |
|-----|------|------|-------|-------------|
| 27 | `position_state` | per-asset | {-1, 0, +1} | Current position: short/flat/long |
| 28 | `unrealized_pnl` | per-asset | [-1, 1] | Signed cumulative return since entry, clipped |
| 29 | `hold_duration` | per-asset | [0, 1] | Steps since last trade / window_size |
| 30 | `heat_fast` | per-asset | [0, 1] | Trade recency, decay=0.80 (half-life ≈ 3 steps) |
| 31 | `heat_slow` | per-asset | [0, 1] | Trade recency, decay=0.99 (half-life ≈ 70 steps) |
| 32 | `capacity_usage` | global | [0, 1] | n_active_positions / k (how full are our slots?) |
| 33 | `confidence` | per-asset | [0.2, 1.0] | Max softmax prob per asset (oracle or model) |

All channels are **tiled across the time axis** — current state only, not state history through the window. This avoids causal leakage and matches inference behavior.

### Why This Layout

**2 heat channels, not 3**: Gemini review correctly notes that global state (capacity) is more valuable than a 3rd heat timescale. Two heat channels (fast=3 steps, slow=70 steps) already span the useful range.

**capacity_usage instead of cash/margin**: Easy Mode has no portfolio simulation — no cash, no margin. But the oracle DOES use k slots (k=4 max positions). `capacity_usage = n_active / k` tells the model "are all slots full?" This is derivable from oracle selections and matches inference.

**confidence channel**: Oracle's multipass soft targets encode signal quality. Model's own softmax encodes prediction uncertainty. Self-correcting: uncertain → low confidence → model learns to hold.

## Position State: The Critical Bug Fix

### The Problem (found in v2 review)

Actions are ambiguous: BUY can mean "open long" (flat→+1) OR "close short" (short→0). SELL can mean "open short" (flat→-1) OR "close long" (long→0). You CANNOT determine resulting position from action type alone — you need the previous position.

### Fix: Two Methods

**Training (oracle state): Use `target_position` from `optimal_path_to_policy()`**

The oracle already computes `target_position` at `path_solver.py:739-743`. This is the canonical source — correct by construction.

```python
# In multipass_oracle_policy / optimal_path_to_policy:
target_position = jnp.where(
    selections > 0.5,
    jnp.where(returns > 0, 1.0, jnp.where(returns < 0, -1.0, prev_pos)),
    0.0
)
# Return this alongside policy targets
```

Extract it and use directly — no re-derivation needed.

**Inference (model state): Action + previous position context**

```python
def position_from_action(prev_pos, action):
    """Determine resulting position from action + context.
    Buy when flat→long(+1). Buy when short→flat(0). Buy when long→noop(+1).
    Sell when flat→short(-1). Sell when long→flat(0). Sell when short→noop(-1).
    """
    is_buy = (action == 2) | (action == 4)
    is_sell = (action == 0) | (action == 3)
    new_pos = jnp.where(
        is_buy,
        jnp.where(prev_pos < 0, 0.0, 1.0),   # short→flat, else→long
        jnp.where(is_sell,
            jnp.where(prev_pos > 0, 0.0, -1.0),  # long→flat, else→short
            prev_pos)  # Hold → keep
    )
    return new_pos
```

## State Computation: `jax.lax.scan`

```python
@partial(jax.jit, static_argnames=['n_assets', 'k', 'window_size'])
def build_state_channels(
    target_positions,   # (T, n_assets) from oracle's target_position {-1, 0, +1}
    returns,            # (T, n_assets) log returns
    n_assets=64,
    k=4,
    window_size=64,
    decay_fast=0.80,
    decay_slow=0.99,
):
    """Build 6 per-step state channels + 1 global channel via lax.scan.

    Returns:
        state_channels: (T, 7, n_assets) — ready to tile per window
    """
    def step(carry, inputs):
        prev_pos, pnl, steps, h_fast, h_slow = carry
        pos, ret = inputs  # pos = target_position at this step

        # Detect trades (position changed)
        traded = (pos != prev_pos).astype(jnp.float32)
        # Flips (long→short or reverse) count as double trade
        flipped = (prev_pos != 0) & (pos != 0) & (pos != prev_pos)
        trade_strength = traded + flipped.astype(jnp.float32)

        # Unrealized PnL: signed relative to position direction
        held = (pos == prev_pos) & (prev_pos != 0)
        signed_ret = ret * prev_pos  # long: +ret, short: -ret
        new_pnl = jnp.where(held, pnl + signed_ret, 0.0)
        new_pnl = jnp.clip(new_pnl, -1.0, 1.0)

        # Hold duration: normalized by window_size
        new_steps = jnp.where(traded > 0.5, 0.0, steps + 1.0)

        # Multi-scale heat
        new_h_fast = h_fast * decay_fast + trade_strength
        new_h_slow = h_slow * decay_slow + trade_strength

        # Global: capacity usage
        n_active = jnp.sum(jnp.abs(pos) > 0.5)
        capacity = n_active / k

        out = jnp.stack([
            pos,                                          # ch27: position
            new_pnl,                                      # ch28: unrealized PnL
            jnp.clip(new_steps / window_size, 0.0, 1.0), # ch29: hold duration
            jnp.clip(new_h_fast, 0.0, 1.0),              # ch30: heat fast
            jnp.clip(new_h_slow, 0.0, 1.0),              # ch31: heat slow
            jnp.full(n_assets, capacity),                 # ch32: capacity (broadcast)
        ])  # (6, n_assets)

        return (pos, new_pnl, new_steps, new_h_fast, new_h_slow), out

    init = tuple(jnp.zeros(n_assets) for _ in range(5))
    _, channels = jax.lax.scan(step, init, (target_positions, returns))
    return channels  # (T, 6, n_assets)
```

### Confidence Channel (7th, computed separately)

```python
# Training: from oracle's multipass soft policy targets
oracle_policy_per_asset = oracle_policy.reshape(T, n_assets, 5)
confidence = oracle_policy_per_asset.max(axis=-1)  # (T, n_assets)

# Inference: from model's own softmax
model_policy_per_asset = jax.nn.softmax(logits.reshape(n_assets, 5), axis=-1)
confidence = model_policy_per_asset.max(axis=-1)  # (n_assets,)
```

### Tiling for Observation

```python
# state_at_t: (6, n_assets) — from build_state_channels at timestep t
# confidence_at_t: (n_assets,) — from oracle/model

# Tile to (7, window_size, n_assets)
state_tiled = jnp.tile(state_at_t[:, None, :], (1, window_size, 1))  # (6, ws, n_assets)
conf_tiled = jnp.tile(confidence_at_t[None, None, :], (1, window_size, 1))  # (1, ws, n_assets)

# Full observation: (34, window_size, n_assets)
full_obs = jnp.concatenate([market_obs, state_tiled, conf_tiled], axis=0)
```

## Oracle Independence: State Corruption (NOT Scheduled Sampling)

Full model-forward scheduled sampling costs 400 sequential forward passes per episode — ~30-60s overhead per buffer refill. This is impractical.

Instead: **state corruption** during training. The model sees oracle state but with random perturbations, forcing it to treat state as a hint, not a command.

```python
def corrupt_state(state_channels, key, epoch, total_epochs):
    """Progressive state corruption to bridge train/inference gap.

    Corruption increases over training: model starts with clean oracle state,
    gradually learns to handle noisy/imperfect state.
    """
    # Corruption intensity: 0.0 at epoch 0, ramps to max_noise
    progress = min(epoch / (total_epochs * 0.5), 1.0)
    noise_level = 0.3 * progress

    k1, k2, k3, k4 = jax.random.split(key, 4)

    # 1. Position bit-flip: randomly invert position direction
    flip_prob = 0.1 * progress  # 0% → 10%
    flip_mask = jax.random.bernoulli(k1, flip_prob, state_channels[0].shape)
    state_channels = state_channels.at[0].set(
        jnp.where(flip_mask, -state_channels[0], state_channels[0])
    )

    # 2. PnL noise: Gaussian perturbation
    pnl_noise = jax.random.normal(k2, state_channels[1].shape) * noise_level
    state_channels = state_channels.at[1].add(pnl_noise)

    # 3. Heat noise: random bumps (simulates over-trading distribution)
    heat_noise = jax.random.uniform(k3, state_channels[3].shape) * noise_level
    state_channels = state_channels.at[3].add(heat_noise)  # fast heat
    state_channels = state_channels.at[4].add(heat_noise * 0.5)  # slow heat (less noise)

    # 4. Full dropout: 5% chance of zeroing ALL state channels
    #    Forces model to still work from price action alone
    zero_mask = jax.random.bernoulli(k4, 0.05)
    state_channels = jnp.where(zero_mask, 0.0, state_channels)

    return state_channels
```

**Why corruption over scheduled sampling:**
- 99% cheaper (no forward passes, just random ops)
- Exposes model to high-heat states it wouldn't see from oracle (via heat noise)
- 5% full dropout ensures model never fully depends on state channels
- Progressive: clean start (learns what channels mean) → noisy later (learns robustness)

## Files to Modify

| File | Change |
|------|--------|
| `mcts_trader/env/jax_encoder.py` | Parameterize `n_channels` (hardcoded 27 → config) |
| `mcts_trader/jax_agent/path_solver.py` | Return `target_position` alongside policy targets |
| `easy_config.py` | `n_channels: int = 34`, `state_corruption: bool = True` |
| `train_easy.py` | Build state channels from oracle target_position, corruption, wire into obs |
| `easy_trainer.py` | Update `dummy_obs` shape to `(1, 34, 64, 64)` |
| `mcts_trader/jax_agent/trm_flax.py` | Verify encoder handles variable channel count |
| `mcts_trader/jax_agent/muzero_model.py` | Thread `n_channels` parameter |
| `tools/benchmark_easy.py` | Autoregressive state tracking with `position_from_action()` |

## Implementation Phases

### Phase A: Core State Channels (get it working)
1. Return `target_position` from oracle path solver
2. Parameterize `n_channels` in encoder (hardcoded 27 → config)
3. Implement `build_state_channels()` with `lax.scan` using oracle positions
4. Wire state channels into observation in `train_easy.py`
5. NO corruption yet — pure oracle state for sanity check
6. **Test**: 50-epoch run — does agree_hold improve?

### Phase B: Oracle Entropy + Dual Dojo (close the gap)

**Revised approach** (2026-02-21): Oracle entropy instead of post-hoc corruption.
Post-hoc corruption creates impossible states (flipped position but unchanged PnL).
Oracle entropy keeps state channels internally consistent.

#### B1: Oracle Entropy
- Random trade drops: oracle "trade" → forced "hold" with probability p
- Random phantom trades: oracle "hold" → forced "trade" with probability q
- Applied before `build_state_channels()` → all derived channels stay consistent
- Exactly simulates inference failure mode (missed trades → wrong position state)

#### B2: Dual Dojo (after CMP 90HX install)
- batch=512 on 3090 (clean gradients, exploitation)
- batch=128 on CMP 90HX 10GB (noisy gradients, exploration)
- Checkpoint merging as natural corruption — merged model doesn't match either oracle
- Batch size asymmetry is a feature, not a bug

**Test**: 200-epoch run — does model trade rate drop below 15%?

### Phase C: Inference Integration
1. Implement `position_from_action()` for inference-time state tracking
2. Update `benchmark_easy.py` with autoregressive state loop
3. **Test**: benchmark on real data — is trade rate calibrated?

## VRAM Budget

```
Observation tensor at batch=512 (current training):
  Old: 512 × 27 × 64 × 64 × 4 bytes = 226 MB
  New: 512 × 34 × 64 × 64 × 4 bytes = 285 MB
  Delta: +59 MB

Plus encoder projection, activations, gradients: ~+100-150 MB total
Current usage: 22.7 / 24.0 GB → ~23.0 GB (fits)
```

Fallback if OOM: reduce batch_size from 512 to 448.

## Expected Impact (Conservative)

| Metric | Current | Phase A (oracle state) | Phase B (+ corruption) |
|--------|---------|----------------------|----------------------|
| agree_hold | 73% | 78-82% | 82-88% |
| agree_trade | 48% | 44-48% | 42-48% |
| balanced_acc | 62% | 63-66% | 65-70% |
| model trade rate | 27.7% | 15-22% | 5-12% |
| over-trading | 7.1× | 3.8-5.6× | 1.3-3.0× |

## Rejected Alternatives

| Alternative | Why Rejected |
|-------------|-------------|
| Run jax_trading_env for ground-truth state | Easy Mode has no portfolio env — oracle is a path solver, not a portfolio manager. Would need to BUILD an env first. |
| Cash/margin channels | Don't exist in Easy Mode — no cash tracking, no margin. Would be phantom data. |
| 3 heat channels (fast/med/slow) | 2 is sufficient. The 3rd channel (med=0.95) is mostly redundant with fast+slow. Slot used for capacity_usage instead. |
| Full model-forward scheduled sampling | 400 sequential forward passes per episode. ~30-60s overhead per buffer refill. State corruption achieves similar robustness at 1% of the cost. |
| Two-stage architecture (trade detector + action selector) | Higher complexity, cascading errors. Try state channels first. Fallback if this fails. |
| Action history attention | Adds computational overhead, discrete/sparse signal is hard to learn. State channels provide dense, interpretable features. |

## Review History

- v1: Initial design — pure teacher forcing, 4 channels, Python loop
- v2: Revised after 3 reviews — multi-scale heat, corruption, scheduled sampling, confidence
- v3: Final revision after 5 reviews (mem-001 Opus, mem-002 Opus, Sonnet, Gemini, fresh):
  - CRITICAL position bug fix: use oracle's `target_position` directly (not action→position)
  - Inference: `position_from_action()` with prev_pos context
  - Corruption replaces scheduled sampling (99% cheaper, per mem-002)
  - Global capacity_usage channel added (per Gemini)
  - 2 heat channels not 3 (per Gemini: capacity > 3rd heat)
  - hold_duration normalized by window_size not hardcoded 64
  - Unrealized PnL clipped to [-1, 1] not [-10, 10]
  - Progressive corruption (ramps with training, not constant)

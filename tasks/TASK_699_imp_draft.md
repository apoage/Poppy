# TASK_699: Safe AR Rollout Speedup — Corrected Implementation

## Context

AR rollout is 83% of epoch time (509s at 89% AR vs 90s at 0%). The rollout runs 336 sequential model forward passes per episode, 2-4 AR episodes per refill (e-dojo: n_envs=4, buffer_size=16), each processed individually at B=1. Three optimizations are mathematically identical (zero quality risk): batched forward pass, bf16 inference, scan tuning.

**Target**: 509s → ~120-200s at 90% AR.

## File: `train_easy.py`

All changes in this single file. Old unbatched function stays for the single-episode call site at line 597.

---

## Bugs Found in Original Draft (v1)

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | Critical | JIT recompilation on variable B (Binomial(4, 0.89) → B=2,3,4) | Pad to fixed B=n_envs, discard padding results |
| 2 | Critical | `capacity = jnp.sum(jnp.abs(pos) > 0.5) / k` sums both B and n_assets | Sum over `axis=-1` only |
| 3 | Critical | `jnp.stack([pos, pnl, ...])` → `(6, B, n_assets)` not `(B, 6, n_assets)` | Use `axis=1` |
| 4 | Medium | scan output stacks to `(T, B, ...)` not `(B, T, ...)` | Transpose after scan |
| 5 | Medium | All tile/concat axes need shifts for batch dim | Complete axis remapping |
| 6 | Low | Option A bf16 is a no-op (TRM casts `x.astype(self.dtype)` immediately) | Skip to Option B or skip bf16 |
| 7 | Low | `jnp.stack` on ragged returns fails | All same length in practice; assert |
| 8 | Minor | `i in list` is O(n) | Use `set()` |
| 9 | Minor | `unroll=4` quadruples XLA code size → longer initial compile | Start with `unroll=2` |

---

## Change 1: scan unroll (1 line, zero risk)

**File**: `train_easy.py:307-309`

```python
# Current:
_, (model_state_trajectory, confidence_trajectory) = jax.lax.scan(
    scan_step, init_state, jnp.arange(T)
)

# Changed:
_, (model_state_trajectory, confidence_trajectory) = jax.lax.scan(
    scan_step, init_state, jnp.arange(T), unroll=2
)
```

Using `unroll=2` (not 4) to limit XLA code bloat. The scan body contains a full 50M-param forward pass — `unroll=4` would quadruple the compiled code size, significantly increasing initial JIT time. `unroll=2` halves kernel dispatches (336→168) with modest compile overhead. Measure first, bump to 4 if compile time is acceptable.

---

## Change 2: Batched `_build_model_state_trajectory_batched` (new function)

Insert after line 310 (after the existing unbatched function). The old function stays for the single-episode path at line 597.

```python
@partial(jax.jit, static_argnames=['model_apply_fn', 'n_assets', 'k', 'window_size', 'n_actions_per_asset', 'B'])
def _build_model_state_trajectory_batched(
    market_obs,        # (B, T, 27, ws, n_assets) pre-encoded market channels
    returns,           # (B, T, n_assets) log returns
    model_params,      # Current model parameters (no gradients)
    model_apply_fn,    # model.apply function
    n_assets=64,
    k=4,
    window_size=64,
    decay_fast=0.80,
    decay_slow=0.99,
    n_actions_per_asset=5,
    B=4,               # static: fixed batch size (padded to n_envs)
):
    """Batched AR rollout: B episodes processed in single scan call (TASK_699).

    Same computation as _build_model_state_trajectory but with batch dimension.
    B is STATIC (passed to jit via static_argnames) to prevent recompilation
    when the actual number of AR episodes varies. Caller pads to fixed B.

    Returns:
        model_state: (B, T, 6, n_assets) state channels
        model_confidence: (B, T, n_assets) softmax confidence
    """
    # Initialize neutral state — all (B, n_assets)
    init_state = (
        jnp.zeros((B, n_assets)),  # position
        jnp.zeros((B, n_assets)),  # pnl
        jnp.zeros((B, n_assets)),  # steps
        jnp.zeros((B, n_assets)),  # h_fast
        jnp.zeros((B, n_assets)),  # h_slow
        jnp.zeros((B, n_assets)),  # prev_confidence
    )

    def scan_step(carry, idx):
        """Single timestep, batched across B episodes."""
        pos, pnl, steps, h_fast, h_slow, prev_conf = carry
        # All carry arrays: (B, n_assets)

        # capacity: sum per-episode, not across batch
        capacity = jnp.sum(jnp.abs(pos) > 0.5, axis=-1, keepdims=True) / k  # (B, 1)

        # Build state channels: stack along channel axis (axis=1)
        state_ch = jnp.stack([
            pos,                                              # (B, n_assets)
            pnl,
            jnp.clip(steps / window_size, 0.0, 1.0),
            jnp.clip(h_fast, 0.0, 1.0),
            jnp.clip(h_slow, 0.0, 1.0),
            jnp.broadcast_to(capacity, (B, n_assets)),       # (B, n_assets)
        ], axis=1)  # (B, 6, n_assets)

        # Tile state across window: (B, 6, n_assets) → (B, 6, ws, n_assets)
        ws = window_size
        state_tiled = jnp.tile(state_ch[:, :, None, :], (1, 1, ws, 1))  # (B, 6, ws, n_assets)

        # Tile confidence: (B, n_assets) → (B, 1, ws, n_assets)
        conf_tiled = jnp.tile(prev_conf[:, None, None, :], (1, 1, ws, 1))  # (B, 1, ws, n_assets)

        # Construct full observation
        market_t = market_obs[:, idx]  # (B, 27, ws, n_assets)
        obs = jnp.concatenate([market_t, state_tiled, conf_tiled], axis=1)  # (B, 34, ws, n_assets)

        # Model forward pass — model handles any batch size natively
        output = jax.lax.stop_gradient(
            model_apply_fn({'params': model_params}, observation=obs, mode='initial_inference')
        )
        policy_logits = output['policy_logits']  # (B, 320)

        # Per-asset actions
        policy_reshaped = policy_logits.reshape(B, n_assets, n_actions_per_asset)
        action_types = jnp.argmax(policy_reshaped, axis=-1)  # (B, n_assets)

        # Model confidence from softmax
        softmax_probs = jax.nn.softmax(policy_reshaped, axis=-1)  # (B, n_assets, 5)
        model_confidence = jnp.max(softmax_probs, axis=-1)  # (B, n_assets)

        # Incremental position stepping (TASK_693)
        buy = (action_types == 2) | (action_types == 4)
        sell = (action_types == 0) | (action_types == 3)
        new_pos = jnp.where(
            buy,
            jnp.minimum(pos + 1.0, 1.0),
            jnp.where(sell, jnp.maximum(pos - 1.0, -1.0), pos)
        )

        # State update
        ret = returns[:, idx]  # (B, n_assets)
        traded = (new_pos != pos).astype(jnp.float32)
        flipped = (pos != 0) & (new_pos != 0) & (new_pos != pos)
        trade_strength = traded + flipped.astype(jnp.float32)

        held = (new_pos == pos) & (pos != 0)
        signed_ret = ret * pos
        new_pnl = jnp.where(held, pnl + signed_ret, 0.0)
        new_pnl = jnp.clip(new_pnl, -1.0, 1.0)

        new_steps = jnp.where(traded > 0.5, 0.0, steps + 1.0)
        new_h_fast = h_fast * decay_fast + trade_strength
        new_h_slow = h_slow * decay_slow + trade_strength

        # Emit prev_conf (BUG-2 alignment: confidence_trajectory[w] = what model SAW at step w)
        return (new_pos, new_pnl, new_steps, new_h_fast, new_h_slow, model_confidence), (state_ch, prev_conf)

    T = market_obs.shape[1]  # time axis is dim 1 (not 0)
    _, (model_state_trajectory, confidence_trajectory) = jax.lax.scan(
        scan_step, init_state, jnp.arange(T), unroll=2
    )
    # scan stacks outputs along new leading axis: (T, B, 6, n_assets), (T, B, n_assets)
    # Transpose to (B, T, 6, n_assets) and (B, T, n_assets)
    model_state_trajectory = jnp.transpose(model_state_trajectory, (1, 0, 2, 3))
    confidence_trajectory = jnp.transpose(confidence_trajectory, (1, 0, 2))

    return model_state_trajectory, confidence_trajectory
```

### Shape trace through scan_step

```
Input shapes (carry):
  pos, pnl, steps, h_fast, h_slow, prev_conf: all (B, n_assets) = (4, 64)

capacity:
  jnp.sum(jnp.abs(pos) > 0.5, axis=-1, keepdims=True) → (B, 1) = (4, 1)
  broadcast_to(capacity, (B, n_assets)) → (4, 64)

state_ch:
  jnp.stack([6 arrays of (B, n_assets)], axis=1) → (B, 6, n_assets) = (4, 6, 64)

state_tiled:
  state_ch[:, :, None, :] → (4, 6, 1, 64)
  tile (1, 1, ws, 1) → (4, 6, 64, 64) = (B, 6, ws, n_assets)

conf_tiled:
  prev_conf[:, None, None, :] → (4, 1, 1, 64)
  tile (1, 1, ws, 1) → (4, 1, 64, 64) = (B, 1, ws, n_assets)

market_t:
  market_obs[:, idx] → (4, 27, 64, 64) = (B, 27, ws, n_assets)

obs:
  concat([market_t, state_tiled, conf_tiled], axis=1)
  = concat([(4,27,64,64), (4,6,64,64), (4,1,64,64)], axis=1)
  → (4, 34, 64, 64) = (B, 34, ws, n_assets)  ✓

policy_logits:
  output['policy_logits'] → (B, 320) = (4, 320)

policy_reshaped:
  reshape(B, n_assets, 5) → (4, 64, 5)

action_types:
  argmax(axis=-1) → (4, 64) = (B, n_assets)  ✓

scan output per step:
  state_ch → (B, 6, n_assets) = (4, 6, 64)
  prev_conf → (B, n_assets) = (4, 64)

scan stacking (T steps):
  model_state_trajectory → (T, B, 6, n_assets) = (336, 4, 6, 64)
  confidence_trajectory → (T, B, n_assets) = (336, 4, 64)

after transpose:
  model_state_trajectory → (B, T, 6, n_assets) = (4, 336, 6, 64)  ✓
  confidence_trajectory → (B, T, n_assets) = (4, 336, 64)  ✓
```

---

## Change 3: Caller refactor in `_generate_batch_episodes` (line 826-928)

Replace the per-episode AR loop with a 3-phase collect→batch→distribute pattern.

### Phase 1: Pre-roll AR coin flips (before the main loop)

Insert between line 824 and 826:

```python
        # TASK_699: Pre-determine which episodes use AR (coin flip phase)
        ar_episode_set = set()
        if use_state_channels and ar_blend > 0 and self._model_params is not None:
            for i in range(n_batch):
                if self.rng.random() < ar_blend:
                    ar_episode_set.add(i)

        # TASK_699: Batched AR rollout (single JIT call for all AR episodes)
        batched_state_ch = {}  # i → (n_valid, 6, n_assets)
        batched_conf = {}      # i → (n_valid, n_assets)
        if ar_episode_set:
            ar_indices = sorted(ar_episode_set)
            B_actual = len(ar_indices)
            B_pad = self.n_envs  # Fixed B for JIT stability (static_argname)

            ws = self.config.window_size
            # Stack market obs for AR episodes, pad to fixed B
            market_obs_list = [all_observations[i] for i in ar_indices]
            returns_list = [log_returns[i, ws - 1:valid_lengths[i] - 1] for i in ar_indices]

            # All episodes should have same length (same config → same generation params)
            ar_lengths = [r.shape[0] for r in returns_list]
            assert len(set(ar_lengths)) == 1, f"TASK_699: Ragged AR lengths {ar_lengths}, cannot batch"
            T_ar = ar_lengths[0]

            # Pad to fixed B with zeros (model runs on padding, results discarded)
            market_obs_padded = jnp.zeros((B_pad, all_observations.shape[1],
                                           all_observations.shape[2], all_observations.shape[3],
                                           all_observations.shape[4]))
            returns_padded = jnp.zeros((B_pad, T_ar, n_assets))
            for b, i in enumerate(ar_indices):
                market_obs_padded = market_obs_padded.at[b].set(all_observations[i])
                returns_padded = returns_padded.at[b].set(returns_list[b])

            state_ch_batch, conf_batch = _build_model_state_trajectory_batched(
                market_obs_padded, returns_padded,
                self._model_params, self._model_apply_fn,
                n_assets=n_assets, k=self.config.oracle_k,
                window_size=ws, decay_fast=0.80, decay_slow=0.99,
                n_actions_per_asset=5, B=B_pad,
            )  # (B_pad, T_ar, 6, n_assets), (B_pad, T_ar, n_assets)

            # Distribute results (discard padding episodes)
            for b, i in enumerate(ar_indices):
                batched_state_ch[i] = state_ch_batch[b]  # (T_ar, 6, n_assets)
                batched_conf[i] = conf_batch[b]           # (T_ar, n_assets)
```

### Phase 2: Modify main loop to use batched results

Replace lines 828-868 (the per-episode processing):

```python
        # Process each episode (alignment and dict creation)
        episode_datas = []
        for i in range(n_batch):
            T = valid_lengths[i] - 1
            start_idx = self.config.window_size - 1
            end_idx = min(start_idx + n_valid, T)
            n_windows = end_idx - start_idx

            obs_i = all_observations[i, :n_windows]

            # TASK_699: Use pre-computed batched AR results
            use_ar_this_ep = (i in ar_episode_set)
            if use_state_channels and use_ar_this_ep:
                state_ch_i = batched_state_ch[i]  # (T_ar, 6, n_assets)
                conf_i = batched_conf[i]           # (T_ar, n_assets)

                ws = self.config.window_size
                state_at_w = state_ch_i[:n_windows]
                state_tiled = jnp.tile(state_at_w[:, :, None, :], (1, 1, ws, 1))
                conf_at_w = conf_i[:n_windows]
                conf_tiled = jnp.tile(conf_at_w[:, None, None, :], (1, 1, ws, 1))
                obs_i = jnp.concatenate([obs_i, state_tiled, conf_tiled], axis=1)

            elif use_state_channels:
                # Oracle state path — unchanged from current code (lines 870-893)
                # ... existing oracle state code ...
```

The rest of the loop (DAgger fix at 896-906, ep_data construction at 908-926) is unchanged. The only change is that `use_ar_this_ep` is now `i in ar_episode_set` instead of a fresh coin flip + individual rollout call.

---

## Change 4: bf16 — Skip Option A, evaluate Option B later

**Option A is a no-op.** The model's TRM encoder immediately casts input to `self.dtype` (trm_flax.py:25: `x = x.astype(self.dtype)`), which is float32. Casting obs to bf16 before the call gets undone on the first line.

**Option B** (rollout-specific model with `dtype=bfloat16, param_dtype=float32`) requires a second model init. This is a separate change with its own verification (need to confirm argmax stability under bf16 quantization). **Defer to a follow-up after measuring batch + scan gains.**

---

## Implementation Order

1. **scan unroll=2** — 1 line in existing function at line 307
2. **Batched rollout function** — new `_build_model_state_trajectory_batched` after line 310
3. **Caller refactor** — pre-roll coin flips + batched call + distribute in `_generate_batch_episodes`
4. **Measure** — compare epoch time at 90% AR
5. **bf16 Option B** — separate task if batch+scan gains are insufficient

## Verification

1. **Correctness**: Temporarily keep old per-episode path. Run 1 epoch with both, compare state trajectories. Must be bit-identical (same computation, just batched).
2. **Speed**: Compare refill time and epoch time. Target: 509s → ~150-200s at 90% AR.
3. **JIT stability**: Confirm no retraces after first epoch (check `jax.log_compiles()`).
4. **VRAM**: Monitor with `nvidia-smi`. Batching 4 episodes at inference (stop_gradient, no backward): ~200-400MB extra activations. Well within 24GB budget.

## Notes

- Old unbatched function stays permanently for the single-episode call site at line 597.
- `B` is static_argname → one trace per unique B value. Using fixed B=n_envs prevents recompilation.
- Padding episodes run the full model forward pass but results are discarded. This wastes ~0-2 episode's worth of compute (B_pad - B_actual) but avoids 10-60s JIT retrace penalty.
- e-dojo config: n_envs=4, buffer_size=16, refill ~3 episodes per epoch → B_pad=4, B_actual=2-4.

# Hypothesis Ranking: Trade Action Learning Collapse

## Methodology

All 5 files read in full (train_easy.py, easy_trainer.py, easy_config.py, benchmark_easy.py, muzero_model.py).
Cross-validated by reading trm_flax.py, trm_encoder_v2.py, and the BEFORE-commit state of easy_trainer.py via `git show 06c44ab~1:easy_trainer.py`.

---

## Ranking (Most → Least Likely)

### #1: Change 8 — Model Construction (hidden_dim 2048 → 1024) — **CONFIRMED**

**Commit**: 06c44ab (TASK_692/693) — added as part of the model construction cleanup
**Confidence**: CONFIRMED

The single biggest change. Before the commits, `easy_trainer.py` created MuZeroNetworks with:
```python
self.model = MuZeroNetworks(
    state_dim=config.state_dim,
    action_space_size=config.action_space_size,
    support_size=VALUE_SUPPORT_SIZE,
    H_cycles=config.H_cycles,
    L_cycles=config.L_cycles,
    use_prediction_head=False,
    n_assets=config.n_assets,
)
```

After the commits:
```python
self.model = MuZeroNetworks(
    ...
    hidden_dim=getattr(config, 'trm_hidden_dim', None),  # TASK_684
    ...
)
```

**Impact trace**:
- Before: `hidden_dim` not passed → MuZeroNetworks default `None` → TRMReasoning computes `state_dim * 16` = `128 * 16` = **2048**
- After: `hidden_dim` = `config.trm_hidden_dim` = **1024** (large config)
- TRMReasoning has 48 latent refiner cycles (6H × 8L) + 6 output refiner cycles, each using a 2-layer FFN with `hidden_dim` hidden units
- Parameter reduction: ~23M params lost (from ~48M to ~25.76M)

**Cross-validation**:
1. `git show 06c44ab~1:easy_trainer.py` — confirms NO `hidden_dim` in constructor before
2. `trm_flax.py:112` — `hidden_dim = self.hidden_dim if self.hidden_dim is not None else self.state_dim * 16`
3. `easy_config.py:175` — `get_large_config()` sets `trm_hidden_dim=1024`
4. `trm_encoder_v2.py:248` — V2 encoder does NOT receive embed_dim/encoder_num_heads from TRM → those params have NO effect

**Why it kills trade learning specifically**: Trades are ~3.9% of decisions (k=4/64 assets). The model needs fine-grained discrimination in the reasoning FFN to distinguish the rare times a trade is optimal from the ~96% hold cases. Halving the FFN (2048→1024) disproportionately hurts rare-class discrimination while leaving majority-class accuracy intact.

---

### #2: Change 1 — Model-State Rollout (TASK_692/693) — **LIKELY**

**Commit**: 06c44ab
**Confidence**: LIKELY (compound effect with #1)

Replaces clean oracle state channels with noisy model-state channels. At epoch 0, the model is random → state channels are noise. At epoch 80, the model is ~50% balanced accuracy → state channels are mediocre.

**Impact**: The model must bootstrap from noisy state channels while having only half the reasoning capacity (#1). Oracle state channels provided perfect position/PnL/duration signals; model state channels provide approximations that require MORE capacity to interpret, not less.

**Cross-validation**:
- `easy_config.py:91` — `autoregressive_state: bool = True` (default on)
- e-dojo_v1.py does NOT override this
- Previous codebase had NO model-state rollout (confirmed by commit list — 06c44ab added it)

---

### #3: Change 6 — Oracle Mask Head (TASK_695) — **POSSIBLE**

**Commit**: 77cb089
**Confidence**: POSSIBLE

CSV evidence shows `oracle_mask_loss = 0.231`, confirming the head IS active. Adds `0.1 * 0.231 = 0.023` to total loss (~0.8% of ~2.8 total). Small gradient mass, but:
- oracle_mask_accuracy = 0.937 from epoch 0 → model predicts "all zeros" baseline (93.75% = 60/64). The head is trivially solvable and adds noise gradients through the shared latent state.
- With half the reasoning capacity (#1), any noise is relatively more harmful.

**Cross-validation**:
- `easy_trainer.py:206` — oracle_mask_logits come from `output['oracle_mask']`
- `muzero_model.py:252` — oracle mask prediction runs on `latent_state` (shared with policy/value)
- The head is a 2-layer MLP (latent → 256 → 64), adding ~34K params

---

### #4: Change 2 — Returns Alignment (BUG-1 fix) — **UNLIKELY**

**Commit**: f6669f4 (TASK_694)
**Confidence**: UNLIKELY (alignment is correct)

**Analysis**:
- Single path: `returns_aligned = log_returns[ws - 1:]` → shape `(T - ws + 1, n_assets)`
- Window idx covers `prices[idx:idx+ws]`, decision at `idx + ws - 1`
- `returns_aligned[idx] = log_returns[idx + ws - 1]` → correct decision-point return
- Batch path: `log_returns[i, ws - 1:valid_lengths[i] - 1]` → same alignment, consistent
- Scan indexes `returns[0..n_valid-1]`, returns has n_valid+1 elements → safe

**Cross-validation**: Traced shapes through both single-episode and batch paths. The returns-to-window mapping is correct. The extra element in returns_aligned is harmless (never indexed).

---

### #5: Change 4 — Trade-Biased Sampling (BUG-7 fix) — **UNLIKELY**

**Commit**: f6669f4 (TASK_694)
**Confidence**: UNLIKELY

**Analysis**:
- `max_offset = max(0, n_available - take_count)` ≈ `424 - 9` = 415
- `trade_indices` are in range [0, 423]
- `valid_trades = trade_indices[trade_indices <= 415]` → filters indices 416-423 only
- At most 8 indices filtered out of typically 15-30 → negligible impact

**Cross-validation**: With episode_length=400, window_size=64, oracle_lookahead=20:
n_samples ≈ 424, take_count ≈ 9, max_offset ≈ 415. Only the last 8 timesteps are filtered.

---

### #6: Change 3 — Confidence Channel (BUG-2 fix) — **UNLIKELY**

**Commit**: a86c206
**Confidence**: UNLIKELY (correct implementation)

**Analysis**:
- Carry has 6 elements (added prev_confidence). Init correctly adds `jnp.zeros(n_assets)`.
- Unpacking `pos, pnl, steps, h_fast, h_slow, prev_conf = carry` matches init order.
- Emits `prev_conf` (what the model SAW) not `model_confidence` (what it computed).
- Assembly: `conf_at_windows = confidence[:n_windows]` uses scan output correctly.

**Cross-validation**: Matches benchmark_easy.py inference (line 533: `prev_confidence` initialized to zeros, updated after forward pass).

---

### #7: Change 5 — Buffer Param ID Fix (LATENT) — **UNLIKELY**

**Commit**: f6669f4 (TASK_694)
**Confidence**: UNLIKELY

Sentinel value -1 sorts FIFO correctly. No functional change to buffer refresh behavior. Could cause minor differences in which episodes get refreshed first, but not a learning regression.

---

### #8: Change 7 — Optimizer Warmup Guard — **NOT THE CAUSE**

**Commit**: f6669f4 (TASK_694)
**Confidence**: NOT THE CAUSE

With large preset: epochs ≥ 2500, steps_per_epoch=400, total_steps ≥ 1,000,000. warmup_steps=1000 << total_steps. The guard has zero effect.

**Cross-validation**: `effective_warmup = min(1000, 999999) = 1000`. `decay_steps = max(1000000, 1001) = 1000000`. Same as before.

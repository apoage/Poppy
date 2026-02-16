# TASK_674: Training Agreement Plateau — Bug Fixes

## Problem

Training with TASK_673 multipass oracle (noise=0.03, clean value PnL) stuck at 52-53% train / 50-51% val agreement. Pre-multipass baseline was 56%. Two confirmed bugs and one checkpoint issue found via deep code review.

---

## Bug 1: Buffer Sample Counter (CRITICAL)

**File**: `train_easy.py:635`

```python
self.buffer_sample_counts[ep_idx] += 1      # BUG: increments by 1
samples_needed -= take_count                  # but take_count can be up to 20
```

With default `batch_size=256`, `N_BUFFER=16`: `samples_per_window = min(256//16 + 1, 20) = 17`. Each visit takes 17 samples but counter only increments by 1. An episode with `SAMPLES_PER_EPISODE=200` gets sampled `200 × 17 = 3400` times before exhaustion instead of 200. The buffer refreshes **17× less frequently** than intended.

With larger batch sizes (e.g. 1024): `samples_per_window = min(1024//16 + 1, 20) = 20`, giving 4000 samples per episode (20× over-sampling).

**Effect**: Catastrophic for generalization. With 16 episodes in the buffer and ~3400 samples each, the model trains on the same 16 episodes for `16 × 3400 / 256 ≈ 213` batches before seeing anything new. The model memorizes specific episodes instead of learning general patterns.

**Post-fix consideration**: After fixing, `SAMPLES_PER_EPISODE=200` with `take_count=17` means only ~11 visits per episode before exhaustion (200/17 ≈ 11.7). The buffer refills ~17× more frequently. **Monitor CPU data generation throughput** — `_refill_buffer()` runs the oracle via `_generate_batch_episodes()`. If CPU generation can't keep up, GPU may stall waiting for new data. Watch for increased idle time in `nvidia-smi`.

**Fix**:
```python
self.buffer_sample_counts[ep_idx] += take_count
```

---

## Bug 2: Optimizer Warmup Reset on Every Merge (HIGH)

**File**: `easy_trainer.py:459-468`

Every merge resets `ScaleByAdamState.count` and `ScaleByScheduleState.count` to 0. The LR schedule is `warmup_cosine_decay_schedule` with `warmup_steps=1000` (from config). With merges every ~20 epochs (8000 steps at 400 steps/epoch):

- Steps 0-1000: LR ramps from 0 → peak (no real learning for ~2.5 epochs)
- Steps 1000-8000: Full LR training
- MERGE → count reset to 0
- Steps 0-1000 again: Another dead warmup period

**12.5% of all training is wasted on redundant warmup from the schedule reset alone.**

**Adam count reset is equally damaging** (not just the schedule count): Resetting `ScaleByAdamState.count` to 0 causes pathological bias correction. At count=1, the bias correction for nu amplifies it by `1/(1 - 0.999^1) = 1000×`. With realistic accumulated nu values, the effective per-parameter learning rate collapses to near-zero for the first ~100 steps until bias correction stabilizes. Combined timeline post-merge:

- Steps 0-100: Near-zero effective LR (adam bias correction on nu)
- Steps 100-1000: Still warming up (schedule LR ramping from 0)
- Steps 1000+: Actual productive training

Real waste is closer to **15-20%** of post-merge training, not just 12.5%.

**Fix**: Don't reset either count on merge. Zero Adam momentum (mu) for fresh gradient directions post-merge. Keep nu (second moment / Fisher approximation) and both counts.

```python
new_mu = jax.tree.map(jnp.zeros_like, adam_state.mu)  # Zero momentum
current_adam_count = adam_state.count
current_schedule_count = schedule_state.count

new_adam_state = optax.ScaleByAdamState(
    count=current_adam_count,       # Keep count (no warmup restart)
    mu=new_mu,
    nu=new_nu,
)
new_schedule_state = optax.ScaleByScheduleState(
    count=current_schedule_count,   # Keep schedule position
)
```

**Why this is safe**: With high count, bias correction is ~1.0 for both terms. Zeroed mu starts clean and rediscovers gradient directions naturally. Merged nu provides good per-parameter gradient scale estimates at face value. The cosine decay continues from its current position — with `decay_steps=2,500,000`, even after 8000 steps the LR is at ~99.97% of peak, so decay is negligible.

**Cleanup**: Remove dead parameter `warmup_steps: int = 200` from `rebuild_optimizer_state()` signature — it is never used in the function body (line 424). The comment at line 458 references it but the code just sets count to 0.

---

## Bug 3: Damaged Checkpoint

Training logs show: `Loaded checkpoint from merged_0049 (epoch 984)`. Merge #49 was created during the **broken** multipass run (noise=0.1, noisy value PnL) — 220 epochs of training with incorrect value targets.

**Fix**: Reset e-dojo and start fresh (no resume from damaged weights).

---

## Bug 4: Validation Uses Training Generator (MEDIUM)

**File**: `train_easy.py:1033`

```python
val_metrics = validate(trainer, generator, config, n_batches=20)
```

`validate()` uses the same `OnTheFlyGenerator` instance as training. It calls `generator.generate_batch()` which samples from the same VRAM buffer the model just trained on.

**With Bug 1 unfixed** (current code): Episodes last ~200 visits per epoch. In 200 training steps, the buffer likely never refreshes. Validation samples from the **exact same 16 episodes** the model just memorized. This is real data leakage.

**With Bug 1 fixed**: Buffer refreshes ~18 times per epoch (200 steps / 11 visits per episode). By validation time, the buffer has been fully replaced many times. Episodes are generated with fresh random seeds each refill. Leakage becomes negligible — but train/val are still drawing from the same generator state.

**Note**: Both train AND val are stuck at ~50-52%, not just val. If this were pure leakage, train would be high and val would be low. The plateau is caused by Bug 1 (data starvation), not leakage. But a separate val_generator is cleaner.

**Fix**: Create a second `OnTheFlyGenerator` with a different seed for validation.

---

## Bug 5: Value Head Support Under-utilization (LOW)

**File**: `easy_config.py:43` — `value_scale: float = 1.0`

The MuZero value support covers [-10, 10] with 401 bins (0.05 resolution). Empirical remaining PnL is typically in [0, 3], occasionally reaching ~5 (historically up to 9.25 with old oracle params). With `value_scale=1.0`, the scaled range is [0, 3], using **~60 out of 401 bins** (15% of support).

**History**: Was `value_scale=10.0` but TASK_146 reduced it because remaining PnL reached 9.25, giving `9.25 × 10 = 92.5` — massively clipped by the [-10, 10] support. Current parameters (k=1, lookahead=20) produce lower PnL, but outliers still reach 3-5.

**Clipping risk at higher scales**:
- `value_scale=2.0`: max scaled ≈ 6-10. Fits support, uses ~120 bins (30%). **Safe.**
- `value_scale=3.0`: max scaled ≈ 9-15. Outliers clipped. **Borderline.**
- `value_scale=5.0`: max scaled ≈ 15-25. **Clipping guaranteed.** Repeats TASK_146 failure.

**Impact**: Primarily degrades value_mse. The noisy value gradients flow through the shared TRM backbone (via `value_weight=0.5`), potentially degrading shared representations. Secondary to Bug 1 and Bug 2.

**Fix**: Set `value_scale = 2.0` (conservative, safe). Monitor `max(batch['returns'] * value_scale)` to verify no clipping.

---

## Bug 6: Zero-Return Direction Defaults to SHORT (LOW)

**File**: `mcts_trader/jax_agent/path_solver.py:723-726`

```python
target_position = jnp.where(
    selections > 0.5,
    jnp.where(returns > 0, 1.0, -1.0),  # ← returns==0 maps to -1 (SHORT)
    0.0
)
```

When an asset has exactly 0.0 return and the oracle selects it (via noisy returns in multipass), the target is SHORT instead of HOLD/neutral.

**Practical impact is LOW**: The oracle's `greedy_lookahead_path` won't select zero-return assets because they can't overcome `min_alpha_threshold=0.001` after transaction costs. Only occurs in multipass when noise pushes a zero-return asset above threshold — rare and diluted across 10 passes. Synthetic data uses floating-point random walks making exact 0.0 returns extremely uncommon.

**Fix** (clean but low priority):
```python
jnp.where(returns > 0, 1.0, jnp.where(returns < 0, -1.0, 0.0))
```

---

## Investigated But Not Bugs

### Validation balanced accuracy
Uses per-asset argmax (same as training) — `easy_trainer.py:302-322` matches `easy_trainer.py:194-216`. ✓

### `softmax_cross_entropy` with soft targets
`optax.softmax_cross_entropy(logits, labels)` computes `-sum(labels * log_softmax(logits))`. Handles soft distributions correctly. ✓

### Multipass oracle value targets
Returns clean PnL (already fixed in TASK_673 v2) — `path_solver.py:268-277` computes clean path separately. ✓

### Hold ratio mismatch (train 2D vs val 1D)
Cosmetic — only affects logging, not `oracle_agreement`. ✓

### Adam bias correction after zeroing mu (raised as concern)
**Not a bug.** With zeroed mu and high count (e.g. 10000), the first step's bias-corrected m̂ = `(1-β₁) * g / (1-β₁^t)` ≈ `0.1 * g / 1.0` = 0.1×g — a 10× reduction in momentum contribution. This is a **44-step soft warm-in** (not a permanent collapse): mu accumulates exponentially via `m_t = 0.9*m_{t-1} + 0.1*g` and reaches 99% of steady-state in ~44 steps (`log(0.01)/log(0.9)`). Compare to the current bug: 1000+ step hard warmup from schedule reset. The alternative (averaging mu from diverged branches) would inject stale/conflicting momentum that could cause oscillation. Zero mu + keep count is the correct approach.

### Soft-target weighting threshold
`is_trade = targets_reshaped[:, :, 1] < 0.5` uses a binary threshold on the hold probability. With multipass oracle (n_passes=10, noise_scale=0.03), targets are highly peaked — most assets are >95% in one action. The 0.5 threshold correctly classifies the vast majority. A continuous weighting scheme would be smoother at the boundary but won't affect the plateau. The multipass soft targets provide gradient smoothness via the cross-entropy loss itself, not the trade weight.

### State-value information gap
Value head predicts remaining PnL without knowing the agent's current position. With `transaction_cost=0.001`, the position-dependent component is ~0.1% of asset price per trade — negligible relative to the PnL signal. Not the cause of a 52% plateau.

---

## Execution Plan

### Step 1: Fix buffer sample counter
- `train_easy.py:635` — change `+= 1` to `+= take_count`

### Step 2: Fix optimizer warmup reset
- `easy_trainer.py:459-468` — preserve both counts across merges, zero mu, keep nu
- Remove dead `warmup_steps` parameter from `rebuild_optimizer_state()` signature

### Step 3: Increase value_scale (conservative)
- `easy_config.py:43` — change `value_scale = 1.0` to `value_scale = 2.0`
- NOT 5.0 — empirical remaining_pnl reaches 3-5, and `5 × 5.0 = 25` clips at [-10, 10] support
- Monitor `max(batch['returns'] * value_scale)` to verify no clipping at 2.0

### Step 4: Add separate validation generator (optional, good practice)
- `train_easy.py` — create second `OnTheFlyGenerator` with different seed for `validate()`

### Step 5: Fix zero-return direction (optional, clean)
- `path_solver.py:725` — three-way where: `>0 → 1, <0 → -1, ==0 → 0`

### Step 6: Fresh start
```bash
./tools/orchestration/reset_edojo.sh
python tools/orchestration/e-dojo_v1.py --run --preset large --epochs 2500 --dual-bella --oracle-multipass
```

### Verification
- After first merge: check logs — LR should NOT restart from 0, should continue at pre-merge level
- Agreement should climb past 56% baseline within first few hundred epochs
- Value MSE should improve with higher value_scale

---

## Status

- [x] Bug 1 fix applied (buffer sample counter) — `+= take_count`
- [x] Bug 2 fix applied (optimizer warmup reset + dead param cleanup) — preserve counts, removed dead param
- [ ] Bug 3 fresh checkpoint (reset e-dojo)
- [x] Bug 4 separate val generator — `val_generator` with seed offset +7919
- [x] Bug 5 value_scale increase — 1.0 → 2.0
- [x] Bug 6 zero-return direction — three-way where (long/short/flat)
- [ ] Fresh start launched
- [ ] Agreement recovery confirmed

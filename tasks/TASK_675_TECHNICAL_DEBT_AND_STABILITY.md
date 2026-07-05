# TASK_675: Technical Debt Cleanup & Stability Guardrails

## Problem
Following the resolution of the 56% agreement plateau in TASK_674, several structural fragilities and "silent" inefficiencies have been identified that pose risks to scaling and long-term maintainability.

---

## HIGH — Affects every training run

### 1. LR Schedule `decay_steps` is 5× Wrong
**File**: `easy_trainer.py:75`

```python
decay_steps=config.epochs * 1000,  # "Approximate"
```

Actual steps per epoch is `args.steps_per_epoch` (default 200, set at `train_easy.py:841`). So `decay_steps` should be `epochs * 200`, not `epochs * 1000`. The cosine decay runs 5× slower than intended — LR stays near peak for the entire run. With 2500 epochs: intended end at step 500k, actual end at step 2.5M (never reached).

With TASK_674's fix preserving schedule count across merges, this matters more — the schedule is now persistent but effectively flat.

**Fix**: Pass actual steps_per_epoch into the trainer or config:
```python
decay_steps=config.epochs * config.steps_per_epoch,  # Exact
```
Requires adding `steps_per_epoch` to `EasyConfig` (currently only an argparse default).

---

### 2. `evaluate_batch` Ignores Latent Cache (2-3× validation slowdown)
**File**: `easy_trainer.py:294-298`

```python
output = self.model.apply(
    {'params': self.params},
    observation=batch['observations'],  # Always re-encodes through TRM
    mode='initial_inference'
)
```

When latent caching is enabled (normal mode), training uses `prediction_only` mode (skips TRM encoder). But `evaluate_batch()` always runs full `initial_inference`, re-encoding every batch through the TRM. Validation runs 20 batches per epoch — each wastes a full TRM forward pass.

**Fix**: Detect and use cached latent states:
```python
if 'latent_states' in batch and batch['latent_states'] is not None:
    output = self.model.apply(
        {'params': self.params},
        latent_state=batch['latent_states'],
        mode='prediction_only'
    )
else:
    output = self.model.apply(
        {'params': self.params},
        observation=batch['observations'],
        mode='initial_inference'
    )
```

---

## MEDIUM — Robustness & correctness edge cases

### 3. Silent Exception Swallowing in Price Generation
**File**: `train_easy.py:222-223`

```python
except Exception as e:
    pass  # Ignore errors, will regenerate
```

CUDA OOM, JAX compilation errors, and real bugs are silently swallowed. If background generation systematically fails, the queue starves and training silently falls back to synchronous generation (slow) with no diagnostic.

**Fix**: Log and count failures:
```python
except Exception as e:
    logger.warning(f"Price generation failed: {e}")
    self._prefetch_failures = getattr(self, '_prefetch_failures', 0) + 1
    if self._prefetch_failures > 50:
        logger.error("Excessive prefetch failures — check GPU/data pipeline")
```

---

### 4. Hardcoded Oracle Lookahead Buffer
**File**: `train_easy.py:185`

```python
n_steps=self.config.episode_length + self.config.window_size + 20,
```

The magic `+ 20` is the oracle lookahead buffer. Should use `self.config.oracle_lookahead` so they stay in sync. If someone increases `oracle_lookahead` to 40, the SyntheticMarket would be too short and the oracle would read into zero-padding.

**Fix**:
```python
n_steps=self.config.episode_length + self.config.window_size + self.config.oracle_lookahead + 5,
```

---

### 5. Hold Ratio Metric Inconsistency
**File**: `easy_trainer.py:238-239` vs `easy_trainer.py:345-346`

Training computes `oracle_hold_ratio` from per-asset argmax over `(batch, 64)` — fraction of all 64 assets that are HOLD. Evaluation computes it from global argmax over `(batch, 320)` then `% 5` — fraction of batch samples where the single top-1 action is HOLD. Different denominators, different semantics.

Only affects logged hold ratio metrics, NOT `oracle_agreement` (correct in both paths). Cosmetic but confusing when comparing train vs val hold ratios.

**Fix**: Use per-asset method in `evaluate_batch` (already computed at line 306-307):
```python
oracle_hold_ratio = float(jnp.mean(oracle_action_type == 1))
pred_hold_ratio = float(jnp.mean(pred_action_type == 1))
```

---

### 6. Zero-Sample Episodes Added to Buffer
**File**: `train_easy.py:408-420`

If `end_idx <= start_idx` for a particular episode in a batched refill (e.g., very short price series), the episode gets `n_samples=0` and is still added to the buffer. When sampled, `take_count = min(17, needed, 0) = 0`, triggering the exhaustion mark at line 622. Harmless (wastes one buffer slot briefly) but should be filtered.

**Fix**: Add guard before appending:
```python
if ep_data['n_samples'] > 0:
    episode_datas.append(ep_data)
```

---

## LOW — Tech debt & defensive hardening

### 7. CSV Logger Not Closed on Training Crash
**File**: `train_easy.py:1117`

No `try/finally` around the training loop. If training crashes, CSV file handle leaks and log may be corrupted.

**Fix**: Wrap in try/finally:
```python
try:
    # ... training loop ...
finally:
    csv_logger.close()
    generator.shutdown()
    val_generator.shutdown()
```

---

### 8. Validation Generator Doubles Buffer Memory
**File**: `train_easy.py:980-991`

`val_generator` allocates the same full-size buffer as the training generator. It's only used once per epoch for 20 batches. Wastes VRAM holding 16 episodes permanently for infrequent use.

**Fix**: Use smaller buffer for validation:
```python
val_buffer_size = max(args.buffer_size // 4, 4)
```

---

### 9. Thread Seed Hash Collision Risk
**File**: `train_easy.py:180-181`

```python
thread_seed = self._base_seed + hash(threading.current_thread().ident) % (2**31)
```

Hash of thread ID is not guaranteed unique. Multiple threads could get the same seed, generating identical markets (breaking diversity). Replace with atomic counter:
```python
self._thread_counter = 0  # in __init__
with self._market_lock:
    thread_seed = self._base_seed + self._thread_counter
    self._thread_counter += 1
```

---

### 10. Configuration Hygiene (Zombie Parameters)

- **`trade_loss_weight`** (`easy_config.py:46`): Overridden by TASK_149 adaptive weighting. Dead code, kept for checkpoint compat.
- **`warmup_steps`** in `EasyConfig` (`easy_config.py:37`): Still used to create the schedule at `easy_trainer.py:74`, but the name collides with the now-removed `warmup_steps` param in `rebuild_optimizer_state`. Not actually dead, but confusing.
- **`min_alpha_threshold`** defaults: `easy_config.py` says 0.001, `path_solver.py` function signatures also default to 0.001 — currently consistent but fragile (two sources of truth).

---

### 11. VRAM Safety (Multipass Scaling)
`batched_multipass_oracle_policy` uses a double-vmap (outer over N episodes, inner over n_passes).

At 10 passes, safe. At 32-64 passes (for higher-fidelity soft targets), intermediate memory spikes linearly. Transition from `jax.vmap` to `jax.lax.map` for the pass dimension would cap VRAM at 1-pass peak.

---

### 12. Streaming Encoder Research
The JAX TRM encoder re-processes 64-step windows for every training step. 63/64 steps are redundant. KV-caching or recurrent-style updates for the temporal dimension could double training throughput.

---

## Investigated — Not Bugs

- **Multipass oracle real vs noisy returns** (`path_solver.py:308`): Intentional. Noise varies SELECTION (which assets), real returns determine DIRECTION (long/short). Using noisy returns for direction would create wrong labels.
- **Empty selection PnL** (`path_solver.py:121-131`): When no assets selected, PnL=0 and cost=change from prev position. Correctly handles closing costs.
- **e-dojo merge interval reset** (`e-dojo_v1.py:639`): `steps_since_merge` is recomputed from epoch deltas each poll (line 527-528), not accumulated. Reset is redundant but not buggy.
- **Buffer sample counter race condition**: Only called from main thread. No concurrent access.
- **Batched oracle variable-length episodes**: All episodes padded to same `max_len` in `_generate_batch_episodes` (line 329). vmap works correctly on padded arrays.

---

## Execution Plan

### Phase 1: Quick wins (apply with TASK_674 fixes)
1. Fix `decay_steps` — add `steps_per_epoch` to config, use exact value
2. Add latent cache to `evaluate_batch` — match training mode
3. Fix hardcoded `+20` → use `oracle_lookahead`
4. Fix hold ratio metric in `evaluate_batch`
5. Add try/finally around training loop

### Phase 2: Robustness (next session)
6. Add exception logging to `_on_prices_ready`
7. Filter zero-sample episodes
8. Reduce val_generator buffer size
9. Fix thread seed to use atomic counter

### Phase 3: Research / scaling
10. Config zombie cleanup
11. Memory-safe multipass (`lax.map`)
12. Streaming encoder research

---

## Status

- [x] #1 LR schedule decay_steps fix — `config.epochs * config.steps_per_epoch`, added to EasyConfig
- [x] #2 evaluate_batch latent cache — detects `latent_states` in batch, uses `prediction_only`
- [x] #3 Exception logging in price generation — log + count failures, error at 50+
- [x] #4 Hardcoded lookahead buffer — `+ config.oracle_lookahead + 5`
- [x] #5 Hold ratio metric consistency — per-asset argmax in evaluate_batch
- [x] #6 Zero-sample episode filter — guard before append
- [x] #7 CSV logger try/finally — wraps training loop, shuts down both generators
- [x] #8 Val generator buffer size — `buffer_size // 4` (min 4)
- [x] #9 Thread seed collision fix — atomic counter replaces hash
- [x] #10 Config zombie cleanup — clearer deprecation markers
- [x] #11 Memory-safe multipass — `jax.lax.map` replaces `jax.vmap` for pass dimension
- [ ] #12 Streaming encoder research — future work (TASK_676)
- [x] #13 Empty buffer crash guard — `RuntimeError` after empty refill (`train_easy.py:619`)
- [x] #14 Optimizer structure assertions — duck-type checks for `.mu`/`.nu`/`.count` (`easy_trainer.py:453-458`)
- [x] #15 Pin optax==0.2.6 — prevents private API breakage (`requirements.txt:5`)

---

## Yarrow Scan Findings (2026-02-17)

Added from multi-model code review (GLM-4.7, gpt-oss-120b, llama-8b).

### 13. Empty Buffer Crash Guard
**File**: `train_easy.py:595-618`
**Severity**: HIGH
**Source**: gpt-oss-120b

`generate_batch()` crashes with `ValueError` if `_refill_buffer()` returns empty
(e.g., data generation thread dies, all episodes fail to generate). No guard after refill.

**Fix** (~2 lines):
```python
if not self.buffer:
    raise RuntimeError("Buffer empty after refill — check price generation pipeline")
```

**Test**:
```python
# Temporarily mock _refill_buffer to return empty, verify RuntimeError
generator.buffer = []
generator.buffer_sample_counts = []
try:
    generator.generate_batch(batch_size=256)
    assert False, "Should have raised RuntimeError"
except RuntimeError as e:
    assert "Buffer empty" in str(e)
```

---

### 14. Brittle Optimizer State Indices
**File**: `easy_trainer.py:346-373` (rebuild_optimizer_state)
**Severity**: HIGH (tech debt)
**Source**: GLM-4.7

`rebuild_optimizer_state()` hardcodes `opt_state[1][0]` = Adam, `[1][2]` = Schedule.
If `optax.chain()` order ever changes (e.g., adding gradient accumulation, changing
clip position), these indices silently corrupt the optimizer state.

**Fix**: Add structural assertion:
```python
# Verify optimizer structure matches expectations
assert isinstance(template[0], optax.EmptyState), \
    f"Expected EmptyState (clip) at [0], got {type(template[0])}"
assert hasattr(template[1][0], 'mu') and hasattr(template[1][0], 'nu'), \
    f"Expected ScaleByAdamState at [1][0], got {type(template[1][0])}"
assert hasattr(template[1][2], 'count') and not hasattr(template[1][2], 'mu'), \
    f"Expected ScaleByScheduleState at [1][2], got {type(template[1][2])}"
```

**Test**:
```python
# Verify assertions pass with current optimizer structure
trainer.rebuild_optimizer_state()  # Should not raise
# Then change optax.chain order and verify it DOES raise
```

---

### 15. optax Private API Dependency
**File**: `easy_trainer.py:346`
**Severity**: HIGH (tech debt)
**Source**: GLM-4.7

```python
from optax._src.base import EmptyState  # Private API
```

Also uses `optax.ScaleByAdamState` and `optax.ScaleByScheduleState` which are
semi-public (not in `optax.__all__` but importable). Any optax version bump could
break the import or rename these types.

**Fix options** (pick one):
1. **Pin optax version** in requirements.txt (cheapest, short-term)
2. **Duck-type instead of import**: Check for `.mu`/`.nu` attributes instead of
   `isinstance(x, ScaleByAdamState)` — works regardless of class name
3. **Use optax serialization API** if available in newer versions

**Test**:
```python
# Verify current optax version exports needed types
import optax
assert hasattr(optax, 'ScaleByAdamState'), "ScaleByAdamState not in optax"
assert hasattr(optax, 'ScaleByScheduleState'), "ScaleByScheduleState not in optax"
# EmptyState: check private import
from optax._src.base import EmptyState  # Will fail if optax restructures
```

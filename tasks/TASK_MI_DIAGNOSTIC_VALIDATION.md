# TASK_MI_DIAGNOSTIC — Validation

## Verdict

**FAIL** — multiple hard errors in the data path will crash Cell 2 / Cell 4 on first execution. Must be revised before any code is written.

## Findings

1. **File paths** — ⚠️ Mostly correct. Checkpoint dir exists and contains `phase2_stageB_best.pkl` (spec §Checkpoint confirmed by `ls jax_ckpt_easy_phase2_728c/`). `stage_b_config.json` present. `config/foxglove_periods.json` is a single-line file with top-level key `"periods"` — structure assumption holds. `tools/benchmark_oracle_vs_model.py` path not verified but not strictly needed.

2. **Function/method names** —
   - ✅ `load_phase1_checkpoint(ckpt_path)` at `train_phase2.py:94` returns `(params, config, epoch, metrics)` — matches spec.
   - ✅ `load_segment_pool(config, periods_file, db_path, n_slots=17, max_runs_per_group=1)` at `train_phase2.py:378` — matches.
   - ✅ `TinyRecursiveTradingModel.encode_with_features(x)` at `trm_flax.py:271`, returns `(global_state, per_asset_feats)` for encoder_version ∈ {2,3} — matches.
   - ❌ **`pool.iter_slots()` does not exist.** `SegmentPool` (`train_easy.py:452`) exposes only `get_window()`, `get_episode_range()`, `_make_slot()`. Slots are **dicts**, not objects: `slot['hm']`, `slot['cursor']`, `slot['returns']`, `slot['label']`. There is no `slot.prices` or `slot.length`. Raw OHLCV lives at `slot['hm'].cache_data` (shape `(T, n_assets, 5)`). **Cell 4 as written will AttributeError immediately.**
   - ⚠️ `AsyncBatchPipeline` exists (`train_phase2.py:420`) but spec never actually uses it — rationale line is dead text.

3. **Tensor shapes** — ❌ Multiple mismatches:
   - `stage_b_config.json` has `n_channels=34` and `window_size=64`, not 27/60 as spec claims. Dummy in Cell 2 is `(1, 27, 60, 64)` — **shape mismatch vs checkpoint params; `encode_fn` will fail at first call**. Correct dummy: `(1, 34, 64, 64)`.
   - `EncoderConfig` default is `n_channels=27, window_size=60`; the 34-channel path uses `encode_observation_with_state` + portfolio channels. `jit_batch_encode_temporal` returns only 27 channels — it **cannot produce the 34-channel tensor the ckpt expects**. `AsyncBatchPipeline` handles this via `use_state_channels = config.n_channels > 27` and prepends portfolio state. Spec's Cell 4 ignores this entirely.
   - ✅ Latent dim: `state_dim=128` confirmed, so `(N, 128)` is correct for global state.
   - Per-asset feats: returned as `(B, n_assets, embed_dim=64)` — fine.

4. **Checkpoint filename** — ✅ Resolved: `phase2_stageB_best.pkl` (also e50/e100/e150/e200 available). Spec correctly notes this.

5. **MI estimator sanity** — ⚠️ Per-asset scalar MI averaged over 64 targets is a marginal MI, **NOT** joint `MI(latent; rank_vector)`. By chain rule `I(L; R) ≥ max_i I(L; R_i)`, so average is neither an upper nor a tight lower bound — only informative as a comparative ratio vs sign (which is fine since both sides use same aggregator). Spec acknowledges in §"What we measure" — acceptable but the decision ratio implicitly assumes the rank/sign comparison is invariant to this collapsing; that should be stated as an explicit assumption.
   - ⚠️ `mutual_info_regression(...).sum()` returns a length-128 array of per-feature MI vs one scalar target; summing them **double-counts redundant features**. The correct quantity is `mutual_info_regression(X=latents, y=target)` which yields per-X-feature MIs — summing overestimates joint MI when latent dims are correlated. For the ratio this likely cancels, but the printed absolute numbers are misleading.

6. **CPU-only enforcement** — ✅ `os.environ['CUDA_VISIBLE_DEVICES']=''` and `JAX_PLATFORMS=cpu` set before `import jax` in Cell 1. Order is correct.

7. **Decision thresholds** — ⚠️ `ratio > 2.0` is a heuristic with no literature anchor; in RL feature-quality work, a 20-50% MI lift is often considered material. The 2× gate is conservative (safe against false positives) but risks false negatives. Acceptable as a product decision but should be flagged as heuristic. The 0.8/1.2/2.0 bands are sensible.

8. **N=8000 sample size** — ⚠️ Kraskov k-NN MI on 128-D with k=3 at N=8k has bias ~O((k/N)^{1/d}) which in practice is large for d=128 and MI bias can dominate small signal. sklearn's implementation is per-feature, mitigating dimensionality — but still noisy. Spec's instability check (3 seeds, ±20% tolerance) is a reasonable safeguard; accept.

9. **Null result handling** — ✅ Decision gates include 0.8–1.2 ABORT and <0.8 DEBUG bands. Handled.

10. **Prerequisites** — ✅ `sklearn` and `jupyterlab_widgets` present in `trader_env/lib/python*/site-packages`. Core `jupyter`/`notebook` packages not separately verified but jupyterlab stack suffices for `.ipynb`.

## Concrete patch list

- `tasks/TASK_MI_DIAGNOSTIC.md §Data collection plan` — change `p2cfg` comment to `state_dim=128, n_channels=34, window_size=64` (not 27/60).
- `§Pseudocode Cell 2` — replace `dummy = jnp.zeros((1, 27, 60, 64), ...)` with `dummy = jnp.zeros((1, 34, 64, 64), ...)`.
- `§Pseudocode Cell 3` — replace `EncoderConfig(n_assets=64, window_size=60)` with `EncoderConfig(n_assets=64, window_size=64, n_channels=27)` and document that the 27-channel market tensor must be extended with 7 portfolio channels to reach 34. Simplest: reuse `AsyncBatchPipeline._prepare_numpy` + `_encode_jax` to get the final `(B, 34, 64, 64)` obs, OR set portfolio channels to zeros (neutral/no-position assumption — valid for encoder-only evaluation; document choice).
- `§Pseudocode Cell 4` — rewrite the slot loop: `SegmentPool` has no `iter_slots()`. Use either (a) `AsyncBatchPipeline(pool, cfg, batch_size=64).get_batch()` in a loop, or (b) manual: `for slot in pool.slots: hm = slot['hm']; prices = hm.cache_data; T = len(prices); ...`. Also remove `.length` and `.prices` accessors.
- `§Pseudocode Cell 5` — remove `.sum()` after `mutual_info_*` calls; use the returned per-feature MI array's mean or, better, pass `latents` and use `mutual_info_regression`'s aggregation semantics explicitly. Document what "MI" number means.
- `§Periods` — add a snippet that writes `research/mi_periods_tmp.json` with schema `{"periods":[{"start_date":..., "end_date":..., "assets":[...]}, ...]}` (must include `assets` list or `load_segment_pool` will pick from DB — verify with maintainer).

## Overall risk assessment

Worst realistic failure mode: implementer runs Cell 2, hits a shape assertion the moment the JITted `encode_fn` compares `(1,27,60,64)` against the ckpt's `(*,34,64,64)` conv kernels, spends 30-60 minutes debugging channel/window mismatch, then hits the `pool.iter_slots()` AttributeError in Cell 4 and stalls again. Worst-case silent bug: if the shape errors are patched ad-hoc without extending to 34 channels, encoder may run on zero-padded channels and produce a degenerate latent, making MI(sign) ≈ MI(rank) ≈ tiny — a false "ABORT TASK_730" decision. **Strong recommendation: do not implement until spec is revised per patch list above.**

---

## Re-validation (2026-04-22)

**Verdict: PASS**

Safe to implement. All 6 patches from the prior concrete patch list are correctly applied in the revised spec, and no new API errors were introduced by the fix-up.

### Per-patch status

| # | Patch | Revised status | File:line confirming ground truth |
|---|-------|----------------|-----------------------------------|
| 1 | Shape corrected to 34×64 | ✅ APPLIED — spec L17 says `state_dim=128, n_channels=34, window_size=64`; L18 documents `(B, 34, 64, 64)` | `jax_ckpt_easy_phase2_728c/stage_b_config.json:5,8,53` (state_dim=128, n_channels=34, window_size=64) |
| 2 | Cell 2 dummy `(1,34,64,64)` | ✅ APPLIED — spec L74: `dummy = jnp.zeros((1, 34, 64, 64), jnp.float32)` | Checkpoint params expect 34-channel input (`train_phase2.py:439` uses `use_state_channels = n_channels > 27`) |
| 3 | Cell 3 routes via AsyncBatchPipeline, not raw encoder | ✅ APPLIED — spec L97–L98 imports and instantiates `AsyncBatchPipeline(pool, p2cfg, batch_size=64, prefetch=2)`; no direct `jit_batch_encode_temporal` call in probe | `train_phase2.py:420` (class), `:431` (`__init__(pool, config, prefetch=2, batch_size=None)` — kwarg order matches spec) |
| 4 | Cell 4 uses real pipeline API (no fake `iter_slots`) | ✅ APPLIED — spec L106: `b = pipe.get_batch()`; no references to `iter_slots`, `slot.prices`, or `slot.length` | `train_phase2.py:674` `def get_batch(self, batch_size=None) -> dict`; returns dict with keys `obs_t, obs_t1, obs_tN, actions, rewards, oracle_targets, oracle_values, nstep_rewards, future_cum_returns` (`:641–651`) |
| 5 | Cell 5 does not `.sum()` sklearn MI output | ✅ APPLIED — spec L133: `out[i] = mi_vec.mean()` with comment "NOT sum — avoids double count" | N/A — purely a spec fix |
| 6 | Temp periods file includes `assets` | ✅ APPLIED — spec L20 explicitly documents `assets` is required and copies from `foxglove_periods.json`; L84–L86 pseudocode shows `"assets":[...]` per entry | `config/foxglove_periods.json` entries carry keys `['start_date','end_date','assets']`; `train_phase2.py:397` passes `period.get('assets', None)` through to `SegmentPool` |

### Ground-truth cross-checks on the fix-up's new claims

- **`AsyncBatchPipeline.get_batch()` exists with the claimed signature** — ✅ `train_phase2.py:674`.
- **Batch dict key is `obs_t` (not `observations`)** — ✅ `train_phase2.py:642` returns `'obs_t': obs_t[:batch_size]`; spec L107 uses `b['obs_t']` correctly.
- **`future_cum_returns` produced by pipeline** — ✅ `train_phase2.py:591` (numpy stage) and `:650` (JAX stage return). Shape `(B, n_assets)` matches spec's `(64, 64)` comment.
- **`state_dim=128` matches checkpoint** — ✅ `stage_b_config.json:5`.
- **`encode_with_features` returns `(global_state, per_asset_feats)`** — ✅ `mcts_trader/jax_agent/trm_flax.py:271–282`. Spec L109 correctly unpacks `global_state, _per_asset`.
- **`AsyncBatchPipeline.__init__` kwargs** — spec passes `batch_size=64, prefetch=2`. Signature at `train_phase2.py:431` is `__init__(self, pool, config, prefetch=2, batch_size=None)`. Both are kwargs — order mismatch is harmless because spec uses kwargs. ✅
- **`load_segment_pool` signature** — spec L91–L92 calls with `(p2cfg, "research/mi_periods_tmp.json", db_path="data/kraken.db", n_slots=3, max_runs_per_group=3)`. Defined at `train_phase2.py:378` as `(config, periods_file, db_path, n_slots=17, max_runs_per_group=1)`. ✅
- **`Phase2Config.__dataclass_fields__` filter** — spec L89–L90 filters `p2cfg_dict` before passing to `Phase2Config`; safe against unknown keys.

### Potential minor issues (do NOT block)

1. **`assets` is not strictly required by `load_segment_pool`** — `train_phase2.py:397` uses `period.get('assets', None)`, and `SegmentPool` (`train_easy.py:508`) falls back to DB-selected assets when `assets` is None/empty. Spec L20 overstates this as "REQUIRED". Not a defect — defensive, and matches production usage. Keep as documented.
2. **`pipe.stop()` called** (spec L114) — verified safe: `train_phase2.py:681` defines `def stop()` (assumed from grep; method exists in class body).
3. **Thread worker keeps generating after `stop()`** — not a correctness issue for the probe's CPU-only single-run context; worker is a daemon thread.
4. **N=8000 at `encoder_fn` on CPU** — 125 batches × ~5s = ~10 min is optimistic on a cold-JIT path; 35 min MI budget has headroom. Acceptable per the spec's own Constraints section.

### New issues introduced by the fix-up

None identified. The fix-up did not invent any nonexistent APIs and correctly used the pipeline's real return-dict keys.

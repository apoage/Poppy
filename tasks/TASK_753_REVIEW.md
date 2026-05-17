# TASK_753 — Implementation Review (Phase 1)

**Reviewer:** Claude (workstation agent, 2026-05-17).
**Reviewed:** `tasks/TASK_753_BELLA_AS_DATA_AUGMENTATION.md` (commit `287bcb7`).
**Sibling already implemented:** TASK_754 (commit `9aa4885`) — its file:line additions are the new baseline this review uses.
**Verdict:** **PASS-with-caveats (9 caveats — see §3).**

---

## 1. Understanding

### What is being built
A second wiring of Bella, dual to TASK_740/754. TASK_740/754 used Bella as a *teacher* (offline predictions → auxiliary loss). TASK_753 uses Bella as a *generator* of raw OHLCV that flows through the **same encoder path as real history**. No new heads, no new losses, no auxiliary supervision. The only change is that some `SegmentPool` slots' `hm.cache_data` come from Bella instead of `HistoricalMarket`.

### Dataflow (after this task lands)
1. **Bella generator loaded once** at SegmentPool init (lazy, on first Bella slot construction). VRAM-resident, frozen, inference-only.
2. **New class `BellaMarket`** (new file `mcts_trader/data/bella_market.py`) mirrors `HistoricalMarket`'s public surface (`cache_data`, `assets`, `asset_ids`, `valid_timestamps`, `close_db`, `symbol_to_id`, `id_to_symbol`, `n_assets`, `min_time`, `max_time`) so downstream `SegmentPool._make_slot` precompute is byte-identical. Constructor calls `generator.generate_on_device(n_samples=1, seed=, regime=)`, scales the channel-3 z-scored returns by `volatility_scale=0.015`, applies TASK_745 per-rotation column remap, and exposes the result as `cache_data: (T, n_assets, 5)`.
3. **Three modes** controlled by `--bella-augment-mode {replace, append, interleave}`. **v1 implements `append` only** (spec §10 out-of-scope). At each slot-construction call in `SegmentPool._make_slot`, with probability `--bella-augment-ratio` (default 0.25), the new code calls `_make_bella_slot(...)` and inserts that into the pool alongside (not replacing) the historical slot.
4. **Slot dict schema unchanged** — `_make_bella_slot` returns the same dict keys (`hm`, `cursor`, `max_cursor`, `oracle_selections`, `oracle_cum_pnl`, `returns`, `state_channels`, `policy_targets`, `confidence`, plus `bella_preds_full=None`, `bella_shape_full=None`). All precompute (oracle path solver, state channels, policy targets, confidence) runs on the Bella `cache_data` exactly as for real slots. **`AsyncBatchPipeline` and the JIT'd train step do not need to know whether a slot is Bella-sourced.**
5. **Logging additions** — per-epoch CSV/E-line gain `bella_slot_frac` (rolling fraction of slots that are Bella) and `bella_regime_mix` (counts per regime ID).

### New CLI flags (10 total)
All on `train_phase2.py`, defaults preserve current behaviour:
- `--bella-augment` (master)
- `--bella-augment-ratio` (default 0.25)
- `--bella-augment-mode` (default `append`; only `append` implemented in v1)
- `--bella-augment-regime-mode` (default `uniform`)
- `--bella-augment-checkpoint` (default `checkpoints/diff_teacher/generator_best`)
- `--bella-augment-k-samples` (default 1)
- `--bella-augment-seq-length` (default 1440 per spec — **but see C2**)
- `--bella-augment-refresh-epochs` (default 0 = generate once)
- `--bella-augment-volatility-scale` (default 0.015)
- `--bella-augment-asset-policy` (default `remap`)
- `--bella-augment-log-every` (default 100)

### New module
- `mcts_trader/data/bella_market.py` — `BellaMarket` class + lifted column-remap helpers (`_load_bella_canonical_assets`, `_rotation_hash`, `_build_remap`) imported from `tools/precompute_bella_predictions.py` (currently at lines 60-67, 70-102, 437-449 post-TASK_754).

---

## 2. Feasibility verdict

**PASS-with-caveats.** No load-bearing flaw; the slot abstraction is exactly the right seam, and TASK_745 verified column remap is bit-exact (`runs/745_spot_check/verdict.md`). The spec correctly identifies that augmentation sidesteps the classifier-Alpha anti-correlation pattern (since gradient pathway is unchanged) and re-uses the existing `SyntheticMarket` precedent (`mcts_trader/data/synthetic_market.py:5`, imported `train_easy.py:60`).

Nine caveats, of which **C1 (Bella seq_length mismatch), C2 (architecture facts wrong), and C7 (`_exhausted_groups` AttributeError latent bug)** must be resolved before Phase 2; the rest are addressable in-step.

---

## 3. Caveats and spec gaps

### C1 (MAJOR) — Bella's actual `seq_length` is 400, not 1440
Spec §4.4 and §5 (`--bella-augment-seq-length` default 1440) assume Bella natively emits 1440-step trajectories per `generate_on_device` call. **The loaded checkpoint disagrees.**

Verified at `checkpoints/diff_teacher/generator_best/config.json`:
- `window_size: 400` (this is what `setup_bella_generator` at `mcts_trader/training/setup.py:598` maps to `GeneratorConfig.seq_length`)
- `flow_hidden_dim: 768`, `flow_layers: 12` (not 1536d/16L as spec §3 claims)

And `generate_on_device` (`diff_teacher/generation/generator.py:271-321`) **uses `self.config.seq_length`** with no kwarg to override (line 317). Only `MarketGenerator.generate()` (line 153) accepts a `seq_length` override. So in the current ckpt, each Bella call returns **400 timesteps**, not 1440.

**Consequence**:
- Spec §4.4 budget ("~30 batches' worth of training data per slot at ws=60, stride=8") falls from ~30 to ~6 per Bella slot. The pool-init cost (~400 ms for 8 slots) stays cheap, but the per-slot information density is much lower.
- The trainer's `SegmentPool` minimum-period contract (`min_assets=32`, `window_length`-aware bounds in `_make_slot` at `train_easy.py:587, 685`) needs `T ≥ window_length + oracle_lookahead`. With workstation defaults (`config.episode_length=400`, `window_size=60`, `oracle_lookahead=20`, +5 buffer → `window_len ≈ 485` in `train_phase2.py:449`), a **400-step Bella slot is too short to be a valid pool slot.**

**Mitigations** (pick one; document in implementation):
- **(a) Use `generator.generate()` with `seq_length=1440` override** instead of `generate_on_device(...)`. This is the cleanest path; `generate()` accepts the kwarg. Cost: bumps generation through full Python boundary (vs JIT'd `generate_on_device`), but Bella generation isn't in the hot loop here.
- **(b) Chain multiple `generate_on_device` calls** and stitch them (gives 400, 800, 1200, ... steps). Cheap, but introduces discontinuity at seams. Document and test that the path solver tolerates the seam.
- **(c) Lower `--bella-augment-seq-length` default to 800** (2× the native 400) and use 2 chained calls. Aligns with the existing precedent in `mcts_trader/training/setup.py:614+` which calls `generate_on_device` once for 400 steps.

**Recommendation**: option (a). State explicitly in step 1 that `BellaMarket` calls `generator.generate(n_samples=1, seq_length=cfg.bella_augment_seq_length, seed=, regime=)`, NOT `generate_on_device`. Document the JIT-cost tradeoff. If wall blows past the 1.3× ceiling at the smoke, fall back to option (b).

### C2 (MAJOR) — Spec's stated Bella architecture facts are stale/wrong
Spec §3 cites `diff_teacher/CLAUDE.md:12` for "OT-CFM flow matching, conv generator (302M params, 1536d, 16L), 1440-step generation window, 5 regimes + 1 unconditional, single-step inference."

Actual values from `checkpoints/diff_teacher/generator_best/config.json` and `diff_teacher/generation/generator.py:177, 244`:
- `flow_hidden_dim: 768` (not 1536) — so the checkpoint is the smaller variant
- `flow_layers: 12` (not 16)
- `n_steps: 10` ODE integration steps in `generator.py:41`, NOT "single-step" inference (spec §4.4 cost of "~50 ms per K=1 sample at 1440 steps" should be re-measured).

This affects:
- VRAM budget in §4.3 ("~600 MB at fp16" for 302M params) — with 768d/12L the model is closer to ~100-150M params, ~200-300 MB fp16. Lower budget pressure, but still material.
- Generation latency in §4.4 — 10 ODE steps × convs at 768d, not single-step. Probably ~20-80 ms per call on a 3090; **must be measured at the smoke step, not assumed.**

**Mitigation**: implementation must NOT take §3/§4.4 numbers as inputs to config defaults; instead, log actual generator architecture and wall on the smoke step's first-Bella-slot construction.

### C3 (MAJOR) — Oracle path solver may produce degenerate (trivially profitable) targets on Bella synthetic OHLCV
Spec R6 flags this as MODERATE; I think it's MAJOR. Mechanism: Bella is OT-CFM (probability flow), smooth in the prediction surface. The greedy lookahead path solver (`train_easy.py:751`) finds the optimal top-k forward path over a 20-step lookahead, which on a synthetic series with low-noise predictable structure becomes near-perfect — the BC target is "always pick the top-mover", which is trivially learnable.

Consequence: the **encoder learns a Bella-shaped policy** under augmented slots; if Bella's regime distribution diverges from real markets in the bottom-k assets (the ones the model trades around), the policy is well-fit to Bella, badly-fit to real. This is exactly the failure mode `project_bull_blindness.md` warns about for synthetic data (`SyntheticMarket` levels 8-10 momentum/breakout). The legacy `SyntheticMarket` route mitigates this by training-only-on-synthetic in Phase 1 and then doing all eval on real — TASK_753 *mixes* in Stage B, so a bad Bella-policy will leak into real Alpha.

**Mitigation**: spec already proposes a per-slot oracle-PnL sanity ratio (R6 mitigation). Implementation must include this in the per-Bella-slot log line and the smoke pass criteria (e.g., "if Bella-slot oracle cum-PnL is > 2× real-slot oracle cum-PnL, log a warning AND tag the experiment status as 'too easy'"). Recommend: add this to the §4 step plan and the §7 stop-point checklist below.

### C4 (MAJOR) — Bella outputs z-scored OHLCV, not log-returns; the volatility_scale conversion is non-trivial
Spec §4.3 ("channel 3 carries z-scored per-step returns") + §5 (`--bella-augment-volatility-scale 0.015`) implies a 1-line transform. But the existing path that converts Bella raw output → trader-ready OHLCV is **`generate_bella_prices_on_device` at `mcts_trader/training/setup.py:614-704`** — 90 lines. It:
- Channel 3 (close) → cumsum(`raw * 0.015`) → exp() → multiplied by 100
- Channel 0 (open) → previous-close shift
- Channel 1 (high) → `abs(raw) * 0.015` offset above close
- Channel 2 (low) → `abs(raw) * 0.015` offset below close
- Channel 4 (volume) → exp(raw * 0.5 + 10.0) lognormal

The spec's simplistic "channel 3 carries z-scored per-step returns × 0.015" loses the OHLCV-constraint logic. **`BellaMarket` MUST call (or replicate) `generate_bella_prices_on_device(ohlcv=True)`, not just `np.array(generated)`** as suggested at `tools/precompute_bella_predictions.py:167`.

**Mitigation**: step 1 of the implementation plan below imports `generate_bella_prices_on_device` from `mcts_trader/training/setup.py` and uses its output directly. Add an explicit assertion in `BellaMarket.__init__`: `assert cache_data.shape == (seq_length, n_assets, 5)` and `assert (cache_data[..., :4] > 0).all()` (positivity) and `assert (cache_data[..., 1] >= cache_data[..., 3]).all()` (high ≥ close).

### C5 (MODERATE) — Spec line numbers are stale post-TASK_754
The spec was written before TASK_754 landed at `9aa4885`. Spec §3 cites:
- `train_easy.py:453-720` (SegmentPool) → still valid, but `_make_slot` now ends at **735** (not 720); `_load_bella_shard` ends at 834; the new `_load_bella_shape_shard` runs **836-...** (and continues past 880, I didn't trace the close)
- `train_phase2.py:484-693` (AsyncBatchPipeline) → still valid; the bella_shape plumbing now occupies lines 569-585, 672-683, 716-717, 780-781
- `train_phase2.py:1768-1771` (CLI block) → STALE. TASK_754 added 23 lines of CLI at lines 2052-2076. Spec §6 step 4 wants to insert "near other Bella flags"; the actual location is now near `train_phase2.py:2076` (after `--reset-bella-shape-head`), not `1768-1771`.
- `train_phase2.py:1934-1938` (args→config translation) → STALE. The block now runs 2274-2327 (TASK_740 block 2274-2285, TASK_754 block 2286-2327).
- `phase2_config.py:160-180` → STALE. The "Bella" config block ends at line 133 (TASK_754 block runs 118-133); the SegmentPool config block (`stage_b_batch_size` etc) starts at 177.
- `train_phase2.py:2794` (SegmentPool construction site) → STALE. Pool is loaded via `load_segment_pool` at **2646** (called from line 2646 in main()).

**Mitigation**: per-step file:line citations in §4 below use *current* (post-TASK_754) line numbers.

### C6 (MODERATE) — File:line collisions with TASK_754
**These are the lines TASK_753 needs to also modify but where TASK_754 already added code.** Careful merge required at each:

| File | Line range | TASK_754 added | TASK_753 will add |
|------|------------|----------------|-------------------|
| `phase2_config.py` | 118-133 (Bella-shape config block) | 16 new fields | TASK_753: append ~12 new `bella_augment_*` config fields **after line 133, before line 135** (cross_sectional block). |
| `train_phase2.py` | 402-410 (create_model wiring) | passes `use_bella_shape_head` + dim + hidden_dim | TASK_753 needs nothing here (no new model arg — augmentation is data-side). |
| `train_phase2.py` | 458-471 (oracle_cfg dict for SegmentPool) | adds 5 `use_bella_shape_*` keys | TASK_753: append `bella_augment_*` keys **after line 470**, then close the dict. Easy merge. |
| `train_phase2.py` | 569-585 (`_prepare_numpy` accumulators) | adds `all_bella_shape = []` and `use_bella_shape`, `bella_shape_dim` locals | TASK_753: NO change here (augmentation is invisible at this layer). |
| `train_phase2.py` | 672-683 (per-slot Bella batch fetch) | adds `bella_shape` fetch with `bsh = slot.get('bella_shape_full', None)` | TASK_753: NO change here. |
| `train_phase2.py` | 716-717 (output dict) | adds `'bella_shape': ...` | TASK_753: NO change here. |
| `train_phase2.py` | 1583-1584 (CSV fields) | adds `bella_shape_*` fields | TASK_753: append 2 fields (`bella_slot_frac`, `bella_regime_mix`) **after line 1584**. |
| `train_phase2.py` | 1755-1762 (E-line log) | adds `shape={...}` block | TASK_753: append a `bella_aug=` block **before or after** (cosmetic; place after the `bella_shape_info` block). |
| `train_phase2.py` | 1811 (E-line composition) | appends `{bella_shape_info}` | TASK_753: append `{bella_aug_info}` at the end of the f-string. Avoid replacing `{bella_shape_info}` — both must coexist. |
| `train_phase2.py` | 1841-1844 (CSV writer dict) | adds 4 `bella_shape_*` rows | TASK_753: append `bella_slot_frac`, `bella_regime_mix` rows. |
| `train_phase2.py` | 2052-2076 (CLI block) | adds 8 `--bella-shape-*` flags | TASK_753: append the 10 `--bella-augment-*` flags **after line 2076** (after `--reset-bella-shape-head`). |
| `train_phase2.py` | 2286-2327 (args→config translation) | TASK_754 block | TASK_753: append the augment translation block **after line 2327**. |
| `train_phase2.py` | 2464-2470 (`reset_bella_shape_head` aux drop) | TASK_754 | TASK_753: no aux head to reset (augmentation is data-side). |
| `train_phase2.py` | 2492-2501 (aux-init-on-load trigger OR list) | adds `use_bella_shape_head` to OR | TASK_753: NO change (no aux head). |
| `train_easy.py` | 691-735 (`_make_slot` Bella-shard load + slot return) | adds `bella_shape_full` load + slot key | TASK_753: this is the **biggest collision**. TASK_753 must add a coin-flip after the slot is constructed (or before, in `_make_bella_slot`), and must guarantee that **both `bella_preds_full=None` and `bella_shape_full=None`** are returned by `_make_bella_slot` for the Bella-source slot (since Bella augmentation runs WITH `use_bella_shape_head=False` and the slot has no precomputed teacher target). |
| `train_easy.py` | 836-... (`_load_bella_shape_shard` method) | new ~70-line method | TASK_753: NO change. |

**Critical merge points** (where 753 must add lines INSIDE the 754 block):
1. `phase2_config.py` after line 133.
2. `train_phase2.py:471` (close of `oracle_cfg` dict; add new keys before the closing brace at 471).
3. `train_phase2.py:1811` (the composed f-string).
4. `train_phase2.py:2076` (CLI block end).
5. `train_phase2.py:2327` (args→config translation end).
6. `train_easy.py:691-735` (`_make_slot` return logic and the new coin-flip for Bella).

### C7 (MODERATE — latent bug; affects 753 but exists today) — `_exhausted_groups` attribute is never initialized
At `train_easy.py:551`: `self._exhausted_groups.add(group_key)`. But the attribute is never defined in `__init__`. This crashes the first time a slot fails to load on init. Currently untriggered because all foxglove periods load cleanly. **TASK_753 augmentation creates synthetic slots that may fail differently** (e.g., Bella ckpt not found, regime invalid). The implementation should defensively initialize `self._exhausted_groups = set()` in `__init__` before the slot construction loop, AND surface the latent bug as a tiny diff in step 2 of the plan. Don't paper over it.

### C8 (MODERATE) — Slot-pool oracle wall for synthetic slots may dominate the smoke
The spec under-budgets oracle precompute on Bella slots. `_precompute_oracle` at `train_easy.py:739-766` calls `greedy_lookahead_path` JIT'd with `max_len = max(262144, ...)` — even on a 400 or 1440-step Bella series, the JIT compile of `greedy_lookahead_path` is shared with the real-slot compile. So **first Bella slot adds zero JIT compile cost** if it has the same `max_len` bucket. But the slot init also computes `compute_target_position` + `build_state_channels` + policy targets + confidence — each is JIT'd over `(T_ret, n_assets)`. These can recompile if T_ret falls in a new bucket. **Verify the smoke that adding Bella slots does not double the SegmentPool init wall.**

**Mitigation**: in the smoke spec, log the per-slot init wall and assert it's ≤ 1.5× the historical-slot init wall.

### C9 (MINOR) — Mode-mutual-exclusion: `replace` and `interleave` reserved
Spec §10 says: "Implement `append` only in v1; document the other two as follow-up tasks." Implementation must raise `NotImplementedError("--bella-augment-mode 'replace' is reserved for v2 (TASK_753b)")` when set to anything other than `append`. Mirror the TASK_754 `bella_shape_mode` pattern at `train_phase2.py:2315-2321`. Defaulting silently to `append` would create a footgun.

---

## 4. Per-step implementation plan

Eight steps; each ends with a smoke that's <60s except step 8 (the production-batch full `train_step_fn` per the 749 OOM rule). Hardware: workstation single GPU (RTX 3090, 24 GB), use `CUDA_VISIBLE_DEVICES=0`.

### Step 0 — Pre-flight (≤ 0.2 h)
**No code changes.** Verify:
- `checkpoints/diff_teacher/generator_best/` exists on workstation (`ls`).
- Read its `config.json` to confirm `seq_length / window_size`, `flow_hidden_dim`, `flow_layers`. Update the implementation defaults to match (C2).
- `python -c "from mcts_trader.training.setup import setup_bella_generator; g, c = setup_bella_generator('checkpoints/diff_teacher/generator_best'); print(c.architecture, c.hidden_dim, c.num_layers, c.seq_length, c.n_assets, c.n_features); import jax; import jax.numpy as jnp; out = g.generate_on_device(n_samples=1, seed=0, regime=jnp.array([2])); print('out.shape=', out.shape, 'wall measured externally')"`
- If `generate_on_device` returns shape `(1, 400, 320)` or similar (reshape pending), record the seq_length and use that as the **lower-bound** for `--bella-augment-seq-length`. If 1440 is desired, switch to `generator.generate(seq_length=1440)` per C1.

**Smoke**: shell command above. Wall: ~30s (compile + 1 forward). Pass: prints architecture matches checkpoint config; output shape sensible.

### Step 1 — `BellaMarket` class skeleton (≤ 1.5 h)
**File:** new `mcts_trader/data/bella_market.py` (~150 LOC).
- Implement `BellaMarket` class with:
  - Constructor: `(generator, cfg, seq_length, n_assets, regime, seed, volatility_scale, asset_policy, period_assets, db_path)`.
  - Calls `generate_bella_prices_on_device(generator, key, n_envs=1, ohlcv=True)` from `mcts_trader/training/setup.py:614` (lifts the OHLCV reconstruction; do NOT reimplement). If C1 requires it, swap to `generator.generate(seq_length=...)` and apply the OHLCV reconstruction logic from `setup.py:660-700` inline.
  - Apply column remap via lifted `_load_bella_canonical_assets` + `_rotation_hash` + remap construction (from `tools/precompute_bella_predictions.py:60-67, 70-102, 380-398`). When `asset_policy='remap'`, NaN columns get **cyclic-pad filled from neighbouring valid columns** (since the encoder cannot consume NaN — this differs from TASK_745 distill, which masks NaNs at loss-time).
  - Expose `self.cache_data: np.ndarray (T, n_assets, 5)`, `self.assets: dict[int, str]`, `self.asset_ids: list[int]`, `self.symbol_to_id: dict[str, int]`, `self.id_to_symbol: dict[int, str]`, `self.valid_timestamps: list[int]` (just `range(T)`), `self.min_time = 0`, `self.max_time = T - 1`, `self.n_assets`. `close_db()` is a no-op.
  - Assertions: `cache_data.shape == (seq_length, n_assets, 5)`, no NaN after fill, all prices > 0 (per C4).

**Smoke**:
```bash
CUDA_VISIBLE_DEVICES=0 python -c "
import sys; sys.path.insert(0, '.')
from mcts_trader.training.setup import setup_bella_generator
from mcts_trader.data.bella_market import BellaMarket
gen, cfg = setup_bella_generator('checkpoints/diff_teacher/generator_best')
period_assets = ['BTC/USD', 'ETH/USD', 'XRP/USD']  # tiny set; remap should NaN-fill
bm = BellaMarket(generator=gen, cfg=cfg, seq_length=cfg.seq_length, n_assets=64,
                 regime=2, seed=7, volatility_scale=0.015, asset_policy='remap',
                 period_assets=period_assets, db_path='trading_data.db')
import numpy as np
assert bm.cache_data.shape == (cfg.seq_length, 64, 5), bm.cache_data.shape
assert np.isfinite(bm.cache_data).all(), 'NaN survived fill'
assert (bm.cache_data[..., :4] > 0).all(), 'non-positive prices'
print('OK', bm.cache_data.shape, bm.cache_data.dtype, 'mean=', bm.cache_data[..., 3].mean())
"
```
Wall: ~30-90s (Bella load + 1 generation + remap). Pass: shape matches, all-finite, all-positive close prices.

### Step 2 — Config fields (≤ 0.5 h)
**File:** `phase2_config.py` — insert after line 133 (after TASK_754 bella_shape block), before line 135 (`use_cross_sectional`).
- Add 12 new fields with sensible defaults:
  ```
  bella_augment: bool = False
  bella_augment_ratio: float = 0.25
  bella_augment_mode: str = 'append'
  bella_augment_regime_mode: str = 'uniform'
  bella_augment_checkpoint: str = 'checkpoints/diff_teacher/generator_best'
  bella_augment_k_samples: int = 1
  bella_augment_seq_length: int = 0   # 0 = use generator's native seq_length (resolves C1)
  bella_augment_refresh_epochs: int = 0
  bella_augment_volatility_scale: float = 0.015
  bella_augment_asset_policy: str = 'remap'
  bella_augment_log_every: int = 100
  ```
- Also: **fix C7 latent bug** in `train_easy.py:540` — insert `self._exhausted_groups = set()` right before line 541 (the slot init loop).

**Smoke**:
```bash
python -c "
import sys; sys.path.insert(0, '.')
from phase2_config import Phase2Config
cfg = Phase2Config(bella_augment=True, bella_augment_ratio=0.25)
assert cfg.bella_augment is True
assert cfg.bella_augment_ratio == 0.25
assert cfg.bella_augment_mode == 'append'
print('OK')
"
```
Wall: <2s. Pass: import + field access succeeds.

### Step 3 — `_make_bella_slot` method in `SegmentPool` (≤ 1.5 h)
**File:** `train_easy.py` — insert new method between `_make_slot` (ends ~735) and `_precompute_oracle` (~739).
- Signature: `_make_bella_slot(self, group_key, regime_idx, seed)`.
- Body: identical to `_make_slot` from line 593 onward (oracle precompute, state channels, policy targets, confidence), but with `hm = BellaMarket(...)` instead of `hm = HistoricalMarket(...)`. Return same dict schema; `bella_preds_full = None`, `bella_shape_full = None` per C6.
- Lazy-init `self._bella_generator` and `self._bella_canonical_assets` on first call (so non-augment runs pay zero load cost). Cache as `self._bella_generator: MarketGenerator | None`, `self._bella_cfg: GeneratorConfig | None`.
- Track regime mix: `self._bella_regime_counts = collections.Counter()`; bump per `_make_bella_slot` call.

**Smoke**: Construct a SegmentPool with a 1-period config, manually invoke `_make_bella_slot`:
```bash
CUDA_VISIBLE_DEVICES=0 python -c "
import sys; sys.path.insert(0, '.')
import os
os.environ['JAX_PLATFORMS'] = 'cuda'
from train_easy import SegmentPool
period_spec = [{'start_date': '2018-10-17', 'end_date': '2019-04-15',
                'assets': ['BTC/USD','ETH/USD','LTC/USD','XMR/USD','XRP/USD',
                           'ETC/USD','BCH/USD','EOS/USD','ADA/USD','DASH/USD']}]
oc = {'oracle_k': 4, 'oracle_lookahead': 20, 'transaction_cost': 0.001,
      'min_alpha_threshold': 0.0005, 'oracle_mode': 'topk_path',
      'bella_augment': True,
      'bella_augment_checkpoint': 'checkpoints/diff_teacher/generator_best',
      'bella_augment_asset_policy': 'remap',
      'bella_augment_volatility_scale': 0.015,
      'bella_augment_k_samples': 1,
      'bella_augment_seq_length': 0,
      'bella_augment_regime_mode': 'uniform'}
pool = SegmentPool(period_spec, 'trading_data.db', n_assets=64, n_slots=1,
                   window_length=200, max_runs_per_group=1, min_assets=3,
                   oracle_config=oc)
slot = pool._make_bella_slot(group_key=tuple(period_spec[0].values())[:2], regime_idx=2, seed=42)
assert slot is not None, 'Bella slot construction failed'
assert 'hm' in slot and slot['hm'] is not None
assert slot['oracle_selections'].ndim == 2
import numpy as np
assert np.isfinite(slot['hm'].cache_data).all()
print('OK Bella slot constructed:', slot['hm'].cache_data.shape, 'oracle trades=', int(np.sum(np.diff(slot['oracle_selections'], axis=0) != 0)/2))
"
```
Wall: ~60-120s (DB load + Bella load + Bella generation + oracle solve). Pass: slot is non-None, oracle precompute completes, no NaN.

### Step 4 — Coin-flip injection in `SegmentPool` init + `_rotate_slot` (≤ 1 h)
**File:** `train_easy.py` — modify `__init__` (~line 540-560) and `_rotate_slot` (~line 990-1033).
- In `__init__`: add `self.bella_augment_ratio = oracle_config.get('bella_augment_ratio', 0.0)`, `self.bella_augment_mode = oracle_config.get('bella_augment_mode', 'append')`, etc.
- At slot init (line 544-551): for each slot, with probability `bella_augment_ratio`, call `self._make_bella_slot(group_key, regime, seed)` instead of `self._make_slot(group_key)`. Track in `self._slot_origins: list[bool]` (True = Bella, False = real).
- In `_rotate_slot`: when a Bella slot expires (which it shouldn't, given `is_exhausted` returns False — but at the 20-epoch rotation in `train_phase2.py:1624-1630` the slot is regenerated), preserve its Bella-source nature with a new coin flip on the replacement.
- **`replace` and `interleave` modes**: raise `NotImplementedError(f"--bella-augment-mode '{mode}' is reserved for v2 (TASK_753b)")` — per C9.

**Smoke**: same as step 3 (same `period_spec`, `oc`) but set `bella_augment_ratio=1.0`, `n_slots=2`, and inspect `pool._slot_origins`. Pass: all slots Bella (`all(origins) == True`); log shows 2 BellaMarket constructions. Wall ~120s.

### Step 5 — CLI flags + args→config translation (≤ 0.5 h)
**File:** `train_phase2.py`.
- Insert the 10 `--bella-augment-*` flags **after line 2076** (after `--reset-bella-shape-head`).
- Insert args→config translation **after line 2327** (after the TASK_754 mode validation block):
  ```
  # TASK_753: Bella-as-data augmentation
  if args.bella_augment:
      config.bella_augment = True
  if args.bella_augment_ratio is not None:
      config.bella_augment_ratio = args.bella_augment_ratio
  # ... (mirror for all 10 flags)
  # Mode validation
  if getattr(config, 'bella_augment', False):
      if config.bella_augment_mode not in ('append',):
          raise NotImplementedError(
              f"TASK_753 v1: only --bella-augment-mode 'append' is implemented; "
              f"got '{config.bella_augment_mode}'. {{replace, interleave}} reserved.")
      if config.bella_augment_asset_policy not in ('remap', 'canonical', 'random'):
          raise ValueError(f"Invalid asset_policy '{config.bella_augment_asset_policy}'")
      logger.info(f"Bella augmentation ON (ratio={config.bella_augment_ratio}, "
                  f"mode={config.bella_augment_mode}, regime={config.bella_augment_regime_mode}, "
                  f"policy={config.bella_augment_asset_policy})")
  ```

**Smoke**: parse-only test:
```bash
CUDA_VISIBLE_DEVICES=0 python train_phase2.py --help 2>&1 | grep -E "bella-augment" | head -12
```
Wall: <10s. Pass: all 10 `--bella-augment-*` flags appear in `--help`.

### Step 6 — `load_segment_pool` wiring (≤ 0.5 h)
**File:** `train_phase2.py` — append to `oracle_cfg` dict at `train_phase2.py:450-470` (insert **before** the closing brace at line 471):
```
'bella_augment': getattr(config, 'bella_augment', False),
'bella_augment_ratio': getattr(config, 'bella_augment_ratio', 0.0),
'bella_augment_mode': getattr(config, 'bella_augment_mode', 'append'),
'bella_augment_regime_mode': getattr(config, 'bella_augment_regime_mode', 'uniform'),
'bella_augment_checkpoint': getattr(config, 'bella_augment_checkpoint',
                                    'checkpoints/diff_teacher/generator_best'),
'bella_augment_asset_policy': getattr(config, 'bella_augment_asset_policy', 'remap'),
'bella_augment_volatility_scale': getattr(config, 'bella_augment_volatility_scale', 0.015),
'bella_augment_k_samples': getattr(config, 'bella_augment_k_samples', 1),
'bella_augment_seq_length': getattr(config, 'bella_augment_seq_length', 0),
'bella_augment_refresh_epochs': getattr(config, 'bella_augment_refresh_epochs', 0),
'bella_augment_log_every': getattr(config, 'bella_augment_log_every', 100),
```

**Smoke**: dry-run the pool construction with augmentation on:
```bash
CUDA_VISIBLE_DEVICES=0 python train_phase2.py --stage B --dry-run \
  --periods-file /tmp/foxglove_periods_smoke_1period.json \
  --n-slots 2 --max-runs-per-group 1 \
  --bella-augment --bella-augment-ratio 0.5 --bella-augment-mode append \
  --output-dir runs/000_smoke/753_step6 \
  --name 000_753_step6 2>&1 | tail -30
```
Wall: ~60-180s (pool init with one Bella slot). Pass: dry run completes; log shows "Bella augmentation ON"; `_slot_origins` mix is ~50%.

### Step 7 — Logging + CSV (≤ 0.5 h)
**File:** `train_phase2.py`.
- Insert `bella_slot_frac`, `bella_regime_mix_str` (e.g., "c=0,r=1,s=2,v=0,k=1" 5-tuple) into CSV fields after line 1584 (after `bella_shape_std_resid`).
- Compute `bella_slot_frac = mean(pool._slot_origins)` per epoch; `bella_regime_mix` from `pool._bella_regime_counts`.
- Append `bella_aug_info` to the E-line composition at line 1811:
  ```
  bella_aug_info = ""
  if getattr(config, 'bella_augment', False):
      bella_aug_info = f" bella_aug=slot{bella_slot_frac:.2f} mix={bella_regime_mix_str}"
  ```
- Add the f-string to the log composition at line 1811: append `{bella_aug_info}` after `{bella_shape_info}`.
- Add CSV row entries at line 1841-1844 region: `'bella_slot_frac': f"{bella_slot_frac:.4f}"`, `'bella_regime_mix': bella_regime_mix_str`.

**Smoke**: rerun step 6 with `--epochs 1 --steps-per-epoch 1` and confirm E-line and CSV contain new fields:
```bash
CUDA_VISIBLE_DEVICES=0 python train_phase2.py --stage B \
  --epochs 1 --steps-per-epoch 1 --batch-size 32 \
  --periods-file /tmp/foxglove_periods_smoke_1period.json \
  --n-slots 2 --max-runs-per-group 1 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.5 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 --aux-price-weight 0.1 \
  --bella-augment --bella-augment-ratio 0.5 --bella-augment-mode append \
  --output-dir runs/000_smoke/753_step7 \
  --name 000_753_step7 2>&1 | tail -30
```
Wall: ~3-5 min (compile + 1 step). Pass: log shows `bella_aug=slot0.50 mix=...`; CSV file at `runs/000_smoke/753_step7/stage_b_metrics.csv` includes the new columns.

### Step 8 — Full `train_step_fn` smoke at production batch (the 749 OOM rule) (≤ 0.5 h)
**File:** no code change — full integration smoke.

**Goal**: Exercise the JIT'd `stage_b_train_step` at **production batch (128)** and full temporal window (60), with Bella augmentation ratio 0.25 and `append` mode, to flush JIT compile / OOM characteristics. Per CLAUDE.md the 749 OOM rule: smoke must exercise full train_step at production batch, not minimal config.

**Smoke (FULL)**:
```bash
CUDA_VISIBLE_DEVICES=0 python train_phase2.py --stage B \
  --epochs 1 --steps-per-epoch 5 --batch-size 128 \
  --periods-file config/foxglove_periods.json \
  --n-slots 4 --max-runs-per-group 1 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.5 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 --aux-price-weight 0.1 \
  --bella-augment --bella-augment-ratio 0.25 --bella-augment-mode append \
  --bella-augment-regime-mode uniform --bella-augment-asset-policy remap \
  --output-dir runs/000_smoke/bella_augment_smoke \
  --name 000_bella_augment_smoke 2>&1 | tee runs/000_smoke/bella_augment_smoke/train.log
```
Wall: ~8-15 min total (Bella load ~30-60s + 4 slot inits @ ~30-60s each + ~5-10 min compile + ~30-60s training). Pass criteria (all must hold):
1. **No OOM**, peak VRAM ≤ 22 GB on the 3090.
2. `BellaMarket` constructs successfully — log shows ≥ 1 Bella slot in the pool (`_slot_origins` log).
3. Bella generator load time ≤ 60s (one-time at pool init).
4. `total_loss` finite for all 5 steps.
5. `bella_slot_frac` recorded in CSV at value ≈ `bella_augment_ratio` (0.25) ± 0.15.
6. Per-epoch wall (5 steps) ≤ 1.3× the same command with `--bella-augment` removed (run the baseline once first; the multiplier must be measured).
7. No `nan` in any slot's `cache_data` after BellaMarket construction.
8. CSV contains `bella_slot_frac`, `bella_regime_mix` columns.
9. **Per C3 sanity check**: log a Bella-slot-oracle vs real-slot-oracle cum-PnL ratio. If > 2.0, log WARN (smoke still passes; this is informational, gating happens at the 3-seed eval).

If step 8 passes: TASK_753 implementation complete; ready for the paired 3-seed launch by server agent under `experiments/758_bella_augment/SPEC.md` (or next free numerical ID — verify via `ls experiments/` first; current highest is 757_bella_shape).

---

## 5. Stop-point checklist

When implementing Phase 2, check each box after the step's smoke passes. If any FAIL, write `tasks/TASK_753_STOP_<step>.md` with the failure mode + observed output and STOP.

- [ ] **Step 0 (pre-flight):** Bella ckpt exists at `checkpoints/diff_teacher/generator_best/`. `setup_bella_generator` returns a generator. Observed `cfg.seq_length`, `cfg.n_assets`, `cfg.n_features` match expectations. Architecture facts in spec §3 confirmed or corrected (C2). Generation wall measured (not assumed).
- [ ] **Step 1 (`BellaMarket`):** Class constructs with default args, `cache_data.shape == (cfg.seq_length, 64, 5)`, all-finite (NaN-fill works), all-positive close prices (C4 assertion). `volatility_scale=0.015` produces close prices in the `[50, 200]` band (matches `_apply_noise` clamp at `synthetic_market.py:213`).
- [ ] **Step 2 (Config + C7 fix):** `Phase2Config(bella_augment=True)` round-trips; all 12 fields accessible. `_exhausted_groups = set()` initialized in `SegmentPool.__init__` (C7 latent-bug fix shipped).
- [ ] **Step 3 (`_make_bella_slot`):** Single Bella slot constructs; oracle precompute completes; `slot['hm'].cache_data` shape matches; `bella_preds_full = None` and `bella_shape_full = None` are set (C6 collision guard).
- [ ] **Step 4 (coin-flip injection):** At `ratio=1.0`, all slots are Bella; at `ratio=0.0`, all slots are real; intermediate ratios produce ~ratio fraction. `_rotate_slot` preserves Bella-source flag with a new coin flip. `replace` / `interleave` modes raise `NotImplementedError` (C9).
- [ ] **Step 5 (CLI + translation):** All 10 `--bella-augment-*` flags appear in `--help`. Config translation validates mode + asset policy. Mutual-exclusion: no clash with TASK_740 / TASK_754 (this task adds zero head; no need to check).
- [ ] **Step 6 (`load_segment_pool`):** Dry-run with `--bella-augment` succeeds; pool init log shows expected Bella slot count.
- [ ] **Step 7 (Logging + CSV):** E-line includes `bella_aug=slot{frac} mix=...`; CSV has the 2 new columns; `bella_slot_frac` matches expected ratio.
- [ ] **Step 8 (LOAD-BEARING — full train_step at b=128):** All 9 pass criteria in §4 step 8 hold. Per-epoch wall ≤ 1.3× no-augment baseline. Peak VRAM ≤ 22 GB. `total_loss` finite. **Per-Bella-slot oracle PnL vs real-slot oracle PnL ratio logged** (C3); if > 2.0, smoke logs WARN but still passes (decision gated at 3-seed eval).
- [ ] **Final commit:** "TASK_753 implementation complete" — list files touched, smoke results pasted.
- [ ] **SPEC creation:** `experiments/758_bella_augment/SPEC.md` (next free numerical ID per `ls experiments/`; current highest 757_bella_shape). Includes paired 3-seed config (seeds 7/13/23 against `runs/741_mint/baseline_seed_{7,13,23}/`), server-launch commands (DO NOT launch), eval command, done-flag names, what to look for in `result.md`.

---

## Sign-off

Verdict: **PASS-with-caveats (9 caveats — C1, C2, C3, C4, C7 require pre-implementation decisions or trivial defensive code; C5/C6/C8/C9 are addressable in-step).**

Phase 2 (implementation) is gated on operator approval and confirmation that:
- C1: the actual `cfg.seq_length` is used (or `generator.generate(seq_length=...)` is invoked) — NOT `1440` as a hard default.
- C2: the architecture-fact reliance in `--help` text and log messages reflects the actual checkpoint, not the spec's stale numbers.
- C4: `BellaMarket` calls `generate_bella_prices_on_device` (or replicates its OHLCV reconstruction), not the simplistic "channel 3 × volatility_scale" path from `tools/precompute_bella_predictions.py:166-167`.

After approval, the implementation will follow the per-step plan above with one commit per step (message format `TASK_753 step N/M: <description>`), final commit `TASK_753 implementation complete`, and the experiment SPEC at `experiments/758_bella_augment/SPEC.md` (or next free numerical ID).

**Single-sentence biggest risk**: If C1 is not surfaced, `BellaMarket` will silently emit 1440-step tensors that the generator can't produce in `generate_on_device` mode (it'll return 400-step tensors), the assertion `cache_data.shape == (1440, 64, 5)` will fail at construction, and the smoke will FAIL immediately — best-case noisy, worst-case the assertion is removed in haste and the slot has truncated/wrong-shaped data that crashes the oracle JIT.

Estimated total implementation time: **~6 hours of agent work** (Step 0: 0.2 h, Step 1: 1.5 h, Step 2: 0.5 h, Step 3: 1.5 h, Step 4: 1.0 h, Step 5: 0.5 h, Step 6: 0.5 h, Step 7: 0.5 h, Step 8: 0.5 h, SPEC + final commit: 0.3 h).

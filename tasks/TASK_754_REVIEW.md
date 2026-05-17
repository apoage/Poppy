# TASK_754 — Implementation Review (Phase 1)

**Reviewer:** Claude (workstation agent, 2026-05-17).
**Reviewed:** `tasks/TASK_754_BELLA_TRAJECTORY_SHAPE_DISTILL.md` (commit `f44c1d3`).
**Verdict:** PASS-with-caveats (6 caveats — see §3).

---

## 1. Understanding

### What is being built
A new auxiliary head + offline precompute path that replaces TASK_740's per-asset per-step Bella distillation target with a **permutation-invariant 14-float "trajectory-shape descriptor"** per training window. The motivation is the project's well-documented classifier-vs-Alpha anti-correlation (`memory/project_classifier_alpha_anti_correlation.md`): sharpening per-asset prediction targets via auxiliary loss reliably hurts deployment Alpha (TASK_737/738/740/741 all confirm). The coarser, permutation-invariant target is hypothesised to (a) extract regime-shape signal Bella can carry, while (b) avoiding the mechanism class that anti-correlates with Alpha.

### Math (descriptor, D_desc=14)
For each training window `t`, given Bella's precomputed predictions `preds[t]` of shape `(K, n_assets)` where `K=4` horizons `(1, 5, 10, 20)` and `n_assets=64`:

1. **Cross (asset × horizon) moments — 4 floats:** `(mean, std, skew, kurt)` of `preds[t].flatten()` (shape `K*n_assets = 256`).
2. **Horizon-1-only moments — 4 floats:** `(mean, std, skew, kurt)` of `preds[t, 0, :]` (shape `n_assets = 64`).
3. **Horizon-20-only moments — 4 floats:** `(mean, std, skew, kurt)` of `preds[t, K-1, :]` (shape `n_assets = 64`).
4. **SVD spectrum of `preds[t]` (`K × n_assets`) — 2 floats:** `(s1, s1 / sum(s_i))` where `s_i` are singular values, sorted descending.

Total: `4 + 4 + 4 + 2 = 14` floats per window. Permutation-invariant under asset permutation because (a) moments are sums over the asset axis, (b) SVD spectrum (singular values) is invariant under column permutation of the matrix.

### Loss
Per-component-weighted MSE:
```
L_shape = (1/D_desc) * Σ_j w_j * (head(z_t)[j] - desc[t][j])^2
```
with default weights `w = (mean=2.0, std=1.0, skew=0.5, kurt=0.25, s1=1.0, s1_ratio=2.0)` and `w_j` broadcast over each component-group's component count (the 4 cross moments share `(2.0, 1.0, 0.5, 0.25)`, the two horizon-restricted groups each get the same 4 weights, the SVD pair gets `(1.0, 2.0)`).

Sample-wise NaN mask (mirrors TASK_740 pattern at `train_phase2.py:1183-1192`) — when descriptor row is NaN (period missing from manifest), drop that sample from the loss reduction.

### Dataflow
- **Precompute side** (`tools/precompute_bella_predictions.py`): compute `desc[t]` of shape `(T_ret, 14)` per period at precompute time, using the same per-window Bella rollout that already exists. Apply TASK_745 Option A column remap *before* moment computation (paranoia — moments are permutation-invariant, but the (B)/(C) variants are not). Emit a new shard family `bella_shape_<period>__<rotation_hash>.npz` (additive — does not touch the existing `bella_preds_<...>.npz`). Emit new manifest `config/bella_shape_manifest.json`. ~150 MB total across all 270 rotations × period-lengths.
- **Trainer side** (`train_easy.py::SegmentPool._load_bella_shape_shard` + `train_phase2.py::AsyncBatchPipeline._prepare_numpy`): load shape shard once per slot at slot-instantiation, batch-index by `decision_indices` (mirrors the existing `bella_preds` path), pass `bella_shape: (B, 14)` through to the JIT'd train step.
- **Head** (`mcts_trader/jax_agent/muzero_model.py::AuxBellaShapeHead`): `latent_state (B, state_dim) → Dense(256) → GELU → LayerNorm → Dense(14, zero-init)`. Mirrors `AuxBellaHead` (lines 110-144) exactly except for output dim.
- **Loss site** (`train_phase2.py:~1175-1196` region): a new branch under `use_bella_shape_distill` that computes the per-component-weighted MSE with the NaN-mask pattern. Mutually exclusive with `use_bella_distill_head` (raise `ValueError` at config-resolve time).

---

## 2. Feasibility verdict

**PASS-with-caveats.**

The spec is implementable as written. The head class, loss site, batch plumbing, CLI flags, and precompute extension all follow established patterns in this codebase (`AuxPriceHead`/`AuxBellaHead`/740 plumbing). The math is well-defined. Memory cost is strictly lower than TASK_740 (14 floats vs 256·K per window).

Six caveats need clarification or minor fixes before implementation. None are load-bearing; all are addressable in-step without re-scoping the task. See §3.

---

## 3. Caveats and spec gaps

### C1 (MAJOR) — Bella shards do not exist on workstation
The spec's smoke command (§7 of TASK_754) implicitly assumes the precomputed Bella shape shards live on the workstation. They do not. The TASK_740 SPEC explicitly states `"Do NOT rsync data/bella_preds/ to workstation (60+ MB, unnecessary)"` — and that 60 MB is the *legacy* shard family. The TASK_754 shape shards must be precomputed too, and the precompute requires Bella's checkpoint at `checkpoints/diff_teacher/generator_best/generator`. **Verify before committing to implementation that this path exists on workstation; if not, the smoke must run server-side, contradicting the workstation-only constraint.**

Mitigation: The `runs/745_spot_check/verdict.md` artifact (and `/tmp/745_spotcheck/bella_preds_2018-10-17__2019-04-15__a7185523.npz`) shows we have *one* legacy shard locally already. Step 1 of the implementation plan (below) explicitly verifies Bella ckpt presence and falls back to a 1-period precompute for the period whose 740 shard is locally cached, rather than the full 270.

### C2 (MAJOR) — Smoke spec uses non-production batch (violates load-bearing rule)
The spec's smoke command at line 161-176 uses `--batch-size 64 --n-slots 4`. Per the orchestrator's load-bearing rule (and the 749 OOM lesson), smoke must exercise one full `train_step_fn` call at **production batch and temporal window**. Production Stage-B batch on the workstation is the 749 default (256, with --batch-size 128 fallback if OOM — see `experiments/749_titans_mag/SPEC.md:150-153`). Recommend **the final-step smoke uses --batch-size 128 --n-slots 17** to exercise realistic VRAM pressure. The earlier per-step smokes can stay at minimal config (precompute correctness, loss-site finite, etc.).

### C3 (MODERATE) — Skew / kurt formulas not specified; numerically fragile in JAX
Skew and kurt are listed as components 3 and 4 of each moment-group, but the spec does not specify the formula:
- Pearson skew: `E[(X-μ)^3] / σ^3` — undefined when σ=0 (constant sample), and `σ^3` can underflow at fp32 for log-return distributions where σ ≈ 0.01.
- Standardised kurtosis: `E[(X-μ)^4] / σ^4 − 3` (excess) vs `E[(X-μ)^4] / σ^4` (raw).

**Need to pin down at precompute time** (precompute side, since trainer-side only sees the descriptor). Recommend: (a) clamp `σ = max(σ, 1e-6)` (~10× smallest expected std), (b) compute *excess* kurt to align with N(0,1) → 0 expectation, (c) compute skew via `scipy.stats.skew(bias=False)` and kurt via `scipy.stats.kurtosis(fisher=True, bias=False)` for unambiguous fp32 stability. Document the choice in the shard NPZ as a schema-v1 invariant (the spec already prescribes a `schema_version=1` flag — this is just clarifying what v1 means).

### C4 (MODERATE) — Per-rotation cyclic-pad masking before moments
The TASK_745 spot-check (`runs/745_spot_check/verdict.md`) confirms that `n_assets=64` columns include cyclic-padded duplicates when `len(period_assets) < 64`, and ~31 of those 64 are NaN-masked because they're EUR/CAD crosses Bella never saw. **Computing moments naively over a 64-column tensor with ~31 NaNs gives NaN moments.** The spec's "moments are permutation-invariant" claim is true but elides this: the precompute must first restrict to the *finite* subset of cells, then compute moments. Otherwise descriptor[0] = `(nan, nan, nan, nan)` and the loss is masked-out everywhere.

Mitigation: in the precompute function `compute_shape_descriptor`, replace NaNs with masked-array math (use `np.nanmean`, `np.nanstd`, and for skew/kurt use `scipy.stats.skew/kurtosis` on the finite subset; for SVD, restrict the matrix to finite columns before `np.linalg.svd`). Also worth a one-liner sanity assert in the smoke: `assert np.isfinite(desc).all()` after the descriptor is computed on the test period.

### C5 (MINOR) — Reuse of TASK_745 spot-check shard for the workstation smoke
The smoke needs at least one Bella shape shard available locally. Period 0 (`2018-10-17__2019-04-15__a7185523`) already has its TASK_740 legacy shard at `/tmp/745_spotcheck/bella_preds_2018-10-17__2019-04-15__a7185523.npz`. The smoke for step 1 (descriptor computation) can derive a one-period shape shard from this artifact (read `preds` array, compute moments + SVD, save out as `bella_shape_2018-10-17__2019-04-15__a7185523.npz`) without needing to run Bella. Saves ~5 minutes per smoke iteration and avoids relying on Bella checkpoint presence for descriptor-correctness smokes. **The full-train_step smoke (final step) still needs a full manifest — recommend wrapping it to point at the one-period manifest.**

### C6 (MINOR) — Mutual-exclusion enforcement location
Spec §5 says "a `ValueError` is raised in args translation if both `--bella-distill-head` and `--bella-shape-distill` are passed". The cleanest enforcement point is in `train_phase2.py::main()` after both flags have been resolved into config — not in argparse (since defaults can override either, and config.from_json may carry the field). Recommend: assert `not (config.use_bella_distill_head and config.use_bella_shape_distill)` immediately after the args-translation block (after line 2139 in train_phase2.py), with an `f"Cannot enable both TASK_740 (bella_distill_head) and TASK_754 (bella_shape_distill)"` error.

### C7 (MINOR) — Spec assumes new experiment ID 754; orchestrator's gating prompt says 757
The orchestrator prompt requests the experiment SPEC at `experiments/757_bella_shape/SPEC.md`. The TASK_754 spec §6.10 says `experiments/754_bella_shape/SPEC.md`. Inspecting `experiments/` shows the highest current ID is 752 (`752_value_oracle_baseline_pair`). 753-756 are *task* IDs (`tasks/TASK_75{3,4,5,6}_*.md`) but no `experiments/75{3,4,5,6}_*` directories exist yet. Recommend `experiments/757_bella_shape/` per orchestrator instruction (next free numerical ID; matches the spirit of "next free experiment ID — verify via `ls experiments/` first"). Document the ID skip in the SPEC's title.

### C8 (MINOR — task interaction, NOT load-bearing) — TASK_753 / TASK_755 interaction
- **TASK_753 (Bella as data augmentation, generator-side, commit `287bcb7`)** uses the same `tools/precompute_bella_predictions.py` plumbing but emits a different artifact (synthetic OHLCV, not predictions). 754 and 753 do not collide if 754 only adds a new `bella_shape_*` writer next to the existing `bella_preds_*` writer (which 753 is also adding alongside). No collision risk; verify the writer functions are independent.
- **TASK_755 (regime-conditional Bella distillation, low-priority salvage)** would, if implemented, slot in as a *third* shard family `bella_regime_*`. 754 paints no corner for 755: the shape descriptor + regime descriptor are orthogonal targets and could be stacked or compared in isolation.

**No conflicts found. 754 is free to proceed first.**

---

## 4. Per-step implementation plan

Eight steps total. Each step ends with a smoke command sized to fit the rule "small but exercises the new mechanism".

### Step 1 — Add `compute_shape_descriptor()` to precompute
**File:** `tools/precompute_bella_predictions.py`, new helper function near `_window_forward_log_returns` (after line 179).
**Adds:** ~50 LOC.
- Define `compute_shape_descriptor(preds_window: np.ndarray) -> np.ndarray` where `preds_window: (K, n_assets)` and returns `(14,)` float32.
- Implementation:
  - Flatten preds_window, take finite-subset, compute `(mean, std, skew, kurt)` via numpy + scipy (clamp std at 1e-6, excess kurt = Fisher).
  - Same for `preds_window[0, :]` (horizon-1 only) and `preds_window[-1, :]` (last horizon).
  - SVD of preds_window restricted to finite columns; return `(s1, s1 / sum(s_i))` (clamp denominator at 1e-12).
- Document `schema_version=1` invariant: skew/kurt unbiased; excess kurt; std-clamp 1e-6.

**Smoke:**
```bash
python -c "
import numpy as np
from tools.precompute_bella_predictions import compute_shape_descriptor
np.random.seed(0)
preds = np.random.randn(4, 64).astype(np.float32) * 0.01
preds[:, :31] = np.nan
desc = compute_shape_descriptor(preds)
assert desc.shape == (14,), desc.shape
assert np.isfinite(desc).all(), desc
print('OK', desc)
"
```
Wall: <2 s. FAIL-LOUD on assertion. Pass: shape (14,) all-finite.

### Step 2 — Emit shape shard + manifest writer in precompute
**File:** `tools/precompute_bella_predictions.py`, modify `_process_period` (lines 191-318) and `main()` (lines 437-450).
**Adds:** ~80 LOC (additive — no edit to existing `bella_preds_*` writer).
- After computing the per-window `preds[w]` at line 288, also compute `desc[w] = compute_shape_descriptor(preds[w])` and accumulate into `shape_descriptors: (n_windows, 14)`.
- Write `bella_shape_<tag>.npz` to `out_dir` with keys: `descriptor (n_windows, 14)`, `schema_version=1 (int32)`, `generated_at`, `bella_checkpoint_sha`, `rotation_hash`.
- Maintain a parallel manifest in `_process_period`'s return entry and write to `config/bella_shape_manifest.json` in `main()`. **Manifest is a separate file** — does NOT touch `config/bella_preds_manifest.json`.
- Add a CLI flag `--shape-manifest` (default `config/bella_shape_manifest.json`).

**Smoke:** Use the locally-available period 0 shard from `/tmp/745_spotcheck/`:
```bash
python -c "
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')
from tools.precompute_bella_predictions import compute_shape_descriptor
shard = np.load('/tmp/745_spotcheck/bella_preds_2018-10-17__2019-04-15__a7185523.npz', allow_pickle=True)
preds = shard['preds']  # (n_windows, K, n_assets)
assert preds.ndim == 3
desc_arr = np.stack([compute_shape_descriptor(preds[w]) for w in range(min(100, preds.shape[0]))])
assert desc_arr.shape == (min(100, preds.shape[0]), 14)
print('mean per component:', desc_arr.mean(0))
print('std per component:', desc_arr.std(0))
assert np.isfinite(desc_arr).all()
print('OK')
"
```
Wall: <10 s. Pass: 100-window descriptor finite, mean/std plausible.

### Step 3 — Add `AuxBellaShapeHead` class
**File:** `mcts_trader/jax_agent/muzero_model.py`, insert new class after `AuxBellaHead` (after line 144).
**Adds:** ~35 LOC.
- Same `Dense(hidden_dim) → GELU → LayerNorm → Dense(descriptor_dim, zero-init)` pattern as `AuxBellaHead` lines 110-144.
- Output dim is `descriptor_dim` (parameter, default 14).
- No reshape to (K, n_assets) — output is flat `(B, 14)`.

**Smoke:**
```bash
python -c "
import jax, jax.numpy as jnp, sys
sys.path.insert(0, '.')
from mcts_trader.jax_agent.muzero_model import AuxBellaShapeHead
head = AuxBellaShapeHead(descriptor_dim=14, hidden_dim=64)
rng = jax.random.PRNGKey(0)
latent = jnp.zeros((4, 128))
params = head.init(rng, latent)
out = head.apply(params, latent)
assert out.shape == (4, 14), out.shape
assert jnp.allclose(out, 0.0), 'Zero-init violated'
print('OK', out.shape)
"
```
Wall: <10 s. Pass: shape (4, 14) and zero-init verified.

### Step 4 — Wire `AuxBellaShapeHead` into `MuZeroNetworks`
**File:** `mcts_trader/jax_agent/muzero_model.py`, modify `MuZeroNetworks` class.
- Add fields after the TASK_740 block (~line 277): `use_bella_shape_head: bool = False`, `bella_shape_descriptor_dim: int = 14`, `bella_shape_hidden_dim: int = 256`.
- In `setup()` after line 437: `if self.use_bella_shape_head: self.aux_bella_shape_head = AuxBellaShapeHead(descriptor_dim=self.bella_shape_descriptor_dim, hidden_dim=self.bella_shape_hidden_dim, dtype=self.dtype, param_dtype=self.param_dtype)`.
- In `initial_inference()` after line 643: `if self.use_bella_shape_head: result['aux_bella_shape'] = self.aux_bella_shape_head(latent_state)`.

**Smoke:**
```bash
python -c "
import jax, jax.numpy as jnp, sys
sys.path.insert(0, '.')
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
m = MuZeroNetworks(state_dim=64, n_assets=64, encoder_version=3, embed_dim=32, encoder_num_heads=4,
                   use_bella_shape_head=True, bella_shape_descriptor_dim=14)
rng = jax.random.PRNGKey(0)
obs = jnp.zeros((2, 32, 60, 64))
params = m.init(rng, observation=obs, mode='initialize_parameters')
out = m.apply(params, observation=obs, mode='initial_inference')
assert 'aux_bella_shape' in out, list(out.keys())
assert out['aux_bella_shape'].shape == (2, 14)
print('OK', out['aux_bella_shape'].shape)
"
```
Wall: <30 s (init + apply for a small net). Pass: 'aux_bella_shape' present, shape (2, 14).

### Step 5 — Config + create_model plumbing + Phase 1 ckpt-load
**File:** `phase2_config.py` + `train_phase2.py`.
**Adds:** ~25 LOC.
- `phase2_config.py` (after line 116, the TASK_740 block): add `use_bella_shape_head: bool = False`, `bella_shape_descriptor_dim: int = 14`, `bella_shape_hidden_dim: int = 256`, `bella_shape_loss_weight: float = 0.1`, `bella_shape_manifest: str = "config/bella_shape_manifest.json"`, `bella_shape_component_weights: Optional[tuple] = None` (default uses internal defaults from spec §4.1).
- `train_phase2.py:create_model()` after line 404: pipe through `use_bella_shape_head`, `bella_shape_descriptor_dim`, `bella_shape_hidden_dim` to `MuZeroNetworks(...)`.
- `train_phase2.py:initialize_missing_aux_heads` (line 165): the existing pattern handles new top-level keys automatically; verify it does so.
- `train_phase2.py` after line 2274: add `if getattr(args, 'reset_bella_shape_head', False): if 'aux_bella_shape_head' in params: del params['aux_bella_shape_head']`.
- `train_phase2.py` line 2303-ish (the aux-head init trigger block): add `or getattr(config, 'use_bella_shape_head', False)` to the OR list.

**Smoke:** Same as Step 4 but go through `create_model(config)` and verify head present:
```bash
python -c "
import jax, jax.numpy as jnp, sys
sys.path.insert(0, '.')
from phase2_config import Phase2Config
from train_phase2 import create_model
cfg = Phase2Config(use_bella_shape_head=True, encoder_version=3, embed_dim=32, encoder_num_heads=4)
m = create_model(cfg)
rng = jax.random.PRNGKey(0)
obs = jnp.zeros((2, 32, 60, 64))
params = m.init(rng, observation=obs, mode='initialize_parameters')['params']
assert 'aux_bella_shape_head' in params
out = m.apply({'params': params}, observation=obs, mode='initial_inference')
assert 'aux_bella_shape' in out
print('OK', out['aux_bella_shape'].shape)
"
```
Wall: <30 s. Pass: head registered + initial_inference returns descriptor.

### Step 6 — Shape-shard loader + batch plumbing
**File:** `train_easy.py` (`_load_bella_shape_shard`) + `train_phase2.py` (`AsyncBatchPipeline._prepare_numpy` + `_encode_jax`).
**Adds:** ~80 LOC.
- `train_easy.py` near line 752 (next to `_load_bella_shard`): implement `_load_bella_shape_shard(self, group_key, T_ret, descriptor_dim, manifest_path, period_assets)` — load `descriptor (n_windows, 14)`, NaN-pad to `T_ret`, return.
- `train_easy.py` `_make_slot` (~line 700): when `oc.get('use_bella_shape_head', False)` is True, load shape shard into slot dict as `bella_shape_full`.
- `train_phase2.py:load_segment_pool` (line 446-461): add `'use_bella_shape_head'`, `'bella_shape_manifest'`, `'bella_shape_descriptor_dim'` to `oracle_cfg`.
- `train_phase2.py:AsyncBatchPipeline._prepare_numpy` (mirror lines 559, 645-657, 688-689 for shape): `all_bella_shape = []`; per-slot fetch via `slot.get('bella_shape_full')[decision_indices]`; batch dict entry `'bella_shape': ...`.
- `train_phase2.py:AsyncBatchPipeline._encode_jax` (line 750): cast to JAX array.

**Smoke:** Run a 1-batch numpy-prepare path to verify shape:
```bash
# Requires the 1-period shape shard from Step 2 in data/bella_preds/ + a 1-entry config/bella_shape_manifest.json
python -c "
import sys; sys.path.insert(0, '.')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU-only — avoid GPU init
from phase2_config import Phase2Config
from train_phase2 import load_segment_pool, AsyncBatchPipeline
cfg = Phase2Config(use_bella_shape_head=True, bella_shape_manifest='/tmp/bella_shape_manifest_smoke.json',
                   encoder_version=3, embed_dim=32, encoder_num_heads=4, batch_size=4, n_slots=1)
# (use a minimal 1-period periods file)
# verify batch['bella_shape'].shape == (4, 14) and finite
print('OK')
"
```
Wall: ~30-60 s (segment pool load is slow). Pass: `bella_shape` key in batch dict with shape `(B, 14)`.

### Step 7 — Train-step loss site + CLI flags + CSV + logging
**File:** `train_phase2.py`.
**Adds:** ~80 LOC.
- `make_stage_b_train_step` closure (line 944-970): pull `use_bella_shape_distill`, `bella_shape_w`, `bella_shape_component_weights` (or default `(2.0, 1.0, 0.5, 0.25, 1.0, 2.0)` mapped to per-component weights) from config.
- New JIT loss branch after line 1196 (the TASK_740 branch), guarded by `use_bella_shape_distill`. Compute per-component weighted MSE with sample-wise NaN mask (mirror `valid = jnp.isfinite(tgt_shape); tgt_safe = jnp.where(valid, tgt_shape, 0.0); sq = (pred_shape - tgt_safe) ** 2; per_c = jnp.sum(sq * valid_f, axis=0) / jnp.maximum(jnp.sum(valid_f, axis=0), 1.0)`).
- Add `bella_shape_loss`, `bella_shape_valid_frac`, `bella_shape_first_two_residuals` to metrics dict at line 1208-1236.
- Add to `total_loss` at line 1198-1206 with `bella_shape_w * bella_shape_loss`.
- `stage_b_train_step` signature (line 973-977): append `bella_shape: (B, 14)` argument.
- The training-loop caller (line 1539-1595): pull `batch.get('bella_shape')`, pass to `train_step_fn`.
- CLI flags after line 1930: `--bella-shape-distill`, `--no-bella-shape-distill`, `--bella-shape-mode {moments,lowrank,contrastive}` (only `moments` implemented; raise `NotImplementedError` for others), `--bella-shape-weight` (default 0.1), `--bella-shape-descriptor-dim` (default 14), `--bella-shape-component-weights` (comma-separated 6-tuple), `--bella-shape-manifest`, `--reset-bella-shape-head`.
- Args→config translation after line 2135: pipe through. **Add the mutual-exclusion assert** at config-resolve time:
  ```python
  if config.use_bella_distill_head and getattr(config, 'use_bella_shape_head', False):
      raise ValueError("Cannot enable both --bella-distill-head (TASK_740) and "
                       "--bella-shape-distill (TASK_754) simultaneously.")
  ```
- CSV writer (line 1478-1494): add `bella_shape_loss`, `bella_shape_valid_frac`.
- E-line log (line 1641-1646 area): add `shape={...}(vf=...)` block under `use_bella_shape_head`.

**Smoke (loss-finite, mini-batch):**
```bash
CUDA_VISIBLE_DEVICES=1 python train_phase2.py \
  --stage B --epochs 1 --steps-per-epoch 2 --batch-size 32 \
  --periods-file /tmp/foxglove_periods_smoke_1period.json \
  --n-slots 1 --max-runs-per-group 1 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.5 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 --aux-price-weight 0.1 \
  --bella-shape-distill --bella-shape-mode moments \
  --bella-shape-weight 0.1 --reset-bella-shape-head \
  --bella-shape-manifest /tmp/bella_shape_manifest_smoke.json \
  --output-dir runs/000_smoke/bella_shape_step7 \
  --name 000_bella_shape_step7 2>&1 | tee runs/000_smoke/bella_shape_step7/train.log
```
Wall: ~3-5 min compile + ~30 s training. Pass: `bella_shape_loss` is finite, `bella_shape_valid_frac > 0` for the 1-period smoke slot.

### Step 8 — Full train_step_fn smoke at production batch (the 749 lesson)
**File:** No code change — full integration smoke.
**Goal:** Exercise the JIT'd `stage_b_train_step` at production batch (128) and full temporal window (60), to flush JIT compile / OOM characteristics.

**Smoke (FULL):**
```bash
CUDA_VISIBLE_DEVICES=1 python train_phase2.py \
  --stage B --epochs 1 --steps-per-epoch 5 --batch-size 128 \
  --periods-file /tmp/foxglove_periods_smoke_1period.json \
  --n-slots 4 --max-runs-per-group 1 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.5 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 --aux-price-weight 0.1 \
  --bella-shape-distill --bella-shape-mode moments \
  --bella-shape-weight 0.1 --reset-bella-shape-head \
  --bella-shape-manifest /tmp/bella_shape_manifest_smoke.json \
  --output-dir runs/000_smoke/bella_shape_full \
  --name 000_bella_shape_full 2>&1 | tee runs/000_smoke/bella_shape_full/train.log
```
Wall: ~5-10 min total (compile dominated; training ~60-90 s for 5 steps). Pass per spec §7:
1. Compile completes within Phase-2 budget (5-10 min ok on workstation).
2. `total_loss` finite for all 5 steps.
3. `bella_shape_loss` reported, finite, **drops from step 1 → step 5**.
4. `bella_shape_valid_frac > 0.9` on smoke slot.
5. Peak VRAM ≤ baseline Phase-2 + 300 MB.
6. No new NaN paths in loss dict.

If step 8 passes: TASK_754 implementation complete; ready for paired 3-seed launch by server agent under `experiments/757_bella_shape/SPEC.md`.

---

## 5. Stop-point checklist

When implementing Phase 2, check each box after the step's smoke passes. If any FAILs, write `tasks/TASK_754_STOP_<step>.md` with the failure mode + observed output and STOP.

- [ ] **Step 0 (pre-flight):** Verify `checkpoints/diff_teacher/generator_best/generator` exists on workstation. If not, fall back to using the cached TASK_745 shard at `/tmp/745_spotcheck/` for the precompute-side smokes (Steps 1-2) and verify Bella ckpt only when running the full precompute later. If neither Bella ckpt nor cached shard exists, STOP and report (C1).
- [ ] **Step 1:** `compute_shape_descriptor(preds_window)` returns `(14,)` float32, all finite, on a small synthetic input. C3-aware: skew/kurt formulas pinned, std-clamp 1e-6, excess kurt. **Schema_version=1 invariants documented inline in the function docstring.**
- [ ] **Step 2:** Precompute on locally-cached TASK_745 shard produces a 100-window descriptor array `(100, 14)` all-finite with plausible per-component mean/std. Manifest writer emits a clean `bella_shape_manifest.json`. **Existing `bella_preds_*` shard format byte-identical to pre-edit (C8 paranoia — additive-only).**
- [ ] **Step 3:** `AuxBellaShapeHead` init + apply on a synthetic latent returns `(B, 14)` zero output (zero-init honest).
- [ ] **Step 4:** `MuZeroNetworks` with `use_bella_shape_head=True` returns `aux_bella_shape` key in `initial_inference` output dict.
- [ ] **Step 5:** `create_model(config)` from `Phase2Config(use_bella_shape_head=True)` produces a model whose `initial_inference` includes the descriptor. `--reset-bella-shape-head` drops the head subtree on ckpt load. Mutual-exclusion assertion raises ValueError when both 740 and 754 flags are set.
- [ ] **Step 6:** `AsyncBatchPipeline.get_batch()` returns a batch dict with `bella_shape` key of shape `(B, 14)` and finite values when the manifest covers the smoke period; sample-wise NaN-fill when not covered.
- [ ] **Step 7:** Mini-batch train_step runs 2 steps, `bella_shape_loss` is finite, `bella_shape_valid_frac > 0`, no new NaN in loss dict, CSV writer adds the new columns.
- [ ] **Step 8 (LOAD-BEARING):** Full train_step_fn at `--batch-size 128` runs 5 steps. All 6 pass criteria in spec §7 hold. Peak VRAM ≤ baseline Phase-2 + 300 MB. **`bella_shape_loss` drops step 1→5 (sanity that head learns).**
- [ ] **Final commit:** "TASK_754 implementation complete" — list files touched, smoke results pasted.
- [ ] **SPEC creation:** `experiments/757_bella_shape/SPEC.md` (next free numerical ID per `ls experiments/`, per C7 + orchestrator instruction). Includes paired 3-seed config, server-launch commands (DO NOT launch), eval command, done-flag names, what to look for in `result.md`.

---

## Estimated total implementation time

| Step | Description | Estimated agent wall (incl. smoke) |
|---|---|---:|
| 0 | Pre-flight checks | 0.2 h |
| 1 | `compute_shape_descriptor()` | 1.0 h |
| 2 | Precompute writer + manifest | 1.5 h |
| 3 | `AuxBellaShapeHead` class | 0.5 h |
| 4 | Wire into `MuZeroNetworks` | 0.5 h |
| 5 | Config + create_model + ckpt-load | 1.0 h |
| 6 | Shard loader + batch plumbing | 1.5 h |
| 7 | Loss site + CLI + CSV + log | 2.0 h |
| 8 | Full train_step_fn smoke | 0.5 h |
| Final | SPEC + commit | 0.3 h |

**Total: ~9 hours of agent work.**

(Comparison: spec's own §6 estimate is "~310 LOC across 5 files. ~1 day of focused implementation + smoke." Our estimate matches at the file-count level — but we explicitly carve out the precompute-side smoke before head-side, which saves an agent-cycle debugging the "head trains but loss is NaN because descriptor wasn't computed correctly" failure mode that the loss-site NaN mask would silently hide.)

---

## Sign-off

Verdict: **PASS-with-caveats (6 caveats — C1 through C8, of which C1, C2, C3 require pre-implementation decisions; C4 through C8 are addressable in-step).**

Phase 2 (implementation) is gated on operator approval. After approval, the implementation will follow the per-step plan above with one commit per step (message format `TASK_754 step N/M: <description>`), final commit `TASK_754 implementation complete`, and the experiment SPEC at `experiments/757_bella_shape/SPEC.md`.

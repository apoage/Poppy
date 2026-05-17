# TASK_754 — Bella Trajectory-Shape Distillation

**Status**: DRAFTED 2026-05-17. Awaiting queue placement.
**Class**: Planning-side / encoder-side auxiliary signal. Successor to TASK_740 (per-asset per-step distillation, soft regression −1.92pp paired 3-seed).
**Blocks**: nothing directly. Should run *after* a sanity confirmation that TASK_745 Option A column remap is in effect on whatever precompute pipeline this task uses (we re-use the existing manifest plumbing but compute a different target).
**Source insight**: User direction 2026-05-17 — "distill the trajectory *shape* over a window into a single auxiliary signal, not per-asset per-step." Per-step targets are noise-dominated and structurally similar to the classifier-sharpening mechanisms that anti-correlate with Alpha (see `memory/project_classifier_alpha_anti_correlation.md`).

---

## 1. Goal

Replace TASK_740's per-asset per-step Bella distillation target with a **coarse trajectory-shape target**: the cross-asset / multi-step Bella prediction is summarised to a low-dimensional descriptor per training window, and a small auxiliary head regresses that descriptor. The loss surface is coarser; the gradient signal is denser per parameter; the failure mode of "Bella column at index i is the wrong asset" cannot scramble the target the same way (the moments-of-distribution descriptor is permutation-invariant; the low-rank descriptor is permutation-covariant in a way the head can absorb).

The hypothesis: a Bella-derived *regime/shape prior* is informative about Alpha but a Bella-derived *per-cell prediction* is not — and the way to find out is to evaluate the coarser signal head-to-head against the existing a1b1 baseline.

---

## 2. Hypothesis

Three independent observations motivate the coarse-target framing:

1. **TASK_740 paired Δα = −1.92pp, 0/3 positive seeds** (`runs/740_bella_distill/result.md:15-23`). Even after the TASK_745 Option A column remap was applied to the precompute manifest, the regression persisted in the provisional eval. The signal that Bella carries is not landing.
2. **Classifier-Alpha anti-correlation (3-way confirmed)** — `memory/project_classifier_alpha_anti_correlation.md`. Mechanisms that sharpen *per-asset* prediction targets (737, 738, 741-mint) consistently improve Bal.Acc / reduce loss but hurt deployment Alpha. Per-asset per-step Bella MSE structurally lives in this class. A coarser trajectory-level target moves the mechanism out of the anti-correlation class.
3. **Information-theoretic argument**: Bella's per-cell prediction at horizon `h` is a noisy estimate of a small-magnitude log-return. Variance dominates mean. The first few **moments of the predicted return distribution over the window** (or the top-k singular vectors of the predicted trajectory matrix) extract the regime-level information Bella *can* express reliably while discarding the per-cell noise it cannot.

We therefore expect: coarser target ⇒ denser per-parameter gradient ⇒ less classifier-sharpening behaviour ⇒ Δα ≥ 0 at the 3-seed gate. If this prediction fails, the Bella-as-prior thesis (Path β in TASK_740) is dead and we retire the mechanism class.

---

## 3. Background / current state

Files and lines load-bearing for this task:

- **TASK_740 SPEC**: `tasks/TASK_740_BELLA_DISTILL_HEAD.md` — original Path β-v2 plan.
- **TASK_740 result**: `runs/740_bella_distill/result.md:15-30` — paired 3-seed Δα −1.92pp, soft regression.
- **TASK_745 column-mismatch**: `tasks/TASK_745_BELLA_SEMANTICS.md:32-50` — Option A per-rotation remap; **Option C aggregate-return regime prior is the conceptual seed for this task** (`TASK_745:71-89`).
- **AuxBellaHead (existing per-cell head, to be replaced/extended)**: `mcts_trader/jax_agent/muzero_model.py:110-144`.
- **Bella distill loss site**: `train_phase2.py:1175-1196` — current per-horizon MSE on `(B, K, n_assets)`.
- **CLI flag block to extend**: `train_phase2.py:1918-1930`.
- **Args→config translation**: `train_phase2.py:2128-2135`.
- **SegmentPool Bella loader**: `train_easy.py:687-700, 752-...` (`_load_bella_shard`).
- **Batch plumbing in trainer**: `train_phase2.py:559-689` (`all_bella_preds`, `bella_preds` field).
- **Precompute pipeline** (re-used): `tools/precompute_bella_predictions.py`.
- **Bella component facts**: `diff_teacher/CLAUDE.md` — Bella outputs OHLCV-shaped sequences `(window, n_assets=64, 5)`; we currently extract horizoned log-returns offline, see `tools/precompute_bella_predictions.py`.
- **Anti-correlation evidence**: `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md`.
- **Architecture constraint** (do NOT propose transformer/attention to reasoning or dynamics): `mcts_trader/CLAUDE.md` — auxiliary heads are encoder-side and acceptable.

**Memory-hierarchy note** (per project `CLAUDE.md`): Bella inference is ~100 ms / window — must remain offline / precomputed in shards (`data/bella_preds/`). The per-window descriptor we compute *from* the precomputed shard is small (≤ ~70 floats per window) and pre-collapsed at precompute time, so the trainer-side memory cost is **strictly lower** than TASK_740's per-cell tensor `(T_ret, K, n_assets)`. Concretely: 740 stores `T_ret × 4 × 64 = 256·T_ret` floats per shard; 754 stores `T_ret × M` floats where `M ≤ 70`. ~3.6× shard-size reduction. Shards stay on disk, loaded once into RAM per slot, streamed to VRAM per batch.

---

## 4. Design

### 4.1 Primary mechanism — distribution moments (selected)

For each training window of length `T_ret` (the look-forward Bella was rolled out over), compute the first **M = 4 moments** of the predicted-log-return distribution **across (assets × horizon-steps)**:

```
preds:   shape (T_ret, K, n_assets)        # log-returns at horizons K=(1,5,10,20)
flat:    shape (T_ret, K * n_assets)       # flatten asset × horizon at each window position
desc[t] = ( mean(flat[t]), std(flat[t]), skew(flat[t]), kurt(flat[t]) )   # (4,)
```

Then **also** compute the same 4 moments restricted to (a) horizon-1-only and (b) horizon-20-only, yielding `4 + 4 + 4 = 12` floats per window. This captures regime-shape at two horizons without per-asset alignment dependence.

For asset-side coarse info (without per-cell alignment), append the **rank-1 SVD energy** of the matrix `M_t = preds[t, :, :]` of shape `(K, n_assets)` — i.e. `s1, s1 / (s1 + s2 + ...)` (top singular value and its relative weight). 2 floats. This carries "is Bella seeing one dominant cross-sectional mode (rally/crash) vs diffuse noise (sideways)".

**Descriptor dimensionality**: `D_desc = 12 + 2 = 14` floats per window.

The auxiliary head is a small MLP `latent_state → R^{D_desc}`, supervised by MSE:

```
L_shape = (1 / D_desc) * sum_j  w_j  * ( head(z_t)[j] - desc[t][j] )^2
```

with per-component weights `w_j` that downweight kurtosis (heavy-tailed, high variance) and upweight `mean` and `s1_ratio` (low variance, regime-discriminative). Default weighting: `w_mean=2.0, w_std=1.0, w_skew=0.5, w_kurt=0.25, w_s1=1.0, w_s1_ratio=2.0`.

**Permutation invariance**: every component of `desc` is invariant under asset permutation (moments are sums; SVD spectrum is invariant under column permutation). This means the residual column-mismatch risk from TASK_745 cannot scramble this gradient — even if Bella's column-i was the "wrong asset" by the trainer's convention, the moments are unchanged.

### 4.2 Alternative mechanisms (listed; not implemented unless primary fails)

- **(B) Low-rank summary** — keep top-`k=3` singular *vectors* (not just values) of the `(T_ret, K·n_assets)` predicted-trajectory matrix; supervise the head against a flattened low-rank reconstruction. Captures "principal modes" of the predicted trajectory. Cost: needs a remap-aware comparison or a permutation-invariant projection. Reserved for TASK_754b if primary lands but is marginal.
- **(C) Contrastive trajectory** — triplet loss: positive = Bella-predicted vs observed future; negative = Bella vs shuffled observed. Tells the encoder "Bella sees the shape" without forcing point match. Reserved for TASK_754c; requires a richer trainer-side loss site than the current aux-head pattern.

Both are documented here so the primary's verdict cleanly decides whether to escalate.

### 4.3 Why (A) over (B) and (C)

- (A) is **fully permutation-invariant**, so it neutralises any residual TASK_745 column-mismatch risk.
- (A) descriptor is **dense** (14 floats) and **smooth** in Bella's predictions — gradients are well-conditioned.
- (B) re-introduces per-asset alignment via singular vectors; we lose the column-mismatch insulation.
- (C) requires negative sampling and a comparator over the observed future — non-trivial trainer-side wiring, and the "observed future" inside a 60-step window is the same noise floor TASK_740 hit.

### 4.4 Gradient path

The head sits on the global latent (mirror of `AuxPriceHead`/`AuxBellaHead` insertion in `MuZeroNetworks.setup`, `muzero_model.py:421-433`). Forward: `z → MLP → R^{D_desc}`. Loss is added to the existing `total_loss` sum at `train_phase2.py:1198-1206`. Zero-init final layer keeps Phase-1 ckpt-load byte-identical (matches the convention used by `AuxPriceHead` line 101-105 and `AuxBellaHead` line 138-142).

---

## 5. CLI surface

New flags, added next to the existing TASK_740 block at `train_phase2.py:1918-1930`. All default to OFF; checkpoint behaviour unchanged.

| Flag | Default | Effect |
|---|---|---|
| `--bella-shape-distill` | `False` (store_true) | Master toggle. Enables the shape head and the shape loss term. |
| `--no-bella-shape-distill` | `False` | Symmetric off-switch (matches `--no-bella-distill-head` style). |
| `--bella-shape-mode` | `moments` | One of `{moments, lowrank, contrastive}`. Only `moments` implemented in this task; `lowrank` and `contrastive` raise `NotImplementedError`. |
| `--bella-shape-weight` | `0.1` | Loss weight (mirrors `--bella-distill-weight` default). |
| `--bella-shape-descriptor-dim` | `14` | Asserted against precompute; mismatch is a hard error at startup. |
| `--bella-shape-component-weights` | `None` | Optional comma-separated 6-tuple overriding the default `w_*` weights for `(mean, std, skew, kurt, s1, s1_ratio)`. |
| `--bella-shape-manifest` | `config/bella_shape_manifest.json` | Manifest of per-period descriptor shards (separate from the TASK_740 manifest, since columns are different). |
| `--reset-bella-shape-head` | `False` | Drop `aux_bella_shape_head` subtree on ckpt load — required on first run from any pre-754 ckpt. |

The legacy TASK_740 flags (`--bella-distill-head`, `--bella-distill-weight`, `--bella-preds-manifest`, `--reset-bella-distill-head`) are **untouched**. The two mechanisms are mutually exclusive at training time; a `ValueError` is raised in args translation if both `--bella-distill-head` and `--bella-shape-distill` are passed.

---

## 6. Implementation steps

Numbered, with target files and rough LOC.

1. **Precompute pipeline extension** (~80 LOC). Edit `tools/precompute_bella_predictions.py` to emit *two* shards per period:
   - existing `bella_preds_<period>__<hash>.npz` (kept for legacy 740 use; no change).
   - new `bella_shape_<period>__<hash>.npz` with `descriptor` of shape `(T_ret, 14)` (float32) plus a `schema_version=1` int. Apply **TASK_745 Option A per-rotation column remap before computing moments** (Phase-1 hygiene; descriptors are permutation-invariant so this is belt-and-suspenders, but it guarantees `lowrank` and `contrastive` variants will see a remapped tensor when added).
   - Emit a new manifest `config/bella_shape_manifest.json` keyed by the same rotation-hashed tag the trainer constructs at `train_easy.py:752+`.
   - Storage: ~`T_ret * 14 * 4 = 56·T_ret` bytes per period. For 18 unique date-ranges × ~15 rotations × ~10k windows × 56 B ≈ 150 MB total. Lives in `data/bella_preds/`.

2. **New aux head** (~30 LOC). In `mcts_trader/jax_agent/muzero_model.py`, add `class AuxBellaShapeHead(nn.Module)` immediately after `AuxBellaHead` (after line 144). Same Dense→GELU→LayerNorm→Dense pattern as `AuxPriceHead`, output dim `= descriptor_dim = 14`. **Zero-init final layer** (`kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros`) — mirrors line 101-105 and 138-142.

3. **Wire head into MuZeroNetworks** (~15 LOC). In `muzero_model.py:208-300`, add four fields: `use_bella_shape_head: bool = False`, `bella_shape_descriptor_dim: int = 14`, `bella_shape_hidden_dim: int = 256`, plus the matching `setup()` instantiation around line 421 and the `initial_inference` output-dict population around line 643 (`result['aux_bella_shape'] = self.aux_bella_shape_head(latent_state)`).

4. **Config + model-builder plumbing** (~20 LOC). Add fields to `phase2_config.Phase2Config`; pipe through `create_model()` at `train_phase2.py:402-404` (next to the existing `use_bella_distill_head` lines).

5. **Per-rotation shape-shard loader** (~50 LOC). In `train_easy.py`, add `_load_bella_shape_shard` next to `_load_bella_shard` (~line 752+); store `bella_shape_full` in the slot dict at line 718-area. The tag construction **must match precompute exactly** — re-use the same `f"{start}__{end}__{rotation_hash}"` pattern.

6. **Batch plumbing** (~30 LOC). In `train_phase2.py`, mirror the `bella_preds` plumbing for `bella_shape`:
   - `all_bella_shape = []` near line 559.
   - Per-decision indexing near line 649-657.
   - Batch dict entry near line 688 (`'bella_shape': ...`).
   - JAX cast near line 750.

7. **Loss site** (~25 LOC). In `train_phase2.py:1175-1196`, add a third branch (or replace the 740 branch when `--bella-shape-distill` is enabled). Per-window MSE with per-component weighting; sample-wise NaN mask matching the existing pattern at line 1183-1192. Sum into `total_loss` at line 1206 with `shape_w * shape_loss`. **Mandatory: assert mutual exclusion with `use_bella_distill_head` at config-resolve time** (one of the two, not both — a stacked variant is out of scope for this task).

8. **CLI block** (~25 LOC). Add the flags from §5 to the argparse block at `train_phase2.py:1918+`. Add the args→config translation at the `2128+` block.

9. **Logging/CSV** (~15 LOC). Add `bella_shape_loss`, `bella_shape_valid_frac`, and per-component `bella_shape_mean_resid`/`...std_resid` (just the first two components) to the CSV writer and to the E-line log. Mirror `bella_valid_frac` from TASK_740.

10. **Smoke and docs** (~20 LOC). `tools/run_754_bella_shape_distill.sh` mirrors `tools/run_740_bella_distill.sh`. Add `experiments/754_bella_shape/SPEC.md` per the standard pattern.

**Estimated total**: ~310 LOC across 5 files. ~1 day of focused implementation + smoke.

---

## 7. Smoke test

Mirrors 749's "full train_step shape" pattern — short, but exercises a complete `stage_b_train_step` so the JIT compile and OOM characteristics are flushed early.

```bash
cd ~/Trader && source trader_env/bin/activate
CUDA_VISIBLE_DEVICES=1 python train_phase2.py \
  --stage B --epochs 1 --steps-per-epoch 5 --batch-size 64 \
  --periods-file config/foxglove_periods.json \
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
  --output-dir runs/000_smoke/bella_shape_smoke \
  --name 000_bella_shape_smoke 2>&1 | tee runs/000_smoke/bella_shape_smoke/train.log
```

**Pass criteria** (must all hold, ≤ 60s wall after compile):

1. Compile completes within the standard Phase-2 budget (~3-5 min on workstation).
2. `total_loss` finite for all 5 steps.
3. `bella_shape_loss` is reported, finite, and **drops from step 1 → step 5** (sanity: head learns *something* in 5 steps; zero-init means the first step's loss = MSE between zero and the descriptor).
4. `bella_shape_valid_frac > 0.9` on the smoke slot (the shape manifest is populated for the smoke period).
5. Peak VRAM ≤ baseline Phase-2 + 300 MB (head is 256·D + 14·256 ≈ small).
6. No new NaN paths anywhere in the loss dict.

If (3) fails: head/gradient-path bug. If (4) fails: precompute / manifest mismatch (most likely rotation-hash tag drift between `precompute_bella_predictions.py` and `train_easy.py:_load_bella_shape_shard`). If (5) fails: shape head accidentally per-asset — re-check `D_desc=14`.

---

## 8. Eval protocol

Standard Tier-2 paired-seed flow.

- **Screening — paired 3-seed**: seeds 7, 13, 23 against the **741_mint a1b1 baseline at the matching seeds** (`runs/741_mint/baseline_seed_{7,13,23}/`). 200 epochs each, server-side, 2-wide GPU packing (TASK_742 rule).
- **Eval**: `tools/benchmark_easy.py` over the standard 6-period benchmark.
- **Screening verdict** (write to `runs/754_bella_shape/result.md`):
  - **PROMOTE to 5-seed** if mean Δα ≥ +1.5pp AND ≥ 2/3 positive AND worst seed ≥ −2pp.
  - **HOLD** if mean Δα ∈ [−1pp, +1.5pp]. Examine per-period split; if `sideways` and `bear` both ≥ 0 then promote anyway (regime-shape prior should land hardest in low-information regimes).
  - **REJECT** if mean Δα < −1pp OR worst seed < −5pp. Retire the trajectory-shape-distill mechanism class; document `project_bella_distill_dead.md` in memory.
- **Promotion criteria — 5-seed extension** (seeds 31, 42 added): mean Δα ≥ +1pp **AND** no seed < −3pp (matches 734's promotion gate). On pass, swap canonical baseline to a1b1 + shape-distill.

Conditional baseline: if 749 (Titans-MAG) lands meanwhile, the screening still pairs against 741_mint a1b1 — Titans is in a different mechanism class and we want orthogonal signal, not stacked.

---

## 9. Risks

**R1 (MAJOR) — Coarse-signal information collapse.** Distribution moments may carry too little information to move Alpha *or* may be carrying only what the existing PopArt value head already learns (mean future return ≈ value target). If true, the gradient through the shape head is redundant with the value head's gradient; we'd see `bella_shape_loss` drop nicely while Δα ≈ 0. **Mitigation**: include the SVD components (`s1`, `s1_ratio`) which carry *cross-sectional* concentration info that the per-asset value head cannot directly express. If 3-seed lands at exactly zero Δα with `bella_shape_loss` healthy, that's the signature of this risk; the response is to escalate to alternative (B) low-rank summary, NOT to raise the loss weight.

**R2 (MAJOR) — Trajectory length / sampling-rate sensitivity.** The moments depend on the Bella rollout's horizon set `K=(1,5,10,20)` and on whether per-horizon log-returns are *cumulative* or *incremental*. Mis-specification (cumulative vs. instantaneous) silently changes the descriptor's distribution — and the head still trains to whatever target we hand it, masking the bug. **Mitigation**: assert `schema_version=1` in precompute → trainer matches; print descriptor stats (`mean, std, min, max` per component) for the first batch and gate the smoke test on plausibility (e.g. `|mean(desc[0])| < 0.05` since log-returns at minute scale).

**R3 (MODERATE) — Bella prediction quality on out-of-distribution regimes.** Bella was trained on historical OHLCV, regime-conditioned via window-statistic estimate (see `tasks/TASK_740_BELLA_DISTILL_HEAD.md:148`). On regimes Bella under-saw (extreme crash with > 30% drawdown over 24h), its predicted moments are unreliable — descriptor is biased. The trainer cannot distinguish "Bella's noisy moments here" from "the regime is genuinely weird". **Mitigation**: log per-period `bella_shape_loss` (already in the CSV) and on rejection, check whether the regression concentrates on the regimes where Bella is known to be weak.

**R4 (MODERATE) — Anti-correlation pattern still bites.** Even a permutation-invariant target can sharpen the encoder's per-asset representation indirectly: the encoder may dedicate capacity to making moments easy to predict, displacing capacity that previously served the policy/value path. We are smaller than the per-cell target (14 vs 256·K per window) so the effect should be smaller, but the mechanism class is still "encoder is pulled toward a non-policy objective". **Mitigation**: keep the shape loss weight low (default 0.1, mirroring 740). If the 3-seed shows the classifier-anti-correlation signature (Bal.Acc up, Alpha down), the class is broader than we hoped and we retire the mechanism entirely.

**R5 (MINOR) — TASK_745 still load-bearing for alternative variants.** While moments are permutation-invariant, the (B) low-rank and (C) contrastive variants are not. If primary fails and we escalate, alternative variants demand that TASK_745 Option A is fully verified by spot-check (see `tasks/TASK_745_BELLA_SEMANTICS.md:97-104`). Don't escalate without that verification.

**R6 (MINOR) — Precompute regression.** The precompute pipeline now emits *two* shard families. A mis-edit could break the existing TASK_740 shard format and corrupt the legacy manifest. **Mitigation**: precompute changes are *additive only* — no edit to the `bella_preds_*` shard writer; only a new `bella_shape_*` writer next to it. Smoke-test that loading TASK_740 shards from the modified script gives byte-identical output to the pre-edit version (one sha256 check, ~5 LOC of test scaffolding).

---

## 10. Out of scope

- Stacking shape-distill with the per-cell TASK_740 head (mutually exclusive at config-resolve time; can be revisited as Tier-D if both individually positive).
- The (B) low-rank and (C) contrastive variants — implemented only if primary lands AND we want to push for a stronger signal. Each becomes TASK_754b/c.
- Retraining Bella with canonical asset-symbol ordering (Tier-D, separate initiative).
- Live Bella inference in the training loop (ruled out by latency; same as TASK_740).
- Re-running TASK_740 with a different loss weight — we already have a 3-seed verdict on that mechanism.
- Cross-window descriptors (e.g. trajectory-shape over consecutive Bella windows). Possible follow-up, not in this task.

---

## 11. Success criteria

- **Code success**: smoke passes all 6 criteria in §7. Phase-1 checkpoint loads byte-identical without `--reset-bella-shape-head` (verifying the zero-init is honest).
- **Screening success (paired 3-seed)**: mean Δα ≥ +1.5pp with ≥ 2/3 positive.
- **Promotion success (5-seed)**: mean Δα ≥ +1pp with no seed < −3pp. Promote a1b1+shape-distill to canonical Stage-B baseline; update `experiments/QUEUE.md` and `lore/notes_<date>.md` with the verdict.
- **Negative-result success**: if screening rejects, document in `runs/754_bella_shape/result.md` with the per-period split AND write `memory/project_bella_distill_dead.md` retiring the Bella-as-aux-target mechanism class. This is the third independent failure of "Bella prior into the agent via aux loss"; recording it as such has strategic value (frees scope for Bella-as-MCTS-rollout in Path γ / TASK_722).

---

## 12. References

- `tasks/TASK_740_BELLA_DISTILL_HEAD.md` — per-cell predecessor.
- `tasks/TASK_745_BELLA_SEMANTICS.md` — column-mismatch bug + Option C is the seed for this task.
- `tasks/TASK_749_TITANS_PIVOT.md` — sibling planning-side experiment; structural template for this spec.
- `runs/740_bella_distill/result.md` — the regression numbers.
- `mcts_trader/jax_agent/muzero_model.py:72-144` — `AuxPriceHead` and `AuxBellaHead` insertion patterns.
- `train_phase2.py:1175-1206` — current aux-loss summation site.
- `train_phase2.py:1918-1930, 2128-2135` — CLI flag block + args translation.
- `train_easy.py:687-700, 752-...` — SegmentPool's Bella shard loader.
- `tools/precompute_bella_predictions.py` — precompute pipeline to extend.
- `diff_teacher/CLAUDE.md` — Bella architecture facts.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — load-bearing for the hypothesis.
- `experiments/QUEUE.md` — register the SPEC under Priority 3 once smoke passes.

---

## Operator notes

- The shape-descriptor schema is **versioned** (`schema_version=1` in the npz). Any future descriptor change bumps the version and triggers a precompute regen — silent target drift is the largest non-obvious risk in this task.
- Re-use the existing `data/bella_preds/` directory; the per-shard naming (`bella_shape_<period>__<hash>.npz`) lives alongside `bella_preds_<period>__<hash>.npz`. The manifest files are separate (`bella_shape_manifest.json` vs `bella_preds_manifest.json`) so the legacy 740 path is untouchable from the 754 code path.
- This is a planning-side / encoder-side mechanism (auxiliary loss on the encoder's latent). The TRM/cross-asset-attention/dynamics-MLP architecture is **not modified** — respects the `mcts_trader/CLAUDE.md` rule. The shape head is structurally identical to `AuxPriceHead`; reviewer can rubber-stamp the head class diff.

# 757_bella_shape — TASK_754 Bella Trajectory-Shape Distillation (paired 3-seed)

> **ID note**: TASK_754 spec §6.10 originally said `experiments/754_bella_shape/`.
> The orchestrator (and reviewer C7) requested the next free numerical ID. As
> of 2026-05-17 the highest experiment dir is `752_value_oracle_baseline_pair/`;
> 753-756 are *task* IDs (`tasks/TASK_75{3,4,5,6}_*.md`) with no matching
> experiment dirs. **757 is the next free.**

## Goal

Test whether the coarse permutation-invariant trajectory-shape descriptor
(TASK_754, β-v3) — a 14-float collapsed view of Bella's prediction window —
gives Alpha headroom over the canonical a1b1 baseline.

The hypothesis: per-asset per-step Bella distillation (TASK_740) hurt Alpha
by −1.92pp because the per-cell target lives in the classifier-Alpha
anti-correlation mechanism class (3-way confirmed in
`memory/project_classifier_alpha_anti_correlation.md`). The coarser,
permutation-invariant target extracts the regime-shape information Bella
*can* express reliably while discarding the per-cell noise it cannot, and
the moments+SVD descriptor structurally moves the mechanism out of the
classifier-sharpening class.

This is a **planning-side / encoder-side mechanism** (auxiliary loss on
the encoder's latent), respecting the `mcts_trader/CLAUDE.md` rule that
the TRM reasoning + dynamics architecture is not modified.

## Source task

`tasks/TASK_754_BELLA_TRAJECTORY_SHAPE_DISTILL.md` (commit `f44c1d3`,
SPEC) + `tasks/TASK_754_REVIEW.md` (commit `f7d49ac`, Phase 1 review).

## Background — multi-seed protocol

Post-TASK_743 (2026-05-14): single-seed Alpha gaps ≤ ~5pp are within noise.
Canonical baseline is a1b1 (rank+aux). Every mechanism claim needs
≥3-seed paired validation.

The a1b1 baseline runs at seeds 7/13/23 already exist from **741_mint**
Wave 1-3 (2026-05-14). This SPEC adds 3 variant runs and **reuses the
741_mint baseline** for the paired comparison — mirrors 734/739/749.

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] TASK_745 Option A column remap verified bit-exact (`runs/745_spot_check/verdict.md`, PASS 2026-05-17). The shape descriptor is permutation-invariant under asset reordering so the remap is belt-and-suspenders here, but the precompute path still applies it (which guarantees future lowrank/contrastive variants will see a remapped tensor).
- [x] `--bella-shape-distill` flag wired (this SPEC's implementation commits 6e12244..deed08b).
- [x] `compute_shape_descriptor()` in `tools/precompute_bella_predictions.py` — emits 14-float descriptor (4 cross moments + 4 horizon-1 + 4 last-horizon + 2 SVD), schema_version=1.
- [x] `AuxBellaShapeHead` class in `mcts_trader/jax_agent/muzero_model.py` — Dense → GELU → LayerNorm → Dense(14, zero-init).
- [x] Workstation smoke PASSED 2026-05-17 (commit `deed08b`):
  - Step 7 mini-batch (32 batch × 2 steps): `shape=0.76384(vf=1.00,mR=0.0003,sR=0.0019)`, total finite.
  - Step 8 full-batch (128 × 5 steps): per-step shape loss step1=0.4847 → step5=0.2477 (DROP), `bella_shape_valid_frac=1.0`, peak VRAM delta +25 MiB, no NaN.
- [ ] **NEW: full Bella precompute on server** — workstation does not have the production Bella shards. Server must run `tools/precompute_bella_predictions.py` (the legacy preds shard + the new shape shard are emitted together; legacy path is byte-additive). Estimated wall: ~16 s per period × 270 rotations ≈ 1.2 h on workstation (server should be comparable). Outputs to `data/bella_preds/` + manifests at `config/bella_preds_manifest.json` and `config/bella_shape_manifest.json`.
- [ ] a1b1 baseline runs at seeds {7, 13, 23} — `runs/741_mint/baseline_seed_{7,13,23}/`. Verify before launching.

## Resources

- 1× RTX 3090 per training process, ~20-22 GB at batch=128 (production batch
  per C2 of the review; 256 is the default but TASK_749 OOM precedent +
  Step 8 smoke at 128 sat at 21.6 GiB → stay at 128 for safety).
- Wall per run: ~2.5 h (no expected slowdown over a1b1; shape descriptor
  is precomputed offline, in-loop overhead is one small Dense forward per batch).
- TASK_742 rule: ≤2 concurrent train_phase2 per host.
- Total: 3 variant runs × 2.5 h with 2-wide batching = **~3.75 h training wall** + ~1.2 h precompute (one-time) + ~0.5 h eval.

## Plan

| Variant | encoder | shape-distill | seeds | source |
|---|---|---|---|---|
| **a1b1 baseline** | V3 minGRU | OFF | 7, 13, 23 | reuse from `runs/741_mint/baseline_seed_*/` |
| **bella_shape** | V3 minGRU + AuxBellaShapeHead | ON (mode=moments, weight=0.1, D=14) | 7, 13, 23 | new runs in this SPEC |

## Wave plan

```
Wave 0 (precompute, T=-1:15, ~1.2h):
  GPU A:  tools/precompute_bella_predictions.py
            (writes bella_preds_*.npz + bella_shape_*.npz + both manifests)

Wave 1 (T=0:00, ~2.5h):
  GPU A:  bella_shape seed 7
  GPU B:  bella_shape seed 13

Wave 2 (T=2:30, ~2.5h):
  GPU A:  bella_shape seed 23

T=5:00 — launch 3 evals in parallel
```

## Commands

### Wave 0 — precompute (server agent runs this)

```bash
cd ~/Trader && source trader_env/bin/activate
mkdir -p data/bella_preds
CUDA_VISIBLE_DEVICES=0 python tools/precompute_bella_predictions.py \
  --bella-checkpoint checkpoints/diff_teacher/generator_best/generator \
  --periods-file config/foxglove_periods.json \
  --output-dir data/bella_preds \
  --manifest config/bella_preds_manifest.json \
  --shape-manifest config/bella_shape_manifest.json \
  --horizons 1,5,10,20 \
  --k-samples 4 --window-size 60 --n-assets 64 \
  2>&1 | tee runs/757_bella_shape/precompute.log
```

The shape shards are emitted alongside the legacy preds shards
(additive — same `_process_period` call). When a legacy shard already
exists from TASK_740 precompute, the writer derives the shape shard
from the cached preds without re-running Bella (no Bella forward
needed for those entries).

### Waves 1-2 — training

```bash
cd ~/Trader && source trader_env/bin/activate

COMMON="--stage B --epochs 200 \
  --batch-size 128 \
  --periods-file config/foxglove_periods.json \
  --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.5 \
  --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 \
  --aux-price-weight 0.1 \
  --bella-shape-distill --bella-shape-mode moments \
  --bella-shape-weight 0.1 \
  --bella-shape-manifest config/bella_shape_manifest.json \
  --reset-bella-shape-head"

# For SEED in {7, 13, 23}, launch on free GPU per TASK_742 rule (≤2 concurrent):
mkdir -p runs/757_bella_shape/bs_seed_${SEED}
screen -d -m -S bs_s${SEED} bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU> python train_phase2.py $COMMON \
  --seed ${SEED} \
  --output-dir runs/757_bella_shape/bs_seed_${SEED} \
  --name 757_bs_s${SEED} 2>&1 \
  | tee runs/757_bella_shape/bs_seed_${SEED}/train.log; \
  echo BS_S${SEED}_DONE > /tmp/bs_s${SEED}_done.flag"
```

**Mutual-exclusion safety**: `--bella-shape-distill` and TASK_740's
`--bella-distill-head` raise ValueError at config-resolve time when set
together (enforced in `train_phase2.py:main()` after the args→config
translation block). Do not stack.

**Batch=128 (not 256)** is intentional per C2 of the review and Step 8
smoke evidence (peak VRAM 21.6 GiB at batch=128 with shape head ON).
Server-side: if VRAM permits, batch=256 is acceptable but should pair
against a re-run 256-batch a1b1 baseline, not the existing 128-or-256
741_mint baselines. See "Notes" below.

### Eval

```bash
# For SEED in {7, 13, 23}:
CUDA_VISIBLE_DEVICES=<GPU> python tools/benchmark_easy.py \
  --checkpoint runs/757_bella_shape/bs_seed_${SEED}/phase2_stageB_best.pkl \
  --periods-file config/benchmark_periods.json \
  --output-dir runs/757_bella_shape/eval_bs_seed_${SEED} \
  2>&1 | tee runs/757_bella_shape/eval_bs_seed_${SEED}/benchmark.log
# Done flag:
echo EVAL_BS_S${SEED}_DONE > /tmp/eval_bs_s${SEED}_done.flag
```

## Success criteria

### Smoke (gate: workstation only, already PASSED 2026-05-17 commit `deed08b`)

- [x] `compute_shape_descriptor` returns (14,) all-finite on synthetic (4, 64) preds with NaN columns. Permutation-invariance verified at max abs diff = 0 (Step 1).
- [x] Precompute path: 100-window descriptor `(100, 14)` all-finite on cached TASK_745 shard, |cross-mean| = 1.1e-5 (well inside R2 plausibility gate of <0.05). Byte-additive contract held (Step 2).
- [x] `AuxBellaShapeHead` init + apply on a (4, 128) latent returns (4, 14) zero — zero-init final Dense honored (Step 3).
- [x] `MuZeroNetworks(use_bella_shape_head=True)` → `aux_bella_shape` key in initial_inference output, shape (B, 14) (Step 4).
- [x] `Phase2Config(use_bella_shape_head=True)` via `create_model` → params has `aux_bella_shape_head`; mutual-exclusion raises ValueError when both 740 and 754 flags set (Step 5).
- [x] `AsyncBatchPipeline.get_batch` returns `bella_shape` of shape (B, 14) all-finite with the 1-period smoke manifest (Step 6).
- [x] Mini-batch (32 × 2 steps) train_step: `shape=0.76384(vf=1.00,mR=0.0003,sR=0.0019)`, total finite, no new NaN paths, CSV writer adds the new columns (Step 7).
- [x] Full-batch (128 × 5 steps) train_step: per-step `bella_shape_loss` step1 → step5 DROP (+0.237 in 5 steps), `bella_shape_valid_frac = 1.0`, peak VRAM delta ≤ +25 MiB (well under the +300 MB budget), no NaN (Step 8).

### Training (per-run)

- [ ] All 3 training runs complete 200 epochs cleanly (no NaN, no OOM)
- [ ] `bella_shape_loss` is finite throughout; reasonable magnitude (~1e-2 to 1.0 depending on residual scale)
- [ ] `bella_shape_valid_frac` ≥ 0.9 at every epoch (occasional dips during slot rotation acceptable)
- [ ] `bella_shape_mean_resid` and `bella_shape_std_resid` decreasing over the first 50 epochs (head is learning the regime-level shape)
- [ ] Wall ≤ 3 h per run

### Screening (paired 3-seed delta)

Per TASK_754 §8 + post-743 multi-seed gate:

- **PROMOTE to 5-seed** if mean Δα ≥ +1.5pp AND ≥ 2/3 positive AND worst seed ≥ −2pp.
- **HOLD** if mean Δα ∈ [−1pp, +1.5pp]. Examine per-period split; if `sideways` and `bear` both ≥ 0 then promote anyway (regime-shape prior should land hardest in low-information regimes).
- **REJECT** if mean Δα < −1pp OR worst seed < −5pp. Retire the trajectory-shape-distill mechanism class; document `memory/project_bella_distill_dead.md` (3rd independent failure of Bella-as-aux-target).

### Promotion (5-seed extension, conditional on PROMOTE above)

Seeds 31 + 42 added, then mean Δα ≥ +1pp AND no seed < −3pp (matches 734's promotion gate). On pass, swap canonical baseline to a1b1+shape-distill and update `experiments/QUEUE.md` + `lore/notes_<date>.md`.

## Done-flag names

- Training: `/tmp/bs_s7_done.flag`, `/tmp/bs_s13_done.flag`, `/tmp/bs_s23_done.flag`
- Eval: `/tmp/eval_bs_s7_done.flag`, `/tmp/eval_bs_s13_done.flag`, `/tmp/eval_bs_s23_done.flag`
- 5-seed extension (if reached): `/tmp/bs_s31_done.flag`, `/tmp/bs_s42_done.flag` + matching eval flags.

## Output expectations

```
runs/757_bella_shape/
├── precompute.log
├── bs_seed_7/         train.log, phase2_stageB_best.pkl, stage_b_metrics.csv, stage_b_config.json
├── bs_seed_13/
├── bs_seed_23/
├── eval_bs_seed_7/    benchmark.json, benchmark.log
├── eval_bs_seed_13/
├── eval_bs_seed_23/
└── result.md          paired delta table + per-period split + diagnostic
                       (bella_shape_loss, bella_shape_mean/std_resid
                       trajectories per seed)
```

`result.md` template should include: per-seed Δα at every benchmark
period, mean/STD across seeds, per-component residual scale at epoch
{50, 100, 200} (so we can tell if a particular descriptor component
is being ignored). If REJECT, also include a "what to retire" section
following the `project_bella_distill_dead.md` pattern.

## Notes

### Baseline status (2026-05-16)

734-x5 did NOT promote (mean Δα +0.20pp at 5-seed after s42 −12.71pp outlier). Canonical Stage-B baseline stays a1b1. Pair this SPEC against `runs/741_mint/baseline_seed_*/`.

### Stacked variant (Tier-D)

If 754 promotes and 749 (Titans-MAG) also promotes (or any other Tier-2 candidate), the stacked variant becomes interesting. Out of scope here.

### What "trajectory shape" means here

We implement only the **moments + SVD-spectrum** descriptor (TASK_754 §4.1 + reviewer §3.C3). Per the spec:

- 4 cross moments (mean, std, skew, excess_kurt) over `flatten(K, n_assets)`
- 4 moments restricted to horizon 1
- 4 moments restricted to the last (h=20) horizon
- 2 SVD spectrum values: `s1`, `s1 / sum(s_i)` of the `(K, n_assets)` matrix

Per-component weight default `(2.0, 1.0, 0.5, 0.25, 1.0, 2.0)` broadcasts as `(w_mean, w_std, w_skew, w_kurt, w_s1, w_s1_ratio)` (cross + h1 + hlast all use the same `(w_mean..w_kurt)` quartet). The lowrank and contrastive variants (TASK_754 §4.2) raise `NotImplementedError` and are reserved for TASK_754b/c.

### Mutual exclusion with TASK_740

`--bella-shape-distill` and `--bella-distill-head` are mutually exclusive at config-resolve time (enforced in `train_phase2.py:main()`). To run TASK_740 instead, drop the shape flag and use the TASK_740 manifest. **Both shards (preds + shape) live in `data/bella_preds/`; they are byte-independent.**

### Anti-correlation hygiene

The descriptor is permutation-invariant under asset reordering. This neutralises the TASK_745 column-mismatch class entirely for the moments + SVD path. The lowrank variant would re-introduce per-asset alignment via singular vectors (TASK_754 §4.3) — but it isn't in scope here, so no extra check needed.

### Compute / VRAM

The head is a single Dense(state_dim → 256) + Dense(256 → 14) module. Adds ~0.04 M params (state_dim=64 × 256 + 256 × 14 + biases ≈ 20 k). In-loop overhead is one small forward pass per batch. The per-window descriptor is 14 floats (~56 bytes); shard size is `T_ret × 14 × 4 ≈ 56·T_ret` bytes — about **3.6× smaller than TASK_740's per-cell tensor** (256·T_ret bytes). Across 270 rotations × ~10k windows × 56 B ≈ 150 MB on disk total.

### Critical flag for engagement

`--reset-bella-shape-head` is REQUIRED on the first run from any pre-754 ckpt. Without it, `initialize_missing_aux_heads` adds the head fresh anyway (which is correct), but having the explicit `--reset-bella-shape-head` flag makes the intent visible in the train.log.

### Operator notes

- **The shape-descriptor schema is versioned** (`schema_version=1`). Any future descriptor change bumps the version + triggers a precompute regen.
- **Server is the only place this gets run for production** — workstation only has the 1-period TASK_745 cached shard. Workstation SPEC validation is the only purpose of the smoke manifest at `/tmp/bella_shape_manifest_smoke.json`; production manifest path is `config/bella_shape_manifest.json` written by the precompute Wave 0 step.
- **Cleanup after rejection**: if REJECT and `project_bella_distill_dead.md` is written, leave the implementation in place for posterity but flip the QUEUE entry to "RETIRED". Re-enabling later for a TASK_754b lowrank attempt should NOT require code changes — only a SPEC.

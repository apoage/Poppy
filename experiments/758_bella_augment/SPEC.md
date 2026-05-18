# 758_bella_augment — TASK_753 Bella as Data Augmentation (paired 3-seed)

> **ID note**: TASK_753 spec §10 originally suggested `experiments/753_bella_augment/`.
> Per the orchestrator (and the same numbering convention TASK_754 used in
> 757_bella_shape) the next free numerical experiment dir is **758**
> (752 used by `value_oracle_baseline_pair`; 757 by `bella_shape`; 753-756
> are *task* IDs in `tasks/TASK_75{3,4,5,6}_*.md` with no matching experiment
> dirs).

## Goal

Test whether wiring Bella as a **data augmentation source** (generator of
raw OHLCV that flows through the same encoder path as real history) gives
Alpha headroom over the canonical a1b1 baseline.

The hypothesis: TASK_740 (Bella as distill aux teacher) regressed by
−1.92pp because the per-asset per-step distill target lives in the
classifier-Alpha anti-correlation mechanism class (3-way confirmed in
`memory/project_classifier_alpha_anti_correlation.md`). TASK_754 (shape
distill) is a coarser version of the same teacher mechanism. **TASK_753
sidesteps both** — Bella's prior reaches the model purely through the
distribution of training data, not through a new loss head. If Bella's
learned market prior carries information the encoder cannot extract from
60-step OHLCV alone, the regularising effect should improve held-out
benchmark Alpha. If Bella's distribution diverges from real markets,
augmentation will either hurt or do nothing — both honest answers.

This is a **data-side mechanism** (no encoder surgery, no new loss head,
no auxiliary signal). The TRM reasoning + dynamics + heads are all
byte-identical to a1b1. Only `hm.cache_data` of some pool slots changes.

## Source task

`tasks/TASK_753_BELLA_AS_DATA_AUGMENTATION.md` (commit `287bcb7`, SPEC)
+ `tasks/TASK_753_REVIEW.md` (commit `94aa8b6`, Phase 1 review).

Implementation commits: `3b6446b..1cb7caa` (step 0/8 through step 7/8).

## Background — multi-seed protocol

Post-TASK_743 (2026-05-14): single-seed Alpha gaps ≤ ~5pp are within
noise. Canonical baseline is a1b1 (rank+aux). Every mechanism claim needs
≥3-seed paired validation.

a1b1 baseline runs at seeds 7/13/23 already exist from **741_mint**
Wave 1-3 (2026-05-14). This SPEC adds 3 variant runs and reuses the
741_mint baseline for the paired comparison — mirrors 734/739/749/757.

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] TASK_745 Option A column remap verified bit-exact
  (`runs/745_spot_check/verdict.md`, PASS 2026-05-17). The `BellaMarket`
  reuse of `_load_bella_canonical_assets` and `_build_remap` matches the
  precompute path exactly.
- [x] Bella generator ckpt at `checkpoints/diff_teacher/generator_best/generator`
  (orbax format, ~1 GB). Loads as: arch=conv, hidden_dim=768, num_layers=12,
  seq_length=400, n_steps=10, n_assets=64, n_features=5. **Note**: NOT
  302M params / 1536d / 16L / single-step as TASK_753 spec §3 claimed.
  Spec was stale (C2 of the review); the implementation uses runtime
  values via `setup_bella_generator`.
- [x] `--bella-augment` flag wired (this SPEC's implementation commits
  `3b6446b..1cb7caa`).
- [x] `BellaMarket` class in `mcts_trader/data/bella_market.py` — drop-in
  replacement for `HistoricalMarket` with chained `generate_on_device`
  calls + OHLCV reconstruction mirroring `setup.py:660-700`. C4 caveat
  (OHLCV reconstruction is non-trivial) addressed by inlining the
  reconstruction logic, NOT by reusing `generate_bella_prices_on_device`
  (we need per-chunk seed control for chained calls).
- [x] Workstation smoke PASSED 2026-05-18 (commit `1cb7caa`):
  - Step 7 mini-batch (32 batch × 1 step, ratio=1.0): `bella_aug=slot1.00 mix=v=1`, total finite (27.77), 30s wall.
  - Step 8 production-batch (128 batch × 5 steps, n_slots=4 → 3 unique):
    - ratio=0.25 trial: rng landed 0 Bella slots — `bella_aug=slot0.00 mix=none`, total finite (36.91), 34s wall.
    - ratio=1.0 forced-Bella trial: 3/3 slots Bella, mix=r=1,s=1,v=1, total finite (36.05), 16s wall. No OOM at batch=128. Bella oracle wall 4.2s vs historical 4.4s (C8 signal: ratio 0.95×, well within 1.5× tolerance).
- [ ] a1b1 baseline runs at seeds {7, 13, 23} — `runs/741_mint/baseline_seed_{7,13,23}/`. Verify before launching.

## Resources

- 1× RTX 3090 per training process. Step 8 smoke at batch=128 with 3
  Bella slots sat comfortably (no OOM with 24 GB VRAM available).
  Recommend **batch=128** for safety (matches TASK_754 SPEC C2; bumping
  to 256 should pair against a re-run 256-batch baseline, not the
  existing 128-or-256 741_mint baselines).
- Wall per run: ~2.5 h estimated (no expected slowdown over a1b1; Bella
  generation is a one-shot at slot construction, ≤ 5s per slot after
  generator load; subsequent training is identical to a1b1).
- TASK_742 rule: ≤2 concurrent train_phase2 per host. Bella augment
  adds ~1.5 GB VRAM for the generator + ~MB per Bella slot's OHLCV
  buffer; recommend ≤1 concurrent run when `--bella-augment` is on.
- Total: 3 variant runs × 2.5 h with single-wide batching =
  **~7.5 h training wall** (no separate precompute step — Bella is loaded
  in-process at pool init).

## Plan

| Variant | encoder | bella-augment | seeds | source |
|---|---|---|---|---|
| **a1b1 baseline** | V3 minGRU | OFF | 7, 13, 23 | reuse from `runs/741_mint/baseline_seed_*/` |
| **bella_augment** | V3 minGRU (identical) | ON (ratio=0.25, mode=append, regime=uniform, asset_policy=remap, seq_length=800) | 7, 13, 23 | new runs in this SPEC |

## Wave plan

```
Wave 1 (T=0:00, ~2.5h):
  GPU A:  bella_augment seed 7

Wave 2 (T=2:30, ~2.5h):
  GPU A:  bella_augment seed 13

Wave 3 (T=5:00, ~2.5h):
  GPU A:  bella_augment seed 23

T=7:30 — launch 3 evals in parallel
```

> If GPU B (second 3090) is available and not running another
> `train_phase2.py`, fold the seeds onto two GPUs (Waves 1+2 in parallel,
> then Wave 3 alone, then evals). The TASK_742 ≤2 concurrent rule
> permits this with Bella augment because the generator weights are
> per-process — no shared VRAM contention.

## Commands

### Training (workstation OR server)

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
  --bella-augment \
  --bella-augment-ratio 0.25 \
  --bella-augment-mode append \
  --bella-augment-regime-mode uniform \
  --bella-augment-asset-policy remap \
  --bella-augment-seq-length 800"

# For SEED in {7, 13, 23}, launch on free GPU per TASK_742 rule (≤2 concurrent):
mkdir -p runs/758_bella_augment/ba_seed_${SEED}
screen -d -m -S ba_s${SEED} bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU> python train_phase2.py $COMMON \
  --seed ${SEED} \
  --output-dir runs/758_bella_augment/ba_seed_${SEED} \
  --name 758_ba_s${SEED} 2>&1 \
  | tee runs/758_bella_augment/ba_seed_${SEED}/train.log; \
  echo BA_S${SEED}_DONE > /tmp/ba_s${SEED}_done.flag"
```

**Mutual-exclusion safety**: `--bella-augment`, `--bella-distill-head`
(TASK_740), and `--bella-shape-distill` (TASK_754) are mutually exclusive
at config-resolve time (enforced in `train_phase2.py:main()` after the
args→config translation block). Do not stack any pair.

**Mode mutual-exclusion**: `--bella-augment-mode replace` and
`--bella-augment-mode interleave` raise `NotImplementedError` (reserved
for TASK_753b). Only `append` is implemented.

**Batch=128 (not 256)** matches TASK_754 SPEC C2 precedent and Step 8
smoke evidence. Server-side: if VRAM permits, batch=256 is acceptable
but should pair against a re-run 256-batch a1b1 baseline.

### Eval

```bash
# For SEED in {7, 13, 23}:
CUDA_VISIBLE_DEVICES=<GPU> python tools/benchmark_easy.py \
  --checkpoint runs/758_bella_augment/ba_seed_${SEED}/phase2_stageB_best.pkl \
  --periods-file config/benchmark_periods.json \
  --output-dir runs/758_bella_augment/eval_ba_seed_${SEED} \
  2>&1 | tee runs/758_bella_augment/eval_ba_seed_${SEED}/benchmark.log
# Done flag:
echo EVAL_BA_S${SEED}_DONE > /tmp/eval_ba_s${SEED}_done.flag
```

## Success criteria

### Smoke (gate: workstation only, already PASSED 2026-05-18 commit `1cb7caa`)

- [x] **Step 0**: `_exhausted_groups = set()` defensive init in
  `SegmentPool.__init__`. Direct `.add()` test passes; pool with empty
  spec exits cleanly through expected `ValueError` (not `AttributeError`).
- [x] **Step 1**: `BellaMarket` 1-chunk + 2-chunk smoke. cache_data.shape
  matches expected, all-finite, OHLC > 0, 22 unknown columns NaN-filled
  (TASK_745 remap path). 1st call 115s (JIT-dominated), 2nd call 17s,
  3rd call 0.08s (cached JIT).
- [x] **Step 2**: `Phase2Config(bella_augment=True)` round-trips; all 11
  fields accessible with expected defaults.
- [x] **Step 3**: `_make_bella_slot` + `_finalize_slot` refactor smoke.
  Bella slot constructs successfully; oracle wall ratio Bella/Hist =
  0.98× (within C8's 1.5× tolerance); oracle cumPnL ratio Bella/Hist =
  0.03× (Bella HARDER than historical, OPPOSITE direction to C3 "too
  easy" warning).
- [x] **Step 4**: Coin-flip injection. At ratio=1.0, all slots Bella;
  at ratio=0.0, all slots historical. `replace`/`interleave` modes raise
  `NotImplementedError`.
- [x] **Step 5**: All 12 `--bella-augment-*` flags registered; argparse
  parses (ratio=0.5) OK; out-of-range ratio (1.5) raises ValueError;
  mutual-exclusion with TASK_740/754 raises ValueError.
- [x] **Step 6**: `load_segment_pool` oracle_cfg wiring. At ratio=1.0
  through CLI → pool init produces all-Bella slots; `bella_meta` on each
  slot populated with regime + seed + unknown_cols + build_wall.
- [x] **Step 7**: Minimal Stage B step. CSV gains `bella_slot_frac`,
  `bella_regime_mix` columns; E-line gains `bella_aug=slot{frac} mix=...`
  segment.
- [x] **Step 8 (LOAD-BEARING)**: Full `stage_b_train_step` at
  batch=128, n_slots=4 (capped to 3 unique groups):
  - ratio=0.25: 0 Bella slots (rng), all losses finite, 34s wall, no OOM.
  - ratio=1.0: 3 Bella slots (r=1,s=1,v=1), all losses finite (36.05),
    16s wall, no OOM. Bella build wall 0.10s per slot post-JIT.

### Training (per-run)

- [ ] All 3 training runs complete 200 epochs cleanly (no NaN, no OOM)
- [ ] `bella_slot_frac` ≈ 0.25 ± 0.10 across epochs (rolling rotation
  preserves the ratio). Larger excursions OK in early epochs.
- [ ] `bella_regime_mix` shows non-trivial spread across regimes (no
  single regime dominates the slot pool).
- [ ] No NaN in any loss across all 200 epochs.
- [ ] Wall ≤ 3 h per run.
- [ ] No `[POOL:BELLA] Bella slot construction failed` warnings (the
  fallback-to-historical path should not engage in production).

### "Too easy" tag protocol (C3 decision from operator)

Per the operator's C3 decision: if at training time the per-slot oracle
PnL ratio Bella/Hist is > 2.0× **on average across a rolling window of
recent Bella slots**, the experiment is tagged "too easy" in
`runs/758_bella_augment/result.md`.

**Operational definition** (for result.md authoring):

1. From `runs/758_bella_augment/ba_seed_*/stage_b_training.log`, grep
   for `[POOL] Loaded ... src=bella oracleCumPnL=X` lines and
   `[POOL] Loaded ... src=historical oracleCumPnL=Y` lines.
2. Compute median(bella PnL / median(historical PnL across same period
   window) per slot construction.
3. If median ratio > 2.0×, tag the experiment "too easy" in the
   verdict section. Do NOT abort the run — the signal is informational,
   gating happens at the 3-seed Alpha gate.

**Workstation step 8 smoke observation**: ratio=1.0 trial gave Bella
PnL {9.88, 10.21, 20.29} vs historical {440.0, 280.8, 331.7}, **ratio
≈ 0.03× — Bella is HARDER than historical, NOT easier**. The "too easy"
flag is not expected to fire on this stack, but the protocol stands.

Note: oracle cum PnL scales with T_ret (Bella slots are 800 steps;
historical slots are ~258k steps). The per-step normalised ratio is
the right object — `bella_pnl / 800 ÷ hist_pnl / hist_T_ret`.
Workstation smoke per-step: Bella ≈ 0.0124/step, Historical ≈ 0.0017/step.
**Per-step Bella IS higher than historical** — 7×. This is the noise
threshold and likely the right calibration point for the C3 flag: if
per-step Bella oracle PnL > 2.0× per-step historical, tag "too easy".
The 7× observed at workstation suggests the augmentation may indeed be
"too easy" relative to the historical baseline. Document this in the
result.md regardless of the final Alpha verdict.

### Screening (paired 3-seed delta)

Per TASK_753 §8:

- **PROMOTE to 5-seed** if mean Δα ≥ **+3pp** AND all 3 seeds positive
  AND no seed worse than −2pp. Same gate that 734_flow_matching passed
  (`project_flow_matching_promising.md`).
- **HOLD** if mean Δα ∈ [+0pp, +3pp]. Examine per-period split; if
  `volatile` and `crash` regimes both show >0, consider follow-up with
  `--bella-augment-regime-mode match` (route real-period regime stats to
  Bella conditioning).
- **REJECT** if mean Δα < 0 OR worst seed < −5pp. Document
  `memory/project_bella_augment_dead.md` (3rd independent failure of
  Bella-as-mechanism on this stack: distill → fail, shape distill →
  conditional on 757, augmentation → fail).

### Promotion (5-seed extension, conditional on PROMOTE)

Seeds 31 + 42 added, then mean Δα ≥ +1pp AND no seed < −3pp (matches
734 / 757 promotion gate). On pass, swap canonical baseline to a1b1 +
bella_augment and update `experiments/QUEUE.md` + `lore/notes_<date>.md`.

## Done-flag names

- Training: `/tmp/ba_s7_done.flag`, `/tmp/ba_s13_done.flag`, `/tmp/ba_s23_done.flag`
- Eval: `/tmp/eval_ba_s7_done.flag`, `/tmp/eval_ba_s13_done.flag`, `/tmp/eval_ba_s23_done.flag`
- 5-seed extension (if reached): `/tmp/ba_s31_done.flag`, `/tmp/ba_s42_done.flag` + matching eval flags.

## Output expectations

```
runs/758_bella_augment/
├── ba_seed_7/         train.log, phase2_stageB_best.pkl, stage_b_metrics.csv, stage_b_config.json
├── ba_seed_13/
├── ba_seed_23/
├── eval_ba_seed_7/    benchmark.json, benchmark.log
├── eval_ba_seed_13/
├── eval_ba_seed_23/
└── result.md          paired delta table + per-period split + diagnostic
                       (per-slot Bella oracle PnL ratio, bella_slot_frac
                       trajectory, bella_regime_mix) + "too easy" tag
                       per protocol above
```

`result.md` template should include: per-seed Δα at every benchmark
period, mean/STD across seeds, the C3 "too easy" computation
(per-step Bella vs historical oracle PnL median ratio), the
`bella_slot_frac` trajectory across the 200 epochs (sanity that
augmentation engaged consistently), and the `bella_regime_mix` final
counts. If REJECT, also include a "what to retire" section following
the `project_bella_distill_dead.md` pattern.

## Notes

### Bella seq_length seam tolerance (C1 from review)

The TASK_753 review's C1 caveat was that `generator.generate(seq_length=1440)`
should be used per the spec's 1440 default. Implementation finding:
**this is BROKEN in the current ckpt** — `generator.generate()` with
`seq_length` override raises a ValueError about conv padding shape
mismatch (`unsupported pad_width with nd=4`). Implementation fell back to
option (b) — chained `generate_on_device` calls — and uses 800 as the
default (2 native-T=400 chunks). The encoder sees a small discontinuity
in price level at every native_T boundary; the oracle path solver
tolerates this for `lookahead ≤ 20` (the lookahead window is small
relative to the chunk size). The smoke confirms oracle PnL is finite
across the seam.

### Bella architecture facts (C2 from review)

TASK_753 spec §3 cited stale architecture facts from `diff_teacher/CLAUDE.md`:
"OT-CFM flow matching, conv generator (302M params, 1536d, 16L),
1440-step generation window, single-step inference." Actual ckpt facts
(from `config.json`):
- `flow_architecture: "conv"`
- `flow_hidden_dim: 768` (NOT 1536)
- `flow_layers: 12` (NOT 16)
- `window_size: 400` (NOT 1440)
- `n_steps: 10` ODE integration steps (NOT single-step)

`setup_bella_generator` logs the actual arch at load time; the implementation
does NOT hardcode any of the stale values. Future ckpt swaps should
update this SPEC's "Prerequisites" section accordingly.

### OHLCV reconstruction (C4 from review)

`BellaMarket` does NOT use `generate_bella_prices_on_device` directly —
that function samples regime randomly per `n_envs` and we need
deterministic regime control for chained chunks. Instead, the
reconstruction logic from `setup.py:660-700` is inlined inside
`BellaMarket.__init__`:
- Channel 3 (close): cumsum(raw × 0.015) → exp() × 100
- Channel 0 (open): previous-close shift
- Channel 1 (high): close × (1 + |raw| × 0.015)
- Channel 2 (low): close × (1 − |raw| × 0.015)
- Channel 4 (volume): max(exp(raw × 0.5 + 10), 1)

NaN columns (where the rotation asset is not in Bella's canonical
universe) get **cyclic-pad filled from neighbouring valid columns** —
the encoder cannot consume NaN, and we are not masking targets here.
This differs from TASK_745 distill, which masks NaNs at loss-time only.

### Mutual exclusion

`--bella-augment` is mutually exclusive with `--bella-distill-head`
(TASK_740) AND with `--bella-shape-distill` (TASK_754) at config-resolve
time. To stack augmentation with shape distill, that would be a TASK_753b
follow-up (mode=append+stack).

### Anti-correlation hygiene

Augmentation is a **data-side mechanism** — gradient pathway is unchanged
from a1b1. It does NOT fit the classifier-Alpha anti-correlation pattern
documented in `memory/project_classifier_alpha_anti_correlation.md`,
which strictly applies to head-side / loss-side sharpening mechanisms.
If augmentation happens to make the encoder a better classifier
(improving Bal.Acc by exposing it to more regime variation), the
trade-rate-explosion failure mode may still trip — but that would be a
genuinely different anti-correlation than 737/738/740/741 documented.
Watch for it in the result.md.

### Compute / VRAM

Bella generator weights ~600 MB at fp32 (~300 MB fp16). Loaded once at
pool init. Stage-B working set ~18 GB + MCTS rollouts on a 24-GB 3090
is tight but Step 8 smoke confirmed no OOM at batch=128 with 3 Bella
slots. Per-slot OHLCV buffer ~1.6 MB (800 × 64 × 5 × 4 bytes).

### Operator notes

- **No precompute step**. Bella is loaded in-process at pool init. The
  first Bella slot pays ~115s (JIT compile + canonical asset load);
  subsequent slots are ~0.1s each (cached JIT, generator weights resident
  in VRAM).
- **`bella_augment_seq_length=800` is the recommended default**. The
  spec's 1440 default is not achievable with the current ckpt; the
  workstation step 8 smoke validated 800. Smaller values (400) work but
  give less training-data density per Bella slot.
- **`--bella-augment-regime-mode match` is a STUB** in v1 (behaves like
  `uniform`). If `uniform` regresses, the follow-up SPEC TASK_753b should
  implement true regime matching (estimate period regime from real
  returns, then condition Bella generation on the matching regime).
- **`bella_meta` is informational only** — `AsyncBatchPipeline` does not
  branch on slot source. Downstream of `_finalize_slot`, real and Bella
  slots are indistinguishable from the trainer's perspective.

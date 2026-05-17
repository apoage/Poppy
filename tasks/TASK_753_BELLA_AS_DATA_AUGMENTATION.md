# TASK_753 — Bella as Data Augmentation (Generator, not Teacher)

**Status**: DRAFTED 2026-05-17. Spec only — no code yet.
**Source**: 740_bella_distill soft regression (paired 3-seed Δα **−1.92pp**, `runs/740_bella_distill/result.md:14-22`).
**Motivation**: Bella as a *teacher* (distillation aux head) hurt Alpha. The mechanism class is suspect on this stack (cf. `project_classifier_alpha_anti_correlation.md`). But Bella as a *concept* — a market generator trained on the full distribution at Foxglove-scale — is still sound. This task explores the alternative wiring: **use Bella to generate the data the model trains on**, not to teach the encoder a forecast target.

**Confidence**: medium. We have working precedent (`SyntheticMarket` is already wired into Stage-A via `train_easy.py:60`), Bella generator is deployed (`checkpoints/diff_teacher/generator_best`), and the column-mismatch lesson from TASK_745 is now well-understood. The hard parts are (a) Bella generation cost vs training throughput, and (b) whether Bella's regime distribution looks anything like real-bull / real-crash.

---

## 1. Goal

Wire Bella into the Stage-B training data pipeline as an **augmentation source** for OHLCV windows, side-by-side with the historical SegmentPool feed (`train_easy.py:453`). Train Poppy on a mix of (real, Bella-generated) windows controlled by a CLI ratio. The model still receives identical `(B, 27, ws, A)` observation tensors and identical oracle/policy/value targets; the only thing that changes is *where the raw price series came from*. No new loss heads, no auxiliary distillation, no encoder surgery. Success = paired 3-seed mean Δα ≥ +3pp against 741_mint baseline_seed_{7,13,23}, all 3 positive.

The hypothesis is that "Bella as data" is a strictly cheaper experiment than "Bella as teacher" because it adds zero alignment surfaces (no per-asset semantic remap, no horizon-matching, no log-return-space alignment) and tests the *same underlying claim* — that Bella's learned market prior carries information the encoder cannot extract from 60-step OHLCV windows alone.

---

## 2. Hypothesis

Distillation (TASK_740) failed because:

1. **Encoder gradient channel narrow** — `AuxBellaHead` injects a per-asset MSE on forward-rollout log-returns. That gradient must squeeze through one bottleneck of the global latent. Anti-correlation pattern (`project_classifier_alpha_anti_correlation.md`) says classifier-shaped signals on this stack consistently trade Bal.Acc/loss gains for Alpha losses.
2. **Alignment surfaces are unforgiving** — TASK_745 documented one silent bug (column order); the remap fix shipped, but the residual regression suggests the *mechanism class* (sharpen-the-encoder) is the problem, not just the column bug.

Augmentation should sidestep both:

- The model's loss surface is **unchanged**: BC / value / consistency / direction / rank / aux_price exactly as in 741_mint. No new head, no new gradient pathway, no "teacher" interpretation.
- Bella shows up only as raw price series → goes through the **same** 27-channel encoder pipeline as historical data. The encoder is forced to extract features that generalise across both real and Bella-distributed markets. If Bella's prior is informative, the regularising effect should improve held-out (real-market) benchmark Alpha; if Bella's distribution is too far from real, augmentation will either hurt or do nothing — both honest outcomes.

This is the "Bella as data, not as teacher" framing. It is also closer to the existing `SyntheticMarket` curriculum path (`mcts_trader/data/synthetic_market.py:5`) — we are replacing/augmenting GBM-style synthetic data with learned-prior synthetic data.

---

## 3. Background / current state

**Files to read first**:

- `~/Trader/diff_teacher/CLAUDE.md` — Bella architecture: OT-CFM flow matching, conv generator (302M params, 1536d, 16L), 1440-step generation window, 5 regimes + 1 unconditional, single-step inference.
- `~/Trader/tools/precompute_bella_predictions.py:140-179` — current Bella usage path. `generator.generate_on_device(n_samples=K, seed=, regime=)` returns `(K, seq_len, n_assets_bella, 5)` OHLCV-shaped output; channel 3 carries z-scored per-step returns (`tools/precompute_bella_predictions.py:166-167`). Volatility scale `0.015` is hard-coded (`tools/precompute_bella_predictions.py:166`).
- `~/Trader/mcts_trader/training/setup.py::setup_bella_generator` — the loader (`tools/precompute_bella_predictions.py:133-134`).
- `~/Trader/train_easy.py:453-720` — `SegmentPool`. Slots carry `hm` (HistoricalMarket), `oracle_selections`, `returns`, `state_channels`, `policy_targets`, `confidence`, plus optional `bella_preds_full` from TASK_740. **Augmentation must add a new slot type** carrying Bella-generated OHLCV in place of `hm.cache_data`, with all downstream precompute (oracle, state channels, policy targets, confidence) computed on the synthetic series exactly as for real data.
- `~/Trader/train_phase2.py:484-693` — `AsyncBatchPipeline`. Pulls `per_slot` positions from `SegmentPool` slots, packs `(B, win_len, n_assets, 5)` (`train_phase2.py:676`). Augmentation hook is at slot selection — pipeline doesn't need to know whether a slot came from history or Bella.
- `~/Trader/mcts_trader/data/synthetic_market.py:5-50` — legacy GBM-style synthetic generator, already used by `train_easy.py:60`. Same OHLCV interface (`(n_steps, n_assets, 5)`), same downstream contract. This is the **precedent**: we already inject synthetic data through the slot abstraction. Bella augmentation extends the same pattern.
- `~/Trader/tasks/TASK_745_BELLA_SEMANTICS.md:1-99` — per-rotation column remap is **mandatory**. Bella's output columns are DB-ID-ordered; trainer expects rotation-specific asset order. The Option A remap (`tools/precompute_bella_predictions.py:222-247`) must be applied here too even though we're not distilling — the oracle/state-channel precompute is per-asset and order-sensitive.
- `~/Trader/runs/740_bella_distill/result.md:14-22` — the regression we are responding to.

**Stage-B batch dims** (load-bearing for memory budget below):

- `B = stage_b_batch_size = 256` (`phase2_config.py:162`)
- `A = 64` assets
- `ws = window_length ≈ 60` (window into history; oracle lookahead is separate)
- Slot `hm.cache_data`: `(T, A, 5)` where `T` is the full period length (often ~50k–100k steps).
- Per-step encoder input: `(B, 27, ws, A)` after `jax_encoder` build-up.

---

## 4. Design

### 4.1 Dataflow

```
                 ┌──────────────────────┐
                 │  SegmentPool (real)  │ ← unchanged path
                 │  hm.cache_data       │   (history slots)
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐    sample_real/sample_bella
                 │   Slot selector      │ ←─ controlled by --bella-augment-ratio
                 └──────────┬───────────┘
                            │
              ┌─────────────┴──────────────┐
              │                            │
   ┌──────────▼───────────┐    ┌───────────▼─────────────┐
   │  BellaSlotFactory    │    │   (existing real slot)  │
   │  - regime sample     │    └───────────┬─────────────┘
   │  - generator.generate│                │
   │  - column remap →    │                │
   │    rotation order    │                │
   │  - oracle precompute │                │
   │  - state/policy/conf │                │
   └──────────┬───────────┘                │
              │                            │
              └─────────────┬──────────────┘
                            │
                 ┌──────────▼───────────┐
                 │  AsyncBatchPipeline  │ ← unchanged
                 │  (B, 27, ws, A)      │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │   train_step_fn      │ ← unchanged
                 └──────────────────────┘
```

**Key invariant**: downstream of the slot selector, **nothing knows whether the slot is real or Bella-generated**. All precompute (oracle path solver, state channels, policy targets, confidence) runs on the synthetic OHLCV the same way it runs on real OHLCV.

### 4.2 Where Bella plugs in

- **Slot factory mode**: a new method `SegmentPool._make_bella_slot(group_key, regime)` that mirrors `_make_slot` (`train_easy.py:562`) but with `hm.cache_data` replaced by a Bella-generated OHLCV tensor of the same shape and the same downstream precompute. It returns the same dict schema as `_make_slot` (so `AsyncBatchPipeline` is unchanged).
- **Selection mode**: a new pool config field `bella_augment_ratio ∈ [0, 1]`. At each call to `SegmentPool._pick_next_slot()` (or wherever the pool advances cursors), with probability `bella_augment_ratio` we take a Bella slot; else a real slot.
- **Three modes** controlled by `--bella-augment-mode`:
  - `replace`: each Bella slot replaces a real slot in the rotation (real slot pool shrinks). Useful for ablation; we'd compare "100% Bella" to "100% real" head-to-head.
  - `append`: Bella slots are added to the pool alongside real slots (pool grows). Default. Pool selection then naturally averages over both per epoch.
  - `interleave`: at each batch composition pass, with probability `bella_augment_ratio` swap one slot's data with a freshly-generated Bella tensor of the same shape. Cheapest at startup; most Bella regen per epoch (cost-heavy unless we cache).

### 4.3 Memory hierarchy placement (per `CLAUDE.md` Efficiency Philosophy)

- **Bella generator weights**: VRAM-resident, loaded once at pool init. ~600 MB at fp16. Hosted on GPU 0 (workstation) — same device as training. On server (GPU 1 = RTX 3090), Bella shares VRAM with training.
- **Bella OHLCV generation**: VRAM during `generate_on_device` call. Output buffer per call: `(K, seq_len=1440, A=64, 5)` fp32 = `K * 1.84 MB`. For `K=4`, that's ~7 MB. Negligible.
- **Generated slot tensor**: copied to RAM via `np.array(generated)` (mirrors current pattern in `tools/precompute_bella_predictions.py:167`). Slot lives in RAM (Tier 3), same as real slots. Stage-B batch pulls into VRAM via the existing pipeline.
- **Generation cadence**: see §4.4. Generate on-the-fly into RAM at slot-construction time, NOT precomputed-to-disk. This matches `CLAUDE.md` Anti-pattern rule "Pre-generate training data to npz/pkl then load" (which is exactly what 740 did and we are now moving away from).
- **Oracle precompute**: CPU/JAX-accelerated on the generated tensor, exactly as for real slots (`train_easy.py:594-651`). Cached in the slot dict for the lifetime of the slot.

### 4.4 Generation cost budget

Bella conv generator: ~50 ms per single-step inference for `K=4` samples at `seq_len=1440` on RTX 3090 (estimate from TASK_740 precompute logs). One Bella slot covers `seq_len = 1440` timesteps → ~30 batches' worth of training data at `B=256, ws=60, stride=8`. A pool of 8 Bella slots costs **~400 ms one-shot at pool init**. After warm-up, regenerate one slot every N training steps (default: never re-generate within an epoch; refresh between epochs).

Compare to TASK_740 precompute (~17h wall for 270 rotations × stride=64). On-the-fly approach is **orders of magnitude cheaper** because we don't need a Bella prediction for every window — we just need *enough Bella-generated trajectories to populate the pool*.

### 4.5 Regime conditioning

Bella supports 5 regimes (`crash`, `rally`, `sideways`, `volatile`, `calm`) + 1 unconditional pseudo-regime (`diff_teacher/CLAUDE.md:12`). Augmentation strategy:

- **Default (`--bella-augment-regime-mode uniform`)**: sample regime uniformly from `{0,1,2,3,4}` per Bella slot. This forces the encoder to generalise across regimes Bella thinks exist.
- **Alternative (`--bella-augment-regime-mode match`)**: estimate regime from real-period statistics (mirrors `_regime_index_from_returns` at `tools/precompute_bella_predictions.py:105-121`) and condition on the matching regime. Tests whether matching distribution helps.
- **Alternative (`--bella-augment-regime-mode unconditional`)**: use the unconditional pseudo-regime. Tests Bella's raw prior without regime conditioning.

---

## 5. CLI surface

New flags on `train_phase2.py` (mirrors style of TASK_749 §4-CLI table and TASK_740 `--bella-distill-*` flags):

| Flag | Type | Default | Effect |
|---|---|---|---|
| `--bella-augment` | store_true | `False` | Master toggle. Off → SegmentPool behaves identically to current. |
| `--bella-augment-ratio` | float | `0.25` | Fraction of slots (in `append`/`interleave`) or replacement fraction (in `replace`) drawn from Bella. Range `[0, 1]`. |
| `--bella-augment-mode` | str | `append` | One of `{replace, append, interleave}`. See §4.2. |
| `--bella-augment-regime-mode` | str | `uniform` | One of `{uniform, match, unconditional}`. See §4.5. |
| `--bella-augment-checkpoint` | str | `checkpoints/diff_teacher/generator_best` | Bella generator dir. Same default as `tools/precompute_bella_predictions.py:323-326`. |
| `--bella-augment-k-samples` | int | `1` | Samples per `generate_on_device` call. `K=1` is enough — each Bella slot is independent; we don't average. |
| `--bella-augment-seq-length` | int | `1440` | Generation horizon. Must be ≥ `window_length + lookahead + headroom`. Default matches Bella's native training horizon. |
| `--bella-augment-refresh-epochs` | int | `0` | If > 0, regenerate all Bella slots every N epochs (memory-mode reset). `0` = generate once at pool init, reuse for the whole run. |
| `--bella-augment-volatility-scale` | float | `0.015` | Scales Bella's z-scored channel-3 returns into log-return space. Matches `tools/precompute_bella_predictions.py:166`. |
| `--bella-augment-asset-policy` | str | `remap` | One of `{remap, canonical, random}`. `remap` = per-rotation column remap (TASK_745 mandatory). `canonical` = use Bella's universe directly, ignore rotation. `random` = random asset permutation per slot (most aggressive augmentation). |
| `--bella-augment-log-every` | int | `100` | Log slot-level Bella generation stats (regime mix, mean-vol, mean-return) every N slot-makes. |

All flags inert when `--bella-augment` is off. Defaults chosen to give a sensible "first real run" with `ratio=0.25, mode=append, regime=uniform, asset-policy=remap`.

---

## 6. Implementation steps

| # | File | LoC | Change |
|---|---|---|---|
| 1 | `mcts_trader/data/bella_market.py` | NEW, ~120 | New `BellaMarket` class mirroring `SyntheticMarket` interface (`mcts_trader/data/synthetic_market.py:5`). Holds a reference to a shared `generator`, exposes `cache_data: (T, A, 5)`, `asset_ids`, `assets`, `valid_timestamps`, `close_db()`. Constructor: `BellaMarket(generator, cfg, seq_length, n_assets, regime, seed, volatility_scale, asset_policy, period_assets, db_path)`. Calls `generator.generate_on_device(n_samples=1, seed=, regime=)`, applies volatility scale, applies column remap (lift the remap helper from `tools/precompute_bella_predictions.py:60-102, 222-247`). Returns a numpy `(T, A, 5)` tensor. **Mandatory**: per-rotation column remap if `asset_policy='remap'` — same logic as TASK_745 fix. |
| 2 | `train_easy.py:453-562` (`SegmentPool`) | edit, ~80 | Add `self._bella_generator`, `self._bella_cfg`, `self._bella_canonical` (lazy-init on first Bella slot construction). Add `self.bella_augment_ratio`, `self.bella_augment_mode`, `self.bella_augment_regime_mode`, etc. as pool fields. Add `_make_bella_slot(group_key, regime)` mirroring `_make_slot` but constructing `hm = BellaMarket(...)`. **Reuse all downstream precompute unchanged** — oracle path solver, state channels, policy targets, confidence are all defined on `hm.cache_data` shape `(T, A, 5)` which `BellaMarket` provides identically. |
| 3 | `train_easy.py` slot-selection path | edit, ~30 | Where the pool advances (currently `_make_slot` is called inside the group-runs loop at `train_easy.py:567-585`), insert a coin-flip: if `mode=append` and Bernoulli draw with prob `bella_augment_ratio` says "Bella", call `_make_bella_slot` instead (or in addition, depending on mode). For `mode=replace`, every `1/ratio`-th slot is Bella. For `mode=interleave`, do the swap at batch-composition time inside `AsyncBatchPipeline._prepare_numpy` (`train_phase2.py:508`). Start with `append`-only support and document the other two as follow-ups if needed. |
| 4 | `train_phase2.py:1768-1771` (CLI block, near other Bella flags) | edit, ~30 | Add the 10 new CLI flags from §5. |
| 5 | `train_phase2.py:1934-1938` (args→config translation) | edit, ~20 | Wire CLI overrides into `Phase2Config`. |
| 6 | `phase2_config.py:160-180` | edit, ~15 | Add `bella_augment: bool = False`, `bella_augment_ratio: float = 0.25`, `bella_augment_mode: str = 'append'`, etc. (one field per CLI flag from §5). |
| 7 | `train_phase2.py` SegmentPool construction site (~`train_phase2.py:2794`) | edit, ~20 | Pass new Bella-augment fields to the `SegmentPool` constructor via `oracle_config` extension or a new explicit kwarg. |
| 8 | `train_phase2.py:1559-1601` (E-line log + CSV writer) | edit, ~15 | Add `bella_slot_frac` (rolling fraction of batch slots from Bella) and `bella_regime_mix` (counts per regime) to the per-epoch log and CSV. |
| 9 | `mcts_trader/data/bella_market.py` — column-remap helper | (inside #1) | Lift `_load_bella_canonical_assets` and `_rotation_hash` from `tools/precompute_bella_predictions.py:60-102`. Expose `BellaMarket._build_remap(period_assets, bella_canonical) → (remap_array, n_unknown)`. Apply gather + NaN-fill exactly as `tools/precompute_bella_predictions.py:235-247, 286-288`. For Bella augmentation, NaN columns get filled with cyclic-pad from neighbouring valid columns (since the encoder cannot consume NaN; we are not masking targets here, we are providing inputs). Document this divergence from TASK_745 explicitly. |
| 10 | smoke + test scaffold | NEW `tests/test_bella_augment_smoke.py`, ~60 | Unit-level: construct a `BellaMarket` slot for one rotation, assert `cache_data.shape == (seq_length, A, 5)`, assert no NaN in `cache_data` after fill, assert remap'd column 0 corresponds to the rotation's asset[0]. |

**Total**: ~390 LoC across 4 files + 1 new module + 1 test.

**Mandatory per TASK_745**: step #9 — column remap is the lesson from 740. Without it the model trains on a column-shuffled view of Bella's output and we re-learn the same lesson the hard way.

---

## 7. Smoke test

**Exact command** (workstation, RTX 3090 = GPU 1 per `MEMORY.md`):

```bash
cd ~/Trader && source trader_env/bin/activate
CUDA_VISIBLE_DEVICES=1 python train_phase2.py \
  --stage B --epochs 2 --steps-per-epoch 30 --batch-size 256 \
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

**Per TASK_749 OOM lesson**: smoke runs at **production batch size (256)** and production window (60), NOT a reduced configuration. It must exercise the **full** `train_step_fn` end-to-end, including AsyncBatchPipeline pulling Bella slots, encoder forward, MCTS, backward, optimizer step.

**Pass criteria** (all must hold):

1. `BellaMarket` slot constructs successfully — first E-line log shows ≥ 1 Bella slot in the pool.
2. Bella generator load time ≤ 60s (one-time at pool init).
3. Per-epoch wall ≤ 1.3× the same command with `--bella-augment` removed (allowing 30% overhead for Bella generation + remap; if higher, generation is not amortizing).
4. `total_loss` finite for every batch — no NaN.
5. `bella_slot_frac` recorded in CSV at value `≈ ratio` (0.25 here) ± 0.1.
6. Peak VRAM ≤ 22 GB on the 3090 (Bella + training co-residency).
7. No `nan` in slot `cache_data` after BellaMarket construction (column remap NaN-fill must work).
8. Wall time ≤ 5 minutes total.

If any of 1, 2, 4, 6, 7 fail → debug fundamentals. If 3 fails → generation cost is too high; consider raising `--bella-augment-refresh-epochs`. If 5 fails → selector logic bug.

---

## 8. Eval protocol

Mirror the Tier-2 paired protocol from TASK_749 §7:

- **Screening**: paired 3-seed, seeds `{7, 13, 23}`, 200 epochs each.
- **Baseline**: `runs/741_mint/baseline_seed_{7,13,23}/` (canonical a1b1, same as 740 and 749).
- **Promotion gate**: mean Δα ≥ **+3pp** AND all 3 seeds positive AND no seed worse than −2pp. This is the same gate that 734_flow_matching passed (`project_flow_matching_promising.md`) and the gate the canonical multi-seed verdict pattern uses.
- **On promotion**: queue a 5-seed extension (add seeds 31, 42) before declaring canonical.
- **On regression**: write a `runs/753_bella_augment/result.md` mirroring the 740 result format (`runs/740_bella_distill/result.md`), note whether the regression is uniform across regimes (per-period Δα table), and consider one follow-up with `--bella-augment-regime-mode match` if `uniform` regressed.
- **SPEC location**: `experiments/753_bella_augment/SPEC.md`, generated from this task after smoke passes.

**Conditional baseline switch**: if 734 promotes flow-matching to canonical before 753 runs, swap the baseline to `runs/734_flow_matching/flow_seed_{7,13,23}/` and re-pair.

---

## 9. Risks

**R1 (MAJOR) — Bella generation wall cost** *Mitigation*: §4.4 budget. Generate once per epoch (or once per run with `--bella-augment-refresh-epochs 0`). Bella's `generate_on_device` is single-step in OT-CFM, ~50 ms per K=1 sample at 1440 steps — small relative to a Stage-B epoch. Monitor with the smoke-test wall-time ceiling (1.3× baseline). If breached, escalate to "refresh every 5 epochs" or "static pool of N Bella slots regenerated at startup only".

**R2 (MAJOR) — Distribution shift** Bella's `rally` regime may not look like real bull markets (cf. `project_bull_blindness.md` — all current models fail in bull regime regardless of mechanism). If Bella's regime distribution is meaningfully different from real markets, augmentation may train the encoder on features that don't transfer. *Mitigation*: (a) per-regime Δα breakdown in the eval result; (b) follow-up with `--bella-augment-regime-mode match` if `uniform` regresses; (c) accept the negative result as a verdict on Bella's regime fidelity, not on the augmentation mechanism per se.

**R3 (MAJOR) — Column-mismatch class of bug** TASK_745 documented exactly this failure mode for distillation. Augmentation has the SAME failure surface: Bella's column k = DB-ID k, trainer expects column k = rotation-specific asset k. *Mitigation*: step #9 in §6 is non-negotiable. Lift the `_load_bella_canonical_assets` + `_build_remap` helpers from the precompute script, unit-test the remap in step #10 smoke, log `n_unknown` per slot. Default `--bella-augment-asset-policy remap`.

**R4 (MODERATE) — Memory budget at full Stage-B batch** Bella generator weights (~600 MB fp16) + Stage-B working set (~18 GB) + MCTS rollouts on a 24-GB 3090 is tight. *Mitigation*: (a) load Bella in fp16 / bf16, not fp32; (b) keep Bella's `_load_bella` (`tools/precompute_bella_predictions.py:132-137`) inference-mode only (no gradients, no optimizer state); (c) Bella forward is done outside any `jit` boundary co-traced with the training step — keep it in its own JIT scope so XLA doesn't blow up memory by fusing; (d) document a fallback `--bella-augment-batch-size 128` flag (not in §5 because we hope to avoid needing it). On server (matching 3090) the same budget applies; on the 10 GB CMP card Bella augmentation is not supported and the flag must error out.

**R5 (MODERATE) — Volatility scale fragility** The `0.015` z-score → log-return scale (`tools/precompute_bella_predictions.py:166`) is hard-coded and matches `mcts_trader/training/setup.py::generate_bella_prices_on_device`. If Bella's actual output stats drift from this assumption, the synthetic OHLCV will have wrong absolute-magnitude returns, and oracle path solver will see unrealistic PnL. *Mitigation*: log per-Bella-slot mean abs log-return and compare to real-slot stats in the same log line; document a sanity range (e.g., real-slot abs return mean is ~3e-4 per step — Bella slots should fall in `[1e-4, 1e-3]`).

**R6 (MODERATE) — Oracle path solver on synthetic data may be degenerate** Bella's flow-matched OHLCV is smoother than real markets (`diff_teacher/CLAUDE.md:14` — single-step OT-CFM tends to produce smooth modes). A smooth synthetic series may give the oracle near-perfect lookahead PnL, which makes the BC target trivially "always buy/sell the right asset". This trains the model on **easy** problems and may regress real-market performance. *Mitigation*: log oracle cum-PnL distribution per Bella slot vs real slot; if Bella slot oracle is >2× real-slot oracle PnL, the augmentation is too easy — add `--bella-augment-noise-floor` (post-hoc additive noise) as escape hatch, or filter to Bella slots in the harder regimes (`volatile`, `crash`).

**R7 (MINOR) — Anti-correlation pattern risk** `project_classifier_alpha_anti_correlation.md` says classifier-side improvements anti-correlate with Alpha on this stack. Augmentation is NOT classifier-side — it's data-side, gradient pathway unchanged — so it should not fit the pattern. But if augmentation happens to make the encoder a better classifier (improving Bal.Acc by exposing it to more regime variation), we may still trip the trade-rate-explosion failure mode. *Mitigation*: log Bal.Acc and trade-rate alongside Δα in the eval result; if Bal.Acc UP and Alpha DOWN with same trade-rate explosion signature, file as another datapoint for the anti-correlation pattern and consider whether the augmentation should be weighted/scheduled differently.

---

## 10. Out of scope

- **Retraining Bella**. Bella is treated as a frozen prior, exactly as in TASK_740. Bella retrain is Tier-D strategic work.
- **Changing the Stage-B loss objective**. No new heads, no new auxiliary losses. The loss stack is identical to 741_mint.
- **MCTS over Bella rollouts** (Path γ from TASK_740 / TASK_722). Different mechanism class; not in scope here.
- **Encoder surgery**. The encoder is byte-identical to baseline.
- **Bella+distill stacking**. We just learned distill regressed; don't stack with augmentation in the first run. Conditional Tier-D if 753 passes the gate.
- **Iris-DEX feeds (Tier-D richer features)**. Separate workstream.
- **`replace` and `interleave` modes**. Implement `append` only in v1; document the other two as follow-up tasks if `append` passes the gate.
- **Multi-period Bella generation** (one Bella tensor spanning multiple rotations). Each Bella slot is a self-contained 1440-step tensor with a single rotation's asset set; no cross-period coupling.

---

## 11. Success criteria

- [ ] `mcts_trader/data/bella_market.py::BellaMarket` constructs a valid slot with `(seq_length, 64, 5)` cache_data, no NaN, remap applied per rotation.
- [ ] `SegmentPool` admits Bella slots when `--bella-augment` is set; pool stats log the slot mix per epoch.
- [ ] Smoke command (§7) passes all 8 pass criteria.
- [ ] Per-epoch wall ≤ 1.3× the no-augment baseline.
- [ ] Bella generator load is one-shot; not re-loaded per slot.
- [ ] Column remap (TASK_745 Option A) lifted from `tools/precompute_bella_predictions.py` and unit-tested.
- [ ] CLI flags added under `train_phase2.py` documented in `train_phase2.py --help`.
- [ ] CSV writer logs `bella_slot_frac`, `bella_regime_mix`.
- [ ] `experiments/753_bella_augment/SPEC.md` drafted post-smoke.
- [ ] Paired 3-seed run launched against `runs/741_mint/baseline_seed_{7,13,23}` baseline.
- [ ] `runs/753_bella_augment/result.md` written with mean Δα, per-seed Δα, per-period Δα, verdict.

---

## 12. References

- TASK_749 spec format precedent — `~/Trader/tasks/TASK_749_TITANS_PIVOT.md` (esp. §5 smoke, §7 eval, §8 risks)
- TASK_740 — `~/Trader/tasks/TASK_740_BELLA_DISTILL_HEAD.md` (the alternative wiring we are NOT pursuing)
- TASK_740 result — `~/Trader/runs/740_bella_distill/result.md` (the regression we are responding to)
- TASK_745 — `~/Trader/tasks/TASK_745_BELLA_SEMANTICS.md` (column-remap fix, mandatory)
- Bella component — `~/Trader/diff_teacher/CLAUDE.md`
- Bella precompute path — `~/Trader/tools/precompute_bella_predictions.py:60-247`
- Bella loader — `~/Trader/mcts_trader/training/setup.py::setup_bella_generator`
- SegmentPool — `~/Trader/train_easy.py:453-720`
- AsyncBatchPipeline — `~/Trader/train_phase2.py:484-693`
- Legacy SyntheticMarket precedent — `~/Trader/mcts_trader/data/synthetic_market.py:5-50`, imported at `~/Trader/train_easy.py:60`
- Anti-correlation pattern — `memory/project_classifier_alpha_anti_correlation.md`
- Bull blindness — `memory/project_bull_blindness.md`
- Flow-matching promising — `memory/project_flow_matching_promising.md` (the gate template)
- Efficiency philosophy — `~/Trader/CLAUDE.md` "Memory Hierarchy" section

---

## Operator notes

- This task **does not** replace TASK_740. 740 documented one wiring (distill); 753 documents the alternative (augment). If 753 passes the gate and 740 didn't, the verdict is "Bella as data > Bella as teacher on this stack" — a clean answer to a binary question.
- Per TASK_742 rule: ≤ 2 concurrent `train_phase2.py` per host. Bella augmentation adds VRAM pressure; consider ≤ 1 concurrent run when `--bella-augment` is on.
- Server GPU-0-forbidden rule still applies (CMP 90HX = 10 GB, insufficient for Bella co-residency).
- Smoke runs land under `runs/000_smoke/bella_augment_smoke/`; SPEC runs under `runs/753_bella_augment/`.
- SPEC sketch: `~/Trader/experiments/753_bella_augment/SPEC.md` (create after smoke passes).

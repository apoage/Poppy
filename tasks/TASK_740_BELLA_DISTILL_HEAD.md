# TASK_740: Bella Distillation Prediction Head (Path β)

**Status**: IMPLEMENTED (2026-04-23). Ready to launch — 1) `python tools/precompute_bella_predictions.py --dry-run` (verify), 2) full pre-compute run, 3) `bash tools/run_740_bella_distill.sh`.
**Priority**: PRIORITY 3 post-736 per `lore/notes_2026-04-22.md §49 line 872`.
**Rationale**: `lore/notes_2026-04-22.md §47 line 626` (Path β definition) + §48 lines 680–681 (β-v1 vs β-v2 trade-off).
**Blocks**: nothing directly. Most-likely ceiling-break if 60-step OHLCV is the information constraint.

---

## The Problem This Addresses

After TASK_736 fixed the ExIt loop, `dir_acc` still caps at 0.574. The working hypothesis (post-TASK_735 result, notes §48 line 688) is that the 0.57 ceiling is not a training-signal artifact but an *information ceiling* of 60-step minute-bar OHLCV. If that's correct, no head-architecture change or oracle reformulation on the same inputs will break it.

Path β's thesis: Bella — our market generator — was trained on the full market distribution at Foxglove-scale and captures priors that a 60-step OHLCV window on the agent side alone cannot. Injecting Bella's forward forecasts as an auxiliary signal gives the agent *new information*, not just a rearrangement of existing signal.

---

## The Two Sub-Options (Pick Based On Feasibility Audit)

### β-v1: Bella predictions as encoder channels

- Bella rolls out K ∈ {1, 4} futures for the current window.
- Per-asset "predicted sign of forward return" (or predicted log-return magnitude) is appended as 33rd–38th channels to the encoder input.
- Forward-looking feature injection — no distillation loss; Bella is a frozen feature extractor.

**Fatal flaw (from notes §48 line 680):** Bella inference is ~100 ms per prediction × batch 256 × 200 steps × 200 epochs = prohibitive. Not viable as-stated.

**Variant β-v1-cached:** pre-compute Bella predictions for all training windows offline (once per data revision), store as shard files alongside the market data. Training loop reads pre-computed tensors. Storage: ~few GB per period. Engineering: offline pipeline + loader glue. Practical.

### β-v2: Distillation into a new aux head

- New `AuxBellaHead(latent) → predicted_log_prices[t+1, …, t+K]` or `→ predicted_sign[t+20]`, shape matching Bella's output.
- `L_distill = MSE(agent_aux_head_output, bella_output)` — pre-computed Bella outputs, same offline shard as β-v1-cached.
- At inference, Bella is not needed. The agent has internalized the prior.

Recommended: **β-v2 (distillation) over β-v1 (channel injection).** Reasons:
1. Cleaner architectural integration (mirrors existing aux heads: direction, rank, price).
2. Decouples inference from Bella availability — deployability.
3. Gradient flow is identical to other aux heads; no encoder input-shape changes.
4. Same offline pipeline as β-v1-cached; no additional cost.

---

## Files To Create / Modify

### Offline pipeline (one-time, per data revision)

| File | Change |
|---|---|
| `tools/precompute_bella_predictions.py` | New. Loads Bella from `jax_ckpt_easy_A/branch_A_bella/phase1_best.pkl` (or B variant), iterates through Foxglove periods, emits `bella_preds_{period}.npz` with `pred[t, asset, horizon]` arrays. |
| `data/bella_preds/` | Storage. Size estimate: ~500 MB per period × 17 slots ≈ 8 GB. Verify disk before committing. |
| `config/bella_preds_manifest.json` | New. Maps each period to its pre-computed shard path. |

### Training wiring

| File | Change |
|---|---|
| `mcts_trader/jax_agent/muzero_model.py` | New `AuxBellaHead` class mirroring `AuxPriceHead` pattern from TASK_735 (shared MLP over per-asset latent → (B, n_assets, K) predictions). Zero-init final layer. |
| `phase2_config.py` | `use_bella_distill_head: bool = False`, `bella_distill_weight: float = 0.1`, `bella_pred_horizons: tuple = (1, 5, 10, 20)`. |
| `train_phase2.py` | Load per-batch Bella predictions from shard alongside the market data pool. Add `bella_distill_loss` to total loss pipeline. CLI `--bella-distill-weight`, `--bella-preds-dir`. |
| `easy_config.py` | No changes (distillation happens in Stage B only). |

### Estimated LoC

- Offline pipeline: ~150 LoC (mostly data wrangling + JAX batch inference).
- AuxBellaHead + config: ~30 LoC.
- Training integration: ~50 LoC.
- **Total: ~230 LoC + 5-7 day infra commitment + 2-3 hr training run.**

---

## Success Criteria

Compare against TASK_736 baseline and (if completed) TASK_739 result.

### Phase 1 sanity check (distillation only; no Stage B yet)

Before committing to full Stage B, run a short Phase 1 experiment:
- Load warmstart checkpoint.
- Add only the bella_distill_loss (weight 0.1).
- Train 20 epochs.
- Gate: `bella_distill_loss` drops monotonically. If it's stuck high, AuxBellaHead can't learn from the latent — architectural problem, fix before Stage B.

### Stage B gates

- **E20**: `bella_distill_loss` < 60% of initial value AND `dir_acc > 0.60` → on track. Continue to E100.
- **E50**: `dir_acc > 0.62` → candidate ceiling-break. Benchmark.
- **E50**: `dir_acc ≤ 0.58` AND distill loss still dropping → latent can't use Bella's prior effectively. Consider β-v1-cached (direct channel injection) as fallback.

### Benchmark

- Bal.Acc > 45% AND Alpha > +6% → **architectural inflection**. Bella prior materially helps. Ship-candidate.
- Bal.Acc 35–45%, Alpha +4–6% → improvement. Stack with TASK_739 (dense oracle) for final push.
- No benchmark movement → 60-step OHLCV ceiling is not what Bella is priced to break. Consider Tier D (richer real-time features from Iris-DEX) or Path γ (full MCTS-over-Bella-rollouts, TASK_722).

---

## Why This Order (after TASK_737 and TASK_739)

1. **Post-737**: We need to know which aux heads are load-bearing before adding a fourth.
2. **Post-739**: Dense sign oracle is cheaper (~100 LoC vs 230+). If A1 alone breaks the ceiling, β becomes a nice-to-have instead of a required infra commitment.
3. **Infra cost is real**: ~8 GB of pre-computed predictions + a new offline pipeline. Worth committing only if the ceiling is demonstrably information-bound (which TASK_739's outcome partially clarifies).

---

## Open Questions / Pre-Launch Decisions

1. **Which Bella checkpoint?** Branch A (`jax_ckpt_easy_A/branch_A_bella`, E2500) or B (`jax_ckpt_easy_B/branch_B_bella`, E4359). Default: B (more training) unless A has reported qualitative advantage.
2. **Prediction horizon matching**: Bella's native horizon vs agent's 20-step oracle horizon. May need resampling. Document the match explicitly in the offline pipeline.
3. **Normalization**: Bella outputs are in log-price space. Agent's aux_price_head (TASK_735) is also in log-price space. Align normalization before distillation, or the MSE is meaningless.
4. **Stochastic Bella**: if Bella is inherently stochastic (samples from a distribution), should we pre-compute K samples per window and distill against the mean, median, or per-sample? First-pass recommendation: mean across K=4 samples.

---

## Dependencies

- TASK_736 baseline (value-weight 0.5).
- TASK_737 suite complete (know which heads are load-bearing).
- TASK_739 complete, ideally (know if oracle density alone breaks ceiling).
- Bella checkpoint (`jax_ckpt_easy_A/branch_A_bella/phase1_best.pkl` or B variant, ~595 MB).
- 8 GB free disk for pre-computed predictions.
- 32-ch Phase 1 checkpoint for warmstart (or 27-ch if 737 legacy_27ch wins).

---

## Out of Scope (explicit non-goals)

- **Path γ (TASK_722)**: Full MCTS-over-Bella-rollouts. 2-3 week commitment. Only revisit if β moves the ceiling but stalls short of ship-gate.
- **Live Bella inference in training loop**: killed by latency. Use offline shards only.
- **Bella retrain**: Bella's current checkpoints are treated as frozen priors. Retraining Bella is a separate initiative.
- **Bella as encoder input (β-v1 uncached)**: ruled out by latency. β-v1-cached subsumed by β-v2 in practice.

---

## Implementation notes (2026-04-23)

- `AuxBellaHead`: added in `mcts_trader/jax_agent/muzero_model.py` immediately after `AuxPriceHead`; shared MLP over global latent → `(B, n_horizons, n_assets)`; zero-init final Dense.
- Wiring: `MuZeroNetworks.setup/initial_inference/init_path` — guarded by `use_bella_distill_head`; default False keeps prior checkpoints byte-identical.
- Config: `phase2_config.Phase2Config` — `use_bella_distill_head=False`, `bella_distill_loss_weight=0.1`, `bella_pred_horizons=(1,5,10,20)`, `bella_preds_manifest='config/bella_preds_manifest.json'`. Default OFF.
- **Prediction space: log-returns**, not log-prices (scale-stable across assets, matches `aux_price_head` so one MSE is meaningful).
- Loss (`train_phase2.py`): per-horizon MSE on valid (non-NaN) cells only; per-sample mask so windows lacking a Bella shard contribute zero gradient. 1/h horizon-weighting matches aux_price.
- CLI: `--bella-distill-head / --no-bella-distill-head`, `--bella-distill-weight`, `--bella-preds-manifest`, `--reset-bella-distill-head`. Mirrors TASK_735's `--no-aux-price-head` style.
- Batch plumbing (`train_phase2.py::AsyncBatchPipeline`): new `bella_preds` field threaded through `_prepare_numpy`/`_encode_jax`/`stage_b_train_step`.
- Segment-pool integration (`train_easy.py::SegmentPool._make_slot`): new `bella_preds_full` slot field; new `_load_bella_shard` method reads per-period NPZ via the manifest. Missing shards → NaN fill → trainer masks out.
- Offline pipeline: `tools/precompute_bella_predictions.py`. **Chosen Bella: `checkpoints/diff_teacher/generator_best` (conv-flow generator, the actual market generator).** The TASK spec's listed paths `jax_ckpt_easy_*/branch_*_bella/phase1_best.pkl` are MuZero-agent checkpoints trained *on* Bella data, not the generator itself — confirmed by inspecting the `.pkl` structure (has `trm/params/action_embedding/...`, not Bella weights). Defaults: K=4 samples, horizons (1,5,10,20), ws=60. Regime conditioning estimated per-window from mean/std of the observed window log-returns (Bella has no native prefix-conditioning mode).
- Storage: `data/bella_preds/bella_preds_<start>__<end>.npz` per period (compressed float32 preds + int32 horizons + metadata). Manifest at `config/bella_preds_manifest.json`.
- Launch: `tools/run_740_bella_distill.sh` — mirrors TASK_736 config + `--bella-distill-head --bella-distill-weight 0.1`; preconditions check Phase 1 ckpt, Bella ckpt, manifest, and all listed shards, plus reth stopped.
- Deviations from spec: (a) uses `checkpoints/diff_teacher/generator_best` (real generator) instead of the `branch_*_bella` path from the spec (those are agents, not generators); (b) approximates per-window conditioning via regime estimate, not prefix inpainting, because Bella's architecture has no prefix mode — single clean abstraction, no ad-hoc inpainting code.

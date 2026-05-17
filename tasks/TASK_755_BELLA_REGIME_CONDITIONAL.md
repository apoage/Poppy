# TASK_755 — Bella Distillation, Regime-Conditional

**Status**: DRAFTED 2026-05-17.
**Priority**: **LOW** — execute only after TASK_753 and TASK_754 have produced verdicts. Rationale: (a) the per-regime signal this is chasing is weak (`bear +0.91pp` on a single 3-seed pair, well within the TASK_743 ~5pp noise floor); (b) regime-gating mechanisms are inherently messier and introduce a confound (gate vs. mechanism) that 753/754 do not. Treat this as a salvage path for the Bella-distill mechanism, not a first-line bet.
**Predecessor**: `tasks/TASK_740_BELLA_DISTILL_HEAD.md` (mechanism), `runs/740_bella_distill/result.md` (the result this responds to), `tasks/TASK_745_BELLA_SEMANTICS.md` (column-mismatch fix that must land before any new Bella run).

---

## 1. Goal

Recover the bella-distill mechanism by gating its auxiliary loss on the detected market regime, so the gradient only flows in regimes where Bella adds value and is silenced where it injects noise. Concretely: pick **hard gating** for v1 (cheap, no new learnable parameters), keep soft regime-conditional weighting as a v2 follow-up if v1 promotes.

---

## 2. Hypothesis

Bella's prediction quality is **regime-dependent**. The 740 paired 3-seed verdict (`runs/740_bella_distill/result.md` lines 24–28) shows:

- Mean Δα **−1.92pp**, 0/3 seeds positive.
- Per-period: **5 of 6 regimes negative** (the headline regression), but **`bear` +0.91pp positive**. Worst hit: `sideways` −5.10pp.

This is consistent with two complementary mechanisms:
1. Bella was trained predominantly on regime-balanced data including bear/crash priors that the agent's 60-step encoder cannot easily recover from minute-bar OHLCV alone. In regimes where those priors are accurate, the distillation gradient helps.
2. In trending bull / quiet sideways regimes, Bella's flow-matching velocity field is closer to a random direction relative to the agent's task — distillation injects gradient noise that resembles the classifier-sharpening anti-pattern documented in `memory/project_classifier_alpha_anti_correlation.md`.

Averaging across regimes destroys the bear-positive signal; gating preserves it.

**Caveat**: the +0.91pp on `bear` is itself within the TASK_743 single-period noise envelope. This task is a hypothesis test, not a confident bet. See §9 (Risks).

---

## 3. Background — current state

- Bella distill head wired and load-bearing:
  - Config: `phase2_config.py:112-116` (`use_bella_distill_head`, `bella_distill_loss_weight=0.1`, `bella_pred_horizons=(1,5,10,20)`, `bella_preds_manifest="config/bella_preds_manifest.json"`).
  - Loss site: `train_phase2.py:1176-1194` (per-horizon MSE, NaN-masked).
  - Batch plumbing: `train_phase2.py:559-689` (collects `bella_preds`), `train_phase2.py:750-751` (host→device).
  - SegmentPool shard load: `train_easy.py:691-718` and `_load_bella_shard` at `train_easy.py:752-786`.
  - CLI: `train_phase2.py:1919-1935` (`--bella-distill-head`, `--bella-distill-weight`, `--bella-preds-manifest`).
- Bella-side regime taxonomy (5 regimes) is already defined: `diff_teacher/flow_matching.py:53-70` (`RegimeEmbedding`, `num_regimes=5`: 0=crash, 1=rally, 2=sideways, 3=volatile, 4=calm). The precompute pipeline already classifies windows: `tools/precompute_bella_predictions.py:105-121` (`_regime_index_from_returns`, rule-based on mean/std).
- Benchmark regime taxonomy (6 regimes — `bull`, `bear`, `sideways`, `ftx_crash`, `etf_approval`, `boring_week`) lives in `config/benchmark_periods.json:5-460`. **These are eval-time labels and do not exist at training time** — training rotations are drawn from `foxglove_periods.json` and are not pre-tagged with a regime.
- TASK_745 column-mismatch fix is **landed** (per `tasks/TASK_745_BELLA_SEMANTICS.md` and `runs/740_bella_distill/result.md` line 30). Any new Bella-distill run must re-precompute shards with the post-745 code path before measuring Δα.

---

## 4. Design (v1 — hard gating)

### 4.1 Regime detector (rule-based, online, CPU)

Reuse and extend the existing rule-based classifier at `tools/precompute_bella_predictions.py:105-121`. The classifier takes a `(ws, n_assets)` log-return window and returns an int regime label. We will compute the **same label per training rotation** using the rotation's first-window log-returns, on CPU, once per rotation (not per batch — cheap).

**Default detector**: rolling 20-period log-returns over the rotation's first window:
- `cum_ret = sum(log_returns)` (cross-asset mean)
- `vol = std(log_returns)`
- Returns one of `{crash, rally, sideways, volatile, calm}` using the same thresholds as `_regime_index_from_returns`.

**Why this and not the benchmark regime labels**: training rotations are not the benchmark periods. The benchmark taxonomy (`bull`, `bear`, `sideways`, `ftx_crash`, `etf_approval`, `boring_week`) is only defined for the 6 eval periods. We need a label at training time that we can compute on any rotation, and the Bella-side 5-class taxonomy already exists and is what Bella's `RegimeEmbedding` was conditioned on during precompute. Using the same taxonomy keeps train and eval consistent.

**Mapping benchmark→detector** (for interpreting per-period results): bull→rally; bear→crash; sideways→sideways; ftx_crash→crash; etf_approval→rally; boring_week→calm. The `bear +0.91pp` evidence from 740 corresponds to detector label **crash**.

### 4.2 Gate

Per-rotation scalar `regime_gate ∈ {0.0, 1.0}` applied as a multiplier on the existing `bella_distill_loss` at `train_phase2.py:1191`. When 0, no Bella gradient flows for that rotation; when 1, the loss behaves exactly as in 740.

The set of "ON" regimes is CLI-controlled. The recommended v1 default is `--bella-distill-regime-gate bear`, which sets gate=1 only when the detector returns `crash` (0). Other supported presets:
- `none` — gate=1 always (equivalent to 740 baseline; sanity).
- `bear` — gate=1 iff regime==crash. **v1 default.**
- `bear+crash` — gate=1 iff regime ∈ {crash}. (Equivalent to `bear` under the 5-class taxonomy; reserved name for documentation symmetry with the 6-class benchmark.)
- `bear+volatile` — gate=1 iff regime ∈ {crash, volatile}.
- `custom` — read regime list from `--bella-distill-regime-set "0,3"`.

### 4.3 Where the gate lives in code

Plumb `regime_label: int` as a new field on the segment-pool slot dict (next to `bella_preds_full` at `train_easy.py:718`). Compute it inside `SegmentPool._make_slot` on CPU once per slot. The trainer reads it alongside `bella_preds` (`train_phase2.py:649-688`) and passes a `(B,)` int32 array `regime_labels` through `_prepare_numpy` / `_encode_jax` into the train step. Inside `stage_b_train_step` (`train_phase2.py:1176-1194`), compute `gate = jnp.isin(regime_labels, gate_set_array).astype(jnp.float32)`, broadcast against the per-sample MSE, and multiply before the sum reduction.

This keeps the gate per-sample (different rotations sharing a batch get different gates), which is mandatory when SegmentPool slots span multiple regimes within a batch.

### 4.4 Defer to v2 (NOT in scope here)

A learnable per-rotation soft weight `w ∈ [0,1]` produced by a small 1–2 layer MLP on rolling stats, applied as a multiplicative weight on the bella loss. This is more powerful but introduces a confound (which is doing the work — Bella or the gate?). Build it only if v1 shows a non-trivial improvement.

---

## 5. CLI surface

Add to `train_phase2.py` near the existing Bella flags (`train_phase2.py:1919-1935`):

| Flag | Default | Effect |
|---|---|---|
| `--bella-distill-regime-gate {none,bear,bear+crash,bear+volatile,custom}` | `none` | Preset gate. `none` matches 740 behaviour exactly. **v1 SPEC default: `bear`**. |
| `--bella-distill-regime-set "<csv ints>"` | `""` | Used only when `--bella-distill-regime-gate custom`. Comma-separated regime indices (0=crash, 1=rally, 2=sideways, 3=volatile, 4=calm). |
| `--bella-distill-regime-detector {ruleset,manifest}` | `ruleset` | `ruleset` = on-the-fly CPU classifier (default). `manifest` = read precomputed labels from the shard (the precompute already records the dominant regime per window in the npz; if absent, fall back to `ruleset`). |
| `--bella-distill-regime-log` | `False` (store_true) | Log per-rotation `regime_label` and `gate_value` to the CSV writer. |

Config fields (mirror in `phase2_config.py:107-116`):

```python
bella_distill_regime_gate: str = "none"
bella_distill_regime_set: tuple = ()       # parsed from CLI csv
bella_distill_regime_detector: str = "ruleset"
```

---

## 6. Implementation steps

Numbered, ~LOC, with file paths. Bar: **agent without context can implement this**.

1. **(MANDATORY) Verify TASK_745 column remap is committed and `bella_preds_manifest` was regenerated post-fix.** Spot-check one shard with the alignment test from `tasks/TASK_745_BELLA_SEMANTICS.md` §"Verification". If not yet verified, STOP — running TASK_755 on a noisy precompute repeats the 740 mistake. ~0 LoC, 30 min.

2. **Refactor regime classifier into a shared module.** Move `_regime_index_from_returns` from `tools/precompute_bella_predictions.py:105-121` into `mcts_trader/utils/regime.py` (NEW). Export `classify_window(returns_window: np.ndarray) -> int` and the constant `REGIME_NAMES = ("crash","rally","sideways","volatile","calm")`. Update `precompute_bella_predictions.py` to import from the new module (no behaviour change). ~40 LoC.

3. **Compute regime label per slot at load time.** In `train_easy.py::SegmentPool._make_slot` (around line 691, next to `bella_preds_full`):
   - Compute log-returns on the first window of the rotation's price tensor (CPU numpy, ~1 ms).
   - Call `classify_window(...)` → int.
   - Add to slot dict as `'regime_label': int(label)` next to `'bella_preds_full'` at line 718.
   - **NO future-data leakage**: use only the first window's returns. Document this with a comment citing §9-R1.
   ~30 LoC.

4. **Plumb `regime_label` through the batch path.** In `train_phase2.py::AsyncBatchPipeline`:
   - Collect per-sample labels alongside `bella_preds` (mirror lines 559–689).
   - Emit `np_batch['regime_labels']` as `(batch_size,) int32`.
   - Convert to jnp in `_encode_jax` at lines 750–751.
   ~25 LoC.

5. **Apply the gate inside the loss.** In `stage_b_train_step` (`train_phase2.py:1176-1194`):
   - Parse `bella_distill_regime_gate` and `bella_distill_regime_set` from config into a static frozenset of allowed regime ints (`static_argnames` on JIT — required per `CLAUDE.md` JAX pitfalls).
   - Build `gate_mask = jnp.isin(regime_labels, gate_array).astype(jnp.float32)` with shape `(B,)`.
   - Multiply `per_sample_bella_loss * gate_mask[:, None, None]` BEFORE the sum reduction; otherwise the horizon-weighted mean is computed over gated-out samples and the loss is silently shrunk by a constant factor.
   - Guard against all-zero gate (entire batch gated out): `bella_distill_loss = jnp.where(gate_mask.sum() > 0, weighted_sum, 0.0)`.
   ~20 LoC.

6. **CLI parsing & config translation.** Add the four flags in §5 to the argparse block around `train_phase2.py:1919-1935`. Translate to config around `train_phase2.py:2128-2135` (next to existing `bella_distill_*` translations). Parse `--bella-distill-regime-set` csv → tuple of ints. ~30 LoC.

7. **CSV / log diagnostics.** When `--bella-distill-regime-log` is set, add two new columns to the existing CSV writer (find `bella_distill_loss` in the log builder — around `train_phase2.py:1571-1601` per pattern from TASK_749): per-batch `regime_label_mode` (most common label in batch) and `gate_fraction` (mean of gate_mask). ~15 LoC.

8. **Optional: manifest-mode detector.** If `--bella-distill-regime-detector manifest`, read regime labels from the npz shard instead of recomputing. Requires `tools/precompute_bella_predictions.py` to save the per-window regime as a new field in the npz (the script already computes it at line 105–121 — just emit it alongside `pred`). ~20 LoC, optional. If skipped, ship `ruleset` only.

9. **Smoke test.** See §7.

10. **SPEC + queue.** Write `experiments/755_bella_regime_conditional/SPEC.md` mirroring `experiments/749_titans_mag/SPEC.md`. Add to `experiments/QUEUE.md`. NOT in this task — write the SPEC only after smoke passes.

**Total estimate**: ~180 LoC across 4 files (`mcts_trader/utils/regime.py` new, `train_easy.py`, `train_phase2.py`, `phase2_config.py`, plus optional `tools/precompute_bella_predictions.py` touch). ~1 day of focused work + smoke.

---

## 7. Smoke test (<60 s if pre-compiled; 5–8 min compile)

Mirror TASK_749's smoke pattern (full production train_step at production batch — the 749 OOM lesson is that a tiny-batch smoke does not catch the real-shape failures).

```bash
cd ~/Trader && source trader_env/bin/activate
CUDA_VISIBLE_DEVICES=1 python train_phase2.py \
  --stage B --epochs 2 --steps-per-epoch 25 --batch-size 256 \
  --periods-file config/foxglove_periods.json \
  --n-slots 4 --max-runs-per-group 1 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.5 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 --aux-price-weight 0.1 \
  --bella-distill-head --bella-distill-weight 0.1 \
  --bella-preds-manifest config/bella_preds_manifest.json \
  --bella-distill-regime-gate bear --bella-distill-regime-log \
  --output-dir runs/000_smoke/755_smoke \
  --name 000_755_smoke 2>&1 | tee runs/000_smoke/755_smoke/train.log
```

**Pass criteria** (must all hold):
1. Compile completes (≤ 8 min).
2. No NaN in `total_loss` for 2 epochs.
3. `gate_fraction` recorded in CSV; over 50 steps, mean ∈ (0.0, 1.0) (i.e., at least one rotation matched and at least one didn't — sanity for the gate logic).
4. `bella_distill_loss` non-zero on at least one step (gate matched somewhere) and zero on at least one step (gate cleared somewhere).
5. Peak VRAM unchanged vs 740 baseline (gate adds ~0 VRAM; if it grows by more than 100 MB something is wrong).
6. With `--bella-distill-regime-gate none`, the loss trace is **byte-identical** to a 740 re-run at the same seed for the first 25 steps. **Required regression test** — run as a second smoke pass.

---

## 8. Eval protocol

Standard paired-seed pattern, but the **headline metric is per-period Δα, not mean Δα**. The mechanism's purpose is to preserve the bear-positive signal without poisoning sideways/bull.

- **Screening**: paired 3-seed (s7, s13, s23) against the **741_mint a1b1 baseline** at the same seeds (the canonical Stage-B baseline; `memory/MEMORY.md` TASK_743 entry). 200 epochs, single-host server, default batch.
- **Promotion gate to 5-seed extension**:
  - `bear` Δα ≥ +1.0pp (preserves or improves the 740 +0.91pp signal), AND
  - `sideways` Δα ≥ −1.5pp (no longer the disaster it was at −5.10pp in 740), AND
  - mean Δα ≥ −0.5pp (the gate has at least pulled the regression toward neutral).
- **Demotion**: if any of bull/etf_approval/sideways Δα < −3pp under the gate, retire the mechanism — the gate is "leaking" Bella's noise even with the conservative `bear`-only preset.
- **Promote-to-canonical (5-seed, s7/13/23/31/42)** only if mean ≥ +1pp AND `bear` ≥ +1pp AND no period < −2pp.

Per-period eval is the standard `tools/benchmark_easy.py` against the 6 regimes in `config/benchmark_periods.json`.

---

## 9. Risks

**R1 (MAJOR) — Regime-detection lookahead bias.** A naive classifier that looks at the *entire* rotation's returns is using the future. Mitigation: classify on the **first window only** (60-step prefix) — the same window the agent sees at decision time. Document explicitly in `_make_slot` and add a unit test that classifying the rotation gives the same label as classifying with only the first 60 timesteps' data.

**R2 (MAJOR) — The `bear +0.91pp` signal may itself be noise.** TASK_743 established that single-seed Alpha gaps ≤ ~5pp are within seed noise on this stack (`memory/MEMORY.md`). +0.91pp on a single bear period in a single 3-seed pair is at most ~0.7σ above zero. If the gating mechanism produces only `bear +1pp / others 0pp` under multi-seed, that may still be indistinguishable from "wash and move on." Mitigation: the §8 promotion gate requires `bear` ≥ +1.0pp **AND** mean improvement, not just bear preservation alone.

**R3 (MAJOR) — Mechanism confound.** Once we gate, we cannot answer "did Bella help?" without also answering "did the gate help?". A flat result is consistent with (a) Bella useless even in bear, gate is doing nothing helpful, or (b) Bella helps in bear, gate works, but bear is too rare in training rotations to move the average. Mitigation: log `gate_fraction` over training; if `gate_fraction < 0.05` we have a base-rate problem that pre-empts the experiment. Consider rebalancing SegmentPool sampling toward crash-labelled rotations as a v2 follow-up (NOT in this task — that's a separate confound).

**R4 (MODERATE) — Classifier-alpha anti-correlation pattern.** Per `memory/project_classifier_alpha_anti_correlation.md`, mechanisms that look like per-asset prediction sharpening tend to collapse Alpha via trade-rate explosion. Bella distill fits the pattern weakly (740 regression). The gate REDUCES the amount of distillation gradient flowing, which moves us toward the safer baseline — so the gate should at worst recover toward baseline, not produce a new failure mode. But: if the gate-on signal is itself net-negative even in bear (R2), we have no escape hatch and the mechanism class is dead.

**R5 (MODERATE) — Train-eval regime taxonomy mismatch.** Training rotations are classified with the Bella 5-class detector (crash/rally/sideways/volatile/calm); eval uses the 6-class benchmark labels (bull/bear/sideways/ftx_crash/etf_approval/boring_week). The §4.1 mapping is defensible but not unique — `etf_approval` could plausibly be `volatile` rather than `rally`. Mitigation: report results with both mappings in the SPEC's results table.

**R6 (MINOR) — Manifest-mode label staleness.** If `--bella-distill-regime-detector manifest`, the labels are baked into the precompute; changing the rule-set requires a re-precompute (~hours of GPU). Mitigation: default to `ruleset` (online) for v1.

---

## 10. Out of scope (explicit non-goals)

- **Do NOT retrain a learned regime classifier in this task.** The detector is rule-based. A learned classifier is v2 (`bella_distill_regime_detector="learned"` reserved but `NotImplementedError`).
- **Do NOT change the benchmark regime definitions** in `config/benchmark_periods.json`. The benchmark taxonomy is a published artifact; modifying it would silently invalidate every prior result.
- **Do NOT re-precompute Bella shards as part of this task.** TASK_744/745 owns the precompute hygiene. This task consumes `config/bella_preds_manifest.json` as-is.
- **Do NOT modify SegmentPool sampling weights.** Regime-balanced rotation sampling is a separate concern (mentioned in R3 mitigation as v2).
- **Do NOT implement soft regime-conditional weighting (v2).** Spec is in §4.4; implementation waits on v1's verdict.
- **Do NOT touch Bella itself.** No retraining, no architectural changes to `diff_teacher/`.
- **Do NOT bundle with TASK_749 (Titans) or TASK_753/754.** Each mechanism gets its own clean paired verdict.

---

## 11. Success criteria

- **v1 SPEC ready**: smoke passes the §7 criteria, especially the byte-identical check for `--bella-distill-regime-gate none`.
- **v1 SPEC promoted**: 3-seed paired Δα satisfies the §8 screening gate.
- **v1 SPEC retired**: 3-seed paired result fails the §8 demotion check OR fails the screening gate without a clear path to a different preset (e.g., `bear+volatile`) recovering the result.
- **Mechanism class retired**: if neither v1 (this task) nor a documented v2 (soft weighting) produces a paired Δα ≥ +1pp with `bear ≥ +1pp`, retire Bella-distill entirely and revise `memory/project_classifier_alpha_anti_correlation.md` to add Bella-distill as a 4th confirmation of the pattern.

---

## 12. References

- Predecessor mechanism: `tasks/TASK_740_BELLA_DISTILL_HEAD.md`
- Predecessor result (the per-period numbers we cite): `runs/740_bella_distill/result.md` (especially lines 24–28 and the caveat at line 30).
- Column-alignment fix that must be verified first: `tasks/TASK_745_BELLA_SEMANTICS.md`.
- Benchmark regime definitions: `config/benchmark_periods.json:5-460`.
- Existing rule-based classifier (to be lifted into `mcts_trader/utils/regime.py`): `tools/precompute_bella_predictions.py:105-121`.
- Bella's own regime taxonomy: `diff_teacher/flow_matching.py:53-70`.
- Bella-distill loss site: `train_phase2.py:1176-1194`.
- SegmentPool shard load (insertion point for `regime_label`): `train_easy.py:691-718`, `_load_bella_shard` at `train_easy.py:752-786`.
- Bella-distill CLI block: `train_phase2.py:1919-1935`.
- Style reference: `tasks/TASK_749_TITANS_PIVOT.md` (structure, depth, CLI table format).
- Noise floor / multi-seed rule: `memory/MEMORY.md` TASK_743 entry.
- Anti-pattern context: `memory/project_classifier_alpha_anti_correlation.md`.
- Bull-blindness context: `memory/project_bull_blindness.md`.

# TASK_806 — Look-ahead-free predictability probe (the "is there entropy" gate)

**Status**: DRAFT — operator-approved direction (2026-06-20). Foundation diagnostic that should have
been experiment #1. Read-only w.r.t. training code / checkpoints / other runs; writes only to
`runs/806_predictability_probe/` and one probe script.

**The question** (one sentence):

> From causally-valid past-only features, can *anything* predict forward returns out-of-sample on
> real data — and does the frozen canon encoder carry *incremental* predictability over a trivial
> linear baseline, anywhere, after multiple-comparison correction?

This is the base-rate-of-predictability test the campaign never ran. It settles the "defeating
statistics" question: is the 0-promotions record a **data** problem (no exploitable entropy → stop,
publish the negative) or a **method** problem (signal exists, the supervised-α objective destroys it
→ fixable).

## Why this, now

`lore/notes_2026-05-31.md:33-41` is the load-bearing reframe: the campaign trains a supervised
classifier that mimics the oracle, and measures on trading α. The oracle is a greedy-lookahead path
solver — it uses future H-step returns by construction, so it is a **teacher, not a strategy**
(operator confirmation, 2026-06-20: "oracle is intentional dataleak, valid only for training").
The oracle is only a *useful* teacher to the degree forward returns are predictable from past-only
features. **That degree has never been measured.** If it is ~0, distilling the oracle teaches the
model to guess randomly *with confidence* → over-trading + Δ-Bal.Acc/Δα anti-correlation
(`[[project_classifier_alpha_anti_correlation]]`) → exactly the 14-experiment graveyard.

TASK_804 already ran one instance of this test (perception surface): `real_fe` failed to beat its
own cyclic-shift control (shift ≥ real, 3/3 seeds, `notes_2026-06-19.md`). This task generalises that
into the direct, representation-agnostic, look-ahead-free predictability gate — including the encoder
itself, which 804 did not isolate.

## Scope guardrails

- **NO oracle.** `path_solver` / `greedy_lookahead_path` / any lookahead function must not be
  imported or called by the probe. (Audited — see LEAK_AUDIT.)
- **NO Bella / synthetic data.** Real `ohlcv` from `trading_data.db` only.
- **Frozen encoder.** The canon encoder is applied as a fixed feature extractor; the probe head is
  ridge/linear only. No gradient flows into encoder weights.
- **No training runs.** This is a probe + bootstrap, not an imprint. Workstation-runnable, minutes
  not hours, GPU optional (only for the encoder forward pass).
- Writes only: `runs/806_predictability_probe/` + the probe script. Do not touch `train_easy.py`,
  `train_phase2.py`, any checkpoint, any other `runs/`.

## Method

### Data & universe
- Real `ohlcv` close prices from `trading_data.db` (246M rows, 1202 assets).
- **Step 0 (must report first):** bar frequency (infer from timestamp gaps) and per-asset coverage.
  State `H` in **bars** throughout.
- **Fixed 64-asset universe** chosen for long contiguous real history (e.g. top-64 by coverage/age
  among the union of `config/benchmark_periods.json` `benchmark_assets`). Universe is fixed and
  reported; do not re-select per cell.

### Temporal split (Option A — the gold standard)
- One contiguous real history per asset, split **forward in time**:
  train `[t0→t1]` · val `[t1+g→t2]` · test `[t2+g→t3]`.
- **Purge gap `g ≥ H_max`** between every pair of splits (de Prado label-overlap purge; the forward
  target of a train sample must not land inside val/test). Document the dates.
- The 6 `benchmark_periods.json` windows (bull/bear/sideways/ftx_crash/etf_approval/boring_week) are
  reported as **additional OOS test cells** wherever they fall inside the test region (they are the
  campaign's actual eval surface — keeps the result comparable). Do not train on them.
- If DB coverage is too thin for a clean 3-way temporal split, STOP and flag reduced power rather
  than silently degrading to cross-period mixing (which conflates predictability with regime
  transfer).

### Features (causally valid — decision time `t`, window `[t−W+1..t]`, `W=64`)
- `linear_uni` — own-asset lagged log-returns over the window → ridge.
- `linear_xs` — all 64 assets' lagged log-returns flattened → ridge. **Strongest reasonable linear
  baseline — the fair fight.** If the encoder can't beat this, the encoder adds nothing.
- `encoder` — frozen canon encoder (`jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`,
  encoder_version 3, embed_dim 64, W=64) applied to the window → embed-dim features → ridge head.
  Reuse `jit_batch_encode_temporal` + `EncoderConfig` + `load_checkpoint` from `tools/benchmark_easy.py`.
- `unconditional` — predict train mean (regression) / majority sign (classification). The floor.
- `persistence` — predict last return repeats.

### Target
- `r(t,H) = log p(t+H) − log p(t)`, per (t, asset). Sweep `H ∈ {1, 5, 20}` bars (20 = campaign
  `oracle_lookahead`). Also sign-of-`r` for the classification view.

### Leakage guards (HARD — each must be enforceable + cited in LEAK_AUDIT)
1. Feature window uses only `timestamp ≤ t`. No future price in any feature.
2. Forward target uses only `p(t+H)` — it is the *label*, not a feature.
3. Train/val/test forward-ordered, contiguous, purge gap `≥ H` between splits. No shuffle.
4. Feature standardisation (mean/std) fit on **train only**, applied to val/test.
5. Encoder weights frozen — probe head is the only thing fit; same fit routine for all arms.
6. Identical split + scaling + target for `linear_*` and `encoder` (apples-to-apples).
7. Ridge α selected on **val only**, frozen for test. Report the chosen α per arm.

### Metric & significance
- **IC** (Spearman rank corr of predicted vs realised `r`) per cell — the canonical quant metric,
  scale-robust. Also OOS `R²` vs unconditional mean, and sign-accuracy vs prior.
- **Block bootstrap** (circular block, block length `≥ max(5, 2H)`) on the test set → 95% CI on IC
  and R². Resamples must respect label-overlap autocorrelation.
- **FDR**: Benjamini–Hochberg at `q=0.05` across the full cell grid (≈ universe-subsets × regimes ×
  horizons × arms). Report raw p-values alongside adjusted.
- Report **base rate** per cell (mean `|r|`, sign split) so regime imbalance is visible.

## Arms / grid

| Arm | Features | Head | Question it answers |
|---|---|---|---|
| `unconditional` | none | mean/sign | the floor (R²=0, IC=0) |
| `persistence` | last return | identity | weak EMH martingale check |
| `linear_uni` | own-asset lags | ridge | is there linear autocorrelation? |
| `linear_xs` | cross-sectional lags | ridge | is there linear cross-asset signal? (fair fight) |
| `encoder` | frozen canon encoder | ridge | does the encoder see anything linear can't? |

Grid: arms × `H∈{1,5,20}` × {test-region slice + any benchmark windows inside it}. Pre-register the
grid before running; do not add cells post-hoc to chase significance.

## Decision gate (three-way, MECHANICAL — verdict must follow from the table)

A cell "has signal" iff its bootstrap IC CI excludes 0 **and** survives BH-FDR. "Material" IC floor =
**0.02** (conventional weak-IC threshold in quant; below is noise — flagged as a judgment call, raw
distribution reported regardless).

1. **NULL — no entropy.** Zero cells survive FDR with material IC, *and* encoder CI ⊇ linear CI
   (encoder adds nothing). ⟹ **Stop trading-model work.** The data has no exploitable structure
   from these features. This is a **publishable negative** ("weak-form predictability tested with
   full encoder vs linear baseline; null across horizons/regimes after FDR"). File under
   `[[project_single_mechanism_exhausted]]` as the root-cause closure.
2. **Signal-but-encoder-blind.** `linear_*` (or persistence) has ≥1 surviving cell with material IC,
   but encoder does not beat linear there. ⟹ **Signal exists; the encoder destroys it.** Method bug,
   fixable. Pivot target: representation / encoder, not the loss.
3. **Signal-and-encoder-sees-it.** Encoder beats linear (CI separation) on ≥1 surviving cell. ⟹
   Signal exists and the encoder perceives it; the supervised-α gap is the real culprit
   (`notes_2026-05-31.md:33-41`). Objective/architecture is wrong, not the data.

**Scope honesty:** outcomes 2/3 prove *predictability*, not post-cost tradable α. The question on the
table is information-theoretic; do not over-read a "signal exists" verdict as "ship."

## Risk (how this could lie)

- **Leakage** (the big one) — a single future leak manufactures fake IC. Mitigation: the LEAK_AUDIT
  artifact + the auditor verdict; guard list above is exhaustive-by-construction.
- **Weak power** — short test windows → wide CIs → false NULL. Mitigation: use the full contiguous
  test region (not just the 6 one-week windows); report n and CI width per cell; flag any cell with
  n too small to resolve IC=0.02.
- **Multiple comparisons** — 36+ cells inflate false positives. Mitigation: BH-FDR + raw p-values
  reported + pre-registered grid.
- **Encoder trained on the leaky oracle** — its features may be tuned to future info it lacks at
  inference, so "encoder blind" could be a symptom, not a clean read. This is *expected* and
  informative (it is the campaign's failure mode made visible); do not "fix" it — test the encoder
  as-is because that is what deploys. Note in result interpretation.
- **Costs/slippage** out of scope by design (predictability ≠ post-cost α) — stated above.

## Deliverables (what the auditor verdicts against)

1. `runs/806_predictability_probe/result.md` — the cell table (per cell: n, base rate, IC + 95% CI,
   R² + CI, sign-acc, raw p, BH-adj p, survive-yes/no) + the **three-way verdict** stated against
   the mechanical rule + a method section (universe, split dates, gaps, scaling source, bootstrap
   params, BH q) + the bar-frequency/coverage report from Step 0.
2. `runs/806_predictability_probe/probe.py` — single reproducible entry point (`python probe.py`
   end-to-end). No oracle import.
3. `runs/806_predictability_probe/LEAK_AUDIT.md` — checklist mapping each leakage guard (1–7) to the
   `probe.py` line that enforces it.

## Acceptance checklist (audited separately — be self-evident)

- [ ] `probe.py` imports/calls **no** lookahead/oracle symbol (grep-clean).
- [ ] Temporal split forward-ordered, contiguous, purge gaps ≥ H documented with dates.
- [ ] Standardisation fit on train only; ridge α chosen on val only.
- [ ] Encoder frozen (ridge/linear head only; no encoder gradient).
- [ ] Same split/scaling/target across `linear_*` and `encoder`.
- [ ] Block bootstrap with stated block length; BH-FDR across the grid; raw p-values reported.
- [ ] Per-cell base rate + IC + R² + CIs + adj-p in result.md (not just a headline number).
- [ ] Three-way verdict follows the mechanical rule; rule quoted; no talking up/down.
- [ ] One command reproduces the whole run.

## References

- `lore/notes_2026-05-31.md:33-41` — the "trading objective never in the loss / loop off" reframe.
- `lore/notes_2026-06-19.md` — TASK_804 NULL (perception can't beat cyclic-shift control).
- `[[project_single_mechanism_exhausted]]`, `[[project_classifier_alpha_anti_correlation]]`,
  `[[project_bull_blindness]]` — the failure-pattern memories this test root-causes.
- `tools/benchmark_easy.py` — `_load_period_prices`, `jit_batch_encode_temporal`, `load_checkpoint`,
  `EncoderConfig` (reuse; do not duplicate the real-data path).
- `config/benchmark_periods.json` — 6 real OOS windows (the campaign eval surface).
- `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl` — frozen canon encoder.

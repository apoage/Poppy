# HANDOFF PROMPT — TASK_806 predictability probe (paste to the executing agent)

> Paste everything below the line to the model that will run it. It is self-contained.

---

You are running **TASK_806 — the look-ahead-free predictability probe** for an ML trading-model
project (Poppy/Bella/Foxglove campaign). The full spec is the file of record:
**`tasks/TASK_806_PREDICTABILITY_PROBE.md` — read it first, in full, before writing any code.**

## Your single mission

Answer one question, rigorously and honestly: **from causally-valid past-only features, can anything
predict forward returns out-of-sample on real data, and does the frozen canon encoder carry
incremental predictability over a strong linear baseline — anywhere, after multiple-comparison
correction?** This is the base-rate-of-predictability test the campaign never ran. It decides whether
the project's 0-promotions record is a *data* problem (no exploitable entropy → stop) or a *method*
problem (signal exists, the objective destroys it → fixable).

## Non-negotiables (these are what you are being verdicted on)

1. **No oracle, no lookahead, anywhere.** Do not import or call `path_solver`,
   `greedy_lookahead_path`, `compute_optimal_path`, or anything that touches future prices. The probe
   must be grep-clean of lookahead. A single future leak manufactures fake IC and voids the result.
2. **No Bella / synthetic data.** Real `ohlcv` from `trading_data.db` only.
3. **Frozen encoder.** The canon encoder is a fixed feature extractor; the only thing you fit is a
   ridge/linear probe head. No gradient into encoder weights.
4. **Temporal split, forward in time.** train → val → test contiguous, with a purge gap `≥ H` between
   splits (forward target of a train sample must not land in val/test). Document the dates. Never
   shuffle across time.
5. **No leakage in scaling/selection.** Standardisation (mean/std) fit on train only; ridge `α`
   chosen on val only. Same split, scaling, and target for `linear_*` and `encoder` — apples to
   apples.
6. **Significance done right.** Block bootstrap (circular block, length `≥ max(5, 2H)`) for 95% CI
   on IC and R². Benjamini–Hochberg FDR at q=0.05 across the full cell grid; report raw p-values too.
7. **Fair linear baseline.** `linear_xs` (cross-sectional, all 64 assets' lagged returns → ridge) is
   the bar the encoder must beat. Do not rig the comparison with a weak linear baseline.
8. **Pre-register the grid** (arms × H∈{1,5,20} × test slices) before running. Do not add cells
   post-hoc to chase significance.
9. **Verdict is mechanical, not narrative.** Apply the three-way rule from the spec exactly. Do not
   talk a NULL up to "promising" or a weak signal down. Quote the rule, show the cells.

## What to produce (exactly these artifacts)

- `runs/806_predictability_probe/probe.py` — one reproducible entry point (`python probe.py` runs the
  whole thing). Reuse the real-data + encoder path from `tools/benchmark_easy.py`
  (`_load_period_prices`, `jit_batch_encode_temporal`, `load_checkpoint`, `EncoderConfig`) — do not
  reinvent it. Frozen encoder: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`.
- `runs/806_predictability_probe/LEAK_AUDIT.md` — a checklist mapping each leakage guard to the
  `probe.py` line that enforces it. This is the first thing the auditor reads.
- `runs/806_predictability_probe/result.md` — the cell table (per cell: n, base rate, IC + 95% CI,
  R² + CI, sign-acc, raw p, BH-adj p, survives-yes/no) + the three-way verdict stated against the
  mechanical rule + a method section (universe, split dates, gaps, scaling source, bootstrap params,
  BH q) + the Step-0 bar-frequency/coverage report.

## Do this first, before anything else

**Step 0:** open `trading_data.db`, infer the **bar frequency** (from timestamp gaps) and report
**per-asset coverage**. If coverage is too thin for a clean forward train/val/test split on a fixed
64-asset universe, **STOP and flag it** — do not silently degrade to cross-period mixing (that
conflates predictability with regime transfer and invalidates the test). State `H` in **bars**.

## What you must NOT touch

`train_easy.py`, `train_phase2.py`, any checkpoint, any other `runs/*/`, the encoder weights, the
campaign's eval harness. You write only to `runs/806_predictability_probe/` and `probe.py`.

## How you will be judged

A separate auditor (not you) will verdict the run against the acceptance checklist in
`tasks/TASK_806_PREDICTABILITY_PROBE.md`. They will not trust your summary — they will read
`probe.py`, re-derive the leakage guards from `LEAK_AUDIT.md`, and check the verdict follows from the
table. Make every step self-evident from the artifacts. If a step is judgment (the IC materiality
floor, the universe choice), say so and report the raw distribution so the judgment is checkable.

Begin by reading the spec, then Step 0, then report the bar frequency + coverage + your planned split
dates before fitting anything.

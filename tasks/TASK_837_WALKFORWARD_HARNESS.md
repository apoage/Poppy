> **STATUS (2026-07-09): SHIPPED as `task837-walkforward`.**
> `experiments/837_walkforward/` — fold rails (pre-registered 2021-23→24 / 2021-24→25-09), PIT
> universes derived module-side (**`build_pit_universe` verified NON-PIT** — unbounded coverage counts;
> deviation loudly SPEC'd, counts on `ohlcv_daily` ≤ train-end; fold1 64-of-219, fold2 64-of-263),
> per-fold corridors via 830 machinery (fold1 net@21 [+0.057, +36.774] ref −0.226; fold2 [−0.291,
> +35.410] ref −0.000), manifests consumable by run_queue.sh + 835. Medium panel: temporal-leak core
> CLEAN; **3 MAJOR gate-coverage roots crossval-CONFIRMED and FIXED**: validate now POLICY-PINS every
> parsed window/universe-rule param to pre-registered constants (an internally-consistent tamper or an
> honest `--min-days` typo is now rejected); eval-job gate enforces file IDENTITY + assets (a leaky-
> universe swap can no longer become the graded number); negative battery de-confounded (no-op control
> must validate CLEAN; 5 negatives each rejected by their NAMED gate). **Remainders (scheduled):** fold
> trainings + the R4 verdict (phase-2+ decision); server pre-flight = `validate --db trading_data.db`
> (read-only) BEFORE any run_queue.sh launch — the minute-ingest assert only gains teeth there.

# TASK_837 — Walk-forward split harness (the R4 graduation instrument)

**Status**: READY 2026-07-08. **Size: M. Risk: medium** (produces the promotion-deciding
numbers; zero canon diffs — orchestration module over existing pieces).
**Pathway**: R4. The definition-of-working says **unseen years** — today NOTHING in the repo
can produce an unseen-years number: training and 835-eval both run the same pinned 2021–2025
window. This harness makes the graduation discipline concrete BEFORE any in-sample result
tempts a promotion claim.

## Pre-registered split policy (planner, 2026-07-08 — the harness implements, never invents)
- **Fold 1**: train 2021-01-01..2023-12-31 → eval 2024-01-01..2024-12-31.
- **Fold 2**: train 2021-01-01..2024-12-31 → eval 2025-01-01..2025-09-01.
- Eval grading: 835 scorecard on the eval window ONLY, vs that window's OWN corridor floors
  (830 pattern, computed by the harness) and the engine-rule ref on the same window.
- **Universe leak flag (load-bearing)**: the pinned 64-asset universe was selected with
  knowledge through 2025 — survivorship leak for fold evals. For each fold the harness must
  re-derive the universe POINT-IN-TIME as of train-end (`trader/data.py::build_pit_universe`
  exists; document any deviation loudly). The pinned-universe run may be reported alongside,
  clearly labeled LEAKY-UNIVERSE, for continuity with 821-lineage numbers.

## Scope — `experiments/837_walkforward/` + SPEC + fixture smoke
1. **Fold materialization**: generate per-fold periods-files (the 821 periods-file mechanism)
   + PIT universes; emit a fold manifest consumable by `run_queue.sh` (training jobs) and by
   835 (eval jobs). No training is run by this task — it BUILDS the rails.
2. **Fold corridor**: per-eval-window floor/ceiling/ref via the 830 corridor machinery.
3. **Aggregation**: per-fold scorecards → one walk-forward summary JSON + LEDGER-convention
   row (per-fold corridor positions, never a single pooled Sharpe without the per-fold view).
4. **Fixture smoke**: fold materialization + corridor + aggregation end-to-end on the fixture
   (tiny stub scorecards); PASS = correct window boundaries (assert no train/eval overlap,
   assert eval data never enters a train periods-file — the leak-free gates as CODE).

## Gates
Fixture smoke green incl. the two leak asserts; zero diffs outside `experiments/837_walkforward/`;
selftest 7/7.

## Out of scope
Running fold trainings (scheduled, phase-2+ decision), changing the definition-of-working bar,
touching 835 internals (import it).

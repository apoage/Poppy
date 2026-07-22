# Factory run 15 — D20 intra-series-gap audit (handoff)

**Date:** 2026-07-22
**Thread:** dark-factory implementer
**Disposition:** diagnostic-only run — NO code change, NO ship, NO commit-of-code. Advanced the
one OPEN wall (DECISIONS D20) with the zero-cost read-only audit its own recommendation named as
the cheap first step. Result is decision-changing.

## Why this run existed

Ready queue was EMPTY (planner: "Next row seeds after D20 rules"). Only wall = **D20**: the
TASK_852 equivalence-gate FLOOR leg still fails after the densification fix — daily net@21
+0.7053 vs minute +0.7608, |Δ|=0.0555 (2.75× the 0.02 tolerance) on the real
2021-01-01..2025-09-01 overlap. D20's recommendation (a) = "cheap next step first — audit
`ohlcv_daily` for per-day gaps *within* each of the 27 covered symbols (SQL only, no training
spend); if closing/normalizing those collapses the 0.0555 the way the coverage fix collapsed the
0.21, ship `--panel-source daily`."

Running that measurement does not commit to any branch of D20 (doesn't ship daily, re-base
tolerance, or pin to minute) — it produces the evidence to rule it. Within-lane, zero GPU, zero
training, read-only against `trading_data.db`.

## Method

`scratchpad/d20_gap_audit.py` (read-only). Reproduced the gate's exact universe:
- minute-ranked top-64 via **per-asset PK-range `COUNT`** (PK=(asset_id,timestamp)) — identical
  ranking to the gate's single `GROUP BY asset_id` but index-fast, not a full 4.5-yr range scan.
  (The naive single-`GROUP BY` form is what made the first attempt time out; the per-asset PK
  loop returns in seconds.)
- daily-covered subset via `_daily_coverage_ids` semantics (`close IS NOT NULL AND close != 0`).

Then, per covered symbol: distinct daily rows in-window, calendar span (first→last day), and
**interior** missing days (gaps strictly between the symbol's own first and last row — leading/
trailing absence is coverage, already handled by the active-mask; interior gaps are what get
stale-forward-filled and could bias the teacher floor).

## Result — candidate (ii) REFUTED

Covered floor_universe (27): XBT ETH EUR DOT SOL LTC XRP ADA XDG GBP LINK XMR XLM TRX ATOM ALGO
MATIC SHIB AVAX KSM BCH UNI AAVE DAI FIL LUNA XTZ.

Daily calendar-union distinct days (covered set): **1704**.

**Interior gap-days across the ENTIRE covered set: 4.**
- XRP/USD: 2 interior gaps (max run 2), 1702 rows / 1704 span, 0.12%
- BCH/USD: 2 interior gaps (max run 2), 1702 rows / 1704 span, 0.12%
- All 25 other covered symbols: **0 interior gaps** (perfectly dense over their own span).

Short-history symbols (not gaps — genuine coverage windows): EUR/GBP/DAI = 413 days from
2024-07-15; MATIC ends 2025-03-31 (1415d); SOL 2021-06-17; SHIB 2021-11-30; AVAX 2021-12-21;
LUNA 2021-12-16.

(NB: D20's context cited BCH/XRP as 1703/1705 rows; this audit measures 1702 under the
close-validity filter. The 1-row delta is the `close != 0`/NULL screen; the structural
conclusion — 2-ish interior gaps on exactly two symbols — is identical.)

## What this means for D20

- **Candidate (ii) [intra-series interior gaps, stale forward-fill] is refuted by direct
  measurement.** 4 forward-filled days across 27 series of ~1704 days cannot move a net@21
  Sharpe by 0.0555. **D20 branch (a) (cheap SQL fix → ship daily) is therefore closed** — there
  is essentially nothing to close.
- **Candidate (i) [axis discretization] is the surviving explanation, by elimination and by
  corroboration:** daily calendar-union axis = 1704 (measured here) vs minute positional
  `[::1440]` stride = 1701 (D20's prior run), plus the heterogeneous per-symbol spans above are
  handled asymmetrically at the panel edges by the two loaders. This is an irreducible
  cross-loader discretization difference, not a fixable data-integrity bug.
- Directly isolating (i) to the exact 0.0555 requires building BOTH full panels off the 25 GB
  minute table (the `[::1440]` stride read) — real-data validation → SCHEDULED, not in-loop.

## D20 RULED concurrently — this audit CONFIRMS it

While this read ran, a concurrent PLANNER session independently reached the identical
conclusion and RULED D20 (moved to DECIDED, 2026-07-22): *"majors gapless (BCH/XRP 2 missing
days each, rest 0) ⇒ residual = DISCRETIZATION (candidate i), branch (a)-ship does NOT fire.
Ruling = (b)-variant: daily source admitted for WITHIN-PANEL grading only (both legs same
panel, Δ cancels); cross-source floor comparisons stay forbidden; minute canon untouched;
eqgate FLOOR FAIL stands as the permanent marker (no silent re-base). Binding OOS re-run
launched on this scoping. LEDGER 852/reconcile."*

This factory audit is therefore an **independent cross-validation** of that ruling — two
sessions, two code paths, same numbers (4 interior gap-days total, XRP 2 / BCH 2 / rest 0).
The cheap-fix branch (a) is confirmed dead; the residual is structural discretization, used
within-panel only. The planner seeded **TASK_853 (inactive-mask OOS panel)** as the next Ready
row on this scoping — the factory picks it up from here.

## State

- Tree clean, factory disarmed. No code touched. `experiments/835_stage4_eval/eval_vector.py`
  unchanged.
- Audit script preserved at `scratchpad/d20_gap_audit.py` (session scratch — not committed;
  re-runnable, ~seconds).
- Repo note: `.git/objects` lives on `/dev/sdb1` (`storage disk`), which was UNMOUNTED at session
  start — git was broken until `udisksctl mount -b /dev/sdb1`. Same as the standing memory note.

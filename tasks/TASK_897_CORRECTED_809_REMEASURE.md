> **STATUS (2026-08-22): INSTRUMENT SHIPPED as `task897-corrected-809-estimator` (c8a406c). THE
> VERDICT IS NOT IN — the real leg is planner-scheduled on gpu-server per D54.**
> Gates (i), (ii), (iv) PASS. Gate (iii) is **PARTIAL and disclosed** (see below). Two BLOCKERs were
> found by review and closed: the leading pre-trade interval was being counted as a *decision* (one
> real bet gave `n_decisions=2`, `sharpe=+22.93`, `CI=[22.93,22.93]` — a trivially-true "CI lo > 0"
> that defeated `sharpe()`'s own `size<2` guard), and `recompute_best.py` printed **"PASS"** for a
> cell with `pnl=-1.0` because its gate omitted the band's `PnL > 0` criterion.
>
> **TWO THINGS THE PLANNER MUST RULE BEFORE THE SRV RUN IS WORTH LAUNCHING — both seeded as
> decision rows:**
> **(1) The CI is measured-UNCALIBRATED, so the pre-registered band cannot be read as written.**
> Coverage through the real code path is **0.530 at n_dec=100** under overlapping dependence
> (nominal 0.95), 0.865–0.868 at n=500–1000, and block length is capped at 20 so longer dependence is
> uncapturable **at any n**. The `CELL CLEARS` band requires `CI lo > 0`; that criterion is not
> readable off this instrument as delivered. Fields are named `ci_lo_UNCALIBRATED` /
> `ci_hi_UNCALIBRATED` so nothing can consume them silently.
> **(2) Gate (iii) contradicts this row's own Out-of-scope.** Gate (iii) requires the 877 alias fold
> applied; Out-of-scope forbids modifying the 809 lineage beyond the estimator fix. The fold is NOT
> applied — `probe.py`/`backtest.py`/`decide.py` contain zero alias references and predate TASK_877,
> and `load_universe_trainlocked` matches symbols by exact string. The artifact previously claimed
> the opposite; that claim is retracted and the gap is stated instead, with a test that re-reads the
> lineage source so the wording cannot drift back into being false.
>
> **Launch notes for the srv leg.** Stage `runs/809_decision_rule_rescue/` + `experiments/897_corrected_809/`
> + the minute-table extract; run `sweep_gated_maker.py --attainable-maker-fee-bp <operator's tiers>`.
> **Do NOT pass `--skip-inherited-ic-check`** — it exists only for non-canonical-scale verification.
> `recompute_best.py`'s repair lives under gitignored `runs/`, so staging it is a manual step that the
> clean-tree gate cannot see.
>
> **Remainders:** the maker-fill model (OHLC-touch + flat adverse selection) is still never validated
> against a real order — unchanged from 809, explicitly out of scope, and still the load-bearing
> assumption. `recompute_best.py` prints its banner at module import time.

# TASK_897 — the corrected 809 re-measurement: was the only gate-passing strategy ever falsified?

**Status**: READY 2026-08-19 (planner; seeded by the pathfinder audit
`docs/pathfinder_audit_2026-08-19.md` §B G1, operator-adopted same day).
**Size: S. Risk: LOW to run, HIGH in consequence** — its verdict either re-opens the minute
surface (execution-engineering path) or closes it honestly, on an estimator that cannot lie the
way 809's did.
**Eval-only, no training. Runs on gpu-server per D54 (2026-08-20):** workstation-only inputs —
the 809 lineage code under gitignored `runs/809_decision_rule_rescue/`, the minute-table extract —
are STAGED to srv as part of the launch (byte-verified copies, read-only use). If a specific
input cannot be staged, the launch note says why and runs data-local as the documented exception.

## Thesis

The 806→811 campaign measured a REAL minute-scale cross-sectional signal (OOS IC 0.06–0.15,
BH-FDR-surviving, n≈217k strided — `runs/806_predictability_probe/result.md`), proved it
cost-bound at taker (808: 0/72 cells), then 809 found conviction-gating + maker execution
passing the post-cost gate: **+5.08, CI[+2.77,+7.63], turnover 0.087, on untouched OOS
2024-11→2025-04**. That PASS was retracted in lore (2026-06-25) as a per-bar annualization
artifact (per-bar returns ×√525600 over overlapping holds — the "809 trap") — **but the
corrected per-decision number was never computed** (`recompute_best.py` exists and never ran).
The cell was therefore never falsified either: the idea died of venue economics and a broken
estimator, not of evidence. This row computes the honest number.

## Scope

1. **Verify the estimator first.** `runs/809_decision_rule_rescue/recompute_best.py` (or its
   successor) must implement per-decision, non-overlapping-hold accounting, annualized by
   realized decisions-per-year — hand-check it against a small worked example before any sweep.
   No per-bar annualization anywhere in the pipeline.
2. **Re-run 809's best realistic cells** (the `gated_maker` family: H ∈ {1,5,20}, k ∈ {16,32},
   long_short + long_only, τ ∈ {0.75,0.9}, N ∈ {0,60}) on the honest estimator, over TWO windows:
   (a) the original test window OOS 2024-11→2025-04, and (b) **the untouched backfilled span
   2025-04→2026-03** — a full year of OOS the minute table now carries that no 809-lineage
   measurement has ever seen.
3. **Record EVERY cell.** The full fee × adverse-selection grid (fee ∈ {0,8,16}bp, adv ∈
   {0,5,10}bp) with per-cell results in the artifact — the 809 run's realistic-sweep per-cell
   data was lost (only the OPT corner survived); that hole closes here. Each cell carries
   turnover, fill rate, and the maker-fill model's assumptions stated inline (OHLC-touch + flat
   adverse selection — flagged since 2026-06 as "the load-bearing assumption and the known weak
   point", never validated against a real order).

## Pre-registered bands (verdict → LEDGER `897/809-corrected` → decision row)

- **CELL CLEARS** — any realistic cell clears the 809 gate (post-cost Sharpe ≥ 1.0, PnL > 0, CI
  lo > 0) at a maker-fee tier the operator can plausibly reach on Kraken (zero-maker-fee is out
  of reach per the Fork-C venue ruling; the attainable set is the operator's, recorded in the
  artifact before the verdict is read). ⇒ the path to profit on this surface is **execution
  engineering**: fill-model validation via small live limit orders, then paper — and the vector
  line's bar becomes that cell, not +0.97.
- **ALL FAIL** — no cell clears at any attainable tier ⇒ the minute surface is **closed for
  real**, on an honest estimator, with the venue/cost question forced to the operator with
  numbers.
- **MIXED** — clears only on the original window or only on the new span ⇒ decay finding; the
  decision is whether the signal lived and died, with the span split reported.

## Acceptance (gates)

- (i) The estimator passes the hand-computed worked example; a mutation re-introducing per-bar
  annualization is caught by a test.
- (ii) Every cell of the grid appears in the artifact (no silent drops — G8(a)'s lesson).
- (iii) Both windows run from the pinned minute table with the 877 alias fold applied; the
  artifact names the data epoch.
- (iv) The attainable-tier set is recorded BEFORE the verdict is read.

## Out of scope

Live orders, paper trading, venue changes, any modification of the 809 lineage code beyond what
the estimator fix requires. Validating the maker-fill model against real orders (that is the
CELL CLEARS follow-up, not this row).

## Launch notes (planner, 2026-08-22)

**D55 (ruled (a)) — calibration gate before the srv leg.** The factory measured the
pre-registered CI at coverage 0.530 (nominal 0.95) with the block cap at 20 — the D49 structural
defect. Before any cell is read: run the 892 calibration METHOD (not its table — different
statistic, different block regime) against this path with the block cap raised; record the
certified method, measured coverage, and chosen block length in the artifact. The CELL CLEARS
band stands unchanged; the interval must earn the right to feed it.

**D56 (ruled (a)) — blast radius MEASURED, and it is nonzero.** Query against
`trading_data.db` (read-only, 2026-08-22): of the 291-symbol 809 candidate universe, 19 slots
have twin spelling-variant ids; **6 are MATERIAL — the 868 identity-split class** (`/USDC/USD`,
`/USDC/USDT`, `/USDCGBP`, `/USDTAUD`, `/USDTCHF`, `/USDTEUR`): the exact-spelling id misses
70k–203k minute rows per window vs the union (33–51% of the window's rows — holed series under
the 809 loader's exact-spelling read). The other 13 twins carry zero missing rows. No XBT/BTC or
XDG/DOGE twin shows a coverage gap. **Method, per the ruling:** 897's runner applies the canon
alias union (`mcts_trader/data/asset_aliases.py`, the 868/877 normalization + agreement gate +
evidence floor) as a READ-TIME layer for BOTH the universe ranking and the panel build — the 809
lineage is not edited (out-of-scope stands); the 6 affected slots are disclosed per-cell, and
any slot whose universe-rank changes under the union read is named in the artifact (the
universe itself is computed from bar counts, so the union read can change the top-64, not just
patch holes in it).

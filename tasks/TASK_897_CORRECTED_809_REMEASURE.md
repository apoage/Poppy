> **STATUS (2026-08-22, factory run 39): D57's NARROWING IS DONE AND VERIFIED — the row HALTED on
> the 3-cycle rule before shipping (D59). NOT SHIPPED. Work preserved in `stash@{0}`, 188 tests +
> selftest 12/12 green. THE SRV LEG MUST NOT LAUNCH until the read-write DB handle below is closed.**
>
> **What is finished and verified (8 lens-runs, 4 default-tier + 4 escalated).** The tau claim is
> narrowed to what tau is: a linear-ACF statistic equal to the dependence RANGE only inside the
> triangular family the calibration was measured against, demonstrably under-reading outside it
> (a 4%-of-variance AR1 at phi=.99 reads tau=1, certifies 100%, covers 0.805; at phi=.999, 0.376).
> The guard is stated as EXPECTED-NOT-TO-BITE here — and a flow-verifier CONFIRMED that empirically,
> driving 36 real cells through the real pipeline and measuring tau=1-3 against block 60, so its
> near-universal pass is disclosed as an absence of measurement, not a clean bill of health.
> `CI lo > 0` is recorded UNAVAILABLE-because-uncalibratable with the D49/D55 structural reason as the
> FIRST box in the artifact body; the calibrated pair is renamed `ci_lo_ADVISORY`/`ci_hi_ADVISORY`,
> never band-readable, with the qualifier travelling in the SAME table cell as the number. TASK_901 is
> named as the range-based replacement that re-reads the conjunct from the same artifacts.
> **A regression judge rebuilt the pre-D57 tree in a worktree and proved all 144 per-cell artifact rows
> BYTE-IDENTICAL: this work changed CLAIMS, never NUMBERS.**
>
> **The provenance wave — the transferable result of this run.** Asking one question of each shipped
> constant (*is this the exact output of the command its own comment advertises?*) found FIVE that were
> not, and they were re-pinned or re-disclosed with superseded records kept visible per D39:
> `MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH` (claimed "MC noise ~6pp, never bit-identically" on a
> BIT-DETERMINISTIC path), `MEASURED_CALIBRATION_TABLE` ("fresh random draws each run" — strata are
> deterministically seeded), `MEASURED_TABLE_SEED_JITTER` (provenance unrecoverable; its shipped clause
> "one of those six seeds landed on 0.9267 exactly" RETRACTED — and the retraction itself was then
> caught contradicting its own paragraph and re-scoped), `MEASURED_DEPENDENCE_REFUSAL_OC` (silently
> rounded to 3dp), `ci_coverage_probe.MEASURED_COVERAGE` (caveat 70 lines from the claim it qualifies).
> Also corrected: `BLOCK_CAP_SELECTION_RATIONALE` said "only cap=80/n_dec=500 read below target" when
> TWO cells do; `MEASURED_BORROWED_PAIRING_COVERAGE`'s row 639 cannot reproduce BY CONSTRUCTION, because
> stratum 620 was demoted by this row.
>
> **Guards hardened, each demonstrated:** the protected-DB guard now refuses `?`/`#`/`file:` spellings
> and compares realpath+samefile on BOTH cwd- and repo-relative interpretations (a judge had
> DEMONSTRATED a writable handle via `trading_data.db?x=1` — sqlite parses `?x=1?mode=ro` as one
> parameter, leaving `mode` unset); `--n-jobs` clamps and refuses above cpu_count (unclamped, 100000 was
> MEASURED spawning 7,549 processes); **gate (iv) now REFUSES tiers absent from the fee grid** — passing
> Kraken's real `14,12,10` previously measured NOTHING while the artifact called them "the
> Kraken-reachable subset", fail-open toward CELL CLEARS since the verdict would be read off cheaper 8bp
> cells; the dependence counter no longer reports "guard refused" for cells that carried no data (the
> banner had contradicted its own table 144-to-0); `_parse_window` refuses unparseable and inverted
> windows.
>
> **WHAT HALTED IT (D59).** A third, unguarded door to the same hazard: `sweep_gated_maker.py:405`
> calls `probe.report_bar_frequency()` — the harness's FIRST data touch on every run — and
> `probe.py:162` opens a bare `sqlite3.connect(cfg.db_path)`, i.e. READ-WRITE-CREATE, on whatever
> `--db-path` names. Demonstrated: a nonexistent `--db-path` had a zero-byte file CREATED at it, and DDL
> through that exact form succeeded on a decoy. This falsifies the harness's own sentence at
> `sweep_gated_maker.py:44` ("this harness never writes to it") and breaches the config invariant — and
> a read-write open of a 24GB sqlite carrying a hot journal performs rollback RECOVERY and writes, which
> `mode=ro` cannot. The remedy is a few lines inside the in-scope file; it was cycle 4, and the rule
> caps at 3.
>
> **REMAINDERS, all named in D59:** the 639 overlay disclosure exists in code but is NOT rendered (the
> borrowed-pairing table renders no `reason` column, unlike the per-stratum table beside it); the
> Scope-honesty prose still cites a "near-100% pass rate" the empty run does not have (errs
> conservatively); the `dep<=block` column prints a bare `False` for cells the guard never ran on; **the
> inherited-IC check is VACUOUS on a non-finite IC** (`nan > tol` is False) while the artifact records it
> as performed; a 100%-failed run gets no top-level statement while a 0-decision run gets a large one;
> `--out` DEFAULTS to the tracked `experiments/897_corrected_809/result.md`, so a bare run writes into
> the repo tree; exit-code semantics are deliberately untouched and seeded as **D58**; the
> `MEASURED_DEPENDENCE_REFUSAL_OC` dep-multiple grid is still {0,1,2,4,8} (at 1.5x the guard still
> certifies 8-13% of draws at 0.905 coverage); the 809 lineage's other plain `sqlite3.connect` sites and
> the never-validated maker-fill model stand unchanged.

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

**D57 (ruled (a), 2026-08-22) — the srv leg's verdict reads on the two readable conjuncts.** The
τ dependence guard was measured inert out-of-family (valid only inside its calibration family;
τ = 1–2 < block 60 on every real per-decision series — it would certify ~100% where containment
measured 0.639). The guard ships with its claim narrowed (it must say it is expected NOT to bite
here). The band's third conjunct, `CI lo > 0`, is **UNAVAILABLE-because-uncalibratable** on this
leg; every interval carries the `UNCALIBRATED` qualifier and the verdict reads on **Sharpe ≥ 1.0
and PnL > 0** plus per-cell point estimates. If a cell clears the two readable conjuncts at an
attainable maker tier, the next step is fill-model validation via small live limit orders — the
backtest CI was always advisory to that gate. **TASK_901** (the range-based dependence probe,
out-of-family validated) upgrades the third conjunct retroactively FROM THE SAME ARTIFACTS.
The stashed D55/D56 work (stash@{0}, 116 tests green) resumes under the narrowed claim. D56(a)
bonus finding: the alias union changes ZERO of 291 universe ranks (the 6 material twins carry 0
train-window rows — the identity split is OOS-only), and window-b missing fractions measured:
/USDTEUR 56.1%, /USDTAUD 56.0%, /USDC/USDT 55.2%, /USDTCHF 53.4%, /USDCGBP 51.3%, /USDC/USD 37.7%.

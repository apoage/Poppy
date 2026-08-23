# TASK_902 — 897 instrument hardening: pay the honesty debt deliberately, not one cycle at a time

**Status**: READY 2026-08-22 (planner; seeded by D59's ruling, option (a) — the remainders of the
run-39 halt are paid here, deliberately, instead of expanding the narrowing row a third time).
**Size: S. Risk: LOW** — every item is a named, demonstrated defect with a known locus; the
discipline is D57's: this row changes CLAIMS and RENDERING, never NUMBERS.

## Why this row exists

Factory run 39 found that TASK_897's instrument carries a systemic honesty debt: asking one
question of each shipped constant — *is this the exact output of the command its own comment
advertises?* — failed five of seven times, and each verification cycle found another instance of
the same shape. The narrowing shipped under D59(a) closes only the launch-blocking read-write DB
handle. Everything else lands here, so the debt is retired by a row scoped for it rather than
discovered one cycle at a time. **Until this row ships, the 897 artifact is readable only with
the banner's disclosures in hand — the srv-leg verdict read (planner) already knows them.**

## Scope (each item named in run 39's halt record; each must be demonstrated, not asserted)

1. **D58(a) — exit-code contract.** Exit NON-ZERO on a gate-(ii) completeness violation
   (`n_cells_recorded != n_cells_expected` — the instrument failing its own gate). Keep exit 0
   for an HONEST empty result (`ok=0` with the artifact stating so — on the fixture this is the
   NORMAL state; a non-zero exit there would train operators to ignore the code). Tests: a
   forced contract violation exits non-zero; a fixture honest-empty run exits 0.
2. **The inherited-IC check must not record "performed" when it was vacuous.** Today
   `nan > tol` is False, so a non-finite IC silently PASSES while the artifact records the check
   as performed — fail-open toward CELL CLEARS. Refuse or record `not-evaluable` explicitly.

   **AMENDED 2026-08-23 (factory run 41, from TASK_897's fail-closed panel): there are TWO
   independent root causes of this vacuity class, and this item originally named only the first.**
   The second: `decide.inherited_signal_check_chunked` reads `tgt =
   backtest.SIGNAL_806_IC_FULL.get(H)` and guards with `if tgt is not None and abs(ic - tgt) >
   tol`. For any H **outside `{1, 5, 20}`** — the only H the 806 pins cover — `tgt is None`
   short-circuits the comparison and the function returns a real, finite, **never-compared** IC.
   An `isfinite` screen cannot catch this: the value IS finite. Reproduced live at H=2 (`ic
   −0.0036549…` recorded `status: "pass"`) against a pins dict of `{1,5,20}`.
   Not reachable from the CLI today (`GRID_H` and `SMOKE_GRID_H` are both subsets of `{1,5,20}`)
   but reachable through the public `run_sweep(grid_h=…)` API — the exact reuse surface D60's own
   text points at for the 898-class decay work.
   TASK_897's D60 fix pass closes this for the pin-era guard it restructured (a `tgt is None`
   guard beside the `isfinite` one). **This row owns the general statement:** wherever the
   instrument records an inherited-IC outcome, "no pin for this H" must record `not-evaluable`
   and never `pass`, and the artifact must never call an uncompared value a fidelity proof.
   The two causes are one rule: **a check that did not execute a comparison is not a passed check.**
2b. **An absent or unreadable `--db-path` crashes uncaught, and NO artifact is written.**
   **NEW 2026-08-23 (factory run 41, TASK_897's flow-verifier; pre-existing, NOT introduced by
   D60).** Reproduced: `--db-path <nonexistent>` raises `sqlite3.OperationalError: unable to open
   database file` at `probe.report_bar_frequency` — the D59-era FIRST data touch, which has no
   exception handling and runs BEFORE the D60 pin-era block and before `write_artifact` are ever
   reached. Result: traceback, exit 1, **no artifact at all**.
   Why this belongs to this row specifically: item 1 above makes the exit code meaningful, but
   the standing orchestration contract (D58, pre-902) is that **orchestration PARSES THE
   ARTIFACT, never the exit code** — and this path produces nothing to parse. A refusal the
   orchestrator cannot read is indistinguishable from a run that never started. The fix is a
   controlled refusal at the first data touch that still writes a legible artifact (or a refusal
   whose contract is explicitly documented as artifact-less, chosen deliberately rather than by
   omission).

3. **Render the 639 overlay disclosure.** The disclosure exists in code but the borrowed-pairing
   table renders no `reason` column (the per-stratum table beside it has one). Row 639 cannot
   reproduce BY CONSTRUCTION (stratum 620 was demoted by this row) — the artifact must say so at
   the row, not only in prose.
4. **Fix the Scope-honesty prose.** It cites a "near-100% pass rate" the empty run does not have
   (errs conservative — still a claim the run doesn't back).
5. **`dep<=block` column honesty.** Cells the dependence guard never ran on print a bare `False`,
   reading as a measurement. Render `not-run` distinctly from `False`.
6. **Top-level statement for a 100%-failed run.** A 0-decision run gets a large banner; a run
   where every cell's decision engine FAILS gets none. Symmetric treatment.
7. **`--out` must not default into the tracked tree.** Today a bare run writes
   `experiments/897_corrected_809/result.md` (tracked). Default to an untracked `runs/` path or
   require `--out` explicitly. This retires the standing "never use the default --out"
   instruction-discipline every 897 brief has carried.
8. **`MEASURED_DEPENDENCE_REFUSAL_OC` grid disclosure.** The OC is disclosed at dep multiples
   {0,1,2,4,8} only; at 1.5x the guard certifies 8–13% of draws at 0.905 coverage. Densify the
   grid or state the gap at the constant.

### Added by run 40 (all pre-existing; routed here by the ship's own remainders list)

9. **The protected-DB gate binds at the wrong layer.** `assert_db_path_allowed` is called only
   from `main()` — `run_sweep`/`_run_sweep_inner` never call it, so a programmatic caller reads
   the DB without the acknowledgment flag. Bind the guard at the data-touch layer.
10. **TOCTOU residual:** the guard returns the *unresolved* string, so a symlink can be
    re-targeted between check and open. Resolve once and feed both the check and the open (the
    run-40 `//localhost` fix normalized the string; resolution is still open).
11. **Refusal echoes are unbounded** (a 900k-char `--db-path` produced a 900k-char refusal).
    Clip with the path still named (the TASK_888 precedent: 1.6M chars → <4k).
12. **The artifact must state which grid ran.** The smoke grid is 144 cells; the full
    pre-registered grid is 864. Today the artifact does not say which — load-bearing for every
    later reader, TASK_901 first among them ("144/144" must never be read as full coverage).

**Explicitly NOT in scope:** the FIFO `<db>-journal` sidecar hang — operator call, run 40, a
documented limitation rather than something to guard (the FIFO guard was TASK_891's failure
mode recurring; complexity inside a fail-closed guard is itself risk).

## Acceptance (gates)

- (i) All eight scope items demonstrated (test or rendered artifact excerpt per item).
- (ii) **Numbers never move:** all 144 per-cell artifact data rows byte-identical against the
  post-D59-ship tree through every cycle (the run-39 regression-judge method).
- (iii) Full 897 suite + selftest green.
- (iv) The banner records which items changed rendering only vs which changed refusal behavior.

## Out of scope

Re-running any srv leg. Touching the 809 lineage (`probe.py`/`backtest.py`/`decide.py`).
The maker-fill model (OHLC-touch + flat adverse selection) — still never validated against a
real order; that is the fill-model validation step in the D57 consequence chain, not this row.
Any change to measurement content — if an item is found to REQUIRE moving a number, stop and
escalate to the planner rather than moving it.

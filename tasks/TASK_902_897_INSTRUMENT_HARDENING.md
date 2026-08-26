> **STATUS (2026-08-27): SHIPPED as `task902-897-instrument-hardening`** — factory run 43, two
> cycles. All 12 scope items landed; the row's honesty debt is retired.
>
> **The discipline held: NO NUMBER MOVED.** Gate (ii) was re-derived twice by an adversarial lens
> using its OWN header-keyed parser — the second time against a `git clone` reconstruction of HEAD
> rather than the working tree — giving **zero numeric field diffs across all 144 rows**. Every
> rendering diff maps to a named scope item. That mattered mechanically: cycle 1 inserted a
> `book_w(mean)` column, and a positional parser would have false-passed the whole comparison.
>
> **Item 2 needed no code.** Its two root causes were already closed by 897's D60 pass, verified
> independently at both sites — the `isfinite` guard and the `tgt_pin is None` guard, both routing to
> `status: "failed"` and dropped by the consumer, and it generalises to the public
> `run_sweep(grid_h=…)` surface because the loop iterates the caller's own `grid_h`. No fabricated
> work; the row says so.
>
> **The BLOCKER cycle 1 introduced, and the sharper thing found while fixing it.** Item 2b's
> `except Exception` rendered a genuine internal bug in **byte-identical banner language** to a
> legitimate refusal, and discarded the traceback entirely — strictly worse than the traceback it
> replaced, since the standing contract is that orchestration parses the artifact and not the exit
> code. The fix keeps a broad catch (a bare traceback writes *nothing* at `--out`, which is exactly
> the unreadable state item 2b closed) but renders an unmistakably different artifact and captures
> `traceback.format_exc()` on both paths. **Narrowing the `except` to a tuple turned out to be
> illusory**: in CPython `ProgrammingError → DatabaseError → Error`, so sqlite3 API misuse — a defect
> — cannot be excluded by any tuple that still admits "file is not a database". Hence an explicit
> predicate plus `NOT_A_REFUSAL_DESPITE_MATCHING`. And **neither artifact quotes the other's banner
> text**, deliberately: an orchestrator greps these files, so a cross-reference would make the two
> kinds match each other's search strings.
>
> **Echo clipping went from 4 sites to all 13** (48 call sites). Measured: 1,800,148 → 4,238 chars
> and 2,700,333 → 6,423. `write_refusal_artifact` had been echoing **all 7 grid axes** unbounded.
> `_clip_echo` is now hardened so it cannot raise — including lone surrogates, which are legal inside
> a `str` but raise `UnicodeEncodeError` at *write* time, so an unhardened clipper would have looked
> fine while the refusal still died.
>
> **A zero-cell run is now loud**: its own banner AND exit 2. The defence against D58 is that gate
> (ii)'s `recorded == expected` is satisfied **trivially by `0 == 0`** and provides no assurance
> whatsoever — that is the instrument failing to act, not an honest empty result. An honest empty
> result (cells recorded, none measured — the normal fixture state) still exits **0**, verified
> against the real fixture.
>
> **NAMED REMAINDERS (both MINOR, both logged, neither blocking).**
> (1) `test_main_does_not_exit_nonzero_on_an_honest_empty_result` still does not pin the state it
> claims to. Cycle 2 moved its premise off the degenerate `0 == 0` case, which was right, but landed
> on `ok == 0 / dropped == 144` — which is by the code's own taxonomy the **100%-failed** scenario.
> The real honest-empty state is `ok == 144` with zero decisions per cell. Not a regression
> (`exit_code_for` never reads `n_cells_ok`, and the real behaviour was verified directly), but no
> unit test pins the actual combination and the docstring wrongly claims it does.
> (2) Five per-cell `reason` fields remain **unclipped**. The reasoning for leaving them is sound and
> was confirmed: they echo into the gate-(ii)-protected data table, so clipping them risks moving
> artifact bytes inside the protected surface. Their inputs are hardcoded grid constants, not
> operator-controlled strings. Residual: an exception whose `str()` embeds a large numpy repr could
> bloat one cell's reason.
>
> **Caveat carried INTO the record, at the field itself and pinned by a test:** the fixture's 7-asset
> universe never clears `k=16`'s `2k` floor, so **no cell of the 144-cell smoke grid ever realizes
> `n_decisions > 0`**. The gate-(ii) fixture rerun is therefore SILENT on `realized_book_width_mean`
> for nonzero-decision cells; only the real 64-asset leg exercises it as shipped.
>
> **Verification:** fail-closed ×1 + regression ×2. Suite **250 passed / 3 failed**, where the 3 are
> pre-existing and out of scope — `test_recompute_gate.py` is unmodified and imports only the also
> unmodified `honest_estimator`, so they cannot be this row's; I confirmed that independently rather
> than accepting it. HEAD 215 → 253 collected, **0 tests removed**, 38 added. selftest 12/12. The
> suite run and selftest were orchestrator-run, not taken on report.
>
> **Process note:** the cycle-2 fix pass died on a transient API error mid-report and was RESUMED
> rather than re-spawned. On resume it re-established state and confirmed the cut had landed after
> all implementation and verification, with nothing half-applied.

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

2c. **A grid cell can silently produce ZERO decisions and still be recorded, and nothing says so.**
   **NEW 2026-08-24 (factory run 41, found by TASK_901's real-data leg, independently verified by
   its correctness judge — both by reading the code and by running it).** `decide._decide_step`
   (`runs/809_decision_rule_rescue/decide.py:198`) opens with
   `if npr < (2*k if variant == "long_short" else k): return tgt` — so on a universe narrower than
   the grid's `k` demands, EVERY bar returns the empty target and the cell yields **0 decisions**.
   Measured: 897's own grid value `k=16`, long_short, against a 7-asset universe → `n_decisions: 0`.
   Not triggered on the real 64-asset leg (16 of 64 clears the `2k` floor), so this is not a defect
   in the shipped numbers — it is a **silent-degenerate-cell class** in the instrument: the cell is
   not a fit failure, so it does not take the gate-(ii) drop path, and an empty measurement is not
   visibly distinguishable from a measured one in the artifact.
   Related and worth stating in the same place: **τ can bind before `k` does.**
   `n_elig = max(1, int(round((1.0 - tau) * npr)))` means at τ=0.9 a 7-slot universe yields
   `n_elig = 1` regardless of `k`, so the realized book is one name per side while the cell is
   labelled by its `k`. A cell's label should not imply a book the run did not hold.
   Fix direction: record realized `n_decisions` and realized book width per cell, and make a
   zero-decision cell state that it is empty rather than merely reporting zeros — an empty cell that
   renders like a measured one is the same honesty class as items 2 and 2b above.

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

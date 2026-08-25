> **STATUS (2026-08-25, planner): COMPLETE — VERDICT READ: ALL FAIL. The minute surface is closed
> for real.** Srv leg #2 (`runs/897_corrected_809/result_897_2026-08-24.md`, 864/864 cells ok,
> runtime 11,438.5s, tiers {8,16} bp pre-registered at `4e79b78`): **0/576 attainable-tier cells
> clear** (Sharpe ≥ 1.0 & PnL > 0); not one cell even reaches Sharpe ≥ 0 or PnL > 0; **0/864 at
> any nonzero fee**. Best attainable cell: window a **−5.045** (H=1 k=16 long-only τ=.9 N=60,
> fee 8, adv 0); window b −8.425. 0/188 clear even at adv=0 (the fill model's most generous
> setting) — the verdict needs no fill-model generosity. The pre-cost edge is real (fee0/adv0
> 96/96 "clear", best +204 — the frictionless fantasy regime) and any measured cost term destroys
> it by orders of magnitude (adv=5 alone at fee0: +204 → −634). **The killer is adverse
> selection, not the venue fee.** 809's retracted PASS is falsified, not merely un-validated.
> LEDGER row `897/809-corrected`; consequence → **D65**. Leg #1's null-run artifact is preserved
> on srv (`result_897_2026-08-23.md`).
>
> **STATUS (2026-08-23, factory run 41): SHIPPED as `task897-d60-guard-rebase` (`5ee6381`).
> The D60 guard re-base landed exactly as ruled — the inherited-IC guard now asserts the 806 pins
> on a PIN-ERA panel (`probe.Config`'s own split defaults, `test_end=2025-04-01`, exact-spelling
> read), **tol UNMOVED at 1e-3**; the sweep-panel IC is a disclosed, non-gated MEASUREMENT.
> 809 lineage byte-untouched (md5-verified at four points). 215 tests (212 baseline + 3 new),
> selftest 12/12, fixture smoke green both ways, `## Cells` through EOF byte-identical vs
> pre-change (`8468be63…`) — no cell number moved.**
>
> **The panel found a SECOND vacuous-pass root cause and it is closed.** The lineage check guards
> with `if tgt is not None and abs(ic - tgt) > tol`, so for any H outside `{1, 5, 20}` the
> comparison short-circuits and a real, **finite, never-compared** IC was recorded `status:
> "pass"` under a bullet calling it "the lineage-fidelity proof; GATED". An `isfinite` screen
> cannot catch it — the value IS finite. Reproduced independently by two lenses (H=2, H=7). Not
> reachable from the CLI (`GRID_H`/`SMOKE_GRID_H` ⊂ {1,5,20}); reachable via the public
> `run_sweep(grid_h=…)` API — the exact reuse surface D60 names for 898-class decay work.
>
> **THE SRV LEG IS READY TO RELAUNCH AND THAT IS THE PLANNER'S, NOT THE FACTORY'S.** Same command
> as the 2026-08-23 launch record (tmux, `safe_run --mem 64G --cpu`, defaults, full 864-cell grid,
> NOT `--smoke`, NOT `--skip-inherited-ic-check`), after deploying `5ee6381`. Expected: the guard
> passes on the real pin-era basis — planner-measured today at **+0.151053 / +0.141160 /
> +0.059755** vs pins ±1e-3. Gate (iv) still requires the operator's attainable Kraken maker-fee
> tiers recorded BEFORE the verdict is read; orchestration PARSES THE ARTIFACT, never the exit
> code (D58 pre-902 contract).
>
> **The verdict reads on TWO conjuncts. `CI lo > 0` stays UNAVAILABLE — but NOT permanently, and
> an earlier draft of this banner overstated that.** TASK_901's first pass reported a total
> negative (no candidate range estimator certifies out-of-family at ≥93%). Its correctness panel
> then measured that the negative is only PARTIAL: `variance_ratio` was condemned by a read-off
> rule its own docstring does not describe — under the documented rule it reads range 256 on the
> persistent component and takes gate (ii)'s REFUSE/PASS branch — so that candidate has never been
> correctly tested. `acf_band` and `spectral_zero` do genuinely fail. **The honest status is
> "not yet established", not "structurally impossible".** TASK_901 is in a corrective cycle; see
> its banner for the settled conclusion.
>
> **REMAINDERS → TASK_902** (both added to that row this run): item 2 now names BOTH root causes of
> the vacuity class (the general rule: *a check that did not execute a comparison is not a passed
> check*); new item 2b — an absent/unreadable `--db-path` crashes uncaught at
> `probe.report_bar_frequency` **before any artifact is written**, so D58's artifact-parsing
> contract has nothing to parse.
>
> _Process note, disclosed not buried: the fix cycle's srv jobs were `safe_run`-wrapped and
> GPU-pinned but not all run inside tmux — they would not have survived an SSH drop. A test-stub
> contract drift was also found and fixed (`fake_run_decision` returned an array where
> `decide._run_decision` returns a scalar `fill_frac`), invisible for four cycles because no prior
> D60 test rendered an artifact._

> _(superseded — the D60 halt banner, kept for the record:)_
> **STATUS (2026-08-23, planner): SRV LEG #1 = HONEST NULL (the inherited-IC guard refused on a
> basis the pins never claimed — 864/864 recorded, ok=0, artifact preserved on srv). D60 RULED:
> re-base the guard to its actual claim (pin-era panel, tol UNMOVED; decomposition measured:
> pins reproduce to −4.5e-5 on today's table, chunked ≡ unchunked exactly, the whole drift is
> the test-span extension). RESUME = the guard re-base in `sweep_gated_maker.py` (in-scope file,
> lineage byte-untouched) + relaunch (~17 min).**
>
> **STATUS (2026-08-22, factory run 40): SHIPPED as `task897-d59-ro-handle`. The D59(a) resume is
> complete — the read-write DB handle is closed, and a protected-DB gate bypass was found and closed
> with it. Verified on gpu-server (GPU-pinned, `safe_run`-wrapped): 207 tests + selftest 12/12.**
>
> **What closed.** `probe.report_bar_frequency` — the harness's FIRST data touch on every run — no
> longer opens `--db-path` READ-WRITE-CREATE. `probe.sqlite3` is rebound to a read-only proxy around
> that one call and restored in `finally` (rebinding the NAME, not the attribute: `probe.sqlite3` and
> `alias_read_layer.sqlite3` are the same cached module object, so mutating `.connect` in place
> recurses infinitely — measured). `probe.py` stays byte-untouched (md5 `ab8e94b0…`, mtime 2026-06-26),
> checked explicitly because `runs/` is gitignored and `git status` cannot see it.
>
> **The bypass, which mattered more than the FIFO work.** A leading `//localhost` is stripped by
> sqlite's URI parser as an ignored authority, so any belt reasoning about the raw string reasons about
> a different file than sqlite opens. Measured against a decoy: `assert_db_path_allowed('<decoy>')`
> REFUSED, `assert_db_path_allowed('//localhost<decoy>')` **PASSED**, and `_connect_ro` then opened the
> protected decoy. Closed by normalizing once and feeding both the quote and the open, plus a leading
> `//` refusal at the gate. Mutation-proven on srv both directions: remove BOTH belts → breach
> reproduces; restore → refuses; either belt alone still refuses.
>
> **The FIFO guard was built and then REMOVED, on operator call.** Cycles 2–3 chased a `mode=ro`-on-FIFO
> hang through ever more exotic spellings (FIFO → symlink-to-FIFO → `//localhost`-prefixed FIFO). That
> is TASK_891's recorded failure mode — chasing total coverage of an unbounded input space — and
> complexity inside a fail-closed guard is itself risk. `--db-path` is an operator-supplied path to a
> real database on a scheduled run. **The hang is now a documented limitation, not a defence.**
>
> **One test's stated red-case was false and is retracted.**
> `test_the_normalized_path_is_the_file_sqlite_opens` claimed dropping `normalize_db_path` from
> `_connect_ro` would make it red; mutation-tested, it stayed GREEN, because sqlite does the identical
> stripping itself. It pins the mirror, not the call site. Corrected in place.
>
> **Byte-identity, honestly.** The 144-cell fixture A/B differs only on `Runtime:` — but every cell is
> zero-decision/NaN (≤6 assets present per bar vs the grid's `k≥16` floor), so it proves STRUCTURAL
> identity only. The numeric proof is `bar_sec` equality plus an off-grid non-NaN A/B (0 mismatches).
> The smoke grid is 144 cells; the full pre-registered grid is 864.
>
> **REMAINDERS → TASK_902**, all pre-existing: `run_sweep`/`_run_sweep_inner` never call
> `assert_db_path_allowed`; the guard returns the unresolved string (TOCTOU residual, not closed);
> unbounded refusal echoes; a FIFO `<db>-journal` sidecar still wedges (hangs identically pre-fix); the
> artifact never states it ran the reduced smoke grid. Plus everything D58/D59 already deferred.
>
> **THE SRV LEG IS NOT LAUNCHED.** It is planner-scheduled, and the operator's attainable Kraken
> maker-fee tiers must be recorded BEFORE the verdict is read (gate iv). Orchestration PARSES THE
> ARTIFACT, never the exit code (D58's pre-902 contract).

> _(superseded — run 39's halt banner, kept for the record:)_
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

**Launch record (2026-08-23, planner). GATE (iv) SATISFIED PRE-LAUNCH: the attainable
maker-fee set is `{8, 16} bp` (operator, 2026-08-23), recorded here before any cell is read.**
Zero-maker-fee stays out of reach per the Fork-C venue ruling; Kraken's real 14/12/10 ladder is
not in the pre-registered grid and gate (iv) refuses off-grid tiers, so the grid subset it is.
**Data basis (amended at launch, stronger than planned):** the workstation and srv DBs are not
byte-identical (sha256 `799dca96…` @ 26,245,337,088 B vs `c19782d1…` @ 26,243,768,320 B), so an
extract was built and verified per D54 (`runs/897_corrected_809/build_extract.py`: full
`assets` + all `ohlcv` rows in [2023-05-16, 2026-03-01) — 120,900,027 rows; per-window counts +
ordered-stream md5 `faa0bbdc…` MATCH). But the transfer was uplink-bound (~0.5 MB/s, ~6 h), and
the extract's own verification tooling then answered the better question directly: **srv's DB
produces the IDENTICAL content hashes as the source of record across the sweep's entire SQL
surface** — `ASSETS=1c26e91f763129a12be3ce1ef1cb2db2`, `OHLCV(span,
120,900,027 rows)=faa0bbdc480352bab90420eeb49f6208`, computed by
`runs/897_corrected_809/hash_db_content.py` on both hosts. The byte difference is page layout
only. **The leg therefore runs on srv's `trading_data.db` directly** (`--db-path
/home/operator/Trader/trading_data.db --i-am-the-scheduled-real-data-run`), on data proven —
not assumed — identical to what the calibration measured. The verified extract remains on the
workstation (`runs/897_corrected_809/minute_extract_897.db`, sha256 `fd1fdd79…`) as a byproduct. **Code basis:** srv deployed to
`f313d9e` (deploy-branch tip carrying `e5c3864`); the 13 launch-critical files (897 instrument
+ 809 lineage incl. the gitignored `recompute_best.py` repair) sha256-verified IDENTICAL on
both hosts post-deploy. **Command as launched (2026-08-23 08:58 CEST, tmux `l897` on srv, `safe_run --name l897 --mem
64G --cpu`, defaults everywhere else — full 864-cell grid, NOT `--smoke`, NOT
`--skip-inherited-ic-check`):**
`sweep_gated_maker.py --db-path /home/operator/Trader/trading_data.db
--i-am-the-scheduled-real-data-run --attainable-maker-fee-bp 8,16
--out runs/897_corrected_809/result_897_2026-08-23.md`, log
`runs/897_corrected_809/launch_897_2026-08-23.log` on srv. Confirmed computing (106% CPU,
single-threaded as designed).
(Correction to this record as first committed: no `--n-jobs` — the sweep's argparse has no such
flag; the `--n-jobs` mentions in docstrings belong to the calibration probe. The sweep runs
single-threaded; runtime goes to tmux.)
Orchestration parses the artifact, never the exit code (D58 pre-902 contract). Verdict reads on
Sharpe ≥ 1.0 & PnL > 0 at the recorded tiers; `CI lo > 0` is UNAVAILABLE (TASK_901 upgrades it
from the same artifacts).

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

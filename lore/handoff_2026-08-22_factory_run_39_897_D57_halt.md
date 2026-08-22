# Factory run 39 — TASK_897 D57 narrowing: verified, HALTED before ship (D59)

**Date:** 2026-08-22. **Lane:** dark-factory implementer (claude-code). **Outcome:** row HALTED on
the 3-cycle rule. No code shipped. Work preserved in `stash@{0}` (188 tests + `trader selftest`
12/12 green). Committed this run: `tasks/DECISIONS.md` (D58, D59), the TASK_897 banner, this file.

## Queue state at entry

Ready: 1 TASK_897 (D57 resume — pop `stash@{0}`, narrow the tau claim, ship) → 2 TASK_881D →
3 TASK_901. FILLER 899 (waiting on the Q2-2026 dump). GATED 880 (D34+D35). Only OPEN wall at
entry: D43. Rows 2 and 3 were never started — row 1 consumed the run.

## What was done and verified

D57(a)'s narrowing is complete. tau is now stated as a linear-ACF statistic (integrated
autocorrelation time) that equals the dependence RANGE only inside the triangular-ACF family the
calibration was measured against, and demonstrably under-reads outside it: a persistent component
worth 4% of variance (`0.2*AR1(phi=.99)`) reads tau=1, certifies 100%, covers 0.805; at phi=.999,
coverage 0.376. The guard is disclosed as EXPECTED-NOT-TO-BITE on this data, and a flow-verifier
CONFIRMED that empirically — driving 36 real cells through the real pipeline (30k-44k decisions
each) and measuring dep_len 1-3 against block 60. `CI lo > 0` is recorded
UNAVAILABLE-because-uncalibratable with the D49/D55 structural reason as the first box in the
artifact body; the calibrated pair is renamed `ci_lo_ADVISORY`/`ci_hi_ADVISORY` and is never
band-readable, with the qualifier rendered in the same table cell as the number. TASK_901 is named
throughout as the range-based replacement that re-reads the conjunct from the same artifacts.

**Numbers did not move.** A regression judge reconstructed the pre-D57 tree in a separate worktree
(116/116, matching the halt note) and diffed generated artifacts: all 144 per-cell data rows
byte-identical. Every subsequent round was re-checked the same way. The row changed claims only.

## The transferable result: one question, five failures

The load-bearing finding is not any single fix. It is that asking one question of each shipped
constant — *is this the exact output of the command its own comment advertises?* — failed five
times out of the seven constants that advertise a command:

| constant | what it claimed | what was true |
|---|---|---|
| `MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH` | reproduces "within MC noise ~6pp, never bit-identically" | path is BIT-DETERMINISTIC; the record did not correspond to its stated seeds (seed 20001 matched the first value exactly, 20105 gave 0.900 where the pin said 0.944, and one cell held the pinned pair in the opposite order) |
| `MEASURED_CALIBRATION_TABLE` | "NOT bit-identical (fresh random draws each run)" | strata are deterministically seeded (`seed + 1000*index`); 11/11 spot-checked strata reproduce exactly |
| `MEASURED_TABLE_SEED_JITTER` | date + budget, no seeds | provenance unrecoverable at any probed index; re-pinned at a named rule |
| `MEASURED_DEPENDENCE_REFUSAL_OC` | agreement implied exact | silently rounded to 3dp (12 of 25 cells) |
| `ci_coverage_probe.MEASURED_COVERAGE` | "recorded verbatim" on the constant | 4 of 8 cells no longer reproduce (the block cap moved 20->60); the caveat sat 70 lines away |

Two more of the same class: `BLOCK_CAP_SELECTION_RATIONALE` said "only cap=80/n_dec=500 read below
target" when TWO cells do (`(100,8000)=0.9267`); `MEASURED_BORROWED_PAIRING_COVERAGE` advertises
"reproducible verbatim" but row 639 cannot reproduce BY CONSTRUCTION — stratum 620 was demoted by
this very row, so `calibrated_k_for(639)` returns None and 639 now prints identically to 619, the
deliberate negative control.

**Process note worth keeping.** Cycle 2 "resolved" the first of these by widening the disclosed
tolerance from ~3pp to ~6pp. That is a bar moved to fit its result, and it was also a false
mechanism claim (there is no run-to-run noise at a fixed seed on a deterministic path). It was
caught by re-deriving rather than reading, and it is why the row escalated to opus. A later round
then shipped a RETRACTION that contradicted its own paragraph four sentences later — the module's
own doctrine already says a wrong retraction is worse than the error it retracts.

## What halted it

`sweep_gated_maker.py:405` calls `probe.report_bar_frequency()` — the harness's first data touch on
every run — and `probe.py:162` opens a bare `sqlite3.connect(cfg.db_path)`: READ-WRITE-CREATE, on
whatever `--db-path` names, needing no URI trick and no typo. Demonstrated: a nonexistent
`--db-path` had a zero-byte file created at it, and DDL through that exact connect form succeeded on
a decoy. This falsifies the harness's own shipped sentence at `sweep_gated_maker.py:44` ("this
harness never writes to it") and breaches the config invariant. Not cosmetic: a read-write open of a
24GB sqlite carrying a hot journal performs rollback recovery and writes, which `mode=ro` cannot —
and the newest backup predates the 858 backfill. `alias_read_layer` had already routed probe.py's
other two connect sites through `_connect_ro`; `report_bar_frequency` is the one left behind.

The remedy is a few lines inside the in-scope file (compute `bar_sec` via `_connect_ro`, or
scope-patch `probe.sqlite3.connect` with a `finally` restore — the pattern already used ~20 lines
above for `decide.CFG`). It was cycle 4; the rule caps at 3. **The srv leg must not launch until it
is closed.**

## Guards that were hardened (all demonstrated, all in the stash)

- Protected-DB guard: refuses `?`/`#`/`file:` spellings, compares realpath+samefile on both
  cwd- and repo-relative interpretations. A judge had DEMONSTRATED a writable handle via
  `trading_data.db?x=1` — sqlite parses `?x=1?mode=ro` as one parameter, leaving `mode` unset, so
  both belts failed on the same string. `_connect_ro` now percent-quotes.
- `--n-jobs`: clamped, and refused above `cpu_count`. Unclamped, `--n-jobs 100000` was MEASURED
  spawning 7,549 processes and still climbing at 65s, on a machine with a documented hard-crash
  history under memory pressure.
- Gate (iv): now REFUSES attainable tiers absent from the fee grid. Previously Kraken's real
  `14,12,10` measured NOTHING while the artifact called them "the Kraken-reachable subset of the fee
  grid below" — fail-open toward CELL CLEARS, since the verdict would then be read off cheaper 8bp
  cells (8 < 10).
- Dependence counters: one classifier read by BOTH banner and table. Previously cells with n_dec=0
  were counted as "REFUSED on dependence grounds" (reading as the safety guard working) while the
  table rendered them "no CI computed" — a banner contradicting its own table 144-to-0.
- `_parse_window`: refuses unparseable and inverted windows. Previously `"not-a-date,also-not"` and
  inverted dates were accepted, silently yielding a measurement-free artifact whose completeness
  counters read green (the documented 863 shape).

## Decisions seeded

- **D59** — the halt. Recommends (a): pop the stash, close ONLY the read-write handle, ship the
  narrowing, and open a SEPARATE instrument-hardening row for the remainders.
- **D58** — exit-code semantics. A gate-(ii) completeness violation and an all-cells-failed run both
  exit 0. Deliberately not changed by the factory: the instrument's design is records-never-verdicts
  and the artifact is unambiguously non-green at `ok=0`. Recommends splitting the cases — exit 0 for
  an honest empty result (on the fixture, zero decisions is the NORMAL outcome, so failing there
  would train the next operator to ignore the exit code), non-zero on a contract violation.

## Open remainders (also in D59 and the task banner)

The 639 overlay disclosure is in code but NOT rendered (the borrowed-pairing table has no `reason`
column, unlike the per-stratum table beside it); the Scope-honesty prose still cites a "near-100%
pass rate" the empty run lacks (errs conservatively); the `dep<=block` column prints a bare `False`
for cells the guard never ran on; **the inherited-IC check is VACUOUS on a non-finite IC** (`nan >
tol` is False) while the artifact records it as performed; a 100%-failed run gets no top-level
statement while a 0-decision run gets a large one; `--out` DEFAULTS to a TRACKED path
(`experiments/897_corrected_809/result.md`), so a bare run writes into the repo tree; the
`MEASURED_DEPENDENCE_REFUSAL_OC` grid is still {0,1,2,4,8} (at 1.5x the guard still certifies 8-13%
of draws at 0.905 coverage); the maker-fill model remains never validated against a real order.

## Suggested next moves

1. Rule D59. Under (a) the row is a short, well-specified resume: pop, close the handle, re-verify,
   ship.
2. Point the same question at other lanes' constants — 892's calibration table and 875's
   walk-forward numbers are the obvious next targets. Nothing suggests this module was unusual; it
   is the one that was asked.
3. TASK_881D and TASK_901 were not started and remain takeable. 901's gate (iv) stays a remainder
   until the srv leg actually produces artifacts to re-read.

## Suite cost note

The 897 suite grew 116 -> 188 tests and ~9s -> ~150s, almost entirely from exact-reproduction tests
that re-run an advertised command and compare bit-identically. These were deliberately NOT marked
slow: a test that proves a command reproduces its constant proves nothing if it does not run, which
is the failure mode behind D43 (857's CLI had been dead since 868, so its own smoke-assert had never
once executed). The canonical factory gate (`python -m trader selftest`) is unaffected at 12/12.

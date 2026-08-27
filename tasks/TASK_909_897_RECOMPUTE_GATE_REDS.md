# TASK_909 — the 897 suite's 3 standing red tests at HEAD

**Status**: READY 2026-08-27 (planner; seeded from run-43's open item 4).
**Size: XS. Risk: LOW, but red-fatigue is the hazard** — a suite that stays red at HEAD trains
every future verification lane to discount red, which is exactly how the next real regression
gets waved through.

## The defect

`experiments/897_corrected_809/test_recompute_gate.py` has 3 failing tests at HEAD, targeting
`recompute_best.py::single_cell_gate_lines` (runs/809 lineage, gitignored). Pre-existing,
unowned, and out of scope of every row that touched the suite (902 shipped green with "3
pre-existing" exclusions named).

## Scope

Own them, one way or the other: **fix** the tests/code if the gate semantics are still live, or
**remove-with-prejudice** (a named `SKIP: superseded by X` is not acceptable — DELETE the test
or repair it; a skipped test that never runs again is a lie of omission in the suite). The
decision turns on whether `recompute_best.py`'s gate still has a consumer after the 897 verdict
(ALL FAIL, lane closed) — if nothing reads it, deleting the dead gate's tests AND saying so in
the commit is the honest close.

## Acceptance (gates)

- (i) `pytest experiments/897_corrected_809/` is fully green at HEAD with zero skips of this
  file's tests.
- (ii) The commit message records which semantics were judged dead vs live, and why.

## Out of scope

Any change to the 897 artifact or its numbers. The 809 lineage beyond the failing gate's
surface.

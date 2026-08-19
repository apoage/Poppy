# TASK_896 — D44 execution: name the 819 prototype's 1-row fill honestly

**Status**: READY 2026-08-19 (**UNBLOCKED — D48 ruled (a) by the operator 2026-08-19**). The
comment work itself is fix-complete and verified in the tree (run 36: the opus fix re-derived
every number independently, matched the orchestrator's measurement element-for-element; selftest
12/12; the 893 positive control still moves 11/12 metrics). What remains is the SHIP, plus the
planner-ruled re-pin change below.
**Size: XS. Risk: comment-only + a deliberate re-pin.** **Parallel-safe FILLER** —
`runs/819_single_asset/portfolio.py`, `.gitignore`, `tools/measure_leading_prefix.py`,
`experiments/LEDGER.md` (one row's digest).

## Scope (remaining)

1. **The D48(a) carve:** add `!runs/819_single_asset/*.py` to `.gitignore` (source files ONLY —
   zero artifacts: no npz, no JSON, no checkpoint enters git), then ship the already-verified
   comment in the same commit.
2. **The planner-ruled re-pin (D48 second item, ruled "correct and re-pin" 2026-08-19):**
   correct the two "verbatim" overclaim strings in `tools/measure_leading_prefix.py` (`:115-118`
   and the emitted `db.admission_predicate` value — `build_close_aligned` applies the
   `% BAR_SEC == 0` lattice test in numpy AFTER the SQL, while `_ADMIT` folds a modulo term into
   the SQL text; functionally equivalent, textually different) and the docstring granularity
   conflation (`:36` — 14 of the 16 assets span 2–24 MINUTE rows; "1 row" is only true of the
   downstream daily subsample). Regenerate the artifact, confirm it reproduces the pinned
   NUMBERS, and RE-PIN the LEDGER `893/prefix-cells` digest in the same commit, recording the
   OLD (`6bede1e9…`) AND NEW sha256 in the row. A repair tool refreshing its own pin pins nothing
   (D39's rule) — this change is planner-supervised, not the instrument's owner row acting alone.

## Acceptance

- (i) Every sentence of the shipped comment is mechanically checkable against the code and the
  893 instrument's outputs (already verified in run 36 — re-confirm, don't redo).
- (ii) `python -m trader selftest` green; zero behaviour change (comment + strings only).
- (iii) The 893 positive-control regression test stays green.
- (iv) `git ls-files runs/819_single_asset/` shows the `.py` sources and NOTHING else from that
  tree; the LEDGER row names both digests.

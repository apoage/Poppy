# TASK_896 — D44 execution: name the 819 prototype's 1-row fill honestly

**Status**: READY 2026-08-17 (planner; D44 operator-ruled DOCUMENT-AND-LEAVE 2026-08-17).
**Size: XS. Risk: comment-only.** **Parallel-safe FILLER** — `runs/819_single_asset/portfolio.py`
only.

## Scope (one item)

`build_close_aligned` admits bars only where `timestamp >= grid_start`, so 16 majors WITH real
earlier history receive a 1-row backward fill at row 0 (outside FULL; no scored number moves).
D44 ruled: the behaviour stays, the claim must exist and be true. Add a comment at the admission
predicate naming: (a) the 1-row fill and its extent (16 assets, row 0 only); (b) its
value-neutrality mechanism and its limit — row 0 == first admitted close exactly, so it is
invisible to log-difference lookbacks **as long as the bfill invariant holds** (measured: the bar
moves +0.9673047900277524 → +0.9672495154672209 only if the real earlier price is admitted);
(c) that `tools/measure_leading_prefix.py`'s `_ADMIT` is a deliberate verbatim copy of the OLD
predicate and must NOT be "fixed" to match any future edit here without a planner ruling —
editing a measurement instrument to change its own answer is self-certification (D44).

## Acceptance

- (i) Every sentence of the new comment is mechanically checkable against the code and the 893
  instrument's outputs.
- (ii) `python -m trader selftest` green; zero behaviour change (comment-only diff).
- (iii) The 893 positive-control regression test stays green.

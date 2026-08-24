# TASK_904A — walkforward smoke out-rooting + materialize gate + unified manifest reader (the verified half)

**Status**: READY 2026-08-24 (planner; re-cut from TASK_904 per D61(a), on the D45/881 precedent —
ship the verified half, don't bury it with the half that halted).
**Size: S. Risk: LOW** — this half passed two independent re-verifies; the work is re-apply, not
rebuild.

## Scope (exactly the three verified components — nothing else from the patch)

Source of the work: `lore/handoff_2026-08-24_factory_run_41_904_cycle3.patch` (1838 lines, incl.
its 32-test suite). Apply ONLY:

1. **Smoke out-rooting** — `run_smoke` (`experiments/837_walkforward/walkforward.py:1080-1147`)
   writes the "real" rails to a temp root and diffs, never to tracked `folds/real/*`.
2. **The `--overwrite-existing` materialize gate** — `materialize` refuses to write a directory
   whose content it did not generate, with the operator flag as the explicit override.
3. **The unified guarded manifest reader** — the two `--protect-file`-class readers unified
   (FIFO exits 2 where it hung at 124).

**EXCLUDE everything in the corridor/aggregate writer family** — that is TASK_904B, and it is
where every cycle-3 defect lived.

## Acceptance (gates)

- (i) `walkforward.py smoke` green with zero writes to tracked paths (content-hash A/B or
  strace); a decoy foreign-content directory is refused and left untouched.
- (ii) The 32-test suite from the patch passes, minus any tests of the excluded 904B surface
  (named in the ship note).
- (iii) The 903-verified 875 numbers and the `folds/real` tree hash (`1b31624a…`) do not move.
- (iv) The artifact/banner states which patch components were applied vs excluded.

## Out of scope

The corridor/aggregate writers (TASK_904B). The residual bare-`open` readers enumerated in the
cycle-3 re-verify (MINOR 4 — the accepted disclosure named 3 of ~10; that list belongs with
904B). `validate --fold-root` path resolution (TASK_905's). The `SPEC.md:60-72` real-db
re-materialization itself.

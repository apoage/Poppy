# TASK_904B — the corridor/aggregate writers, two-phase (the half that halted 904)

**Status**: PROPOSED 2026-08-24 (planner; re-cut from TASK_904 per D61(a)). Not queue-ranked until
904A ships — the two halves share files and the verified half goes first.
**Size: M. Risk: MEDIUM** — this is a redesign, not a patch re-apply: the halted row's BLOCKER
was structural (a per-fold write-then-refuse loop cannot be fail-closed).

## The defect and the required shape

`walkforward.py`'s corridor path writes inside the per-fold loop (`:1185`) *before* a later
fold's refusal (`:1174-1182`) — cycle 3's demonstrated BLOCKER: the corridor refusal mutated the
tracked fold tree while exiting 1 and printing "Nothing was written". The required design, from
the halt record: **two-phase — compute and decide every fold, then write.** A refusal anywhere
means zero writes, and the artifact can say so truthfully.

Also in scope: the residual bare-`open` readers enumerated in the cycle-3 re-verify (MINOR 4;
the previously accepted disclosure named 3 of ~10 — the full list is in that report).

## Acceptance (gates)

- (i) A refusal at ANY fold leaves the tracked tree byte-identical (hash A/B), exits non-zero,
  and never prints a false "nothing was written" — the message is generated from what actually
  happened.
- (ii) A bare `walkforward.py corridor` cannot rewrite tracked `corridor.json` from the fixture
  DB (D61's option-(b) hazard, closed here).
- (iii) The two-phase boundary is demonstrated: a test proves all decides precede all writes.
- (iv) 904A's gates stay green; 875's numbers and the fold-tree hash do not move.

## Out of scope

Re-opening 904A's shipped surface. The `SPEC.md:60-72` real-db re-materialization itself.

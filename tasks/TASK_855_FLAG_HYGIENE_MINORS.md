# TASK_855 — Flag-hygiene MINORs from the 854 panel (filler, XS)

**Status**: READY 2026-07-23 (run-16 wrap MINORs, operator-approved seeding).
**Size: XS. Risk: low** (two one-liners + artifact refresh; zero behavior change at defaults).

## Scope
1. `value_shift`: identical silent-inert hazard as the fixed `value-scale` — extend the same
   fail-loud guard (one line, same combo detection).
2. `results_fixture_smoke.json` (835 committed baseline): predates the `prior_weight` operator
   metadata key — refresh the artifact (additive only; parity must hold byte-for-byte on every
   pre-existing key; regenerate with the default operator and diff-verify).

## Gates
Selftest green; default tiers-SHA `ca72a38c7462ce24` unchanged; small panel (flag-wiring lens).

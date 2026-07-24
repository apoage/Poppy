# TASK_855 — Flag-hygiene MINORs from the 854 panel (filler, XS)

> **STATUS (2026-07-24): SHIPPED as `task855-flag-hygiene-minors`.** (1) `value_shift` fail-loud guard
> (sibling to 854's `value_scale`; inert combo `leaf_eval='true-env'` + no v̂ tail + `value_shift != 0.0`,
> default 0.0); ST11 extended to pin both guards. (2) `results_fixture_smoke.json` refreshed — additive
> metadata keys (851/854/856) only, every scored value byte-identical, tiers-SHA `ca72a38c7462ce24`
> unchanged, honest `ledger_row` date 2026-07-24 (panel caught the initial 2026-07-18 revert as
> provably false). selftest 12/12. Closes the D23 Ready queue.

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

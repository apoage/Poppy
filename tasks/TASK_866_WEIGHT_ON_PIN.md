# TASK_866 — Weight-ON parity pin for the 832 module (close the CLASS the 864 incident exposed) — FILLER

> **STATUS (2026-08-02): SHIPPED as `task866-weight-on-pin` (`ac8dfe1`).** Factory run 24. All
> three acceptance gates verified FIRST-HAND by the correctness judge (GREEN, 0 BLOCKER/MAJOR):
> (i) mutation tests both directions independently reproduced — the judge re-ran BOTH imprint
> configurations itself (weight-OFF matches CANON_832_PINS float-exact; weight-ON matches the
> tracked oracle on every key minus `db`/`checkpoint`), then confirmed run-side AND oracle-side
> single-field perturbations FAIL the pin; (ii) cross-configuration refusal fires both directions
> naming `vector_weight_rebalance` (in-memory artifact surgery; a schema-broken artifact missing
> `imprint.config` also refuses, never degrades to a numeric diff); (iii) selftest 12/12, 861
> smoke 11/11 legs green exit 0. Fail-closed oracle load proven live: the judge dirtied the
> tracked oracle in place and the pin REFUSED (git-porcelain dirty check), restored byte-identical.
> Scope-3 sweep result: `experiments/832_linear_imprint/` has exactly ONE committed baseline
> artifact — now guarded by an own-configuration oracle; no unguarded siblings. 1 MINOR found and
> closed in the wrap (an unrelated CLAUDE.md deletion in the working tree, reverted before commit).

**Was**: READY 2026-07-29 (seeded from the factory run-21 wrap, row B).
**Size: XS/S. Risk: low.** SAME-SURFACE constraint: `experiments/832_linear_imprint/` — serial
with TASK_865/867, never parallel.

## Why (the 864 root cause, named in its banner)

`CANON_832_PINS` is a **weight-OFF** oracle guarding a **weight-ON** baseline: the 861 smoke pins
the flagless default invocation, while the committed parity artifact is the weight-ON deliverable
(`--vector-weight-rebalance`, per `SPEC.md:205-211`). So nothing in the repo pins the weight-ON
numbers — which is exactly how `c2f1ba5`/TASK_846 moved them silently for ~80 commits while every
gate stayed green. The 864 misdiagnosis ("479 stale leaves") happened for the same reason: the
only oracle available compared the wrong configuration. This task closes the class, not the
instance: **every committed parity baseline must be guarded by an oracle of ITS OWN
configuration.**

## Scope

1. **Pin the weight-ON configuration**: either add a weight-ON pin set alongside
   `CANON_832_PINS`, or make the pins READ the tracked baseline (`results_fixture_smoke.json`,
   now recipe-pinned + validity-gated by 864) as the oracle — implementer's choice, but the
   chosen design must make "baseline moves without a pin failing" impossible for BOTH
   configurations. If reading the tracked file, the pin must fail loud when the file is missing/
   dirty/recipe-mismatched (no fail-open on a deleted oracle).
2. **Recipe identity**: the pin must verify the configuration that produced the baseline (the
   864 promote gate records it) — a weight-OFF artifact checked against weight-ON pins, or vice
   versa, is a REFUSE with the mismatch named, not a numeric diff.
3. **Sweep for siblings**: one read-only pass over the module's committed artifacts — any other
   baseline guarded by a different-configuration oracle (or none) gets a one-line report in the
   wrap (fix only if XS; otherwise report).

## Acceptance (gates)

- (i) Mutation test: perturb one weight-ON field in the baseline (or its producing path) → the
  new pin FAILS; restore → green. Same demonstrated for the weight-OFF pins (still live).
- (ii) Cross-configuration refusal fires with the mismatch named (scope 2).
- (iii) All existing smoke/selftest/e2e legs green; no behavior change outside the check paths.

## Out of scope

The 861 baseline promote port (TASK_867); the minute loader (TASK_865); refreshing any baseline
values (864 already reconciled them — this task only guards them).

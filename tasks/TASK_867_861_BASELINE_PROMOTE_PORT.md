# TASK_867 — Port the 864 promote gate to the 861 baselines (module-wide self-clobbering, confirmed) — FILLER

> **STATUS (2026-08-02): SHIPPED as `task867-861-baseline-promote-port` (`d670d57`).** Factory
> run 24. Default + `--quick` invocations write gitignored `*_local.json` outputs; the two
> tracked 861 baselines move ONLY via `--promote-baselines` (recipe-pinned from argparse canon,
> validity-gated, atomic, dirty-refusing, per-field diff). The `--quick` seeds-subset downgrade
> is a named REFUSAL. `smoke_e2e.py:428/459`'s vacuous within-run check is replaced by leg
> `vii_baseline_pins`, which pins the COMMITTED baselines; a pre-run fail-closed gate refuses
> missing/git-dirty baselines on every invocation (tamper → FAIL before any leg runs).
> Correctness judge GREEN first-hand, 0 BLOCKER/MAJOR, 1 NIT (non-blocking, `smoke_e2e.py:1273`
> demo isolation). Selftest 12/12; 861 smoke 12/12 legs (11 → 12, nothing shrank). Baseline
> attribution: collection volatile-only; results +21 vii leaves, 3 wall-clock moves; zero
> unattributable. Release order: promote → commit → re-verify (dirty gate refuses in between
> by design).

**Was**: READY 2026-07-29 (seeded from the factory run-21 wrap, row C).
**Size: XS/S. Risk: low.** SAME-SURFACE constraint: `experiments/832_linear_imprint/861/` +
`smoke_e2e.py` — serial with TASK_865/866, never parallel.

## Why (confirmed during run 21 — hit and restored twice)

The self-clobbering pattern 864 fixed for `results_fixture_smoke.json` is confirmed module-wide:
`861/chosen_vectors_fixture.json` and `861/results_fixture_smoke.json` are both rewritten in
place by ordinary verification runs, and `smoke_e2e.py --quick` silently **DOWNGRADES** the
latter (writes falsification seed 0 where the committed baseline holds seeds 0–3) while still
reading complete and green. Separately, `861/smoke_e2e.py:428/459` is only a within-run
byte-identity check — it passes against arbitrary committed content, so it guards nothing about
the baseline itself.

## Scope

1. **Port the shipped 864 pattern** to both 861 baselines: default invocations write untracked
   `*_local` outputs; overwriting a tracked baseline requires the explicit promote flag —
   recipe-pinned (the producing invocation's expected values re-derived from argparse defaults),
   validity-gated independent of `--smoke-assert`, atomic (`.tmp` + `os.replace`), dirty-refuse,
   per-field diff printed.
2. **Kill the `--quick` downgrade**: `--quick` must never be promotable over the full-seed
   baseline; a seeds-subset artifact promoted onto a seeds-0–3 baseline is a recipe-mismatch
   REFUSE (scope-1 machinery should catch this for free — assert it explicitly in the smoke).
3. **Make `smoke_e2e.py:428/459` check the committed baseline**, not within-run byte-identity:
  the checks must fail if the tracked file's content does not match what the current code +
  recipe reproduces (or, where exact reproduction is seed-dependent, validate the recorded
  recipe/seed-set identity). "Passes against arbitrary content" is the named defect — the fixed
  check must be demonstrated to FAIL against tampered content.

## Acceptance (gates)

- (i) Default + `--quick` invocations leave `git status` clean (before/after shown).
- (ii) Promote-path refusals fire: dirty file, recipe mismatch, seeds-subset-over-full, invalid
  artifact. Legit promote works and prints the diff.
- (iii) Tamper test: hand-edit a committed baseline field → `smoke_e2e.py` FAILS; restore →
  green.
- (iv) All existing legs green; the 861 e2e count does not silently shrink (`--quick` may skip
  legs only by SAYING so).

## Out of scope

Baseline value changes (guard, don't move); TASK_865/866 surfaces; verification-harness design
beyond the two named files.

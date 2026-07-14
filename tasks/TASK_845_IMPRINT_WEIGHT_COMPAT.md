# TASK_845 — 832 imprint weight-env compat (D13 redirect: thread TASK_840 through the E5 runner)

**Status**: READY 2026-07-14 (D13 DECIDED: **redirect** — E5/832 imprint→ExIt is the critical
path; H6 widening = named contingency).
**Size: S–M. Risk: medium** (`experiments/832_linear_imprint/` module only; ZERO canon diffs;
the ExIt half already takes trainer flags).
**Sequencing note**: ranked BEHIND TASK_843 (netting/snap-to-flat changes weight-path fee0
behavior — the expressibility gate below should measure against the env's FINAL form; if the
factory ships 845 first anyway, the gate numbers must be re-run after 843 lands).

## Why
D13 (operator, 2026-07-14) made the E5 imprint→ExIt path critical. The 832 runner shipped
2026-07-08 — BEFORE the TASK_840 weight-rebalance env (factory run 7) — and is weight-blind
in two distinct ways:
1. **No weight-env params.** CLI audit: no `--vector-weight-rebalance` / unit / max / deadband
   anywhere in the argparse block (`imprint.py:533-579`); both in-env walks (teacher trajectory
   generation and the imprint-sanity deterministic replay, walk seam `imprint.py:~127-141`,
   `jax.jit(partial(step_vector, prices, config=cfg))`) build a pre-840 env config. Training
   under order semantics and ExIt-ing under weight semantics would be a cross-env imprint —
   exactly the mismatch the p1b→p1c arc taught us to never leave unmeasured.
2. **Teacher mapping is order-semantics.** `_to_action_vec` (imported from
   `experiments/830_corridor/corridor.py`, `imprint.py:86-91`) emits action codes that
   reproduce the hold-shares book under ORDER semantics. Under WEIGHT semantics the same codes
   mean ±1-unit budget-scaled target-weight steps (`jax_trading_env.py:1613-1630`) — the
   replayed book, its net@21, AND the BC reward/value stream all change. Unverified teacher
   expressibility is the core risk: imprinting on trajectories that don't reproduce ~floor in
   the weight env is training on a broken teacher (H5's disease, re-created by plumbing).

## Scope — `experiments/832_linear_imprint/` only
1. **Thread the weight-env config** through the runner: accept the phase-1c weight flags
   (`--vector-weight-rebalance` + unit/max/deadband; the 834/836/844 `--config-json`
   TrainingConfig-overrides pattern is acceptable if cleaner) and apply them to BOTH in-env
   walks — teacher trajectory generation AND the sanity replay. The ExIt half needs no code
   (trainer flags exist); document the ExIt launch flags in the SPEC.
2. **Teacher-expressibility gate (pre-registered, the real deliverable):** replay the teacher
   `action_vec` through the WEIGHT env on the fixture, identical decision rows as the
   order-semantics replay; report BOTH net@21 values in the smoke JSON. Fixture gate =
   |teacher_weight − teacher_order| ≤ 10% relative. If FAIL: ship a weight-native mapping
   variant (per step per asset: action = sign(w_target − w_cur) while |Δw| > deadband — the
   natural ladder encoding) and re-gate. The binding ±0.10-absolute read (DEC-2 tolerance)
   happens on the REAL-panel scheduled run, planner-owned.
3. **BC targets from the weight env:** values = discounted MC returns over the teacher's
   WEIGHT-env rewards (same env the ExIt loop runs in); one-hot marginals unchanged.
4. Fixture smoke re-run weight-ON (CE decreasing, teacher agreement, sanity-replay plumbing) +
   selftest; zero diffs outside `experiments/832_linear_imprint/` (+ SPEC).

## Gates
Fixture smoke green (weight ON); expressibility gate reported (PASS, or weight-native mapping
shipped + PASS); zero canon diffs; medium panel (mapping fidelity is the leak-check focus —
830-audit pattern).

## Out of scope
The real imprint-sanity verdict, the imprint run, and the ExIt runs (scheduled planner runs
per DEC-2, ε=+0.10 + per-window consistency leg); any env change (that is TASK_843);
H6 widening (named contingency, activates only if the IMPRINTED policy can't hold the floor).

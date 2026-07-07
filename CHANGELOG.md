# Changelog

The dark-factory release ritual appends one entry per shipped task (newest first).
Format: `## YYYY-MM-DD — <task> — <one-line what shipped>` + remainders if any.

## 2026-07-08 — TASK_830 — E0 instrument corridor (factory half): linear-book floor + oracle ceiling through the vector env
- `experiments/830_corridor/`: `SPEC.md` (pre-registered ±0.10 Sharpe tolerance + attribution note), `corridor.py` runner (replays the 819 linear book as `action_vec`s through `step_vector` for the floor; per-asset `path_solver` oracle for the ceiling — instrument only, H5 respected), `results_fixture_smoke.json`.
- `experiments/LEDGER.md`: E0 fixture-smoke row, real-panel verdict PENDING the scheduled CPU run.
- Fixture smoke: floor +0.42 net@21 / +0.75 gross, oracle ceiling ~+30, vs 819 reference +0.75 net@21. Floor does NOT reproduce within tolerance — **panel-verified as a REPLAY-MAPPING artifact, NOT a `step_vector` defect** (independent cash-account sim reproduces the env to 3–4 sig figs; dominant term = fixed-weight vs renormalized `1/n_present`; BEAR22 = faithful hold-shares de-leveraging; env adds +0.013 over the pure mapping). Runner + LEDGER reframed off the wrong "filed vs Stage 1" attribution.
- Seeded DECISIONS **D6**: which book defines the official corridor floor (hold-shares vs constant-dollar-weight daily-rebalance) — flags that the discrete action space may make exact +0.97 reproduction inexpressible in-env.
- Verified by medium panel (correctness/regression + flow-verifier): runner correct, zero diffs outside `experiments/`, env byte-identical, numbers reproduce byte-identical.
- Remainder: real-panel corridor numbers = scheduled CPU run (24GB db read-only), NOT a factory-loop job.

## 2026-07-08 — TASK_828 Stage 1 — vector env step (`step_vector`), flag-gated, canon bit-identical
- `mcts_trader/env/jax_trading_env.py`: new `step_vector(state, action_vec, prices, config)` — per-asset action vector `(n_assets,)` ∈ {0..4} executed as one simultaneous portfolio rebalance; vectorized fill/fee/margin/PnL mirroring the scalar `step`/`execute_*`; base log-return reward + optional `reward_attributable` realized-PnL sum; `batch_step_vector`/`jit_step_vector`/`jit_batch_step_vector` wrappers.
- Gated by new `EnvConfig.vector_action: bool = False` (appended last; default OFF). Scalar canon path proven bit-identical — scalar-rollout SHA-256 identical HEAD vs working-tree.
- `tools/tests/test_vector_env.py`: 13 tests (equivalence-vs-scalar-sequence, intended simultaneous-rebalance divergence, cash/margin conservation on synthetic + fixture prices, JIT/batch, `reward_attributable` sign/entry/sum equivalence, forced-liquidation sentinel suppression). Default-pytest import-mode robust.
- Verified by full high-tier panel (correctness/fail-closed/regression/security judges + flow-verifier): no BLOCKER/MAJOR; the one MINOR coverage gap fixed and teeth-verified.
- Remainders: Stage 2 (Sampled-MuZero search) not started — needs this banner; `--vector-action` not yet threaded through training/search/env_factory (Stage 2/3). Pre-existing stale `test_jax_env.py` `order_reserved>0` assertions (2) left untouched — planner cleanup candidate.

## 2026-07-07 — factory setup — dark-factory config, QUEUE/DECISIONS surfaces, fixture, repo prune
- `.dark-factory/config.json` (invariants inlined), `tasks/QUEUE.md` (828 stages ranked), `tasks/DECISIONS.md` (D1–D5 open).
- `fixtures/make_fixture.py` → 51MB instant-load fixture (full daily panel + 90d minute slice, top-8 coverage).
- Repo pruned to the reasoning-trading-system core (manifest in `lore/prune_2026-07-07.md`).

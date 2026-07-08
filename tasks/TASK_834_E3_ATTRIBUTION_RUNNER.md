> **STATUS (2026-07-08): SHIPPED as `task834-e3-runner`.**
> `experiments/834_e3_probe/` — checkpoint-loading DEC-1 runner (R_model vs R_true, calibrated
> verdict + flagged fallback). Fixture smoke green (post-fix: R_model 0.928 / R_true 0.503, 0
> illegal probe vectors, cap binds); selftest 7/7; zero canon diffs. Medium panel found 2 MAJORs
> (legality mask omitted from proposal sampling; silent fresh-init measurement — crossval reproduced
> a FALSE DEC-1 PASS on a missing snapshot path) + 2 MINORs — ALL fixed + adversarially re-verified
> (fail-loud guards attacked 3 ways, all refuse). **Remainder (scheduled/planner):** apply the DEC-1
> threshold to real battery snapshots (pass the run's `--config-json`; hold-bias/dither must match
> the operator actually launched — the 3-arm diagnostic may change it).

# TASK_834 — E3/DEC-1 attribution runner (checkpoint-loading R_model vs R_true)

**Status**: READY 2026-07-08 (queue row 1 — the running Stage-3 battery needs it within hours,
when mid/late snapshots exist). **Size: S. Risk: low-medium** (new small module in
`experiments/`; zero canon diffs — 830/831 precedent).

## Why
The shipped E3 probe (`tools/tests/test_vector_reward_probe.py`) is an untrained-model,
in-process harness: one fixed latent, sensitivity-vs-numerical-noise (its own docstring says
so). The REAL DEC-1 gate — **`R_model ≥ 0.5 × R_true`** with
`R = E_book[Var_action(reward)] / Var_book[E_action(reward)]` — needs a runner that works on
SAVED training checkpoints and computes the true-env leg. Neither exists (verified 2026-07-08:
no checkpoint path in the probe; no R_true code anywhere — grep hits docs only).

## Scope — `experiments/834_e3_probe/probe_runner.py` + SPEC + fixture smoke
1. **Checkpoint load**: build the model with the RUN'S TrainingConfig (not the probe's
   hard-coded small/64-asset shapes — shape mismatch fails restore), then
   `trainer.load_checkpoint(abspath)` (`mcts_trader/jax_agent/trainer.py:1700`; snapshots at
   `jax_ckpt_fast/<name>/snapshot_ep<N>/`).
2. **Distribution sampling**: N book/states drawn from the run's state distribution (walk the
   training env or sample the replay pattern — pre-register which in SPEC), M action vectors
   per state from the dithered-book proposal (hold-bias 3.0).
3. **R_model**: decoded reward via `recurrent_inference_vector` (`muzero_model.py:814`,
   dispatch `:1292`), variance decomposition over (state, action) sample →
   `E_book[Var_action] / Var_book[E_action]`.
4. **R_true**: SAME states/actions through the true env reward —
   `batch_step_vector_unique` / `jit_batch_step_vector_unique`
   (`mcts_trader/env/jax_trading_env.py:1940/:2015`). Same decomposition.
5. **Output JSON**: `R_model`, `R_true`, ratio, N/M, seed, checkpoint path, plus the Stage-B
   ~1/60 baseline line for distance-travelled. Verdict text applies DEC-1 (calibrated primary,
   absolute-0.5 fallback flagged if `R_true` unavailable).
6. **Fixture smoke**: untrained fresh-init model on the fixture db — must run end-to-end and
   show R_model ≈ noise-level vs a trained head's expected movement (no threshold applied to
   the untrained case; the smoke proves plumbing, not the verdict).

## Gates
Fixture smoke green on CPU (`JAX_PLATFORMS=cpu`); zero diffs outside `experiments/834_e3_probe/`;
selftest 7/7; the DEC-1 reading procedure quoted verbatim in the SPEC.

## Out of scope
Applying the threshold to the real run (planner/scheduled), touching the shipped probe test,
any trainer/model edits.

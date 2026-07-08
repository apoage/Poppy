# E3 — DEC-1 attribution runner (TASK_834, gates the 828 Stage-3 battery snapshots)

**Hypothesis**: none of its own — this is a **measurement instrument** (830 precedent). It
computes the DEC-1 gate quantity on SAVED training checkpoints so the planner can decide,
per mid/late snapshot, whether the Stage-3 search plans over a reward model that feels its
own actions (the D2 question, made empirical).

## DEC-1 — quoted VERBATIM from `tasks/DECISIONS.md`

> | DEC-1 | E3 acceptance threshold: how action-sensitive must the reward head be? (`R = E_book[Var_action(reward)] / Var_book[E_action(reward)]`) | **CALIBRATED threshold (planner ruling, pre-registered before any training run; operator may override): `R_model ≥ 0.5 × R_true`**, where `R_true` = the SAME attribution ratio computed with the TRUE env reward on the same state/action distribution (the probe machinery already does true-env evals — 831 pattern). Rationale: an ABSOLUTE `R ≥ 0.5` silently assumes the real reward's action-sensitivity is ~its state-sensitivity; if the env's own `R_true` is, say, 0.3, an absolute 0.5 would demand the model OVERWEIGHT actions vs reality. Calibrating to `R_true` makes the gate self-honest. Report alongside: absolute `R_model`, `R_true`, and the Stage-B ~1/60 baseline as distance-travelled. Fallback if `R_true` is unavailable at run time: the absolute `R ≥ 0.5` as recommended. Does not block the Stage-3 CODE build (harness ships regardless; `R_true` is an env computation, addable at run time). | 2026-07-08 |

The runner applies this rule verbatim: **primary** `R_model >= 0.5 × R_true`; **fallback**
(only when the `R_true` leg is unavailable at run time, e.g. an env error or `--skip-true`)
absolute `R_model >= 0.5`, and the output JSON carries `dec1.fallback_used = true` — the
fallback is always FLAGGED, never silent. Reported alongside in every JSON: absolute
`R_model`, `R_true`, their ratio, and `stage_b_baseline_R = 1/60 ≈ 0.0167` (the Stage-B
reward head's measured ~60× action-blind variance gap) as the distance-travelled reference.

## The measurement

For N book/states and M action-vectors per state (same states, same actions for BOTH legs):

- **R_model** — `initial_inference(obs)` → latent; `recurrent_inference_vector(latent,
  action_vec)` (`muzero_model.py:814`, dispatch `:1292`) → reward logits → expected value
  over the 401-bin support (`SampledMuZeroVectorSearch._reward_to_scalar`, the canon search
  decode).
- **R_true** — the identical (state, action-vector) pairs stepped through the true env
  reward via `jit_batch_step_vector_unique` (`mcts_trader/env/jax_trading_env.py:2015`),
  chunked over states (`--chunk-states`, default 8) with per-chunk broadcast prices to
  bound the tiled-price footprint.
- **Decomposition** (identical for both legs): per state, population variance (ddof=0) of
  the M rewards → mean over N = `E_book[Var_action]`; per state, mean of the M rewards →
  population variance over N = `Var_book[E_action]`; `R = num / max(denom, 1e-10)`.
  Small-sample bias is identical across the two legs, so it cancels in the DEC-1
  comparison. R is invariant to affine reward transforms, so trainer-side scaling of the
  reward TARGET (`--reward-hybrid` PopArt normalization) does not distort the calibrated
  comparison; both legs are decoded/measured in the env's symlog reward units regardless.

## Pre-registered choices

1. **Config provision** — the run's `TrainingConfig` is supplied as `--config-json`, a JSON
   object of TrainingConfig field overrides (e.g. transcribed from the run's
   `jobs.manifest` CLI line). Known dataclass fields go through the constructor; unknown
   keys are attached via `setattr`, mirroring `mcts_trader/training/cli.py`'s dynamic attrs
   (`reward_hybrid`, `use_jax_env`, …) and logged. `--n-assets` (default 64) sets the model
   shape from args — nothing is hard-coded. Before restore, the snapshot's
   `trainer_metadata.json` (persisted by `trainer.save_checkpoint`, `trainer.py:1656`) is
   cross-checked against the built model (state_dim / h_cycles / l_cycles /
   use_prediction_head) and the runner fails loudly on mismatch. Restore =
   `trainer.load_checkpoint(abspath)` (`trainer.py:1700`), snapshots at
   `jax_ckpt_fast/<name>/snapshot_ep<N>/`. **Anti-silent-no-op guards** (flax
   `restore_checkpoint(step=None)` returns the target UNCHANGED on a missing/empty dir —
   a bad path would otherwise probe the fresh init and emit a false DEC-1 verdict): the
   runner hard-fails (SystemExit) if `trainer_metadata.json` is absent, if the restored
   `step_count <= 0`, or if the restored params are identical to the pre-restore fresh
   init; the restored `step_count` is recorded in the output JSON.
2. **State distribution** — N states are collected by walking the run's OWN environment:
   `create_jax_env_historical` (`env_factory.py:289`) with the exact
   `train_jax_fast.py:240` argument mapping (same DB loader, `--periods-file` pinned
   universe + order — or the `assets_file` branch, `train_jax_fast.py:233-236` —
   `bar_minutes` resample, fee/slip, burn flags, `reward_attributable`,
   `vector_action=True`), reset with `PRNGKey(--seed)`, then `n_envs` parallel walks
   (default = config `parallel_envs`) stepped with actions drawn from the **zero-logit
   dithered-book proposal** (canon `_build_proposal` on zeros; HOLD-book v1, per the task's
   "simplest defensible v1"), **legality-masked per state exactly as battery self-play**:
   `jit_batch_compute_action_mask` with the run's `max_positions`/`min_cash_threshold`
   (`self_play.py:661-674`; threaded from config as in `train_jax_fast.py:486`). The env
   does NOT enforce `max_positions` — the mask is the only order-cap constraint — so an
   unmasked walk leaves the training-reachable distribution; the DEC-1 gate must be read
   ON that distribution. After `--burn-in` steps (default 8; positions accumulate), one
   pre-step (decision-point) state + its `compute_observation_27ch` observation + its
   legality mask are collected per env per step until N states exist. **Post-done
   exclusion**: states at/after an env's done flag are excluded (post-done states are
   degenerate); the exclusion count is recorded in the JSON and the runner fails loud if
   fewer than half the requested N survive. The mask is computed ONCE per vector step
   (self-play semantics), so several simultaneous opens can overshoot `max_positions` in
   the crossing step — training-faithful, bounded by `(max_positions − 1) + n_assets`.
   Choosing the zero-logit (not policy-logit) proposal for the WALK makes the state
   sequence checkpoint-independent: every snapshot of a run is probed on the IDENTICAL
   seed-fixed states, so the E3 trend across snapshots measures the reward head, not
   proposal drift.
3. **Action distribution** — M action-vectors per state (default 16) from the canon
   proposal builder `SampledMuZeroVectorSearch._build_proposal`
   (`sampled_search.py:171`; instantiated with `model_apply_fn=None`, the 831 pattern —
   sampling machinery only, no search): one dither draw per state, then M i.i.d. per-asset
   categorical draws, **with the same per-state legality mask passed as `action_mask`**
   (illegal slots → −1e9, `sampled_search.py:209-211`) — exactly the canon root sampling
   (`_search` step 2) including its mask leg, so the probe's action distribution is the
   training-reachable one. The JSON records mask diagnostics: fraction of illegal slots on
   the collected states, fraction of probe vectors containing a training-illegal action
   (asserted exactly 0 in the smoke), and per-state order counts vs the cap. Hold-bias /
   dither / scheme come from the run config (`vector_hold_bias` / `vector_dither_scale` /
   `vector_proposal_scheme`); the 828 Stage-3 battery operator is **hold-bias 3.0, dither
   1.0, dithered-book** (`experiments/828_stage3_train/SPEC.md`; the CLI default 0.0 is a
   documented trap — the runner warns if the config says 0.0). Proposal BASE logits
   default to the checkpoint's factored policy at each state (`--proposal-base policy`,
   battery-faithful: training's proposal is policy+bias+dither); `--proposal-base zero` is
   the checkpoint-independent control.
4. **Both legs paired by construction** — the env state carries its own `rng_key`, so all
   M true-env variants of a state see identical stochasticity (limit fills, burns); the
   action-independent step preamble is shared. Nothing about the pairing is left to chance.
5. **Fixture smoke** (`--fixture-smoke`) — fresh-init UNTRAINED model on
   `fixtures/trading_fixture.db` (pre-registered: the fixture db, not a synthetic env),
   `bar_minutes=60` (the fixture's 1-min `ohlcv` table spans only ~90 days — 1440-min bars
   would leave too few rows for window 60 + episode; hourly bars keep the same resample
   code path), `n_assets=16` (fixture has 8 minute-table assets; the factory's cyclic pad
   fills to 16), model small, fee16/slip5, `reward_attributable=true`, hold-bias 3.0,
   dither 1.0, N=16, M=8. It exercises the REAL save→load path: the fresh state is written
   with `trainer.save_checkpoint` to a temp dir, reloaded by a SECOND (differently
   initialized) trainer via `trainer.load_checkpoint`, and the param round-trip is
   asserted both ways (restored == saved AND restored != the second fresh init). Smoke
   asserts: finite `R_model` and `R_true` components, complete JSON schema,
   `dec1.threshold_applied == false` (plumbing proof, NOT a verdict on an untrained
   model), zero training-illegal probe actions, and the masked order-cap bound. Writes
   `results_fixture_smoke.json`.

## Semantics / caveats (read before applying the verdict)

- The state walk uses the run's own env-creation path, but its proposal-driven trajectory
  is an approximation of the search-driven training trajectory (the walk samples ONE
  proposal action per step instead of running the full Gumbel search — pre-registered v1
  simplification; the proposal is the distribution the search's candidates are drawn from).
- `R_true` depends on the action/state distribution; that is WHY DEC-1 calibrates to it —
  both legs always use the same draw, so the comparison is self-honest under any proposal.
- The verdict on real snapshots belongs to the SCHEDULED probe run (planner-owned); the
  factory deliverable is this runner + the fixture smoke. No threshold is ever applied to
  the untrained smoke case.

## Compute / seeds / kill

- **Compute**: CPU-only (`JAX_PLATFORMS=cpu`) everywhere; the smoke is seconds–minutes.
  The real snapshot probe is a scheduled CPU one-shot per snapshot (small model, N=64,
  M=16 ⇒ ~1k recurrent inferences + ~1k env steps — no GPU needed, invariant #7).
- **Seeds**: `--seed` (default 0) fixes reset, walk, dither, and sampling. Multi-seed
  probing = rerun with different `--seed`; per-snapshot trend claims should use ≥3 seeds.
- **Kill date**: 2026-07-22 (with the 828 battery review). If the runner cannot produce
  finite R legs on a real snapshot by then, E3 is re-scoped.

## Deliverables

- `probe_runner.py` — the runner (this dir). Imports canon machinery only
  (`setup_model_and_trainer`, `create_jax_env_historical`, `batch_step_vector` /
  `jit_batch_step_vector_unique`, `compute_observation_27ch`,
  `SampledMuZeroVectorSearch` proposal builder + reward decode); zero canon diffs.
- Fixture smoke: `JAX_PLATFORMS=cpu python experiments/834_e3_probe/probe_runner.py
  --fixture-smoke` → writes `results_fixture_smoke.json`.
- Output JSON schema: `R_model` / `R_true` legs (each with `E_book[Var_action]`,
  `Var_book[E_action]`, `R`), `ratio_model_over_true`, `dec1` (rule + verdict text +
  `passed` + `fallback_used` + `threshold_applied`), N (surviving + collected + excluded
  counts), M, seed, checkpoint path, `restored_step_count`, sampling block (method +
  operator knobs + mask diagnostics + pct-hold), config block, `stage_b_baseline_R`,
  jax version, timestamp.

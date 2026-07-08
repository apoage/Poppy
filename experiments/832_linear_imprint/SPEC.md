# SPEC — TASK_832 / E5 linear-book imprint runner (pathway R2, FACTORY half)

**Pre-registered 2026-07-08, before any run.** Owner of the real verdict: the scheduled
<=1-GPU-day run (DEC-2). This factory half ships the runner + fixture smoke ONLY and is
**epsilon-independent**: the DEC-2 eps = +0.10 net@21 tolerance is a RUN-TIME interpretation
parameter and is NEVER applied by the smoke.

Spec sources: `tasks/TASK_832_LINEAR_IMPRINT_EXIT.md`; the CONFIRMED scope note in
`tasks/QUEUE.md` row 3 (planner 2026-07-08 — pins the shape, overrides older readings);
`tasks/DECISIONS.md` DEC-2 (ratified eps/methodology).

## Mechanism (pinned): BUFFER-INJECTION

1. **Teacher generation (on-the-fly, no npz/pkl):** the 819 linear trend book is walked
   through the vector env (`step_vector`) on the TRAINING panel; per-decision
   `action_vec` comes from the TASK_830 mapping (`corridor.py::_to_action_vec`,
   imported by path — not reinvented), book signs from
   `corridor.py::book_target_signs` = `trader.strategy.positions_from_config` on the
   SAME panel's log-closes.
2. **One-hot marginals:** `marginals[t,i,a] = 1{a == teacher_action[t,i]}` (float32,
   `(T, n_assets, 5)`).
3. **Injection:** `JAXReplayBuffer(vector_action=True).add_trajectory(obs, action_vecs,
   rewards, values, marginals, dones)` (replay_buffer.py:99; vector routing
   replay_buffer.py:146-165), chunked at <= `--chunk-len` (default 400, canon episode
   scale; tail chunk merged if < sequence_length).
4. **Imprint = `trainer.train`** on the injected buffer: the shipped factored CE
   (mean-over-assets, `trainer.py:408 _vector_policy_ce`, consumed at trainer.py:914-922
   and 996-1005) **IS the BC loss** — trainer metric `policy_loss` is the imprint CE.

No trainer/env/search/model edits. `amax_imprint` / `expert_bc` (scalar-only by design)
are NOT extended, NOT enabled (`amax_method=None`, `ExpertBCConfig` default disabled).

## Pre-registered choices

### (a) Imprint-phase entropy knobs (CLI args; trainer DEFAULTS untouched)
- `--entropy-coef 0.001` (canon 0.01, cli.py:130). One-hot targets drive entropy to 0;
  the canon coefficient actively fights BC. 10x reduction keeps a token anti-degeneracy
  pull without competing with the CE.
- `--min-entropy 0.0` (canon 3.0, cli.py:295). The adaptive 5x push
  (trainer.py:1075-1079) fires only when entropy < eff_min (vector-rescaled,
  trainer.py:1069-1074); 0.0 makes it unreachable => the push is exactly OFF for the
  imprint phase, via a constructor argument, not a default change.
- `--ckpt-entropy-coef 0.01` (canon cli.py:130): the entropy_coef HANDED TO THE EXIT
  HALF via checkpoint metadata — see the restore side channel under PIN (c). The
  imprint knobs above never leak past the imprint phase.

### (b) Panel consistency (load-bearing)
- Panel = `create_jax_env_historical` (env_factory.py:289): 1-min `ohlcv` closes,
  coverage-ranked universe, ffill, row-resample `prices[::bar_minutes]`
  (env_factory.py:367-369). The REAL run uses the pinned `--bar-minutes 1440` daily
  resample; the fixture smoke uses 240 (below).
- Book signs: `logc = log(max(panel_prices, 1e-12))` on THAT panel ->
  `book_target_signs(logc)` (819 rule: sign(logC_t - logC_{t-30}), equal-weight,
  vol-window 30, warm = 31). NOT the 830 corridor's daily-close/liquidity-filtered
  panel — an obs-distribution mismatch would evaporate the warm start.
- Observations: the exact self-play rule (self_play.py:646-658) —
  `window_start = episode_start_idx + step - window_size` (window_size = 60 from the
  TRAINING EnvConfig), encoded with `jit_batch_encode_simple` on the SAME panel prices.
  Obs depend only on (prices, window_start) => computed in blocks after the walk.
- First decision row `start_row = max(warm, window_size)`; decisions
  `k = 0..N-1` act at panel row `start_row + k` with target sign `tgt[start_row + k]`.
- Teacher value targets: discounted MC returns over the teacher's own env rewards
  (`compute_returns(use_gae=False)`, returns.py:179), bootstrap 0 (teacher has no value
  net), discount 0.997. PopArt (ON, canon) normalizes at train time.
- Imprint trains on the COST-HONEST walk (fee 16 / slip 5 = net@21, the 828 Stage-3
  primary operator); the gross tier is replayed only for reporting.
- Trainer knobs mirrored from canon where they matter: lr 1e-4 (cli.py:121),
  discount 0.997, PopArt ON (cli.py:140-141), value_loss_weight 2.0 (setup.py:530),
  reward_scale 100, `--reward-hybrid` + `--reward-attributable` default ON (Stage-3
  primary battery, experiments/828_stage3_train/SPEC.md) so imprint reward/value units
  match the ExIt continuation. Small model: state_dim 64, H4/L6 (config.py:135-137).

### (c) ExIt half = already-shipped flags (VERIFIED anchors, no new flags)
- `--load-checkpoint` mcts_trader/training/cli.py:250
- `--reset-optimizer` mcts_trader/training/cli.py:252
- `--clear-buffer` mcts_trader/training/cli.py:254
- Consumed: train_jax_fast.py:443-462 (load + `trainer.reset_optimizer_state`,
  trainer.py:1781), train_jax_fast.py:465-467 (`replay_buffer.clear`,
  replay_buffer.py:423). Checkpoint I/O: trainer.py:1656 (save) / 1700 (load).
- The runner saves the imprinted checkpoint via `trainer.save_checkpoint` (flax
  `checkpoints.save_checkpoint` + trainer_metadata.json + popart_state.npz) — directly
  consumable by `train_jax_fast.py --vector-action --load-checkpoint <dir>
  --reset-optimizer --clear-buffer` provided the arch flags match (small / state_dim 64
  / H4 L6 / no prediction head / --vector-action). The ExIt run itself is the
  SCHEDULED half — not launched by this runner.
- **entropy_coef restore side channel (MAJOR fix, verify panel 2026-07-08):**
  `save_checkpoint` writes `trainer.entropy_coef` into trainer_metadata.json
  (trainer.py:1673-1674) and `load_checkpoint` restores it UNCONDITIONALLY after
  trainer construction (trainer.py:1719-1720) — so whatever the metadata carries
  OVERRIDES the ExIt run's constructed value, including an explicit
  `--entropy-coef` on train_jax_fast.py. Left unfixed, the whole ExIt improvement
  phase would silently run at the imprint's reduced 0.001 (10x weaker regularization
  AND 10x weaker adaptive anti-collapse push) exactly at the collapse-prone
  post-imprint moment (entropy ~0.3), contaminating the DEC-2/R2 verdict with a
  non-canon operator. Fix: the runner sets `trainer.entropy_coef` to
  `--ckpt-entropy-coef` (default **0.01** = canon, cli.py:130) immediately BEFORE
  `save_checkpoint` (training is already complete) and logs the handoff loudly
  (`[832] ENTROPY HANDOFF: ...`). **Operator instruction:** at ExIt launch, verify
  the `Restored entropy_coef: 0.01` line in the train_jax_fast.py log.

### Imprint-sanity replay (pre-registered operator)
- **Direct policy-argmax rollout** (cheaper than visit-argmax; acceptable for the
  sanity check per the scope note): `initial_inference` -> `policy_logits` reshaped
  `(A, 5)` -> per-asset argmax -> `step_vector`, deterministic, on the IDENTICAL
  decision rows/panel as the teacher. Reported at gross and net@21 with the teacher
  floor and `corridor_position = policy − floor`, plus per-asset action agreement.
- Sharpe = audited `backtest._sharpe(rets, period)` with `period = bar_minutes`
  (annualization sqrt(525600/period); = the canon K=1440 daily convention on the real
  run).

## Fixture smoke (factory gate — the ONLY run the factory performs)

`fixtures/trading_fixture.db` minute slice = 8 assets, 2025-09-13 .. 2025-12-12,
~27.2k distinct minute rows => `--bar-minutes 1440` would leave ~19 bars < warm(31)+window(60).
Pre-registered smoke sizing (SAME code path, denser resample):

- window `2025-09-14 .. 2025-12-12`, `--bar-minutes 240`, `--n-assets 8`
  (~114 panel bars -> start_row 60, N ~ 53 decisions), `--iters 80`, `--batch-size 32`,
  seed 0, CPU (`JAX_PLATFORMS=cpu` default in-module).
- Command: `python experiments/832_linear_imprint/imprint.py`
  (fixture defaults auto-apply; writes `results_fixture_smoke.json`, checkpoint to
  gitignored `ckpt_smoke/`).
- PASS = end-to-end run with (1) all imprint losses finite, (2) imprint policy CE
  DECREASES on the teacher targets (head-5 mean -> tail-5 mean), (3) the sanity replay
  runs and emits the schema. **No performance threshold** — smoke numbers are signal
  of learning machinery only, never a corridor verdict.

## Output JSON (schema)

`imprint` (mechanism, full config, n_injected_steps, chunks, per-loss summaries incl.
CE trajectory, finite/decrease booleans), `teacher` (in-env floor gross/net@21,
action histogram, value-target stats), `sanity` (method, per-tier policy/floor/
corridor_position + diagnostics, action agreement), `ledger_row` (imprint corridor
position net@21 per DEC-2 shape; post-ExIt row = the scheduled run), `exit_half`
(verified flag anchors), `checkpoint`.

## Scheduled real run (NOT this factory half — recorded for the operator)

`--db trading_data.db --start 2021-01-01 --end 2025-09-01 --bar-minutes 1440
--n-assets 64 --iters 2000 --batch-size 256`, `JAX_PLATFORMS=cuda`, single verified-idle
GPU, tmux, then the ExIt continuation via the PIN-(c) flags. Verdicts (DEC-2 primary +
consistency legs, ±0.10 imprint-sanity tolerance) belong to that run and `experiments/
LEDGER.md` — not to the fixture smoke.

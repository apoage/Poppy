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
  (fixture defaults auto-apply; writes the **gitignored** `results_fixture_smoke_local.json`,
  checkpoint to gitignored `ckpt_smoke/`). **TASK_864**: the default invocation never writes the
  TRACKED `results_fixture_smoke.json` — that requires an explicit `--write-parity-baseline`
  (see "Committed fixture artifact" below), so no verification run can dirty the baseline.
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

## TASK_845 — weight-env compat (D13 redirect; added 2026-07-16)

The runner shipped (2026-07-08) BEFORE the TASK_840/843 target-weight rebalance env, so it
was weight-blind. TASK_845 threads the phase-1c weight-env config through it and adds the
pre-registered teacher-expressibility gate. **Default OFF => byte-identical to the pre-840
order-semantics path** (verified: with the weight flags off the results JSON's core fields
are identical to the pre-845 baseline; the only additions are the always-emitted
`expressibility_gate` block and the four `vector_weight_*` config echoes). ZERO canon diffs.

### Weight-env flags (mirror `cli.py:216-229` names/sentinels)
- `--vector-weight-rebalance` (store_true, default OFF): applies the TASK_840/843 target-weight
  rebalance `step_vector` to **BOTH** in-env walks — teacher trajectory generation (whose
  rewards become the injected BC value targets) AND the imprint-sanity deterministic replay.
  Threaded via `create_jax_env_historical` in `load_training_panel`, so `env_cfg` and every
  `env_cfg.replace(...)` tier config carry it.
- `--vector-weight-unit` (float, sentinel <=0 => 1/n_assets), `--vector-weight-max` (units of
  w_unit, default 4), `--vector-rebalance-deadband` (sentinel <0 => 0.1*w_unit).

### Teacher-expressibility gate (pre-registered, the real deliverable)
The order-code teacher (`corridor._to_action_vec`) reproduces the ~floor under ORDER
semantics; under WEIGHT semantics the same codes mean ±1-unit budget-scaled target-weight
steps, so the book, its net@21, and the injected BC reward/value stream all change. The gate
measures whether the teacher still expresses ~floor in the weight env:
- **Reference (fixed):** order-code teacher walked at net@21 through the ORDER env (weight OFF)
  = the 830/831 corridor floor (`order_floor_reference`).
- **Primary:** order-code teacher walked at net@21 through the WEIGHT env (weight ON), on the
  IDENTICAL decision rows. Fixture gate = `|teacher_weight - teacher_order| <= 0.10` relative.
- **Contingency (pre-registered, ACTIONABLE):** if the primary misses, re-gate the
  **weight-native ladder** mapping (`_make_weight_native_teacher_fn`: per step per asset
  `action = sign(w_target - w_cur)` while `|dw| > deadband`, driving to ±w_max in the 819
  target direction, `w_cur` reconstructed from the state as `jax_trading_env.py:1608-1612` —
  see the pre-preamble caveat in that function's comment) against the SAME order reference.
  Both mappings' `{order,weight}` net@21 + rel diff + PASS/FAIL are in the JSON under
  `expressibility_gate` (+ `expressibility_gate.weight_native_regate`). **The weight-native
  ladder is not diagnostic-only:** `--imprint-mapping {order-code,weight-native}` selects which
  teacher is actually IMPRINTED (default `order-code`; `weight-native` re-walks the same
  per-cost-tier configs with the ladder and rebinds the injected teacher / floor / agreement /
  values / ledger so the whole run is self-consistent, and records `imprint.imprint_mapping`).
  So if the real-panel gate shows order-code FAIL / weight-native PASS, the operator can imprint
  on the passing mapping — the contingency is wired, not just measured.
- **Fixture read (REPORTED, not asserted; PASS/FAIL INCONCLUSIVE):** the ~50-decision fixture
  slice is a losing period (order net@21 -3.44, weight net@21 -4.69 order-code / -4.59
  weight-native), so a ±10% *relative* threshold on negative Sharpes is ill-defined and its
  PASS/FAIL is inconclusive — consistent with this module's "no performance threshold on the
  fixture smoke" stance. BUT the ~1.25 net@21 gap is **NOT noise**: it is a real **structural
  sizing divergence**. At the default weight params (`w_unit = 1/n_assets`, `w_max = 4*w_unit
  = 0.5`) the weight book expresses ~2× (order-code) to ~8×-then-budget-clipped, Σ|w|≤1
  (weight-native) the order env's per-asset exposure — a real ~4% terminal-wealth gap that
  ±0.10-**ABSOLUTE** (the DEC-2 real-panel tolerance) would ALSO fail. The smoke asserts the
  gate is REPORTED (both values emitted) and that the weight-native variant is re-gated when the
  order-code mapping misses; the **binding read is the real-panel scheduled run** (planner-owned,
  DEC-2 ±0.10), which must confront the weight-param sizing (unit/max/deadband are exposed
  precisely so the planner can tune the expressible book). No mapping/threshold/fixture was
  tuned to force a pass.

### BC targets under weight ON
Injection is unchanged in mechanism; because the teacher walk now runs the WEIGHT env, the
injected `values = compute_returns(use_gae=False)` are discounted MC returns over the teacher's
WEIGHT-env rewards (the same env the ExIt loop runs in). One-hot marginals unchanged.

### Imprinted-mapping selector (`--imprint-mapping`, fail-closed coupling)
`weight-native` is only coherent when the WHOLE run is weight semantics (teacher walk, obs
encoding, and the ExIt continuation env). The runner therefore **asserts**
`--imprint-mapping weight-native` is passed together with `--vector-weight-rebalance`; the
contradictory combo (weight-native ladder walked through the ORDER env while the gate certifies
the WEIGHT-env teacher — two different teachers) is rejected fast with a clear message. Default
`order-code` is unconstrained (it is the pre-840 order path).

### Committed fixture artifact (regeneration — pinned)
`results_fixture_smoke.json` committed in this dir is the **weight-ON, default order-code**
deliverable. It is NOT reproduced by the bare flagless command (that writes the weight-OFF
numbers). Regenerate it with EXACTLY:
`python experiments/832_linear_imprint/imprint.py --vector-weight-rebalance --write-parity-baseline`
(fixture defaults + `--imprint-mapping order-code` default auto-apply). **Reproducibility
precondition:** `/db` and `/checkpoint` are recorded as ABSOLUTE paths, so the artifact reproduces
byte-identically only when the command is run from THIS tree root (`/home/lukas/Trader`) with the
default `--ckpt-out` — a different clone path or an explicit `--ckpt-out` changes those two fields
(and `--write-parity-baseline` refuses the latter outright, see below).

A weight-native (or any other off-recipe) run is **not** a `--write-parity-baseline` run at all: it
writes the gitignored `results_fixture_smoke_local.json` by default, or a scratch `--out` if you
want to keep it — the two flags are mutually exclusive, so "use a scratch `--out` to protect the
committed artifact" applies only to plain runs, never in combination with the promote flag.

**TASK_864** made that recipe enforceable rather than advisory. Every other invocation — including
the bare flagless one — writes the gitignored `results_fixture_smoke_local.json`, so the tracked
baseline can only move through `--write-parity-baseline`, which prints a per-field diff first and
REFUSES when:
- the run is **not the CANON RECIPE** — every argparse dest except `--out` and the promote flag
  itself is compared against the recipe above, with the expected values *re-derived from this
  module's own argparse defaults* (`CANON_BASELINE_ARGV` + `_resolve_run_defaults`, so the gate
  cannot drift from the defaults). Off-recipe `--imprint-mapping` / `--seed` / `--iters` /
  `--bar-minutes` / `--n-assets` / `--start` / `--end` / `--position-size-pct` / `--batch-size` /
  weight unit-max-deadband / `--tag` / `--ckpt-out` / lr / discount / entropy knobs all refuse,
  naming the offending field and its canon value. **The flagless weight-OFF default is refused by
  this check** (it previously promoted onto the weight-ON baseline, replacing 462/545 leaves at
  exit 0 with no downstream detector);
- `--no-smoke-assert` is given, or the finished run's own artifact fails re-validation
  (`imprint.all_losses_finite`, `imprint.policy_ce_decreased`, a full-length loss trajectory, both
  sanity-replay tier Sharpes, both expressibility-gate net@21 values) — checked INDEPENDENT of
  `--smoke-assert`, because `run()` writes the artifact BEFORE its smoke asserts, so an unchecked
  or truncated run could otherwise become the oracle;
- the tracked file is git-dirty, `--out` is also given, the db is not the fixture, or the run is a
  suffixed different experiment (861 `--teacher-vectors`, 863 `--panel-frame-from-checkpoint`).

All of those refuse fail-fast (before the ~1 min run) except the artifact re-validation, which
necessarily runs after it; the promoted bytes are re-serialised from the run's in-memory result
(never re-read from the shared local path) and installed with `os.replace` via `<baseline>.tmp`, so
the tracked oracle is never left truncated. Rationale for the whole apparatus: the flagless default
used to land on this tracked path, so a stray run left a dirty baseline that read like a legitimate
"refresh".

**Promoting recipe of the CURRENT baseline (the record of what produced these bytes).**
`python experiments/832_linear_imprint/imprint.py --vector-weight-rebalance --write-parity-baseline`
run from `/home/lukas/Trader` → `results_fixture_smoke.json` md5
`16736523175d25659405d568608a2a07` (555 → 545 leaves vs the TASK_845 bytes; 460 moved, 10 removed).
Resolved recipe fields: `vector_weight_rebalance True`, `imprint_mapping order-code`, `db
fixtures/trading_fixture.db`, window `2025-09-14 .. 2025-12-12`, `bar_minutes 240`, `n_assets 8`,
`position_size_pct 0.0625`, `iters 80`, `batch_size 32`, `seed 0`, `fee_bps 16.0`, `slip_bps 5.0`,
`vector_weight_unit -1.0`, `vector_weight_max 4.0`, `vector_rebalance_deadband -1.0`,
`ckpt_out <exp>/ckpt_smoke`, `smoke_assert True`. Every promote prints this recipe and the md5 of
the bytes it installs; **a refresh commit MUST update this block** — the absence of such a record
is why the run-20 clobber went unnoticed for ~80 commits.

**Baseline provenance / last refresh.** The artifact was regenerated at TASK_864 (2026-07-29). Its
values moved from the originally-committed (TASK_845) ones for exactly one reason:
`c2f1ba5`/**TASK_846** (weight-budget fee headroom) changed weight-env behavior at fee16, which is
the tier this smoke runs (`fee_bps 16.0`, `reward_tier_bp 21`). Verified by reproduction from
`c2f1ba5`'s parent (`e17dc2d`, reproduces the old bytes exactly) and from `c2f1ba5` itself
(reproduces the new bytes exactly), with every intervening commit to HEAD a no-op on this
artifact. Most visible consequence: `expressibility_gate.passed` flipped `false -> true`
(`rel_diff` 0.364 -> 0.095), so the pre-registered `weight_native_regate` contingency block is no
longer emitted — it is conditional on the primary gate MISSING (`imprint.py`, `if not
gate["passed"]`), not a schema removal.

### ExIt launch flags (weight-ON continuation — no runner code; trainer flags exist)
The ExIt half MUST run the SAME env semantics as the imprint. On `train_jax_fast.py`, in
addition to the PIN-(c) flags, pass the weight-env flags (`cli.py:216-229`, consumed at
`train_jax_fast.py:265-268`) with the SAME values used for the imprint run:
`--vector-action --load-checkpoint <dir> --reset-optimizer --clear-buffer
--vector-weight-rebalance [--vector-weight-unit U --vector-weight-max M
--vector-rebalance-deadband D]` (plus the matching arch flags). Mismatched weight flags =
the cross-env imprint the gate exists to prevent.

## Scheduled real run (NOT this factory half — recorded for the operator)

`--db trading_data.db --start 2021-01-01 --end 2025-09-01 --bar-minutes 1440
--n-assets 64 --iters 2000 --batch-size 256 --vector-weight-rebalance` (D13 phase-1c weight
env; add `--vector-weight-unit/-max/-rebalance-deadband` only to override the sentinels),
`JAX_PLATFORMS=cuda`, single verified-idle GPU, tmux, then the ExIt continuation via the
PIN-(c) flags **plus the SAME weight-env flags** (see "ExIt launch flags" above). The
real-panel run is also where the ±10%-relative expressibility gate gets its BINDING read
(the fixture read is REPORTED-only). Verdicts (DEC-2 primary + consistency legs, ±0.10
imprint-sanity tolerance) belong to that run and `experiments/LEDGER.md` — not to the
fixture smoke.

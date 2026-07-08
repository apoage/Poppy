# Stage-4 — faithful vector eval harness (TASK_835)

**Hypothesis**: none. This is an **instrument** (830 precedent): the ONLY evaluator that can
grade the running 828 Stage-3 battery. Vector mode fail-closed skips the legacy in-training
eval (shipped panel finding: it routed vectors through an `action_points`-capped env), so
Stage-4's pre-registered PASS (beat the engine ref, no HOLD collapse, BEAR-positive) has no
instrument until this one. Fidelity errors here are verdict errors — every design choice below
is a *mirror* of the training stack, pre-registered, and never tuned to the numbers.

## What the harness does

checkpoint → deterministic full-timeline rollout through the REAL vector env with the SHIPPED
Sampled-MuZero vector search → 830/831-convention corridor scorecard (JSON per checkpoint +
comparison table over a checkpoint list + one LEDGER-convention row per run).

## Fidelity contract (mirrors, not reimplementations)

| Piece | Source of truth mirrored | Where |
|---|---|---|
| Model+trainer build | `TrainingConfig` → `setup_model_and_trainer` → `init_train_state(dummy (1,27,A,A))` | `mcts_trader/training/setup.py:439`, `train_jax_fast.py:366,439` |
| Checkpoint load | `trainer.load_checkpoint(abspath)` then `trainer.state = loaded` (the TASK_834 pattern) | `mcts_trader/jax_agent/trainer.py:1700`, `train_jax_fast.py:443` |
| Panel | `create_jax_env_historical` with the run's periods-file/db/`--bar-minutes` — 1-min `ohlcv` table → `[::bar_minutes]` stride (the 832 panel-consistency rule) | `mcts_trader/training/env_factory.py:289` |
| Decision loop | `run_self_play_jax` vector branch: leak-free obs window `[t-window, t-1]` via `jit_batch_encode_simple` (27ch), `jit_batch_compute_action_mask` (max_positions 16 / min_cash 100), book = HOLD everywhere (DEC-3) | `mcts_trader/training/self_play.py:640-705` |
| Search | `create_sampled_vector_search` with the `train_jax_fast.py:375-384` parameters; **eval difference (Stage-2 carry-note): `deterministic=True` ⇒ selected action = `visit_probs` argmax** | `mcts_trader/jax_agent/sampled_search.py` |
| Env step | `jit_batch_step_vector` (the real `step_vector`) | `mcts_trader/env/jax_trading_env.py:1327` |
| Accounting | PV marked at the executed row (`current_step - 1`), per-decision returns, `trader.backtest._sharpe` annualization — the 830/831 convention | `experiments/830_corridor/corridor.py` |

`vector_gumbel_scale_eval` is reserved/unwired upstream — NOT touched, NOT wired here.
Run shapes (model size, h/l cycles, n_assets, K, sims, hold-bias, dither, scheme, discount,
gumbel scale) come from CLI args; defaults = the 828 Stage-3 battery manifest
(`experiments/828_stage3_train/jobs.manifest`), including **hold-bias 3.0 explicit** (the
TrainingConfig CLI default 0.0 is the documented trap).

## Pre-registered choices

1. **Cost-tier rollout convention** (mirrors 830/831 `GATE_COSTS`): gross (cost 0) and
   net@21 (fee16 + slip5 per side) are **separate rollouts** — fees change the trajectory
   through the env (execution prices, cash, legal mask, hence search decisions), so each tier
   is walked independently on the identical panel with the identical seed and scored on its
   own realized return series. The tier enters via `EnvConfig.replace(fee_bps, slippage_bps)`
   on the ONE panel built by `create_jax_env_historical` (prices identical by construction).
2. **Determinism**: single env; one fixed PRNGKey chain (`--eval-seed`, default 0) drives
   proposal sampling + search gumbel; `deterministic=True` (visit-count argmax; ties → first
   index); env RNG seeded from the same key and inert (probabilistic_fills False, burns off);
   no exploration schedule, no replay buffer. Same args ⇒ bit-identical output. The residual
   *seed sensitivity* (the K-candidate proposal is sampled — inherent to Sampled MuZero) is
   surfaced, not hidden: rerun with another `--eval-seed` to measure it.
3. **Full-timeline episode**: `episode_start_idx = window_size` (=60, first obs-valid row),
   `max_episode_steps = N+2`, N = T−1−window_size (capped by `--max-decisions` for smokes).
   Training episodes are 256-step random-start; Stage-4 evaluates the full timeline in one
   pass — that is the point of the instrument, not a fidelity break (env semantics identical).
4. **Window floors/ceilings are LOADED, never hardcoded**: the four 831-EXT regime windows
   come from `runs/831_ext/{bear22w,bull23w,y2024,recent}.json` (keys `window`, `floor`,
   `ceiling`; per-tier "0"/"21"); the full-window corridor comes from
   `experiments/830_corridor/results_real_panel.json` (`corridor["21"].floor_inenv` +0.424,
   `.reference` +0.751, `.ceiling` +29.698; gross tier from `corridor["0"]`). Pinned fallback
   constants with the same provenance (830/E0-REAL LEDGER row) apply only if a file is absent;
   the output records which source was used. Window slicing maps each window's date range onto
   decision rows via the reconstructed row-timestamp axis, using the loader's own date parse.
5. **Row-timestamp axis**: `create_jax_env_historical` discards timestamps, so the harness
   reconstructs them with the loader's own queries (read-only mirror of
   `VectorBTTradingEnv._load_database_prices`: same asset-id resolution preserving symbol
   order, same DISTINCT-timestamp union, same date parse, then the same `[::bar_minutes]`
   stride) and **asserts** `len(row_ts) == prices.shape[0]` — a mismatch aborts the run as a
   panel-consistency violation.
6. **Annualization**: per-decision Sharpe = `trader.backtest._sharpe(rets, period)` with
   `period = --bar-minutes` minutes/decision (=1440 canonical ⇒ identical to the 830/831
   corridor annualization). At smoke settings (`bar_minutes=60` on sparse fixture minutes) the
   period is nominal; the measured median row spacing is emitted as a diagnostic. Corridor
   comparisons are only meaningful at `bar_minutes=1440` on the real panel.
7. **Position sizing**: `position_size_pct` stays the training default 0.1
   (`create_jax_env_historical` default; the battery does not override it) — the agent is
   evaluated in the SAME env it was trained in. Caveat carried, not tuned: the 830 corridor
   floor was replayed at 0.5/A sizing; Sharpe was measured leverage-invariant on the corridor
   book (830 finding), but margin-call exposure differs — the scorecard's `bankrupt_at` and
   per-window floors make this visible.
8. **HOLD-collapse counters** (the 821 signature): pct_hold per asset + aggregate (% of
   (decision, asset) pairs with action==HOLD), avg long/short position counts per decision,
   full per-asset dict keyed by loaded symbols (padded columns marked `#pad<i>` when the
   universe is smaller than `--n-assets`, mirroring the factory's modulo padding). Collapse
   flag = aggregate pct_hold > 99.
9. **Verdict scope**: the harness emits numbers + a LEDGER-convention row; applying the
   Stage-4 PASS (beat ref / no collapse / BEAR-positive) is the planner's job (gate ladder).
10. **Per-window floor comparability caveat** (explicit — not an identical-book replay):
   the 831-EXT per-window floors/ceilings were measured on those windows' OWN panels
   (95–273 asset liquidity-filtered universes on the real db, warm-excluded decision rows,
   0.5/A replay sizing), while the agent's per-window Sharpe here is a **date-slice of the
   64-asset run-panel rollout** at `position_size_pct` 0.1. The per-window `above_floor`
   flags are therefore a calibrated yardstick, indicative only; the graded number is the
   full-window corridor position (same-panel, same-annualization). Never promote on a
   per-window flag alone.
11. **Rollout completeness guard**: each tier records `steps_completed` vs `steps_expected`.
   If ANY tier terminates early (env `done` — e.g. bankruptcy/liquidation), the scorecard is
   marked **PARTIAL ROLLOUT** (JSON `corridor.partial_rollout: true`, loud banner in the
   printed summary and in the LEDGER row) and the corridor PASS booleans
   (`beats_floor_net21`, `beats_reference_net21`, `position_in_corridor_net21`) are
   **withheld (null)** — a partial walk must never emit an unflagged PASS/FAIL. Raw Sharpe /
   PV / position stats stay visible for diagnosis.
12. **LEDGER pointer integrity** (the 734-819 stale-pointer audit hazard): the pointer field
   of the emitted LEDGER row is derived from the ACTUAL output path
   (`os.path.relpath(out_dir/out_name, repo)`), never a hardcoded directory — fixture runs
   point at this experiment dir, scheduled runs at `runs/835_stage4/<file>`.

## Pre-registered fixture smoke config

The fixture's `ohlcv` (1-min) table only covers **8 symbols × 2025-09-13..2025-12-12**
(XBT/USD, ETH/USD, LTC/USD, XMR/USD, ZEC/USD, ETC/USD, DASH/USD, REP/USD; ~27k union minute
rows), while the daily table is full-history — and `create_jax_env_historical` reads the
minute table. The smoke therefore sizes to the slice:

- db = `fixtures/trading_fixture.db`, symbols = the 8 above, window 2025-09-13 → 2025-12-12;
- `bar_minutes = 60` (exercises the `[::bar_minutes]` resample path; 447 rows > warm 60 + 40
  decisions — measured, the loader's local-midnight end filter trims the last partial day),
  `n_assets = 8`, `--max-decisions 40`, `eval_seed = 0`;
- operator = the battery operator verbatim (dithered-book, K=16, sims=32, hold-bias 3.0,
  dither 1.0, gumbel 3.0, small model);
- **real save→load roundtrip**: fresh-init trainer (seed 0) → `save_checkpoint` → second
  fresh-init trainer (seed 1) → `load_checkpoint` → params proven **different before** load
  and **identical after** (both asserted; smoke fails otherwise). The roundtrip checkpoint
  lives at `experiments/835_stage4_eval/smoke_ckpt/` (module-local, gitignored) so the
  committed smoke artifact carries stable provenance, not an ephemeral tmp path;
- PASS = runs clean end-to-end on CPU (`JAX_PLATFORMS=cpu`), emits the FULL scorecard schema
  (both tiers, all four 831-EXT window keys — no coverage on the fixture slice ⇒ null values,
  keys present), untrained numbers near-noise — **no threshold applied** (plumbing, not verdict).
- The four 831-EXT windows do not intersect the fixture minute slice ⇒ `per_window[*].sharpe`
  is null in the smoke by design; real coverage comes from the scheduled run on the 821
  train window (2021-01-01..2025-12-01), which contains all four.

## Compute / IO envelope

CPU-only default (`JAX_PLATFORMS` setdefault "cpu"); fixture db by default; never touches
`trading_data.db` unless the scheduled run points `--db` at it (read-only SELECTs + the
training factory's own loader). No GPU, no network, no writes outside the chosen out-dir
(fixture → this experiment dir; real → `runs/835_stage4`). Smoke ≈ minutes (JIT-dominated).
Real run: ~1670 decisions × 2 tiers per checkpoint, Python-loop over jitted steps — a
scheduled one-shot per battery snapshot batch (invariant #7), not a factory-loop job.

## Deliverables

- `eval_vector.py` — the runner (this dir). Zero canon diffs.
- `results_fixture_smoke.json` — smoke output (`python eval_vector.py --smoke`).
- Scheduled-run usage (battery snapshots × seeds; snapshots every 100 epochs):
  ```
  python experiments/835_stage4_eval/eval_vector.py \
      --db trading_data.db \
      --periods-file experiments/821_planner_daily/period_821_train.json \
      --checkpoint jax_ckpt_fast/828s3_vec_fee16_s0/snapshot_ep1000 \
      --checkpoint jax_ckpt_fast/828s3_vec_fee16_s0/snapshot_ep2000 \
      ...                    # or --checkpoint-list <file>
  ```
  (bar-minutes 1440 / 64 assets / battery operator are the defaults; one JSON per checkpoint
  + `comparison_<tag>.json` + a LEDGER row per checkpoint.)

## Kill / notes

If the panel-consistency assert (row-timestamp axis vs factory panel) ever fires, the harness
is out of sync with the training loader and MUST NOT be used for a verdict until re-mirrored.

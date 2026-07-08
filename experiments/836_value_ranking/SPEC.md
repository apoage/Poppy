# E4 — value-ranking probe (TASK_836; does the learned VALUE net rank states above momentum?)

**Hypothesis**: H1/H2 (`docs/pathway.md` R2 — E4 specced there, unbuilt until now). This is a
**measurement instrument** (830/834 precedent): E3 (834) gates the reward head, 835 grades full
rollouts; E4 isolates the VALUE net — the missing diagnostic between "value can't rank states"
and "search/policy can't exploit a good value". It runs on the phase-1 cost0 snapshots the
moment they exist.

## Interpretation — quoted VERBATIM from `tasks/TASK_836_E4_VALUE_RANKING_PROBE.md`

> PASS/interpretation (pre-registered): value beats momentum's correlation on ≥3/5 windows =
> the value net carries signal beyond the obvious; below-momentum everywhere = the value net
> is the bottleneck (feeds the phase-2 postmortem either way — this probe is DIAGNOSTIC, it
> gates nothing by itself).

The runner emits this text verbatim in every JSON (`interpretation.rule_verbatim`) and applies
it only in checkpoint mode — never to the untrained fixture smoke (834/835 precedent).

## The measurement

One DETERMINISTIC, checkpoint-independent env walk over the full panel timeline; per
decision-point state `t`:

- **Predicted value** — `initial_inference(obs)['value']` → the canon root decode
  `SampledMuZeroVectorSearch._categorical_to_scalar` (`sampled_search.py:245` — expected value
  over the support, × `value_scale` + `value_shift` from the run config; affine, so
  Spearman-invariant). Obs = the training rule: leak-free window `[t-window, t-1]` via
  `jit_batch_encode_simple` (27ch, `self_play.py:646-658` — the 832 PIN b / 835
  panel-consistency rule). NOTE: the 27ch obs is **price-only** — the value input is
  position-blind; the fairness consequences are pre-registered under "Momentum" below.
- **Realized forward return** over horizon `k` — `pv[t+k]/pv[t] − 1` along the SAME walk: the
  k-decision continuation under the same pre-registered deterministic rule, computed by the
  TRUE env (`jit_batch_step_vector`; PV marked pre-decision at each decision row via
  `compute_portfolio_value`). Realized returns are therefore **conditional on the walk rule**
  (zero-logit dithered-book proposal continuation), NOT on the trained policy — stated
  explicitly here and in every output JSON.
- **Metric** — Spearman rank correlation of (predicted value → realized k-return) per window
  per horizon, against the same correlation for two momentum rankings (below). Rank
  correlation is monotone-transform-invariant, so the value's symlog/support units never
  need calibrating against return units.

## Pre-registered choices

1. **Checkpoint load** — the 834 pattern REUSED, not re-derived: `probe_runner.py` is imported
   via `importlib` (the 832 cross-experiment precedent, `imprint.py:85-94`) and this probe
   calls its `build_training_config` (config via `--config-json`, TrainingConfig fields +
   setattr extras, `probe_runner.py:113-130`), `build_model` (`:163-172`), and
   `check_ckpt_metadata` (`:175-201` — fails loud if `trainer_metadata.json` is absent or
   mismatches the built model, BEFORE flax's silent no-op restore can bite). The inline
   restore guards are mirrored verbatim from `probe_runner.py:722-739`: SystemExit if restored
   `step_count <= 0` or if restored params are identical to the pre-restore fresh init;
   `restored_step_count` is recorded in the JSON.
2. **State sample** — a single-env (`n_envs=1`), fixed-start, full-timeline walk (the 835
   deterministic-rollout pattern: `episode_start_idx = warm = window_size`,
   `max_episode_steps = n_decisions + 2`), through the run's OWN env creation path
   (`create_jax_env_historical` with the 834 `build_env` argument mapping: periods-file /
   assets-file universe, `bar_minutes` resample, the run's fee/slip/burn/`reward_attributable`).
   **Walk actions = the zero-logit dithered-book proposal with a fixed `--seed`** (canon
   `_build_proposal` on zeros, hold-bias/dither/scheme from the run config, legality-masked per
   state exactly as battery self-play — the 834 precedent), so the walk is
   **checkpoint-independent**: every snapshot of a run is ranked on the IDENTICAL seed-fixed
   states. A HOLD-walk is pre-registered AGAINST: from the all-cash reset it never opens a
   position, PV is constant, and the realized-return leg degenerates to all-zeros.
   The first `--burn-in` (default 8) decisions are excluded from the ranked sample (books
   accumulate); a walk that hits env done/liquidation stops, and later states simply don't
   exist (post-done exclusion by construction). The panel row-timestamp axis is reconstructed
   with the 835 loader mirror (`load_row_timestamps`) and length-asserted against the panel.
3. **Windows** — full window + the four 831-EXT regime windows, whose definitions are LOADED
   from `runs/831_ext/{bear22w,bull23w,y2024,recent}.json` (key `window`; via 835's
   `load_ext_windows` — never hardcoded). Per-window samples are timestamp-filtered subsets of
   the ONE full-window walk (the EXT windows are sub-ranges of the full panel; 835
   `score_tier` convention). Missing window file → key emitted with null (835 precedent).
4. **Horizons** — `k=3` primary (the 831x/C value-horizon finding: steep gain to R=3, flat
   beyond), `k=1` and `k=6` secondary. Per horizon, a state enters the sample iff
   `burn_in <= t` and `t+k <= n_exec`.
5. **Momentum baseline** — trailing `m=21`-bar return; m=21 ≈ one month at the canonical
   daily bars — the standard time-series-momentum lookback, and `< warm = window_size = 60`
   so it is always computable without leaving the panel.
   - **PRIMARY ("book")**: current holdings revalued at rows `t−1` vs `t−1−m`
     (`compute_portfolio_value` at two price rows; positions/cash/entries frozen —
     **holdings-weighted trailing asset momentum, NOT the trailing walk PV**; past
     rebalances/fees do not enter this leg). All-cash states give exactly 0 (ties;
     average-rank handling).
   - **SECONDARY ("market")**: equal-weight cross-asset mean trailing return over the same
     rows — information-symmetric to the value net's price-only obs. Reported for the
     fairness read (the value net cannot see the book; if it loses to the book baseline but
     beats the market baseline, the gap is book-blindness, not value quality). The
     pre-registered interpretation counts the PRIMARY baseline only. Padded (cyclic-repeat)
     columns double-count in the equal-weight mean — diagnostic-only caveat.
   - **SECONDARY ("walk_pv")**: the true trailing-PV time-mirror of the realized leg —
     `pv_pre[t]/pv_pre[t−m] − 1` on the same walk PV series (rows ≤ `t`, the realized-leg
     mark convention; never beyond `t`). Requires `t >= m`, so earlier states are excluded
     from this leg only (its own `n` reflects the sub-sample). Unlike the book leg it DOES
     absorb past walk rebalances/fees. Diagnostic column; the pre-registered PRIMARY
     baseline and the interpretation rule are unchanged.
6. **Spearman source** — `scipy.stats.spearmanr` (scipy 1.15.3 verified present in
   `trader_env`; pre-registered). Average-rank tie handling. An internal rank-correlation
   fallback exists only for scipy-less hosts and self-identifies in
   `spearman_source`. Correlations are emitted as null (with a note) when `n < MIN_N = 8`,
   when an input series is constant, or when the result is non-finite — a degenerate window
   can never masquerade as signal. **Pre-registered p-value caveat**: p-values assume
   independent samples; consecutive k-step forward windows overlap by `k−1` steps
   (effective n ≈ n/k), making the reported p anti-conservative for k>1 — treat p as
   descriptive only; the interpretation rule uses ρ comparisons exclusively (echoed
   verbatim in the JSON `horizons.p_caveat` field).
7. **Leak protocol** — value obs rows `[t−window, t−1]` (`window_starts + window_size ==
   price_idx` by construction; first state's window starts at row 0); momentum rows
   `[t−1−m, t−1]` (never beyond `t−1`; `warm−1−m >= 0` hard-asserted); the realized leg reads
   rows `[t, t+k]` — FUTURE relative to the state, which is its purpose. All three stated in
   the JSON `leak_check` block.
8. **Fixture smoke** (`--fixture-smoke`) — fresh-init UNTRAINED model on
   `fixtures/trading_fixture.db`, config mirrored from 834's pre-registered smoke
   (`probe_runner.py:133-156`: `bar_minutes=60` for the fixture's ~90-day 1-min table,
   n_assets=16 over the fixture's 8 minute-table symbols via the factory's cyclic pad,
   model small, fee16/slip5, `reward_attributable=true`, hold-bias 3.0, dither 1.0), walk
   capped at 96 decisions. Exercises the REAL save→load path (834 pattern:
   `trainer.save_checkpoint` to a temp dir → `check_ckpt_metadata` → a SECOND
   differently-initialized trainer restores; params asserted equal-to-saved AND
   different-from-fresh). Smoke asserts: finite full-window correlations at k=1/3/6 for all
   four rankings (value + book/market/walk_pv momentum), EXT window keys present with
   n=0/null (the fixture slice
   2025-09-13..2025-12-12 does not intersect any 831-EXT window — 835 precedent), zero
   training-illegal walk actions, `interpretation.applied == false`. Correlations ≈ 0
   expected; NO threshold applied. Writes `results_fixture_smoke.json`.

## Semantics / caveats (read before applying the interpretation)

- Realized returns are conditional on the pre-registered walk rule (proposal continuation,
  the run's fee tier). The value net was trained to predict returns under its OWN improving
  policy — the probe measures RANKING transfer onto a fixed reference continuation, which is
  exactly what makes snapshots comparable; it is not an absolute value-accuracy claim.
- The value input is price-only (27ch); the PRIMARY momentum baseline sees the book. A
  below-book-momentum result therefore has two readings, which is why the SECONDARY
  market-momentum and walk_pv columns ship in the same JSON.
- **Degenerate-value disambiguation** (interpretation integrity): if NO window is evaluable
  at the primary horizon, the runner distinguishes WHY — windows that had enough samples
  but whose VALUE leg was nulled as `degenerate (constant series)` / non-finite read as
  "value output degenerate everywhere — value net is trivially the bottleneck (strongest
  collapse evidence)"; only genuine `n < MIN_N` coverage gaps read as "coverage failure;
  re-scope". The nulled value-leg notes are surfaced in `interpretation.value_leg_notes`.
- The verdict on real snapshots belongs to the SCHEDULED probe run (planner-owned, ≥3 seeds
  via `--seed` reruns); the factory deliverable is this runner + the fixture smoke.

## Compute / seeds / kill

- **Compute**: CPU-only (`JAX_PLATFORMS=cpu`) everywhere; smoke walk ≈ 4 s; a full real-panel
  walk (~1670 daily decisions, small model, initial_inference only — no search) is minutes on
  CPU. Never GPU in the factory loop (invariant #7).
- **Seeds**: `--seed` (default 0) fixes reset + walk dither/sampling. Multi-seed claims =
  rerun with different `--seed`; per-snapshot trend claims should use ≥3 seeds.
- **Kill date**: 2026-07-22 (with the 828 battery review), matching E3.

## Deliverables

- `rank_probe.py` — the runner (this dir). Canon imports only (`create_jax_env_historical`,
  `jit_batch_step_vector`, `jit_batch_compute_action_mask`, `compute_portfolio_value`,
  `jit_batch_encode_simple`, `SampledMuZeroVectorSearch`); 834/835 reuse via importlib;
  zero canon diffs.
- Fixture smoke: `JAX_PLATFORMS=cpu python experiments/836_value_ranking/rank_probe.py
  --fixture-smoke` → writes `results_fixture_smoke.json`.
- Checkpoint mode: `JAX_PLATFORMS=cpu python experiments/836_value_ranking/rank_probe.py
  --checkpoint jax_ckpt_fast/<name>/snapshot_ep<N> --config-json <run>.json [--db …]`.
- Output JSON schema: `windows.{full,bear22w,bull23w,y2024,recent}.per_horizon.{1,3,6}` each
  with `n` + `{value, momentum_book, momentum_market, momentum_walk_pv}` (each
  `{n, rho, p, note?}`) + `value_beats_primary` + `realized_std` (all keys present — null in
  the n=0 / missing-window branches too); `interpretation` (verbatim rule + counts + reading
  + `applied` + `value_leg_notes`); `walk` (rule + operator knobs + pct-hold +
  order/legality diagnostics + final PV); `momentum` / `horizons` (incl. `p_caveat`) /
  `leak_check` (booleans derived from the actually-used values) provenance blocks;
  `restored_step_count`; `spearman_source`; config block; jax version; timestamp;
  LEDGER-convention row (`ledger_row`, pointer derived from the actual output path).

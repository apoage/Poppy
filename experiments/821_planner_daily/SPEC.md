# SPEC — 821_planner_daily (the clean cell: fixed planner × daily bars × absolute PnL)

**Task**: `tasks/TASK_821_PLANNER_DAILY_CLEAN_CELL.md`.
**Question**: can the MuZero/MCTS planner trade when — for the first time — given an honest
objective at the predictable (daily) timescale, with all five prior contaminations removed
(foresight-BC targets, differential beat-market reward, pre-818 reward wiring, bull-only 812
window, unpredictable 1-min timescale)?

## Design (all flags committed — verified `4d73268`; consumption verified `trainer.py:807/944/950/1464`)

- `--bar-minutes 1440` — env subsamples the 1-min close array `prices_np[::1440]` -> daily bars.
- `--reward-hybrid --reward-attributable` — TASK_818 fixed wiring, absolute realized-PnL on close.
  HARD GATE: attributable never without hybrid (both present here).
- **NO** `--differential-reward` (default OFF), **NO** `--imprint` / `--amax` (oracle BC OFF;
  defaults `None`/`False`), `use_prediction_loss=False` (config default). Oracle = EVALUATION
  YARDSTICK ONLY (`episode_optimal_alpha` column).
- Model: `--model-size small --sims 32 --batch-size 256 --parallel-envs 4`.
- Episode length: `--steps 256` daily bars (~8.5 months) — long enough to open/hold/close a full
  trend cycle, short enough that ~1473 distinct start offsets tile 2021→2025.
- Two passes: (1) `--fee 0 --slippage 0` cost=0 learnability; (2) `--fee 16 --slippage 5` the real gate.

## Data — multi-regime daily, one contiguous window

- `--db-path trading_data.db` + `--periods-file experiments/821_planner_daily/period_821_train.json`.
- **Window `2021-01-01 → 2025-12-01`** covering BULL21 / BEAR22 / REC23 / BULL24 (the 819 broad span).
  `train_jax_fast.py` uses `periods[0]` ONLY (single window) — regime coverage comes from the JAX
  `reset()` sampling `episode_start` uniformly across the loaded array, NOT from multiple periods.
- 64 assets, ORDER pinned identical to the 812/806 audited universe (TASK_740 landmine: never reorder).
  Post-2021 coins back-fill flat at first real price (no synthetic gap, no P&L on the flat prefix).
- **Measured footprint** (2026-07-05): 65.1M 1-min rows fetched (~7.8 GB transient Python fetchall,
  under the 10 GB naive-path ceiling) → 2.49M distinct 1-min timestamps → **~1730 daily bars** after
  `::1440` → final on-device price array `(≈1730, 64)` f32 ≈ 0.4 MB. One-time load at startup (~2–4 min);
  episodes thereafter re-sample on-device via `jit_batch_reset` (no reload). `--bar-minutes` path chosen
  over `ohlcv_daily` direct-read to avoid new env code (committed path per task prep note).

## Launch command — cost=0 pass (launch-ready; workstation single 3090 = GPU 0)

```bash
# on workstation, from ~/Trader — ONE GPU job, desktop GPU apps quiet, correct-allocation contract
tmux new-session -d -s exp_821_planner_daily \
  "cd ~/Trader && source trader_env/bin/activate && \
   tools/safe_run.sh --name 821_planner_daily --mem 24G --gpu 0 --gpu-frac 0.7 -- \
   python -u train_jax_fast.py \
     --jax-env --db-path trading_data.db \
     --periods-file experiments/821_planner_daily/period_821_train.json \
     --bar-minutes 1440 \
     --reward-hybrid --reward-attributable \
     --model-size small --sims 32 --batch-size 256 --parallel-envs 4 --steps 256 \
     --fee 0 --slippage 0 \
     --epochs 2000 --name 821_planner_daily \
   2>&1 | tee runs/821_planner_daily/train_cost0.log"
```

**Fee pass** (the real gate): identical, but `--fee 16 --slippage 5`, `--name 821_planner_daily_fee16`,
tmux session `exp_821_planner_daily_fee16`, log `runs/821_planner_daily/train_fee16.log`. Optionally
add `--assert-postcost` on the fee pass (guards fee/slip>0 + oracle/predloss OFF); do NOT add it to the
cost=0 pass — it hard-asserts `fee_bps>0` (`train_jax_fast.py:260`) and would crash.
`--gpu-frac 0.7` → ~16.8 GB VRAM cap (≤17 GB job budget). On gpu-server change `--gpu 0` to a
verified-idle GPU (1/2/3 — never 0, VirtualBox-reserved).

## Flag/spec deviations from the TASK_821 sketch (with reasons)

- `--model small` → **`--model-size small`** (actual CLI flag; `cli.py:84`).
- `--mcts-sims 32` → **`--sims 32`** (actual CLI flag; `cli.py:91`).
- **`--seed 42` DROPPED** — no `--seed` flag exists in `train_jax_fast.py`/`cli.py` (it lives only in
  the Foxglove `train_phase2.py`). Env RNG is time-seeded (`env_factory.py:337`) so separate launches
  differ naturally, but the ≥3-seed promotion rule needs either a new `--seed` flag (code change) or
  documented reliance on time-seed variation. **FLAG for main agent before any promote claim.**
- Added flags absent from the sketch but REQUIRED: `--jax-env`, `--db-path`, `--periods-file`,
  `--parallel-envs 4`, `--steps 256` (daily episode length — the 1-min default `--steps 400` would be
  ~13 months of daily bars and blow past the window).

## Smoke (workstation, done at prep — see report)

`--epochs 40` background run; pass = env-shape `≈1730 steps, 64 assets`, `reward_loss` moving off the
~5.1–5.5 marginal floor, `pct_hold < 99` (trades ≠ 0) by smoke end, VRAM ≤17 GB / RAM ≤20 GB.

## Kill criteria

- HOLD collapse: `pct_hold → ~99+` sustained with `policy_entropy↓ + mcts_visit_entropy↓` trending
  together before alpha reacts (the 812 death-spiral signature).
- `reward_loss` pinned flat at the marginal-entropy floor (~5.1) with no movement (wiring not engaging).
- No convergence: value/policy loss flat AND `eval_alpha` not developing in budget.

## Pass/fail (pre-registered — from TASK_821)

- **CONFIRMS "MCTS can trade"**: held-out post-cost return positive-trending, no HOLD collapse,
  2022 BEAR positive, gross-beats B&H, and — the real bar — **beats the 819 linear diversified-TSMOM
  book (net@21 Sharpe +0.97 CI[+0.10,+1.84]) on the same data**. ≥3 seeds before any promote claim.
- **CLOSES the planner**: honest wiring + learnable regime + absolute reward and it still collapses to
  HOLD or cannot beat the linear book → "MCTS cannot trade this data" is a clean verdict.

## STOP

Report at `runs/821_planner_daily/RESULT.md` (main agent owns the long launch; sub-agent stops at
prep + smoke). No full-build commitment until both passes + ≥3 seeds + eval vs the 820 linear book.

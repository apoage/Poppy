# TASK_821 — The clean cell: fixed planner × daily bars × absolute PnL × oracle-as-instrument

**Status**: SPEC'D 2026-07-05 (from the same-day forensic audit; see lore + memory
[[project_oracle_design_intent]]). **Question**: can the MuZero/MCTS planner trade when — for the
first time — it is given an honest objective at the predictable timescale? Every prior "MCTS cannot
trade" datapoint carried at least one contamination: foresight-oracle BC targets (Foxglove imprint,
Stage-B), differential beat-the-market reward (short-tilt math: short-in-bear ≈ +2×move,
long-in-bull ≈ 0), the pre-fix reward wiring (818 bug), the bull-only 812 window, or the
unpredictable 1-min timescale. This run removes all five simultaneously.

## Design (all flags already committed — verified in `4d73268`)

- `--bar-minutes 1440` — daily bars (env subsamples `prices_np[::K]`); the 816/819-confirmed trend
  edge lives at this duration. Price array shrinks ~1440× vs 1-min → light run.
- `--reward-hybrid --reward-attributable` — TASK_818 fixed wiring, absolute realized-PnL on close.
  HARD GATE: attributable never without hybrid.
- **NO** `--differential-reward` (stays default-OFF), **NO** `--imprint` / `--amax` (oracle BC off;
  the 812-probe assert `train_jax_fast.py:263` guards this). `use_prediction_loss=False`.
- Oracle role: EVALUATION YARDSTICK ONLY (`episode_optimal_alpha`).
- Window: multi-regime incl. the 2022 bear (2021→2025, the 819 broad-universe span) — NOT the
  bull-only 812 window. Prep item: build the periods/env config for daily-bar multi-regime
  episodes (the 812 config was 1-min single-window).
- Model: `--model small --mcts-sims 32 --batch-size 256` (~6-7 GB on one 3090 at 1-min; less here).
- Two passes: (1) cost=0 learnability; (2) `--fee 16 --slippage 5` the real gate.

## Pass/fail (pre-registered)

- **CONFIRMS "MCTS can trade"**: held-out post-cost return positive-trending, no HOLD collapse
  (hold% stable, trades retained), 2022 BEAR positive, gross-beats B&H, and — the real bar —
  **beats the 819 linear diversified-TSMOM book (net@21 Sharpe +0.97 CI[+0.10,+1.84]) on the same
  data**. Planner must earn its complexity over a linear rule.
- **CLOSES the planner**: with honest wiring + learnable regime + absolute reward, it still
  collapses to HOLD or cannot beat the linear book. Then "MCTS cannot trade this data" is finally
  a clean verdict, and the campaign's planner chapter closes without an asterisk.

## Execution

- Host: gpu-server GPU when network returns (preferred), OR workstation 3090 under the 2026-07-05
  correct-allocation contract (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.7`, PREALLOCATE default,
  `safe_run --mem 24G`, ONE GPU job, desktop GPU apps quiet). Smoke first (≤10 min, few epochs):
  verify daily-bar env shape, reward_loss ≁ marginal-entropy floor, trades ≠ 0 at epoch 50.
- Launch sketch (fill periods config at prep):
  `tools/safe_run.sh --name 821 --mem 24G -- python -u train_jax_fast.py --bar-minutes 1440
  --reward-hybrid --reward-attributable --model small --mcts-sims 32 --batch-size 256
  --fee 0 --slippage 0 --seed 42 ...` then the fee16/slip5 pass; ≥3 seeds before any promote claim
  (the ≥3-seed rule stands — single-seed gaps ≤5pp are noise).

## Relation to TASK_820

Independent. 820 (linear CTA ship-path: parity ✅ bit-exact, gate 2 OOS pending) proceeds
regardless — it is the deploy candidate. 821 is the science: whether the planner can beat 820's
engine. If yes, the planner becomes the 820 signal upgrade path; if no, 820 is the terminal
architecture for this data and the next lever is DATA (Iris-DEX per-block / futures).

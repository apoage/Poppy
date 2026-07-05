# SPEC — 812_mcts_probe (genuine-RL MuZero/ExIt probe, post-cost, death-spiral-gated)

**Task**: `tasks/TASK_812_GENUINE_RL_PLANNER_PROBE.md` (Phase B minimal loop).
**Prior**: LOW (811 linear-optimal + cost-bound; Leg-A wall; death-spiral unsolved). Probe for the clean
yes/no on the last open seam: can a multi-step *planner* beat linear single-step *prediction* post-cost.

## The three non-negotiables (enforced)

1. **Post-cost reward ACTIVE** — `--fee 16 --slippage 5` (21 bp/side all-in) → `EnvConfig.fee_bps/slippage_bps`;
   `--assert-postcost` hard-asserts fee>0 & slip>0 & oracle/prediction-loss OFF before training (the
   zero-cost default is 810's turnover trap). Leak-free fill is structural: obs window `[t-W, t-1]`,
   execute at `t` (TASK_124). Value target = `compute_returns` (GAE/MC over env rewards) — NOT path_solver.
2. **Death-spiral triad + convergence logged every epoch** — base CSV columns `policy_entropy`,
   `mcts_visit_entropy` (TASK_812, raw visit-distribution entropy), `pct_hold`→trade_rate, `value_loss`,
   `policy_loss`, `alpha`, `eval_alpha`. ε-greedy target-mix decay **0.25→0.05 from step 0**
   (`--epsilon-schedule`) severs MCTS-sharpening→policy-sharpening.
3. **Gross-beat + post-cost-beat gate** — evaluated post-hoc by `runs/812_mcts_trader/eval_*` on the
   **held-out OOS test 2024-11-01 → 2025-04-01**, same 64 assets, vs `linear_xs` (808), gross(cost=0) AND
   post-cost Sharpe, per-rebalance H-strided, CI-separated, leak-free. Net-beat without gross-beat = artifact.

## Data (pinned to the audited 806 universe)

- `--db-path trading_data.db` + `--periods-file experiments/812_mcts_probe/period_812_train.json`
  (train window **2023-05-16 → 2024-06-01**, the 64 train-window-locked assets, asset-order pinned —
  TASK_740 landmine). Test window held out for eval.

## Launch command (GPU 2, OOM-safe tmux)

```bash
# on gpu-server, from ~/Trader
tmux new-session -d -s exp_812_mcts_probe \
  "cd ~/Trader && source trader_env/bin/activate && \
   tools/safe_run.sh --name 812_mcts_probe --mem 40G --gpu 2 --gpu-frac 0.5 -- \
   python -u train_jax_fast.py \
     --jax-env --db-path trading_data.db \
     --periods-file experiments/812_mcts_probe/period_812_train.json \
     --model-size small --sims 32 --batch-size 256 --steps 400 --parallel-envs 4 \
     --fee 16 --slippage 5 --assert-postcost \
     --epsilon-schedule --epsilon-init 0.25 --epsilon-final 0.05 --epsilon-decay-epochs 1000 \
     --epochs 2000 --name 812_mcts_probe \
   2>&1 | tee runs/812_mcts_probe/train.log"
```

GPU: **2** (verified idle; GPU 0 = VirtualBox-forbidden). VRAM est. ~5 GB (replay capacity 5000 on-device
≈ 2.2 GB + model/search/activations). Never the workstation display GPU.

## Kill criteria (die in days, not weeks)

- Death-spiral: `policy_entropy↓ + mcts_visit_entropy↓ + trade_rate→rail` trending together before alpha reacts.
- Net-beat without gross-beat (turnover artifact).
- No convergence (value/policy loss flat, alpha not developing OOS) AND no post-cost gross-beat in budget.

## STOP

At `runs/812_mcts_trader/MINIMAL_LOOP_RESULT.md` — triad+convergence curves, peak VRAM, gross + post-cost
Sharpe vs linear + CI, RSS. **No full-build commitment** until walls cleared + auditor confirms.

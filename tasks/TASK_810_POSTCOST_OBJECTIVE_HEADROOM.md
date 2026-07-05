# TASK_810 — Post-cost-objective headroom test (can a model beat linear on Kraken 1-min?)

**Status**: DRAFT — operator-approved direction (2026-06-22). The decisive test of the operator's goal:
**a working model on Kraken 1-min across all regimes.** Cheap small-model experiment that **gates** the
expensive full-encoder effort. GPU/server-side.

**The question** (one sentence):

> 806 tested an *oracle-trained* encoder (wrong objective) and it didn't beat linear. **Can a model
> trained on a post-cost TRADING objective beat the linear baseline post-cost, across regimes, on
> Kraken 1-min?** This is the one genuinely-untested lever for the operator's goal.

## Why this, now (the open lever)

- 808/809 **closed the linear signal** on Kraken (real edge ~3bps < cost ~5–21bps under every rule +
  execution; the +5.08 "rescue" was a per-bar-Sharpe annualization artifact, now caught). Linear is
  cost-bound.
- But they did **NOT** close "a model." The encoder 806 tested was *oracle-distilled* (666) — its loss
  had no trading gradient. **Nobody has trained a model on a post-cost trading objective** (realized
> P&L − cost, in the gradient). That formulation is untested, and it's the one that actually points at
  "a model that trades profitably" — because the gradient optimizes net post-cost P&L, not
  oracle-imitation.
- If a small model trained on the right objective **can't** beat linear post-cost, the signal is the
  ceiling and Kraken-1-min is closed for models too (clean Fork-C). If it **can** (even one regime),
  there's headroom → escalate to the full encoder (TASK_811).

## Scope guardrails

- **Reuse, don't rebuild.** Inherit 806's audited `linear_xs` features (causal ≤ t) + 808's cost model
  (Kraken taker 21bps/side on turnover) + split + guards. Only the **model + differentiable objective**
  are new.
- **No oracle, no leaky teacher.** The forward return is the **label (in the loss only)**, never a
  feature. This is the #1 audit item (see GUARDS).
- **Small model** — MLP or tiny cross-sectional transformer over the 64 assets. **Not Poppy/MuZero** —
  this is a headroom probe; if a small model can't beat linear, the big one won't either.
- **Train→val→test, strict OOS.** Select on val, report test only.
- **Server-side (GPU), OOM-safe** (cgroup cap, the chunked-fit lesson from 809; RSS peak reported).
  Never the workstation display GPU.

## Method

### Features + costs (inherited, unchanged)
- `linear_xs` inputs (64 assets × 63 lagged log-returns, causal), same train/val/test split, same 21bps
  taker cost model. Optionally a maker arm, but **taker is the apples-to-apples bar** vs 808.

### Model (NEW, small)
- Input: the per-asset lagged features (+ optional cross-sectional context). Output: per-asset weight
  (or score → rank → top-k weight). Document architecture + size. Keep it small/fast.

### Objective — THE crux (NEW)
A **differentiable post-cost portfolio P&L**, end-to-end:
`loss = − mean_t[ Σ_a w_a(t) · r_a(t→t+H)  −  turnover(t) · cost_per_side ]`
- `w_a(t)` = model output from features ≤ t (causal).
- `r_a(t→t+H)` = **realized** forward return (the LABEL — enters the loss only, never the input).
- `turnover(t) = Σ|w_a(t) − w_a(t−1)|` (actual, like 808).
- The gradient flows **through the portfolio construction and the cost** → a real trading gradient that
  forces the model to learn **cost-aware, low-turnover, selective** decisions (exactly the "tactical
  decisions / decide when to trade" the operator wants). Optionally a differentiable-Sharpe surrogate
  (mean/std) instead of mean-PnL — report which.
- Soft position regularisation / turnover cap as needed; pre-register, don't tune to the test.

### Comparison + gate
- **vs linear_xs, POST-COST**, on the test window AND **all regime windows** (`benchmark_periods.json`,
  1-min data): bull / bear / sideways / ftx_crash / etf_approval / boring_week. Report **per regime**.
- **Primary gate:** does the model beat linear post-cost (Sharpe) on the test window or any regime?
- **Secondary:** does it clear **Sharpe ≥ 1 post-cost** anywhere linear couldn't?
- **Sharpe annualized CORRECTLY:** per-rebalance / holding-period returns × √(rebalances/year) — **NOT**
  per-bar × √525600 (that was the 809 artifact; do not repeat it). Block-bootstrap CI on the
  correctly-annualized Sharpe.

## Deliverables (audited)

1. `runs/810_postcost_headroom/result.md`: per-regime table (model post-cost Sharpe + CI vs linear
   post-cost, turnover, PnL) + the gate verdict + the model/objective spec + (if headroom found) a
   note that TASK_811 (full encoder on this objective) is the escalation.
2. `runs/810_postcost_headroom/model.py`: the model + differentiable objective, reproducible.
3. `runs/810_postcost_headroom/GUARDS.md`: inherited guards (cite 806/808) + NEW audit items —
   (a) the differentiable portfolio layer does **not** feed the realized return back into the weights
   (label-only); (b) Sharpe annualization is per-rebalance, not per-bar; (c) features stay causal.

## Acceptance checklist (audited separately)

- [ ] Differentiable objective is end-to-end; gradient flows through portfolio + cost (a trading
      gradient, not MSE/oracle-imitation).
- [ ] Cost in the objective == 808's Kraken taker model (apples-to-apples vs linear).
- [ ] **No leak:** features causal (≤ t); forward return is the label (loss only); portfolio layer
      doesn't feed realized return into weights.
- [ ] **Sharpe annualized correctly** (per-rebatch/holding-period, NOT per-bar × √525600).
- [ ] Model vs linear POST-COST, per regime, val-selected, test-reported.
- [ ] Server-side, OOM-safe, no workstation GPU; no oracle/Bella/leaky-teacher.
- [ ] Verdict honest: if no headroom above linear, say so plainly (Kraken-1-min closed for models).

## Risk

- **Model doesn't beat linear (likely).** The 1-min signal is largely *linear* (806: the big
  oracle-trained encoder didn't beat linear at H=1). A right-objective model learns *when not to trade*
  (cost-aware), so it's not foregone — but the prior is it confirms the ceiling. If so → Fork C (the
  1-min signal is the ceiling on Kraken; the methodology is the harvest).
- **Differentiable-backtest leak.** The portfolio layer mixing realized return into weights is the
  classic trap. Audit explicitly.
- **Overfitting the objective on train.** Strict val→test; watch for the model fitting the
  differentiable surrogate without generalizing.
- **Headroom found → TASK_811** (retrain the actual encoder/Poppy on this objective) is the follow-up,
  not part of this task.

## References

- `runs/806_predictability_probe/` (audited features), `runs/808_linear_backtest/` (audited cost model +
  the CORRECT per-rebalance Sharpe), `config/benchmark_periods.json` (regime windows).
- `runs/HANDOFF_2026-06-21_path-to-working.md`, `runs/QUEUE_DISPOSITION_2026-06-21.md`.

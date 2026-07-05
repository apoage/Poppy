# HANDOFF PROMPT — 2026-06-21 — path to a working trader (paste to Opus)

> Paste everything below the line to Opus. Self-contained.

---

You're picking up the campaign after a pivotal audit. **Read these in order before doing anything:**
1. `runs/QUEUE_DISPOSITION_2026-06-21.md` — authoritative state + **your execution order (top section)**. READ FIRST.
2. `tasks/TASK_808_LINEAR_COSTAWARE_BACKTEST.md` — your primary task spec.
3. `runs/HANDOFF_2026-06-21_path-to-working.md` — strategic context.

## What just changed (you don't have this context yet)

**TASK_806 is DONE and independently AUDITED.** Verdict: real 1-min crypto returns have **clean-OOS
predictability** — a cross-sectional linear model gets IC 0.06–0.15 (H=1/5/20), FDR-surviving, N_eff up
to 217k. Guards re-derived clean (causal window, no leak; persistence negative = real mean-reversion,
not a leak artifact). **Conclusion: the campaign's 0-promotions record is a METHOD failure (TASK_666's
oracle-distillation spine has no trading gradient), NOT a data failure. The data can support a trader.**

Two things that clean your queue (details in QUEUE_DISPOSITION):
- **787 and 790 are DEAD** — both already RAN and returned NULL (`lore/notes_2026-06-02.md:9-10`: 787
  PnL-in-loss +0.35pp FAIL, 790 realized-PnL-value +0.06pp FAIL). Their task files say "unrun" — **stale**.
  Do NOT re-run them or any 780–784 spine patch. The minimal "fix the objective on the spine" idea is a
  proven dead end.
- **Tokenization trader-mechanism is DEAD** (804 = NULL; and that test was confounded — it ran on the
  spine 806 proved can't trade). The **tokenizer itself works** (frozen codebooks, cross-asset transfer
  validated vs permutation floors, grammar above null) — it's a *deferred tool*, not dead. Don't revive
  it as a trader mechanism now.

## Your mission: ship the first working trader. Simply. In this order.

### STEP 1 (PRIORITY) — TASK_808: cost-aware OOS backtest of the proven linear signal
- **Reuse the AUDITED 806 `linear_xs` path + guards** (G1–G7 + UNIVERSE-LOCK, already proven in
  `runs/806_predictability_probe/`). Only re-audit the portfolio + cost code you ADD — don't rebuild the
  data/feature/guard path.
- **The whole point is the POST-COST gate.** Does any cell in the pre-registered grid
  (`H∈{1,5,20}` × rebalance `∈{1,5,20,60}min` × `{long-short,long-only}` × `k∈{8,16,32}`) clear
  **post-cost annualized Sharpe ≥ 1.0** with a block-bootstrap 95% CI excluding 0, over the full test
  window, with **realistic Kraken costs** (taker fees ~16bps side + a slippage model + the ACTUAL
  turnover the strategy generates)?
- **Expect 1-min to get eaten by fees** — that's fine; the sweep finds the horizon/frequency that
  survives (longer H / lower rebalance = lower turnover = costs bite less). Report **pre-cost vs
  post-cost** side by side so the cost ceiling is visible.
- **If a cell passes → that's the first working trader.** Spec the paper-trade wiring (a thin sibling
  of `real_trade_jax.py`). If nothing passes, the cost ceiling is the finding and the learned model
  (Step 2) must beat whatever linear achieves post-cost.
- **CPU / server-side, no GPU.** It's a ridge + backtest.

### STEP 2 (BACKGROUND — parallel, read-only, not a blocker) — TASK_807
Scope the genuine-RL rebuild: was pre-666 RL a lead or a wall? Does `train_jax_fast.py` still run
post-666? **Do NOT start any rebuild until 808 ships AND 807 reports a lead.** If 808's linear trader
works and 807 says genuine RL was a wall → stop; the linear trader is the system.

## Constraints (every step, non-negotiable)
- **All training/compute on gpu-server, never the workstation display GPU** (it hard-locks the box).
- **OOM safety by cgroup construction**: cap = verified_peak × ~1.5 (well under host), `python -u`,
  opt out of oomd pressure where possible, report + persist the true RSS peak. Trip→clean-cgroup-kill→
  raise→relaunch is the designed contingency (806 did 64G→100G this way, host never touched).
- **Audit discipline**: an independent auditor (separate model) will audit 808 against its acceptance
  checklist — re-derive the guards from `backtest.py`, check the post-cost verdict on the table, **not
  the summary**. Make every step self-evident from the artifacts.

## DROP (do not touch) / DEFER (keep, don't run now)
- **DROP:** 780, 781, 782, 783, 784, 787, 790, 789, 791, 792, 793, 794, 795, 796, 797, 798, 799, 804.
- **DEFER:** 785, 786, 800, 801, 802, 803, 805.
- (Full rationale in `runs/QUEUE_DISPOSITION_2026-06-21.md`.)

## Hand back
When `runs/808_linear_backtest/` has `result.md` + `backtest.py` + `GUARDS.md`, **STOP and surface for
audit.** Don't chain into the rebuild. The auditor verifies the post-cost verdict before anything next.
Report: per-cell pre/post-cost Sharpe+CI, the gate verdict, the cost-assumption set, the RSS peak, and
(if it passed) the deploy plan.

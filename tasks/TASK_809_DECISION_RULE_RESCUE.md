# TASK_809 — Decision-rule rescue: can a smarter rule + maker capture the proven edge?

**Status**: DRAFT — operator-approved direction (2026-06-21). The live lever after 808. **Reuses the
audited 806 signal + 808 cost-aware backtest; adds the two UNTESTED decision levers** (conviction/cost
gating, maker execution) that 808 held fixed. CPU/server-side.

**The question** (one sentence):

> 808 proved the linear signal is real but dies to taker costs *under a dumb fixed-schedule rebalance*
> (turnover ~1.0/rebalance). Can a **smarter, event-driven decision rule** (trade only when edge > cost)
> and/or **maker execution** (earn the spread) slash turnover/cost enough to make the proven edge
> post-cost profitable?

## Why this, now (the defined unknown)

The 2026-06-21 lever-reconciliation established what's tried vs live:
- **TRIED/dead**: cost-aware backtest (808, done); token-informativeness gate (803, dead — gates on input
  entropy, baseline 98.5% hold); turnover penalty in the loss (787, ran NULL +0.35pp; loss-side, not a
  gate).
- **UNTESTED/live**: conviction/cost-gated event-driven decisions; maker execution. (Lower frequency is
  OFF THE TABLE — operator directive: the signal is a 1-min phenomenon and resampling collapses sample
  size. Solve it at 1-min or not at all.)

808's failure was the *decision rule* (a timer), not the signal — it traded ~every bar regardless of
conviction, so turnover × taker-cost > edge. The signal's per-trade edge (~3bps from IC 0.15) is real
and positive; it just needs a rule that trades **selectively** (your "decide right" / "tactical
decisions" point) and/or a cheaper execution regime. Both are untested.

## Scope guardrails

- **Reuse, don't rebuild.** Inherit 806's `linear_xs` signal (audited) + 808's cost-aware backtest +
  guards (G1–G10 + UNIVERSE-LOCK + the 808 cost model). Only the **decision rule** and **execution
  model** are new — audit those, cite the inherited rest.
- **No oracle, no Bella, no encoder, no MCTS.** Same linear signal as 806/808 (the learned model is a
  later, separate question).
- **CPU/server-side, OOM-safe** (cgroup cap, `python -u`, RSS peak reported). Never workstation GPU.
- Writes only: `runs/809_decision_rule_rescue/`. Don't touch 806/808 artifacts, training code, ckpts.

## Method

### Signal + costs (inherited, unchanged from 806/808)
- `linear_xs` predicted cross-sectional forward return per (t, asset), same split, same 21bps/side taker
  cost model + sensitivity sweep, same post-cost gate. Inherited-signal check (fit IC == 806 within 1e-3).

### Arm A — conviction/cost-gated, event-driven decision rule (NEW)
Instead of rebalancing on a fixed clock, decide **per bar whether to trade at all**:
- Compute per-asset predicted edge. A trade is taken only when the **expected net edge exceeds a
  threshold**: `Σ |predicted_edge| over the changed positions − cost_of_those_trades > τ`. Sweep `τ`.
- Add **hysteresis** (a position is only changed if the target weight differs from current by more than
  a band `h`) so the book doesn't churn on noise. Sweep `h`.
- Report the resulting **turnover per rebalance** — the hypothesis is it drops far below 808's ~1.0,
  which is what lets the edge survive.
- This is the direct embodiment of "trade only when it's worth it" / event-driven / no-timer.

### Arm B — maker / limit-order execution (NEW)
Model a maker regime honestly (NOT free fills):
- Limit order posted at the relevant side; **filled only if price crosses it** within the bar/window
  (fill probability from historical touch rates, or a conservative model).
- **Adverse selection**: a fill means the market moved toward/through your limit — model the cost of
  that (the filled trades are the adverse ones). This is the hard part; state the model + sensitivity.
- **Maker fee**: ~−2 to −5 bps/side (rebate) instead of +21 taker. Sweep the maker rate.
- Combine with Arm A (gated decisions executed maker) and standalone (fixed-rebalance + maker).

### Grid (pre-registered)
- Arms: {gated-taker, gated-maker, fixed-maker} × `τ` ∈ {0, small, med} × `h` ∈ {0, small, med} ×
  maker-rate ∈ {−2, −5}bps × `H ∈ {1,5,20}` × `k ∈ {16,32}` (subset of 808's, focused on the
  lowest-turnover survivors). Pre-register; don't chase.

### Gate (same as 808, post-cost)
A cell WORKS iff post-cost Sharpe ≥ 1.0 AND net PnL > 0 over full test AND block-bootstrap 95% CI on
post-cost Sharpe excludes 0. **The verdict question: does any (rule × execution) cell clear it?**

## Deliverables (audited)

1. `runs/809_decision_rule_rescue/result.md`: per-cell table (arm, τ, h, maker-rate, H, k: turnover,
   pre/post-cost Sharpe + CI, PnL, maxDD, fill-rate for maker) + gate verdict + the decision-rule +
   maker-fill model documentation + (if any cell passes) deploy plan.
2. `runs/809_decision_rule_rescue/decide.py`: reproducible entry point, reusing 806/808 paths.
3. `runs/809_decision_rule_rescue/GUARDS.md`: inherited guards (cite 806/808) + the NEW audit items —
   (a) the gate/hysteresis uses no future info (decision at t from features ≤ t); (b) the maker fill
   model is honest (adverse selection modeled, not free fills; no look-ahead in fill determination).

## Acceptance checklist (audited separately)

- [ ] Reuses 806 signal + 808 cost path (cite); inherited-IC check passes.
- [ ] Arm A turnover is **measured** and shown to drop vs 808's ~1.0 (the whole point — if it doesn't
      drop, the gate isn't working).
- [ ] Arm B maker fills model **adverse selection** (not free); fill-rate reported; maker rate swept.
- [ ] Decision/gate/hysteresis use no future info (positions at t from predictions on features ≤ t).
- [ ] Post-cost gate is Sharpe ≥ 1.0 + PnL>0 + CI excludes 0; pre vs post reported; sensitivity sweep.
- [ ] No oracle/Bella/encoder/MCTS. Server-side, OOM-safe, no workstation GPU.
- [ ] Verdict follows the gate mechanically; honest if nothing passes — **stay on 1-min** (lower
      frequency is out of scope: the signal decays 0.15→0.06 from H=1→20 and resampling collapses
      sample size). If neither arm clears costs at 1-min, the 1-min signal is ceiling-bound on this
      data → surface to operator (a data/problem decision, not a rule tweak).

## Risk

- **Maker adverse selection eats the rebate.** Likely partial. Model honestly; the sweep shows margin.
- **The gate reduces trades so much that gross edge drops too.** There's a trade-off (trade less = less
  cost but fewer edge-capturing trades); the sweep finds the sweet spot. If no τ/h works, that's the
  finding.
- **Decision-rule look-ahead.** The one new leak surface — the gate/hysteresis must use only features ≤
  t. Audit explicitly.
- If both arms fail at 1-min → the 1-min signal is ceiling-bound on this data. **Do not retreat to
  lower frequency** (signal decays with horizon; sample size collapses). Surface to operator — the
  decision is then about the data/problem, not the rule.

## References

- `runs/806_predictability_probe/` (audited signal), `runs/808_linear_backtest/` (audited cost path +
  guards G8–G10) — reuse both.
- `runs/HANDOFF_2026-06-21_path-to-working.md`, `runs/QUEUE_DISPOSITION_2026-06-21.md`.
- `real_trade_jax.py` (limit-order logic lives here for live; backtest it instead of assuming taker).

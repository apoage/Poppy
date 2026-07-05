# HANDOFF PROMPT — TASK_809 — decision-rule rescue (paste to Opus)

> Paste everything below the line. Self-contained.

---

You're continuing the path-to-working. **Read before doing anything:**
1. `tasks/TASK_809_DECISION_RULE_RESCUE.md` — your spec.
2. `runs/808_linear_backtest/` (result.md, backtest.py, GUARDS.md) — the audited cost path you REUSE.
3. `runs/806_predictability_probe/` — the audited signal you REUSE.

## Where we are (you don't have this context)
- **806 (audited):** real 1-min crypto returns are predictable — cross-sectional linear IC 0.06–0.15,
  clean OOS, no leak. The signal is REAL.
- **808 (audited):** that signal, traded under a **dumb fixed-schedule rebalance + taker costs**, is
  annihilated — 0/72 cells pass post-cost Sharpe ≥ 1.0. Best cell +3.66 pre → −18.92 post. Cause: the
  rule trades ~every bar regardless of conviction (turnover ~1.0/rebalance), so 21bps × turnover > edge.
- **The failure was the decision RULE (a timer), not the signal.** The per-trade edge (~3bps) is real and
  positive; it just needs a rule that trades *selectively* and/or a cheaper execution regime. **Both are
  untested** (confirmed: 803 gated on *input* entropy not confidence — dead; 787 was a *loss penalty*
  not a gate — ran NULL).

## Your mission (TASK_809): does a smarter decision rule + maker capture the proven edge?
**Reuse the audited 806 signal + 808 cost backtest verbatim.** Only the decision rule and execution
model are new. Two arms:

- **Arm A — conviction/cost-gated, event-driven rule.** Don't rebalance on a clock. Per bar, trade only
  when the expected net edge (predicted edge − cost of the trades) exceeds a threshold τ, with
  hysteresis h so positions don't churn on noise. Sweep τ, h. **The whole point: turnover per rebalance
  must drop far below 808's ~1.0 — measure it and show it.** This is the "trade only when it's worth it
  / no timer" idea made concrete.
- **Arm B — maker / limit execution.** Model maker fills **honestly**: filled only if price crosses the
  limit (fill prob from historical touch rates), **adverse selection modeled** (fills are the adverse
  ones — NOT free), maker rebate ~−2 to −5bps/side vs +21 taker. Sweep the maker rate. Combine with A.

**Gate (same as 808, post-cost):** a cell WORKS iff post-cost Sharpe ≥ 1.0 AND net PnL > 0 over the full
test window AND block-bootstrap 95% CI excludes 0. **Verdict question: does any (rule × execution) cell
clear it?**

## Audit items (an independent auditor re-derives these from `decide.py`, not the summary)
- **Arm A turnover must be measured and shown to drop** vs 808's ~1.0 — if the gate isn't reducing
  turnover, it isn't working; report that honestly.
- **Arm B maker fills must model adverse selection** — not free fills. Report fill-rate. No look-ahead
  in fill determination (fill at t from info ≤ t).
- **Decision/gate/hysteresis use no future info** — positions at t from predictions on features ≤ t
  only (the one new leak surface; assert it).
- Inherited-signal check: fitted IC == 806's within 1e-3 (aborts on drift).

## Constraints (non-negotiable)
- **All compute on gpu-server, never the workstation display GPU** (hard-locks the 45GB box — burned
  twice). Workstation = edits / `py_compile` only.
- **OOM safety by cgroup construction**: cap = verified_peak × ~1.5, `python -u`, opt out of oomd
  pressure, persist the RSS peak. Trip→clean-kill→raise→relaunch is the designed contingency.
- **Audit discipline**: every result re-derived from artifacts before lore/canon.

## Hand back
When `runs/809_decision_rule_rescue/` has `result.md` + `decide.py` + `GUARDS.md`, **STOP for audit.**
Don't chain into a rebuild. Report: per-cell turnover + pre/post-cost Sharpe + CI,
the gate verdict, the decision-rule + maker-fill models, the RSS peak, and (if any cell passed) the
deploy plan. If nothing passes, say so plainly — **the 1-min signal is ceiling-bound on this data**.
Stay on 1-min: lower frequency is OFF THE TABLE (operator directive — the signal is intrinsically
short-horizon, IC decays 0.15→0.06 from H=1→20, and resampling collapses sample size). Surface to
operator for the data/problem decision; there is no fallback lever.

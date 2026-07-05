# HANDOFF PROMPT — TASK_810 — post-cost-objective headroom test (paste to Opus)

> Paste everything below the line. Self-contained.

---

You're testing the operator's core goal: **a working model on Kraken 1-min, all regimes.** **Read first:**
(1) `tasks/TASK_810_POSTCOST_OBJECTIVE_HEADROOM.md` — your spec; (2) `runs/806_predictability_probe/` —
the audited features you REUSE; (3) `runs/808_linear_backtest/` — the audited cost model + the CORRECT
per-rebalance Sharpe annualization you REUSE.

## Where we are (you don't have this context)
- **806 (audited):** real 1-min crypto returns are predictable — linear cross-sectional IC 0.06–0.15,
  clean OOS, no leak.
- **808 (audited):** that linear signal, under taker costs + any decision rule, is cost-bound — 0 cells
  pass post-cost Sharpe ≥ 1.0. Real edge (~3bps) < cost (~5–21bps).
- **809 (audited):** the decision-rule + maker "rescue" was an **artifact** — per-bar Sharpe annualized
  by √525600 inflated a +5.08 that's really ~+0.7–1.5. Ceiling-bound confirmed at 1-min on Kraken spot
  for the *linear* signal.
- **The open lever:** 806's encoder was *oracle-trained* (666 leaky distillation — wrong objective).
  **Nobody has trained a model on a post-cost TRADING objective.** That's the one untested formulation,
  and it's what actually points at "a model that trades profitably."

## Your mission (TASK_810): can a right-objective model beat linear post-cost, all regimes?
**Reuse the audited 806 features + 808 cost model + split.** Only the model + differentiable objective
are new. Train a **small model** (MLP / tiny cross-sectional transformer — NOT Poppy; this is a headroom
probe) on a **differentiable post-cost portfolio objective**:

`loss = − mean_t[ Σ_a w_a(t)·r_a(t→t+H) − turnover(t)·cost_per_side ]`

- `w_a(t)` from features ≤ t (causal); `r_a(t→t+H)` is the **realized forward return = the LABEL, in the
  loss only, never an input**; `turnover = Σ|Δw|`; `cost_per_side` = 808's Kraken taker 21bps.
- The gradient flows **through the portfolio + the cost** → a real trading gradient that forces the model
  to learn cost-aware, low-turnover, selective trading (the "decide when to trade" point).
- **Compare vs linear_xs POST-COST**, on the test window AND **all regime windows**
  (`benchmark_periods.json`: bull/bear/sideways/ftx_crash/etf/boring, 1-min). Report **per regime**.
- **Gate:** does the model beat linear post-cost (Sharpe) on test or any regime? Clear Sharpe ≥ 1
  post-cost anywhere linear couldn't?

## Audit items (an independent auditor re-derives these from `model.py`, not the summary)
- **No leak (the #1 item):** the differentiable portfolio layer must NOT feed the realized return back
  into the weights — `r_a(t→t+H)` is the label (loss only), inputs stay causal (≤ t). Assert/verify this.
- **Sharpe annualized CORRECTLY:** per-rebalance / holding-period returns × √(rebalances/year) — **NOT**
  per-bar × √525600 (that was the 809 artifact; do not repeat it). Block-bootstrap CI on the
  correctly-annualized Sharpe.
- **Apples-to-apples:** cost in the objective == 808's Kraken taker model (same definition), so the
  model-vs-linear comparison is honest.
- Train→val→test, strict OOS; report val-selected, test-only numbers.

## Constraints (non-negotiable)
- **All training/compute on gpu-server, never the workstation display GPU** (hard-locks the 45GB box —
  burned twice). Workstation = edits / `py_compile` only.
- **OOM safety by cgroup construction**: cap = verified_peak × ~1.5, `python -u`, opt out of oomd
  pressure, persist the RSS peak. Use the chunked-fit pattern from 809 if the fit is large.
- **Audit discipline:** every result re-derived from artifacts before lore/canon.

## Hand back
When `runs/810_postcost_headroom/` has `result.md` + `model.py` + `GUARDS.md`, **STOP for audit.** Don't
chain into a full-encoder rebuild. Report: per-regime model-vs-linear post-cost Sharpe + CI, turnover,
the gate verdict, the model/objective spec, RSS peak. **Honest verdict either way:** if no headroom
above linear, say so plainly — Kraken-1-min is closed for models and the operator moves to the
accept/harvest decision. If headroom found (model beats linear post-cost in any regime), flag it — that
unblocks TASK_811 (retrain the actual encoder on this objective), which is the path to the operator's
working-model goal.

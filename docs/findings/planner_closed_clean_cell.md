# planner closed clean cell

> TASK_821 (2026-07-06): the SCALAR-action MuZero planner converges to 100% HOLD (0/1670 trades, Sharpe 0.00 vs linear +0.97) on daily crypto — because a single-action-per-step agent can only place one sub-noise bet per decision. This is an ACTION-SPACE defect, not proof planners can't trade. The vector-action rebuild (TASK_828) is the diagnosed, untested fix.

*Migrated from local agent memory 2026-08-31 (`project_planner_closed_clean_cell.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
**TASK_821 measurement (2026-07-06, `runs/821_planner_daily/EVAL_PASS1.md`) — stands as fact:**
the planner with every prior contamination removed (818 reward-fix, daily bars, absolute PnL, no
oracle BC, multi-regime incl. BEAR22, cost=0, seed 42) trained clean (2000 ep, entropy ROSE
1.06→1.88, no collapse) and the deterministic policy holds 100% cash — **0 trades in 1670 daily
decisions**, net@21 Sharpe +0.00 vs the +0.97 linear book, in-sample.

**The diagnosis (this is the load-bearing part, and it points FORWARD):** the failure is the
ACTION SPACE, not the planner concept. The env takes a SCALAR action and touches ONE asset per
step (`jax_trading_env.py:1154`), so a single decision is a single per-asset bet whose EV is below
its own noise → the value function rationally declines every one. But the winning strategy (819's
book) is a SIMULTANEOUS 64-asset portfolio whose edge is the SUM of 64 tiny edges — inexpressible
in a one-asset-per-step vocabulary. The planner didn't fail to trade; it was never given the
action space that can hold the strategy.

**The fix (TASK_828, live, untested):** vector-action MuZero — action = per-asset weight vector,
all 64 assets rebalanced per step, so one decision's EV = Σ(64 edges), above noise. The policy head
is already factored (`muzero_model.py:256`); the work is a vector env step + Sampled-MuZero search.
This is the honest test the campaign never ran.

**What this does NOT mean (prune the old fatalism):** NOT "planners are structurally incapable,"
NOT "MCTS cannot trade this data," NOT "ship the linear rule and stop." Those over-generalizations
came from reading a scalar-architecture result as a universal law. TASK_821 is a clean verdict on
ONE formulation. Related scoped negatives: [single_mechanism_exhausted](single_mechanism_exhausted.md),
[modeling_surface_exhausted](modeling_surface_exhausted.md) — all pre-vector-action.

Interim ship candidate meanwhile: 820 linear CTA (paper gate 3). Breadth lever independent of the
planner: [diversified_trend_shipcandidate](diversified_trend_shipcandidate.md) → TASK_825 cross-asset.

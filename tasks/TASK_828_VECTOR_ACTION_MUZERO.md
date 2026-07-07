# TASK_828 — Vector-action MuZero: one decision = the whole portfolio

**Status**: ACTIVE 2026-07-06. Operator directive: THIS is the fix. TASK_821 closed the planner
because a single-action-per-step agent evaluates each trade's SOLO EV against noise and rationally
declines every one — while the edge is breadth-aggregated (sum of 64 tiny per-asset edges). The fix
is to change WHAT A DECISION IS: the action becomes a **per-asset vector** (all 64 assets acted on
in one step), so one decision's EV is the SUM of 64 edges — above noise, rankable by the value
function. Not a reward-head patch, not the linear rule. The action space itself.

## Current defect (grounded in code)
- Env `step()` takes a SCALAR action, decodes ONE asset: `asset_idx = action//5; action_type =
  action%5` (`mcts_trader/env/jax_trading_env.py:1154-1155`). One asset per timestep → to even
  HOLD the diversified book the planner needs ~10× its action bandwidth. The book is INEXPRESSIBLE.
- Policy head is ALREADY factored: `FactoredPolicyHead` → `(B, n_assets, 5)` (`muzero_model.py:
  256-270`), then flattened to `(B, 320)` and collapsed to a single categorical. **The per-asset
  structure exists in the network and is discarded at decode.** Big head start.
- MCTS (`mctx_search.py`) searches a single categorical over 320 → cannot represent the 5^64 joint
  space. This is the piece that needs the real work.

## The build (flag-gated `--vector-action`, canon scalar path preserved bit-identical)

### Stage 1 — env vector step (foundation, testable in isolation)
`step_vector(state, action_vec)` where `action_vec: (n_assets,)` ∈ {0..4} per asset. Execute ALL
assets' action-types simultaneously = a one-shot portfolio rebalance. Vectorize the existing
per-asset fill/fee/margin/PnL logic (reuse `--reward-attributable` realized-PnL-on-close, `:1157+`).
Reward = the whole vector step's portfolio log-return (already the base reward at `:1210`). Verify:
a hand-built action_vec produces the same end-state as the equivalent sequence of scalar steps
(equivalence test); self-tests for cash/margin conservation under simultaneous rebalance.

### Stage 2 — Sampled-MuZero MCTS over joint action-vectors
Single-categorical mctx can't hold 5^64. Use **Sampled MuZero** (Hubert et al. 2021): at each node,
SAMPLE K candidate action-VECTORS from the factored policy `(n_assets, 5)` (independent per-asset
draws, K≈16–32), treat those K as the node's action set, run MCTS/backup over them, correct visit
counts for the sampling distribution. The factored policy head already gives the per-asset
distribution to sample from — no new head needed. Dynamics/reward/value nets take the sampled
vector as the action input (embed the vector, not a scalar index). Keep gumbel/qtransform machinery.

### Stage 3 — training loop + targets
Self-play emits vector-action trajectories. Policy target = the factored visit distribution over the
sampled vectors (per-asset marginals or the sampled-set weighting). Value/reward targets unchanged
(portfolio PnL). Thread `--vector-action` through `train_jax_fast.py` / config / env_factory /
self_play. Default OFF = today's scalar path, bit-identical (canon gate at every stage).

### Stage 4 — evaluation vs the bar
Deterministic full-timeline rollout (reuse `runs/821_planner_daily/eval_pass1.py` pattern), daily
bars, absolute PnL, 2021–2025 incl. BEAR22, cost=0 then fee16/slip5. Pre-registered PASS: beats the
819 linear book (net@21 Sharpe +0.97) AND holds positions (no HOLD collapse) AND BEAR positive.
≥3 seeds before any promotion claim (`--seed` exists, `687c146`).

## Why this can work where scalar couldn't
The scalar planner's per-decision EV was a single sub-noise trade → rationally HOLD. A vector
decision's EV is Σ(64 per-asset edges) — the SAME aggregate the linear book harvests, but now inside
a value function that can also learn regime-conditioning, risk sizing, and cross-asset structure the
fixed rule can't. This is the one architecture that lets the planner express the strategy that
actually works on this data.

## Constraints / discipline
Flag-gated, default-OFF, canon bit-identical proven at EVERY stage. Substantial build — stage it,
land each stage green before the next. GPU: server (gpu-server, back online) preferred for training;
workstation 3090 under the correct-allocation contract (MEM_FRACTION=0.7, cgroup RAM cap) for smokes.
Sub-agents: commit + report, never deploy. Reference: Sampled MuZero (Hubert et al. 2021,
arXiv:2104.06303) for the sampling/visit-correction math.

## Relation to the closed planner
TASK_821 closed the SCALAR planner cleanly. This does not reopen that — it changes the action space,
a different architecture. If Stage 4 beats the linear book, the planner returns on the RIGHT
formulation; if it still can't, the closure extends to vector actions too and the answer is truly
architectural, not action-bandwidth. Either way it's the honest test the campaign never ran.

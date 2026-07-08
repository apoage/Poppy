> **STATUS (2026-07-08): Stage 3 SHIPPED as `task828-stage3`.**
> — Vector-action training loop threaded end-to-end (`--vector-action`, default OFF = canon scalar,
>   parity hash == PINNED `a0627ce…4754`). Item K landed: the **D2 action-conditional reward head**
>   (`recurrent_inference_vector` embeds the per-step action vector into the dynamics MLP), factored
>   per-asset **marginal policy targets** (asset-major CE, MEAN over assets), per-asset entropy (ln5).
> — 3 CPU gates green (orchestrator-run): parity MATCH==PINNED; flag-ON e2e smoke; **E3 reward-vs-action
>   probe** (sensitivity/control ratio 6.16e9, single-slot flip moves reward → the head is genuinely
>   action-conditional). selftest 7/7. Full high-tier panel: 0 BLOCKER/MAJOR; 1 MINOR (vector eval was
>   an `action_points`-capped legacy-env approximation logged as faithful → now fail-closed SKIP) fixed
>   + re-verified. DEC-5 confirmed (base 320 head asset-major).
> **Remainders (SCHEDULED gpu-server GPU run, NOT factory-loop):** real E3 R-threshold PASS (DEC-1
>   `R_model ≥ 0.5×R_true`), multi-epoch convergence, ≥3-seed sweeps, Stage-4 full-timeline eval vs the
>   +0.97 bar. Non-blocking: `vector_gumbel_scale_eval` reserved/unwired; DS24 visit-entropy 0.0 in
>   vector mode (cosmetic); `jit_batch_step_vector_unique` not gate-exercised. **Unblocks TASK_832.**
>
> **STATUS (2026-07-08): Stages 1 & 2 SHIPPED.**
> — **Stage 1**: vector env step `step_vector` flag-gated (`EnvConfig.vector_action`, default OFF),
>   canon bit-identical, panel-green (13 tests).
> — **Stage 2**: `SampledMuZeroVectorSearch` (`mcts_trader/jax_agent/sampled_search.py`) + gated
>   `VectorActionEmbedding`/`dynamics_vector` in the model (`use_vector_action`, default OFF),
>   `mctx_search.py` untouched, canon param-tree + search hash bit-identical, full high-tier panel
>   green — Sampled-MuZero math verified (β=π on-policy ⇒ uniform-per-slot prior exact); search
>   confirmed to use its value net (11 tests).
> **Stage 3 wire-up carry-notes (panel-surfaced — honor these when threading self-play):**
>   (1) mask via −inf per-asset logits BEFORE the search (`sampled_search` takes no `action_mask` arg);
>   (2) build the model with the 320-wide asset-major factored head (`idx=asset*A+action`);
>   (3) deterministic eval selects via `visit_probs` argmax / lower `gumbel_scale` (default returned
>   `action_vec` is exploration-dominated); (4) interior depth-reuse of root K vectors is a documented
>   simplification (per-node resampling = follow-up); the **action-conditional reward head (D2) + E3
>   variance-attribution probe is the hard gate inside Stage 3**.
> **Remainders:** Stage 3 (training loop + factored targets + action-conditional reward head — CODE
> may land but TRAINING LAUNCH waits on TASK_831 E1 PASS), Stage 4 (scheduled eval). `use_vector_action`
> not yet threaded through training/self-play/`env_factory`/`train_jax_fast.py`.
> **GATE UPDATE (2026-07-08, planner): real-panel 831 E1 = PASS ⇒ Stage-3 TRAINING GATE OPEN.**
> E2 carried proposal scheme for Stage-3 self-play: **dithered-book|K16** (budget fallback K8);
> linear-prior proposals rejected (pin search to the rule's neighborhood). Corridor for Stage-4
> grading: net@21 [floor_inenv +0.424, ceil +29.698], engine ref +0.751 on 2021-01-01..2025-09-01
> (LEDGER 830/E0-REAL, 831/E1-REAL, 831/E2-REAL).

> **SCHEDULED-RUN LOG (planner, 2026-07-08 — battery pre-flight; to be folded into
> `experiments/828_stage3_train/SPEC.md` when the factory disarms).**
> Smoke #1 (fee16, hold-bias 3.0, 40 ep, srv GPU 2): env/losses/timing PASS (~10.6 s/ep steady
> ⇒ ~6h/seed @2000 ep) but **pct_hold FAILED** — 94.7→99.97%, policy entropy 0.19→0.005, visit
> entropy 0.0009 (the 821 collapse signature at epoch 40). Diagnosis: hold-bias 3.0 was tuned
> under a PERFECT value (831-EXT); with an untrained value the 3.0 logit gap (Gumbel σ≈1.28)
> makes all K candidates ~90%-HOLD clones — search can't diversify, the self-play book never
> grows, dithered-book has no book to dither. **BATTERY NOT LAUNCHED on a failed criterion.**
> Pre-registered 3-arm diagnostic (80 ep, GPUs 1/2/3): A hb0.0/d1.0 · B hb1.0/d1.0 · C hb3.0/d2.0.
> Arm PASS = ep80 pct_hold <99 AND < its ep40 value AND policy_entropy ≥0.05. Selection among
> passers: pct_hold ∈ [60,95] preferred; tie → closest to carried scheme (C > B > A). All arms
> fail ⇒ STOP, no battery — structural finding (exploration under learned values needs its own
> design, e.g. proposal-bias annealing), escalate to the pathway.

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
(portfolio PnL). **D2 carry (operator: promising, 2026-07-07):** the reward head must be
ACTION-CONDITIONAL — the Stage-B head was measured action-blind (60× variance gap between state
and action contributions). The dynamics/reward nets take the sampled action VECTOR as a real input
(embedded, not a scalar index); verify post-training that reward predictions actually move with
the action input (variance-attribution probe), or the search is planning over a reward model that
ignores what it does. Thread `--vector-action` through `train_jax_fast.py` / config / env_factory /
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

## Pathway wiring (2026-07-08)
This build is the SPINE of `docs/pathway.md`, not the whole pathway. Rung experiments interleave
the stages: Stage 1 → TASK_830 (E0 corridor) → Stage 2 → TASK_831 (E1/E2 search probe) →
Stage 3 (E3 probe inside) → TASK_832 (E5 imprint) → Stage 4. Code may land ahead of verdicts;
GPU spend may not — Stage-3 training launch waits on 831's E1 PASS.

## Factory notes (dark-factory execution contract, added 2026-07-07)
- **Risk: high** at every stage (touches the canon env/search/training path) → full high-tier panel.
- Queue rows = one stage each (`tasks/QUEUE.md`); banner per stage, do not roll stages together.
- Verification data = `fixtures/trading_fixture.db` (51MB, full daily panel; regen
  `fixtures/make_fixture.py`) — never the 24GB `trading_data.db`, never GPU-hours in the loop.
- Canon gate per stage: with `--vector-action` OFF, existing behavior bit-identical (hash a short
  scalar-path rollout before/after and diff). Stage-1 equivalence test: one `step_vector` ==
  the equivalent sequence of scalar `step`s on the same state (allowing for the documented
  simultaneous-rebalance fee/margin semantics — assert and document any intended divergence).
- Stage-4 (real-data training/eval) is NOT a factory job — scheduled server run, gated in the queue.

## Relation to the closed planner
TASK_821 closed the SCALAR planner cleanly. This does not reopen that — it changes the action space,
a different architecture. If Stage 4 beats the linear book, the planner returns on the RIGHT
formulation; if it still can't, the closure extends to vector actions too and the answer is truly
architectural, not action-bandwidth. Either way it's the honest test the campaign never ran.

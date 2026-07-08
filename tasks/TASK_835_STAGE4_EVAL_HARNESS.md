# TASK_835 — Stage-4 faithful vector eval harness (checkpoint → corridor scorecard)

**Status**: READY 2026-07-08. **Size: M. Risk: medium** (it produces the numbers the promotion
decision reads — fidelity errors here are verdict errors; zero canon diffs required).

## Why
The Stage-3 battery (running) has NO in-training held-out signal — vector mode fail-closed
skips the legacy eval (panel finding: it routed vectors through an `action_points`-capped env).
Stage-4's pre-registered PASS (beat the engine ref, no HOLD collapse, BEAR-positive) therefore
needs a faithful evaluator that does not exist yet. This is the last missing instrument between
"training done" and "verdict".

## Scope — `experiments/835_stage4_eval/eval_vector.py` + SPEC + fixture smoke
1. **Load** a training checkpoint (same config-then-`trainer.load_checkpoint` pattern as
   TASK_834; snapshots `jax_ckpt_fast/<name>/snapshot_ep<N>/` or final).
2. **Deterministic full-timeline rollout** through the REAL vector env (`step_vector`,
   `create_jax_env_historical` panel — the SAME panel construction as training; the 832
   panel-consistency rule applies here too): search with `deterministic=True`, action =
   `visit_probs` argmax (Stage-2 carry-note; `vector_gumbel_scale_eval` is reserved/unwired —
   do not touch it). No exploration, no buffer, pure evaluation pass.
3. **Scorecard** (the 830/831 shapes): net@21 AND gross Sharpe (both rollouts — fee-aware
   trajectories differ), full window + the four 831-EXT regime windows, position stats
   (pct_hold per asset, avgL/avgS — the HOLD-collapse counters), final PV, corridor position
   vs [floor +0.424, ceil +29.698] and the engine ref +0.751.
4. **Output JSON** per checkpoint + a compact comparison table over a checkpoint list
   (mid/late snapshots × seeds — the battery emits every 100 epochs).
5. **Fixture smoke**: untrained model on the fixture db end-to-end; PASS = runs clean, emits
   the full scorecard schema, and the untrained policy's numbers land near-noise (no threshold
   applied — plumbing proof, not verdict).

## Gates
Fixture smoke green (CPU); zero diffs outside `experiments/835_stage4_eval/`; selftest 7/7;
scorecard fields match the LEDGER corridor convention (one row emittable per run).

## Out of scope
Applying the Stage-4 PASS verdict (planner/scheduled run, per the gate ladder); walk-forward
unseen-years design (R4, separate); any edits to trainer/env/search.

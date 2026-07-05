---
status: accepted
date: 2026-04-23
deciders: Lukas
---

# 0004. Oracle modes — topk_path as default, dense_sign retained as diagnostic tool

## Status

Accepted.

## Context

The Hindsight Oracle (`mcts_trader/jax_agent/path_solver.py`) provides labels for Phase 1 supervised training. Two formulations are implemented:

- **`topk_path`** (default): `greedy_lookahead_path(k=4, n=64)` — greedy near-optimal path through the next 64 steps, allowed to act on at most k=4 assets. By construction this emits **~93.75 % HOLD** labels (4 of 64 assets active).
- **`dense_sign`** (TASK_739): per-asset sign of the cumulative forward return over a horizon, with a dead-zone ε around zero. Trade density is controlled by ε; calibrated empirically to ~13 % non-HOLD via `tools/calibrate_739_epsilon.py` (ε = 7.5e-3 for our 1-min crypto data).

The oracle's class-imbalance was identified as a contributing cause of the 0.57 dir_acc plateau (`lore/notes_2026-04-19.md §1` — encoder direction audit). The `dense_sign` formulation was built to test "is the oracle the bottleneck?" — same encoder, denser per-asset directional supervision.

Empirical result (`lore/notes_2026-04-24.md §62`): TASK_739 v2 with calibrated ε produced Alpha **+1.68 %** vs TASK_736 reference **+3.87 %** — within the ±5–6 pp 6-period noise band. dir_acc 0.570 vs 0.574 — same ceiling.

The first attempt with the spec's default ε=1e-4 (`lore/notes_2026-04-24.md §59`) was pathological: with that ε the cumulative return crossed zero so often that per-asset labels alternated near-randomly, the BC gradient averaged to zero, and `policy_loss` locked at `log(5)` for 20 epochs. ε calibration is mandatory before using this mode.

## Decision

`topk_path` remains the default Phase-1 oracle. `dense_sign` is **retained as a validated diagnostic tool**, not promoted to default.

CLI surface (preserved, do not silently remove):
- `--oracle-mode {topk_path,dense_sign,hybrid}` — selects the oracle.
- `--sign-epsilon` — dead-zone for `dense_sign`. Must be calibrated per-dataset (`tools/calibrate_739_epsilon.py`); the spec default of 1e-4 is pathological for our data.

## Consequences

Gain:
- `dense_sign` exists as a clean ablation knob: any future "is the oracle the bottleneck?" question can be answered in one training run instead of weeks of speculation.
- The benchmark equivalence (within noise) of the two oracle modes is itself a significant finding — it pins the 0.57 dir_acc ceiling as upstream of oracle formulation (architecture/information, not labels).

Give up:
- Two oracle code paths to maintain.
- The ε calibration step is operationally fragile — every new dataset/horizon combination needs `calibrate_739_epsilon.py` re-run.

## Alternatives considered

- **Replace the oracle with `dense_sign` outright.** Rejected — same benchmark within noise, no win to justify the change. Keeping `topk_path` as default preserves all prior baselines' interpretability.
- **Drop `dense_sign` after the v2 result.** Rejected — it is the only orthogonal-axis test against an "oracle is wrong" hypothesis; removing it would force a re-implementation if the question arose again. Hard rule from `lore/notes_2026-04-24.md §68`: "DO NOT silently remove `--oracle-mode dense_sign` or its calibration."

## References

- `lore/notes_2026-04-19.md §1` — encoder direction audit identifying oracle class imbalance.
- `lore/notes_2026-04-23.md §55` — TASK_739 implementation.
- `lore/notes_2026-04-24.md §59` — ε=1e-4 pathology and kill.
- `lore/notes_2026-04-24.md §61–§62` — calibration via `tools/calibrate_739_epsilon.py`, v2 result.
- `lore/notes_2026-04-24.md §68` — hard rule preserving the mode.
- `tasks/TASK_739_DENSE_SIGN_ORACLE.md`.
- `mcts_trader/jax_agent/path_solver.py`, `tools/calibrate_739_epsilon.py`.

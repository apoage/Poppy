# TASK_764 — Oracle k-Reformulation (Phase 1 Re-Imprint)

**Status**: DRAFT — operator-approval pending. Activates if TASK_762 + TASK_763 close null (per `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` Direction 3).
**Created**: 2026-05-20.
**Estimated wall**: ~35h server (single-GPU bottleneck on Phase 1).

## Hypothesis

`project_bull_blindness.md` traces bull-regime failure to k=4/64 oracle class imbalance: oracle picks top-4 assets out of 64 universe over horizon H, so "long top-4" labels are rare in broad bull regimes where many assets rally simultaneously. Encoder learns to predict the wrong class distribution.

Increasing k (k=8/64 or k=12/64) increases label density and reduces class imbalance, which should improve bull-regime classifier Bal.Acc AND give the planner access to a richer action space (more assets pre-labeled as profitable longs).

**Falsifiable in two stages**:
- Phase 1 sanity gate: k=8 imprint must lift bull-regime classifier Bal.Acc by ≥ +5pp on Phase 1 metrics. Otherwise hypothesis is dead before Phase 2.
- Phase 2 promotion gate: paired Δα ≥ +3pp under eval-MCTS sims=16 joint.

## Code changes required

1. **`mcts_trader/jax_agent/path_solver.py`**: parameterize `k` (currently hardcoded 4). Pre-implementation read to confirm where k lives.
2. **`train_easy.py`**: add `--oracle-k N` CLI flag for Phase 1 imprint.
3. **`train_phase2.py`**: ensure Phase 2 inherits k from the Phase 1 imprint metadata (or accept explicit `--oracle-k` matching the imprint).
4. **Oracle re-precompute**: if oracle labels are precomputed (check `data/` for cached label files), rebuild for k=8 and k=12.

Estimated code change: ~30 lines + ~2-4h oracle re-precompute wall.

## Variants

**Phase 1 — k-sweep at imprint level**:

| cell | oracle_k | Phase 1 seeds | wall |
|---|---:|---|---:|
| 764-k4 (canon) | 4 | reuse `jax_ckpt_easy_phase1_xs32/easy_medium/` | 0 |
| 764-k8 | 8 | {7,13,23} | ~12h |
| 764-k12 | 12 | {7,13,23} | ~12h |

**Phase 1 sanity gate**: bull-regime Bal.Acc lift ≥ +5pp on at least one k value. If neither k=8 nor k=12 passes, STOP — hypothesis is dead, save the ckpts as data points but don't run Phase 2.

**Phase 2 — a1b1 Stage-B on best-k Phase 1 imprint**:

| cell | Phase 1 base | Phase 2 seeds |
|---|---|---|
| 764P2-canon | k=4 reuse | reuse 760 a1b1 sims=32 baselines |
| 764P2-best | best-k from Phase 1 sanity gate | {7,13,23} |

## Promotion gate

- Phase 1 sanity: bull-regime Bal.Acc lift ≥ +5pp.
- Phase 2: paired Δα ≥ +3pp under eval-MCTS sims=16 joint AND no seed < −5pp.
- 5-seed extension if Phase 2 3-seed clears.

## Risk

Higher than 762/763. Touches Phase 1 — known good imprint becomes the variable. **Worst case**: new Phase 1 imprint is BETTER on bull but WORSE on other regimes (sideways, bear) and we regress globally. Mitigation:
1. Run Phase 1 sanity diagnostic across all regimes, not just bull. Reject k value if it loses ≥ 3pp Bal.Acc on any non-bull regime.
2. Bull-only ablation: train Phase 1 on bull-period data only at each k value before committing to full Phase 1 retrain.

## Why this comes AFTER reward shaping and entropy reg

Phase 1 retrain breaks the imprint contract — every prior experiment used the same Phase 1 ckpt. Re-imprinting means every prior paired-Δα number ceases to be directly comparable. This is the most disruptive single-component change. Save it for after the CLI-side mechanism class is fully exhausted.

## Bloom-arc relevance

Bull-blindness has been documented since the early campaign (`project_bull_blindness.md`) but never directly tested. If k-reformulation works, that's a strong "we knew the failure mode for months, finally tested it, here's the result" arc — publishable independent of the campaign verdict.

## References

- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` Direction 3.
- `lore/notes_2026-05-20.md` §151.
- `~/.claude/projects/-home-lukas-Trader/memory/project_bull_blindness.md` — primary evidence base.
- `mcts_trader/jax_agent/path_solver.py` — oracle implementation, k parameterization point.
- `train_easy.py` + `tools/orchestration/e-dojo_v1.py` — Foxglove Phase 1 entry points.

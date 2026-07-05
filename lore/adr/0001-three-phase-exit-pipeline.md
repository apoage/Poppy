---
status: accepted
date: 2026-01-13
deciders: Lukas
---

# 0001. Three-phase Expert Iteration pipeline (Foxglove → Poppy)

## Status

Accepted.

## Context

End-to-end MuZero (learning representation, dynamics, policy, and value simultaneously from a cold start) is "Hard Mode": it requires very large batch sizes to stabilize gradients and frequently enters non-stationary "death spirals" (bad model → bad search → bad data → worse model). On a single RTX 3090 (24 GB VRAM) this is not tractable.

We have a Hindsight Oracle (`mcts_trader/jax_agent/path_solver.py`, see `TASK_119_OPTIMAL_PATH_SOLVER.md` and `TASK_125_CAUSAL_EXPERT_IMPRINTING.md`) that can produce near-optimal action labels on historical data after the fact. This makes a supervised pre-training stage possible.

The design is documented at `docs/easy_mode.md` ("Easy Mode: System 2 Trading on a Single GPU"), which formalises the three-phase split.

## Decision

We split agent training into three sequential phases, separated into named components:

1. **Phase 1 — "The Cheat"** (Foxglove, `train_easy.py` + `easy_trainer.py` + `tools/orchestration/e-dojo_v1.py`). Supervised behavioural cloning of the Hindsight Oracle. Produces a strong policy and value prior. No MCTS in this phase.
2. **Phase 2 — "The Dream"** (Bella, `diff_teacher/`). A frozen world model — a market generator trained on real data — serves as the dynamics simulator that MCTS rolls out against. Decouples representation from rollout policy.
3. **Phase 3 — "The Thought"** (Poppy, `mcts_trader/`). MuZero+TRM agent runs MCTS at inference time on top of the Phase-1 priors and the Phase-2 dynamics. Self-play improvement on top of the imprinted prior.

Each phase has clearly bounded artefacts (Phase-1 checkpoint, Phase-2 frozen dynamics, Phase-3 trained policy) and can be debugged in isolation.

## Consequences

Gain:
- Stable, large-batch supervised gradients in Phase 1 — no death-spiral dynamics.
- The hard problem (dynamics learning) is isolated into Bella where it can be evaluated as a standalone generative model, not buried inside an RL loop.
- Each phase produces a benchmarkable artefact, making regressions traceable.
- Fits the single-GPU memory budget (24 GB).

Give up:
- We cannot reach behaviour beyond what the oracle + Bella can express, unless Phase 3 self-play breaks past the imprint.
- More moving parts: three trainers, three checkpoint conventions, three configs.
- The Phase-1 oracle becomes load-bearing — its formulation directly shapes what the Phase-1 policy can learn (see ADR-0004).

## Alternatives considered

- **End-to-end MuZero from scratch.** Rejected: VRAM-infeasible at the batch sizes needed for stability on a 3090; documented "death spiral" risk.
- **Behavioural cloning only (skip Phases 2/3).** Insufficient — BC alone cannot improve past the oracle, and the oracle is itself bounded (see ADR-0004).

## References

- `docs/easy_mode.md` — full design.
- `tasks/TASK_666_EASY_MODE_BRANCH.md`, `tasks/TASK_666_IMPLEMENTATION_PLAN.md`, `tasks/TASK_666_DEPENDENCY_MAP.md` — implementation planning.
- `tasks/TASK_125_CAUSAL_EXPERT_IMPRINTING.md` — original imprint motivation.
- `lore/notes_2026-04-22.md §49` — confirmation that the ExIt loop must close (value training enabled) for Phase 2 to actually function.

> **STATUS (2026-07-08): factory half SHIPPED.** `experiments/831_search_probe/` runner + SPEC +
> fixture smoke landed; panel-green (E1 gate logic verified sound — floor-vs-search apples-to-apples).
> Fixture smoke: floor net@21 +5.59, **E1=PASS** (8/9 scheme×K reach floor), cheapest reacher
> dithered-book|K8. **E1 measures PROPOSAL CAPACITY** — PASS is necessary-but-not-sufficient for
> Stage-3; the real deliverable is the E2 scheme×K ranking. **Remainder:** the load-bearing E1/E2
> verdict = SCHEDULED single-GPU sweep (≤1 GPU-day, verified-idle GPU) — NOT a factory-loop job;
> its **real E1 PASS gates the 828 Stage-3 TRAINING launch** (pathway R1). Kill logic: E1 FAIL across
> all E2 schemes ⇒ proposal mechanism is the defect ⇒ Stage-3 does not launch, planner respecs sampler.

# TASK_831 — True-model search probe: can Sampled-MuZero FIND the book at all? (E1/E2, R1)

**Status**: GATED on 828 Stage 2 banner (needs the sampled-vector search machinery) + TASK_830
corridor recorded.
**Pathway**: `docs/pathway.md` rung R1. Hypothesis: H1 (search side).
**Size: M. Risk: medium** (no canon edits; wires Stage-2 search to the simulator). Budget:
≤1 GPU-day total incl. ablations. This verdict GATES the 828 Stage 3 training launch.

## Purpose
Separate "search can't find it" from "model can't learn it" BEFORE training spends server-weeks.
Run the Stage-2 Sampled-MuZero search with the ENV SIMULATOR as dynamics and TRUE reward
(no learned nets; value = truncated rollout return or the 830 oracle bound as a heuristic —
pre-register which in SPEC.md).

- **E1 (capacity)**: on held daily-panel slices, does K-sample search with a perfect model
  recover ≥ corridor-floor (linear book) performance? PASS/FAIL pre-registered vs the 830
  floor. If a perfect-model search can't reach the floor, no learned model will.
- **E2 (sampling scheme ablation)** — the design input Stage 3 actually needs:
  - independent per-asset draws from the factored policy (the 828 Stage 2 default),
  - mixture with a linear-rule prior (proposal distribution seeded by the working book),
  - dithered-current-book (local moves around the held portfolio),
  - K ∈ {8, 16, 32}.
  Output: which scheme×K reaches the floor cheapest. That choice is carried into Stage 3.

## Deliverables
- `experiments/831_search_probe/SPEC.md` + small runner importing Stage-2 search + env
  (no monster-file edits). Fixture smoke = factory; GPU sweep = scheduled single-GPU run
  (verified-idle GPU, never blind `--gpu 1,2,3`).
- LEDGER lines per scheme×K with corridor position. E1 verdict named explicitly.

## Kill logic (from pathway)
E1 FAIL across all E2 schemes → the action-proposal mechanism is the defect; 828 Stage 3 does
NOT launch; planner respecs the sampler (new structural hypothesis) first.

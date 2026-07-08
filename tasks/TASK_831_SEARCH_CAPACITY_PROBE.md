> **STATUS (2026-07-08): COMPLETE — real E1/E2 OFFICIAL.** Full-panel sweep done same day
> (workstation CPU under safe_run — GPU unneeded; ~7 min wall; 279 assets, ~1670 decisions,
> seeds {0,1,2}). **E1 = PASS, 9/9 reach floor ⇒ Stage-3 TRAINING GATE OPEN.** E2 carried
> scheme: **dithered-book|K16** (+5.29 net@21, per-seed [4.18, 5.94, 5.76]); budget fallback
> dith|K8. linear-prior pins search to the rule's neighborhood (≤+2.19, near-zero seed
> variance) — rejected as proposal scheme. Perfect-model search +5.29 vs rule ref +0.75:
> wide learnable corridor. Proposal-capacity caveat stands. LEDGER 831/E1-REAL + E2-REAL;
> artifacts `runs/831_search_probe/`.

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

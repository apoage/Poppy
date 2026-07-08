# Handoff 2026-07-08 — Dark-factory run: the vector-action spine + R0/R1 instruments

Dark-factory implementer run on `master`. Four tasks shipped, each flag-gated (canon
bit-identical, default OFF), each verified by a risk-tiered adversarial panel, each committed
and tagged atomically. This is the SPINE of `docs/pathway.md` (H1 breadth): the vector env
step, the Sampled-MuZero search over it, and the two instrument rungs that will score it.

## Shipped this run

| Commit | Tag | Task | What landed | Verification |
|--------|-----|------|-------------|--------------|
| `3d4be4b` | `task828-stage1` | 828 Stage 1 | `step_vector` — per-asset action vector `(n_assets,)∈{0..4}` as one simultaneous portfolio rebalance; `EnvConfig.vector_action=False` gate | Full high-tier panel green; 13 tests; canon scalar rollout hash bit-identical |
| `bea9a52` | `task830-e0-corridor` | 830 E0 (factory half) | `experiments/830_corridor/` — replay 819 linear book through `step_vector` (floor) + `path_solver` oracle bound (ceiling) | Medium panel green; runner correct, zero canon diffs |
| `11fbc53` | `task828-stage2` | 828 Stage 2 | `SampledMuZeroVectorSearch` (`mcts_trader/jax_agent/sampled_search.py`) + `VectorActionEmbedding`/`dynamics_vector` in the model; `use_vector_action=False` gate; `mctx_search.py` untouched | Full high-tier panel green; Sampled-MuZero math verified (β=π); param tree byte-identical OFF; search uses its value net |
| `03b3d97` | `task831-e1e2-probe` | 831 E1/E2 (factory half) | `experiments/831_search_probe/` — true-model sampled search (env as perfect model), E1 capacity check + E2 scheme×K ablation | Medium panel green; E1 gate logic sound; zero canon diffs |

All four: `python -m trader selftest` 7/7 throughout; fixture-only (`fixtures/trading_fixture.db`),
never the 24GB `trading_data.db`, never GPU in-loop.

## The two decision-changing findings

1. **The corridor-floor gap is a REPLAY-MAPPING artifact, NOT a `step_vector` bug.** TASK_830's
   fixture floor (net@21 +0.42) misses the +0.97 engine bar — the implementer first filed this
   against Stage 1. The panel overturned it: an independent from-scratch cash-account simulator
   reproduces `step_vector` to 3–4 sig figs (finalPV to the dollar). The gap is dominated by the
   runner's fixed `0.5/A` weighting vs the 819 book's renormalized `1/n_present`, plus faithful
   hold-shares de-leveraging in BEAR22 vs the book's daily constant-dollar rebalance. The env adds
   only +0.013 over the pure mapping. **The env is clean; the "reproduce the bar" question is a
   mapping choice** → seeded **DECISIONS D6**.

2. **Sampled-MuZero is on-policy, so the visit-count correction is exactly a uniform-per-slot
   prior.** Stage 2 samples K action-vectors from the factored policy marginals themselves (β = π),
   so the importance weight π/β = 1 and the empirical β̂ = n_a/K falls out of uniform 1/K slots +
   duplicate accumulation. Verified empirically (20k-sample draw matches the factored product to
   <0.7%). The search demonstrably uses its value net (visits follow value; selection flips when
   Gumbel exploration is dialed down). β̂ is diagnostic-only, never fed to mctx.

## Where it stands (pathway R0→R1 built, R1 verdict pending real data)

```
R0 instrument  E0 corridor (830) ...... FACTORY HALF SHIPPED · real-panel numbers SCHEDULED
R1 search      E1/E2 probe (831) ....... FACTORY HALF SHIPPED · real E1/E2 verdict SCHEDULED
                                          └─ real E1 PASS GATES 828 Stage-3 training launch
R2 learnability E3 (in Stage 3) / E5 (832) ... BLOCKED on Stage 3 banner + real E1 PASS
R3 full loop   828 Stage 3 training ..... CODE may land ahead; TRAINING waits on real E1 PASS
R4 graduation  Stage 4 eval ............. SCHEDULED, gate ladder
```

Fixture SMOKE signals (NOT verdicts): 830 floor net@21 +0.42 (mapping-limited, see D6);
831 E1 = PASS (8/9 scheme×K reach floor), cheapest reacher **dithered-book|K8**, and E1 reframed
as a **proposal-capacity** test (PASS necessary-but-not-sufficient; the real deliverable is the
**E2 scheme×K ranking** carried into Stage 3).

## The critical path is now SCHEDULED runs, not factory-loop code

The next load-bearing actions are the two scheduled real-panel jobs — CPU/single-GPU, verified-idle
GPU, never blind `--gpu 1,2,3`, server-preferred (gpu-server), in tmux:
1. **830 real corridor** — `experiments/830_corridor/corridor.py --db trading_data.db` (read-only).
   Records the real `[floor, ceiling]` per window/cost tier in `experiments/LEDGER.md`. Honor D6:
   report BOTH the hold-shares and constant-dollar-weight floors (or label which is official).
2. **831 real E1/E2 sweep** — `experiments/831_search_probe/search_probe.py --db trading_data.db
   --max-decisions None` + the K∈{8,16,32} × 3-scheme grid. **Its real E1 PASS is the R1 gate that
   unlocks 828 Stage-3 training**, and its E2 ranking picks the Stage-3 sampler.

## Open decisions (walls)

- **D6** (seeded this run): which book defines the official E0 corridor floor — env-native
  hold-shares (+0.42) or constant-dollar-weight daily-rebalance (matches +0.97)? Recommendation:
  report both on the scheduled run, grade the agent against the hold-shares in-env floor.
  **Deeper flag for Stage 2/3:** the discrete per-asset action space {sell/hold/buy/limit×2} cannot
  express an exact continuous weight target → exact +0.97 reproduction may be INEXPRESSIBLE in-env.
  Relevant to what "beat the bar" can mean.

## Next factory work (when unblocked)

- **TASK_829 (retire Foxglove)** — unblocked filler (D5 decided). Decouple `train_phase2.py:484` +
  `benchmark_easy.py:41` imports, then delete the Foxglove files. **Do it BEFORE Stage 3** threads
  `use_vector_action` through `train_phase2.py` (import-surface collision). Left for a fresh context —
  medium-risk import-retargeting deletion deserves one.
- **828 Stage 3** — only after the real E1 PASS. Carries the panel-surfaced wire-up notes: mask via
  −inf per-asset logits (`sampled_search` has no `action_mask` arg); build the 320-wide asset-major
  factored head; deterministic eval selects via `visit_probs` argmax / lower `gumbel_scale`; the
  action-conditional reward head (D2) + E3 variance-attribution probe is the hard gate inside Stage 3;
  interior depth-reuse of root K vectors is a documented follow-up (per-node resampling).

## Process note (for the next factory run)

Two verify panels observed the working-tree `muzero_model.py`/env file **oscillating** — canon-parity
subagents were doing `git show HEAD > <tracked file>` then restoring, racing each other and risking
loss of uncommitted work (one judge's `git stash` was correctly blocked by the push-guard). Fix
applied mid-run: (a) commit each stage's work atomically the moment it's green (done — all four are
committed); (b) the 831 panel prompts explicitly **forbade tracked-file overwrites** for parity
checks (`git show HEAD:path > /scratchpad/...` and read that) — no oscillation observed after.
Bake that instruction into all future canon-parity verify prompts.

## Unrelated footguns noticed (not fixed — planner cleanup candidates)

- `MuZeroNetworks.action_space_size` default 192 (=64×3) vs production 320 (=64×5) — overridden at
  every call site, but a footgun for anyone relying on the default.
- `tools/tests/test_jax_env.py` `TestLimitOrders`: 2 stale `order_reserved>0` assertions contradict
  the current lazy no-reservation env (fail identically on HEAD — pre-existing, not ours).

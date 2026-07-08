> **STATUS (2026-07-09): SHIPPED as `task838-pernode-resample`.**
> Flag-gated `per_node_resampling` (config `vector_per_node_resample` / `--vector-per-node-resample`,
> default OFF): interior nodes resample K vectors from the factored policy at THEIR latent (embedding
> pytree carries per-node candidates; β=π exact per node, Hubert 2021). **OFF bit-identical**: pre-edit
> pin `22b0c7f5…f23e` MATCH (a panel judge independently reproduced the pin from the pre-edit commit —
> genuine provenance); Stage-3 scalar pin still MATCH; 11 existing search tests untouched+green; OFF
> branch byte-identical by construction. Full high-tier panel: ON semantics proven by runtime tree
> capture; 1 MAJOR fixed (835 eval now threads the flag — no silent train/eval operator asymmetry);
> evidence-integrity verified after a self-caught mid-build fabrication (all numbers bit-reproduce).
> **Acceptance FINDING (pre-registered read NOT held)**: constant-policy true-model probe, equal K:
> net@21 OFF +6.421 vs ON +3.634 (Δ −2.787, 0/3 seeds) — fresh random per-node portfolios hurt under
> an uninformed policy. The informed-proposal mechanism the flag exists for needs a TRAINED policy:
> **any ON run must re-measure first (scheduled Stage-3/4 stream). Default stays OFF.**

# TASK_838 — Per-node action resampling in the sampled vector search (Stage-2 named follow-up)

**Status**: READY 2026-07-08 (take AFTER 836/837 — lowest urgency of the batch).
**Size: M. Risk: high** (touches `mcts_trader/jax_agent/sampled_search.py` — canon-adjacent;
full flag-gate + parity discipline).

## Why
The shipped `SampledMuZeroVectorSearch` reuses the ROOT's K sampled action vectors at every
interior node — a documented Stage-2 simplification (828 banner carry-note 4). Consequence:
depth-2+ planning can only consider root-proposed portfolios, which caps H4 (multi-step
cost-aware netting) exactly where deeper search should pay. This is the panel's own named
follow-up, not a knob.

## Scope
1. Flag-gated `per_node_resampling` (config + `--vector-per-node-resample`, default OFF =
   today's root-reuse behavior BIT-IDENTICAL — parity test pinned like the Stage-1/2 gates).
2. ON: at each expanded node, sample K vectors from the factored policy at THAT node's latent
   (same dithered-book machinery, node's book state), with the Sampled-MuZero visit/prior
   correction kept exact (β=π per node — cite the Hubert et al. correction in the SPEC).
3. **Acceptance experiment (fixture, true-model — the 831 pattern)**: per-node ON vs OFF at
   equal K on the fixture probe; report net@21 at depth>1 sims. Pre-registered read: ON ≥ OFF
   (it may only tie at sims/depth used today — that is a finding, not a failure; record it).
4. JIT discipline: no dynamic shapes (fixed K per node), no `lax.cond`-in-vmap (JAX pitfalls).

## Gates
OFF-path parity hash == pinned; new tests alongside `tools/tests/test_sampled_search.py`
(do not modify existing 11); fixture acceptance probe runs green both ON and OFF;
selftest 7/7; full high-tier panel (canon-adjacent).

## Out of scope
Changing default behavior (stays OFF until a scheduled run shows ON wins), training-loop edits,
proposal-scheme changes (that is the phase-2 exploration question, separate evidence stream).

---
status: accepted
date: 2026-04-25
deciders: Lukas
---

# 0007. Per-asset scalar injection via FiLM, post-temporal-pool (TASK_741, codename Mint)

## Status

Accepted (designed, implementation pending at time of writing).

## Context

ADR-0005 establishes that the encoder caps at ~87 % Val Agree on the cheat-channel diagnostic (TASK_733). The hypothesised mechanism (`lore/notes_2026-04-24.md §65–§66`): per-asset scalar features fed in as broadcast input channels are partially averaged out by V3's minGRU temporal pool — a scalar replicated across 60 timesteps acts like a DC offset and recurrent attenuation eats some of it.

If true, then any future per-asset scalar feature (Bella's predicted sign, cross-sectional rank, regime tag, etc.) suffers the same dilution. The cheat-channel result quantifies the cost: ~13 pp of extractable signal lost between input and output even when the input *is* the answer.

Two injection alternatives were considered:
- **Concat + Dense** (the obvious fix): re-introduces dilution — the scalar is averaged into a larger vector.
- **Cross-attention from scalars into latent**: violates the no-attention-in-reasoning rule (ADR-0002).

FiLM (Feature-wise Linear Modulation) — `(1 + γ)·feats + β` where γ, β are produced by a small MLP from the scalars — applies per-feature multiplicative scaling, forcing each scalar to act on every feature dim explicitly. No averaging. No attention.

## Decision

Implement per-asset scalar injection as a **FiLM module post-temporal-pool**, on the per-asset features `(batch, n_assets, embed_dim)` at `mcts_trader/jax_agent/trm_encoder_v3.py:149` (after LN, before `global_proj`).

Spec details (`tasks/TASK_741_PER_ASSET_INJECTION_FIX.md`):
- New `FiLMProjector` Flax module, 2-layer MLP from scalars `(B, A, n_scalars)` → `(γ, β)`.
- **Zero-init final Dense** so default behaviour is byte-identical to the current encoder. Opt-in via `--per-asset-injection`.
- Cross-sectional channels 27–31 (TASK_732) stay as input channels for the first cycle — additive A/B against existing baseline. Removal is post-validation cleanup, not a precondition.

Validation gates:
1. Re-run TASK_733 cheat with `--per-asset-injection`. Spec gate: Val Agree **≥ 95 %**. **Binary falsifier** — if it stalls at 87 % again, the hypothesis is dead and the design is abandoned.
2. Re-run TASK_739 dense_sign with `--per-asset-injection`. Spec gate: dir_acc > 0.62.
3. Benchmark Gate-2 winner.

## Consequences

Gain (if Gate 1 passes):
- Architectural half of the 0.57 ceiling (ADR-0005) becomes movable. Information-side pivots (Bella distillation, richer inputs) are no longer bottlenecked at 87 %.
- A general-purpose mechanism for any future per-asset scalar — solves the same problem the cheat channel surfaced.

Give up:
- ~150–200 LoC of new code in a load-bearing module (`trm_encoder_v3.py`).
- One open design question deferred (`lore/notes_2026-04-24.md §66`): cross-asset attention at line 146 might already be smoothing what FiLM is meant to sharpen. The first run uses canonical post-LN ordering; an "inject before cross-attn" experiment is a follow-up if Gate 1 marginally fails.

## Decision uncertainty

The hypothesis has not yet been tested. Gate 1 is binary and cheap (~6 hr training); the design is therefore documented here as the agreed-on next move, with the explicit understanding that **a Gate-1 failure kills this ADR cleanly** and the project would need to revisit the architectural half of ADR-0005's decomposition.

## Alternatives considered

See Context above (concat+Dense, cross-attention) — both rejected on principled grounds before the spec was written. No empirical alternatives have been tested.

## References

- `lore/notes_2026-04-24.md §66` — TASK_741 (Mint) design rationale.
- `lore/notes_2026-04-24.md §65` — TASK_733 cheat result that motivates this fix.
- `lore/notes_2026-05-05.md §69` — A1+B1 stacked test confirming label/loss-side reformulations are exhausted; escalation to TASK_741 triggered.
- `tasks/TASK_741_PER_ASSET_INJECTION_FIX.md` — full 231-line spec.
- ADR-0002 (no attention in reasoning), ADR-0005 (ceiling decomposition).

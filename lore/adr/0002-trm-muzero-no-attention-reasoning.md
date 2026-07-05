---
status: accepted
date: 2026-01-15
deciders: Lukas
---

# 0002. TRM + MuZero as the planner architecture; no attention in reasoning or dynamics

## Status

Accepted.

## Context

The agent has to plan over 320 actions (5 action types × 64 assets) per timestep with a strict latency budget (live trading). Standard transformer-style backbones blow up VRAM with sequence-length-squared attention and are hard to fit alongside MCTS state caching on a single 3090. There was an early Pytorch transformer-encoder + wrapper agent (now under `mcts_trader/agent/`, marked LEGACY) that did not scale well.

The Tiny Recursive Model (TRM) family offers MLP-based recursion (Dense + LN + GELU in H×L cycles) as a substitute for transformer-stack reasoning, with much lower memory cost.

## Decision

The planner is **TRM + MuZero** in JAX/Flax (`mcts_trader/jax_agent/`):

- **Encoder**: a single cross-asset attention block — the *only* attention in the model. Produces per-asset features and a global latent.
- **Reasoning**: pure MLP loops in H×L cycles. **No attention.**
- **Dynamics**: a 2-layer MLP for MCTS state prediction. **No attention.**
- **Search**: Gumbel MuZero via Google's `mctx` library.
- **Value head**: PopArt-normalised, 401 categorical bins over `[-10, +10]`.

The "no attention in reasoning or dynamics" rule is hard. It is enforced in `mcts_trader/CLAUDE.md` and reaffirmed in TASK_741's design rationale (FiLM injection chosen over cross-attention specifically to respect this rule — see `lore/notes_2026-04-24.md §66`).

## Consequences

Gain:
- Reasoning + dynamics fit comfortably alongside MCTS state caches in 24 GB VRAM.
- MCTS rollout cost is bounded by MLP-not-attention inference, keeping search latency low enough for live trading.
- `mctx` provides battle-tested Gumbel MuZero search out of the box.

Give up:
- The encoder is the only place the model can do cross-asset reasoning. If a feature genuinely requires multi-step attention reasoning between assets, this architecture cannot represent it.
- PopArt is mandatory: disabling it forces a manual rescaling of the value support range (raw range can hit `[-750, +750]`).

## Alternatives considered

- **Pure transformer planner.** Rejected for VRAM and latency reasons; the legacy PyTorch wrapper was the proof.
- **Cross-attention from scalar features into the latent** (suggested in early TASK_741 design). Rejected because it conflicts with the no-attention-in-reasoning rule. FiLM modulation chosen instead (`lore/notes_2026-04-24.md §66`).

## References

- `mcts_trader/CLAUDE.md` — "Architecture (do not suggest transformer/attention changes to reasoning or dynamics)".
- `mcts_trader/jax_agent/muzero_model.py`, `trm_flax.py`, `mctx_search.py`.
- `lore/TRM+MCTS Agent.md` — original design notes.
- `lore/notes_2026-04-24.md §66` — TASK_741 design choosing FiLM over cross-attention to respect this rule.
- `tasks/TASK_741_PER_ASSET_INJECTION_FIX.md`.

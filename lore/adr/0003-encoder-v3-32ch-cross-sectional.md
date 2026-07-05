---
status: accepted
date: 2026-04-22
deciders: Lukas
---

# 0003. Encoder V3 (minGRU temporal pool, 32 channels, embed_dim=64, heads=8)

## Status

Accepted.

## Context

The encoder has gone through three measurable revisions:

- **V2** (`mcts_trader/jax_agent/trm_encoder_v2.py`): flatten the 60-step window into a single Dense, then cross-asset attention. Loses temporal ordering. `embed_dim=32`, `num_heads=4`.
- **V3** (`trm_encoder_v3.py`): minGRU temporal pool over the window before cross-asset attention. Preserves temporal ordering.
- **Channel count**: 27 channels (legacy) vs 32 channels (TASK_732 — adds five cross-sectional channels: per-asset rank-of-return inside the asset universe at each step, etc.).

Two empirical findings drove the consolidation (see `lore/notes_2026-04-19.md §22`, `lore/notes_2026-04-22.md §39`, §44, §46, and `lore/notes_2026-04-24.md §60`):

1. With the prior V2, 27-ch baseline, `dir_acc` was ceiling-bounded around 0.55–0.57. A series of TASK_727/728/728c experiments isolated this as upstream of the policy head — the latent did not carry enough per-asset direction.
2. Re-running the TASK_737 ablation with TASK_736's value-weight fix (ADR-0006) showed `legacy_27ch` costs **−21.27 pp Alpha** vs the 32-channel baseline (−17.40 % vs +3.87 %). The largest single regression of any ablation in the suite.

Phase 1 trade-agreement also moved with these changes: V2/27-ch ~60 %, V3/27-ch 76.7 %, V3/32-ch 83.3 % (`lore/notes_2026-04-22.md §49` table).

## Decision

The default encoder is **V3 minGRU temporal + 32 channels + embed_dim=64 + heads=8**. CLI: `--encoder-version 3 --embed-dim 64 --encoder-heads 8`. Channel count is implied by `n_channels` and the `use_cross_sectional` config.

The 27-channel path is preserved behind `--legacy-27ch` (TASK_738) for ablation only. It is *not* a candidate for production warmstart unless retraining the 27-ch baseline.

## Consequences

Gain:
- Phase 1 trade-agreement on the validation set climbs from ~60 % (V2/27) to 83 % (V3/32) — a real signal that more of the oracle's behaviour is being learned.
- The five cross-sectional channels are strongly load-bearing for trading skill: removing them costs ~21 pp Alpha (TASK_737 legacy_27ch run, `lore/notes_2026-04-24.md §60`).
- Temporal ordering is preserved, which matters for any future autoregressive or sequence-aware head.

Give up:
- Larger encoder (~2× params): more VRAM during Phase 1.
- 32-channel input means upstream feature pipelines must compute the cross-sectional channels; the data-prep code is more complex than the V2 era.
- Encoder ceiling on extracting per-asset scalar features remains (~87 % on the cheat diagnostic — see ADR-0005). Widening alone did not fix it.

## Alternatives considered

- **Per-asset latent routing (TASK_729a)**: route per-asset features around the 64→128 global bottleneck directly into the policy head. Tried; the dir_acc plateau followed it through (`lore/notes_2026-04-19.md §21`). Rejected — the bottleneck was not the global projection.
- **Wider V2 only (TASK_729b)**: widen `embed_dim` 32→64 in V2 without switching to V3. Tried; failed the gate in `lore/notes_2026-04-22.md §25`. Rejected.
- **Stay on V2/27-ch.** Rejected after TASK_737 legacy_27ch quantified the −21 pp Alpha cost.

## References

- `lore/notes_2026-04-19.md §18–§22` — encoder ceiling diagnostic, TASK_729a/b/c decision.
- `lore/notes_2026-04-22.md §24–§39, §44–§46` — V3 launch, TASK_730/731/732 results, ceiling reframing.
- `lore/notes_2026-04-24.md §60` — TASK_737 ablation including legacy_27ch (−21.27 pp Alpha).
- `tasks/TASK_729b_WIDER_ENCODER.md`, `tasks/TASK_729c_V3_TEMPORAL_ENCODER.md`, `tasks/TASK_732_CROSS_SECTIONAL_CHANNELS.md`, `tasks/TASK_738_LEGACY_27CH_ABLATION.md`.
- `mcts_trader/jax_agent/trm_encoder_v3.py`.

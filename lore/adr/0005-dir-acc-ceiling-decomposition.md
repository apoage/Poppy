---
status: accepted
date: 2026-04-25
deciders: Lukas
---

# 0005. The 0.57 dir_acc ceiling is decomposed ~50 % architectural, ~50 % informational

## Status

Accepted.

## Context

Across six orthogonal Stage-B configurations (`lore/notes_2026-05-05.md §69`, decomposing further `lore/notes_2026-04-24.md §60, §62, §65`), per-asset direction accuracy sits at **0.569–0.574** — a hard plateau:

1. Oracle: `topk_path` vs `dense_sign` — same dir_acc.
2. Encoder channels: 27-ch vs 32-ch — same dir_acc.
3. Head suite: ref / no_aux / no_rank / minimal (TASK_737) — same dir_acc.
4. BC on vs BC off — same dir_acc (TASK_733 A1+B1 stacked, `lore/notes_2026-05-05.md §69`).
5. ε calibration on `dense_sign` — same dir_acc.
6. Cheat channel: ground-truth future signs handed to the encoder via channel 32 — caps Val Agree at **87 %** (TASK_733 Phase-1 cheat result, `lore/notes_2026-04-24.md §65`).

The cheat-channel diagnostic (TASK_733) was the load-bearing experiment. The TASK_733 spec defines an answer matrix:
- Val Agree ≥ 90 % → information-bounded; pivot to richer inputs.
- 80–90 % → partial signal extraction; mixed.
- < 80 % → bug in cheat input or encoder.

Result: 87 % peak. Train-Val gap ~1–2 pp throughout. We landed in the partial-extraction band.

The interpretation (`lore/notes_2026-04-24.md §65`): even with the answer broadcast as channel 32, V3's minGRU temporal pool partially averages the DC offset out, capping the encoder at ~87 %. The remaining gap from 87 % to 100 % is the architecture's intrinsic ceiling. The 0.57 dir_acc seen in normal training therefore equals (architectural ceiling 0.87) × (information available in 60-step OHLCV) — i.e. roughly half the deficit is fixable by richer inputs alone, the other half needs architecture changes.

The literature SOTA on minute-bar crypto direction is 0.53–0.58, which matches the information-bounded half exactly.

## Decision

We treat the 0.57 dir_acc plateau as a **decomposed ceiling, ~50 % architectural and ~50 % informational**, not a bug. Future work is allocated against this decomposition:

- Architectural half → ADR-0007 (TASK_741 / FiLM injection, designed but not yet implemented at the time of writing).
- Informational half → Bella distillation (TASK_740, β path) and richer inputs from Iris-DEX (Tier D).

Hard rule (`lore/notes_2026-04-24.md §68`): cheat-channel checkpoints are diagnostic only. **Never warmstart a production Stage B from `jax_ckpt_easy_phase1_cheat/*`**; `tools/benchmark_easy.py` has a safety rail that refuses any checkpoint with `use_cheat_channel=True`. Do not bypass.

## Consequences

Gain:
- Quantitative target for downstream pivots. Bella distillation alone is now a known-bounded gain (≤ ~+15 pp dir_acc on top of 0.57 ≈ 0.65 best case) — not a magic bullet.
- TASK_741's gate is binary: cheat re-run with `--per-asset-injection` either reaches Val Agree ≥ 95 % or stays at 87 % and the FiLM hypothesis is killed.
- Bal.Acc is now known to be an unreliable proxy for trading skill (it rewards HOLD-dominance). Alpha and beats-random are the load-bearing signals (`lore/notes_2026-04-22.md §52`).

Give up:
- Any pivot that targets only one half of the decomposition has a known ceiling. Information-only pivots (Bella, longer windows) can move at most ~+5–8 pp dir_acc; architecture-only pivots (FiLM) face the same constraint from the other side. To break 0.65 dir_acc both halves must move.

## Alternatives considered

- **Treat the 0.57 plateau as a bug to find and fix.** Rejected after six orthogonal axes pinned it — no single config moved it.
- **Accept the ceiling and ship.** Rejected — the model is still net-negative PnL at this ceiling (TASK_736: −2.83 % AVG). Not deployable.
- **Pivot directly to Tier E (different game, e.g. market-making / MEV).** Deferred but still on the table if both architectural and informational pivots stall.

## References

- `lore/notes_2026-04-22.md §46` — original tier/pathway framework.
- `lore/notes_2026-04-22.md §52` — TASK_736 benchmark, Bal.Acc unreliability finding.
- `lore/notes_2026-04-24.md §60, §62, §65` — TASK_737 ablation, TASK_739 v2, TASK_733 cheat result.
- `lore/notes_2026-05-05.md §69` — six-axis confirmation post A1+B1.
- `tasks/TASK_733_CEILING_DIAGNOSTIC.md`, `tasks/TASK_741_PER_ASSET_INJECTION_FIX.md`.
- ADR-0007 (FiLM injection), ADR-0006 (value-weight fix).

# single mechanism exhausted

> All single-mechanism perturbations in the 738–758 campaign are null-or-worse under eval-MCTS (deploy-truthful operator). Recal asymmetric-lift finding shows eval-MCTS lifts baseline more than variants. Next moves are structural.

*Migrated from local agent memory 2026-08-31 (`project_single_mechanism_exhausted.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
**Verdict landed 2026-05-20.** Per recal sweep in `runs/recommendation_2026-05-20/analysis.md` §10 (server agent's autonomous synthesis, operator-reviewed in `lore/notes_2026-05-20.md` §151):

Every single-mechanism perturbation tried in the 738–758 wave is null-or-worse under eval-MCTS sims=16 joint (the deploy-truthful operator per `real_trade_jax.py:131,326`):

| Variant | argmax paired Δα | eval-MCTS paired Δα |
|---|---:|---:|
| 746 oracle 5-seed | +0.72pp | **−1.20pp** |
| 734 flow 5-seed | +0.20pp | −0.15pp |
| 749 Titans b128 3-seed | −3.11pp | −0.33pp |
| 757 bella_shape 3-seed | −5.80pp | −0.52pp |

**Why this matters:** the original 759 result.md claimed eval-MCTS uniformly shifts all variants up by ~+3.66pp (would re-promote 2–3 mechanisms). The recal proved this wrong — eval-MCTS is a **ceiling-bounded** correction: rescues low-Alpha ckpts where argmax decode under-uses value head, doesn't move high-Alpha ckpts already saturating value head. Variants with high-α ckpts don't ride the baseline lift.

**Mechanism class verdict map (CLOSED at this stack's precision):**
- Encoder swap, encoder aux distill, encoder aux head — CLOSED
- Data augmentation — CLOSED (catastrophic)
- Memory module (Titans) — CLOSED
- Dynamics loss (flow-matching) — CLOSED
- Value target (nstep / TD(0) / oracle V*) — CLOSED
- Eval-side decoder — RESCUE EXISTS (eval-MCTS = new canonical benchmark, but doesn't lift variants)
- Train-side search depth (760 sims=64) — in flight as final small-perturbation data point

**How to apply:**
- Stop proposing single-mechanism encoder/aux/distill/memory variants. They're exhausted.
- Future paired-Δα work must use eval-MCTS sims=16 joint as the scorecard, NOT argmax.
- Paired baselines must be retrained with same config as variant — DO NOT reuse 759's frozen baselines (asymmetric-lift breaks the pairing).
- Structural pivot framework: see `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md`. Four directions drafted: reward shaping (TASK_762), entropy reg (TASK_763), oracle k-reformulation (TASK_764), incompatible-dataset retrain (TASK_765).

**Related:** [benchmark_live_gap](benchmark_live_gap.md) (original 759 finding, partially superseded by asymmetric-lift), [classifier_alpha_anti_correlation](classifier_alpha_anti_correlation.md) (the pattern was largely a measurement artifact at argmax — 757 shape collapsed from −5.80 to −0.52 under eval-MCTS).

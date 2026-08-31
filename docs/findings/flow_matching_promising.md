# flow matching promising

> Flow-matching dynamics loss (--flow-matching, velocity_weight=0.8) looked promising at paired 3-seed (mean Δα +3.38pp, all 6 periods positive) on 2026-05-15, but 5-seed extension on 2026-05-16 REVERSED the verdict: s31 +3.56pp, s42 −12.71pp (variant-only estimate), driving 5-seed mean to +0.20pp. NOT PROMOTED. Same single→multi-seed reversal pattern as TASK_737 → TASK_743. Canonical Stage-B remains a1b1. Lesson: even 3-seed all-positive can mask 5-seed outliers on this stack.

*Migrated from local agent memory 2026-08-31 (`project_flow_matching_promising.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
**HISTORICAL — superseded by 5-seed extension on 2026-05-16.** See `lore/notes_2026-05-16.md` §115 for the reversal. This file preserved for the original 3-seed evidence and the lesson.

**Original verdict (2026-05-15 from paired 3-seed eval, seeds 7/13/23)**: Flow-matching looked PROMISING — first Tier-2 mechanism on this stack to show positive Δα without the "loss DOWN / Alpha DOWN" failure mode of 738/741.

## Quantitative

| Metric | Δ AVG | Per-seed Δα | Verdict |
|---|---:|---|---|
| Alpha | **+3.38pp** | s7 −0.54, s13 +3.31, s23 +7.36 | Mean clears +3pp threshold; strict "all positive" fails on s7 (flat) |
| Per-period Δα | **+0.89 to +4.89** | ALL 6 regimes positive | No regime damage — unlike 738/741 which hurt bull/bear/crash |
| Bal.Acc | +1.36pp | flat | Doesn't sharpen classifier (different mechanism class than 738/741) |
| dir_acc | 0.568–0.569 | flat | Doesn't move dir_acc — Alpha gain is **planning-side**, not classification |
| Training loss | +0.06 AVG | flat | Velocity-loss numerically shifts loss but doesn't dominate |
| Trade rate | toward selectivity | s13 −76%, s23 −79% trades | Opposite signature from over-trading-collapse pattern |

## Why this matters

Three preceding regressions (737 no_aux_solo single-seed outlier, 741 mint_xs, 738 legacy_27ch) all share the same failure mode: training loss DOWN, Bal.Acc UP, trade rate explodes, Alpha collapses. Flow-matching is the **existence proof** that an Alpha-side fix is possible without touching dir_acc — the dynamics loss improves *planning*, not *classification*.

This re-validates that the bottleneck for this stack is downstream of the encoder. Don't keep trying encoder/classifier sharpening mechanisms.

## 5-seed reversal (2026-05-16)

Variant-only extension at seeds 31, 42:
- s31 absolute α = +6.36% (Δα ≈ +3.56pp vs 3-seed baseline mean +2.80%)
- s42 absolute α = **−9.91%** (Δα ≈ −12.71pp)

5-seed mean Δα: **+0.20pp** (down from 3-seed mean +3.38pp). Dead-center inconclusive band.

**Verdict: NOT PROMOTED.** Canonical Stage-B baseline stays a1b1 (rank+aux), not flow-matching.

## Lesson (load-bearing for future SPECs)

Even 3-seed paired Δα with all 6 periods positive is not enough confidence to promote on this stack. The noise floor is wider than the 3-seed protocol assumed. Updated rule on 2026-05-16 (per 740/749 SPECs):

- 3-seed = **screening test**, not promotion decision
- Promotion = **5-seed** with mean ≥ +3pp AND no seed < −5pp
- Always extend to 5-seed before declaring promotion

## How to apply

- **Do NOT use flow-matching as canonical** — baseline stays a1b1.
- **Stacked variants with flow_matching are not justified** (e.g. flow + Titans, flow + Bella) — flow itself isn't load-bearing.
- **Don't escalate to α-v2** (prediction_head=True + NeuralMarketODE) yet — identity-collapse symptom (state_loss → 0 by E50) did NOT appear, so α-v1 has headroom.
- **Investigate the mechanism class**: flow-matching = continuous-time velocity prediction in the dynamics MLP. Other "improve planning without touching classification" candidates worth thinking about: alternative dynamics losses (consistency, EMA target), MCTS search-depth tweaks, value-target reformulation.
- **Reference for Tier-D Bella retrain**: if Bella ever retrains, consider flow-matching there too (same Vincent et al. framework).

## How NOT to apply

- Don't promote on 3-seed alone — TASK_741 demonstrated single-seed deceiveing the strict rule. 5-seed required.
- Don't claim it "fixes bull-blindness" — bull Δα is +3.98pp, real but small. The model is still bearish in bull regimes (base −18.67%, var −14.69%).

See `runs/734_flow_matching/result.md`, `lore/notes_2026-05-15.md` §109.
Related findings: [task741_result](task741_result.md) (the "loss DOWN, Alpha DOWN" anti-pattern this contrasts with), [bull_blindness](bull_blindness.md) (still unsolved).

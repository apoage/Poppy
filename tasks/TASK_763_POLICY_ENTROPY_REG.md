# TASK_763 — Policy Entropy Regularization Sweep

**Status**: DRAFT — operator-approval pending. Activates if TASK_762 closes null (per `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` Direction 2).
**Created**: 2026-05-20.
**Estimated wall**: ~19h server (2-wide).

## Hypothesis

Classifier-Alpha anti-correlation pattern (`project_classifier_alpha_anti_correlation.md`) says sharper per-asset direction classifier consistently degrades Alpha. Mechanism: policy/value head re-uses sharper classifier readout to trade aggressively in wrong magnitude or direction. Entropy regularization on the policy head should constrain this — force the policy to leave more probability mass on alternative actions, breaking the over-confidence pathway.

**Falsifiable**: a working cell must show BOTH Δ-Bal.Acc < 0 (classifier less sharp) AND Δα > 0 (Alpha up). If Alpha goes up but Bal.Acc also goes up, the anti-correlation mechanism wasn't broken — the hypothesis didn't fire even if the result is good.

## Code changes required

1. **`train_phase2.py`**: add CLI flags `--entropy-coef F` (default 0.01, current hardcoded in `trainer.py:382,435`) and `--min-entropy F` (default 3.0).
2. **`phase2_config.py`**: expose both as Phase2Config fields.
3. **`trainer.py:382,435`**: replace hardcoded constants with config values.

Estimated code change: ~20 lines across 3 files. Pre-implementation review subagent before launch.

## Variants

**Single-axis sweep** (entropy_coef, with min_entropy held at canon 3.0):

| cell | entropy_coef | min_entropy | seeds |
|---|---:|---:|---|
| 763-ec005 | 0.005 | 3.0 | {7,13,23} |
| 763-ec01 (canon) | 0.01 | 3.0 | reuse a1b1 sims=32 baselines from 760 |
| 763-ec02 | 0.02 | 3.0 | {7,13,23} |
| 763-ec05 | 0.05 | 3.0 | {7,13,23} |
| 763-ec10 | 0.10 | 3.0 | {7,13,23} |

If best cell shows the anti-correlation signature (Δ-Bal.Acc < 0, Δα > 0), Phase 2 = `min_entropy` sweep at best `entropy_coef`.

## Promotion gate

- 3-seed paired Δα under eval-MCTS sims=16 joint, paired against 760's a1b1 sims=32 baselines.
- **Anti-correlation diagnostic gate** (early exit): if best Δα cell does NOT also show Δ-Bal.Acc < 0, the mechanism didn't break the anti-correlation pattern — STOP regardless of Alpha.
- Standard gate: ≥ +3pp Δα AND no seed < −5pp → 5-seed extension.

## Diagnostics

Per-epoch logging must capture:
- Policy entropy (mean over batch).
- Per-asset Bal.Acc.
- Trade-rate per period in benchmark.
- Per-period Δα.

Expected mechanistic pattern (if hypothesis holds): higher entropy_coef → lower Bal.Acc → lower trade rate → higher Alpha in volatile regimes (bear, ftx_crash) where over-confident wrong trades hurt most.

## Risk

Medium. Entropy regularization is well-understood in RL/MuZero but the effective sweep range on this stack is unknown. High entropy_coef may regularize the policy into ~uniform random action distribution (Alpha collapses to ~0). Mitigation: log entropy per epoch; early-stop any cell that hits min_entropy floor consistently.

## Why this comes AFTER reward shaping

Reward shaping (TASK_762) is a more direct test of the over-trade hypothesis — it directly penalizes the failure mode in the loss. Entropy reg attacks the same failure mode via a more indirect mechanism (probability-mass constraint, not penalty). If reward shaping fails, entropy reg may still work via a different channel. If reward shaping works, entropy reg is a follow-on tuning task, not a primary mechanism test.

## References

- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` Direction 2.
- `lore/notes_2026-05-20.md` §151.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — primary mechanism evidence.
- `trainer.py:382,435` — current hardcoded entropy_coef/min_entropy.

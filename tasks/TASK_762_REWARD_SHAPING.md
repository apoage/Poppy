# TASK_762 — Reward Shaping / Cost Penalty Sweep

**Status**: DRAFT — operator-approval pending. Activates if 760 closes null (per `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` Direction 1).
**Created**: 2026-05-20.
**Estimated wall**: ~15h server (2-wide).

## Hypothesis

Over-trade collapse signature observed across 738/741/740/757/758 (Δ-trade-rate ≫ 0 with Δα ≪ 0 — see `project_classifier_alpha_anti_correlation.md`) says noise trades are under-penalized in current loss. Current `reward_scale=100` is baked in `phase2_config.py:28`; explicit fee/cost penalty doesn't exist.

**Falsifiable**: sweep `reward_scale` and `fee_penalty`; if best cell yields paired Δα ≥ +3pp under eval-MCTS sims=16 joint, hypothesis confirmed. If all cells null/negative, over-trade collapse is not driven by reward scaling and the hypothesis is dead.

## Code changes required

1. **`phase2_config.py:28`**: expose `reward_scale` as Phase2Config field (currently constant).
2. **`phase2_config.py`**: add `fee_penalty_bps: float = 0.0` field (new).
3. **`trainer.py`**: thread `fee_penalty_bps` into Stage-B reward computation. Per-trade cost = `abs(position_change) × fee_penalty_bps / 10000 × current_price`.
4. **`train_phase2.py`**: add CLI flags `--reward-scale N` (default 100, current canon) and `--fee-penalty-bps F` (default 0).

Estimated code change: ~50 lines across 3 files. Pre-implementation review subagent before launch.

## Variants

**Phase 1 — `reward_scale` sweep** (single axis, 3-seed each):

| cell | reward_scale | fee_penalty_bps | seeds |
|---|---:|---:|---|
| 762A-rs25 | 25 | 0 | {7,13,23} |
| 762A-rs50 | 50 | 0 | {7,13,23} |
| 762A-rs100 (canon) | 100 | 0 | reuse a1b1 sims=32 baselines from 760 |
| 762A-rs200 | 200 | 0 | {7,13,23} |

**Phase 2 — `fee_penalty` sweep on best `reward_scale`** (conditional, only if Phase 1 finds a cell with paired Δα ≥ +1pp):

| cell | reward_scale | fee_penalty_bps | seeds |
|---|---:|---:|---|
| 762B-fp0 (canon) | best | 0 | reuse Phase 1 |
| 762B-fp05 | best | 0.5 | {7,13,23} |
| 762B-fp10 | best | 1.0 | {7,13,23} |
| 762B-fp20 | best | 2.0 | {7,13,23} |

## Promotion gate

- 3-seed paired Δα under eval-MCTS sims=16 joint, paired against 760's a1b1 sims=32 baselines.
- Phase 1 gate: ≥ +1pp → escalate to Phase 2. < +1pp → STOP, hypothesis dead.
- Phase 2 gate: ≥ +3pp AND no seed < −5pp → 5-seed extension + Bloom-trigger.

## Diagnostics

Per-run logging must include Δ-trade-rate vs a1b1 baseline. The hypothesis is mechanistic — we expect: best cell shows Δ-trade-rate < 0 (over-trade collapse reduced) AND Δα > 0. If Alpha goes up but trade-rate also goes up, that's a different mechanism and the hypothesis didn't fire even if the result is positive.

## Risk

Low. Reward scaling is a well-studied lever in policy-gradient methods. Worst case: numerically destabilizes training (PopArt should absorb scale changes, but verify). Mitigation: smoke run at `reward_scale=50` before launching full sweep.

## References

- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` Direction 1.
- `lore/notes_2026-05-20.md` §151.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — mechanism justification.
- `experiments/760_train_mcts_sims/SPEC.md` — baseline pairing source.

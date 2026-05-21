# 762_reward_shaping — Reward-Scale + Cost-Penalty Sweep

**Status**: BLOCKED-ON-CODE (do not launch until code section below lands)
**Owner**: workstation implements code; server agent executes after deploy
**Reference**: `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` Direction 1, `tasks/TASK_762_REWARD_SHAPING.md`, `lore/notes_2026-05-21.md` §153

## Goal

Test whether the **over-trade collapse signature** seen across the closed single-mechanism regressions (`[[classifier-alpha-anticorrelation]]`) can be broken by directly penalizing noise trades in the reward channel — either by amplifying the existing reward signal (`reward_scale`) or by adding an explicit per-trade fee penalty (`fee_penalty_bps`).

This is a **planning-side perturbation, not encoder-side** — it doesn't touch the Phase-1 imprint and respects the comparability contract with the campaign baselines.

## Hypothesis

The current Phase-2 loss treats per-step rewards as raw scalars passed to `scalar_to_categorical(rewards)` in `train_phase2.py:347, 1169`. Per `phase2_config.py:30`, rewards are ~0.001 magnitude — small relative to the 401-bin categorical width. The `reward_scale: float = 100.0` field exists in `Phase2Config` but is **not currently read** anywhere in `train_phase2.py`.

Two related interventions:
1. **`reward_scale` sweep**: amplify the existing reward signal so the value head receives a stronger gradient on per-step PnL. Sweep ∈ {25, 50, 100 (canon), 200}.
2. **`fee_penalty_bps` sweep**: add an explicit per-trade penalty (subtract `fee_penalty_bps × |trade_indicator|` from the reward target) so the loss directly discourages noise trades. Sweep ∈ {0 (canon), 2, 5, 10}.

**Falsifiable**: a winning cell must show **Δ-trade-rate < 0 AND Δα > 0** simultaneously. If Alpha rises but trade rate also rises, the mechanism isn't the over-trade pathway and the hypothesis didn't fire even if the result is good.

## Code prerequisites (workstation, BLOCKING)

Per `lore/notes_2026-05-21.md` §153 pre-flight: the Phase-2 path does not currently use `reward_scale` and has no fee-penalty term. Required:

1. **Wire `phase2_config.reward_scale` into the Phase-2 reward target** at `train_phase2.py:347` and `train_phase2.py:1169`. Apply `rewards * config.reward_scale` before `scalar_to_categorical`.
2. **Add `fee_penalty_bps` field** to `Phase2Config` (default 0.0 = unchanged) + subtract `fee_penalty_bps × 1e-4 × trade_indicator` from `rewards` before scaling. `trade_indicator` = sum of per-asset `abs(action ≠ Hold)` (5-class action space — define inside the Phase-2 batch construction).
3. **CLI flags** in `train_phase2.py` argparse: `--reward-scale FLOAT`, `--fee-penalty-bps FLOAT`. Plumb to config like other overrides at `train_phase2.py:2305+`.
4. **Smoke test**: 1 epoch on small SegmentPool with `--reward-scale 50 --fee-penalty-bps 5` to confirm no NaN, no shape mismatch, trade rate observable in metrics.
5. **Logging**: per-epoch `mean_trade_rate` must appear in metrics CSV (compute from the same action distribution used by the loss).

Estimated wall: ~3h workstation (read paths + wire + smoke + commit + deploy).

## Variants

**Wave 1 — `reward_scale` solo sweep** (3 seeds per cell, paired against 760's a1b1 sims=32 baselines):

| cell | reward_scale | fee_penalty_bps | seeds |
|---|---:|---:|---|
| 762-rs025 | 25 | 0 | {7,13,23} |
| 762-rs050 | 50 | 0 | {7,13,23} |
| 762-rs100 (canon) | 100 | 0 | reuse 760 a1b1 baselines |
| 762-rs200 | 200 | 0 | {7,13,23} |

**Wave 2 — `fee_penalty_bps` solo sweep at best `reward_scale`**: gated on Wave 1 mean Δα > 0 AND winning cell shows Δ-trade-rate < 0.

| cell | reward_scale | fee_penalty_bps | seeds |
|---|---:|---:|---|
| 762-fp02 | best | 2 | {7,13,23} |
| 762-fp05 | best | 5 | {7,13,23} |
| 762-fp10 | best | 10 | {7,13,23} |

## Promotion gate

- **Mechanism gate (early exit)**: best Wave-1 cell must show Δ-trade-rate < 0 AND Δα > 0. Otherwise the over-trade hypothesis didn't fire — STOP, write `result.md` with negative-but-informative verdict.
- **Standard gate**: paired Δα ≥ +3pp under eval-MCTS sims=16 joint AND no seed < −5pp → 5-seed extension.

## Wall estimate

- Wave 1: ~12h server (4 cells × 3 seeds, 2-wide).
- Wave 2 (conditional): ~9h server.
- Total: ~15-21h depending on Wave-1 outcome.

## Operator decisions

- Launch order: TASK_761 default is 762 → 763 → 764. Confirm or override.
- Wave 2 auto-continuation: server agent may auto-continue if Wave-1 mechanism gate passes, or wait for operator confirmation?

## References

- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` Direction 1.
- `tasks/TASK_762_REWARD_SHAPING.md` — full task draft (predates code-precondition finding).
- `lore/notes_2026-05-21.md` §153 — pre-flight finding on Phase-2 code paths.
- `phase2_config.py:30` — `reward_scale` field (currently unread).
- `train_phase2.py:347, 1169` — reward target construction (where wiring lands).
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — over-trade pathway evidence.

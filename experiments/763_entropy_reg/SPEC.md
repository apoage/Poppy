# 763_entropy_reg — Policy Entropy Regularization Sweep

**Status**: BLOCKED-ON-CODE (do not launch until code section below lands; runs after 762 closes)
**Owner**: workstation implements code; server agent executes after deploy
**Reference**: `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` Direction 2, `tasks/TASK_763_POLICY_ENTROPY_REG.md`, `lore/notes_2026-05-21.md` §153

## Goal

Direct attack on `[[classifier-alpha-anticorrelation]]`. Sharper per-asset direction classifier (via 738/740/741/757/758 mechanisms) consistently degraded Alpha via the over-confidence pathway. Entropy regularization on the **policy head** constrains the planner's action distribution so probability mass cannot collapse onto a single confident-but-wrong action — even when the upstream classifier sharpens.

Sibling to TASK_762: same target failure mode, different lever. 762 attacks the reward channel; 763 attacks the policy distribution directly.

## Hypothesis

Add an entropy bonus to the policy loss:

```
total_policy_loss = policy_ce - entropy_coef * policy_entropy
                  + adaptive_floor(min_entropy, policy_entropy)
```

Sweep `entropy_coef` ∈ {0.005, 0.01 (canon-equivalent), 0.02, 0.05, 0.1}. Hold `min_entropy` at 3.0 (vs `ln(192) ≈ 5.26` uniform — room to constrain without collapsing to random).

**Falsifiable mechanism gate**: a winning cell MUST show Δ-Bal.Acc < 0 (classifier less sharp at the readout, by construction) AND Δα > 0. If Alpha rises but Bal.Acc also rises, the anti-correlation pathway wasn't broken — hypothesis didn't fire. STOP regardless of Alpha sign.

## Code prerequisites (workstation, BLOCKING)

Per `lore/notes_2026-05-21.md` §153: `train_phase2.py` has no entropy bonus in its Stage-B loss (`train_phase2.py:1103` computes `policy_loss = optax.softmax_cross_entropy(...).mean()` with no regularization). Required:

1. **Compute `policy_entropy`** in the Stage-B loss path immediately after `policy_logits` are formed: `entropy = -sum(softmax(logits) * log_softmax(logits))`, mean over valid mask.
2. **Add `entropy_coef`, `min_entropy` fields** to `Phase2Config` (defaults 0.0 and 0.0 — preserves canon = no regularization).
3. **Modify the policy loss term**: subtract `entropy_coef * policy_entropy`; add adaptive floor bonus when `entropy < min_entropy` (5× stronger push, mirroring `trainer.py:1244`).
4. **CLI flags**: `--entropy-coef FLOAT`, `--min-entropy FLOAT`. Plumb at `train_phase2.py:2305+`.
5. **Per-epoch logging**: `policy_entropy`, `policy_entropy_below_floor_fraction` in metrics CSV.
6. **Smoke test**: 1 epoch with `--entropy-coef 0.05 --min-entropy 3.0`, confirm entropy decreases monotonically and stays above floor, no NaN.

Estimated wall: ~4h workstation (read existing trainer.py:1242–1257 as reference, port to Phase-2, smoke + commit + deploy).

## Variants

**Single-axis sweep on `entropy_coef`** at `min_entropy = 3.0`, 3 seeds, paired against 760's a1b1 sims=32 baselines:

| cell | entropy_coef | min_entropy | seeds |
|---|---:|---:|---|
| 763-ec005 | 0.005 | 3.0 | {7,13,23} |
| 763-ec01 (canon-ish) | 0.01 | 3.0 | {7,13,23} |
| 763-ec02 | 0.02 | 3.0 | {7,13,23} |
| 763-ec05 | 0.05 | 3.0 | {7,13,23} |
| 763-ec10 | 0.10 | 3.0 | {7,13,23} |

**Wave 2 — `min_entropy` sweep at best `entropy_coef`**: gated on Wave 1 mechanism + standard gates.

## Promotion gate

- **Mechanism gate (early exit)**: best Wave-1 cell shows Δ-Bal.Acc < 0 AND Δα > 0. Else STOP.
- **Standard gate**: paired Δα ≥ +3pp under eval-MCTS sims=16 joint AND no seed < −5pp → 5-seed extension.
- **Entropy collapse early-stop**: any cell that hits min_entropy floor in >50% of epochs is degenerate; mark FAIL, save ckpt for diagnostics.

## Wall estimate

- Wave 1: ~15h server (5 cells × 3 seeds, 2-wide).
- Wave 2 (conditional): ~9h.
- Total: ~19-24h.

## Why this comes AFTER 762

If reward shaping (762) breaks the over-trade pathway, entropy reg may be redundant (same failure mode, two levers). If 762 fails, 763 attacks the same failure mode from a different angle — distribution constraint instead of reward penalty. Sequential gives stronger interpretation than parallel.

If 762 mechanism gate fires (Δ-trade-rate < 0 AND Δα > 0) and gives ≥ +1pp Δα, operator should decide: extend 762 to 5-seed first, or pivot to 763 to confirm same mechanism via different lever.

## Operator decisions

- Wave 2 auto-continuation: same question as 762.
- If 762 yields clean ≥ +1pp mechanism evidence, does 763 still run, or skip?

## References

- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` Direction 2.
- `tasks/TASK_763_POLICY_ENTROPY_REG.md` — full task draft (predates code-precondition finding).
- `lore/notes_2026-05-21.md` §153.
- `mcts_trader/jax_agent/trainer.py:1242–1257` — reference implementation for adaptive entropy bonus (port to Phase-2 path).
- `train_phase2.py:1103, 1086` — current Stage-B policy loss (where bonus lands).
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md`.

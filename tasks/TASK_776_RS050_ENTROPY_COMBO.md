# TASK_776 — rs050 + Policy Entropy Combination

**Status**: DRAFT — code change (factored-path entropy term) landed in this worktree on branch `worktree-agent-a55e2ae10d6a66896`; experiment runnable once merged + deployed.
**Created**: 2026-05-25
**Ship-path candidate**: #2 of 10 per `lore/ship_path_recommendations_2026-05-25.md`.
**Promotion gate (canon)**: paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint, no seed < −5pp.

## Hypothesis

The campaign has tested mechanisms one-axis-at-a-time. 762 Wave 1 found `reward_scale=50` (rs050) reliable at paired Δα **+0.27pp** with Δtrade-rate **−0.243%** — a small but reliable mechanism-gate PASS that lifted Alpha by attenuating per-step reward magnitude and reducing reward-clip incidence. 762 Wave 2 found stacking `fee_penalty_bps` on top did NOT extend the mechanism. The remaining unexplored combination is **rs050 + entropy regularization** on the policy head.

763 (entropy reg) was queued *standalone* at canon `reward_scale=100`, never with the proven rs050 leg attached. Both are already-shipped levers attacking the same `[[classifier-alpha-anticorrelation]]` pathway from orthogonal angles:

- **rs050** is a *reward-magnitude* mechanism — attenuates the per-step PnL signal driving the value target.
- **entropy_coef** is a *probability-distribution* mechanism — constrains policy mass from collapsing onto any single confident action.

The campaign-stage hypothesis: a winning cell super-adds rs050's small reliable gain with an entropy_coef that breaks the over-confident-wrong-trade failure mode. If both legs are independently zero-to-positive, the joint should exceed the sum of individual contributions if the mechanisms act on the same pathway from different angles.

**Falsifiable in two stages**:
- Mechanism gate: best cell shows Δ-Bal.Acc ≤ 0 (entropy de-sharpens classifier) AND Δα > 0. If Bal.Acc rises and Alpha rises, the anti-correlation pathway wasn't engaged — hypothesis didn't fire; STOP regardless of Alpha sign.
- Promotion gate (canon): paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint, no seed < −5pp.

## Verified code state (workstation tip `79430e3`)

**rs050 leg** is fully wired:
- `phase2_config.py:30` — `reward_scale: float = 100.0` default.
- `train_phase2.py:637, 679` — reward shaping in `SegmentBatchGenerator` (per-step PnL scaled by `self.config.reward_scale` before clip and target).
- CLI flag `--reward-scale` at `train_phase2.py:2320-2321` — overrides `config.reward_scale` when non-None.
- 762 Wave 1 production reference: `runs/762_reward_shaping/rs050_seed_*/stage_b_config.json` confirms `reward_scale: 50.0` and `use_factored_search: true`.

**Entropy leg** had a partial gap that this TASK closes:
- `phase2_config.py:52-53` — `entropy_coef: float = 0.0`, `min_entropy: float = 0.0` defaults (canon = no regularization).
- `train_phase2.py:1058-1059` — captured at JIT-trace time inside `make_stage_b_train_step`.
- CLI flags `--entropy-coef`, `--min-entropy` at `train_phase2.py:2325-2328`.
- **Joint-path entropy term** at `train_phase2.py:1237-1247` (pre-776 code) was correct but gated to `not use_factored=True` only. The whole 762 W1 campaign uses `use_factored_search=true` (`runs/762_reward_shaping/*/stage_b_config.json:147`), so 763 entropy could not be combined with rs050 cells on the canon search path.

The cited factored-path gap was called out explicitly in `lore/ship_path_recommendations_2026-05-25.md` §2 "What's missing":
> "763's entropy loss is gated to `not use_factored=True` (`train_phase2.py:1237`). Either flip the production runs to joint MCTS for the entropy variant, or expose a factored-path entropy term first (~10 LOC, per-asset Shannon entropy averaged across the 64-asset axis). The factored-path extension is small and load-bearing; without it 763 is a one-line null risk."

## Code change scope

**File**: `train_phase2.py` (~30 LOC, flag-gated, canon bit-identical when `entropy_coef == 0.0`).

**Location**: `train_phase2.py:1235-1249` — the policy-entropy regularization block inside `make_stage_b_train_step`'s closure.

**Pre-776 behavior**:
- Joint path with `entropy_coef > 0`: computes one Shannon entropy over the 320-way joint softmax → bonus + adaptive floor.
- Factored path with `entropy_coef > 0`: skipped entirely (the `not use_factored` clause), `policy_entropy = 0.0`.

**Post-776 behavior**:
- Joint path: unchanged math (same Shannon entropy computation over the 320-way joint softmax).
- Factored path: per-asset Shannon entropy computed on `policy_logits.reshape(B, 64, 5)`, softmax-then-entropy along the per-asset action axis, mean over `(B, 64)` → scalar with units ≤ ln(5) ≈ 1.61.
- Same `entropy_bonus`/`floor_penalty` math applied in both paths after `policy_entropy` is produced.
- Canon (`entropy_coef == 0.0`): identical `else` branch runs in both pre- and post-776 code; loss is bit-identical.

**Why this gate is safe**: the new factored branch is only reached when `entropy_coef > 0.0`, which is OFF by default in `phase2_config.py:52`. Every shipped baseline (760 a1b1, 762 W1 rs050) used `entropy_coef = 0.0` (verified in their `stage_b_config.json:79`), so the new branch is inert on every recorded baseline and on every variant cell that doesn't explicitly turn entropy on.

**Diff** (worktree `worktree-agent-a55e2ae10d6a66896`):

```diff
-            # TASK_763: policy entropy regularization (joint case only; factored
-            # has different semantics — sweep gated by use_factored=False).
-            if entropy_coef > 0.0 and not use_factored:
+            # TASK_763 + TASK_776: policy entropy regularization. Joint and
+            # factored paths use different semantics.
+            if entropy_coef > 0.0:
+                if use_factored:
+                    # TASK_776: per-asset Shannon entropy. Reshape (B, 320) →
+                    # (B, 64, 5), per-asset softmax, per-asset entropy, then
+                    # mean over (B, 64) yielding a scalar with units of nats
+                    # per asset (max ln(5) ≈ 1.61).
+                    pol_r = policy_logits.reshape(-1, n_assets, n_actions_per_asset)
+                    per_asset_probs = jax.nn.softmax(pol_r, axis=-1)
+                    per_asset_entropy = -jnp.sum(
+                        per_asset_probs * jnp.log(per_asset_probs + 1e-10),
+                        axis=-1,
+                    )  # (B, 64)
+                    policy_entropy = per_asset_entropy.mean()
+                else:
                     policy_probs = jax.nn.softmax(policy_logits, axis=-1)
                     policy_entropy = -jnp.sum(
                         policy_probs * jnp.log(policy_probs + 1e-10), axis=-1
                     ).mean()
                 entropy_bonus = entropy_coef * policy_entropy
                 ...
```

**Companion docstring update** in `phase2_config.py:47-58` — reflect that factored-path entropy is now supported, with explicit guidance that `min_entropy` units differ ~3.6× between joint (max ≈ 5.77) and factored (max ≈ 1.61).

**Smoke verification** (this worktree):
- Static read-back of the diff confirms the new factored branch is unreachable when `entropy_coef == 0.0` (the outer `if` gate). The pre-776 `else` branch (`policy_entropy = jnp.array(0.0)`) is preserved exactly. Canon is bit-identical.
- Standalone JAX smoke (numpy-init random logits): factored per-asset H ≈ 1.31 (vs ln(5)=1.61 max), joint H ≈ 5.31 (vs ln(320)=5.77 max), ratio ≈ 4.06 — sane units, no NaN.

## Variants

3 entropy_coef × 3 seeds = 9 cells, all with `reward_scale=50.0` + `use_factored_search=True` (canon search path) + `entropy_coef > 0.0` (new factored entropy term).

| cell | reward_scale | entropy_coef | min_entropy | seed | wall (server, est.) |
|---|---:|---:|---:|---:|---:|
| 776-ec005_s7  | 50.0 | 0.005 | 0.7 | 7  | ~2.5 h |
| 776-ec005_s13 | 50.0 | 0.005 | 0.7 | 13 | ~2.5 h |
| 776-ec005_s23 | 50.0 | 0.005 | 0.7 | 23 | ~2.5 h |
| 776-ec02_s7   | 50.0 | 0.02  | 0.7 | 7  | ~2.5 h |
| 776-ec02_s13  | 50.0 | 0.02  | 0.7 | 13 | ~2.5 h |
| 776-ec02_s23  | 50.0 | 0.02  | 0.7 | 23 | ~2.5 h |
| 776-ec05_s7   | 50.0 | 0.05  | 0.7 | 7  | ~2.5 h |
| 776-ec05_s13  | 50.0 | 0.05  | 0.7 | 13 | ~2.5 h |
| 776-ec05_s23  | 50.0 | 0.05  | 0.7 | 23 | ~2.5 h |

**min_entropy = 0.7** is the factored-path analog of 763's `min_entropy = 3.0` (which targeted joint-path's max ~5.77 → ~52% of max). On the factored path with max ~1.61 nats per asset, 0.7 ≈ 43% of max — keeps the floor active enough to push back against fully-collapsed per-asset distributions while leaving headroom for the policy to commit decisively when it should.

**Wall**: 9 cells × ~2.5 h ≈ 22.5 h serial; **~12-15 h wall at 2-wide** on server (per `lore/ship_path_recommendations_2026-05-25.md` §2).

## Baselines

Two paired comparisons are required to isolate the contribution:

1. **vs 760 a1b1 sims=32 (canon scorecard)** — absolute Δα for promotion-gate decision. Reference: `runs/760_*/result.md` (canon baseline used across the post-eval-MCTS-recal campaign).
2. **vs 762 W1 rs050 alone** — isolates the entropy contribution from the rs050 leg. Reference: `runs/762_reward_shaping/rs050_seed_{7,13,23}/result.md` (rs050 paired Δα +0.27pp). If 776 cells super-add (Δα > 762's +0.27pp by ≥ +2pp per seed), the entropy term is contributing on top of rs050.

Both comparisons must use eval-MCTS sims=16 joint (deploy-truthful scorecard).

## Sanity gate (pre-promotion)

- Cell must not collapse: per-step `policy_entropy` (logged at `train_phase2.py:1514`) must stay above 0.3 nats per asset for ≥ 80% of epochs. If a cell sits at the floor (`< min_entropy = 0.7`) for > 50% of epochs, mark FAIL (degenerate — entropy_coef is too high or min_entropy too tight); save ckpt as a data point.
- No NaN in any loss term across 50 epochs.

## Promotion gate (canon)

- **Mechanism gate (early-exit)**: best Wave-1 cell shows Δ-Bal.Acc ≤ 0 (entropy de-sharpened classifier as designed) AND Δα > 0. Otherwise STOP regardless of Alpha sign — the anti-correlation pathway wasn't engaged.
- **Canon promotion**: paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint AND no seed < −5pp → trigger 5-seed extension.
- **5-seed promotion**: 5-seed mean Δα ≥ +3pp with all seeds > −5pp → promote to canon (replaces a1b1 + rs050 as the new baseline if both legs are kept).

## Anti-correlation check

**LOW risk by construction**. Entropy regularization is the *direct attack* on the anti-correlation pattern (its purpose is to keep the classifier softer; the entropy bonus penalizes over-confident distributions). The risk asymmetry:

- **Upside**: if the anti-correlation pathway is the dominant blocker, entropy reg should be a precise lever — small `entropy_coef` (0.005-0.05) should produce the largest gains, with the rs050 leg providing baseline lift.
- **Downside**: high `entropy_coef` pushes the policy toward uniform-random and Alpha collapses to ~0. Mitigated by sweeping {0.005, 0.02, 0.05} and the sanity gate (entropy-floor-hit fraction).

**Diagnostic to log per-epoch**: `policy_entropy` (already logged at `train_phase2.py:1514` via the existing metrics dict — but the joint-only pre-776 path logged `0.0` for factored runs; post-776 the factored path will log the new per-asset entropy, so 776 cells get real entropy traces).

## Risk

- **Same as 763 standalone** plus the orthogonal-mechanism risk that rs050 already absorbed most of the available gain (760 a1b1 → +rs050 → +0.27pp, leaving little for entropy to add). If 776 cells cluster around 762 W1's rs050 baseline (±1pp of +0.27pp), the entropy mechanism is null on this surface and the combination doesn't super-add — closes the joint pathway as a small-mechanism candidate.
- **Path-specific min_entropy calibration**: the 0.7 default is an educated guess; if Wave 1 cells all hit the floor in > 50% of epochs (sanity gate FAIL), a Wave 2 with `min_entropy ∈ {0.3, 0.5, 0.7}` at the best `entropy_coef` would calibrate the floor before re-running. Wave 2 is gated on operator decision.

## Why this comes AFTER 762 / 763 individually

- 762 W1 closed (+0.27pp paired, mechanism gate PASSED). 762 W2 closed FAIL (`lore/notes_2026-05-23.md` §167). The reward-magnitude lever is well-mapped.
- 763 standalone has not been run — the joint-path-only gate at `train_phase2.py:1237` (pre-776 code) made it inapplicable on the canon factored search.
- 776 stacks the two on the canon search path with the factored-path entropy term unlocked. Combination-first ordering is justified per `lore/ship_path_recommendations_2026-05-25.md` §"Recommended execution order" item 2: candidate #2 is the next-cheapest probe after #5 (rs050 + 767), and is the most prior-favorable LOW-anti-correlation joint test.

## References

- `lore/ship_path_recommendations_2026-05-25.md` §2 — primary spec source for this candidate, including the "what's missing" identification of the factored-path entropy gap.
- `experiments/763_entropy_reg/SPEC.md` — entropy-only SPEC; 776 stacks on it.
- `tasks/TASK_763_POLICY_ENTROPY_REG.md` — full design doc for the entropy mechanism.
- `tasks/TASK_762_REWARD_SHAPING.md` (implicit via `lore/notes_2026-05-23.md` §167) — rs050 Wave-1 result.
- `train_phase2.py:1235-1249` — modified policy-entropy block.
- `phase2_config.py:30, 47-58` — `reward_scale`, `entropy_coef`, `min_entropy` config fields.
- `runs/762_reward_shaping/rs050_seed_{7,13,23}/result.md` — rs050 paired baselines.
- `runs/760_*/result.md` — canon a1b1 sims=32 baselines.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — primary evidence base for the anti-correlation pathway.
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` — context: single-mechanism class is closed; combinations are the next probe class.
- `lore/handoff_2026-05-24_764-launch.md` — server wall-time calibration baseline.

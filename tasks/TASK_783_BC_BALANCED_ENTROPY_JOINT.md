# TASK_783 — Class-Balance + Entropy Joint Sweep

**Status**: DRAFT — zero code in TASK_783 itself. **BLOCKED on TASK_776 merge** (factored-path entropy term). Canon trains under `use_factored_search=True`, and the committed entropy term at `train_phase2.py:1237` (HEAD `79430e3`) is **gated to the joint path only** (`if entropy_coef > 0.0 and not use_factored`). Without TASK_776 the sweep is either a no-op on canon or must defy the factored-search contract — both unacceptable.
**Created**: 2026-05-25
**Ship-path candidate**: #9 of 10 per `lore/ship_path_recommendations_2026-05-25.md`.
**Promotion gate (canon)**: paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint, no seed < −5pp.

## Hypothesis

**Canonical claim**: balanced BC + entropy regularization trades classifier sharpness for trade selectivity at the right rate.

The TASK_726 class-balanced BC mechanism (`bc_class_balanced=True`, in production on every campaign run) gives trades up to `bc_max_trade_weight` more gradient weight than holds, by construction sharpening the classifier into the trade-positive direction. Per `project_classifier_alpha_anti_correlation.md`, sharper classifiers on this stack reliably degrade Alpha via trade-rate explosion (the over-confidence pathway: 737, 738, 740, 741, 757 all collapse via Δ-Bal.Acc up, Δ-trade-rate up, Δα down). Class-balanced BC is itself a sharpening lever — current canon keeps it ON because the alternative (HOLD-marginal collapse) is worse, but the on-state is still sharpening.

Entropy regularization on the policy head (TASK_763 mechanism, factored-path extension shipping in TASK_776) is the *direct counterweight*: it adds a softmax-spreading bonus `-entropy_coef * H(policy)` to the loss, which pushes probability mass away from the most confident action.

**The two mechanisms have never been swept jointly.** Individually:
- TASK_726 class-balanced BC: in canon, no Δα measured "with vs without" because the canon baseline already has it on.
- TASK_763 entropy reg: queued but not yet run; the SPEC at `experiments/763_entropy_reg/SPEC.md` is single-axis on `entropy_coef` at `entropy_coef ∈ {0.005, 0.01, 0.02, 0.05, 0.10}`.

Joint sweep tests whether the over-confidence the class-balanced BC creates is *exactly* the surface entropy regularization can neutralize. Net direction depends on coefficient balance — this is the falsifiable claim.

**Falsifier mechanism gate**: a winning cell must show Δ-Bal.Acc ≤ 0 (classifier de-sharpened by the entropy term) AND Δα > 0 simultaneously vs the canon a1b1 baseline. If both Bal.Acc and Alpha rise, the anti-correlation pathway wasn't broken — entropy term wasn't load-bearing.

## Verified code state (workstation tip `79430e3`, HEAD-committed)

### Class-balanced BC (TASK_726) — fully shipped, default ON, in production

`phase2_config.py`:
- `:74` — `bc_class_balanced: bool = True` (DEFAULT ON; this is current canon)
- `:75` — `bc_max_trade_weight: float = 64.0` (config default; **production runs override to 4.0** — see canon spot-check below)

`train_phase2.py`:
- `:1053` — `use_bc_balance = config.bc_class_balanced  # TASK_726`
- `:1054` — `bc_max_tw = config.bc_max_trade_weight     # TASK_726`
- `:1217-1230` — **MCTS visit-distribution BC site**: when `use_bc_balance` is True, computes per-asset trade-strength weighting on the MCTS-target policy CE (joint path uses the global 320-action softmax, factored path uses per-asset 5-action softmaxes — both branches inside `if use_factored:`)
- `:1338-1349` — **Oracle BC site**: same weighting applied to the oracle-target CE (the bc_weight*bc_loss term)
- `:2315-2316` — `--no-bc-class-balance` CLI override (sets `config.bc_class_balanced = False`)

### Entropy regularization (TASK_763) — shipped commit `b4e0158`, gated to JOINT path only

`phase2_config.py`:
- `:52` — `entropy_coef: float = 0.0` (canon = no regularization)
- `:53` — `min_entropy: float = 0.0` (canon = no floor)

`train_phase2.py`:
- `:1058-1059` — `entropy_coef = float(getattr(config, 'entropy_coef', 0.0))`, `min_entropy = float(getattr(config, 'min_entropy', 0.0))` (JIT-trace-time captures)
- **`:1237` — CRITICAL: `if entropy_coef > 0.0 and not use_factored:`** — entropy bonus is computed ONLY when `use_factored=False`. This is the load-bearing constraint that blocks TASK_783 on canon (which uses `use_factored=True`).
- `:1238-1249` — entropy bonus computation: `softmax → log_softmax → mean entropy → bonus = entropy_coef * policy_entropy → floor_penalty = entropy_coef * 5.0 * relu(min_entropy - policy_entropy) → policy_loss -= bonus; policy_loss += floor_penalty`
- `:2366-2369` — `--entropy-coef FLOAT` + `--min-entropy FLOAT` argparse + override block

### Factored-search canon — confirmed via production stage_b_config

`runs/762_reward_shaping/rs050_seed_7/stage_b_config.json` (spot-checked):
- `"use_factored_search": true` (line 147 across 5 sampled cells: rs050_seed_7, rs025_seed_13, fp02_seed_7, fp05_seed_7, rs200_seed_13)
- `"bc_class_balanced": true`
- `"bc_max_trade_weight": 4.0` (**not** 64.0; canon override)
- `"entropy_coef": 0.0`, `"min_entropy": 0.0`

**Consequence**: under HEAD-committed `train_phase2.py:1237`, setting `--entropy-coef 0.02 --min-entropy 3.0` on a canon Stage-B run is a **silent no-op** because `use_factored=True` makes the `if` branch False. The metric `policy_entropy` is still exported (TASK_763 logging at `:1547`) but no gradient flows.

## Dependency on TASK_776 — explicit

**TASK_776 (rs050 + entropy combination)** introduces the factored-path entropy extension as part of its prep ceremony. Per `lore/notes_2026-05-25.md` §"TASK number allocation" lines 54-55:

> | 2 — rs050 + entropy sweep | TASK_776 | Code (factored-path entropy ~10 LOC) + SPEC |

TASK_776 lands a code change in its worktree that replaces the line-1237 joint-only gate with a unified path-aware extension:
- Joint MCTS (`use_factored=False`): one Shannon entropy over the 320-way global softmax (max ≈ ln(320) ≈ 5.77 nats), same as today.
- Factored MCTS (`use_factored=True`): per-asset Shannon entropy over the 5-action softmax, averaged over the 64-asset axis (max ≈ ln(5) ≈ 1.61 nats). `min_entropy` floor is interpreted in the same per-asset units.

**TASK_783 cannot launch on canon until TASK_776's code change is merged to `master`.** Two interlocks:

1. **Path-correctness**: without TASK_776, every TASK_783 cell with `entropy_coef > 0` produces an identical metric profile to canon (no gradient → no Bal.Acc change, no Alpha change, no `policy_entropy` movement beyond what would happen without the flag). The sweep would report 9 cells of zero Δ vs 760 a1b1 — a non-result.
2. **Unit-correctness**: TASK_783's `--min-entropy 3.0` value is sized for the joint-path 320-way distribution (ln(320) ≈ 5.77 → floor at 3.0 leaves headroom). On the factored-path per-asset entropy (max 1.61), `min_entropy 3.0` is **above the path's maximum** — every batch hits the floor penalty unconditionally, the optimizer fights it with maximum force, and the cell degenerates. The factored-path equivalent floor is **`min_entropy ≈ 0.7`** (≈ 43% of max, mirroring the joint's 3.0 / 5.77 ≈ 52% ratio). **TASK_776's path-aware extension expects the operator to set `min_entropy` per-path** — TASK_783 must use the factored-path units once TASK_776 lands.

**Read order**: TASK_776 (code + SPEC) must merge to `master` before TASK_783's SPEC at `experiments/783_bc_entropy_joint/SPEC.md` is unlocked for launch.

## Variants

3 cells × 3 seeds = 9 paired-against-760 cells. All cells use the canon a1b1 flag stack PLUS `--bc-class-balanced --bc-max-trade-weight 4.0` (current canon, matches `runs/762_reward_shaping/rs050_seed_7/stage_b_config.json:75`).

The `min_entropy` value below assumes TASK_776's factored-path units (per-asset, max ≈ 1.61). If TASK_776 lands the joint-path semantics for both paths (treating `min_entropy` as global), the value is **0.7** instead of **3.0** — operator must confirm against TASK_776's final SPEC before launch.

| cell | bc_class_balanced | bc_max_trade_weight | entropy_coef | min_entropy (factored) | seeds | wall (server, est.) |
|---|---|---:|---:|---:|---|---:|
| 783-canon | True | 4.0 | 0.0 | 0.0 | {7, 13, 23} | reuse 760 a1b1 baselines |
| 783-balanced-ec02 | True | 4.0 | 0.02 | 0.7 | {7, 13, 23} | ~2.5 h/seed |
| 783-balanced-ec05 | True | 4.0 | 0.05 | 0.7 | {7, 13, 23} | ~2.5 h/seed |

**Note**: cell 783-canon reuses the existing 760 a1b1 baselines (`runs/760_train_mcts_sims/baseline_paired_seed_*`) rather than re-running them — no incremental cost, paired-Δα is computed against the same baseline as the other 9 cells across the campaign.

**Cells with non-zero coefficient**: 6 cells × 2.5 h = 15 h serial → **~12-15 h wall at 2-wide on server GPU 1/2** (matches the ship-path table estimate). Add ~1h orchestration overhead.

## Promotion gate (canon)

- Paired Δα ≥ +3pp vs 760 a1b1 sims=32 baselines under eval-MCTS sims=16 joint.
- No seed < −5pp.
- Eval command per the canon contract: `python tools/benchmark_easy.py --checkpoint <ckpt> --periods-file config/benchmark_periods.json --eval-mcts-sims 16 --eval-mcts-decode joint --output runs/783_bc_entropy_joint/eval_<cell>/benchmark.json`.
- 5-seed extension (seeds 31, 42) only if 3-seed clears.

## Mechanism gate (early-exit)

Per `experiments/763_entropy_reg/SPEC.md` standing mechanism gate (this TASK inherits the same anti-correlation falsifier):

- **A winning cell MUST show Δ-Bal.Acc ≤ 0 AND Δα > 0 simultaneously vs canon a1b1.** This is the load-bearing constraint that distinguishes "entropy regularization broke the anti-correlation pathway" from "the classifier somehow improved on a different axis (margin? coverage?) and Alpha followed by coincidence."
- **If Δ-Bal.Acc > 0 AND Δα > 0**: not a TASK_783 win; the entropy term didn't fire as the load-bearing mechanism. Result is informative but should be tagged "not a path-aware ship."
- **Entropy-collapse early-stop**: any cell where the post-cell `policy_entropy` hits the floor in >50% of training epochs is degenerate (operator-pushed-too-hard); mark FAIL, save ckpt for diagnostics, do not include in the paired Δα aggregation.

## Anti-correlation check — MEDIUM risk

**Why MEDIUM (not LOW)**: class-balanced BC is in the encoder-direct gradient pathway. The trade-weight `(1 + trade_strength * (trade_weight - 1.0))` applied to `per_asset_ce_policy` at `train_phase2.py:1226-1228` (factored) and `:1349` (oracle BC term) flows directly into the encoder + policy head via backward. This is *exactly* the surface `project_classifier_alpha_anti_correlation.md` implicates.

Entropy is the counterweight, but the net direction depends on coefficient balance. Two failure modes:
1. **Entropy too weak** (`ec=0.02` may not be enough at factored-path scale): class-balance dominates, classifier sharpens, Alpha collapses via over-trade — same signature as the closed mechanism class.
2. **Entropy too strong** (`ec=0.05` may push policy toward uniform-random): policy entropy hits floor, optimizer fights the floor penalty, policy decays to random, Alpha collapses to ~0.

**Diagnostic — Δ-Bal.Acc vs Δα plot at each entropy_coef**:
- Read per-epoch CSV metrics from each cell (`policy_entropy`, `direction_acc`, `bal_acc`, `mean_trade_rate`).
- Compute paired Δ vs the matched 760 baseline at end-of-training.
- Plot 6 cells (3 entropy values × 3 seeds) in (Δ-Bal.Acc, Δα) space.
- A *winning cell* sits in the lower-right quadrant (Bal.Acc down, Alpha up). A *failed cell* sits in upper-left (Bal.Acc up, Alpha down) — the anti-correlation signature.
- The plot is the load-bearing artifact for the `result.md` writeup.

## Cost estimate

- Code in TASK_783 itself: **0 LOC.**
- Dependency: TASK_776's factored-path entropy code change (~10 LOC, lives in TASK_776, not TASK_783).
- 6 cells × ~2.5 h server wall = 15 h serial / **~12-15 h wall at 2-wide on GPU 1/2** per the ship-path estimate. Add ~1h for paired-Δα aggregation + result.md.
- Total: ~13-16 h wall + 0 code-hours, conditional on TASK_776 merge.

## References

- `lore/ship_path_recommendations_2026-05-25.md` §9 — primary spec source for this TASK.
- `tasks/TASK_726_CLASS_BALANCED_BC.md` — class-balanced BC original mechanism (shipped, in production).
- `tasks/TASK_763_POLICY_ENTROPY_REG.md` — entropy regularization full task draft.
- `experiments/763_entropy_reg/SPEC.md` — sibling SPEC (single-axis entropy sweep, joint path).
- **TASK_776 (the dependency)** — rs050 + entropy joint, ships the factored-path entropy extension that unblocks this TASK.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — over-confidence pathway evidence (737/738/740/741/757 all closed via this signature).
- `phase2_config.py:52-53` — `entropy_coef`, `min_entropy` field defaults.
- `phase2_config.py:74-75` — `bc_class_balanced=True` (canon ON), `bc_max_trade_weight=64.0` (default; production override 4.0).
- `train_phase2.py:1053-1054` — TASK_726 BC class-balance capture site.
- `train_phase2.py:1217-1230` — TASK_726 MCTS-target BC weighting (factored + joint branches).
- `train_phase2.py:1237` — **CRITICAL: TASK_763 entropy gate, joint-only** (`if entropy_coef > 0.0 and not use_factored`).
- `train_phase2.py:1338-1349` — TASK_726 oracle-BC weighting.
- `train_phase2.py:1547` — `policy_entropy` metric export (TASK_763).
- `train_phase2.py:2366-2369` — `--entropy-coef` / `--min-entropy` CLI.
- `runs/762_reward_shaping/rs050_seed_7/stage_b_config.json:75, 147` — canon production confirms `bc_max_trade_weight=4.0` and `use_factored_search=true`.
- `runs/760_train_mcts_sims/baseline_paired_seed_*/` — paired-Δα baseline (a1b1 sims=32, eval-MCTS sims=16 joint).
- `runs/recommendation_2026-05-20/analysis.md` — eval-MCTS recal baseline narrative; +3.66pp uniform lift on baselines (lifts baseline more than variants on average — per `project_single_mechanism_exhausted.md`).
- `lore/notes_2026-05-25.md` §"TASK number allocation" — TASK_776 dependency contract.

# 783_bc_entropy_joint — Class-Balanced BC × Entropy Regularization Joint Sweep

**Status**: **BLOCKED on TASK_776 merge** (factored-path entropy term).
**Owner**: server agent executes once TASK_776 lands on `master`; workstation reviews.
**Reference**: `lore/ship_path_recommendations_2026-05-25.md` §9, `tasks/TASK_783_BC_BALANCED_ENTROPY_JOINT.md`, `tasks/TASK_726_CLASS_BALANCED_BC.md`, `tasks/TASK_763_POLICY_ENTROPY_REG.md`, **TASK_776 (dependency)**.

## Why this is BLOCKED

Canon trains under `use_factored_search=True` (production confirmed via `runs/762_reward_shaping/rs050_seed_7/stage_b_config.json:147`). The committed entropy term at `train_phase2.py:1237` (HEAD `79430e3`) gates on `not use_factored`, so every TASK_783 cell with non-zero `entropy_coef` produces a silent no-op on canon. TASK_776 lands the factored-path extension (per-asset Shannon entropy over the 5-action softmax, averaged over the 64-asset axis); without it this SPEC's "balanced + ec0.02 / + ec0.05" cells are indistinguishable from canon.

**Unblock contract**: TASK_776 code merged to `master`. Verify with `grep "if entropy_coef > 0.0" train_phase2.py` showing the path-aware extension (not `and not use_factored`).

## Goal

Test whether the over-confidence created by canon's `bc_class_balanced=True` (per-asset trade-weight 4.0× hold-weight; current production mechanism per TASK_726) can be neutralized by policy-head entropy regularization (TASK_763 mechanism, factored-path extension TASK_776). The two have never been swept jointly. Canonical claim: "balanced BC + entropy reg trades classifier sharpness for trade selectivity at the right rate."

The TASK_783 falsifier inherits the TASK_763 mechanism gate: a winning cell must show **Δ-Bal.Acc ≤ 0 AND Δα > 0 simultaneously** vs canon a1b1. Both rising = wrong mechanism fired.

## Hypothesis

`bc_class_balanced=True` is the canonical production mechanism (`phase2_config.py:74`). It sharpens the per-asset classifier on the trade direction (weights trade-strength up to `bc_max_trade_weight=4.0` more than hold-strength). Per `project_classifier_alpha_anti_correlation.md`, this is exactly the surface where every closed mechanism's gradient lived. The campaign never tested "what happens if we keep balanced BC on (because turning it off hits the HOLD-marginal collapse) but counterweight it with explicit entropy regularization."

If the joint configuration finds a `entropy_coef` value where the classifier is held at its current sharpness (Bal.Acc neutral or down) while the policy's action distribution stays soft enough to prevent over-trade collapse, this is the path-aware ship.

## Variants

| cell | bc_class_balanced | bc_max_trade_weight | entropy_coef | min_entropy (factored) | seeds |
|---|---|---:|---:|---:|---|
| 783-canon | True | 4.0 | 0.0 | 0.0 | {7, 13, 23} (reuse 760 a1b1) |
| 783-balanced-ec02 | True | 4.0 | 0.02 | 0.7 | {7, 13, 23} |
| 783-balanced-ec05 | True | 4.0 | 0.05 | 0.7 | {7, 13, 23} |

**`min_entropy` unit note**: TASK_776's factored-path semantics interpret `min_entropy` in **per-asset units** (max ≈ ln(5) ≈ 1.61). `0.7` is ≈ 43% of max, mirroring the joint-path 3.0 / 5.77 ≈ 52% ratio canonized in `experiments/763_entropy_reg/SPEC.md`. **Operator must confirm against TASK_776's final SPEC** before launch — if TASK_776 keeps the joint-path semantics for both paths, `min_entropy=3.0` is the equivalent value.

**Baseline policy**: cell `783-canon` reuses `runs/760_train_mcts_sims/baseline_paired_seed_*` directly (no incremental cost, paired-Δα computed against the same baseline as the other 9 cells across the campaign).

## Launch commands (per cell)

All launches go through `train_phase2.py` with the canon a1b1 flag stack. **Critical**: do not omit `--bc-class-balanced` / `--bc-max-trade-weight 4.0` — production canon overrides the config default of 64.0 to 4.0.

### Smoke check (always run before paired-3-seed)

Verify the entropy term fires under factored search after TASK_776 merge:

```bash
python train_phase2.py \
  --stage B \
  --base-ckpt runs/recommendation_2026-05-20/<canon_basis_ckpt>.pkl \
  --epochs 1 \
  --bc-class-balanced --bc-max-trade-weight 4.0 \
  --entropy-coef 0.05 --min-entropy 0.7 \
  --seed 7 \
  --name 783_smoke_factored_entropy
```

Expected: per-epoch `policy_entropy` metric shows a non-trivial value (not 0.0); `policy_entropy_below_floor_fraction` <= 50%; no NaN. If `policy_entropy` is 0.0 or unchanged from canon, TASK_776 did not actually land the factored-path extension and the cell-launch must be blocked.

### Cell 783-balanced-ec02 (seeds 7, 13, 23)

```bash
python train_phase2.py \
  --stage B \
  --base-ckpt <canon_a1b1_base_ckpt> \
  --epochs <canon_epochs> \
  --bc-class-balanced --bc-max-trade-weight 4.0 \
  --entropy-coef 0.02 --min-entropy 0.7 \
  --seed <SEED> \
  --name 783_balanced_ec02_seed_<SEED>
```

### Cell 783-balanced-ec05 (seeds 7, 13, 23)

```bash
python train_phase2.py \
  --stage B \
  --base-ckpt <canon_a1b1_base_ckpt> \
  --epochs <canon_epochs> \
  --bc-class-balanced --bc-max-trade-weight 4.0 \
  --entropy-coef 0.05 --min-entropy 0.7 \
  --seed <SEED> \
  --name 783_balanced_ec05_seed_<SEED>
```

**Note on canon flags**: every other Phase-2 setting inherits from `Phase2Config` defaults + the a1b1 production override file. Specifically `use_factored_search=True` (the entire point of TASK_776 dependency), `reward_scale=100.0` (canon, not rs050), `mcts_sims=32`, `use_gumbel=True`, `use_direction_head=True`, `use_rank_head=True`, `use_per_asset_policy_head=True`, etc.

## GPU placement + wall

- Server `gpu-server`, GPU 1 + GPU 2 (per `TASK_742` rule: ≤2 concurrent `train_phase2.py` per host; GPU 0 reserved for VirtualBox, GPU 3 free for eval).
- 6 paired cells × ~2.5 h/cell = 15 h serial.
- 2-wide concurrent: **~12-15 h wall** (matches ship-path table estimate).
- Eval-MCTS sims=16 joint benchmark per cell: ~5-10 min on GPU 3.
- Total: ~13-16 h wall.

## Eval protocol (paired Δα, eval-MCTS sims=16 joint)

For each cell ckpt:

```bash
python tools/benchmark_easy.py \
  --checkpoint runs/783_bc_entropy_joint/<cell_name>/checkpoint_best.pkl \
  --periods-file config/benchmark_periods.json \
  --eval-mcts-sims 16 \
  --eval-mcts-decode joint \
  --output runs/783_bc_entropy_joint/eval_<cell_name>/benchmark.json
```

Paired comparison: variant Alpha minus the matched-seed 760 a1b1 baseline at `runs/760_train_mcts_sims/baseline_paired_seed_<SEED>/eval_baseline_seed_<SEED>/benchmark.json` (matched seed → matched evaluation periods → paired Δα).

**Aggregation**: per-cell mean Δα across 3 seeds; cell with mean ≥ +3pp AND no seed < −5pp clears the promotion gate.

## Anti-correlation diagnostic logging

Per-epoch logging from each cell's training-metrics CSV must include (already exported in HEAD `79430e3`):

- `policy_entropy` (TASK_763, `train_phase2.py:1547`) — must be > 0 and < max-path-entropy for valid cell
- `mean_trade_rate` (from action distribution)
- `bal_acc` (per-asset direction balanced accuracy)
- `direction_acc` (TASK_727)
- `bull_bal_acc` / `bear_bal_acc` (regime-conditional) if available

**Diagnostic artifact**: scatter plot of (Δ-Bal.Acc, Δα) for each of the 6 paired cells in `runs/783_bc_entropy_joint/result.md`. Each point = (Bal.Acc_variant − Bal.Acc_baseline, Alpha_variant − Alpha_baseline) at matched seed.

- Lower-right quadrant (Bal.Acc ↓, Alpha ↑) = mechanism fired correctly. Ship candidate.
- Upper-right (Bal.Acc ↑, Alpha ↑) = Alpha rose for non-mechanism reasons. NOT a TASK_783 ship — flag for separate investigation.
- Upper-left (Bal.Acc ↑, Alpha ↓) = classic anti-correlation signature. Reaffirms the standing finding; entropy didn't break the pathway.
- Lower-left (Bal.Acc ↓, Alpha ↓) = entropy over-strong, policy decayed toward random; consider lower `entropy_coef` value for follow-up.

## Promotion gate (canon)

- **Standard gate**: paired Δα ≥ +3pp under eval-MCTS sims=16 joint AND no seed < −5pp → 5-seed extension (seeds 31, 42).
- **Mechanism gate (early exit)**: best cell must show Δ-Bal.Acc ≤ 0 AND Δα > 0. If both rise, write `result.md` with "mechanism-gate-FAIL but Alpha-positive" verdict and refer to follow-up TASK.

## Entropy-collapse early-stop

Per `experiments/763_entropy_reg/SPEC.md` standing rule (inherited): any cell where the per-epoch `policy_entropy` hits `min_entropy` in >50% of training epochs is degenerate. Mark FAIL, save ckpt for diagnostics, do not include in paired Δα aggregation. If both `ec=0.02` and `ec=0.05` cells hit this floor (`min_entropy=0.7` is too aggressive for the factored-path units), launch follow-up at `min_entropy=0.4` instead.

## Operator decisions

- Confirm `min_entropy` unit interpretation against TASK_776's final SPEC (0.7 factored-units vs 3.0 joint-units).
- 5-seed extension auto-continue gate: if 3-seed paired Δα clears +3pp, server agent may auto-launch seeds {31, 42} without operator wait (per `feedback_server_stays_under_load.md`).
- Result.md write owner: server agent drafts; workstation reviews + tags Bloom-relevant.

## Wall estimate summary

- Smoke: ~30 min on workstation.
- Paired 6 cells: ~12-15 h wall server (2-wide GPU 1/2).
- Eval-MCTS sims=16 joint per cell: ~5-10 min on GPU 3.
- Aggregation + result.md: ~1 h.
- **Total: ~13-16 h wall + 0 code-hours** (in TASK_783; the ~10 LOC factored-path extension lives in TASK_776).

## References

- `tasks/TASK_783_BC_BALANCED_ENTROPY_JOINT.md` — full task spec including verified code state.
- `lore/ship_path_recommendations_2026-05-25.md` §9 — primary candidate spec.
- **TASK_776 (the dependency)** — rs050 + entropy joint, ships factored-path entropy extension.
- `experiments/763_entropy_reg/SPEC.md` — sibling SPEC, single-axis joint-only entropy sweep.
- `phase2_config.py:52-53, 74-75` — config fields (`entropy_coef`, `min_entropy`, `bc_class_balanced`, `bc_max_trade_weight`).
- `train_phase2.py:1053-1054` — TASK_726 BC class-balance capture.
- `train_phase2.py:1237` — **CRITICAL: TASK_763 entropy gate (joint-only at HEAD)**; TASK_776 unblocks factored path.
- `train_phase2.py:1338-1349` — TASK_726 oracle-BC weighting.
- `runs/762_reward_shaping/rs050_seed_7/stage_b_config.json:75, 147` — canon production confirms `bc_max_trade_weight=4.0`, `use_factored_search=true`.
- `runs/760_train_mcts_sims/baseline_paired_seed_*/` — canon paired baselines.
- `runs/recommendation_2026-05-20/analysis.md` — eval-MCTS sims=16 joint recal baseline narrative.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — over-confidence pathway evidence.
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` — single-mechanism class closed verdict (joint sweeps are the response).
- `lore/notes_2026-05-25.md` §"TASK number allocation" — ship-path ceremony contract + TASK_776 dependency.

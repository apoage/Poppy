# 776_rs050_entropy — rs050 + Policy Entropy Combination Sweep

**Status**: SPEC — DRAFT. Code change (factored-path entropy term) landed in worktree `worktree-agent-a55e2ae10d6a66896`; not yet merged + deployed.
**Owner**: server agent executes; workstation operator reviews. Server-stays-under-load contract per `feedback_server_stays_under_load.md`.
**Reference**: `tasks/TASK_776_RS050_ENTROPY_COMBO.md`, `lore/ship_path_recommendations_2026-05-25.md` §2, `experiments/763_entropy_reg/SPEC.md` (parent SPEC for the entropy mechanism).
**Promotion gate (canon)**: paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint, no seed < −5pp.

## Goal

Stack two already-shipped levers on the canon factored search path:

- **rs050** (`reward_scale=50.0`) — 762 W1 reliable, paired Δα +0.27pp, Δtrade-rate −0.243%.
- **entropy_coef** — 763 SPEC mechanism, never run because it was joint-MCTS-only prior to TASK_776 unlocking the factored path.

Both attack `[[classifier-alpha-anticorrelation]]` from orthogonal angles (reward-magnitude vs probability-distribution). The 3-entropy_coef × 3-seed cell matrix tests whether the legs super-add on the canon eval-MCTS sims=16 joint scorecard.

**Hypothesis quick-recap** (full version in TASK_776):
- Sweep `entropy_coef ∈ {0.005, 0.02, 0.05}` × seeds {7, 13, 23} at fixed `reward_scale=50.0`, `min_entropy=0.7`, `use_factored_search=true`.
- Mechanism gate: best cell must show Δ-Bal.Acc ≤ 0 AND Δα > 0 (anti-correlation pathway was the intervention target).
- Promotion gate: paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint, no seed < −5pp.

## Code prerequisites — SHIPPED (worktree `worktree-agent-a55e2ae10d6a66896`)

The factored-path entropy term lands in this worktree at `train_phase2.py:1235-1271` (replaces the joint-only block at the prior `:1235-1249` range), plus a phase2_config docstring refresh at `phase2_config.py:47-58`. ~30 LOC additions, flag-gated, canon bit-identical when `entropy_coef == 0.0`.

**Verification (this worktree)**:
- Diff inspected: outer gate `if entropy_coef > 0.0:` controls both new and original branches. When `entropy_coef == 0.0`, the `else` branch (`policy_entropy = jnp.array(0.0)`) runs unchanged → loss is bit-identical to pre-776 code.
- Standalone JAX smoke (random logits, B=2): factored per-asset H ≈ 1.31 (vs ln(5)=1.61 max), joint H ≈ 5.31 (vs ln(320)=5.77 max), ratio ≈ 4.06 — sane units, no NaN.
- Both `phase2_config.py` and `train_phase2.py` pass `ast.parse(...)` syntax check.

**Path-aware `min_entropy`**: the joint-path `min_entropy ≈ 3.0` used in 763 SPEC targeted ~52% of joint max (5.77). The factored-path analog at 776 is `min_entropy ≈ 0.7` ≈ 43% of factored max (1.61). Operator MUST keep `min_entropy` matched to the path — same `entropy_coef` does not mean same regularization strength across paths because the max-H differs ~3.6×.

## Cells matrix

3 entropy_coef × 3 seeds = 9 training cells; all `reward_scale=50.0`, `use_factored_search=true`, `min_entropy=0.7`.

| cell | reward_scale | entropy_coef | min_entropy | seed | output dir | wall (server) |
|---|---:|---:|---:|---:|---|---:|
| ec005_seed_7  | 50.0 | 0.005 | 0.7 | 7  | `runs/776_rs050_entropy/ec005_seed_7`  | ~2.5 h |
| ec005_seed_13 | 50.0 | 0.005 | 0.7 | 13 | `runs/776_rs050_entropy/ec005_seed_13` | ~2.5 h |
| ec005_seed_23 | 50.0 | 0.005 | 0.7 | 23 | `runs/776_rs050_entropy/ec005_seed_23` | ~2.5 h |
| ec02_seed_7   | 50.0 | 0.02  | 0.7 | 7  | `runs/776_rs050_entropy/ec02_seed_7`   | ~2.5 h |
| ec02_seed_13  | 50.0 | 0.02  | 0.7 | 13 | `runs/776_rs050_entropy/ec02_seed_13`  | ~2.5 h |
| ec02_seed_23  | 50.0 | 0.02  | 0.7 | 23 | `runs/776_rs050_entropy/ec02_seed_23`  | ~2.5 h |
| ec05_seed_7   | 50.0 | 0.05  | 0.7 | 7  | `runs/776_rs050_entropy/ec05_seed_7`   | ~2.5 h |
| ec05_seed_13  | 50.0 | 0.05  | 0.7 | 13 | `runs/776_rs050_entropy/ec05_seed_13`  | ~2.5 h |
| ec05_seed_23  | 50.0 | 0.05  | 0.7 | 23 | `runs/776_rs050_entropy/ec05_seed_23`  | ~2.5 h |

**Total wall**: 9 × 2.5 h = 22.5 h serial; **~12-15 h at 2-wide** on server GPU 1+2 (TASK_742 cap).

## Launch command per cell (Stage-B)

Canon a1b1 Stage-B args + the three sweep knobs (`--reward-scale`, `--entropy-coef`, `--min-entropy`). Base ckpt is the standard `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`, same as every other 76x/77x cell.

```
python -u train_phase2.py \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --stage B \
  --seed 7 \
  --output-dir runs/776_rs050_entropy/ec005_seed_7 \
  --name 776_ec005_s7 \
  --reward-scale 50 \
  --entropy-coef 0.005 \
  --min-entropy 0.7 \
  --factored \
  [... other canon a1b1 flags from runs/762_reward_shaping/rs050_seed_7/stage_b_config.json ...]
```

**Operator pre-launch diff**: each cell's resolved `stage_b_config.json` should differ from `runs/762_reward_shaping/rs050_seed_7/stage_b_config.json` only on `entropy_coef`, `min_entropy`, `seed`, and the output-dir/name strings. Anything else is a flagged misconfiguration — abort and reconcile.

The exact a1b1 Stage-B flag set (from `runs/762_reward_shaping/rs050_seed_7/stage_b_config.json`, post-deploy verified bit-identical to canon for 762 W1):
- `epochs=200`, `batch_size=256`, `steps_per_epoch=200`
- `learning_rate=3e-4`, `weight_decay=1e-4`, `grad_clip=5.0`
- `value_target_mode=nstep`, `nstep_horizon=20`, `td_discount=0.95`
- `use_factored_search=true`, `factored_temperature=0.1`, `factored_sub_batch=4096`
- `bc_class_balanced=true`, `bc_max_trade_weight=4.0`
- `use_direction_head=true`, `direction_loss_weight=0.5`
- `use_rank_head=true`, `rank_loss_weight=1.0`
- `use_aux_price_head=true`, `aux_price_loss_weight=0.1`
- `use_popart=true`, `popart_beta=3e-4`
- `value_clip_mode=pre_popart`

## GPU placement

- **GPU 0**: FORBIDDEN — VirtualBox-reserved (workstation memory standing rule).
- **GPU 1 + GPU 2**: 2-wide concurrent `train_phase2.py`. TASK_742 cap (≤2 concurrent `train_phase2.py`) binds — do not exceed.
- **GPU 3**: parallel — runs eval-MCTS sims=16 joint `benchmark_easy.py` on each completed `phase2_stageB_best.pkl`. Eval is not a `train_phase2.py` process so it does not count against the TASK_742 cap.

Server-stays-under-load priority (per `feedback_server_stays_under_load.md`): the moment a GPU frees, launch the next cell from the matrix. Seed-7 cells first across entropy_coef (early read of all 3 ec values), then seed-13, then seed-23. Standing 15-min-idle-slot rule per `lore/handoff_2026-05-24_764-launch.md` applies to GPU 1+2+3.

## Eval command per cell (after `phase2_stageB_best.pkl` writes)

```
python tools/benchmark_easy.py \
  --checkpoint runs/776_rs050_entropy/ec005_seed_7/phase2_stageB_best.pkl \
  --periods-file config/benchmark_periods.json \
  --eval-mcts-sims 16 --eval-mcts-decode joint \
  --output runs/776_rs050_entropy/eval_ec005_seed_7/benchmark.json
```

Eval ≈ 30 min per cell on GPU 3, overlaps training wall. Eval contract identical to 759/760/762 (`runs/762_reward_shaping/AUTHORIZATION_2026-05-21.md` §"Eval (GPU 3, parallel)").

## Diagnostics to log per epoch

The trainer already logs everything below in `stage_b_metrics.csv` — no new instrumentation needed for 776:

| metric | source | gate / check |
|---|---|---|
| `policy_entropy` | `train_phase2.py:1514` (was 0 on factored runs pre-776; now real per-asset H) | should track between 0.3 and 1.6; floor-hit fraction < 50% |
| `policy_loss` | `train_phase2.py:1513` | should be similar magnitude to 762 W1 rs050 (entropy bonus is small) |
| `mean_trade_rate` | `train_phase2.py:1467-1498` | Δ vs 762 rs050 baseline: ↓ = mechanism engaged (entropy de-sharpens → fewer high-confidence trades) |
| `policy_acc` / Bal.Acc | per-epoch validation metrics | Δ-Bal.Acc ≤ 0 vs 762 rs050 = mechanism engaged (per anti-correlation diagnostic) |

**Floor-hit early-stop**: any cell that sits at `policy_entropy < min_entropy = 0.7` for > 50% of epochs is degenerate (entropy_coef too high or floor too tight for this seed). Mark FAIL, save ckpt as diagnostic data, do NOT include in promotion gate.

## Mechanism gate (early exit before promotion gate)

Compute on Wave-1 completion, BEFORE invoking the promotion gate:

- Δα (vs 762 rs050 paired baseline) > 0 AND Δ-Bal.Acc (vs 762 rs050 paired) ≤ 0 — the anti-correlation pathway was engaged as designed.
- If Δα > 0 AND Δ-Bal.Acc > 0 simultaneously: the gain came from somewhere OTHER than entropy regularization (the pathway was not engaged). Mechanism gate FAIL. Save data, do NOT extend to 5-seed.
- If Δα < 0 across all cells: entropy reg too strong on this stack at all tested coefficients. Mechanism gate FAIL. Save data, close the joint mechanism candidate.

## Promotion gate (canon)

Two paired comparisons (both required to gate):

- **Primary (canon)**: paired Δα ≥ +3pp vs **760 a1b1 sims=32** (`runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}`) under eval-MCTS sims=16 joint, no seed < −5pp.
- **Mechanism isolation**: paired Δα vs **762 rs050** (`runs/762_reward_shaping/rs050_seed_{7,13,23}`). If 776 super-adds, the entropy term contributes on top of rs050 (per-seed Δα ≥ +2pp vs 762 W1's +0.27pp paired result).

**5-seed extension** (post Wave-1 promotion): seeds 31, 42 added at the winning `entropy_coef`. Promote to canon if 5-seed mean Δα ≥ +3pp AND all seeds > −5pp.

## Anti-correlation diagnostic table — write on Wave-1 completion

| cell | Δα vs 760 (pp) | Δα vs 762-rs050 (pp) | ΔBal.Acc vs 760 (pp) | ΔBal.Acc vs 762-rs050 (pp) | mean_trade_rate Δ | floor-hit fraction |
|---|---:|---:|---:|---:|---:|---:|
| ec005_seed_7 | … | … | … | … | … | … |
| (8 more rows) | | | | | | |

The mechanism gate evaluates the **ΔBal.Acc vs 762-rs050** column (the cleaner contrast — isolates the entropy contribution from the rs050 leg). The promotion gate evaluates the **Δα vs 760** column (canon, deploy-truthful).

## Wave 2 (conditional)

If Wave 1 cells consistently hit the floor (`floor-hit fraction > 50% in ≥ 6/9 cells`) but the mechanism gate fires on at least one cell, Wave 2 sweeps `min_entropy ∈ {0.3, 0.5, 0.7}` at the best `entropy_coef`, 3 seeds. Gate on operator decision; default is auto-stop after Wave 1 unless `Wave 1 best paired Δα ≥ +1.5pp vs 760` (mid-quality signal worth tuning) AND ≥ 50% of cells hit the floor (suggests floor was a meaningful constraint).

If Wave 1 closes null on the mechanism gate, do NOT launch Wave 2 — close the joint mechanism candidate and pivot to the next ship-path candidate (`lore/ship_path_recommendations_2026-05-25.md` §"Recommended execution order" item 3 onward).

## Cost summary

| stage | cells | wall (serial) | wall (2-wide) |
|---|---:|---:|---:|
| Wave 1 (3 ec × 3 seeds) | 9 | ~22.5 h | ~12-15 h |
| Wave 1 eval (GPU 3, parallel) | 9 | ~4.5 h serial | overlapped |
| Wave 2 min_entropy sweep (conditional) | 9 | ~22.5 h | ~12-15 h |

Default expected cost (Wave 1 only): ~12-15 h server wall.

## Pre-launch gating on operator (workstation)

This SPEC blocks on:
1. **Worktree merge**: `tasks/TASK_776_RS050_ENTROPY_COMBO.md` and the `train_phase2.py` + `phase2_config.py` edits in this worktree must be merged to master and deployed via `tools/deploy/deploy.sh` (main agent only). The deploy is the load-bearing prerequisite — the factored-path entropy term is what makes 776 distinct from 763.
2. **Existing campaign queue**: 775 (oracle-mask Phase-1 imprint) currently authorized per `lore/handoff_2026-05-24_764-launch.md` may be running on GPU 1/2 — 776 launches when GPU 1/2 free up OR runs concurrently with 775's eval-only stages (GPU 3-bound).

## References

- `tasks/TASK_776_RS050_ENTROPY_COMBO.md` — full task spec.
- `lore/ship_path_recommendations_2026-05-25.md` §2 — primary spec source.
- `experiments/763_entropy_reg/SPEC.md` — entropy-only parent SPEC; 776 stacks rs050 on top.
- `tasks/TASK_763_POLICY_ENTROPY_REG.md` — entropy mechanism design.
- `runs/762_reward_shaping/AUTHORIZATION_2026-05-21.md` — eval contract + GPU scheduling reference.
- `runs/762_reward_shaping/rs050_seed_{7,13,23}/result.md` — rs050 paired baselines (Δα +0.27pp paired W1 reference).
- `runs/762_reward_shaping/rs050_seed_7/stage_b_config.json` — canon a1b1 stage-B flag set with `reward_scale=50.0`.
- `runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}` — canon a1b1 sims=32 baselines (eval-MCTS sims=16 joint already computed).
- `train_phase2.py:1235-1271` (post-776) — modified policy-entropy block.
- `phase2_config.py:47-58` (post-776) — entropy config docstring (factored path noted).
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — primary evidence base.
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` — context: combinations are the next probe class.

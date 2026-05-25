# 781_return_weighted_stack — Return-Weighted BC + 767 + rs050 (3-stack)

**Status**: READY-TO-LAUNCH after deploy of the worktree's TASK_781 commit (`--bc-return-weighted` CLI flag + return-weighted BC site).
**Owner**: workstation implements + smoke-traces + deploys; server agent executes; workstation reviews.
**Reference**: `tasks/TASK_781_RETURN_WEIGHTED_BC_STACK.md`, `lore/ship_path_recommendations_2026-05-25.md` §7, `tasks/TASK_725_BULL_EXPERIMENTS.md` §"Experiment D".

## Goal

Test whether combining three orthogonal levers — return-weighted BC (TASK_725-D), reward-magnitude attenuation (rs050 from TASK_762), and limit-order action prune (TASK_767) — into a **single 3-stack** clears the +3pp Δα promotion gate vs canon. Each leg is individually small-positive-or-pending; the joint hypothesis is super-additive coverage of three failure surfaces.

## Hypothesis

| Lever | Targets | Prior individual result |
|---|---|---|
| `--bc-return-weighted` (TASK_725-D) | Bull-period BC under-weighting (`project_bull_blindness.md`) | Never run — code shipped in worktree TASK_781 commit |
| `--reward-scale 50` (rs050, TASK_762 W1) | Over-trade reward magnitude | Paired Δα +0.27pp at 3 seeds; Δ-trade-rate −0.243% |
| `--disable-limit-orders` (TASK_767) | MCTS branching factor + limit-channel drift | Standalone authorized 2026-05-23; result pending |

Joint hypothesis: each lever attacks a **distinct failure surface** (BC weighting, reward magnitude, action-space branching). Super-additivity would be visible as Δα ≥ sum of individuals (≥ ~0.5pp from rs050 + 767 individual estimates + the rwbc lift) **AND** a bull-regime Bal.Acc lift unique to the rwbc contribution.

## Code prerequisites — SHIPPED in TASK_781 worktree commit

All changes are flag-gated, default-OFF, canon bit-identical:

1. `phase2_config.py:76-81` — `bc_return_weighted: bool = False` (documented field; deferred stub from TASK_726 replaced with a load-bearing comment).
2. `train_phase2_cli.py:71-76` — `--bc-return-weighted` `store_true` flag.
3. `train_phase2.py:1059` — closure-level `use_bc_rwt = bool(getattr(config, 'bc_return_weighted', False))`.
4. `train_phase2.py:1343-1384` — BC loss site: nested return-weight branch inside both `use_bc_balance` and `else` paths. Each path's `else` arm (`use_bc_rwt=False`) is **literally the pre-781 line**.
5. `train_phase2.py:2356-2357` — argparse override.

**Return signal used: `oracle_values`** (oracle remaining PnL over `oracle_value_horizon=400` steps, precomputed from `oracle_cum_pnl` at `train_phase2.py:640-665`). `nstep_rewards` (discounted 20-step reward) is also in the batch but represents the **short-horizon value target**, not the **full-episode expected return** the TASK_725-D spec asks for. Mean-normalization (`|oracle_values| / mean(|oracle_values|)`) keeps the total gradient magnitude comparable to canon — the variant **redistributes** BC gradient by sample, it does not amplify the BC term overall.

**Smoke verification (static trace, no JAX run required)**: the canon path (`use_bc_rwt=False`) in both `if use_bc_balance:` and `else:` arms contains the literal pre-781 expression. Python-level dead-code elimination guarantees a bit-identical JIT trace on canon. Verified separately via `python -c` checks on default config + CLI parse (default = False, `--bc-return-weighted` flips True).

**Required deploy**: worktree commit must be merged to master + deployed to server before launching this experiment. Until deployed, the server's `train_phase2.py` does not recognize `--bc-return-weighted` and the cell would silently train as `rs050+767` only.

## Variants

**1 training cell × 3 seeds**, plus 3 evals.

| cell | reward_scale | disable_limit_orders | bc_return_weighted | use_factored_search | seed | Status |
|---|---:|:---:|:---:|:---:|---:|---|
| 781-rwbc3stack-s7  | 50 | True | True | True | 7  | QUEUED |
| 781-rwbc3stack-s13 | 50 | True | True | True | 13 | QUEUED |
| 781-rwbc3stack-s23 | 50 | True | True | True | 23 | QUEUED |

`use_factored_search=True` is the canon Stage-B search path (per `runs/762_reward_shaping/*/stage_b_config.json:147`). `bc_class_balanced=True` is also canon (default in `phase2_config.py:74`) — no need to flip it on; the **`--bc-return-weighted` flag composes with** the class-balance weights (per-asset trade/hold weight × per-sample return-magnitude weight).

**Baseline pair selection (ranked preference, server agent picks the strongest available):**
1. **First choice — TASK_779 rs050+767 paired result** if its 3 cells are eval'd by the time this cell completes. This is the immediate-prior pair: `(rs050+767+rwbc) − (rs050+767)` isolates the rwbc contribution.
2. **Fallback — 760 a1b1 sims=32 baselines** (`runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}` + `eval_baseline_paired_seed_*`). The canon scorecard. Pair the 3-stack against canon and report both numbers if TASK_779 lands mid-flight.

**Promotion gate evaluates against 760 a1b1** in both cases (deploy-truthful canon).

## Launch command (per cell)

Canon a1b1 Stage-B flags, plus `--reward-scale 50 --disable-limit-orders --bc-return-weighted --seed S`. Run order: seed 7 first (early-exit gate check at epoch 50), then 13 and 23 in parallel.

```bash
mkdir -p runs/781_return_weighted_stack/rwbc3stack_seed_${SEED}

nohup python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --stage B --epochs 200 \
  --periods-file config/foxglove_periods.json \
  --factored \
  --no-reset-value-head \
  --value-target-mode oracle \
  --reward-scale 50 \
  --disable-limit-orders \
  --bc-return-weighted \
  --seed ${SEED} \
  --output-dir runs/781_return_weighted_stack/rwbc3stack_seed_${SEED} \
  > runs/781_return_weighted_stack/rwbc3stack_seed_${SEED}/stage_b_training.log 2>&1 &
```

Note: copy the exact arg set used by `runs/762_reward_shaping/rs050_seed_7/stage_b_training.log` and `runs/767_action_space/*/stage_b_training.log` (when TASK_779 is queued/running) as the canon-a1b1 base — diff against `runs/760_train_mcts_sims/baseline_paired_seed_7/stage_b_config.json` before launching each cell to confirm only `reward_scale`, `disable_limit_orders`, `bc_return_weighted`, and `seed` differ from canon.

## GPU placement

Per workstation↔server contract + TASK_742 ≤2-concurrent rule:

- **Training (GPU 1 + GPU 2, 2-wide)**: cell s7 first (early-exit gate). When s7 passes E50 trade-rate check, launch s13 + s23 to fill both GPUs.
- **Eval (GPU 3, parallel)**: as each cell writes `phase2_stageB_best.pkl`, launch its eval on GPU 3 immediately — `benchmark_easy.py` is inference, not `train_phase2.py`, so the cap does not apply.

GPU 0 stays forbidden (VirtualBox-reserved per MEMORY.md).

## Wall estimate

- 3 training cells × ~2.5 h/cell, 2-wide on server ≈ **5 h training wall**.
- 3 eval cells × ~30 min/cell, on GPU 3 in parallel with training ≈ **0 h additional wall** (overlapped).
- Total: **~5 h wall**, plus the upstream deploy.

## Early-exit gate (epoch 50, seed 7)

Anti-correlation risk for this stack is **MEDIUM** (per ship-path memo §7). Return-weighted BC could sharpen the classifier in the high-return regime — the 738 failure mode (−18.44pp via trade-rate explosion).

**At epoch 50 of `rwbc3stack_seed_7`, read `stage_b_metrics.csv`**:

| Diagnostic | Compute | Threshold | Action if triggered |
|---|---|---|---|
| `mean_trade_rate` (epoch 50) | from CSV col | ≤ 2× the canon a1b1 trade-rate at matched epoch (`runs/760_train_mcts_sims/baseline_paired_seed_7/stage_b_metrics.csv`, row epoch=50) | Continue if pass |
| Same | Same | > 2× canon | **ABORT all three cells**; write `result.md` documenting the over-trade-explosion signature; do NOT launch s13/s23 |

If the gate fires only on rwbc3stack but rs050+767 (TASK_779) at the same epoch did **not** explode trade-rate, **peel rwbc off first**. That isolates the return-weighting term as the cause; future variants might try rwbc with a per-sample weight cap (`min(weight, 5.0)`) or a smaller mean-normalization radius.

## Promotion gate

- **Standard gate**: paired Δα ≥ **+3pp** vs 760 a1b1 sims=32 under `benchmark_easy.py --eval-mcts-sims 16 --eval-mcts-decode joint`. **No seed < −5pp.**
- **Bull diagnostic (mechanism gate)**: Δ-Bal.Acc on bull-regime benchmark periods ≥ **+5pp** vs canon (`project_bull_blindness.md` target). If standard gate clears but bull diagnostic does not, the win came from rs050/767 and not from the rwbc leg — flag it; rwbc gets demoted to "neutral" and is not promoted into canon for follow-up stacks.
- **5-seed extension** if 3-seed clears both gates: add seeds {31, 42} → paired with the matching 760 a1b1 5-seed if available, else extend the 760 baseline first.

## Eval protocol

Per the eval-MCTS recal contract (canon since 2026-05-19, `project_benchmark_live_gap.md`):

```bash
CUDA_VISIBLE_DEVICES=3 python tools/benchmark_easy.py \
  --checkpoint runs/781_return_weighted_stack/rwbc3stack_seed_${SEED}/phase2_stageB_best.pkl \
  --periods-file config/benchmark_periods.json \
  --eval-mcts-sims 16 --eval-mcts-decode joint \
  --output runs/781_return_weighted_stack/eval_rwbc3stack_seed_${SEED}/benchmark.json
```

The eval-MCTS sims=16 joint scorecard. Matches 759/760/762 protocol — no recompute drift.

## On completion — write result.md

When all 3 cells trained + eval'd:

1. Compute paired Δα vs 760 a1b1 sims=32 (the canon-pair). Also report unpaired vs 779 rs050+767 if those numbers exist.
2. Compute per-period Bal.Acc Δ on bull regimes (the `project_bull_blindness.md` mechanism diagnostic).
3. Write `runs/781_return_weighted_stack/result.md` with the verdict:
   - **PROMOTE** if standard gate clears AND bull diagnostic clears: queue 5-seed extension.
   - **NEUTRAL (alpha win, no bull lift)** if standard gate clears but bull diagnostic fails: report; do not promote into canon.
   - **NULL** if neither: report; close TASK_781 with a negative-but-informative entry.
   - **REGRESSION** if Δα < 0 or any seed < −5pp: report; if early-exit gate already fired, paste E50 trade-rate evidence.

## References

- `tasks/TASK_781_RETURN_WEIGHTED_BC_STACK.md` — task spec.
- `lore/ship_path_recommendations_2026-05-25.md` §7 — candidate framing.
- `tasks/TASK_725_BULL_EXPERIMENTS.md` §"Experiment D" — original ~10 LOC code-change design.
- `tasks/TASK_726_*` — class-balanced BC (canon, composes with rwbc).
- `tasks/TASK_762_REWARD_SHAPING.md` + `runs/762_reward_shaping/AUTHORIZATION_2026-05-21.md` — rs050 Wave 1 +0.27pp result.
- `tasks/TASK_767_ACTION_SPACE_RESTRUCTURE.md` + `experiments/767_action_space/SPEC.md` — limit-order action prune.
- `runs/762_reward_shaping/rs050_seed_*/stage_b_config.json` — rs050 launch config to template against.
- `runs/760_train_mcts_sims/baseline_paired_seed_*/` — canon a1b1 sims=32 baselines + their eval-MCTS sims=16 evals.
- `~/.claude/projects/-home-lukas-Trader/memory/project_bull_blindness.md` — bull-regime failure root cause.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — anti-correlation failure mode (738 sharpening).
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` — single-mechanism class closed; combinations are the unexplored space.

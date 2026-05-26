# TASK_779 — rs050 + 767 Combination (Reward-Scale × Action-Space)

**Status**: DRAFT — zero code, ready to launch. Both flags shipped + production-deployed; the combination has never been jointly tested.
**Created**: 2026-05-25
**Ship-path candidate**: #5 of 10 per `lore/ship_path_recommendations_2026-05-25.md`. **Lowest-cost candidate on the list.**
**Promotion gate (canon)**: paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint, no seed < −5pp.

## Hypothesis

Two production-deployed mechanisms attack different bottlenecks of the `[[classifier-alpha-anticorrelation]]` pathway and have never been jointly tested:

- **rs050** (`reward_scale=50`, half-canon): attenuates per-step reward magnitude — a **gradient-magnitude lever**. 762 W1 (`lore/notes_2026-05-23.md` §167) produced paired Δα **+0.27pp** with Δtrade-rate **−0.243%** across 3 seeds. Mechanism gate PASS (over-trade pathway responsive), standard gate FAIL (+0.27pp ≪ +3pp). Wave 2 (rs050 + fee_penalty_bps) closed FAIL — the fee-on-top mechanism is exhausted. rs050 standalone is the small-positive lower bound of the reward-shaping class.
- **767** (`disable_limit_orders=True`): masks LimitSell (3) + LimitBuy (4) from the policy logits and the MCTS expansion — a **search-branching-factor lever**, collapsing 64×5=320 → effective 64×3=192. The oracle never emits limit actions (`mcts_trader/jax_agent/path_solver.py:406, 473`), so the Phase-1 imprint never trains policy mass into the limit channel; 767 prevents Phase-2 search from drifting into it. The canon factored-search smoke (`experiments/767_action_space/SPEC.md` §6) measured `limit_action_fraction = 0.398` on the 5-action baseline — ~40% of the per-asset probability mass sits on the limit channel under canon, so 767 is a genuine intervention with substantial capacity to remove, not a cheap branching-factor confirmation. 767 standalone is queued/in-flight per `experiments/767_action_space/SPEC.md` (status `READY-TO-LAUNCH` post-TASK_771/F1 fix).

**Joint hypothesis**: rs050 reduces the magnitude of the reward signal feeding the value head while 767 narrows the policy / search action surface. Both are CLI-orthogonal — combining them in one cell costs the same wall as either alone. If the two levers attack non-overlapping failure modes, their combined Δα is **super-additive** (mechanism interaction → ship-worthy). If they attack the same surface, the sum collapses to either-leg's individual contribution. The campaign has not seen this interaction tested.

**Falsifiable in three gates**:
- **Anti-correlation diagnostic (early-exit)**: combination must NOT show Δ-Bal.Acc > 0 AND Δα < 0 at 3-seed mean. Triggers TIER-3 closure of the joint mechanism class on this stack.
- **Mechanism interaction gate**: combination Δα must beat the better of {rs050-only, 767-only} at the 3-seed mean (super-additive or at least non-destructive). Else the two levers were redundant and the combination is no improvement over picking one.
- **Standard promotion gate (canon)**: paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint AND no seed < −5pp → 5-seed extension (seeds 31, 42).

## Verified code state (workstation tip `79430e3`)

Both flags are production-deployed with config-default OFF. Verified by reading the cited file:line ranges directly.

**rs050 (reward-scale)** — shipped commit `b4e0158` ("TASK_762/763: wire reward_scale + fee_penalty_bps + entropy_coef + min_entropy"):
- Config field: `phase2_config.py:30` — `reward_scale: float = 100.0` (canon = 100, sweep ∈ {25, 50, 100, 200} per 762 SPEC).
- Per-step reward wire: `train_phase2.py:637` — `rewards = np.clip(step_pnl * self.config.reward_scale, -10.0, 10.0)`.
- N-step value-target wire: `train_phase2.py:679` — `per_step_r = per_step_pnl * self.config.reward_scale`.
- CLI: `train_phase2_cli.py:71-73` — `--reward-scale FLOAT` (default None = inherit).
- Override plumb: `train_phase2.py:2348-2349` — `if args.reward_scale is not None: config.reward_scale = args.reward_scale`.
- (Brief cited `train_phase2.py:632-634` for the rs050 wire — that range falls inside the same numpy block but the actual `* config.reward_scale` multiplication sits at **line 637 and 679** at current tip post the TASK_762 F1 fix and TASK_772.1 modularity refactor. See §"Code-state divergences from brief" below.)

**767 (disable_limit_orders)** — shipped commit `7f41586` ("TASK_767: flag-gated action-space restructure (drop limit orders)"), with bug-fixes in commit `931f81f` ("TASK_771: 767 bug-fixes — F1 (P0) mean-shift, F2 (P1) mctx interior mask, F4 (P2) Stage-B resume"):
- Config field: `phase2_config.py:45` — `disable_limit_orders: bool = False` (canon = False, bit-identical pre-767).
- JIT-trace-time mask build: `train_phase2.py:1060-1090` — builds `limit_keep_mask` `(320,) {0,1}` and `limit_logit_mask` `(320,) {-1e9, 0}` arrays (factored + joint policy paths). The `-1e9` (not `-inf`/`finfo.min`) is load-bearing per TASK_767 §"Code prerequisites" — keeps `log_softmax` finite while driving softmax probability to exactly 0.0 on masked actions. (Brief cited `1060-1081`; actual range at `79430e3` extends to **1090** because the keep_mask + helper closure are still inside that block.)
- Train-time MCTS expansion mask: `train_phase2.py:1742-1762` — builds `(B, 320)` int8 mask passed to joint MCTS as `invalid_actions`. Factored-search masking lives in `mcts_trader/jax_agent/factored_search.py:190-197` (Q-values clamped to -1e9 before per-asset softmax + action-selection argmax) plus the TASK_771/F1 mean-shift removal in `_q_to_policy` (load-bearing P0 fix — without it the policy training target collapsed to uniform across valid actions). (Brief cited `1742-1754`; actual range at `79430e3` extends to **1762**.)
- CLI: `train_phase2_cli.py:89-98` — `--disable-limit-orders` (`BooleanOptionalAction`, default None = inherit; `--no-disable-limit-orders` also accepted post TASK_771/F4).
- Override plumb: `train_phase2.py:2355-2356` — `if args.disable_limit_orders is not None: config.disable_limit_orders = bool(args.disable_limit_orders)`. (Brief cited `2332-2333`; actual is **2355-2356** post the TASK_772.1 argparse extraction.)
- Eval-MCTS mask auto-detection: `tools/benchmark_easy.py:931-942` reads `disable_limit_orders` from the checkpoint config and threads the mask through `eval-mcts-sims` calls at `:544-548 / :633-638`. Eval-time consistency is automatic — no extra CLI needed at benchmark step.

**Per `lore/notes_2026-05-23.md` §165**, the 767 ship was P0-bugged at first (uniform `mean_trade_rate ≈ 0.667`); TASK_771 batch caught and fixed it pre-launch via the multi-agent QA pass. Post-fix smoke (`experiments/767_action_space/SPEC.md` §6) shows `mean_trade_rate` off the uniform fingerprint and `limit_action_fraction = 0.000` (search excludes limits) on the flag-ON cell vs `0.398` on canon.

## Code-change required: NONE

Both flags are CLI-overridable at production tip. The cell launches with `--reward-scale 50 --disable-limit-orders` and no other code touches the Phase-1 ckpt or the Stage-B canon — the comparability contract with the 760 a1b1 baselines and the 762 rs050 cells is preserved.

## Variants

Single experimental cell, 3 seeds. Three baselines for paired-Δ analysis (all already trained + eval'd):

| cell | reward_scale | disable_limit_orders | seeds | trained? | wall (server, est.) |
|---|---:|---|---|---|---:|
| 779-rs050_767 | 50 | True | {7,13,23} | NO — this TASK launches them | ~2.5 h/seed |
| **Baselines (paired, already on disk):** | | | | | |
| 760-a1b1 (canon absolute) | 100 | False | {7,13,23} | reuse `runs/760_train_mcts_sims/baseline_paired_seed_*` | 0 |
| 762-rs050 (rs050 isolated leg) | 50 | False | {7,13,23} | reuse `runs/762_reward_shaping/rs050_seed_*` + their `eval_rs050_seed_*` | 0 |
| 767-3action (767 isolated leg) | 100 | True | {7,13,23} | reuse 767 standalone result whenever it lands per `experiments/767_action_space/SPEC.md` | 0 (in-flight or queued) |

The 767-3action leg is the only baseline that may not yet exist at TASK_779 launch time. If `runs/767_action_space/baseline_seed_*` is not yet on disk, the mechanism-interaction gate (rs050+767 vs best of {rs050-only, 767-only}) falls back to comparing against **rs050-only** until 767-3action lands — this is acceptable because rs050 is the harder lower bound (+0.27pp paired vs canon).

## Promotion gate

Three gates, evaluated in order (early-exit on FAIL):

1. **Anti-correlation diagnostic (early-exit)**:
   - Read Δ-Bal.Acc and Δα at 3-seed mean from the per-cell eval JSON.
   - **FAIL** if Δ-Bal.Acc > 0 AND Δα < 0 → TIER-3 closure of joint reward+search mechanism class on this stack. Stop.
   - **PASS** otherwise → proceed to mechanism-interaction gate.

2. **Mechanism-interaction gate (combination meaningful)**:
   - Compute Δα_combo (vs 760 canon), Δα_rs050 (rs050 W1 result, +0.27pp), Δα_767 (767 standalone, value TBD).
   - **FAIL** if Δα_combo ≤ max(Δα_rs050, Δα_767) → combination is non-additive (the two levers were redundant; pick the cheaper one). Stop the campaign branch but log the result.
   - **PASS** if Δα_combo > max(Δα_rs050, Δα_767) → mechanism interaction is real (super-additive or at least non-destructive); proceed to standard gate.

3. **Standard promotion gate (canon)**:
   - paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint.
   - No seed < −5pp.
   - **PASS** → 5-seed extension (seeds 31, 42) → if confirmed (mean Δα ≥ +3pp, no seed < −5pp), ship.

## Cost estimate

- **Training**: 3 seeds × 2.5 h/seed = 7.5 h serial, **~4-5 h wall at 2-wide on server** (GPU 1 + GPU 2).
- **Eval**: 3 cells × ~30 min on GPU 3 (parallel with training).
- **Total**: ~5 h wall, ~8-9 GPU-hours. **Lowest cost candidate on the ship-path list** (per `lore/ship_path_recommendations_2026-05-25.md` Recommended execution order).

## Risk + anti-correlation check

**Anti-correlation risk: LOW.** Per `lore/ship_path_recommendations_2026-05-25.md` §5 anti-correlation check:
- 767 prunes options that the oracle never selects (`mcts_trader/jax_agent/path_solver.py:406, 473` — 0% limit usage). It does not sharpen the policy head's per-asset 5-class softmax (the surface that `project_classifier_alpha_anti_correlation.md` line 27 implicates as the bottleneck).
- rs050 attenuates reward magnitude going into the value-target. It does not sharpen the per-asset classifier output either.

Both interventions are downstream of the encoder representation and orthogonal to the classifier sharpening pathway. Closest priors:
- 762 rs050 W1: Δα +0.27pp paired, mechanism gate PASS, no anti-correlation signature observed.
- 767 standalone: pre-fix smoke had a uniform-collapse bug (TASK_771/F1, P0) that would have masqueraded as a null; post-fix smoke (commit `931f81f`) shows clean trade-rate behavior. The TASK_771 multi-agent QA pass (`lore/notes_2026-05-23.md` §165) added the bug-catch protocol that makes this trustable.

**Worst-case**: combination is null vs both legs (mechanism-interaction gate FAILs at step 2) — the two levers attack overlapping failure modes and the campaign learns the over-trade pathway has only one degree of freedom on this stack. Cost-of-falsification: ~8-9 GPU-hours.

**Best-case**: combination clears +3pp, 5-seed confirms, ship. The single-mechanism class is closed (`project_single_mechanism_exhausted.md`); a 2-mechanism stack that ships would be the first paired-Δα win since 746 +0.72pp 5-seed.

## Why this is candidate #5 (and the recommended first launch)

Per `lore/ship_path_recommendations_2026-05-25.md` "Recommended execution order":
- **#1 priority**: TASK_775 oracle-mask aux head (Phase-1 re-imprint, ~13-15 GPU-h, requires re-imprint).
- **#5 priority**: this TASK — 0 new code, ~8-9 GPU-h, no re-imprint, both legs are production-deployed.

The brief framing: "the cheapest probe on the list" — both legs of the combination were independently zero-to-positive (rs050 +0.27pp confirmed, 767 hypothesis is pending standalone result), the anti-correlation risk is LOW by mechanism construction, and the cost is bounded by the existing 2.5 h/seed Stage-B baseline. If small-perturbation combinations are going to ship, this and TASK_776 (rs050 + entropy) are the two most prior-favorable candidates.

If TASK_779 clears the canon gate, the 5-seed extension confirms before TASK_775 finishes Phase-1 re-imprint — a fast "ship from small candidate" path.
If TASK_779 closes null, server pivots to TASK_775 / TASK_776 per the ranked queue.

## References

- `lore/ship_path_recommendations_2026-05-25.md` §5 — primary spec source for this TASK.
- `tasks/TASK_767_ACTION_SPACE_RESTRUCTURE.md` — original 767 design (Gemini-proposed 2026-05-21).
- `experiments/767_action_space/SPEC.md` — 767 standalone SPEC, READY-TO-LAUNCH (post TASK_771/F1 fix in commit `931f81f`). Cites the oracle audit + smoke + bug history.
- `experiments/762_reward_shaping/SPEC.md` — 762 SPEC, rs050 wave 1 source (READY-TO-LAUNCH, shipped `b4e0158`).
- `runs/762_reward_shaping/AUTHORIZATION_2026-05-21.md` + `runs/762_reward_shaping/AUTHORIZATION_WAVE2_2026-05-22.md` — style references for AUTHORIZATION format.
- `runs/762_reward_shaping/wave1_status.md` + `lore/notes_2026-05-23.md` §167 — rs050 W1 +0.27pp result.
- `runs/762_reward_shaping/rs050_seed_{7,13,23}` (+ `eval_rs050_seed_*`) — rs050-only baseline checkpoints + benchmarks (paired-Δ source).
- `runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}` (+ `eval_baseline_paired_seed_*`) — canon absolute baseline.
- `lore/notes_2026-05-23.md` §165 — TASK_771 multi-agent QA pass (the 767 P0 catch).
- `phase2_config.py:30, 45` — `reward_scale` + `disable_limit_orders` config fields (verified at `79430e3`).
- `train_phase2.py:637, 679` — rs050 reward-scale wire (verified at `79430e3`).
- `train_phase2.py:1060-1090, 1742-1762` — 767 mask build + MCTS expansion guard (verified at `79430e3`).
- `train_phase2.py:2348-2349, 2355-2356` — CLI override plumbing (verified at `79430e3`).
- `train_phase2_cli.py:71-73, 89-98` — `--reward-scale` + `--disable-limit-orders` argparse (verified at `79430e3`).
- `mcts_trader/jax_agent/path_solver.py:406, 473` — oracle 0% limit-action usage (audit closing TASK_767 §Risk).
- `mcts_trader/jax_agent/factored_search.py:190-197` — factored-search Q-value mask + the TASK_771/F1 mean-shift removal.
- `tools/benchmark_easy.py:931-942, 544-548, 633-638` — eval-time mask auto-detection.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` (line 27 = bottleneck is downstream of encoder).
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` — single-mechanism class closed verdict.

## Code-state divergences from brief

For full transparency, the user's brief cited a small number of line numbers that drifted at workstation tip `79430e3`. The cited file:line ranges have been re-verified by reading the source directly; the cited *mechanism* is correct in every case but the line numbers have shifted, primarily because of:
- TASK_772.1 (commit `55404a0`) extracted argparse from `train_phase2.py` to `train_phase2_cli.py` — every flag definition moved out of train_phase2.py.
- TASK_771 batch (commits `931f81f` + `12c61bb`) added the value-clip-mode switch + the 767 mean-shift fix, shifting downstream line numbers in `train_phase2.py`.
- TASK_762 F1 (commit `bbcb4b0`) extended `fee_penalty_bps` to value targets, inserting lines in the reward-target block.

| Brief citation | Actual at `79430e3` | Status |
|---|---|---|
| `phase2_config.py:30` (rs050 default 100.0) | `phase2_config.py:30` — `reward_scale: float = 100.0` | CONFIRMED (no drift) |
| `phase2_config.py:45` (767 default False) | `phase2_config.py:45` — `disable_limit_orders: bool = False` | CONFIRMED (no drift) |
| `train_phase2.py:1060-1081` (767 mask) | `train_phase2.py:1060-1090` — same block, extends 9 lines further to include keep_mask helper closure | DRIFTED (+9) |
| `train_phase2.py:1742-1754` (MCTS mask) | `train_phase2.py:1742-1762` | DRIFTED (+8) |
| `train_phase2.py:2332-2333` (767 CLI) | `train_phase2.py:2355-2356` (override plumb); `train_phase2_cli.py:89-98` (argparse) | DRIFTED (+23) + moved to cli module |
| `train_phase2.py:632-634` (rs050 wire) | `train_phase2.py:637, 679` (the actual `* config.reward_scale` multiplications) | DRIFTED — brief range is rs050-adjacent but not the load-bearing wire |
| `runs/767_action_space/AUTHORIZATION_2026-05-23.md` | Does not exist | NOT FOUND — closest is `experiments/767_action_space/SPEC.md` (status `READY-TO-LAUNCH`) + `runs/762_reward_shaping/AUTHORIZATION_WAVE2_2026-05-22.md` |
| `runs/762_reward_shaping/rs050_seed_*/result.md` | Does not exist per-seed | NOT FOUND — W1 outcome lives in `runs/762_reward_shaping/wave1_status.md` + `lore/notes_2026-05-23.md` §167 (the +0.27pp number is confirmed). Per-cell artifacts at `runs/762_reward_shaping/rs050_seed_*/` are training-only (`phase2_stageB_best.pkl`, `stage_b_config.json`, etc); eval-MCTS sims=16 joint outputs live at `runs/762_reward_shaping/eval_rs050_seed_*/benchmark.json`. |

None of these divergences are load-bearing for TASK_779 — the CLI flags + mechanism are intact. The SPEC at `experiments/779_rs050_767/SPEC.md` cites the corrected file:line ranges.

## Implementation log

(reserved — populate at execution time)

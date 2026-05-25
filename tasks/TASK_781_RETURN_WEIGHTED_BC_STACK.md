# TASK_781 — Return-Weighted BC (TASK_725-D) + 767 + rs050 (3-stack)

**Status**: DRAFT — small code change (~12 LOC) + 3-way combination experiment.
**Code shipped 2026-05-26** in this worktree commit (see commit SHA in report). Flag-OFF default keeps canon path bit-identical.
**Created**: 2026-05-25 (ship-path ceremony candidate #7).
**Estimated wall**: ~2 h code (done) + ~5 h experiment wall (3 seeds × 2.5 h × 2-wide).

## Hypothesis

The class-balanced BC loss (TASK_726, shipped) gives trades up to 64× more weight than holds, but treats **all trades as equally informative**. **Return-weighted BC** (TASK_725-D) weights each oracle decision by the magnitude of its expected forward return — bull-period BUY signals get more gradient than sideways BUY signals. This directly attacks `project_bull_blindness.md` root cause.

Combined with **rs050** (reward-magnitude attenuation, TASK_762 W1 +0.27pp paired) and **767** (action-space prune: drop limit orders, TASK_767), the stack covers three orthogonal failure surfaces:

| Surface | Lever | Class |
|---|---|---|
| Bull-period BC gradient under-weighting | TASK_725-D `bc_return_weighted` | BC weighting |
| Over-trade reward magnitude | TASK_762 `reward_scale=50` | Reward channel |
| Search drift to never-emitted limit actions | TASK_767 `disable_limit_orders` | Action-space prune |

## Verified code state (read pre-edit)

- TASK_726 class-balanced BC: `train_phase2.py:1053-1054` (closure capture) and `:1339` (BC loss site; specifically the per-asset CE → weighted sum branch).
- `use_bc_balance` (line 1053), `bc_max_trade_weight` (line 1054).
- `phase2_config.py:81` already had `bc_return_weighted: bool = False` declared as a deferred field (TASK_726 comment: "Deferred: return-magnitude weighted boost (Change 3)").
- `train_phase2_cli.py:66-69` had `--no-bc-balance` and `--bc-max-trade-weight` but **no** `--bc-return-weighted` CLI flag (confirmed via grep before edit).
- BC site closure (`make_stage_b_train_step`) already receives `oracle_values` as an argument (`train_phase2.py:1146` signature, `:1164` docstring), populated from `oracle_cum_pnl` via `train_phase2.py:640-665` (`oracle_cum_pnl[end_idx] - oracle_cum_pnl[decision_indices]`).

## Code-change scope (shipped)

**Total: ~30 LOC across 3 files, flag-gated, default-OFF, canon bit-identical.**

1. `phase2_config.py:76-81` — replace the deferred 1-line stub with a documented field declaration. No semantic change to the canon default (`bc_return_weighted: bool = False`).
2. `train_phase2_cli.py:71-76` — add `--bc-return-weighted` `store_true` flag.
3. `train_phase2.py:1059` — closure-level constant `use_bc_rwt = bool(getattr(config, 'bc_return_weighted', False))`.
4. `train_phase2.py:1343-1384` — BC loss site: nest the new `if use_bc_rwt:` branch **inside** each of the existing `use_bc_balance` and `else` paths. The `use_bc_rwt=False` branch in each path is **the exact pre-781 line** — bit-identical to TASK_726.
5. `train_phase2.py:2356-2357` — argparse override `if args.bc_return_weighted: config.bc_return_weighted = True`.

## Return-signal choice — `oracle_values` (oracle remaining PnL)

The TASK_725-D spec (`tasks/TASK_725_BULL_EXPERIMENTS.md` §"Experiment D", line 225) uses `oracle_values`. Two reasons it is the right signal here:

- **Already in batch** — `oracle_values` is precomputed for every sample in the segment batch at `train_phase2.py:640-665` from `oracle_cum_pnl` and threaded all the way to the train step (signature line 1146). No new precompute, no new closure plumbing.
- **Right horizon** — `oracle_values = oracle_cum_pnl[end_idx] - oracle_cum_pnl[decision_indices]`, i.e. remaining PnL over `oracle_value_horizon` (default 400 steps, full episode). This is the **expected forward return magnitude** the spec calls for (high in bull periods where the oracle's chosen-asset path has big PnL, low in sideways periods).

`nstep_rewards` is the alternative — discounted N-step reward sum over `nstep_horizon=20`. It is also in the batch, but it represents the **short-horizon reward** (used for n-step value targets), not the full-episode expected return. The TASK_725-D spec is explicit on using the oracle remaining PnL signal; we follow it.

Mean-normalization (`|oracle_values| / mean(|oracle_values|)`) keeps the **total gradient magnitude comparable** to canon (the per-sample weights average to 1.0), so the BC term contribution to the joint loss stays in the same order of magnitude — the variant is a **redistribution** of BC gradient across samples, not an amplification.

## Anti-correlation risk — MEDIUM

Return-weighted BC could **sharpen the classifier in the high-return regime** — exactly the failure mode TASK_738 hit (`project_classifier_alpha_anti_correlation.md`, −18.44pp). Mitigations baked into the 3-stack:

- **767 (limit-order prune)** narrows what the classifier can choose — 320 → effective 192 actions.
- **rs050 (reward_scale=50)** attenuates the per-step reward magnitude that drives over-confidence.

**EARLY-EXIT GATE**: at epoch 50 of seed-7, if trade-rate > 2× canon (i.e. ≥ 2× the 760 a1b1 trade-rate at the same epoch), **abort the cell and ship a negative result.md**. The over-trade-explosion signature is the classic anti-correlation flag (`project_single_mechanism_exhausted.md`).

If the early-exit fires only on rwbc-3-stack but not rs050+767 (TASK_779), peel rwbc off first — that isolates return-weighting as the cause.

## Variants

**1 cell × 3 seeds** — small probe, paired against the strongest available canon.

| cell | reward_scale | disable_limit_orders | bc_return_weighted | seeds | baseline |
|---|---:|:---:|:---:|---|---|
| 781-rwbc3stack | 50 | True | True | {7, 13, 23} | (a) 779 rs050+767 (if results in by then), else (b) 760 a1b1 sims=32 |

Baseline pick rationale: if TASK_779 (rs050+767, candidate #5 in ship-path memo) has paired Δα and per-seed numbers by the time this cell completes, use it as the immediate-prior pair — that isolates the rwbc contribution. If 779 has not yet completed, fall back to 760 a1b1 sims=32 — the canon scorecard. In either case the **promotion gate compares the 3-stack to 760 a1b1** (the deploy-truthful canon).

## Promotion gate

- **Standard gate**: paired Δα ≥ **+3pp** vs 760 a1b1 sims=32 under `benchmark_easy.py --eval-mcts-sims 16 joint`, **no seed < −5pp**.
- **Bull diagnostic** (mechanism check): Δ-Bal.Acc on bull-regime benchmark periods ≥ **+5pp** vs canon (per `project_bull_blindness.md` implied target). This is the "did the hypothesis fire" gate. If Δα ≥ +3pp but bull Bal.Acc Δ < +5pp, the win came from elsewhere — flag it.
- **5-seed extension** if 3-seed clears both gates.

## Cost

- Code: ~2 h workstation (shipped in this worktree).
- Smoke: static trace through flag-OFF path documented in commit message (no JAX run required — Python-level branch dead under JIT trace).
- Experiment wall: 3 seeds × ~2.5 h × 2-wide = ~5 h server wall.

## References

- Ship-path memo: `lore/ship_path_recommendations_2026-05-25.md` §7.
- `tasks/TASK_725_BULL_EXPERIMENTS.md` §"Experiment D" — spec for the return-weighted BC change (~10 LOC).
- `tasks/TASK_726_*` — class-balanced BC, in production (`use_bc_balance`, `bc_max_trade_weight`).
- `tasks/TASK_762_REWARD_SHAPING.md` + `runs/762_reward_shaping/wave1_status.md` — rs050 Wave 1 +0.27pp paired (`reward_scale=50`).
- `tasks/TASK_767_*` + `runs/767_action_space/AUTHORIZATION_2026-05-23.md` — limit-order action-space prune.
- `~/.claude/projects/-home-lukas-Trader/memory/project_bull_blindness.md` — bull-regime failure root cause (oracle k=4/64 imbalance).
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — anti-correlation failure mode (738-style sharpening).
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` — single-mechanism class closed; combinations are the unexplored space.
- `phase2_config.py:76-81` — `bc_return_weighted` field declaration.
- `train_phase2.py:1059` — closure capture.
- `train_phase2.py:1343-1384` — BC loss site with return-weight branch.
- `train_phase2_cli.py:71-76` — `--bc-return-weighted` CLI flag.

## Smoke verification (static trace, no JAX run)

Performed at edit time. Path inspected: `make_stage_b_train_step` BC block with `config.bc_return_weighted=False` (default).

- Closure-level `use_bc_rwt = bool(getattr(config, 'bc_return_weighted', False))` evaluates to Python `False` at JIT-trace time.
- Both `if use_bc_balance: … if use_bc_rwt:` and `else: … if use_bc_rwt:` branches: the `else:` arm contains **the literal pre-781 line** (`bc_loss = jnp.sum(per_asset_ce * bc_weights) / jnp.sum(bc_weights)` and `bc_loss = per_asset_ce.mean()` respectively). Python-level dead-code elimination guarantees the JIT trace records only the canon expression.
- No new JAX ops on the canon path; no closure-state changes; no batch-dict additions.

**Conclusion**: `--bc-return-weighted` absent (or `config.bc_return_weighted=False`) produces a JAX graph bit-identical to pre-781 in both `use_bc_balance` and `not use_bc_balance` modes.

Verified separately:
- `python -c "from phase2_config import Phase2Config; assert Phase2Config().bc_return_weighted is False"` — passes.
- `python -c "import sys; sys.argv=['t','--checkpoint','x']; from train_phase2_cli import parse_args; assert parse_args().bc_return_weighted is False"` — passes.
- Adding `--bc-return-weighted` flips `args.bc_return_weighted` to True — passes.
- `python -c "import ast; ast.parse(open('train_phase2.py').read())"` — passes.

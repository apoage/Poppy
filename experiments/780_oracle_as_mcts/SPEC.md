# 780_oracle_as_mcts — Oracle-as-MCTS Phase-1 distillation via `bridge_trainer.py`

**STATUS**: BLOCKED on code-budget authorization. ~2-3 weeks of new code in `bridge_trainer.py` + `oracle_to_mcts_targets()` + dynamics integration + buffer-format change — see `tasks/TASK_780_ORACLE_AS_MCTS_DISTILLATION.md` §"Code-change scope". DO NOT launch until operator approves the code budget and the bridge_trainer.py implementation passes Gate 1 (code review) + Gate 2 (full-shape smoke).

**Owner**: workstation implements + smoke-tests + deploys; server agent executes the imprint + paired Phase-2; workstation reviews.
**Reference**: `tasks/TASK_780_ORACLE_AS_MCTS_DISTILLATION.md` (full design); `tasks/TASK_697_ORACLE_AS_MCTS_DISTILLATION.md` (original Feb-2026 design, this SPEC's parent); `lore/ship_path_recommendations_2026-05-25.md` §6 (ship-path candidate slot).

## Goal

Test whether training Phase-1 against MCTS-shaped targets — multi-pass oracle visit-count distributions, per-step policies, per-action Q-values, and oracle-derived dynamics targets — closes the representation gap between Foxglove (classification + scalar regression) and Poppy (visit distributions + backed-up Q-values + multi-step latent dynamics), shipping a paired Δα ≥ +3pp at Phase-2 Stage-B retrain.

Closest prior on the encoder-target-redesign surface: 740_bella_distill (Δα −1.92pp), 757_bella_shape (Δα −5.80pp), 758_bella_augment (Δα −16.54pp). All three closed negative; **mechanism class CLOSED** per `project_encoder_bella_class_closed.md`. TASK_780 differs structurally — it replaces target shape (planning-structure enrichment) rather than adding a regression aux loss or removing signal. UNKNOWN anti-correlation outcome.

## Hypothesis (one-line)

Phase-1 trains on flat one-hot oracle action labels + scalar oracle PnL → representation gap with Phase-2's MCTS-shaped outputs → bridge that gap by training Phase-1 on visit-count distributions, per-step policies, Q-values, and oracle-derived multi-step dynamics.

## Code prerequisites (ALL UNSHIPPED — gated on operator code-budget greenlight)

Per `tasks/TASK_780_ORACLE_AS_MCTS_DISTILLATION.md` §"Code-change scope":

1. **`oracle_to_mcts_targets()` helper** — ~1 day, location TBD (`mcts_trader/jax_agent/path_solver.py` adjacent to `multipass_oracle_policy` OR new `mcts_trader/jax_agent/oracle_to_mcts.py`). Maps un-collapsed multi-pass output (`(n_passes, T, n_assets * n_actions_per_asset)` from `path_solver.py:320`) → `{'policy_target': visit-count distribution, 'value_target': root_value, 'q_values': per-action mean return, 'step_policies': per-step visit distributions, 'step_values': per-step value targets, 'step_actions': trajectory for dynamics targets}`.
2. **New `bridge_trainer.py`** — ~3-5 days, top-level alongside `easy_trainer.py`. Uses `OnTheFlyGenerator` (from `train_easy.py:376`) + multi-pass oracle + the new helper; runs MuZero-style loss `policy * CE(logits, visit_dist) + value * hybrid_value(value_logits, root_value) + reward * CE(reward_logits, oracle_step_reward) + consistency * ‖encode(s_t) − dynamics(h_{t−1}, a_{t−1})‖²` per TASK_697 §A.2.
3. **Dynamics training inside the bridge** — ~2 days, multi-step `recurrent_inference` unroll against the oracle trajectory. Reuses the Stage-A unroll pattern at `train_phase2.py:1613-1700` (`train_stage_a`).
4. **Buffer-format change** — included in `bridge_trainer.py` budget above. From `{observations, actions [soft policy], returns [scalar remaining PnL]}` at `train_easy.py:924-935` to `{observations, actions [trajectory], rewards [per-step], visit_distributions, values [per-step root], dones}`. New `_generate_batch_episodes_bridge` method on `OnTheFlyGenerator` (do not mutate the existing path during the transition).
5. **Pass-count schedule** — ~0.5 day. Epoch-conditional `n_passes` anneal (32 → 16 → 8) per TASK_697 §A.3. Config field on the bridge config.
6. **Phase A loop + checkpoint emission** — ~1 day. Same pickle format as `easy_trainer.save_checkpoint` (`easy_trainer.py:428-447`) + sibling `config.json` so `train_phase2.load_phase1_checkpoint(:96-100)` and `get_phase2_config(:2275)` work unchanged.
7. **Phase B freeze + Phase C unfreeze pipeline** — ~1 day. Reuses `optax.multi_transform` at `train_phase2.py:919-921, 2217-2253` for freeze, plus discriminative LR ramps for unfreeze.
8. **Action-space alignment decision (operator pick)** — keep 320 (recommended; LimitSell/LimitBuy visit-counts are zero-by-construction in the multipass output per `path_solver.py:406, 473`) or drop to 192 (rewires `n_actions_per_asset = 5` → 3 in `easy_trainer.py:163`, `path_solver.py:232`, `train_easy.py:749, 831, 852, 900`). **Default in this SPEC**: 320, matches existing model surface, no architecture change required.
9. **Smoke** — ~2 days. Per Gate 2 below.

**Total**: ~2 weeks code per TASK_697 §"Implementation Estimate" line 242. Refine after Gate-1 code-review pass.

## Two-gate review protocol (per `lore/notes_2026-05-18.md` §138)

- **Gate 1 — Code review (pre-implementation, read-only sub-agent)**: read this SPEC + the parent TASK_780 + the cited code sites + the proposed `bridge_trainer.py` skeleton. Return ≤ 1 page with (a) proposed code-change locations + line counts, (b) two hard risks, (c) two computation sites that both need the update, (d) out-of-scope items, (e) refined effort estimate. Workstation approves in-step mitigations.
- **Gate 2 — Smoke discipline (post-implementation)**: full-shape smoke per the 749 OOM rule, exercising `bridge_trainer._train_step` at production `batch_size = 64`, `window_size = 64`, `n_passes_max = 32`, `unroll_steps = 5`. Verify:
  - `total_loss / policy_loss / value_loss / consistency_loss / reward_loss / grad_norm` all finite, no NaN.
  - `oracle_to_mcts_targets()` returns expected shapes against a known 10-pass oracle output.
  - Multi-step unroll reaches `unroll_steps = 5` without VRAM OOM.
  - `phase1_best.pkl` written by the bridge loads cleanly under `train_phase2.load_phase1_checkpoint` (`:96-100`). Parameter-tree structure matches `MuZeroNetworks(use_prediction_head=True, ...)`.
  - Sibling `config.json` is read cleanly by `get_phase2_config(:2275)`.
  - Smoke artifacts go to `runs/000_smoke/780_bridge/{seed_*}/`.

## Variants

### Phase A — 3-cell minimal (default)

Paired imprint, 3 seeds {7, 13, 23}:

| cell | n_passes_schedule | action_space | seed | wall (server, est.) |
|---|---|---:|---:|---:|
| 780A-default_s7  | (0:32, 50:16, 150:8) | 320 | 7  | ~165 min |
| 780A-default_s13 | (0:32, 50:16, 150:8) | 320 | 13 | ~165 min |
| 780A-default_s23 | (0:32, 50:16, 150:8) | 320 | 23 | ~165 min |

Wall: 3 × ~165 min = ~8.25 h serial, ~5 h at 2-wide on server (TASK_768 implementation log + TASK_764 actual).

### Phase B — paired 3-seed Stage-B at canon a1b1

| cell | base ckpt | seed | wall (server, est.) |
|---|---|---:|---:|
| 780B-best_s7  | 780A-default_s7  | 7  | ~2.5 h |
| 780B-best_s13 | 780A-default_s13 | 13 | ~2.5 h |
| 780B-best_s23 | 780A-default_s23 | 23 | ~2.5 h |

Wall: 3 × 2.5 h = 7.5 h serial, ~4-5 h at 2-wide.

Stage-B config matches canon a1b1 (`runs/762_reward_shaping/rs050_seed_*/stage_b_config.json` as canon reference, post-rs050 baseline). Bridge-trained dynamics are loaded as-is; Stage-A becomes either a short fine-tune or is skipped — decision at smoke time based on the Phase-A consistency-loss landing point.

### Phase C — gradual unfreeze + ExIt (OUT OF SCOPE for ship gate)

Only triggered if Phase B clears the +3pp paired gate at 3-seed. See `tasks/TASK_780_ORACLE_AS_MCTS_DISTILLATION.md` §"Phase 3 follow-on".

## Phase-A sanity gate (early-exit before Phase-B cost)

Both required to advance to Phase-B:

1. **Per-asset Bal.Acc on benchmark ≥ canon imprint − 3pp at matched epoch**. Argmax of `visit_distribution[asset, :]` per asset is the predicted action — directly comparable to current `oracle_agreement` decomposed as `agree_trade + agree_hold` at `easy_trainer.py:287`. Canon reference: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl` final-epoch balanced agreement on the segment-pool validation split. **A regression > 3pp signals the visit-distribution target is sharpening the classifier → leading anti-correlation indicator → FAIL → diagnose loss shape before paying Phase-B cost.**
2. **Dynamics consistency loss < 0.01 at Phase A end**. Matches Stage-A success criterion at `train_phase2.py:1679-1680`. If above, dynamics block did not learn multi-step composition from the oracle trajectory — Phase B Stage-A from this ckpt is no faster than from a vanilla Foxglove ckpt, the bridge's load-bearing claim fails. Operator decision: extend Phase-A OR fall back to running Phase-2 Stage-A as canon (treating Phase-A as a policy/value-only imprint).

If both gates pass on ≥ 2/3 seeds, proceed to Phase-B. Otherwise diagnose; do NOT launch Phase-B on a degraded Phase-A.

## Phase-B promotion gate (canon)

- Paired Δα ≥ +3pp vs 760 a1b1 sims=32 baselines under eval-MCTS sims=16 joint.
- No seed < −5pp.
- 5-seed extension (seeds 31, 42) only if 3-seed clears.

Eval command:

```
python tools/benchmark_easy.py \
  --checkpoint <p2_ckpt> \
  --periods-file config/benchmark_periods.json \
  --eval-mcts-sims 16 --eval-mcts-decode joint \
  --output runs/780_oracle_as_mcts/eval_p2_<cell>/benchmark.json
```

## Planned imprint command (PROVISIONAL — for the post-build pre-launch checklist)

```
python bridge_trainer.py \
  --preset medium \
  --epochs 300 \
  --batch-size 64 \
  --steps-per-epoch 200 \
  --n-envs 8 \
  --levels 0-14 \
  --oracle-multipass \
  --bridge-pass-schedule "0:32,50:16,150:8" \
  --bridge-unroll-steps 5 \
  --action-space 320 \
  --seed {7,13,23} \
  --output-dir jax_ckpt_bridge_780 \
  --name "bridge_780A_s{seed}" \
  --db-path data/trading_data.db \
  --periods-file config/benchmark_periods.json
```

The flag set is illustrative — final CLI surface is decided at Gate 1 review. The named flags above match the bridge's design: pass-count schedule, unroll-depth, action-space pick, plus the canon-equivalent arg surface from `train_easy.py` argparse `:1741-1845`.

## Pre-launch checklist (post-build)

- [ ] Gate 1 code review complete; in-step mitigations applied.
- [ ] Gate 2 full-shape smoke clean at production batch + window + unroll. Artifacts in `runs/000_smoke/780_bridge/`.
- [ ] `bridge_trainer.py` + `oracle_to_mcts_targets()` + buffer-format change merged to workstation `master`.
- [ ] Workstation-tested: bridge-trained `phase1_best.pkl` loads under `train_phase2.load_phase1_checkpoint` without raising.
- [ ] Workstation-tested: Phase-2 Stage-B can run from a bridge ckpt for 1 epoch without crash (deploy-safety gate).
- [ ] Server deploy: workstation owner runs `tools/deploy/deploy.sh --code-only`.
- [ ] Canon a1b1 paired baselines confirmed available on server: `runs/760_train_mcts_sims/a1b1_seed_{7,13,23}` (or whichever is current canon at launch time).
- [ ] Eval-MCTS sims=16 joint decoder canonical scorecard contract confirmed (see `lore/notes_2026-05-20.md` §151).
- [ ] AUTHORIZATION_2026-MM-DD.md written for this experiment, stating: code budget approved by operator on YYYY-MM-DD, Gate 1 + Gate 2 passes recorded, launch window confirmed (host = gpu-server, GPU 1+2, 2-wide).
- [ ] Server agent has the queue placement: Phase-A 3-seed → wait for completion → sanity-gate check → Phase-B 3-seed → eval.
- [ ] Pre-launch reminder to operator: **this is the largest re-imprint in the project. Subsequent paired Δα measurements are NO LONGER comparable to the 738-769 campaign data.** Accept the campaign reset as the cost of the structural pivot.

## Expected wall after build

- Code: ~2 weeks workstation (TASK_697 §"Implementation Estimate"; refine post-Gate-1).
- Phase A: ~5 h wall (2-wide on server).
- Phase B: ~4-5 h wall (2-wide on server).
- Eval + result.md write: ~1-2 h.
- **Total to first Phase-B verdict**: ~2.5 weeks from code-budget greenlight.

## Risk + anti-correlation check (summary; full version in TASK_780)

**Anti-correlation hypothesis**: UNKNOWN. The mechanism replaces the target shape (visit distributions enriched with planning structure) — different surface from 740/757 (encoder regression aux) and 738 (signal removal). But because a richer policy target *should* lower Phase-1 policy CE, there is non-zero risk of triggering the classifier-sharpening pathway that anti-correlates with Alpha on this stack (`project_classifier_alpha_anti_correlation.md`).

**Phase-A sanity gate #1 (Bal.Acc ≥ canon − 3pp) is the leading anti-correlation indicator** — if the visit-distribution target sharpens the per-asset classifier vs the current soft-one-hot target, Bal.Acc rises *during Phase-1* and the bull-blindness pathway is reactivated before Phase-B. Sanity-gate failure exits the experiment cheaply.

**Diagnostic during Phase-B Stage-B**: read `policy_entropy` and `dir_acc` per-epoch metrics. If Δ-Bal.Acc > 0 AND Δα < 0 at 3-seed mean vs 760 a1b1, the anti-correlation pathway fired through unmodeled coupling — the result is then a **closure** (visit-distribution target as a class is closed on this stack).

## References

- `tasks/TASK_780_ORACLE_AS_MCTS_DISTILLATION.md` (parent TASK design — full hypothesis, code-change-scope, verified-code-state).
- `tasks/TASK_697_ORACLE_AS_MCTS_DISTILLATION.md` (original Feb-2026 design that TASK_780 revives).
- `lore/ship_path_recommendations_2026-05-25.md` §6 — ship-path candidate slot.
- `docs/easy_mode.md` §"Phase 2: The Dream" (lines 45-63) — the dynamics warmup that Phase A replaces.
- `docs/easy_mode.md` §"Phase 3: The Thought" (lines 65-127) — the autodidact / ExIt loop that Phase C completes.
- `mcts_trader/jax_agent/path_solver.py:222-339` — `multipass_oracle_policy` (single-episode).
- `mcts_trader/jax_agent/path_solver.py:342-392` — `batched_multipass_oracle_policy` (vmapped multi-episode).
- `easy_trainer.py:46-65, 131-302, 317-333, 428-447` — current Phase-1 trainer to replace.
- `train_easy.py:376, 659-945, 1740, 1847, 2117, 2202-2256` — current Phase-1 entry & generator path.
- `train_phase2.py:96-100, 2261-2267, 2650` — Phase-2 load contract & missing-aux-head init.
- `train_phase2.py:277-359, 1613-1700` — current Stage-A dynamics warmup; the loop Phase A replaces.
- `tasks/TASK_768_BASE_MODEL_REFRESH.md` §"Implementation log" — 165-min server imprint baseline + Option A bit-identical landing.
- `tasks/TASK_775_ORACLE_MASK_PHASE1_REIMPRINT.md` — companion Phase-1 imprint design (oracle-mask head); together with TASK_780 forms the Phase 3 prerequisite.
- `lore/notes_2026-05-18.md` §138 — two-gate review protocol.
- `lore/handoff_2026-05-24_764-launch.md` — 165-min server imprint actual.
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` — campaign-class-closure verdict that freed this candidate's budget.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — anti-correlation finding.
- `~/.claude/projects/-home-lukas-Trader/memory/project_encoder_bella_class_closed.md` — closest prior on encoder-target redesign (740/757/758 closed).

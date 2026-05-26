# Ship-Path Ceremony — Validation Pass (2026-05-26)

Validator: parallel agent invoked after the 10-candidate ship-path ceremony documented in `lore/notes_2026-05-25.md` §169 and `lore/ship_path_recommendations_2026-05-25.md`. Mandate: READ-ONLY check of TASK + SPEC + code-worktree deliverables for accuracy, completeness, deploy-contract compliance, and inter-task consistency.

Workstation tip checked: `79430e3` (master). Cross-checks performed against current source code, not the brief's cited line numbers (the brief itself drifts in places — see per-task findings).

## Summary table

| Task | Verdict | Confidence | Key issues |
|---|---|---|---|
| TASK_775 | READY | HIGH | Two minor line-citation drifts (`:2217-2226` actually `:2258-2261`; `:2650` actually `:2698`). Mechanism understanding intact. |
| TASK_776 | READY-TO-MERGE | HIGH | Diff is clean, canon bit-identical when `entropy_coef=0`. Pre-pre-merge bit-identity smoke on GPU still recommended (operator policy). |
| TASK_777 | READY-TO-MERGE w/ caveat | MID | Diff is clean. Master ALREADY has a stale copy of `tasks/TASK_777_*.md` (81-line diff vs worktree) and `experiments/777_mixture_value/SPEC.md`. Worktree branch is the source of truth; the master untracked files should be deleted before merge to avoid a confusing "already there" condition. |
| TASK_778 | READY (design only) | HIGH | Status correctly marked BLOCKED on code-budget. No code committed. Cited line ranges verified accurate (dynamics MLP at `:380-388` + `:583-610`; factored search expansion `:80-114`; mctx recurrent_fn `:89-140`). |
| TASK_779 | READY | HIGH | Self-corrected line citations. Two residual citation drifts (the MCTS expansion mask claim `:1742-1762` is actually `:1783-1795`; CLI override claim `:2348-2349, 2355-2356` is actually `:2361-2362, 2373-2374`). Mechanism + flags verified. |
| TASK_780 | READY (design only) | HIGH | Status correctly marked BLOCKED on code-budget. Path solver citations `:222`, `:345`, `:380` verified accurate. ~2-3 week budget acknowledged. |
| TASK_781 | READY-TO-MERGE | HIGH | Diff is clean. `bc_return_weighted` flag-gated default OFF; canon paths in both BC arms are literal pre-781 expressions. |
| TASK_782 | READY w/ patch caveat | HIGH | Important canon-correction discovered ('canon a1b1 already has encoder_lr=3e-5'). Schedule A is byte-equivalent to canon; Schedule B needs 4-line CLI patch OR JSON-override path. Sub-agent correctly declined to commit the patch. |
| TASK_783 | READY-with-dependency | HIGH | Explicit dependency on TASK_776 merge documented. SPEC will be no-op on canon (factored search) without 776's factored-path entropy term. Dependency note is unambiguous. |
| TASK_784 | READY-TO-MERGE | HIGH | Diff is clean, λ=0 branch-skipped path is bit-identical. One harmless comment misstatement (`(T-2,)` shape claim — actual array length matches `oracle_cum_pnl` length after concat). |

**Ship-ready (training can launch tomorrow)**: TASK_775, TASK_779, TASK_782 (Schedule A) — zero code required.
**Ship-ready after worktree merge**: TASK_776, TASK_777, TASK_781, TASK_784.
**Ship-ready after worktree merge + dependency**: TASK_783 (depends on TASK_776).
**Design-only, no launch yet**: TASK_778, TASK_780 (both correctly marked BLOCKED on code budget).

---

## Per-task findings

### TASK_775 — Oracle-mask aux head (Phase-1 re-imprint)

- **TASK file** (`/home/lukas/Trader/tasks/TASK_775_ORACLE_MASK_PHASE1_REIMPRINT.md`): PASS.
  - Status, hypothesis, mechanism, verified-code-state, code-change-scope (NONE for default 3-cell), variants, sanity gate, promotion gate, anti-correlation check, cost estimate, references — all present.
  - Promotion gate correctly states paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint, no seed < −5pp (lines 6, 16, 116-118).
  - Anti-correlation discussion (lines 134-146) is rigorous and distinguishes per-asset binary selection-set from per-action softmax (the surface implicated in the campaign anti-correlation finding).
- **Citation spot-checks** (against `79430e3`):
  - `muzero_model.py:286` use_oracle_mask_head: **VERIFIED** at line 286.
  - `muzero_model.py:417-421` setup branch: **VERIFIED** at 417-421.
  - `muzero_model.py:564-568` `oracle_mask_prediction`: **VERIFIED** at 564-568.
  - `easy_trainer.py:60-61` constructor forward: **VERIFIED** at 60-61.
  - `easy_trainer.py:215-225` loss term: **VERIFIED** at 215-225.
  - `easy_config.py:86-88`: **VERIFIED**.
  - `train_easy.py:1885-1886` CLI: **VERIFIED**.
  - `train_phase2.py:392-393`: **VERIFIED** (factory forward).
  - `train_phase2.py:919-920` Stage-B freeze: **VERIFIED** (`if 'oracle_mask' in key: return 'freeze'`).
  - **MINOR DRIFT**: TASK file cites `train_phase2.py:2217-2226` for Stage-A optimizer label_fn; actual location is `:2258-2261` (the cited range is the dynamic encoder-unfreeze handler at `:2213-2220`). Cited mechanism is correct, line range is off by ~40.
  - **MINOR DRIFT**: TASK file cites `train_phase2.py:2650` for `initialize_missing_aux_heads(...)` call; actual call site is `:2698`. The function definition is at `:167`. Off by ~48.
  - `jax_ckpt_easy_phase1_xs32/easy_medium/config.json:44`: **VERIFIED** — `"use_oracle_mask_head": false`.
- **SPEC** (`/home/lukas/Trader/experiments/775_oracle_mask/SPEC.md`): PASS.
  - Default 3-cell minimal + extended 9-cell variants documented.
  - Launch command (line 49-62) uses `--preset small --bella --oracle-mask-head --seed 7 --output-dir ... --encoder-version 3 --embed-dim 64 --encoder-heads 8 --epochs 200`. Reasonable.
  - GPU placement: GPU 1+2 train, GPU 3 eval. **GPU 0 explicitly forbidden** (line 89).
  - Phase-1 sanity gate (lines 95-110): oracle_mask_accuracy > 60% AND bull-regime Bal.Acc Δ ≥ −3pp on ≥ 2/3 seeds.
  - Eval protocol (lines 134-142) uses `--eval-mcts-sims 16 --eval-mcts-decode joint`. CORRECT canon scorecard.
- **Code-state drift**: 2 minor line-citation drifts (above). Not load-bearing for the experiment (mechanism is intact).
- **Verdict**: READY.
- **Recommended action**: launch.

### TASK_776 — rs050 + entropy combination

- **Worktree**: `/home/lukas/Trader/.claude/worktrees/agent-a55e2ae10d6a66896/`, branch `worktree-agent-a55e2ae10d6a66896`, commit `1b4fcdf`.
- **TASK file** (`tasks/TASK_776_RS050_ENTROPY_COMBO.md`, worktree-resident): PASS.
- **SPEC** (`experiments/776_rs050_entropy/SPEC.md`, worktree-resident): PASS.
- **Code diff** (`train_phase2.py` + `phase2_config.py`):
  - `phase2_config.py:47-58` — docstring update reflecting factored-path entropy support + explicit operator note about scaling `min_entropy` per path.
  - `train_phase2.py:1235-1271` — entropy block restructured. Outer gate `if entropy_coef > 0.0:` controls all new behavior. Inside, factored vs joint paths each compute their entropy. Canon (`entropy_coef == 0.0`) falls through to the `else` branch that assigns `policy_entropy = jnp.array(0.0)` — **bit-identical to pre-776**.
  - Variables `n_assets` and `n_actions_per_asset = 5` are in scope at line 1038-1039 (captured at JIT-trace time). The factored-path entropy uses `pol_r = policy_logits.reshape(-1, n_assets, n_actions_per_asset)` — correctly typed.
  - **No new imports**. Pure JAX (`jax.nn.softmax`, `jnp.sum`, `jnp.log`).
  - **No fixture/dependency surprises**.
- **Flag-gated default OFF**: verified. `entropy_coef: float = 0.0` is unchanged (line 52 of `phase2_config.py`).
- **Canon bit-identical when flag OFF**: verified by source read of the outer `if entropy_coef > 0.0:` guard.
- **Inter-task consistency**: TASK_783 explicitly depends on TASK_776 (see §TASK_783 below). The factored-path extension is exactly what 783's sweep needs to be non-trivial on canon.
- **Verdict**: READY-TO-MERGE.
- **Recommended action**: merge after a GPU bit-identity smoke (run 5 epochs with `entropy_coef=0.0`, diff `value_loss/policy_loss/grad_norm` against canon 760 baseline — same protocol as 769/777). Worktree static-smoke (random logits → sane H ≈ 1.31 / 5.31) is documented.

### TASK_777 — Mixture value target

- **Worktree**: `/home/lukas/Trader/.claude/worktrees/agent-ad35b376f93d08f62/`, branch `worktree-agent-ad35b376f93d08f62`, commit `8972cf7`.
- **TASK file** (`tasks/TASK_777_MIXTURE_VALUE_TARGET.md`, worktree-resident): PASS. Comprehensive: hypothesis, mechanism, anti-correlation rationale (LOW risk, same class as 746/734), code-change scope (~50 LOC), three bit-identity smoke gates (α=1.0 → oracle bit-identical, α=0.0 → nstep bit-identical, plus oracle-branch untouched), promotion gate, mechanism diagnostic, out-of-scope items.
- **SPEC** (`experiments/777_mixture_value/SPEC.md`, worktree-resident): PASS.
- **Code diff** (5 files changed, 554 insertions):
  - `phase2_config.py:206-227` — extends `value_target_mode` docstring; adds `value_mixture_alpha: float = 0.5`.
  - `phase2_config.py:310-326` — `__post_init__` validators for both new fields.
  - `train_phase2_cli.py` — extends `--value-target-mode choices=`; adds `--value-mixture-alpha`, `--oracle-horizon`.
  - `train_phase2.py:1051-1055` — closure-time capture of `value_mixture_alpha` (Python float, baked into HLO).
  - `train_phase2.py:1289-1332` — new `elif value_target_mode == "mixture":` block placed between `nstep` and `td0` branches. Oracle leg mirrors the dedicated `oracle` branch's logic; nstep leg mirrors the dedicated `nstep` branch. `td_target_raw = α * oracle_scaled + (1−α) * nstep_target_raw`. Both then PopArt-normalized + clipped identically.
  - `train_phase2.py:2609-2627` — arg translation with operator-friendly guard: setting `--value-mixture-alpha` without `--value-target-mode mixture` raises `ValueError`.
- **Bit-identity argument** (TASK file §"Smoke gate 2"): rigorous. `td_target_raw = 1.0 * oracle_scaled + 0.0 * nstep_target_raw = oracle_scaled` exactly by IEEE 754. Verified by reading the diff.
- **No new imports**. No new dependencies.
- **Inter-task inconsistency / DUPLICATION**:
  - Master tip ALREADY has `/home/lukas/Trader/tasks/TASK_777_MIXTURE_VALUE_TARGET.md` and `/home/lukas/Trader/experiments/777_mixture_value/SPEC.md` as untracked files. They differ from the worktree versions by 81 lines (TASK) and a few lines (SPEC). The master untracked copies must NOT be the basis for merge; the worktree copy is the source of truth (matches the committed code).
  - This is likely a sub-agent dispatch glitch where the same TASK was begun on master then duplicated into a worktree for the actual code change. Operator should `git rm` the master copies (or stage them as removed) before merging the worktree branch, or use `git merge worktree-agent-ad35b376f93d08f62` which will overwrite.
- **Flag-gated default OFF**: verified. `value_target_mode: str = "oracle"` (canon default) at `phase2_config.py:212`; `value_mixture_alpha: float = 0.5` only consumed when mode is `"mixture"`.
- **Verdict**: READY-TO-MERGE w/ caveat (delete master untracked copies of TASK + SPEC first, or merge with `-X theirs`).
- **Recommended action**: merge after GPU bit-identity smoke per worktree's own §"Smoke gate" plan (also same 769 protocol).

### TASK_778 — Bella-as-rollouts (design-only)

- **TASK file** (`/home/lukas/Trader/tasks/TASK_778_BELLA_AS_ROLLOUTS.md`): PASS.
  - Status: DRAFT, design only, BLOCKED on code-budget authorization (lines 3, 7).
  - Hypothesis (§1), why-reopens-CLOSED-Bella-class (§2 — explicit world-model carve-out cited verbatim from `project_encoder_bella_class_closed.md`), verified-code-state (§3), code-change scope (§4, ~630 LOC estimated), two-gate review protocol (§5), variants (§6), promotion gate + mechanism diagnostic (§7), cost (§8), risks + anti-correlation (§9), references (§10).
  - **Codecorrection by sub-agent**: §3.4 explicitly flags that the brief's citation of `muzero_model.py:415-420` for "dynamics_dense1/2" was WRONG — those lines hold the TASK_695 oracle_mask aux head. Sub-agent corrected: dynamics MLP definition is at `:380-388`, forward at `:583-610`, recurrent_inference at `:701-728`. This is a valuable self-correction.
  - Promotion gate: paired Δα ≥ +3pp vs 760 a1b1 sims=32, eval-MCTS sims=16 joint (lines 6, 228).
- **Citation spot-checks**:
  - `muzero_model.py:380-388` dynamics_dense1/2 definition: **VERIFIED** at 380-388.
  - `muzero_model.py:583-610` dynamics forward: **VERIFIED** at 583-610.
  - `factored_search.py:80-114` model_apply call with mode='recurrent_inference': **VERIFIED** at 86-91 inside 80-114.
  - `mctx_search.py:89-140` `_recurrent_fn`: **VERIFIED** at 89-140 (109-114 is the model.apply call).
- **SPEC** (`/home/lukas/Trader/experiments/778_bella_rollouts/SPEC.md`): PASS.
  - STATUS clearly marked BLOCKED on code-budget authorization (line 3).
  - Planned launch command is "do NOT run before gate 2 clears" (line 31).
  - GPU 0 explicitly forbidden (line 69).
  - Pre-launch checklist comprehensive (lines 107-119): Bella ckpt deployed, dynamics-MLP-consistency baseline measured, qa-hunt + qa-crossval reviews, bit-identity smoke, flag-ON smoke, VRAM headroom, 762 W2 / 763 / 764 status snapshot, prior small-probe verdict before committing to this large build.
  - Cancellation triggers (lines 123-128): any of TASK_775/776/777/779 returns +3pp at 3-seed AND clears 5-seed → freeze TASK_778 build, ship the survivor.
- **Verdict**: READY (design only — no code to verify).
- **Recommended action**: wait (per the ship-path memo §"Recommended execution order", run cheap probes first). Authorize code budget only after TASK_775/776/777/779 close.

### TASK_779 — rs050 + 767 combination

- **TASK file** (`/home/lukas/Trader/tasks/TASK_779_RS050_767_COMBO.md`): PASS.
  - Hypothesis, mechanism, verified-code-state, code-change-required=NONE, variants, promotion gate (three-stage early-exit), cost (~5h wall, ~8-9 GPU-h — "lowest cost candidate on the ship-path list"), anti-correlation check (LOW), references.
  - **Excellent**: explicit "Code-state divergences from brief" section (§lines 134-150) listing every brief citation that drifted (TASK_771 batch / TASK_772.1 CLI extraction caused line shifts). Sub-agent re-verified each load-bearing wire.
- **Citation spot-checks** (against `79430e3`):
  - `phase2_config.py:30` reward_scale 100.0: **VERIFIED**.
  - `phase2_config.py:45` disable_limit_orders False: **VERIFIED**.
  - `train_phase2.py:637, 679` rs050 multiplications: **VERIFIED** at 637 (`rewards = np.clip(step_pnl * self.config.reward_scale, ...)`) and 679 (`per_step_r = per_step_pnl * self.config.reward_scale`).
  - `train_phase2.py:1060-1090` 767 mask build: **VERIFIED** — disable_limit_orders capture at 1080, masks built 1081-1090.
  - `train_phase2_cli.py:71-73` `--reward-scale`: **VERIFIED** at line 71.
  - `train_phase2_cli.py:89-98` `--disable-limit-orders`: **VERIFIED** at line 89 (`BooleanOptionalAction`).
  - **RESIDUAL DRIFT (not flagged in task)**:
    - TASK cites `train_phase2.py:1742-1762` as the train-time MCTS expansion mask. Actual: the MCTS expansion mask (which feeds `invalid_actions` to `MuZeroMCTXSearch`) is at `:1783-1795`. Lines 1742-1762 are the value-head reset block. Brief cites `:1742-1754` and TASK_779 self-correction says `:1742-1762`, but both are wrong — the actual location is `:1783-1795`. Off by ~40.
    - TASK cites `train_phase2.py:2348-2349, 2355-2356` for CLI override plumb. Actual: `:2361-2362, 2373-2374`. Off by ~13.
- **SPEC** (`/home/lukas/Trader/experiments/779_rs050_767/SPEC.md`): PASS.
  - Launch command with full flag set (lines 41-69), flag-name verification table (lines 73-105) cross-checked against `train_phase2_cli.py`.
  - GPU 0 forbidden (line 111); GPU 1/2 train 2-wide, GPU 3 eval.
  - 3-gate promotion (lines 169-177): anti-correlation early-exit, mechanism-interaction (combo > max(rs050-only, 767-only)), standard +3pp canon gate.
- **Verdict**: READY.
- **Recommended action**: launch — recommended first on the ship-path list per memo §"Recommended execution order" (lowest cost, both legs production-deployed).

### TASK_780 — Oracle-as-MCTS distillation (design-only)

- **TASK file** (`/home/lukas/Trader/tasks/TASK_780_ORACLE_AS_MCTS_DISTILLATION.md`): PASS.
  - Status: DRAFT, revival of TASK_697 (parked Feb 2026). Operator authorization required for ~2-3 week code budget (line 8).
  - Hypothesis, verified-code-state (multi-pass oracle infrastructure, Phase-1 imprint loop, Phase-2 expected interface), code-change scope table (~2 weeks per TASK_697 estimate), action-space alignment decision (defaults to 320, operator-confirmable), two-gate review protocol, variants (Phase A 3-cell + Phase B paired 3-seed + Phase C out-of-scope), Phase-A sanity gate (Bal.Acc ≥ canon − 3pp AND consistency loss < 0.01), Phase-B canon promotion gate, anti-correlation check (UNKNOWN by direct precedent — closest priors are 740/757/738), cost estimate (~2.5-3 weeks to first measurable result).
- **Citation spot-checks**:
  - `path_solver.py:222` multipass_oracle_policy: **VERIFIED** at 222.
  - `path_solver.py:345` batched_multipass_oracle_policy: **VERIFIED** at 345.
  - `path_solver.py:380` lambda calling multipass_oracle_policy: **VERIFIED** at 380.
- **SPEC** (`/home/lukas/Trader/experiments/780_oracle_as_mcts/SPEC.md`): PASS.
  - STATUS: BLOCKED on code-budget authorization (line 3).
  - Code prerequisites listed (lines 18-32), two-gate review protocol (lines 34-43), variants matrix, sanity gate, promotion gate, planned imprint command (provisional), pre-launch checklist (lines 123-135), expected wall.
  - Important warning (line 135): **"this is the largest re-imprint in the project. Subsequent paired Δα measurements are NO LONGER comparable to the 738-769 campaign data. Accept the campaign reset as the cost of the structural pivot."** This is the right warning to surface.
- **Verdict**: READY (design only — no code to verify).
- **Recommended action**: wait. Authorize the ~2-3 week code budget only if TASK_775/776/777/779 + TASK_781/782/783/784 all close null.

### TASK_781 — Return-weighted BC + 767 + rs050 stack

- **Worktree**: `/home/lukas/Trader/.claude/worktrees/agent-a0379d27c0a8ee2c8/`, branch `worktree-agent-a0379d27c0a8ee2c8`, commit `9f2fd61`.
- **TASK file** (`tasks/TASK_781_RETURN_WEIGHTED_BC_STACK.md`, worktree-resident): PASS.
- **SPEC** (`experiments/781_return_weighted_stack/SPEC.md`, worktree-resident): PASS.
- **Code diff** (5 files, 316 insertions):
  - `phase2_config.py:73-81` — replaces deferred-stub comment for `bc_return_weighted` with load-bearing comment + documents default False.
  - `train_phase2_cli.py:71-76` — `--bc-return-weighted` `store_true` flag with help text.
  - `train_phase2.py:1056-1059` — closure-time `use_bc_rwt = bool(getattr(config, 'bc_return_weighted', False))`.
  - `train_phase2.py:1351-1384` — return-weighted branch inside BOTH `use_bc_balance` arm AND the else arm. Each path's canon expression is literally the pre-781 line (verified by reading the diff). When `use_bc_rwt=False`, dead branches stripped under JIT trace.
  - `train_phase2.py:2354-2359` — argparse override.
- **Return signal**: `oracle_values` (oracle remaining PnL over `oracle_value_horizon=400` steps, already in batch dict). Mean-normalization (`|oracle_values| / mean(|oracle_values|)`) keeps total gradient magnitude comparable to canon.
- **No new imports**.
- **Flag-gated default OFF**: verified. `bc_return_weighted: bool = False` (line 76 of `phase2_config.py`).
- **Canon bit-identical when flag OFF**: verified by reading diff — both BC arms have literal pre-781 expressions in their `else` clauses.
- **SPEC dependency on TASK_779**: the recommended Wave-1 paired baseline is TASK_779 result (rs050+767) when available, with fallback to 760 a1b1. This is sensible.
- **Inter-task consistency with TASK_779**: TASK_781 uses both `--reward-scale 50` (rs050) and `--disable-limit-orders` (767) plus the new `--bc-return-weighted`. The two underlying flags do NOT conflict — they touch independent paths (reward signal vs action mask vs BC weighting). Verified no overlapping mutation.
- **Verdict**: READY-TO-MERGE.
- **Recommended action**: merge after GPU smoke; launch can either replace TASK_779 in the ship-queue (TASK_781 is a 3-stack including 779's two flags + adds rwbc) OR follow TASK_779 to isolate the rwbc contribution.

### TASK_782 — Encoder unfreeze schedule

- **TASK file** (`/home/lukas/Trader/tasks/TASK_782_ENCODER_UNFREEZE.md`): PASS.
  - **Critical discovery surfaced by sub-agent** (lines 14, 30-31): canon a1b1 in `runs/760_train_mcts_sims/baseline_paired_seed_7/stage_b_config.json` already shows `"encoder_lr": 3e-05, "encoder_unfreeze_epoch": null`, and the training log line `Encoder UNFROZEN at lr=3e-05 (TASK_728)` confirms encoder is unfrozen from epoch 0 in 760 canon. **The dataclass default `encoder_lr=0.0` is the config-file default, not the runtime canon.** Schedule A (`encoder_lr=3e-5` from epoch 0) is therefore byte-identical to canon — the null-control. Schedule B (delayed unfreeze at epoch 100) is the actual experimental lever.
  - This is an important correction to the ship-path memo's framing (which assumed "encoder unfrozen" was the variant). The TASK file fixes it inline.
  - Status, hypothesis, verified code state, code-change-required (Schedule A: NONE; Schedule B: 4-line optional CLI patch), variants (6 cells), anti-correlation diagnostic (encoder gradient norm logging with hard/soft thresholds), promotion gate, risk + anti-correlation check (MEDIUM, with surface-distinction argument).
- **Citation spot-checks**:
  - `phase2_config.py:247-248`: **VERIFIED** — `encoder_lr: float = 0.0` and `encoder_unfreeze_epoch: Optional[int] = None`.
  - `train_phase2.py:957-972`: **VERIFIED** — encoder unfreeze branch in `create_stage_b_optimizer`.
  - `train_phase2.py:2213-2220`: **VERIFIED** — dynamic encoder-unfreeze handler inside the epoch loop.
  - `train_phase2.py:2538-2539`: **VERIFIED** — `--encoder-lr` CLI override.
  - `train_phase2_cli.py:120-121`: **VERIFIED** — argparse line for `--encoder-lr`.
  - `runs/760_train_mcts_sims/baseline_paired_seed_7/stage_b_config.json:142` shows `"encoder_lr": 3e-05` — **VERIFIED** the canon-correction.
- **SPEC** (`/home/lukas/Trader/experiments/782_encoder_unfreeze/SPEC.md`): PASS.
  - 6 cells (Schedule A × 3 + Schedule B × 3), Wall ~5 h at 2-wide.
  - GPU 0 forbidden; GPU 1+2 train, GPU 3 eval.
  - Anti-correlation diagnostic protocol: encoder grad norm logging, hard abort at 5×canon between protected epoch range.
  - Early-exit protocol (4 conditions, lines 141-146): sanity-control failure, hard grad-norm abort, Schedule B catastrophic Alpha collapse on first seed, both schedules >2× canon trade rate.
- **Code-change-required-but-deferred**: Schedule B requires `--encoder-unfreeze-epoch` CLI flag (4 LOC patch in `train_phase2_cli.py:120-121` + `train_phase2.py:2538-2539`). The TASK file correctly declares this out-of-scope for a sub-agent per the deploy contract ("DO NOT edit existing source files" was the rule; sub-agent declined). Operator decides: land the 4-line patch as TASK_782 pre-flight, OR use a JSON config-override via `get_phase2_config(phase1_config_path=...)`.
- **Verdict**: READY w/ patch caveat.
- **Recommended action**: operator lands the 4-line CLI patch (cheap, well-scoped) and launches Schedule A+B together. Schedule A's `Δα ≈ 0 ± seed noise vs 760` is the harness-reproducibility null; Schedule B is the actual test.

### TASK_783 — Class-balance + entropy joint sweep

- **TASK file** (`/home/lukas/Trader/tasks/TASK_783_BC_BALANCED_ENTROPY_JOINT.md`): PASS.
  - **Critical dependency surfaced**: TASK_783 is explicitly marked BLOCKED on TASK_776 merge (line 3). At master `79430e3`, the entropy term at `train_phase2.py:1237` is **gated to the joint path only** (`if entropy_coef > 0.0 and not use_factored:`); canon training uses `use_factored_search=True`. Without TASK_776's factored-path extension, 783's sweep is a silent no-op on canon.
  - The TASK file documents the dependency at multiple levels:
    - Header line 3 — BLOCKED status.
    - §"Dependency on TASK_776 — explicit" (lines 61-76) — Read order specified.
    - §"min_entropy units" — sweep value of 3.0 (joint-path) vs 0.7 (factored-path) depends on which path TASK_776 lands.
- **Citation spot-checks**:
  - `phase2_config.py:74-75`: bc_class_balanced=True default, bc_max_trade_weight=64.0 default: **VERIFIED** (file line 74-75).
  - `phase2_config.py:52-53`: entropy_coef, min_entropy defaults: **VERIFIED**.
  - `train_phase2.py:1053-1054`: BC capture site: **VERIFIED**.
  - `train_phase2.py:1237`: joint-only entropy gate: **VERIFIED** (`if entropy_coef > 0.0 and not use_factored:`).
  - `train_phase2.py:1217-1230`: BC visit-distribution weighting: **VERIFIED**.
  - `train_phase2.py:1338-1349`: BC oracle weighting: **VERIFIED**.
  - `runs/762_reward_shaping/rs050_seed_7/stage_b_config.json:147`: `use_factored_search: true`: **VERIFIED**.
- **SPEC**: `experiments/783_bc_entropy_joint/SPEC.md` exists (master tip).
- **Inter-task dependency on TASK_776**: documented correctly — both legs must use the post-776 factored-path entropy implementation. Without 776, every 783 cell with `entropy_coef > 0` produces an identical metric profile to canon (silent no-op).
- **Verdict**: READY-with-dependency.
- **Recommended action**: wait until TASK_776 merges. Then immediately launch.

### TASK_784 — Smoothed-oracle value target

- **Worktree**: `/home/lukas/Trader/.claude/worktrees/agent-ad1e54ad52f9aeb0f/`, branch `worktree-agent-ad1e54ad52f9aeb0f`, commit `4e72167`.
- **TASK file** (`tasks/TASK_784_SMOOTHED_ORACLE_VALUE_TARGET.md`, worktree-resident): PASS.
- **SPEC** (`experiments/784_smoothed_oracle/SPEC.md`, worktree-resident): PASS.
- **Code diff** (5 files, 506 insertions):
  - `phase2_config.py:206-228` — new `oracle_smoothing: float = 0.0` field with comprehensive comment explaining the EWMA recursion + hypothesis.
  - `train_phase2.py:640-668` — EWMA branch inserted in `AsyncBatchPipeline._build_batch`:
    - Reads `oracle_smoothing = getattr(self.config, 'oracle_smoothing', 0.0)`.
    - When `> 0.0`: computes `deltas = np.diff(oracle_cum_pnl)`, EWMA-smooths, reconstructs `cum_pnl_eff` anchored at `oracle_cum_pnl[0]`.
    - When `== 0.0`: `cum_pnl_eff = oracle_cum_pnl` (the literal pre-784 reference). Bit-identical.
    - **MINOR**: comment at line 645 says `deltas = np.diff(oracle_cum_pnl) # (T-2,) — first-difference`. `np.diff` returns length T−1 (not T−2). The comment is wrong but the code is correct — `np.concatenate([oracle_cum_pnl[:1], oracle_cum_pnl[0] + np.cumsum(smoothed)])` produces length 1 + (T−1) = T, matching `oracle_cum_pnl`. Cosmetic only.
  - `train_phase2_cli.py:257-265` — `--oracle-smoothing` argparse flag with bounded help.
  - `train_phase2.py:2588-2598` — argparse override with explicit validation: `0.0 ≤ λ < 1.0` (`λ=1.0` is degenerate). Raises `ValueError` on out-of-range.
- **No new imports**. Pure numpy.
- **Flag-gated default OFF**: verified. `oracle_smoothing: float = 0.0` and `if oracle_smoothing > 0.0:` gate.
- **Canon bit-identical when flag OFF**: verified by reading the diff. `cum_pnl_eff = oracle_cum_pnl` (the literal pre-784 reference) when λ=0.
- **Verdict**: READY-TO-MERGE.
- **Recommended action**: merge after GPU smoke (3-λ sweep × 3 seeds, paired vs 760 a1b1 AND vs 746 oracle). Note ship-path memo flags this as "Low EV" — likely null but cheap.

---

## Cross-task issues

### Inter-task dependency: TASK_783 → TASK_776

Documented correctly in TASK_783 lines 3, 61-76. Without TASK_776 merge, TASK_783's canon-path entropy sweep is a silent no-op. **Action**: merge TASK_776 first.

### Inter-task overlap: TASK_779 (rs050+767) ⊂ TASK_781 (rs050+767+rwbc)

TASK_781 explicitly uses the same `--reward-scale 50 --disable-limit-orders` flags as TASK_779 plus its new `--bc-return-weighted`. The two are not in conflict; they are 1-leg vs 3-leg variants of the same stack. Operator picks order:
- Run TASK_779 first → if it ships +3pp, no need for TASK_781.
- Run TASK_781 first → if it ships +3pp, optionally also run TASK_779 to isolate the rwbc contribution.
TASK_781 SPEC documents this baseline-selection logic (lines 49-53 of SPEC).

### Inter-task overlap: TASK_777 master-vs-worktree duplication

Master tip has untracked copies of `tasks/TASK_777_*.md` and `experiments/777_mixture_value/SPEC.md` that DIFFER from the worktree versions (81 lines of TASK diff, several lines of SPEC diff). This is a sub-agent dispatch collision. Worktree commit `8972cf7` is the source of truth (matches the committed code). Operator should delete the master untracked copies before merging the worktree.

### Inter-task issue: 776 vs 783 path-aware min_entropy semantics

TASK_776 implements per-asset entropy with max ≈ ln(5) = 1.61. TASK_783 sweeps `min_entropy = 3.0` (joint-path units). After 776 merges, every TASK_783 cell MUST use `min_entropy ≈ 0.7` (factored-path units) for `min_entropy 3.0` would be above the path's maximum and the floor penalty would fire on every batch. TASK_783 documents this at lines 72-74. Documentation is clear; operator must remember to scale the unit when copying 783's spec.

### No "deploy" / "commit-to-master" instructions left in artifacts

Spot-checked every TASK + SPEC file: none contain `deploy.sh` calls, commit-to-master instructions, or push directives. The sub-agent contract was honored across all 10 deliverables. TASK_777 and TASK_781 implementation logs explicitly state "Deploy contract: Sub-agent worktree-only. NO `deploy.sh` call. NO push. Commit in worktree only."

### No new external dependencies introduced by the 4 code worktrees

Verified by reading every code diff:
- TASK_776: uses `jax.nn.softmax`, `jnp.sum`, `jnp.log`. All already imported in `train_phase2.py`.
- TASK_777: uses existing `jax.nn.softmax`, `jnp.sum`, `get_value_support()`, `jnp.clip`. All already in scope.
- TASK_781: uses `jnp.abs`, `jnp.mean`, `jnp.maximum`. All already in scope.
- TASK_784: uses `np.diff`, `np.empty_like`, `np.cumsum`, `np.concatenate`. Plain numpy at batch-build time, no JIT crossing. All already in scope.

---

## Worktree-merge readiness (the 4 code tasks)

| Task | Branch | Commit | Files changed | Insertions | Same-line conflicts with master? |
|---|---|---|---|---|---|
| TASK_776 | `worktree-agent-a55e2ae10d6a66896` | `1b4fcdf` | `train_phase2.py`, `phase2_config.py`, plus tasks/ + experiments/ | code: ~30 LOC additions, 0 deletions in `train_phase2.py`; ~7 LOC docstring update in `phase2_config.py` | LOW. Touches `train_phase2.py:1232-1268` and `phase2_config.py:47-58`. No other worktree touches these ranges. |
| TASK_777 | `worktree-agent-ad35b376f93d08f62` | `8972cf7` | 5 files, 554 insertions | `train_phase2.py` +62, `train_phase2_cli.py` +19, `phase2_config.py` +34 | LOW with master (touches `train_phase2.py:1051-1055` + `:1289-1332` + `:2609-2627`). **MEDIUM with TASK_781**: TASK_781 touches `:1059` (closure capture, adjacent to TASK_777's `:1055`) and `:2356-2357` (override, adjacent to TASK_777's `:2620-2627`). Different lines but same blocks — interleaving merge should be clean, but a 3-way merge is recommended. |
| TASK_781 | `worktree-agent-a0379d27c0a8ee2c8` | `9f2fd61` | 5 files, 316 insertions | `train_phase2.py` +43, `train_phase2_cli.py` +6, `phase2_config.py` +8 | LOW with master. **MEDIUM with TASK_777**: see above. |
| TASK_784 | `worktree-agent-ad1e54ad52f9aeb0f` | `4e72167` | 5 files, 506 insertions | `train_phase2.py` +38, `train_phase2_cli.py` +9, `phase2_config.py` +17 | LOW. Touches `train_phase2.py:640-668` (batch-build) and `:2588-2598` (CLI override). No other worktree touches these. |

**Recommended merge order**:
1. TASK_776 (unblocks TASK_783).
2. TASK_784 (independent, lowest-risk).
3. TASK_777 (independent from 776/784; standalone code).
4. TASK_781 (touches lines adjacent to TASK_777; merge with a manual diff review on `train_phase2.py:1055-1059` and `:2350-2360`).

Each merge should be followed by a 5-epoch canon smoke (run with all flags OFF, diff key metrics against pre-merge 760 trajectory; max abs diff ≤ 1e-6 per the 769 bit-identity protocol).

---

## Overall: which tasks are ship-ready vs which need a follow-up before launch

**Immediate launch (no merge required)**:
- TASK_775 (Oracle-mask Phase-1 re-imprint, default 3-cell, ~8h wall server).
- TASK_779 (rs050 + 767 combo, ~5h wall server — the lowest-cost ship-path probe).
- TASK_782 (encoder unfreeze, Schedule A is byte-equivalent to canon; Schedule B needs a 4-LOC patch or a JSON-override path — operator picks).

**Launch after merge of corresponding worktree**:
- TASK_776 (after `1b4fcdf` lands on master + GPU bit-identity smoke).
- TASK_777 (after `8972cf7` lands on master with master-side stale-duplicate cleanup + GPU bit-identity smoke).
- TASK_781 (after `9f2fd61` lands; can stack with TASK_779 if 779 lands first).
- TASK_784 (after `4e72167` lands + GPU smoke for the 3 λ values).

**Launch after dependency cleared**:
- TASK_783 (after TASK_776 merges; sweep is no-op without 776's factored-path entropy term).

**Design-only, hold for operator code-budget greenlight**:
- TASK_778 (Bella-as-rollouts, ~3-5 days code + smoke; only authorize if TASK_775/776/777/779 all close null).
- TASK_780 (oracle-as-MCTS distillation via bridge_trainer.py, ~2-3 weeks code; only authorize after the full small-perturbation queue closes).

**Verdict count**:
- Ship-ready (no follow-up): 3
- Ship-ready after worktree merge: 4
- Ship-ready with dependency: 1
- Design-only (no launch yet): 2
- Total: 10

**Blockers**: none. Every task is a viable next-step. The only hard inter-task ordering constraint is TASK_776 → TASK_783.

**Risk surfaces flagged**:
- TASK_777 master/worktree duplication needs cleanup before merge (avoid confusing operator).
- TASK_782 needs the 4-LOC CLI patch or a JSON-override decision before Schedule B launches.
- TASK_775's two minor line citation drifts are cosmetic — the mechanism understanding is intact.
- TASK_779's residual citation drifts (MCTS expansion mask `:1742-1762` should be `:1783-1795`; CLI plumb `:2348-2349, 2355-2356` should be `:2361-2362, 2373-2374`) are cosmetic — flags and SPEC are correct.

**No conflicting requirements** between TASK_779 and TASK_781 (which both use `rs050` + `767` flags) — they are 1-leg and 3-leg variants of the same stack with explicit baseline-selection logic documented in TASK_781 SPEC.

**Deploy contract compliance**: confirmed across all 10 deliverables. No file contains `deploy.sh` instructions, no commit-to-master directives, no force-push. The 4 code-touching sub-agents committed to worktree branches only.

# Tech Debt Audit — 2026-05-27

Scope: post-TASK_780 ship. Pure findings, no remediation tasks created.
Reviewed `train_easy.py` / `train_phase2.py` / `bridge_trainer.py` / `easy_trainer.py`,
`phase2_config.py` / `easy_config.py`, `tasks/`, `experiments/`, `runs/`,
`tools/`, deploy plumbing, lore citations from the last week.

---

## 1. Dead / orphaned code

### 1.1 `real_trade_jax.py` imports a deleted module (HIGH)

- `real_trade_jax.py:34` imports `RealPortfolioManager` from `mcts_trader.agent.real_portfolio`.
- `mcts_trader/agent/real_portfolio.py` is `D` (staged deletion) in current `git status`; file is no longer on disk. Same goes for `mcts_trader/agent/portfolio.py`, `mcts_trader/env/trading_env.py`, `mcts_trader/env/wrapper.py`, `mcts_trader/tactics/__init__.py`.
- **What's wrong**: live-trading entry point is *broken at import time*. Anyone who runs `python real_trade_jax.py` today gets `ModuleNotFoundError`.
- **Cleanup**: either restore `real_portfolio.py` (if live trading is still intended to work) or move `real_trade_jax.py` to `_old/` and document the deprecation. Same call for `tools/test_env_init.py` (imports `mcts_trader.env.trading_env`) and `tools/orchestration/dojo_manager.py` (deprecated per `tools/orchestration/CLAUDE.md`, also imports deleted modules).
- **Risk if left**: HIGH — silent decay of the only live-trading surface; will not be caught until the next time you go to deploy.

### 1.2 Root-level `.bak` / `.backup` files

- `/home/lukas/Trader/train_easy.py.backup.20260222` — 4-month-old snapshot.
- `/home/lukas/Trader/train_jax_fast.py.bak` — backup of a router that is itself stale (see §1.3).
- `/home/lukas/Trader/mcts_trader/env/vectorbt_env.py.bak`
- `/home/lukas/Trader/training_schedule_old.json`
- **Cleanup**: delete.
- **Risk**: LOW (size negligible) but they show up in audits and lure grep into noise.

### 1.3 `train_jax_fast.py` no longer an entry point

- `CLAUDE.md`, `GEMINI.md`, `GLM.md`, `KIMI.md`, `README.md`, `SERVER.md`, `tools/parallel_epoch_training.py`, `tools/test_episode_length.py`, `tools/run_scheduled_training.md` all still mention `train_jax_fast.py` as "the Poppy training entry".
- Current production routes are `train_easy.py` → `easy_trainer.py` / `bridge_trainer.py` → `train_phase2.py`. `train_jax_fast.py` is not on any active path (no caller in code, no recent log mentions, no `tools/deploy/launch-experiment.sh` references in the last month of lore).
- **Cleanup**: either delete or relegate to `_old/`. At minimum update CLAUDE.md's top-level layout table and the GEMINI/GLM/KIMI markdowns to point at the real entry points.
- **Risk**: MED — false signposting wastes onboarding/sub-agent time; an agent following CLAUDE.md will pollute the wrong file.

### 1.4 Stale dojo scripts referenced as "deprecated" but still in `tools/orchestration/`

- `tools/orchestration/dojo_v2.py`, `dojo_manager.py`, `dojo_dynamic_curriculum.py`, `train_manager.py` are marked **"DO NOT EDIT — deprecated"** in `tools/orchestration/CLAUDE.md` yet remain checked in (≈85 KB of unused code) and `dojo_manager.py` imports the now-deleted `mcts_trader.tactics`.
- **Cleanup**: move to `_old/`.
- **Risk**: LOW.

### 1.5 Ad-hoc one-shot launch scripts that target closed experiments

- `tools/run_737_ablations.sh`, `tools/run_739_dense_sign.sh`, `tools/run_740_bella_distill.sh`, `tools/run_a1b1_stacked.sh`, `tools/run_overnight.sh`. All target experiments that are CLOSED / NULL per MEMORY (737 superseded by 743; 739 inconclusive; 740 Bella class closed).
- `train_diff_teacher_small.sh`, `run_dual_easy.sh` at repo root — same category.
- **Cleanup**: remove (or move to `tools/_old/`).
- **Risk**: LOW.

### 1.6 Root-level ad-hoc logs / metrics

- `a1b1_run.log`, `ablation_737.log`, `bench_phase1_v3.log`, `hang_debug.log`, `overnight_2026-04-23.log` (+ `.failed1`, `.failed2`), `task_739_run_20260424_1101.log`, `nohup.out`, `benchmark_results.json`, `last_run_metrics.json`, `task_739_result.md`, `task_a1b1_result.md`, `ablation_737_summary.md`.
- These predate the `runs/<id>/` discipline (see `experiments/QUEUE.md` + `tools/deploy/README.md`).
- **Cleanup**: prune (they're already in `.training-exclude` so server is unaffected; but they remain in workstation working-copy and git status churn).
- **Risk**: LOW.

---

## 2. Stale tasks (status field out of sync with reality)

All task files below claim DRAFT / ACTIVE / IMPLEMENTED / DRAFTED status, but their experiment cells are CLOSED in `experiments/QUEUE.md` and/or `MEMORY.md` findings. None has been re-stamped with a closing verdict.

| Task file | Claimed status | Actual status (per QUEUE / MEMORY) |
|---|---|---|
| `TASK_734_PREDICTION_HEAD_ALPHA.md` | ACTIVE | DONE — 5-seed REVERSAL Δα +0.20pp, NOT PROMOTED |
| `TASK_738_LEGACY_27CH_ABLATION.md` | ACTIVE | DONE — REGRESSION −18.44pp |
| `TASK_739_DENSE_SIGN_ORACLE.md` | IMPLEMENTED — Ready to launch | DONE — INCONCLUSIVE −2.63pp, ceiling hypothesis closed |
| `TASK_740_BELLA_DISTILL_HEAD.md` | IMPLEMENTED — Ready to launch | DONE — REGRESSION; Bella class CLOSED |
| `TASK_741_PER_ASSET_INJECTION_FIX.md` | PROPOSED | DONE — REGRESSION −6.98pp; abandoned |
| `TASK_749_TITANS_PIVOT.md` | DRAFTED | DONE — REGRESSION −3.11pp; Titans-MAG closed |
| `TASK_749_OOM_FIX_REVIEW.md` | REVIEW — blocked | Bf16 carry fix shipped; reviewable as historical |
| `TASK_753_BELLA_AS_DATA_AUGMENTATION.md` | DRAFTED | DONE — CATASTROPHIC −16.54pp |
| `TASK_754_BELLA_TRAJECTORY_SHAPE_DISTILL.md` | DRAFTED 2026-05-17 | DONE — REGRESSION −5.80pp; Bella class CLOSED |
| `TASK_755_BELLA_REGIME_CONDITIONAL.md` | DRAFTED 2026-05-17 | Class CLOSED 2026-05-19 |
| `TASK_756_VALUE_TARGET_DEEPER_PROBES.md` | DRAFTED | Value-target roadmap CLOSED (per QUEUE) |
| `TASK_763_POLICY_ENTROPY_REG.md` | DRAFT — pending | CLOSED NULL 2026-05-25 (per MEMORY) |
| `TASK_767_ACTION_SPACE_RESTRUCTURE.md` | GEMINI-PROPOSED | CLOSED NULL 2026-05-25 (per MEMORY) |
| `TASK_764_ORACLE_K_REFORMULATION.md` | DRAFT | Roadmap supersedes (see TASK_761) |
| `TASK_775` / `776` / `777` / `778` / `779` / `780` / `781` / `782` / `783` / `784` | ACTIVE-ish | Mixed; see `lore/validation_2026-05-26_ship_path.md` for live status — file at minimum out-of-sync with the recent ceremony |
| `TASK_730_RANK_ORACLE.md`, `TASK_735_AUX_PRICE_HEAD.md`, `TASK_722`, `TASK_737` | ACTIVE | All landed long ago (rank+aux is `a1b1` baseline) |

Also stale at directory level:
- `tasks/TASK_690_DUAL_DOJO.md`, `tasks/TASK_062_HISTORICAL_DOJO_PREPARATION.md`, `tasks/TASK_108_LATENT_DIFFUSION_MARKET_GENERATOR.md`, `tasks/TASK_109_DIFF_TEACHER_VALIDATION.md`, `tasks/TASK_110_ADVERSARIAL_MARKET_GENERATOR.md`, `tasks/TASK_115_BUGHUNT_SHAPE_MISMATCHES.md`, `tasks/TASK_115b_*`, `tasks/TASK_116_*` all marked DRAFT/IN-PROGRESS — pre-Foxglove-pivot era. Not active.

**Risk**: MED. Future sub-agents / sessions read these files and waste time triaging "what's still live" or worse, re-launching closed mechanisms.

**Cleanup pattern**: append a one-line `**Closed**: YYYY-MM-DD — see runs/<id>/result.md` to each closed task and either leave in place or move to `tasks/_old/` (already exists with 85 BUG_*.md historical entries).

---

## 3. Stale experiments / runs

`runs/` is ~19 GB; many experiment dirs are closed-null and can be archived.

- **Confirmed CLOSED / NULL (safe to compress or archive off-disk)**:
  `runs/738_legacy_27ch/` (497M), `runs/740_bella_distill/` (501M), `runs/741_mint/` (992M), `runs/747_value_td0/`, `runs/748_flow_velsweep/`, `runs/749_titans_mag/` (993M), `runs/757_bella_shape/` (497M), `runs/758_bella_augment/` (496M), `runs/760_train_mcts_sims/` (991M), `runs/764_oracle_k/` (831M), `runs/767_action_space/` (496M), `runs/763_entropy_reg/` (2.4G), `runs/769_value_clip/`, `runs/739_dense_sign/`, `runs/737_ablations/` (690M), `runs/734_flow_matching/` (826M), `runs/746_value_oracle/` (826M), `runs/762_reward_shaping/` (3.0G — Wave-1 superseded by Wave-2).
- **Spec dirs without runs (`experiments/750_titans_hero/`, `experiments/751_titans_lr_sweep/`)** — cancelled per QUEUE. Retain SPEC, drop the entry from QUEUE active section.
- **`runs/000_smoke/` (2.3G)** — accumulating smokes; rotate.

**Cleanup**: per the new SPEC discipline, `result.md` + `metrics.csv` + final checkpoint are the keepers; per-seed Orbax trees can be deleted once closed. A `tools/archive_closed_run.sh <id>` would be appropriate, but per the audit mandate that's a wishlist item, not debt. Concrete debt: the disk usage is on the workstation only (server has its own copy), so this is purely a workstation hygiene issue.

**Risk**: LOW (no correctness impact); MED if backup/disk tightens.

---

## 4. Inconsistent gitignore / tracking drift

### 4.1 ~1080 stale `D` entries in `git status` (MED)

`git status --short` shows ~1085 paths changed (mostly `D _old/...`, `D jax_ckpt_fast/...`, `D experiments/.../checkpoints/...`). These are deletions that have never been committed. Every session opens with this ~1k-line wall of noise, obscuring real working-tree changes.

- **Cleanup**: one focused commit (`git rm -r _old/ jax_ckpt_fast/experiment\ 141220251839/ research/ output/ tensor_snapshots/ out_ad03/ lore/arXiv-2512.11525v1/`) plus `.gitignore` rules for the dirs that should never have been tracked.
- **Risk**: MED — high noise floor masks real anomalies. Also: large staged deletions inflate `git gc` cost.

### 4.2 `tools/deploy/.training-exclude` does NOT exclude `.claude/worktrees/` (HIGH)

- `du -sh .claude/worktrees` → **5.2 GB**, 6 agent worktrees on the workstation.
- Looking at `tools/deploy/.training-exclude`: `.claude/` is partially excluded (`/.claude/settings.local.json`, `/.claude/scheduled_tasks.lock`) but `/.claude/worktrees/` is NOT, and `/.claude/agents/` isn't either.
- **What's wrong**: every `deploy.sh` rsyncs ≥5 GB of agent worktree state to the server. Multiplied by every sub-agent launch.
- **Cleanup**: add `/.claude/worktrees/` and `/.claude/agents/` to `.training-exclude`. Also add `**/.claude/` patterns to `.gitignore` for safety.
- **Risk**: HIGH — burns deploy bandwidth, can ship in-flight sub-agent work to the server (the very risk called out in `feedback_sub_agent_no_deploy.md`).

### 4.3 `.training-exclude` is itself untracked

- It's listed as `?? tools/deploy/.training-exclude` in git status. Combined with `?? tools/deploy/deploy.sh`, `?? tools/deploy/launch-experiment.sh`, `?? tools/deploy/pull-results.sh`, `?? tools/deploy/bootstrap-server.sh`. The deploy plumbing is uncommitted on master.
- **Cleanup**: commit `tools/deploy/*.sh` + `.training-exclude` to master (or to a deploy branch and merge).
- **Risk**: HIGH — server-deploy is run from working-tree files that can drift / be lost.

### 4.4 Untracked `tools/` content

A large chunk of `tools/` is `??` (e.g. `tools/INDEX.md`, `tools/orchestration/*`, `tools/DS*` dashboards, `tools/smoke_780_*.py`, `tools/precompute_bella_predictions.py`, `tools/calibrate_739_epsilon.py`, `tools/test_*.py`). Many of these are actively imported / referenced.

- **Cleanup**: triage and commit the live ones; delete the rest. `tools/smoke_780_phase_a.py` and `tools/smoke_780_phase_b_bootstrap.py` are referenced from TASK_780 and ship-path validation — **these must be committed**.
- **Risk**: MED — load-bearing tools can silently disappear.

### 4.5 Bella checkpoint dirs tracked but listed in `.training-exclude`

- `jax_ckpt_fast/` is staged-deleted at the file level; the dir is also in `.training-exclude`. Mixed signals.
- `output/`, `runs/`, `logs/`, `checkpoints/`, `wandb/` are now in `.gitignore` (new today) but pre-existing tracked files inside them still appear as `D`. Cleanup needs `git rm -r --cached` for those.
- **Risk**: LOW.

---

## 5. TODOs / half-finished work in active code paths

Skipping `_old/`, `docs/`, `trader_env/`, `tests/`, `.claude/`. Active-code TODOs:

- `real_trade_jax.py:388` — `# Warmup... (TODO: Full implementation)`. Live-trading initialization left as a stub (combined with §1.1, the live path is effectively non-functional).
- `phase2_config.py:205` — `# TODO(TASK_739 follow-up)` for `oracle_mode='hybrid'`. TASK_739 closed inconclusive; the hybrid mode will never ship via the original plan. Either implement or remove the reserved value.
- `mcts_trader/data/segment_pool.py:423` — same `oracle_mode='hybrid'` TODO; pairs with the above. The raise message is correct (refuses 'hybrid'), but the TODO comment is misleading.
- `mcts_trader/jax_agent/redo.py:245` — `# TODO: Implement proper optimizer state reset if needed`. ReDo (TASK_042) is not in any active path per recent lore — verify whether ReDo is still wired before fixing; otherwise delete.

**Risk**: LOW on each individually; MED for `real_trade_jax.py` given live-trading framing.

---

## 6. Deploy / plumbing gaps

Already covered §4.2 (worktrees) and §4.3 (uncommitted deploy scripts). Additional:

- **`tools/deploy/.training-exclude` excludes `/jax_ckpt_easy` and `/jax_ckpt_fast` but NOT `/jax_ckpt_easy_phase1_v3`, `/jax_ckpt_easy_phase1_xs32`, `/jax_ckpt_easy_phase2_*`** — root-level Phase-1/Phase-2 checkpoint dirs. They are caught by the wildcard `/jax_ckpt_*/` further down, but the wildcard appears after the named excludes; rsync's behavior here depends on ordering. Spot-check the next deploy. (`/jax_ckpt_*/` is the catchall — confirm intended; if so, simplify to just the catchall.)
- **`tools/deploy/.training-exclude` is missing `/data/`** at the explicit anchored form but has `/data/` listed (line 14). However, `data/` (no leading slash) is not specified — rsync should handle this fine, just flagging the lone source-of-truth.
- **`requirements.frozen.txt` divergence** — `MEMORY.md` cites this as the server install path. Not audited for actual drift, but the file is committed and is shipped on every deploy. Worth a `pip freeze` diff next deploy.

**Risk**: LOW–MED.

---

## 7. Broken / missing markdown citations

Spot-checked the most recent week of lore against `train_phase2.py` (the most-cited file).

- `lore/notes_2026-05-23.md` cites `phase2_config.py:212-220` for `value_clip_mode: str = "pre_popart"`. Actual line is **`phase2_config.py:264`** (the surrounding block runs `:256-264`). Drift: ~50 lines.
- Same line cites `train_phase2.py:680-689` (numpy per-step clip), `:1260-1269` (oracle JIT), `:1281-1284` (nstep JIT), `:1293-1297` (td0 JIT), `:1043-1050` (closure capture), `:2306-2316` (argparse), `:2631-2633` (override). Verified:
  - `:637` (immediate-reward clip) — accurate.
  - `:680-689` numpy gate — **actual is `:713`**. Drift +25 lines.
  - `:1043-1050` closure capture — **actual is `:1075`** (`value_clip_mode = getattr(config, ...)`). Drift +30 lines.
  - `:1260-1269` oracle JIT gate — **actual is `:1318+`** (5 gates between `:1318` and `:1395`). Drift ~+55 lines.
  - `:2306-2316` argparse — **actual is `:2468`**. Drift +160 lines.
- `lore/notes_2026-05-22.md` cites `train_phase2.py:637` (rs reward clip) and `:1198-1200` (PopArt clip) — `:637` still accurate; `:1198-1200` did not verify cleanly (PopArt-related blocks now around `:1395+`).
- `lore/notes_2026-05-22.md` cites `train_easy.py:88,188,327` for hardcoded `k=4` — not re-verified; likely fine, file size unchanged.

These citations were accurate the day they were written; subsequent commits (TASK_771 chain bugfix `bbcb4b0`, TASK_780 build, etc.) shifted the lines. The standing convention "cite by line number for navigation, by content for permanence" is good, but the divergence is now ≥150 lines in `train_phase2.py`.

**Cleanup**: nothing to fix in code. Flag: future lore should prefer symbol references (`scalar_to_categorical`, `make_stage_b_train_step`) for anything you might want to find ≥1 week later, line numbers only for the same-day pointer.

**Risk**: LOW.

---

## 8. Duplicate / divergent code

### 8.1 `scalar_to_categorical` lives in two places (MED)

- `train_phase2.py:78-87` — canon `scalar_to_categorical(values)`.
- `bridge_trainer.py:64-83` — `_scalar_to_categorical(values)`, docstring says "Mirrors train_phase2.scalar_to_categorical (:78-87) so bridge value targets share the same 401-bin layout".
- The mirror is literal: identical math, identical clip range, identical bin count. The TASK_780 author chose to copy rather than import — defensible (avoids a circular import on `train_phase2`'s heavy module-level JAX setup) but now any change to the 401-bin support has to happen in both files.
- **Cleanup**: hoist the helper into a shared module (`mcts_trader/jax_agent/__init__.py` already exports `VALUE_SUPPORT_SIZE`, `get_value_support` — a natural home).
- **Risk**: MED — silent divergence on PopArt / value-support changes.

### 8.2 Trainer base-class opportunity (LOW)

- `easy_trainer.py:EasyTrainer` (lines 37-573) and `bridge_trainer.py:BridgeTrainer` (112-715) both implement: `__init__` (config, RNG, model init), `train_step`, `evaluate_batch`, `save_checkpoint`, `load_checkpoint`, `create_*_trainer` factory. The TASK_780 plan deliberately chose "parallel module, not shared base" — that's fine. But once TASK_780 promotes (or doesn't), one of the two is going to be the orphan. Flag now, decide post-Phase-A-gate.

### 8.3 `train_phase2_cli.py` extraction is partial (LOW)

- `train_phase2_cli.py` was carved out in TASK_772.1 (2026-05-23, per file docstring) to hold `parse_args`. But `train_phase2.py` still has ≈300 lines of post-parse config-mutation logic (the `if args.X is not None: config.X = args.X` cascade around `:2300-2700`). The extraction is half-done; either complete it or document the split.

### 8.4 Wide passthrough flags (LOW–MED)

`phase2_config.py:282-322` keeps several "passthrough" / "documented no-op" fields:
- `use_cheat_channel` (`:286`) — passthrough only.
- `use_per_asset_injection`, `per_asset_inject_xs`, `per_asset_inject_cheat` (`:291-293`) — TASK_741 closed REGRESSION; CLI still accepts these.
- `use_flow_matching`, `flow_matching_velocity_weight` (`:126-127`) — flagged in-config as "documented no-op, do not rely on it"; TASK_734 closed; still wired into `train_phase2.py:1142, 2123, 2487`.
- `use_titans_memory` + 10 sibling `titans_*` fields (`:302-322`) — Titans-MAG class closed. Still wired into encoder selection (`train_phase2.py:415, 1007, 2182, 2207, 2665, 2689, 2821`).
- `use_bella_distill_head` (`:146`), `use_bella_shape_head` (`:158`), `bella_augment*` (`:176-190`) — Bella class CLOSED. Still wired (multiple `config.bella_*` sites in `train_phase2.py`).

Each is "harmless by default" (gates default-False), but each represents a code path that has to be reasoned about during every read of the loss function. The closed mechanisms (Bella, Titans, mint, flow_matching) collectively add ~15 config fields and ~50 conditional branches.

- **Cleanup**: pick one closed mechanism (Bella is the biggest) and either remove or fence under `# DEPRECATED — Bella class closed 2026-05-19, see project_encoder_bella_class_closed.md` so a future reader knows not to touch.
- **Risk**: MED — every closed mechanism left in the JIT trace is a latent correctness hazard (any of those branches firing accidentally costs days of debugging — TASK_770 / TASK_771 chain bug is precisely this pattern).

---

## Priority cleanup table

| # | Item | Effort | Impact | Risk if left | One-liner |
|---|---|---|---|---|---|
| 1 | `.claude/worktrees/` not in `.training-exclude` | 1 min | HIGH | HIGH | Add `/.claude/worktrees/` + `/.claude/agents/`; stops 5+ GB shipping per deploy + closes the sub-agent-deploy race described in `feedback_sub_agent_no_deploy.md`. |
| 2 | `tools/deploy/*.sh` + `.training-exclude` uncommitted | 5 min | HIGH | HIGH | Commit the deploy plumbing to master; right now the deploy path is a working-tree artifact that can disappear. |
| 3 | `real_trade_jax.py` imports deleted `real_portfolio` | 30 min | HIGH | HIGH | Either restore `mcts_trader/agent/real_portfolio.py` or move `real_trade_jax.py` to `_old/`. Live-trading entry is broken at import. |
| 4 | ~1080 stale `D` entries in `git status` | 15 min | MED | MED | One commit: `git rm -r` the deleted trees + commit `.gitignore`. Restores a usable `git status` baseline. |
| 5 | Stale task statuses (734/738/739/740/741/749/753/754/755/763/767…) | 30 min | MED | MED | Append `**Closed**: <date> — see runs/<id>/result.md` to ~15 task files OR bulk-move closed ones to `tasks/_old/`. |
| 6 | Stale entry-point docs (CLAUDE.md, README.md, GEMINI/GLM/KIMI.md) | 10 min | MED | MED | Replace `train_jax_fast.py` references with `train_easy.py` / `train_phase2.py`. Affects new-session orientation. |
| 7 | Closed-mechanism config flags still in JIT trace (Bella, Titans, mint, flow_matching) | 1-2 h | MED | MED | At minimum, mark each block `# DEPRECATED — see <memory_note>`. Prevents accidental re-enable + clarifies loss function for next reader. |
| 8 | Duplicate `scalar_to_categorical` (bridge_trainer + train_phase2) | 15 min | LOW | MED | Hoist into `mcts_trader/jax_agent/__init__.py` alongside `VALUE_SUPPORT_SIZE`. |
| 9 | Root-level `.bak`/`.backup`/scratch logs | 5 min | LOW | LOW | `rm -i *.log *.json.backup train_*.bak train_easy.py.backup.20260222 training_schedule_old.json nohup.out`. |
| 10 | Archive 19 GB of closed `runs/` | 30 min | LOW | LOW | Keep `result.md` + final ckpt + `metrics.csv`, drop per-seed Orbax trees for closed cells (738/740/741/747/748/749/757/758/760/763/764/767/769…). |

---

## Out-of-scope notes (flagged, not actioned)

- **`requirements.frozen.txt`** drift vs current `pip freeze` not verified; worth a snapshot before the next server deploy.
- **`tasks/_old/` already exists** with 85 historical BUG_*.md entries — established pattern, just underutilized for the post-Foxglove task wave.
- **`experiments/QUEUE.md`** is well-maintained as live truth — most reliable source of experiment status in the repo. Use it as the canonical pointer when reconciling §2.

End of audit.

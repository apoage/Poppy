# TASK_785 — Tech Debt Cleanup

**Status**: DRAFT
**Created**: 2026-05-27
**Source audit**: `lore/tech_debt_audit_2026-05-27.md`
**Scope**: Post-TASK_780 hygiene pass. Code + repo state, no new mechanisms.

---

## Hypothesis

The single-mechanism + ship-path waves left behind dead imports, stale tasks, ~1080 uncommitted deletions, untracked deploy plumbing, and ~15 closed-mechanism flags still wired into the JIT trace. Each item costs (a) onboarding/sub-agent time, (b) latent correctness risk, or (c) deploy bandwidth. Cleanup is mechanical, low-risk, and unlocks faster reading of the loss function.

## Cleanup units (each landable as its own commit)

### Unit A — Deploy plumbing (HIGH priority, ~10 min)

1. Add to `tools/deploy/.training-exclude`:
   ```
   /.claude/worktrees/
   /.claude/agents/
   ```
   Stops the 5.2 GB of agent worktree state shipping on every `deploy.sh`.
2. Commit the previously-untracked deploy plumbing to master:
   - `tools/deploy/deploy.sh`
   - `tools/deploy/launch-experiment.sh`
   - `tools/deploy/pull-results.sh`
   - `tools/deploy/bootstrap-server.sh`
   - `tools/deploy/sync-db.sh`
   - `tools/deploy/.training-exclude` (post Unit A.1 edit)
   - `tools/deploy/README.md`
3. Also commit load-bearing tools currently untracked:
   - `tools/smoke_780_phase_a.py`
   - `tools/smoke_780_phase_b_bootstrap.py`

**Verifier**: `deploy.sh --dry-run --code-only` shows no `.claude/worktrees/` entries; deploy plumbing absent from `git status` untracked list.

### Unit B — Live-trading import break (HIGH priority, ~30 min)

`real_trade_jax.py:34` imports `RealPortfolioManager` from `mcts_trader.agent.real_portfolio`, which is `D` in `git status` (file gone from disk).

**Decision required from operator**: is live trading still a deployable surface?
- If YES: restore `mcts_trader/agent/real_portfolio.py` (find last working version in git history pre-deletion) and refresh the import.
- If NO: move `real_trade_jax.py` to `_old/` and update `CLAUDE.md` § "Top-level layout" accordingly.

Same cleanup applies to:
- `tools/test_env_init.py` (imports `mcts_trader.env.trading_env`)
- `tools/orchestration/dojo_manager.py` (imports deleted `mcts_trader.tactics`; already marked deprecated in `tools/orchestration/CLAUDE.md`)

**Verifier**: `python -c "import real_trade_jax"` exits 0.

### Unit C — Git status restoration (MED priority, ~15 min)

`git status --short` currently shows ~1080 lines (mostly `D _old/...`, `D jax_ckpt_fast/...`, `D experiments/.../checkpoints/...`). Commit the deletions:

```bash
git rm -r _old/ jax_ckpt_fast/ research/ output/ tensor_snapshots/ \
  out_ad03/ lore/arXiv-2512.11525v1/ "experiments/14122025"  # etc per audit
git add .gitignore CLAUDE.md  # already modified
git commit -m "TASK_785 unit C: commit long-standing deletions + ignore"
```

Also `git rm -r --cached` for the now-gitignored dirs (`runs/`, `logs/`, `checkpoints/`, `output/`, `wandb/`) where pre-existing tracked files still show as `D`.

**Verifier**: `git status --short` returns < 100 lines.

### Unit D — Stale task status refresh (MED priority, ~30 min)

Re-stamp the 15 task files whose claimed status is out of sync with reality (per audit §2 table). Pattern per file:

```markdown
**Closed**: 2026-MM-DD — <one-liner verdict>. See `runs/<id>/result.md` or `MEMORY.md`.
```

Files: `TASK_734`, `TASK_738`, `TASK_739`, `TASK_740`, `TASK_741`, `TASK_749`, `TASK_753`, `TASK_754`, `TASK_755`, `TASK_756`, `TASK_763`, `TASK_767`, `TASK_730`, `TASK_735`, `TASK_722`, `TASK_737`. Also pre-Foxglove drift: `TASK_690`, `TASK_062`, `TASK_108-110`, `TASK_115`, `TASK_115b`, `TASK_116`.

Optionally `git mv` confirmed-closed files into `tasks/_old/` (pattern already exists with 85 BUG_*.md entries).

**Verifier**: sub-agent grep for `Status: DRAFT` / `Status: ACTIVE` returns only currently-active tasks (775-784 minus closed cells).

### Unit E — Signposting refresh (MED priority, ~10 min)

`CLAUDE.md`, `README.md`, `SERVER.md`, `GEMINI.md`, `GLM.md`, `KIMI.md`, `tools/parallel_epoch_training.py`, `tools/test_episode_length.py` all reference `train_jax_fast.py` as the Poppy entry. Replace with the current entry chain:
- Phase 1: `train_easy.py` → `easy_trainer.py` (or `bridge_trainer.py` post-TASK_780)
- Phase 2: `train_phase2.py` → `train_phase2_cli.py`

**Verifier**: `grep -rn "train_jax_fast" --include='*.md'` outside `_old/` and `lore/` returns nothing.

### Unit F — Closed-mechanism fencing (MED priority, ~1-2 h)

`phase2_config.py:282-322` and ~50 sites in `train_phase2.py` still carry config flags for closed mechanisms. Don't delete (the JIT traces are load-bearing for paired-Δα reproducibility of historical results), but fence each block with a single-line marker so the next reader knows it's frozen.

Block markers (one per mechanism):
- Bella distill / shape / augment: `# DEPRECATED — Bella class closed 2026-05-19 (project_encoder_bella_class_closed.md)`
- Titans-MAG: `# DEPRECATED — Titans class closed 2026-05-19 (notes_2026-05-19.md §146)`
- Mint xs (per-asset injection): `# DEPRECATED — TASK_741 regression (project_task741_result.md)`
- Flow matching: `# DEPRECATED — TASK_734 not promoted (notes_2026-05-15.md)`
- Entropy reg: `# DEPRECATED — TASK_763 closed null (project_763_767_closed_null.md)`
- Limit-orders disable: `# DEPRECATED — TASK_767 closed null (project_763_767_closed_null.md)`

**Verifier**: `grep -n "DEPRECATED" phase2_config.py train_phase2.py` lists all six mechanisms.

### Unit G — Duplicate helper (LOW priority, ~15 min)

`scalar_to_categorical` lives in two places (`train_phase2.py:78-87` + `bridge_trainer.py:64-83`, literal mirror per docstring). Hoist into `mcts_trader/jax_agent/__init__.py` alongside `VALUE_SUPPORT_SIZE` / `get_value_support`. Both call sites import from there.

**Verifier**: post-edit smoke — `python -c "from bridge_trainer import BridgeTrainer; from train_phase2 import scalar_to_categorical; print('ok')"` exits 0; canon Stage-B 5-step bit-identity smoke passes.

### Unit H — Scratch / `.bak` removal (LOW priority, ~5 min)

```bash
rm train_easy.py.backup.20260222 train_jax_fast.py.bak \
   mcts_trader/env/vectorbt_env.py.bak training_schedule_old.json \
   a1b1_run.log ablation_737.log bench_phase1_v3.log hang_debug.log \
   overnight_2026-04-23.log* task_739_run_*.log nohup.out \
   benchmark_results.json last_run_metrics.json \
   task_739_result.md task_a1b1_result.md ablation_737_summary.md
```

Same for one-shot scripts targeting closed experiments (`tools/run_737_*.sh`, `tools/run_739_*.sh`, `tools/run_740_*.sh`, `train_diff_teacher_small.sh`, `run_dual_easy.sh`).

**Verifier**: `ls *.log *.bak` shows nothing at repo root.

### Unit I — Archive 19 GB of closed `runs/` (LOW priority, ~30 min)

Workstation-local hygiene. For each closed cell in audit §3, keep `result.md` + `metrics.csv` + final ckpt; drop per-seed Orbax trees. No server impact (server keeps its own copy independently). Defer to disk pressure.

---

## Out of scope

- Trainer base-class consolidation (`EasyTrainer` / `BridgeTrainer` shared parent) — defer until TASK_780 promotes or doesn't.
- `train_phase2_cli.py` extraction completion — defer until next refactor sweep.
- Markdown citation drift in older lore — convention going forward should prefer symbol references; no remediation needed for historical notes.
- Adding `tools/archive_closed_run.sh` helper — wishlist, not debt.

## Falsifier / acceptance gates

- Unit A: `deploy.sh --dry-run --code-only` shows zero `.claude/worktrees/` entries.
- Unit B: live-trade entry-point imports clean OR file relocated to `_old/`.
- Unit C: `git status --short | wc -l` returns < 100.
- Unit D-F: as above.
- All units: canon Stage-B 5-step bit-identity smoke passes; canon `a1b1` benchmark run produces unchanged metrics (sanity).

## Execution order

A → B → C in one sitting (correctness-critical, ~1 hour total).
D → E → F as a documentation pass (~2 hours).
G → H → I as workstation hygiene (anytime).

Each unit is its own commit. No worktree isolation needed; no flag-gating (this is cleanup, not mechanism).

## References

- Audit: `lore/tech_debt_audit_2026-05-27.md`
- Sub-agent deploy contract: `~/.claude/projects/-home-lukas-Trader/memory/feedback_sub_agent_no_deploy.md`
- Closed-mechanism findings: `project_encoder_bella_class_closed.md`, `project_task741_result.md`, `project_763_767_closed_null.md`, `project_single_mechanism_exhausted.md`

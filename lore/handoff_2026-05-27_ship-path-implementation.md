# Handoff — 2026-05-27 — Ship-path ceremony + TASK_780 implementation

**Sequence**: #3
**Previous**: `lore/handoff_2026-05-24_764-launch.md`
**Branch**: `master`
**Working dir**: `/home/lukas/Trader`
**Session focus**: Ship-path ceremony delivered TASKs 775-784 with SPECs + 4 worktree code changes + 10 AUTHORIZATION artifacts. Validation pass clean. TASK_780 oracle-as-MCTS distillation built end-to-end in 3 phases (~2200 LOC, all smoke gates PASS). Server findings (763 + 767 closed null) reshape combination priorities. Server NVML driver under maintenance — training blocked, code shipping unblocked.

## Confirmed rubric

```
Compaction rubric (chandoff 2026-05-27):

KEEP IN FULL:
- Ship-path ceremony (2026-05-25): 10 candidates → TASKs 775-784 with SPECs (7dabf65, 0fc8dbd, ee4aa4b, 7aa2420, 2e1828a, 372cca0) + 4 code commits (73ff976 776 entropy-factored, ba882bd 777 mixture-value, 96bd8c5 781 return-weighted-BC, f9d5aa4 784 EWMA-smoothed-oracle). 10 AUTHORIZATIONs in runs/77*/AUTHORIZATION_2026-05-26.md (gitignored).
- TASK_780 oracle-as-MCTS distillation fully implemented in 3 phases: bb7b500 (Phase 1 skeleton+helper), 148e8e9 (Phase 2 training-loop math), 006d265 (Phase 3-5 integration+smoke+TASK_782 CLI patch bonus). ~2200 LOC. All smoke gates PASS (latent_std parity 0.95, 5-epoch smoke loss 11.70→4.78). Operator authorized Phase-A launch as soon as server maintenance ends.
- Server findings reshape priorities: 763 entropy CLOSED NULL (+0.55pp best, anti-corr never fired — ΔBa=+0.19pp wrong direction); 767 limit-orders CLOSED NULL (−0.65pp, seed-7 trades exploded +448). Combinations 776/779/781/783 risk-elevated; 775/777/782/784 unaffected.
- Server NVML driver/library mismatch — training blocked, SSH/rsync fine. Workstation pulled ~7 GB of 762/763/764/767/769 checkpoints. Server training tip 096ad66 (PRE-TASK_780-deploy — needs re-deploy to push the 4 new commits).
- Next code action approved: land TASK_775 4-LOC --oracle-mask-weight CLI patch as sister to the TASK_782 --encoder-unfreeze-epoch patch already shipped in 006d265.

COMPRESS + POINT:
- 10-candidate triage → see lore/ship_path_recommendations_2026-05-25.md
- Per-task validation verdicts → see lore/validation_2026-05-26_ship_path.md
- TASK_780 phased build narrative → see lore/notes_2026-05-26.md §170
- Ceremony framing + Bloom-arc → see lore/notes_2026-05-25.md §169
- 763 + 767 null findings → see memory/project_763_767_closed_null.md

DROP:
- The "not everything is vectors" lesson from lore §170 — operator rejected as not durable.
- Cherry-pick conflict resolution mechanics on phase2_config.py / train_phase2.py / train_phase2_cli.py (resolved, additive merges).
- Stale-TASK_777-on-master git-mechanics workaround (resolved).
- Worktree branch names (4 locked, harness auto-cleans).
- mkdir runs/77*_* glob bug + fix (transient).
- TaskCreate/TaskList progress churn.

VERBATIM ANCHORS:
- Master tip 5268872 (Lore §170 commit).
- TASK_780 chain: bb7b500 → 148e8e9 → 006d265.
- TASK 775-784 SHAs (order): 7dabf65, 73ff976, ba882bd, 0fc8dbd, ee4aa4b, 7aa2420, 96bd8c5, 2e1828a, 372cca0, f9d5aa4.
- Server training tip: 096ad66 (stale vs workstation).
- Phase-2 smoke: 11.70 → 4.78 (5 epochs).
- Phase-3 latent_std parity ratio: 0.95.
- 763 best Δα: +0.55pp at ec02, ΔBa +0.19pp.
- 767 mean Δα: −0.65pp; s7 trade explosion +448.

UNCERTAIN:
- Whether the 10 AUTHORIZATIONs in runs/77*/ deserve a snapshot into lore/ for durability or should stay gitignored only (current convention says gitignored; depends on whether workstation backup is enough).
```

## Git state at handoff

```
5268872 Lore §170: TASK_780 build — oracle-as-MCTS distillation lands in 3 phases
006d265 TASK_780 Phase 3-5: integration tests + smoke protocol + Phase-A gate helper
148e8e9 TASK_780 Phase 2: bridge_trainer training-loop math + bridge buffer + wire
bb7b500 TASK_780 Phase 1: bridge_trainer skeleton + oracle_to_mcts_targets helper + config/CLI wiring
b0ea7e8 Lore 2026-05-25: ship-path ceremony — 10-candidate triage + validation
```

Working tree: only long-standing `_old/` deletion drift + `.gitignore`/`CLAUDE.md` edits unchanged from prior sessions. Nothing new uncommitted.

Server-side state (not in workstation git): server training tip `096ad66` is from the deploy that happened mid-session, BEFORE TASK_780 Phase-1/2/3-5 landed. Re-deploy required to push the 4 new commits (`bb7b500`, `148e8e9`, `006d265`, `5268872`) to server training branch.

## Next action

1. Land TASK_775 4-LOC `--oracle-mask-weight` CLI patch at `train_easy.py:1885-1886` (sister to the TASK_782 patch already shipped in `006d265`). Trivial follow-up: `parser.add_argument('--oracle-mask-weight', type=float, default=None)` + `if args.oracle_mask_weight is not None: config.oracle_mask_weight = args.oracle_mask_weight`. Single commit.

2. When server maintenance ends + NVML driver/userspace reconciled: `tools/deploy/deploy.sh` to push workstation → server (will bring server from `096ad66` to current workstation tip + TASK_775 patch). Then authorize Phase-A launch per `runs/780_oracle_as_mcts/AUTHORIZATION_2026-05-26.md` (3 seeds × ~165 min on GPUs 1+2).

3. While waiting for Phase-A, consider the cheaper sibling probes per `lore/validation_2026-05-26_ship_path.md`: TASK_779 (rs050+767, ~5h wall, BUT now risk-elevated by 767 closed-null), TASK_777 mixture-value (~13h wall, unaffected by recent nulls), TASK_775 oracle-mask Phase-1 (~3-4h wall, unaffected). Priority ordering depends on whether to ship the cheapest-falsifier first or the highest-EV-residual first.

4. AUTHORIZATIONs durability: decide whether to snapshot `runs/77*/AUTHORIZATION_2026-05-26.md` into `lore/` for git tracking, or rely on workstation backup (current convention is gitignored runs/).

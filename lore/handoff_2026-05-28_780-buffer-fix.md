# Handoff — 2026-05-28 — TASK_780 buffer-init fix + 769/779 closure + tech debt task

**Sequence**: #4
**Previous**: `lore/handoff_2026-05-27_ship-path-implementation.md`
**Branch**: `master`
**Working dir**: `/home/lukas/Trader`
**Session focus**: Tech debt audit + TASK_785 written. Server agent closed 779 (NULL) + 769 (MECHANISM PASS — F2 PopArt pre-clip fix structurally validated, rs200 Δα +1.02pp vs 762-rs200). TASK_780 Phase-A pre-flight crashed with KeyError 'visit_distributions' (canon prefill populating buffer used by bridge); ws shipped 8-LOC fix `a73d727` and redeployed (`62fa34a → 4f33daa`). Server agent now authorized to launch 775 on GPU 1+2 and retry 780 pre-flight on GPU 3.

## Confirmed rubric

```
Compaction rubric (chandoff 2026-05-28):

KEEP IN FULL:
- TASK_780 buffer-init bug + ship cycle: server pre-flight smoke crashed KeyError 'visit_distributions' at train_easy.py:1289 (canon _refill_buffer populates ep_data without bridge keys). Fix a73d727 — branch both prefill sites (lines 527, 570) on use_oracle_mcts_distill. Redeployed 62fa34a→4f33daa. Server agent has REPLY_2026-05-28.md instructing 775 launch on GPU 1+2 + 780 retry on GPU 3.
- 769_value_clip MECHANISM PASS: rs200 Δα +1.02pp vs 762-rs200 baseline (3/3 seeds positive) — F2 PopArt pre-clip fix STRUCTURALLY VALIDATED. rs200 ≥ rs100 by +0.25pp → reward_scale lever re-opened. Standard gate FAIL (rs100 −0.56pp). result.md on server.
- 779_rs050_767 NULL: mean Δα −0.50pp, all 3 seeds negative. Side-finding: rs050 throttle suppressed 767 seed-7 over-trade (s7 trades=138, not the +448 stand-alone outlier from project_763_767_closed_null.md). result.md on server.
- TASK_785 written (c55861c) — 9 lettered units A-I, each landable as own commit. HIGH: deploy.sh missing .claude/worktrees/ exclude; real_trade_jax.py imports staged-deletion module; ~1080 stale D entries in git status.
- eval_seed_7 wedge resolved as TRANSIENT (not harness bug): killed PID 160009 at 9h11m wall mid-period-5; clean 22-min rerun did not reproduce. Possibly cgroup/driver hiccup during the Ubuntu 26.04 + NVIDIA 580 upgrade window.

COMPRESS + POINT:
- Tech debt findings → see lore/tech_debt_audit_2026-05-27.md (~290 lines, 10 categories)
- TASK_785 cleanup units A-I → see tasks/TASK_785_TECH_DEBT_CLEANUP.md
- Server agent overnight tasks 1-4 verdicts → see runs/STATUS_2026-05-28.md (on server)
- Server buffer-bug Q1 + resolution → see runs/QUESTIONS_2026-05-28.md + runs/REPLY_2026-05-28.md (server-side)

DROP:
- "PRIVATE WITH FREE PACK" interrupt — UI noise, ignore.
- Lore time-bias scrub mechanics (specific phrases removed: "~2-3 weeks code", "~4 hours of agent wall", "21 min after Phase 1", "~3-5 days code", "in a single day"). Already in commit c55861c.
- Sleep/tired/time-of-day grep findings — confirmed memory was clean; only feedback_memory_layer_model.md mention is the defensive rule, kept.
- `deploy.sh` rsync line-by-line of stale .claude/worktrees/ paths (known harmless garbage per chandoff #2).
- Cherry-pick mechanics from prior sessions — already in prior handoffs.

VERBATIM ANCHORS:
- Master tip: a73d727 (TASK_780 buffer-init fix).
- Chain this session: 5268872 → c55861c (cleanup) → a73d727 (780 fix).
- Server training tip: 4f33daa (post-deploy of a73d727).
- TASK_780 bug site: train_easy.py:526-531 + :570-574 (8 LOC patch).
- 769 verdicts: rs200 Δα +1.02pp vs 762-rs200, rs100 mean −0.56pp.
- 779 verdict: mean Δα −0.50pp, s7 trades=138.
- Server killed PID: 160009 (9h11m, GPU 1, 18 GB).

UNCERTAIN:
- Whether the `--oracle-mask-weight` CLI flag is actually wired on server (REPLY tells server agent to verify and use env-override fallback if not). Server agent's check is the source of truth — ws hasn't confirmed.
- Whether the next session should fold STATUS_2026-05-28.md + 769 F2-validation finding into experiments/QUEUE.md immediately, or wait for TASK_775 result first.
```

## Git state at handoff

```
a73d727 TASK_780 buffer-init branch: route prefill to _refill_buffer_bridge when oracle-mcts-distill
c55861c Lore + tasks cleanup: scrub impl-time bias, audit + cleanup task
5268872 Lore §170: TASK_780 build — oracle-as-MCTS distillation lands in 3 phases
006d265 TASK_780 Phase 3-5: integration tests + smoke protocol + Phase-A gate helper
148e8e9 TASK_780 Phase 2: bridge_trainer training-loop math + bridge buffer + wire
```

Working tree: only long-standing `_old/` deletion drift + assorted M files from prior sessions (config, diff_teacher, mcts_trader). The two cleanup commits this session were lore + train_easy.py only; the dirty working tree is pre-existing.

Server training tip: `4f33daa` (matches `a73d727 + deploy wrapper commit`). Server has `runs/REPLY_2026-05-28.md` instructing TASK_775 launch on GPU 1+2 and TASK_780 retry on GPU 3.

## Next action

1. Wait for server agent to verify `--oracle-mask-weight` CLI wiring (or fall back to env override) and launch TASK_775 paired 3-seed on GPU 1+2.
2. Server agent retries TASK_780 pre-flight smoke at tip `4f33daa` on GPU 3; on PASS, launches Phase-A 3-seed (serialized after 775 finishes one cell, OR on GPU 3 only).
3. On next ws session: fold `runs/STATUS_2026-05-28.md` (769 F2 validation + 779 NULL) into `experiments/QUEUE.md` and update memory `project_763_767_closed_null.md` with the rs050-throttle-mitigates-s7-over-trade finding.
4. TASK_785 cleanup units A → C remain the highest-impact follow-up (deploy.sh worktree exclude + real_trade_jax import fix + ~1080 stale D entries).

# Handoff 2026-07-06 — vector-action pivot + doubt-prune

**Previous:** `lore/handoff_2026-06-25_campaign-closed-linear-optimal.md` (seq +1)
**Branch:** master · **Working dir:** /home/lukas/Trader
**Session focus:** "build a working trader on OHLC" → shipped the linear CTA, ran the full forensic
+ crash-recovery gauntlet, then PIVOTED on operator direction to **vector-action MuZero** (TASK_828)
as the real fix and pruned the project's accumulated fatalism (MEMORY.md 20.8→9.8KB).

## Compaction rubric

```
Compaction rubric (chandoff 2026-07-06):

KEEP IN FULL:
- The PIVOT: operator rejected deploying the linear rule ("too trivial") AND the reward-head
  patch (TASK_827); directed vector-action MuZero (TASK_828) as THE fix — action = per-asset
  weight vector, policy head already factored (muzero_model.py:256), scalar env step
  (jax_trading_env.py:1154) is the defect. Supersedes "planner closed / ship the rule."
- The doubt-prune mandate: memory reframed so old CLOSED/NULL/EXHAUSTED verdicts are SCOPED to a
  pre-Opus-4.8 scalar architecture, NOT laws. Next session must NOT default to "marginal/
  impossible/ship the rule." Rebuild-freely posture.
- Operator tone context: frustration was at defeatist framing; resolution = the rebuild direction.

COMPRESS + POINT:
- TASK_820 linear CTA (parity/gates/paper-live) → lore/notes_2026-07-06 + TASK_820 status.
- Forensic wave (oracle-intent, short-bias math, reward-head action-blind, contamination audit,
  758→null recon, stale result.md fixes) → lore/notes_2026-07-05 + the project_* memory files.
- TASK_821 planner run (5 attempts, seed 42, 0/1670 HOLD, closed) → runs/821_planner_daily/
  EVAL_PASS1.md + project_planner_closed_clean_cell.
- Crash night (3 sync floods → RAM slots A1/B1 + BIOS revert + idle voltage → fixed) → TASK_822 +
  runs/822_stability/BIOS_CHEATSHEET.md.
- Git corruption + recovery (840 empty objects, reset to 58e3f65, recommit) → one line.

DROP:
- Per-attempt 821 relaunch mechanics, screen/tmux churn, storage disk remount repeats.
- MCP connect/disconnect churn, /context + /usage dumps, stress-ladder pass-by-pass numbers,
  tripwire-monitor plumbing.

VERBATIM ANCHORS (must survive unchanged):
- HEAD 4b7171c; 687c146 (--seed); 88b2f02 (per_asset decode + q-no-reward); tag
  archive/pre-reorg-2026-07-05.
- TASK_828 (vector-action MuZero); reward-head 60× action-variance gap; 821 = 0/1670 vs +0.97 bar.
- Sync-flood reset 0x08000800 / MCE baa00000000c0135; fix = A2/B2 slots + BIOS 3641 + Typical
  Current Idle. MEMORY.md now 9.8KB.

UNCERTAIN (operator: correct me):
- Do 824/825/826 stay active, or is 828 the sole focus?
- Is TASK_827 (reward-head fix) fully dead or a fallback if 828 stalls?
- Keep the 820 paper loop running (systemd timer 02:21) or kill it given the pivot off the rule?
- Iris-DEX: pursue the node-less hosted-RPC path (TASK_826) or fully shelve?
```

## Git state
```
4b7171c Final stray sweep: AI-tool pointers, root result summaries, legacy scripts
6ea7fe7 Commit tools/config legacy working-tree state (audit plan #5)
2f5ae11 Commit Bella package legacy working-tree state (audit plan #3)
9144d29 Session artifacts 2026-07-06: crash-night lore, 821/822 task closures, RAM stress tool
88b2f02 benchmark_easy: add --q-no-reward eval flag, document per_asset decode
```
Untracked: tasks/TASK_823..828 (new specs), pingping.md, START_AFTER_REBOOT.txt, .claude/

## Next action
Start **TASK_828 Stage 1** — the vector env step: `step_vector(state, action_vec)` in
`mcts_trader/env/jax_trading_env.py` (vectorize the per-asset fill/fee/PnL currently gated by the
scalar `asset_idx = action//5` at :1154), flag-gated `--vector-action`, canon scalar path
bit-identical. Equivalence-test: a vector action == the equivalent sequence of scalar steps.

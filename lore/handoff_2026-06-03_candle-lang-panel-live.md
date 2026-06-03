# Handoff — 2026-06-03 — Candle-language panel BUILT, LAUNCHED, and running on gpu-server

**Sequence**: #10
**Previous**: `lore/handoff_2026-06-02_candle-language-design-build-scoped.md` (#9)
**Branch**: `master`
**Working dir**: `/home/lukas/Trader`

## Session focus
Built `tools/candle_lang/` (the branch-C candle-language tokenizer panel, 15 modules), CPU-verified it,
deployed it to gpu-server, and launched the full-corpus panel run. First server contact found three bugs
(hardcoded DB path, a 25 GB `quick_check` gate, and a ~9×-re-read I/O bottleneck) — all fixed and
committed. The run is live and self-sufficient. Full narrative: `lore/notes_2026-06-03.md` §176/§177/§178.

## Confirmed rubric
```
KEEP IN FULL:
- The TASK_789 candle-language PANEL is LIVE on gpu-server: tmux `789_panel`, full corpus
  (9 build + 5 holdout, cap 20k), 4 CPU members compute-bound -> L4 on GPU 3 after. Self-
  sufficient (linger on; repo-root-relative DB path; survives workstation-off). Re-entry: pull
  runs/789_token_c0/{L1_BPE,l2_abba,L3,L5,L4}_RESULT.md (+.json); if missing check logs/<m>.log.
- A8 cross-asset transfer is THE headline gate (build-vocab covers sub-cent holdout vs BOTH
  shuffle AND random-merge controls, mega-tokens excluded). A0=1.0 is structural; A1/A2/A9 support.
- tools/candle_lang/ built + CPU-verified (15 modules). The I/O fix is the load-bearing mechanism:
  db_stream load-once per-process window cache + tail-seek, GATED on max_windows set (unbounded path
  stays true-streaming). Took members 5%->100% CPU. No member code changed (all copy before shuffle).
- 3 server-launch fixes: config.db_path() -> repo-root-relative (was hardcoded workstation path);
  panel quick_check -> cheap indexed sanity (25GB scan too slow); the ~9x re-read I/O bottleneck.

COMPRESS + POINT:
- Full build -> verify -> launch -> I/O-fix narrative -> lore/notes_2026-06-03.md §176/§177/§178. Point.
- Workflow build (wl3uffw6j, 22 agents) + 6 blocking corrections SATISFIED -> §176 / TASK_795.

DROP:
- The 1610-file dirty working tree (pre-existing TASK_785 _old/ deletions; untouched all session).
- safe_run/tmux/ssh poll mechanics; per-poll progress snapshots; the two bad-path relaunches
  (db-path + quick_check) beyond "found+fixed"; verify-pass crossval false-positives (only IF-02 real).

VERBATIM ANCHORS:
- commits 88b686c, 4142295, 5ece1f7, 85369c2, 674ec18; prior #9 handoff 7d7e5f6.
- gpu-server = operator@<redacted-ip>:22; tmux `789_panel`; runs/789_token_c0/.
- db_stream: cache+tail-seek gated on max_windows; config.db_path() repo-root-relative.

UNCERTAIN:
- L5 matrix-profile is the compute long-pole at cap 20k -- may be very slow / not finish; cap may need lowering.
- Members lack a --build-assets CLI (only --max-windows-per-asset / --window[l1,l3] / --out); add for ablations.
- B (Poppy ExIt, pkl->Orbax warm-start) still unbuilt; TASK_785 deploy hygiene (git-push broken) open.
```

## Git state
```
88b686c lore §178: session close — panel live on gpu-server, re-entry checklist
4142295 lore §177 + TASK_795: server launch, the I/O wall, and the load-once fix
5ece1f7 candle-lang db_stream: load-once corpus cache + tail-seek (fix the I/O bottleneck)
85369c2 candle-lang: fix server-launch DB path + slow quick_check gate (found in first gpu-server run)
674ec18 candle-lang BUILD: tools/candle_lang/ panel (15 modules), CPU-verified
```
`git status --short`: ~1610 entries — ALL pre-existing TASK_785 debt (`_old/` deletions, untracked churn),
untouched this session. Every session commit added only `tools/candle_lang/`, `tasks/TASK_795*`, and
`lore/notes_2026-06-03.md`. Do NOT sweep the dirty tree.

## Next action
When back online: **pull the 789 scorecards** — SSH `operator@<redacted-ip> -p 22`, read
`runs/789_token_c0/{L1_BPE,l2_abba,L3,L5,L4}_RESULT.md`. Read **A8 cross-asset transfer first** (the
branch-C verdict). If a member's RESULT is missing it's still running or died — check
`runs/789_token_c0/logs/<member>.log`. If L5 is still grinding, that's expected (MP compute long-pole);
the L1/L4 headline lands first. Then decide promote-or-kill on branch C, and whether to start B
(pkl→Orbax warm-start adapter).

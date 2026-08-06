# Handoff 2026-08-05 (#18, planner) — 861 chain read + FOUR operator architectural rulings; the ledger's negatives go PROVISIONAL

**Previous:** `lore/handoff_2026-07-29_frame-pin-chain-planner.md` (#17).
Factory-side this window: `lore/handoff_2026-08-03_factory_run25.md` + `lore/handoff_2026-08-04_factory_run_25.md`.
**Branch:** master · **Working dir:** /home/lukas/Trader
**Session role:** PLANNER/orchestrator — QUEUE/DECISIONS/LEDGER/SPECs + planner-scheduled GPU runs;
factory executes Ready rows in its own session; no planner writes to shared surfaces while armed.
**Session focus (Aug 3→5):** ratified factory runs 24+25 → launched and READ the 861 real-leg chain
(the program's biggest science block: search-teacher FAIL, reference re-base, OOS collapse) → then
spent the second half being corrected by the operator on four architectural points, three of which
invalidated framings I had already committed to the record. Net effect: the science is honest, the
ledger's whole negative history is now marked PROVISIONAL pending re-runs on the fixed framework,
and checkpoint lifecycle went from "queued task" to shipped mechanism.

## Compaction rubric

```
[confirmed by operator "seems ok" 2026-08-05 — canonical copy]

KEEP IN FULL:
- Role contract: planner (QUEUE/DECISIONS/LEDGER/SPECs + scheduled GPU runs) vs dark factory
  (separate session, executes Ready rows); planner NEVER writes tasks//DECISIONS/QUEUE or commits
  while factory ARMED (run-23 quarantine precedent). Factory NOT armed at handoff.
- LIVE STATE: Ready = 876 (score emission) → 878 (failed-fill/coinflip) → 872 (live TZ) → 873
  (fixture pin) → 877 (XBT fold) → 874 (provenance), physically ordered. 879 = planner-scheduled
  re-run program (not a factory row). RUNNING: WF generalization battery on srv, tmux `wf875`,
  5 folds × 3 seeds, ~2.5h/fold (~12h total), pre-registered bar IN the script: per fold seed
  MEDIAN; ≥4/5 folds positive AND above own floor = real OOS edge · 2-3/5 inconclusive · ≤1/5 none.
  Tree clean @ 67c8176, nothing pushed.
- OPERATOR ARCHITECTURAL RULINGS (all load-bearing, correct my framing where they conflict):
  (1) FORWARD-FILL IS ARCHITECTURE (panel must be dense) and a stale price IS valid while nothing
  trades — "fabricated PnL" was my wrong frame; real defects were the ZERO price (fixed D30) and
  the identity split (fixed 868). (2) A no-print bar = order PLACED, UNFILLED, and it CAN fill on
  a later step ⇒ COINFLIP execution (pending-order queue = "heavy", out of scope); p from trailing
  print density, seeded path-independent hash or every parity gate dies; 3-arm ablation answers
  "flat-before-move = signal or noise". (3) UNSCORED WEIGHTS ARE GARBAGE — "no reason to store
  weights that i dont have info what they are". (4) RERUN EXPERIMENTS FROM BROKEN FRAMEWORKS —
  LEDGER now carries FRAMEWORK EPOCHS E-a..E-f and every pre-E-c verdict is PROVISIONAL (history,
  not "closed"). (5) The DB is DERIVED/reproducible (snapshots + rebuild scripts) and the server is
  disposable — my backup urgency was misplaced. (6) There are no "other people": ~20 AI models
  across 5 families built this repo; keep attribution accurate, the discipline exists to catch
  MODEL-generated claims including mine.
- SCIENCE RESULTS (861 chain, all ledgered): teacher walk +8.118 (behavior class confirmed, hard
  checks pass) · re-based reference median +0.9914 (spread 0.041 — old ±10 seed noise does NOT
  reproduce on clean data) · search-teacher imprint FAIL 0/3, median +0.692 ⇒ better-teacher
  hypothesis CLOSED (imitability, not quality, is what hard-target BC rewards) · OOS: oos_full
  +0.064, wf_q4 −1.767, wf_q1 −1.041 · OLD ckpt through the clean battery negative everywhere
  (regimes −0.881/−1.388/−0.410/−0.170, form FAIL 0/4) · MASK EXONERATED (mask-off vs on: 5/7
  bit-identical, max |Δ| 0.032).
- MY WITHDRAWN CLAIM (do not let it come back): I asserted the collapse proved contamination
  ("fabricated PnL, not alpha"). WITHDRAWN as unisolated and refuted by its own numbers — the old
  ckpt goes negative on 2021–2023 windows where the 2025 gap has no overlap. Cause UNISOLATED
  among alias union (adding REAL data, not contamination) / mask / UTC / frame pin; isolation legs
  B (tag task867) and C (tag task868) UNRUN. Verdict stands only as "not reproducible under
  current semantics".
- SHIPPED BY PLANNER (not factory): tools/ckpt_gc.py (score-based retention, 6 gates incl. the
  COLLAPSE case where flax keep=3 keeps the broken ckpts; unscored never score-pruned; disk floor
  protects the DB) + tools/ckpt_archive.py (download-interesting/clear-rest, never uploads,
  verifies before deleting, manifest first, LEDGER-derived protection, --unscored drop default,
  --no-keep-newest for dead runs). Executed: 28.82 GB / 29 dirs cleared from srv (136 → 163 GB
  free); all were 828s3 phase-1 (trained INSIDE the cash-wall defect), twice-failed ExIt, or smokes.
- XBT FINDING (TASK_877): resolver folds slash-only, deliberately NOT XBT→BTC; XBTCHF (id 1175)
  holds 92,639 real rows inside the exact gap where BTCCHF (frame slot 23) has 697; 48,780-row
  overlap agrees to max rel close diff 0.0. Identity repair INCOMPLETE, genuine sparse set 5 → 4.

COMPRESS + POINT:
- Factory run 24/25 wrap details + their fix inventories → task banners (866/867/868/869/870/871)
  + lore/handoff_2026-08-03_factory_run25.md + lore/handoff_2026-08-04_factory_run_25.md.
- Full scope/acceptance of 872/873/874/876/877/878/879 → their task files (878 carries the
  coinflip amendment + ablation; 879 carries the epoch table and the ranked re-run ladder).
- D30/D31 full texts, the D31 attribution correction, framework epochs → tasks/DECISIONS.md +
  the LEDGER header block.
- 861 chain execution mechanics (orchestrator script, per-ckpt provenance derivation with
  order-exact assert, srv tmux) → LEDGER rows 861/ref-rebase · 861/search-teacher-imprint ·
  861/ref-oos-rebase · 861/ref-oos-maskoff + runs/861_*.
- DB backup thread (integrity-gated copy, coverage 2013-09-10..2026-03-31, sha256 799dca96c9…)
  → memory "DB backups" line; operator ruled it low-value, do not repeat the crusade.
- CLAUDE.md lost-edit recovery (22-line context trim, snapshot 08:30:01) → commit 40b90d1.

DROP:
- ckpt_gc test-fixture debugging (tmpfs disk-floor surprise, two wrong test assumptions).
- .gitignore negation mechanics, queue row renumbering/reordering mechanics.
- Buffered-log / python2-vs-venv / killed-background-task churn.
- The 2.4 GB archive pull that was started then deleted when the unscored ruling landed.
- /usage and ping turns; the srv row-level DB sync (measured a NO-OP: 1130 assets identical,
  30-asset deep sample zero mismatches — keep the one-line conclusion only).

VERBATIM ANCHORS:
- 67c8176 (878 amendment) · 4cdacb2 (878+879 rulings) · 0072a89 (877) · 3e74a1e (ckpt mechanism +
  28.8 GB clear) · 3030da9 (attribution withdrawal) · 858fda0 (OOS re-measure) · 482364e (861
  verdict) · b20ac63 (chain launch) · 40b90d1 (CLAUDE.md restore).
- +8.118 · +0.9914 (spread 0.041) · +0.692 · +0.064 / −1.767 / −1.041 · −0.881/−1.388/−0.410/−0.170
  · max |Δ| 0.032 · 0.94% frozen · 9.74% not-real · 92,639 / 48,780 / max rel diff 0.0 · 28.82 GB
  (136→163 GB) · 5 folds × 3 seeds · E-a..E-f.
- tools/ckpt_gc.py · tools/ckpt_archive.py · runs/875_wf · tmux wf875 · asset_aliases.py ·
  XBTCHF id 1175 / BTCCHF id 1085 slot 23.

UNCERTAIN (operator: correct me):
- The WF battery is running on the PRE-877 panel (BTCCHF still sparse, 1 of 64 slots). Accept its
  verdict as-is, or re-run after the XBT fold lands?
- Should 878 (coinflip) land BEFORE the 879 re-run ladder starts? It changes what a re-run measures,
  but it also delays the base-training block you want the idle GPUs on.
- When 876 ships: do I launch 879 rung 1 (828s3 phase-1 self-play = the long base training) by
  planner default, or wait for your explicit go?
- Isolation legs B/C (tag worktrees) — still worth running now that the mask is exonerated and the
  re-run ladder supersedes those numbers anyway, or drop them?
```

## Git state
```
67c8176 TASK_878 amended: unfilled != dead — coinflip execution, data-derived p, seeded path-independent, 3-arm ablation
4cdacb2 operator rulings: no-print = FAILED FILL (878), and every pre-E-c verdict is PROVISIONAL pending re-run (879)
d5fe729 QUEUE: 877 as Ready #4 + two OPEN QUESTIONS of the forward-fill class for the operator
0072a89 TASK_877: the alias union misses XBT<->BTC — 92,639 real rows still unread (same class as D31 a1)
3e74a1e ship the checkpoint prune/archive mechanism (planner, not a factory row) + clear 28.8GB of dead-lane weights from srv
```
Status: **this handoff file is UNCOMMITTED on disk — the factory ARMED between the rubric
confirmation and the commit attempt (git gate fired, correctly: commits belong to the implementer
session while armed). First planner action after the next disarm: commit this file.** Everything
else committed and clean through `67c8176`. Nothing pushed to origin — owner pushes after
review. (Exactly the handoff-#17 precedent, and the reason the no-shared-writes-while-armed rule
exists.) gpu-server: 163 GB free, GPUs 1–3 running the WF battery under tmux
`wf875`; `trading_data.db` untouched all session (mtime 2026-07-26 20:47:32) and confirmed
row-identical to the workstation copy.

## Next action
Read the WF battery when it lands (`runs/875_wf/WF_SUMMARY.json`, tmux `wf875`, ~12 h from
2026-08-05 19:52Z): per fold take the seed median, then grade ≥4/5 positive-and-above-floor = real
OOS edge · 2–3/5 inconclusive · ≤1/5 no edge, and ledger the row either way — it replaces the n=2
quarter evidence the withdrawn +0.898 claim rested on. Then the four UNCERTAIN calls above, of
which the load-bearing one is whether 879 rung 1 (the 828s3 phase-1 self-play re-run = the long
base-training block the idle GPUs are for) launches on planner default once 876 ships. Factory is
green to arm on Ready = 876 → 878 → 872 → 873 → 877 → 874 at any time; the WF battery runs on the
deployed snapshot and does not collide with it.

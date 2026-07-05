# Handoff — 2026-05-30 — s7 hang discovered, 780 OOM closed, 775+780 launches authorized

**Sequence**: #5
**Previous**: `lore/handoff_2026-05-28_780-buffer-fix.md`
**Branch**: `master`
**Working dir**: `/home/lukas/Trader`
**Session focus**: Continuation after compact. Checked server, found 775p2 s7 hung 19h+ at E135/200 + 19h GPU 2/3 idle window (s13 done not chained, 780 preflight C passed at 21:40 yesterday). 780 OOM Q2 resolved via `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`. Sent `REPLY_2026-05-30.md` authorizing kill of PID 744270 + s23 launch on GPU 2 + 780 Phase-A 3-seed serialized on GPU 3 (all with MEM_FRACTION=0.95). Wrote `lore/notes_2026-05-29.md` §171 covering buffer-init bug + Q2.

## Confirmed rubric

```
Compaction rubric (chandoff 2026-05-30):

KEEP IN FULL:
- 780 OOM Q2 CLOSED via Option C: XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 lets n_passes=32 default stay. Preflight passed at 21:40 UTC 2026-05-29 (104 MB phase1_best.pkl). No code change.
- 775p2 s7 HUNG at E135/200 for 19h+ (PID 744270, last log 2026-05-29 22:54). Second seed-7 wedge in campaign (first was eval PID 160009 9h11m last session). Pattern worth tracking — may not be transient.
- s13 finished cleanly E199, best total=4.5496 at E93. GPU 2 idle since 23:20 UTC 2026-05-29 — s23 never chained (orch script `/tmp/775p2_orch.sh` lacks chain logic).
- REPLY_2026-05-30.md sent: authorize kill 744270, launch s23 on G2, launch 780 Phase-A 3-seed on G3 serialized with MEM_FRACTION=0.95, write STATUS_2026-05-30.md noting s7 wedge + 19h idle + 780 OOM closure.
- lore/notes_2026-05-29.md §171 written — covers buffer-init bug + fix a73d727 + 775 P1 PASS + 780 OOM + Q2 verdict. Includes Bloom-arc framing "the bug ship-in-a-day didn't catch".

COMPRESS + POINT:
- 780 buffer-init bug + 8-LOC fix a73d727 → see lore/notes_2026-05-29.md §171
- 769/779 overnight verdicts → see runs/STATUS_2026-05-28.md (server) + §171 anchors
- 775 P1 PASS details → see §171
- Tech debt + TASK_785 9-unit cleanup plan → see tasks/TASK_785_TECH_DEBT_CLEANUP.md

DROP:
- /tmp/claude-1000/response.md /copy noise.
- Orch script COMMON-args full text — already in REPLY_2026-05-30.md and lore/.
- Per-GPU HBM snapshot tables — only the wedge + idle facts matter going forward.
- "Server agent offline overnight" speculation — not actionable for compaction.

VERBATIM ANCHORS:
- Master tip: a73d727 (TASK_780 buffer-init fix; same as last handoff — no new ws commits this session).
- Server training tip: 4f33daa.
- Hung process: PID 744270 on GPU 1 (775p2 s7 E135/200, 23h+ etime, last log 2026-05-29 22:54 UTC).
- Server time of session-end check: 2026-05-30 18:33 UTC.
- 780 OOM workaround: XLA_PYTHON_CLIENT_MEM_FRACTION=0.95.
- 780 preflight C output: /tmp/780_smoke_c/780_preflight_C/phase1_best.pkl (104,558,956 B at 21:40 UTC).
- s13 verdict: best total=4.5496 at E93.

UNCERTAIN:
- Whether seed-7 wedge is reproducible enough to skip s7 entirely vs rerun. REPLY says "rerun unless evidence it'll wedge again" — server agent's call post-kill.
- Whether 780 Phase-A epoch count should match SMOKE_PROTOCOL's spec or AUTHORIZATION's — REPLY pointed to both rather than naming a number. Server agent reads protocol; ws doesn't have the spec value memorized.
- Whether STATUS_2026-05-30.md should also fold in 769 F2 validation + 779 NULL → experiments/QUEUE.md update (deferred from §171 anchors list since last handoff).
```

## Git state at handoff

```
a73d727 TASK_780 buffer-init branch: route prefill to _refill_buffer_bridge when oracle-mcts-distill
c55861c Lore + tasks cleanup: scrub impl-time bias, audit + cleanup task
5268872 Lore §170: TASK_780 build — oracle-as-MCTS distillation lands in 3 phases
006d265 TASK_780 Phase 3-5: integration tests + smoke protocol + Phase-A gate helper
148e8e9 TASK_780 Phase 2: bridge_trainer training-loop math + bridge buffer + wire
```

No new ws commits this session. Only new file created: `lore/notes_2026-05-29.md` (§171). Working tree dirty pre-session per usual `_old/` deletion drift.

Server training tip: `4f33daa` (matches `a73d727 + deploy marker`). Server has `runs/REPLY_2026-05-30.md` (4782 B) instructing kill + 2 launches.

## Next action

1. Wait for server agent to ack `REPLY_2026-05-30.md`: kill PID 744270, launch 775p2 s23 on G2, launch 780 Phase-A 3-seed on G3 serialized with `MEM_FRACTION=0.95`.
2. Server writes `STATUS_2026-05-30.md` covering s7 wedge + 19h idle + 780 OOM closure.
3. Next ws session: fold s7-wedge-pattern (twice now) into memory if it reproduces a third time, OR mark as transient if s7 reruns clean. Also fold 769 F2 + 779 NULL into `experiments/QUEUE.md` if not yet done.
4. TASK_785 units A–C remain highest-impact follow-up.

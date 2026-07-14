# Handoff 2026-07-14 (#12) — Compaction safety-net (factory runs 7–9)

Previous: `lore/handoff_2026-07-14_factory-run9-short-bias-probe.md` (#11 — read it and its Previous chain #10/#9 for the substantive per-run detail)
Branch: `master` · Working dir: `/home/lukas/Trader`
Session focus: ORCHESTRATOR ran the dark factory for three runs — run 7 (TASK_840 weight-rebalance env + TASK_841 wedge watchdog), run 8 (TASK_842 weight close-fee), run 9 (TASK_844 short-bias probe). This file carries ONLY the compaction rubric + git state; all per-run detail is in lore #9/#10/#11.

## Confirmed compaction rubric

```
KEEP IN FULL:
- LIVE STATE: factory DISARMED, tree clean, Ready queue EMPTY behind the D13 OPERATOR
  WALL. D13 = phase-1c split verdict (weight env FIXED / from-scratch policy below floor
  2/2 with systematic ~10-short bias): redirect to E5/832 imprint→ExIt vs follow the
  pre-registered D10 ladder-widening. Nothing new starts until D13 is answered or a new
  D13-independent row lands. TASK_843 gated on D13.
- ORCHESTRATOR discipline that recurred every run: capture parity pins on the CLEAN
  pre-edit tree BEFORE any canon edit; independently re-run every objective gate before
  shipping; fix passes route to the ORIGINAL implementer via SendMessage; when a finding
  is interpretive/decision-owned, ship the faithful impl + seed a planner DECISIONS row
  rather than guess past a ratified decision.
- The 4 shipped tags + one-line each + the D-rows they seeded (see ANCHORS).

COMPRESS + POINT (collapse each run to its tag-line + "N MAJOR fixed"; full detail in lore):
- Run 7 (TASK_840 weight env + TASK_841 watchdog) → lore/handoff_2026-07-11_factory-run7-weight-env-and-watchdog.md (#9)
- Run 8 (TASK_842 weight close-fee, D9 remainder) → lore/handoff_2026-07-12_factory-run8-weight-close-fee.md (#10)
- Run 9 (TASK_844 short-bias postmortem probe) → lore/handoff_2026-07-14_factory-run9-short-bias-probe.md (#11)

DROP:
- All "hold until agents settle" bash: foreground/background watchdog loops, transcript-idle
  & size-stability heuristics, stop-gate re-fire churn, "no notification yet, re-hold" cycles.
- Session-limit incident mechanics (subagents dying ~2am Prague, 0-tool-use respawn) — lesson
  already in lore #10/#11.
- The 07-12↔07-14 date-notification glitch.
- Intermediate green gate dumps (conclusions only); /usage /effort /model interstitials; MCP noise.

VERBATIM ANCHORS:
- Three canon parity pins: canon 48e1f467c4e7e37316e3555015a89ec8e266297c8515e55c66ea4b54a536d25b
  (test_vector_step_parity.py); 839-budget 4f8b692ead92248f16850f4874d9b9fca7d85fb9447fe26721442d9f164c19d3
  (test_vector_weight_parity.py); weight-ON-fee0 1c50d58c570d10cfe237e4b7c630cd01058a7b5a68766eb0001a14e301474c20
  (test_vector_weight_fee0_parity.py).
- Tags@commits: task840-weight-rebalance-step@31a545f, task841-checkpoint-wedge@52b155f,
  task842-weight-close-fee@30013a4, task844-short-bias-probe@b53ecfe.
- Open wall DECISIONS D13; ratified D9/D10/D11/D12.

UNCERTAIN:
- Whether D12 (weight close+reopen over-tax cost model) and the 844 open_count NIT still need
  a factory follow-up task, or the planner has fully absorbed them.
- Whether ultracode being ON now should change the next armed run's orchestration (larger
  panels / Workflow fan-out) — next run is the first under it.
- Whether the next session needs any panel-finding detail beyond the lore pointers + "answer D13".
```

## Git state
```
b953b04 Planner handoff #8: D13 wall, 844 postmortem running                                   [planner]
744fa47 QUEUE + handoff #11: TASK_844 shipped (factory run 9) → Ready empty behind D13 wall     [factory]
b53ecfe TASK_844: short-bias postmortem probe (read-only diagnostic instrument)   [tag: task844-short-bias-probe]
800f015 p1c s2 record folded: split verdict confirmed 3/3; battery closed clean under watchdog [planner]
f3e9b3b p1c verdicts: env FIXED / policy below floor with short bias; D13 OPEN; TASK_844 seeded [planner]
```
HEAD moved to `b953b04` (planner committed its handoff #8 on top of the factory's run-9 work; the run-7/8/9 tags below are the durable anchors). Working tree: clean. Factory: DISARMED. Nothing pushed.

## Next action
Factory idle behind the D13 operator wall. Highest-value unblock = the **D13 ruling** ("redirect" to E5/832 imprint→ExIt, or "follow D10" ladder-widening). TASK_844's scheduled postmortem run (planner) locates the short-bias locus (value/policy/credit) and should inform the D13 fork. Resume the factory with "armed" once D13 is answered or a new D13-independent Ready row lands.
```
```

# Handoff 2026-07-29 (#17, planner chain) — 861 chain twice re-framed; minute pin seeded; oracle-configuration class rule

**Previous:** `lore/handoff_2026-07-27_imprint-validated-search-parked-planner.md` (#16).
Factory-side this window: `lore/handoff_2026-07-29_factory_run21.md` (+ run-20 wrap @ ded162b).
**Branch:** master · **Working dir:** /home/lukas/Trader
**Session role:** PLANNER/orchestrator — QUEUE/DECISIONS/LEDGER/SPECs + scheduled runs; factory
executes Ready rows in its own session; git gate while armed.
**Session focus (Jul 27→29):** D26 ruled ("prepare dark factory tasks") → 861/862 seeded and
shipped (run 20); the factory's verification then caught TWO successive framing defects in the
861 real-leg pre-registration — universe drift (D28, 59/64 slots) and the daily-pin breadth/
source trap (D29, 33/64 vs minute 62/64) — each ruled planner-side opt (a); the chain is now
minute-pin-gated on TASK_865 with a 3-seed median bar. Runs 20+21 also produced the
oracle-configuration class rule (866/867) and two owned corrections (imprint.py WARN claim;
864's stale-leaves misdiagnosis).

## Compaction rubric

```
[confirmed by operator "ok" 2026-07-29 — canonical copy]

KEEP IN FULL: role contract; LIVE STATE (factory GREEN to arm — Ready = 865 → 866 → 867 STRICTLY
SERIAL, shared surface experiments/832_linear_imprint/; row D ExIt-frame-pin PARKED until an ExIt
continuation is pre-registered; NO open decisions — D27/D28/D29 planner-ruled opt (a), operator
may override; gpu-server idle; nothing pushed; tree clean @ 3413534); THE 861 CHAIN (factory-half
SHIPPED; real legs re-registered TWICE, final = minute-pinned frame_provenance_imprint.json BOTH
legs, gated on 865; leg 2/3 = 3 imprint seeds, MEDIAN > +0.839 with ≥2/3 above — single-seed
uninterpretable, sanity-replay noise −8.93…+12.74/10 seeds; leg 1 "+8.02" SOFT behavior-class
check; leg 4 OOS/WF same frame; frame facts: unpinned 59/64 drift · daily 33/64 (D20 forbidden +
ST3 2.76× breadth) · minute 62/64, 2 inactive = FTM→S / MATIC→POL; teacher operator pinned
verbatim in the task file; imprint.py:620 "only WARNs" = FACTUAL ERROR, hard RuntimeError at
627–631, corrected at both authored sites); science anchors unchanged (imprint +0.839 IS /
+0.898 OOS Δ+2.33 / WF +3.561 & +0.611; teacher +8.024; floor +0.424; search-leaf lane PARKED
857 −4.23 / 860 −4.02 structural); verification meta-lesson driving 866/867 (every MAJOR =
"green gate on wrong work"; 78.8% = 315/400 cells admit trajectory-inert BC-target rewrites, all
gates green; class rule = every committed baseline needs an oracle of ITS OWN configuration —
CANON_832_PINS weight-OFF guarded a weight-ON baseline, 846 drifted silently ~80 commits).

COMPRESS + POINT: runs 20/21 findings/fixes → task banners 861/862/863/864 + QUEUE shipped
blocks + lore/handoff_2026-07-29_factory_run21.md + .dark-factory/verdicts/; D27/D28/D29 full
texts → tasks/DECISIONS.md (D29 row carries the framing argument); 865/866/867 scope → their
task files (865 carries the re-registered real leg); data/ops (858 ingest, D25 quarantine, srv
sync, 160G caps, Q2-2026 dump watch) → handoff #16 + memory (unchanged); 864 misdiagnosis
correction (~461/468 leaves = weight-ON flag; real drift ~7 fields from c2f1ba5/846) → TASK_864
banner + CHANGELOG.

DROP: mid-run factory status pastes + run-20 pre-wrap; DECISIONS row-surgery mechanics; /usage +
ping turns; the run-19 duplicate wrap-paste at session start.

VERBATIM ANCHORS: 861@9077528 · 862@7303e74 · 863@609d6d3 · 864@6b71e0e+086b1c4 · ba6dfa9
(D27+D28) · 3413534 (D29) · wraps ded162b, 1676d11 · 59/64 · 33/64 · 62/64 · +0.839 · +8.024 ·
+0.898 (Δ+2.33) · WF +3.561/+0.611 · floor +0.424 · −8.93…+12.74 · ST3 2.76× · 78.8% (315/400) ·
imprint.py 627–631 RuntimeError.

UNCERTAIN: D27/D28/D29 stand as planner rulings with override right — re-open any?; when 865
ships, launch the 861 real legs by planner default (srv, 3-seed battery) or wait for explicit
go?; keep 866/867 queued behind 865, or clear them to single-track the critical path?
```

## Git state
```
3413534 D29 planner-RULED opt (a): seed 865 (minute frame pin) + 866/867 (pin+promote hygiene) — factory GREEN, serial
1676d11 factory run 21 wrap: queue groomed (Ready EMPTY, 4 recommended rows) + dated handoff
086b1c4 task864 scope 2: refresh the weight-ON parity baseline — every delta attributed to c2f1ba5/TASK_846
6b71e0e task864-parity-baseline-hygiene: tracked baseline unwritable by default; promote recipe- and validity-gated
609d6d3 task863-imprint-frame-pin: 853 frame pin ported to imprint.py; both-directions fail-closed provenance gate
```
Status: this handoff file is UNCOMMITTED on disk — the factory ARMED between the file write and
the commit attempt (git gate fired, correctly). First planner action after the next disarm:
commit this file. Nothing pushed to origin. Tags through
task864-parity-baseline-hygiene. gpu-server idle, row-synced to Q1-2026.

## Next action
Factory arms on Ready = 865 → 866 → 867 (SERIAL). When 865 ships + verifies: planner launches
the 861 real-leg chain per the D29-amended registration in TASK_861 — minute-pinned teacher pass
(workstation or srv CPU-heavy true-env walk) → 3-seed imprint battery on gpu-server (160G caps,
--stall-timeout 2700, verified-idle GPUs) → leg-3 MEDIAN read vs +0.839 → OOS/WF only if it
clears. Per D17 everything is benchmark reads, no gate status. Q2-2026 Kraken dump watch stands
for the next OOS extension.

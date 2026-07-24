# Handoff 2026-07-23 (#15, planner chain) — E5 diagnosis chain closed; 856/857 seeded

**Previous:** `lore/handoff_2026-07-16_e5-gate-read-planner.md` (#14). Factory-side handoffs
(runs 11–16) live separately: `lore/handoff_2026-07-18_factory-run-12.md`,
`lore/factory_run16_handoff_2026-07-23.md`, et al.
**Branch:** master · **Working dir:** /home/lukas/Trader
**Session role:** PLANNER/orchestrator — queue/DECISIONS/SPECs + scheduled server runs; factory
(separate session) executes Ready rows; git gate while armed.
**Session focus (Jul 16→23):** the E5 arc end-to-end — gate-read → imprint (+0.839, first
above-floor learned policy) → ExIt destroyed it → three-step operator diagnosis (proposals →
leaf → selection prior), each step measured, fixed by 847/851/854 → fixed operator +8.02 on the
frozen imprint → ExIt re-try under the fixed operator FAILED at a new locus (argmax
distillation) → D23 ruled: 856 (decision-rule probe + grading fix) + 857 (v̂-leaf distill, the
deployable-search path). OOS track: blocked on data (D22, minute backfill, OPEN, non-blocking).

## Compaction rubric

```
Compaction rubric (chandoff 2026-07-23):

KEEP IN FULL:
- Role contract: this session = PLANNER (queue/DECISIONS/SPECs/scheduled server runs); factory
  = separate session on Ready rows; planner never commits while factory ARMED (git gate).
- LIVE STATE: factory GREEN, Ready = TASK_856 (decision-rule probe + net_dependent grading
  fix, XS/S) → TASK_857 (counterfactual value imprint → v̂-leaf search, M) → TASK_855 (XS
  filler). ONE OPEN decision: D22 (minute backfill for the OOS benchmark — ops call,
  non-blocking). Server GPUs idle; nothing running anywhere.
- THE SCIENCE (load-bearing): (1) imprint ckpt runs/832_e5/ckpt_imprint_full replays +0.839
  net@21 (first above-floor learned policy; floor +0.424, ref +0.751). (2) The search operator
  was fixed in three MEASURED steps — policy-blind proposals (847), action-blind value leaf
  (851, v̂ corr +0.03), prior term in Gumbel selection (854) — frozen-imprint search-eval went
  −6.92 → +8.02 (sweep winner: dithered-book K16, hb3.0, --leaf-eval true-env
  --no-leaf-vhat-tail --prior-weight 0.0, R=3; monotone in prior-weight; E1 anchor +5.29).
  (3) BUT the true-env leaf is R-step LOOKAHEAD — teacher/ceiling only, never deployable.
  (4) ExIt2 re-try under the fixed operator FAILED at ARGMAX DISTILLATION: policy-argmax 100%
  HOLD from ep50 all seeds, CE flat ~0.29 (info in the DISTRIBUTION, argmax kills it); search
  grading leg was VACUOUS (net-free operator returns bit-identical numbers for any ckpt —
  planner miss, owned; fix = 856's net_dependent flag).
- Post-ship planner runs (pre-registered in the task files): 856 → rule-grid sweep on
  jax_ckpt_fast/e5exit2_pq_s{0,1,2}/snapshot_ep300 vs floor +0.424; 857 → real calibration
  (v̂-vs-true across candidates, bar ≥ +0.5 from +0.03) + the first DEPLOYABLE-operator
  corridor read (v̂-leaf search; ceiling +8.02).
- OOS track state: blocked on DATA not instruments — daily table covers 27/64 in-sample
  (853 round-trip MISS ⇒ daily-source OOS unattributable; the +1.254 read is recorded
  UNATTRIBUTABLE), minute table dies 2025-10 (thin n=31 read = only honest OOS datum).
  D22 recommendation = minute backfill via the existing fetcher.
- Ops invariants: fresh server tmux needs `source trader_env/bin/activate`; training arms
  --stall-timeout 2700 (D11), evals unarmed; wedge = XLA post-save hang, watchdog+auto-retry
  is the mitigation; deploy planner-owned, never mid-battery; storage disk mount holds
  .git/objects (udisksctl mount -b /dev/sdb1 if git breaks); server DB STALE (daily ends
  2025-10-04) — sync-db.sh before any server run touching post-2025-10 data.

COMPRESS + POINT:
- Factory runs 11–16 mechanics/panels → tasks/QUEUE.md shipped-log + the factory lore handoffs.
- Every verdict of the arc → experiments/LEDGER.md rows: 832/E5-{gate,imprint,exit,erosion,
  autopsy,inversion,inversion2,inversion3,exit2,oos}, 849/leaf, 850/eqgate, 852/reconcile,
  853/numeric-leg.
- The isolation-ladder variant table + operator sweep → LEDGER 832/E5-inversion{2,3} +
  experiments/828_stage3_train/SPEC.md Amendment 6 + addendum.
- D14–D23 rulings verbatim → tasks/DECISIONS.md.

DROP:
- Watcher/scp/tmux construction mechanics and their false trips (the *nt*-matches-"imprint"
  glob trap, banner-matching stall pattern), session-restart recoveries, /usage noise,
  DECISIONS formatting-glitch fixes, GPU idle checks.

VERBATIM ANCHORS:
- Numbers: imprint +0.839 · floor +0.424 · ref +0.751 · E1 +5.29 · fixed operator +8.024
  (bar +0.74) · old operator −6.92 · v̂ calibration +0.03 → 857 bar ≥ +0.5 · ExIt2 CE ~0.29.
- Winner operator flags (train=grade, one operator): --vector-proposal-scheme dithered-book
  --vector-k 16 --vector-hold-bias 3.0 --vector-dither-scale 1.0 --leaf-eval true-env
  --no-leaf-vhat-tail --prior-weight 0.0 (R=3 default).
- Artifacts: runs/832_e5/ckpt_imprint_full (srv + local) · jax_ckpt_fast/e5exit2_pq_s{0,1,2}/
  snapshot_ep{50,300} (srv) · runs/832_e5/frame_provenance_imprint.json (local).
- Tags@commits: 846@c2f1ba5 · 847@873effe · 848@371e7c5 · 849@d836d46 · 850@97a2710 ·
  851@72df551 · 853@c8ee701 · 854@f5807f6; HEAD 2384d63; srv training branch at deploy 9853496.

UNCERTAIN (operator: correct me):
- On 856/857 ship, do the planner post-ship runs launch automatically (my default: yes, per
  standing server-never-idles autonomy) or wait for your go?
- D22 (minute backfill): parked as OPEN ops call — fine to leave parked, or schedule the
  fetcher run?
- The 100%-hold ep300 ckpts: keep for 856's probe (assumed yes) — never warm-start from them.
```

## Git state
```
2384d63 D23 RULED (operator): 856 (decision-rule probe + grading fix) and 857 (v-hat-leaf distill, deployable-search path) seeded to Ready
5e05d5f 832/E5-exit2 FAIL: argmax distillation is the new locus (100% hold from ep50, CE flat; search leg vacuous — grading miss owned); D23 seeded
153c442 832/E5-inversion3: PASS +8.02 (bar +0.74) — isolation confirmed, search now adds value over the prior; sweep then ExIt re-arm
ebff2f5 Amendment 6 addendum: sweep table, ExIt operator chosen (pw0.0/R3/hb3.0 true-env no-tail), battery pre-registered
b687a68 TASK_855 (flag-hygiene MINORs) seeded as filler; Ready row swapped post-854 ship
```
Status: clean. Nothing pushed to origin (owner pushes after review); srv training branch at
deploy 9853496.

## Next action
Operator says "armed" → factory takes 856 then 857 (855 filler). On 856 ship: planner runs the
rule-grid sweep on the ep300 ckpts (floor bar +0.424) → LEDGER. On 857 ship: real calibration
read (bar ≥ +0.5) → v̂-leaf corridor read (floor +0.424 bar, +8.02 ceiling reference) → LEDGER —
the program's first deployable-operator number. D22 stays parked unless the operator schedules
the backfill.

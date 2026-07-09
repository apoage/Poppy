# Handoff 2026-07-09 — phase-1 battery running, all instruments shipped (planner #7)

**Previous:** `lore/handoff_2026-07-09_factory-run5-instruments2.md` (factory #6; planner chain:
`lore/handoff_2026-07-08_factory-line-ready.md`)
**Branch:** master · **Working dir:** /home/lukas/Trader
**Session role:** PLANNER/orchestrator — queue/decisions/specs + scheduled server runs; the
factory (separate session) executes queue rows; lane guard + git gate enforce while armed.
**Session focus:** pathway execution day — R0/R1 official, Stage-3 spine + 6 instruments shipped
via factory runs 2–5, fee16-from-scratch collapse finding → curriculum redesign, phase-1 cost0
battery launched on gpu-server.

## Compaction rubric

```
Compaction rubric (chandoff 2026-07-09):

KEEP IN FULL:
- Role contract: this session = PLANNER/orchestrator (queue, DECISIONS, SPECs, scheduled server
  runs); factory = separate session executing queue rows; lane guard + git gate block planner
  writes to code lanes while the factory is ARMED.
- LIVE STATE: Stage-3 phase-1 battery RUNNING on gpu-server (tmux q_828s3, run_queue.sh, GPUs
  1/2/3): cost0 ×3 seeds from scratch, dithered-book|K16 with hb0.0/d1.0, 2000 epochs,
  snapshots every 100 → jax_ckpt_fast/828s3p1_cost0_s{0,1,2}. At last check ~ep1500, ~2h left,
  ~18s/ep. Watcher = background task bemf7vs64 (25-min cadence, completion OR tripwire pct_hold
  >99 & ent <0.02 on ≥2 seeds past ep300). Workstation may hibernate — watcher freezes/resumes.
- NEXT ACTION on battery end: instrument passes on server snapshots WITH THE PHASE-1 FLAGS
  (hb 0.0, dither 1.0, fee0/slip0): 835 eval (corridor grade; gross = trained objective, net@21
  = phase-2 preview), 834 E3/DEC-1 (R_model ≥ 0.5×R_true; its hb-0.0 "trap" warning is EXPECTED
  noise here), 836 E4 rank probe (diagnostic only). Then 832 imprint-sanity run (~1h). Then
  phase-2 fee16 CONTINUATION manifests: (a) from cost0 ckpts, (b) from 832 imprint. LEDGER line
  per verdict.
- Collapse finding + redesign (LEDGER 828s3/preflight): fee16-from-scratch collapses regardless
  of proposal bias (fees punish exploration before the value learns; hb3.0 is perfect-value-only
  — does not transfer to learning). Curriculum = cost0 → fee16 continuation + imprint warm-start.
  Proposal-bias-annealing contingency DELIBERATELY not seeded; activates only if phase 2 also
  collapses (SPEC amendment 2).
- Strategic fact (837 fold corridors): the linear rule's ref is NEGATIVE on unseen-years folds
  (2024 −0.226, 2025 −0.000) — the +0.751/+0.97 bar is in-sample; the rule decayed OOS; the
  fallback-shippable-rule story is weak; unseen-years graduation bar ≈ beat ~0 + be positive.
  Also: build_pit_universe verified NON-PIT; 837 derives PIT universes module-side.
- Instrument fleet all SHIPPED+panel-verified, all take operator flags as CLI: 834 E3 runner,
  835 Stage-4 eval, 832 imprint (ExIt = --load-checkpoint --reset-optimizer --clear-buffer;
  verify "Restored entropy_coef: 0.01" at ExIt launch), 836 E4 rank probe, 837 walk-forward
  rails, 838 per-node resample (default OFF; ON measured WORSE under uninformed policy, Δ−2.79
  — re-measure only with a trained policy).
- Canonical numbers: corridor net@21 [floor +0.424, ceil +29.698], engine ref +0.751 (in-sample
  full window 2021-01-01..2025-09-01); E1 real PASS 9/9; carried scheme dithered-book|K16
  (K>32 net-negative = turnover tax; value-horizon bar low: R=1 beats ref).

COMPRESS + POINT:
- Factory runs 2–5 mechanics/panel details → tasks/QUEUE.md shipped-log + task-file banners.
- 831-EXT battery (10-seed grid, regime windows, R-sweep, K-ext) → experiments/LEDGER.md 831x/A–D.
- Server ops saga (db parity, run_queue repair/redeploy, vintage correction) → QUEUE.md
  server-status block + TASK_833 banner.
- Rung-ladder rationale/history → docs/pathway.md.

DROP:
- ssh/tmux/screen quoting mechanics, poller/watcher construction, per-arm CSV extraction
  scripts, deploy push logs, workflow prompts/schemas, /usage-/model-command noise.

VERBATIM ANCHORS:
- Phase-1 job names 828s3p1_cost0_s{0,1,2}; manifest experiments/828_stage3_train/
  jobs_phase1.manifest; SPEC amendments 1+2 in experiments/828_stage3_train/SPEC.md.
- DEC-1 = R_model ≥ 0.5×R_true (fallback abs ≥0.5); DEC-2 = ε +0.10 net@21 + consistency leg
  (≥3/4 regime windows ≥ own floor); zero OPEN decision rows.
- 837 folds: f1 train 2021-23 eval 2024 corridor [+0.057, +36.774] ref −0.226; f2 train
  2021-24 eval 2025-09 corridor [−0.291, +35.410] ref −0.000.
- Watcher task bemf7vs64; server tmux q_828s3; HEAD 54c86d0.

UNCERTAIN (operator: correct me):
- Phase-1 self-play telemetry (hold 96–98%, ent ~0.2, cost0 rewards NEGATIVE, ma20 −13..−26) —
  read as worrying-but-not-verdict; the 835/834/836 passes decide. Confirm no early kill wanted.
- 832 imprint-sanity run folded into the post-battery server batch — confirm ordering (or run
  it DURING the battery on a free CPU? GPUs are occupied).
```

## Git state
```
54c86d0 Server db vintage corrected: full parity verified, minute jobs cleared
ebff440 Factory run 5 wrap-up: queue regroom (Ready empty, 836/837/838 shipped) + handoff #6
39ea3de TASK_838: per-node action resampling in the sampled vector search, flag-gated default-OFF
2396c55 TASK_837: walk-forward split harness (R4 graduation instrument), zero canon diffs
2b235ae TASK_836: E4 value-ranking probe (value net vs momentum, per window), zero canon diffs
```
Status: clean. Nothing pushed (owner pushes after review).

## Next action
Wait for watcher bemf7vs64 (battery end or tripwire) → run 835/834/836 on the server snapshots
with the phase-1 flags → LEDGER verdicts → 832 imprint-sanity → phase-2 fee16-continuation
manifests. Factory Ready is empty; new rows come from those verdicts.

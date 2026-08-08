# Handoff 2026-08-07 (#19, planner) — three pre-registered verdicts land, and the last one puts the measuring instruments themselves in doubt

**Previous:** `lore/handoff_2026-08-05_operator-rulings-rerun-doctrine-planner.md` (#18).
Factory-side this window: `lore/handoff_2026-08-06_factory_run_26.md` +
`lore/handoff_2026-08-07_factory_run_27.md`.
**Branch:** master · **Working dir:** /home/lukas/Trader
**Session role:** PLANNER/orchestrator — QUEUE/DECISIONS/LEDGER/SPECs + planner-scheduled GPU
runs; the factory executes Ready rows in its own session; no planner writes to shared surfaces
while it is armed (practised twice this window, worked both times).
**Session focus (Aug 5→7):** read the 875 walk-forward battery to a verdict → ratified factory
runs 26 and 27 and ruled D32/D33/D34 → ran the 879 rung-3 re-measure, which voided two ledger
verdicts, showed the search stack is subtractive, and ended by catching two instruments
disagreeing about the same weights. Net effect: the imprint lane's OOS question is settled
(negative), the distill lane's parking reason is withdrawn, and the program cannot compare
numbers across harnesses until TASK_882 lands.

## Compaction rubric

```
[confirmed by operator "ok" 2026-08-07 — canonical copy]

KEEP IN FULL:
- Role contract: planner (QUEUE/DECISIONS/LEDGER/SPECs + planner-scheduled GPU runs) vs dark
  factory (own session, executes Ready rows). Planner NEVER writes tasks//DECISIONS/QUEUE/LEDGER
  nor commits while the factory is ARMED — stage in scratchpad, land at disarm. Practised twice
  this window and it worked both times.
- LIVE STATE: Ready = 876 (re-rescoped: emission + flax_keep=3/overwrite=False + recency default,
  ZERO deleter surface) → 882 (NEW, instrument reconciliation) → 873 → 877 → 874 → 878. GATED:
  880 behind DEPLOYED emission (real scores.jsonl on gpu-server, not "876 shipped"), 881
  (protection model, now also owns the followlinks=False BLOCKER + the disputed
  _order_newest_first). Three stashes: {0}=876 verified, {1}=880 cycles 1-3, {2}=run-26 876.
  Tree clean @ 19ad94b, nothing pushed. srv GPUs IDLE, 162 GB free. Rung-1 SPEC drafted in
  scratchpad (RUNG1_DRAFT_SPEC.md), gated on 876 emission.
- THE THREE SCIENCE VERDICTS OF THIS WINDOW (all ledgered, all pre-registered before launch):
  (1) 875/wf-generalization — imprint lane has NO OOS EDGE, 1/5 folds, the pre-registered "none"
  band; fold medians +1.4957/+0.3995/+0.3108/+0.4121/−2.9566 vs floors
  +3.8987/+1.4301/+1.3952/+0.2635/+0.4244; the OOS median is FLAT while the floor swings 15×, so
  it clears only when the baseline has a bad half-year; fold 5 = 33% drawdown on the half-year
  closest to deployment. (2) 879/rung3 — the 857/860 distill headlines (−4.23/−4.02) are BANKRUPT
  RUNS, dead at step 27 of ~1670 ("best" = survived longest); corrected medians −2.84/−2.73 vs
  floor +0.5164, and 6/6 distilled seeds BELOW the un-distilled baseline. CONTROL: the SEARCH
  STACK IS SUBTRACTIVE — argmax −0.564 vs search+v̂-leaf −2.166 on identical weights, Δ1.602 >
  the 1.0 bar, behaviour flipping 5.4% → 94.1% hold. (3) 882/instrument-disagreement — imprint.py
  sanity +0.8394 / PV 299,547 vs eval_vector policy-argmax −0.6145 / PV 41,919, SAME weights and
  window, window eliminated by measurement. One of two load-bearing reference families is wrong.
- WHAT 882 PUTS IN DOUBT, precisely: +0.9914 (the in-window reference, LEDGER 861/ref-rebase) is
  an imprint.py number; the 875 WF battery and the 879 corridor are eval_vector numbers. The 875
  VERDICT STANDS (internally consistent — every fold vs a floor from the same run). The
  "+0.9914 in-window, no OOS edge" FRAMING does NOT — it straddles two harnesses, and on
  eval_vector the same ckpt reads −0.61 in-window. Do not compare across harnesses until 882.
- FOUR PLANNER ERRORS, owned and load-bearing (the pattern matters more than any one):
  (a) 872's queue ranking claimed "the only defect result-bearing on the LIVE path" — both
  consumers are dead code. (b) 876 sized "S" with irreversible-deletion semantics. (c) D32's
  rescope put _is_protected + TTL re-scan (= the protection model) into the row I declared free
  of the protection model → three cycles died there. (d) The queue never encoded that 880's
  safety rests on an artifact 876 writes ON THE HOST. Three of four are boundary/decomposition
  errors, not implementation errors.
- DOCTRINE FINDING (transferable, mine): a VERBATIM re-run does NOT deliver "the fixed framework"
  when a fix is FLAG-GATED. E-b/E-c/E-d are unconditional semantics; E-a is a flag
  (--vector-weight-rebalance), so re-running the recorded command faithfully re-runs the BROKEN
  env — which is exactly what happened. Every rung must diff its command against the flags each
  epoch introduced, and record that diff, before it runs.
- THE READER-DEFECT CLASS, three instances in one window: a summary block whose missing-key
  default wrote WF_SUMMARY.json as `[]` (a silent fake-null for a battery that returned a
  decisive verdict); --unscored drop reading "nobody wrote the file" as "measured and worthless"
  (one command from deleting 27 of 28 srv checkpoints incl. the 0.9914 anchor); and the factory's
  own "a guard that normalizes looks identical to a guard that corrects". Any reader or guard
  added anywhere must state which of the two it is, and fail loud on a missing key.

COMPRESS + POINT:
- Factory runs 26 + 27 blow-by-blow → lore/handoff_2026-08-06_factory_run_26.md +
  lore/handoff_2026-08-07_factory_run_27.md + the task banners.
- D32/D33/D34 full texts and their evidence → tasks/DECISIONS.md (planner-rulings table).
- 879 rung-3 execution mechanics (manifests, run_queue, the two launch failures) →
  experiments/879_rerun/SPEC.md + amendments 1-2.
- Per-fold WF numbers and the grader → LEDGER 875/wf-generalization + runs/875_wf/grade_correct.py
  + NOTE_summary_key_mismatch.md (on srv).
- Checkpoint-lifecycle history (ckpt_gc/ckpt_archive, the 28.82 GB clear) → tools/ + TASK_876/880/881.

DROP:
- The srv python2-vs-venv trap and the acquire-timeout mis-set (both now queued as an XS runner
  fix; keep the one-line lesson, drop the debugging).
- My self-inflicted `pkill -f r879_` that killed its own ssh session.
- Monitor/watcher plumbing, backlog-replay notifications, GPU-idle polling.
- Queue row renumbering mechanics and the DECISIONS row-move scripts.

VERBATIM ANCHORS:
- 19ad94b · e6e4829 · 702b2e8 · 75f7653 · 43ccf1d · 3cb0486 (handoff #18) · b695487 (srv deploy).
- +0.8394 / PV 299,547 · −0.6145 / PV 41,919 · +0.9914 / PV 465,545 · −0.564 vs −2.166 (Δ1.602)
  · −2.843 / −2.734 vs floor +0.5164 · 1/5 folds · −2.9566 vs +0.4244 · ARCHIVE 1 / DROP 27
  (2.51 GB) · 28 of 28 ckpts, 0 scores.jsonl on srv · eval_vector.py:2857 ·
  position_size_pct 0.0078125.
- stash@{0}/{1}/{2} — do not drop any of them.

UNCERTAIN (operator: correct me):
- Does rung 1 (the long base training) launch before 882 resolves? Its primary bar is SURVIVAL
  (S1-S4, instrument-independent) but its score read is not — I lean launch, with the score read
  explicitly deferred.
- Should +0.9914 be marked PROVISIONAL in the LEDGER right now, ahead of 882? It is the imprint
  lane's headline and it is currently one of two contradictory numbers.
- If 882 rules imprint.py canonical, does the 875 WF verdict need re-running on that harness? It
  is internally consistent either way, but "no OOS edge" would then rest on the losing instrument.
- Does "the search stack is subtractive" re-open D26 (search-leaf lane parked), or does it
  redirect the whole search line — i.e. is the next question "fix the search" rather than
  "distil a better leaf"?
```

## Git state
```
19ad94b D33+D34 RULED; 879 rung-3 lands: distill verdicts VOID, search stack subtractive, instruments disagree
e6e4829 run 27: 876 and 880 both HALTED on the 3-cycle rule — D33 + D34 seeded, no code shipped
702b2e8 879 rung 3 VOID as designed — and the 857/860 verdicts rest on BANKRUPT runs
75f7653 879 rung 3: pre-register + launch the distill re-measure on the fixed framework
43ccf1d D32 RULED (split + explicit fail-loud protection, destructive default OFF); 875 WF verdict: NO OOS EDGE 1/5
```
Status: **this handoff file is UNCOMMITTED on disk — the factory ARMED between the rubric
confirmation and the commit attempt, and the git gate refused, correctly (commits belong to the
implementer session while armed). First planner action after the next disarm: commit this file.**
This is the second consecutive handoff to land this way (#18 did too), which is itself the
argument for the no-shared-writes-while-armed rule rather than a nuisance. Everything else is
committed and clean at `19ad94b`, nothing pushed — the owner pushes after review. Three stashes are
load-bearing and none may be dropped: `stash@{0}` (876, all three rescoped items verified),
`stash@{1}` (880 cycles 1-3, 33 tests), `stash@{2}` (the run-26 876 artifact).
`trading_data.db` and the fixture (`b0f8de8b…`) are byte-untouched across both factory runs and
every planner leg. gpu-server: `training @ b695487`, GPUs 1-3 IDLE, 162 GB free, no tmux jobs
running. The `--apply` freeze on `tools/ckpt_archive.py` STANDS.

## Next action
Factory is GREEN to arm on Ready = 876 → 882 → 873 → 877 → 874 → 878. Planner side, in order:
(1) settle the four UNCERTAIN calls with the operator — the load-bearing one is whether rung 1
launches before TASK_882 resolves the instrument question; (2) when 876 ships, DEPLOY it to
gpu-server and confirm real `scores.jsonl` files appear there, because that — not the merge — is
what un-gates TASK_880 and what makes a long run's checkpoints worth keeping; (3) launch rung 1
from `scratchpad/RUNG1_DRAFT_SPEC.md` (40-epoch smoke on one GPU first to measure epoch
wall-time, then the 3-seed battery), grading SURVIVAL first (S1-S4) and deferring the score read
until 882 lands; (4) pull the score-bearing anchors only (~0.4 GB: `runs/861_ref/ckpt_ref_s*`,
`runs/832_e5/ckpt_imprint_full`) via `--unscored archive`, never the 2.61 GB bulk.

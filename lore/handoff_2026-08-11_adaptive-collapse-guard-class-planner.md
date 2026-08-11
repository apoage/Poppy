# Handoff 2026-08-11 (#20, planner) — the collapse was adaptive, and three runs of guards taught one law

**Previous:** `lore/handoff_2026-08-07_instrument-disagreement-subtractive-search-planner.md` (#19).
Factory-side this window: `lore/handoff_2026-08-08_factory_run_28.md` ·
`…_run_29.md` (2026-08-08) · `…_run_30.md` (2026-08-09) · `…_run_31.md` (2026-08-11).
**Branch:** master · **Working dir:** /home/lukas/Trader
**Session role:** PLANNER/orchestrator — QUEUE/DECISIONS/LEDGER/SPECs + planner-scheduled GPU runs;
the factory executes Ready rows in its own session.
**Session focus (Aug 8→11):** ran the decisive 882 leg and restored `+0.9914` as an
instrument-independent number → ruled D35/D36/D37/D38 across four factory runs → and then took the
idle server through a three-step experiment chain (884 smoke → 885 screen → 887 confirmation) that
ended by inverting the premise the whole self-play lane rested on.

## The one thing to carry forward

**The hold collapse was ADAPTIVE, not pathological.** The epsilon target-mix genuinely prevents it —
confirmed at 3 seeds and 300 epochs, at a stable fixed point — and preventing it makes the policy
*lose more*, monotonically in activity, at zero costs. The policy was not failing to explore. It had
correctly concluded that its own actions have negative expected value, and every mechanism that
makes it act more makes it lose more. Read with `879`'s subtractive search and `875`'s absent OOS
edge, all three point at the same thing underneath: a policy with no edge to protect.

## Compaction rubric

```
[confirmed by operator "ok" 2026-08-11 — canonical copy]

KEEP IN FULL:
- Role contract: planner (QUEUE/DECISIONS/LEDGER/SPECs + planner-scheduled GPU runs) vs dark
  factory (own session, executes Ready rows). No planner writes to tasks//DECISIONS/QUEUE/LEDGER
  and no commits while armed. **Operator's contract clarification (2026-08-08): "arming" is a
  CONDITIONAL note — they wait for the turn to end, so the tree IS available during the turn; land
  shared-surface writes before finishing rather than deferring to a later disarm.** And:
  `.dark-factory/state.json`'s `armed: false` is NOT trustworthy — background judges disarm to
  escape the stop gate. The operator's explicit statement is authoritative; state.json is
  corroboration only, never the basis for a write.
- THE RESULT THAT REFRAMES THE PROGRAM — LEDGER `887/collapse-confirm`. C1 PASS 3/3 and 3/3,
  C2 PASS: the TASK_812 epsilon target-mix genuinely prevents the hold collapse, at a stable fixed
  point (D holds 91.234/91.412/91.357, seed spread 0.18pp, vs A 99.072/98.976/98.756; final-50
  slopes −0.0189/+0.0031/−0.0040). **But C3 — pre-registered as reported-never-a-gate — is the
  finding: median return A −2.86% · E −9.11% · D −13.84%, perfectly MONOTONE IN ACTIVITY, at ZERO
  costs.** The collapse was ADAPTIVE: the policy correctly concluded its own actions have negative
  EV, and every mechanism making it act more makes it lose more. 885 stands as a mechanism result,
  re-read as a symptom. Distillation loop real (885) + search subtractive (879 control) + imprint
  no OOS edge (875) — underneath all three, a policy with no edge to protect.
- `+0.9914` RESTORED and instrument-independent (LEDGER `882/decisive-legA`): pinned eval_vector
  +0.9915919 vs imprint.py +0.9913783, Δ 0.000214, PV within 0.060%, residual = exactly 3 of
  107,072 decision cells (float32 argmax tie-breaks). 3 seeds +0.99159/+0.99355/+0.95212. Also
  exonerates it of D37 by construction — random-init cannot agree to four decimals.
- THE GUARD CLASS, three consecutive runs, now a law: run 28 "a guard that normalizes looks
  identical to one that corrects" → run 29 "a guard that asserts looks identical to one that
  verifies" → run 30 "A GUARD MUST JUDGE IDENTITY BY THE SAME FUNCTION THE LOADERS USE." Every one
  read correct from the writing context and wrong from the reading context.
- THE CUT THAT WORKS: separate by CAPACITY, not by lane. D33 "zero deleter surface" shipped 876 on
  attempt 3; D36 "ship what can only refuse or disclaim, never bless" shipped 883R first try; D38
  splits 886's verified contract from its load path. Corollary: `universe_pinned: true` exists
  NOWHERE in the codebase — 886 earns it back.
- SIX PLANNER ERRORS, ONE FAMILY (872 ranking · 876 sizing · D32 rescope · D34 gate · D34 ordering
  · 883 sizing "XS/S arg validation" for a universe-identity contract across 2 pinning mechanisms
  and 3 checkpoint producers). Always a mis-drawn boundary, always found by a factory cycle dying
  on it.
- LIVE STATE: Ready 888 → 886 (contract only) → 878 → 881; Gated 880. **888 is rank 1 on
  consequence: real_trade_jax.py:145 prints "loaded successfully" on a bad path and trades
  random-init weights; dry-run-only is a policy, not a mechanism.** Second: train_jax_fast.py:544 —
  a silent resume-discard is invisible because a fresh run looks like a fresh run. Stashes:
  {0}=886, {1}=883R-superseded, then 876-retained/880 — resume by MESSAGE, never index.
  `manifest_884_battery.txt` staged and deliberately UNLAUNCHED.
- D37's unfalsifiability: 367 results JSONs carry a `checkpoint` field, NOT ONE records whether it
  loaded. No historical number can be EXONERATED. The fix is backfill-shaped — 7 checkpoints
  survive locally, every load-bearing one.
- Provenance is now TRACKED (.gitignore → `runs/*/frame_provenance*.json`). Two near-destructions
  in three runs; sha256 sidecars share the disk with what they check, version control does not.

COMPRESS + POINT:
- Factory runs 28-31 → lore/handoff_2026-08-0{8,8,9,11}_factory_run_{28,29,30,31}.md.
- D35-D38 full texts + preserved option lists → tasks/DECISIONS.md (planner-rulings table).
- 884/885/887 execution mechanics → experiments/88{4,5,7}_*/SPEC.md + manifests.
- 877's recovery numbers (667→92,637 · 182→0 · 5→4) → the TASK_877 banner.

DROP:
- The 3-way-contention misread I corrected within one turn.
- The `grep -o` on tqdm logs that stalled the srv link.
- My 2.7 h ETA error (7 jobs / 3 GPUs = three waves, not one).
- Queue renumbering mechanics and the row-move scripts.

VERBATIM ANCHORS:
- 4d347af · 61b402c · 72c0c87 · 146abca · 2ea68c2 · 946c07a · a1ea5e7 · f49e999 · ea8510d ·
  srv deploy 60f77f7.
- SPEC hashes 2b265701 (884) · fc2edd4d (885) · 4dd38d80 (887) · periods 68d2d7fd.
- 14.17 s/ep · pct_hold 82.3 → 95.6@ep3 → 99.0 · mcts_visit_entropy 0.1554 vs 0.0246
  (per-asset max ln(3)=1.0986) · sims 8/32/128 → 98.854/99.016/98.987 (0.13pp, non-monotone) ·
  −1.796 vs −1.866 · net@21 −3.953 at 6/8 slots wrong.

UNCERTAIN (operator: correct me):
- Given 887, does the SELF-PLAY LANE STAY OPEN AT ALL, or does the program pivot to "is there any
  edge in this feature/reward representation" before more training compute?
- Should 878 still ship? It rewrites execution semantics for policies that currently have no edge
  to measure.
- Is 887 strong enough to move D26 from PARKED to CLOSED?
- Both learned lanes now read no-edge. Does the linear CTA's paper-live gate get revived as the
  only thing with a measured (marginal) edge?
```

## Git state
```
4d347af 887: the epsilon fix works, and the collapse it fixes was ADAPTIVE — plus D38 ruled
61b402c factory run 31 handoff: 877 SHIPPED, 886 HALTED on D38
72c0c87 TASK_877: fold XBT->BTC, and make the fold earn it
3165de5 factory run 31: TASK_886 HALTS on D38 — the row welded two concerns, one never failed
146abca D37 RULED: fold into 886 as item 0 — and the defect is unfalsifiable retrospectively
```
Tree clean, factory DISARMED, **nothing pushed** — the owner pushes after review. `trading_data.db`
byte-untouched across all four factory runs and every planner leg; fixture never regenerated
(`b0f8de8b…`). Four stashes are load-bearing and none may be dropped — **resume by MESSAGE, not
index**, because run 29 and run 31 both shifted them while D35's text still names `stash@{0}`.
gpu-server: deployed at `60f77f7`, GPUs 1/2/3 idle after the 887 battery, ~159 GB free. The
`--apply` freeze on `tools/ckpt_archive.py` STANDS.

Shipped by the factory this window: **876** (`ae127b4`), **882** (`bbb2d38`), **873** (`261077b`),
**874** (`cecfcfd`), **883R** (`2ea68c2`), **877** (`72c0c87`). Halted: **886** (D38, split).

## Next action
Factory is GREEN to arm on Ready = **888 → 886 (contract only) → 878 → 881**. Planner side, in
order: (1) settle the four UNCERTAIN calls — the load-bearing one is whether the self-play lane
stays open after 887, because it decides whether any further training compute is justified;
(2) do NOT re-spec rung 1 as "rung 1 with epsilon mixing" — 887 measured that as trading more and
losing more; (3) when 888 ships, run the fingerprint backfill over the 7 surviving checkpoints,
since that and not the merge is what makes `verified: true` reachable; (4) keep the server under
load, but on a question worth the compute — after 887 the open question is edge in the
representation, not exploration in the policy.

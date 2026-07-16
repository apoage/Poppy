# QUEUE — the ranked dark-factory work order

The factory's single entry point (planner-maintained). STATUS banners on task files remain the
truth for what's done; this file only orders what's OPEN. The factory takes from the top, skips
gated rows (noting why), and treats FILLER rows as parallel/anytime slices. Last groomed:
**2026-07-16 (factory run 10 wrap-up) — 843 + 845 SHIPPED, Ready now EMPTY.** The env reached its
final fee0 form (843: netting + snap-to-flat; winning-short reservation BLOCKER fixed) and the 832
imprint runner is weight-compat with the pre-registered teacher-expressibility gate (845). Two OPEN
planner decisions came out of the run: **D14** (pre-existing fee16 all-short overdraft wipe — before
phase-2 fee16) and **D15** (E5 teacher-expressibility structural sizing divergence — mapping/params
before the imprint→ExIt spend; on D13's critical path). Prior context: **2026-07-14 D13 DECIDED:
REDIRECT** (operator; H6 widening = named "rainy day" contingency) → critical path = E5/832
imprint→ExIt (DEC-2 bar ε=+0.10 + per-window consistency leg); the 844 postmortem showed no locus
tripped 4/4 (the failure is "no profitable behavioral prior", which imprint injects). **Next moves
are PLANNER-owned**: the scheduled E5 imprint-sanity→imprint→ExIt real-panel runs (un-blocked by 845)
+ ruling D14/D15. No new factory-loop task until a Ready row lands.

## Ready — take from the top

| # | Task | Size | Note |
|---|------|------|------|
| _(none)_ | Ready EMPTY after factory run 10 (843 + 845 shipped). Next moves are PLANNER-owned: the scheduled E5 imprint-sanity→imprint→ExIt real-panel runs (un-blocked by 845) + the OPEN decisions **D14** (fee16 all-short overdraft, before phase-2 fee16) and **D15** (E5 teacher-expressibility mapping/params, before the imprint→ExIt spend). No new factory-loop task until a Ready row lands. | | |

_Shipped 2026-07-16 factory run 10 (843 → 845 in sequence, the D12-cleanup → E5-compat chain):_
- _**TASK_843** (`task843-weight-incremental-netting`, `ed27849`) — weight-path incremental netting + snap-to-flat
  (D12): delta-only re-trim (entry preserved) + snap-to-flat kills the short over-tax (185.2→≈canon) and
  slippage dust. Canon-OFF `48e1f467…` + budget `4f8b692e…` pins UNCHANGED; weight-ON/fee0 RE-BASED
  `1c50d58c…`→`6293668f…`. Escalated 4-member panel: **1 BLOCKER (winning-short survivor under-reservation
  → all-short margin wipe at fee0 from a valid state; orchestrator-crossval-CONFIRMED vs pre-edit) FIXED**
  (footprint-fraction reservation `(2E−P)/P` clamped ≥1; re-run fail-closed judge confirmed the fee0 hole
  closed, 3,840-case fuzz + sweep 0 hits). Acceptance (i)–(iv) PASS; E1 fee16 edge +2.1667 (from 842's −0.134).
  **Remainder → DECISIONS D14** (PRE-EXISTING fee16 all-short overdraft wipe, stash-confirmed 842-identical)._
- _**TASK_845** (`task845-imprint-weight-compat`, `4ac7721`) — 832 imprint runner threaded through the weight
  env + teacher-expressibility gate; ZERO canon diffs. Medium panel + fresh fix-judge: **1 MAJOR (weight-native
  remediation mapping was DIAGNOSTIC-ONLY — runner always imprinted the order-code teacher) FIXED** via
  `--imprint-mapping {order-code,weight-native}` (default order-code = byte-identical) + **2 MINOR guards**
  (fail-closed flag-coupling assert; None-guard the fee16 bankrupt-teacher WARN crash). Expressibility gate
  REPORTED, fixture PASS/FAIL INCONCLUSIVE (losing slice) → **DECISIONS D15** (structural sizing divergence
  under default w_unit/w_max; binding read = real-panel run per DEC-2). Un-blocks the scheduled E5 runs._

_Shipped 2026-07-14 factory run 9 (solo, read-only diagnostic):_
- _**TASK_844** (`task844-short-bias-probe`, `b53ecfe`) — short-bias postmortem instrument (mirror-book
  value / flat-book marginals / leg decomposition + fixture smoke), zero canon diffs, D13-independent.
  Medium panel: regression clean; **2 MAJOR FIXED** — probe-1 mirror-Δ was uncalibrated (independently-
  init BUY/SELL embedding columns ⇒ a FRESH model trips |t|~4; now gated on a matched fresh-init null
  z-stat, smoke robust across seeds 0/7/8 z +0.10/−0.85/−1.12) + probe-3 mislabeled an all-close SUPERSET
  as the EXACT reward_attributable term (relabeled + action-attributed subset now gates credit_misassign).
  MINOR (smoke tol false-alarm) fixed; NIT (open_count counts limit-fill opens) → planner. Real numbers =
  the SCHEDULED planner run (p1c s0/s1(+s2) + p1b s1 control, weight-env flags)._

_Shipped 2026-07-12 factory run 8 (solo, canon-adjacent fee fix):_
- _**TASK_842** (`task842-weight-close-fee`, `30013a4`) — canon-equivalent explicit close/open-short fee
  on the TASK_840 weight re-trim (was undercharging ~16bps/close, D9). Every term ×fee_rate ⇒ **fee0
  bit-identical** (THIRD pin weight-ON/fee0 `1c50d58c…` captured pre-edit at ba53294; canon `48e1f467…`
  + budget `4f8b692e…` also MATCH) ⇒ **RUNNING phase-1c cost0 untouched**. Medium panel: correctness+
  regression session-limit-truncated (trending clean, orchestrator-substituted); flow-verifier re-run
  CONFIRMED **no undercharge** (weight ≥ canon per re-trim — the short RT is an OVER-charge from 840
  close+reopen/residual geometry, fee-independent, → **D12**) + canon-faithful arithmetic. NIT: within-1e-6
  is float32-summation fragile (float64-exact). **Un-gates phase-2 fee16.** 840 (iii) E1 fee16 edge honestly
  moved +0.235→−0.134 (the +0.235 was flattered by the fixed undercharge)._

_Shipped 2026-07-11 factory run 7 (840 + 841 parallel lanes, disjoint surfaces):_
- _**TASK_840** (`task840-weight-rebalance-step`, `31a545f`) — target-weight rebalance step (H6 env,
  D8 opt (c)): flag `--vector-weight-rebalance` default OFF (canon + 839-budget both bit-identical,
  TWO pins `48e1f467…`/`4f8b692e…`). Full high-tier panel: **1 BLOCKER (held-winning-short budget
  under-reservation, crossval-CONFIRMED) + 1 flow-verifier-CONFIRMED held-leg drift bleed → BOTH FIXED**
  (whole-book posval/PV reservation + cap-clip re-trims every drifted leg → cash≥0, Σ|w|≤1, margin no-op).
  Acceptance (i) survives full 1697-step walk (invariant gate cash≥0/Σ|w|≤1) / (ii) sticky-short PASS;
  **(iii) E1 edge +0.235 (flipped from −0.23, below +0.436) = FINDING, non-gating**. Residuals → **D9**
  (fee undercharge, gates fee16 phase-2) + **D10** (expressibility). Unblocks phase-1c re-train._
- _**TASK_841** (`task841-checkpoint-wedge`, `52b155f`) — `run_queue.sh --stall-timeout` wedge watchdog,
  **default OFF (opt-in)** per invariant #1 (unarmed = byte-identical to HEAD; CSV-declaring jobs only when
  armed). Medium panel: **1 MAJOR (default-armed false-kills healthy CSV-less/`--name=` jobs at ~55min)
  double-CONFIRMED → FIXED** + 3 MINOR. Root cause characterized (orbax post-save GC vs RLock), STOPPED at
  out-of-scope trainer surgery; `--debug-stall-dump` wired for the next real wedge. Default → **D11**._

_Shipped 2026-07-09 factory run 6 (solo, canon-adjacent env fix):_
- _**TASK_839** (`task839-budget-vector-step`) — the env-wall fix: flag-gated (`--vector-budget-step`,
  default OFF = canon bit-identical, pin `48e1f467…` MATCH pre-edit 39ae0bf) budget-constrained
  equal-split sizing in `step_vector` (RATIFIED D7 opt (a)). Full high-tier panel: **1 MAJOR
  crossval-CONFIRMED → RESOLVED** ("≤cash by construction" holds only at PV≈cash → honest wording +
  PV>cash test + **DECISIONS D8** seeded; ratified formula untouched) + security MINOR (symlink
  write-guard realpath) & fail-closed MINOR (probe exit code) FIXED. Acceptance: OFF bankrupt step 0 /
  **ON 4/4 survive ≥64 steps zero-overdraft**; E1 byte-identical (narrow-regime cap non-binding).
  **Unblocks the line**: phase-1 RE-RUN under the fixed step → phase-2 fee16 + 832 imprint-sanity +
  834/836 re-runs. (NB: phase-2 gate row below still lists TASK_839 as a gate — now SATISFIED; D8 is
  the residual planner call on PV-relative vs cash-relative before flipping the flag ON.)_

_Shipped 2026-07-09 factory run 5 (836+837 parallel lanes, then 838 canon-adjacent solo):_
- _**TASK_836** (`task836-e4-rank-probe`) — E4 value-vs-momentum rank probe; 0 MAJOR; 3 MINORs fixed
  (honest baseline labels + true walk-PV mirror, overlap p-caveat, degenerate-value reading)._
- _**TASK_837** (`task837-walkforward`) — R4 walk-forward rails; `build_pit_universe` verified NON-PIT
  → module-side PIT; 3 MAJOR gate-coverage roots fixed (POLICY-PIN to pre-registered constants,
  eval-file identity, de-confounded 5-negative battery + no-op control)._
- _**TASK_838** (`task838-pernode-resample`) — flag-gated per-node resampling, OFF bit-identical
  (pin 22b0c7f5 MATCH, provenance independently reproduced); 1 MAJOR fixed (835 threads the flag —
  train/eval operator symmetry); **acceptance FINDING: OFF > ON under an uninformed policy
  (Δnet@21 −2.79) — any ON run must re-measure with a trained policy first; default stays OFF**._

_Shipped 2026-07-08 factory run 4 (parallel-lane batch, all zero-canon-diff instrument modules,
each medium-panel verified with adversarial crossval + fix re-verify):_
- _**TASK_834** (`task834-e3-runner`) — checkpoint E3/DEC-1 runner; 2 MAJORs found+fixed (legality
  mask on both proposal legs; fail-loud vs flax silent fresh-init restore — a reproduced FALSE PASS)._
- _**TASK_835** (`task835-stage4-eval`) — faithful Stage-4 vector eval; 0 MAJOR; 3 MINORs fixed
  (LEDGER pointer, partial-rollout guard, provenance)._
- _**TASK_832** (`task832-linear-imprint`) — E5 buffer-injection BC runner; 1 MAJOR found+fixed
  (imprint entropy_coef leaking into the ExIt half via checkpoint metadata)._

_Shipped 2026-07-08 factory run 3: **TASK_828 Stage 3** (`task828-stage3` — vector-action training
loop + D2 action-conditional reward head + factored-marginal targets; parity MATCH==PINNED, E3 probe
green, full high-tier panel 0 BLOCKER/MAJOR + 1 MINOR fixed). Stage-3 TRAINING itself is the scheduled
gpu-server GPU run below, not a factory-loop job._
_Shipped 2026-07-08 factory run 2: TASK_833 (`task833-run-queue-repair` — repaired scheduler
redeployed to srv same day) and TASK_829 (Foxglove retired, `40afa14`)._

## Scheduled runs — owner: planner/main session (NOT factory-loop jobs)

**Server status (updated 2026-07-09)**: gpu-server db verified at FULL workstation parity —
daily tables synced 2026-07-08, and the minute `ohlcv` table measured at max ts 1765558500
(2025-12-12, == workstation; the "Jan-2026 vintage" concern was file-mtime, not data). All
daily AND minute jobs cleared. Server code current; the 7 drifted server files remain STASHED
(`git stash list`).
Stage-3 training destination = gpu-server GPUs 1/2/3 — run_queue.sh REPAIRED (TASK_833) and
redeployed 2026-07-08; the scheduler is cleared for use.

| Run | What | Where | Status |
|-----|------|-------|--------|
| 830 real corridor | full-panel floors+ceiling+expressibility gap, `trading_data.db` read-only | workstation CPU (screen+safe_run) | **DONE 2026-07-08** — corridor official [+0.424, +29.698] net@21, ref +0.751; H6 armed; LEDGER 830/E0-REAL |
| 831 real E1/E2 sweep | 9 scheme×K combos, full panel, seeds 0,1,2 | workstation CPU (screen+safe_run, ~7 min) | **DONE 2026-07-08** — E1 PASS 9/9 ⇒ Stage-3 training gate OPEN; carried scheme dithered-book\|K16; LEDGER 831/E1-REAL + E2-REAL |
| 828 Stage-3 training **phase 1** | cost0 ×3 seeds from scratch, dithered-book\|K16 **hb0.0/d1.0** (3-arm diagnostic winner; fee16-from-scratch REJECTED — collapse finding, LEDGER 828s3/preflight), snapshots every 100; `jobs_phase1.manifest` | gpu-server GPUs 1/2/3 via run_queue.sh (tmux `q_828s3`) | **DONE 3/3 2026-07-10 05:18** — s1/s2 ep2000 first pass; s0 wedged ep1581 (log-freeze post-checkpoint-save) → killed, archived `*_wedged_ep1581`, solo retry completed clean (rc 0, 20 snapshots; wedge did NOT reproduce at ep1581 — non-deterministic, not data-positional). Ops note: fresh tmux needs `source trader_env/bin/activate` (bare `python` = system Py2) |
| 828 phase-1 **instrument passes** | 835 eval + 834 E3/DEC-1 + 836 E4 on s1/s2 `snapshot_ep2000`, phase-1 flags EXPLICIT (`--vector-hold-bias 0.0` — 835's CLI default is 3.0!), fee0/slip0 run configs `runs/828s3p1_instruments/cfg_*.json` (server) | gpu-server GPUs 2/3 via run_queue.sh (tmux `q_828s3_inst`) | **DONE 2026-07-09 20:06** — 835 FAIL both seeds (bankrupt, below floor; LEDGER 835/S4-p1); 834 fail-loud walk-bankrupt (834/E3-p1); 836 n=0 (836/E4-p1). Root = **828s3/env-wall**. 834/836 re-run AFTER phase-1b. s0's 835 DONE 2026-07-10: same FAIL (bankrupt 501/676, hold 98.2) — verdict 3/3, LEDGER row folded |
| 828 Stage-3 training **phase 1b** | RE-RUN under the fixed step: phase-1 command + `--vector-budget-step` (D7 opt (a), **D8 ratified ship-and-measure**), cost0 ×3 seeds, `jobs_phase1b.manifest`; pre-registration = SPEC amendment 3 | gpu-server GPUs 1/2/3 via run_queue.sh | **s1/s2 DONE 2026-07-10 ~16:15** (ep2000, 20 snapshots, 9.7h). **s0 WEDGED at ep162** — SAME signature as phase-1's wedge: frozen immediately after "Finished synchronous save" (orbax post-checkpoint deadlock, now 2/6 seed-runs — pattern, candidate factory row) → killed 2026-07-11 21:08, archived `*_wedged_ep162`, solo retry RUNNING (tmux `q_828s3p1b_s0`, GPU 1) |
| 828 phase-1b **instrument passes** | 835 (D8 primary evidence) + 834 DEC-1 + 836 E4 on s1/s2 `snapshot_ep2000`, phase-1b flags (budget ON) | gpu-server GPUs 2/3 | **DONE 2026-07-11 21:35, 6/6 rc0 — the decisive mixed verdict**: 835 still bankrupt (step 132/181, s1 book = 10–11 shorts → D8 escalation FIRED ⇒ TASK_840); **834 DEC-1 PASS ×2** (R 1.30/1.12 vs bars 0.33/0.24 — D2 rebuild works); **836 value>momentum ×2** (ρ +0.21/+0.30 vs −0.03 — first positive learning signal). LEDGER 835/S4-p1b · 834/E3-p1b · 836/E4-p1b |
| 828 Stage-3 **phase 1c** | re-train under the TASK_840 weight env: phase-1 recipe + `--vector-weight-rebalance` (NO budget flag — one variable), cost0 ×3 seeds, `jobs_phase1c.manifest`; pre-registration = SPEC amendment 4; **D10 ratified PROCEED**; watchdog ARMED per D11 (`--stall-timeout 2700`) | gpu-server via run_queue (tmux `q_828s3p1c`, `--gpus 1,2,3 --acquire-timeout 0`) | **s0/s1 DONE 2026-07-12 07:32 + instruments graded same morning (LEDGER 835/S4-p1c · 834/E3-p1c · 836/E4-p1c): ENV FIXED (zero bankruptcies, first-ever complete rollouts) / POLICY BELOW FLOOR 2/2 (short-biased book) ⇒ D13 OPEN**; s2 DONE 14:16 + record pass graded (−1.385, complete rollout, 7L/8.4S — verdict 3/3). **Battery closed CLEAN: zero failures, zero wedges, watchdog armed throughout.** Earlier: s2 start was delayed by the **p1b-s0 retry wedge #3 at ep982** → py-spy stack CAPTURED (841 root-cause: XLA/CUDA dispatch hang post-save, orbax-lock hypothesis REFUTED — see TASK_841 evidence note + `lore/notes_2026-07-12.md`), killed, NO re-retry (p1b verdict stands on s1/s2). Graded read: corridor vs floor +0.424 on ≥2/3 seeds; any bankruptcy = TASK_840 defect |
| 844 **short-bias postmortem** | `bias_probe.py` on p1c s0/s1/s2 + p1b s1 control (`jobs_844.manifest`, weight/order run flags from `cfg_*.json`) | gpu-server GPUs 2/3 via run_queue.sh (tmux `q_844`) | **DONE 2026-07-14 20:40, 4/4 rc0 — NO LOCUS TRIPPED (LEDGER 844/postmortem)**: value clean (calibrated \|z\|≤0.9, the raw-t would have false-tripped ×4), prior BUY-leaning (0.65–0.84), credit-misassignment unsupported. Planner post-hoc: probe leg tracker is FLIP-BLIND in weight semantics — shorts exit ONLY by BUY-flip to long (exact arithmetic 3/3); longs culled by non-action re-trim closes (margin ruled out — fee0 no-op invariant). Folded into D13 |
| 832 **imprint-sanity** (E5 leg 1) | teacher replay floor (weight env, real panel — the binding ±0.10-absolute expressibility read) + BC imprint + deterministic sanity replay vs teacher ±0.10 (DEC-2 tolerance); cost0, weight flags = phase-1c set; run BOTH mappings via `--imprint-mapping {order-code,weight-native}` (845); ≤1 GPU-day | gpu-server single verified-idle GPU, run_queue + D11 watchdog | **UN-BLOCKED 2026-07-16** (TASK_845 shipped; TASK_843 shipped before it → env at final form, no re-base pending). **Rule DECISIONS D15 first** (which mapping / whether to retune w_unit/w_max — fixture shows a structural sizing divergence; this run IS the ±0.10-absolute binding read). FAIL here = H5 falsification clause fires (BC/representation path implicated) → planner, before any ExIt spend |
| 832 **ExIt** (E5 leg 2, = the phase-2 continuation under redirect) | short ExIt from the imprint ckpt (`--load-checkpoint --reset-optimizer --clear-buffer`; verify `Restored entropy_coef: 0.01` at launch — SPEC PIN c), ≥3 seeds, fee16 per DEC-2; PASS = floor+0.10 net@21 ≥3 seeds (per-seed above bare floor) + ≥3/4 regime windows at/above their window floor | gpu-server, run_queue + D11 watchdog | GATED on imprint-sanity PASS. PASS ⇒ R3 gate opens (warm-start ranked first); FAIL with sanity green + E1/E3 green ⇒ H1 weakens, planner pivots (pathway kill logic). H6 widening = contingency if the IMPRINTED policy can't hold the floor |
| ~~828 Stage-3 phase 2 (from-scratch continuations)~~ | ~~fee16 continuation from phase-1b/1c ckpts~~ | — | **SUPERSEDED by D13 redirect 2026-07-14**: the ExIt row above IS the phase-2 fee16 continuation, warm-started from the imprint. From-scratch ckpts (p1/p1b/p1c) = forensics + probe inputs only, never warm-start (cell measured negative twice: LEDGER 835/S4-p1b, 835/S4-p1c) |

## Gated — do NOT start (named gate must open first)

| Task | Gate |
|------|------|
| ~~`tasks/TASK_843_WEIGHT_INCREMENTAL_NETTING.md`~~ | gate SATISFIED 2026-07-14 → moved to Ready #1 |
| `tasks/TASK_828_VECTOR_ACTION_MUZERO.md` (Stage 4 eval) | Stages 1–3 shipped ✓ (2026-07-08) + Stage-3 training run's rung verdicts green; SCHEDULED server run, not a factory-loop job |

_TASK_832 (E5 linear imprint) moved to Ready 2026-07-08 — its gate (828 Stage 3 banner) is now satisfied._

## Parked (D1/D4, 2026-07-07 — non-binding "interesting outview"; not factory work)

- `tasks/TASK_824_SERVER_HONEST_REMEASURE.md`, `tasks/TASK_825_CROSSASSET_TREND_CTA.md`,
  `tasks/TASK_826_ONCHAIN_SIGNAL_PROBE.md` (D4: OHLC-only, live within the 24GB DB),
  `tasks/TASK_823_VPS_MICROLIVE_CTA.md` (D0: rejected).

## Grooming rules
Planner re-ranks after each factory run + banners; a row leaves this file only by shipping
(banner) or being explicitly parked. If the factory finds the top row blocked in practice, it
takes the next and records why in its wrap-up.

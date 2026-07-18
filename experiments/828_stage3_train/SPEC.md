# SPEC — 828 Stage-3 training battery (gpu-server, scheduled run)

**Pathway**: R3 (`docs/pathway.md`). The first LEARNED-model test of H1 — everything until now
was perfect-model. Pre-registered 2026-07-08 before launch. Owner: planner/main session.
Mechanism: TASK_833-repaired `tools/deploy/run_queue.sh`, GPUs {1,2,3}, 1 job/GPU, tmux.

## Jobs (manifest `experiments/828_stage3_train/jobs.manifest`)
Two passes × 3 seeds, all with the 831-EXT-carried operator — dithered-book, K=16,
**hold-bias 3.0 explicit** (CLI default 0.0 is a trap), dither 1.0; daily bars via
`--bar-minutes 1440` on the pinned 821 window/universe (64 assets, never reorder);
`--reward-hybrid --reward-attributable`; small model, sims 32; `--snapshot-every 100`
(immutable mid/late checkpoints — required by the E3/DEC-1 probe, TASK_834).
- **Primary: fee16/slip5 ×3 seeds** (`--assert-postcost`) — the bar condition. The known
  failure mode (821 HOLD collapse) must be beaten UNDER COSTS or not at all.
- **Diagnostic: cost0 ×3 seeds** (no assert-postcost) — existence proof / cost-attribution
  if the fee pass degenerates.

## Pre-flight gates (in order, before the battery)
1. **Deploy** current master (Stage-3 code) to srv; verify tag reachable.
2. **Minute-vintage check** (the training path reads the 1-min `ohlcv` table — recon-verified
   `env_factory.py:289-379` → `vectorbt_env.py:582+`): server `MAX(timestamp) FROM ohlcv` ≥
   **1764543600** (2025-12-01, window end) + spot-check coverage for the pinned 64 symbols.
   FAIL ⇒ full `sync-db.sh` before launch (table-level daily parity does NOT cover this).
3. **GPU smoke**: fee16 template, `--epochs 40 --no-checkpoint`, one verified-idle GPU.
   PASS = env ~1731 steps × 64 assets, finite losses, pct_hold trending <99, and a steady-state
   s/epoch measurement. **Pre-registered budget rule**: if s/epoch × 2000 > 20h/seed, cut
   `--epochs` to fit ≤20h/seed (snapshots every 100 keep partials usable) — recorded, not tuned.

## Pre-flight amendment 2 (2026-07-08, post 3-arm diagnostic — REVISED BATTERY)
Smoke #1 (hold-bias 3.0): pct_hold 94.7→99.97, entropy→0.005 — FAILED. 3-arm diagnostic
(80 ep, fee16 from scratch): **B hb1.0/d1.0 COLLAPSED** (99.4%, ent 0.05); **C hb3.0/d2.0
letter-pass only** (97.0%, ent 0.26→0.18 sliding — slow collapse); **A hb0.0/d1.0 the only
live policy** (ent 0.74) but hold rising 17→55→80% while fees bleed it (α ≈ −100%).
**Structural finding**: fee16-from-scratch punishes exploration before the value learns —
proposals only set collapse speed. Stop condition applied in spirit: NO fee16-from-scratch
battery. **Revised design (pre-registered)**:
- **Phase 1 (launched)**: cost0 ×3 seeds from scratch, proposals **hb0.0/d1.0** (the
  empirically-alive arm; the fee-punishment driver is absent at cost0). 2000 ep, snapshots
  every 100. Deviation from the 831-EXT carried hb3.0 RECORDED — E2 measured search quality
  under a true value; training exploration is a different regime.
- **Phase 2 (design gate: phase-1 snapshots + 832 imprint sanity)**: fee16 as CONTINUATION,
  two arms — (a) warm-start from cost0 checkpoints (cost curriculum), (b) warm-start from the
  832 linear-book imprint (E5 promoted: the book exists from epoch 0, dithered-book has a real
  book to dither, value learns on sensible trajectories). `--load-checkpoint
  --reset-optimizer --clear-buffer` (shipped, 832-verified); verify the `Restored
  entropy_coef: 0.01` line (832 PIN c).
- Instruments (834 E3 probe / 835 eval) must be invoked with the flags ACTUALLY launched:
  phase 1 = `--vector-hold-bias 0.0 --vector-dither-scale 1.0`, cost0.

## Monitoring & verdicts
- In-loop generalization eval is fail-closed SKIPPED in vector mode (panel finding) — monitor
  `logs/training_<name>.csv`: reward/alpha, per-asset action histograms, pct_hold,
  `mcts_visit_entropy` (vector-correct in CSV; DS24 shows cosmetic 0.0 — known), losses.
- **HOLD-collapse tripwire (the 821 signature)**: pct_hold pinned >99 with entropy floor-crash
  past epoch ~300 on ≥2 fee seeds ⇒ record FAIL early, don't ride it to 2000.
- **E3/DEC-1 gate (mid/late snapshots)**: `R_model ≥ 0.5 × R_true` via the TASK_834 runner
  (fallback absolute ≥0.5 if R_true leg late). This decides whether the search plans over a
  reward model that feels its own actions — the D2 question, now empirical.
- Stage-4 full-timeline eval vs the corridor/ref stays a SEPARATE gated run (its faithful
  per-asset eval harness is also not built yet — future factory row).
- Artifacts: server `jax_ckpt_fast/<name>/` + `logs/training_<name>.csv` + `runs/828_stage3/`
  → pulled via `pull-results.sh`; verdict lines → `experiments/LEDGER.md` (per pass, per seed).

## Amendment 3 (2026-07-10, planner — PHASE-1b: re-run under the fixed step; pre-registered before launch)

Phase-1 verdict was environmental, not model (LEDGER **828s3/env-wall**, 835/S4-p1 s0/s1/s2 all
bankrupt): `step_vector`'s snapshot-cash accounting made breadth lethal at any cost tier.
TASK_839 shipped the fix (`--vector-budget-step`, D7 opt (a), OFF bit-identical); **D8 ratified**
ship-as-is-and-measure (planner ruling 2026-07-10).

**Phase-1b battery** — identical to phase 1 except the flag and names:
- Jobs `828s3p1b_cost0_s{0,1,2}` = the phase-1 command + `--vector-budget-step`
  (`vector_budget_pct` default 0.99), manifest `jobs_phase1b.manifest`, gpu-server GPUs 1/2/3.
- **Pre-registered reads**:
  1. Mid-run tripwire UNCHANGED (pct_hold >99 & entropy <0.02, ≥2 seeds past ep300 ⇒ FAIL early).
  2. Health signal (not a gate): self-play episodes should no longer mass-terminate by
     bankruptcy; cost0 self-play reward ma20 materially above the phase-1 −13..−26 band.
  3. Post-run instruments (phase-1b flags = phase-1 flags + `--vector-budget-step`): **835
     rollouts must COMPLETE (partial=false, no bankruptcy) — this is the D8 primary evidence**;
     corridor grading per D6; 834 DEC-1 and 836 E4 become measurable for the first time.
  4. If 835 rollouts still die by margin cascade on invested books ⇒ D8 escalation: layer
     opt (c) Σw≤1 as a NEW task; do not tweak 839 in place.
- Phase-2 fee16 continuation re-targets phase-1b checkpoints (phase-1 ckpts are wall-domain
  artifacts — keep for forensics, never warm-start from them).

## Amendment 4 (2026-07-11, planner — PHASE-1c: the weight-env re-train; pre-registered before launch)

Phase-1b verdict (LEDGER 835/S4-p1b + 834/E3-p1b + 836/E4-p1b): **the nets learn — the env
kills**. DEC-1 PASS ×2 and value>momentum ×2, while incremental-order semantics bankrupted every
rollout (sparse books, unhedged shorts, margin cascade). TASK_840 shipped the target-weight
rebalance env (D8 opt (c)); **D10 ratified PROCEED** (E1 edge positive +0.235; the trained
policy is the real expressibility test).

**Phase-1c battery** — the phase-1 recipe on the WEIGHT env:
- Jobs `828s3p1c_cost0_s{0,1,2}` = phase-1 command + `--vector-weight-rebalance`
  (weight-unit/deadband/max at shipped defaults: 1/n_assets, 0.1×unit, 4×unit). NO
  `--vector-budget-step` (independent flag; weight semantics replace trade sizing wholesale —
  one variable at a time). Manifest `jobs_phase1c.manifest`, gpu-server, run_queue with
  `--stall-timeout 2700` (D11: training launches arm the watchdog) and `--acquire-timeout 0`
  (third seed waits for GPU 1 to free from the p1b-s0 retry).
- **Pre-registered reads**:
  1. Collapse tripwire UNCHANGED (pct_hold >99 & ent <0.02, ≥2 seeds past ep300 ⇒ FAIL early).
  2. Invariant health (env property, not model): ZERO bankruptcy terminations in self-play —
     any bankruptcy under weight-ON is a TASK_840 defect ⇒ stop, file, no verdict.
  3. Post-run instruments (phase-1c flags: hb0.0/d1.0, fee0/slip0, `--vector-weight-rebalance`;
     NO budget flag): **835 rollouts MUST complete (guaranteed by env) — the graded number is
     the corridor position**: primary = beats floor +0.424 net@21 on ≥2/3 seeds; the D6 engine
     ref +0.751 and per-window floors reported alongside. 834 DEC-1 + 836 E4 re-run for
     continuity (pass/diagnostic per their own pre-registrations).
  4. **D10 granularity trigger**: phase-1c below floor on ≥2/3 seeds ⇒ H6 ladder-widening task
     (weight-unit sweep / finer action set) becomes the next build; NOT a tweak to 840.
  5. Phase-2 fee16 stays GATED on TASK_842 (D9 close-fee) + phase-1c verdicts.
- s0-p1b retry (record-completion) runs to term on GPU 1; phase-1c s2 queues behind it.

## Amendment 5 (2026-07-17, planner — ExIt battery: the D13-redirect phase 2; pre-registered before launch)

Gates now OPEN: imprint-sanity PASSED (LEDGER 832/E5-imprint: replay +0.839 net@21 vs teacher
floor +0.542, zero bankruptcies) and TASK_846 SHIPPED (`task846-weight-fee-headroom`, c2f1ba5 —
the D14 fee16 all-short wipe is closed; binding distortion read Δ=0.0008, D16). This is the
phase-2 fee16 continuation under the D13 redirect: warm-start from the E5 imprint, NEVER from
the from-scratch ckpts (superseded, LEDGER 835/S4-p1{b,c}).

**ExIt battery** — phase-1c recipe + fee16 + the imprint warm start:
- Jobs `e5exit_fee16_s{0,1,2}` = the phase-1c command with: `--fee 16 --slippage 5` (the canon
  fee16/slip5 = net@21 cost model, DEC-2), `--load-checkpoint runs/832_e5/ckpt_imprint_full
  --reset-optimizer --clear-buffer` (PIN-(c), all three), `--vector-weight-rebalance` at the
  SAME shipped sentinels as the imprint (unit 1/64, max 4, deadband 0.1×unit — SPEC 832 "ExIt
  launch flags": mismatched weight flags = the cross-env imprint the gate exists to prevent),
  seeds {0,1,2}, `--epochs 1000` (SHORT continuation: warm start ≠ from-scratch 2000; extension
  past 1000 only via a pre-grading amendment), `--snapshot-every 100`.
  Manifest `runs/832_e5/jobs_exit.manifest`, gpu-server, run_queue `--gpus 1,2,3
  --acquire-timeout 0 --stall-timeout 2700` (D11 armed; wedge watchdog is the mitigation).
- **Launch verification (hard)**: each log must show `Restored entropy_coef: 0.01` (832 PIN c —
  the imprint's 0.001 knob must NOT leak into ExIt). Missing line ⇒ kill before ep50, fix, relaunch.
- **Pre-registered reads**:
  1. Collapse tripwire UNCHANGED (pct_hold >99 & ent <0.02, ≥2 seeds past ep300 ⇒ FAIL early).
     Additional imprint-erosion tripwire: pct_hold falling BELOW 80% with alpha collapsing
     (fee-bleed churn) on ≥2 seeds past ep300 ⇒ record, planner reads before riding to 1000.
  2. Invariant health: bankruptcy termination in self-play at fee16 is now a TASK_846 defect
     (the wipe is supposed to be closed) ⇒ stop, file, no verdict.
  3. **Graded read (DEC-2, the E5/R3 verdict)**: 835 corridor eval on the FINAL snapshot
     (ep1000), ExIt flags at eval: PASS = net@21 ≥ floor+0.10 on ≥3/3 seeds (per-seed above the
     bare floor +0.424) AND ≥3/4 regime windows at/above their window floors (consistency leg).
     Baseline anchor: the imprint itself replays at +0.839 — an ExIt that DEGRADES below the
     imprint replay is itself a finding (search/self-play subtracting value), graded honestly.
  4. FAIL with sanity green + E1/E3 green ⇒ H1 weakens, planner pivots (pathway kill logic);
     the H6 rainy-day contingency (D13) activates only if the IMPRINTED policy can't hold the
     floor.

### Amendment 5 addendum (2026-07-18, planner — post-verdict erosion-curve diagnostic; pre-registered before results)
ExIt graded FAIL 2/2 (s1 −1.223, s2 −0.791 net@21 vs bar +0.524; s0 wedge-failed ep706, moot —
verdict determined). Both seeds landed in the from-scratch attractor (99% hold, sparse book)
FROM an above-ref warm start (+0.839). Diagnostic before the pivot decision (D17): grade the
SNAPSHOT CURVE on s2 (better final seed), snapshots ep{100,300,500,700,900}, same 835 eval
operator as the verdict run. Read: net@21 vs epoch = the erosion speed. Interpretation guide
(pre-registered): above-floor at ep100 with monotone decay ⇒ slow erosion — a BC-anchor /
KL-regularizer or shorter ExIt may hold the prior; already below floor at ep100 ⇒ the fee16
self-play target destroys the prior ~immediately — training-objective-level fix required
(curriculum or anchored objective), not a duration tweak. Diagnostic-only: no corridor claim,
folds into D17 context.

### Amendment 5 addendum 2 (2026-07-18, planner — targets-autopsy read A; pre-registered before result)
Erosion curve read: IMMEDIATE decay (ep100 already −0.105 net@21, 99.4% hold, sparse book vs
the +0.839 dense-book warm start; ep300..1000 = noise around the attractor). Per the addendum-1
guide: training-objective-level fault ⇒ autopsy the TARGETS. **Read A (zero training):** 835
eval (visit-argmax, sims 32, dithered-book K16 — the ExIt search operator) of the UNMODIFIED
`runs/832_e5/ckpt_imprint_full` on the SAME panel where its policy-argmax replay scored +0.839
(2% hold, 29L/32S). Interpretation (pre-registered): search-book ≈ policy-book (dense, ~floor)
⇒ search layer NOT the erosion source — suspicion moves to the BC/replay-buffer interaction;
search-book = HOLD/sparse ⇒ the search visit distribution OVERRIDES the prior toward HOLD at
decision time — value/reward leaf evaluation at fee16 is the demonstrated erosion mechanism
(then the anchored-ExIt design must fix LEAF EVAL or target mixing, not just add a BC loss).
Diagnostic-only; folds into D17.

## Amendment 6 (2026-07-18, planner — ExIt re-try under the policy-prior operator; pre-registered BEFORE the gating read)

Gate: LEDGER-bound read-A INVERSION (835 search-eval of frozen `runs/832_e5/ckpt_imprint_full`
with `--vector-proposal-scheme policy-prior`, same panel/flags as 832/E5-autopsy otherwise).
**Launch the battery ONLY if inversion net@21 ≥ +0.74** (policy-argmax +0.839 − 0.10; the same
configuration scores −6.92 under dithered-book). Below +0.74 ⇒ NO training spend — the residual
gap is leaf-eval territory (read B: value ranking of prior-book vs candidates) → back to D17's
decided tree via a new DECISIONS row.

**Battery `e5exit2_pp_s{0,1,2}`** = the Amendment-5 ExIt command with TWO changes:
`--vector-proposal-scheme policy-prior` and `--epochs 300` (the erosion window — Amendment-5
addendum showed annihilation by ep100, so 300 is decisive), `--snapshot-every 50` (fine curve).
Everything else IDENTICAL (fee16/slip5, PIN-(c) from `ckpt_imprint_full`, weight sentinels,
hb0.0/d1.0 dither params feed the explore slots, D11 watchdog 2700, `Restored entropy_coef:
0.01` hard check).

**Pre-registered reads**:
1. PRIMARY (prior-survival): ep300 snapshot net@21 ≥ +0.74 under BOTH operators — 848
   policy-argmax replay AND 835 policy-prior search-eval — on ≥2/3 seeds. PASS ⇒ the improvement
   operator no longer subtracts value ⇒ resume the DEC-2 ladder (longer ExIt = next planner
   schedule, its own amendment). FAIL with the inversion green ⇒ the TRAINING-side targets
   (buffer/CE/value drift), not decision-time search — read B before any further spend.
2. Erosion curve at ep{50,100,...,300} via 848 policy-argmax (cheap, batched post-run) — decay
   shape vs the Amendment-5 curve is the mechanism read.
3. Tripwires + invariants carry over from Amendment 5 unchanged (collapse, bankruptcy=846
   defect, wedge watchdog).

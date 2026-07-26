# Dark-factory run 19 — handoff (2026-07-26)

Neutral technical continuity record. One task shipped; queue now empty of Ready rows.

## Shipped

| Tag | Task | Verification | Canon |
|-----|------|--------------|-------|
| `task860-reward-head-codistill` (`736fdc6`) | TASK_860 — reward-head co-distill (D24 "seed co-distill", D19 opt (c)) | medium panel (correctness+regression+fail-closed) + fresh fail-closed re-verify; 1 BLOCKER + 2 MINOR fixed | OFF byte-identical (615/615 leaves diff 0, selftest 12/12) |

Only file changed: `experiments/857_vhat_distill/distill.py` (+286/−50). Flag `--codistill-reward`
default OFF = shipped-857 bit-identical.

## What it does
Trains the reward head on true-r0 counterfactual targets ALONGSIDE the 857 value imprint,
to attack the frozen-832-reward-head cap on the deployable ranking `Q = r_model + γ·v̂`
(LEDGER 857/vhat-corridor: corr(r_model, true r0) ≈ 0.27 dragged the deployed-Q read below
grade even though v̂-alone lifted). Emits `r0` as its own target column (reuses the array
already computed for `G = r0 + γ·V` — RAM only); loss = value CE + λ·reward CE on the reward
head's own `reward_popart` two-hot; optax `multi_transform` unfreezes ONLY `dynamics_dense2`
reward columns (policy head + encoder stay frozen). Reward CE is pure CE (matches 857's value
path), NOT the trainer's `0.8·Huber+0.2·CE` reward_hybrid blend.

## The BLOCKER worth remembering (fail-closed lens)
The first implementation gated the ON smoke on **deployed-Q held-out per-state improvement**.
At fixture scale (n=12 held-out states) that quantity is **pure noise and inverts per-seed** —
`--codistill-reward --shuffle-targets` (the spec's mandatory falsification, acceptance iv)
PASSED the smoke (exit 0, green) on seeds 1 and 2. The now-trainable reward head fitting the
shuffled r0 *inflates* r_model, propping Q back over the −0.25 floor — the very mechanism the
task adds defeated the frozen-reward assumption that made shuffle collapse Q in shipped 857.

Fix: replaced it with a **high-SNR train-set fit gate** — pooled correlation of the POST-distill
heads vs the never-shuffled TRUE targets over the full train set (~n=972); both `r_model·true-r0`
AND `v̂·true-G` must clear floor **+0.45**. Legit fits true → ~+0.9; shuffle fits a permutation
of true → ~0. Clean separation across seeds: legit band [+0.833, +0.943], sabotage band
[−0.187, +0.055], ~0.38 cushion each side. Seed sweep 0–3 both modes: legit 4/4 exit 0,
shuffle 4/4 exit non-zero. deployed-Q + held-out r_model demoted to reported-only (explicitly
labeled non-discriminative at fixture scale). Lesson for future fixture-scale falsification
hooks: don't gate on a noisy small-n held-out metric; a full-train-set fit-vs-true certificate
robustly separates learned-signal from sabotage.

## Planner remainder (scheduled, pre-registered in the spec — NOT a factory job)
The fixture gate only certifies "the pipeline learned the true signal, not noise." The REAL bars:
1. Real distill: 3 seeds, warm-start `runs/832_e5/ckpt_imprint_full`, `--codistill-reward`.
2. **deployed-Q held-out per-state ≥ +0.5** (the 857 bar, now on the co-distilled Q).
3. Then the SAME corridor read (`--leaf-eval vhat-distilled` on the co-distilled ckpt):
   floor **+0.424** bar; **+8.02** true-leaf ceiling reference; **−4.23** = the 857 value-only
   best-seed reference for attribution. Report BOTH v̂-alone and deployed-Q calibration.
OOS routes through the 858-extended canon separately, after the in-sample corridor.

If the co-distilled deployed-Q clears +0.5 and the corridor lifts materially over −4.23 toward
the floor, the deployable operator gains a real ranking; if not, the reward-head axis is also
capped and the next fork is a fresh DECISIONS row (encoder/value-target, not more reward tuning).

## Queue state after this run
- **Ready = EMPTY.** TASK_860 was the sole Ready row (D24 seeded it).
- Next Ready rows seed from the planner's scheduled 857/860 real-panel verdicts (D24 fork).
- **D22 (minute backfill, ops)** stays OPEN non-blocking: TASK_858 shipped the guarded ingest
  tool; the real ingest awaits the operator downloading Kraken quarterly OHLCVT dumps
  (Q4-2025…Q3-2026) + a fresh DB backup, run on the WORKSTATION. This gates all binding OOS reads.
- Factory IDLE pending those planner/operator legs.

## Open questions for the operator (highest-value unblocks)
1. **Kraken dump download (unblocks every binding OOS read + walk-forward battery).** TASK_858's
   guarded backfill tool is shipped and fixture-green but cannot self-serve the data. Download the
   quarterly OHLCVT CSV dumps and take a fresh backup so the workstation ingest can run?
2. **857/860 real-panel scheduling.** The 857 distill+corridor chains were reported running on
   gpu-server overnight (run 17/18 context); the 860 co-distill real leg is now also ready to
   schedule from `ckpt_imprint_full`. Confirm the server is free to run the co-distill leg, or
   should it queue behind the in-flight 857 reads?

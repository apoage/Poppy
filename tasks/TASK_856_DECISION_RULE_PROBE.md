# TASK_856 — Decision-rule probe on the distilled marginals + net-dependent grading fix (D23 step 1)

**Status**: READY 2026-07-23 (D23 ruled — operator: "scope both to factory .. aswell as fix ..").
**Size: XS/S. Risk: low** (eval operator extension + grading hygiene; zero training-path code).

## Why (LEDGER 832/E5-exit2)
The ExIt2 policy trains stably toward the fixed search's visit targets (~94% hold / ~6% sparse
trades) but per-asset ARGMAX rounds to 100% HOLD from ep50 — while CE ≈ 0.29 (far from
point-mass) says the trade information survives IN THE DISTRIBUTION. Whether a better decision
rule extracts it decides if distillation is a tuning problem or an architecture problem —
zero training spend to find out: the ep300 checkpoints already exist
(`jax_ckpt_fast/e5exit2_pq_s{0,1,2}/snapshot_ep300`, srv).

## Scope
1. `--decision-rule {argmax,threshold,top-m}` on the 848 policy-replay operator:
   - `threshold`: per asset, act on the best non-HOLD action iff its marginal mass > τ
     (`--decision-tau`, sweepable);
   - `top-m`: per step, act on the M assets with the highest non-HOLD mass (`--decision-m`,
     M≈4 ≈ the search's ~6%×64 trade rate), argmax the action within each;
   - default `argmax` = bit-identical today.
2. **Grading fix (the vacuous-leg hole, LEDGER 832/E5-exit2 item 2)**: `run_queue`-facing
   assertion in the 835 search-eval path — when the configured operator has ZERO learned
   components (prior-weight 0 + true-env leaf + no v̂ tail + a non-policy proposal scheme),
   the result JSON carries `"net_dependent": false` and prints a WARNING that the read cannot
   grade a checkpoint; the 848 replay and any pw>0 / v̂-leaf operator set `true`. Fail-loud
   beats silently-vacuous — this exists because a 3-checkpoint grading run returned bit-identical
   teacher numbers.
3. Fixture smoke: rules produce differing books on the fixture ckpt; property tests (τ→1 ⇒
   argmax-when-confident-else-hold; M=64+τ=0 ⇒ argmax).

## Acceptance (gates)
- (i) Default bit-identical (SHAs); selftest green; property tests.
- (ii) Fixture smoke as above. The REAL sweep (ep300 ckpts × rule grid, floor bar +0.424) =
  planner's scheduled run, same discipline as every binding leg.
- (iii) Small panel (correctness + flag-wiring lenses).

## Out of scope
Any training; TASK_857 (independent build); retuning the search operator.

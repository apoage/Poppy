# TASK_882 — Two "argmax replay" instruments disagree by 1.6 net@21 on the same weights

**Status**: READY 2026-08-07 (planner; from the 879 rung-3 control leg). **Blocks every
cross-harness comparison in the program until it lands.**
**Size: S/M. Risk: low to run, HIGH in consequence** (it decides which of two load-bearing
reference numbers is real). Surface: `experiments/832_linear_imprint/imprint.py` sanity replay ·
`experiments/835_stage4_eval/eval_vector.py --operator policy-argmax` · whatever the 862
scorecard actually runs.

## The measurement (LEDGER `882/instrument-disagreement`)

Same checkpoint `runs/832_e5/ckpt_imprint_full`, same bar size, same cost tier, both claiming to
replay per-asset argmax:

| instrument | window | net@21 | final PV | book |
|---|---|---|---|---|
| `imprint.py` sanity replay | 2021-01-01..2025-09-01 | **+0.8394** | **299,547** (3× gain) | 28.7L / 32.0S ≈ 61 of 64 held |
| `imprint.py` sanity replay (861 ref s0) | 2021-01-01..2025-10-04 | **+0.9914** | **465,545** (4.7× gain) | 25.8L / 30.9S |
| `eval_vector --operator policy-argmax` | 2021-01-01..2025-10-04 | **−0.6145** | **41,919** (58% LOSS) | 5.9% hold |

1673/1673 decisions, no bankruptcy, weight env ON, in-run teacher floor on the eval_vector side.
**The two harnesses produce opposite economic outcomes on identical weights.** The window was
already eliminated as the cause — that is what the third row measures.

## Why this is urgent rather than interesting

`+0.9914` is the program's current in-window reference (LEDGER `861/ref-rebase`), the number that
superseded `+0.839` and the one the imprint lane's "real in-window learning" claim rests on. If
eval_vector is right, that claim is an instrument artifact. If imprint.py is right, then every
eval_vector-based read — including the 875 walk-forward battery and the 879 rung-3 corridor —
is measuring something other than what it claims. **One of those two families is wrong and both
are load-bearing.**

## Scope

1. **Classify every instrument.** Enumerate which LEDGER numbers came from which harness:
   `imprint.py` sanity replay · `eval_vector --operator policy-argmax` · `eval_vector --operator
   search` · the 862 scorecard (which wraps one of these — determine which, and whether
   eval-MCTS is a fourth). Output: a table in this task file, one row per LEDGER row, naming its
   instrument. This is mostly artifact reading, not compute.
2. **Per-decision diff on the fixture** (small, deterministic, no GPU needed at scale): run both
   instruments on the same fixture checkpoint and window, dump per-decision `(action_vector,
   position, cash, pv, fee)`, and find the FIRST decision where they diverge. Report that
   decision's full state from both sides. A single divergent step localises the defect far
   faster than arguing about aggregates.
3. **Test the leading hypothesis explicitly**: the imprint artifact records
   `position_size_pct 0.0078125` (1/128) while eval_vector runs its own `w_unit`/
   `vector_weight_max`. If sizing differs, the same argmax produces different leverage and the
   PV paths separate exactly as observed. Confirm or eliminate — and if it IS sizing, state which
   one matches the trained policy's own semantics, because that is the one that is correct.
4. **Rule which instrument is canonical for which claim**, and write it where it cannot be
   missed: in-sample learning claims, OOS claims, and floor comparisons must each name their
   harness. A number without an instrument is not a result.

## Acceptance (gates)

- (i) The instrument table exists and covers every LEDGER row that carries a net@21 number.
- (ii) The first divergent decision is identified, with both instruments' state printed side by
  side, and the mechanism named.
- (iii) A regression test pins the agreed semantics so the two cannot silently drift apart again.
- (iv) Any LEDGER row whose number is invalidated gets a correction stamp naming the instrument
  (the `857/860` MEASUREMENT VOID stamps are the format).
- (v) A one-line rule added to the LEDGER header: every future row names its instrument.

## Out of scope

Deciding whether the imprint lane has an edge — that question cannot even be asked until this
lands. Changing either harness's behaviour beyond what (iii) needs to pin it.

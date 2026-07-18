> **STATUS (2026-07-18): SHIPPED as `task848-policy-replay-instrument`.** `--operator policy-argmax`
> (default `search` = byte-identical) + `--start/--end` window + on-window `--teacher-floor` added to
> the 835 instrument; only `experiments/835_stage4_eval/` touched, no canon leak. Round-trip (i)
> byte-exact vs imprint (gross −6.409/net@21 −6.778/hist [9,388,3,0,0]); default-search (ii)
> bit-identical (net@21 −43.972…); fail-loud (iii) on malformed date / incoherent teacher-floor combo /
> empty+short panel / wrong arch; selftest 8/8. Medium+ panel + 2 fix re-verifies: 1 BLOCKER (malformed
> `--start/--end` → in-sample contamination) + 2 MAJOR (teacher floor through wrong env; delta emitted on
> partial rollout) + MINORs — ALL fixed and re-verified. **Remainder → planner**: the scheduled real OOS
> run (`--operator policy-argmax --checkpoint runs/832_e5/ckpt_imprint_full --start 2025-09-01 --end
> <latest> --teacher-floor --vector-weight-rebalance`) is planner-owned, NOT launched by the factory;
> verdict → LEDGER 832/E5-oos; a PASS routes to the BENCHMARK sequence only (operator D17 correction),
> not the D-gate ladder. Order-code teacher-floor mapping intentionally out of scope (incoherent combo
> is fail-closed-rejected).

# TASK_848 — Policy-argmax replay instrument (ckpt-in + arbitrary window; the OOS read mechanics)

**Status**: SHIPPED 2026-07-18 (was READY; D17 (2)/Track 2 ruled — operator).
**Size: S. Risk: low** (new eval mode on the 835 instrument, zero training-path code).

## Why
D17 Track 2: benchmark the FROZEN imprint checkpoint out-of-sample (2025-09-01 → latest DB) —
does the imprinted net track its teacher on unseen data? No existing tool can do this: 835
evaluates checkpoints ONLY under the search operator (known-broken for a prior, LEDGER
832/E5-autopsy: −6.92 vs +0.839 same weights), and imprint.py replays ONLY the net it just
trained on its own training window. **Operator correction (binding, D17): this is a BENCHMARK
read, NOT a deployment gate** — a pass qualifies the artifact for the full testing/benchmarking
sequence (canonical scorecard conventions, regime windows, walk-forward) and only then the
4-gate ladder.

## Scope
1. `experiments/835_stage4_eval/eval_vector.py`: add `--operator {search,policy-argmax}`
   (default `search` = byte-identical today). `policy-argmax` = the deterministic direct-replay
   operator already specified in 832's imprint-sanity (initial_inference → per-asset argmax →
   step_vector; no search, no dither) — reuse/lift that machinery, do not reimplement.
2. Window freedom: accept `--start/--end` (same panel loader semantics as imprint.py) as an
   alternative to `--periods-file`, so the OOS window needs no new periods JSON.
3. **On-window teacher floor**: when `--teacher-floor` is passed, also walk the weight-native
   ladder teacher (832's `_make_weight_native_teacher_fn`, same sentinels) on the identical
   decision rows and report `teacher_floor_inenv` per cost tier + policy-vs-floor delta — the
   OOS read is meaningless without the floor measured on the SAME window.
4. JSON schema: existing fields + `operator`, `teacher_floor_inenv`, `corridor_position`.

## Acceptance (gates)
- (i) Round-trip: `--operator policy-argmax` on `ckpt_imprint_full` over the ORIGINAL panel
  reproduces the imprint sanity numbers (net@21 +0.839 ± float noise, hold ~2%, 29L/32S) — the
  instrument agrees with imprint.py's replay before it is trusted anywhere else. Fixture-scale
  leg for the factory; real-panel leg is the planner's scheduled run.
- (ii) Default `search` operator BIT-IDENTICAL (existing result JSONs reproduce).
- (iii) Fail-loud: ckpt/window/flag mismatches (wrong arch, empty panel, warm > rows) abort
  with a message, never emit a silently-degenerate JSON.
- (iv) Selftest 7/7; small panel (correctness lens; fail-closed lens on (iii)).

## Scheduled real runs (planner-owned, after ship)
OOS read: `--operator policy-argmax --checkpoint runs/832_e5/ckpt_imprint_full
--start 2025-09-01 --end <latest DB day> --teacher-floor --vector-weight-rebalance` (sentinels),
tiers {0,21}. Pre-registered read: imprint tracks teacher OOS if net@21 ≥ teacher_floor − 0.10
on the OOS window; verdict → LEDGER row 832/E5-oos; a PASS routes to the benchmark sequence
(operator correction above), a FAIL is an H5 generalization finding (in-sample-only imprint).

## Out of scope
Deployment-ladder promotion (explicitly out — operator D17 correction); scorecard integration
(follow-up row after the OOS verdict); any search-path change (that is TASK_847).

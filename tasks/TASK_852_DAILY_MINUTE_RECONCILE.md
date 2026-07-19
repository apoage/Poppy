# TASK_852 — Reconcile the daily-vs-minute panel divergence (un-gates the binding OOS read)

**Status**: READY 2026-07-19 (LEDGER 850/eqgate — data-integrity finding).
**Size: S/M. Risk: low-medium** (eval loader + diagnosis; zero training-path code; the FIX must
not touch the minute path, which is the canon panel source for every existing verdict).

## Why (LEDGER 850/eqgate)
The TASK_850 equivalence gate FAILED on the real overlap window: closes are EXACT on shared
dates, but only 27/64 assets share dates and the teacher floor moves +0.5414 → +0.7979
(Δ=0.2565 ≫ 0.02). The daily and minute sources build DIFFERENT panels — coverage and/or
bar-timestamp convention — and the inflation direction is consistent with a +1-day alignment
shift (LOOKAHEAD risk). The binding OOS benchmark (832/E5-oos re-run) is gated on this.

## Scope
1. **Diagnose (report first, fix second):**
   - per-asset `ohlcv_daily` coverage windows vs the minute-resample coverage on the overlap
     (why do 37/64 assets share zero dates?);
   - the bar-timestamp convention of each source (minute-resample bucket label vs
     `ohlcv_daily.timestamp` epoch: same trading day? off-by-one?); include `ohlcv_daily.source`
     (`agg1m` vs `kraken_rest`) in the breakdown;
   - a decisive lookahead probe: shift the daily panel by −1 day and re-run the FLOOR leg — if
     Δ collapses, the convention shift is proven.
2. **Fix in the DAILY loader only** (alignment offset and/or coverage mask), minute path
   untouched byte-identical. If the divergence is genuinely coverage (assets absent from
   `ohlcv_daily` pre-2025), the daily panel must reproduce the minute panel's ACTIVE-MASK
   semantics on the overlap, not silently densify.
3. Re-run the equivalence gate: PASS = both legs within tolerance (CLOSE exact-or-rtol,
   FLOOR |Δ| ≤ 0.02). If a residual justified divergence remains (e.g. provably better daily
   coverage), STOP and report — accepting a re-based floor is a DECISIONS row, not a code call.

## Acceptance (gates)
- (i) Diagnosis report in the task file / run log: coverage table + convention finding +
  lookahead-probe result (numbers, not prose).
- (ii) Equivalence gate PASSES on the real overlap window after the fix (both legs), OR the
  task STOPs with the documented divergence and a seeded DECISIONS row.
- (iii) Minute path bit-identical (existing result JSONs reproduce); fixture + selftest green;
  small panel (fail-closed lens mandatory — this task exists because a fail-closed gate fired).

## Scheduled real run after PASS (planner-owned)
The binding OOS benchmark re-run: `--panel-source daily --operator policy-argmax
--start 2025-07-03 --end 2026-06-30 --teacher-floor --vector-weight-rebalance` → replaces the
preliminary 832/E5-oos row (BENCHMARK read only — operator D17 correction stands).

## Out of scope
Minute-data backfill (separate ops decision); any training-path change; D19/TASK_851 (the leaf
fix — independent track).

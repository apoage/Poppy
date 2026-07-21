# TASK_852 — Reconcile the daily-vs-minute panel divergence (un-gates the binding OOS read)

> **STATUS (2026-07-21): STOPPED at a residual data-integrity finding — gates on DECISIONS D20
> (OPEN).** Diagnosis complete (below); the densification fix closes 78% of the original gap
> (0.2565→0.0555) and is verified (CLOSE leg PASS, minute path byte-identical, selftest green), but
> the FLOOR leg still fails its own ±0.02 tolerance on the residual. Per this task's own
> acceptance (ii) fallback, this is a STOP-and-report, not a forced pass — see D20 for the
> options. Fix code lives uncommitted in `experiments/835_stage4_eval/eval_vector.py` pending
> D20 + a verification panel (deferred to avoid a working-tree collision with a concurrent
> TASK_851 edit to the same file).

## Diagnosis (acceptance i — 2026-07-21)

**Coverage** (64-asset minute-ranked universe, real `trading_data.db`, window
`2021-01-01..2025-09-01`): 27/64 assets have ≥1 real `ohlcv_daily` row in-window; 37/64 have zero.
Of the 27: 19 full-window majors (`source=agg1m`, ~1700-1705 rows), 5 real-but-partial
(listed mid-window: SOL, MATIC, SHIB, AVAX, LUNA), 3 tail-only (`source=kraken_rest`, 414 rows,
2024-07-15→2025-09-01: EUR/USD, GBP/USD, DAI/USD). Per-asset shared-date count between minute and
daily exactly equals each asset's own `ohlcv_daily` row count wherever coverage exists — no
truncation, no per-asset offset.

**Convention finding**: CLOSE leg matches EXACTLY (0.000e+00 relative diff, all 40,665 shared
asset-days) — no date-label offset in the raw data.

**Lookahead-probe result (decisive — REFUTES this task's own Why-section hypothesis)**: shifting
the daily panel −1 day (pull tomorrow's close back) moves FLOOR |Δ| from baseline 0.2565 → 0.2497;
the +1-day control moves it to 0.2590. Neither direction collapses the gap — the "+1-day alignment
shift / lookahead" hypothesis is refuted.

**Root cause (isolated numerically)**: the daily loader forward-filled the 37 zero-coverage
symbols from a nonexistent anchor, producing fake flat-zero columns that silently densified the
panel. Coverage-symmetric isolation (same 27 real symbols, same `n_assets=27` both panels): minute
floor +0.7608 vs daily floor +0.7053, |Δ|=0.0555 — **78% of the original 0.2565 gap is this
densification bug** (0.2565→0.0555).

**Fix implemented** (daily loader + gate orchestration only; minute path untouched byte-identical),
`experiments/835_stage4_eval/eval_vector.py`: `load_daily_prices()` gets an active-mask filter
(new `_daily_coverage_ids()` helper) that drops zero-coverage symbols instead of densifying them;
`run_equivalence_gate()`'s FLOOR leg intersects the comparison universe against real daily
coverage before pinning `args.symbols`/`args.n_assets`, so both FLOOR panels build on the
identical coverage-safe set.

**Gate re-run** (real DB, run twice, byte-identical both times): CLOSE leg PASS (27/64 shared,
0.000e+00, tol 1e-4). FLOOR leg: ACTIVE-MASK excludes the same 37/64 zero-coverage symbols from
both panels; daily=+0.7053, minute=+0.7608, |Δ|=0.0555 (tol ≤0.02) → **FAIL**, exit 2, no artifact.
Minute path byte-identical pre/post fix (`--smoke` net@21 −43.972, gross −17.697, finalPV 96935,
pct_hold 64.7%, verified twice). `python -m trader selftest`: all checks PASS both times (count
includes TASK_851's concurrently-added ST9/ST10, unrelated to this task).

**Residual not fully explained** (~0.0555, 2.75× tolerance) even on the identical-universe
comparison. Leading unresolved candidates (not isolated further — see D20): (1) row-count
mismatch T=1704 (daily, calendar-day union) vs T=1701 (minute, positional `[::1440]` stride) on
the identical 27-symbol window are structurally different discretizations; (2) small intra-series
gaps within a couple of the 27 covered majors (BCH/USD, XRP/USD: 1703 rows vs 1705 for XBT/ETH),
forward-filled with stale closes on gap days. → **DECISIONS D20 (OPEN)**.

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

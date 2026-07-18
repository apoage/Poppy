# TASK_850 — Daily-table panel source for the eval instruments (unlocks the real OOS benchmark)

**Status**: READY 2026-07-18 (planner seed under D17 Track 2 / D18 ops note).
**Size: S. Risk: low** (panel LOADER only, eval instruments; zero training-path code).

## Why (LEDGER 832/E5-oos)
The OOS benchmark is data-starved by the LOADER, not the DB: `build_panel` (835/848, mirroring
imprint.py's `load_training_panel`) resamples the MINUTE table (`ohlcv`), whose 64-asset
coverage dies ~2025-10-03 (XBT/USD minute ends 2025-12-12; only 146 assets past Oct 15) — while
**`ohlcv_daily` is current to 2026-06 for 376 assets**. The 10-month OOS panel exists; the
instrument can't reach it. Resampling 65M minute rows to daily is also plain waste (CLAUDE.md
efficiency rules) when a daily table exists.

## Scope
1. `--panel-source {minute,daily}` on the 835/848 instrument (default `minute` = byte-identical
   today). `daily` reads `ohlcv_daily` directly (schema: asset_id/timestamp/OHLCV), same symbol
   selection, same warm/decision-row semantics, valid ONLY with `--bar-minutes 1440`
   (fail-loud otherwise).
2. **Equivalence gate (the trust anchor)**: on the overlap window (e.g. 2021-01-01→2025-09-01),
   `daily` and `minute` panels must agree — per-asset close series equal within float/aggregation
   tolerance and the teacher-floor walk net@21 within ±0.02 of the minute-source number (the
   830 corridor floor +0.424 is the reference). Any real divergence = data-integrity finding,
   STOP and report (do not paper over).
3. Optional propagation to imprint.py is OUT of scope (training panels stay minute-sourced —
   no training-path change).

## Acceptance (gates)
- (i) Equivalence gate above (fixture leg on the fixture DB's daily slice; real leg = planner's
  scheduled run, same discipline as 846/847).
- (ii) Default `minute` source BIT-IDENTICAL (existing result JSONs reproduce).
- (iii) Fail-loud: `daily` + non-1440 bar, missing table, empty window.
- (iv) Selftest green; small panel (correctness + fail-closed lenses).

## Scheduled real runs (planner-owned, after ship)
Re-run the 832/E5-oos read with `--panel-source daily --start 2025-07-03 --end 2026-06-30`
(expect ~240 OOS decisions vs the thin 31) → binding OOS verdict to LEDGER (replaces the
preliminary row's claim; still a BENCHMARK read only — operator D17 correction stands).

## Out of scope
Deployment-ladder anything; minute-data backfill (separate ops decision — the fetcher stopped
mid-Oct 2025 for most assets); training-panel changes.

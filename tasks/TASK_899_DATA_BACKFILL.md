# TASK_899 — data backfill to present: catch up, then make pulling a habit

**Status**: READY 2026-08-19 (planner; seeded by the pathfinder audit
`docs/pathfinder_audit_2026-08-19.md` §B G6, operator-adopted same day).
**Size: XS/S. Risk: LOW** — the 858 backfill machinery exists and is proven; this is running it
forward, plus the checks.
**Planner/server ops row, not factory.** Runs on gpu-server.

## Thesis

Data rot is now the binding constraint on any future OOS claim: the canon minute panel ends
2026-03-31, the daily table ends 2026-07-04, and today is 2026-08-19 — every "recent" window in
the record is ≥4.5 months stale on minute bars. Operator's framing (2026-08-19): data only
accrues forward — we cannot will future data into existence — so this row is **catch-up to
present, then a pulling habit**, nothing more mystical than that.

## Scope

1. **Minute canon:** pull Kraken dumps 2026-03-31 → present via the 858 machinery, with the 877
   alias fold (XBT→BTC) applied at ingest, per-asset coverage gap-checked against 858's
   conventions, and a diff report before any existing row is touched (append-only by default).
2. **Daily table:** same, 2026-07-04 → present.
3. **Record the new reachable windows:** the artifact names what OOS spans now exist that did
   not before (notably: 892's binding OOS leg can extend past 2026-03-31, and 898's "recent"
   claim gets a fresh table end).
4. **The habit:** a documented, one-command re-pull invocation (manifest + cron-sketch, not a
   running cron — scheduling is the operator's), so the next backfill is a habit, not a row.

## Acceptance (gates)

- (i) Per-asset coverage report: new rows contiguous with the old at the seam (no gap, no
  overlap) for every canon asset; migrated tickers named (the 861 FTM/MATIC precedent).
- (ii) Zero pre-existing rows modified without a printed diff and an explicit flag.
- (iii) A spot-check: three assets × three recent days against the venue's public data,
  hand-verified.
- (iv) The artifact states the new table end dates — every downstream "recent window" claim must
  quote them.

## Out of scope

Re-running any verdict on the new data (that is each measurement row's own choice, recorded in
its own artifact). Any schema change to the canon tables.

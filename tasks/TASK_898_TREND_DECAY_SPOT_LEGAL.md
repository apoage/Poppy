# TASK_898 — trend-decay aggregation + the spot-legal variant: is the fallback still a fallback?

**Status**: READY 2026-08-19 (planner; seeded by the pathfinder audit
`docs/pathfinder_audit_2026-08-19.md` §B G3, operator-adopted same day).
**Size: S. Risk: LOW to run, HIGH in consequence** — `docs/pathway.md:106` says the fixed rule
"stays the fallback shippable at every rung"; the record says that is currently FALSE, scattered
across three rows nobody aggregated.
**Eval-only on the existing daily table. Runs on gpu-server per D54 (2026-08-20):**
workstation-only inputs (the 819 broad panel npz / daily-table extract) are STAGED to srv as
part of the launch (byte-verified, read-only); if a specific input cannot be staged, the launch
note says why and runs data-local as the documented exception.

## Thesis

The program's only positive-expectancy artifact is the 819 trend rule — and its edge is measured
DEAD in the most recent data, but the measurement is scattered: 875's fold floors decay +3.90 →
+1.43 → +1.40 → +0.26 → +0.42 (2023H2→2025H2, monotone-ish); the 861 scorecard's OOS floors are
negative (oos_full **−0.818**, wf_q1 **−2.806**); 820's gate-2 on truly unseen
2025-12-13→2026-07-04 read net@8 **−1.31** CI[−4.45,+1.29], beating B&H only by avoiding the
bear. No row aggregates this, and a **long-flat / no-short spot-legal variant** — the rule's
edge is concentrated in crash avoidance ("beats B&H by shorting the 2022 crash"), and shorting
was never spot-possible on this venue anyway — has never been pre-registered or measured.

## Scope

1. **Aggregate the decay** into one graded measurement: the 819 config re-scored on the daily
   table from its original window forward through the table's present end (2026-07-04), with
   per-year and rolling-window net@8/net@21, same seed/bootstrap conventions as 893's instrument
   (unmodified `trader.backtest.cell_metrics`).
2. **Pre-register and measure the LONG-FLAT variant**: identical rule, shorts replaced by flat;
   same windows, same scorer. (This is also the only variant deployable without perps.)
3. Report both against B&H and against the cash floor, per regime, with CIs.

## Pre-registered bands (verdict → LEDGER `898/trend-decay` → pathway.md:106 rewritten by the planner)

- **FALLBACK STANDS** — the full rule's recent-data edge is statistically alive ⇒ pathway.md:106
  stands, with the decay curve attached.
- **FALLBACK DEAD, VARIANT LIVES** — the full rule is dead but the long-flat variant clears ⇒
  pathway.md:106 is rewritten to name the VARIANT as the fallback (and the vector line's bar
  inherits the variant's number, not +0.97).
- **BOTH DEAD** ⇒ pathway.md:106 is struck; the program records that it currently has NO
  positive-expectancy fallback, which reprices every "measured against the rule" comparison in
  the ledger.

## Acceptance (gates)

- (i) The scorer is the unmodified `trader.backtest.cell_metrics` (893's discipline); seed and
  bootstrap recorded; `python -m trader selftest` green.
- (ii) Every number cites its window and data epoch; the daily table's end date is asserted in
  the artifact (data rot is G6 — a "recent" claim on a stale table is a named failure mode).
- (iii) The variant is pre-registered before its first score is read (the artifact carries the
  spec hash).

## Out of scope

Changing the 819 rule's parameters (that is tuning, not measurement). The +0.97 bar's
survivorship tilt (caveated in pathway.md by the 2026-08-19 record corrections; a full PIT
re-price of the vector-line floors is the pathfinder's G2, not this row). Sequenced AFTER
TASK_899 if the daily table is to be extended first — otherwise runs on the table as it stands
and says so.

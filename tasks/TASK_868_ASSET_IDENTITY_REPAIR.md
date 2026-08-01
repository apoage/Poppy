> **[AUTHORIZED 2026-08-01 — D31 RULED by the OPERATOR: opt (a) as amended. Attribution
> resolved: the "unattributed actor" was the planner session writing while the factory was
> armed; the orchestrator's quarantine was correct and is preserved in the D31 row. This is now
> a QUEUED task. The evidence was independently confirmed by the run-23 orchestrator (identity
> split; twins agreeing to max rel close diff 0.0 over 125,737 overlapping rows).]**

# TASK_868 — Asset identity repair + interior frozen-cell mask (D31 ruled opt (a)-as-amended)

**Status**: READY 2026-08-01 (D31 RULED by the OPERATOR — opt (a) as amended).
**Size: M. Risk: medium** (touches the canon unpinned minute loader — every minute-panel number
re-bases, which is the ruling's point; nothing downstream may treat pre-D31 numbers as bars).
SAME-SURFACE constraint: `experiments/835_stage4_eval/` + `mcts_trader/env/vectorbt_env.py` +
`mcts_trader/training/env_factory.py` + the smoke suites — SERIAL after TASK_866/867.

## Why (DECISIONS D31 — run-23 panel finding + planner-measured amendment)

Run 23 proved interior frozen spans are tradeable and fabricate PnL (re-anchor jumps to +64.52%;
synthetic ±25%-of-PV repro through the real loader), gating the 861 chain a fourth time. The
planner's follow-up measurement changed the diagnosis for 6 of the 11 slots: they are an INGEST
IDENTITY SPLIT, not missing data — twin `assets` rows under spelling variants (`USDTEUR` id 1144,
`USDTCHF` 1172, `USDCUSD` 1145, `USDCGBP` 1160, `USDCUSDT` 1153, `USDTAUD` 1176; added
2025-12-12) carry 128k–204k minute rows spanning exactly 2025-01-01→2025-10-01, and in the
124k-row overlap with the originals the closes agree with **max relative diff 0.00** (same feed,
split key). 17/64 frame slots have duplicate asset rows (11 empty stubs + these 6); 55 normalized
duplicates table-wide. The genuine sparse case is 5 slots (`ETHBTC`, `BTCGBP`, `BTCCHF`,
`BTCCAD`, `XRPBTC` — illiquid crosses, 446–1,139 minute prints over the 183-day span), ≈1% of the
decision grid.

## Scope

1. **Alias union (a1)**: at load time, resolve each requested symbol to the set of `assets` rows
   whose normalized form matches, and union their `ohlcv` rows into one series. **NO DB writes**
   — `trading_data.db` stays read-only source of record and the fix must work identically on the
   srv row-synced copy. Fail-loud agreement gate: where aliased series overlap in time, any close
   disagreement (rel > 1e-12) on a shared timestamp = REFUSE with both `asset_id`s and the first
   disagreeing timestamp named — never average, never prefer silently. Duplicate timestamps with
   AGREEING prices dedupe cleanly. Alias resolution is DISCLOSED in the artifact (per-slot:
   `asset_ids` used, per-id row counts). Applies to BOTH the framed loaders and the canon
   unpinned minute path (`vectorbt_env.py:645-655` region) — the split slots' coverage ranking is
   itself distorted today.
2. **Interior frozen-cell mask (a2)**: cells whose slot did not print anywhere inside the
   feature bar are force-HOLD via the shipped `price_valid` mechanic — derived CAUSALLY
   (staleness computed from data up to and including the decision bar; no look-ahead into
   whether the gap later resumes), uniform on the pinned AND canon unpinned minute loaders.
   Post-repair this bites ≈1% of cells on the 5 sparse slots. The D30 forward-fill stays for
   VALUATION (PV-exact); the mask only gates OPENING — the D30 PV-conservation smoke leg must
   stay green.
3. **Table-wide duplicate sweep**: one read-only pass over `assets` — report all 55 normalized
   duplicates (id pairs, row counts, span, overlap agreement), fix nothing beyond scope 1.
   Report lands in the wrap + a committed `runs/`-side JSON.
4. **Fixture**: `fixtures/make_fixture.py` gains (i) a twin-identity pair with an orphaned span
   and (ii) a genuine multi-bar interior hole (adopting run-23 rec (C)) so both mechanisms are
   fixture-provable forever.

## Acceptance (gates)

- (i) On the leg-1 window + frame: the 6 split slots read as REAL data across
  2025-03-31→2025-10-01 (frozen-interior cell count drops from 2,013 to the 5-slot residue,
  ≈1%); the disclosure names the alias ids used. Coverage read re-run and recorded (the honest
  inert share moves from 10.67% accordingly).
- (ii) Agreement gate demonstrated BOTH ways on the fixture: agreeing twins union cleanly;
  a tampered disagreeing overlap REFUSES with ids + timestamp named.
- (iii) Mask causality: a probe proving the mask at decision t uses only data ≤ t (e.g. the mask
  for a gap that resumes at t+k is identical whether or not rows after t exist in the DB copy).
- (iv) PV conservation stays exact (the D30 first-class leg, both books, to the cent) with mask
  ON; the pre-fix fabricated re-anchor repro now nets 0 on the masked slot.
- (v) All suites green (selftest, 863, 865, 861 e2e, scorecard); byte-identity claims are NOT
  expected for minute-panel numerics (this task re-bases them — that is disclosed, per-field
  diffs shown); daily paths bit-identical.

## Out of scope

DB surgery / merging `assets` rows (operator-level decision, not needed by the loader fix);
`_to_epoch` UTC + daily teacher-floor mask (TASK_869, same wave); re-measuring the references
(planner-scheduled 861 leg 0 after 868+869 ship); true liquidation (parked, D30 opt (ii)).

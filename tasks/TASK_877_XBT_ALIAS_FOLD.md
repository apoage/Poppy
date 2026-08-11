# TASK_877 — The alias union misses the XBT↔BTC split: 92,639 real rows still unread

> **STATUS (2026-08-11): SHIPPED as `task877-xbt-alias-fold`** — see the CHANGELOG entry. 3 cycles,
> 5 lens-runs, 3 MAJORs closed and re-verified, 0 BLOCKER/MAJOR outstanding.
>
> Gate (i): BTCCHF's gap-window coverage 667 → **92,637** (UTC); its frozen-interior cells for that
> span **182 → 0**, exact. Program-wide frozen-interior **1,012 → 830**, NOT-REAL **9.7334% → 9.5635%**;
> genuine sparse set 5 → **4**. Gate (ii) demonstrated both ways, mutation-proven. Gate (iii): every
> substring trap (`AIXBT/*`, `XBTPY`, `WBTC`, `TBTC`, `XXBTZ*`) stays split, verified against real ids.
> Gate (iv): all suites green. **Scope 4 answered: the fold does NOT change the coverage ranking** —
> zero symbols enter or exit the top-64 set in either window.
>
> **Two corrections to this task file, both by re-measurement:** (1) the XDG→DOGE collision is not
> "stub-only" — there are **zero** collisions; no DOGE-spelled row exists in either DB. (2) The 697-row
> figure above is naive-LOCAL time; under UTC it is 667. The reconciliation is stated inline in
> `asset_aliases.py`, and `harvest_gate_i.py` — which produced the local-time number — is now migrated
> to UTC (the TASK_869/D31(a3) class surviving in an unmigrated diagnostic).
>
> **Scope grew beyond the file in one respect, deliberately:** the gate had no minimum-evidence floor,
> so a 0- or 1-shared-timestamp merge was blessed as confidently as a 48,780-timestamp one.
> `MIN_ALIAS_OVERLAP_TS = 64` now enforces it — judged on the pair's whole history, because a
> per-window floor was measured to refuse **seven already-shipped folds**.
>
> **Known remainder, disclosed not fixed:** the daily loaders never apply the alias union at all
> (pre-existing to 868). Live instance: frame slot 11 `/USDC/USD` (id 996, 0 daily rows) reads
> INACTIVE while its twin `USDCUSD` (id 1145) holds 721 days. Wiring them moves daily-pinned
> numerics → separate row.

**Status**: READY 2026-08-05 (planner-measured; same defect class as D31 (a1), which 868 fixed only
for slash-normalization). **Size: XS/S. Risk: low-medium** (touches the shipped alias resolver, so
minute-panel numbers move for the affected slot — disclosed, per-field diff).
Surface: `mcts_trader/data/asset_aliases.py` (+ the disclosure fields and the coverage numbers).

## Why (measured 2026-08-05, read-only)

`asset_aliases.py` normalizes **slash-only** and its docstring records that it deliberately does
**NOT** fold `XBT`→`BTC`. That caution has a measurable cost: Kraken's `XBT` *is* `BTC`, and one
collision carries real data on both sides.

| asset | id | rows | span | rows inside 2025-03-31 → 2025-10-01 |
|---|---|---|---|---|
| `BTCCHF` (frame slot 23) | 1085 | 979,758 | 2019-12-06 → 2026-03-31 | **697** |
| `XBTCHF` | 1175 | 140,750 | 2025-01-01 → 2025-09-30 | **92,639** |

Overlap: **48,780 shared timestamps, max relative close difference 0.0** — the same signature that
identified the six twins in D31 (a1). So `BTCCHF`'s "genuine sparse gap" is not a gap in the
market; it is the same instrument logged under a second spelling, unread by the resolver.

Consequence: the D31 (a1) repair is INCOMPLETE. Of the 5 slots recorded as genuinely sparse
(`ETHBTC`, `BTCGBP`, `BTCCHF`, `BTCCAD`, `XRPBTC`), **`BTCCHF` is recoverable**, so the genuine
set is 4 and the frozen-interior residue is smaller than the 0.94% now recorded. This also means
`XBT`-spelled data may be missing from the coverage RANKING wherever both spellings exist.

## Scope

1. Extend the resolver's normalization to fold the known Kraken ticker aliases — at minimum
   `XBT`→`BTC`, and check `XDG`→`DOGE` (collision found, but stub-only: no rows on the XDG side)
   — as an explicit, documented alias TABLE, not a blanket string rewrite. The existing
   fail-loud price-agreement gate must apply unchanged to every newly folded group: any
   disagreement in overlap is a REFUSE naming both ids, never an average.
2. Sweep for the same class: report every `assets` collision that is a collision only under a
   ticker-alias mapping and carries rows on BOTH sides (today: exactly one, `BTCCHF`/`XBTCHF`).
   Any additional hit gets named in the wrap, fixed if XS.
3. Re-derive the disclosure numbers the fold changes: frame-slot coverage, `frozen_spans`,
   frozen-interior cell count/pct, not-real share, `n_aliased_slots`. Per-field diff attributed
   to this task (the 864 promote gate is the write path for tracked baselines).
4. State plainly in the wrap whether folding changes the coverage RANKING of the canon unpinned
   universe (it may, since the split halves a series' apparent history).

## Acceptance (gates)

- (i) `BTCCHF` reads XBTCHF's rows: its gap-window coverage goes from 697 to ~93k, and its
  frozen-interior cells for that span go to ~0. Numbers shown before/after.
- (ii) Agreement gate demonstrated on a folded group both ways (agreeing folds cleanly; a
  tampered overlap REFUSES with both ids + first disagreeing timestamp named).
- (iii) The alias table is explicit and tested — a symbol that merely CONTAINS "xbt" as a
  substring of something else must not be folded (no blanket `.replace`).
- (iv) All suites green; every moved numeric attributed; daily paths unaffected or disclosed.

## Out of scope

The 4 remaining genuinely-sparse slots (illiquid crosses — real market absence, nothing to
recover); the frozen-cell mask policy itself (see the planner note in QUEUE — that needs an
operator ruling, not a fix); the walk-forward battery now running, which used the pre-fold panel
and carries `BTCCHF` as sparse — a disclosed limitation affecting 1 of 64 slots, not a corruption.

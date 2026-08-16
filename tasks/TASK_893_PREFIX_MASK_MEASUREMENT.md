# TASK_893 — measure D41: how many decision cells sit in leading bfill prefixes, and does the bar move?

> **STATUS (2026-08-16): SHIPPED as `task893-prefix-mask-measurement`.** Instrument
> `tools/measure_leading_prefix.py` + the `893/prefix-cells` LEDGER row. 2 cycles; verified by a
> 3-lens panel then a 2-lens re-verification. **`trader/` source deliberately UNMODIFIED — D41 stays
> OPEN and this row only supplies its evidence.**
>
> **Numbers for the ruling:** 4933 prefix rows / 39 assets · **4904 decision cells in FULL = 7.3793%**
> (4518 on the scored grid) · concentrated BULL21/BEAR22/REC23, none after 2023 · masking every prefix
> cell inactive moves the bar by **exactly 0.000000** (net@21 `+0.9673047900277524` both sides) · **0**
> as-shipped positions ever land inside a prefix.
>
> **Read the Δ=0 correctly — this is the load-bearing caveat.** It is a STRUCTURAL COROLLARY of
> `present_mask`'s `warm=30`, not an independent test: returns inside a bit-identical prefix are exactly
> zero, so `first_active` can never precede the prefix boundary. The instrument IS responsive (a planted
> position moves it `+0.000749`; that positive control ships as a permanent regression test that fails
> loudly if it ever goes vacuous), but **a D41 ruling resting on "0 measured exploitability" alone would
> be circular.** Any consumer reading `close_d`/`logc` WITHOUT `present_mask` gets no protection and this
> instrument would not catch it.
>
> **What survives regardless of which D41 option is taken:** the `_ffill_bfill_cols` docstring's
> *"no cross-time leak"* claim is **DISPROVEN** at second-level resolution — SOL's row 167 carries a price
> from **59,760 s in its future**, XDG's row 0 from 420 s. That is measured, not argued.
>
> **Remainder → planner:** (1) rule D41 on this evidence; (2) a SECOND defect found here is unruled and
> outside this row — `build_close_aligned` (`runs/819_single_asset/portfolio.py`) admits only
> `timestamp >= grid_start`, so 16 majors with real earlier history still get a 1-row backward fill
> (this moved the count 4917 → 4933; all 16 land on row 0, outside FULL, so no scored number moves).
> It sits in a planner-owned prototype and was deliberately not touched.

**Status**: READY 2026-08-16 (planner/orchestrator; the measurement D41 pre-demands before any ruling).
**Size: XS. Risk: read-only** — a diagnostic script and a re-run of existing numbers; `trader/`
source is NOT modified (the candidate fix is D41's call, after the measurement).
**Parallel-safe**: `trader/` + `tools/` surface only — zero overlap with every other Ready row.
**Resolves the evidence half of DECISIONS D41** ("whichever way this goes, it needs a measurement,
not an argument").

## The defect being measured (not fixed)

`trader/data.py:97` `_ffill_bfill_cols` forward-fills, then sets every row BEFORE a column's
first valid print to that first price — a backward fill in a data path, under a standing
operator rule of forward-fill only, with a docstring claiming "no cross-time leak". It is
bounded (only the first price travels backward; returns inside the constant prefix are exactly
zero) but it is what it is, and it sits in the one lane with a measured marginal edge and an
open paper-live gate.

## Scope

A new diagnostic script (`tools/measure_leading_prefix.py` or `experiments/893_prefix_cells/` —
implementer's pick, new files only) that, on the canonical daily panel the 819/820 lineage uses:

1. **Counts**: per asset and in total — how many panel rows precede the asset's first valid
   print, how many DECISION cells that is, and what share of the traded window they are.
   Report per-decade/regime too: if prefixes concentrate in the early window the finding
   distributes differently than if they are uniform.
2. **Re-measures**: the 819 linear-book bar number (net@21 Sharpe **+0.97** and its components)
   twice on identical code — (a) as shipped, (b) with leading-prefix cells masked INACTIVE
   (positions forced flat there, no PnL attribution) instead of constant-filled. Report the Δ
   with the same seed set the bar was published on. The 820 paper-trading config number too if
   it is reproducible from committed artifacts.
3. **Answers the exploitability question with data**: does any held position's ENTRY fall inside
   a leading prefix? (Returns there are exactly zero, so a strategy cannot profit from the fill
   directly — but an entry timed on a constant series' "signal" is the channel to rule out.)

## Acceptance (gates)

- (i) The script runs against `fixtures/trading_fixture.db` AND the real panel (planner executes
  the real leg; the script itself must take `--db` and default to the fixture).
- (ii) Counts are cross-checked two ways (independent computation paths) and agree exactly.
- (iii) The masked-variant re-run touches nothing in `trader/` — masking happens inside the
  diagnostic's own panel copy.
- (iv) Output = one JSON + one paragraph in the LEDGER (`893/prefix-cells`), and the numbers
  needed to rule D41 (a)/(b)/(c) are all present.

## Out of scope

Any change to `trader/data.py` (the docstring correction, the masking fix, or the leave-it
ruling are D41 options, decided AFTER this measurement). Any change to mcts_trader/ (verified
bfill-free already). Re-basing any published bar — if the number moves, that is D41's ruling,
not this row's.

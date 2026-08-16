# TASK_893 — measure D41: how many decision cells sit in leading bfill prefixes, and does the bar move?

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

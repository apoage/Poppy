# TASK_895 — execute D41(a): kill the false docstring, fix the 819-prototype prefix fill

**Status**: READY 2026-08-16 (planner; D41 operator-ruled option (a) 2026-08-16).
**Size: XS. Risk: LOW** — a comment correction + a one-line alignment fix in a prototype.
**Parallel-safe FILLER**: surfaces are `trader/data.py` (docstring only) and
`runs/819_single_asset/portfolio.py` (prototype) — zero overlap with every other Ready row.

## Two items, both from the D41 ruling

1. **Correct the `_ffill_bfill_cols` docstring** (`trader/data.py:97`). The claim *"Uses only
   same-series past/adjacent bars (no cross-time leak)"* is **DISPROVEN by measurement** (TASK_893,
   LEDGER `893/prefix-cells`: SOL's row 167 carries a price from 59,760 s in its future; XDG's
   row 0 from 420 s). The replacement must say what the function actually does — forward-fill,
   then set rows before a column's first valid print to that first price — and name the
   protection that makes it safe today: consumers routing through `present_mask` (`warm=30`)
   never trade those rows (0 positions ever land inside a prefix, measured), and **a consumer
   reading `close_d`/`logc` raw gets no such protection**. No behaviour change; the ruling is
   explicit that the behaviour stays.
2. **The 819 rider (D41's rider, ruled its own row here):** `build_close_aligned`
   (`runs/819_single_asset/portfolio.py`) admits bars only where `timestamp >= grid_start`, so
   16 majors WITH real earlier history still receive a 1-row backward fill (all land on row 0,
   outside FULL — no scored number moves, measured). Admit the earlier history so the first
   grid row is a REAL bar, or mask row 0 for those 16 — implementer's choice, but the fill
   disappears or is masked, and the 893 instrument's count is re-run to prove it
   (4933 → expected 4917).

## Acceptance (gates)

- (i) The new docstring's every sentence is checkable against the code; a reviewer can verify
  each claim mechanically.
- (ii) `python -m trader selftest` green; 819/820 numbers byte-identical (behaviour unchanged).
- (iii) The 819 fix: `tools/measure_leading_prefix.py` re-run shows the 16-asset 1-row fills
  gone (count 4933 → 4917), and the bar still reads net@21 `+0.9673047900277524`.
- (iv) 893's positive-control regression test (planted position moves the number +0.000749)
  stays green.

## Out of scope

Any change to the fill BEHAVIOUR of `trader/data.py` itself (D41 ruled (a) — keep it). Any
masking of prefixes in `trader/`. Re-grading historical 819/820 numbers.

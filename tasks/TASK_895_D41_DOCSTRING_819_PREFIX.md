# TASK_895 — execute D41(a): kill the false docstring, fix the 819-prototype prefix fill

> **STATUS (2026-08-17): SHIPPED IN PART as `task895-d41-docstring` — item 1 only.**
> **Item 1 (the docstring) SHIPPED.** The disproven "no cross-time leak" claim is gone; the
> replacement states the actual mechanism (forward-fill, then a BACKWARD fill of the leading gap),
> cites 893's measured disproof, names the `present_mask` warm=30 protection AND the honest limit
> that a raw `close_d`/`logc` reader gets none of it, and preserves D41's load-bearing caveat that
> the Δ=0 is a STRUCTURAL COROLLARY, not a general fence. Comment-only diff; selftest 12/12,
> parity GATE 1a PASS, 893 positive control green (+0.000749).
>
> **Item 2 (the 819 `build_close_aligned` rider) NOT SHIPPED — gate (iii) is INTERNALLY
> UNSATISFIABLE. Escalated to DECISIONS `D44` (OPEN).** Both halves verified first-hand, not taken
> on report: (α) `tools/measure_leading_prefix.py:118` hardcodes `_ADMIT`, a verbatim copy of the
> OLD admission SQL, and Paths A/B plus the value anchor all query raw `ohlcv` through it — the
> instrument never reads the panel, so NO edit to `runs/819_single_asset/portfolio.py` can move the
> 4933 count (the implementer applied the fix, regenerated the panel, re-ran the tool, got 4933
> bit-for-bit). (β) row 0's fabricated value EQUALS the first admitted close, which is exactly why
> it is invisible to log-difference signals (`logc[t]−logc[0] == logc[t]−logc[1]`); admitting the
> real earlier price necessarily breaks that identity for every lookback reaching row 0 from a
> scored row, measured net@21 `+0.9673047900277524 → +0.9672495154672209`. Masking does not escape
> it either — masking suppresses POSITIONS while the fabricated PRICE stays in the signal input.
> **Correction this forces on the D41 record: 893's Δ=0 held because the fill is VALUE-NEUTRAL, not
> because row 0 is unreached. Row 0 IS read, by every lookback spanning it.**
> Tree verified: `portfolio.py` byte-identical to HEAD, frozen panel `_close_broad_daily.npz`
> untouched (sha256 `7017e6c3…` AND mtime still 2026-06-28).

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

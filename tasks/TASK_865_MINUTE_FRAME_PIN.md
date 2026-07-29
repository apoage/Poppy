# TASK_865 — Frame pin on the MINUTE path (D29 ruled opt (a): the only basis comparable to +0.839 on universe, source, and breadth at once)

**Status**: READY 2026-07-29 (D29 RULED — planner, opt (a); operator may override).
**Size: S/M. Risk: medium-low** (a new framed loader on a proven pattern; both existing pin
paths — 853 daily in `eval_vector.py`, 863 in `imprint.py` — stay untouched in behavior when the
new path is not selected).

## Why (DECISIONS D29 — measured during factory run 21)

The 861 real-leg chain is blocked a second time, for a precise reason: TASK_863's pin is sound
but its only available loader is DAILY (`load_daily_prices_framed`/`create_daily_panel_framed`),
and `ohlcv_daily` covers **33/64** frame slots on the leg-1 window (31 slots zero rows → forced
HOLD all window, ~48% of decision cells structurally inert), while the 858 backfill brought the
MINUTE table to **62/64** for the same frame. The +0.839 bar is minute-panel-native with a
64-asset book (frame provenance `source`: "835 minute-panel slot order"; LEDGER 832/E5-imprint:
29L/32S, 97.5% agreement). Pinning through daily therefore substitutes a SOURCE confound (D20:
|Δ|=0.0555 coverage-symmetric; cross-source comparisons RULED forbidden, daily = within-panel
grading only) plus a **64→33 breadth confound** (first-order: ST3 measures 24-asset breadth at
2.76× single-asset Sharpe; sign vs the bar unestablished) for the 59/64 universe drift D28
removed. A minute-framed loader is the only option that makes leg 3 apples-to-apples on universe
AND source AND breadth simultaneously — i.e. the only one that delivers what D28 opt (a) was
chosen to achieve.

## Scope

1. **Minute-framed loader**: teach the PIN-b `create_jax_env_historical` minute path (the
   `ORDER BY COUNT(*) DESC LIMIT n_assets` derivation at `vectorbt_env.py:579-590` and its
   consumers) to accept the recorded frame — fixed 64-slot panel in the frame's positional
   order over the minute `ohlcv` table, mirroring the SHIPPED 853/863 semantics exactly:
   zero-coverage and pre-listing slots INACTIVE and dual-disclosed (flat-zero features +
   price-validity force-HOLD), fail-loud on identity scrambling, same-membership reorder
   REFUSED, `active_cells_only` companion scalars emitted. Reuse the shipped validator rather
   than re-deriving it.
2. **Wire both consumers**: `eval_vector.py --panel-frame-from-checkpoint` on a minute-bar run
   uses the new loader (today it is daily-only); `imprint.py`'s 863 pin path likewise when the
   run is minute-sourced. Selection must be explicit and recorded in the artifact
   (`panel_source` + frame block); no silent source switching.
3. **Guard the known trap on the real leg's exact configuration**: the 863 panel's MAJOR
   (degenerate pinned run exits 0 with null Sharpes when `--smoke-assert` is off — the default
   for any non-fixture DB) was fixed pin-gated on the daily path; the minute path must carry the
   same degenerate-run detector from day one.
4. **Fixture smoke**: minute-framed collect → imprint → sanity replay end-to-end; reorder
   refused; inactive-slot disclosure asserted (the 863 lesson: disclosures are ASSERTED, not
   just printed); the 861 e2e suite extended to cover the minute-pinned path.

## Acceptance (gates)

- (i) All defaults / unpinned / daily-pinned paths bit-identical to
  `task864-parity-baseline-hygiene` HEAD (the shipped 863 daily pin included); selftest + 861
  e2e green.
- (ii) Fixture smoke per scope 4, including both refusal directions and the asserted disclosure.
- (iii) Coverage honesty: a read-only real-DB check leg (no GPU) prints the frame-slot coverage
  of the minute table on the leg-1 window — expected **62/64** with the two inactive slots
  NAMED (the FTM→S / MATIC→POL ticker migrations); a materially different count is a STOP, not
  a warning.
- (iv) Degenerate-run detector demonstrated to fire on the minute-pinned path with
  `--smoke-assert` off (the real leg's configuration).

## Real leg (planner post-ship — re-registers the 861 chain, third and final framing)

TASK_861 legs 1+2 run minute-pinned on `runs/832_e5/frame_provenance_imprint.json`
(`--bar-minutes 1440` over minute `ohlcv`, 62/64 active slots, 2 disclosed-inactive ≈ 3% of
cells — accepted as nominal vs the daily path's 48%). Leg 2/3 = 3 imprint seeds, median > +0.839
with ≥2/3 seeds above (unchanged from the D28 addendum). Leg 4 (OOS/WF) reuses the same frame
provenance. Per D17: benchmark reads only.

## Out of scope

Grading bars (TASK_861 owns them); the ExIt-continuation frame pin (row D — PARKED until an ExIt
continuation is actually scheduled); baseline/pin hygiene (TASK_866/867); any change to the
daily loaders' behavior.

# TASK_920 — scorer honesty pack (the four corrections the panels verified)

**Status**: PROPOSED 2026-09-05 (planner; four record corrections surfaced by the tactics panel
and verified in primary files — `lore/fable_tactics_panel_2026-09-04.md` §"Record corrections"
+ magnitudes from `lore/harvest_6b32_appendix/tactics_gaps.md`).
**Size XS/S. Risk: MEDIUM-SENSITIVE surface (`trader/backtest.py`) — therefore strictly
flag-gated, OFF = bit-identical, pins mandatory.**

## The four corrections (receipts re-verified by two panels independently)

1. **The "B&H" column is not buy-and-hold** (`trader/backtest.py:225-227`): it is a
   daily-rebalanced equal-weight book charged ZERO turnover. A **true B&H book (fixed shares,
   annual re-form) has never been computed** — add `--bh-book {legacy-label, true-bh}` (B4).
2. **Log-return summation cannot see a rebalancing premium by construction**
   (`trader/backtest.py:221-223`): add `--accounting {log, simple}`; the bias is ≈ +0.3
   Sharpe-equivalent (≈5 bp/day; 1/N −0.40 → ~−0.1 on simple returns).
3. **Constant-dollar drift re-trims are uncharged** (turn counts target-weight changes only;
   estimate +9–12×/yr ≈ −0.03..−0.04 Sharpe at 21 bp on the +0.97 bar): add
   `--charge-drift-rebalance` with the measured decomposition printed.
4. **RiskConfig default-off rows never recorded** (`trader/config.py:108-130`: 5% per-asset
   cap, shadow-scale DD throttle, 0.2% band): one run each on the frozen trend book — closes
   TASK_820's never-written gate-2 overlay number. Expected FAIL on Sharpe (vol-target cuts
   exposure where the book earns); the yield is the missing number itself.

## Gates

- **G1: OFF = bit-identical** on the pinned trend-book read (SHAs recorded) — non-negotiable,
  this scorer grades every row.
- **G2: drift tripwire (pre-registered)**: if charging drift re-trims moves net@21 by
  > −0.10, every prior 819/898 number gets a dated footnote in LEDGER — disclosed, not re-run.
- **G3**: the two lore corrections land: tactics-panel §"Proposed rows" row-1 reversal-arm
  relabel (7d = time-series, only 21d is cross-sectional — already fixed in TASK_918's
  library), and the AS-correction (0.04–0.05, not 0.3–0.5 Sharpe) added to the tactics file's
  record-corrections list where it was dropped.

## Record-fix riders (notes only, no code)

- **819 TEST B2 is the correct receipt for "adaptive forecaster fails"** (per-regime refit
  lost 4/4 OOS regimes −2.34/−2.41/−1.60/−2.09 vs TSMOM; the commonly cited −0.16 is
  survivor-universe net@8, never run PIT) — cite this, not −0.16, going forward.
- **898's spot-legal variant was flattered by cash drag** (gross de-levered to 43%;
  leverage-matched FULL +0.321 → −1.308) — footnote the 898 row.
- **892's anti-edge concentrates in maintenance re-trims** (−0.060/−0.062/−0.050; entries/
  exits NULL) — the receipt any GP-style book's re-trim channel guard cites.
- **893's leading backward-fill** (7.38% of cells, 2021–23): any new featuriser reading
  `close_d` unprotected gets future-stamped prices — add the assert to the shared panel
  builder used by 915/917/918.

## Out of scope

`eval_vector.py` semantics (separate surface, 882-pinned), re-running dead experiments, any
change to a gate number.

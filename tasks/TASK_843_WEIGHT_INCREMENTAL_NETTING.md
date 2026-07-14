# TASK_843 — Weight-path incremental netting + snap-to-flat (D12 cleanup)

**Status**: READY 2026-07-14 — gate SATISFIED (phase-1c battery complete 3/3 + instruments
graded 2026-07-12; D13 ruled). Ranked #1: the env must reach its FINAL fee0 form before
TASK_845's teacher-expressibility gate measures against it. The weight-ON/fee0 pin
`1c50d58c…` RE-BASES with this change — capture fresh pins pre/post (838/842 discipline).
**Size: S–M. Risk: medium** (TASK_840 weight branch only; canon + 839-budget pins unchanged).

## Why
D12 (ratified 2026-07-12): the 840 weight env re-trims by FULL close+reopen each step instead
of netting incrementally against the held position, and `w_cur` uses market notional so
slippage leaves a residual micro-short after a unit-step close. Consequences (panel/orchestrator
repro): a SHORT round-trip at fee16 charges 185.2 vs canon 105.0 (~76% over-tax on short
turnover) + residual dust positions. Conservative (never flatters), but it distorts phase-2+
cost structure and adds noise trades.

## Scope
1. Re-trim legs NET against the held position (canon-style incremental adjust: only the DELTA
   notional trades, with 842's canon-equivalent fees on the closed portion only).
2. Snap-to-flat: |target|=0 with |current| below the deadband closes the FULL residual (kills
   slippage dust; no micro-shorts).
3. Parity: canon + 839-budget pins unchanged; NEW weight-ON/fee0 pin captured pre-edit and
   re-based deliberately (record old `1c50d58c…` → new in the banner).
4. **Acceptance (fixture)**: (i) short round-trip at fee16 within 1e-3 relative of the canon
   equivalent (fixes the 185.2-vs-105.0 gap); (ii) zero residual positions after a
   close-everything action sequence; (iii) 840 acceptance walks re-run green (survival +
   invariants cash≥0 / Σ|w|≤1 / margin-noop); (iv) E1-pattern probe at fee16 reported
   (continuity with the 842 honest edge −0.134 — direction expected to improve).

## Gates
Pins as specified; acceptance (i)–(iv); selftest 7/7; medium panel.

## Out of scope
Action-ladder granularity (H6 — separate evidence stream), proposal changes, deadband retuning
beyond snap-to-flat, deploying to gpu-server while ANY battery runs (planner owns deploy).

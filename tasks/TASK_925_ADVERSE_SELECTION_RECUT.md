# TASK_925 — the 907-B2 re-cut: adverse selection on resting fills (gap-aware markouts)

**Status**: PROPOSED 2026-09-28 (planner, seeded at run 57's close — **GATED on D81(a) +
D87(a): do NOT start before the operator rules**). This is the re-cut vehicle D81's halt
always pointed at, now with run 57's evidence behind it: it is the ONE experiment that
bears on all three cost-capped rows (915 D89, 918 D86, 923 D88).
**Size: M. Risk: MEDIUM — it reopens the halted 907-B2 surface; every D81 safeguard below
is load-bearing. Verify: full.**

## Why this row exists (run 57's synthesis)

Every surviving signal line is cost-capped at the 80 bp taker tier: 915's best G-B survivor
+0.04 vs +0.97; 918's re-optimised hull −1.31 same-close / −0.90 T+1; 923's net CIs straddle
zero. 924 measured the maker side's QUOTING costs (median BTC half-spread 0.0064 bp, quote
life p50 32.5 s) — but what a resting fill COSTS AFTER it happens (adverse selection) is
unmeasured, and that is 907-B2's estimand. D87's recommendation: measure it before any
executor build. D81's halt was about the probe's defect-handling, not the estimand — the
re-cut is the sanctioned path, and run 52 preserved 5596 lines of reusable round-4 work.

## Scope (to be confirmed/refined by the D81(a) ruling)

- **The estimand**: post-fill markouts on resting (maker) fills, gap-aware (stall gaps are
  recorded gaps and EXCLUDED — D74's ruling stands), per pair and hour-of-day, vs the
  pre-committed ≤ 2 bp AS(1h) line that gates any executor build.
- **Reuse first**: the preserved round-4 patch (`lore/handoff_2026-09-05_factory_run_52_
  907B2_round4.patch`, sha `e0d8f5d5…`; apply-check was vs `1cd2c70` — REBASE CHECK against
  current HEAD is cycle-1 work, and 913's write-hardening has since touched `capture.py`'s
  writer path: the read path was verified unchanged at `3461b15`, but the patch context
  will not apply clean by name).
- **Carry 924's fix**: B1's frozen-mid numerator is gap-blind (`capture.py:2422-2436`) —
  the re-cut's gap-aware markouts inherit the gap-aware frozen-mid denominator.
- **The four round-4 BLOCKER shapes stay fixed** (flagged-span keepalive endpoint; MidQuote
  provenance gate; emission-policy detector bypass; D77 interval-scoped exclusion +
  two-sided backstop) — the run-52 halt record named them; the re-cut must not regress any.

## Gates (binary; refined by the ruling)

- **G1**: the four named round-4 fixes are re-proven on the CURRENT capture set (not the
  2026-09-01 snapshot) — including the interval-scoped exclusion with disclosure-not-void
  pair flags and the ≤1.0% / ≤60s backstop.
- **G2**: gap-aware markout tables publish for all pairs × horizons, with the D74 gap
  exclusion visible and the frozen-mid denominator gap-aware (924's 30.42% BTC, not B1's
  43.4% gap-blind figure).
- **G3**: the AS(1h) ≤ 2 bp line is answered with a CI, not a point estimate — and the
  verdict names its power (D85 discipline at spec time).
- **G4**: zero live orders; the 907 tree's recorder/capture behavior UNCHANGED except as
  the ruling's scope names (D67(b)/D81 constraints); no DB writes; srv read-only + wrapped
  jobs only.

## Out of scope (unless the ruling adds them)

Any executor/book build (that is D87(b), explicitly NOT taken); venue expansion; L2 signal
mining (parked, operator's pre-registration call); reopening the B1/B2 halt record beyond
what the re-cut needs.

## Pre-committed consequences

- AS(1h) > 2 bp with a powered CI → maker feasibility measured-dead; the recomposition's
  Option B closes as "taker-only planner"; 915/918/923's cost-capped verdicts stand as
  final for this venue/tier.
- AS(1h) ≤ 2 bp (powered) → the FIRST honest green light the record has produced for any
  execution path; D87(b)'s executor build row opens with numbers, and 806's parked
  minute-IC 0.1226 becomes re-activable.

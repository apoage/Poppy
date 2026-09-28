> **STATUS (2026-09-28): SHIPPED phase A (run 57, tag `task923-listing-book-A`).** UNDERPOWERED at every
> exit (D85, spec-time MDE 470–1573 bp vs +100 bp band). Direction is SHORT (P3 profile). Gross edge
> separates from placebo at h21/h60/age* (pooled-SE z ≈ 2.9–3.2, thin); net at 80 bp/leg + borrow-high
> every CI straddles zero. Capacity median ≈ $3.6k/event; shortability at listing unverified. First
> "NULL" withdrawn (zero-centred criterion defect). Next step → **D88**.

# TASK_923 — post-listing event book, phase A (Option A of the profitability recomposition)

**Status**: PROPOSED 2026-09-28 (planner; operator handed unattended time off
`docs/profitability_paths_2026-09-28.md` — this is its Option A).
**Size S. Risk: LOW compute (CPU-only, reads the existing DB read-only) — HIGH honesty
weight: this is the first BOOK-shaped claim since 916 restated the cost stack, so every
guard below is load-bearing.**
**Verify: full.**

## Why this row exists

917 P3 (shipped `8f411d1`) measured the FIRST placebo-clean positive since 897 closed the
surfaces: post-listing IC **+0.0304**, NW-t **+4.15**, circular-shift placebo clean,
leakage-audited (listing-date artifact refuted, leave-one-out market-factor stable). It was
shipped explicitly as "an instrument reading, not an edge thread". This row converts the
instrument into a **tradability read** at the honest cost stack — or kills it as
cost-negative like everything else. Either outcome is productive; a vague "promising" is
the only forbidden outcome.

## What (phase A — the only phase authorized)

An event-book simulation over the same PIT daily panel 917 P3 used (its
`daily_panel.py` builder, unchanged), same listing-date source, same age windows:

- **Entry**: first PIT-tradable daily bar post-listing (taker — maker entry is NOT claimed
  and not simulated here; that is 924's question, not this row's).
- **Exits, pre-registered BEFORE the first run** (fixed set, no tuning): fixed horizons
  {5, 10, 21, 60} trading days, and age-exit at the P3-measured IC-decay age. Every exit is
  reported; there is no "best" selection — the FAMILY verdict is the max over the
  pre-registered set with Bonferroni across it, stated.
- **Costs**: the 916-verified stack — 80 bp base taker tier round-trip, borrow charged on
  any short leg per 916's model. Also report the legacy 21 bp column, clearly labelled
  NON-BINDING (record-keeping only; no claim may cite it).
- **Estimand**: per-event net markout distribution (bp), mean with event-clustered
  (by listing cohort week) CI, NOT a Sharpe A/B. D85 applies in spirit: at SPEC time,
  compute n_events and the event-level SE from P3's data, then the MDE against the
  pre-registered band (mean net markout > +100 bp round-trip, i.e. beats cost with a
  margin). **If MDE > band, the row refuses its own "tradable" verdict and reports the
  distribution + the re-powering requirement (more exchanges/assets), which is itself the
  deliverable.**
- **Honesty battery (all mandatory)**: circular-shift placebo re-run on the book (not just
  the IC); leave-one-listing-out; capacity disclosure (n events, assumed notional per
  event, liquidity ceiling from first-N-days volume); by-cohort-year table (is the edge
  decaying as listing quality changed 2021→2026?); zero-events and one-event degenerate
  guards fail loud.

## Gates (binary)

- **G1 SPEC-PIN**: the exit family, cost constants, estimand, band, and MDE computation are
  written into `experiments/923_post_listing_book/SPEC.md` with sha256 recorded BEFORE the
  first real-panel run (fixture smoke first per validation tiers).
- **G2 PLACEBO-FAIL-LOUD**: the circular-shift placebo, run through the SAME book code,
  must produce a distribution the real book's CI excludes; if the placebo book matches the
  real book, the row's verdict is NULL (P3's IC did not survive tradability), recorded.
- **G3 COST-TRUTH**: no verdict line may cite a number below the 80 bp + borrow stack.
- **G4 MDE-DISCIPLINED**: the verdict states n, SE, MDE vs band explicitly; an underpowered
  PASS is reported as UNDERPOWERED, never as tradable.

## Out of scope

Maker/limit entry (924's feasibility numbers gate that); multi-venue listings (Kraken-only
data); any portfolio/Sharpe framing; any live trading; any DB write. Phase B (if G1–G4 all
pass AND 924 says maker entry is feasible) is a separate row.

## Consequences (pre-committed)

- G2 NULL or cost-negative at the honest stack → Option A of the recomposition closes as
  "instrument-positive, book-negative" — recorded, no second attempt on this venue/data.
- G1–G4 PASS (powered) → 923 keeps the event thread alive with numbers; the recomposition's
  alpha leg has its first honest floor.

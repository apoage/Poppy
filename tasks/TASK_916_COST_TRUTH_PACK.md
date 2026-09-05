# TASK_916 — cost truth pack: borrow charge, fee tier, restated benchmark

**Status**: PROPOSED 2026-09-05 (planner; from `lore/harvest_dead_session_6b32cd7f_2026-09-05.md` —
the margin-rollover finding from dead session 6b32cd7f + the fee-ladder finding from
`lore/fable_self_learning_loops_panel_2026-09-05.md`).
**Size XS/S. Risk: LOW (flag-gated scorer addition; OFF = bit-identical), HIGH decision weight** —
this row re-prices the bar TASK_915 is graded against. CPU only.

## Why (the two load-bearing facts, receipts in the harvest file)

1. **Shorting on Kraken goes through margin and the record never charges it.** Rollover
   0.01–0.02%/4h on BTC, 0.02–0.04%/4h on alts (22–88%/yr on shorted notional) + opening fee
   + 2% liquidation fee (Kraken fee schedule, fetched 2026-09-05 by the dead session;
   re-verify before relying). `trader/` has no margin cost; `borrow_rate_annual` defaults 0.0
   (`mcts_trader/env/jax_trading_env.py:115`); LEDGER/DECISIONS/pathway never mention rollover.
   At alt rates and a 50%-short leg the drag ≈ 22%/yr ≈ **twice the entire 21 bp trading-cost
   stack**. The +0.97 survivor / +0.56 PIT benchmark book is itself ~50% short in BEAR22 —
   the bar is borrow-uncorrected on BOTH sides.
2. **"fee16" does not exist on Kraken's current ladder** (schedule changed 2026-07-09;
   `POWERED_RESULT.md:7` calls 21 bp "taker16+slip5" vs `TASK_897:284` recording the attainable
   maker set as {8,16} bp — inconsistent). July-2026 ladder (fetched 2026-09-05, verify):
   base 0.40/0.80, $50k 0.15/0.30, $100k 0.12/0.25, $250k 0.10/0.22, $500k 0.08/0.20,
   $1M 0.06/0.18, $2.5M 0.04/0.15 (maker/taker %). Below ~$100k/30d every net@21 column in
   the record is optimistic. The operator's actual tier is unrecorded.

## What (four legs, all on `trader/`, PIT + survivor panels)

1. **Borrow charge, flag-gated**: `--borrow-model {none, kraken-margin}` charging shorted
   notional per the fetched schedule, parameterised BTC vs alts. **OFF = bit-identical**
   (pin an existing read). Applied to the SCORER, so both benchmark and candidates pay it.
2. **Fee tier honesty**: cost columns restated at (a) the operator's recorded tier once D80
   rules, (b) worst-case base tier, (c) the legacy 16+5 for continuity. Until D80 rules, run
   (b)+(c) so the row is not blocked.
3. **Benchmark restate**: recompute the 819 trend book net@{8,21} WITH borrow at each tier,
   survivor AND PIT; re-grade the BEAR-sensitive reads (BEAR22 +1.10, H=3d +1.35, 830
   hold-shares BEAR22 +1.218, 898 by-year). New reference numbers published in one table.
4. **Disclosure ledger**: which historical LEDGER rows move directionally and by roughly how
   much (note lines, no re-runs of dead experiments).

## Gates (binary)

- **G1**: OFF bit-identical pin MATCH.
- **G2**: the restated benchmark table exists with both borrow models × ≥2 tiers, both panels,
  and the BEAR re-grades.
- **G3**: **D80 seeded** (operator): actual 30d tier + whether the goalpost NUMBER in
  `docs/pathway.md` is restated net of borrow (pathway is the operator's — this row only
  publishes the restated numbers and names the deltas).

## Pre-committed consequences

- If the borrow-inclusive PIT bar reads < +0.56, that number becomes the honest reference bar
  for TASK_915's G-B/G-C MARGINAL band — **no softening of any gate without an operator
  ruling**; the restatement can only make bars harder, never easier, and that is the point.
- Every market-neutral design's odds quoted in the harvest (ARC-GP, Valerian, Comfrey,
  XS-AUTOCORR, HCB books) are already re-graded under this fact — see the harvest file §1;
  no re-litigation.

## Out of scope

Changing `docs/pathway.md` goalposts (operator), 907 estimands, any new signal, `eval_vector`
semantics (the env's `borrow_rate_annual` exists already — wiring it into runs is a later row
only if a book survives).

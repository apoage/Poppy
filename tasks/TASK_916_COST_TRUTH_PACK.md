> **STATUS (2026-09-06): SHIPPED as `task916-cost-truth-pack` (`18d42d3`, factory run 54).**
> All four legs landed. G1 OFF byte-identical verified twice in detached worktrees at HEAD
> `191e2ef`; G2 table present (2 scenarios x 4 tiers x 2 panels, 144 cells, all move down);
> G3 D80 seeded not ruled (`docs/pathway.md` untouched). Suite 16 -> 17 checks (ST17), proven
> non-vacuous against 6/6 planted mutations. Verification: 4-lens panel -> fix pass (2 MINORs
> closed: WBTC/TBTC alias bucketing, unregistered-window anchor) -> delta lens. No BLOCKER,
> no MAJOR at any point.
>
> **Headline numbers.** net21 borrow-charged, low scenario: SURVIVOR FULL +0.967 -> +0.582,
> PIT FULL +0.836 -> +0.424, PIT BEAR22 +0.990 -> +0.517, H=3d FULL +1.345 -> +0.934. At the
> verified base taker tier (80bp) instead of the legacy 21bp, PIT FULL reads +0.279 -> -0.132.
> Registered-window anchor reproduces n=2800 exactly: uncharged +0.5849, charged +0.2223 (low)
> / -0.1402 (high) -> **the pre-committed consequence FIRES**; the borrow-inclusive bar is the
> honest reference for TASK_915's G-B/G-C MARGINAL band. **Point estimates only** — the
> bootstrap CI does not exclude +0.56 (low [-0.47,+0.90], high [-0.86,+0.56]); the artifact
> carries this caveat and it must travel with the number.
>
> **REMAINDERS (not shipped, deliberately).** D80(ii) restating the `docs/pathway.md` goalpost
> net of borrow and D80(iii) remain OPEN — operator's call; this row only publishes numbers.
> The operator's actual 30d volume tier is still unrecorded, and it decides which ladder row is
> honest. Margin OPENING fee and the 2% liquidation fee are NOT charged (no margin-call hook in
> a backtest). `830 hold-shares BEAR22 (+1.218)` disclosed directionally only — lives in
> `eval_vector`, out of scope. `XBTPY` deliberately left in the ALT bucket (dead-only universe,
> ~0.0002 Sharpe effect, and its series diverges from XBT by 0.65-358x non-constant).

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
   **[CORRECTION 2026-09-06, run-53 catch — the original text here compared "drag ≈ 22%/yr"
   against the fee stack, using one number for two quantities (the D81(c) shape, in the
   planner's own file). Correct arithmetic: book-level drag = short_fraction × notional
   rate. At a ~50%-short book: BTC-low 0.5 × 22% = ~11%/yr — comparable to the trend book's
   11.9%/yr fee stack (0.155/day × 21 bp); alt-low 0.5 × 44% = ~22%/yr ≈ 1.9× the stack;
   alt-high 0.5 × 88% = ~44%/yr ≈ 3.7× the stack. The PIT book's short leg is mostly alts,
   so the honest summary is "comparable-to-4× the fee stack, asset-mix dependent," and it
   lands disproportionately in BEAR22 when the short leg is on — not a flat "twice."]** The
   +0.97 survivor / +0.56 PIT benchmark book is itself ~50% short in BEAR22 — the bar is
   borrow-uncorrected on BOTH sides; the restatement below charges both.
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
2. **Fee tier honesty**: cost columns restated at (a) the operator's recorded situation —
   **D80(i) ANSWERED 2026-09-05: the operator holds a Kraken subscription ("free things on
   first 10k traded")**; planner web verification reads this as **Kraken+**: $4.99/mo, 0%
   fees on up to $10k/month volume — but the published terms **exclude Kraken Pro / API
   trades from the waiver** (app-only; spreads still apply). VERIFY THIS EXCLUSION against
   the primary sources at leg-2 time; if it holds, the program's API-traded book is
   unaffected and the honest tier is the Pro ladder; if the operator would trade via the
   app, the app's spread is the cost surface instead. Also model: $10k/mo free volume vs the
   book's ~56×/yr turnover (fully covered only at ≲$2k account, ~21% at $10k). Then
   (b) worst-case base tier, (c) the legacy 16+5 for continuity.
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

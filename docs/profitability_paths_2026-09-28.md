# Profitability recomposition — where a net-positive system can come from

**2026-09-17, interactive session (planner lane). PROPOSAL — not a queue row; nothing here
is authorized to build. Every row it would seed needs the operator/planner framing call and,
where it is an A/B on Sharpe, the D85 spec-time MDE gate.**

## The measured premise (why this doc exists)

The record has already measured the binding constraint, and it is not signal:

- **916 (shipped `18d42d3`)**: at the verified Kraken base taker tier (~80 bp effective with
  borrow charged), the PIT FULL book reads **+0.279 → −0.132**. The record's 21 bp assumption
  was 2–5× optimistic. The +0.97/+0.56 bars are ~50% short in BEAR22 and margin rollover terms.
- **897** closed the minute surfaces as cost-negative at these tiers.
- **917 P3** delivered the first placebo-clean positive since the surfaces closed
  (post-listing IC +0.0304, NW-t 4.15, leakage-audited) — an *instrument reading*, not a book.
- **907's live capture** measured the execution reality: BTC/USD mid frozen ~46.5% of the
  time; 20m/1h markout horizons structurally 0-sample (longest gap-free stretch 561.8s).

Conclusion the panels already reached seven times (mechanism-design panel: all seven designs
died on cost/borrow arithmetic): **more direction alpha on the existing surfaces cannot clear
80 bp. The recomposition must either earn per-trade edge large enough to pay taker, or cut
effective cost per trade by an order of magnitude.**

## Option A — post-listing event thread (the only clean positive)

**Idea.** Trade the 917 post-listing drift as an event book: enter on first traded print
(917 already proved close-entry is bid-ask bounce and built the first-print instrument),
exit on a pre-registered horizon; event-level markout is the estimand, never a Sharpe A/B
(sparse n would fail D85's own gate — say so at spec time).

- Payoff: per-trade edge of potentially hundreds of bp — the only kind that survives 80 bp.
- Costs/risks: few events per year (capacity tiny, MDE on n events must be computed at spec
  time); needs first-print entry mechanics the record has only in instrument form.
- Prerequisites: none on the GPU ladder; CPU-only; folds into any surviving book as its alpha.

## Option B — planner repurposed to execution (the novel-algorithmic spend)

**Idea.** Stop asking MCTS to predict daily/minute direction (the thing the record repeatedly
fails to confirm OOS) and point the model-based planning machinery at the one surface where
short-horizon predictability is real and measurable: **our own execution**.

- State: current position, quote/fill state, short-horizon vol forecast, book imbalance.
- Actions: {place/cancel/take at level k} instead of {flat/long/short × assets}.
- Reward: realized markout net of fees, learned from the 907 live capture (and future
  captures — now cheap: 913 write-hardening + 914 streaming reader made capture analysis
  O(1) memory).
- This is learned optimal execution with a world-model — the MuZero thesis survives the 915
  verdict even if direction-selection dies, because the model is applied where its horizon
  advantage exists (minutes, not days).

**Payoff if it works:** 80 bp taker → ~5–15 bp effective, which (i) revives 806's
frozen-encoder minute-IC 0.1226 (parked exactly because costs killed it), and (ii) flips the
916 restatement (−0.132) toward positive without any new signal.

**Honest gate, measurable before any build:** 907-B2's re-cut already showed the danger —
46.5% frozen mid means maker fills may be structurally thin on this venue. Phase B was
correctly gated on AS(1h) ≤ 2 bp. So: **the B2 re-cut (D81) IS option B's pre-gate.** If
spread/spread-stability fails on the real capture, B dies cheaply, before any build.

## Option C — turnover-minimal book (boring, near-free, capped)

Fold the already-measured pieces into whatever book survives 915/918:

- θ-hysteresis (+0.3263 of +0.6523 flip removal; no Bonferroni clearance — disclosed lead,
  rides per D85),
- a no-trade band sized from the measured per-decision SE,
- the vol-target arm (analytic ceiling √(1+CV²), net ≈ +0.10 − 0.08; the 918 library arm).

Creates no edge; stops the book donating fees. Minutes of CPU, no data. Capped around
+0.10–0.15 Sharpe — additive polish, never the answer alone.

## Option D — carry outside the current record (funding/basis, delta-neutral)

The standard honest answer to "profitable on crypto with taker fees" is carry, not direction:
perp funding premia, cross-venue basis. Explicitly noted because it sidesteps the whole
direction-prediction question — and it is **outside this repo's data record**: funding rates
are not in the Kraken spot OHLCVT DB, so it is a data-ingestion row (new feed, new pipeline)
before it is an algorithm row. Trade-offs: capacity-competitive and crowd-tradeable (edge
decays as the crowd does it), and it abandons the MuZero thesis the repo is built around.
Recorded for completeness; the operator should treat it as a different system, not a row here.

## Recommendation (ordering, after the pending kill-tests)

1. **Let 915 and 918 fire first.** 915's two-sided ceiling (PASS → closed-form trader on the
   winning cell; FAIL → OHLC-repeal recommendation + data pivot) and 918's STEP-0 hull test
   (85% prior it closes the selector class incl. MuZero-as-selector, minutes of CPU) are
   cheap and gate everything downstream. Do not pre-empt them.
2. **If the selector class dies**, recompose: **C as the book, A as the alpha, B as the
   cost engine.** B is where the novel-algorithm budget goes — it attacks the only leak the
   record has measured end-to-end, and its pre-gate (B2/AS(1h)) is already specified.
3. **Do NOT** (record-anchored): more signal mining on the closed daily/minute surfaces; a
   book build before 915's verdict (the loops-panel buried seven designs on exactly this
   arithmetic); any new Sharpe A/B whose MDE vs its pre-registered band refuses itself at
   spec time (D85).

## Provenance

Every number above is a shipped, panel-verified record: 916 (`18d42d3`), 917 (`8f411d1`),
919 phase A (`f16b310`), 907 B1 (`1360250`), 913/914/921 (`3461b15`/`db8ed48`/`6a1ab2f`),
897 + the loops-panel harvest (QUEUE rows 5–7). Session context:
`lore/notes_2026-09-17.md`. Nothing here overrides D85, D79, or the grooming rules.

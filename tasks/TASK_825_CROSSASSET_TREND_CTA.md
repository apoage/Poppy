> **STATUS (2026-07-07): PARKED per DECISIONS D1/D4 — "interesting outview", non-binding; 828 is the focus. OHLC-only constraint stands (D4).**

# TASK_825 — Cross-asset-class trend CTA: attack the real ceiling (correlation, not model)

**Status**: SPEC'D 2026-07-06. This is the honest strategic frontier now that the model surface is
closed ([[project_planner_closed_clean_cell]]) and the on-chain data fork (Iris-DEX) is parked for
storage. It targets the ONE thing the mathematics says actually moves the number.

## The thesis (why this, and why it's not "more model")

The fundamental law of active management: `IR ≈ IC × √(effective breadth)`. The 819/820 crypto
trend edge is REAL per-market (IC ≈ 0.03) but **crypto majors are 0.7–0.9 correlated**, so nominal
breadth of ~40 assets collapses to an *effective* breadth of maybe 3–5 → IR ceiling ~0.6–1.0,
exactly where every crypto-only experiment landed (816→819, and 820's 2026 OOS bled in the bear).
No architecture can manufacture independent information that isn't in the data. The lever is
**more INDEPENDENT bets = uncorrelated markets**. A trend signal that is marginal on 40 correlated
crypto assets is the textbook building block of a *multi-market* managed-futures CTA that reaches
Sharpe 1.5–2.0 — because trends in FX, metals, energy, equity indices and rates are largely
uncorrelated with each other and with crypto. Same signal, same engine, genuinely more breadth.

## What's reused (cheap — the engine is already asset-agnostic)

`trader/` is a daily-close-panel machine; it does not care what the columns are. `strategy.py`
(30d sign trend), `risk.py` (overlay), `backtest.py` (per-decision Sharpe, block-bootstrap CI) all
operate on an arbitrary daily panel. So this is mostly a **DATA** task + a diversification
measurement, not new modelling. CPU only — no GPU (workstation under caps, or server CPU).

## Phase 1 — ingest uncorrelated daily markets (free data)

Build `tools/fetch_daily_multiasset.py` → a `ohlcv_daily_x` table (or parallel sqlite) in the same
schema. Free daily sources (pick per reliability): stooq (CSV, no key, generous), Yahoo via a thin
fetcher, or similar. Target a genuinely diversified set:
- **FX majors**: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF …
- **Commodities**: gold (GC/XAU), silver, WTI crude (CL), Brent, copper, nat-gas …
- **Equity indices**: S&P (ES/SPX), Nasdaq (NQ), DAX, Nikkei, FTSE …
- **Rates**: 10y note (ZN/TNX), Bund, … (trend works classically on rates)
**Real gotchas to handle (these are where free data lies):** futures contract rolls (use
continuous/back-adjusted series or roll correctly — a naive stitch fabricates trends), splits/divs
for indices, FX quote-convention consistency, holiday-calendar misalignment across venues (align on
a common trading-day grid, forward-fill leak-free exactly as `data.py` present-mask does), and
survivorship (indices change constituents — the index level is fine, single-name equities are not
in scope). Log any series that fails a sanity check; do not silently include it.

## Phase 2 — measure whether breadth lifts the edge

Same frozen strategy + discipline (leak-free, cost-honest, per-decision Sharpe, block-bootstrap
CI, cross-regime, beat-B&H). Pre-registered comparisons:
1. **Crypto-only** (the 820 baseline) vs **crypto + each new class added cumulatively** vs
   **full multi-asset** book. Metric: FULL net Sharpe + CI, and does the CI TIGHTEN (breadth
   should shrink the interval) and the worst-regime improve.
2. **The correlation proof**: compute the cross-market return correlation matrix and the *effective
   breadth* (e.g. `(Σλ)²/Σλ²` on the correlation eigenvalues, or 1/Σwᵢ²·ρ). Show effective breadth
   rises from ~3–5 (crypto-only) toward the nominal count as uncorrelated classes are added — this
   is the mechanism, measured, not assumed.
3. **The 2026 OOS re-test**: the crypto-only book went absolutely-negative in the 2026 bear
   (TASK_820 gate 2). Does the multi-asset book survive that window (crypto crashing while, say,
   gold/rates trend the other way)? This is the single most operator-relevant number.
4. Cost model per class (futures/FX spreads differ from crypto taker) — don't apply 21bp crypto
   costs to FX. Use per-class realistic costs.

## Gates (pre-registered)
- **Promotes** if the multi-asset book's FULL net Sharpe CI excludes 0 AND beats crypto-only PAIRED
  AND the 2026-OOS worst-regime is materially better AND effective breadth demonstrably rose.
  A multi-asset book that clears where crypto-only couldn't = the campaign's first genuinely
  deployable-grade result, and it upgrades TASK_820/823's live book directly.
- **Honest null** if adding uncorrelated markets does NOT lift Sharpe/tighten CI — then the crypto
  trend edge really is just marginal-and-alone, and the CTA stays a small crypto-only book. Record it.

## Why this is the right next bet
Cheap (CPU, free data, existing engine), attacks the actual measured ceiling (correlation, not
model capacity — [[project_planner_closed_clean_cell]]), needs no terabytes (Iris-DEX-independent),
and its success directly makes the SHIPPED trader (TASK_823) better and more deployable rather than
producing another research verdict. Highest strategic value / cost of anything on the board.

## Run location
CPU — workstation under `safe_run --mem` caps, or server CPU. No GPU. Sub-agents: build + report;
keep the leak-free / cost-honest discipline non-negotiable (a fabricated futures-roll trend is the
classic way this kind of study lies to itself).

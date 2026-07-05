# TASK_816 — Operating-point probe (same OHLC, better cost / frequency / instrument / universe)

**Status**: DRAFT — operator direction (2026-06-26). **A competing take to TASK_814.** Where 814 says
"the binding constraint was the *data* → add richer info," this task says the binding constraint was the
**operating point** (edge-per-trade vs cost), not OHLC's information content — and tests the *same* OHLC
signal at a better operating point. Reuses the 806 apparatus. Kill-gated.

**816 is logically UPSTREAM of 814 and 817 — run it FIRST.** It is the cheapest way to decide whether the
other two are even worth their (expensive) data-sourcing *at this operating point*. Running 814 (richer
data) or 817 (execution) first risks **building on a cost-bound foundation**: if the *operating point* — not
the data, not the execution — is the actual wall, 814/817 inherit it. 816 falsifies that for ~free on
existing OHLC before anyone sources news, funding, or order-book data.

## The question (one sentence)
Does the **same price-only (OHLC) signal** clear the cost floor when moved off the worst-case operating
point — i.e. at a **cleaner universe** (liquid majors, drop the junk pairs), **lower cost** (perp/futures
fees ≈ 2–5 bp vs spot taker 21 bp; maker), and **lower trade frequency** (accumulated per-trade edge,
low turnover) — OOS, leak-free, FDR-surviving?

## Why this, now (the diagnosis 806→813 actually supports)
- 806 proved **OHLC carries real signal** (clean-OOS IC 0.06–0.15 — *larger* than the cross-sectional
  momentum ICs that fund real equity strategies). The campaign did NOT find "no signal in OHLC."
- The binding constraint was **edge-per-trade (~3 bp) vs round-trip cost (~42 bp) ≈ a 14× gap.** That gap
  is set by **frequency × venue × instrument × universe**, NOT by OHLC being information-empty.
- 806→813 tested essentially the **worst case for OHLC**: **1-minute** (tiny per-trade edge) × **spot
  taker** (high cost) × a **64-asset mid-cap universe with several near-junk pairs** (e.g. `/USDTEUR`,
  `/USDTCHF`). The classic price/trend edge lives at **lower frequency, lower cost, on liquid majors /
  futures** — none of which the campaign tested. The managed-futures industry is the existence proof.
- So before concluding "OHLC is closed, must add data" (814), test the cheaper hypothesis: **OHLC is fine;
  the operating point was wrong.**

## The four levers (each shrinks the 14× gap; test individually + combined)
1. **Universe quality (genuinely new — never tested).** Restrict to liquid majors / top-N by dollar volume;
   drop the **junk** (low-liquidity, wide-spread, stale-price pairs — e.g. `/USDTEUR`, `/USDTCHF`). Junk
   **inflates *effective* cost** (you cross a wide spread + slip) AND **adds noise** to the cross-section.
   The 64-asset universe's *quality* was never a variable in 806→813. **Free on existing OHLC** — the
   single cheapest lever, run it first.
2. **Cost regime — the disciplined re-test of the question 809 fumbled.** Re-price the SAME signal at maker
   (~5–10 bp) and **perp/futures taker+maker (~2–5 bp)**, not just spot taker (21 bp); a 5× cut closes most
   of the 14× gap. **Not redundant with 809:** 809 tested maker but its verdict was confounded by the
   **per-bar-Sharpe annualization artifact** and the **turnover mirage** (net-beat-without-gross-beat). 816
   re-prices cleanly under the full 810/813 disciplines — per-decision H-strided Sharpe, **gross AND net**,
   honest turnover, FDR. The honest version of "does lower cost save it?", not a re-run.
3. **Frequency / horizon** — genuine low-frequency rebalancing (daily/4 h holds, low turnover) so per-trade
   edge accumulates and total cost amortizes. (Note: 813 multi-day was cost-bound *on the junk universe* —
   re-test on the clean-majors universe before concluding.)
4. **Instrument** — perps/futures (lower cost; funding-carry is a *separate* structural edge — overlaps
   814's funding signal; flag the spot-vs-perp decision in the verdict).

## Method (phased, probe-gated; reuse 806 guards)
- **Phase 0 — data**: source the better-operating-point OHLC — daily/4 h bars for liquid majors, and/or
  **perp/futures OHLC** (Kraken/Binance futures, historical, free, clean bar timestamps — *easier* than
  814's news/funding leak-free alignment, same data TYPE). DB today is spot `ohlcv` + `fx_rates` only.
- **Phase 1 — probe**: re-run the 806/808/813 edge-vs-cost machinery (linear_xs + the vol/trend map) on
  each lever and the combination. Report per-decision H-strided Sharpe, **gross AND net**, block-bootstrap
  CI, BH-FDR, at each cost schedule. The decisive number is **edge-per-trade vs the (lower) cost floor.**

## Gate (kill or go)
- **GO**: the same OHLC signal clears the cost floor at ≥1 better operating point — net post-cost Sharpe
  > 0, CI-separated, FDR-surviving, gross-beat reported → **OHLC is tradeable; the campaign closed an
  operating point, not the data.** → a deployable price-only strategy + (optionally) the 812 planner on it.
- **KILL**: even the best operating point (clean majors × futures cost × low frequency) is cost-bound →
  OHLC is genuinely closed *across operating points* — a much stronger close than 806→813, and it makes
  814's "need richer data" the live hypothesis.

## Honest prior
**Moderate.** The 14× gap is large; futures cost (~5×) + a clean universe + low turnover *plausibly* closes
it, and the managed-futures existence proof supports OHLC tradeability — but crypto majors are competitive
and the campaign's multi-day vol probe (813) was already cost-bound on the junk universe. Arguably a
*cleaner* test than 814 (same data type, no new leak surface). Not a sure thing; Phase 1 settles it in days.

## Constraints / audit (inherit the campaign discipline)
Leak-free (causal features ≤ t; forward return = label; clean bar timestamps); honest cost (state the exact
fee schedule per run; gross AND net — 810); per-decision H-strided Sharpe, never per-bar ×√525600 (809);
BH-FDR across the grid; OOS test only; reuse `probe`/`backtest` (cite); server-side, OOM-safe (the
oomd-safe cache + slice pattern from 813), never the workstation display GPU. STOP for audit before any
build; the auditor re-derives edge-vs-cost from artifacts.

## References
- `runs/806_predictability_probe/` (the IC 0.06–0.15 price-only baseline + guards), `runs/808_linear_backtest/`
  (cost model + per-rebalance Sharpe), `runs/813_volharvest/` (the multi-day vol/trend map + event-level audit).
- `tasks/TASK_814_RICHERINFO_PROBE.md` (the sibling "richer data" take — 816 is the "better operating point"
  take; run both, pick the cheaper-to-falsify first).
- `[[project_modeling_surface_exhausted]]` — the 806→813 close was *Kraken 1-min spot taker, junk-heavy
  universe*; this task tests whether that scope, not OHLC itself, was the binding constraint.

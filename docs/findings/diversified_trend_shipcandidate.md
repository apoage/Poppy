# diversified trend shipcandidate

> TASK_819 bottom-up pivot — diversified per-asset TSMOM trend edge is REAL but MARGINAL; Phase-2's gate-cross was survivorship (Phase-3 point-in-time FOLDED it, CI straddles 0); model capacity signal-limited. OHLC-trend surface honestly closed.

*Migrated from local agent memory 2026-08-31 (`project_diversified_trend_shipcandidate.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
**TASK_819 (2026-06-28): the bottom-up pivot produced the campaign's FIRST positive, gate-crossing result — a diversified per-asset trend (CTA) portfolio — and definitively closed the "more model" question.** Reverses the all-null pattern of 738→818.

**The two converging results (both `runs/819_single_asset/`):**
- **CAPACITY ladder → SIGNAL-LIMITED** (`CAPACITY_RESULT.md`, ran on gpu-server). Pooled daily panel (20 assets,
  ~34k rows), strict OOS. OOS IC: ridge 0.011 → ridge-rich 0.018 → GBM 0.037 → MLP 0.036 → GRU 0.033 — **all
  below the shuffled-label noise floor |0.059|**; all net@21 CIs include 0. Every deep model CONVERGED (GBM
  OVER-iterated). **Answers "are we missing an edge by training longer/deeper?" → NO.** Nonlinear/sequence
  ceiling = linear ceiling. Capacity is not the lever — corroborates [modeling_surface_exhausted](modeling_surface_exhausted.md) /
  [collapse_fixed_predictability_wall](collapse_fixed_predictability_wall.md).
- **PORTFOLIO → CONFIRMED (per-cell gate), ship-CANDIDATE** (`PORTFOLIO_RESULT.md`). A risk-weighted PORTFOLIO of
  per-asset TSMOM bets (diversification damping per-asset noise — NOT cross-sectional ranking, which is
  cost-bound 808/811). Best cell = **broad ~39-asset universe, EQUAL-weight, single-30d trend**: FULL net@8
  Sharpe **+1.09 CI[+0.21,+1.95]**, net@21 **+0.97 CI[+0.10,+1.84]** (both exclude 0), **BEAR22 +1.36** (B&H
  −2.29), equity ×21.7 @8bp, max DD −38%, turnover 56.6×/yr. **Crossed the gate BTC-alone (819 Phase-1, CI incl 0)
  AND the 816 naive-20-equal basket (CI incl 0) could not.** ROBUST (clears noLUNA + H=3d + tanh → not a LUNA
  artifact). **BREADTH-DRIVEN:** more paths +0.22 was the ONLY lever that helped; risk-sizing −0.06 and
  lookback-ensemble −0.16 both HURT.

**The lesson (predicted in Phase-1, paid off): MORE PATHS, NOT MORE MODEL.** The per-asset trend edge is real but
noisy; diversification (breadth) is what averages the noise down enough to cross the gate; model capacity does
nothing.

**PHASE-3 POWERED CONFIRMATION → FOLDED (2026-06-28, `POWERED_RESULT.md`, server).** ONE pre-registered config
(broad/equal/single-30d, no sweep → no multiplicity), POINT-IN-TIME 276-asset universe (each asset from its OWN
listing, dead coins included), FULL history 2018-2025 (TWO independent bears). Result: **FULL net@8 +0.67
CI[−0.01,+1.34], net@21 +0.56 CI[−0.13,+1.23] — both CIs now STRADDLE 0** (Phase-2's [+0.21,+1.95] was
survivorship-inflated; survivorship was worth ~0.42 Sharpe and the difference between clearing and not).
**BUT the directional edge is REAL, just-misses:** BOTH bears positive (BEAR18 +0.53 vs B&H −2.28; BEAR22 +1.16 vs
−2.54), FULL Sharpe +0.67 beats B&H −0.40, equity ×30.8. The fold is marginal + partly a power artifact (2018 bear
only 5%-covered ≈14 coins, drags the FULL CI; lower bound only −0.01).

**FINAL VERDICT: crypto-OHLC trend is REAL but MARGINAL** — directional, crash-protective, NOT statistically
confirmable under honest point-in-time conditions. Consistent across every angle: single-asset too noisy → 816
FDR-fail → Phase-2 confirms only on survivors → Phase-3 just-misses point-in-time → capacity signal-limited.
**The OHLC-trend surface is honestly CLOSED: neither more model nor more (survivor) paths makes it cleanly
shippable.** Strategic fork (operator's): (a) ACCEPT the marginal edge — deploy a trend CTA with risk mgmt
(positive expectancy + crash hedge, but thin, ~4-12%/yr turnover cost); or (b) CHANGE THE DATA — Iris-DEX
per-block on-chain OHLCV / futures, where a cleaner edge may live.

**Infra note:** capacity training ran entirely on gpu-server (operator: never OOM/GPU the workstation display
GPU); portfolio is CPU/numpy (24s). Both reuse the 816 backtest infra.

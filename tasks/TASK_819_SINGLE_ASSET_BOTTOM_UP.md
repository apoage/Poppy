# TASK_819 — Bottom-up trader: single-asset edge → diversified trend portfolio → cross-asset

**Status**: ACTIVE (started 2026-06-28). Strategic pivot from the top-down 64-asset MuZero planner
([[project_collapse_fixed_predictability_wall]]: wiring fixed but the cross-sectional/1-min signal is a
predictability floor). New thesis (operator): **build the simplest debuggable profitable thing first — one
asset, absolute P&L — then add paths.** The campaign's only confirmed edge (816 TSMOM trend) is fundamentally a
single-asset time-series edge, so single-asset-first stands on a proven floor.

## Phase 1 — single-asset BTC edge (DONE 2026-06-28, `runs/819_single_asset/`)
Probe `single_asset.py` (CPU, 9s, self-tests pass). BTC = col 0 of `_close_crossregime.npz`, 2021-2025, sweep
15m/1h/4h/1d × L/H durations, net {0,3,8,21}bp, 2022 BEAR as judge.
- **TSMOM floor:** best cell L=30d/H=1d returns **×6.26 vs buy-and-hold ×3.30** (beats B&H by shorting the 2022
  crash) — a REAL directional edge. BUT **0/67 cells clear the strict gate** (FULL net@21 Sharpe +0.70, CI
  **[−0.19,+1.54] includes 0**). One path is too noisy to confirm.
- **Learned predictor (ridge, strict OOS):** IC ≈ **+0.03, R² ≈ 0**, net-negative everywhere — **no richer
  learnable single-asset signal beyond the slow trend itself.** "Build something that predicts" hits a wall:
  the prediction IS the trend.
- **Lever identified: MORE PATHS (diversification), not more model.** The cross-section damps the noise (why
  816's 20-asset portfolio nearly confirmed where BTC-alone can't). TSMOM at fixed (L,H) duration is
  timeframe-invariant on FULL (the sweep collapses to a duration grid); BEAR Sharpe is phase-fragile single-asset.

## Phase 2 — diversified trend portfolio (IN FLIGHT, workflow wcd61thm4)
Portfolio of single-asset trend bets (diversification damps noise) — **NOT cross-sectional ranking** (that's the
cost-bound thing 811/816 closed). Three engineering levers vs 816's naive 20-equal-weight: **(1) more paths**
(broaden universe ~40-60 assets), **(2) risk sizing** (inverse-vol / risk parity), **(3) lookback ensemble**
(average L∈{7,14,30,60,90}d, kill param noise). Cross-regime, net-of-cost, strict gate (FULL net CI excludes 0 +
beats B&H + BEAR positive + FDR), + adversarial verify (leak/cost, artifact/regime, FDR/overfit). Deployable
metrics: equity curve, max DD, annual turnover. Output `runs/819_single_asset/PORTFOLIO_RESULT.md`.

## Phase 2 RESULT (2026-06-28, `runs/819_single_asset/PORTFOLIO_RESULT.md`, CPU 24s) — CONFIRMED (ship-candidate)
**MORE PATHS crossed the gate that BTC-alone + the 816 naive-20 basket could not.** Best cell = broad ~39-asset
universe, EQUAL-weight, single-30d trend: **FULL net@8 Sharpe +1.09 CI[+0.21,+1.95], net@21 +0.97 CI[+0.10,+1.84]**
(both exclude 0), **BEAR22 +1.36** (B&H −2.29), equity ×21.7 @8bp, max DD −38%, turnover 56.6×/yr. ROBUST: clears
noLUNA + H=3d + tanh (NOT a LUNA artifact). BREADTH-DRIVEN: more paths +0.22 was the ONLY lever that helped;
risk-sizing −0.06 and lookback-ensemble −0.16 both HURT (so it's diversification, not a tuned overlay). **Caveats:
FDR 0/8** (best p≈0.01 fails BH q=0.05 across 8 cells — power/multiplicity wall, same as 816) + **survivorship**
(39-asset universe hindsight-selected; extra paths help mostly post-2022; bear carried by OG names). The
campaign's STRONGEST crypto-CTA result — deployable-ish on low-cost perps; not yet a closed spot edge.

## CAPACITY result (2026-06-28, `runs/819_single_asset/CAPACITY_RESULT.md`, server) — SIGNAL-LIMITED
Settles the operator's "are we missing an edge by not training longer/deeper?" → **NO.** Pooled daily panel (20
assets, ~34k rows), strict OOS. OOS IC: ridge 0.011 → ridge-rich 0.018 → GBM 0.037 → MLP 0.036 → GRU 0.033 —
**all BELOW the shuffled-label noise floor |0.059|; all net@21 CIs include 0.** Every deep model CONVERGED
(GBM is OVER-iterated, val-IC peaks ~iter 200 then declines). The nonlinear/sequence ceiling = the linear
ceiling. **Capacity does nothing; the lever is DATA/PATHS, not model** — corroborates 806→818.

## Phase 3 RESULT — POWERED point-in-time → FOLDED (2026-06-28, `runs/819_single_asset/POWERED_RESULT.md`, server)
ONE pre-registered config (broad/equal/single-30d, NO sweep → no multiplicity), POINT-IN-TIME 276-asset universe
(each asset from its OWN listing, dead coins included), FULL history 2018-2025 (TWO independent bears). Coverage
per regime: BEAR18 5%, REC1920 9%, BULL21 22%, BEAR22 52%, REC23 76%, BULL24 88%, CHOP25 98%, FULL 43%.
**Result: FULL net@8 +0.67 CI[−0.01,+1.34], net@21 +0.56 CI[−0.13,+1.23] — both CIs STRADDLE 0** (Phase-2's
[+0.21,+1.95] was survivorship-inflated; survivorship ≈ +0.42 Sharpe = the difference between clearing and not).
**Verdict: FOLDED on the strict gate.** BUT the directional edge is REAL/just-misses: BOTH bears positive
(BEAR18 +0.53 vs B&H −2.28; BEAR22 +1.16 vs −2.54), FULL +0.67 beats B&H −0.40, equity ×30.8, max DD −57%. The
fold is marginal + partly a power artifact (2018 bear only 5%-covered ≈14 coins drags the FULL CI; lower bound
only −0.01).

## FINAL — crypto-OHLC trend = REAL but MARGINAL; surface honestly CLOSED
Consistent across every angle: single-asset too noisy → 816 FDR-fail → Phase-2 confirms only on survivors →
Phase-3 just-misses point-in-time → capacity SIGNAL-LIMITED. **Neither more model nor more (survivor) paths makes
it cleanly shippable.** Strategic fork (operator's): **(a) ACCEPT** the marginal edge — deploy a trend CTA with
risk mgmt (positive expectancy + crash hedge, but thin, ~4-12%/yr turnover cost), gate-3 paper ladder
[[project_deployment_gate_ladder]]; or **(b) CHANGE THE DATA** — Iris-DEX per-block on-chain OHLCV / futures,
where a cleaner edge may live. NEVER blindly clone the foresight oracle (it doesn't transfer — the −85% blind result).

## Discipline (806→819, non-negotiable)
Leak-free; cost-honest (per-fire turnover); per-decision H-strided Sharpe (NOT per-bar×√525600 — the 809 trap);
cross-regime with the 2022 BEAR as judge; OOS for learned models; beat-buy-and-hold (strip beta); BH-FDR across
sweeps; adversarially verify every positive result (net-beat without gross-beat = artifact). Never blindly clone the
foresight oracle (it doesn't transfer — −85% blind result). All probes CPU/numpy on the workstation OR sandboxed on gpu-server.

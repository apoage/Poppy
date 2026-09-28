# TASK_915 — predictability ceiling of the daily OHLC surface, then the GP book

**Status**: PROPOSED 2026-09-03 (planner; adopted by operator instruction from
`lore/fable_direction_memo_2026-09-03.md`, which IS the authoritative spec — this file is the
queue pointer + gates. Planner verified the memo's A-2 receipt first-hand: the vector line was
close-only, `jax_encoder.py:124-134` + `env_factory.py:417`.)
**Size: M. Risk: LOW compute (CPU only, ≤2 CPU-days), HIGH decision weight** — this row is
built to END a question, two-sidedly.

## What it is (one paragraph; the memo's §4 has the mathematics)

A pre-registered ceiling sweep: does the PIT Kraken daily OHLC panel carry a market-residual,
vol-standardised (Yang–Zhang) forward-return signal with per-date cross-sectional IC ≥ ~0.02–0.03
on 2024-01→2026-06, at h ∈ {1,3,7,21}d, from any causal feature family (F1 trend/reversal,
F2 range/vol — the OHLC-native content never used, F3 cross-sectional, F4 log-signatures)?
Ridge + GBM only (811 measured nonlinear ≤ linear on the MINUTE surface; the daily 819 CAPACITY
point estimates are inconclusive with nonlinear numerically ABOVE linear — corrected 2026-09-05
per the loops panel; GBM stays in, neural nets excluded as scope, not evidence). IC → net-Sharpe bound via
the fundamental law with MEASURED breadth (N_eff from the residual correlation spectrum) and
Gârleanu–Pedersen turnover under the 21 bp tier. On PASS: the GP closed-form book, graded on
BOTH `trader/` and `eval_vector` (the 882 two-instrument agreement discipline), then the
existing gate ladder.

## Preconditions

- **TASK_899's staged daily backfill INGESTED first** (the seam matters for 2026H1 cells) —
  planner-owned; the staged CSVs are ready.
  **AMENDMENT 2026-09-28 (planner): satisfied READ-ONLY — load DB-daily (through 2026-06-21)
  + the staged CSVs (`runs/858_backfill/api_2026-08-31/daily/`, seam-verified 2026-08-31,
  62 overlap rows, zero mismatches) with an in-loader seam re-verification that refuses on
  any mismatch; pin the loader + seam hash in the artifact. NO DB writes; the ingest itself
  stays gated (RO02 / operator go-word) and is not needed for the seam.**
- **TASK_916's restated cost stack landed first** (2026-09-05 amendment): G-B/G-C convert IC
  to net Sharpe through the 21 bp tier — the honest stack is tier + BORROW (margin rollover
  is real and never charged; the benchmark itself is short in BEAR22). Grade at the 916-restated
  bar; do not hand-tune.
- True UTC daily bars from `trader/data.py` aggregation (NOT the minute stride); PIT universe
  via the 819 POWERED builder; epoch named + pinned per `LEDGER.md:28-41`.

## Gates (binary, pre-registered before the first fit; SPEC sha256 recorded)

- **G-A Signal exists**: ≥1 cell, mean IC ≥ 0.02 on 2024-01→2026-06, Newey–West t ≥ 3 after
  Bonferroni (≈160 cells), IC > 0 in each of 2024/2025/2026H1 separately.
- **G-B Ceiling clears**: fundamental-law net-Sharpe bound at 21 bp ≥ +0.97 (MARGINAL ≥ +0.56)
  for some G-A cell.
- **G-C Control realises it**: the GP book at that cell, PIT, both instruments agreeing within
  0.1: net@21 ≥ +0.97, CI lo > 0, BEAR22 form-check positive, 2026H1 unseen positive, holds
  positions, ≥3 seeds.

## Pre-committed consequences (adopted with the memo — D79)

- **G-A fails everywhere → recommend repealing the OHLC-only constraint (D4); the next row is
  a DATA row (funding/basis, on-chain flow, or the news-LLM hybrid), not a model row. No
  further policy machinery on daily bars, any architecture, MuZero included.**
- G-A passes / G-B fails → record the measured IC/N_eff/turnover triple; breadth/cost are
  operator decisions with numbers; no trader build.
- G-B passes / G-C fails → the first legitimate re-entry point for a search-based controller,
  with the forecast already measured.
- Kill date: 14 days from SPEC commit. The cell grid is fixed in the SPEC; additions are new
  rows. The 828 training lane stays DEAD regardless (killed by D79; env + eval_vector kept as
  the cost-honest simulator).

## Out of scope

The minute surface (897/907 own it; 907 completes for D65 closure and does NOT gate this row).
Re-opening the 828 lane. Neural nets in the ceiling sweep. Any book build before G-A.

## Amendments folded 2026-09-05 (before SPEC commit — legal pre-registration edits; from the
loops panel `lore/fable_self_learning_loops_panel_2026-09-05.md` and the dead-session harvest)

- **Null family**: replace the E-2 four-draw floor with a **block-Rademacher / circular-shift
  placebo family** (dependence-preserving, zero fitting variance). Reflection/time-reversal
  surrogates are WRONG for sign-flip targets (PnL-invariant) — circular shift + per-date
  permutation.
- **Two SDs, both stated**: per-date cross-sectional IC null SD ≈ 1/√(N−1) ≈ 0.06–0.07 over
  live assets; N_eff governs the fundamental-law breadth term only. Do not divide by √N_eff
  when testing IC ≠ 0.
- **Monthly-IC autocorrelation τ** with an overlap-aware null, one added statistic (the
  continual-learning survivor; G-A cells also report the τ_IC ≥ 5d gate read).
- **SR0(N) multiplicity line** beside every best cell (oracle-null selection inflation
  ≈ +0.5–0.75 at this N on this window).
- **MC-attenuation note**: any generator-vs-ridge IC comparison needs K ≥ ~1/(0.1·IC²) ≈ 25k
  paths or the closed-form mean (a K=512 rollout halves an IC of 0.02).
- **DRO position gate** (Wasserstein μ̂ᵀφ − √δ‖φ‖ − cost > 0) pre-named as the GP book's
  position gate IF G-A passes (the adversarial-loop survivor).

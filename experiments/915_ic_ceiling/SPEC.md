# TASK_915 SPEC — predictability ceiling of the daily OHLC surface (G-A/G-B only, tonight)

Authoritative math: `lore/fable_direction_memo_2026-09-03.md` Sec.4 (lines 244-380).
Task file: `tasks/TASK_915_IC_CEILING_AND_GP_BOOK.md`. This SPEC is the pre-registration: the
cell grid, targets, gates and kill date below are FIXED before the first real-data fit; any
addition afterward is a new row (per the task's "no metric shopping" clause). **G-C (the GP
book, graded on `trader/` + `eval_vector`) is explicitly NOT built by this SPEC or this run** —
tonight's scope is G-A (does the signal exist) and G-B (does the fundamental-law ceiling clear
costs), per the orchestrator's steering. G-C is the named remainder.

Pinned: this file's own sha256 is recorded in `RUN_MANIFEST.json` at the moment it is computed,
which happens AFTER the panel-only MDE table below (asset counts, no fitting) and BEFORE the
first cell of the OFFICIAL 160-cell grid is fit. "First real-data fit" is read as the first
cell of that official, checkpointed run (`/var/tmp/915-run/`) — NOT the fixture-smoke and
timed-dry-run cells the task itself calls for as pre-registration validation steps (a handful
of real cells run BEFORE this pin, on gpu-server, to (a) confirm the loader/gate mechanism
fires correctly on real data — see Sec.12's adversarial-pass disclosure — and (b) measure the
per-cell wall time needed to produce the CPU-day estimate in Sec.11). This is a disclosed
interpretation, not a silent exemption: Sec.12 states exactly which cells ran pre-pin and what
they found (including a real bug the dry run itself surfaced and this SPEC's code fixed before
the official launch).

## 1. Epoch and PIT frame

Post-E-f (`LEDGER.md:28-41`; E-e XBT/BTC fold, E-f no-print=FAILED-FILL both apply generally,
neither is load-bearing for a CPU-only IC/turnover instrument with no fill simulation). Data:
`fixtures/trading_fixture.db` `ohlcv_daily` (verified TRUE daily OHLCV, both `kraken_rest` and
`agg1m` sources, zero NULL O/H/L/C/V cells, 499,430 rows as of this SPEC — see Sec.6) through
2026-06-21, seam-extended with `runs/858_backfill/api_2026-08-31/daily/*.csv` (56 pulled pairs,
31 USD-quoted after the STABLE_FIAT filter) to 2026-08-31, re-verified at LOAD TIME on every
column used (open/high/low/close/volume, not close alone — see Sec.6). Fixture sha256
`b0f8de8b108b50521f301df1d0856aed72bc760c57b67a842fa62fa918206cf9` (verified identical on the
workstation and on gpu-server's own checkout before this run and re-verified after).

## 2. Universe

Point-in-time via `trader.data.load_db_backtest_panel(include_dead=True)` (the same delegated
builder `898/916/917/918` use — "the 819 POWERED builder" per the task's naming; dead/delisted
coins retained, PIT membership resolved downstream by `trader.strategy.present_mask`), then
`trader.data.apply_liquidity_filter` (819's own thresholds: active-window zero-return fraction
< 0.15, >= 365 active days). `present_mask` warm-up = max(beta_window=90, vol_window=30) + 1 =
91 days of real (non-backfilled) history before an asset counts as live.

## 3. Targets (4) — `targets.py`

- **T1 PRIMARY** `y1_primary_resid_yz`: market-residual, Yang-Zhang-standardised forward
  return. `y1_{i,t,h} = (r_{i,t->t+h} - beta_hat_{i,t}*r_{mkt,t->t+h}) / sigma_hat_YZ_{i,t}`.
  beta_hat: trailing 90-day causal OLS of asset log-returns on the equal-weight-market
  log-return (live assets only). sigma_hat_YZ: Yang-Zhang (2000) range estimator, trailing
  30 true daily OHLC bars (`yang_zhang.yang_zhang_vol`).
- **T2 CONTROL (A-2 test)** `y2_control_resid_c2c`: identical construction, standardised by
  close-to-close sigma instead of Yang-Zhang — isolates whether the OHLC-native range
  estimator earns anything over a close-only one.
- **T3 CONTROL** `y3_control_abs_return`: `|r_{i,t->t+h}|`, unstandardised, NOT residualised —
  a magnitude/vol-clustering control (if F2/F4 "predict" T3 but not T1/T2, the apparent edge is
  volatility, not direction).
- **T4 "sign x vol-scaled"** `y4_sign_volscaled`: `r_{i,t->t+h} / sigma_hat_YZ_{i,t}^2`, raw
  (not residualised) — this row's operational definition of the memo's Barroso-Santa-Clara /
  Daniel-Moskowitz vol-managed-momentum arm (the memo names the concept, not a formula;
  disclosed here as inverse-variance-scaled outright forward return, the standard
  vol-managed-momentum construction).
- Market-factor-timing arm (single series, breadth~=1): `targets.market_timing_target`,
  reported, never pooled into the 160-cell grid (memo Sec.4.2).

Leak-free: every feature/beta/sigma at row t reads rows <= t only; every target's label starts
at row t+1 (`forward_log_return`). Verified in `tests/test_targets.py`
(`test_leak_free_targets_unaffected_by_future_mutation`: mutating data after a cutoff does not
move any target at-or-before the cutoff).

## 4. Horizons (4) and feature sets (5)

Horizons: h in {1, 3, 7, 21} days.

- **F1** trend/reversal: lagged log-returns at {1,3,7,21,30,63}d (30d added explicitly to
  literally include the 819 single-30-day-trend signal as a control, per the memo's
  instruction — the memo's own stated set {1,3,7,21,63} omits it).
- **F2** range/vol: Yang-Zhang / Parkinson / Garman-Klass vol (30d), their ratios (OHLC-native
  efficiency, A-2's untested content), vol-of-vol, range-to-close, HAR-RV (Corsi) daily/
  weekly(5d)/monthly(22d) terms.
- **F3** cross-sectional: beta (T1's own estimate, reused not recomputed), idiosyncratic vol
  (rolling std of beta-residual 1d returns, 30d), dollar-volume (rolling mean(close*volume),
  30d). Cross-sectional RANKING of every feature happens centrally (Sec.5), not inside F3.
- **F4** truncated (level<=3) log-signature of the trailing 30-day path of (log-close,
  log-range, log-volume, time) — `signature.py`, Chen's identity, numpy (no
  `iisignature`/`esig` in gpu-server's `trader_env`, verified 2026-09-28 — DISCLOSED
  SUBSTITUTION per the task instruction). 4+16+64=84 raw columns (level1/2/3 tensors
  flattened); level-2 is exactly antisymmetric (the doubled Levy area) by construction.
- **F5 = ALL**: F1 union F2 union F3 union F4 (the memo names 4 families and "state the
  fifth"; ALL is the natural fifth cell, letting the conjunctive estimator (GBM) use every
  family at once).

4 targets x 4 horizons x 5 feature sets x 2 estimators = **160 cells** (`grid.ALL_CELLS`,
asserted `== 160` at import time).

Every feature is cross-sectionally z-scored PER DATE over LIVE assets only
(`grid.cross_sectional_zscore`) before fitting — a same-date operation, hence leak-free
regardless of when it is computed.

## 5. Estimators (2)

- **Ridge**: `sklearn.linear_model.Ridge(alpha=1.0)` (closed-form solver), pinned alpha, no
  per-cell tuning.
- **GBM**: `sklearn.ensemble.HistGradientBoostingRegressor(max_iter=100, max_depth=4,
  learning_rate=0.1, l2_regularization=0.0, max_leaf_nodes=15, min_samples_leaf=30)`, fixed
  hyperparameters (no per-cell tuning), **3 seeds {0,1,2}** (`GBM_SEEDS`), predictions averaged
  — applied uniformly to every GBM cell in the 160-cell grid (cheap enough: measured wall time
  in Sec.8), not only to a G-C book.

Expanding-window **quarterly** refit: for each calendar-quarter test window, train on every
prior row up to `test_start - h` (**purge >= h**, enforced by construction in
`grid.fit_cell` — the last usable training label is the one whose h-day-forward window ends
strictly before the test quarter starts), minimum 500 days / 200 valid rows of training history
before a quarter is fit. OOS coverage: 2022-01-01 through 2026-06-30 (by-year reporting from
2022; G-A gates only 2024-01-01..2026-06-30, per Sec.3 below).

## 6. Data check (memo's A-2 receipt — must NOT repeat close-only construction)

Verified against `fixtures/trading_fixture.db` before writing this SPEC: `ohlcv_daily` schema
carries `open, high, low, close, volume, trade_count, source` (not close-only); zero NULL
cells in any of open/high/low/close/volume across 499,430 rows (391,123 `agg1m` +
108,307 `kraken_rest`). The 858 staged CSVs (`time,open,high,low,close,vwap,volume,count`)
likewise carry full OHLCV. `loader.py`'s seam re-verification checks **all five** columns
(open/high/low/close/volume) independently in the DB/CSV overlap window and refuses (raises)
on ANY mismatch on ANY column — extending TASK_918's close-only seam check
(`experiments/918_tactic_hull/loader.py`), which is disclosed as insufficient for this row.
**Data check: PASS — true daily OHLC is present; no close-only substitution made.**

## 7. Statistics (G-A)

Per-date cross-sectional Spearman IC (`stats_utils.spearman_ic`, live-asset pairs only, n>=3).
Newey-West SE of the mean IC, Bartlett kernel, **lag = h** (`stats_utils.newey_west_se_of_mean`,
reduces exactly to the iid SE at lag=0 — tested). Bonferroni over the 160-cell grid: alpha=0.05
two-sided, per-cell alpha = 0.05/160, critical NW-t = **3.6047** (`stats_utils.
bonferroni_t_crit`, asymptotic normal — the panel's dates-per-cell count (Sec.9) is large
enough that this is the standard convention rather than a small-sample t).

**G-A pass** (`gates.ga_pass`): mean IC over 2024-01-01..2026-06-30 >= 0.02 AND NW-t (lag=h) >=
3.6047 AND mean IC > 0 in EACH of 2024, 2025, 2026H1 separately (a cell with zero coverage in a
required year, e.g. from a short synthetic test panel, correctly FAILS this check rather than
vacuously passing — verified in `tests/test_positive_control.py`).

**2026-09-05 amendment — null family** (`stats_utils.circular_shift_null`,
`block_rademacher_null`, `per_date_permutation_null`): dependence-preserving placebos (never
reflection/time-reversal, wrong for sign-flip PnL-invariant targets), run as a SANITY CHECK on
the NW SE, never as the gate itself (the gate is the real per-date IC series's own NW t).

**Two SDs, both stated, never conflated**: the per-date cross-sectional IC null SD used to
test IC != 0 is ~1/sqrt(N-1) over live assets that date (what NW-t above tests against);
`N_eff` (Sec.8) governs BREADTH in the fundamental law ONLY — never divide by sqrt(N_eff) when
testing IC != 0.

**tau_IC**: monthly-aggregated IC series' lag-1 autocorrelation, reported per cell
(`stats_utils.tau_ic`), with an overlap-aware NW SE (`tau_ic_overlap_aware_se`) — the
continual-learning-survivor persistence statistic, not a gate.

**SR0(N) multiplicity line**: `stats_utils.sr0_multiplicity(160)` = sqrt(2*ln(160)) = **3.24**
(max-of-160-iid-N(0,1) order-statistic approximation), reported beside the best cell's NW-t —
oracle-null selection-inflation context, not a gate deduction.

### D85 MDE at spec time (panel asset counts ONLY, no fitting)

`MDE ~= t_crit * (1/sqrt(N_bar-1)) * sqrt(nw_inflation) / sqrt(n_dates)`
(`stats_utils.mde_ic`, `mde_report.py`). **Assumed NW inflation = 2h-1** (h=1: 1x / h=3: 5x /
h=7: 13x / h=21: 41x) — the standard overlapping-window variance-inflation convention for an
h-day-forward-return series sampled daily (MA(h-1)-type serial correlation induced by the
overlap even under the null); this is a CONVENTION disclosed explicitly, not a measurement (the
true inflation is only knowable after fitting, from the real IC series' own autocorrelation —
reported per cell as part of the NW SE itself). `t_crit` = the Sec.7 Bonferroni value, 3.6047.

**MDE table (computed from the real panel, live-asset counts over 2024-01-01..2026-06-30,
liquidity-filtered universe, before any fit — values appended here and in
`RUN_MANIFEST.json`/`mde_table.json` by `run_grid.py`'s startup step, which runs
`mde_report.mde_table` BEFORE the first cell fit):**

**Measured 2026-09-28, real panel (fixture + 858 seam, liquidity-filtered, 384 assets after
the seam extension, before the by-cell present-mask restricts further), via `mde_report.py`
run BEFORE any model fit (`/var/tmp/915-run-smoke/mde_table.json`):**

| h | n_dates | N_bar (live assets) | NW inflation | MDE @ NW-t=3 (Bonferroni) | vs 0.02 band |
|---|---|---|---|---|---|
| 1 | 912 | 301.11 | 1x | 0.00573 | POWERED |
| 3 | 912 | 301.11 | 5x | 0.01282 | POWERED |
| 7 | 912 | 301.11 | 13x | 0.02068 | **UNDERPOWERED** |
| 21 | 912 | 301.11 | 41x | 0.03672 | **UNDERPOWERED** |

h=1 and h=3 are powered to detect the 0.02 band at this panel's breadth; **h=7 and h=21 are
pre-registered UNDERPOWERED under the 2h-1 NW-inflation convention** — a G-A FAIL at h in
{7,21} is uninformative about whether a real 0.02-IC signal exists there and must be reported
as such, never folded silently into "G-A fails everywhere" (Sec.10's kill-everywhere
consequence is conditioned on the POWERED cells, i.e. effectively h in {1,3}, unless a
passing h in {7,21} cell renders the power question moot by clearing anyway). This table's
qualitative CONVENTION (2h-1 inflation, the Bonferroni t_crit, the 0.02 band) is what this
SPEC's own sha256 pins; the measured N_bar/n_dates above are the real panel's, not assumed.

## 8. Fundamental law conversion (G-B) — DISCLOSED SCOPING

`gp_fundamental_law.py`. **N_eff and Sigma measured exactly as specified**: N_eff =
(sum(eig))^2/sum(eig^2) of the Ledoit-Wolf-shrunk (`sklearn.covariance.LedoitWolf`) MARKET-
RESIDUAL return correlation matrix (`resid_fwd` = the T1/T2 numerator, i.e. return net of the
measured beta*market — not raw returns, so the dominant crypto factor is removed before N_eff
is measured, per the memo's own framing of why N_eff << 64). BR = N_eff * 365/h.

**GP turnover/TC: a disclosed, fully-tested simplification, not the closed-form matrix
Riccati solution.** Solving the general (non-diagonal Lambda, non-diagonal Sigma)
Gârleanu-Pedersen matrix Riccati equation from scratch under this row's time budget risks
shipping an unverified closed form. Instead `gp_fundamental_law.simulate_gp_tradeoff` MEASURES
the GP trade-off DIRECTLY on the panel's own realised forecast path: aim_t = inverse-
(residual-)variance-weighted forecast, normalised to a fixed dollar-neutral gross per date
(Lambda assumed PROPORTIONAL to Sigma, `Lambda=kappa*Sigma`, the textbook special case that
collapses the matrix Riccati equation to a scalar one, applied uniformly across assets); a
scalar partial-adjustment trading rate `a` (x_t = x_{t-1} + a*(aim_t - x_{t-1})) is grid-
searched over 20 points in (0,1] to MAXIMISE the realised net Sharpe at the graded cost tier.
Turnover, gross Sharpe and net Sharpe at the optimal a* are all measured from the simulation,
not assumed. **TC is reported as the measured Clarke-de Silva-Thorley ratio** (net Sharpe at
a* / frictionless gross Sharpe at a=1), cross-checked against — never assumed equal to — the
memo's literature range 0.5-0.7 (`gross_ir_bound_literature_tc_lo/hi` reported alongside
`gross_ir_bound_measured_tc`). Tested (`tests/test_gp_fundamental_law.py`): turnover falls
monotonically as `a` falls; a zero-cost sweep prefers `a=1` (no reason to smooth without
cost); a higher cost tier never picks a HIGHER trading rate than a lower one.

**Denominations (D81(c))**: G-A's denominator is DATES (n_dates in the 2024-01..2026-06
window, overlapping for h>1, hence Newey-West lag=h); the per-date IC's own denominator is
live assets that date (N_bar in Sec.7's MDE table). G-B's BR is bets/year = N_eff (a
population-of-assets breadth measure, via the correlation spectrum) times 365/h (a rate); net
Sharpe is annualised (decisions_per_year = 365/h) throughout, matching the gate's own +0.97/
+0.56 bar, which is itself an annualised net Sharpe (per `docs/pathway.md`/819's convention).

**Cost bar — which one binds.** Two columns are computed and reported for every G-A-passing
cell: **legacy21** (21bp/side, the historical taker16+slip5 assumption the memo's G-B text
uses verbatim) and **restated_916** (80bp base-taker per `experiments/916_cost_truth/
build_benchmark.py`'s `FEE_TIERS_BP["base_taker80"]`, the July-2026 Kraken Level-1 schedule).
**The BINDING bar is restated_916**, per the task's precondition amendment ("TASK_916's
restated cost stack landed first... Grade at the 916-restated bar; do not hand-tune") — the
legacy21 column is reported for continuity with the memo's literal G-B text but never used to
call PASS/MARGINAL/FAIL (`gates.gb_result`, `binding: "restated_916"`). **Borrow charge
disclosure**: 916's full restatement additionally charges margin-rollover borrow on short
notional via `trader.backtest.borrow_window_report` — that machinery grades a REALISED BOOK
(short-fraction, holding period), which does not exist yet at the G-B stage (no book, no
positions to compute short-fraction from — the book is G-C, out of scope tonight). G-B's GP
simulation therefore charges the 80bp base-taker RATE as a linear cost on turnover (the
quantity GP's cost model consumes) and does NOT additionally charge borrow; this is flagged
explicitly as **a scoping gap, not a resolution** — a cell that clears G-B at 80bp base-taker
should be re-graded with borrow BEFORE being handed to G-C, and this SPEC does not claim the
GP bound already reflects borrow. G-B verdict thresholds unchanged from the memo: **PASS >=
+0.97, MARGINAL >= +0.56** annualised net Sharpe, applied to the restated_916 (80bp,
borrow-undercharged as just disclosed) column.

## 9. Gates (verbatim from the task file)

- **G-A**: >=1 cell, mean IC >= 0.02 on 2024-01->2026-06, NW t >= 3.6047 (Bonferroni over 160),
  IC > 0 in each of 2024/2025/2026H1 separately.
- **G-B**: fundamental-law net-Sharpe bound at the 916-restated tier >= +0.97 (MARGINAL >=
  +0.56) for some G-A cell.
- **G-C**: NOT BUILT TONIGHT (orchestrator scope). Named remainder: the GP closed-form book
  at the winning G-A/G-B cell, graded PIT on BOTH `trader/` (`trader.backtest.cell_metrics`)
  and `eval_vector.py --operator policy-argmax --vector-weight-rebalance` (the 882 two-
  instrument agreement discipline), net@21(916-restated) >= +0.97 CI lo > 0, BEAR22 positive,
  2026H1 unseen positive, holds positions, >=3 seeds. A future row picks this up if G-A/G-B
  clear; the DRO position gate (Wasserstein) is pre-named for that row, not built here.

## 10. Kill date and pre-committed consequences

Kill date: **14 days from this SPEC's commit** (2026-09-28 -> 2026-10-12). Cell grid fixed
above; any addition is a new row. Consequences per the task file Sec. "Pre-committed
consequences" (D79) apply verbatim: G-A fails everywhere -> recommend repealing D4 (OHLC-only
constraint), next row is a DATA row, not a model row, no further policy machinery on daily
bars under any architecture including MuZero; G-A passes/G-B fails -> record the measured
IC/N_eff/turnover triple, no trader build, breadth/cost are operator decisions; G-B passes ->
G-C is the next row.

## 11. Compute and provenance

CPU only, numpy/scipy/scikit-learn, `trader_env` on gpu-server (verified: sklearn 1.7.2, scipy
1.15.3; NOT installed: iisignature, esig, lightgbm, xgboost, statsmodels — no `pip install`
anywhere, `trader_env/` is PROTECTED). Loader/seam/module sha256 pinned in
`RUN_MANIFEST.json`. No DB writes anywhere in this module (`fixtures/trading_fixture.db`
opened `file:...?mode=ro`; the 858 CSVs are plain read-only files).

**Timed dry run and CPU-day estimate (measured 2026-09-28, real panel, gpu-server,
`OMP/OPENBLAS/MKL_NUM_THREADS=8`, `safe_run.sh --cpu --mem 16G`).** 10 real cells (target
y1, h=1, all 5 feature sets x 2 estimators — chosen because `grid.ALL_CELLS`'s iteration
order puts exactly this slice first, and it deliberately covers every feature-set family
including the two most expensive, F4 and ALL) over the FULL 2022-01-01..2026-06-30 expanding-
window range (18 quarterly refits per cell, not a shortened smoke window): ridge sum 20.9s,
GBM (3 seeds) sum 129.7s, total 150.6s for these 10 cells; panel load+feature build 152s
(ONE-TIME, shared by the whole grid); G-B evaluation for all 10 G-A-passing cells added
<1s (folded into the 301.5s measured total). **Extrapolation**: horizon and target do not
materially change per-cell fit cost (same expanding-window row counts; GBM/ridge cost is
driven by feature-set width and estimator, not by which target column or purge-by-a-few-days
h value is used), so the 10-cell sum scales by 4 targets x 4 horizons / (1 target x 1
horizon) = 16x: 150.6s x 16 = **2,409.6s (~40 minutes)** of cell-fitting, plus the 152s
one-time build and an estimated <200s of G-B overhead (10/10 cells passed G-A in the dry run,
so this is a conservative upper bound assuming most of the 160 cells also pass) = **~2,760s
(~46 minutes) total estimated wall time for the full 160-cell grid, single-threaded-equivalent
BLAS at 8 threads** — several orders of magnitude under the 2-CPU-day (172,800s) budget, so
the full run IS launched (Sec.13).

## 12. Adversarial-pass disclosure (D83(i)) — a real bug the dry run found and this SPEC fixes

The FIRST 10-cell dry run (pre-fix) showed F4 and ALL cells returning `mean_ic=NaN` for
every one of the 4 cells tested (F4/ridge, F4/gbm, ALL/ridge, ALL/gbm) — indistinguishable,
on its face, from "no signal in F4/ALL". Diagnosis (before trusting that result): F4's
`f4_l1_3` (the log-signature level-1 entry for the "time" path dimension — a DETERMINISTIC
constant of the window length, identical for every asset on every date) and every
`f4_l2_i_i` (log-sig level-2's antisymmetric diagonal, always exactly 0) have ZERO
cross-sectional variance. `grid.cross_sectional_zscore`'s original implementation mapped a
zero-std date to NaN, and `fit_cell`'s joint-finiteness mask
(`np.isfinite(X).all(axis=-1)`) requires EVERY selected column finite — so ONE always-
constant column silently zeroed out every training/test row for every cell that included it
(F4 and ALL: 64/160 cells, 40% of the grid), producing `NaN` mean_ic that reads identically
to "genuinely no signal" (exactly the D83(i) failure shape: "the cap never binds" and "the
cap was never applied" produce the same number). Verified directly on gpu-server
(`f4_l1_3` finite fraction after z-scoring: 0.0; joint-finite fraction across all 84 F4
columns: 0.0, confirmed by an ad-hoc diagnostic before any fix was applied).

**Fix**: a zero-cross-sectional-variance date now maps to `z=0` for every live asset that
date (the standard z-score convention for a constant column — centred value is identically
0 — rather than an undefined/NaN value), so it contributes no rank information without
poisoning any OTHER column's finiteness. Regression-tested
(`tests/test_zero_variance_column_guard.py`, two cases: a constant column alongside a real
one must not NaN the real one; a genuinely single-live-asset date is still handled). Re-run
post-fix: F4 (ridge mean_ic=0.0957 nw_t=30.05, gbm 0.1046/31.61) and ALL (ridge 0.1236/37.34,
gbm 0.1413/44.40) all clear G-A — the fix was necessary for the grid to measure what it
claims to measure, not merely cosmetic.

**Second adversarial finding, disclosed rather than quietly launched past**: the SAME dry
run's F1/F2/F3 cells (unaffected by the bug above) showed a per-date cross-sectional Spearman
IC of 0.04-0.13 at h=1 (NW-t 11-44) for `y1_primary_resid_yz` — roughly 3-4x the prior
literature ceiling this memo itself cites (`819 CAPACITY`'s pooled max 0.037). Before trusting
this for the launch decision, TWO independent negative controls were run on the SAME real
forecast array: (a) the target's DATE axis independently permuted (breaks the true forecast
<->target correspondence while preserving each series' own marginal/serial structure) —
mean_ic collapsed to -0.0006 (NW-t -0.17); (b) the per-date cross-sectional asset-label
permutation null (`stats_utils.per_date_permutation_null`, 100 draws) — mean -0.0004, sd
0.0033, against the real cell's 0.1124. Both controls collapse to noise while the correctly-
aligned data does not, which is consistent with a genuine (if economically surprising)
signal rather than a trivial alignment bug — and is ECONOMICALLY plausible: a broad
point-in-time universe includes many illiquid alt-coins (memecoins among the 858 extension
list), where strong 1-day bid-ask-bounce-driven reversal is a documented microstructure
effect, exactly the kind of signal 897 already found gets "killed by immediacy" once realistic
costs are charged. **This is exactly what happened**: all 10 dry-run cells that cleared G-A
(mean IC 0.04-0.14) FAILED G-B at the 916-restated (80bp) tier (`n_eff` measured 303.9 — a
generous, market-residual-spectrum breadth, so it is turnover/cost, not breadth, doing the
killing) — a coherent, on-theme result, not a red flag, but recorded here because a
surprising number this load-bearing gets its own adversarial pass before being trusted, per
this repo's own conduct rule, rather than being silently used to justify (or block) the
launch decision.

---

## POST-HOC ERRATA (appended 2026-09-28, TASK_915 cycle 2 — AFTER the official run's G-A results were seen)

Everything above this line is the pre-registration as pinned (sha256
`d9b7d4050345adee403e41c5918edce0957418f740a3d1739476c0d6b8c2c95e`, `RUN_MANIFEST.json`) and
is NOT edited. This section is post-hoc: it corrects statements above, records defects found
by the verification panel, and names the post-hoc instruments added. It changes no
pre-registered gate, cell, threshold or consequence. The pre-registered G-A/G-B outcome is the
official run's (`/var/tmp/915-run/grid_results.json` on gpu-server, frozen code shas in
`RUN_MANIFEST.json`); post-hoc results are reported beside it, never substituted for it.

### E.1 Same-close alignment (BLOCKER-grade, correctness judge) — Sec.3 statement is wrong

Sec.3 says "every target's label starts at row t+1". It does not: `targets.forward_log_return`
is `lc[t+h] - lc[t]`, so every label STARTS at close_t, and every F1 feature ends at close_t
(`features.build_f1`: `lc[t] - lc[t-lb]`). No future data is read (the leak-free test in
`tests/test_targets.py` is correct as far as it goes), but feature and label share the SAME
print, close_t. Any idiosyncratic noise e_t in that print (bid-ask bounce, a stale/last-trade
close) enters the F1 feature with sign +1 and the label with sign -1, producing a negative
feature/label covariance of about -var(e) that no strategy trading after close_t can earn.
Judge measurement (gpu-server `/var/tmp/915-judge-leak/skipday_test.py`, frozen module code
as pinned in `RUN_MANIFEST.json`, target y1, ridge): the one-day-skip label (close_{t+1} ->
close_{t+1+h}, features/beta/sigma still at t) gives

| cell | official mean IC (NW-t) | skip1 mean IC (NW-t) |
|---|---|---|
| F1 h1 | 0.1191 (35.21) | 0.0123 (3.56 < crit 3.60) |
| F1 h3 | 0.0887 (22.29) | 0.0034 (0.61) |
| F2 h1 | 0.0442 (11.49) | 0.0401 (11.14) |
| F2 h3 | 0.0603 (11.30) | 0.0539 (10.15) |

3.73% of present (asset, day) cells have close == previous close (judge,
`/var/tmp/915-judge-leak/stale_check.py`, same panel) — consistent with stale closes, not
proof of them. The cause of the F1 timing sensitivity is therefore named as **same-close
alignment (bid-ask bounce / stale-close consistent, not individually established)**. F3/F4/ALL
were untested by the judge; `posthoc_skip1.py` re-scores every official G-A pass.

**Why the pre-registered null family is blind to it.** The three nulls
(`stats_utils.circular_shift_null`, `block_rademacher_null`, `per_date_permutation_null`,
lines 130-183) and Sec.12's two negative controls all DESTROY the date-t forecast<->target
alignment and ask whether the aligned statistic beats that. A same-close artifact IS an
aligned association (the shared e_t exists only at the true alignment), so it collapses under
every such null exactly as a genuine signal would; Sec.12's reading of the collapse as
"consistent with a genuine signal rather than a trivial alignment bug" does not follow. The
discriminating control is a TIMING control — delay the label, keep the alignment — which none
of the nulls implement. `tests/test_skip1_bounce.py` plants a pure bid-ask-bounce process
(driftless efficient price, iid +/- half-spread prints): the official pipeline PASSES G-A on
it, the skip1 re-score collapses it and flags `timing_fragile`; a genuine AR(1) process stays
passing under skip1.

### E.2 MDE table used t=3.0, not the stated t_crit (MAJOR, pre-registration judge)

Sec.7 says `t_crit = 3.6047` enters the MDE; `mde_report.py:41` calls `stats_utils.mde_ic(...,
power_t=3.0)`, so the pinned table is at t=3.0. Recomputed from the same panel counts
(n_dates=912, N_bar=301.11, 2h-1 NW inflation; `annotate_results.mde_both`):

| h | MDE @ t=3.0 (pinned) | flag | MDE @ t=3.6047 | flag | flips? |
|---|---|---|---|---|---|
| 1 | 0.00573 | POWERED | 0.00689 | POWERED | no |
| 3 | 0.01282 | POWERED | 0.01541 | POWERED | no |
| 7 | 0.02068 | UNDERPOWERED | 0.02484 | UNDERPOWERED | no |
| 21 | 0.03672 | UNDERPOWERED | 0.04412 | UNDERPOWERED | no |

No POWERED/UNDERPOWERED flag flips; the pinned table's values are reproduced exactly at t=3.0.
Both flags are carried per cell in `results/grid_results_annotated.json`.

### E.3 G-B carries no machine-readable borrow flag (MAJOR)

Sec.8 discloses that G-B charges the 80bp base-taker rate on turnover and NOT 916's borrow,
but the frozen `gates.gb_result` returned only `verdict`, so a consumer of
`grid_results.json` reading a G-B PASS would be fail-open. Annotation adds `borrow_charged:
false, provisional_if_pass: true` to every official G-B entry (without changing any number).
For FUTURE runs `gates.gb_result` now returns both keys (post-run code change, see E.5).

### E.4 MINOR

Per-cell rows lacked the horizon's underpowered flag (added by annotation, both t
conventions). `N_TESTS = 160` was a literal in `gates.py:14` duplicating `grid.ALL_CELLS`; now
derived as `len(grid.ALL_CELLS)` for future runs (value unchanged). `mde_report.py` still
carries its own `N_TESTS = 160` and `power_t=3.0` — left as frozen (not in this pass's scope).

### E.5 Post-hoc instruments and post-run code changes

- `posthoc_skip1.py` (POST-HOC): every official G-A-passing cell re-scored with the one-day-skip
  label through `grid.fit_cell` / `gates.cell_ic_stats` / `gates.ga_pass` (same seeds, refit
  schedule, purge, Bonferroni crit, per-year rule); G-B via `gates.gb_result` on skip1
  survivors. Output `results/posthoc_skip1.json`.
- `annotate_results.py`, `make_verdict.py`: annotation and `results/VERDICT.json`, computed
  from the result files.
- `gates.py` post-run change (does not affect any number the official run produced):
  frozen sha `4d9ffad1372ad6de8aaec4ed5a78b48d1ac254f0c2fcd46f5e37e7bea8a3b818` -> post-run sha
  `8ecafd13f09b64c6d931197b7d0974ae62d4299fe100740e3232a9b28069f0a6`. The official run used the frozen file.

### E.6 Post-hoc outcome (numbers from `results/posthoc_skip1.json` / `results/VERDICT.json`, gpu-server run 2026-09-28, official `grid_results.json` sha256 `19eef63abafa0edffa71b39c3d229fa29b96e4d4942f30f872f5b20ac16e17d9`)

Pre-registered: G-A 128/160 pass; G-B 127 FAIL + 1 MARGINAL (`y4_sign_volscaled__h1__ALL__ridge`,
+0.693 at restated_916, borrow uncharged), no PASS. Post-hoc skip1: 109/128 survive G-A
(F1 13/27, F4 24/27, ALL 24/26, F2 24/24, F3 24/24; y4 0/10); the MARGINAL cell does not
survive (skip1 IC 0.0039, t 0.46); all 109 survivors FAIL G-B (best +0.037). The
pre-registered consequence is unchanged by the post-hoc result: G-A passes / G-B fails.

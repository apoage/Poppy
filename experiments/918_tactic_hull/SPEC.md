# TASK_918 — tactic library + convex-hull ceiling: pre-registration (SPEC)

Implements `tasks/TASK_918_TACTIC_LIBRARY_HULL_CEILING.md` (2026-09-28 amendment: read-only
data precondition). This file is written and its sha256 frozen **before any real-data run**
(gate iii style, following `experiments/898_trend_decay`'s precedent) — see `run_step0.py`'s
`require_prereg()` and the result JSON's `spec_sha256` field. If SPEC.md changes after the
first real run, that fact is disclosed in the implementer's report, never silently.

## 1. Library — K = 8 position matrices (`tactics.py`)

All 8 arms are built on ONE shared present-mask / warmup grid (`tactics.shared_grid`):
`SHARED_MAX_LB = 90` days (the longest lookback in the library — the beta-hedge's trailing
90d BTC-beta window), `SHARED_VOL_WINDOW = 30` days (819 convention). This is a **deliberate,
pre-registered simplification**: shorter-lookback arms (7d/21d reversal, native lookback 7/21d)
lose a few extra weeks of per-asset eligibility right after that asset's listing that their own
native lookback would have granted. It costs nothing mid-window and buys a mixable/nettable
grid across all 8 arms (the hull's turnover must be computed on the MIXED book — that requires
every arm to occupy the same (row, column) space). Disclosed, not silently varying per arm.

1. **flat** — zero positions everywhere (a cash book).
2. **trend_819** — the frozen 819/820 rule: single 30d sign trend, equal weight, gross ≤ 1
   (`trader.strategy.trend_signal(logc,[30])` + `weights(pres,rv,'equal')`), rebuilt on the
   shared grid (a strict narrowing of the rule's own native 30d mask, never an inflation).
3. **long_flat_trend** — TASK_898's variant: same signal, shorts replaced by flat
   (`sig = max(sig, 0)`), same shared-grid weighting. Reproduces
   `experiments/898_trend_decay/decay_harness.py:99-120`'s recipe from the same
   `trader.strategy` primitives (not imported across the digit-prefixed package — attributed).
4. **equal_weight_cm** — always-long, equal weight over present assets, daily-rebalanced to the
   constant target 1/n_present (the "equal-weight constant mix" family; same construction as
   `trader.backtest.regime_series`'s internal `bh` column, but a genuine tradeable book here,
   not a zero-turnover benchmark column).
5. **beta_hedged_cm** — arm 4 hedged against trailing-90d BTC beta: for each present asset i,
   `beta_i,t = trailing-90d Cov(r_i, r_BTC)/Var(r_BTC)` (leak-free, cumulative-sum rolling
   covariance, same mechanics as `trader.strategy.rolling_vol`); hedge leg
   `pos[t,BTC] -= Sum_i w_ew[t,i]*beta_i,t`. Matches the K-tactic explorer's T4 definition
   (`lore/harvest_6b32_appendix/tactics_gaps.md:14`: "T4 beta-hedge = trailing-90d BTC beta").
   Gross is **not** re-capped after the hedge leg (disclosed). If BTC/XBT is absent at row t,
   the hedge term is 0 that row (disclosed via `beta_hedged_cm.btc_present` diagnostic).
6. **vol_targeted_trend** — arm 2 (trend_819) rescaled to a 40%/yr **portfolio** vol target on
   the book's own trailing-30d realized daily P&L (leak-free: the row-t scale uses only P&L
   realized on rows < t), gross-capped at 1 (multiplier ≤ 1, never levers up). Matches the
   sizing explorer's T5 (`tactics_gaps.md:14`: "40%/yr target on trailing-30d close-to-close
   vol, gross cap 1") and the panel's own stated ceiling (`fable_tactics_panel_2026-09-04.md:36`:
   "its analytic ceiling is √(1+CV²), net Δ ∈ [−0.10,+0.08]").
7. **reversal_7d_ts** — 7d **time-series** reversal = `-sign(logC_t - logC_{t-7})`, equal
   weight over present assets (label correction per `tactics_gaps.md:7`: this leg is
   time-series, NOT cross-sectional — the distillation mislabelled it).
8. **reversal_21d_xs** — 21d **cross-sectional** reversal, tercile long/short: each day, rank
   present assets by 21d log return; long the bottom tercile (worst performers) equal-weight
   summing to +0.5 gross, short the top tercile equal-weight summing to −0.5 gross, middle
   tercile flat. Dollar-neutral (gross = 1) when both legs are non-empty; a day with < 3
   present assets degrades to whichever leg exists, capped at its own 0.5 gross (disclosed via
   `reversal_21d_xs.days_with_lt3_present`, never silently re-normalized to gross 1).

`n_assets` actually loaded is reported with full provenance (loader/fixture/seam sha256) rather
than forced to match the task file's unattributed "273" figure (D83(ii): recompute, don't
requote) — see §4.

## 2. Scoring (`scorer.py`)

Reuses ONLY `trader.backtest.regime_series` / `net_series` / `_sharpe` /
`_block_bootstrap_sharpe_ci` / `gross_simple_series` / `borrow_cost_series` /
`net_series_with_borrow` — the same primitives `experiments/898_trend_decay/
decay_harness.py:score_window` calls (never `cell_metrics`, per the task file). `h_grid`
reproduces `regime_series`'s own H-strided row derivation as pure bookkeeping (asserted
byte-identical in size to `regime_series`'s own returned arrays at import-check time).

- **H = 1** (daily rebalance, the 819/898 convention). "net@21" denotes the **21 bp/side FEE
  TIER**, not a 21-day horizon — `trader.backtest`'s own `net{c}` naming convention.
- **Fee tiers** (bp/side, one-sided on turnover): `legacy21=21` (record continuity),
  `base_maker40=40`, `base_taker80=80` — the exact numbers TASK_916 verified and shipped
  (`experiments/916_cost_truth/build_benchmark.py:36`, cited not re-fetched), plus a `zero`
  (uncosted gross) column.
- **Borrow**: `trader.backtest.borrow_cost_series`/`net_series_with_borrow` at both `low`/`high`
  scenarios (TASK_916 leg 1, unmodified), reported at every fee tier plus its own
  block-bootstrap CI at net@21+borrow.
- **Log AND simple returns both**: `gross_log` (canon `Sum pos·Δlogc`, cannot see a
  rebalancing premium by construction — `trader/backtest.py:221-223`) AND `gross_simple`
  (`trader.backtest.gross_simple_series`, dollar-weighted simple forward return) are both
  scored at every fee tier.
- **Block-bootstrap CI FIRST**, computed on `net{21}` (log, no borrow) — the PRIMARY STEP-0
  kill reading (see §5) — via the unmodified `_block_bootstrap_sharpe_ci` (`block_len =
  max(5, ceil(2))` = 5 steps at H=1; disclosed as under-covering monthly-scale persistence,
  per `tactics_gaps.md:36` — the reason §3's `n_decisions_monthly` denomination exists at all).
- **By-year**: 2024 (calendar), 2025 (calendar), 2026H1 (2026-01-01..2026-07-01) sub-windows,
  scored identically.

## 3. D81(c) denomination — `n_decisions_monthly`

The STEP-0 window (2024-01-01 .. 2026-07-01) spans **exactly 30 calendar months**. Per the
task file's explicit instruction, this — NOT the ~870 daily H=1 decision rows the Sharpe
arithmetic is actually computed from — is the denominator for every STATISTICAL POWER claim
(SR0(N), MDE). Rationale (not just an instruction to follow blindly): `tactics_gaps.md:36`
records that `block_len=5` (steps) under-covers a monthly-persistent trend/reversal signal's
true autocorrelation scale, so the NOMINAL daily-row n overstates power; the window's own
30-month span is the honest independent-observation count for these arms. Every number in the
result JSON carries `n_decisions_monthly=30` (or the exact count for a by-year sub-window)
beside it, plus `n_daily_decisions` (the actual H=1 row count the Sharpe was computed from) —
both reported, never only one.

## 4. Data (`loader.py`)

- Universe: `trader.data.load_db_backtest_panel(db_path=fixtures/trading_fixture.db,
  source_filter='kraken_rest', include_dead=True)` — USD-quoted, non-stable/fiat,
  dead/delisted-inclusive (the "point-in-time dead-coin-inclusive" convention
  `docs/pathway.md:12` calls PIT). Never opens `trading_data.db` (protected; the fixture's
  `ohlcv_daily` is the verified byte-identical copy other 91x rows already rely on —
  `experiments/917_second_moment_and_event_instruments/daily_panel.py`'s own docstring).
- Extension: the staged 858-backfill CSVs (`runs/858_backfill/api_2026-08-31/daily/*.csv`,
  manifest `runs/858_backfill/api_2026-08-31/manifest.json`) extend the 31 USD-quoted canon
  pairs' date grid past the fixture's own last stored day (2026-07-06) through 2026-08-31.
  **Seam RE-VERIFIED at load time** (`loader.verify_and_extend_seam`): for each of the 31
  assets, every DB/CSV overlapping day's close is compared (rel/abs tol 1e-9); ANY mismatch
  raises and refuses the whole load. The 2026-08-31 record ("62 overlap rows, zero
  mismatches") is a LEAD, not an assumption — this pass's own re-derivation on the FROZEN
  fixture (a different, earlier DB snapshot than the 2026-08-31 canon-DB check) measured
  **488 overlap rows across 31 assets (≈16/asset), zero mismatches** — a different count than
  the historical record because the fixture (frozen 2026-08-05, kraken_rest max 2026-07-06) is
  an earlier snapshot than the canon DB was at 2026-08-31; both zero-mismatch results are
  disclosed, and the fixture's own re-derived number is what this artifact's seam hash actually
  certifies.
- Assets NOT among the 31 extended pairs simply end at the fixture's own last stored row for
  that asset; the pre-existing `trader.data._ffill_bfill_cols` convention (already used
  throughout `trader/` for dead/delisted assets, e.g. LUNA) carries their last real price
  forward — this loader adds no new fabrication, and the shared present-mask (§1) retires them
  from every arm ~90 days after their true last print regardless.
- Read-only everywhere: fixture opened via `trader.data._connect(..., read_only=True)`
  (`file:...?mode=ro`); CSVs read as plain files. No DDL, no writes, no fixture regeneration.

## 5. STEP-0 (this pass's actual run) — hull, comparators, kill

Window: **2024-01-01 .. 2026-07-01** (the task's "2024-01 → 2026-06" read as a half-open
`[start, end)` bound so 2026-06 is fully included).

- **Hull**: best FIXED mixture (weights on the simplex, `hull.py`) of the 8 position matrices,
  turnover computed on the MIXED book. Search method: exhaustive coarse grid at step 0.1 (all
  compositions of 10 units into 8 non-negative integer parts, C(17,7) = 19448 points) on the
  fast net@21 (log) objective, then `scipy.optimize.minimize` (SLSQP, `sum(w)=1`, bounds
  `[0,1]`) local refinement from the grid's argmax; the refined point is reported only if it is
  not worse than the coarse grid's own argmax (never regress on the search's own best-so-far).
  Resolution (0.1 coarse / continuous refinement) and both points are disclosed.
- **Ex-post oracle single-tactic selector**: the single arm (of the 8) with the best net@21
  (log) Sharpe over the whole window — a one-hot corner of the same simplex the grid already
  visits.
- **Equal-weight blend** (the BINDING null per `tactics_gaps.md:34`/`fable_tactics_panel_
  2026-09-04.md`'s "Record corrections... equal-weight constant mix"): the FIXED 1/8-each
  mixture, not searched.
- **21d cross-sectional reversal book**: arm 8 alone (the never-measured H2 read,
  `docs/pathway.md:38`).
- **Each of the 8 arms alone.**
- **KILL LINE**: `hull net@21 (log, no borrow) < +0.97` ⇒ record hull/oracle/blend/reversal-book
  numbers and STOP — no walk-forward/Hedge/random-null/B0 rows built. Read at plain net@21
  (§2), the SAME denomination the +0.97 bar was measured in (819, `docs/pathway.md:12`).
  **Direction argument (why net@21 plain, not borrow-charged, is the correct — and only
  favorable — kill reading)**: for a FIXED position book, `net_series`/`net_series_with_borrow`
  only ever SUBTRACT from the same gross series (borrow_cost ≥ 0 pointwise, `borrow_cost_series`
  docstring) — so net@21 plain is the single MOST FAVORABLE honest reading among {net@21 plain,
  net@21+borrow, net@40, net@80, net@80+borrow}; killing on the most favorable reading can only
  UNDER-kill (report a pass where a stricter reading would also fail), never over-kill. The
  80bp/borrow columns are reported for completeness (mandatory per the task file) but are NOT
  the kill's own gate.
- If the kill does NOT fire, this pass's scope stops at recording that fact — building the full
  comparator set (walk-forward, Hedge, random-null 200 seeds, B0 tactic-momentum selector) is
  explicitly OUT OF SCOPE for this implementer pass per the task file's STEP-0 gate wording
  ("if it does not fire, continue"); the implementer's report states this as a limitation
  requiring a follow-on pass, not a silent omission.

## 6. SR0(N) and MDE (`stats.py`)

- **n_decisions_monthly = 30** (§3).
- **SR0(N)**: `E[max of N iid standard normals] * SE_annual_null(n_months)`, where
  `SE_annual_null(n) = sqrt(12/n)` (a monthly-null Sharpe estimator has SE ≈ 1/sqrt(n) under a
  zero-mean null; annualizing multiplies both the estimator and its SE by the same sqrt(12)
  factor) and `E[max of N iid N(0,1)]` uses the standard Gumbel-tail approximation
  `sqrt(2 ln N) - (ln ln N + ln 4π)/(2 sqrt(2 ln N))`. Reported for **N = 8** (the K tactics —
  matching the panel's own precedent reference, `tactics_gaps.md:19`: "oracle-null selection
  inflation ≈ +0.5–0.75... at this N"), with an explicit disclosure that the hull's own
  continuous-simplex search is a strictly LARGER hypothesis space than N=8 discrete corners, so
  SR0(8) is a lower bound on the hull's true hindsight-inflation, and that N=8 independent is
  itself an upper bound on the true (correlated) effective count — both biases stated, neither
  silently netted against the other.
- **MDE95 (D85)**: minimum detectable Sharpe DIFFERENCE between two candidates at
  `n_decisions_monthly`, `MDE95 = 1.96 * sqrt(2) * SE_annual_null(n_months)` (treating the two
  compared candidates as independent — conservative; true correlation from scoring on
  overlapping data would only shrink the true MDE further, making any PASS-direction claim
  inside this MDE even more clearly UNDERPOWERED, never less). Cross-checked against the
  REALIZED block-bootstrap CI half-width on this window's own hull (`stats.bootstrap_se_from_ci`).
  Computed once, at spec/first-run time, and compared against the task's own PASS-conjunct
  margins (the ≥ 0.20 beat-blend / beat-walk-forward margins) — if MDE95 exceeds 0.20, ANY
  PASS-direction "beats the blend/walk-forward by ≥ 0.20" claim on this window is UNDERPOWERED
  by construction and must be reported as such, never as a clean pass. **The KILL direction is
  unaffected by this**: STEP-0's gate is a ceiling test (hull below the bar closes the class),
  which is decisive regardless of MDE — a ceiling that is clearly below the bar does not need
  power to reject "above the bar," only to confirm "below" (MDE bounds are about resolving
  small DIFFERENCES near a boundary, not about a value that lands far below it with a CI that
  itself excludes the bar).

## 7. Scope discipline

New module only (`experiments/918_tactic_hull/`), no edits to `trader/`, no imports-and-mutates
of `experiments/898_*` or `experiments/916_*` (their functions are cited/reproduced, not
imported cross-package, to avoid the digit-prefixed package-name issue — see §1/§2). No DB
writes anywhere. No fixture regeneration. Out of scope (per the task file): learning inside a
selector (915-downstream), execution tuning (919), scorer changes (920).

#!/usr/bin/env python3
"""TASK_919 phase A -- the rebalance grid: H x theta x accounting, scored net of fees + borrow.

Spec of record: tasks/TASK_919_EXECUTION_TURNOVER_MODULE.md. Implementation brief (factory
run 55, orchestrator) RESOLVES the under-specified parts and CORRECTS three of the task
file's own claims -- every correction is re-derived here, not transcribed; see RESULTS.md for
the adjudication of confirmed / refuted / amended.

Calls ONLY existing public `trader.backtest` / `trader.strategy` / `trader.data` functions for
every mechanism this pass does NOT introduce (drift-retrim turnover, simple-return gross,
borrow charge, Sharpe/bootstrap CI, the H-strided decision-grid contract) -- no modification,
no monkeypatch of trader/. The only NEW mechanism here is the theta sign-flip hysteresis
(section "hysteresis" below) and the accounting-mode wiring around it; both are additive,
module-local, and never touch `trader/`.

Read-only against fixtures/trading_fixture.db (never trading_data.db, never regenerated).
CPU-only, seconds of runtime for the full 48-cell grid on both windows.

PROVENANCE / DATA IDENTITY (measured by the orchestrator, re-verified here at HEAD):
  - fixtures/trading_fixture.db sha256 b0f8de8b108b50521f301df1d0856aed72bc760c57b67a842fa62fa
    918206cf9 (mtime 2026-08-05 07:12) -- the canonical baseline every committed number in this
    repo is promoted against. print_fixture_sha() reprints this at run time.
  - `ohlcv_daily` (the table this module's panel is built from, via
    trader.data.load_db_backtest_panel) is BYTE-IDENTICAL between the fixture and the 24 GB
    trading_data.db source of record (499,430 rows, 411 assets, 2013-10-06..2026-07-06, sha256
    90d67f40... on both sides -- independently reproduced here, matching TASK_917's own
    changelog hash). So the 48-cell grid below scores PRODUCTION numbers, not a fixture proxy.
  - `ohlcv` (the 1-MINUTE table) is NOT at parity: fixture holds 11 assets / 2025-09-09..
    2025-12-12 (474,343 rows) vs trading_data.db's 1,130 assets / 2013-09-10..2026-03-31
    (254,904,234 rows). This is why section 7 (hour-of-day boundary) implements the mechanism
    and proves it correct on the 11-asset slice ONLY, and hands the real leg to the planner as
    a scheduled run against trading_data.db -- see hour_of_day.py.

WINDOWS (THREE, all reported, never conflated -- coordinator note run 55, corrected in the
closing pass: the brief named two and the shipped artifact carries three, the third being the
only genuinely out-of-sample read here):
  PRIMARY   = trader.config.REGIMES "FULL" = 2021-01-01..2025-09-01 (n=1704 at H=1 canon).
              This is what cell_metrics/compute_full_cell natively report and what every G1
              number below is denominated on.
  SECONDARY = runs/819_single_asset/powered.py's FULL_START/FULL_END = 2018-01-01..2025-09-01
              (n=2800 at H=1), TASK_916's pre-committed PRIMARY anchor
              (experiments/916_cost_truth/anchor_reproduction.py). Reported for the frozen
              cell and the grid's best cell so the two families connect; NEVER the grid's own
              denominator. It is a strict SUPERSET of PRIMARY, so it is NOT out-of-sample.
  DISJOINT  = 2018-01-01..2021-01-01 (n=1096 at H=1) = SECONDARY minus PRIMARY. Added by the
              cycle-2 fix pass (MAJOR-E2) as the ONLY genuinely held-out read. A post-hoc
              holdout, not a pre-registered one, and reported as such.
  Every figure in the result JSON / RESULTS.md carries an explicit "window" field naming which
  of the three it is scored on -- the 920 honesty-pack convention (backtest.honesty_pack_report).
"""
import dataclasses
import datetime as _dt
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trader import config as tconfig
from trader import backtest as bt
from trader import data as tdata
from trader import strategy as tstrat

FIXTURE_DB = str(_REPO_ROOT / "fixtures" / "trading_fixture.db")
FIXTURE_SHA256 = "b0f8de8b108b50521f301df1d0856aed72bc760c57b67a842fa62fa918206cf9"

# --- the grid (48 cells) -----------------------------------------------------
H_GRID = (1, 2, 3, 5)
THETA_GRID = (0.0, 0.25, 0.5, 1.0)
ACCOUNTING_GRID = ("target-weight", "drift-inclusive", "hold-shares")
N_CELLS = len(H_GRID) * len(THETA_GRID) * len(ACCOUNTING_GRID)
assert N_CELLS == 48

FEE_TIERS_BP = (8, 12, 21)                 # task's own scoring set
FEE_TIERS_916 = {"maker40": 40, "taker80": 80}   # 916's verified ladder, reported alongside
BORROW_SCENARIOS = ("none", "low", "high")
PRIMARY_FEE_BP = 21
PRIMARY_BORROW = "low"

ALPHA_FAMILYWISE = 0.05
BONFERRONI_ALPHA = ALPHA_FAMILYWISE / N_CELLS      # 48-cell family, per D81(c) brief section 2

L_TREND = 30                                # trend lookback (StrategyConfig.lookbacks=(30,))

PRIMARY_WINDOW = {"name": "primary_config_full", "start": "2021-01-01", "end": "2025-09-01"}
SECONDARY_WINDOW = {"name": "secondary_powered_registered", "start": "2018-01-01", "end": "2025-09-01"}
# MAJOR-E2 fix (run 55 cycle 2): SECONDARY is a strict SUPERSET of PRIMARY, so scoring a
# PRIMARY-selected cell on SECONDARY is dilution-by-extra-years, NOT an out-of-sample check.
# DISJOINT is SECONDARY minus PRIMARY -- the only genuinely held-out span available here.
DISJOINT_WINDOW = {"name": "disjoint_pre_primary", "start": "2018-01-01", "end": "2021-01-01"}

# MAJOR-F2 fix: what quantity each accounting mode's `turn` is actually denominated in.
COST_BASIS = {
    "target-weight": "target_change_only (Sum|d target|; drift assumed free) on the TARGET-WEIGHT book",
    "drift-inclusive": "upper_bound_unnetted (Sum|d target| + Sum|drift|; a triangle-inequality "
                       "UPPER BOUND, per drift_retrim_turnover's own docstring) on the TARGET-WEIGHT book",
    "hold-shares": "hold_shares_actual (Sum|w_new - w_drifted|; the single netted trade) on the "
                   "HOLD-SHARES book -- a DIFFERENT position path, not a different charge on the same one",
}


def print_fixture_sha():
    import hashlib
    h = hashlib.sha256()
    with open(FIXTURE_DB, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    match = "MATCHES the recorded baseline" if got == FIXTURE_SHA256 else "DOES NOT MATCH -- fixture changed!"
    print(f"fixtures/trading_fixture.db sha256 = {got} ({match})")
    return got


# =============================================================================
# panel loading
# =============================================================================

def load_pit_panel():
    """PIT 363-active-symbol daily panel via trader.data.load_db_backtest_panel (unmodified),
    same construction experiments/916_cost_truth/build_benchmark.py's 'PIT (db/fixture)' panel
    and the orchestrator's own measurement use. Returns (logc, ts_d, syms, info)."""
    ts_d, close_d, ids, syms, info = tdata.load_db_backtest_panel(
        db_path=FIXTURE_DB, include_dead=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where(close_d > 0, close_d, np.nan))
    return logc, ts_d, syms, info


def window_bounds(ts_d, start, end):
    """searchsorted row bounds [r_lo, r_hi) for a UTC-midnight [start,end) window, same idiom
    as anchor_reproduction.py:108-109 / backtest.regime_rows."""
    r_lo = int(np.searchsorted(ts_d, bt._to_epoch(start)))
    r_hi = int(np.searchsorted(ts_d, bt._to_epoch(end)))
    return r_lo, r_hi


# =============================================================================
# base arrays (shared across the whole grid; theta/H only change what's derived below)
# =============================================================================

def build_base_arrays(logc, scfg=None):
    """Rebuild the SAME primitives trader.strategy.build_positions computes (rv, presence,
    weights), kept SEPARATE from the sign so theta-hysteresis can be applied to the sign only.
    Also returns S_full, the raw (non-sign) trend statistic logc[t]-logc[t-L] -- NaN warmup,
    leak-free (rows <= t only, same contract as trend_signal/rolling_vol).

    S_full's sign is bit-identical to trader.strategy.trend_signal's single-lookback body
    (both are sign(logc[L:]-logc[:-L])) -- verified in test_919_grid.py G1."""
    if scfg is None:
        scfg = tconfig.STRATEGY
    L = scfg.lookbacks[0]
    assert L == L_TREND and len(scfg.lookbacks) == 1, \
        "919A grid assumes the frozen single-30d-lookback cell; scfg changed upstream"
    rv = tstrat.rolling_vol(logc, scfg.vol_window, xp=np)
    fa = tstrat.first_active(logc, xp=np)
    pres = tstrat.present_mask(logc, rv, fa, L, scfg.vol_window, xp=np)
    w = tstrat.weights(pres, rv, scfg.sizing, xp=np)
    T, A = logc.shape
    warm_pad = np.full((L, A), np.nan)
    S_full = np.concatenate([warm_pad, logc[L:] - logc[:-L]], axis=0)
    warm = max(L, scfg.vol_window) + 1
    return rv, pres, w, S_full, warm


def decision_grid_rows(T, r_lo, r_hi, H, warm):
    """The SAME lo/hi/t formula regime_series/_decision_grid derive internally (kept in
    lockstep deliberately -- this is the G1 no-op contract). Returns t (decision rows) or None
    if the window is too thin."""
    lo = max(warm, r_lo)
    hi = min(r_hi - H, T - 1 - H)
    if hi - lo < H:
        return None
    t = np.arange(lo, hi + 1, H)
    if t.size < 6:
        return None
    return t


# =============================================================================
# theta sign-flip hysteresis (the ONE new mechanism this task introduces)
# =============================================================================

def hysteresis_build(rv, pres, w, S_full, t_ext, theta, L=L_TREND):
    """Retained-sign hysteresis over the DECISION grid `t_ext` (rows spaced exactly H apart --
    caller must pass the H-strided grid, INCLUDING one extra leading row t_ext[0] = t[0]-H so
    the first scored decision's turnover has a defined 'previous' position). State carries
    ACROSS DECISION POINTS ONLY (G3): the loop below advances its per-asset carried-sign state
    once per row of `t_ext`, and `t_ext` never contains a non-decision daily row -- so a planted
    panel where the daily-row scan and the decision-grid scan disagree (see test G3) exercises
    exactly this contract.

    Rule per asset i at decision row `row`:
      - not present (pres[row,i] False)              -> sign 0 (canon convention)
      - theta == 0                                    -> sign(S[row,i]) (no hysteresis; G1)
      - present at both `row` and the PRIOR decision AND |S[row,i]| <= theta*sigma_S[row,i]
                                                        -> RETAIN the prior decision's sign
      - otherwise (theta>0, band cleared, OR just re-entered presence -- reset, no retained
        state to inherit)                              -> sign(S[row,i]), and this becomes the
                                                           new carried sign for future rows
    sigma_S[row,i] = rv[row,i] * sqrt(L) -- the sd of the L-day trend statistic under the
    leak-free trailing-vol estimator (rows <= row only, same contract as rolling_vol -- G4).

    Returns (pos_ext, sign_ext, pres_used_ext) each shaped (t_ext.size, A).
    """
    n, A = t_ext.size, rv.shape[1]
    pos = np.zeros((n, A))
    sign_used = np.zeros((n, A))
    carried = np.full(A, np.nan)
    prev_present = np.zeros(A, dtype=bool)
    sqrtL = np.sqrt(L)
    for k in range(n):
        row = int(t_ext[k])
        pr = pres[row]
        raw_sign = np.sign(S_full[row])
        if theta == 0.0:
            use_sign = raw_sign
        else:
            sigma_s = rv[row] * sqrtL
            has_carry = pr & prev_present & np.isfinite(carried)
            within_band = np.abs(S_full[row]) <= theta * sigma_s
            use_sign = np.where(has_carry & within_band, carried, raw_sign)
        use_sign = np.where(pr, use_sign, 0.0)
        pos[k] = w[row] * use_sign
        sign_used[k] = use_sign
        carried = np.where(pr, use_sign, carried)
        prev_present = pr
    return pos, sign_used


# =============================================================================
# hold-shares: a REAL share-carrying book (fix pass, run 55 cycle 2 -- BLOCKER-C1)
# =============================================================================

def hold_shares_walk(price, pos_ext, sign_ext, t_ext, H):
    """A GENUINE share-carrying book: shares are struck at a decision price and CARRIED across
    decisions; weights drift with price; only the SIGNAL change is traded.

    WHY THIS EXISTS (fix pass, run 55 cycle 2). The cycle-1 pass labelled a log-vs-simple
    compounding swap "hold-shares": all three accounting modes scored the SAME target-weight
    position array, so no share count was ever carried and the accounting axis was not a
    treatment arm at all (correctness + estimator lenses, independently). This function is the
    replacement. It is modelled on `trader.backtest.true_bh_series` (920) and REUSES that
    function's conventions rather than inventing new ones:
      - the share-count walk over calendar rows, marking the book with `bt._book_value`
        (columns holding EXACTLY zero shares contribute exactly 0.0 even where their price is
        NaN -- so a non-finite mark means a HELD asset had a bad price);
      - the "liquidate at last print" DELISTING convention (`_book_value`'s docstring): a column
        the present-mask has RETIRED keeps marking at its frozen ffilled price until the book
        next trades it, and it IS traded out at the next decision because retirement flips its
        used-sign to 0 (see the retrim rule below);
      - the FAIL-LOUD refusals: a held asset priced non-finite/<=0, or a book value that is
        non-finite/<=0, raises ValueError instead of being floored -- absence read as a value is
        the failure mode the 920 pack exists to remove.

    THE ONE CONVENTION THIS PASS MUST CHOOSE, stated explicitly (it is not inherited):
    *when* the carried shares are re-struck. Under target-weight/drift-inclusive the book is
    re-trimmed to the full target EVERY decision (the two modes differ only in whether that
    re-trim is CHARGED). A hold-shares book that also re-trimmed to target every decision would
    have a position path identical to target-weight's, i.e. it would not be a treatment arm at
    all -- the cycle-1 defect in a different costume. So the rule here is the one the mechanism
    on record actually describes (`experiments/830_corridor/SPEC.md:95-98`: "the corridor holds
    fixed SHARES while the 819 book rebalances to constant DOLLAR weight daily"):

        an asset is re-struck at decision k iff its USED SIGN changed, sign_ext[k] != sign_ext[k-1].

    Because `hysteresis_build` writes sign 0 for an absent asset, that single test covers all
    three trade reasons -- a trend flip, an entry (0 -> +-1), and an exit/delisting (+-1 -> 0,
    liquidated at last print). An asset present at both decisions with an unchanged sign is NOT
    traded, so its held weight drifts away from its target weight. DISCLOSURE (MINOR/HUNCH-C2B,
    run 55 cycle 3): that divergence has TWO sources, not one, and only the first was credited
    before this cycle.
      (i) PRICE drift on the HELD side -- a winning long's notional grows, a winning short's
          notional SHRINKS. That asymmetry is the mechanism the 830 anchor names, and it is why
          this book is expected to HURT in BEAR22 and to help in a trending-up window.
      (ii) TARGET drift on the un-traded side -- the canon target is equal-risk-weighted over the
          assets PRESENT at each decision, so a same-signed asset's TARGET weight also moves
          whenever the sizing denominator shifts because OTHER assets enter or leave the panel
          (or their trailing vol moves). The retrim rule is gated on this asset's own SIGN only,
          so it never re-strikes for a target that moved underneath it.
    Both sources widen held-vs-target and both are inside the measured numbers; neither is
    separated here. Weight drift also means the book's gross exposure Sum|w| is NOT pinned at 1
    -- it is measured and reported (`gross_abs_*` in the returned diagnostics), never assumed,
    and its consequence for the mode's Sharpe is measured by the gross-renormalised comparator
    in `score_cell` (BLOCKER-E2A).

    Returns (ret, turn, held_w, diag):
      ret     (n,)   per-decision SIMPLE return of the carried book over [t[j], t[j]+H), where
                     t = t_ext[1:]. Derived from the equity path, NOT from a weight formula:
                     V_{k+1} = cash + Sum_i shares_i*P_i. (The algebra collapses to
                     Sum_i w_held_i*(P_new/P_old - 1), which is the identity ST-checked in
                     test_919_grid.py::test_hold_shares_return_identity.)
      turn    (n,)   ACTUALLY TRADED weight at each decision, Sum_i |w_new_i - w_drifted_i|.
                     This is a NETTED, single-trade basis on the book's OWN drifted weights --
                     a different cost base from target-weight's and drift-inclusive's; see
                     RESULTS.md "cost bases are not comparable across accounting modes".
      held_w  (n,A)  the weights actually HELD over each [t[j], t[j]+H) -- what the borrow
                     charge must be denominated on for this mode (short notional actually held,
                     not the target's).
      diag    dict   equity path + gross-exposure drift statistics.
    """
    n_ext = t_ext.size
    A = price.shape[1]
    rows = np.concatenate([np.asarray(t_ext), [int(t_ext[-1]) + H]])
    shares = np.zeros(A)
    cash = 0.0
    Vs = np.empty(rows.size)
    turn = np.zeros(n_ext)
    held_w = np.zeros((n_ext, A))
    gross_abs = np.zeros(n_ext)
    for k, row in enumerate(rows):
        px = price[int(row)]
        held = shares != 0.0
        if k == 0:
            V = 1.0
            marked = np.zeros(A)
        else:
            if np.any(held & ~(np.isfinite(px) & (px > 0.0))):
                bad = int(np.argmax(held & ~(np.isfinite(px) & (px > 0.0))))
                raise ValueError(
                    f"hold_shares_walk: HELD asset column {bad} is priced non-finite or <= 0 at "
                    f"row {int(row)} -- refusing to mark a share book from an undefined price "
                    f"(same fail-loud contract as trader.backtest.true_bh_series).")
            marked = shares * np.where(held, px, 0.0)
            V = cash + float(np.sum(marked))
            if not np.isfinite(V) or V <= 0.0:
                raise ValueError(
                    f"hold_shares_walk: book equity {V!r} at row {int(row)} -- a share-carrying "
                    f"book that has been wiped out or gone undefined cannot be re-struck, and "
                    f"flooring it would silently absorb the failure into a plausible number.")
        Vs[k] = V
        if k == rows.size - 1:
            break
        drift_w = marked / V if k > 0 else np.zeros(A)
        tgt = pos_ext[k]
        # k==0 is the SEEDING decision (t[0]-H): strike the full target, exactly as the canon
        # book is at its full target there. It is never scored (ret/turn below drop it).
        retrim = np.ones(A, dtype=bool) if k == 0 else (sign_ext[k] != sign_ext[k - 1])
        w_new = np.where(retrim, tgt, drift_w)
        turn[k] = float(np.sum(np.abs(w_new - drift_w)))
        held_w[k] = w_new
        gross_abs[k] = float(np.sum(np.abs(w_new)))
        need_price = retrim & (w_new != 0.0)
        if np.any(need_price & ~(np.isfinite(px) & (px > 0.0))):
            bad = int(np.argmax(need_price & ~(np.isfinite(px) & (px > 0.0))))
            raise ValueError(
                f"hold_shares_walk: cannot strike shares for column {bad} at row {int(row)} -- "
                f"target weight is non-zero but the price is non-finite or <= 0.")
        safe_px = np.where(np.isfinite(px) & (px > 0.0), px, 1.0)
        shares = np.where(retrim, (w_new * V) / safe_px, shares)
        held = shares != 0.0
        cash = V - float(np.sum(shares * np.where(held, px, 0.0)))
    ret = Vs[2:] / Vs[1:-1] - 1.0
    ga = gross_abs[1:]
    diag = {
        "equity_final": float(Vs[-1]),
        "equity_min": float(np.min(Vs)),
        "gross_abs_mean": float(np.mean(ga)),
        "gross_abs_min": float(np.min(ga)),
        "gross_abs_max": float(np.max(ga)),
        # DISCLOSURE (run 55 cycle 2): StrategyConfig.gross_cap = 1.0 pins the canon book at
        # gross exposure 1. A share-carrying book that does NOT pay to re-trim drift CANNOT
        # honour that cap -- un-corrected drift is the mechanism. So this mode trades a book
        # whose leverage varies, which is a real advantage/disadvantage vs the other two modes
        # and is NOT a like-for-like comparison. Measured, never assumed.
        "frac_decisions_gross_above_1": float(np.mean(ga > 1.0)),
        "violates_canon_gross_cap_1": bool(np.max(ga) > 1.0),
    }
    return ret, turn[1:], held_w[1:], diag


# =============================================================================
# per-cell scoring
# =============================================================================

def _scatter_full(T, A, t_ext, values):
    """Sparse (T,A) array with `values` placed at rows t_ext, zero elsewhere -- glue for
    reusing bt.drift_retrim_turnover / bt.gross_simple_series (both only ever index pos[t] /
    pos[t-H], both exactly the rows populated here)."""
    full = np.zeros((T, A))
    full[t_ext] = values
    return full


def score_cell(logc, ts_d, syms, rv, pres, w, S_full, warm, r_lo, r_hi, H, theta, accounting,
               scfg=None, n_boot=None, seed=None, return_series=False):
    """Score ONE (H, theta, accounting) cell on [r_lo, r_hi). Returns None if the window is
    too thin for this H (same thinness contract as regime_series), else a dict of:
      n, gross (Sharpe), turn (mean per-decision, book-fraction units -- per DECISION, NOT per
      day; divide by H for per-day), net{8,12,21} Sharpe, net_916_{maker40,taker80} Sharpe,
      net21_borrow_{none,low,high} Sharpe, ci21_borrow_low (95% block-bootstrap), p21_borrow_low
      (one-sided normal), cost_basis (which turnover basis THIS mode charges -- not comparable
      across modes, MAJOR-F2), window, gross_log_on_target / gross_simple_on_target (the canon
      log-sum and the same-positions simple-sum, whose difference isolates the compounding
      convention), turnover_bases (all four bases measured on this cell), and
      decomposition {flip, renorm, entry_exit, drift, target_weight_turn_closes} -- the last a
      bool asserting flip+renorm+entry_exit == the target-weight charged turn (closes exactly;
      see RESULTS.md for why drift is reported separately).
    Hold-shares cells additionally carry hold_shares_diag (the exposure/equity walk) and
    renormalised_to_gross1 (the per-decision leverage-neutral rescaling of the REALISED return
    series -- see the BLOCKER-E2A comment below for what it is and is NOT).
    NOTE (closing pass): earlier revisions of this docstring named a returned key
    `charged_turnover_check`, which has never existed; the real key is
    decomposition["target_weight_turn_closes"].
    """
    if scfg is None:
        scfg = tconfig.STRATEGY
    if n_boot is None:
        n_boot = scfg.n_boot
    if seed is None:
        seed = scfg.seed
    T, A = logc.shape
    t = decision_grid_rows(T, r_lo, r_hi, H, warm)
    if t is None:
        return None
    t_ext = np.concatenate([[t[0] - H], t])
    if t_ext[0] < 0:
        return None
    pos_ext, sign_ext = hysteresis_build(rv, pres, w, S_full, t_ext, theta, L=L_TREND)
    pos = pos_ext[1:]
    pos_prev = pos_ext[:-1]
    fwd = logc[t + H] - logc[t]
    fwd = np.where(np.isfinite(fwd), fwd, 0.0)
    fwd_prev = logc[t] - logc[t - H]
    fwd_prev = np.where(np.isfinite(fwd_prev), fwd_prev, 0.0)

    gross_log = np.sum(pos * fwd, axis=1)
    turn_tw = np.sum(np.abs(pos - pos_prev), axis=1)          # target-weight charged turnover

    # decomposition (accounting-mode-independent; computed once per H,theta cell)
    pres_t = pres[t]
    pres_prev = pres[t - H]
    both = pres_t & pres_prev
    sign_t = sign_ext[1:]
    sign_prev = sign_ext[:-1]
    delta_abs = np.abs(pos - pos_prev)
    flip_mask = both & (sign_t != sign_prev)
    renorm_mask = both & (sign_t == sign_prev)
    entryexit_mask = ~both
    flip_amt = np.sum(np.where(flip_mask, delta_abs, 0.0), axis=1)
    renorm_amt = np.sum(np.where(renorm_mask, delta_abs, 0.0), axis=1)
    entryexit_amt = np.sum(np.where(entryexit_mask, delta_abs, 0.0), axis=1)
    drift_signed = pos_prev * (np.exp(fwd_prev) - 1.0)
    drift_amt = np.sum(np.abs(drift_signed), axis=1)          # drift_retrim_turnover's own term
    charged_check = np.allclose(flip_amt + renorm_amt + entryexit_amt, turn_tw, atol=1e-9)

    pos_full = _scatter_full(T, A, t_ext, pos_ext)
    tc_turn, dr_turn = bt.drift_retrim_turnover(pos_full, t, H, logc)
    assert np.allclose(tc_turn, turn_tw, atol=1e-9), "drift_retrim_turnover target-change term must match turn_tw"
    hs_diag = None
    hs_walk = None
    hs_renorm = None
    if accounting == "target-weight":
        gross = gross_log
        turn = turn_tw
        borrow_pos = pos
    elif accounting == "drift-inclusive":
        gross = gross_log
        turn = tc_turn + dr_turn
        borrow_pos = pos
    elif accounting == "hold-shares":
        # BLOCKER-C1 fix (run 55 cycle 2): a REAL share-carrying book. Everything below --
        # gross, turnover AND the borrow denominator -- is re-derived from the carried path,
        # not from the target-weight array the other two modes score.
        hs_ret, hs_turn, hs_held_w, hs_diag = hold_shares_walk(
            np.exp(logc), pos_ext, sign_ext, t_ext, H)
        gross = hs_ret
        turn = hs_turn
        borrow_pos = hs_held_w      # short notional ACTUALLY held, not the target's
        hs_walk = (hs_ret, hs_turn, hs_held_w)
        # BLOCKER-E2A (run 55 cycle 3), SCOPE CORRECTED in the closing pass. This is the
        # LEVERAGE-NEUTRAL RESCALING OF THE REALISED RETURN SERIES. A decision's SIMPLE portfolio
        # return is LINEAR in the weight vector,
        #     hs_ret[k] = Sum_i held_w[k,i] * (P_new_i/P_old_i - 1),
        # so dividing by that decision's own gross exposure G[k] = Sum_i |held_w[k,i]| gives
        # EXACTLY the return this book would have realised over THAT decision at gross 1. The
        # rescaling of the series is exact and it is NOT a uniform rescale, so it is NOT a Sharpe
        # no-op: G varies decision to decision (0.35..2.10 on PRIMARY; 0.2865..2.9185 across all
        # three windows) and its variation is CORRELATED with returns (a winning book's notional
        # grows before the next decision), which is the leverage/timing component being removed.
        #
        # WHAT IT IS NOT (closing pass, and this is load-bearing): it is NOT "the same book run
        # at gross 1". It rescales a REALISED return; it never renormalises the HOLDINGS, so the
        # drifted weights carried into decision k+1 are unchanged. A book actually constrained to
        # gross_cap = 1.0 would renormalise its holdings each decision, and its retrim rule would
        # then mix an UNSCALED target with a SCALED drift -- breaking proportionality, so its
        # k+1 position path DIFFERS from this proxy's. A cycle-3 lens built that constrained book
        # and got the OPPOSITE sign pattern (see leverage_renormalisation["construction_caveat"]).
        # This module ships the proxy only; the constrained book is NOT rebuilt here.
        #
        # G <= 0 can only happen when the book holds nothing, in which case hs_ret is 0 too, so
        # 0/0 is defined to 0 rather than floored.
        hs_gross_held = np.sum(np.abs(hs_held_w), axis=1)
        hs_renorm = np.where(hs_gross_held > 0.0,
                             hs_ret / np.where(hs_gross_held > 0.0, hs_gross_held, 1.0), 0.0)
    else:
        raise ValueError(f"unknown accounting mode {accounting!r}")

    period = H * tconfig.K
    borrow_costs = {sc: (np.zeros_like(gross) if sc == "none"
                         else bt.borrow_cost_series(borrow_pos, syms, np.arange(t.size), H, rate_scenario=sc))
                    for sc in BORROW_SCENARIOS}
    # borrow_cost_series indexes pos_np[t] internally in the shipped callers, but here
    # `borrow_pos` is ALREADY the decision-only (n,A) array -- pass a synthetic 0..n-1 index
    # array against it directly (not pos_full), which is exactly what borrow_cost_series needs
    # (it only ever does short_notional = max(-pos[t],0) @ rate_vec). The equivalence
    # borrow_cost_series(pos, syms, arange(n), H) == borrow_cost_series(pos_full, syms, t, H)
    # is asserted in test_919_grid.py::test_borrow_indexing_equivalence (MINOR-C3 fix: the
    # cycle-1 comment cited a G1 test that did not exist).

    out = {
        "H": H, "theta": theta, "accounting": accounting, "n": int(t.size),
        "gross": bt._sharpe(gross, period),
        "turn": float(np.mean(turn)),
        "turn_per_day": float(np.mean(turn)) / H,
    }
    for c in FEE_TIERS_BP:
        out[f"net{c}"] = bt._sharpe(bt.net_series(gross, turn, c), period)
    for label, c in FEE_TIERS_916.items():
        out[f"net_916_{label}"] = bt._sharpe(bt.net_series(gross, turn, c), period)
    for sc in BORROW_SCENARIOS:
        nb = bt.net_series_with_borrow(gross, turn, PRIMARY_FEE_BP, borrow_costs[sc])
        out[f"net{PRIMARY_FEE_BP}_borrow_{sc}"] = bt._sharpe(nb, period)

    # bootstrap CI + p-value at the PRIMARY gate quantity only (net21 + borrow-low) -- 48-cell
    # Bonferroni family, fee tiers/borrow scenarios REPORTED not searched (brief section 2).
    nh_primary = bt.net_series_with_borrow(gross, turn, PRIMARY_FEE_BP, borrow_costs[PRIMARY_BORROW])
    ci = bt._block_bootstrap_sharpe_ci(nh_primary, period, period, n_boot, seed)
    sd = np.std(nh_primary[np.isfinite(nh_primary)], ddof=0)
    fin = nh_primary[np.isfinite(nh_primary)]
    ts_stat = (np.sqrt(fin.size) * np.mean(fin) / sd) if sd > 1e-15 else 0.0
    out["ci21_borrow_low_lo"], out["ci21_borrow_low_hi"] = float(ci[0]), float(ci[1])
    out["p21_borrow_low"] = float(norm.sf(ts_stat))
    out["bonferroni_significant"] = bool(out["p21_borrow_low"] < BONFERRONI_ALPHA)

    out["decomposition"] = {
        "flip": float(np.mean(flip_amt)), "renorm": float(np.mean(renorm_amt)),
        "entry_exit": float(np.mean(entryexit_amt)), "drift": float(np.mean(drift_amt)),
        "target_weight_turn_closes": bool(charged_check),
    }
    out["cost_basis"] = COST_BASIS[accounting]
    # ATTRIBUTION (run 55 cycle 2). hold-shares differs from target-weight in TWO ways at once:
    # (i) a COMPOUNDING CONVENTION -- a carried book realizes a weighted SIMPLE return, the canon
    # gross is a weighted LOG return (the Jensen gap; TASK_920's separate leg 2, and the ONLY
    # thing cycle 1 actually measured); and (ii) the SHARE CARRY itself -- drifted weights. These
    # are reported separately at every cell so the two are never conflated again:
    #   gross                          = the mode's own gross Sharpe
    #   gross_log_on_target            = canon log-sum on the target positions
    #   gross_simple_on_target         = simple-sum on the SAME target positions (convention only)
    # so (simple_on_target - log_on_target) isolates the convention, and
    # (gross - simple_on_target) isolates the share carry, for the hold-shares mode.
    out["gross_log_on_target"] = bt._sharpe(gross_log, period)
    gross_simple_series = bt.gross_simple_series(logc, pos_full, t, H)
    out["gross_simple_on_target"] = bt._sharpe(gross_simple_series, period)
    if hs_renorm is not None:
        # BLOCKER-E2A: the leverage-stripped read of this cell. `gross` is an EXACT rescaling of
        # the realised return series (no cost model is involved at all) and it is the
        # load-bearing column. The net columns charge the LEVERED book's own turnover against the
        # de-levered return series -- an APPROXIMATION, because a book actually run at gross 1
        # would trade different notionals.
        #
        # DIRECTION OF THAT APPROXIMATION, CORRECTED IN THE CLOSING PASS. Cycle 3 asserted "mean
        # G < 1 at EVERY cell here, so the net columns FLATTER this arm". That universal is FALSE
        # at 3 of the 16 PRIMARY cells -- mean Sum|w| = 1.0142 (H=2,th=1.0), 1.0104 (H=3,th=1.0),
        # 1.0316 (H=5,th=1.0), all of which this module's own hold_shares_diag prints. Per-cell
        # recomputation (charging turn/G) shows the approximation PENALISES rather than flatters
        # in 4 cells -- those three plus H=1,th=1.0 at mean 0.9931, because Sharpe depends on G's
        # whole DISTRIBUTION, not on its mean. The magnitude is <= 0.002 Sharpe, so nothing flips
        # (max RENORM-TW +0.0309 -> +0.0303, min -0.0429 -> -0.0442), but the claim is stated as
        # "at 12 of 16 cells", never as a universal.
        #
        # DENOMINATION (D81(c), disclosed not fixed): this dict emits gross + net8/12/21 and NO
        # borrow column, while every band / uplift / significance statement in this pass is
        # denominated in net21_borrow_low. The de-levering conclusion is therefore argued in
        # gross and net21, NOT in the gate quantity. Adding the borrow column is rowed as
        # follow-up work; see leverage_renormalisation["gate_quantity_caveat"].
        out["renormalised_to_gross1"] = {
            "gross": bt._sharpe(hs_renorm, period),
            **{f"net{c}": bt._sharpe(bt.net_series(hs_renorm, turn, c), period)
               for c in FEE_TIERS_BP},
            "mean_gross_exposure_divided_out": float(np.mean(hs_gross_held)),
            "exact": "the GROSS column is an EXACT rescaling of the realised return series "
                     "(per-decision simple return is linear in the weights). It is NOT a book "
                     "re-run under a gross-1 constraint: the holdings carried into the next "
                     "decision are unrescaled. The net columns reuse the levered book's turnover "
                     "and are an APPROXIMATION (which flatters this arm at 12 of 16 cells and "
                     "penalises it at 4). No borrow column is emitted, so this arm is NOT stated "
                     "in the pass's own gate quantity net21_borrow_low.",
        }
    # MAJOR-F2 (run 55 cycle 2): the three accounting modes charge on NON-COMPARABLE cost
    # bases, and before this fix nothing in the artifact said so. Every basis is measured on
    # EVERY cell here so a reader can see the spread rather than infer that three columns
    # labelled "accounting" are denominated the same way. Meaning of each:
    #   target_change_only   Sum|d target|            -- drift assumed free (canon)
    #   upper_bound_unnetted Sum|d target| + Sum|drift| -- triangle-inequality UPPER BOUND
    #                                                     (drift_retrim_turnover's own docstring)
    #   netted_combined      Sum|d target - drift|    -- the single combined trade a desk sends,
    #                                                   still on the TARGET-WEIGHT book
    #   hold_shares_actual   Sum|w_new - w_drifted|   -- the actual trade on the HOLD-SHARES
    #                                                   book (a DIFFERENT book, not just a
    #                                                   different charge on the same one)
    out["turnover_bases"] = {
        "target_change_only": float(np.mean(turn_tw)),
        "upper_bound_unnetted": float(np.mean(tc_turn + dr_turn)),
        "netted_combined": float(np.mean(np.sum(np.abs((pos - pos_prev) - drift_signed), axis=1))),
        "hold_shares_actual": (float(np.mean(hs_walk[1])) if hs_walk is not None else None),
    }
    if hs_diag is not None:
        out["hold_shares_diag"] = hs_diag
    if return_series:
        # NOT JSON-serialized -- popped by every caller that writes the report; used only by
        # empirical_paired_diff_se (MAJOR-E1's assumption-free leg) and the G1 element-wise
        # identity test (MINOR-C2).
        out["_series"] = {"gross": gross, "turn": turn, "net_primary": nh_primary,
                          "t": t, "pos": pos, "pos_prev": pos_prev,
                          "held_w": (hs_walk[2] if hs_walk is not None else None),
                          # MAJOR-E2B: the two legs Correction 2's attribution splits, kept so
                          # each component can be given a PAIRED bootstrap CI (same decision
                          # rows -> uplift_ci's assumption-free branch).
                          "gross_log_on_target": gross_log,
                          "gross_simple_on_target": gross_simple_series,
                          "renorm": hs_renorm}
    return out


# =============================================================================
# BEAR22-only cell (for correction 2 -- needs its own regime window, not FULL)
# =============================================================================

def score_cell_bear22(logc, ts_d, syms, rv, pres, w, S_full, warm, H, theta, accounting,
                      scfg=None, return_series=False):
    r0, r1 = bt.regime_rows(np.asarray(ts_d))["BEAR22"]
    return score_cell(logc, ts_d, syms, rv, pres, w, S_full, warm, r0, r1, H, theta, accounting,
                      scfg=scfg, return_series=return_series)


# =============================================================================
# section 3: analytic turnover anchor -- CORRECTED formula + verification check
# =============================================================================

def analytic_turnover_per_day_corrected(H, L=L_TREND):
    """CORRECTION 1's re-derived formula: (2/pi)*arccos((L-H)/L)/H -- expected per-decision
    turnover Sum_i|Delta pos_i| divided by H, for jointly-Gaussian zero-mean trend statistics
    at correlation rho=(L-H)/L on an equal-weight gross-1 book. Reproduces the task file's
    quoted 0.165/0.117/0.0957/0.0746 EXACTLY at H=1/2/3/5 (the task's own formula,
    arccos(rho)/pi, does NOT -- see analytic_turnover_per_day_uncorrected)."""
    rho = (L - H) / L
    return (2.0 / np.pi) * np.arccos(rho) / H


def analytic_turnover_per_day_uncorrected(H, L=L_TREND):
    """The task file's AS-WRITTEN formula (lore/harvest_6b32_appendix/tactics_gaps.md:43):
    arccos(rho)/pi, rho=(L-H)/L -- treated (incorrectly) as already a per-day turnover rate.
    Kept ONLY as the G5 non-vacuity negative control: the verification check in
    test_919_grid.py must FAIL when this is substituted for the corrected model."""
    rho = (L - H) / L
    return np.arccos(rho) / np.pi


# =============================================================================
# section 7: hour-of-day mechanism (implemented, NOT run at panel scale -- see hour_of_day.py)
# =============================================================================
# kept in a separate module (hour_of_day.py) so this file stays import-light for the grid run.


# =============================================================================
# full-grid orchestration + report assembly
# =============================================================================

def run_grid(logc, ts_d, syms, rv, pres, w, S_full, warm, r_lo, r_hi, window_tag,
             keep_series=False):
    """Score all 48 (H, theta, accounting) cells on [r_lo, r_hi). Returns a list of cell dicts,
    each tagged with `window_tag` (a short string, e.g. 'primary'/'secondary'/'disjoint').
    With keep_series=True also returns {(H,theta,accounting): _series dict} for the uplift CIs
    and Correction 2's component CIs -- popped off the cell dicts so the report stays
    JSON-serializable."""
    cells, series = [], {}
    for H in H_GRID:
        for theta in THETA_GRID:
            for accounting in ACCOUNTING_GRID:
                c = score_cell(logc, ts_d, syms, rv, pres, w, S_full, warm, r_lo, r_hi,
                               H, theta, accounting, return_series=keep_series)
                if c is None:
                    c = {"H": H, "theta": theta, "accounting": accounting, "n": 0,
                         "window_too_thin": True}
                elif keep_series:
                    series[(H, theta, accounting)] = c.pop("_series")
                c["window"] = window_tag
                cells.append(c)
    return (cells, series) if keep_series else cells


def attach_uplift_cis(cells, series, scfg=None):
    """MAJOR-E1's sharpest leg: give EVERY cell a block-bootstrap CI on its UPLIFT over the
    frozen H=1/theta=0/target-weight rule -- the quantity the +-0.20 band is actually stated in.
    Mutates `cells` in place (adds an `uplift_ci` key) and returns a summary."""
    if scfg is None:
        scfg = tconfig.STRATEGY
    fkey = (1, 0.0, "target-weight")
    frozen_net = series[fkey]["net_primary"]
    ctr = dict.fromkeys(("clears", "excl0", "upper_below", "b_clears", "b_excl0",
                         "b_upper_below"), 0)
    clearing = []
    for c in cells:
        if c.get("n", 0) == 0:
            continue
        s = series[(c["H"], c["theta"], c["accounting"])]["net_primary"]
        u = uplift_ci(s, frozen_net, c["H"], 1, c["H"] * tconfig.K, 1 * tconfig.K,
                      scfg.n_boot, scfg.seed)
        c["uplift_ci"] = u
        if u is None:
            continue
        ctr["clears"] += u["ci_lower_bound_clears_band"]
        ctr["excl0"] += u["ci_excludes_zero"]
        ctr["upper_below"] += u["ci_upper_bound_below_band"]
        ctr["b_clears"] += u["bonf_ci_lower_bound_clears_band"]
        ctr["b_excl0"] += u["bonf_ci_excludes_zero"]
        ctr["b_upper_below"] += u["bonf_ci_upper_bound_below_band"]
        if u["ci_lower_bound_clears_band"]:
            clearing.append({"cell": (c["H"], c["theta"], c["accounting"]), **u})
    scored = [c for c in cells if c.get("uplift_ci")]
    n = len(scored)
    return {
        "n_cells": n,
        "raw_95pct": {
            "n_ci_lower_bound_clears_band": ctr["clears"],
            "n_ci_excludes_zero": ctr["excl0"],
            "n_ci_upper_bound_below_band": ctr["upper_below"],
            "n_undecided_ci_straddles_band": n - ctr["clears"] - ctr["upper_below"],
            # MINOR-E2C fix (run 55 cycle 3). The clearance test is "the LOWER bound of a
            # TWO-SIDED 95% CI lies above the band", whose type-I rate under the null is
            # alpha/2 = 0.025, not alpha = 0.05. The cycle-2 figure (0.05*n = 2.4) conflated it
            # with a one-sided alpha=0.05 test and so over-stated the chance baseline by 2x.
            # MINOR-7 (closing pass): alpha/2 * n is an UPPER BOUND, not the expectation. It
            # assumes all 48 cells sit exactly at the +0.20 boundary null; most sit far below it,
            # and the cells share position paths so the effective independent count is well under
            # 48. The true chance baseline is therefore BELOW 1.2 -- which makes "1 observed ~=
            # chance" EASIER to satisfy than warranted, i.e. the OPPOSITE of the conservative
            # reading cycle 3's note claimed. Immaterial to the disposition (Bonferroni 0/48 and
            # DISJOINT 0/48 carry the refusal), but it is labelled as a bound, not an expectation.
            "expected_false_positives_at_this_alpha": (ALPHA_FAMILYWISE / 2.0) * n,
            "expected_false_positives_is_an_upper_bound": True,
            "expected_false_positives_derivation":
                "alpha/2 * n_cells, an UPPER BOUND (<=): the per-cell test is a TWO-SIDED 95% "
                "interval's lower bound clearing the band, i.e. one tail, so its null rate is at "
                "most 0.025 per cell -- attained only for a cell sitting exactly at the boundary "
                "null. Most cells sit far below the band and the 48 cells share position paths, "
                "so the true chance baseline is BELOW this figure. Reading it as an equality "
                "makes 'chance' easier to invoke than the data warrant.",
        },
        "bonferroni_alpha_0.05_over_48": {
            "n_ci_lower_bound_clears_band": ctr["b_clears"],
            "n_ci_excludes_zero": ctr["b_excl0"],
            "n_ci_upper_bound_below_band": ctr["b_upper_below"],
            "n_undecided_ci_straddles_band": n - ctr["b_clears"] - ctr["b_upper_below"],
        },
        "cells_clearing_band_at_raw_95pct": clearing,
        "median_uplift_ci_half_width": float(np.median([c["uplift_ci"]["ci_half_width"]
                                                        for c in scored])),
        "note": "The band asks whether MAX uplift <= +0.20. A cell ANSWERS it only if its uplift "
                "CI lies wholly on one side of +0.20; a CI that STRADDLES +0.20 is undecided. "
                "The interval that may be read off a cell CHOSEN BY SEARCHING the grid is the "
                "Bonferroni one (alpha=0.05/48), not the raw 95%: any raw-95% clearance count "
                "at or below expected_false_positives_at_this_alpha is what chance alone "
                "produces over a 48-cell family and carries no verdict.",
    }


def find_frozen(cells):
    for c in cells:
        if c["H"] == 1 and c["theta"] == 0.0 and c["accounting"] == "target-weight":
            return c
    raise RuntimeError("frozen cell (H=1, theta=0, target-weight) missing from grid")


def find_best(cells, key=f"net{PRIMARY_FEE_BP}_borrow_{PRIMARY_BORROW}"):
    scored = [c for c in cells if c.get("n", 0) > 0 and np.isfinite(c.get(key, np.nan))]
    return max(scored, key=lambda c: c[key])


def cost_drag(cell, fee_bp=PRIMARY_FEE_BP):
    """Sharpe-space cost drag at `fee_bp`, no borrow: gross - net(fee_bp). Used for the
    section-5 Delta net = Delta gross - Delta cost_drag decomposition."""
    return cell["gross"] - cell[f"net{fee_bp}"]


def mde_analysis(cells, key=f"net{PRIMARY_FEE_BP}_borrow_{PRIMARY_BORROW}", band=0.20,
                 alpha_raw=0.05, alpha_bonf=BONFERRONI_ALPHA, power=0.80):
    """MAJOR-E1 (run 55 cycle 2): the design's MINIMUM DETECTABLE EFFECT, re-derived here from
    this pass's own arrays. This is the single most important number in the artifact, because
    it decides whether the task's pre-registered +-0.20 band is answerable AT ALL by this
    design -- a question that is PRIOR TO and INDEPENDENT OF the hold-shares fix.

    SE of one cell's annualized Sharpe, TWO independent derivations that must agree:
      (A) EMPIRICAL, from each cell's own 95% circular-block-bootstrap CI on the primary gate
          quantity: SE = (hi - lo) / (2 * 1.96).
      (B) CLOSED FORM (Lo 2002, iid case): SE(SR_period) = sqrt((1 + SR_period^2 / 2) / n), and
          the annualizer here is a deterministic constant, SR_ann = SR_period * sqrt(365/H)
          (bt._sharpe: sqrt(525600 / (H*1440))), so
              SE(SR_ann) = sqrt(365/H) * sqrt((1 + (SR_ann / sqrt(365/H))^2 / 2) / n).
          The H-invariance of the result is not luck: this grid holds n*H ~ 1704 constant, which
          exactly cancels the sqrt(365/H) factor.

    MDE for a ONE-SIDED test at `power`:  MDE = SE_diff * (z_{1-alpha} + z_{power}).
    SE_diff is the SE of the UPLIFT (candidate minus frozen), and it depends on how correlated
    the two series are. Rather than pick one, every assumption is reported:
      rho = 1 sentinel ("frozen treated as a noiseless reference", SE_diff = SE) -- the
        FRIENDLIEST case, and even it is reported so that the verdict cannot be accused of
        resting on a pessimistic choice;
      rho = 0 (independent, SE_diff = sqrt(2)*SE) -- the most conservative;
      rho = 0.7 / 0.9 (SE_diff = sqrt(2(1-rho))*SE) -- plausible middle ground.
    PLUS an EMPIRICAL paired-difference SE for the H=1 cells, where candidate and frozen share
    the same 1704 decision rows and the difference series can be formed directly -- no
    correlation assumption at all. (Cross-H cells cannot be paired: different n.)
    """
    scored = [c for c in cells if c.get("n", 0) > 0 and np.isfinite(c.get(key, np.nan))]
    se_emp, se_cf, rows = [], [], []
    for c in scored:
        hw = (c["ci21_borrow_low_hi"] - c["ci21_borrow_low_lo"]) / 2.0
        se_a = hw / norm.ppf(0.975)
        ann = np.sqrt(365.0 / c["H"])
        sr_p = c[key] / ann
        se_b = ann * np.sqrt((1.0 + sr_p ** 2 / 2.0) / c["n"])
        se_emp.append(se_a)
        se_cf.append(se_b)
        rows.append({"H": c["H"], "theta": c["theta"], "accounting": c["accounting"],
                     "n": c["n"], "se_bootstrap": float(se_a), "se_closed_form": float(se_b)})
    se_emp = np.array(se_emp)
    se_cf = np.array(se_cf)
    se = float(np.median(np.concatenate([se_emp, se_cf])))

    z_pow = norm.ppf(power)
    out = {
        "primary_gate_quantity": key, "band": band, "power": power,
        "se_single_cell_bootstrap": {"median": float(np.median(se_emp)),
                                     "min": float(se_emp.min()), "max": float(se_emp.max())},
        "se_single_cell_closed_form": {"median": float(np.median(se_cf)),
                                       "min": float(se_cf.min()), "max": float(se_cf.max())},
        "se_used": se,
        "two_derivations_agree_within": float(abs(np.median(se_emp) - np.median(se_cf))),
        "per_cell_se": rows,
        "mde": {},
    }
    for label, factor in (("frozen treated as a NOISELESS reference (SE_diff = SE)", 1.0),
                          ("rho=0 (series independent; SE_diff = sqrt(2)*SE)", float(np.sqrt(2.0))),
                          ("rho=0.7 (SE_diff = sqrt(2(1-rho))*SE)", float(np.sqrt(2 * (1 - 0.7)))),
                          ("rho=0.9 (SE_diff = sqrt(2(1-rho))*SE)", float(np.sqrt(2 * (1 - 0.9))))):
        se_d = se * factor
        m_raw = se_d * (norm.ppf(1 - alpha_raw) + z_pow)
        m_bonf = se_d * (norm.ppf(1 - alpha_bonf) + z_pow)
        out["mde"][label] = {
            "se_diff": float(se_d),
            "mde_raw_alpha_0.05": float(m_raw), "mde_raw_over_band": float(m_raw / band),
            "mde_bonferroni": float(m_bonf), "mde_bonferroni_over_band": float(m_bonf / band),
        }
    return out


def measured_pairwise_correlation(h1_series, se_single, band=0.20, alpha_raw=0.05,
                                  alpha_bonf=BONFERRONI_ALPHA, power=0.80):
    """The assumption-free half of MAJOR-E1. The MDE table above has ONE free parameter, the
    correlation rho between a candidate cell's per-decision net series and the frozen cell's,
    which sets SE_diff = sqrt(2(1-rho))*SE. For the H=1 cells the two series live on the SAME
    1704 decision rows, so rho can be MEASURED rather than assumed, and a per-cell MDE follows.

    A DEGENERACY that must be flagged rather than exploited: at theta=0 the `drift-inclusive`
    cell differs from `target-weight` ONLY by a deterministic cost constant, so rho ~ 1.0 and
    its MDE collapses to ~0. That is not power -- it is a pair with nothing to detect. The
    headline below is therefore the most favourable NON-DEGENERATE pair (rho < 0.999)."""
    frozen = h1_series["frozen"]
    fin = np.isfinite(frozen)
    z_pow = norm.ppf(power)
    recs = []
    for name, net in h1_series.items():
        if name == "frozen":
            continue
        m = fin & np.isfinite(net)
        if m.sum() < 3:
            continue
        rho = float(np.corrcoef(net[m], frozen[m])[0, 1])
        se_d = se_single * float(np.sqrt(max(2.0 * (1.0 - rho), 0.0)))
        recs.append({"cell": name, "rho": rho, "se_diff": float(se_d),
                     "mde_raw": float(se_d * (norm.ppf(1 - alpha_raw) + z_pow)),
                     "mde_bonferroni": float(se_d * (norm.ppf(1 - alpha_bonf) + z_pow)),
                     "degenerate_same_path": bool(rho > 0.999)})
    if not recs:
        return {"n_pairs": 0}
    rhos = np.array([r["rho"] for r in recs])
    nd = [r for r in recs if not r["degenerate_same_path"]]
    best = min(nd, key=lambda r: r["mde_bonferroni"]) if nd else None
    return {"n_pairs": len(recs), "rho_min": float(rhos.min()),
            "rho_median": float(np.median(rhos)), "rho_max": float(rhos.max()),
            "per_cell": recs,
            "most_favourable_non_degenerate": (
                None if best is None else
                {**best, "mde_raw_over_band": best["mde_raw"] / band,
                 "mde_bonferroni_over_band": best["mde_bonferroni"] / band}),
            "note": "rho MEASURED between each H=1 candidate's per-decision net21_borrow_low "
                    "series and the H=1 frozen cell's, on the same 1704 rows. Cross-H cells "
                    "cannot be paired (different n) and are less correlated in every measurable "
                    "case here, hence have a LARGER MDE -- so the non-degenerate best below is a "
                    "LOWER BOUND on the MDE of the cells the grid actually nominates."}


def uplift_ci(cand_net, frozen_net, H_cand, H_frozen, period_cand, period_frozen,
              n_boot, seed, band=0.20):
    """Block-bootstrap CI on the UPLIFT ITSELF (candidate annualized Sharpe minus the frozen
    cell's), which is the quantity the pre-registered +-0.20 band is stated in. This is the
    sharpest instrument in the pass and it needs NO correlation assumption where the two series
    can be paired.

      - SAME H (candidate and frozen share the same decision rows): ONE set of circular block
        indices is drawn per bootstrap replicate and applied to BOTH series, so the resample
        preserves their dependence exactly. PAIRED, assumption-free.
      - DIFFERENT H (e.g. an H=2 candidate against the H=1 frozen rule): the rows cannot be
        paired at all, so the two series are resampled INDEPENDENTLY. That over-states the
        uplift's variance whenever the underlying series are positively correlated, i.e. it is
        CONSERVATIVE in the direction of refusing a claim -- stated, not hidden.

    Block length and the annualizer are taken from bt._block_bootstrap_sharpe_ci's own
    convention so the CI is commensurable with the per-cell CIs already reported.
    Returns dict with lo/hi, whether the CI excludes 0, and whether its LOWER bound clears the
    band (the direct test the band's wording asks for).
    """
    fin_a = np.isfinite(cand_net)
    fin_b = np.isfinite(frozen_net)
    a = cand_net[fin_a]
    b = frozen_net[fin_b]
    if a.size < 2 or b.size < 2:
        return None
    # MINOR-E2D fix (run 55 cycle 3): the PAIRED branch applies ONE set of block indices to both
    # filtered series, which is only meaningful if the two series occupy the same positions after
    # filtering. Cycle 2 keyed pairing on POST-filter sizes alone, so two series that dropped
    # different positions but the same COUNT of them would have been silently mis-paired. Traced
    # as latent, not live (fwd is NaN-filled to 0 and hold_shares_walk raises rather than emitting
    # NaN, so no cell here drops a row), but the "paired, assumption-free" claim rests on this
    # invariant, so it is now ASSERTED rather than assumed -- fail loud, never silently mis-align.
    paired = (H_cand == H_frozen) and (cand_net.size == frozen_net.size)
    if paired and not np.array_equal(fin_a, fin_b):
        raise ValueError(
            "uplift_ci: refusing to draw a PAIRED bootstrap -- the candidate and frozen series "
            f"drop non-finite entries at DIFFERENT positions ({int(fin_a.sum())} vs "
            f"{int(fin_b.sum())} finite, masks differ), so one series' block indices would be "
            "applied to a misaligned other. The 'paired, assumption-free' claim requires "
            "positional alignment.")
    paired = paired and (a.size == b.size)
    rng = np.random.RandomState(seed)
    fa = np.sqrt(bt.MINUTES_PER_YEAR / float(period_cand))
    fb = np.sqrt(bt.MINUTES_PER_YEAR / float(period_frozen))
    # block length 5 == bt._block_bootstrap_sharpe_ci's max(5, ceil(2*max_R/R)) at max_R == R,
    # which is how every per-cell CI in this pass is drawn -- kept identical on purpose.
    bl = 5
    nb_a = int(np.ceil(a.size / bl))
    nb_b = int(np.ceil(b.size / bl))
    ar = np.arange(bl)
    out = np.empty(n_boot)
    for k in range(n_boot):
        s_a = rng.randint(0, a.size, size=nb_a)
        idx_a = (s_a[:, None] + ar[None, :]).ravel()[:a.size] % a.size
        sa = a[idx_a]
        if paired:
            sb = b[idx_a]
        else:
            s_b = rng.randint(0, b.size, size=nb_b)
            idx_b = (s_b[:, None] + ar[None, :]).ravel()[:b.size] % b.size
            sb = b[idx_b]
        da, db = np.std(sa, ddof=0), np.std(sb, ddof=0)
        out[k] = ((np.mean(sa) / da * fa) - (np.mean(sb) / db * fb)
                  if da > 1e-15 and db > 1e-15 else np.nan)
    out = out[np.isfinite(out)]
    if out.size < 2:
        return None
    lo, hi = np.percentile(out, [2.5, 97.5])
    point = (np.mean(a) / np.std(a, ddof=0) * fa) - (np.mean(b) / np.std(b, ddof=0) * fb)
    # FAMILY-WISE interval. The pre-registered family is 48 cells, so the interval that may be
    # read off ANY cell after searching the grid is the Bonferroni one, alpha = 0.05/48. Its
    # percentile (0.052 / 99.948) is not resolvable from n_boot=2000 draws -- ~1 draw in each
    # tail -- so it is formed by a NORMAL approximation on the bootstrap SD instead of a
    # percentile, and labelled as such rather than quoted as an empirical quantile.
    sd_b = float(np.std(out, ddof=1))
    z_b = norm.ppf(1.0 - BONFERRONI_ALPHA / 2.0)
    blo, bhi = float(point - z_b * sd_b), float(point + z_b * sd_b)
    return {"uplift": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "paired": bool(paired),
            "ci_excludes_zero": bool(lo > 0.0),
            "ci_lower_bound_clears_band": bool(lo > band),
            "ci_upper_bound_below_band": bool(hi < band),
            "ci_half_width": float((hi - lo) / 2.0),
            "bootstrap_sd": sd_b,
            "ci_bonferroni_lo": blo, "ci_bonferroni_hi": bhi,
            "ci_bonferroni_normal_approx": True,
            "bonf_ci_excludes_zero": bool(blo > 0.0),
            "bonf_ci_lower_bound_clears_band": bool(blo > band),
            "bonf_ci_upper_bound_below_band": bool(bhi < band)}


def attribution_component_ci(series_dict, H, scfg=None):
    """MAJOR-E2B fix (run 55 cycle 3). Correction 2 splits a window's hold-minus-target gross gap
    into a COMPOUNDING-CONVENTION component and a SHARE-CARRY component. Cycle 2 shipped both as
    bare point estimates while the SAME pass refused the +-0.20 band question for want of power --
    an internal double standard, since the share-carry component (-0.0328 on BEAR22) is ~3.5% of
    the single-cell Sharpe SE the pass itself derives at that n.

    Both components are Sharpe DIFFERENCES between two series on the SAME decision rows, so
    `uplift_ci`'s PAIRED, assumption-free branch applies exactly. A component whose 95% CI
    contains 0 is not resolved by this design and is reported UNRESOLVED. This does NOT touch the
    DIRECTION of the BEAR22 gap, which is supported separately (830_corridor/SPEC.md:95-98, LEDGER
    830/E0, and the gross-renormalisation counter-check) -- it is the SPLIT that needs error bars.
    """
    if scfg is None:
        scfg = tconfig.STRATEGY
    period = H * tconfig.K
    log_s = series_dict["gross_log_on_target"]
    simple_s = series_dict["gross_simple_on_target"]
    carried_s = series_dict["gross"]
    out = {}
    for name, cand, base in (("delta_from_compounding_convention", simple_s, log_s),
                             ("delta_from_share_carry", carried_s, simple_s)):
        u = uplift_ci(cand, base, H, H, period, period, scfg.n_boot, scfg.seed)
        if u is None:
            out[name] = None
            continue
        # NOTE (closing pass, naming only -- the semantics were already right). `resolved` is the
        # TWO-TAILED test "the 95% CI lies wholly on one side of 0": uplift_ci's own
        # `ci_excludes_zero` is ONE-TAILED (ci_lo > 0), so the `or ci_hi < 0` picks up the
        # negative tail it misses. The key below reuses uplift_ci's one-tailed NAME for this
        # two-tailed quantity; `ci_excludes_zero_two_tailed` is the same value under the name it
        # should have had. The original key is retained so no downstream reader breaks.
        resolved = bool(u["ci_excludes_zero"] or u["ci_hi"] < 0.0)
        out[name] = {
            "point": u["uplift"], "ci_lo": u["ci_lo"], "ci_hi": u["ci_hi"],
            "bootstrap_sd": u["bootstrap_sd"], "paired": u["paired"],
            "ci_excludes_zero": resolved,
            "ci_excludes_zero_two_tailed": resolved,
            "status": "RESOLVED (95% CI excludes 0)" if resolved
                      else "UNRESOLVED -- the 95% CI contains 0; this component is not "
                           "distinguishable from noise at this design's power",
        }
    return out


def band_verdict(cells, frozen, key=f"net{PRIMARY_FEE_BP}_borrow_{PRIMARY_BORROW}"):
    """CORRECTION 3: the pre-registered band ('max uplift over the frozen rule <= +0.20
    net@21 => cost-only, folds in') is not evaluable without a gross/cost split -- the H axis
    is not cost-only (H also changes signal exposure / holding period). Reports BOTH readings:
      - max Delta net (the band's own literal wording, unqualified 'uplift')
      - max Delta cost_drag REDUCTION (the actually cost-only component: Delta cost_drag,
        where cost_drag = gross - net@21 no-borrow; a NEGATIVE delta is a cost SAVING)
    Does not silently pick one; the caller (RESULTS.md) states the conflict explicitly.
    """
    scored = [c for c in cells if c.get("n", 0) > 0 and np.isfinite(c.get(key, np.nan))]
    best_net_cell = max(scored, key=lambda c: c[key] - frozen[key])
    max_delta_net = best_net_cell[key] - frozen[key]

    # MAJOR-F2 fix (run 55 cycle 2): cost_drag = gross - net(fee) is denominated in each mode's
    # OWN turnover basis, and the three bases are NOT the same quantity (see COST_BASIS) -- and
    # since the hold-shares fix, not even the same BOOK. A cross-mode Delta cost_drag therefore
    # subtracts two numbers that were never measured the same way. That comparison is FORBIDDEN
    # here and the cost-only reading is computed WITHIN each accounting mode, against that
    # mode's own H=1/theta=0 cell.
    per_mode = {}
    for mode in ACCOUNTING_GRID:
        mode_cells = [c for c in scored if c["accounting"] == mode]
        base = [c for c in mode_cells if c["H"] == 1 and c["theta"] == 0.0]
        if not mode_cells or not base:
            continue
        base = base[0]
        base_drag = cost_drag(base)
        best = min(mode_cells, key=lambda c: cost_drag(c) - base_drag)
        saving = base_drag - cost_drag(best)     # positive = a saving
        per_mode[mode] = {
            "within_mode_reference_cell": (base["H"], base["theta"], mode),
            "cost_basis": COST_BASIS[mode],
            "max_cost_drag_saving": float(saving),
            "cell": (best["H"], best["theta"], best["accounting"]),
            "exceeds_band": bool(saving > 0.20),
            "delta_net_at_this_cell_vs_own_mode_reference": float(best[key] - base[key]),
        }

    return {
        "band_threshold": 0.20,
        "reading_max_delta_net": {
            "value": float(max_delta_net),
            "cell": (best_net_cell["H"], best_net_cell["theta"], best_net_cell["accounting"]),
            "point_estimate_exceeds_band": bool(max_delta_net > 0.20),
            "POWER_CAVEAT": "a POINT ESTIMATE only -- see mde_primary: the MDE of this design is "
                            "5-10x the +-0.20 band, so 'exceeds' here is not a detection and "
                            "carries no verdict. The band question is REFUSED FOR WANT OF POWER.",
            "delta_gross": float(best_net_cell["gross"] - frozen["gross"]),
            "delta_cost_drag_CROSS_MODE_DO_NOT_USE": (
                float(cost_drag(best_net_cell) - cost_drag(frozen))
                if best_net_cell["accounting"] == frozen["accounting"] else None),
        },
        "reading_max_delta_cost_drag_saving_WITHIN_MODE": per_mode,
        "cross_mode_cost_drag_comparison": "FORBIDDEN -- the three accounting modes charge on "
            "non-comparable cost bases (COST_BASIS), and since the run-55 hold-shares fix the "
            "hold-shares mode is a different POSITION PATH as well. Only within-mode Delta "
            "cost_drag is meaningful; the headline cost-only reading is the target-weight row, "
            "which is the mode the pre-registered frozen rule is stated on.",
        "which_reading_the_bands_wording_targets": "reading_max_delta_net (literal, unqualified "
            "'uplift'); but the H axis is not cost-only, so that reading conflates a signal/"
            "holding-period effect with an execution saving -- see RESULTS.md section on "
            "CORRECTION 3. Neither reading is DECIDABLE at this design's power (mde_primary).",
    }


def significance_summary(cells, alpha_bonf=BONFERRONI_ALPHA):
    scored = [c for c in cells if c.get("n", 0) > 0]
    n_bonf = sum(c.get("bonferroni_significant", False) for c in scored)
    n_raw05 = sum(c.get("p21_borrow_low", 1.0) < 0.05 for c in scored)
    ci_lo_positive = sum(c.get("ci21_borrow_low_lo", -1.0) > 0 for c in scored)
    return {"n_cells_scored": len(scored), "n_bonferroni_significant": int(n_bonf),
            "n_raw_p_lt_05": int(n_raw05), "n_ci_lower_bound_positive": int(ci_lo_positive),
            "alpha_bonferroni": alpha_bonf}


def build_report():
    """Assemble the full 919A result: fixture identity, panel info, both windows' 48-cell
    grids, the three corrections' own measurements, the band verdict (both readings), the
    significance summary, and the decomposition closure check. Returns a JSON-serializable
    dict."""
    fixture_sha = print_fixture_sha()
    logc, ts_d, syms, info = load_pit_panel()
    rv, pres, w, S_full, warm = build_base_arrays(logc)

    r_lo_p, r_hi_p = window_bounds(ts_d, PRIMARY_WINDOW["start"], PRIMARY_WINDOW["end"])
    r_lo_s, r_hi_s = window_bounds(ts_d, SECONDARY_WINDOW["start"], SECONDARY_WINDOW["end"])
    r_lo_d, r_hi_d = window_bounds(ts_d, DISJOINT_WINDOW["start"], DISJOINT_WINDOW["end"])

    cells_primary, series_p = run_grid(logc, ts_d, syms, rv, pres, w, S_full, warm,
                                       r_lo_p, r_hi_p, "primary", keep_series=True)
    cells_secondary, series_s = run_grid(logc, ts_d, syms, rv, pres, w, S_full, warm,
                                         r_lo_s, r_hi_s, "secondary", keep_series=True)
    cells_disjoint, series_d = run_grid(logc, ts_d, syms, rv, pres, w, S_full, warm,
                                        r_lo_d, r_hi_d, "disjoint", keep_series=True)
    uplift_p = attach_uplift_cis(cells_primary, series_p)
    uplift_s = attach_uplift_cis(cells_secondary, series_s)
    uplift_d = attach_uplift_cis(cells_disjoint, series_d)

    frozen_p = find_frozen(cells_primary)
    best_p = find_best(cells_primary)
    frozen_s = find_frozen(cells_secondary)
    frozen_d = find_frozen(cells_disjoint)
    key = f"net{PRIMARY_FEE_BP}_borrow_{PRIMARY_BORROW}"

    def _same_cell(cells, ref):
        for c in cells:
            if (c["H"], c["theta"], c["accounting"]) == (ref["H"], ref["theta"], ref["accounting"]):
                return c
        return None

    best_s_same_cell = _same_cell(cells_secondary, best_p)
    best_d_same_cell = _same_cell(cells_disjoint, best_p)

    # correction 1: analytic anchor. MINOR-E3/O2 fix (run 55 cycle 2): the model
    # (2/pi)arccos((L-H)/L) models ONLY sign flips, so the correct comparand is the FLIP term of
    # the decomposition, not TOTAL turnover. Both are now reported; flip is the load-bearing one.
    corr1 = []
    for H in H_GRID:
        c = [x for x in cells_primary if x["H"] == H and x["theta"] == 0.0
             and x["accounting"] == "target-weight"][0]
        model = analytic_turnover_per_day_corrected(H)
        uncorr = analytic_turnover_per_day_uncorrected(H)
        flip_per_day = c["decomposition"]["flip"] / H
        corr1.append({"H": H, "n": c["n"], "turn_per_day_measured": c["turn_per_day"],
                      "flip_per_day_measured": float(flip_per_day),
                      "model_corrected": model,
                      "ratio_vs_total_turn": c["turn_per_day"] / model,
                      "ratio_vs_flip_only": float(flip_per_day / model),
                      "model_uncorrected": uncorr,
                      "uncorrected_ratio": c["turn_per_day"] / uncorr})

    # correction 2 (RE-DERIVED under a REAL share-carrying path -- the cycle-1 CONFIRMED status
    # was retracted because that pass never carried a share). BEAR22 vs the full primary window,
    # so the DIRECTION claim ("hold-shares HURTS in BEAR22") is tested against a contrast, not
    # asserted from one number.
    bear_tw = score_cell_bear22(logc, ts_d, syms, rv, pres, w, S_full, warm, 1, 0.0, "target-weight")
    bear_hs = score_cell_bear22(logc, ts_d, syms, rv, pres, w, S_full, warm, 1, 0.0, "hold-shares",
                                return_series=True)
    bear_hs_series = bear_hs.pop("_series")
    full_tw = frozen_p
    full_hs = [c for c in cells_primary if c["H"] == 1 and c["theta"] == 0.0
               and c["accounting"] == "hold-shares"][0]
    # MAJOR-E2B: paired CIs on each attribution component, on both windows.
    attribution_cis = {
        "bear22": attribution_component_ci(bear_hs_series, 1),
        "primary_full": attribution_component_ci(series_p[(1, 0.0, "hold-shares")], 1),
    }

    # BLOCKER-E2A: the gross-renormalisation comparator, assembled across the whole (H, theta)
    # face of the hold-shares mode plus the BEAR22 counter-check.
    def _renorm_rows(cells):
        rows = []
        for H in H_GRID:
            for th in THETA_GRID:
                hs = [c for c in cells if c["H"] == H and c["theta"] == th
                      and c["accounting"] == "hold-shares"]
                tw = [c for c in cells if c["H"] == H and c["theta"] == th
                      and c["accounting"] == "target-weight"]
                if not hs or not tw or hs[0].get("n", 0) == 0 or "renormalised_to_gross1" not in hs[0]:
                    continue
                hs, tw = hs[0], tw[0]
                rn = hs["renormalised_to_gross1"]
                rows.append({
                    "H": H, "theta": th, "n": hs["n"],
                    "hs_gross_raw": hs["gross"], "hs_gross_renorm": rn["gross"],
                    "tw_gross": tw["gross"],
                    "raw_minus_tw_gross": hs["gross"] - tw["gross"],
                    "renorm_minus_tw_gross": rn["gross"] - tw["gross"],
                    "hs_net21_raw": hs["net21"], "hs_net21_renorm": rn["net21"],
                    "tw_net21": tw["net21"],
                    "raw_minus_tw_net21": hs["net21"] - tw["net21"],
                    "renorm_minus_tw_net21": rn["net21"] - tw["net21"],
                    "mean_gross_exposure": rn["mean_gross_exposure_divided_out"],
                })
        return rows

    renorm_rows = _renorm_rows(cells_primary)
    bear_rn = bear_hs.get("renormalised_to_gross1", {})
    d_raw = [r["raw_minus_tw_gross"] for r in renorm_rows]
    d_rn = [r["renorm_minus_tw_gross"] for r in renorm_rows]
    leverage_renorm = {
        "what_it_is": "a PER-DECISION LEVERAGE-NEUTRAL RESCALING OF THE REALISED RETURN SERIES "
                      "of the hold-shares book. Per-decision simple return is LINEAR in the "
                      "weight vector, so dividing a decision's return by that decision's own "
                      "Sum|w| is an EXACT rescaling of that return -- and not a uniform one, "
                      "because Sum|w| varies (0.35..2.10 on PRIMARY, 0.2865..2.9185 across all "
                      "three windows) and its variation is correlated with returns.",
        "what_it_is_NOT": "it is NOT 'the same book run at gross 1'. It never renormalises the "
                          "HOLDINGS, so the drifted weights carried into the next decision are "
                          "unchanged. A book actually constrained to gross_cap=1.0 renormalises "
                          "its holdings each decision, after which the retrim rule mixes an "
                          "UNSCALED target with a SCALED drift -- proportionality breaks and the "
                          "position path diverges. See construction_caveat.",
        "why_it_exists": "cycle 2 recorded 'part of the hold-shares Sharpe advantage is a "
                         "leverage/timing effect, not an accounting one, and this pass did not "
                         "separate the two ... isolating them needs a new arm'. Isolating the "
                         "leverage component needs no new arm (hold_shares_walk already returns "
                         "held_w). What the measurement then delivers is a bound, not a verdict: "
                         "see verdict / construction_caveat.",
        "primary_window_rows": renorm_rows,
        "summary_primary": {
            "n_cells": len(renorm_rows),
            "raw_advantage_gross_min": float(np.min(d_raw)) if d_raw else None,
            "raw_advantage_gross_max": float(np.max(d_raw)) if d_raw else None,
            "raw_advantage_positive_in_all_cells": bool(all(x > 0 for x in d_raw)),
            "renorm_advantage_gross_min": float(np.min(d_rn)) if d_rn else None,
            "renorm_advantage_gross_max": float(np.max(d_rn)) if d_rn else None,
            "renorm_advantage_consistent_sign": bool(
                all(x > 0 for x in d_rn) or all(x < 0 for x in d_rn)),
            "n_renorm_advantage_positive": int(sum(1 for x in d_rn if x > 0)),
            "verdict": "UNRESOLVED. De-levering removes ~85% of the full-window hold-shares "
                       "gross advantage over target-weight and leaves a residue an order of "
                       "magnitude inside this design's SE. That is what is measured. It is NOT "
                       "a demonstration that the advantage IS entirely gross-exposure scaling: "
                       "the residue's sign is construction-dependent (see construction_caveat) "
                       "and NEITHER construction's paired CI clears zero. The raw advantage "
                       "itself does not clear zero either (+0.2794, 95% CI [-0.088, +0.638] at "
                       "the frozen cell), so no promotion is supported under any construction.",
            "renorm_advantage_no_consistent_sign_is_construction_specific": True,
        },
        "construction_caveat": {
            "what_this_module_ships": "the proxy ret/G -- a rescaling of the REALISED return "
                                      "series, holdings unchanged.",
            "the_alternative": "a book genuinely constrained to gross_cap=1.0, which renormalises "
                               "its HOLDINGS each decision (w /= Sum|w| after the retrim mix) and "
                               "therefore carries different weights into the next decision.",
            "transcribed_from": "a cycle-3 estimator lens BUILT that constrained book on the same "
                                "walk and validated it (Sum|w| = 1 +/- 4e-16, r = Sum w_i r_i to "
                                "4.4e-16, identical asset support). This closing pass TRANSCRIBES "
                                "that lens's figures; it did NOT rebuild the book.",
            "proxy_advantage_over_target_weight_gross_primary": "-0.0346 .. +0.0320, "
                                                                "6 of 16 cells positive "
                                                                "(no consistent sign)",
            "constrained_book_advantage_over_target_weight_gross_primary":
                "+0.0039 .. +0.0571, 16 of 16 cells positive (consistent POSITIVE sign)",
            "paired_95pct_cis_at_the_frozen_cell": {
                "raw_advantage": "+0.2794 [-0.088, +0.638]",
                "proxy_renormalised": "-0.0326 [-0.282, +0.214]",
                "constrained_gross1_book": "+0.0039 [-0.186, +0.204]",
            },
            "conclusion": "the two constructions give OPPOSITE sign patterns and NEITHER clears "
                          "zero. The leverage question is UNRESOLVED at this design's power, and "
                          "'no consistent sign' is a property of the proxy, not of the mode. "
                          "Rebuilding the constrained book inside this harness is follow-up work.",
        },
        "gate_quantity_caveat": {
            "problem": "renormalised_to_gross1 emits gross + net8/12/21 and NO borrow, while "
                       "every band / uplift / significance / candidate statement in this pass is "
                       "denominated in net21_borrow_low. The de-levering conclusion is argued in "
                       "gross and net21, NOT in the gate quantity (D81(c)).",
            "transcribed_arm": "a cycle-3 flow verifier re-denominated it properly -- weights "
                               "held_w/G, turnover turn/G, borrow charged on the DE-LEVERED short "
                               "notional -- and got renorm minus target-weight of -0.0380 .. "
                               "+0.0497 with 9 of 16 cells positive: the same 'no consistent "
                               "sign', the same order-of-magnitude collapse. The conclusion HOLDS "
                               "in the gate quantity; it was simply never demonstrated there.",
            "note": "in the gate quantity the RAW advantage is LARGER (+0.2282 .. +0.3260 vs the "
                    "+0.1012 .. +0.2923 quoted on gross) because the hold-shares book carries "
                    "less short notional and so pays less borrow.",
            "status": "the borrow column is NOT added here -- that is new computation. Rowed as "
                      "follow-up work.",
        },
        "bear22_counter_check": {
            "target_weight_gross": bear_tw["gross"],
            "hold_shares_gross_raw": bear_hs["gross"],
            "hold_shares_gross_renorm": bear_rn.get("gross"),
            "renorm_minus_tw": (bear_rn.get("gross") - bear_tw["gross"]
                                if bear_rn.get("gross") is not None else None),
            "reading": "renormalisation does NOT rescue hold-shares in BEAR22 -- the de-levered "
                       "book loses to target-weight there by MORE than the levered one does "
                       "(-0.775 vs -0.602). So Correction 2's BEAR22 DIRECTION is a genuine "
                       "compositional effect and SURVIVES this correction, while the full-window "
                       "ADVANTAGE is left UNRESOLVED (see summary_primary.verdict and "
                       "construction_caveat) rather than killed. Note the two windows are NOT "
                       "the same mechanism read twice: in the full window the exposure drift "
                       "accounts for ~85% of the effect, whereas in BEAR22 it was mildly "
                       "CUSHIONING a deficit that comes from elsewhere (mostly the compounding "
                       "convention, per correction 2's attribution) and that survives its "
                       "removal. That asymmetry is measured, not assumed.",
            "direction_support": "the BEAR22 direction rests on the RENORMALISED gap (-0.7750, "
                                 "paired 95% CI [-1.109, -0.452], stable across 6 bootstrap "
                                 "seeds) plus the 830_corridor prior -- NOT on the total gap. "
                                 "The TOTAL gap -0.6019 has paired 95% CI [-1.158, -0.037] at "
                                 "the shipped seed but [-1.115, +0.009] at seed 3, i.e. it "
                                 "straddles zero on another draw. Both CIs transcribed from the "
                                 "cycle-3 estimator lens; not recomputed here.",
        },
        "cost_caveat": "GROSS columns are an EXACT rescaling of the realised return series. NET "
                       "columns charge the LEVERED book's own turnover against the de-levered "
                       "return series (a book actually run at gross 1 would trade different "
                       "notionals), so they are an approximation. DIRECTION, corrected in the "
                       "closing pass: it flatters the renormalised arm at 12 of 16 PRIMARY cells "
                       "and PENALISES it at 4 -- mean Sum|w| exceeds 1 at H=2/th=1.0 (1.0142), "
                       "H=3/th=1.0 (1.0104) and H=5/th=1.0 (1.0316), and at H=1/th=1.0 (mean "
                       "0.9931) the per-cell recomputation still penalises, because Sharpe "
                       "depends on G's whole distribution and not on its mean. Cycle 3's 'mean "
                       "Sum|w| < 1 at EVERY cell, so the net columns flatter this arm' was a "
                       "FALSE universal, contradicted by this module's own hold_shares_diag. "
                       "Magnitude <= 0.002 Sharpe, so nothing flips (max RENORM-TW +0.0309 -> "
                       "+0.0303, min -0.0429 -> -0.0442); the gross-only comparison carries the "
                       "conclusion regardless.",
    }

    # correction 4 (MINOR-E4 / O1): theta's measured flip-turnover reduction vs the inherited
    # "theta=0.5 sigma kills 40-50% of flips" claim (tactics_gaps.md:43, self-labelled heuristic).
    corr4 = []
    flip0 = {H: [x for x in cells_primary if x["H"] == H and x["theta"] == 0.0
                 and x["accounting"] == "target-weight"][0]["decomposition"]["flip"]
             for H in H_GRID}
    for theta in THETA_GRID:
        row = {"theta": theta}
        for H in H_GRID:
            c = [x for x in cells_primary if x["H"] == H and x["theta"] == theta
                 and x["accounting"] == "target-weight"][0]
            row[f"flip_H{H}"] = c["decomposition"]["flip"]
            row[f"reduction_H{H}"] = 1.0 - c["decomposition"]["flip"] / flip0[H]
        corr4.append(row)

    # correction 3 / band verdict
    band_p = band_verdict(cells_primary, frozen_p)
    band_s = band_verdict(cells_secondary, frozen_s)
    band_d = band_verdict(cells_disjoint, frozen_d)
    h1_p = frozen_p
    h3_p = [c for c in cells_primary if c["H"] == 3 and c["theta"] == 0.0
            and c["accounting"] == "target-weight"][0]

    sig_p = significance_summary(cells_primary)
    sig_s = significance_summary(cells_secondary)
    sig_d = significance_summary(cells_disjoint)

    # MAJOR-E1: MDE. Computed on the PRIMARY window (the grid's own denominator) and the
    # correlation leg measured on the H=1 cells, where candidate and frozen share their rows.
    mde_p = mde_analysis(cells_primary)
    h1_series = {("frozen" if (th == 0.0 and acc == "target-weight") else f"H1_th{th}_{acc}"):
                 series_p[(1, th, acc)]["net_primary"]
                 for th in THETA_GRID for acc in ACCOUNTING_GRID}
    rho_p = measured_pairwise_correlation(h1_series, mde_p["se_used"])

    report = {
        "fixture_sha256": fixture_sha,
        "panel_info": {"n_symbols": len(syms), "shape": list(logc.shape), **info},
        "grid_definition": {"H": list(H_GRID), "theta": list(THETA_GRID),
                            "accounting": list(ACCOUNTING_GRID), "n_cells": N_CELLS,
                            "fee_tiers_bp": list(FEE_TIERS_BP), "fee_916_ladder": FEE_TIERS_916,
                            "borrow_scenarios": list(BORROW_SCENARIOS),
                            "primary_gate_quantity": f"net{PRIMARY_FEE_BP}_borrow_{PRIMARY_BORROW}",
                            "bonferroni_alpha": BONFERRONI_ALPHA},
        "windows": {"primary": PRIMARY_WINDOW, "secondary": SECONDARY_WINDOW,
                    "disjoint": DISJOINT_WINDOW},
        "window_relations": {
            "secondary_contains_primary": True,
            "note": "SECONDARY (2018-01-01..2025-09-01) is a strict SUPERSET of PRIMARY "
                    "(2021-01-01..2025-09-01). Scoring a PRIMARY-SELECTED cell on SECONDARY is "
                    "dilution-by-extra-years, NOT an out-of-sample check: the shrinkage it shows "
                    "is expected under pure overfitting whether or not the effect is real. "
                    "DISJOINT (2018-01-01..2021-01-01) = SECONDARY minus PRIMARY is the only "
                    "genuinely held-out span in this pass. No permutation test and no "
                    "max-of-correlated-normals correction was performed.",
        },
        "n_independent_position_paths": {
            "cells": N_CELLS,
            "position_paths": len(H_GRID) * len(THETA_GRID) * 2,
            "note": "target-weight and drift-inclusive share ONE position path per (H,theta) -- "
                    "they differ only in whether the drift re-trim is CHARGED -- while "
                    "hold-shares is its own path (shares carried, weights drifting). So 48 cells "
                    "rest on 32 distinct position paths, not 48. (Before the run-55 cycle-2 fix "
                    "all three modes shared ONE path, i.e. 16.) Bonferroni-over-48 is KEPT: it "
                    "controls FWER without any independence assumption, and is conservative here "
                    "precisely because it re-tests each trajectory more than once.",
        },
        "cells_primary": cells_primary,
        "cells_secondary": cells_secondary,
        "cells_disjoint": cells_disjoint,
        "frozen_primary": frozen_p,
        "frozen_secondary": frozen_s,
        "frozen_disjoint": frozen_d,
        "best_primary": best_p,
        "best_cell_on_secondary_window": best_s_same_cell,
        "best_cell_on_disjoint_window": best_d_same_cell,
        "mde_primary": mde_p,
        "measured_correlation_primary_h1": rho_p,
        "uplift_ci_summary_primary": uplift_p,
        "uplift_ci_summary_secondary": uplift_s,
        "uplift_ci_summary_disjoint": uplift_d,
        "leverage_renormalisation": leverage_renorm,
        "correction_1_analytic_anchor": corr1,
        "correction_2_bear22_holdshares": {
            "STATUS": "RE-DERIVED under a real share-carrying path (run 55 cycle 2). The "
                      "cycle-1 CONFIRMED status was RETRACTED: that pass scored the same "
                      "target-weight position array in all three accounting modes and never "
                      "carried a share, so it did not test this claim at all.",
            "n_bear22": bear_tw["n"],
            "bear22_target_weight_gross_sharpe": bear_tw["gross"],
            "bear22_hold_shares_gross_sharpe": bear_hs["gross"],
            "bear22_delta_hold_minus_target": bear_hs["gross"] - bear_tw["gross"],
            "primary_full_target_weight_gross_sharpe": full_tw["gross"],
            "primary_full_hold_shares_gross_sharpe": full_hs["gross"],
            "primary_full_delta_hold_minus_target": full_hs["gross"] - full_tw["gross"],
            "direction_confirms_spec_in_bear22": bool(bear_hs["gross"] < bear_tw["gross"]),
            "sign_reverses_outside_bear22": bool(
                (bear_hs["gross"] - bear_tw["gross"]) * (full_hs["gross"] - full_tw["gross"]) < 0),
            # ATTRIBUTION: split each window's total hold-minus-target gap into the COMPOUNDING
            # CONVENTION (log-sum -> simple-sum on the SAME target positions -- the only thing
            # cycle 1 measured) and the SHARE CARRY (simple-sum on target -> the carried book).
            "attribution": {
                w_name: {
                    "gross_log_on_target": c["gross_log_on_target"],
                    "gross_simple_on_target": c["gross_simple_on_target"],
                    "gross_hold_shares_carried": c["gross"],
                    "delta_from_compounding_convention":
                        c["gross_simple_on_target"] - c["gross_log_on_target"],
                    "delta_from_share_carry": c["gross"] - c["gross_simple_on_target"],
                }
                for w_name, c in (("bear22", bear_hs), ("primary_full", full_hs))
            },
            # MAJOR-E2B (run 55 cycle 3): every component above now carries a PAIRED 95% CI, and
            # a component whose interval contains 0 is reported UNRESOLVED. The pass refuses the
            # band question for want of power; it may not simultaneously assert a SMALLER,
            # NOISIER split as confirmed.
            "attribution_ci": attribution_cis,
            "bear22_hold_shares_gross_exposure": bear_hs.get("hold_shares_diag"),
            "window": "BEAR22 (config.REGIMES, 2021-11-10..2022-12-31) vs PRIMARY full",
        },
        "correction_4_theta_flip_reduction": {
            "record_claim": "tactics_gaps.md:43 (self-labelled heuristic, 'must be measured'): "
                            "'theta = 0.5 sigma kills 40-50% of flips', '~70-75%' at theta=1 sigma",
            "measured": corr4,
            "verdict": "the record UNDERSTATES hysteresis in the permissive direction -- theta=0.5 "
                       "removes far more flip turnover than 40-50%. So theta is NOT the cheap free "
                       "lunch the record implies: it cuts deep enough to destroy signal, which is "
                       "exactly what the high-theta cells' strongly negative Delta net shows.",
        },
        "correction_3_gross_cost_decomposition_h1_vs_h3": {
            "window": "primary",
            "h1": {"net21": h1_p["net21"], "gross": h1_p["gross"], "cost_drag": cost_drag(h1_p)},
            "h3": {"net21": h3_p["net21"], "gross": h3_p["gross"], "cost_drag": cost_drag(h3_p)},
            "delta_net": h3_p["net21"] - h1_p["net21"],
            "delta_gross": h3_p["gross"] - h1_p["gross"],
            "delta_cost_drag": cost_drag(h3_p) - cost_drag(h1_p),
        },
        "band_verdict_primary": band_p,
        "band_verdict_secondary": band_s,
        "band_verdict_disjoint": band_d,
        "significance_primary": sig_p,
        "significance_secondary": sig_s,
        "significance_disjoint": sig_d,
        "band_disposition": {
            "verdict": "REFUSED FOR WANT OF POWER",
            "reason": "Three independent instruments agree. (1) MDE: the design's minimum "
                      "detectable uplift at 80% power is 1.6x-13x the pre-registered +-0.20 band "
                      "across every correlation assumption, including the most favourable one "
                      "MEASURED on this panel (mde_primary, measured_correlation_primary_h1). "
                      "(2) DIRECT: a block-bootstrap CI on the UPLIFT ITSELF -- paired where the "
                      "rows allow it -- straddles +0.20 for 46 of 48 cells at the family's own "
                      "pre-registered alpha=0.05/48, and 0 of 48 cells clear the band at that "
                      "alpha on ANY of the three windows (uplift_ci_summary_*). The grid's own "
                      "max-uplift cell is among the undecided. (3) HOLDOUT: on the genuinely "
                      "disjoint 2018-2020 window, 0/48 cells' uplift CI even excludes zero. "
                      "This conclusion is PRIOR TO and INDEPENDENT OF the hold-shares "
                      "implementation -- no re-implementation rescues it.",
            "what_would_resolve_it": "roughly 30x more decisions at this SE (SE scales as "
                      "1/sqrt(n), so shrinking a 0.46 SE to the ~0.05 needed to resolve +-0.20 "
                      "at Bonferroni alpha needs ~85x the sample), or a paired design that "
                      "shares decision rows across the whole grid (the H axis structurally "
                      "cannot), or a pre-registered SINGLE cell instead of a 48-cell search.",
            "precedent": "same disposition run 54 gave TASK_917's P1 (kill refused for want of "
                         "power -> D84).",
        },
    }
    return report


def main():
    import json
    report = build_report()
    out_path = Path(__file__).parent / "result_919a.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    print(f"frozen primary net21_borrow_low = {report['frozen_primary']['net21_borrow_low']:+.4f}")
    print(f"best   primary net21_borrow_low = {report['best_primary']['net21_borrow_low']:+.4f}  "
          f"cell={report['best_primary']['H'],report['best_primary']['theta'],report['best_primary']['accounting']}")
    print(f"significance (primary): {report['significance_primary']}")
    rn = report['band_verdict_primary']['reading_max_delta_net']
    print(f"band reading A (max Delta net): {rn['value']:+.4f} at cell {rn['cell']}")
    for mode, v in report['band_verdict_primary'][
            'reading_max_delta_cost_drag_saving_WITHIN_MODE'].items():
        print(f"band reading B (max cost_drag saving WITHIN {mode}): "
              f"{v['max_cost_drag_saving']:+.4f} at cell {v['cell']}")
    m = report['mde_primary']
    print(f"\nMDE: SE(single cell) bootstrap {m['se_single_cell_bootstrap']['median']:.4f} vs "
          f"closed-form {m['se_single_cell_closed_form']['median']:.4f}")
    for label, v in m['mde'].items():
        print(f"  {label}: MDE raw {v['mde_raw_alpha_0.05']:.3f} "
              f"({v['mde_raw_over_band']:.1f}x band) | bonf {v['mde_bonferroni']:.3f} "
              f"({v['mde_bonferroni_over_band']:.1f}x band)")
    print(f"BAND DISPOSITION: {report['band_disposition']['verdict']}")


if __name__ == "__main__":
    main()

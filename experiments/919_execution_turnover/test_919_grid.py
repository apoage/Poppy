#!/usr/bin/env python3
"""TASK_919 phase A -- gates G1-G5 (brief section 8) + the analytic-anchor / hour-of-day
verification checks + the cycle-2 fix-pass gates.

Hermetic where possible (planted panels); the fixture-dependent checks SKIP with a notice if
fixtures/trading_fixture.db is absent, following the project's own ST8/ST17 import-guard
convention (never red a determinism gate on a fixture-less host).

CYCLE-2 (fix pass) ADDITIONS, each closing a named panel finding:
  test_g1_noop_bit_identical            MINOR-C2: element-wise, not 3 scalar reductions
  test_g2_no_trader_edits               MINOR-R1: actually runs git, instead of returning a string
  test_borrow_indexing_equivalence      MINOR-C3: the test a code comment claimed but never had
  test_hold_shares_return_identity      BLOCKER-C1: the carried book's equity path == the
                                        weighted simple return of the weights it actually held
  test_g5_hold_shares_poison            BLOCKER-C1 / G5: a variant that RE-STRIKES shares every
                                        decision (i.e. exactly the cycle-1 defect) must produce
                                        DIFFERENT numbers -- proves these tests would have FAILED
                                        against the old implementation
  test_g4_hold_shares_leak_free         G4 extended to the new share-carrying path
  test_hold_shares_fail_loud            the fail-loud contract inherited from true_bh_series
  test_results_tables_match_json        BLOCKER-F1: RESULTS.md's tables cannot drift from the json

CYCLE-3 (fix pass) ADDITIONS:
  test_prose_number_gate                MAJOR-C2A: the drift gate now reaches PROSE numbers, and
                                        the judge's own demonstrated poison is shown to be
                                        ACCEPTED by the cycle-2 gate and REJECTED by this one
  test_gross_renormalisation_exact      BLOCKER-E2A: the gross-renormalised comparator is exact,
                                        is not a constant rescale, and erases the hold-shares
                                        advantage it was built to test
  test_uplift_ci_pairing_alignment      MINOR-E2D: the paired bootstrap's positional-alignment
                                        invariant is asserted, not assumed

Run: trader_env/bin/python experiments/919_execution_turnover/test_919_grid.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import grid_harness as gh
import render_tables as rt
from trader import backtest as bt
from trader import config as tconfig
from trader import strategy as tstrat

FIXTURE_PRESENT = Path(gh.FIXTURE_DB).exists()

_PANEL = {}


def _panel():
    """Load the PIT panel + base arrays once and reuse (the fixture read is the slow part)."""
    if not _PANEL:
        logc, ts_d, syms, info = gh.load_pit_panel()
        rv, pres, w, S_full, warm = gh.build_base_arrays(logc)
        r_lo, r_hi = gh.window_bounds(ts_d, gh.PRIMARY_WINDOW["start"], gh.PRIMARY_WINDOW["end"])
        _PANEL.update(logc=logc, ts_d=ts_d, syms=syms, rv=rv, pres=pres, w=w,
                      S_full=S_full, warm=warm, r_lo=r_lo, r_hi=r_hi)
    return _PANEL


# =============================================================================
# G1 -- no-op bit-identity, now ELEMENT-WISE (MINOR-C2)
# =============================================================================

def test_g1_noop_bit_identical():
    """G1: at theta=0, H=1, accounting=target-weight, borrow=none, the harness reproduces
    compute_full_cell on the PRIMARY window bit-identically (net@21 +0.8358, turn 0.1561,
    n=1704, gross +1.0339).

    MINOR-C2 fix (run 55 cycle 2): cycle 1 compared THREE SCALAR REDUCTIONS (mean gross Sharpe,
    mean turn, net21), which cannot rule out a compensating per-decision error. This now asserts
    np.array_equal on the FULL per-decision gross, turn AND net@21 series at every H in the
    grid, against a hand-built regime_series call on the canon positions_from_config output.

    Non-vacuity: a ONE-CELL perturbation (planted sign flip on a single asset at a single
    decision) must FAIL the identity."""
    if not FIXTURE_PRESENT:
        return "SKIPPED (fixtures/trading_fixture.db absent)"
    p = _panel()
    logc, ts_d, syms = p["logc"], p["ts_d"], p["syms"]
    rv, pres, w, S_full, warm = p["rv"], p["pres"], p["w"], p["S_full"], p["warm"]
    r_lo, r_hi = p["r_lo"], p["r_hi"]

    cell = gh.score_cell(logc, ts_d, syms, rv, pres, w, S_full, warm, r_lo, r_hi,
                         1, 0.0, "target-weight", return_series=True)
    ref = bt.compute_full_cell(logc, ts_d)["reg"]["FULL"]
    assert cell["n"] == ref["n"] == 1704, (cell["n"], ref["n"])
    assert abs(ref["net21"] - 0.8357541266542514) < 1e-12
    assert abs(ref["turn"] - 0.1560866077894317) < 1e-12
    assert abs(ref["gross"] - 1.0338583098255947) < 1e-12
    assert cell["gross"] == ref["gross"], (cell["gross"], ref["gross"])
    assert cell["turn"] == ref["turn"], (cell["turn"], ref["turn"])
    assert cell["net21"] == ref["net21"], (cell["net21"], ref["net21"])

    # ELEMENT-WISE identity at every H, theta=0, against canon regime_series.
    pos_c, pres_c, wgt_c, warm_c = tstrat.positions_from_config(logc, tconfig.STRATEGY)
    for H in gh.H_GRID:
        cell_h = gh.score_cell(logc, ts_d, syms, rv, pres, w, S_full, warm, r_lo, r_hi,
                               H, 0.0, "target-weight", return_series=True)
        g_ref, tu_ref, _bh = bt.regime_series(logc, pos_c, pres_c, r_lo, r_hi, H, warm_c)
        g_got = cell_h["_series"]["gross"]
        tu_got = cell_h["_series"]["turn"]
        assert g_got.shape == g_ref.shape == (cell_h["n"],), (g_got.shape, g_ref.shape)
        assert np.array_equal(g_got, g_ref), \
            f"G1 element-wise: H={H} gross differs at {int(np.sum(g_got != g_ref))} of " \
            f"{g_ref.size} decisions (max abs {float(np.max(np.abs(g_got - g_ref)))})"
        assert np.array_equal(tu_got, tu_ref), \
            f"G1 element-wise: H={H} turn differs at {int(np.sum(tu_got != tu_ref))} of " \
            f"{tu_ref.size} decisions (max abs {float(np.max(np.abs(tu_got - tu_ref)))})"
        n21_got = bt.net_series(g_got, tu_got, 21)
        n21_ref = bt.net_series(g_ref, tu_ref, 21)
        assert np.array_equal(n21_got, n21_ref), f"G1 element-wise: H={H} net@21 differs"

    # G5 non-vacuity: perturb ONE cell of the canon position array and confirm the ELEMENT-WISE
    # comparison above would have caught it (not just the scalar mean).
    pos_bad = pos_c.copy()
    t_probe = np.arange(max(warm_c, r_lo), min(r_hi - 1, logc.shape[0] - 2))
    row = int(t_probe[len(t_probe) // 2])
    col = int(np.argmax(pres[row]))
    pos_bad[row, col] = -pos_bad[row, col] if pos_bad[row, col] != 0 else 0.01
    g_bad, tu_bad, _ = bt.regime_series(logc, pos_bad, pres_c, r_lo, r_hi, 1, warm_c)
    g_ok, tu_ok, _ = bt.regime_series(logc, pos_c, pres_c, r_lo, r_hi, 1, warm_c)
    assert not np.array_equal(tu_bad, tu_ok), \
        "G5 non-vacuity: a one-cell perturbation must break the element-wise turn identity"
    assert float(np.mean(tu_bad)) != ref["turn"], \
        "G5 non-vacuity: a one-cell perturbation must also move the mean turnover off canon"
    return "PASS"


# =============================================================================
# G2 -- trader/ untouched. MINOR-R1: this now actually checks (it used to return a string).
# =============================================================================

TRADER_LAST_COMMIT_AT_TASK_START = "18d42d3"


def test_g2_no_trader_edits():
    """G2 (flag OFF / bit-identical): this task must touch NO file under trader/.

    MINOR-R1 fix (run 55 cycle 2). Cycle 1's version was `import trader; return "PASS (...)"` --
    structurally unable to reach the claim it certified, so a future edit to trader/ would still
    have printed PASS. This runs git and ASSERTS three ways: no unstaged diff, no staged diff,
    and no untracked/modified paths under trader/. It also pins the last commit that touched
    trader/, so a COMMITTED edit slipped in by another lane is visible too.

    `python -m trader selftest` stays a separate invocation (not called from here, to avoid a
    circular-import surprise inside the suite)."""
    def git(*args):
        return subprocess.run(["git", "-C", str(_REPO_ROOT), *args],
                              capture_output=True, text=True, check=True).stdout
    unstaged = git("diff", "--stat", "HEAD", "--", "trader/").strip()
    staged = git("diff", "--cached", "--stat", "--", "trader/").strip()
    status = git("status", "--porcelain", "--", "trader/").strip()
    assert unstaged == "", f"G2: trader/ has uncommitted edits:\n{unstaged}"
    assert staged == "", f"G2: trader/ has STAGED edits:\n{staged}"
    assert status == "", f"G2: git status is not clean under trader/:\n{status}"
    last = git("log", "-1", "--format=%h", "--", "trader/").strip()
    assert last.startswith(TRADER_LAST_COMMIT_AT_TASK_START), \
        (f"G2: trader/ last changed at {last}, expected "
         f"{TRADER_LAST_COMMIT_AT_TASK_START} -- something committed to trader/ since this "
         f"task began. Re-verify the canon parity claim before trusting any number here.")
    return f"PASS (trader/ clean; last touched at {last})"


# =============================================================================
# G3 / G4 -- theta path-dependence and leak-freedom (unchanged; panel verified these clean)
# =============================================================================

def test_g3_theta_path_dependence():
    """G3: the retained-sign scan carries across DECISION points (t, t+H, 2H, ...), never
    across every daily row, and resets on presence-mask entry. Planted panel: an asset whose
    raw sign flips on a day that falls BETWEEN two decisions (H=3) but is back to the original
    sign again by the next decision. An INCORRECT per-day implementation is planted here as the
    negative control and shown to disagree with the per-decision one."""
    T, A = 60, 1
    rng = np.random.RandomState(3)
    L = 5
    ret = np.full(T, 0.01)
    blip_row = 32   # strictly between decisions at H=3 starting from row>=L+1
    ret[blip_row] = -0.30
    logc = np.cumsum(ret).reshape(T, 1) + rng.randn(T, 1) * 1e-6
    pres = np.ones((T, A), dtype=bool)
    rv = np.full((T, A), 0.01)   # fixed, so sigma_S is deterministic and known
    w = np.ones((T, A))
    warm_pad = np.full((L, A), np.nan)
    S_full = np.concatenate([warm_pad, logc[L:] - logc[:-L]], axis=0)

    H, theta = 3, 1.0
    t = np.arange(L + 2, T - H - 1, H)
    t_ext = np.concatenate([[t[0] - H], t])
    pos_ext, sign_ext = gh.hysteresis_build(rv, pres, w, S_full, t_ext, theta, L=L)

    def per_day_wrong(rv, pres, w, S_full, row_lo, row_hi, theta, L):
        A = rv.shape[1]
        carried = np.full(A, np.nan)
        out_sign = np.zeros(row_hi - row_lo, dtype=float)
        for k, row in enumerate(range(row_lo, row_hi)):
            pr = pres[row]
            raw_sign = np.sign(S_full[row])
            sigma_s = rv[row] * np.sqrt(L)
            has_carry = pr & np.isfinite(carried)
            within = np.abs(S_full[row]) <= theta * sigma_s
            use = np.where(has_carry & within, carried, raw_sign)
            use = np.where(pr, use, 0.0)
            out_sign[k] = use[0]
            carried = np.where(pr, use, carried)
        return out_sign

    per_day_signs = per_day_wrong(rv, pres, w, S_full, int(t_ext[0]), int(t_ext[-1]) + 1, theta, L)
    per_day_at_decisions = per_day_signs[t_ext - t_ext[0]]
    assert not np.array_equal(sign_ext[:, 0], per_day_at_decisions), \
        "G3: a per-DAY hysteresis scan and the per-DECISION scan must disagree on this planted " \
        "panel (the blip sits strictly between two decisions)"

    pres2 = pres.copy()
    pres2[t_ext[1]] = False    # absent at the decision right after the first
    _pos2, sign_ext2 = gh.hysteresis_build(rv, pres2, w, S_full, t_ext, theta, L=L)
    raw_at_2 = float(np.sign(S_full[int(t_ext[2])])[0])
    assert sign_ext2[2, 0] == raw_at_2, \
        "G3: re-entry after an absence must reset to raw sign, not retain stale state"
    return "PASS"


def test_g4_leak_free():
    """G4: sigma_S (and the raw trend statistic S) use rows <= t only. ST6 wrecking pattern --
    corrupt closes > t0 and assert hysteresis positions <= t0 are byte-identical."""
    rng = np.random.RandomState(13)
    T, A = 200, 4
    logc = np.cumsum(rng.randn(T, A) * 0.01, axis=0)
    rv, pres, w, S_full, warm = gh.build_base_arrays(logc)
    H, theta, t0 = 2, 0.5, 150
    t = np.arange(warm, t0, H)
    t_ext = np.concatenate([[t[0] - H], t])
    pos1, sign1 = gh.hysteresis_build(rv, pres, w, S_full, t_ext, theta, L=gh.L_TREND)

    logc2 = logc.copy()
    logc2[t0 + 1:] += np.cumsum(rng.randn(T - t0 - 1, A) * 0.05, axis=0)
    rv2, pres2, w2, S_full2, _warm2 = gh.build_base_arrays(logc2)
    pos2, sign2 = gh.hysteresis_build(rv2, pres2, w2, S_full2, t_ext, theta, L=gh.L_TREND)

    assert np.array_equal(pos1, pos2, equal_nan=True), \
        "G4 LEAK: wrecking close>t0 altered hysteresis positions at decisions <= t0"
    assert np.array_equal(sign1, sign2, equal_nan=True), \
        "G4 LEAK: wrecking close>t0 altered retained-sign state at decisions <= t0"
    return "PASS"


# =============================================================================
# hold-shares: the BLOCKER-C1 replacement path
# =============================================================================

def _planted_share_panel(T=140, A=6, seed=11):
    """A small synthetic panel with real presence/vol structure, for the hold-shares tests."""
    rng = np.random.RandomState(seed)
    logc = np.cumsum(rng.randn(T, A) * 0.02, axis=0) + 1.0
    rv, pres, w, S_full, warm = gh.build_base_arrays(logc)
    return logc, rv, pres, w, S_full, warm


def _hold_shares_restrike_every_decision(price, pos_ext, sign_ext, t_ext, H):
    """THE POISON (G5). Identical to gh.hold_shares_walk EXCEPT that it re-strikes shares to the
    full target at EVERY decision instead of only where the used sign changes. That is exactly
    the cycle-1 defect: the position path collapses back onto target-weight's and no share is
    ever carried. Any test that cannot tell this apart from the real walk is worthless."""
    A = price.shape[1]
    rows = np.concatenate([np.asarray(t_ext), [int(t_ext[-1]) + H]])
    shares = np.zeros(A)
    cash = 0.0
    Vs = np.empty(rows.size)
    turn = np.zeros(t_ext.size)
    held_w = np.zeros((t_ext.size, A))
    for k, row in enumerate(rows):
        px = price[int(row)]
        held = shares != 0.0
        if k == 0:
            V, marked = 1.0, np.zeros(A)
        else:
            marked = shares * np.where(held, px, 0.0)
            V = cash + float(np.sum(marked))
        Vs[k] = V
        if k == rows.size - 1:
            break
        drift_w = marked / V if k > 0 else np.zeros(A)
        w_new = pos_ext[k]                      # <-- the poison: ALWAYS the target
        turn[k] = float(np.sum(np.abs(w_new - drift_w)))
        held_w[k] = w_new
        safe_px = np.where(np.isfinite(px) & (px > 0.0), px, 1.0)
        shares = (w_new * V) / safe_px
        cash = V - float(np.sum(shares * np.where(shares != 0.0, px, 0.0)))
    return Vs[2:] / Vs[1:-1] - 1.0, turn[1:], held_w[1:]


def test_hold_shares_return_identity():
    """The carried book's return must be an EQUITY-PATH quantity, and it must equal the weighted
    SIMPLE return of the weights the book actually held over each period:
        ret[j] == Sum_i held_w[j,i] * (P[t_j + H, i] / P[t_j, i] - 1)
    This pins the cash/share accounting (V_{k+1} = cash + Sum shares*P collapses to exactly this)
    and would fail if the walk silently scored TARGET weights instead of held ones."""
    logc, rv, pres, w, S_full, warm = _planted_share_panel()
    price = np.exp(logc)
    T = logc.shape[0]
    for H in (1, 3):
        for theta in (0.0, 0.5):
            t = np.arange(warm, T - H - 1, H)
            t_ext = np.concatenate([[t[0] - H], t])
            pos_ext, sign_ext = gh.hysteresis_build(rv, pres, w, S_full, t_ext, theta)
            ret, turn, held_w, diag = gh.hold_shares_walk(price, pos_ext, sign_ext, t_ext, H)
            simp = price[t + H] / price[t] - 1.0
            expect = np.sum(held_w * np.where(np.isfinite(simp), simp, 0.0), axis=1)
            assert np.allclose(ret, expect, atol=1e-12), \
                f"hold-shares return identity broken at H={H}, theta={theta}: max abs " \
                f"{float(np.max(np.abs(ret - expect)))}"
            # a carried book is NOT pinned at gross 1 -- that is the mechanism, assert it moves
            assert diag["gross_abs_max"] > 1.0 + 1e-6 or diag["gross_abs_min"] < 1.0 - 1e-6, \
                "hold-shares: gross exposure never left 1.0, so nothing drifted -- the book is " \
                "being re-trimmed to target and no share is actually being carried"
    return "PASS"


def test_g5_hold_shares_poison():
    """G5 for the NEW mechanism, planted where it is WEAKEST. `_hold_shares_restrike_every_
    decision` is the cycle-1 defect reproduced exactly: same equity accounting, same simple
    returns, same fail-loud shape -- but shares re-struck to the target at EVERY decision, so no
    share is ever carried. This test asserts the real walk and the poison DISAGREE, i.e. that the
    hold-shares tests in this file would have FAILED against the old implementation.

    It also pins the poison's identity to the OTHER two accounting modes: re-striking every
    decision reproduces the target-weight turnover exactly and the simple-return gross exactly,
    which is precisely what the correctness lens found cycle 1 was computing."""
    logc, rv, pres, w, S_full, warm = _planted_share_panel()
    price = np.exp(logc)
    T, A = logc.shape
    disagreements = 0
    for H in (1, 2, 3):
        for theta in (0.0, 0.25, 0.5):
            t = np.arange(warm, T - H - 1, H)
            t_ext = np.concatenate([[t[0] - H], t])
            pos_ext, sign_ext = gh.hysteresis_build(rv, pres, w, S_full, t_ext, theta)
            real_ret, real_turn, real_w, _d = gh.hold_shares_walk(price, pos_ext, sign_ext,
                                                                  t_ext, H)
            p_ret, p_turn, p_w = _hold_shares_restrike_every_decision(price, pos_ext, sign_ext,
                                                                      t_ext, H)
            assert not np.allclose(real_ret, p_ret, atol=1e-10), \
                f"G5 VACUOUS: at H={H}, theta={theta} the real share-carrying walk and a variant " \
                f"that re-strikes shares EVERY decision produce the same returns. The test " \
                f"cannot detect the cycle-1 defect."
            assert not np.allclose(real_turn, p_turn, atol=1e-10), \
                f"G5 VACUOUS: at H={H}, theta={theta} the poison's turnover is indistinguishable"
            assert not np.array_equal(real_w, p_w), \
                f"G5 VACUOUS: at H={H}, theta={theta} the poison holds the same weights"
            disagreements += 1
            # The poison IS the cycle-1 book: prove it, so the equivalence the correctness lens
            # diagnosed is PINNED rather than asserted. Two exact identities:
            #   (a) it holds the TARGET weights, i.e. no share is carried -- the defect itself;
            #   (b) its return is bt.gross_simple_series on the target positions -- exactly the
            #       log-vs-simple compounding swap cycle 1 mislabelled "hold-shares".
            # Its TURNOVER is only first-order equal to cycle 1's Sum|d target - drift_signed|:
            # cycle 1 dropped the /(1+r_prev) book-growth normalisation that a real equity path
            # carries, so the two agree to O(r) and are deliberately NOT compared exactly.
            pos = pos_ext[1:]
            pos_prev = pos_ext[:-1]
            assert np.allclose(p_w, pos, atol=1e-12), \
                "the poison must hold exactly the TARGET weights (that is what makes it the bug)"
            g_simple = bt.gross_simple_series(logc, gh._scatter_full(T, A, t_ext, pos_ext), t, H)
            assert np.allclose(p_ret, g_simple, atol=1e-12), \
                "the poison's return must equal gross_simple_series on the target positions -- " \
                "i.e. exactly the log-vs-simple swap cycle 1 mislabelled 'hold-shares'"
            fwd_prev = logc[t] - logc[t - H]
            fwd_prev = np.where(np.isfinite(fwd_prev), fwd_prev, 0.0)
            cycle1_turn = np.sum(np.abs((pos - pos_prev) - pos_prev * (np.exp(fwd_prev) - 1.0)),
                                 axis=1)
            assert abs(float(np.mean(p_turn)) - float(np.mean(cycle1_turn))) < 0.02, \
                "the poison's turnover must reproduce cycle 1's netted-retrim turnover to " \
                "first order (the only difference is the book-growth normalisation)"
    assert disagreements == 9
    return f"PASS ({disagreements}/9 poison variants correctly rejected)"


def test_g4_hold_shares_leak_free():
    """G4 extended to the share-carrying path (the cycle-1 G4 only covered hysteresis_build).
    Wreck every close AFTER t0 and assert the carried book's returns, turnover and held weights
    at decisions ENDING at or before t0 are byte-identical."""
    rng = np.random.RandomState(29)
    T, A = 220, 5
    logc = np.cumsum(rng.randn(T, A) * 0.02, axis=0) + 1.0
    rv, pres, w, S_full, warm = gh.build_base_arrays(logc)
    H, theta, t0 = 2, 0.5, 160
    t = np.arange(warm, t0 - H, H)          # every scored period [t, t+H) ends at or before t0
    t_ext = np.concatenate([[t[0] - H], t])
    pos_ext, sign_ext = gh.hysteresis_build(rv, pres, w, S_full, t_ext, theta)
    r1, tu1, hw1, _ = gh.hold_shares_walk(np.exp(logc), pos_ext, sign_ext, t_ext, H)

    logc2 = logc.copy()
    logc2[t0 + 1:] += np.cumsum(rng.randn(T - t0 - 1, A) * 0.08, axis=0)
    rv2, pres2, w2, S_full2, _ = gh.build_base_arrays(logc2)
    pos_ext2, sign_ext2 = gh.hysteresis_build(rv2, pres2, w2, S_full2, t_ext, theta)
    r2, tu2, hw2, _ = gh.hold_shares_walk(np.exp(logc2), pos_ext2, sign_ext2, t_ext, H)

    assert np.array_equal(r1, r2), "G4 LEAK (hold-shares): wrecking close>t0 moved the carried " \
                                   "book's returns at decisions ending <= t0"
    assert np.array_equal(tu1, tu2), "G4 LEAK (hold-shares): turnover moved"
    assert np.array_equal(hw1, hw2), "G4 LEAK (hold-shares): held weights moved"
    return "PASS"


def test_hold_shares_fail_loud():
    """The fail-loud contract inherited from true_bh_series: a HELD asset priced non-finite (or
    <= 0) must RAISE, never be floored into a plausible-looking number. Planted two ways: a NaN
    close and a zero close on an asset the book is actually holding."""
    logc, rv, pres, w, S_full, warm = _planted_share_panel()
    T, A = logc.shape
    H, theta = 1, 0.0
    t = np.arange(warm, T - H - 1, H)
    t_ext = np.concatenate([[t[0] - H], t])
    pos_ext, sign_ext = gh.hysteresis_build(rv, pres, w, S_full, t_ext, theta)
    # sanity: the clean panel walks fine
    gh.hold_shares_walk(np.exp(logc), pos_ext, sign_ext, t_ext, H)

    # Run 55 cycle 3 (flow-verifier MINOR): cycle 2 picked its poisoned column with
    # argmax(|pos| > 0), which takes whichever column comes first and therefore did not
    # EXPLICITLY cover the delisting-while-SHORT path. Both signs are now exercised by name.
    held_at_2 = pos_ext[2]
    long_cols = np.flatnonzero(held_at_2 > 0)
    short_cols = np.flatnonzero(held_at_2 < 0)
    assert long_cols.size and short_cols.size, \
        "the planted panel must hold BOTH a long and a short at t_ext[2] for this test to " \
        "cover both delisting paths; it holds long=%d short=%d" % (long_cols.size, short_cols.size)

    hit = 0
    for side, col in (("long", int(long_cols[0])), ("short", int(short_cols[0]))):
        for bad_value in (np.nan, 0.0, -1.0):
            price = np.exp(logc)
            row = int(t_ext[3])
            price[row, col] = bad_value
            try:
                gh.hold_shares_walk(price, pos_ext, sign_ext, t_ext, H)
            except ValueError as e:
                assert "hold_shares_walk" in str(e)
                hit += 1
            else:
                raise AssertionError(
                    f"hold_shares_walk absorbed a HELD {side} asset priced {bad_value!r} instead "
                    f"of raising -- that is the exact failure mode the 920 fail-loud contract "
                    f"exists to remove.")
    assert hit == 6
    return f"PASS ({hit}/6 bad prices refused, long AND short legs)"


# =============================================================================
# BLOCKER-E2A -- the gross-renormalised comparator (run 55 cycle 3)
# =============================================================================

def test_gross_renormalisation_exact():
    """BLOCKER-E2A. Cycle 2 declared the leverage-vs-carry separation "a new arm, not a fix" and
    left it undone. Isolating the leverage COMPONENT needs no new arm, and this pins both halves
    of the proxy the module ships:

      (a) EXACTNESS -- per-decision simple return is LINEAR in the weight vector, so the
          rescaled series is exactly ret / Sum|held_w|, recomputed here from the walk's own
          returned arrays and required to be BIT-IDENTICAL to what score_cell reports.
      (b) NON-VACUITY -- a rescaling that were a mere constant would leave Sharpe untouched
          (asserted directly on bt._sharpe), so any movement must come from Sum|w| VARYING with
          returns. It moves, and it moves enough to reverse the sign of this cell's advantage
          over the same-cell target-weight rule.

    SCOPE, corrected in the closing pass: `ret/G` rescales the REALISED RETURN SERIES. It is NOT
    a book re-run under a gross-1 constraint -- the holdings carried into the next decision are
    unrescaled -- so (b)'s sign reversal is a property of THIS construction. A genuinely
    gross_cap=1.0-constrained book (built by a cycle-3 lens, not rebuilt here) reverses the other
    way, and neither construction's paired CI clears zero. This test is a DRIFT GUARD on the
    shipped proxy, not a verdict on the accounting mode.
    """
    if not FIXTURE_PRESENT:
        return "SKIPPED (fixtures/trading_fixture.db absent)"
    p = _panel()
    H, theta = 1, 0.0
    hs = gh.score_cell(p["logc"], p["ts_d"], p["syms"], p["rv"], p["pres"], p["w"],
                       p["S_full"], p["warm"], p["r_lo"], p["r_hi"], H, theta, "hold-shares",
                       return_series=True)
    tw = gh.score_cell(p["logc"], p["ts_d"], p["syms"], p["rv"], p["pres"], p["w"],
                       p["S_full"], p["warm"], p["r_lo"], p["r_hi"], H, theta, "target-weight")
    s = hs.pop("_series")
    G = np.sum(np.abs(s["held_w"]), axis=1)
    assert np.all(G > 0.0), "a decision holding nothing would make the rescaling undefined"
    # (a) exact
    assert np.array_equal(s["renorm"], s["gross"] / G), \
        "the renormalised series must be EXACTLY ret / Sum|held_w|, element-wise"
    period = H * tconfig.K
    rn = hs["renormalised_to_gross1"]
    assert abs(bt._sharpe(s["renorm"], period) - rn["gross"]) < 1e-12
    # (b) non-vacuity: a CONSTANT rescale is a Sharpe no-op, so the move is G's variation
    assert abs(bt._sharpe(s["gross"] * 0.37, period) - bt._sharpe(s["gross"], period)) < 1e-12, \
        "bt._sharpe is not scale-invariant -- the non-vacuity argument below does not hold"
    assert rn["gross"] < hs["gross"] - 0.2, \
        f"renormalisation must MOVE the Sharpe materially; got {rn['gross']} vs {hs['gross']}"
    # the finding itself: a large raw advantage that renormalisation erases and reverses
    raw_adv = hs["gross"] - tw["gross"]
    rn_adv = rn["gross"] - tw["gross"]
    assert raw_adv > 0.2, f"expected a large RAW hold-shares advantage here; got {raw_adv}"
    # SCOPE NOTE (closing pass). This asserts a property of the PROXY ret/G at THIS cell, and
    # nothing more. It is NOT evidence that "the advantage does not survive de-levering" as a
    # verdict: the sign of the residue is CONSTRUCTION-DEPENDENT (a cycle-3 lens's genuinely
    # gross-1-constrained book gives +0.0039..+0.0571, positive in all 16 cells), and neither
    # construction's paired CI clears zero. The leverage question is UNRESOLVED; the assertion
    # stays because it pins the proxy the module actually ships against silent drift.
    assert rn_adv < 0.0, \
        f"the proxy's residue at this cell must stay negative (drift guard on ret/G, NOT a " \
        f"verdict on the mode); got raw {raw_adv:+.4f} -> renorm {rn_adv:+.4f}"
    return f"PASS (raw advantage {raw_adv:+.4f} -> renormalised {rn_adv:+.4f})"


# =============================================================================
# MINOR-E2D -- uplift_ci's paired branch must not mis-align
# =============================================================================

def test_uplift_ci_pairing_alignment():
    """MINOR-E2D. The PAIRED bootstrap applies ONE set of block indices to both filtered series,
    which is only meaningful if they occupy the same positions after filtering. Cycle 2 keyed
    pairing on POST-filter sizes alone. Planted here: two series that drop non-finite entries at
    DIFFERENT positions but in EQUAL COUNTS -- the exact input cycle 2 would have accepted (the
    equal-count precondition is asserted, so this is a proof, not an assertion) -- must now
    RAISE. Aligned drops must still pair normally."""
    rng = np.random.RandomState(0)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    a[3] = np.nan
    b[7] = np.nan
    assert int(np.isfinite(a).sum()) == int(np.isfinite(b).sum()), \
        "the plant must have EQUAL finite counts, or it does not reproduce the cycle-2 hole"
    try:
        gh.uplift_ci(a, b, 1, 1, 1440, 1440, 50, 7)
    except ValueError as e:
        assert "paired" in str(e).lower(), str(e)
    else:
        raise AssertionError(
            "uplift_ci mis-aligned a PAIRED bootstrap: the two series drop non-finite entries at "
            "different positions, so one series' block indices were applied to the other.")
    a2, b2 = a.copy(), b.copy()
    a2[7] = np.nan
    b2[3] = np.nan          # now the masks agree positionally
    u = gh.uplift_ci(a2, b2, 1, 1, 1440, 1440, 50, 7)
    assert u is not None and u["paired"], "aligned series must still pair"
    u_cross = gh.uplift_ci(a2, b2[:150], 2, 1, 2880, 1440, 50, 7)
    assert u_cross is not None and not u_cross["paired"], "cross-H must not claim pairing"
    return "PASS (misaligned refused, aligned paired, cross-H unpaired)"


# =============================================================================
# MAJOR-C2A -- the drift gate must reach PROSE, not just AUTOGEN blocks
# =============================================================================

# The exact mutation a cycle-2 judge ran BY HAND to show `--check` was vacuous outside the
# markers. It sits in prose, outside every AUTOGEN block.
_PROSE_POISON_FROM = "within rounding (0.078 here)"
_PROSE_POISON_TO = "within rounding (9.9999 WRONG NUMBER PLANTED here)"


def test_prose_number_gate():
    """MAJOR-C2A fix (run 55 cycle 3). Cycle 2's `--check` diffed only the bytes INSIDE the
    AUTOGEN markers, so prose appeared identically on both sides and could never trigger a diff;
    a judge demonstrated it by planting a wrong number in prose and getting exit 0.

    This test does three things, in order of what they are worth:
      1. proves the CYCLE-2 comparison ACCEPTS the judge's exact mutation (so the test is not
         one that would already have passed against the holed implementation -- the same standard
         cycle 2's G5 poison had to meet);
      2. proves the CYCLE-3 gate REJECTS it;
      3. proves the whitelist is not an escape hatch -- every declared entry must be NEEDED by
         the current document, so it cannot be padded to manufacture a pass.
    """
    if not Path(rt.RESULT_JSON).exists():
        return "SKIPPED (result_919a.json absent -- run grid_harness.py first)"
    import json
    rep = json.loads(Path(rt.RESULT_JSON).read_text())
    cur = Path(rt.RESULTS_MD).read_text()

    clean = rt.check_prose_numbers(cur, rep)
    assert clean == [], (
        "RESULTS.md prose quotes numbers that appear nowhere in result_919a.json and are not "
        f"whitelisted: {clean}")

    assert _PROSE_POISON_FROM in cur, (
        f"the poison site {_PROSE_POISON_FROM!r} is gone from RESULTS.md -- re-point this test "
        f"at another prose number rather than deleting the check")
    mutated = cur.replace(_PROSE_POISON_FROM, _PROSE_POISON_TO, 1)
    assert mutated != cur

    # 1. the cycle-2 gate is BLIND to it -- this is what makes the new test non-vacuous
    assert rt.render(mutated, rep) == mutated, (
        "the AUTOGEN-only comparison unexpectedly reacted to a PROSE edit; if that is now true "
        "this test no longer demonstrates the hole it was written for")
    # 2. the cycle-3 gate catches it
    bad = rt.check_prose_numbers(mutated, rep)
    assert any(tok.lstrip("+-").startswith("9.9999") for tok, _ in bad), (
        "PROSE gate is VACUOUS: the judge's demonstrated poison was accepted. Found: %r" % bad)

    # 3. the whitelist cannot be padded: with it emptied, every declared entry must be one the
    #    document actually needs.
    needed = set()
    for tok, _ctx in rt.check_prose_numbers(cur, rep, whitelist={}):
        needed.add(abs(float(tok.rstrip().rstrip("%").strip().replace(",", ""))))
    dead = [v for v in rt.PROSE_NUMBER_WHITELIST if not any(abs(v - n) < 1e-9 for n in needed)]
    assert not dead, (
        f"PROSE_NUMBER_WHITELIST contains {len(dead)} entry/entries the document does not need "
        f"({dead}). A whitelist that is not minimal is an escape hatch: remove them.")
    assert len(rt.PROSE_NUMBER_WHITELIST) <= 30, \
        "the whitelist has grown past the point where design (A) is a gate rather than a bypass"
    return (f"PASS (cycle-2 gate blind to the poison, cycle-3 gate catches it; "
            f"{len(rt.PROSE_NUMBER_WHITELIST)} whitelist entries, none dead)")


# =============================================================================
# MINOR-C3 -- the borrow-indexing equivalence a code comment used to claim
# =============================================================================

def test_borrow_indexing_equivalence():
    """MINOR-C3 fix (run 55 cycle 2). grid_harness passes `borrow_pos` (an already-decision-only
    (n,A) array) with a synthetic 0..n-1 index to bt.borrow_cost_series, whereas the shipped 916
    callers pass a full (T,A) array indexed by `t`. Cycle 1's code comment claimed this
    equivalence was "confirmed in test_919_grid.py (G1)" -- grep found no such test. Here it is:
    the two calls must agree EXACTLY, at every H and every borrow scenario."""
    logc, rv, pres, w, S_full, warm = _planted_share_panel()
    syms = [f"SYM{i}" for i in range(logc.shape[1])]
    syms[0] = "XBT"          # exercise the BTC-vs-ALT rate split, not just one rate
    T, A = logc.shape
    checks = 0
    for H in (1, 3):
        t = np.arange(warm, T - H - 1, H)
        t_ext = np.concatenate([[t[0] - H], t])
        pos_ext, _sign = gh.hysteresis_build(rv, pres, w, S_full, t_ext, 0.25)
        pos = pos_ext[1:]
        pos_full = gh._scatter_full(T, A, t_ext, pos_ext)
        for sc in gh.BORROW_SCENARIOS:
            if sc == "none":
                continue
            a = bt.borrow_cost_series(pos, syms, np.arange(t.size), H, rate_scenario=sc)
            b = bt.borrow_cost_series(pos_full, syms, t, H, rate_scenario=sc)
            assert np.array_equal(a, b), \
                f"borrow indexing NOT equivalent at H={H}, scenario={sc}: max abs " \
                f"{float(np.max(np.abs(a - b)))}"
            assert np.all(a >= 0.0), "borrow cost must be >= 0 pointwise (D81(c) direction)"
            checks += 1
    assert checks == 4
    return f"PASS ({checks} index-form comparisons exact)"


# =============================================================================
# G5 -- the analytic anchor, with the flip comparand (MINOR-E3/O2)
# =============================================================================

def test_g5_anchor_nonvacuous():
    """G5 applied to section 3's anchor check: the corrected model must be within a band of the
    MEASURED turn/day at every H in {1,2,3,5}, AND the same band must REJECT the task file's
    as-written (uncorrected) formula -- a band wide enough to pass both is vacuous."""
    if not FIXTURE_PRESENT:
        return "SKIPPED (fixtures/trading_fixture.db absent)"
    p = _panel()
    BAND_LO, BAND_HI = 0.85, 0.98   # measured ratio is ~0.92-0.95; chosen to fail the 2x-off
                                    # uncorrected formula at H=1 and H=3/5.
    n_uncorrected_fail = 0
    for H in gh.H_GRID:
        cell = gh.score_cell(p["logc"], p["ts_d"], p["syms"], p["rv"], p["pres"], p["w"],
                             p["S_full"], p["warm"], p["r_lo"], p["r_hi"], H, 0.0,
                             "target-weight")
        model = gh.analytic_turnover_per_day_corrected(H)
        ratio = cell["turn_per_day"] / model
        assert BAND_LO <= ratio <= BAND_HI, \
            f"H={H}: corrected-model ratio {ratio:.4f} outside [{BAND_LO},{BAND_HI}]"
        # MINOR-E3/O2: the model describes FLIPS only, so also check the flip comparand. It is
        # further below the model than total turn is, and it is NOT flat in H -- both facts are
        # asserted so a future change cannot quietly restore the wrong reading.
        flip_ratio = (cell["decomposition"]["flip"] / H) / model
        assert flip_ratio < ratio, \
            f"H={H}: flip-only ratio must be BELOW the total-turn ratio (flip <= turn by " \
            f"construction); got {flip_ratio:.4f} vs {ratio:.4f}"
        assert 0.85 <= flip_ratio <= 0.98, f"H={H}: flip-only ratio {flip_ratio:.4f} out of band"
        if not (BAND_LO <= cell["turn_per_day"] / gh.analytic_turnover_per_day_uncorrected(H)
                <= BAND_HI):
            n_uncorrected_fail += 1
    assert n_uncorrected_fail >= 3, \
        f"G5 non-vacuity: the band must reject the uncorrected formula at >=3 of 4 H values " \
        f"(it is only coincidentally close at H=2); got {n_uncorrected_fail}/4 rejections"
    # the flip-only ratio DECLINES from H=1 to H=3 -- the estimator lens's finding, pinned.
    p_ = _panel()
    fr = {}
    for H in gh.H_GRID:
        c = gh.score_cell(p_["logc"], p_["ts_d"], p_["syms"], p_["rv"], p_["pres"], p_["w"],
                          p_["S_full"], p_["warm"], p_["r_lo"], p_["r_hi"], H, 0.0,
                          "target-weight")
        fr[H] = (c["decomposition"]["flip"] / H) / gh.analytic_turnover_per_day_corrected(H)
    assert fr[1] > fr[2] > fr[3], \
        f"the flip-only ratio must DECLINE H=1->3 (measured 0.933/0.919/0.898); got {fr}"
    return "PASS"


def test_decomposition_closes():
    """The flip/renorm/entry_exit partition must sum EXACTLY to the target-weight charged
    turnover, and adding drift must give the drift-inclusive charged turnover exactly."""
    if not FIXTURE_PRESENT:
        return "SKIPPED (fixtures/trading_fixture.db absent)"
    p = _panel()
    for H in gh.H_GRID:
        for theta in gh.THETA_GRID:
            tw = gh.score_cell(p["logc"], p["ts_d"], p["syms"], p["rv"], p["pres"], p["w"],
                               p["S_full"], p["warm"], p["r_lo"], p["r_hi"], H, theta,
                               "target-weight")
            di = gh.score_cell(p["logc"], p["ts_d"], p["syms"], p["rv"], p["pres"], p["w"],
                               p["S_full"], p["warm"], p["r_lo"], p["r_hi"], H, theta,
                               "drift-inclusive")
            d = tw["decomposition"]
            assert d["target_weight_turn_closes"], (H, theta)
            assert abs((d["flip"] + d["renorm"] + d["entry_exit"]) - tw["turn"]) < 1e-12, (H, theta)
            assert abs((d["flip"] + d["renorm"] + d["entry_exit"] + d["drift"]) - di["turn"]) < 1e-12, \
                (H, theta, "drift-inclusive closure")
    return "PASS"


# =============================================================================
# BLOCKER-F1 -- RESULTS.md's tables cannot drift from result_919a.json
# =============================================================================

def test_results_tables_match_json():
    """BLOCKER-F1 fix (run 55 cycle 2). Cycle 1 hand-wrote the 48-cell table and 26 of 48 rows
    disagreed with result_919a.json by up to +102%. Every table in RESULTS.md now lives inside
    AUTOGEN markers and is emitted by render_tables.py from the json; this asserts the shipped
    file is byte-identical to a fresh render.

    Non-vacuity: a one-character edit to a rendered table must FAIL it -- proven inline, not
    asserted, by mutating the text in memory and confirming the comparison rejects it."""
    if not Path(rt.RESULT_JSON).exists():
        return "SKIPPED (result_919a.json absent -- run grid_harness.py first)"
    import json
    rep = json.loads(Path(rt.RESULT_JSON).read_text())
    cur = Path(rt.RESULTS_MD).read_text()
    fresh = rt.render(cur, rep)
    assert cur == fresh, (
        "RESULTS.md's AUTOGEN tables are STALE vs result_919a.json. Run "
        "`python render_tables.py`. NEVER hand-edit a table between AUTOGEN markers -- that is "
        "the cycle-1 BLOCKER-F1 defect.")
    n_blocks = len(rt.blocks(rep))
    assert n_blocks >= 15, f"expected the full block set, got {n_blocks}"

    # G5 non-vacuity for THIS gate: corrupt one rendered digit and confirm rejection.
    marker = "<!-- AUTOGEN:grid_primary BEGIN -->\n"
    i = cur.index(marker) + len(marker)
    j = cur.index("\n", cur.index("| 1 | 0.0 | target-weight", i))
    mutated = cur[:j].replace("+1.0339", "+9.9999", 1) + cur[j:]
    assert mutated != cur, "the mutation did not apply -- the non-vacuity check is itself vacuous"
    assert rt.render(mutated, rep) != mutated, \
        "table-drift gate is VACUOUS: a corrupted number in a rendered table was accepted"
    return f"PASS ({n_blocks} AUTOGEN blocks match the json; mutation correctly rejected)"


def test_hour_of_day_mechanism():
    """Section 7: day_boundary_utc=0 reproduces ohlcv_daily's own agg1m rows on the 11-asset
    fixture slice (see hour_of_day.verify_hour0_matches_ohlcv_daily's own docstring for the
    disclosed trailing-snapshot-staleness exception)."""
    if not FIXTURE_PRESENT:
        return "SKIPPED (fixtures/trading_fixture.db absent)"
    import hour_of_day as hod
    r = hod.verify_hour0_matches_ohlcv_daily()
    assert r["n_compared"] > 100, r
    assert r["max_abs_close_diff"] == 0.0, r
    return f"PASS ({r})"


TESTS = [
    ("G1_noop_bit_identical_elementwise", test_g1_noop_bit_identical),
    ("G2_no_trader_edits", test_g2_no_trader_edits),
    ("G3_theta_path_dependence", test_g3_theta_path_dependence),
    ("G4_leak_free", test_g4_leak_free),
    ("G4_hold_shares_leak_free", test_g4_hold_shares_leak_free),
    ("G5_anchor_nonvacuous", test_g5_anchor_nonvacuous),
    ("G5_hold_shares_poison", test_g5_hold_shares_poison),
    ("hold_shares_return_identity", test_hold_shares_return_identity),
    ("hold_shares_fail_loud", test_hold_shares_fail_loud),
    ("borrow_indexing_equivalence", test_borrow_indexing_equivalence),
    ("decomposition_closes", test_decomposition_closes),
    ("results_tables_match_json", test_results_tables_match_json),
    ("prose_number_gate", test_prose_number_gate),
    ("gross_renormalisation_exact", test_gross_renormalisation_exact),
    ("uplift_ci_pairing_alignment", test_uplift_ci_pairing_alignment),
    ("hour_of_day_mechanism", test_hour_of_day_mechanism),
]


def main():
    n_pass = n_fail = n_skip = 0
    for name, fn in TESTS:
        try:
            result = fn()
        except AssertionError as e:
            print(f"[{name}] FAIL: {e}")
            n_fail += 1
            continue
        except Exception as e:
            print(f"[{name}] ERROR: {type(e).__name__}: {e}")
            n_fail += 1
            continue
        if isinstance(result, str) and result.startswith("SKIPPED"):
            print(f"[{name}] {result}")
            n_skip += 1
        else:
            print(f"[{name}] {result}")
            n_pass += 1
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped (of {len(TESTS)})")
    print(f"919A_SUITE_EXIT: {0 if n_fail == 0 else 1}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

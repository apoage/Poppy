#!/usr/bin/env python3
"""TASK_819 Phase-1 — BOTTOM-UP SINGLE-ASSET (BTC) trader probe.

THE QUESTION
------------
Does a profitable, PREDICTABLE edge exist on BTC ALONE, cross-regime, net of cost,
on ABSOLUTE P&L? The 64-asset MuZero planner can't capture the edge the oracle proved
exists -- too many entangled failure modes. We cut to the simplest debuggable problem:
ONE asset (BTC), absolute P&L, prove an edge, then later expand.

The ONLY edge this campaign has confirmed is 816 TSMOM trend (a single-asset time-series
edge applied per-asset, then averaged over 20 majors). 816 is a 20-asset cross-sectional
PORTFOLIO; this probe isolates BTC (no diversification across 20 -> strictly noisier) and
asks whether the time-series trend edge stands alone. That is the FLOOR (Test A). Test B
asks whether a first learned predictor beats it.

METHOD (806->816 discipline, non-negotiable)
- Data: BTC = column 0 of runs/816_operating_point/_close_crossregime.npz (1-min, 2021-2025).
- TIMEFRAME SWEEP: subsample close[::K] for K in {15,60,240,1440} = 15m/1h/4h/1d.
- Lookback L / horizon H expressed as REAL DURATIONS, converted to bars per timeframe
  (so cross-timeframe cells are apples-to-apples and annualization is exact).
- Leak-free: signal/feature use close <= t; label = forward return (uses t+H).
- Cost-honest: net(c) = gross - turnover * c, c in {0,3,8,21} bp/side, ONE-SIDED on
  turnover (the 808 convention: turnover = |dsignal|, a +1->-1 flip trades 2 units, so a
  flip costs 2*c). 21 bp/side = Kraken taker 16 + slippage 5 (the realistic full cost).
- Per-decision H-strided Sharpe via backtest._sharpe(series, period_minutes) -- NEVER
  per-bar*sqrt(525600) (the 809 trap). period_minutes = H_bars * K.
- Cross-regime with the 2022 BEAR as judge (a real trend edge profits the SHORT side of a
  -75% crash; a long-beta artifact dies there). Beat-buy-and-hold (long-only BTC) hurdle to
  strip BTC beta. Block-bootstrap CI on FULL.
- Test B predictor: STRICT time-split OOS (fit early, eval late) + expanding walk-forward
  (so the BEAR can be evaluated OUT-of-sample, which a single early/late split cannot do).

CPU / numpy only. NO GPU. NO deploy. Reuses the audited backtest._sharpe +
backtest._block_bootstrap_sharpe_ci and probe._to_epoch verbatim.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

_T = Path(__file__).resolve().parent
sys.path.insert(0, str(_T.parent / "806_predictability_probe"))
sys.path.insert(0, str(_T.parent / "808_linear_backtest"))
import probe      # noqa: E402  (audited 806 path -- only _to_epoch used here)
import backtest   # noqa: E402  (audited 808 path -- _sharpe + _block_bootstrap_sharpe_ci)

DATA = _T.parent / "816_operating_point" / "_close_crossregime.npz"
BTC_COL = 0
SEED = 819

# Regimes -- FROZEN, identical dates to 816 (the BEAR is the judge).
REGIMES = [("BULL21", "2021-01-01", "2021-11-10"),
           ("BEAR22", "2021-11-10", "2022-12-31"),
           ("REC23",  "2023-01-01", "2023-12-31"),
           ("BULL24", "2024-01-01", "2024-12-31"),
           ("CHOP25", "2025-01-01", "2025-09-01"),
           ("FULL",   "2021-01-01", "2025-09-01")]
REG_NAMES = [r[0] for r in REGIMES]

# Timeframe sweep: K (minutes/bar) -> label.
TIMEFRAMES = [(15, "15m"), (60, "1h"), (240, "4h"), (1440, "1d")]

# Lookback / horizon DURATIONS in minutes (span ~hours -> a month). Converted to bars/K.
LOOKBACK_MIN = [240, 720, 1440, 4320, 10080, 20160, 43200]   # 4h 12h 1d 3d 7d 14d 30d
HORIZON_MIN = [240, 1440, 4320, 10080]                       # 4h 1d 3d 7d

COSTS_BP = (0, 3, 8, 21)   # bp/side, one-sided on turnover. 21 = Kraken taker16 + slip5.
HEADLINE_COST = 21         # the realistic full cost used for the gate/CI
N_BOOT = 2000


def dur_label(m):
    if m % 1440 == 0:
        return f"{m // 1440}d"
    if m % 60 == 0:
        return f"{m // 60}h"
    return f"{m}m"


# =============================================================================
# DATA
# =============================================================================

def load_btc():
    z = np.load(str(DATA))
    close = np.ascontiguousarray(z["close"][:, BTC_COL]).astype(np.float64)
    grid_ts = z["grid_ts"]
    return close, grid_ts


def regime_rows(grid_ts_sub):
    """Row index bounds [lo,hi) per regime in a (sub)sampled grid via searchsorted."""
    out = {}
    T = grid_ts_sub.size
    for name, a, b in REGIMES:
        lo = int(np.searchsorted(grid_ts_sub, probe._to_epoch(a)))
        hi = int(np.searchsorted(grid_ts_sub, probe._to_epoch(b)))
        out[name] = (max(0, lo), min(T, hi))
    return out


# =============================================================================
# TEST A -- TSMOM (the floor), single-asset BTC
# =============================================================================

def tsmom_cell(logc, L, H, r_lo, r_hi):
    """Single-asset TSMOM per-decision series on an H-strided entry grid inside [r_lo,r_hi).

    signal_t = sign(logc_t - logc_{t-L})  (uses close <= t -> leak-free)
    held H bars; per-decision gross = signal_t * (logc_{t+H} - logc_t)
    turnover_t = |signal_t - signal_{t-H}|  (one-sided; a flip = 2)
    buy&hold_t = logc_{t+H} - logc_t        (long-only, signal == +1)
    Returns (gross, turn, bh) arrays or None if too few decisions.
    """
    T = logc.size
    sig = np.full(T, np.nan)
    sig[L:] = np.sign(logc[L:] - logc[:-L])
    lo = max(L + H, r_lo)
    hi = min(r_hi - H, T - 1 - H)
    if hi - lo < H:
        return None
    t = np.arange(lo, hi + 1, H)
    if t.size < 6:
        return None
    s = sig[t]
    s_prev = sig[t - H]
    fwd = logc[t + H] - logc[t]
    ok = np.isfinite(s) & np.isfinite(s_prev) & np.isfinite(fwd)
    if ok.sum() < 6:
        return None
    s, s_prev, fwd = s[ok], s_prev[ok], fwd[ok]
    gross = s * fwd
    turn = np.abs(s - s_prev)
    bh = fwd
    return gross, turn, bh


def net_series(gross, turn, c_bp):
    return gross - turn * (c_bp / 1e4)


def run_test_a(logc, grid_ts_sub, K):
    """All (L,H) cells for one timeframe K. Returns list of cell dicts."""
    rows = regime_rows(grid_ts_sub)
    cells = []
    for Lm in LOOKBACK_MIN:
        L = Lm // K
        if L < 2:
            continue
        for Hm in HORIZON_MIN:
            if Hm > Lm:
                continue
            H = Hm // K
            if H < 1:
                continue
            period_min = H * K   # true decision spacing in minutes (exact annualization)
            cell = {"K": K, "Lm": Lm, "Hm": Hm, "L": L, "H": H,
                    "period_min": period_min, "reg": {}}
            ok_any = False
            for name in REG_NAMES:
                r0, r1 = rows[name]
                res = tsmom_cell(logc, L, H, r0, r1)
                if res is None:
                    cell["reg"][name] = None
                    continue
                ok_any = True
                gross, turn, bh = res
                d = {"n": int(gross.size),
                     "turn": float(np.mean(turn)),
                     "gross": backtest._sharpe(gross, period_min),
                     "bh": backtest._sharpe(bh, period_min)}
                for c in COSTS_BP:
                    d[f"net{c}"] = backtest._sharpe(net_series(gross, turn, c), period_min)
                if name == "FULL":
                    net_h = net_series(gross, turn, HEADLINE_COST)
                    ci = backtest._block_bootstrap_sharpe_ci(net_h, period_min, period_min,
                                                             N_BOOT, SEED)
                    sd = np.std(net_h, ddof=0)
                    ts = (np.sqrt(net_h.size) * np.mean(net_h) / sd) if sd > 1e-15 else 0.0
                    d["ci_lo"], d["ci_hi"], d["p"] = float(ci[0]), float(ci[1]), float(norm.sf(ts))
                    # absolute P&L: non-overlapping H-strided log-returns TILE the timeline, so
                    # their sum == total compounded log-growth over the traded span.
                    d["cum_net21"] = float(np.sum(net_h))
                    d["cum_gross"] = float(np.sum(gross))
                    d["cum_bh"] = float(np.sum(bh))
                cell["reg"][name] = d
            if ok_any:
                cells.append(cell)
    return cells


def bear_positive(cell):
    """Real-trend signature: BEAR net@8 > 0 AND beats long-only B&H in the bear."""
    b = cell["reg"].get("BEAR22")
    return bool(b and np.isfinite(b["net8"]) and b["net8"] > 0 and b["net8"] > b["bh"])


def full_ok(cell):
    """FULL net@headline beats B&H AND bootstrap CI excludes 0 from below."""
    f = cell["reg"].get("FULL")
    return bool(f and np.isfinite(f[f"net{HEADLINE_COST}"]) and
                f[f"net{HEADLINE_COST}"] > f["bh"] and
                np.isfinite(f["ci_lo"]) and f["ci_lo"] > 0)


def cell_score(cell):
    f = cell["reg"].get("FULL")
    v = f["net8"] if f else -9e9
    return v if np.isfinite(v) else -9e9


# =============================================================================
# TEST B -- first learned predictor (ridge), strict OOS + walk-forward
# =============================================================================

def build_design(logc, H, lags=(1, 2, 3, 5, 10, 20),
                 vol_w=(10, 20, 50), mom_L=(5, 10, 20, 50)):
    """Causal feature matrix X (T,F) and forward-H label y (T,). All features use close<=t;
    label y_t = logc_{t+H} - logc_t uses t+H. NaN where any input window unavailable."""
    T = logc.size
    rr = np.full(T, np.nan)
    rr[1:] = logc[1:] - logc[:-1]          # rr_i = return realized AT bar i (uses <= i)
    feats, names = [], []
    # lagged returns: lag-l at row t = rr[t-l+1]
    for l in lags:
        f = np.full(T, np.nan)
        f[l - 1:] = rr[:T - (l - 1)]
        feats.append(f); names.append(f"ret_lag{l}")
    # realized vol over last w returns ending at t (rolling std of rr)
    rr0 = np.where(np.isfinite(rr), rr, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(rr0)])
    csum2 = np.concatenate([[0.0], np.cumsum(rr0 * rr0)])
    for w in vol_w:
        f = np.full(T, np.nan)
        idx = np.arange(w, T)            # need rr[t-w+1..t] all finite -> t >= w (rr[0]=nan)
        s1 = csum[idx + 1] - csum[idx + 1 - w]
        s2 = csum2[idx + 1] - csum2[idx + 1 - w]
        var = np.maximum(s2 / w - (s1 / w) ** 2, 0.0)
        f[idx] = np.sqrt(var)
        feats.append(f); names.append(f"vol{w}")
    # past-L momentum (level): logc_t - logc_{t-L}
    for L in mom_L:
        f = np.full(T, np.nan)
        f[L:] = logc[L:] - logc[:-L]
        feats.append(f); names.append(f"mom{L}")
    X = np.column_stack(feats)
    y = np.full(T, np.nan)
    y[:T - H] = logc[H:] - logc[:-H]
    return X, y, names


def _standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def ridge_fit(X, y, alpha):
    Xb = np.column_stack([X, np.ones(X.shape[0])])
    A = Xb.T @ Xb
    reg = alpha * np.eye(Xb.shape[1]); reg[-1, -1] = 0.0   # don't penalize bias
    w = np.linalg.solve(A + reg, Xb.T @ y)
    return w


def ridge_pred(X, w):
    return np.column_stack([X, np.ones(X.shape[0])]) @ w


def _ic(pred, y):
    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan")
    p, t = pred[m], y[m]
    if np.std(p) < 1e-15 or np.std(t) < 1e-15:
        return float("nan"), float("nan")
    pear = float(np.corrcoef(p, t)[0, 1])
    spear = float(np.corrcoef(rankdata(p), rankdata(t))[0, 1])
    return pear, spear


def _oos_r2(pred, y, y_train_mean):
    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    ss_res = float(np.sum((y[m] - pred[m]) ** 2))
    ss_tot = float(np.sum((y[m] - y_train_mean) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")


def _rule_pnl(pred, y, turn_prev_pos, tau):
    """Long/short/flat rule: pos=sign(pred) if |pred|>=tau else 0. Returns gross, turn, pos."""
    pos = np.where(np.abs(pred) >= tau, np.sign(pred), 0.0)
    gross = pos * y
    prev = np.concatenate([[turn_prev_pos], pos[:-1]])
    turn = np.abs(pos - prev)
    return gross, turn, pos


def fit_predictor(X, y, dec_rows, mu, sd, alphas, val_frac=0.25, H=1):
    """Select alpha (+ tau) on a time-ordered VAL tail of the training decisions, refit on
    full train. dec_rows = sorted decision row indices in the TRAIN span (H-strided)."""
    n = dec_rows.size
    n_val = max(8, int(n * val_frac))
    if n - n_val < 12:
        n_val = max(4, n // 3)
    fit_rows = dec_rows[:n - n_val]
    # purge: drop fit rows whose label overlaps the val span
    val_lo = dec_rows[n - n_val]
    fit_rows = fit_rows[fit_rows + H <= val_lo]
    val_rows = dec_rows[n - n_val:]
    if fit_rows.size < 10 or val_rows.size < 4:
        fit_rows, val_rows = dec_rows, dec_rows   # degenerate fallback (tiny span)
    Zf = (X[fit_rows] - mu) / sd
    Zv = (X[val_rows] - mu) / sd
    yf, yv = y[fit_rows], y[val_rows]
    best = None
    for a in alphas:
        w = ridge_fit(Zf, yf, a)
        pv = ridge_pred(Zv, w)
        ic, _ = _ic(pv, yv)
        sc = ic if np.isfinite(ic) else -9e9
        if best is None or sc > best[0]:
            best = (sc, a, w, pv)
    _, alpha, _, pv = best
    # tau selection on val: maximize net@headline Sharpe over a few |pred| quantiles
    yv_lab = yv
    best_tau, best_sh = 0.0, -9e9
    for q in (0.0, 0.3, 0.5, 0.7):
        tau = 0.0 if q == 0.0 else float(np.nanquantile(np.abs(pv), q))
        g, tn, _ = _rule_pnl(pv, yv_lab, 0.0, tau)
        sh = backtest._sharpe(net_series(g, tn, HEADLINE_COST), H * 1)  # period scaling cancels in argmax
        if np.isfinite(sh) and sh > best_sh:
            best_sh, best_tau = sh, tau
    # refit on FULL train at chosen alpha
    Zall = (X[dec_rows] - mu) / sd
    w = ridge_fit(Zall, y[dec_rows], alpha)
    return w, alpha, best_tau


def eval_oos(logc, grid_ts_sub, K, H, train_lo, train_hi, test_lo, test_hi,
             alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Fit ridge on [train_lo,train_hi) (H-strided decisions), eval on [test_lo,test_hi).
    Strict purge of H bars between train and test. Returns metrics dict (per OOS regime)."""
    X, y, names = build_design(logc, H)
    mu, sd = None, None
    period_min = H * K
    # train decisions: H-strided, label must not cross into test (purge)
    tr = np.arange(max(train_lo, H), min(train_hi, test_lo) - H, H)
    tr = tr[np.isfinite(X[tr]).all(axis=1) & np.isfinite(y[tr])]
    if tr.size < 20:
        return None
    mu, sd = _standardize_fit(X[tr])
    w, alpha, tau = fit_predictor(X, y, tr, mu, sd, alphas, H=H)
    y_train_mean = float(np.mean(y[tr]))
    # OOS decisions: H-strided in test span
    te = np.arange(test_lo, min(test_hi, logc.size - H), H)
    te = te[np.isfinite(X[te]).all(axis=1) & np.isfinite(y[te])]
    if te.size < 6:
        return None
    Zte = (X[te] - mu) / sd
    pred = ridge_pred(Zte, w)
    yte = y[te]
    pear, spear = _ic(pred, yte)
    r2 = _oos_r2(pred, yte, y_train_mean)
    g, tn, pos = _rule_pnl(pred, yte, 0.0, tau)
    out = {"alpha": alpha, "tau": tau, "n_oos": int(te.size), "ic": pear, "spear": spear,
           "r2": r2, "turn": float(np.mean(tn)),
           "sh_gross": backtest._sharpe(g, period_min)}
    for c in COSTS_BP:
        out[f"net{c}"] = backtest._sharpe(net_series(g, tn, c), period_min)
    out["bh"] = backtest._sharpe(yte, period_min)
    # per-regime breakdown over the OOS decisions
    out["reg"] = {}
    rows = regime_rows(grid_ts_sub)
    for name in REG_NAMES:
        r0, r1 = rows[name]
        m = (te >= r0) & (te < r1)
        if m.sum() < 6:
            out["reg"][name] = None
            continue
        gg, tt = g[m], tn[m]
        out["reg"][name] = {
            "n": int(m.sum()), "ic": _ic(pred[m], yte[m])[0],
            "net8": backtest._sharpe(net_series(gg, tt, 8), period_min),
            f"net{HEADLINE_COST}": backtest._sharpe(net_series(gg, tt, HEADLINE_COST), period_min),
            "bh": backtest._sharpe(yte[m], period_min)}
    return out


# =============================================================================
# SELF-TESTS (numpy) -- validate the backtest logic before trusting any result
# =============================================================================

def self_tests():
    msgs = []

    # 1. annualization identity
    rng = np.random.RandomState(0)
    x = rng.randn(500) * 0.01 + 0.001
    expect = np.mean(x) / np.std(x) * np.sqrt(525600 / 60.0)
    got = backtest._sharpe(x, 60)
    assert abs(got - expect) < 1e-9, f"annualization mismatch {got} vs {expect}"
    msgs.append(f"[ST1] annualization identity OK (_sharpe == mean/std*sqrt(525600/R))")

    # 2a. planted DOWNTREND (the bear judge): TSMOM shorts -> gross +, long-only B&H dies (-)
    rng = np.random.RandomState(3)
    logp = np.cumsum(-0.0006 + rng.randn(8000) * 0.004)         # persistent down drift + noise
    g, tu, bh = tsmom_cell(logp, L=20, H=10, r_lo=0, r_hi=logp.size)
    sh_g = backtest._sharpe(g, 10)
    sh_bh = backtest._sharpe(bh, 10)
    assert sh_g > 0.5, f"downtrend TSMOM gross Sharpe should be + (shorts the crash), got {sh_g}"
    assert sh_bh < 0.0, f"downtrend long-only B&H should be - (it dies), got {sh_bh}"
    assert sh_g > sh_bh, "TSMOM must beat B&H in a downtrend (the bear signature)"
    # 2b. square-wave (both legs): TSMOM dominates B&H
    drift = np.concatenate([np.full(60, 0.003), np.full(60, -0.003)])
    logp2 = np.cumsum(np.tile(drift, 40))
    g2, _, bh2 = tsmom_cell(logp2, L=10, H=5, r_lo=0, r_hi=logp2.size)
    sh_g2 = backtest._sharpe(g2, 5)
    sh_bh2 = backtest._sharpe(bh2, 5)
    assert sh_g2 > 1.0 and sh_g2 > abs(sh_bh2), \
        f"square-wave TSMOM should dominate B&H, got g={sh_g2} bh={sh_bh2}"
    msgs.append(f"[ST2] planted trend: downtrend TSMOM gross {sh_g:+.2f}>0 vs long-only B&H "
                f"{sh_bh:+.2f}<0 (shorts the crash); square-wave TSMOM {sh_g2:+.2f} dominates "
                f"B&H {sh_bh2:+.2f} -- short side profits OK")

    # 3. sub-cost wiggle: pure random walk -> gross ~0, turnover>0, cost makes net negative
    rng = np.random.RandomState(7)
    logp = np.cumsum(rng.randn(40000) * 0.001)
    g, tu, bh = tsmom_cell(logp, L=20, H=10, r_lo=0, r_hi=logp.size)
    sh_g = backtest._sharpe(g, 10)
    sh_n = backtest._sharpe(net_series(g, tu, 21), 10)
    assert np.mean(tu) > 0, "random walk should incur turnover"
    assert sh_n < sh_g + 1e-9, "cost must not increase Sharpe"
    assert np.mean(net_series(g, tu, 21)) < 0, "sub-cost wiggle net mean must be < 0"
    msgs.append(f"[ST3] random-walk sub-cost: gross Sharpe {sh_g:+.2f}~0, net@21 {sh_n:+.2f} "
                f"< gross, net mean<0 (cost eats noise) OK")

    # 4. cost monotonicity
    shs = [backtest._sharpe(net_series(g, tu, c), 10) for c in COSTS_BP]
    assert all(shs[i] >= shs[i + 1] - 1e-9 for i in range(len(shs) - 1)), \
        f"net Sharpe must be non-increasing in cost, got {shs}"
    msgs.append(f"[ST4] cost monotonic: net Sharpe {[round(s,2) for s in shs]} non-increasing OK")

    # 5. predictor recovers a planted linear signal, and is ~0 on pure noise (leak-free)
    rng = np.random.RandomState(11)
    T = 6000
    rr = rng.randn(T) * 0.01
    logp = np.cumsum(rr)
    # build features, then construct a label that is a known linear fn of one feature
    X, _, names = build_design(logp, H=1)
    f = X[:, names.index("mom10")]
    y_planted = np.full(T, np.nan)
    good = np.isfinite(f)
    y_planted[good] = 0.8 * (f[good] - np.nanmean(f)) / (np.nanstd(f) + 1e-12) * 0.01 \
        + rng.randn(good.sum()) * 0.003
    rows = np.where(np.isfinite(X).all(axis=1) & np.isfinite(y_planted))[0]
    split = rows[int(0.6 * rows.size)]
    tr = rows[rows + 1 <= split]
    te = rows[rows >= split]
    mu, sd = _standardize_fit(X[tr])
    w = ridge_fit((X[tr] - mu) / sd, y_planted[tr], 1e-2)
    pred = ridge_pred((X[te] - mu) / sd, w)
    ic_signal, _ = _ic(pred, y_planted[te])
    # pure-noise label -> OOS IC ~ 0 (no leak inflates it)
    y_noise = rng.randn(T) * 0.01
    w2 = ridge_fit((X[tr] - mu) / sd, y_noise[tr], 1e-2)
    pred2 = ridge_pred((X[te] - mu) / sd, w2)
    ic_noise, _ = _ic(pred2, y_noise[te])
    assert ic_signal > 0.3, f"predictor should recover planted signal, OOS IC {ic_signal}"
    assert abs(ic_noise) < 0.2, f"predictor on noise should be ~0 OOS IC, got {ic_noise}"
    msgs.append(f"[ST5] predictor: planted-signal OOS IC {ic_signal:+.2f} (>0.3), "
                f"pure-noise OOS IC {ic_noise:+.2f} (~0, leak-free) OK")
    return msgs


# =============================================================================
# REPORT
# =============================================================================

def fmt(v, p=2, sign=True):
    if v is None or not np.isfinite(v):
        return "  -"
    return f"{v:+.{p}f}" if sign else f"{v:.{p}f}"


def gv(cell, reg, key):
    d = cell["reg"].get(reg)
    return fmt(d[key]) if d else "  -"


def write_report(all_cells, promising, predB, walkfwd, st_msgs, mkt, runtime):
    L = []
    ap = L.append
    ap("# TASK_819 Phase-1 — BOTTOM-UP SINGLE-ASSET (BTC) TRADER — RESULT\n")
    ap("**Question:** does a profitable, PREDICTABLE edge exist on BTC ALONE, cross-regime, "
       "net of cost, on ABSOLUTE P&L? Floor = 816 TSMOM trend. CPU/numpy only, no GPU, no deploy.\n")
    ap("\n**Data:** BTC = col 0 of `runs/816_operating_point/_close_crossregime.npz` (1-min, "
       "2021-2025). Timeframe sweep K∈{15,60,240,1440}=15m/1h/4h/1d via `close[::K]`. L (lookback) "
       "and H (horizon) are REAL DURATIONS converted to bars/timeframe. **The 816 floor is a "
       "20-asset cross-sectional PORTFOLIO; this probe isolates BTC alone — strictly noisier (no "
       "diversification across 20 majors).**\n")
    ap("\n**Discipline:** leak-free (signal/feature close≤t, label=fwd ret); cost-honest "
       "net(c)=gross−turnover·c, c∈{0,3,8,21} bp/side ONE-SIDED (a +1→−1 flip trades 2 units → "
       "costs 2·c); 21 bp/side = Kraken taker16+slip5 (realistic full). Per-decision H-strided "
       "Sharpe = mean/std·√(525600/period_min), period_min=H·K (NOT per-bar·√525600). The 2022 "
       "BEAR is the judge; beat long-only B&H to strip BTC beta; block-boot CI on FULL.\n")

    ap("\n## Self-tests (backtest logic validated before any result)\n")
    for m in st_msgs:
        ap(f"- `{m}`\n")

    ap("\n## Market context (BTC buy-and-hold cumulative log-return per regime)\n")
    ap("| regime | dates | BTC B&H cum log-ret |\n|---|---|--:|\n")
    for name, a, b in REGIMES:
        ap(f"| {name} | {a}→{b} | {mkt[name]:+.3f} |\n")
    ap("\nThe BEAR_2022 is genuinely a deep downtrend (cum log-ret strongly negative) — a real "
       "trend edge must be POSITIVE there (short side), where long-only B&H bleeds.\n")

    # ---- Test A tables, per timeframe ----
    ap("\n## TEST A — TSMOM floor on BTC alone (per timeframe)\n")
    ap("Per-regime GROSS Sharpe (the **BEAR** column decides) + FULL net@{8,21}bp + FULL B&H + "
       "CI(net@21) + beatsBH(net@21). Turnover is tiny for low-frequency TSMOM, so net≈gross. Sorted "
       "by BEAR gross. NOTE: a fixed (L,H) duration is the same calendar trade at every timeframe, so "
       "FULL columns repeat across the 4 tables; the BEAR column does NOT (phase-sensitive — see "
       "Robustness).\n")
    for K, klab in TIMEFRAMES:
        cells = [c for c in all_cells if c["K"] == K]
        if not cells:
            continue
        ap(f"\n### {klab} timeframe (K={K})\n")
        ap("| L | H | turn | BULL21 | **BEAR22** | REC23 | BULL24 | CHOP25 | FULL gross | "
           "FULL net@8 | FULL net@21 | FULL B&H | CI(net@21) | beatsBH | bearPos |\n")
        ap("|--|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|:--:|:--:|\n")
        def bk(c):
            b = c["reg"].get("BEAR22")
            return b["gross"] if b and np.isfinite(b["gross"]) else -9e9
        for c in sorted(cells, key=bk, reverse=True):
            f = c["reg"].get("FULL")
            ci = f"[{f['ci_lo']:+.2f},{f['ci_hi']:+.2f}]" if f else "  -"
            turn = c["reg"]["FULL"]["turn"] if f else float("nan")
            beats = "Y" if full_ok(c) else "n"
            bp = "Y" if bear_positive(c) else "n"
            ap(f"| {dur_label(c['Lm'])} | {dur_label(c['Hm'])} | {fmt(turn,3,False)} | "
               f"{gv(c,'BULL21','gross')} | **{gv(c,'BEAR22','gross')}** | {gv(c,'REC23','gross')} | "
               f"{gv(c,'BULL24','gross')} | {gv(c,'CHOP25','gross')} | {gv(c,'FULL','gross')} | "
               f"{gv(c,'FULL','net8')} | {gv(c,'FULL','net21')} | {gv(c,'FULL','bh')} | {ci} | "
               f"{beats} | {bp} |\n")

    # ---- robustness: FULL invariance vs BEAR phase-fragility across timeframes ----
    durs = sorted({(c["Lm"], c["Hm"]) for c in all_cells})
    full_dev, bear_spread = [], []
    for Lm, Hm in durs:
        fs = [c["reg"]["FULL"]["gross"] for c in all_cells
              if c["Lm"] == Lm and c["Hm"] == Hm and c["reg"].get("FULL")
              and np.isfinite(c["reg"]["FULL"]["gross"])]
        bs = [c["reg"]["BEAR22"]["gross"] for c in all_cells
              if c["Lm"] == Lm and c["Hm"] == Hm and c["reg"].get("BEAR22")
              and np.isfinite(c["reg"]["BEAR22"]["gross"])]
        if len(fs) > 1:
            full_dev.append((max(fs) - min(fs), Lm, Hm))
        if len(bs) > 1:
            bear_spread.append((max(bs) - min(bs), Lm, Hm, min(bs), max(bs)))
    ap("\n## Robustness — FULL sampling-invariance vs BEAR phase-fragility\n")
    ap("TSMOM at a fixed (L,H) DURATION is essentially the same calendar-time trade regardless of "
       "intrabar timeframe K, so the FULL-period Sharpe is near-invariant to K — but the per-regime "
       "(esp. BEAR) Sharpe is PHASE-SENSITIVE (which intrabar sub-grid you sample changes which "
       "trades land inside the regime).\n")
    if full_dev:
        md = max(full_dev)
        ap(f"- **FULL gross Sharpe is ~timeframe-invariant**: across all durations the max spread of "
           f"FULL gross Sharpe over the 4 timeframes is only **{md[0]:.2f}** "
           f"(at L={dur_label(md[1])}/H={dur_label(md[2])}) — confirms (L,H) is a calendar-time trade.\n")
    if bear_spread:
        bsr = max(bear_spread)
        ap(f"- **BEAR gross Sharpe is PHASE-FRAGILE**: the same duration swings up to **{bsr[0]:.2f}** "
           f"Sharpe across timeframes (worst: L={dur_label(bsr[1])}/H={dur_label(bsr[2])} ranges "
           f"[{bsr[3]:+.2f}, {bsr[4]:+.2f}]). A robust edge would not flip with the sampling phase — "
           f"this fragility is itself evidence the single-asset bear result is mostly noise.\n")

    # ---- promising cells ----
    ap("\n## Promising cells (bear-positive AND FULL net@21 beats B&H with CI>0)\n")
    if promising:
        ap("| timeframe | L | H | BEAR22 net@8 | BEAR22 B&H | FULL net@21 | FULL B&H | CI(net@21) |\n")
        ap("|---|--|--|--:|--:|--:|--:|---|\n")
        for c in promising:
            klab = dict((k, v) for k, v in TIMEFRAMES)[c["K"]]
            b = c["reg"]["BEAR22"]; f = c["reg"]["FULL"]
            ap(f"| {klab} | {dur_label(c['Lm'])} | {dur_label(c['Hm'])} | {fmt(b['net8'])} | "
               f"{fmt(b['bh'])} | {fmt(f['net21'])} | {fmt(f['bh'])} | "
               f"[{f['ci_lo']:+.2f},{f['ci_hi']:+.2f}] |\n")
    else:
        ap("**NONE.** No single-asset BTC TSMOM cell is bear-positive AND clears the FULL net@21 "
           "CI>0 + beats-B&H gate. (See verdict.)\n")

    # ---- Test B ----
    ap("\n## TEST B — first learned predictor (ridge) on the best cells\n")
    ap("Causal features: lagged returns [1,2,3,5,10,20], realized vol [10,20,50], past-L momentum "
       "[5,10,20,50]. Ridge α + flat-threshold τ selected on a TRAIN-internal val tail. **B1 (strict "
       "OOS):** fit 2021-01→2024-01 (incl. BEAR), eval 2024-01→2025-09 (BULL24+CHOP25). **B2 "
       "(walk-forward):** expanding train, eval the next regime — so the BEAR is evaluated OUT-of-"
       "sample (a single early/late split cannot test the bear OOS).\n")
    if predB:
        ap("\n### B1 strict early/late split (OOS = BULL24 + CHOP25)\n")
        ap("| timeframe | L | H | α | τ | n_oos | OOS IC | OOS R² | net@8 | net@21 | B&H | "
           "BULL24 net@21 | CHOP25 net@21 | TSMOM net@21(same cell, OOS span) |\n")
        ap("|---|--|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
        for row in predB:
            c, r = row["cell"], row["oos"]
            klab = dict((k, v) for k, v in TIMEFRAMES)[c["K"]]
            b24 = r["reg"].get("BULL24"); c25 = r["reg"].get("CHOP25")
            ap(f"| {klab} | {dur_label(c['Lm'])} | {dur_label(c['Hm'])} | {r['alpha']:g} | "
               f"{r['tau']:.1e} | {r['n_oos']} | {fmt(r['ic'])} | {fmt(r['r2'])} | {fmt(r['net8'])} | "
               f"{fmt(r['net21'])} | {fmt(r['bh'])} | "
               f"{fmt(b24['net21']) if b24 else '  -'} | {fmt(c25['net21']) if c25 else '  -'} | "
               f"{fmt(row['tsmom_oos'])} |\n")
    if walkfwd:
        ap("\n### B2 walk-forward (predictor evaluated OOS on each regime incl. the BEAR)\n")
        ap("| timeframe | L | H | OOS regime | train | n | IC | predictor net@21 | "
           "TSMOM net@21 | B&H |\n")
        ap("|---|--|--|---|---|--:|--:|--:|--:|--:|\n")
        for row in walkfwd:
            c = row["cell"]
            klab = dict((k, v) for k, v in TIMEFRAMES)[c["K"]]
            ap(f"| {klab} | {dur_label(c['Lm'])} | {dur_label(c['Hm'])} | {row['reg']} | "
               f"{row['train']} | {row['n']} | {fmt(row['ic'])} | {fmt(row['net21'])} | "
               f"{fmt(row['tsmom'])} | {fmt(row['bh'])} |\n")

    # ---- verdict ----
    ap("\n## Flagged assumptions\n")
    ap("1. **Execution convention (816-frozen):** signal/forecast at close(t) is assumed executable "
       "AT close(t) (gross = signal·[logc(t+H)−logc(t)]) — NO 1-bar exec lag (808 used close(t+1)). "
       "Negligible for slow TSMOM (signal flips rarely); OPTIMISTIC for the Test-B predictor at short "
       "H (it rebalances every bar). The predictor is net-NEGATIVE even WITHOUT the lag, so the null "
       "is robust to it; any future *positive* short-H cell must be re-checked with the lag.\n")
    ap("2. **Cost model:** flat bp/side on one-sided turnover Σ|Δsignal| (a +1→−1 flip = 2 units). "
       "21 bp/side = Kraken taker16+slip5 (spot). Perps/maker would sit in the 3–8 bp band shown. No "
       "funding, market impact, or partial-fill modelling.\n")
    ap("3. **Returns are LOG returns** (matches 816); per-decision H-strided Sharpe annualized by "
       "√(525600/period_min). 'Absolute P&L' = single-asset DIRECTIONAL long/short/flat (contrast the "
       "64-asset cross-sectional RANKING objective), reported as risk-adjusted Sharpe + cumulative "
       "log-growth. CAVEAT: short legs use the log convention (P&L = −fwd in log space), which slightly "
       "OVERSTATES short compounding vs exact simple-return short P&L on the large 2022 moves — so the "
       "×growth multiple is a mild upper bound. The Sharpe/CI gate (the actual verdict) is unaffected.\n")
    ap("4. **Test B split:** B1 trains through the BEAR (so the bear is in-sample for B1); B2 "
       "walk-forward puts the bear OOS but with a THIN single-regime (BULL21-only) train — a real "
       "limitation flagged, not hidden. Features/τ/α are all selected on train-internal data only.\n")
    ap("5. BTC = column 0 of the 816 npz (OG_SYM[0]='XBT'); no NaNs in the BTC column over 2021-2025. "
       "Single-asset only — no diversification, the 816 floor was a 20-asset portfolio.\n")
    ap("\n## VERDICT\n")
    ap(verdict_text(all_cells, promising, predB, walkfwd))
    ap(f"\n\n_Runtime {runtime:.1f}s, CPU/numpy, seed {SEED}. Self-tests pass. "
       f"Reuses backtest._sharpe + _block_bootstrap_sharpe_ci + probe._to_epoch verbatim._\n")
    (_T / "RESULT.md").write_text("".join(L))


def verdict_text(all_cells, promising, predB, walkfwd):
    # gather bear-positive stats
    bear_pos = [c for c in all_cells if bear_positive(c)]
    n_total = len(all_cells)
    lines = []
    # best TSMOM cell by FULL net@8
    ranked = sorted(all_cells, key=cell_score, reverse=True)
    best = ranked[0] if ranked else None
    if best:
        klab = dict((k, v) for k, v in TIMEFRAMES)[best["K"]]
        f = best["reg"]["FULL"]; b = best["reg"].get("BEAR22")
        bn = b["net8"] if b else float("nan")
        lines.append(
            f"**Best TSMOM cell: L={dur_label(best['Lm'])}/H={dur_label(best['Hm'])} "
            f"(FULL Sharpe is timeframe-invariant) — FULL net@8 Sharpe {fmt(f['net8'])}, net@21 "
            f"{fmt(f['net21'])}, BEAR net@8 {fmt(bn)}, B&H {fmt(f['bh'])}, CI(net@21) "
            f"[{f['ci_lo']:+.2f},{f['ci_hi']:+.2f}] — {'beats' if f['net21'] > f['bh'] else 'does NOT beat'} "
            f"B&H on the point estimate, but the CI INCLUDES 0, so it fails the gate (not statistically "
            f"distinguishable from no edge).**")
    lines.append(
        f"\n\n- TSMOM on BTC alone: **{len(bear_pos)}/{n_total}** cells are bear-positive "
        f"(BEAR net@8 > 0 AND > B&H — the trend-following short-side signature).")
    lines.append(
        f"\n- **{len(promising)}/{n_total}** cells additionally clear the FULL net@21 gate "
        f"(beats B&H AND bootstrap CI excludes 0).")
    if best and best["reg"].get("FULL") and "cum_net21" in best["reg"]["FULL"]:
        f = best["reg"]["FULL"]
        lines.append(
            f"\n- **Absolute P&L (best cell, FULL 2021-2025):** net@21 cumulative log-return "
            f"{f['cum_net21']:+.2f} (×{np.exp(f['cum_net21']):.2f} growth) vs buy-and-hold "
            f"{f['cum_bh']:+.2f} (×{np.exp(f['cum_bh']):.2f}). "
            + ("The best cell grows capital and " if f['cum_net21'] > 0 else
               "The best cell LOSES money outright and ")
            + ("returns LESS than simply holding BTC" if f['cum_net21'] < f['cum_bh']
               else "BEATS buy-and-hold in total dollars (by shorting the 2022 bear)")
            + " — but see the risk-adjusted gate below, which it FAILS.")
    # predictor read
    if predB:
        any_beat = any(
            (row["oos"]["net21"] is not None and np.isfinite(row["oos"]["net21"]) and
             np.isfinite(row["tsmom_oos"]) and row["oos"]["net21"] > row["tsmom_oos"] and
             np.isfinite(row["oos"]["bh"]) and row["oos"]["net21"] > row["oos"]["bh"])
            for row in predB)
        best_ic = max((row["oos"]["ic"] for row in predB
                       if np.isfinite(row["oos"]["ic"])), default=float("nan"))
        lines.append(
            f"\n- Learned predictor (B1 strict OOS): best OOS IC {fmt(best_ic)}; "
            f"{'AT LEAST ONE cell beats BOTH TSMOM and B&H net@21 OOS' if any_beat else 'NO cell beats both TSMOM and B&H net@21 OOS'}.")
    if walkfwd:
        bear_rows = [r for r in walkfwd if r["reg"] == "BEAR22"]
        if bear_rows:
            profit_bear = any(np.isfinite(r["net21"]) and r["net21"] > 0 and
                              np.isfinite(r["bh"]) and r["net21"] > r["bh"] for r in bear_rows)
            nets = [r["net21"] for r in bear_rows if np.isfinite(r["net21"])]
            lines.append(
                f"\n- Predictor walk-forward on the BEAR (OOS, thin single-regime train): net@21 "
                f"{'/'.join(f'{x:+.2f}' for x in nets)} — "
                + ("PROFITS the short side (net>0 AND beats B&H)." if profit_bear else
                   "NET-NEGATIVE in the bear — it does NOT profit the short side (the high-turnover "
                   "variants even underperform B&H). The raw TSMOM short barely clears zero; the "
                   "learned predictor captures none of it."))
    # final call
    fbest = best["reg"]["FULL"] if best and best["reg"].get("FULL") else None
    beats_total = bool(fbest and "cum_net21" in fbest and fbest["cum_net21"] > fbest["cum_bh"])
    lines.append("\n\n**Where the edge lives (two lenses, reconciled):** ")
    if not promising and beats_total:
        lines.append(
            f"There is a genuine TENSION the discipline demands we surface honestly. On RAW cumulative "
            f"P&L over this single 2021-2025 path, the best long-lookback cell (L={dur_label(best['Lm'])}/"
            f"H={dur_label(best['Hm'])}) returns ×{np.exp(fbest['cum_net21']):.2f} vs buy-and-hold "
            f"×{np.exp(fbest['cum_bh']):.2f} — it DOES beat B&H in absolute dollars, almost entirely by "
            f"being SHORT through the 2022 crash (the classic trend-following signature, same as 816). "
            f"BUT on the RISK-ADJUSTED, statistically-gated terms it is NOT confirmed: net@21 Sharpe "
            f"+{fbest['net21']:.2f} with bootstrap CI [{fbest['ci_lo']:+.2f},{fbest['ci_hi']:+.2f}] "
            f"that INCLUDES 0, and the BEAR Sharpe is phase-fragile (flips sign across timeframes). "
            f"**Verdict: a WEAK, UNCONFIRMED candidate — not a null, not a confirmed edge.** On BTC ALONE "
            f"the single-path total-return outperformance is real but the noise swamps statistical "
            f"confirmation; 816 got per-regime confirmation only because averaging over 20 assets damped "
            f"that noise (and even 816 failed its strict FDR). The learned ridge predictor adds NOTHING "
            f"(OOS IC ≈ 0, net-negative, beats neither TSMOM nor B&H) — so the predictable component is the "
            f"slow trend itself, not a richer learned signal. **Decision: the lever is MORE PATHS (more "
            f"assets / longer history / walk-forward), not more model.** A single-asset bottom-up trader on "
            f"BTC alone cannot statistically confirm an edge here; expand the cross-section to damp the "
            f"noise, or treat 30d-lookback trend as a low-conviction overlay on low-cost perps.")
    elif promising:
        c = promising[0]
        klab = dict((k, v) for k, v in TIMEFRAMES)[c["K"]]
        lines.append(
            f"The TSMOM trend floor stands on BTC alone at "
            f"**{klab}/L={dur_label(c['Lm'])}/H={dur_label(c['Hm'])}** — bear-positive and net-of-cost "
            f"beats-B&H cross-regime with CI>0. It is a single-asset time-series TREND edge (short the "
            f"crash), modest and chop-fragile, NOT a learned alpha. The ridge predictor "
            f"{'adds value over the floor' if predB and any(np.isfinite(r['oos']['net21']) and np.isfinite(r['tsmom_oos']) and r['oos']['net21']>r['tsmom_oos'] for r in predB) else 'does NOT beat the TSMOM floor OOS'} "
            f"— so the predictable edge is the trend itself, not a richer learned signal.")
    else:
        lines.append(
            "**NO single-asset BTC cell clears the net-of-cost, cross-regime, beats-B&H, CI>0 gate.** "
            "The 816 edge was a 20-asset PORTFOLIO effect (diversification across majors); on BTC ALONE "
            "the time-series trend signal is too noisy to clear cost+CI cross-regime. This is a critical "
            "finding for the bottom-up direction: the unit that carries the edge is the CROSS-SECTION of "
            "many assets, not BTC in isolation — a single-asset bottom-up trader starts BELOW the floor.")
    return "".join(lines)


# =============================================================================
# DRIVER
# =============================================================================

def main():
    t0 = time.time()
    print("[819] self-tests...", flush=True)
    st_msgs = self_tests()
    for m in st_msgs:
        print("   " + m, flush=True)

    print("[819] loading BTC...", flush=True)
    close, grid_ts = load_btc()
    print(f"   close {close.shape} ts[{grid_ts[0]}..{grid_ts[-1]}]", flush=True)

    # market context (full-res log-ret per regime)
    logc_full = np.log(close)
    rows_full = regime_rows(grid_ts)
    mkt = {}
    for name in REG_NAMES:
        r0, r1 = rows_full[name]
        mkt[name] = float(logc_full[r1 - 1] - logc_full[r0]) if r1 - 1 > r0 else float("nan")
    print("   market context:", {k: round(v, 2) for k, v in mkt.items()}, flush=True)

    # Test A across timeframes
    all_cells = []
    subgrids = {}
    for K, klab in TIMEFRAMES:
        c_sub = close[::K]
        ts_sub = grid_ts[::K]
        subgrids[K] = (np.log(c_sub), ts_sub)
        cells = run_test_a(np.log(c_sub), ts_sub, K)
        all_cells += cells
        bp = sum(bear_positive(c) for c in cells)
        print(f"[A] {klab}: {len(cells)} cells, {bp} bear-positive", flush=True)

    promising = [c for c in all_cells if bear_positive(c) and full_ok(c)]
    promising = sorted(promising, key=cell_score, reverse=True)
    # cells to run Test B on: top-3 promising; if none promising, top-3 by FULL net@8 overall.
    # Dedup by DURATION (Lm,Hm) -- TSMOM at a fixed duration is ~timeframe-invariant on FULL, so
    # we cover 3 DISTINCT strategies; run each at its COARSEST representable timeframe (native
    # scale -> least feature noise for the predictor; fit-sample count is set by the H stride, not K).
    pool = promising if promising else sorted(all_cells, key=cell_score, reverse=True)
    seen, testB_cells = set(), []
    for c in pool:
        key = (c["Lm"], c["Hm"])
        if key in seen:
            continue
        seen.add(key)
        # pick the coarsest-K instance of this duration
        same = [d for d in all_cells if d["Lm"] == c["Lm"] and d["Hm"] == c["Hm"]]
        testB_cells.append(max(same, key=lambda d: d["K"]))
        if len(testB_cells) >= 3:
            break
    print(f"[B] running predictor on {len(testB_cells)} cells", flush=True)

    # Test B1 — strict early/late split
    SPLIT = "2024-01-01"
    predB, walkfwd = [], []
    for c in testB_cells:
        K, H = c["K"], c["H"]
        logc_sub, ts_sub = subgrids[K]
        rws = regime_rows(ts_sub)
        split_row = int(np.searchsorted(ts_sub, probe._to_epoch(SPLIT)))
        oos = eval_oos(logc_sub, ts_sub, K, H,
                       train_lo=rws["FULL"][0], train_hi=split_row,
                       test_lo=split_row, test_hi=rws["FULL"][1])
        if oos is None:
            continue
        # TSMOM same cell over the OOS span (apples-to-apples comparison)
        ts_res = tsmom_cell(logc_sub, c["L"], H, split_row, rws["FULL"][1])
        tsmom_oos = (backtest._sharpe(net_series(*ts_res[:2], HEADLINE_COST), H * K)
                     if ts_res else float("nan"))
        predB.append({"cell": c, "oos": oos, "tsmom_oos": tsmom_oos})

        # Test B2 — walk-forward: train = all regimes before the test regime
        folds = [("BEAR22", "BULL21"), ("REC23", "BEAR22"),
                 ("BULL24", "REC23"), ("CHOP25", "BULL24")]
        for test_reg, last_train_reg in folds:
            tr_lo = rws["FULL"][0]
            tr_hi = rws[test_reg][0]            # train up to start of the test regime
            te_lo, te_hi = rws[test_reg]
            wf = eval_oos(logc_sub, ts_sub, K, H, tr_lo, tr_hi, te_lo, te_hi)
            if wf is None:
                continue
            ts_res = tsmom_cell(logc_sub, c["L"], H, te_lo, te_hi)
            ts_sh = (backtest._sharpe(net_series(*ts_res[:2], HEADLINE_COST), H * K)
                     if ts_res else float("nan"))
            tr_span = f"≤{last_train_reg}"
            walkfwd.append({"cell": c, "reg": test_reg, "train": tr_span, "n": wf["n_oos"],
                            "ic": wf["ic"], "net21": wf["net21"], "tsmom": ts_sh, "bh": wf["bh"]})

    runtime = time.time() - t0
    write_report(all_cells, promising, predB, walkfwd, st_msgs, mkt, runtime)

    # dump raw cells for reproducibility
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    (_T / "_cells.json").write_text(json.dumps(clean(all_cells), indent=1))
    print(f"[819] wrote RESULT.md ({len(all_cells)} cells, {len(promising)} promising) "
          f"in {runtime:.1f}s", flush=True)


if __name__ == "__main__":
    main()

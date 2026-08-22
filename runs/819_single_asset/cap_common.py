#!/usr/bin/env python3
"""TASK_819 capacity ladder — SHARED utilities (single source of truth).

All four capacity-test arms (ridge-rich, GBM, MLP, GRU) import data loading,
the causal feature builder, the leak-free TIME split, the per-DECISION net-Sharpe
evaluator, and the ridge / TSMOM reference baselines from HERE, so every model is
scored on byte-identical features, splits, costs and Sharpe convention.

DISCIPLINE (matches cap_seq.py / 806->816)
- leak-free: features use close(s) <= t only; target = strictly forward H-d log-return.
- strict TIME split: TRAIN 2021-2023, VAL 2024-H1, TEST 2024-H2..2025 (frozen).
- per-asset normalization stats fit on TRAIN ONLY.
- cost-honest: net Sharpe at {0,3,8,21} bp/side, per-fire (|Δpos| one-way).
- per-DECISION Sharpe = mean/std*sqrt(525600/period_min), period_min = H*1440.
- OOS only for the verdict; report the TRAIN-vs-TEST gap explicitly.
"""
import sys
from pathlib import Path
import datetime as dt
import numpy as np
from scipy.stats import spearmanr

_T = Path(__file__).resolve().parent
sys.path.insert(0, str(_T.parent / "808_linear_backtest"))
import backtest  # noqa: E402  (audited 808 path — provides _sharpe + cost convention)

# ----------------------------------------------------------------------------- config
DATA = _T.parent / "816_operating_point" / "_close_crossregime.npz"
OG_SYM = ["XBT", "ETH", "LTC", "XRP", "XDG", "ADA", "MATIC", "LINK", "KSM", "SHIB",
          "XMR", "TRX", "XLM", "ALGO", "BCH", "INJ", "ATOM", "LUNA", "GRT", "AAVE"]
BTC_COL = 0
MIN_HIST = 91                      # common min history (momentum L=90 needs it)
COSTS_BP = [0.0, 3.0, 8.0, 21.0]   # bp/side, per-fire
CLIP_Z = 8.0

# day-index split boundaries (verified 2026-06-28 against grid_ts):
#   train days [MIN_HIST, 1096)  = ..2024-01-01
#   val   days [1096, 1278)      = 2024-01-01 .. 2024-07-01
#   test  days [1278, Td-1-H]    = 2024-07-01 .. 2025-08-31
VAL_START_DAY = 1096
TEST_START_DAY = 1278

# REGIME windows for per-regime OOS breakdown inside TEST
REGIMES = [("BULL24", "2024-01-01", "2024-12-31"),
           ("CHOP25", "2025-01-01", "2025-09-01")]


def epoch(ds):
    return int(dt.datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())


def regime_epochs():
    return {nm: (epoch(a), epoch(b)) for nm, a, b in REGIMES}


# ----------------------------------------------------------------------------- data
def load_daily():
    """Daily panel from the 1-min cross-regime cache. close[::1440] = 1 bar/day."""
    z = np.load(str(DATA))
    close = z["close"][::1440].astype(np.float64)    # (Td, A) daily
    ts = z["grid_ts"][::1440].astype(np.int64)
    logc = np.log(close)
    r = np.diff(logc, axis=0)                         # (Td-1, A); r[j]=logc[j+1]-logc[j]
    return close, ts, logc, r


def per_asset_stats(r, lo, hi):
    """mean/std of daily returns over TRAIN return rows [lo,hi)."""
    seg = r[lo:hi]
    return seg.mean(axis=0), seg.std(axis=0) + 1e-12


def rank_ic(pred, y):
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(pred) < 1e-12:
        return 0.0
    ic = spearmanr(pred, y).correlation
    return float(ic) if np.isfinite(ic) else 0.0


# ----------------------------------------------------------------------------- features
def _base_feature_names():
    lags = [1, 2, 3, 5, 10, 20]
    vols = [10, 20, 50]
    moms = [7, 14, 30, 60, 90]
    names = [f"lag{L}" for L in lags] + [f"vol{W}" for W in vols] + [f"mom{L}" for L in moms]
    return lags, vols, moms, names


def build_features(logc, r, mu, sd, H, rich=False):
    """Causal hand-crafted features per (decision-day t, asset a).
      base:  lagged log-returns [1,2,3,5,10,20]; realized vol [10,20,50];
             momentum past-L [7,14,30,60,90]  (14 features — the spec set).
      rich:  + extra lags [4,7,15,30]; vol-of-vol (std of rolling-10 vol over 20);
             + |mom30|, sign(mom30); + interactions mom30*vol20, lag1*vol10,
               mom30*mom7, vol20*vol50, lag1*lag2  (the capacity control).
    Target = forward H-d log-return; normalized target yn = fwd/(sd*sqrt(H)).
    Per-feature z-score uses TRAIN samples only (pooled across assets).
    Returns dict(Xz,yn,fwd,days,tr,va,te,A,names,rich)."""
    Td, A = logc.shape
    t_hi = Td - 1 - H
    days = np.arange(MIN_HIST, t_hi + 1)
    lags, vols, moms, names = _base_feature_names()
    nd = days.shape[0]

    base_cols = []
    for L in lags:
        base_cols.append(np.stack([r[t - L] for t in days]))                 # (nd,A)
    for W in vols:
        base_cols.append(np.stack([r[t - W:t].std(axis=0) for t in days]))
    for L in moms:
        base_cols.append(np.stack([logc[t] - logc[t - L] for t in days]))
    feats = {nm: c for nm, c in zip(names, base_cols)}

    use_names = list(names)
    if rich:
        # extra lags
        for L in [4, 7, 15, 30]:
            feats[f"lag{L}"] = np.stack([r[t - L] for t in days]); use_names.append(f"lag{L}")
        # vol-of-vol: std over a 20-day window of the rolling-10 realized vol
        vov = np.empty((nd, A))
        for k, t in enumerate(days):
            rv10 = np.array([r[s - 10:s].std(axis=0) for s in range(t - 20, t)])  # (20,A)
            vov[k] = rv10.std(axis=0)
        feats["volofvol"] = vov; use_names.append("volofvol")
        # nonlinear trend transforms
        feats["absmom30"] = np.abs(feats["mom30"]); use_names.append("absmom30")
        feats["sgnmom30"] = np.sign(feats["mom30"]); use_names.append("sgnmom30")
        # pairwise interactions
        inter = {"mom30_x_vol20": feats["mom30"] * feats["vol20"],
                 "lag1_x_vol10": feats["lag1"] * feats["vol10"],
                 "mom30_x_mom7": feats["mom30"] * feats["mom7"],
                 "vol20_x_vol50": feats["vol20"] * feats["vol50"],
                 "lag1_x_lag2": feats["lag1"] * feats["lag2"]}
        for nm, c in inter.items():
            feats[nm] = c; use_names.append(nm)

    X = np.stack([feats[nm] for nm in use_names], axis=-1)   # (nd, A, F)
    fwd = logc[days + H] - logc[days]                        # (nd, A) raw forward
    yn = fwd / (sd * np.sqrt(H))[None, :]

    tr = days < VAL_START_DAY
    va = (days >= VAL_START_DAY) & (days < TEST_START_DAY)
    te = days >= TEST_START_DAY

    # per-feature z-score on TRAIN samples only (pooled across assets)
    Xtr = X[tr].reshape(-1, X.shape[-1])
    fmu = Xtr.mean(axis=0)
    fsd = Xtr.std(axis=0) + 1e-12
    Xz = (X - fmu[None, None, :]) / fsd[None, None, :]
    return dict(Xz=Xz, yn=yn, fwd=fwd, days=days, tr=tr, va=va, te=te,
                A=A, names=use_names, rich=rich)


def split_xy(feat):
    """Flatten (day,asset) -> samples for tabular models."""
    F = feat["Xz"].shape[-1]
    Xz, yn = feat["Xz"], feat["yn"]
    tr, va, te = feat["tr"], feat["va"], feat["te"]
    return dict(
        Xtr=Xz[tr].reshape(-1, F), ytr=yn[tr].reshape(-1),
        Xva=Xz[va].reshape(-1, F), yva=yn[va].reshape(-1),
        Xte=Xz[te].reshape(-1, F), yte=yn[te].reshape(-1),
        F=F)


# ----------------------------------------------------------------------------- eval
def eval_signal(forecast, fwd, days, ts, H, pos_mode="tanh", regime=None,
                ci_costs=(), n_boot=2000, ci_seed=819):
    """Per-DECISION net-Sharpe over a contiguous daily grid with rebalance step=H.
    forecast,fwd: (nd, A) over consecutive decision days `days`.
    pos = tanh(z) (per-asset time z-score) or sign(forecast); rebalance every H rows
    (non-overlapping per-decision). Returns IC/R2/Sharpe(by cost) pooled + BTC.
    ci_costs: bp values for which to add a block-bootstrap 95% CI on the pooled net
    Sharpe (the campaign statistical gate — CI excluding 0 = a confirmed edge)."""
    nd, A = forecast.shape
    if regime is not None:
        lo, hi = regime
        e = ts[days]
        m = (e >= lo) & (e < hi)
    else:
        m = np.ones(nd, dtype=bool)
    fc = forecast[m]
    fw = fwd[m]
    if fc.shape[0] < 6:
        return None
    ic_pooled = rank_ic(fc.reshape(-1), fw.reshape(-1))
    ic_btc = rank_ic(fc[:, BTC_COL], fw[:, BTC_COL])
    ic_per = np.array([rank_ic(fc[:, a], fw[:, a]) for a in range(A)])

    mu = fc.mean(axis=0, keepdims=True)
    sd = fc.std(axis=0, keepdims=True) + 1e-12
    z = (fc - mu) / sd
    pos = np.tanh(z) if pos_mode == "tanh" else np.sign(fc)
    ridx = np.arange(0, fc.shape[0], H)
    P = pos[ridx]
    F = fw[ridx]
    prev = np.zeros((1, A))
    dpos = np.abs(P - np.vstack([prev, P[:-1]]))
    R_min = H * 1440
    out = {"n_rebal": int(P.shape[0]), "ic_pooled": ic_pooled, "ic_btc": ic_btc,
           "ic_mean_asset": float(np.mean(ic_per)), "sharpe": {}, "ci": {}}
    for cbp in COSTS_BP:
        c = cbp / 1e4
        net = P * F - c * dpos
        port = net.mean(axis=1)
        out["sharpe"][cbp] = {"pooled": backtest._sharpe(port, R_min),
                              "btc": backtest._sharpe(net[:, BTC_COL], R_min)}
        if cbp in ci_costs:
            lo_ci, hi_ci = backtest._block_bootstrap_sharpe_ci(
                port, R_min, R_min, n_boot, ci_seed)
            out["ci"][cbp] = (lo_ci, hi_ci)
    return out


def oos_r2(pred_flat, y_flat):
    pred_flat = np.asarray(pred_flat, dtype=np.float64)
    y_flat = np.asarray(y_flat, dtype=np.float64)
    sst = np.sum((y_flat - y_flat.mean()) ** 2)
    if sst < 1e-30:
        return float("nan")
    return float(1.0 - np.sum((y_flat - pred_flat) ** 2) / sst)


def eval_model_from_flat(pred_te_flat, feat, ts, H, pos_mode="tanh"):
    """Reshape a flat (day,asset)-ordered TEST prediction into the dense matrix and
    run eval_signal + per-regime breakdown + OOS R2. Returns (eval_dict, r2, regimes)."""
    te = feat["te"]
    A = feat["A"]
    n_te = int(te.sum())
    pred_mat = np.asarray(pred_te_flat, dtype=np.float64).reshape(n_te, A)
    fwd_te = feat["fwd"][te]
    days_te = feat["days"][te]
    ev = eval_signal(pred_mat, fwd_te, days_te, ts, H, pos_mode, ci_costs=(8.0, 21.0))
    r2 = oos_r2(pred_te_flat, feat["yn"][te].reshape(-1))
    regs = {}
    for nm, win in regime_epochs().items():
        regs[nm] = eval_signal(pred_mat, fwd_te, days_te, ts, H, pos_mode, regime=win)
    return ev, r2, regs


# ----------------------------------------------------------------------------- baselines
def ridge_fit(Xtr, ytr, alpha):
    P = Xtr.shape[1]
    return np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(P), Xtr.T @ ytr)


def ridge_baseline(feat, ts, H, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    """Ridge on `feat` (standard OR rich). VAL-selected alpha (by val rank-IC).
    Returns dict with OOS IC/R2/Sharpe, train IC (overfit signature), alpha."""
    s = split_xy(feat)
    best = None
    for a in alphas:
        w = ridge_fit(s["Xtr"], s["ytr"], a)
        ic = rank_ic(s["Xva"] @ w, s["yva"])
        if best is None or ic > best[0]:
            best = (ic, a, w)
    val_ic, alpha, w = best
    pred_te = s["Xte"] @ w
    ev, r2, regs = eval_model_from_flat(pred_te, feat, ts, H)
    ic_train = rank_ic(s["Xtr"] @ w, s["ytr"])
    return dict(name=f"ridge{'_rich' if feat['rich'] else ''}", alpha=float(alpha),
                val_ic=float(val_ic), oos_ic=ev["ic_pooled"], oos_ic_btc=ev["ic_btc"],
                oos_r2=float(r2), ic_train=float(ic_train),
                gap=float(ic_train - ev["ic_pooled"]), sharpe=ev["sharpe"], ci=ev.get("ci"),
                n_rebal=ev["n_rebal"], regimes=regs, nfeat=int(s["F"]))


def tsmom_signal(logc, H):
    """forecast = 30-day past log-return; position = sign(forecast). TEST grid only."""
    Td, A = logc.shape
    t_hi = Td - 1 - H
    days = np.arange(MIN_HIST, t_hi + 1)
    mom30 = logc[days] - logc[days - 30]
    fwd = logc[days + H] - logc[days]
    te = days >= TEST_START_DAY
    return mom30[te], fwd[te], days[te]


def tsmom_baseline(logc, ts, H):
    ts_pred, ts_fwd, ts_days = tsmom_signal(logc, H)
    ev = eval_signal(ts_pred, ts_fwd, ts_days, ts, H, "sign", ci_costs=(8.0, 21.0))
    regs = {}
    for nm, win in regime_epochs().items():
        regs[nm] = eval_signal(ts_pred, ts_fwd, ts_days, ts, H, "sign", regime=win)
    return dict(name="tsmom_sign30d", oos_ic=ev["ic_pooled"], oos_ic_btc=ev["ic_btc"],
                sharpe=ev["sharpe"], ci=ev.get("ci"), n_rebal=ev["n_rebal"], regimes=regs)


def sharpe_row(sharpe, ci=None):
    row = {str(int(c)): {"pooled": sharpe[c]["pooled"], "btc": sharpe[c]["btc"]} for c in COSTS_BP}
    if ci:
        for c, (lo, hi) in ci.items():
            row[str(int(c))]["ci"] = [float(lo), float(hi)]
    return row

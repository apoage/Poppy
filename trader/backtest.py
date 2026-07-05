#!/usr/bin/env python3
"""TASK_820 backtest / accounting layer.

Audited functions are COPIED here (not imported across runs/) with attribution:

  from runs/808_linear_backtest/backtest.py  (TASK_808, audited):
      _sharpe                       annualized Sharpe = mean/std * sqrt(525600/period)
      _cumulative_pnl_and_maxdd     cumprod(1+r) equity + max drawdown
      _block_bootstrap_sharpe_ci    circular block bootstrap 95% CI on post-cost Sharpe

  from runs/806_predictability_probe/probe.py  (TASK_806, audited):
      _to_epoch                     'YYYY-MM-DD' UTC midnight -> epoch seconds
      build_aligned_ohlcv           aligned minute-grid OHLCV loader (ffill/bfill convention)
      bh_fdr                        Benjamini-Hochberg FDR survive-mask

  from runs/819_single_asset/portfolio.py  (TASK_819 Phase-2, audited research reference):
      net_series, regime_rows, regime_series, cell_metrics, bear_positive, gate_at, clears_gate

The per-decision H-strided Sharpe, one-sided turnover cost model, block-bootstrap CI, and
exp(cumsum) equity/maxDD reproduce the 819 numbers bit-for-bit on numpy float64 (gate 1a).

BACKEND: the scoring/bootstrap layer runs on numpy for RNG determinism (np.random.RandomState
has no clean jax equivalent). The pure-arithmetic kernels (_sharpe, net_series, regime_series)
accept an `xp` handle and are xp-generic; cell_metrics materializes to numpy at the boundary so
gate/parity numbers are deterministic regardless of how the upstream signal was computed.
"""

import datetime as _dt
import sqlite3

import numpy as np
from scipy.stats import norm

from . import config
from .config import K, MINUTES_PER_YEAR, REGIMES, REG_NAMES
from . import strategy
from . import risk as riskmod


# =============================================================================
# probe.py (TASK_806) — copied verbatim with attribution
# =============================================================================

def _to_epoch(date_str: str) -> int:
    """[copied from runs/806_predictability_probe/probe.py] YYYY-MM-DD (UTC midnight)
    -> epoch seconds. DB timestamps are epoch seconds."""
    return int(
        _dt.datetime.strptime(date_str, "%Y-%m-%d")
        .replace(tzinfo=_dt.timezone.utc)
        .timestamp()
    )


def bh_fdr(pvals, q):
    """[copied from runs/806_predictability_probe/probe.py] Benjamini-Hochberg FDR at level q.
    Returns a boolean survives-mask aligned to pvals. NaN p-values never survive."""
    p = np.asarray(pvals, dtype=float)
    finite = np.isfinite(p)
    survive = np.zeros_like(p, dtype=bool)
    idx = np.where(finite)[0]
    if len(idx) == 0:
        return survive
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        survive[order[: kmax + 1]] = True
    return survive


def build_aligned_ohlcv(db_path, asset_ids, grid_start, grid_end, bar_sec, W=0):
    """[adapted from runs/806_predictability_probe/probe.build_aligned_ohlcv] Aligned
    (T, A, 5) minute-grid [O,H,L,C,V] over [grid_start - W*bar_sec .. grid_end), forward-filled
    intra-series (carry last real bar), single leading back-fill. Parameterized (no probe.Config)
    so it is self-contained. NOTE: this is the audited grid/ffill REFERENCE; the production daily
    path (data.build_daily_panel_from_db) uses streaming SQL aggregation and reproduces this same
    gap-fill semantics without loading the full table into RAM."""
    n_assets = len(asset_ids)
    g0 = int(grid_start) - int(W) * int(bar_sec)
    grid_ts = np.arange(g0, int(grid_end), int(bar_sec), dtype=np.int64)
    T = len(grid_ts)
    ts_to_row = {int(t): i for i, t in enumerate(grid_ts)}
    ohlcv = np.full((T, n_assets, 5), np.nan, dtype=np.float64)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    for col, aid in enumerate(asset_ids):
        cur.execute(
            "SELECT timestamp, open, high, low, close, volume FROM ohlcv "
            "WHERE asset_id=? AND timestamp>=? AND timestamp<? ORDER BY timestamp",
            (int(aid), g0, int(grid_end)),
        )
        for ts, o, h, lo, c, v in cur.fetchall():
            r = ts_to_row.get(int(ts))
            if r is not None:
                ohlcv[r, col, 0] = o
                ohlcv[r, col, 1] = h
                ohlcv[r, col, 2] = lo
                ohlcv[r, col, 3] = c
                ohlcv[r, col, 4] = v
    con.close()
    close = ohlcv[:, :, 3]
    have = ~np.isnan(close)
    for col in range(n_assets):
        valid_rows = np.where(have[:, col])[0]
        if len(valid_rows) == 0:
            continue
        last = None
        for r in range(T):
            if have[r, col]:
                last = ohlcv[r, col].copy()
            elif last is not None:
                ohlcv[r, col] = last
        first_r = valid_rows[0]
        if first_r > 0:
            ohlcv[:first_r, col] = ohlcv[first_r, col]
    return ohlcv, grid_ts


# =============================================================================
# 808 backtest.py — copied with attribution (Sharpe / equity / bootstrap CI)
# =============================================================================

def _sharpe(returns, period, xp=np):
    """[copied from runs/808_linear_backtest/backtest._sharpe] Annualized Sharpe =
    mean/std * sqrt(P), P = 525600 / period (period in minutes-per-decision). In 819/820,
    period = H * K (K=1440 daily) -> P = 525600/1440 = 365 for H=1d. xp-generic."""
    r = returns[xp.isfinite(returns)]
    if r.size < 2:
        return float("nan")
    sd = xp.std(r)
    if sd < 1e-15:
        return float("nan")
    return float(xp.mean(r) / sd * xp.sqrt(MINUTES_PER_YEAR / float(period)))


def _cumulative_pnl_and_maxdd(net):
    """[copied from runs/808_linear_backtest/backtest._cumulative_pnl_and_maxdd] Cumulative net
    PnL via PRODUCT compounding prod(1+net)-1 and max drawdown of that equity curve. (819 uses
    the exp(cumsum) log-return equity instead — see cell_metrics; this is kept for attribution
    and simple-return reporting.)"""
    r = net[np.isfinite(net)]
    if r.size == 0:
        return float("nan"), float("nan")
    equity = np.cumprod(1.0 + r)
    pnl = float(equity[-1] - 1.0)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    maxdd = float(np.min(drawdown))
    return pnl, maxdd


def _block_bootstrap_sharpe_ci(net, R, max_R, n_boot, seed):
    """[copied from runs/808_linear_backtest/backtest._block_bootstrap_sharpe_ci] Circular block
    bootstrap 95% CI on the post-cost annualized Sharpe. Block length L=max(5, ceil(2*max_R/R))
    STEPS. numpy RNG (RandomState) -> deterministic; not ported to jax."""
    r = net[np.isfinite(net)]
    n = r.size
    if n < 2:
        return float("nan"), float("nan")
    block_len = max(5, int(np.ceil(2.0 * max_R / R)))
    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(n / block_len))
    factor = np.sqrt(MINUTES_PER_YEAR / float(R))
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.randint(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel() % n
        idx = idx[:n]
        sample = r[idx]
        sd = np.std(sample, ddof=0)
        sharpes[b] = (np.mean(sample) / sd * factor) if sd > 1e-15 else np.nan
    sharpes = sharpes[np.isfinite(sharpes)]
    if sharpes.size < 2:
        return float("nan"), float("nan")
    lo, hi = np.percentile(sharpes, [2.5, 97.5])
    return float(lo), float(hi)


# =============================================================================
# 819 portfolio.py — accounting (net-of-cost, per-decision, regime metrics)
# =============================================================================

def net_series(gross, turn, c_bp, xp=np):
    """[copied from runs/819 portfolio.net_series] net(c) = gross - turnover * c/1e4. xp-generic."""
    return gross - turn * (c_bp / 1e4)


def regime_rows(ts_sub):
    """[copied from runs/819 portfolio.regime_rows] Map each regime window to (lo, hi) row
    indices in ts_sub via searchsorted on UTC-midnight epoch boundaries."""
    out = {}
    T = ts_sub.size
    for name, a, b in REGIMES:
        lo = int(np.searchsorted(ts_sub, _to_epoch(a)))
        hi = int(np.searchsorted(ts_sub, _to_epoch(b)))
        out[name] = (max(0, lo), min(T, hi))
    return out


def regime_series(logc, pos, pres, r_lo, r_hi, H, warm, xp=np):
    """[copied from runs/819 portfolio.regime_series] Per-decision portfolio gross / turnover /
    equal-weight B&H on an H-strided entry grid inside [r_lo, r_hi).
      gross_t = Sum_i pos[t,i]*fwd_i ; turn_t = Sum_i |pos[t,i]-pos[t-H,i]| ;
      bh_t = equal-weight long over present assets at t. Returns (gross, turn, bh) or None.
    Uses the forward label logc[t+H]-logc[t] (the ONLY future-touching quantity)."""
    T = logc.shape[0]
    lo = max(warm, r_lo)
    hi = min(r_hi - H, T - 1 - H)
    if hi - lo < H:
        return None
    t = xp.arange(lo, hi + 1, H)
    if t.size < 6:
        return None
    fwd = logc[t + H] - logc[t]
    fwd = xp.where(xp.isfinite(fwd), fwd, 0.0)
    pos_t = pos[t]
    pos_prev = pos[t - H]
    gross = xp.sum(pos_t * fwd, axis=1)
    turn = xp.sum(xp.abs(pos_t - pos_prev), axis=1)
    pr = pres[t].astype(logc.dtype)
    wbh = pr / xp.maximum(pr.sum(axis=1, keepdims=True), 1e-300)
    bh = xp.sum(wbh * fwd, axis=1)
    return gross, turn, bh


def cell_metrics(logc, pos, pres, ts_sub, H, warm, scfg=None):
    """[copied from runs/819 portfolio.cell_metrics] All-regime metrics for one portfolio config.
    FULL also gets bootstrap CI (net@8 & @21) + deployable metrics (equity multiple, maxDD,
    annualized Sharpe, annual turnover). Equity = exp(cumsum(net)) (819 log-return convention).

    Scoring runs on numpy (materialized here) for RNG determinism; upstream `pos` may be any
    backend array."""
    if scfg is None:
        scfg = config.STRATEGY
    costs_bp = scfg.costs_bp
    gate_costs = scfg.gate_costs
    n_boot = scfg.n_boot
    seed = scfg.seed
    # materialize to numpy float64 for deterministic scoring / bootstrap
    logc = config.to_numpy(logc)
    pos = config.to_numpy(pos)
    pres = np.asarray(pres)
    ts_sub = np.asarray(ts_sub)

    rows = regime_rows(ts_sub)
    out = {"H": H, "reg": {}}
    for name in REG_NAMES:
        r0, r1 = rows[name]
        res = regime_series(logc, pos, pres, r0, r1, H, warm, xp=np)
        if res is None:
            out["reg"][name] = None
            continue
        gross, turn, bh = res
        period = H * K
        d = {"n": int(gross.size), "turn": float(np.mean(turn)),
             "gross": _sharpe(gross, period), "bh": _sharpe(bh, period)}
        for c in costs_bp:
            d[f"net{c}"] = _sharpe(net_series(gross, turn, c), period)
        if name == "FULL":
            for c in gate_costs:
                nh = net_series(gross, turn, c)
                ci = _block_bootstrap_sharpe_ci(nh, period, period, n_boot, seed)
                sd = np.std(nh, ddof=0)
                ts_stat = (np.sqrt(nh.size) * np.mean(nh) / sd) if sd > 1e-15 else 0.0
                d[f"ci{c}_lo"], d[f"ci{c}_hi"], d[f"p{c}"] = float(ci[0]), float(ci[1]), float(norm.sf(ts_stat))
                eq = np.exp(np.cumsum(nh))
                peak = np.maximum.accumulate(eq)
                d[f"cum{c}"] = float(np.sum(nh))
                d[f"mult{c}"] = float(eq[-1])
                d[f"maxdd{c}"] = float(np.min(eq / peak - 1.0))
            d["cum_bh"] = float(np.sum(bh))
            d["mult_bh"] = float(np.exp(np.sum(bh)))
            years = (d["n"] * H * K) / MINUTES_PER_YEAR
            d["years"] = float(years)
            d["ann_turn"] = float(np.sum(turn) / years) if years > 0 else float("nan")
        out["reg"][name] = d
    return out


def bear_positive(cell, c):
    """[copied from runs/819 portfolio.bear_positive]"""
    b = cell["reg"].get("BEAR22")
    return bool(b and np.isfinite(b[f"net{c}"]) and b[f"net{c}"] > 0 and b[f"net{c}"] > b["bh"])


def gate_at(cell, c):
    """[copied from runs/819 portfolio.gate_at]"""
    f = cell["reg"].get("FULL")
    if not f or not np.isfinite(f.get(f"ci{c}_lo", np.nan)):
        return False
    return bool(f[f"ci{c}_lo"] > 0 and f[f"net{c}"] > f["bh"] and bear_positive(cell, c))


def clears_gate(cell, gate_costs=(8, 21)):
    """[copied from runs/819 portfolio.clears_gate]"""
    return any(gate_at(cell, c) for c in gate_costs)


# =============================================================================
# TASK_820 helpers — full frozen cell + gate-1a parity
# =============================================================================

def compute_full_cell(logc, ts_sub, scfg=None, xp=None):
    """Build the frozen-strategy positions and score all regimes. Returns the cell dict."""
    if scfg is None:
        scfg = config.STRATEGY
    pos, pres, w, warm = strategy.positions_from_config(logc, scfg, xp=xp)
    return cell_metrics(logc, pos, pres, ts_sub, scfg.H_rebal, warm, scfg=scfg)


def run_overlay_backtest(logc, ts_sub, scfg=None, rcfg=None, equity_cost_bp=21.0):
    """Sequential backtest WITH the risk overlay ON (overlays 1-4 are stateful — DD throttle
    tracks running equity/HWM, rebalance band holds the prior applied position). Replays each
    historical bar as if fresh (now_ts == bar_ts, so the staleness guard never fires in-sample).
    Returns the applied per-decision (gross, turn, bh) FULL series + overlay activity counters.
    NOT a parity path — the overlay deliberately perturbs the 819 positions."""
    if scfg is None:
        scfg = config.STRATEGY
    if rcfg is None:
        rcfg = config.RiskConfig(enabled=True)
    logc = config.to_numpy(logc)
    ts_sub = np.asarray(ts_sub)
    pos, pres, w, warm = strategy.positions_from_config(logc, scfg)
    pos = config.to_numpy(pos)
    pres = np.asarray(pres)
    A = logc.shape[1]
    H = scfg.H_rebal
    T = logc.shape[0]
    r0, r1 = regime_rows(ts_sub)["FULL"]
    lo = max(warm, r0)
    hi = min(r1 - H, T - 1 - H)
    t = np.arange(lo, hi + 1, H)
    equity, hwm = 1.0, 1.0
    applied_prev = np.zeros(A)
    cost_rate = equity_cost_bp / 1e4
    G, TU, BH = [], [], []
    n_stale = n_halt = n_capped = 0
    for ti in t:
        new, info = riskmod.apply_overlay(
            pos[ti], applied_prev, equity, hwm,
            latest_ts=int(ts_sub[ti]), now_ts=int(ts_sub[ti]), rcfg=rcfg,
        )
        fwd = logc[ti + H] - logc[ti]
        fwd = np.where(np.isfinite(fwd), fwd, 0.0)
        g = float(np.sum(new * fwd))
        turn = float(np.sum(np.abs(new - applied_prev)))
        pr = pres[ti].astype(np.float64)
        wbh = pr / max(pr.sum(), 1e-300)
        bh = float(np.sum(wbh * fwd))
        net = g - turn * cost_rate
        equity *= float(np.exp(net))
        hwm = max(hwm, equity)
        G.append(g); TU.append(turn); BH.append(bh)
        n_stale += int(info["stale"]); n_halt += int(info["halt"]); n_capped += int(info["n_capped"] > 0)
        applied_prev = new
    return {"t": t, "gross": np.asarray(G), "turn": np.asarray(TU), "bh": np.asarray(BH),
            "warm": warm, "n_stale": n_stale, "n_halt": n_halt, "n_capped": n_capped,
            "equity_cost_bp": equity_cost_bp, "n": len(t)}


# Gate-1a parity targets (819 Phase-2 broad/equal/single30 H=1d, overlay OFF) with tolerances.
PARITY_TARGETS = [
    ("FULL net@8 Sharpe", ("FULL", "net8"), 1.0865, 0.02),
    ("FULL net@21 Sharpe", ("FULL", "net21"), 0.9673, 0.02),
    ("FULL turnover/day", ("FULL", "turn"), 0.1551, 0.005),
    ("FULL equity x@8bp", ("FULL", "mult8"), 21.69, 0.25),
    ("FULL maxDD@8bp", ("FULL", "maxdd8"), -0.3776, 0.01),
    ("BEAR22 net@8 Sharpe", ("BEAR22", "net8"), 1.3622, 0.05),
]

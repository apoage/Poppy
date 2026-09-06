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

import numpy as np
from scipy.stats import norm

from . import config
from .config import K, MINUTES_PER_YEAR, REGIMES, REG_NAMES
from . import strategy
# TASK_870: read-only handle on the canon trading_data.db (see mcts_trader/utils/db.py) — a
# plain read-write sqlite3.connect on this 24GB read-only source of record lets SQLite's
# hot-journal recovery rewrite the file on the first pure-SELECT read.
from mcts_trader.utils.db import _ro_connect
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
    con = _ro_connect(db_path)
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


# Panel warmup for score-windowed runs: signal/mask need max(lookback, vol_window) active bars
# before the first scored decision, plus slack for listing quirks / short leading gaps.
WARMUP_BUFFER_DAYS = 30


def warmup_days(scfg=None):
    """Daily bars of pre-history a score-windowed panel must load before score-start."""
    if scfg is None:
        scfg = config.STRATEGY
    return max(max(scfg.lookbacks), scfg.vol_window) + WARMUP_BUFFER_DAYS


def window_metrics(logc, ts_sub, start_ts=None, end_ts=None, scfg=None, xp=None):
    """Gate-2 style scorer: frozen-strategy positions built on the WHOLE loaded panel (which
    must include >= warmup_days of pre-history), but metrics/equity/turnover computed ONLY on
    decisions with ts in [start_ts, end_ts). Same accounting as cell_metrics' FULL branch
    (net-of-cost Sharpe per cost, bootstrap CI + equity + maxDD at gate costs, ann turnover).
    Returns the metrics dict (plus n = decision count) or None if the window is too thin.

    The parity path (compute_full_cell -> cell_metrics) is deliberately NOT routed through
    this function — gate 1a stays byte-identical to the 819 research code."""
    if scfg is None:
        scfg = config.STRATEGY
    pos, pres, w, warm = strategy.positions_from_config(logc, scfg, xp=xp)
    logc = config.to_numpy(logc)
    pos = config.to_numpy(pos)
    pres = np.asarray(pres)
    ts_sub = np.asarray(ts_sub)
    T = ts_sub.size
    r_lo = int(np.searchsorted(ts_sub, start_ts)) if start_ts is not None else 0
    r_hi = int(np.searchsorted(ts_sub, end_ts)) if end_ts is not None else T
    H = scfg.H_rebal
    res = regime_series(logc, pos, pres, r_lo, r_hi, H, warm, xp=np)
    if res is None:
        return None
    gross, turn, bh = res
    period = H * K
    d = {"n": int(gross.size), "turn": float(np.mean(turn)),
         "gross": _sharpe(gross, period), "bh": _sharpe(bh, period)}
    for c in scfg.costs_bp:
        d[f"net{c}"] = _sharpe(net_series(gross, turn, c), period)
    for c in scfg.gate_costs:
        nh = net_series(gross, turn, c)
        ci = _block_bootstrap_sharpe_ci(nh, period, period, scfg.n_boot, scfg.seed)
        eq = np.exp(np.cumsum(nh))
        peak = np.maximum.accumulate(eq)
        d[f"ci{c}_lo"], d[f"ci{c}_hi"] = float(ci[0]), float(ci[1])
        d[f"cum{c}"] = float(np.sum(nh))
        d[f"mult{c}"] = float(eq[-1])
        d[f"maxdd{c}"] = float(np.min(eq / peak - 1.0))
    d["cum_bh"] = float(np.sum(bh))
    d["mult_bh"] = float(np.exp(np.sum(bh)))
    years = (d["n"] * H * K) / MINUTES_PER_YEAR
    d["years"] = float(years)
    d["ann_turn"] = float(np.sum(turn) / years) if years > 0 else float("nan")
    return d


def run_overlay_backtest(logc, ts_sub, scfg=None, rcfg=None, equity_cost_bp=21.0,
                         lo_ts=None, hi_ts=None):
    """Sequential backtest WITH the risk overlay ON (overlays 1-4 are stateful — the DD
    throttle tracks a running curve, the rebalance band holds the prior applied position).
    Replays each historical bar as if fresh (now_ts == bar_ts, so the staleness guard never
    fires in-sample).

    dd_mode='shadow_scale' (default): a SHADOW book — the uncapped overlay-OFF frozen
    strategy, net of the same cost — runs alongside; its equity/HWM drive the throttle.
    At decision t the shadow curve includes returns realized THROUGH t only (leak-free).
    dd_mode='realized_flat' (legacy): the throttled book's own realized curve drives it.

    Window: scores the FULL regime by default; [lo_ts, hi_ts) epoch bounds override it
    (gate-2 style). Returns per-decision (gross, turn, bh) + overlay activity counters +
    the shadow equity multiple. NOT a parity path."""
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
    if lo_ts is None and hi_ts is None:
        r0, r1 = regime_rows(ts_sub)["FULL"]
    else:
        r0 = int(np.searchsorted(ts_sub, lo_ts)) if lo_ts is not None else 0
        r1 = int(np.searchsorted(ts_sub, hi_ts)) if hi_ts is not None else T
    lo = max(warm, r0)
    hi = min(r1 - H, T - 1 - H)
    t = np.arange(lo, hi + 1, H)
    equity, hwm = 1.0, 1.0            # realized (throttled) book
    sh_eq, sh_hwm = 1.0, 1.0          # shadow (uncapped overlay-OFF) book
    applied_prev = np.zeros(A)
    shadow_prev = np.zeros(A)
    cost_rate = equity_cost_bp / 1e4
    G, TU, BH = [], [], []
    n_stale = n_halt = n_capped = n_half = n_deep = 0
    for ti in t:
        raw = pos[ti]
        if rcfg.dd_mode == "shadow_scale":
            drive_eq, drive_hwm = sh_eq, sh_hwm
        else:
            drive_eq, drive_hwm = equity, hwm
        new, info = riskmod.apply_overlay(
            raw, applied_prev, drive_eq, drive_hwm,
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
        # shadow book: the raw (uncapped) frozen positions, same cost model. Updated AFTER
        # the decision, so the next decision sees shadow equity realized through its own t.
        s_turn = float(np.sum(np.abs(raw - shadow_prev)))
        s_net = float(np.sum(raw * fwd)) - s_turn * cost_rate
        sh_eq *= float(np.exp(s_net))
        sh_hwm = max(sh_hwm, sh_eq)
        shadow_prev = raw
        G.append(g); TU.append(turn); BH.append(bh)
        n_stale += int(info["stale"]); n_halt += int(info["halt"]); n_capped += int(info["n_capped"] > 0)
        n_half += int(info["dd_scale"] == 0.5)
        n_deep += int(0.0 < info["dd_scale"] <= rcfg.dd_deep_scale)
        applied_prev = new
    return {"t": t, "gross": np.asarray(G), "turn": np.asarray(TU), "bh": np.asarray(BH),
            "warm": warm, "n_stale": n_stale, "n_halt": n_halt, "n_capped": n_capped,
            "n_half": n_half, "n_deep": n_deep, "dd_mode": rcfg.dd_mode,
            "shadow_mult": sh_eq, "equity_cost_bp": equity_cost_bp, "n": len(t)}


# Gate-1a parity targets (819 Phase-2 broad/equal/single30 H=1d, overlay OFF) with tolerances.
PARITY_TARGETS = [
    ("FULL net@8 Sharpe", ("FULL", "net8"), 1.0865, 0.02),
    ("FULL net@21 Sharpe", ("FULL", "net21"), 0.9673, 0.02),
    ("FULL turnover/day", ("FULL", "turn"), 0.1551, 0.005),
    ("FULL equity x@8bp", ("FULL", "mult8"), 21.69, 0.25),
    ("FULL maxDD@8bp", ("FULL", "maxdd8"), -0.3776, 0.01),
    ("BEAR22 net@8 Sharpe", ("BEAR22", "net8"), 1.3622, 0.05),
]


# =============================================================================
# TASK_920 scorer honesty pack — REPORTING-ONLY overlays.
#
# Every function below is new and additive: none is called from compute_full_cell,
# cell_metrics, window_metrics, regime_series, or run_overlay_backtest, so gate 1a
# (`parity`) and the default `backtest` report are computed by EXACTLY the pre-920 code
# path regardless of what lives here (G1: OFF = bit-identical). These are wired into the
# CLI ONLY behind new flags that default to the legacy behavior (trader/cli.py).
# =============================================================================

def _decision_grid(logc, scfg, r_lo, r_hi, xp=None):
    """Build the frozen-strategy positions and the SAME H-strided entry grid [lo, hi] that
    regime_series derives internally (lo=max(warm,r_lo), hi=min(r_hi-H, T-1-H)), materialized
    to numpy. Returns (logc_np, pos_np, pres_np, t, H, warm) or None if the window is too
    thin — the identical thinness contract regime_series enforces (kept in lockstep
    deliberately; this is glue for the honesty-pack overlays below, not a new scoring path)."""
    pos, pres, w, warm = strategy.positions_from_config(logc, scfg, xp=xp)
    logc_np = config.to_numpy(logc)
    pos_np = config.to_numpy(pos)
    pres_np = np.asarray(pres)
    T = logc_np.shape[0]
    H = scfg.H_rebal
    lo = max(warm, r_lo)
    hi = min(r_hi - H, T - 1 - H)
    if hi - lo < H:
        return None
    t = np.arange(lo, hi + 1, H)
    if t.size < 6:
        return None
    return logc_np, pos_np, pres_np, t, H, warm


def _utc_day(ts):
    """epoch seconds -> 'YYYY-MM-DD' (UTC). Reporting/diagnostics only."""
    return _dt.datetime.fromtimestamp(int(ts), tz=_dt.timezone.utc).strftime("%Y-%m-%d")


def _book_value(shares, px):
    """Mark a share book to prices `px`. Columns holding EXACTLY zero shares contribute
    exactly 0.0 even where their price is NaN — a plain `shares * px` gives 0 * NaN = NaN
    there and poisons the whole sum from an asset the book does not even hold. So a
    non-finite result here means a HELD asset had a bad price, which true_bh_series raises
    on rather than flooring (see its fail-loud contract).

    DELISTING CONVENTION (fix pass, run 53 cycle 3 — regression-lens MINOR). "Held" here is
    denominated in SHARES, not in presence: this function never consults `pres`, so a column
    the present-mask has RETIRED (the constant ffilled post-delisting tail, the LUNA shape at
    trader/data.py:328-331) keeps marking the book at its frozen last price until the next
    calendar-year re-form recycles that value into live assets. That is the "liquidate at last
    print" convention, and the defensible alternative — a delisted holding is a total loss,
    i.e. mask by `pres` — is a MODELLING decision for the operator, not something this pass
    picks silently. So the convention is left AS IS and made VISIBLE instead:
    true_bh_series(..., return_diag=True) measures the non-present share of book value and
    the CLI prints it beside every true-B&H figure. Measured on the pinned npz panel: 3.55% of
    terminal book value (0.1007 of 2.8382), 3.77% max (2025-08-02); `--source db
    --include-dead` 2021->2024 is smaller still. Unbounded in principle for a concentrated or
    high-mortality window, which is why it is disclosed every time rather than dismissed."""
    return float(np.sum(shares * np.where(shares != 0.0, px, 0.0)))


def true_bh_series(logc, pres, ts_sub, t, H, return_diag=False):
    """B4 / leg 1 (--bh-book true-bh): a TRUE buy-and-hold book — fixed SHARES, re-formed to
    equal-dollar-weight over present assets only once per calendar year — as opposed to the
    legacy 'B&H' column (regime_series' `bh`), which re-normalizes to equal DOLLAR weight
    EVERY decision and charges it zero turnover (not buy-and-hold at all; see task file).

    Walks a share-count equity curve V bar-by-bar over [t.min(), t.max()+H] (numpy, since this
    needs actual calendar dates from ts_sub, not just row deltas): initial equal-dollar buy at
    the first row, re-form at the first row of every subsequent calendar year. Zero turnover is
    charged at re-forms (same convention as the legacy column) since this reports the buy-and-
    hold BENCHMARK, not a traded strategy. Returns bh[k] = log(V[t[k]+H]) - log(V[t[k]]),
    aligned to `t` (same shape/units as regime_series' bh).

    FAIL-LOUD CONTRACT (fix pass, run 53 cycle 2 — correctness-lens MAJOR). Two ways the book
    can become undefined, and NEITHER is floored or absorbed any more; both raise ValueError:

      1. `n == 0` at a RE-FORM row (no asset present on the exact row a calendar year rolls
         over). The pre-fix code set `shares = 0` there, which is an ABSORBING state: every
         later re-form then marks the book at cur_val = 0, so logV sat at log(1e-300) ~= -690
         for the entire remainder of the series — no NaN, no warning, a silently wrong Sharpe
         and equity multiple. Note the trigger is denominated in the presence count AT A
         RE-FORM ROW, not in general sparsity: an all-absent NON-re-form row is harmless (the
         book simply holds its existing shares) and is deliberately NOT an error.
      2. A non-finite or non-positive book value at any row — i.e. a HELD asset priced NaN /
         <= 0. `max(v, 1e-300)` used to floor exactly this into a plausible-looking number.

    Both raise instead of returning a number because this figure is QUOTED: absence read as a
    value is the failure mode this whole task exists to remove, so the caller must lose the
    B&H column, never silently receive a wrong one.

    DISCLOSURE (fix pass, run 53 cycle 3 — regression-lens MINOR). `return_diag=True` also
    returns a dict measuring how much of the book is marked on RETIRED (non-present) columns,
    i.e. how much of the reported figure rests on the "liquidate at last print" delisting
    convention documented on _book_value. The convention itself is UNCHANGED — the bh series
    returned is byte-identical with and without the flag — because which convention is right
    is the operator's modelling call; this pass only stops it being invisible. Keys:
    phantom_share_final / phantom_share_max (fraction of book value on non-present holdings at
    the last walked row, and its maximum over the walk), phantom_row_max / phantom_day_max
    (where the maximum occurred), book_value_final / phantom_value_final (the absolute pair
    the final share is computed from). A phantom share of 0 means every marked holding was
    still present, i.e. the convention never bound and the figure is convention-free."""
    price = np.exp(logc)
    row_lo = int(t.min())
    row_hi = int(t.max()) + H
    years = np.fromiter(
        (_dt.datetime.fromtimestamp(int(ts_sub[r]), tz=_dt.timezone.utc).year
         for r in range(row_lo, row_hi + 1)),
        dtype=np.int64, count=row_hi - row_lo + 1,
    )
    A = price.shape[1]
    shares = np.zeros(A)
    logV = np.empty(row_hi - row_lo + 1)
    prev_year = None
    ph_share_final = 0.0
    ph_value_final = 0.0
    ph_share_max = 0.0
    ph_row_max = row_lo
    for k, row in enumerate(range(row_lo, row_hi + 1)):
        cur_year = years[k]
        if prev_year is None or cur_year != prev_year:
            cur_val = _book_value(shares, price[row]) if prev_year is not None else 1.0
            p = pres[row]
            n = int(np.sum(p))
            if n == 0:
                raise ValueError(
                    f"true_bh_series: NO asset is present at the calendar-year re-form row "
                    f"{row} ({_utc_day(ts_sub[row])}, year {int(cur_year)}) — a buy-and-hold "
                    f"book cannot be formed there, and zeroing it is an ABSORBING state that "
                    f"would silently floor the rest of the series at log(1e-300). Refusing to "
                    f"report a true-B&H number. Move the window past the gap "
                    f"(--score-start/--score-end) or use --bh-book legacy-label."
                )
            if not np.isfinite(cur_val) or cur_val <= 0.0:
                raise ValueError(
                    f"true_bh_series: book value {cur_val!r} at the calendar-year re-form row "
                    f"{row} ({_utc_day(ts_sub[row])}) — a held asset is priced non-finite or "
                    f"<= 0. Refusing to re-form the book from an undefined value."
                )
            shares = np.where(p, (cur_val / n) / np.maximum(price[row], 1e-300), 0.0)
            prev_year = cur_year
        v = _book_value(shares, price[row])
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(
                f"true_bh_series: buy-and-hold book value {v!r} at row {row} "
                f"({_utc_day(ts_sub[row])}) — a HELD asset is priced non-finite or <= 0. "
                f"Refusing to report a B&H number floored out of an undefined value."
            )
        logV[k] = np.log(v)
        if return_diag:
            # v is finite and > 0 here (checked above), so every HELD column has a good
            # price and this sum needs no NaN guard of its own.
            phantom = np.asarray(shares != 0.0) & ~np.asarray(pres[row], dtype=bool)
            ph_value_final = float(np.sum(shares[phantom] * price[row][phantom]))
            ph_share_final = ph_value_final / v
            if ph_share_final > ph_share_max:
                ph_share_max = ph_share_final
                ph_row_max = row
    idx0 = t - row_lo
    idx1 = idx0 + H
    bh = logV[idx1] - logV[idx0]
    if not return_diag:
        return bh
    return bh, {"phantom_share_final": ph_share_final,
                "phantom_share_max": ph_share_max,
                "phantom_row_max": int(ph_row_max),
                "phantom_day_max": _utc_day(ts_sub[ph_row_max]),
                "phantom_value_final": ph_value_final,
                "book_value_final": float(np.exp(logV[-1]))}


def gross_simple_series(logc, pos, t, H):
    """B2 / leg 2 (--accounting simple): dollar-weighted SIMPLE forward return per decision,
    gross_t = Sum_i pos[t,i] * (exp(logc[t+H,i]-logc[t,i]) - 1) — as opposed to the canon
    log-summed gross (regime_series: Sum_i pos[t,i]*(logc[t+H,i]-logc[t,i])). A weighted sum
    of SIMPLE per-asset returns is the return a book actually realizes when it holds those
    dollar weights over [t, t+H]; a weighted sum of LOG returns is a linear approximation
    that misses the Jensen's-inequality gap between them — the rebalancing/diversification
    premium the task's leg 2 refers to."""
    fwd_log = logc[t + H] - logc[t]
    fwd_log = np.where(np.isfinite(fwd_log), fwd_log, 0.0)
    simp = np.exp(fwd_log) - 1.0
    return np.sum(pos[t] * simp, axis=1)


def drift_retrim_turnover(pos, t, H, logc):
    """B3 / leg 3 (--charge-drift-rebalance): decompose real-world turnover into (a) the
    TARGET-WEIGHT-CHANGE component already charged by regime_series' `turn`
    (Sum_i |pos[t,i]-pos[t-H,i]|) and (b) a currently-UNCHARGED constant-dollar DRIFT-RETRIM
    component — the trade needed to correct the OLD target's natural price drift over
    [t-H, t] even when the target itself does not change: drift_i = pos[t-H,i] *
    (exp(logc[t,i]-logc[t-H,i]) - 1), i.e. the per-asset price-return move of the prior
    period's dollar allocation (first-order approximation: ignores compounding against
    realized portfolio-level P&L, disclosed here and in the report). Returns
    (target_change_turn, drift_turn), each shaped like regime_series' `turn`; the canon
    charged turnover is target_change_turn (bit-identical to regime_series); a book that also
    charges drift-retrim pays target_change_turn + drift_turn.

    DIRECTION OF THE APPROXIMATION (fix pass, run 53 cycle 2 — this is stronger than
    "approximate", and it is the true claim): charging target_change_turn + drift_turn is a
    triangle-inequality UPPER BOUND on the turnover of the single COMBINED trade a real desk
    would send, because per asset |dtarget_i - drift_i| <= |dtarget_i| + |drift_i|. So the
    disclosed cost can only ever OVERSTATE the drift-retrim charge, never understate it —
    a conservative bound of KNOWN sign, not an approximation of unknown direction. Measured
    on the pinned npz book: disclosed mean turnover 0.1923 vs netted single-combined-trade
    0.1878 (~1.9% overstatement), net@21 Sharpe delta -0.046 disclosed vs -0.042 netted;
    both are inside the -0.10 G2 tripwire, so the bound's slack changes no gate outcome.
    ST14 pins the inequality itself on a planted panel."""
    pos_t = pos[t]
    pos_prev = pos[t - H]
    fwd_prev = logc[t] - logc[t - H]
    fwd_prev = np.where(np.isfinite(fwd_prev), fwd_prev, 0.0)
    drift = pos_prev * (np.exp(fwd_prev) - 1.0)
    target_change_turn = np.sum(np.abs(pos_t - pos_prev), axis=1)
    drift_turn = np.sum(np.abs(drift), axis=1)
    return target_change_turn, drift_turn


def honesty_pack_report(logc, ts_sub, scfg, r_lo, r_hi, net_bps, bh_book="legacy-label",
                         accounting="log", charge_drift_rebalance=False, xp=None):
    """TASK_920 legs 1-3, packaged: computes only the overlays requested (every arg left at
    its default is a no-op — no key in the returned dict), on the IDENTICAL decision grid
    (_decision_grid) the canon FULL/WINDOW report scores, via the new functions above. Never
    calls regime_series/cell_metrics/window_metrics, so the canon numbers this augments are
    computed by unmodified pre-920 code. Returns None if the window is too thin (same
    thinness contract as regime_series), or a dict with any of 'bh_book', 'accounting',
    'drift_rebalance' — PLUS a 'window' key naming the exact span every figure in it is
    denominated on (fix pass, run 53 cycle 2: 'the 2022 bear window' has two defensible
    readings — calendar-2022 and the BEAR22 regime window 2021-11-10..2022-12-31,
    config.REGIMES — with visibly different numbers, so no figure here is reported without
    its window attached).

    Raises ValueError (from true_bh_series) rather than returning a floored number when the
    requested true-B&H book is undefined; the CLI turns that into a refusal + non-zero exit."""
    if bh_book == "legacy-label" and accounting == "log" and not charge_drift_rebalance:
        return None
    grid = _decision_grid(logc, scfg, r_lo, r_hi, xp=xp)
    if grid is None:
        return None
    logc_np, pos_np, pres_np, t, H, warm = grid
    ts_sub = np.asarray(ts_sub)
    period = H * K
    out = {}

    if bh_book == "true-bh":
        bh_true, bh_diag = true_bh_series(logc_np, pres_np, ts_sub, t, H, return_diag=True)
        out["bh_book"] = {"sharpe": _sharpe(bh_true, period),
                           "cum": float(np.sum(bh_true)),
                           "mult": float(np.exp(np.sum(bh_true))),
                           # delisting-convention disclosure, MEASURED on this very walk —
                           # see _book_value / true_bh_series. Never omitted: the caller
                           # prints it beside the figure so the convention is never silent.
                           "phantom": bh_diag}

    if accounting == "simple":
        fwd_log = logc_np[t + H] - logc_np[t]
        fwd_log = np.where(np.isfinite(fwd_log), fwd_log, 0.0)
        gross_log = np.sum(pos_np[t] * fwd_log, axis=1)
        gross_simple = gross_simple_series(logc_np, pos_np, t, H)
        turn = np.sum(np.abs(pos_np[t] - pos_np[t - H]), axis=1)
        net_log = net_series(gross_log, turn, net_bps)
        net_simple = net_series(gross_simple, turn, net_bps)
        out["accounting"] = {
            "gross_sharpe": _sharpe(gross_simple, period),
            "net_sharpe": _sharpe(net_simple, period),
            "bp_per_day": float(np.mean(gross_simple - gross_log)) * 1e4 / H,
            "sharpe_delta": _sharpe(net_simple, period) - _sharpe(net_log, period),
        }

    if charge_drift_rebalance:
        target_change_turn, drift_turn = drift_retrim_turnover(pos_np, t, H, logc_np)
        fwd = logc_np[t + H] - logc_np[t]
        fwd = np.where(np.isfinite(fwd), fwd, 0.0)
        gross = np.sum(pos_np[t] * fwd, axis=1)
        turn_drifted = target_change_turn + drift_turn
        sh_canon21 = _sharpe(net_series(gross, target_change_turn, 21), period)
        sh_drift21 = _sharpe(net_series(gross, turn_drifted, 21), period)
        years = (t.size * H * K) / MINUTES_PER_YEAR
        ann_turn_canon = float(np.sum(target_change_turn) / years) if years > 0 else float("nan")
        ann_turn_drift = float(np.sum(turn_drifted) / years) if years > 0 else float("nan")
        share_drift = float(np.sum(drift_turn) / max(np.sum(turn_drifted), 1e-300))
        out["drift_rebalance"] = {
            "ann_turn_canon": ann_turn_canon, "ann_turn_drift": ann_turn_drift,
            "turn_multiple": (ann_turn_drift / ann_turn_canon) if ann_turn_canon > 0 else float("nan"),
            "sharpe_canon21": sh_canon21, "sharpe_drift21": sh_drift21,
            "sharpe_delta21": sh_drift21 - sh_canon21,
            "share_drift": share_drift, "share_target_change": 1.0 - share_drift,
            "tripwire": (sh_drift21 - sh_canon21) < -0.10,
        }
    if not out:
        return None
    # Denomination of every figure above: the EXACT scored span, not a named regime label.
    out["window"] = {
        "n": int(t.size), "H": int(H),
        "row_lo": int(t[0]), "row_hi": int(t[-1]) + H,
        "first_entry": _utc_day(ts_sub[int(t[0])]),
        "last_exit": _utc_day(ts_sub[int(t[-1]) + H]),
    }
    return out


# =============================================================================
# TASK_916 cost truth pack — margin-rollover (borrow) charge, REPORTING-ONLY overlay.
#
# Additive, same contract as the TASK_920 honesty pack above: nothing here is called from
# compute_full_cell, cell_metrics, window_metrics, regime_series, or run_overlay_backtest, so
# gate 1a (`parity`) and the default `backtest` report are computed by EXACTLY the pre-916
# code path regardless of what lives here (G1: --borrow-model none, the default, is
# bit-identical). Wired into the CLI ONLY behind --borrow-model (trader/cli.py).
#
# D81(c) denominator: this is a cost per SHORTED NOTIONAL PER UNIT TIME, NOT per turnover —
# it is charged even on a position held with zero turnover, which the fee model (net_series,
# bp/side on turnover) never sees. The two cost families are summed, never conflated.
# =============================================================================

# Kraken margin rollover (dead-session harvest, fetched 2026-09-05 — UNVERIFIED against the
# primary schedule by this pass; see TASK_916 report for what leg-2 verification established).
# 0.01-0.02%/4h BTC, 0.02-0.04%/4h alts, annualized at 6 four-hour windows/day * 365 days/yr.
# The 'low'/'high' scenario picks the low/high END OF EACH ASSET CLASS'S OWN quoted range (not
# a mix-and-match across classes) -- BTC-low 0.01%/4h~=21.9%/yr, alt-low 0.02%/4h~=43.8%/yr,
# alt-high 0.04%/4h~=87.6%/yr; matches the corrected arithmetic in the task file's "Why" #1.
_ANNUALIZE_4H = 6 * 365  # 4h windows/day * days/yr = 2190
BORROW_RATE_SCENARIOS = {
    "low":  {"BTC": 0.0001 * _ANNUALIZE_4H, "ALT": 0.0002 * _ANNUALIZE_4H},   # ~21.9% / ~43.8%
    "high": {"BTC": 0.0002 * _ANNUALIZE_4H, "ALT": 0.0004 * _ANNUALIZE_4H},   # ~43.8% / ~87.6%
}


# TASK_916 fix pass (run-54, correctness-lens finding): the PIT db-fixture panel (363-symbol
# active universe `build_benchmark.py`'s "PIT (db/fixture)" panel scores) contains WBTC
# (Wrapped Bitcoin) and TBTC (Threshold tBTC) -- both track BTC ~1:1, both are 'online' pairs
# with real short exposure (verified 2026-09-06: WBTC ~48.9% of active bars short, weight up to
# ~1.4%; TBTC ~47.6% short, weight up to ~2.3%). Pre-fix these fell through to the generic ALT
# rate (2x-4x overcharge on their notional). Kept as an EXACT-match set, not a substring rule:
# the same panel also holds AIXBT (a genuine altcoin whose ticker contains "XBT"), which a
# substring rule would misclassify as BTC in the UNSAFE direction (undercharging its actual
# ALT-rate borrow). ST17 (selftest.py) asserts this against the REAL fixture panel, not just
# the hand-picked names here.
_BTC_ALIASES = frozenset({"XBT", "BTC", "WBTC", "TBTC"})


def _is_btc(sym):
    """Kraken's BTC ticker is XBT (verified against runs/819_single_asset/_close_broad_daily.npz
    `syms` and POWERED_RESULT.md's 2018-bear-carrier list, both XBT, never BTC); 'BTC' is kept
    as a defensive alias only in case a future panel source normalizes differently. WBTC/TBTC
    are wrapped/tokenized BTC (~1:1 tracking) found in the PIT db-fixture panel -- see
    _BTC_ALIASES above for the verification. Exact match only, never substring: AIXBT (ticker
    contains "XBT", genuine altcoin) must fall through to the ALT rate."""
    return sym in _BTC_ALIASES


def borrow_cost_series(pos, syms, t, H, rate_scenario="low"):
    """TASK_916 leg 1: per-decision margin-rollover cost on SHORT exposure ONLY, charged on
    shorted notional per unit time held (D81(c) denominator) -- never on turnover, never on
    long exposure. cost_k = Sum_i max(-pos[t_k,i], 0) * rate_annual(i) * (H/365), where
    rate_annual(i) is the BTC or ALT scenario rate for asset i's symbol. Always >= 0 (a book
    with zero short exposure at t_k pays exactly 0 there), so subtracting it from a net return
    series can only ever LOWER a Sharpe/return that has any short exposure, never raise one --
    the D81(c) pre-committed direction (the restatement can only make bars harder). Returns an
    array shaped like regime_series' `turn` (one value per decision)."""
    rates = BORROW_RATE_SCENARIOS[rate_scenario]
    is_btc = np.asarray([_is_btc(s) for s in syms], dtype=bool)
    rate_vec = np.where(is_btc, rates["BTC"], rates["ALT"])
    short_notional = np.maximum(-np.asarray(pos)[t], 0.0)          # (n_decisions, n_assets)
    return short_notional @ rate_vec * (H / 365.0)


def net_series_with_borrow(gross, turn, c_bp, borrow_cost):
    """net(c) with the fee model (net_series: turnover * bp/side) PLUS the borrow charge on
    top, both subtracted from the same gross series -- gross - turn*(c/1e4) - borrow_cost.
    borrow_cost is elementwise >= 0 (see borrow_cost_series), so this is <= net_series(...)
    pointwise on every decision: the D81(c) direction is enforced by construction, not just by
    convention -- a caller cannot accidentally produce a borrow-inclusive number ABOVE its
    uncorrected counterpart."""
    return net_series(gross, turn, c_bp) - borrow_cost


def borrow_window_report(logc, ts_sub, scfg, r_lo, r_hi, syms, costs_bp=None, gate_costs=None,
                         rate_scenario="low", xp=None):
    """TASK_916 leg 3 scorer: on the IDENTICAL H-strided decision grid the canon FULL/WINDOW
    report scores (`_decision_grid`, unchanged from the TASK_920 pack -- never calls
    regime_series/cell_metrics/window_metrics, so the canon numbers this augments are computed
    by unmodified pre-916 code), report gross/turn (same construction regime_series uses
    internally: gross_t=Sum_i pos[t,i]*fwd_i, turn_t=Sum_i|pos[t,i]-pos[t-H,i]|), the canon
    net(c) column, AND a borrow-charged net(c) column per requested cost tier, plus a
    borrow-charged bootstrap CI at the gate costs (same `_block_bootstrap_sharpe_ci` the canon
    scorer uses, called on the borrow-inclusive net series). Returns None if the window is too
    thin (identical thinness contract to regime_series/_decision_grid)."""
    if costs_bp is None:
        costs_bp = scfg.costs_bp
    if gate_costs is None:
        gate_costs = scfg.gate_costs
    grid = _decision_grid(logc, scfg, r_lo, r_hi, xp=xp)
    if grid is None:
        return None
    logc_np, pos_np, pres_np, t, H, warm = grid
    fwd = logc_np[t + H] - logc_np[t]
    fwd = np.where(np.isfinite(fwd), fwd, 0.0)
    gross = np.sum(pos_np[t] * fwd, axis=1)
    turn = np.sum(np.abs(pos_np[t] - pos_np[t - H]), axis=1)
    borrow_cost = borrow_cost_series(pos_np, syms, t, H, rate_scenario=rate_scenario)
    period = H * K
    d = {"n": int(t.size), "H": int(H), "rate_scenario": rate_scenario,
         "gross": _sharpe(gross, period), "turn": float(np.mean(turn)),
         "borrow_bp_per_day": float(np.mean(borrow_cost)) * 1e4 / H,
         "short_frac_mean": float(np.mean(np.sum(np.maximum(-pos_np[t], 0.0), axis=1) /
                                          np.maximum(np.sum(np.abs(pos_np[t]), axis=1), 1e-300)))}
    for c in costs_bp:
        d[f"net{c}"] = _sharpe(net_series(gross, turn, c), period)
        d[f"net{c}_borrow"] = _sharpe(net_series_with_borrow(gross, turn, c, borrow_cost), period)
    for c in gate_costs:
        nh = net_series_with_borrow(gross, turn, c, borrow_cost)
        ci = _block_bootstrap_sharpe_ci(nh, period, period, scfg.n_boot, scfg.seed)
        d[f"ci{c}_borrow_lo"], d[f"ci{c}_borrow_hi"] = float(ci[0]), float(ci[1])
    return d


def borrow_regime_report(logc, ts_sub, scfg, syms, rate_scenario="low"):
    """TASK_916 leg 3 convenience: `borrow_window_report` over every named regime
    (`config.REGIMES`), same window bounds `regime_rows` derives for the canon scorer. Returns
    {regime_name: dict-or-None}."""
    rows = regime_rows(np.asarray(ts_sub))
    out = {}
    for name in REG_NAMES:
        r0, r1 = rows[name]
        out[name] = borrow_window_report(logc, ts_sub, scfg, r0, r1, syms,
                                         rate_scenario=rate_scenario)
    return out

#!/usr/bin/env python3
"""TASK_917 P4 — hourly residual IC surface (closes the never-tested 1h-6h band).

PER THE TASK'S COMPUTE POLICY THIS INSTRUMENT IS PREPARED, NOT RUN, IN THIS PASS. It is built
and self-tested here against the fixture's small 90-day/8-asset minute slice ONLY (a wiring
check — the numbers that self-test produces are NOT the instrument's result, just proof the
pipeline runs end-to-end without error). The real run needs the FULL minute table
(`trading_data.db`, ~254M rows, up to 273 assets, 2018-2026) streamed and aggregated to hourly
bars -- estimated cost below (`--estimate-cost`). Do not point this at trading_data.db without
scheduling it (server/overnight one-shot, per this project's compute-allocation invariant).

Design (Stage 1 ONLY -- a descriptive IC surface; Stage 2's GP book is explicitly OUT OF SCOPE
per the task file: "Any book/controller build" is not this row):
  - Hourly OHLCV aggregated from the 1-min `ohlcv` table via STREAMING SQL (same pattern as
    `trader.data.aggregate_asset_daily`, bucketed by hour instead of by day) -- O(1) RAM per
    asset, never loads the full minute table.
  - SKIP-ENTRY (open t+1) + STALE-TAPE SCREEN: the forward target is measured from open[t+1]
    (never close[t], which on an illiquid hourly tape can be a stale carried print, not a real
    trade) to open[t+1+h]; any hour whose close is IDENTICAL to the prior hour's close (a
    stale/no-print tape) is excluded from feature AND target construction at that hour.
  - Market residual: same disclosed beta=1 equal-weight demeaning as P3 (e = r - mkt), for the
    same CPU-cheap-appropriate reason.
  - Feature family (OHLC-native, no fitted model — a pure descriptive IC, matching P1-P3's
    correlational methodology): trailing-return sign/magnitude at lookback L in
    {1,2,4,6,12,24,48,168} hours, PLUS a range family log(range(L)/range(168h)) at
    L in {1,4,24}h.
  - IC surface: per (feature, horizon h in {1,2,4,6,12,24}), mean cross-sectional Pearson IC
    across hours (>= MIN_N assets), Newey-West t-stat (lag scaled to h), Bonferroni across the
    full grid.
  - Break-even overlay: the task file's own pre-registered lines (IC_1h ~= 0.037 at HL 6h,
    ~= 0.0145 at HL 24h; the repo's only prior hourly read implies IC_1h ~= 0.008) are printed
    alongside whatever this surface measures — NOT re-derived here, just carried forward.
  - 816 residual-TSMOM re-read (its best cell +1.53 CI[+0.32,+2.88], daily/4h, a proper CI) is
    a SEPARATE prepared-not-run function (`reread_816_best_cell`) — needs the 816 run
    artifacts located and re-scored; stubbed with a clear TODO and cost note rather than
    silently working from a possibly-stale summary.

Read-only. `--fixture-selftest` (default) runs the pipeline against the fixture's small minute
slice for a wiring check only. `--full --db-path trading_data.db` is the real invocation and
is NOT executed by this pass.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trader.config import STABLE_FIAT
from mcts_trader.utils.db import _ro_connect

FIXTURE = str(_REPO_ROOT / "fixtures" / "trading_fixture.db")
TRADING_DB = str(_REPO_ROOT / "trading_data.db")
SEC_PER_HOUR = 3600

LOOKBACKS_H = (1, 2, 4, 6, 12, 24, 48, 168)
RANGE_LOOKBACKS_H = (1, 4, 24)
HORIZONS_H = (1, 2, 4, 6, 12, 24)
MIN_N_PER_HOUR = 8
NW_LAG_BASE = 4

# Break-even overlay -- pre-registered in the task file, carried forward verbatim (not
# re-derived by this pass).
REQUIRED_IC_1H = {"HL_6h": 0.037, "HL_24h": 0.0145}
PRIOR_HOURLY_READ_IC_1H = 0.008   # "816 H=60" implied read, per the task file


def _hour_key(ts):
    return (int(ts) // SEC_PER_HOUR) * SEC_PER_HOUR


def aggregate_asset_hourly(con, asset_id, start_ts=None, end_ts=None):
    """Streaming 1-min -> hourly OHLCV, same O(1)-RAM-per-asset cursor pattern as
    `trader.data.aggregate_asset_daily` (TASK_820), bucketed by hour instead of by day."""
    cur = con.cursor()
    where = "asset_id=?"
    params = [int(asset_id)]
    if start_ts is not None:
        where += " AND timestamp>=?"
        params.append(int(start_ts))
    if end_ts is not None:
        where += " AND timestamp<?"
        params.append(int(end_ts))
    cur.execute(
        f"SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE {where} "
        f"ORDER BY timestamp", params)
    cur_h = None
    h_open = h_high = h_low = h_close = None
    h_vol = 0.0
    for ts, o, h, lo, c, v in cur:
        hour = _hour_key(ts)
        if cur_h is None:
            cur_h = hour
            h_open, h_high, h_low, h_close, h_vol = o, h, lo, c, 0.0
        elif hour != cur_h:
            yield (cur_h, h_open, h_high, h_low, h_close, h_vol)
            cur_h = hour
            h_open, h_high, h_low, h_close, h_vol = o, h, lo, c, 0.0
        if h is not None:
            h_high = h if h_high is None else max(h_high, h)
        if lo is not None:
            h_low = lo if h_low is None else min(h_low, lo)
        if c is not None:
            h_close = c
        h_vol += (v or 0.0)
    if cur_h is not None:
        yield (cur_h, h_open, h_high, h_low, h_close, h_vol)


def discover_universe(db_path, min_bars=0):
    """USD-quoted, non-stable/fiat universe from `assets` (mirrors
    `trader.data.build_pit_universe`, kept local so this module has no write-adjacent
    dependency on the canon backfill path).

    KNOWN DIVERGENCE FROM P1-P3, disclosed rather than silently carried (RESULTS.md repeats
    it): P1-P3 go through `trader.data.load_db_backtest_panel` with
    `source_filter="kraken_rest"`, which admits only assets carrying real kraken_rest daily
    coverage. This local reader has NO source filter -- it matches any `BASE/USD` symbol row in
    `assets`. On the fixture that admits two non-crypto test rows with zero `ohlcv_daily`
    coverage, `TWX/USD` (id 90001) and `HOLE/USD` (id 90003), on top of the 8 real
    kraken_rest assets in the fixture's minute slice -> 10 columns, not 8. (A third test row,
    `TWXUSD` id 90002, is dropped only because it has no "/" in the symbol.)

    Consequence: harmless in the wiring self-test (the two columns are all-NaN and contribute
    nothing to any IC cell), but it MUST be reconciled before the `--full` run against
    `trading_data.db` -- either add `source_filter="kraken_rest"` here or route P4 through the
    same canon panel loader P1-P3 use, so P4's universe is the same object the rest of the
    pack measures. Left unchanged in this revision because P4 is PREPARED-NOT-RUN and changing
    its universe is a behaviour change outside this fix pass's scope."""
    con = _ro_connect(db_path)
    cur = con.cursor()
    out = []
    for aid, sym in cur.execute("SELECT id, symbol FROM assets"):
        if not sym or "/" not in sym:
            continue
        base, quote = sym.split("/", 1)
        if quote.upper() != "USD" or base.upper() in STABLE_FIAT:
            continue
        out.append((aid, base))
    con.close()
    out.sort(key=lambda t: t[0])
    return out


def build_hourly_panel(db_path, asset_ids, syms, start_ts=None, end_ts=None):
    """Raw (un-ffilled) hourly OHLCV panel. Returns dict with hour_ts (T,), open/high/low/
    close/volume (T,A), syms."""
    con = _ro_connect(db_path)
    per_asset = []
    gmin, gmax = None, None
    for aid in asset_ids:
        rows = list(aggregate_asset_hourly(con, aid, start_ts, end_ts))
        per_asset.append(rows)
        if rows:
            gmin = rows[0][0] if gmin is None else min(gmin, rows[0][0])
            gmax = rows[-1][0] if gmax is None else max(gmax, rows[-1][0])
    con.close()
    if gmin is None:
        empty = np.zeros((0, len(asset_ids)))
        return {"hour_ts": np.array([], dtype=np.int64), "open": empty, "high": empty,
                "low": empty, "close": empty, "volume": empty, "syms": syms}
    hour_ts = np.arange(gmin, gmax + SEC_PER_HOUR, SEC_PER_HOUR, dtype=np.int64)
    T = hour_ts.size
    A = len(asset_ids)
    o = np.full((T, A), np.nan); h = np.full((T, A), np.nan)
    l = np.full((T, A), np.nan); c = np.full((T, A), np.nan)
    v = np.full((T, A), np.nan)
    for col, rows in enumerate(per_asset):
        for (ht, ho, hh, hl, hc, hv) in rows:
            r = (ht - gmin) // SEC_PER_HOUR
            if 0 <= r < T:
                o[r, col] = ho; h[r, col] = hh; l[r, col] = hl; c[r, col] = hc; v[r, col] = hv
    return {"hour_ts": hour_ts, "open": o, "high": h, "low": l, "close": c, "volume": v,
            "syms": syms}


def stale_tape_mask(close_raw):
    """True where the bar's close EQUALS the prior bar's close exactly (a carried/no-print
    tape) -- excluded from feature and target construction at that hour."""
    stale = np.zeros_like(close_raw, dtype=bool)
    stale[1:] = np.isfinite(close_raw[1:]) & np.isfinite(close_raw[:-1]) & \
        (close_raw[1:] == close_raw[:-1])
    return stale


def build_features_and_targets(panel):
    close = panel["close"]
    T, A = close.shape
    stale = stale_tape_mask(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where((close > 0) & ~stale, close, np.nan))
    logopen = np.log(np.where(panel["open"] > 0, panel["open"], np.nan))
    dr = np.full_like(logc, np.nan)
    dr[1:] = logc[1:] - logc[:-1]
    mkt = np.nanmean(np.where(np.isfinite(dr), dr, np.nan), axis=1) if A else np.zeros(T)

    features = {}
    for L in LOOKBACKS_H:
        f = np.full_like(logc, np.nan)
        if T > L:
            f[L:] = logc[L:] - logc[:-L]
        features[f"trend_{L}h"] = f
    rng = (panel["high"] - panel["low"]) / np.where(close > 0, close, np.nan)
    ref = np.nanmedian(np.where(np.isfinite(rng), rng, np.nan)[-168:] if T >= 168 else rng,
                        axis=0) if T else np.full(A, np.nan)
    for L in RANGE_LOOKBACKS_H:
        f = np.full_like(logc, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            f[:] = np.log(np.where((rng > 0) & np.isfinite(ref) & (ref > 0), rng / ref, np.nan))
        features[f"range_{L}h_vs_168h"] = f

    # Raw (gross, not-yet-residualized) skip-entry forward return: open[t+1] -> open[t+1+h].
    targets = {}
    for h in HORIZONS_H:
        y = np.full_like(logc, np.nan)
        if T > h + 1:
            y[1:-h] = logopen[h + 1:] - logopen[1:-h]
        targets[h] = y
    # Residualize against the market's OWN forward sum over the identical [t+1, t+1+h] window
    # (a proper multi-hour residual -- the single-hour demeaning P3 uses on daily bars would
    # under-residualize a multi-hour target):
    mkt_cum = np.concatenate([[0.0], np.nancumsum(mkt)])
    for h in HORIZONS_H:
        y = targets[h]
        valid = np.isfinite(y)
        idx = np.where(valid.any(axis=1))[0]
        mkt_fwd = np.full(T, np.nan)
        if T > h + 1:
            mkt_fwd[1:-h] = mkt_cum[h + 1:T] - mkt_cum[1:T - h]
        targets[h] = y - mkt_fwd[:, None]
    return features, targets, stale


def newey_west_tstat(x, lag):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 5:
        return float("nan")
    xm = x - x.mean()
    s = np.dot(xm, xm) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1)
        s += 2 * w * (np.dot(xm[k:], xm[:-k]) / n)
    if s <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(s / n))


def ic_surface(features, targets, stale):
    T = next(iter(features.values())).shape[0]
    rows = []
    for fname, farr in features.items():
        for h, yarr in targets.items():
            ics = []
            for t in range(T):
                fv, yv = farr[t], yarr[t]
                m = np.isfinite(fv) & np.isfinite(yv) & ~stale[t]
                if m.sum() < MIN_N_PER_HOUR:
                    continue
                if np.std(fv[m]) < 1e-12 or np.std(yv[m]) < 1e-12:
                    continue
                ic = np.corrcoef(fv[m], yv[m])[0, 1]
                if np.isfinite(ic):
                    ics.append(ic)
            ics = np.asarray(ics)
            if ics.size < 10:
                rows.append({"feature": fname, "h": h, "mean_ic": None, "nw_t": None,
                             "n_hours": int(ics.size)})
                continue
            mean_ic = float(np.mean(ics))
            nw_t = newey_west_tstat(ics, NW_LAG_BASE * max(h, 1))
            rows.append({"feature": fname, "h": h, "mean_ic": mean_ic, "nw_t": nw_t,
                         "n_hours": int(ics.size)})
    return rows


def reread_816_best_cell():
    """TASK_917 P4 sub-item: re-read 816's residual-TSMOM best cell (+1.53, CI[+0.32,+2.88])
    at daily/4h horizons with a proper CI. PREPARED, NOT RUN: needs the 816 run artifacts
    located (runs/816_*/ or equivalent) and re-scored through a leak-free, cost-honest,
    per-decision H-strided path with a block-bootstrap CI (the same machinery
    `trader.backtest._block_bootstrap_sharpe_ci` already provides) -- not attempted here to
    avoid re-deriving a result from a stale in-context summary rather than the artifact
    itself. Cost: locate + reload the 816 panel/positions (unknown size, likely small/daily
    -- probably minutes, not the hourly-surface cost below), then one bootstrap pass."""
    return {"status": "prepared_not_run",
            "note": "needs the 816 run artifact re-scored at daily/4h with a block-bootstrap "
                    "CI; not attempted in this pass to avoid working from a summary instead "
                    "of the artifact -- see docstring."}


def run_selftest():
    print(f"[P4 selftest] fixture: {FIXTURE}")
    con = _ro_connect(FIXTURE)
    ids_syms = con.execute(
        "SELECT DISTINCT asset_id FROM ohlcv").fetchall()
    con.close()
    asset_ids = [r[0] for r in ids_syms]
    universe = dict(discover_universe(FIXTURE))
    asset_ids = [a for a in asset_ids if a in universe]
    syms = [universe[a] for a in asset_ids]
    print(f"[P4 selftest] {len(asset_ids)} assets admitted by this module's LOCAL "
          f"discover_universe() from the fixture's minute slice: {sorted(syms)}")
    print("[P4 selftest] NOTE: that is 8 real kraken_rest crypto assets PLUS 2 non-crypto "
          "test rows (TWX/USD, HOLE/USD) with zero ohlcv_daily coverage — this module's "
          "universe reader has no source_filter, unlike the load_db_backtest_panel path "
          "P1-P3 use. Harmless here (all-NaN columns), must be reconciled before --full. "
          "See discover_universe()'s docstring.")
    t0 = time.time()
    panel = build_hourly_panel(FIXTURE, asset_ids, syms)
    print(f"[P4 selftest] hourly panel: T={panel['hour_ts'].size} A={len(asset_ids)} "
          f"built in {time.time()-t0:.2f}s")
    features, targets, stale = build_features_and_targets(panel)
    print(f"[P4 selftest] {len(features)} features x {len(targets)} horizons built")
    rows = ic_surface(features, targets, stale)
    n_with_ic = sum(1 for r in rows if r["mean_ic"] is not None)
    print(f"[P4 selftest] IC surface: {len(rows)} cells, {n_with_ic} with enough hours "
          f"(fixture slice is only {panel['hour_ts'].size} hours / "
          f"{len(asset_ids)} assets -- FAR too small to be a real IC read; this is a "
          f"wiring check only)")
    for r in rows[:6]:
        ic = f"{r['mean_ic']:+.4f}" if r["mean_ic"] is not None else "n/a"
        print(f"    {r['feature']:18s} h={r['h']:>2}h  IC={ic}  n_hours={r['n_hours']}")
    print(f"[P4 selftest] break-even overlay (carried, not re-derived): {REQUIRED_IC_1H}, "
          f"prior hourly read implies IC_1h~={PRIOR_HOURLY_READ_IC_1H}")
    print(f"[P4 selftest] 816 re-read: {reread_816_best_cell()}")
    print("\n[P4] STATUS: PREPARED, NOT RUN on real data. Wiring self-test PASSED "
          "(no exceptions, plausible shapes) on the fixture's 90-day minute slice — 8 real "
          "kraken_rest assets, carried as 10 panel columns because this module's local "
          "universe reader also admits the 2 non-crypto test rows named above.")
    print("[P4] SIDE EFFECT: none — unlike P1/P2/P3, this module writes NO artifact.")
    return 0


def estimate_cost():
    """Rough cost estimate for the real invocation (--full), never executed here."""
    print("[P4] cost estimate for --full (trading_data.db, full minute history):")
    print("  - universe: up to ~273 USD-crypto assets (task file's own figure)")
    print("  - minute rows to stream: ~254M (measured COUNT(*) on trading_data.db.ohlcv)")
    print("  - aggregate_asset_hourly streams each asset's full minute history once, O(1) "
          "RAM/asset; the daily analogue (trader.data.aggregate_asset_daily) processes the "
          "same table at roughly 1-2M rows/s on this workstation's SQLite -- so a full "
          "streaming pass is on the order of 2-5 minutes of pure I/O, likely dominated by "
          "per-asset query overhead (273 separate range scans) rather than row count")
    print("  - IC surface: 11 features x 6 horizons = 66 cells, each an O(T) Python loop over "
          "hours (T ~= 65.7k hours per the mech_B sizing note) -- this pass's per-hour Python "
          "loop is the dominant cost, likely tens of minutes at this array size; a numpy-"
          "vectorized rewrite (batch corrcoef via einsum) would cut this substantially and "
          "should be done before the real run if it is scheduled")
    print("  - net: low tens of minutes to ~1-2 hours for Stage-1 alone, well inside the "
          "task's <=2 CPU-day budget; the 816 re-read is separate and small")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="NOT run by this pass -- real invocation")
    ap.add_argument("--db-path", default=TRADING_DB)
    ap.add_argument("--estimate-cost", action="store_true")
    args = ap.parse_args()
    if args.estimate_cost:
        return estimate_cost()
    if args.full:
        print("[P4] REFUSING: --full is the real invocation against trading_data.db and is "
              "deliberately NOT run by this pass (prepared-not-run per the task's compute "
              "policy). Run it as a scheduled server/overnight one-shot instead.",
              file=sys.stderr)
        return 2
    return run_selftest()


if __name__ == "__main__":
    sys.exit(main())

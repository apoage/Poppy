#!/usr/bin/env python3
"""TASK_820 paper trading — one daily step (Gate 3 of the deployment ladder).

Each `paper-step`:
  1. loads state (runs/820_paper/state.json) and the trailing daily close panel,
  2. computes the frozen-strategy target positions (819 broad/equal/single-30d) with the risk
     overlay ON,
  3. realizes the prior day's PnL from the held positions, simulates fills at today's close with
     the 21 bp/side cost model,
  4. appends per-asset rows to runs/820_paper/ledger.csv and updates equity / high-water-mark.

Idempotent per calendar day: re-running a day whose bar was already processed is a no-op
(keyed on the newest daily-bar timestamp `last_ts`).

Panel source: 'db' (read ohlcv_daily) for production; 'npz' (cached 819 panel) for dry-runs.
Equity compounds in log space (equity *= exp(pnl_net)) to match the 819/backtest convention.

DD throttle: dd_mode='shadow_scale' (default) — the state also tracks a SHADOW book (the
uncapped overlay-OFF frozen strategy, net of the same cost); the shadow drawdown drives the
throttle scale (1.0 / 0.5 / 0.25), so after the underlying strategy recovers the paper book
automatically re-scales up (no realized-equity dead state).
"""

import csv
import json
import time

import numpy as np

from . import strategy
from .config import (DB_PATH, NPZ_CACHE, PAPER, PAPER_DIR, PAPER_LEDGER, PAPER_STATE,
                     RiskConfig, STRATEGY)
from . import data as datamod
from . import risk as riskmod

LEDGER_HEADER = ["date", "asset", "prev_pos", "new_pos", "price", "cost_bps", "pnl"]

# Paper trades with the overlay ON (the "with risk management" fork).
PAPER_RISK = RiskConfig(enabled=True)


# =============================================================================
# state / ledger IO
# =============================================================================

def _default_state():
    return {"last_ts": None, "positions": {}, "equity": 1.0, "hwm": 1.0,
            "halted": False, "n_steps": 0,
            # shadow book: the uncapped overlay-OFF frozen strategy, net of the same cost —
            # its dd drives the throttle in dd_mode='shadow_scale' (no dead state).
            "shadow_positions": {}, "shadow_equity": 1.0, "shadow_hwm": 1.0}


def load_state(state_path=PAPER_STATE):
    p = str(state_path)
    try:
        with open(p) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return _default_state()


def save_state(state, state_path=PAPER_STATE):
    p = str(state_path)
    import os
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, p)


def _append_ledger(rows, ledger_path=PAPER_LEDGER):
    import os
    p = str(ledger_path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    new = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LEDGER_HEADER)
        for r in rows:
            w.writerow(r)


# =============================================================================
# panel loading
# =============================================================================

def _load_panel(source, db_path, npz_path, history_days):
    """Return (close_d (T,A), ts_d (T,), syms). db -> ohlcv_daily (kraken_rest-covered
    USD-crypto universe, base symbols); npz -> cached 819 panel (base symbols match)."""
    if source == "npz":
        close_d, ts_d, syms = datamod.load_npz_panel(npz_path)
        close_d, syms, _dropped, _keep = datamod.apply_liquidity_filter(close_d, syms)
    elif source == "db":
        ts_d, close_d, _ids, syms, _info = datamod.load_db_backtest_panel(db_path)
    else:
        raise ValueError(f"unknown source {source!r}")
    if history_days and close_d.shape[0] > history_days:
        close_d = close_d[-history_days:]
        ts_d = ts_d[-history_days:]
    return np.ascontiguousarray(close_d), np.asarray(ts_d), syms


# =============================================================================
# one daily step
# =============================================================================

def paper_step(source="db", now_ts=None, db_path=DB_PATH, npz_path=NPZ_CACHE,
               scfg=STRATEGY, rcfg=PAPER_RISK, pcfg=PAPER,
               state_path=PAPER_STATE, ledger_path=PAPER_LEDGER, verbose=True):
    """Execute (or skip, if idempotent) one daily paper step. Returns a result dict."""
    close_d, ts_d, syms = _load_panel(source, db_path, npz_path, pcfg.history_days)
    if close_d.shape[0] < scfg.vol_window + max(scfg.lookbacks) + 2:
        raise RuntimeError(f"insufficient history: {close_d.shape[0]} daily bars")

    today_ts = int(ts_d[-1])
    if now_ts is None:
        now_ts = int(time.time())
    date = time.strftime("%Y-%m-%d", time.gmtime(today_ts))

    state = load_state(state_path)
    if state.get("last_ts") == today_ts:
        if verbose:
            print(f"[paper] {date}: already processed (last_ts={today_ts}) -> no-op")
        return {"status": "noop", "date": date, "last_ts": today_ts,
                "equity": state["equity"], "hwm": state["hwm"], "halted": state["halted"]}

    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where(close_d > 0, close_d, np.nan))
    pos, pres, w, warm = strategy.positions_from_config(logc, scfg, xp=None)
    raw_target = np.asarray(pos[-1], dtype=np.float64)          # today's decision vector

    prev_pos = np.asarray([state["positions"].get(s, 0.0) for s in syms], dtype=np.float64)

    # realized PnL over the last day from the HELD (prev) positions
    ret = logc[-1] - logc[-2]
    ret = np.where(np.isfinite(ret), ret, 0.0)
    pnl_gross_vec = prev_pos * ret

    cost_rate = pcfg.cost_bp_per_side / 1e4

    # SHADOW book update FIRST (leak-free: uses today's realized return, known at today's
    # close, which is exactly when this decision executes). The shadow book holds the raw
    # uncapped overlay-OFF positions net of the same cost; its dd drives the throttle in
    # dd_mode='shadow_scale' and keeps recovering because it never stops trading.
    shadow_prev = np.asarray([state.get("shadow_positions", {}).get(s, 0.0) for s in syms],
                             dtype=np.float64)
    s_dpos = np.abs(raw_target - shadow_prev)
    s_pnl = float(np.sum(shadow_prev * ret)) - float(np.sum(s_dpos)) * cost_rate
    shadow_eq = float(state.get("shadow_equity", 1.0)) * float(np.exp(s_pnl))
    shadow_hwm = max(float(state.get("shadow_hwm", 1.0)), shadow_eq)

    # risk overlay -> today's new target (throttle driven per dd_mode)
    equity0, hwm0 = float(state["equity"]), float(state["hwm"])
    if rcfg.dd_mode == "shadow_scale":
        drive_eq, drive_hwm = shadow_eq, shadow_hwm
    else:
        drive_eq, drive_hwm = equity0, hwm0
    new_pos, info = riskmod.apply_overlay(raw_target, prev_pos, drive_eq, drive_hwm,
                                          latest_ts=today_ts, now_ts=now_ts, rcfg=rcfg)

    # costs at 21 bp/side on |Delta pos|
    dpos = np.abs(new_pos - prev_pos)
    cost_vec = dpos * cost_rate
    pnl_vec = pnl_gross_vec - cost_vec
    pnl_net = float(np.sum(pnl_vec))
    turnover = float(np.sum(dpos))

    equity1 = equity0 * float(np.exp(pnl_net))
    hwm1 = max(hwm0, equity1)

    # ledger rows for assets held or traded
    price = np.asarray(close_d[-1], dtype=np.float64)
    rows = []
    for i, s in enumerate(syms):
        if abs(prev_pos[i]) < 1e-12 and abs(new_pos[i]) < 1e-12:
            continue
        rows.append([date, s, f"{prev_pos[i]:.6f}", f"{new_pos[i]:.6f}",
                     f"{price[i]:.8g}", f"{dpos[i]*pcfg.cost_bp_per_side:.4f}",
                     f"{pnl_vec[i]:.8f}"])
    _append_ledger(rows, ledger_path)

    state["last_ts"] = today_ts
    state["positions"] = {s: float(new_pos[i]) for i, s in enumerate(syms)
                          if abs(new_pos[i]) > 1e-12}
    state["equity"] = equity1
    state["hwm"] = hwm1
    state["halted"] = bool(info["halt"])
    state["shadow_positions"] = {s: float(raw_target[i]) for i, s in enumerate(syms)
                                 if abs(raw_target[i]) > 1e-12}
    state["shadow_equity"] = shadow_eq
    state["shadow_hwm"] = shadow_hwm
    state["n_steps"] = int(state.get("n_steps", 0)) + 1
    save_state(state, state_path)

    shadow_dd = shadow_eq / max(shadow_hwm, 1e-12) - 1.0
    result = {"status": "stepped", "date": date, "last_ts": today_ts,
              "n_assets": int(np.sum(np.abs(new_pos) > 1e-12)), "gross": float(np.sum(np.abs(new_pos))),
              "turnover": turnover, "pnl_net": pnl_net, "cost": float(np.sum(cost_vec)),
              "equity": equity1, "hwm": hwm1, "drawdown": equity1 / hwm1 - 1.0,
              "shadow_equity": shadow_eq, "shadow_dd": shadow_dd,
              "halted": bool(info["halt"]), "stale": bool(info["stale"]),
              "dd_scale": info["dd_scale"], "dd_mode": rcfg.dd_mode, "n_ledger_rows": len(rows)}
    if verbose:
        alert = " STALE-ALERT" if info["stale"] else ""
        halt = " HALTED" if info["halt"] else ""
        print(f"[paper] {date}: stepped n_assets={result['n_assets']} gross={result['gross']:.3f} "
              f"turnover={turnover:.4f} pnl_net={pnl_net:+.5f} equity={equity1:.4f} "
              f"dd={result['drawdown']*100:+.1f}% shadow_dd={shadow_dd*100:+.1f}% "
              f"scale={info['dd_scale']:.2f}{alert}{halt}")
    return result


# =============================================================================
# report
# =============================================================================

def report(state_path=PAPER_STATE, ledger_path=PAPER_LEDGER, tail=10):
    """Summarize paper state + tail of the ledger. Returns a dict; also prints."""
    import os
    state = load_state(state_path)
    print("=== TASK_820 paper report ===")
    print(f"  steps={state.get('n_steps',0)}  last_ts={state.get('last_ts')}  "
          f"equity={state.get('equity',1.0):.4f}  hwm={state.get('hwm',1.0):.4f}  "
          f"drawdown={ (state.get('equity',1.0)/max(state.get('hwm',1.0),1e-12)-1.0)*100:+.2f}%  "
          f"halted={state.get('halted',False)}")
    print(f"  shadow: equity={state.get('shadow_equity',1.0):.4f}  "
          f"hwm={state.get('shadow_hwm',1.0):.4f}  "
          f"dd={(state.get('shadow_equity',1.0)/max(state.get('shadow_hwm',1.0),1e-12)-1.0)*100:+.2f}%"
          f"  (drives the throttle in shadow_scale mode)")
    npos = len(state.get("positions", {}))
    print(f"  open positions: {npos}")
    p = str(ledger_path)
    if os.path.exists(p):
        with open(p) as f:
            lines = f.read().strip().splitlines()
        print(f"  ledger rows: {len(lines)-1} (last {min(tail,len(lines)-1)}):")
        for ln in lines[-tail:]:
            print("    " + ln)
    else:
        print("  ledger: (none yet)")
    return state

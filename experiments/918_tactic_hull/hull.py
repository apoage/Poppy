#!/usr/bin/env python3
"""TASK_918 hull search — best FIXED mixture (weights on the simplex) of the K=8 position
matrices, chosen ex-post on the full STEP-0 window at net@21 (log returns), turnover computed
on the MIXED book (so netting between e.g. trend-long and reversal-short is honest, per the
task file). Uses only `trader.backtest.regime_series`/`net_series`/`_sharpe` for the objective
-- no reimplemented scoring math, only mixture/search bookkeeping.

Search method (pre-registered in SPEC.md): (1) an EXHAUSTIVE coarse grid over the 7-simplex at
step 0.1 -- all compositions of 10 units into K=8 non-negative integer parts, C(17,7) = 19448
points -- scored by the fast net@21 objective; (2) local refinement via
`scipy.optimize.minimize` (SLSQP, equality constraint sum(w)=1, bounds [0,1]) initialized at
the coarse grid's argmax, maximizing the same objective. The reported hull is the refined
point; the coarse grid's own argmax is also reported (resolution disclosure).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trader import backtest as bt        # noqa: E402
from trader.config import K as K_BARS    # noqa: E402

GRID_STEP = 0.1
GRID_UNITS = int(round(1.0 / GRID_STEP))   # 10


def _compositions(total, parts):
    """All non-negative integer compositions of `total` into `parts` parts (stars and bars),
    yielded as tuples summing to `total`."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def mix_positions(weights, pos_stack):
    """pos_stack: (K,T,A). weights: (K,). Returns the mixed (T,A) book = sum_k w_k*pos_stack[k]."""
    return np.tensordot(np.asarray(weights, dtype=np.float64), pos_stack, axes=(0, 0))


def fast_net21_sharpe(weights, pos_stack, pres, warm, ts_sub, H, r_lo, r_hi):
    """Objective used by the search: net@21bp (log-summed gross), no CI, no borrow -- CI/borrow
    are computed once, after the search, on the chosen point (see `run_step0.py`)."""
    pos_mix = mix_positions(weights, pos_stack)
    res = bt.regime_series(_LOGC, pos_mix, pres, r_lo, r_hi, H, warm, xp=np)
    if res is None:
        return float("-inf")
    gross, turn, _bh = res
    period = H * K_BARS
    nh = bt.net_series(gross, turn, 21)
    sh = bt._sharpe(nh, period)
    return sh if np.isfinite(sh) else float("-inf")


_LOGC = None   # set by search_hull before any objective call (module-level to avoid re-passing
              # the (T,A) logc array through scipy's flat x-vector callback signature)


def search_hull(logc, pos_stack, pres, warm, ts_sub, H, start_ts, end_ts, tactic_names):
    """Coarse simplex grid (step 0.1) + SLSQP local refinement. Returns dict with grid_best
    (weights, sharpe), refined (weights, sharpe), and the full coarse-grid table (weights ->
    sharpe) for the resolution disclosure."""
    global _LOGC
    _LOGC = logc
    Kn = pos_stack.shape[0]
    ts_sub = np.asarray(ts_sub)
    T = ts_sub.size
    r_lo = int(np.searchsorted(ts_sub, start_ts)) if start_ts is not None else 0
    r_hi = int(np.searchsorted(ts_sub, end_ts)) if end_ts is not None else T

    def obj(w):
        return fast_net21_sharpe(w, pos_stack, pres, warm, ts_sub, H, r_lo, r_hi)

    grid_rows = []
    best_w, best_s = None, float("-inf")
    for comp in _compositions(GRID_UNITS, Kn):
        w = np.asarray(comp, dtype=np.float64) / GRID_UNITS
        s = obj(w)
        grid_rows.append((tuple(comp), s))
        if s > best_s:
            best_s, best_w = s, w

    def neg_obj(w):
        return -obj(w)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * Kn
    result = minimize(neg_obj, best_w, method="SLSQP", bounds=bounds, constraints=cons,
                      options={"maxiter": 200, "ftol": 1e-6})
    refined_w = np.clip(result.x, 0.0, None)
    refined_w = refined_w / max(refined_w.sum(), 1e-12)
    refined_s = obj(refined_w)
    if refined_s < best_s:
        # SLSQP is a local method against a piecewise/noisy objective; never report a
        # "refinement" that is worse than the coarse grid's own argmax.
        refined_w, refined_s = best_w, best_s

    return {
        "grid_step": GRID_STEP, "grid_n_points": len(grid_rows),
        "grid_best_weights": dict(zip(tactic_names, [float(x) for x in best_w])),
        "grid_best_net21_sharpe": float(best_s),
        "refined_weights": dict(zip(tactic_names, [float(x) for x in refined_w])),
        "refined_net21_sharpe": float(refined_s),
        "scipy_success": bool(result.success), "scipy_message": str(result.message),
    }


def equal_weight_blend(pos_stack):
    """The binding null (S0/C3, `tactics_gaps.md:34`): the FIXED equal-weight point on the
    simplex (1/K each), not searched/optimized."""
    Kn = pos_stack.shape[0]
    w = np.full(Kn, 1.0 / Kn)
    return w, mix_positions(w, pos_stack)


def oracle_single_best(per_arm_net21_log):
    """Ex-post oracle single-tactic selector: the single arm with the best net@21 (log) Sharpe
    over the whole window -- a degenerate one-hot point on the simplex, reported for
    completeness even though the grid search already visits every corner."""
    name = max(per_arm_net21_log, key=lambda k: (per_arm_net21_log[k] if np.isfinite(
        per_arm_net21_log[k]) else float("-inf")))
    return name, per_arm_net21_log[name]

#!/usr/bin/env python3
"""TASK_918 STEP-0 runner. Read SPEC.md FIRST (this file enforces the pre-registration: it
refuses to run if SPEC.md's sha256 does not match the one pinned in `PREREG.json`, writing
that pin on first use). Loads the panel (`loader.py`), builds the K=8 tactics (`tactics.py`),
scores every arm + the hull + the equal-weight blend + the 21d reversal book alone
(`scorer.py`/`hull.py`), computes SR0(8)/MDE95 (`stats.py`), evaluates the pre-registered kill,
and writes the result JSON. Per the task file's STEP-0 gate: if the kill fires, this script
stops there (walk-forward/Hedge/random-null/B0 are out of scope for this pass).

CPU-only; minutes. Run wrapped (`tools/safe_run.sh --cpu`) on gpu-server per the repo's
compute-allocation rule; never against `trading_data.db`.
"""
import argparse
import datetime as _dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import loader                                   # noqa: E402
import tactics                                  # noqa: E402
import scorer                                   # noqa: E402
import hull as hullmod                          # noqa: E402
import stats                                    # noqa: E402
from trader import backtest as bt               # noqa: E402

SPEC_PATH = _HERE / "SPEC.md"
PREREG_PATH = _HERE / "PREREG.json"

WINDOW_START = "2024-01-01"
WINDOW_END = "2026-07-01"     # half-open: includes all of 2026-06
N_DECISIONS_MONTHLY = 30      # exact number of calendar months in [WINDOW_START, WINDOW_END)
KILL_BAR = 0.97
N_ARMS = len(tactics.TACTIC_NAMES)


def spec_sha256():
    return hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()


def require_prereg(force=False):
    current = spec_sha256()
    if PREREG_PATH.exists():
        existing = json.loads(PREREG_PATH.read_text())
        if existing.get("spec_sha256") == current:
            return existing
        if not force:
            raise RuntimeError(
                f"SPEC.md changed since pre-registration ({existing.get('spec_sha256', '?')[:12]}"
                f"... -> {current[:12]}...). Re-run with --force-reregister only if deliberate.")
    reg = {"spec_sha256": current,
          "registered_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    PREREG_PATH.write_text(json.dumps(reg, indent=2) + "\n")
    return reg


def score_full_and_by_year(logc, pos, pres, ts_sub, warm, syms):
    full = scorer.score_arm(logc, pos, pres, ts_sub, warm, syms, H=1,
                            start_ts=bt._to_epoch(WINDOW_START), end_ts=bt._to_epoch(WINDOW_END),
                            n_decisions_monthly=N_DECISIONS_MONTHLY)
    by_year = {}
    for name, s_ts, e_ts in scorer.by_year_windows():
        months = round((e_ts - s_ts) / 86400.0 / 30.4368)
        by_year[name] = scorer.score_arm(logc, pos, pres, ts_sub, warm, syms, H=1,
                                         start_ts=s_ts, end_ts=e_ts,
                                         n_decisions_monthly=months)
    return {"FULL": full, "by_year": by_year}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE / "results" / "step0_result.json"))
    ap.add_argument("--force-reregister", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    prereg = require_prereg(force=args.force_reregister)

    panel = loader.load_panel()
    day_ts, close, ids, syms = panel["day_ts"], panel["close"], panel["ids"], panel["syms"]
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where(close > 0, close, np.nan))

    pos_arms, pres, warm, tac_diag = tactics.build_all(logc, syms)
    pos_stack = np.stack([pos_arms[n] for n in tactics.TACTIC_NAMES], axis=0)

    per_arm = {}
    for name in tactics.TACTIC_NAMES:
        per_arm[name] = score_full_and_by_year(logc, pos_arms[name], pres, day_ts, warm, syms)

    search = hullmod.search_hull(logc, pos_stack, pres, warm, day_ts, 1,
                                 bt._to_epoch(WINDOW_START), bt._to_epoch(WINDOW_END),
                                 tactics.TACTIC_NAMES)
    refined_w = np.asarray([search["refined_weights"][n] for n in tactics.TACTIC_NAMES])
    pos_hull = hullmod.mix_positions(refined_w, pos_stack)
    hull_scored = score_full_and_by_year(logc, pos_hull, pres, day_ts, warm, syms)

    blend_w, pos_blend = hullmod.equal_weight_blend(pos_stack)
    blend_scored = score_full_and_by_year(logc, pos_blend, pres, day_ts, warm, syms)

    per_arm_net21_full = {n: per_arm[n]["FULL"]["net_log"]["legacy21"]
                          if per_arm[n]["FULL"] else float("nan") for n in tactics.TACTIC_NAMES}
    oracle_name, oracle_sharpe = hullmod.oracle_single_best(per_arm_net21_full)

    sr0_8 = stats.sr0(N_ARMS, N_DECISIONS_MONTHLY)
    mde95 = stats.mde95(N_DECISIONS_MONTHLY)
    hull_ci_lo = hull_scored["FULL"]["ci_net21_log_lo"]
    hull_ci_hi = hull_scored["FULL"]["ci_net21_log_hi"]
    bootstrap_se = stats.bootstrap_se_from_ci(hull_ci_lo, hull_ci_hi)

    hull_net21 = hull_scored["FULL"]["net_log"]["legacy21"]
    kill_fired = bool(np.isfinite(hull_net21) and hull_net21 < KILL_BAR)

    out = {
        "task": "TASK_918", "revision": "step0",
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "spec_sha256": prereg["spec_sha256"],
        "spec_registered_at_utc": prereg["registered_at_utc"],
        "loader_provenance": {
            "fixture_sha256": panel["fixture_sha256"], "loader_sha256": panel["loader_sha256"],
            "seam_block": panel["seam_block"], "loader_info": panel["loader_info"],
        },
        "panel": {"n_assets": int(logc.shape[1]), "n_days": int(logc.shape[0]),
                  "day_ts_min": int(day_ts[0]), "day_ts_max": int(day_ts[-1])},
        "window": {"start": WINDOW_START, "end": WINDOW_END,
                  "n_decisions_monthly": N_DECISIONS_MONTHLY},
        "tactics_diag": tac_diag,
        "per_arm": per_arm,
        "hull_search": search,
        "hull_scored": hull_scored,
        "equal_weight_blend": {"weights": dict(zip(tactics.TACTIC_NAMES,
                                                   [float(x) for x in blend_w])),
                               "scored": blend_scored},
        "oracle_single_best": {"tactic": oracle_name, "net21_log_sharpe": oracle_sharpe},
        "stats": {"N_ARMS": N_ARMS, "n_decisions_monthly": N_DECISIONS_MONTHLY,
                 "sr0_N8": sr0_8, "mde95": mde95,
                 "bootstrap_se_from_hull_ci": bootstrap_se,
                 "hull_ci_net21_log": [hull_ci_lo, hull_ci_hi]},
        "kill": {"bar": KILL_BAR, "hull_net21_log_full": hull_net21,
                "hull_ci_net21_log_lo": hull_ci_lo, "hull_ci_net21_log_hi": hull_ci_hi,
                "kill_fired": kill_fired,
                "denomination": "net@21bp, log returns, no borrow -- the most favorable "
                                "honest reading (see SPEC.md S5 direction argument)"},
        "runtime_seconds": time.time() - t0,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str) + "\n")

    verdict = "KILL FIRED" if kill_fired else "NOT FIRED"
    print(f"STEP-0: hull net@21 = {hull_net21:+.4f} "
          f"(CI {hull_ci_lo:+.4f}..{hull_ci_hi:+.4f}, n={hull_scored['FULL']['n_daily_decisions']}"
          f", n_decisions_monthly={N_DECISIONS_MONTHLY}) vs bar +{KILL_BAR} -> {verdict}")
    print(f"[918] wrote {out_path}  ({time.time()-t0:.1f}s)")
    return 0 if kill_fired or True else 1


if __name__ == "__main__":
    sys.exit(main())

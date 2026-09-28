#!/usr/bin/env python3
"""TASK_918 -- POST-HOC execution-timing x cost-tier sensitivity.

POST-HOC: this script was added AFTER the STEP-0 real-data result was seen (fix-pass cycle 2,
2026-09-28). It is EXCLUDED from the pre-registered verdict. SPEC.md, PREREG.json and
results/step0_result.json are the pre-registered record and are never written here; this
script refuses to run if SPEC.md's sha256 no longer matches PREREG.json.

What it measures, for all 8 arms + the equal-weight blend + the ex-post oracle + a RE-OPTIMISED
hull, in each of two execution-timing conditions:
  * same_close -- the pre-registered STEP-0 timing: the position set at close t is paid
    logc[t+1]-logc[t] (`tactics.py` + `trader/backtest.py:regime_series`).
  * t_plus_1   -- every arm's position matrix delayed ONE extra day, pos_T1[t] = pos[t-1]
    (pos_T1[0] = 0). All arms are delayed (the mixed book is linear in the arms, so delaying
    the mixture == mixing the delayed arms). The present-mask is NOT shifted.
at gross / net21 / net40 / net80 (log returns, TASK_916 fee tiers from `scorer.FEE_TIERS_BP`)
and net80 + borrow (low / high, `trader.backtest.borrow_cost_series` via `scorer.score_arm`).

Two hulls per condition, both via the module's own UNMODIFIED `hull.search_hull` (coarse
simplex grid step 0.1 + SLSQP):
  * hull_net21_chosen  -- objective = the pre-registered net@21 (`hull.fast_net21_sharpe`).
  * hull_honest_chosen -- objective = net@80bp + borrow(low). Implemented by temporarily
    substituting `hull.fast_net21_sharpe` for the duration of that one call (the search
    algorithm itself is byte-identical; the objective is built only from the same
    `trader.backtest` primitives `scorer.score_arm` calls). A self-check asserts that the
    search's reported objective equals `scorer.score_arm`'s independently computed value at
    that tier for the chosen weights -- i.e. that the substitution actually took effect.

All scoring goes through `run_step0.score_full_and_by_year` -> `scorer.score_arm`. The only
new arithmetic here is the one-day delay, bookkeeping, and a block-bootstrap CI at the honest
tier for the hulls (the unmodified `trader.backtest._block_bootstrap_sharpe_ci`, same
N_BOOT/SEED as `scorer`).

Positive control (fail-closed): the same_close condition must reproduce step0_result.json's
hull net@21 and reversal_21d_xs net@21 to 1e-6, else this script exits non-zero without
writing. Judge reproduction (finding 3 of the cycle-1 panel, srv scratch
/var/tmp/918-judge-correctness/) is checked to 0.01 and REPORTED, never asserted.

CPU-only, run wrapped (`tools/safe_run.sh --cpu`) on gpu-server. Read-only on the fixture.
"""
import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import loader                                   # noqa: E402
import tactics                                  # noqa: E402
import scorer                                   # noqa: E402
import hull as hullmod                          # noqa: E402
import stats                                    # noqa: E402
import run_step0                                # noqa: E402
from trader import backtest as bt               # noqa: E402
from trader.config import K as K_BARS           # noqa: E402

RESULTS = _HERE / "results"
STEP0_PATH = RESULTS / "step0_result.json"
OUT_PATH = RESULTS / "sensitivity_posthoc.json"

HONEST_FEE_LABEL = "base_taker80"
HONEST_BORROW = "low"
POSITIVE_CONTROL_TOL = 1e-6
JUDGE_TOL = 0.01
# Cycle-1 correctness-judge numbers (srv /var/tmp/918-judge-correctness/judge_918*.py),
# quoted from the orchestrator brief -- compared, not trusted.
JUDGE_REF = {
    "reversal_21d_xs_t1_gross": 2.527,
    "reversal_21d_xs_t1_net21": 0.733,
    "trend_819_t1_net21": 0.399,
    "reversal_7d_ts_t1_net21": -0.553,
    "equal_weight_cm_t1_net21": -1.388,
    "hull_reversal_only_delayed_reopt_net21": 0.805,
}
TIERS = ("gross", "net21", "net40", "net80", "net80_borrow_low", "net80_borrow_high")
HONEST_TIER = "net80_borrow_low"

CODE_FILES = ["loader.py", "tactics.py", "scorer.py", "hull.py", "stats.py", "run_step0.py",
              "run_sensitivity_posthoc.py"]
TRADER_FILES = ["trader/backtest.py", "trader/strategy.py", "trader/data.py",
                "trader/config.py"]


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def delay_one_day(pos):
    """pos_T1[t] = pos[t-1]; pos_T1[0] = 0. Works on (T,A) or (K,T,A) along the time axis."""
    pos = np.asarray(pos, dtype=np.float64)
    out = np.zeros_like(pos)
    out[..., 1:, :] = pos[..., :-1, :]
    return out


def tiers_of(d):
    """Map a `scorer.score_arm` dict to the TIERS readout (None -> all NaN)."""
    if d is None:
        return {k: float("nan") for k in TIERS}
    return {
        "gross": d["gross_log_sharpe"],
        "net21": d["net_log"]["legacy21"],
        "net40": d["net_log"]["base_maker40"],
        "net80": d["net_log"]["base_taker80"],
        "net80_borrow_low": d["borrow"]["low"]["net_log_base_taker80"],
        "net80_borrow_high": d["borrow"]["high"]["net_log_base_taker80"],
    }


def compact(scored):
    """Compact per-book readout from `run_step0.score_full_and_by_year`'s output."""
    f = scored["FULL"]
    return {
        "tiers": tiers_of(f),
        "n_daily_decisions": f["n_daily_decisions"] if f else None,
        "n_decisions_monthly": f["n_decisions_monthly"] if f else None,
        "ci_net21_log": [f["ci_net21_log_lo"], f["ci_net21_log_hi"]] if f else None,
        "turn_mean": f["turn_mean"] if f else None,
        "by_year": {y: {"tiers": tiers_of(v),
                        "n_daily_decisions": v["n_daily_decisions"] if v else None,
                        "n_decisions_monthly": v["n_decisions_monthly"] if v else None}
                    for y, v in scored["by_year"].items()},
    }


def make_honest_objective(syms):
    """Drop-in replacement for `hull.fast_net21_sharpe` (same signature) at net@80 + borrow
    low. Same primitives, same order, as `scorer.score_arm`'s borrow block."""
    c_bp = scorer.FEE_TIERS_BP[HONEST_FEE_LABEL]

    def honest(weights, pos_stack, pres, warm, ts_sub, H, r_lo, r_hi):
        logc = hullmod._LOGC
        pos_mix = hullmod.mix_positions(weights, pos_stack)
        res = bt.regime_series(logc, pos_mix, pres, r_lo, r_hi, H, warm, xp=np)
        if res is None:
            return float("-inf")
        gross, turn, _bh = res
        t = scorer.h_grid(logc.shape[0], warm, r_lo, r_hi, H)
        bc = bt.borrow_cost_series(pos_mix, syms, t, H, rate_scenario=HONEST_BORROW)
        sh = bt._sharpe(bt.net_series_with_borrow(gross, turn, c_bp, bc), H * K_BARS)
        return sh if np.isfinite(sh) else float("-inf")
    return honest


@contextlib.contextmanager
def hull_objective(fn):
    orig = hullmod.fast_net21_sharpe
    hullmod.fast_net21_sharpe = fn
    try:
        yield
    finally:
        hullmod.fast_net21_sharpe = orig


def honest_tier_ci(logc, pos, pres, warm, day_ts, syms):
    """Block-bootstrap CI at net80+borrow_low over the FULL window (unmodified primitives)."""
    r_lo = int(np.searchsorted(day_ts, bt._to_epoch(run_step0.WINDOW_START)))
    r_hi = int(np.searchsorted(day_ts, bt._to_epoch(run_step0.WINDOW_END)))
    gross, turn, _ = bt.regime_series(logc, pos, pres, r_lo, r_hi, 1, warm, xp=np)
    t = scorer.h_grid(logc.shape[0], warm, r_lo, r_hi, 1)
    bc = bt.borrow_cost_series(pos, syms, t, 1, rate_scenario=HONEST_BORROW)
    nh = bt.net_series_with_borrow(gross, turn, scorer.FEE_TIERS_BP[HONEST_FEE_LABEL], bc)
    lo, hi = bt._block_bootstrap_sharpe_ci(nh, K_BARS, K_BARS, scorer.N_BOOT, scorer.SEED)
    return [float(lo), float(hi)]


def run_hull(label, objective_tier, obj_fn, logc, stack, pres, warm, day_ts, syms):
    w0 = bt._to_epoch(run_step0.WINDOW_START)
    w1 = bt._to_epoch(run_step0.WINDOW_END)
    ctx = hull_objective(obj_fn) if obj_fn is not None else contextlib.nullcontext()
    t0 = time.time()
    with ctx:
        search = hullmod.search_hull(logc, stack, pres, warm, day_ts, 1, w0, w1,
                                     tactics.TACTIC_NAMES)
    assert hullmod.fast_net21_sharpe.__name__ == "fast_net21_sharpe", "objective not restored"
    # hull.search_hull labels its objective values "*_net21_sharpe"; relabel to the objective
    # actually used so no consumer reads an honest-tier number as net@21.
    search = {k.replace("net21_sharpe", "objective_sharpe"): v for k, v in search.items()}
    search["objective_tier"] = objective_tier
    w = np.asarray([search["refined_weights"][n] for n in tactics.TACTIC_NAMES])
    pos = hullmod.mix_positions(w, stack)
    scored = run_step0.score_full_and_by_year(logc, pos, pres, day_ts, warm, syms)
    c = compact(scored)
    independent = c["tiers"][objective_tier]
    gap = abs(independent - search["refined_objective_sharpe"])
    # D83(i) guard: proves the objective the search optimised IS the tier we label it as.
    assert gap < 1e-9, (label, objective_tier, independent, search["refined_objective_sharpe"])
    c.update({"search": search, "objective_self_check_abs_gap": gap,
              "ci_net80_borrow_low_log": honest_tier_ci(logc, pos, pres, warm, day_ts, syms),
              "full_score": scored, "search_seconds": time.time() - t0})
    print(f"[918-posthoc] {label}: {objective_tier}-chosen hull -> "
          + " ".join(f"{k}={v:+.3f}" for k, v in c["tiers"].items())
          + f"  w={ {k: round(v, 3) for k, v in search['refined_weights'].items() if v > 1e-3} }",
          flush=True)
    return c


def run_condition(label, pos_arms, logc, pres, warm, day_ts, syms, honest_fn):
    stack = np.stack([pos_arms[n] for n in tactics.TACTIC_NAMES], axis=0)
    per_arm = {}
    for n in tactics.TACTIC_NAMES:
        per_arm[n] = compact(run_step0.score_full_and_by_year(logc, pos_arms[n], pres, day_ts,
                                                              warm, syms))
        print(f"[918-posthoc] {label} {n:>20s}: "
              + " ".join(f"{k}={v:+.3f}" for k, v in per_arm[n]["tiers"].items()), flush=True)
    blend_w, pos_blend = hullmod.equal_weight_blend(stack)
    blend = compact(run_step0.score_full_and_by_year(logc, pos_blend, pres, day_ts, warm, syms))
    oracle = {}
    for tier in ("net21", HONEST_TIER):
        name, val = hullmod.oracle_single_best({n: per_arm[n]["tiers"][tier]
                                                for n in tactics.TACTIC_NAMES})
        oracle[f"at_{tier}"] = {"tactic": name, "sharpe": val}
    return {
        "per_arm": per_arm,
        "equal_weight_blend": blend,
        "oracle": oracle,
        "hull_net21_chosen": run_hull(label, "net21", None, logc, stack, pres, warm, day_ts,
                                      syms),
        "hull_honest_chosen": run_hull(label, HONEST_TIER, honest_fn, logc, stack, pres, warm,
                                       day_ts, syms),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args(argv)
    t_start = time.time()

    prereg = json.loads(run_step0.PREREG_PATH.read_text())
    spec_sha = run_step0.spec_sha256()
    if prereg.get("spec_sha256") != spec_sha:
        raise SystemExit(f"SPEC.md sha {spec_sha} != PREREG.json {prereg.get('spec_sha256')}")
    step0 = json.loads(STEP0_PATH.read_text())

    panel = loader.load_panel()
    day_ts, close, syms = panel["day_ts"], panel["close"], panel["syms"]
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where(close > 0, close, np.nan))
    pos_arms, pres, warm, _diag = tactics.build_all(logc, syms)
    honest_fn = make_honest_objective(syms)

    conditions = {}
    conditions["same_close"] = run_condition("same_close", pos_arms, logc, pres, warm, day_ts,
                                             syms, honest_fn)

    # ---- positive control (fail-closed) -------------------------------------------------
    sc = conditions["same_close"]
    pc = {
        "hull_net21": {"step0": step0["kill"]["hull_net21_log_full"],
                       "posthoc": sc["hull_net21_chosen"]["tiers"]["net21"]},
        "reversal_21d_xs_net21": {
            "step0": step0["per_arm"]["reversal_21d_xs"]["FULL"]["net_log"]["legacy21"],
            "posthoc": sc["per_arm"]["reversal_21d_xs"]["tiers"]["net21"]},
        "hull_net80": {"step0": step0["hull_scored"]["FULL"]["net_log"]["base_taker80"],
                       "posthoc": sc["hull_net21_chosen"]["tiers"]["net80"]},
    }
    for v in pc.values():
        v["abs_diff"] = abs(v["step0"] - v["posthoc"])
        v["ok"] = bool(v["abs_diff"] < POSITIVE_CONTROL_TOL)
    print(f"[918-posthoc] positive control: {pc}", flush=True)
    if not all(v["ok"] for v in pc.values()):
        raise SystemExit("positive control FAILED -- same_close does not reproduce STEP-0")

    pos_t1 = {n: delay_one_day(p) for n, p in pos_arms.items()}
    conditions["t_plus_1"] = run_condition("t_plus_1", pos_t1, logc, pres, warm, day_ts, syms,
                                           honest_fn)

    # ---- judge reproduction (reported, not asserted) ------------------------------------
    t1 = conditions["t_plus_1"]["per_arm"]
    rev_only = dict(pos_arms)
    rev_only["reversal_21d_xs"] = pos_t1["reversal_21d_xs"]
    rev_only_hull = run_hull("reversal_only_delayed", "net21", None, logc,
                             np.stack([rev_only[n] for n in tactics.TACTIC_NAMES], axis=0),
                             pres, warm, day_ts, syms)
    measured = {
        "reversal_21d_xs_t1_gross": t1["reversal_21d_xs"]["tiers"]["gross"],
        "reversal_21d_xs_t1_net21": t1["reversal_21d_xs"]["tiers"]["net21"],
        "trend_819_t1_net21": t1["trend_819"]["tiers"]["net21"],
        "reversal_7d_ts_t1_net21": t1["reversal_7d_ts"]["tiers"]["net21"],
        "equal_weight_cm_t1_net21": t1["equal_weight_cm"]["tiers"]["net21"],
        "hull_reversal_only_delayed_reopt_net21": rev_only_hull["tiers"]["net21"],
    }
    judge = {k: {"judge": JUDGE_REF[k], "measured": measured[k],
                 "abs_diff": abs(JUDGE_REF[k] - measured[k]),
                 "within_tol": bool(abs(JUDGE_REF[k] - measured[k]) <= JUDGE_TOL)}
             for k in JUDGE_REF}
    print(f"[918-posthoc] judge reproduction: {judge}", flush=True)

    # ---- summary table ------------------------------------------------------------------
    table = []
    for cond, c in conditions.items():
        for n in tactics.TACTIC_NAMES:
            table.append({"condition": cond, "book": n, **c["per_arm"][n]["tiers"]})
        table.append({"condition": cond, "book": "equal_weight_blend",
                      **c["equal_weight_blend"]["tiers"]})
        for h in ("hull_net21_chosen", "hull_honest_chosen"):
            table.append({"condition": cond, "book": h, **c[h]["tiers"]})

    n_months = run_step0.N_DECISIONS_MONTHLY
    out = {
        "task": "TASK_918", "revision": "posthoc_sensitivity",
        "POSTHOC": ("Added after seeing STEP-0 data (fix-pass cycle 2). EXCLUDED from the "
                    "pre-registered verdict; see results/VERDICT.json for how the two combine."),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "spec_sha256": spec_sha, "prereg": prereg,
        "step0_result_sha256": sha256_file(STEP0_PATH),
        "code_sha256": {**{f: sha256_file(_HERE / f) for f in CODE_FILES},
                        **{f: sha256_file(_REPO_ROOT / f) for f in TRADER_FILES}},
        "fixture_sha256": panel["fixture_sha256"],
        "loader_seam_sha256": panel["seam_block"]["seam_sha256"],
        "window": {"start": run_step0.WINDOW_START, "end": run_step0.WINDOW_END,
                   "n_decisions_monthly": n_months},
        "timing_definitions": {
            "same_close": "pos[t] set at close t, paid logc[t+1]-logc[t] (pre-registered)",
            "t_plus_1": "pos_T1[t]=pos[t-1], pos_T1[0]=0, ALL arms; present-mask unshifted",
        },
        "tier_definitions": {
            "gross": "log, zero cost", "net21": "log, 21bp/side (legacy, the bar's own)",
            "net40": "log, 40bp/side (916 base maker)", "net80": "log, 80bp/side (916 base taker)",
            "net80_borrow_low": "net80 + trader.backtest.borrow_cost_series(low)",
            "net80_borrow_high": "net80 + trader.backtest.borrow_cost_series(high)",
        },
        "honest_tier": HONEST_TIER,
        "stats": {"N_ARMS": run_step0.N_ARMS, "n_decisions_monthly": n_months,
                  "sr0_N8": stats.sr0(run_step0.N_ARMS, n_months),
                  "mde95": stats.mde95(n_months),
                  "note": "SR0(8)/MDE95 are functions of N and n_decisions_monthly only; "
                          "identical for every condition/tier here"},
        "bar": {"value": run_step0.KILL_BAR,
                "denomination": "net@21 same-close (819's window). Comparing it to any other "
                                "tier/timing is a post-hoc reading, not a pre-registered gate."},
        "positive_control": pc,
        "judge_reproduction": judge,
        "judge_reproduction_hull_reversal_only_delayed": {
            k: rev_only_hull[k] for k in ("tiers", "search", "n_daily_decisions",
                                          "n_decisions_monthly", "ci_net21_log")},
        "conditions": conditions,
        "summary_table": table,
        "runtime_seconds": time.time() - t_start,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"[918-posthoc] wrote {args.out} ({time.time() - t_start:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

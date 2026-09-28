#!/usr/bin/env python3
"""TASK_918 -- writes results/VERDICT.json, the machine-readable verdict any consumer should
read. POST-HOC tooling (fix-pass cycle 2, added after seeing STEP-0 data). Every value is
sourced from results/step0_result.json (pre-registered record) and
results/sensitivity_posthoc.json (post-hoc, `run_sensitivity_posthoc.py`); nothing is typed in
by hand except the bar-denomination and cause-claim prose. The status string is DERIVED: it
only says POSTHOC_FRAGILE if at least one post-hoc re-optimised hull is below the bar.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RESULTS = _HERE / "results"
STEP0 = RESULTS / "step0_result.json"
SENS = RESULTS / "sensitivity_posthoc.json"
OUT = RESULTS / "VERDICT.json"


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    s0 = json.loads(STEP0.read_text())
    ps = json.loads(SENS.read_text())
    if ps["spec_sha256"] != s0["spec_sha256"]:
        raise SystemExit("spec sha mismatch between step0 and posthoc results")
    if ps["step0_result_sha256"] != _sha(STEP0):
        raise SystemExit("posthoc result was computed against a different step0_result.json")
    if not all(v["ok"] for v in ps["positive_control"].values()):
        raise SystemExit("posthoc positive control not OK")

    bar = s0["kill"]["bar"]
    kill_fired = bool(s0["kill"]["kill_fired"])
    c = ps["conditions"]
    ht = ps["honest_tier"]

    # The four re-optimised hulls: (timing, objective tier) -> value at that same tier.
    evaluated = {
        "same_close__net21": c["same_close"]["hull_net21_chosen"]["tiers"]["net21"],
        f"same_close__{ht}": c["same_close"]["hull_honest_chosen"]["tiers"][ht],
        "t_plus_1__net21": c["t_plus_1"]["hull_net21_chosen"]["tiers"]["net21"],
        f"t_plus_1__{ht}": c["t_plus_1"]["hull_honest_chosen"]["tiers"][ht],
    }
    fires = [k for k, v in evaluated.items() if not (math.isfinite(v) and v >= bar)]
    # "same_close__net21" is the pre-registered cell; it must agree with step0's own kill.
    if ("same_close__net21" in fires) != kill_fired:
        raise SystemExit("posthoc same_close/net21 cell disagrees with the pre-registered kill")

    if kill_fired:
        status = "PREREG_FIRED"
    elif fires:
        status = "PREREG_NOT_FIRED__POSTHOC_FRAGILE"
    else:
        status = "PREREG_NOT_FIRED__POSTHOC_ROBUST"

    verdict = {
        "task": "TASK_918",
        "preregistered": {
            "denomination": "net@21, same-close",
            "hull": s0["kill"]["hull_net21_log_full"],
            "hull_ci_net21": [s0["kill"]["hull_ci_net21_log_lo"],
                              s0["kill"]["hull_ci_net21_log_hi"]],
            "bar": bar,
            "kill_fired": kill_fired,
        },
        "posthoc_sensitivity": {
            "honest_tier": ht,
            "honest_tier_hull": evaluated[f"same_close__{ht}"],
            "net21_chosen_hull_at_net80": c["same_close"]["hull_net21_chosen"]["tiers"]["net80"],
            "t_plus_1_hull_net21": evaluated["t_plus_1__net21"],
            "t_plus_1_hull_honest_tier": evaluated[f"t_plus_1__{ht}"],
            "reoptimised_hulls_evaluated": evaluated,
            "kill_would_fire_under": fires,
            "bar_denomination_note": ("the +0.97 bar was measured at net@21 same-close on 819's "
                                      "window; reading it against other tiers/timings is "
                                      "post-hoc, not a pre-registered gate"),
        },
        "status": status,
        "cause_claim": "none — timing sensitivity measured; bid-ask-bounce NOT established",
        "decision_owner": "operator (DECISIONS row)",
        "sources": {
            "step0_result.json": {"sha256": _sha(STEP0)},
            "sensitivity_posthoc.json": {"sha256": _sha(SENS)},
            "spec_sha256": s0["spec_sha256"],
            "readme": "README.md (headline section)",
        },
    }
    OUT.write_text(json.dumps(verdict, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

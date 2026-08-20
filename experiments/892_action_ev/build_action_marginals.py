#!/usr/bin/env python3
"""TASK_892 — derive the --marginals-sidecar sidecar (eval_vector.py's
ACTION_MARGINALS_SCHEMA) from a subject trace (trace.py's output, or any JSON carrying a
``decisions.chosen_vectors`` (n_decisions, n_assets) array plus ``provenance.col_symbols``).

Per-asset marginal = the empirical distribution over the 5 action codes in that asset's column
of ``chosen_vectors`` (the ATTEMPT stream — the same convention TASK_861/878 already use for
"what the policy chose"). This is deliberately the SIMPLEST honest estimator: it draws the null
from exactly the subject's own observed per-asset rates, so a null built from it is
matched-by-construction (SPEC.md, gate ii) up to sampling noise.
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

# cycle-3 minor (non-atomic artifact writes): ONE definition of the atomic write, in trace.py.
import trace as trace_mod                                     # noqa: E402  (path set above)

EV_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "eval_vector.py")
N_ACTIONS_PER_ASSET = 5
ACTION_MARGINALS_SCHEMA = "892/action-marginals/v1"


def marginals_from_chosen_vectors(chosen_vectors, n_assets):
    """(n_decisions, n_assets) int codes -> (n_assets, 5) row-stochastic float64 array. An
    asset column with zero observations (n_decisions==0) gets a degenerate all-HOLD row (index
    1) rather than a NaN/uniform guess — "no evidence" defaults to the inert action, never to a
    fabricated trading rate."""
    chosen_vectors = np.asarray(chosen_vectors, dtype=np.int64)
    n_decisions = chosen_vectors.shape[0]
    out = np.zeros((n_assets, N_ACTIONS_PER_ASSET), dtype=np.float64)
    for a in range(n_assets):
        if n_decisions == 0:
            out[a, 1] = 1.0
            continue
        counts = np.bincount(chosen_vectors[:, a], minlength=N_ACTIONS_PER_ASSET)
        out[a] = counts / counts.sum()
    return out


def build_sidecar(trace_doc):
    prov = trace_doc["provenance"]
    symbols = prov["col_symbols"]
    n_assets = prov["n_assets"]
    chosen_vectors = trace_doc["decisions"]["chosen_vectors"]
    marginals = marginals_from_chosen_vectors(chosen_vectors, n_assets)
    return {
        "schema": ACTION_MARGINALS_SCHEMA,
        "symbols": list(symbols),
        "n_assets": int(n_assets),
        "marginals": marginals.tolist(),
        "provenance": {
            "source_trace_schema": trace_doc.get("schema"),
            "source_checkpoint_label": prov.get("checkpoint_label"),
            "source_decision_rule": prov.get("decision_rule"),
            "source_tier_bp": prov.get("tier_bp"),
            "source_eval_seed": prov.get("eval_seed"),
            "n_decisions": int(np.asarray(chosen_vectors).shape[0]),
            "note": "per-asset marginal = empirical distribution over action codes in the "
                    "source trace's chosen_vectors (the ATTEMPT stream); a null drawn from "
                    "this sidecar is matched to the source walk's activity/mix up to sampling "
                    "noise (SPEC.md gate ii), never exactly by construction of THIS function.",
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, help="a trace.py JSON (subject walk)")
    ap.add_argument("--out", required=True, help="sidecar JSON path")
    args = ap.parse_args()
    with open(args.trace) as fh:
        trace_doc = json.load(fh)
    sidecar = build_sidecar(trace_doc)
    # atomic: a kill mid-dump must never leave a truncated sidecar where load_action_marginals
    # expects a complete one (it would fail loud, but on the wrong error).
    trace_mod.write_json_atomic(args.out, sidecar, indent=2)
    print(f"[892/build_action_marginals] wrote {args.out} "
         f"(n_assets={sidecar['n_assets']}, source={args.trace})")


if __name__ == "__main__":
    main()

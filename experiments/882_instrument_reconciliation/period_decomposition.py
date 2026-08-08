#!/usr/bin/env python3
"""TASK_882 FIX 1 — is the imprint lane's in-window number carried by 2021?

The review challenge: LEDGER `861/ref-oos-rebase` leg `old_ckpt_clean` runs the SAME
checkpoint (`runs/832_e5/ckpt_imprint_full`) through `eval_vector --operator policy-argmax`
CORRECTLY PINNED to `runs/832_e5/frame_provenance_imprint.json`, with
`vector_weight_rebalance=true`, and reads NEGATIVE in all four regime windows. The frame-pin
mechanism cannot explain that, because the pin was present.

One reconciling hypothesis is arithmetic rather than mechanical: every regime window starts
at/after 2021-12-15, while imprint.py's sanity replay starts 2021-01-01. If the policy's
whole gain sits in the excluded first ~11.5 months of 2021, a continuous 4.7-year window can
be strongly positive while four ~1-year sub-windows are each negative, with no contradiction.

This script tests that hypothesis on RETAINED ARTIFACTS ONLY — no DB access, no eval run. It
converts every retained final_pv into a compound annual growth rate (CAGR), which is the one
statistic that IS comparable between a 4.7-year continuous walk and a 1-year sub-walk (each
sub-window restarts the book at initial_cash, so PV multiples cannot simply be chained, but
each window's own growth RATE can be placed on one axis).

Reading the output: if the full-window CAGR sits INSIDE the band of the sub-window CAGRs, the
gain is spread across the window and 2021-concentration is refuted for that checkpoint. If
the full-window CAGR sits far ABOVE the band, the excluded 2021 stub must be carrying it, and
the script prints the multiple that stub would have to deliver.

Run:  python experiments/882_instrument_reconciliation/period_decomposition.py
"""
import glob
import json
import os
import sys
from datetime import date

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

INITIAL_CASH = 100_000.0
DAYS_PER_YEAR = 365.2425

# (label, imprint-sanity artifact, its scorecard window dir)
CKPTS = [
    ("ckpt_imprint_full (OLD, trained 2026-07-17, pre-D31 semantics)",
     "runs/832_e5/imprint_full.json", "runs/861_ref_scorecard/old_ckpt_clean/windows"),
    ("ckpt_ref_s0 (clean-trained 2026-08-03, post-D31)",
     "runs/861_ref/results_s0.json", "runs/861_ref_scorecard/s0/windows"),
    ("ckpt_ref_s1 (clean-trained 2026-08-03, post-D31)",
     "runs/861_ref/results_s1.json", "runs/861_ref_scorecard/s1/windows"),
    ("ckpt_ref_s2 (clean-trained 2026-08-03, post-D31)",
     "runs/861_ref/results_s2.json", "runs/861_ref_scorecard/s2/windows"),
]
# The four IN-WINDOW regime slices. oos_full / wf_q4 / wf_q1 are OUT of the training window
# and are reported separately — they are not part of the 2021-concentration arithmetic.
IN_WINDOW = ("bear22w", "bull23w", "y2024", "recent")


def _years(a, b):
    ya, ma, da = (int(x) for x in a.split("-"))
    yb, mb, db = (int(x) for x in b.split("-"))
    return (date(yb, mb, db) - date(ya, ma, da)).days / DAYS_PER_YEAR


def _cagr(mult, yrs):
    return (mult ** (1.0 / yrs) - 1.0) * 100.0


def main():
    out = []
    w = out.append
    w("TASK_882 FIX 1 — period decomposition of the imprint lane's in-window PnL")
    w("=" * 100)
    w("Source: RETAINED artifacts only (no DB read, no eval run). PV multiple = final_pv /")
    w(f"        {INITIAL_CASH:,.0f} initial cash; CAGR = mult**(1/years) - 1.")
    w("        Every scorecard window restarts the book at initial cash, so PV multiples are")
    w("        NOT chainable across windows — CAGR is what places them on one axis.")
    w("")
    doc = {"schema": "882/period-decomposition/v1", "checkpoints": []}

    for label, imprint_path, win_dir in CKPTS:
        ip = os.path.join(_REPO, imprint_path)
        if not os.path.exists(ip):
            w(f"[SKIP] {label}: {imprint_path} missing")
            continue
        d = json.load(open(ip))
        t21 = d["sanity"]["per_tier"]["21"]
        start, end = d["window"]
        yrs = _years(start, end)
        mult = t21["final_pv"] / INITIAL_CASH
        full_cagr = _cagr(mult, yrs)
        entry = {"checkpoint": label, "imprint_sanity": {
            "artifact": imprint_path, "window": [start, end], "years": round(yrs, 3),
            "net21": t21["policy_inenv"], "final_pv": t21["final_pv"],
            "pv_multiple": round(mult, 4), "cagr_pct": round(full_cagr, 2)}, "windows": {}}

        w("-" * 100)
        w(f"{label}")
        w(f"  imprint.py sanity replay   {start}..{end}  ({yrs:.2f}y)")
        w(f"      net@21 {t21['policy_inenv']:+.4f}   final_pv {t21['final_pv']:>12,.0f}"
          f"   x{mult:.3f}   CAGR {full_cagr:+.1f}%/yr")
        w(f"  eval_vector policy-argmax, frame-pinned (862 scorecard, {win_dir}):")

        band = []
        for name in IN_WINDOW + ("oos_full", "wf_q4", "wf_q1"):
            hits = glob.glob(os.path.join(_REPO, win_dir, name, "results_*.json"))
            if not hits:
                continue
            r = json.load(open(hits[0]))
            wt = r["tiers"]["21"]
            ws, we = r["window"]
            wy = _years(ws, we)
            wm = wt["final_pv"] / INITIAL_CASH
            wc = _cagr(wm, wy)
            tag = "IN-WINDOW" if name in IN_WINDOW else "oos      "
            w(f"      {name:9s} {ws}..{we} ({wy:.2f}y) {tag}  net@21 {wt['sharpe_full']:+.4f}"
              f"   final_pv {wt['final_pv']:>10,.0f}   x{wm:.3f}   CAGR {wc:+.1f}%/yr")
            entry["windows"][name] = {"window": [ws, we], "years": round(wy, 3),
                                      "net21": wt["sharpe_full"], "final_pv": wt["final_pv"],
                                      "pv_multiple": round(wm, 4), "cagr_pct": round(wc, 2),
                                      "in_training_window": name in IN_WINDOW}
            if name in IN_WINDOW:
                band.append((name, wc, wm, wy, ws, we))

        if band:
            lo = min(b[1] for b in band)
            hi = max(b[1] for b in band)
            inside = lo <= full_cagr <= hi
            w("")
            w(f"      IN-WINDOW sub-window CAGR band : [{lo:+.1f}%/yr, {hi:+.1f}%/yr]")
            w(f"      continuous full-window CAGR    :  {full_cagr:+.1f}%/yr"
              f"   -> {'INSIDE the band' if inside else 'OUTSIDE the band'}")
            entry["cagr_band_in_window"] = [round(lo, 2), round(hi, 2)]
            entry["full_window_cagr_inside_band"] = bool(inside)
            if inside:
                w("      VERDICT: growth is spread across the window. The full-window number is")
                w("               NOT carried by the excluded 2021 stub — 2021-concentration is")
                w("               REFUTED for this checkpoint, and the two instruments agree in")
                w("               sign and magnitude band.")
                entry["verdict"] = "no 2021 concentration; instruments agree"
            else:
                # what the excluded stub would have to do
                stub_start, cover_start = start, min(b[4] for b in band)
                stub_y = _years(stub_start, cover_start)
                rest_y = yrs - stub_y
                mean_c = sum(b[1] for b in band) / len(band)
                rest_mult = (1.0 + mean_c / 100.0) ** rest_y
                need = mult / rest_mult if rest_mult > 0 else float("inf")
                w("      VERDICT: the continuous number is OUTSIDE the sub-window band, so on this")
                w("               arithmetic the excluded head of the window would have to carry it.")
                w(f"               Excluded stub {stub_start}..{cover_start} ({stub_y:.2f}y);")
                w(f"               covered stretch at the mean sub-window rate {mean_c:+.1f}%/yr")
                w(f"               over {rest_y:.2f}y => x{rest_mult:.3f}; the stub would then need")
                w(f"               x{need:.2f} in {stub_y*12:.1f} months to reach x{mult:.3f}.")
                w("               NOT TESTABLE from retained artifacts: imprint.py's sanity block")
                w("               records only aggregate net@21 / final_pv — no PV path, no")
                w("               per-period breakdown, no equity curve. See the RESOLUTION for")
                w("               the competing (and better-supported) EPOCH explanation.")
                entry["verdict"] = "outside band; stub magnitude required but untestable"
                entry["excluded_stub"] = {
                    "span": [stub_start, cover_start], "years": round(stub_y, 3),
                    "required_multiple": round(need, 3),
                    "assumed_rest_cagr_pct": round(mean_c, 2)}
        w("")
        doc["checkpoints"].append(entry)

    w("=" * 100)
    text = "\n".join(out)
    print(text)
    with open(os.path.join(_HERE, "period_decomposition.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(_HERE, "period_decomposition.json"), "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"[882] wrote {os.path.join(_HERE, 'period_decomposition.txt')}")
    print(f"[882] wrote {os.path.join(_HERE, 'period_decomposition.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

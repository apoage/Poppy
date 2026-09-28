#!/usr/bin/env python3
"""TASK_923 age-decay exit derivation (D85's pre-registered rule, stated BEFORE the first
book run and never revisited after seeing book results).

Rule: age* = the right edge of the FIRST of P3's own 7 age buckets (0-9,10-19,20-29,30-44,
45-59,60-74,75-89 -- `p3_post_listing_event_study.py:370`, its own `main()`) whose mean
market-residual forward return is within 1 STANDARD ERROR of zero, where the per-bucket SE
is computed from the SAME PIT panel P3 used (bucket residual-return SD / sqrt(bucket n) --
an unclustered per-observation SE, matching P3's own profile-table methodology, which does
not cluster at that level either). If no bucket satisfies this in [0,89], age* defaults to
89 (AGE_MAX, the end of P3's studied window) -- the task file's own named example rule.

This module computes age* from PANEL DATA ONLY (P3's own age/residual arrays, via
`p3.true_ic_for_panel`) -- no book P&L, no cost, no listing-book simulation is touched, so
running it is a legitimate SPEC-time pre-registration input per D85.
"""
import numpy as np

# Exactly p3_post_listing_event_study.py:370's own buckets (mirrored, not re-derived --
# any drift from that literal tuple would silently change what "P3's profile" means here).
P3_AGE_BUCKETS = ((0, 9), (10, 19), (20, 29), (30, 44), (45, 59), (60, 74), (75, 89))
AGE_STAR_DEFAULT = 89


def bucket_profile_with_se(panel, p3_module):
    """Returns list of dicts (one per P3_AGE_BUCKETS entry): age_lo, age_hi, n, mean_bp,
    sd_bp, se_bp, within_1se_of_zero (bool)."""
    res = p3_module.true_ic_for_panel(panel)
    age_true = res["_age_true"]
    fwd_resid = res["_fwd_resid"]
    out = []
    for lo, hi in P3_AGE_BUCKETS:
        m = (age_true >= lo) & (age_true <= hi) & np.isfinite(fwd_resid)
        vals = fwd_resid[m]
        n = int(vals.size)
        if n < 2:
            out.append({"age_lo": lo, "age_hi": hi, "n": n, "mean_bp": float("nan"),
                       "sd_bp": float("nan"), "se_bp": float("nan"),
                       "within_1se_of_zero": False})
            continue
        mean_bp = float(np.mean(vals)) * 1e4
        sd_bp = float(np.std(vals, ddof=1)) * 1e4
        se_bp = sd_bp / np.sqrt(n)
        out.append({"age_lo": lo, "age_hi": hi, "n": n, "mean_bp": mean_bp, "sd_bp": sd_bp,
                   "se_bp": se_bp, "within_1se_of_zero": bool(abs(mean_bp) <= se_bp)})
    return out


def derive_age_star(panel, p3_module, buckets=P3_AGE_BUCKETS, default=AGE_STAR_DEFAULT):
    profile = bucket_profile_with_se(panel, p3_module)
    for row in profile:
        if row["within_1se_of_zero"]:
            return int(row["age_hi"]), profile
    return int(default), profile

#!/usr/bin/env python3
"""TASK_915 cell grid: 4 targets x 4 horizons x 5 feature sets x 2 estimators = 160 cells.

Expanding-window quarterly refit, strictly causal, purge >= h between train end and test
start (train rows whose OWN label horizon [t, t+h] would overlap the test quarter are
excluded). Cross-sectional per-date z-scoring of every feature (mean/std over LIVE assets
that date only -- leak-free regardless of when computed, since it never reads another date).
Ridge (sklearn, closed-form solver) and HistGradientBoostingRegressor (fixed hyperparameters,
>=3 seeds, averaged), per SPEC.md.
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import features as feat
import targets as tg
import stats_utils as su

HORIZONS = (1, 3, 7, 21)
FEATURE_SETS = ("F1", "F2", "F3", "F4", "ALL")
ESTIMATORS = ("ridge", "gbm")
RIDGE_ALPHA = 1.0                       # pinned; features are per-date z-scored (unit-ish scale)
GBM_PARAMS = dict(max_iter=100, max_depth=4, learning_rate=0.1, l2_regularization=0.0,
                   max_leaf_nodes=15, min_samples_leaf=30)
GBM_SEEDS = (0, 1, 2)
BETA_WINDOW = 90
VOL_WINDOW = 30
MIN_TRAIN_DAYS = 500                    # >= beta_window + a couple of vol windows of history


def _to_epoch(date_str):
    y, m, d = (int(x) for x in date_str.split("-"))
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp())


def quarter_starts(day_ts, first_test_date, last_date):
    """Calendar-quarter start epochs from `first_test_date` through `last_date` inclusive."""
    lo = _to_epoch(first_test_date)
    hi = _to_epoch(last_date)
    y = dt.datetime.fromtimestamp(lo, tz=dt.timezone.utc).year
    q = (dt.datetime.fromtimestamp(lo, tz=dt.timezone.utc).month - 1) // 3
    out = []
    while True:
        month = q * 3 + 1
        start = dt.datetime(y, month, 1, tzinfo=dt.timezone.utc).timestamp()
        if start > hi:
            break
        out.append(int(start))
        q += 1
        if q > 3:
            q = 0
            y += 1
    return out


def cross_sectional_zscore(panel_2d, present_mask):
    """Per-date z-score over LIVE assets only; non-live entries -> NaN.

    A date whose cross-sectional STD is ~0 (every live asset carries the identical value --
    e.g. F4's "time" path-dimension increment, which is a deterministic constant of the
    window length, not of the asset, and log-sig level-2's antisymmetric DIAGONAL, which is
    always exactly 0) is NOT a missing-data event: it is a column that carries zero
    cross-sectional information that date. Mapping it to NaN (the earlier, BUGGY behaviour)
    poisons the joint-finiteness mask every downstream cell uses (`np.isfinite(X).all(axis=-1)`
    in `fit_cell`) -- ONE always-constant column (discovered empirically: F4's `f4_l1_3` time
    increment and every `f4_l2_i_i` diagonal) silently zeroed out EVERY row of EVERY cell that
    includes it (F4 and ALL: 64/160 cells), which looked identical to "no signal" (NaN mean_ic)
    -- exactly the D83(i) failure mode ("the cap never binds" vs "the cap was never applied"
    produce identical tables). Fix: a zero-std date maps to z=0 for every live asset (the
    conventional definition -- x-mean=0 identically, so the z-score IS 0, not undefined),
    which contributes no cross-sectional rank information without poisoning any other column's
    finiteness. Regression test: tests/test_grid_smoke.py /
    tests/test_positive_control.py re-run post-fix; F4/ALL coverage verified non-zero in
    RUN_MANIFEST.json's dry-run log (see SPEC.md Sec.11 disclosure)."""
    x = np.where(present_mask, panel_2d, np.nan)
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    zero_var = np.isfinite(sd) & (sd <= 1e-12)
    sd_safe = np.where(zero_var | ~np.isfinite(sd), 1.0, sd)
    z = (x - mu) / sd_safe
    z = np.where(zero_var & np.isfinite(x), 0.0, z)   # zero-var date -> 0 for LIVE assets only
    return np.where(np.isfinite(z), z, np.nan)


def build_present_mask(panel, beta_window=BETA_WINDOW, vol_window=VOL_WINDOW):
    from trader import strategy as tstrat
    lc = tg.log_close(panel["close"])
    fa = tstrat.first_active(lc)
    from yang_zhang import close_to_close_vol
    rv = close_to_close_vol(panel["close"], window=vol_window)
    warm = max(beta_window, vol_window) + 1
    return tstrat.present_mask(lc, rv, fa, warm, vol_window), fa


def build_panel_features_and_targets(panel):
    """One-time (per-run, not per-cell) construction: present mask, all 4 targets per
    horizon, and all raw F1-F4 features, then cross-sectionally z-scored. Returns a dict
    consumed by `fit_cell`."""
    present, fa = build_present_mask(panel)
    targets_by_h = {}
    beta_by_h = {}
    sigma_yz = None
    for h in HORIZONS:
        td = tg.build_all_targets(panel, present, h, beta_window=BETA_WINDOW,
                                   vol_window=VOL_WINDOW)
        targets_by_h[h] = td
        beta_by_h[h] = td["beta"]
        if sigma_yz is None:
            sigma_yz = td["sigma_yz"]

    raw_feats = feat.build_all_features(panel, targets_by_h[HORIZONS[0]])
    feature_names = sorted(raw_feats.keys())
    z_feats = {name: cross_sectional_zscore(raw_feats[name], present) for name in feature_names}
    return {"present": present, "first_active": fa, "targets_by_h": targets_by_h,
            "feature_names": feature_names, "z_features": z_feats, "sigma_yz": sigma_yz,
            "raw_features": raw_feats}


def _fit_predict_ridge(Xtr, ytr, Xte):
    m = Ridge(alpha=RIDGE_ALPHA)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def _fit_predict_gbm(Xtr, ytr, Xte, seeds=GBM_SEEDS):
    preds = []
    for s in seeds:
        m = HistGradientBoostingRegressor(random_state=s, **GBM_PARAMS)
        m.fit(Xtr, ytr)
        preds.append(m.predict(Xte))
    return np.mean(preds, axis=0)


def fit_cell(built, day_ts, h, target_name, feature_set, estimator,
             first_test_date="2022-01-01", last_date="2026-06-30",
             min_train_days=MIN_TRAIN_DAYS):
    """Expanding-window quarterly refit for ONE cell. Returns dict with the (T,A) OOS
    forecast panel (NaN outside any test quarter / where present_mask is False) and per-
    quarter bookkeeping (train/test row ranges, n_train)."""
    present = built["present"]
    y = built["targets_by_h"][h][target_name]
    cols = feat.feature_set_columns(built["feature_names"], feature_set)
    X = np.stack([built["z_features"][c] for c in cols], axis=-1)   # (T, A, F)
    T, A = present.shape
    forecast = np.full((T, A), np.nan)
    q_starts = quarter_starts(day_ts, first_test_date, last_date)
    q_starts.append(_to_epoch(last_date) + 86400)  # sentinel end
    quarters_info = []

    for qi in range(len(q_starts) - 1):
        q_lo_ts, q_hi_ts = q_starts[qi], q_starts[qi + 1]
        test_lo = int(np.searchsorted(day_ts, q_lo_ts))
        test_hi = int(np.searchsorted(day_ts, q_hi_ts))
        if test_hi <= test_lo:
            continue
        train_hi = test_lo - h                      # purge >= h: last usable label ends here
        if train_hi < min_train_days:
            continue
        Xtr_rows = X[:train_hi]
        ytr_rows = y[:train_hi]
        ptr_rows = present[:train_hi]
        mask_tr = ptr_rows & np.isfinite(ytr_rows) & np.isfinite(Xtr_rows).all(axis=-1)
        n_train = int(mask_tr.sum())
        if n_train < 200:
            continue
        Xtr = Xtr_rows[mask_tr]
        ytr = ytr_rows[mask_tr]

        Xte_rows = X[test_lo:test_hi]
        pte_rows = present[test_lo:test_hi]
        mask_te = pte_rows & np.isfinite(Xte_rows).all(axis=-1)
        if mask_te.sum() == 0:
            continue
        Xte = Xte_rows[mask_te]
        if estimator == "ridge":
            pred = _fit_predict_ridge(Xtr, ytr, Xte)
        elif estimator == "gbm":
            pred = _fit_predict_gbm(Xtr, ytr, Xte)
        else:
            raise ValueError(estimator)
        sub = forecast[test_lo:test_hi]
        sub[mask_te] = pred
        forecast[test_lo:test_hi] = sub
        quarters_info.append({"q_start_ts": q_lo_ts, "test_lo": test_lo, "test_hi": test_hi,
                              "n_train": n_train, "n_test": int(mask_te.sum())})
    return {"forecast": forecast, "quarters": quarters_info, "target_name": target_name,
            "h": h, "feature_set": feature_set, "estimator": estimator, "columns": cols}


ALL_CELLS = [(t, h, fs, est) for t in tg.TARGET_NAMES for h in HORIZONS
             for fs in FEATURE_SETS for est in ESTIMATORS]
assert len(ALL_CELLS) == 160, len(ALL_CELLS)

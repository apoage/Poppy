#!/usr/bin/env python3
"""TASK_901 gate (iii) — the real minute per-decision series.

Builds ONE real per-decision return series through the SAME audited 806/808/809 lineage
(`probe.py` / `decide.py`, imported verbatim, never re-implemented — the identical discipline
`experiments/897_corrected_809/sweep_gated_maker.py` follows) and the SAME honest per-decision
segmentation (`honest_estimator.segment_by_decision`), against a READ-ONLY, byte-verified extract
of the real minute table (`runs/897_corrected_809/minute_extract_897.db`, built and content-hash
verified against the real `trading_data.db` by that row's own `build_extract.py` — reused here
rather than re-reading the 24GB source of record, per this task's own "prefer an existing
extract" instruction).

WHAT "7 MAJORS" MEANS HERE — DISCLOSED, NOT ASSUMED. TASK_901's task file and the D57 halted
cycle's pinned constants (`experiments/897_corrected_809/calibration.py::
MEASURED_REAL_DATA_TAU_INERT_HALTED_CYCLE`) both refer to "7 majors, OOS 2024-11 -> 2025-04" as
the series measured. THAT HALTED CYCLE'S OWN CONSTRUCTION (which 7 symbols, which exact
selection rule) IS NOT PRESERVED IN THIS REPO — it ran interactively and was halted; only the
numeric result was pinned, with provenance explicitly marked "NOT re-run in this loop". This
module does NOT claim to reproduce that construction bit-for-bit. Instead it builds an
INDEPENDENT real-data series using the SAME real lineage, the SAME OOS window
(`probe.Config`'s own defaults: test_start=2024-11-01, test_end=2025-04-01 — these ARE the
2024-11->2025-04 window, unmodified here), and `n_universe=7` (the same TRAIN-window-locked
top-N-by-coverage ranking `probe.load_universe_trainlocked` already uses for every other row in
this lineage, just asked for 7 assets instead of 64) — which is what "7 majors" means under this
module's own construction. This is disclosed in every artifact this module writes so a reader
never mistakes it for a reproduction of the pinned 0.639.

AND IT IS NOT "THE 7 MAJORS" IN ANY ORDINARY SENSE — SAY SO PLAINLY (fix cycle 2). The ranking
resolves to `XBT/USD, /USDTEUR, XBT/EUR, ETH/USD, LTC/EUR, LTC/USD, XRP/USD`: four underlyings
(BTC, ETH, LTC, XRP), two of them listed twice against different quote currencies, plus a
stablecoin FX pair (`/USDTEUR`). Calling that "the 7 majors" would overstate the breadth of the
cross-section by roughly a factor of two. It is A real per-decision series built through the real
lineage — which is all gate (iii) needs — and this module claims nothing more for it.

NEVER TOUCHES THE REAL trading_data.db. `EXTRACT_DB` below is the only path this module opens,
and it is opened through the SAME read-only sqlite shim `sweep_gated_maker.py` uses for its own
DB reads (`_ReadOnlySqlite3ForProbe`, imported here, never re-implemented) so a stray write is
structurally impossible even against the extract.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
_897 = _REPO / "experiments" / "897_corrected_809"
sys.path.insert(0, str(_REPO / "runs" / "806_predictability_probe"))
sys.path.insert(0, str(_REPO / "runs" / "808_linear_backtest"))
sys.path.insert(0, str(_REPO / "runs" / "809_decision_rule_rescue"))
sys.path.insert(0, str(_897))

import probe     # noqa: E402  (806 lineage — verbatim, never reimplemented)
import decide    # noqa: E402  (809 lineage)
import honest_estimator as hest          # noqa: E402
from sweep_gated_maker import _ReadOnlySqlite3ForProbe  # noqa: E402  (897's read-only DB shim,
                                                          # imported not copied)

EXTRACT_DB = str(_REPO / "runs" / "897_corrected_809" / "minute_extract_897.db")


def build_real_series(H: int, variant: str = "long_short", k: int = 2, tau: float = 0.9,
                      N: int = 60, fee_bp: float = 8.0, adv_bp: float = 0.0,
                      n_universe: int = 7, db_path: str = EXTRACT_DB) -> Tuple[np.ndarray, Dict]:
    """Build ONE real per-decision net-return series (see module docstring for what "7 majors"
    means here). `tau`/`N`/`fee_bp` mirror 897's own pre-registered grid defaults (fee_bp=8 is
    the smaller of 897's two attainable maker-fee tiers, per D60's launch record).

    `k` DOES NOT carry over from 897's grid (k in {16,32}) — `decide._decide_step` REFUSES
    (returns an all-zero target book, every bar, forever) whenever the present-asset count is
    below `2*k` for `variant="long_short"`. 897's k=16 assumes its 64-asset universe; against
    THIS module's `n_universe=7` it silently produces ZERO decisions for the entire series
    (measured while building this module, and independently re-verified in fix cycle 2:
    n_decisions=0 at k=16, n_universe=7 — not a data problem, a k/universe-size mismatch; the
    SILENT-DEGENERATE-CELL class this exposes is routed to TASK_902, not fixed here). `k=2` is
    the value used.

    WHAT k=2 DOES AND DOES NOT BUY — CORRECTED IN FIX CYCLE 2. An earlier revision of this
    docstring justified k=2 as keeping "the same long_short shape at a comparable ratio (2 of 7
    ~= 29% vs 16 of 64 = 25%)". THAT JUSTIFICATION IS WRONG, because `tau` binds BEFORE `k` in
    BOTH configurations: `_decide_step` sets `n_elig = max(1, round((1 - tau) * npr))`, which at
    tau=0.9 is **1** for every npr in {4..7} (round(0.1*7) = 1) against **6** at npr=64. So the
    fresh-entry pool here is a SINGLE name per side regardless of k — the realized book fills at
    most one fresh long and one fresh short per bar, each at weight 1/k = 1/2, with previously
    held names retained only by the min-hold rule (N=60). 897's 64-asset config, by contrast,
    has a 6-name eligible pool per side against k=16. The two are NOT the same shape at a
    comparable ratio; k=2 is simply the largest k that leaves this 7-asset universe able to
    trade at all. `meta["present_count_hist"]` / `meta["n_elig_hist"]` record the realized
    distribution so a reader can check this rather than take it on trust.

    Returns `(reb_ret, meta)` — `reb_ret` is the per-decision (holding-period) net-return series
    (`honest_estimator.segment_by_decision`), `meta` records the exact construction (symbols,
    window, bar_sec, n_decisions, n_bars) for the artifact.
    """
    cfg = probe.Config(db_path=db_path, n_universe=int(n_universe))
    orig_decide_cfg = decide.CFG
    orig_probe_sqlite3 = probe.sqlite3
    decide.CFG = cfg
    probe.sqlite3 = _ReadOnlySqlite3ForProbe()
    try:
        bar_sec = probe.report_bar_frequency(cfg)
        symbols, asset_ids, _ranking = probe.load_universe_trainlocked(cfg)
        ohlcv, grid_ts = probe.build_aligned_ohlcv(cfg, asset_ids, bar_sec)
        close = np.ascontiguousarray(ohlcv[:, :, 3])
        high = np.ascontiguousarray(ohlcv[:, :, 1])
        low = np.ascontiguousarray(ohlcv[:, :, 2])
        del ohlcv
        gc.collect()

        w, mu, sd, alpha = decide.fit_xs_signal_chunked(close, grid_ts, int(H))
        _, _, test = probe.make_splits(cfg, grid_ts, int(H))
        dec_rows = np.arange(test.lo, test.hi + 1, dtype=np.int64)
        dec_rows = dec_rows[dec_rows + 2 < close.shape[0]]
        pred = decide._chunked_predict(close, dec_rows, w, mu, sd, cfg.W)
        present = np.isfinite(pred) & np.isfinite(close[dec_rows]) & (close[dec_rows] > 0)

        npr = present.sum(axis=1).astype(np.int64)          # present-asset count per decision bar
        present_count_hist = {int(v): int(c) for v, c in zip(*np.unique(npr, return_counts=True))}
        # `_decide_step`'s OWN eligibility arithmetic, evaluated (not re-implemented) on the
        # realized present-counts, so the "tau binds before k" claim in this module's docstring is
        # a measurement rather than an assertion.
        n_elig = np.maximum(1, np.round((1.0 - float(tau)) * npr).astype(np.int64))
        n_elig_hist = {int(v): int(c) for v, c in zip(*np.unique(n_elig, return_counts=True))}

        gross, turn, fill = decide._run_decision(close, high, low, pred, present, dec_rows,
                                                 int(k), variant, float(tau), int(N), "maker")
        cost = (float(fee_bp) + float(adv_bp)) * 1e-4
        net = gross - turn * cost
        reb_ret = hest.segment_by_decision(net, turn)
    finally:
        decide.CFG = orig_decide_cfg
        probe.sqlite3 = orig_probe_sqlite3

    meta = {
        "H": int(H), "variant": variant, "k": int(k), "tau": float(tau), "N": int(N),
        "fee_bp": float(fee_bp), "adv_bp": float(adv_bp), "n_universe": int(n_universe),
        "symbols": symbols, "bar_sec": bar_sec,
        "test_start": cfg.test_start, "test_end": cfg.test_end,
        "n_bars": int(net.shape[0]) if hasattr(net, "shape") else None,
        "n_decisions": int(reb_ret.shape[0]), "fill_rate": float(fill),
        "db_path": db_path,
        "present_count_hist": present_count_hist, "n_elig_hist": n_elig_hist,
        "universe_note": "n_universe=7 resolves (train-window-locked, top-N-by-coverage) to "
                         "XBT/USD, /USDTEUR, XBT/EUR, ETH/USD, LTC/EUR, LTC/USD, XRP/USD — i.e. "
                         "FOUR distinct underlyings (BTC, ETH, LTC, XRP) listed against two quote "
                         "currencies plus one stablecoin FX pair (/USDTEUR). It is A real "
                         "per-decision series through the audited lineage; it is NOT 'the 7 "
                         "majors' the task file names, and cross-sectional breadth here is "
                         "smaller than the asset count suggests (XBT/USD vs XBT/EUR and LTC/USD "
                         "vs LTC/EUR are near-duplicate price processes).",
        "construction_note": "independent real-data leg (see real_series.py docstring) — NOT a "
                             "reproduction of the D57 halted-cycle's pinned 0.639 construction, "
                             "whose exact symbol selection is not preserved in this repo.",
    }
    return reb_ret, meta


if __name__ == "__main__":
    import json
    reb_ret, meta = build_real_series(H=1, variant="long_short")
    print(json.dumps(meta, indent=2))
    print(f"n_decisions={reb_ret.size} mean={reb_ret.mean():.6g} std={reb_ret.std():.6g}")

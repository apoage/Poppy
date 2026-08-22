#!/usr/bin/env python3
"""TASK_893 — measure D41: leading-bfill-prefix cells + masked-vs-shipped re-measurement.

D41 (`tasks/DECISIONS.md`): `trader/data.py:97` `_ffill_bfill_cols` forward-fills a daily close
panel, then sets every row BEFORE a column's first valid print to that first price — a genuine
backward fill, despite its docstring's "no cross-time leak" claim. D41 explicitly refuses to
rule without a measurement. This script IS that measurement. It is READ-ONLY with respect to
`trader/` behavior: it does not import-and-mutate, monkeypatch, or edit any `trader/` function —
every number below is produced by calling existing public `trader.*` functions with plain
arguments (`bt.cell_metrics` already accepts `pos` as a caller-supplied array, so the "masked"
re-run is just a different `pos` array passed into the unmodified, audited scorer).

WHAT `--db` DOES AND DOES NOT SELECT
------------------------------------
The PANEL is ALWAYS `trader.config.NPZ_CACHE` (the frozen 819 broad daily npz behind the
published +0.97 bar). `--db` selects ONLY the source for the two DB-side cross-check paths and
the value anchor. It NEVER feeds the panel, the positions, or the re-measurement. A fixture run
and a real run therefore score the SAME panel and MUST produce identical `remeasure_*` numbers;
the fixture leg is a smoke test of this script, NOT a second measurement of the bar. This is
asserted in the JSON (`db.role`, `panel.note`) and printed at the top of every run.

HOW THE FROZEN PANEL WAS BUILT (settled 2026-08-16; this is what makes the counting subtle)
-------------------------------------------------------------------------------------------
`runs/819_single_asset/portfolio.py::load_broad` built the npz as:

    close_1m = build_close_aligned(BROAD_IDS, grid_ts)   # ONE-MINUTE grid, ffill + leading bfill
    close_d  = close_1m[::K]                             # K = 1440 -> every 1440th MINUTE row
    ts_d     = grid_ts[::K]

so the daily panel is a *minute-subsample*, and `ts_d` is sampled at the 816 grid's offset
(22:56:00 UTC), not midnight. Two consequences drive every count below:

  * `build_close_aligned` admits a raw bar only if
    `timestamp >= grid_ts[0] AND timestamp < grid_ts[-1]+60 AND (timestamp-grid_ts[0]) % 60 == 0`.
    Bars BEFORE `grid_ts[0]` exist in the DB for several majors and are simply not admitted, so
    those assets still get a leading backward fill even though they are not "new listings" —
    measured on the 1-MINUTE panel this spans 2-24 rows for 14 of the 16 such majors (only KSM
    and CRV are genuine 1-row fills); "1 row" is a property of the DOWNSTREAM daily subsample
    (`close_1m[::1440]`), not of this predicate.
  * The bfill's SOURCE row is a MINUTE row (e.g. SOL's first admitted print, 2021-06-17 15:32
    UTC). K-subsampling almost never lands on it, so the source row is NOT present in the daily
    panel. The leading bit-identical run in a daily column is therefore made up ENTIRELY of
    fabricated rows — its length IS the prefix length, not one more than it.

Three legs (task Scope §1-3):

  (1) COUNT   — how many daily-panel rows precede each asset's first admitted print (i.e. carry a
                backward-filled price), in total, per regime window, and as scored decision cells.
                Computed by THREE paths (acceptance ii):
                  Path A ("min_searchsorted") — one `MIN(timestamp)` query per asset under
                    `build_close_aligned`'s exact admission predicate, then `searchsorted` onto
                    `ts_d` (side='left' => count of grid slots strictly before the first print).
                  Path B ("db_rowwise") — independent computation: walk rows r = 0,1,2,... and
                    ask the DB, per row, whether ANY admitted bar exists at-or-before `ts_d[r]`.
                    The prefix is the first r for which one does. No MIN, no searchsorted, a
                    different SQL shape, and every row tested on its own terms.
                  Path C ("npz_run") — npz-only structural check, no DB: the leading
                    bit-identical run length. Exact where the run is >= 2 rows. PROVABLY
                    AMBIGUOUS where the run length is 1: "row 0 fabricated, row 1 real" and
                    "row 0 real, row 1 simply differs" are indistinguishable from the close
                    column alone. Such assets are reported as AMBIGUOUS, never as agreement.
                Path A/B are the authoritative pair wherever `--db` has coverage; Path C is the
                only source on a DB that cannot reach the panel start (e.g. the fixture), and
                then the total is reported as a lower/upper bound instead of a point value.
                A VALUE ANCHOR is checked alongside the counts: for every fabricated prefix the
                panel's row-0 close must equal the close of the first admitted bar, whose
                timestamp is LATER than the timestamps of every row carrying it. That lead time
                (`last_prefix_row_lead_seconds`) is the backward fill, measured in seconds.
  (2) RE-MEASURE — the gate-1a FULL cell (net@8/@21 Sharpe, equity, maxDD, turnover, BEAR22),
                as shipped vs. with every leading-prefix cell's POSITION forced to exactly 0.0
                (masked inactive — no PnL attribution), same seed/bootstrap (`config.STRATEGY`,
                seed=819) both sides.
  (3) EXPLOIT  — does the as-shipped `pos` array hold ANY nonzero position inside an asset's own
                leading prefix? Reported together with a STRUCTURAL COROLLARY proof (see
                `structural_corollary` in the JSON) showing the answer is 0 a priori for any
                consumer that routes through `strategy.present_mask` — and a POSITIVE CONTROL
                that forces a nonzero position into a prefix cell without touching the function
                under test, proving the mask->scorer path is responsive rather than vacuous.

Nothing here writes to `trading_data.db` (opened `mode=ro`) or to any `trader/` source file.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trader import config as tconfig  # noqa: E402
from trader import data as datamod    # noqa: E402
from trader import backtest as bt     # noqa: E402

# 819 broad universe, verbatim from runs/819_single_asset/portfolio.py BROAD_IDS/BROAD_SYM
# (read-only reference to a planner-owned research file — copied as data, not imported, so
# this script never executes or edits that prototype). Same order as the cached npz's `syms`.
BROAD_IDS = [1091, 359, 309, 879, 589, 1121, 61, 1102, 561, 1111, 1108, 970, 162, 89, 607,
             862, 174, 544, 200, 987, 48, 404, 596, 1131, 599, 412, 853, 340, 749, 722,
             395, 461, 482, 669, 846, 515, 492, 24, 911]
BROAD_SYM = ["XBT", "ETH", "DOT", "SOL", "LTC", "XRP", "ADA", "XDG", "LINK", "XMR",
             "XLM", "TRX", "ATOM", "ALGO", "MATIC", "SHIB", "AVAX", "KSM", "BCH", "UNI",
             "AAVE", "FIL", "LUNA", "XTZ", "MANA", "FLOW", "SC", "EOS", "PEPE", "CRV",
             "FET", "GRT", "ICP", "NEAR", "SAND", "KAVA", "INJ", "ZEC", "SUI"]
SYM_TO_ID = dict(zip(BROAD_SYM, BROAD_IDS))

# 819 build constants (runs/819_single_asset/portfolio.py:67-68) — the npz's own grid geometry.
K_SUBSAMPLE = 1440
BAR_SEC = 60

DEFAULT_DB = str(REPO_ROOT / "fixtures" / "trading_fixture.db")
OUT_DIR = REPO_ROOT / "runs" / "893_prefix_cells"

# FUNCTIONALLY EQUIVALENT to `build_close_aligned`'s admission predicate, NOT textually
# identical to it: `build_close_aligned` applies the `% BAR_SEC == 0` lattice test in numpy
# AFTER the SQL fetch, while this folds an equivalent modulo term into the SQL text itself. A
# bar outside the grid window, or off the minute lattice anchored at grid_start, was never
# written into the 1-min panel and therefore cannot be the source of (or an escape from) the
# leading backward fill.
_ADMIT = ("asset_id=? AND timestamp>=? AND timestamp<? AND (timestamp-?)%60=0")


def _iso(ts):
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _npz_frozen_logc():
    """Same panel `python -m trader parity`/`backtest --source npz` score (trader/cli.py
    `_npz_frozen_logc`): the cached 819 broad daily panel, liquidity-filtered."""
    close_d, ts_d, syms = datamod.load_npz_panel()
    close_d, syms, dropped, keep = datamod.apply_liquidity_filter(close_d, syms)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where(close_d > 0, close_d, np.nan))
    return close_d, logc, ts_d, syms, dropped


# ---------------------------------------------------------------------------- counting paths

def path_c_npz_run(close_col):
    """Path C (npz-only, no DB): length of the leading bit-identical run.

    `_ffill_bfill_cols` sets `out[:R] = a[R]` where R is the first admitted row *of the 1-minute
    panel*. The daily panel is that panel subsampled every K=1440 rows, so R is essentially never
    a daily row: every daily row carrying the bfilled value is fabricated and the run length IS
    the prefix length. (Verified 2026-08-16 against the raw DB on all 39 universe assets — see
    module docstring and `value_anchor`.)

    Returns (run_len, ambiguous). `ambiguous` is True exactly when run_len == 1: a single-row
    leading run is equally consistent with prefix 0 (row 0 real, row 1 simply a different real
    close) and prefix 1 (row 0 fabricated, row 1 the first real close). The close column alone
    cannot separate those; only the DB can. This is a real limitation of Path C and is reported,
    never explained away.
    """
    v0 = close_col[0]
    neq = np.flatnonzero(close_col != v0)
    run_len = int(neq[0]) if neq.size else int(close_col.size)
    return run_len, (run_len == 1)


def path_a_min_searchsorted(cur, aid, ts_d, grid_start, grid_end):
    """Path A: first ADMITTED raw print (single MIN query) -> searchsorted onto the daily grid.
    Returns (prefix_rows, first_admitted_ts) or (None, None) if this DB has no admitted bar."""
    row = cur.execute(f"SELECT MIN(timestamp) FROM ohlcv WHERE {_ADMIT}",
                      (int(aid), grid_start, grid_end, grid_start)).fetchone()
    if not row or row[0] is None:
        return None, None
    first_ts = int(row[0])
    # side='left' => number of grid slots STRICTLY before the first print. A slot AT the print
    # time carries the real bar, so it is not fabricated and must not be counted.
    return int(np.searchsorted(ts_d, first_ts, side="left")), first_ts


def path_b_db_rowwise(cur, aid, ts_d, grid_start, grid_end):
    """Path B: independent per-row existence probe. For r = 0,1,2,... ask whether ANY admitted
    bar exists at-or-before `ts_d[r]`; the prefix is the first r for which one does. Distinct
    computation from Path A (no MIN aggregate, no searchsorted, one test per row against that
    row's own timestamp). Returns None if the asset has no admitted bar anywhere in the grid."""
    probe = (f"SELECT 1 FROM ohlcv WHERE {_ADMIT} AND timestamp<=? LIMIT 1")
    for r in range(int(ts_d.size)):
        hit = cur.execute(probe, (int(aid), grid_start, grid_end, grid_start,
                                  int(ts_d[r]))).fetchone()
        if hit:
            return r
    return None


def value_anchor(cur, aid, ts_d, close_col, prefix, first_ts, grid_start, grid_end):
    """The evidence that a leading prefix row is FABRICATED rather than merely stale: the price
    it carries is the close of a bar timestamped LATER than the row itself.

    Checks (all DB-grounded, independent of the counting arithmetic):
      * `row0_close_equals_first_admitted_close` — the panel's row 0 is literally the first
        admitted bar's close.
      * `last_prefix_row_lead_seconds` — `first_admitted_ts - ts_d[prefix-1]`, i.e. how far into
        the future the LAST fabricated row reaches. Strictly positive => backward fill, proven.
      * `first_real_row_matches_ffill` — row `prefix` equals the close of the last admitted bar
        at-or-before its own timestamp (an ordinary, accepted forward fill).
    """
    out = {"first_admitted_ts": first_ts, "first_admitted_iso": _iso(first_ts)}
    if first_ts is None:
        return out
    fc = cur.execute(f"SELECT close FROM ohlcv WHERE {_ADMIT} AND timestamp=?",
                     (int(aid), grid_start, grid_end, grid_start, int(first_ts))).fetchone()
    first_close = float(fc[0]) if fc and fc[0] is not None else None
    out["first_admitted_close"] = first_close
    out["panel_row0_close"] = float(close_col[0])
    out["row0_close_equals_first_admitted_close"] = (
        first_close is not None and float(close_col[0]) == first_close)
    if prefix > 0:
        out["last_prefix_row_ts"] = int(ts_d[prefix - 1])
        out["last_prefix_row_iso"] = _iso(int(ts_d[prefix - 1]))
        out["last_prefix_row_lead_seconds"] = int(first_ts) - int(ts_d[prefix - 1])
        out["is_backward_fill"] = out["last_prefix_row_lead_seconds"] > 0
    else:
        out["is_backward_fill"] = False
    if prefix < ts_d.size:
        ff = cur.execute(
            f"SELECT close FROM ohlcv WHERE {_ADMIT} AND timestamp<=? ORDER BY timestamp DESC LIMIT 1",
            (int(aid), grid_start, grid_end, grid_start, int(ts_d[prefix]))).fetchone()
        if ff and ff[0] is not None:
            out["first_real_row_matches_ffill"] = float(close_col[prefix]) == float(ff[0])
    return out


def _overlap(lo_a, hi_a, lo_b, hi_b):
    return max(0, min(hi_a, hi_b) - max(lo_a, lo_b))


def _default_out(db_path):
    resolved = str(Path(db_path).resolve())
    if resolved == str(Path(DEFAULT_DB).resolve()):
        stem = "fixture"
    elif resolved == str(Path(tconfig.DB_PATH).resolve()):
        stem = "real"
    else:
        stem = Path(db_path).stem
    return str(OUT_DIR / f"prefix_cells_{stem}.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description="TASK_893 — measure D41 leading-bfill-prefix cells")
    ap.add_argument("--db", type=str, default=DEFAULT_DB,
                    help="sqlite db for the DB-side CROSS-CHECK paths ONLY — it never feeds the "
                         f"panel, which is always trader.config.NPZ_CACHE (default: {DEFAULT_DB})")
    ap.add_argument("--out", type=str, default=None,
                    help="output JSON (default: runs/893_prefix_cells/prefix_cells_<fixture|real>.json)")
    args = ap.parse_args(argv)
    out_path = Path(args.out) if args.out else Path(_default_out(args.db))

    scfg = tconfig.STRATEGY
    close_d, logc, ts_d, syms, dropped = _npz_frozen_logc()
    T, A = close_d.shape
    grid_start = int(ts_d[0])
    grid_end = int(ts_d[-1]) + K_SUBSAMPLE * BAR_SEC

    print("[893] PANEL   : %s" % tconfig.NPZ_CACHE)
    print("[893]           %d daily rows x %d assets (liquidity-dropped: %s); grid %s .. %s, "
          "step %ds" % (T, A, dropped or "none", _iso(ts_d[0]), _iso(ts_d[-1]),
                        int(ts_d[1] - ts_d[0])))
    print("[893] --db    : %s" % args.db)
    print("[893]           ROLE = cross-check + value-anchor source ONLY. The panel is NEVER read")
    print("[893]           from --db, so fixture and real legs score the SAME panel and MUST")
    print("[893]           report identical remeasure_FULL_and_BEAR22 numbers. The fixture leg is")
    print("[893]           a smoke test of this script, not a second measurement of the bar.")

    rows = bt.regime_rows(ts_d)
    full_lo, full_hi = rows["FULL"]

    # ---------------------------------------------------------------- (1) counts
    con = datamod._connect(args.db, read_only=True)
    cur = con.cursor()
    try:
        # Reliability guard: the DB-side paths are only usable if `--db` reaches back to the panel
        # start. A truncated slice (e.g. the ~90-day fixture) reports each covered asset's "first
        # admitted print" as just the slice's own start — indistinguishable, per asset, from a
        # genuine late listing. Data-driven check (no dates hardcoded): if NOT ONE universe asset
        # shows an admitted print at-or-before ts_d[0], this DB cannot be reaching back that far
        # for ANY of them (implausible for a real full history — XBT/ETH/LTC predate 2021).
        path_a = {}
        for sym in syms:
            aid = SYM_TO_ID.get(sym)
            path_a[sym] = (None, None) if aid is None else path_a_min_searchsorted(
                cur, aid, ts_d, grid_start, grid_end)
        db_reaches_back = any(ts is not None and ts <= grid_start for _, ts in path_a.values())

        per_asset = []
        n_ab_checked = n_ab_agree = 0
        n_c_checked = n_c_agree = n_c_ambiguous = 0
        n_anchor_checked = n_anchor_backfill = n_anchor_row0_ok = n_anchor_ffill_ok = 0
        for j, sym in enumerate(syms):
            aid = SYM_TO_ID.get(sym)
            run_len, c_ambig = path_c_npz_run(close_d[:, j])
            vr0_a, first_ts = path_a[sym]
            trusted = db_reaches_back and vr0_a is not None
            vr0_b = (path_b_db_rowwise(cur, aid, ts_d, grid_start, grid_end)
                     if trusted else None)

            ab_agree = None
            if trusted:
                n_ab_checked += 1
                ab_agree = (vr0_a == vr0_b)
                n_ab_agree += int(ab_agree)

            # Path C consistency: only assessable where C is unambiguous (run >= 2) AND the DB
            # pair is available. An ambiguous C is counted as ambiguous, NOT as agreement.
            # Path C's run==1 ambiguity is a STRUCTURAL property of the npz column — it exists
            # whether or not a DB is available to resolve it, so it is counted unconditionally.
            # Consistency against the DB is only assessable where C is unambiguous AND trusted.
            c_consistent = None
            n_c_ambiguous += int(c_ambig)
            if trusted and not c_ambig:
                n_c_checked += 1
                c_consistent = (run_len == vr0_a)
                n_c_agree += int(c_consistent)

            if trusted:
                vr0 = int(vr0_a)
                src = "db_cross_checked"
                lo_b, hi_b = vr0, vr0
            else:
                # No usable DB coverage: fall back to the npz run, and carry the run==1 ambiguity
                # as an explicit interval rather than pretending to a point value.
                vr0 = run_len
                src = "npz_run_only"
                lo_b, hi_b = (0, 1) if c_ambig else (run_len, run_len)

            anchor = (value_anchor(cur, aid, ts_d, close_d[:, j], vr0, first_ts,
                                   grid_start, grid_end) if trusted else None)
            if anchor is not None:
                n_anchor_checked += 1
                n_anchor_backfill += int(bool(anchor.get("is_backward_fill")))
                n_anchor_row0_ok += int(bool(anchor.get("row0_close_equals_first_admitted_close")))
                n_anchor_ffill_ok += int(bool(anchor.get("first_real_row_matches_ffill")))

            dec_full = _overlap(0, vr0, full_lo, full_hi)
            per_regime = {name: _overlap(0, vr0, r0, r1)
                          for name, (r0, r1) in rows.items() if name != "FULL"}
            per_asset.append({
                "sym": sym, "asset_id": aid,
                "prefix_rows_pathA_min_searchsorted": vr0_a,
                "prefix_rows_pathB_db_rowwise": vr0_b,
                "prefix_rows_pathC_npz_run": run_len,
                "pathC_ambiguous_run1": bool(c_ambig),
                "pathA_pathB_agree": ab_agree,
                "pathC_consistent_with_db": c_consistent,
                "prefix_rows_authoritative": vr0,
                "prefix_rows_lower_bound": lo_b,
                "prefix_rows_upper_bound": hi_b,
                "authoritative_source": src,
                "value_anchor": anchor,
                "decision_cells_in_FULL": dec_full,
                "decision_cells_by_regime": per_regime,
            })
    finally:
        con.close()

    total_prefix_rows = sum(p["prefix_rows_authoritative"] for p in per_asset)
    total_lo = sum(p["prefix_rows_lower_bound"] for p in per_asset)
    total_hi = sum(p["prefix_rows_upper_bound"] for p in per_asset)
    total_decision_cells_full = sum(p["decision_cells_in_FULL"] for p in per_asset)
    full_window_cells = A * (full_hi - full_lo)
    share = total_decision_cells_full / full_window_cells if full_window_cells else float("nan")
    regime_totals = {name: sum(p["decision_cells_by_regime"].get(name, 0) for p in per_asset)
                     for name in rows if name != "FULL"}

    # Pre-existing regime-window coverage gap (NOT introduced here): trader/config.py REGIMES
    # names half-open windows whose boundaries are resolved independently by
    # backtest.regime_rows, so a row can fall between two named windows and belong to none.
    # Report it explicitly so the per-regime split visibly failing to sum to the FULL total
    # reads as the known window gap it is, and not as an arithmetic error in this script.
    covered = np.zeros(T, dtype=bool)
    for name, (r0, r1) in rows.items():
        if name != "FULL":
            covered[r0:r1] = True
    gap_rows = [int(r) for r in np.flatnonzero(~covered[full_lo:full_hi]) + full_lo]
    cells_in_gap = total_decision_cells_full - sum(regime_totals.values())

    print("\n[893] (1) leading-prefix rows (whole panel, all %d assets): %d%s"
          % (A, total_prefix_rows,
             "" if total_lo == total_hi else "  [bounds %d..%d — see path_c ambiguity]"
             % (total_lo, total_hi)))
    print("[893]     decision cells inside FULL [%d,%d): %d / %d (%.4f%% of the FULL window)"
          % (full_lo, full_hi, total_decision_cells_full, full_window_cells, share * 100))
    print("[893]     per-regime overlap: %s" % regime_totals)
    print("[893]     per-regime sums to %d, i.e. %d cell(s) short of the FULL total — those land"
          % (sum(regime_totals.values()), cells_in_gap))
    print("[893]     in rows %s, which belong to NO named regime (pre-existing half-open gap "
          "between\n[893]     trader/config.py REGIMES windows as resolved by backtest.regime_rows; "
          "not a defect of\n[893]     this script and out of scope for TASK_893)." % gap_rows)
    print("[893]     db depth: %s" % ("OK — reaches the panel start; Path A/B authoritative"
                                      if db_reaches_back else
                                      "INSUFFICIENT — never reaches ts_d[0]; DB paths unusable, "
                                      "Path C (npz run) only"))

    xcheck = {
        "pathA_vs_pathB": {
            "n_checked": n_ab_checked, "n_exact_agree": n_ab_agree,
            "n_disagree": n_ab_checked - n_ab_agree,
            "n_checked_with_nonzero_prefix": sum(
                1 for p in per_asset if p["pathA_pathB_agree"] is not None
                and p["prefix_rows_authoritative"] > 0),
            "n_agree_with_nonzero_prefix": sum(
                1 for p in per_asset if p["pathA_pathB_agree"] and p["prefix_rows_authoritative"] > 0),
        },
        "pathC_npz_run_vs_db": {
            "n_unambiguous_checked": n_c_checked, "n_consistent": n_c_agree,
            "n_inconsistent": n_c_checked - n_c_agree,
            "n_ambiguous_run1": n_c_ambiguous,
            "n_assets_resolved_by_pathC_alone": sum(
                1 for p in per_asset if p["authoritative_source"] == "npz_run_only"),
            "note": ("A run of exactly 1 row cannot distinguish prefix 0 from prefix 1 from the "
                     "close column alone; those assets are reported AMBIGUOUS, not agreeing. "
                     "When no DB reaches the panel start, Path C is the only source and the "
                     "ambiguous assets are the reason the whole-panel total is emitted as the "
                     "interval [total_prefix_rows_lower_bound, total_prefix_rows_upper_bound] "
                     "rather than a point value."),
        },
        "value_anchor": {
            "n_checked": n_anchor_checked,
            "n_row0_close_equals_first_admitted_close": n_anchor_row0_ok,
            "n_proven_backward_fill": n_anchor_backfill,
            "n_first_real_row_matches_ffill": n_anchor_ffill_ok,
        },
    }
    ab = xcheck["pathA_vs_pathB"]
    pc = xcheck["pathC_npz_run_vs_db"]
    print("[893]     cross-check A(min+searchsorted) vs B(per-row DB probe): %d/%d exact agree, "
          "%d disagree\n[893]       — of which %d/%d agree on a NONZERO prefix (non-trivial "
          "corroboration)" % (ab["n_exact_agree"], ab["n_checked"], ab["n_disagree"],
                              ab["n_agree_with_nonzero_prefix"], ab["n_checked_with_nonzero_prefix"]))
    print("[893]     cross-check C(npz run) vs DB: %d/%d consistent where C is unambiguous; "
          "%d asset(s)\n[893]       have a 1-row run and are AMBIGUOUS by construction (never "
          "counted as agreement);\n[893]       %d asset(s) had to be resolved by Path C alone"
          % (pc["n_consistent"], pc["n_unambiguous_checked"], pc["n_ambiguous_run1"],
             pc["n_assets_resolved_by_pathC_alone"]))
    va = xcheck["value_anchor"]
    print("[893]     value anchor: %d/%d assets' row-0 close == first ADMITTED bar's close; "
          "%d proven\n[893]       backward fills (last prefix row carries a price timestamped "
          "AFTER it); %d/%d first\n[893]       real rows match the ordinary forward fill"
          % (va["n_row0_close_equals_first_admitted_close"], va["n_checked"],
             va["n_proven_backward_fill"], va["n_first_real_row_matches_ffill"], va["n_checked"]))
    worst = max((p for p in per_asset if p["value_anchor"]
                 and p["value_anchor"].get("last_prefix_row_lead_seconds")),
                key=lambda p: p["prefix_rows_authoritative"], default=None)
    if worst is not None:
        w = worst["value_anchor"]
        print("[893]       e.g. %s: rows 0..%d all carry %r, the close of the bar at %s; the LAST "
              "of\n[893]       those rows is timestamped %s — %ds BEFORE the price it reports."
              % (worst["sym"], worst["prefix_rows_authoritative"] - 1, w["panel_row0_close"],
                 w["first_admitted_iso"], w["last_prefix_row_iso"],
                 w["last_prefix_row_lead_seconds"]))

    # ---------------------------------------------------------------- (2) re-measure
    from trader import strategy as stratmod
    pos, pres, w_, warm = stratmod.positions_from_config(logc, scfg)

    vr0_by_col = np.array([p["prefix_rows_authoritative"] for p in per_asset], dtype=np.int64)
    row_idx = np.arange(T)[:, None]
    prefix_mask = row_idx < vr0_by_col[None, :]           # (T, A) bool

    # Cells the scorer can actually reach: regime_series starts at max(warm, r_lo) and stops at
    # min(r_hi - H, T-1-H). Reported alongside the raw FULL overlap because the raw overlap
    # counts rows the backtest never evaluates.
    scored_lo = max(int(warm), full_lo)
    scored_hi = min(full_hi - scfg.H_rebal, T - 1 - scfg.H_rebal) + 1
    scored_cells = int(sum(_overlap(0, int(v), scored_lo, scored_hi) for v in vr0_by_col))
    print("[893]     of those, cells on the SCORED decision grid [%d,%d) (backtest warm=%d, H=%d): %d"
          % (scored_lo, scored_hi, warm, scfg.H_rebal, scored_cells))

    # ---------------------------------------------------------------- (3) exploitability
    entries_in_prefix = int(np.count_nonzero((pos != 0.0) & prefix_mask))
    present_in_prefix = int(np.count_nonzero(pres & prefix_mask))

    # STRUCTURAL COROLLARY — why (3) is 0 a priori, not as an empirical surprise.
    #   Inside a fabricated prefix [0, R) every close is bit-identical, so every log return
    #   logc[t+1]-logc[t] for t <= R-2 is EXACTLY 0.0. strategy.first_active returns the first
    #   return-row index with |r| > 1e-12, hence fa >= R-1. present_mask requires
    #   age = t - fa >= present_warm = max(max_lookback, vol_window) = 30 (note: NOT the same as
    #   build_positions' returned `warm` = that + 1, which is the backtest's start offset). For
    #   any t < R: t < R <= fa+1 < fa+present_warm, so age < present_warm and the cell is NOT
    #   present. Therefore no prefix cell can ever be present, positions there are 0, and every
    #   masked-vs-shipped delta is EXACTLY 0.0 by arithmetic — a corollary of first_active's
    #   definition, not an independent test of the fill's safety. Checked numerically, not
    #   merely asserted.
    present_warm = max(max(scfg.lookbacks), scfg.vol_window)
    fa = np.asarray(stratmod.first_active(logc))
    fa_ge_bound = bool(np.all(fa >= np.maximum(vr0_by_col - 1, 0)))
    corollary = {
        "claim": ("No leading-prefix cell can ever be `present`, so pos there is 0 and every "
                  "masked-vs-shipped delta is exactly 0.0 BY CONSTRUCTION."),
        "proof": ("Returns inside a bit-identical prefix [0,R) are exactly 0 => "
                  "strategy.first_active(fa) >= R-1 => for t < R, age = t-fa < present_warm => "
                  "present_mask False. (trader/strategy.py first_active + present_mask; "
                  "trader/data.py _ffill_bfill_cols.)"),
        "present_mask_warm": int(present_warm),
        "backtest_start_warm": int(warm),
        "first_active_ge_prefix_minus_1_all_assets": fa_ge_bound,
        "present_cells_inside_prefix": present_in_prefix,
        "nonzero_positions_inside_prefix": entries_in_prefix,
        "implication_for_D41": (
            "A zero measured exploitability here is NOT evidence that the backward fill is "
            "harmless in general — it is present_mask's own warm-up excluding the region by "
            "definition. Any future consumer that reads close_d / logc WITHOUT routing through "
            "strategy.present_mask (a different signal, a model featurizer, an eval harness, an "
            "external notebook) gets no such protection, and THIS measurement would not catch "
            "it. Cite the 0 as 'the shipped trend book cannot touch these cells', never as "
            "'the fill is safe'."),
    }
    print("\n[893] (3) nonzero as-shipped positions inside an asset's own leading prefix: %d"
          % entries_in_prefix)
    print("[893]     present-mask cells inside a leading prefix: %d" % present_in_prefix)
    print("[893]     STRUCTURAL COROLLARY: this 0 is determined a priori — first_active >= R-1 "
          "for all\n[893]     assets (%s), so age < present_mask warm=%d everywhere in the prefix "
          "and no cell is\n[893]     ever present. The delta below is 0 by arithmetic, not by "
          "measurement. It does NOT\n[893]     clear the fill for consumers that bypass "
          "present_mask." % (fa_ge_bound, present_warm))

    pos_masked = np.where(prefix_mask, 0.0, pos)
    n_forced = int(np.count_nonzero(pos_masked != pos))
    print("[893]     cells the mask actually changed (pos != pos_masked): %d" % n_forced)

    cell_shipped = bt.cell_metrics(logc, pos, pres, ts_d, scfg.H_rebal, warm, scfg=scfg)
    cell_masked = bt.cell_metrics(logc, pos_masked, pres, ts_d, scfg.H_rebal, warm, scfg=scfg)

    def _f(cell, reg, key):
        d = cell["reg"].get(reg)
        return d.get(key) if d else None

    compare_keys = [("FULL", "net8"), ("FULL", "net21"), ("FULL", "turn"),
                    ("FULL", "mult8"), ("FULL", "mult21"), ("FULL", "maxdd8"),
                    ("FULL", "ci8_lo"), ("FULL", "ci8_hi"),
                    ("FULL", "ci21_lo"), ("FULL", "ci21_hi"),
                    ("BEAR22", "net8"), ("BEAR22", "net21")]

    def _compare(cell_a, cell_b):
        out = {}
        print("\n  metric                as-shipped     variant       delta")
        print("  " + "-" * 58)
        for reg, key in compare_keys:
            a = _f(cell_a, reg, key)
            b = _f(cell_b, reg, key)
            d = (b - a) if (a is not None and b is not None) else None
            out[f"{reg}.{key}"] = {"shipped": a, "masked": b, "delta": d}
            print(f"  {reg+'.'+key:22s} {a:+10.4f}   {b:+10.4f}   {d:+10.6f}"
                  if a is not None and b is not None else f"  {reg+'.'+key:22s}  n/a")
        return out

    deltas = _compare(cell_shipped, cell_masked)

    # ---------------------------------------------------------------- positive control
    # NON-VACUITY REGRESSION TEST. The corollary above guarantees the real delta is 0, which on
    # its own is indistinguishable from a broken instrument. So: plant a nonzero position in a
    # prefix cell directly on the `pos` ARRAY — exactly what a consumer that bypasses
    # present_mask would produce — and re-run the SAME unmodified masking + SAME unmodified
    # bt.cell_metrics. Nothing is stubbed or monkeypatched; the function under test is the one
    # scoring. If the delta is not nonzero, the masking->scorer path is dead and every 0 above
    # is meaningless. Deterministic: deepest prefix, midpoint row.
    j_ctrl = int(np.argmax(vr0_by_col))
    control = {"applied": False, "reason": "no asset has a leading prefix on this panel"}
    if int(vr0_by_col[j_ctrl]) > scored_lo + 1:
        r_ctrl = (scored_lo + int(vr0_by_col[j_ctrl])) // 2
        pos_ctrl = pos.copy()
        pos_ctrl[r_ctrl, j_ctrl] = 0.5
        pos_ctrl_masked = np.where(prefix_mask, 0.0, pos_ctrl)
        print("\n[893] POSITIVE CONTROL (non-vacuity regression test): planting pos[%d, %s]=0.5 "
              "inside\n[893]   %s's %d-row prefix, bypassing present_mask the way an unprotected "
              "consumer would."
              % (r_ctrl, per_asset[j_ctrl]["sym"], per_asset[j_ctrl]["sym"],
                 int(vr0_by_col[j_ctrl])))
        ctrl_shipped = bt.cell_metrics(logc, pos_ctrl, pres, ts_d, scfg.H_rebal, warm, scfg=scfg)
        ctrl_masked = bt.cell_metrics(logc, pos_ctrl_masked, pres, ts_d, scfg.H_rebal, warm,
                                      scfg=scfg)
        ctrl_deltas = _compare(ctrl_shipped, ctrl_masked)
        moved = [k for k, v in ctrl_deltas.items()
                 if v["delta"] is not None and v["delta"] != 0.0]
        control = {
            "applied": True, "row": r_ctrl, "sym": per_asset[j_ctrl]["sym"],
            "asset_col": j_ctrl, "planted_position": 0.5,
            "prefix_rows_for_that_asset": int(vr0_by_col[j_ctrl]),
            "cells_mask_changed": int(np.count_nonzero(pos_ctrl_masked != pos_ctrl)),
            "metrics_that_moved": moved,
            "deltas": ctrl_deltas,
            "instrument_responsive": bool(moved),
        }
        print("[893]   control moved %d/%d metrics -> instrument RESPONSIVE: %s"
              % (len(moved), len(ctrl_deltas), bool(moved)))
        if not moved:
            print("[893]   *** CONTROL FAILED — the masking->scorer path is VACUOUS; every zero "
                  "delta above is\n[893]   *** meaningless. Investigate before citing any number "
                  "from this run.")

    result = {
        "task": "TASK_893",
        "panel": {
            "source": tconfig.NPZ_CACHE,
            "note": ("The panel is ALWAYS this frozen npz. --db never feeds it. Fixture and real "
                     "legs therefore score the SAME panel and MUST produce identical "
                     "remeasure_FULL_and_BEAR22 numbers; the fixture leg is a smoke test of this "
                     "script, not an independent measurement of the bar."),
            "n_assets": A, "n_panel_rows": T,
            "grid_start_iso": _iso(ts_d[0]), "grid_end_iso": _iso(ts_d[-1]),
            "grid_step_seconds": int(ts_d[1] - ts_d[0]),
            "build": ("runs/819_single_asset/portfolio.py::load_broad — build_close_aligned on a "
                      "1-MINUTE grid (ffill + leading bfill), then close_1m[::1440]. The bfill's "
                      "source row is a minute row and is essentially never on the daily grid, so "
                      "the leading bit-identical run in a daily column is entirely fabricated."),
        },
        "db": {
            "path": args.db,
            "role": ("CROSS-CHECK AND VALUE-ANCHOR SOURCE ONLY — never the panel, never the "
                     "positions, never the re-measurement."),
            "reaches_back_to_panel_start": db_reaches_back,
            "admission_predicate": ("timestamp >= grid_start AND timestamp < grid_end AND "
                                    "(timestamp - grid_start) % 60 == 0  (FUNCTIONALLY "
                                    "EQUIVALENT to build_close_aligned's admission, not "
                                    "textually identical: that function applies the modulo "
                                    "lattice test in numpy AFTER the SQL, this folds it into "
                                    "the SQL text; bars before grid_start were NOT admitted "
                                    "and do not prevent a leading backward fill)"),
        },
        "full_regime_rows": [full_lo, full_hi],
        "counts": {
            "per_asset": per_asset,
            "total_prefix_rows_whole_panel": total_prefix_rows,
            "total_prefix_rows_lower_bound": total_lo,
            "total_prefix_rows_upper_bound": total_hi,
            "total_decision_cells_in_FULL": total_decision_cells_full,
            "full_window_cells": full_window_cells,
            "share_of_FULL_window": share,
            "decision_cells_by_regime": regime_totals,
            "decision_cells_on_scored_grid": scored_cells,
            "scored_grid_rows": [scored_lo, scored_hi],
            "regime_coverage_gap": {
                "rows_in_FULL_belonging_to_no_named_regime": gap_rows,
                "decision_cells_in_those_rows": cells_in_gap,
                "note": ("PRE-EXISTING, not introduced by TASK_893: trader/config.py REGIMES "
                         "windows are resolved independently by backtest.regime_rows, leaving "
                         "these rows between two half-open windows. This is why "
                         "decision_cells_by_regime sums to less than "
                         "total_decision_cells_in_FULL. Out of scope for TASK_893."),
            },
            "path_cross_check": xcheck,
        },
        "exploitability": {
            "nonzero_positions_inside_own_prefix": entries_in_prefix,
            "present_cells_inside_own_prefix": present_in_prefix,
            "cells_mask_actually_changed": n_forced,
            "pos_masked_is_bit_identical_to_pos": n_forced == 0,
        },
        "structural_corollary": corollary,
        "remeasure_FULL_and_BEAR22": deltas,
        "positive_control": control,
        "paper_820_number": None,
        "paper_820_note": ("Task Scope §2 asks for the 820 paper-trading config number 'if it is "
                           "reproducible from committed artifacts'. CONSIDERED AND DELIBERATELY "
                           "DROPPED: the 820 paper lane's output is a running daily ledger "
                           "(config.PAPER_LEDGER / PAPER_STATE) accumulated by live wall-clock "
                           "steps, not a scalar recomputable from committed artifacts, so there "
                           "is nothing to re-measure deterministically and any figure quoted "
                           "would depend on when it was read. Recorded as absent-on-purpose."),
        "seed": scfg.seed, "n_boot": scfg.n_boot,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=lambda o: None))
    print("\n[893] wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

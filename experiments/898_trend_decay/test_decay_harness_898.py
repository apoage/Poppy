#!/usr/bin/env python3
"""TASK_898 — regression tests for the trend-decay + long-flat-variant instrument.

Read-only w.r.t. `trader/` and `fixtures/trading_fixture.db` (never regenerated, never
written). Run directly: `python test_decay_harness_898.py`.

  CHECK 1 (gate i)   -- score_window() and trader.backtest.window_metrics() are byte-identical
                        when handed the SAME (pos, pres, warm) on the SAME bounds. Proves
                        score_window is not a reimplementation with different math -- it is the
                        same primitives, just parameterized for an externally-built pos.
                        Swept over a boundary/thin/reversed/out-of-range window battery (not a
                        single window), on EVERY key of the dict, and separately for the
                        VARIANT path against the unmodified cell_metrics on the REGIMES bounds.
  CHECK 2 (gate ii)  -- a deliberately STALE synthetic table REFUSES by default, and proceeds
                        with a recorded epoch_caveat under --allow-stale.
  CHECK 3 (gate ii)  -- a deliberately SHORT synthetic table REFUSES by default.
  CHECK 4 (gate ii)  -- the real fixture (not stale/short enough to hard-refuse) asserts its
                        table epoch plainly and does NOT silently pass through data rot.
  CHECK 5 (gate iii) -- require_preregistered() refuses when PREREG.json is absent, refuses on
                        hash drift, and succeeds after preregister() -- ordering is enforced by
                        the code, not by discipline.
  CHECK 6            -- build_long_flat_positions never emits a negative position (positive
                        control: the construction actually differs from the frozen rule wherever
                        the frozen rule was short). Routed THROUGH require_preregistered() so
                        the checked-in test observes the gate-(iii) ordering it describes.
  CHECK 7            -- python -m trader selftest stays green (trader/ untouched by this row).
  CHECK 8            -- decision_grid() recovers exactly the rows regime_series scores, and
                        decision_exposure() therefore reports realized Sum|pos| on the scorer's
                        own grid; the long-flat variant measurably DE-LEVERAGES vs the baseline
                        (SPEC.md Amendment 1 / MAJOR 1(b)).
  CHECK 9            -- the leverage-matched comparator holds the baseline's gross exposure on
                        every matchable decision, never shorts, and is NOT a no-op on Sharpe --
                        while a CONSTANT multiple of the same book IS a no-op, which is exactly
                        why the match has to be per-decision (MAJOR 1(c)).
  CHECK 10 (gate ii) -- run()/assert_panel_epoch emit the FILTERED panel's own ts_sub min/max,
                        and a deliberately narrowed panel is reported as narrower_than_table
                        with a note (MAJOR 2).
"""
import datetime as _dt
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import decay_harness as dh                      # noqa: E402
from trader import config as tconfig             # noqa: E402
from trader import backtest as bt                # noqa: E402
from trader import strategy as tstrat            # noqa: E402
from trader import data as tdata                 # noqa: E402

FIXTURE_DB = _REPO_ROOT / "fixtures" / "trading_fixture.db"


def _dicts_identical(a, b):
    """Byte-identical comparison over the UNION of keys (rtol=0, atol=0; NaN == NaN). Returns
    (ok, first_mismatching_key)."""
    if a is None or b is None:
        return (a is None and b is None), ("<None-vs-dict>" if a is not b else None)
    keys = set(a) | set(b)
    for k in sorted(keys):
        if k not in a or k not in b:
            return False, k
        va, vb = a[k], b[k]
        if isinstance(va, (int, np.integer)) and not isinstance(va, bool):
            if int(va) != int(vb):
                return False, k
        elif not np.isclose(va, vb, equal_nan=True, rtol=0, atol=0):
            return False, k
    return True, None


def _window_battery():
    """Boundary / thin / reversed / out-of-range / straddling window bounds for the gate-(i)
    equivalence sweep. `None` means 'unbounded on that side' (the scorer's own default)."""
    E = bt._to_epoch
    return [
        ("unbounded_both", None, None),
        ("open_start_2021", E("2021-01-01"), None),
        ("open_end_2021", None, E("2021-01-01")),
        ("panel_head_2013", E("2013-10-06"), E("2014-01-01")),
        ("pre_panel", E("1990-01-01"), E("1995-01-01")),
        ("post_panel", E("2030-01-01"), E("2031-01-01")),
        ("straddles_panel_end", E("2026-01-01"), E("2027-01-01")),
        ("degenerate_equal_bounds", E("2022-01-01"), E("2022-01-01")),
        ("reversed_bounds", E("2023-01-01"), E("2022-01-01")),
        ("thin_1d", E("2022-06-01"), E("2022-06-02")),
        ("thin_3d", E("2022-06-01"), E("2022-06-04")),
        ("thin_6d", E("2022-06-01"), E("2022-06-07")),
        ("thin_8d", E("2022-06-01"), E("2022-06-09")),
        ("thin_14d", E("2022-06-01"), E("2022-06-15")),
        ("month_2022_06", E("2022-06-01"), E("2022-07-01")),
        ("year_2021", E("2021-01-01"), E("2022-01-01")),
        ("year_2022", E("2022-01-01"), E("2023-01-01")),
        ("year_2025", E("2025-01-01"), E("2026-01-01")),
        ("half_2022H1", E("2022-01-01"), E("2022-07-01")),
        ("half_2026H1", E("2026-01-01"), E("2026-07-01")),
        ("half_2026H2_past_end", E("2026-07-01"), E("2027-01-01")),
        ("leap_day_2024", E("2024-02-28"), E("2024-03-02")),
        ("master_forward", E("2021-01-01"), None),
        ("very_long_2013_2026", E("2013-01-01"), E("2026-07-06")),
    ]


def _make_synthetic_db(path, n_assets=6, n_days=40, day0="2026-01-01"):
    """A tiny standalone ohlcv_daily-only sqlite db (independent of the fixture) so the
    stale/short-table refusals can be exercised deterministically without touching or copying
    the real fixture."""
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE assets (id INTEGER PRIMARY KEY, symbol TEXT)")
    con.execute("""CREATE TABLE ohlcv_daily (
        asset_id INTEGER, timestamp INTEGER, open REAL, high REAL, low REAL, close REAL,
        volume REAL, trade_count INTEGER, source TEXT,
        PRIMARY KEY (asset_id, timestamp))""")
    t0 = int(_dt.datetime.strptime(day0, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc).timestamp())
    rng = np.random.RandomState(0)
    for a in range(n_assets):
        con.execute("INSERT INTO assets (id, symbol) VALUES (?, ?)", (a, f"SYN{a}/USD"))
        price = 100.0
        for d in range(n_days):
            price *= float(np.exp(rng.randn() * 0.01))
            ts = t0 + d * 86400
            con.execute(
                "INSERT INTO ohlcv_daily (asset_id, timestamp, open, high, low, close, volume,"
                " trade_count, source) VALUES (?,?,?,?,?,?,?,?,?)",
                (a, ts, price, price, price, price, 1.0, 1, "kraken_rest"))
    con.commit()
    con.close()


def main():
    failures = []

    # ---- CHECK 0 (gate iii): the live registration is checked BEFORE this file builds any ---
    # variant at all. SPEC.md's gate-(iii) claim is scoped to run(); this test honours the same
    # ordering rather than stepping around it, so no number below is derived from an unregistered
    # spec. (CHECK 5 exercises the refusal paths against a scratch PREREG.json.)
    print("[898] CHECK 0 -- live PREREG.json matches SPEC.md before any variant is built ...")
    gate_ok, gate_err = False, None
    try:
        dh.require_preregistered()
        gate_ok = True
    except RuntimeError as e:
        gate_err = str(e)
    print(f"    [{'PASS' if gate_ok else 'FAIL'}] live_prereg_ok={gate_ok}"
          + ("" if gate_ok else f" err={gate_err}"))
    if not gate_ok:
        failures.append(f"CHECK 0 (gate iii, live registration) — {gate_err}")

    # ---- CHECK 1 (gate i): score_window == window_metrics on identical (pos, bounds) --------
    print("[898] CHECK 1 -- score_window() vs trader.backtest.window_metrics() equivalence ...")
    tconfig.set_backend("numpy")
    ts_sub, close_d, ids, syms, dinfo = tdata.load_db_backtest_panel(db_path=str(FIXTURE_DB))
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where(close_d > 0, close_d, np.nan))
    scfg = tconfig.STRATEGY
    pos_base, pres_base, _, warm_base = tstrat.positions_from_config(logc, scfg)
    pos_var, pres_var, _, warm_var = dh.build_long_flat_positions(logc, scfg)

    # 1a: baseline path, swept over the boundary/thin/reversed/out-of-range battery, all keys
    bad, n_scored, n_none = [], 0, 0
    for label, s_ts, e_ts in _window_battery():
        d_ref = bt.window_metrics(logc, ts_sub, s_ts, e_ts, scfg)
        d_mine = dh.score_window(logc, pos_base, pres_base, ts_sub, scfg.H_rebal, warm_base,
                                  scfg, s_ts, e_ts)
        same, key = _dicts_identical(d_ref, d_mine)
        if not same:
            bad.append(f"{label}(key={key})")
        if d_ref is None:
            n_none += 1
        else:
            n_scored += 1
    # 1b: VARIANT path against the unmodified cell_metrics on the fixed REGIMES bounds
    cell_var = bt.cell_metrics(logc, pos_var, pres_var, ts_sub, scfg.H_rebal, warm_var, scfg)
    n_reg = 0
    for reg_name, reg_a, reg_b in tconfig.REGIMES:
        d_ref = cell_var["reg"][reg_name]
        d_mine = dh.score_window(logc, pos_var, pres_var, ts_sub, scfg.H_rebal, warm_var, scfg,
                                  bt._to_epoch(reg_a), bt._to_epoch(reg_b))
        if d_ref is None or d_mine is None:
            if (d_ref is None) != (d_mine is None):
                bad.append(f"variant/{reg_name}(None-mismatch)")
            continue
        # cell_metrics computes the CI/equity block only for FULL; compare the shared keys
        shared = set(d_ref) & set(d_mine)
        sub_ref = {k: d_ref[k] for k in shared}
        sub_mine = {k: d_mine[k] for k in shared}
        same, key = _dicts_identical(sub_ref, sub_mine)
        if not same:
            bad.append(f"variant/{reg_name}(key={key})")
        n_reg += 1
    ok1 = not bad and n_scored >= 8 and n_none >= 3 and n_reg >= 5
    print(f"    [{'PASS' if ok1 else 'FAIL'}] baseline windows: {n_scored} scored + {n_none} "
          f"thin/empty (both sides agreed None); variant vs cell_metrics: {n_reg} regimes; "
          f"mismatches={bad or 'none'}")
    if not ok1:
        failures.append("CHECK 1 (gate i equivalence) — score_window and window_metrics/"
                         f"cell_metrics disagree: {bad}")

    # ---- CHECK 2 (gate ii): stale table refuses, --allow-stale caveats ---------------------
    print("[898] CHECK 2 -- stale synthetic table refuses by default ...")
    tmp_dir = _HERE / "_scratch_test"
    tmp_dir.mkdir(exist_ok=True)
    stale_db = tmp_dir / "stale.db"
    if stale_db.exists():
        stale_db.unlink()
    _make_synthetic_db(stale_db, day0="2021-01-01", n_days=800)   # ends ~2023-03, long stale by "now"
    refused = False
    try:
        dh.assess_epoch(str(stale_db), now_utc=_dt.datetime(2026, 8, 22, tzinfo=_dt.timezone.utc))
    except RuntimeError:
        refused = True
    caveated = False
    try:
        block = dh.assess_epoch(str(stale_db),
                                 now_utc=_dt.datetime(2026, 8, 22, tzinfo=_dt.timezone.utc),
                                 allow_stale=True)
        caveated = "epoch_caveat" in block and block["stale"]
    except RuntimeError:
        caveated = False
    ok2 = refused and caveated
    print(f"    [{'PASS' if ok2 else 'FAIL'}] refused_by_default={refused} "
          f"caveated_with_allow_stale={caveated}")
    if not ok2:
        failures.append("CHECK 2 (gate ii stale) — stale table did not refuse by default and/or "
                         "did not caveat under --allow-stale.")

    # ---- CHECK 3 (gate ii): short table refuses ---------------------------------------------
    print("[898] CHECK 3 -- short synthetic table refuses by default ...")
    short_db = tmp_dir / "short.db"
    if short_db.exists():
        short_db.unlink()
    now = _dt.datetime.now(_dt.timezone.utc)
    _make_synthetic_db(short_db, day0=(now - _dt.timedelta(days=30)).strftime("%Y-%m-%d"),
                        n_days=20)
    refused_short = False
    try:
        dh.assess_epoch(str(short_db), now_utc=now)
    except RuntimeError:
        refused_short = True
    print(f"    [{'PASS' if refused_short else 'FAIL'}] refused_by_default={refused_short}")
    if not refused_short:
        failures.append("CHECK 3 (gate ii short) — a table spanning only ~30 days since "
                         "2021-01-01 did not hard-refuse.")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- CHECK 4 (gate ii): the real fixture asserts its epoch plainly ----------------------
    print("[898] CHECK 4 -- fixture table epoch is asserted plainly ...")
    lo, hi = dh.table_epoch(str(FIXTURE_DB))
    ok4 = lo < hi and dh._fmt_date(hi) is not None
    print(f"    [{'PASS' if ok4 else 'FAIL'}] fixture ohlcv_daily epoch "
          f"{dh._fmt_date(lo)} .. {dh._fmt_date(hi)}")
    if not ok4:
        failures.append("CHECK 4 (gate ii epoch assertion) — could not derive a sane table "
                         "epoch from the fixture.")

    # ---- CHECK 5 (gate iii): pre-registration ordering is code-enforced --------------------
    print("[898] CHECK 5 -- pre-registration ordering enforced by code ...")
    real_prereg = dh.PREREG_PATH
    real_bytes = real_prereg.read_bytes() if real_prereg.exists() else None
    tmp_prereg = _HERE / "_scratch_prereg.json"
    try:
        dh.PREREG_PATH = tmp_prereg
        if tmp_prereg.exists():
            tmp_prereg.unlink()
        missing_refused = False
        try:
            dh.require_preregistered()
        except RuntimeError:
            missing_refused = True
        dh.preregister()
        after_register_ok = False
        try:
            dh.require_preregistered()
            after_register_ok = True
        except RuntimeError:
            after_register_ok = False
        # tamper with the registered hash -> drift must refuse
        reg = json.loads(tmp_prereg.read_text())
        reg["spec_sha256"] = "0" * 64
        tmp_prereg.write_text(json.dumps(reg))
        drift_refused = False
        try:
            dh.require_preregistered()
        except RuntimeError:
            drift_refused = True
        ok5 = missing_refused and after_register_ok and drift_refused
        print(f"    [{'PASS' if ok5 else 'FAIL'}] missing_refused={missing_refused} "
              f"after_register_ok={after_register_ok} drift_refused={drift_refused}")
        if not ok5:
            failures.append("CHECK 5 (gate iii ordering) — require_preregistered() did not "
                             "enforce missing/drift refusal + post-register success.")
    finally:
        dh.PREREG_PATH = real_prereg
        if tmp_prereg.exists():
            tmp_prereg.unlink()
        if real_bytes is not None and not real_prereg.exists():
            real_prereg.write_bytes(real_bytes)

    # ---- CHECK 6: long-flat variant never shorts, and actually differs from baseline -------
    print("[898] CHECK 6 -- long-flat variant has zero negative positions and differs from "
          "baseline wherever baseline was short (built AFTER the live gate) ...")
    # gate_ok comes from CHECK 0, which ran before this file constructed any variant.
    no_shorts = bool(np.all(pos_var >= -1e-12))
    was_short = pos_base < -1e-12
    differs_where_short = bool(was_short.any() and np.all(pos_var[was_short] == 0.0))
    ok6 = gate_ok and no_shorts and differs_where_short
    print(f"    [{'PASS' if ok6 else 'FAIL'}] live_prereg_ok={gate_ok} no_shorts={no_shorts} "
          f"n_short_cells_flattened={int(was_short.sum())} "
          f"differs_where_short={differs_where_short}")
    if not ok6 and gate_ok:
        failures.append("CHECK 6 (variant positive control) — long-flat positions still short, "
                         "or the variant is vacuously identical to the baseline.")

    # ---- CHECK 7: python -m trader selftest stays green -------------------------------------
    print("[898] CHECK 7 -- python -m trader selftest ...")
    r = subprocess.run([sys.executable, "-m", "trader", "selftest"], cwd=str(_REPO_ROOT),
                        capture_output=True, text=True)
    ok7 = r.returncode == 0
    print(f"    [{'PASS' if ok7 else 'FAIL'}] returncode={r.returncode}")
    if not ok7:
        failures.append(f"CHECK 7 (trader selftest) — exited {r.returncode}:\n{r.stdout}\n{r.stderr}")

    # ---- CHECK 8: exposure is measured on the SCORER'S OWN grid, and the variant de-levers --
    print("[898] CHECK 8 -- decision_grid()/decision_exposure() vs the scorer's own decisions; "
          "long-flat de-leverage is measured, not assumed ...")
    grid_bad, expo_rows = [], []
    for label, s_ts, e_ts in _window_battery():
        grid = dh.decision_grid(logc.shape, pres_base, ts_sub, scfg.H_rebal, warm_base,
                                 s_ts, e_ts)
        d_ref = bt.window_metrics(logc, ts_sub, s_ts, e_ts, scfg)
        if d_ref is None:
            if grid is not None:
                grid_bad.append(f"{label}(grid present where scorer refused)")
            continue
        if grid is None:
            grid_bad.append(f"{label}(no grid where scorer scored)")
            continue
        # same decision count as the scorer, and an independent reconstruction of the grid
        T = logc.shape[0]
        r_lo = int(np.searchsorted(ts_sub, s_ts)) if s_ts is not None else 0
        r_hi = int(np.searchsorted(ts_sub, e_ts)) if e_ts is not None else int(ts_sub.size)
        lo = max(warm_base, r_lo)
        hi = min(r_hi - scfg.H_rebal, T - 1 - scfg.H_rebal)
        t_expect = np.arange(lo, hi + 1, scfg.H_rebal)
        if grid.size != d_ref["n"] or not np.array_equal(grid, t_expect):
            grid_bad.append(f"{label}(grid n={grid.size} vs scorer n={d_ref['n']})")
            continue
        eb = dh.decision_exposure(pos_base, grid)
        ev = dh.decision_exposure(pos_var, grid)
        # the exposure is literally mean Sum|pos| on those rows
        if not np.isclose(eb["gross_exposure_mean"],
                          float(np.mean(np.sum(np.abs(pos_base[grid]), axis=1))),
                          rtol=0, atol=0):
            grid_bad.append(f"{label}(exposure != mean Sum|pos| on grid)")
        if eb["gross_exposure_max"] > scfg.gross_cap + 1e-12:
            grid_bad.append(f"{label}(baseline gross {eb['gross_exposure_max']} > gross_cap)")
        expo_rows.append((label, eb["gross_exposure_mean"], ev["gross_exposure_mean"]))
    de_levered = [r for r in expo_rows if r[2] < r[1] - 1e-6]
    ok8 = (not grid_bad) and len(expo_rows) >= 8 and len(de_levered) == len(expo_rows)
    print(f"    [{'PASS' if ok8 else 'FAIL'}] windows={len(expo_rows)} "
          f"all_de_levered={len(de_levered)}/{len(expo_rows)} problems={grid_bad or 'none'}")
    for label, b, v in expo_rows[:4]:
        print(f"        {label:24} baseline gross={b:.3f}  long-flat={v:.3f} "
              f"({100.0 * v / b:.1f}% of baseline)")
    if not ok8:
        why = grid_bad or ["variant did not de-leverage on every window, so the MAJOR-1 "
                           "confound is not actually being measured"]
        failures.append(f"CHECK 8 (exposure instrumentation) — {why}")

    # ---- CHECK 9: the leverage-matched comparator actually matches, and is not a no-op ------
    print("[898] CHECK 9 -- leverage-matched comparator holds the baseline's exposure and "
          "changes Sharpe (a CONSTANT multiple would not) ...")
    pos_lm, lm_scale, lm_matchable = dh.build_leverage_matched_positions(pos_base, pos_var)
    gb = np.sum(np.abs(pos_base), axis=1)
    gl = np.sum(np.abs(pos_lm), axis=1)
    matched_ok = bool(np.allclose(gl[lm_matchable], gb[lm_matchable], rtol=1e-9, atol=1e-12))
    unmatched_flat = bool(np.all(gl[~lm_matchable] == 0.0)) if (~lm_matchable).any() else True
    lm_no_shorts = bool(np.all(pos_lm >= -1e-12))
    lm_cap_ok = bool(np.max(gl) <= scfg.gross_cap + 1e-9)
    # the row-wise match changes Sharpe; a constant multiple provably does not
    s_ts, e_ts = bt._to_epoch("2021-01-01"), None
    d_var = dh.score_window(logc, pos_var, pres_var, ts_sub, scfg.H_rebal, warm_var, scfg,
                             s_ts, e_ts)
    d_lm = dh.score_window(logc, pos_lm, pres_var, ts_sub, scfg.H_rebal, warm_var, scfg,
                            s_ts, e_ts)
    d_const = dh.score_window(logc, pos_var * 2.317, pres_var, ts_sub, scfg.H_rebal, warm_var,
                               scfg, s_ts, e_ts)
    const_is_noop = bool(np.isclose(d_var["net8"], d_const["net8"], rtol=1e-9, atol=0)
                         and np.isclose(d_var["net21"], d_const["net21"], rtol=1e-9, atol=0))
    rowwise_is_not_noop = not np.isclose(d_var["net8"], d_lm["net8"], rtol=1e-6, atol=0)
    ok9 = (matched_ok and unmatched_flat and lm_no_shorts and lm_cap_ok
           and const_is_noop and rowwise_is_not_noop)
    print(f"    [{'PASS' if ok9 else 'FAIL'}] matched={matched_ok} unmatched_rows_flat="
          f"{unmatched_flat} no_shorts={lm_no_shorts} gross<=cap={lm_cap_ok} "
          f"const_multiple_is_noop={const_is_noop} rowwise_changes_sharpe={rowwise_is_not_noop}")
    print(f"        MASTER net8: baseline={bt.window_metrics(logc, ts_sub, s_ts, e_ts, scfg)['net8']:.4f} "
          f"long-flat={d_var['net8']:.4f} lev-matched={d_lm['net8']:.4f}")
    if not ok9:
        failures.append("CHECK 9 (leverage-matched comparator) — the third series does not hold "
                         "the baseline's exposure, or is a no-op, or leaks shorts/leverage.")

    # ---- CHECK 10 (gate ii): the FILTERED panel's own ts_sub epoch is emitted ---------------
    print("[898] CHECK 10 -- filtered-panel ts_sub epoch asserted (gate ii, second assertion) ...")
    blk = dh.assess_epoch(str(FIXTURE_DB), now_utc=_dt.datetime.now(_dt.timezone.utc),
                          allow_stale=True)
    dh.assert_panel_epoch(blk, ts_sub)
    pe = blk.get("panel_epoch", {})
    required = ("panel_min_ts", "panel_min_date", "panel_max_ts", "panel_max_date",
                "panel_n_rows", "panel_span_days", "head_gap_days", "tail_gap_days",
                "narrower_than_table", "note")
    fields_ok = all(k in pe for k in required)
    real_ok = (fields_ok and pe["panel_min_ts"] == int(ts_sub[0])
               and pe["panel_max_ts"] == int(ts_sub[-1])
               and pe["panel_n_rows"] == int(ts_sub.size))
    # a deliberately NARROWED panel must be reported as narrower, with a note
    narrow_ts = ts_sub[int(ts_sub.size * 0.1): int(ts_sub.size * 0.8)]
    blk2 = dict(blk)
    dh.assert_panel_epoch(blk2, narrow_ts)
    pe2 = blk2["panel_epoch"]
    narrow_ok = (pe2["narrower_than_table"] and pe2["head_gap_days"] > 0
                 and pe2["tail_gap_days"] > 0 and "NARROWER" in pe2["note"])
    ok10 = fields_ok and real_ok and narrow_ok
    print(f"    [{'PASS' if ok10 else 'FAIL'}] panel epoch {pe.get('panel_min_date')} .. "
          f"{pe.get('panel_max_date')} rows={pe.get('panel_n_rows')} "
          f"narrower_than_table={pe.get('narrower_than_table')}; "
          f"narrowed-panel detected={narrow_ok}")
    if not ok10:
        failures.append("CHECK 10 (gate ii panel epoch) — the filtered panel's ts_sub min/max "
                         "is not asserted, or a narrower-than-table panel is not flagged.")

    print()
    if failures:
        print(f"[898] {len(failures)} FAILURE(S):")
        for f in failures:
            print("  - " + f)
        return 1
    print("[898] all checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

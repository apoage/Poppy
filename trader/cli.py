#!/usr/bin/env python3
"""TASK_820 CLI — `python -m trader {selftest|backtest|parity|update-data|paper-step|report}`."""

import argparse
import sys

import numpy as np

from . import config
from . import backtest as bt
from . import data as datamod
from . import paper as papermod
from . import selftest as selftestmod


# =============================================================================
# helpers
# =============================================================================

def _npz_frozen_logc():
    """Load the cached 819 broad panel, apply the liquidity filter, return (logc, ts_d, syms)."""
    close_d, ts_d, syms = datamod.load_npz_panel()
    close_d, syms, dropped, keep = datamod.apply_liquidity_filter(close_d, syms)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where(close_d > 0, close_d, np.nan))
    return logc, ts_d, syms, dropped


def _fmt(v, p=2, sign=True):
    if v is None or not np.isfinite(v):
        return "  -"
    return f"{v:+.{p}f}" if sign else f"{v:.{p}f}"


def _isolated_rcfg(which):
    """TASK_920 leg 4: isolate ONE default-off RiskConfig overlay mechanism instead of all
    three together, by neutralizing the other two (dd thresholds pushed past -100%, so dd
    (always in (-1,0]) never trips them; per-asset cap raised to 1.0, above the gross cap so
    it never binds; rebalance band zeroed so it never skips a trade). 'all' (default) returns
    RiskConfig(enabled=True) unchanged — the pre-920 --overlay behavior."""
    base = config.RiskConfig(enabled=True)
    if which == "all":
        return base
    if which == "cap":
        return config.RiskConfig(enabled=True, per_asset_cap=base.per_asset_cap,
                                  dd_halve=-2.0, dd_flat=-2.0, rebalance_band=0.0)
    if which == "dd":
        return config.RiskConfig(enabled=True, per_asset_cap=1.0,
                                  dd_halve=base.dd_halve, dd_flat=base.dd_flat,
                                  dd_deep_scale=base.dd_deep_scale, rebalance_band=0.0)
    if which == "band":
        return config.RiskConfig(enabled=True, per_asset_cap=1.0, dd_halve=-2.0, dd_flat=-2.0,
                                  rebalance_band=base.rebalance_band)
    raise ValueError(f"unknown --overlay-only {which!r}")


# (cli flag, args attribute, legacy default) for the three honesty-pack legs.
_HONESTY_FLAGS = (("--bh-book", "bh_book", "legacy-label"),
                  ("--accounting", "accounting", "log"),
                  ("--charge-drift-rebalance", "charge_drift_rebalance", False))


def _print_borrow_pack(args, logc, ts_d, net_bps, r_lo, r_hi, syms):
    """TASK_916 leg 1/3: print the margin-rollover (borrow) charge, additively, on top of the
    canon report printed ABOVE this call (unaffected). No-op when --borrow-model is the
    default 'none' (G1: bit-identical OFF)."""
    if args.borrow_model == "none":
        return
    rep = bt.borrow_window_report(logc, ts_d, config.STRATEGY, r_lo, r_hi, syms,
                                  rate_scenario=args.borrow_rate_scenario)
    if rep is None:
        print("[backtest] borrow overlay: window too thin, skipped", file=sys.stderr)
        return
    print(f"\n  --- TASK_916 borrow charge (--borrow-model {args.borrow_model} "
          f"--borrow-rate-scenario {args.borrow_rate_scenario}; reporting-only, canon numbers "
          f"above unchanged) ---")
    print(f"  mean short fraction of gross {rep['short_frac_mean']*100:.1f}%  "
          f"borrow drag {rep['borrow_bp_per_day']:.3f} bp/day-equivalent")
    print(f"    net@{net_bps} Sharpe {_fmt(rep.get(f'net{net_bps}'))} -> borrow-charged "
          f"{_fmt(rep.get(f'net{net_bps}_borrow'))}")
    for c in config.STRATEGY.costs_bp:
        print(f"    net@{c:<2d}: Sharpe {_fmt(rep[f'net{c}'])} -> borrow-charged "
              f"{_fmt(rep[f'net{c}_borrow'])}")
    for c in config.STRATEGY.gate_costs:
        print(f"    @{c}bp borrow-charged CI [{rep[f'ci{c}_borrow_lo']:+.2f},"
              f"{rep[f'ci{c}_borrow_hi']:+.2f}]")


def _unsupported_combo(args):
    """TASK_920 fix pass (run 53 cycle 3): flag combinations this build does NOT implement.
    Returns a refusal message (str) or None when the combination is supported.

    Both cases below were previously accepted SILENTLY — exit 0, empty stderr, and a printed
    number that was not the one requested:

      (a) any honesty-pack flag (legs 1-3) together with `--overlay`. `_print_honesty_pack`
          is called from the two overlay-OFF branches of `cmd_backtest` only, so the overlay
          branch dropped the request and printed its OWN `B&H` column — the LEGACY per-bar
          equal-DOLLAR reweight figure this task exists to declare "not buy-and-hold"
          (-0.15 on the pinned npz panel, OPPOSITE IN SIGN to the +0.29 true-B&H book the
          operator asked for). `--bh-book`'s own help text promises "REFUSES (non-zero exit,
          no number printed) if that book is undefined"; under `--overlay` the refusal
          machinery was never reached, so it could neither honour nor refuse the request.

      (b) `--overlay-only` without `--overlay`. Leg 4's entire yield IS the isolated-overlay
          Sharpe; without `--overlay` the plain overlay-OFF report printed instead (+1.09 on
          the pinned panel), so a harness reading that line would record the overlay-OFF
          number as the cap-only result.

    REFUSAL, not a note (cf. the `--include-dead` note in `cmd_backtest`): a note is enough
    when the printed number is still correctly labelled, and neither of these is — the figure
    on stdout is actively MISLABELED, so there is nothing safe to leave there. Wiring the
    honesty pack into the overlay branch is a real feature (the overlay path scores its own
    positions through `run_overlay_backtest`, not the `_decision_grid` the pack is built on)
    and is deliberately left as FOLLOW-UP work; until it exists the combination is refused
    loudly rather than dropped quietly."""
    if args.overlay_only != "all" and not args.overlay:
        return (f"[backtest] REFUSING: --overlay-only={args.overlay_only} requires --overlay "
                f"(TASK_920 leg 4 isolates a mechanism of the risk overlay, so there is "
                f"nothing to isolate with the overlay OFF). Continuing would print the plain "
                f"overlay-OFF report and silently drop the isolation, which is the entire "
                f"number this flag exists to produce. No figure printed. Re-run with "
                f"--overlay --overlay-only {args.overlay_only}.")
    if args.overlay:
        dropped = [flag for flag, attr, dflt in _HONESTY_FLAGS if getattr(args, attr) != dflt]
        if dropped:
            return (f"[backtest] REFUSING: {', '.join(dropped)} "
                    f"{'is' if len(dropped) == 1 else 'are'} not implemented together "
                    f"with --overlay (TASK_920 legs 1-3 are wired into the overlay-OFF report "
                    f"only). Continuing would DROP the request and print the legacy per-bar "
                    f"equal-dollar-reweight `B&H` column instead — the mislabeled figure this "
                    f"task exists to remove, and the opposite sign to the true-B&H book on the "
                    f"pinned panel. No figure printed. Re-run the honesty pack without "
                    f"--overlay (the canon overlay-OFF report is what it augments).")
        if args.borrow_model != "none":
            return (f"[backtest] REFUSING: --borrow-model {args.borrow_model} is not "
                    f"implemented together with --overlay (TASK_916's borrow charge is wired "
                    f"into the overlay-OFF report only, same reason as the TASK_920 honesty "
                    f"pack — the overlay path scores its own positions through "
                    f"run_overlay_backtest, not the _decision_grid the borrow report is built "
                    f"on). No figure printed. Re-run the borrow charge without --overlay.")
    return None


def _print_honesty_pack(args, logc, ts_d, net_bps, r_lo, r_hi):
    """TASK_920 legs 1-3: print the requested reporting-only overlays. No-op (prints nothing)
    when every flag is at its legacy default.

    Returns True normally; False when an overlay REFUSED to produce a number (the fail-loud
    path of bt.true_bh_series — an undefined buy-and-hold book). The caller turns that into a
    non-zero exit code: no honesty-pack figure is printed, and the run does not read as clean.
    The canon numbers printed ABOVE this call are unaffected and remain valid."""
    if args.bh_book == "legacy-label" and args.accounting == "log" and not args.charge_drift_rebalance:
        return True
    try:
        rep = bt.honesty_pack_report(logc, ts_d, config.STRATEGY, r_lo, r_hi, net_bps,
                                      bh_book=args.bh_book, accounting=args.accounting,
                                      charge_drift_rebalance=args.charge_drift_rebalance)
    except ValueError as e:
        print(f"[backtest] TASK_920 overlay REFUSED — no number produced: {e}", file=sys.stderr)
        return False
    if rep is None:
        print("[backtest] honesty-pack overlays: window too thin, skipped", file=sys.stderr)
        return True
    print("\n  --- TASK_920 scorer-honesty overlays (reporting-only; canon numbers above unchanged) ---")
    w = rep["window"]
    print(f"  window: {w['first_entry']} -> {w['last_exit']} ({w['n']} decisions, H={w['H']}d, "
          f"rows {w['row_lo']}..{w['row_hi']}) — every figure below is denominated on THIS span")
    if "bh_book" in rep:
        b = rep["bh_book"]
        print(f"  --bh-book true-bh: B&H Sharpe {_fmt(b['sharpe'])} (fixed shares, annual "
              f"re-form)  equity x{b['mult']:.2f}")
        # Delisting convention is DISCLOSED, never chosen silently (fix pass, run 53 cycle 3).
        ph = b["phantom"]
        print(f"    delisting convention: a holding the present-mask has RETIRED is marked at "
              f"its frozen last price until the next annual re-form (not written off). "
              f"Non-present share of book value: {ph['phantom_share_final']*100:.2f}% at the "
              f"last row ({ph['phantom_value_final']:.4f} of {ph['book_value_final']:.4f}), "
              f"{ph['phantom_share_max']*100:.2f}% max ({ph['phantom_day_max']}). The "
              f"alternative convention (delisting = total loss) is an operator modelling "
              f"call; this figure is computed under the one named here.")
    if "accounting" in rep:
        a = rep["accounting"]
        print(f"  --accounting simple: gross Sharpe {_fmt(a['gross_sharpe'])}  "
              f"net@{net_bps} Sharpe {_fmt(a['net_sharpe'])}  "
              f"(vs log accounting: {a['bp_per_day']:+.2f} bp/day, Sharpe delta {a['sharpe_delta']:+.3f})")
    if "drift_rebalance" in rep:
        d = rep["drift_rebalance"]
        trip = "  [G2 TRIPWIRE: net@21 moved > -0.10 -- disclosure, not re-run]" if d["tripwire"] else ""
        print(f"  --charge-drift-rebalance: ann_turn {d['ann_turn_canon']:.1f}x -> "
              f"{d['ann_turn_drift']:.1f}x ({d['turn_multiple']:.2f}x); net@21 Sharpe "
              f"{_fmt(d['sharpe_canon21'])} -> {_fmt(d['sharpe_drift21'])} "
              f"(delta {d['sharpe_delta21']:+.3f}); turnover share target-change="
              f"{d['share_target_change']*100:.0f}% drift-retrim={d['share_drift']*100:.0f}%{trip}")
    return True


# =============================================================================
# commands
# =============================================================================

def cmd_selftest(args):
    msgs = selftestmod.self_tests()
    for m in msgs:
        print("  " + m)
    print(f"\n[selftest] all {len(msgs)} checks PASS (numpy float64 determinism reference)")
    return 0


def cmd_parity(args):
    """Gate 1a: reproduce the 819 Phase-2 best cell (broad/equal/single30 H=1d, overlay OFF)."""
    config.set_backend("numpy")   # determinism reference
    logc, ts_d, syms, dropped = _npz_frozen_logc()
    print(f"[parity] npz panel: {logc.shape[0]} daily rows x {logc.shape[1]} assets "
          f"(dropped by liquidity: {dropped or 'none'})")
    cell = bt.compute_full_cell(logc, ts_d, config.STRATEGY)
    f = cell["reg"]["FULL"]
    b = cell["reg"]["BEAR22"]

    def get(reg_key, key):
        d = cell["reg"].get(reg_key)
        return d.get(key) if d else None

    print("\n  metric                    expected      got        tol      PASS")
    print("  " + "-" * 62)
    all_pass = True
    for label, (reg, key), exp, tol in bt.PARITY_TARGETS:
        got = get(reg, key)
        ok = got is not None and np.isfinite(got) and abs(got - exp) <= tol
        all_pass &= ok
        print(f"  {label:24s}  {exp:+9.4f}   {got:+9.4f}   {tol:6.3f}   {'PASS' if ok else 'FAIL'}")
    print("  " + "-" * 62)
    print(f"  CI(net@8) [{f['ci8_lo']:+.2f},{f['ci8_hi']:+.2f}]  "
          f"CI(net@21) [{f['ci21_lo']:+.2f},{f['ci21_hi']:+.2f}]  "
          f"ann_turn {f['ann_turn']:.1f}x  years {f['years']:.2f}")
    print(f"\n[parity] GATE 1a: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


def cmd_backtest(args):
    # Refuse unimplemented flag combinations BEFORE anything is loaded or printed, so the
    # refusal cannot be mistaken for a report (TASK_920 fix pass — see _unsupported_combo).
    refusal = _unsupported_combo(args)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 2
    config.set_backend(args.backend)
    net_bps = args.net_bps
    score_start = bt._to_epoch(args.score_start) if args.score_start else None
    score_end = bt._to_epoch(args.score_end) if args.score_end else None
    if score_start is not None and score_end is not None and score_end <= score_start:
        print("[backtest] --score-end must be after --score-start", file=sys.stderr)
        return 2
    windowed = score_start is not None or score_end is not None

    if args.source == "npz":
        logc, ts_d, syms, dropped = _npz_frozen_logc()
        skip_note = ""
        if args.include_dead:
            print("[backtest] note: --include-dead only affects --source db (npz universe is fixed)")
    else:
        # panel loads warmup history BEFORE score-start so signals/masks are live at the
        # first scored decision; universe = kraken_rest-covered USD-crypto (agg1m-only
        # assets skipped cleanly), or ALL stored assets incl. dead/delisted coins with
        # --include-dead (survivorship control; present-mask retires their constant tails).
        load_start = (score_start - bt.warmup_days() * config.SEC_PER_DAY
                      if score_start is not None else None)
        ts_d, close_d, ids, syms, dinfo = datamod.load_db_backtest_panel(
            start_ts=load_start, end_ts=score_end, include_dead=args.include_dead)
        with np.errstate(divide="ignore", invalid="ignore"):
            logc = np.log(np.where(close_d > 0, close_d, np.nan))
        if args.include_dead:
            skip_note = (f" dead-included: {dinfo['n_dead_included']} no-{dinfo['source_filter']}; "
                         f"skipped: {dinfo['n_skipped_nonusd']} non-USD/stable, "
                         f"{dinfo['n_skipped_empty']} empty-in-window")
        else:
            skip_note = (f" skipped: {dinfo['n_skipped_source']} no-{dinfo['source_filter']}, "
                         f"{dinfo['n_skipped_nonusd']} non-USD/stable, "
                         f"{dinfo['n_skipped_empty']} empty-in-window")

    win_lo = args.score_start or "panel-start"
    win_hi = args.score_end or "latest"
    print(f"[backtest] source={args.source} backend={config.backend_name()} "
          f"overlay={'ON' if args.overlay else 'OFF'} net-bps={net_bps}")
    print(f"[backtest] n_assets={logc.shape[1]} panel_rows={logc.shape[0]} "
          f"window=[{win_lo} .. {win_hi})" + skip_note)

    if not args.overlay and not windowed:
        cell = bt.compute_full_cell(logc, ts_d, config.STRATEGY)
        f = cell["reg"]["FULL"]
        print(f"[backtest] n_decisions={f['n']}")
        print(f"\n  FULL  net@{net_bps} Sharpe {_fmt(f.get(f'net{net_bps}'))}  "
              f"gross {_fmt(f['gross'])}  B&H {_fmt(f['bh'])}  turnover/day {f['turn']:.4f}")
        for c in config.STRATEGY.costs_bp:
            print(f"    net@{c:<2d}: Sharpe {_fmt(f[f'net{c}'])}")
        print(f"  equity x@{net_bps if net_bps in (8,21) else 8}bp "
              f"{f.get('mult8' if net_bps<=8 else 'mult21', f['mult8']):.2f}  "
              f"maxDD {f['maxdd8']*100:+.1f}%  ann_turn {f['ann_turn']:.1f}x")
        print(f"  per-regime net@8: " + "  ".join(
            f"{n}={_fmt(cell['reg'][n]['net8']) if cell['reg'].get(n) else ' -'}"
            for n in config.REG_NAMES if n != "FULL"))
        r_lo, r_hi = bt.regime_rows(ts_d)["FULL"]
        if not _print_honesty_pack(args, logc, ts_d, net_bps, r_lo, r_hi):
            return 1
        _print_borrow_pack(args, logc, ts_d, net_bps, r_lo, r_hi, syms)
    elif not args.overlay:
        d = bt.window_metrics(logc, ts_d, score_start, score_end, config.STRATEGY)
        if d is None:
            print("[backtest] score window too thin (needs >= 6 decisions after warmup)",
                  file=sys.stderr)
            return 1
        print(f"[backtest] n_decisions={d['n']} ({d['years']:.2f} yr)")
        print(f"\n  WINDOW  net@{net_bps} Sharpe {_fmt(d.get(f'net{net_bps}'))}  "
              f"gross {_fmt(d['gross'])}  B&H {_fmt(d['bh'])}  turnover/day {d['turn']:.4f}")
        for c in config.STRATEGY.costs_bp:
            print(f"    net@{c:<2d}: Sharpe {_fmt(d[f'net{c}'])}")
        for c in config.STRATEGY.gate_costs:
            print(f"  @{c}bp: CI[{d[f'ci{c}_lo']:+.2f},{d[f'ci{c}_hi']:+.2f}]  "
                  f"equity x{d[f'mult{c}']:.2f}  maxDD {d[f'maxdd{c}']*100:+.1f}%")
        print(f"  B&H over window: x{d['mult_bh']:.2f} (Sharpe {_fmt(d['bh'])})  "
              f"ann_turn {d['ann_turn']:.1f}x")
        r_lo = int(np.searchsorted(ts_d, score_start)) if score_start is not None else 0
        r_hi = int(np.searchsorted(ts_d, score_end)) if score_end is not None else ts_d.size
        if not _print_honesty_pack(args, logc, ts_d, net_bps, r_lo, r_hi):
            return 1
        _print_borrow_pack(args, logc, ts_d, net_bps, r_lo, r_hi, syms)
    else:
        rcfg = _isolated_rcfg(args.overlay_only)
        r = bt.run_overlay_backtest(logc, ts_d, config.STRATEGY, rcfg,
                                    equity_cost_bp=float(net_bps),
                                    lo_ts=score_start, hi_ts=score_end)
        period = config.STRATEGY.H_rebal * config.K
        g, tu, bh = r["gross"], r["turn"], r["bh"]
        nh = bt.net_series(g, tu, net_bps)
        sh = bt._sharpe(nh, period)
        ci = bt._block_bootstrap_sharpe_ci(nh, period, period, config.STRATEGY.n_boot, config.STRATEGY.seed)
        eq = np.exp(np.cumsum(nh))
        peak = np.maximum.accumulate(eq)
        maxdd = float(np.min(eq / peak - 1.0))
        tag = "WINDOW" if windowed else "FULL"
        print(f"[backtest] n_decisions={r['n']}")
        if args.overlay_only != "all":
            print(f"[backtest] --overlay-only={args.overlay_only} (TASK_920 leg 4: isolated "
                  f"overlay, other two mechanisms neutralized)")
        print(f"\n  {tag} (overlay ON, dd_mode={r['dd_mode']})  net@{net_bps} Sharpe {_fmt(sh)} "
              f"CI[{ci[0]:+.2f},{ci[1]:+.2f}]  gross {_fmt(bt._sharpe(g, period))}  "
              f"B&H {_fmt(bt._sharpe(bh, period))}")
        for c in config.STRATEGY.costs_bp:
            print(f"    net@{c:<2d}: Sharpe {_fmt(bt._sharpe(bt.net_series(g, tu, c), period))}")
        print(f"  turnover/day {np.mean(tu):.4f}  equity x{eq[-1]:.2f}  maxDD {maxdd*100:+.1f}%  "
              f"shadow x{r['shadow_mult']:.2f}")
        print(f"  overlay activity: steps={r['n']} capped_steps={r['n_capped']} "
              f"half_steps={r['n_half']} deep_steps={r['n_deep']} halt_steps={r['n_halt']} "
              f"stale_steps={r['n_stale']}")
    return 0


def cmd_update_data(args):
    if not args.asset_ids and not args.universe:
        print("[update-data] refusing blind full run: pass --asset-ids 1091,359,... or --universe",
              file=sys.stderr)
        return 2
    if args.universe:
        univ = datamod.build_pit_universe(min_bars=args.min_bars)
        ids = [a for a, s in univ]
        print(f"[update-data] PIT universe: {len(ids)} USD-crypto assets")
    else:
        ids = [int(x) for x in args.asset_ids.split(",")]
    start = bt._to_epoch(args.start) if args.start else None
    end = bt._to_epoch(args.end) if args.end else None
    written = datamod.update_data(args.db, ids, start, end, source="agg1m")
    total = sum(written.values())
    print(f"[update-data] wrote {total} daily bars across {len(ids)} assets -> ohlcv_daily "
          f"(db={args.db})")
    for aid, n in list(written.items())[:20]:
        print(f"    asset {aid}: {n} days")
    return 0


def cmd_paper_step(args):
    now_ts = int(args.now_ts) if args.now_ts else None
    if args.dry_run and args.source == "npz" and now_ts is None:
        # fake "today" = last cached day so the staleness guard does not fire
        close_d, ts_d, syms = datamod.load_npz_panel()
        now_ts = int(ts_d[-1])
    kw = {}
    if args.state_dir:
        from pathlib import Path
        d = Path(args.state_dir)
        kw["state_path"] = d / "state.json"
        kw["ledger_path"] = d / "ledger.csv"
    res = papermod.paper_step(source=args.source, now_ts=now_ts, **kw)
    return 0


def cmd_report(args):
    kw = {}
    if args.state_dir:
        from pathlib import Path
        d = Path(args.state_dir)
        kw["state_path"] = d / "state.json"
        kw["ledger_path"] = d / "ledger.csv"
    papermod.report(**kw)
    return 0


# =============================================================================
# parser
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(prog="trader", description="TASK_820 trend CTA")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="run ST1-ST6 construction self-tests")

    pb = sub.add_parser("backtest", help="frozen strategy backtest")
    pb.add_argument("--net-bps", type=int, default=8, help="report/equity cost bp/side (default 8)")
    pb.add_argument("--overlay", action="store_true", help="risk overlay ON (else 819 parity math)")
    pb.add_argument("--backend", choices=["numpy", "jax"], default="numpy")
    pb.add_argument("--source", choices=["npz", "db"], default="npz")
    pb.add_argument("--score-start", type=str, default=None, metavar="YYYY-MM-DD",
                    help="score decisions only from this UTC date (panel warms up before it)")
    pb.add_argument("--score-end", type=str, default=None, metavar="YYYY-MM-DD",
                    help="score decisions only before this UTC date (exclusive)")
    pb.add_argument("--include-dead", action="store_true",
                    help="db source: include dead/delisted assets (agg1m-only, no kraken_rest "
                         "rows) for survivorship-honest historical windows; present-mask "
                         "retires their constant post-delisting tails")
    pb.add_argument("--bh-book", choices=["legacy-label", "true-bh"], default="legacy-label",
                    help="TASK_920 leg 1: B&H accounting. legacy-label (default) is the "
                         "existing column -- per-bar equal-DOLLAR reweight, zero turnover "
                         "charged; NOT buy-and-hold. true-bh computes a real buy-and-hold "
                         "book (fixed shares, annual re-form) and REFUSES (non-zero exit, no "
                         "number printed) if that book is undefined -- no asset present on a "
                         "calendar-year re-form row, or a held asset priced non-finite")
    pb.add_argument("--accounting", choices=["log", "simple"], default="log",
                    help="TASK_920 leg 2: gross/net return accounting. log (default) is the "
                         "canon 819/820 summed-log-return convention (cannot see a "
                         "rebalancing premium by construction). simple uses dollar-weighted "
                         "SIMPLE returns instead")
    pb.add_argument("--charge-drift-rebalance", action="store_true",
                    help="TASK_920 leg 3: also charge the turnover needed to re-trim "
                         "constant-dollar POSITION DRIFT between rebalances (uncharged by "
                         "default, target-weight changes only); prints the target-change "
                         "vs drift-retrim decomposition")
    pb.add_argument("--overlay-only", choices=["all", "cap", "dd", "band"], default="all",
                    help="TASK_920 leg 4 (with --overlay only): isolate ONE risk-overlay "
                         "mechanism -- cap (5%% per-asset cap) | dd (shadow-scale DD "
                         "throttle) | band (0.2%% rebalance band) -- instead of all (default, "
                         "pre-920 --overlay behavior, all three together)")
    pb.add_argument("--borrow-model", choices=["none", "kraken-margin"], default="none",
                    help="TASK_916: margin-rollover (borrow) charge on SHORT exposure, per "
                         "unit time held (D81(c) denominator -- shorted notional, NOT "
                         "turnover). none (default) prints nothing, canon report unaffected "
                         "(G1: bit-identical). kraken-margin adds a borrow-charged net(c) "
                         "column + CI, reporting-only, below the canon report")
    pb.add_argument("--borrow-rate-scenario", choices=["low", "high"], default="low",
                    help="TASK_916 (with --borrow-model kraken-margin only): which end of "
                         "Kraken's quoted per-asset-class rollover range to charge -- low = "
                         "0.01%%/4h BTC, 0.02%%/4h alts (~21.9%%/~43.8%% annualized); high = "
                         "0.02%%/4h BTC, 0.04%%/4h alts (~43.8%%/~87.6%% annualized)")

    sub.add_parser("parity", help="gate 1a: reproduce the 819 Phase-2 best cell")

    pu = sub.add_parser("update-data", help="aggregate 1-min -> ohlcv_daily (streaming)")
    pu.add_argument("--asset-ids", type=str, default="", help="comma-separated asset ids")
    pu.add_argument("--universe", action="store_true", help="use the discovered PIT USD-crypto universe")
    pu.add_argument("--min-bars", type=int, default=0, help="universe coverage floor (1-min bars)")
    pu.add_argument("--start", type=str, default=None, help="YYYY-MM-DD start (optional)")
    pu.add_argument("--end", type=str, default=None, help="YYYY-MM-DD end exclusive (optional)")
    pu.add_argument("--db", type=str, default=config.DB_PATH, help="sqlite db path")

    pp = sub.add_parser("paper-step", help="one daily paper step (overlay ON)")
    pp.add_argument("--source", choices=["npz", "db"], default="db")
    pp.add_argument("--dry-run", action="store_true", help="fake today=last cached day (npz)")
    pp.add_argument("--now-ts", type=str, default=None, help="override wallclock (epoch sec)")
    pp.add_argument("--state-dir", type=str, default=None, help="state/ledger dir override")

    pr = sub.add_parser("report", help="paper state + ledger tail")
    pr.add_argument("--state-dir", type=str, default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {
        "selftest": cmd_selftest, "parity": cmd_parity, "backtest": cmd_backtest,
        "update-data": cmd_update_data, "paper-step": cmd_paper_step, "report": cmd_report,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

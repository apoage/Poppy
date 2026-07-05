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
    else:
        rcfg = config.RiskConfig(enabled=True)
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

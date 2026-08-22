#!/usr/bin/env python3
"""TASK_898 — trend-decay aggregation + the long-flat/spot-legal variant.

Builds and verifies the INSTRUMENT (per DECISIONS D54 the real leg runs planner-scheduled on
gpu-server, never inside this factory loop). This script emits NUMBERS, never a verdict —
FALLBACK STANDS / FALLBACK DEAD / BOTH DEAD is the planner's ruling against `docs/pathway.md`.

Read `SPEC.md` in this directory first — it is the pre-registration this file enforces.

`trader/` is never modified and never imported-and-mutated. Every scored number is produced by
calling existing public `trader.backtest` / `trader.strategy` functions with plain arguments:

  - the frozen 819/820 rule's positions come from `trader.strategy.positions_from_config`
    (unmodified).
  - the long-flat variant's positions come from `build_long_flat_positions` below, which calls
    the SAME unmodified `trader.strategy` primitives (`rolling_vol`, `first_active`,
    `trend_signal`[_tanh], `present_mask`, `weights`) and differs from
    `trader.strategy.build_positions` by exactly one line (`sig = np.maximum(sig, 0.0)` — every
    short signal replaced by flat). NOTE (SPEC.md Amendment 1): that one-line diff is NOT a
    portfolio-level no-op. `weights()` never sees the signal's sign, so the flattened short
    book's budget is NOT reallocated onto the longs — it becomes IMPLICIT CASH, and the variant
    therefore runs at a STATE-DEPENDENT lower gross exposure than the frozen rule (worst
    exactly in bear regimes, where the frozen rule was mostly short). Sharpe is invariant to a
    CONSTANT leverage multiplier but not to a state-dependent one, so this is a real confound
    and is made visible two ways: every scored series carries its own realized gross exposure
    (`gross_exposure_*`, from `decision_exposure` below), and a THIRD leverage-matched series
    is scored alongside (`build_leverage_matched_positions`).
  - the leverage-matched comparator's positions come from `build_leverage_matched_positions`
    below: the long-flat book rescaled ROW-WISE so its per-decision gross exposure equals the
    frozen rule's on that same row. It isolates signal quality from sitting in cash. It is a
    DIAGNOSTIC comparator, not a deployable book (the deployable one is the de-leveraged
    long-flat variant — see SPEC.md Scope 2).
  - the fixed-REGIMES report (Scope 3) calls `trader.backtest.cell_metrics` UNMODIFIED for both
    the frozen rule and the variant (it already accepts caller-supplied `pos`/`pres`/`warm` —
    that is how TASK_820/893 always used it).
  - the arbitrary decay windows (Scope 1/2, which extend past `REGIMES`' fixed FULL end) are
    scored by `score_window` below, which is NOT a reimplementation of the scorer: it calls the
    exact same four primitives `trader.backtest.window_metrics` calls internally
    (`regime_series`, `net_series`, `_sharpe`, `_block_bootstrap_sharpe_ci`) — `window_metrics`
    itself cannot be reused unmodified here because it always rebuilds `pos` from
    `strategy.positions_from_config` and has no parameter for an externally supplied `pos`.
    `test_decay_harness_898.py` proves `score_window` and `window_metrics` produce
    byte-identical dicts when both are handed the frozen rule's own positions on identical
    bounds.

CLI:
    python decay_harness.py preregister                 # write/verify PREREG.json (gate iii)
    python decay_harness.py run --db <path> [--allow-stale] [--out <path>]
"""
import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trader import config as tconfig          # noqa: E402
from trader import backtest as bt              # noqa: E402
from trader import strategy as tstrat          # noqa: E402
from trader import data as tdata               # noqa: E402

SPEC_PATH = _HERE / "SPEC.md"
PREREG_PATH = _HERE / "PREREG.json"

# --- gate (ii) thresholds (this row's own encoded discipline; see SPEC.md) -------------------
WARN_STALE_DAYS = 14
HARD_REFUSE_STALE_DAYS = 120
MIN_SPAN_DAYS = 365
RESEARCH_WINDOW_START = "2021-01-01"   # 819's original research-window start (unchanged)

# Carried into every artifact so the amended pre-registration is visible where the numbers are
# read, not only in SPEC.md's history (see SPEC.md "Amendment 1").
SPEC_AMENDMENT_NOTE = (
    "SPEC.md carries Amendment 1 (2026-08-22) and was RE-REGISTERED with "
    "`preregister --force`, so the spec_sha256 below is the AMENDED spec's hash, not the "
    "original registration's. The amendment (a) corrects a false claim that the long-flat "
    "variant is 'identical in every respect with exactly one change' — true at line-diff level, "
    "false at portfolio level, because the clipped short budget becomes implicit cash rather "
    "than being reallocated onto longs; (b) requires realized gross exposure beside every "
    "Sharpe; (c) adds the leverage-matched comparator series. This is a legitimate amendment "
    "rather than p-hacking because NO real-data leg has run: the only numbers in existence at "
    "amendment time are fixture-scale instrument-verification numbers, which are not scores "
    "this row's verdict bands can be tuned against. The real-data leg (D54, gpu-server) runs "
    "strictly after this hash."
)


# =============================================================================
# Scope 2 — long-flat / no-short variant (pre-registered in SPEC.md)
# =============================================================================

def build_long_flat_positions(logc, scfg, xp=None):
    """Identical to `trader.strategy.build_positions` (called with `scfg`'s own lookbacks/
    sizing/vol_window/signal) except every SHORT signal is replaced by FLAT before weighting:
    `sig = max(sig, 0)`. Calls only the unmodified `trader.strategy` primitives; `trader/`
    itself is never edited. Returns (pos, pres, w, warm) — same shape contract as
    `trader.strategy.build_positions` / `positions_from_config`."""
    lookbacks = list(scfg.lookbacks)
    max_lb = max(lookbacks)
    rv = tstrat.rolling_vol(logc, scfg.vol_window, xp=xp)
    fa = tstrat.first_active(logc, xp=xp)
    if scfg.signal == "tanh":
        sig = tstrat.trend_signal_tanh(logc, lookbacks, scfg.vol_window, xp=xp)
    else:
        sig = tstrat.trend_signal(logc, lookbacks, xp=xp)
    xp_ = tstrat._xp(xp)
    sig = xp_.maximum(sig, 0.0)                                      # <-- the ONLY rule change
    pres = tstrat.present_mask(logc, rv, fa, max_lb, scfg.vol_window, xp=xp)
    sig = xp_.where(pres, sig, 0.0)
    w = tstrat.weights(pres, rv, scfg.sizing, xp=xp)
    pos = w * sig
    warm = max(max_lb, scfg.vol_window) + 1
    return pos, pres, w, warm


# =============================================================================
# Scope 2b — leverage-matched comparator (SPEC.md Amendment 1)
# =============================================================================

LM_EPS = 1e-12


def build_leverage_matched_positions(pos_base, pos_var):
    """The long-flat book rescaled ROW-WISE to the frozen rule's own realized gross exposure:

        pos_lm[t,:] = pos_var[t,:] * (Sum_i|pos_base[t,i]| / Sum_i|pos_var[t,i]|)

    so `Sum_i|pos_lm[t,i]| == Sum_i|pos_base[t,i]|` on every row where the variant holds
    anything at all. This strips out the state-dependent cash drag that the long-flat
    construction silently introduces (see SPEC.md Amendment 1) and leaves only the question the
    row exists to ask: is the LONG-ONLY signal itself worth anything, once it is not being
    quietly compared at a fraction of the baseline's leverage?

    A CONSTANT multiplier would be a no-op on Sharpe; this one is per-decision and therefore is
    NOT. Rows where the variant is entirely flat (every present asset short) cannot be matched —
    the long-only book has nothing to hold — so they stay in cash and are counted as
    `unmatched` in the reported exposure block: a residual, disclosed confound, not a fixed one.

    Gross stays capped: Sum|pos_lm| == Sum|pos_base| <= gross_cap by construction. Returns
    (pos_lm, scale, matchable) where `scale` is the (T,1) row multiplier and `matchable` the
    (T,) bool mask of rows that could be matched."""
    gb = np.sum(np.abs(pos_base), axis=1, keepdims=True)
    gv = np.sum(np.abs(pos_var), axis=1, keepdims=True)
    matchable = (gv > LM_EPS)
    scale = np.where(matchable, gb / np.maximum(gv, LM_EPS), 0.0)
    return pos_var * scale, scale, matchable[:, 0]


# =============================================================================
# realized gross exposure, measured on the SCORER'S OWN decision grid
# =============================================================================

def decision_grid(shape, pres, ts_sub, H, warm, start_ts=None, end_ts=None):
    """The EXACT row indices `trader.backtest.regime_series` would score for this window —
    recovered from the unmodified primitive itself, never re-derived.

    `regime_series(logc, pos, ...)` returns `gross_t = Sum_i pos[t,i] * (logc[t+H]-logc[t])`.
    Called with a synthetic unit-ramp panel (`logc_ramp[t,i] = t`, so every forward label is
    exactly `H`) it degenerates to `gross_t = H * Sum_i pos[t,i]`. Feeding it `pos[t,i] = t`
    gives `A*H*t`, and feeding it `pos[t,i] = 1` gives `A*H`; the ratio is `t` itself. Same
    panel shape => same grid, same warmup, same bounds, and the same thin-window refusal
    (returns None exactly when the scorer would).

    Everything downstream (exposure, leverage stats) is then a plain lookup on those rows —
    no window arithmetic of our own anywhere."""
    T_rows, A = int(shape[0]), int(shape[1])
    if A == 0:
        return None
    pres = np.asarray(pres)
    ts_sub = np.asarray(ts_sub)
    r_lo = int(np.searchsorted(ts_sub, start_ts)) if start_ts is not None else 0
    r_hi = int(np.searchsorted(ts_sub, end_ts)) if end_ts is not None else int(ts_sub.size)
    ramp = np.broadcast_to(np.arange(T_rows, dtype=np.float64)[:, None], (T_rows, A))
    res_idx = bt.regime_series(ramp, ramp, pres, r_lo, r_hi, H, warm, xp=np)
    if res_idx is None:
        return None
    res_one = bt.regime_series(ramp, np.ones((T_rows, A), dtype=np.float64), pres,
                               r_lo, r_hi, H, warm, xp=np)
    return np.rint(res_idx[0] / res_one[0]).astype(int)


def decision_exposure(pos, t_grid):
    """Realized gross exposure `Sum_i |pos[t,i]|` over exactly the decisions the scorer scores.

    This is the number MAJOR-1 exists for: the long-flat variant's clipped shorts are not
    reallocated onto longs (`trader.strategy.weights` never sees the signal's sign), so the
    variant runs at a lower, STATE-DEPENDENT leverage than the frozen rule. No Sharpe in this
    artifact may be read without this beside it."""
    if t_grid is None or t_grid.size == 0:
        return None
    expo = np.sum(np.abs(tconfig.to_numpy(pos)[t_grid]), axis=1)
    return {
        "gross_exposure_mean": float(np.mean(expo)),
        "gross_exposure_min": float(np.min(expo)),
        "gross_exposure_max": float(np.max(expo)),
        "gross_exposure_n": int(expo.size),
    }


def leverage_match_stats(scale, matchable, t_grid):
    """Diagnostics for the leverage-matched comparator on the scorer's own decision grid:
    how hard the rescale had to push, and how often it could not be applied at all (every
    present asset short => the long-only book has nothing to hold and stays in cash — a
    residual, DISCLOSED confound that leverage-matching cannot remove)."""
    if t_grid is None or t_grid.size == 0:
        return None
    s = np.asarray(scale).reshape(-1)[t_grid]
    m = np.asarray(matchable).reshape(-1)[t_grid]
    s_ok = s[m] if m.any() else np.array([0.0])
    return {
        "lm_scale_mean": float(np.mean(s_ok)),
        "lm_scale_max": float(np.max(s_ok)),
        "lm_unmatched_frac": float(np.mean(~m)),
    }


def annotate_exposure(d, expo):
    """Attach a scored series' realized gross exposure to its metrics dict AFTER scoring, so
    `score_window`'s dict shape stays byte-identical to `window_metrics`'s (gate i, CHECK 1).
    No Sharpe number in this artifact is readable without its leverage sitting beside it."""
    if d is None or expo is None:
        return d
    d.update(expo)
    return d


def _leverage_summary(base_m, var_m, lm_m):
    """The confound, stated once per window/regime in one place a reader cannot miss: how much
    of the long-flat book was actually deployed versus the frozen rule's, and therefore how much
    of any Sharpe difference could be cash rather than signal."""
    def _g(d):
        return None if d is None else d.get("gross_exposure_mean")
    b, v, m = _g(base_m), _g(var_m), _g(lm_m)
    out = {"baseline_gross_mean": b, "variant_gross_mean": v, "lev_matched_gross_mean": m,
           "variant_over_baseline": (float(v / b) if (b and v is not None and b > 0) else None)}
    if lm_m is not None:
        for k in ("lm_scale_mean", "lm_scale_max", "lm_unmatched_frac"):
            out[k] = lm_m.get(k)
    if out["variant_over_baseline"] is not None:
        out["note"] = (
            f"the long-flat variant deployed {100.0 * out['variant_over_baseline']:.1f}% of the "
            "frozen rule's gross exposure here; the remainder sat in IMPLICIT CASH (the clipped "
            "short budget is not reallocated onto longs). Sharpe is invariant to a CONSTANT "
            "leverage multiplier but NOT to this state-dependent one, so compare the variant's "
            "Sharpe to the baseline's only alongside `variant_long_flat_lev_matched`, which "
            "holds the baseline's exposure path with the long-only signal.")
    return out


# =============================================================================
# arbitrary-window scorer — same primitives `trader.backtest.window_metrics` calls internally,
# parameterized to accept an externally-built (pos, pres, warm) triple. See SPEC.md gate (i).
# =============================================================================

def score_window(logc, pos, pres, ts_sub, H, warm, scfg, start_ts=None, end_ts=None):
    """Windowed metrics for a caller-supplied (pos, pres, warm) triple, using ONLY
    `trader.backtest`'s own unmodified `regime_series` / `net_series` / `_sharpe` /
    `_block_bootstrap_sharpe_ci`. Same dict shape as `trader.backtest.window_metrics`. Returns
    None if the window has fewer than 6 decisions (same thin-window refusal as window_metrics)."""
    logc = tconfig.to_numpy(logc)
    pos = tconfig.to_numpy(pos)
    pres = np.asarray(pres)
    ts_sub = np.asarray(ts_sub)
    T = ts_sub.size
    r_lo = int(np.searchsorted(ts_sub, start_ts)) if start_ts is not None else 0
    r_hi = int(np.searchsorted(ts_sub, end_ts)) if end_ts is not None else T
    res = bt.regime_series(logc, pos, pres, r_lo, r_hi, H, warm, xp=np)
    if res is None:
        return None
    gross, turn, bh = res
    period = H * tconfig.K
    d = {"n": int(gross.size), "turn": float(np.mean(turn)),
         "gross": bt._sharpe(gross, period), "bh": bt._sharpe(bh, period)}
    for c in scfg.costs_bp:
        d[f"net{c}"] = bt._sharpe(bt.net_series(gross, turn, c), period)
    for c in scfg.gate_costs:
        nh = bt.net_series(gross, turn, c)
        ci = bt._block_bootstrap_sharpe_ci(nh, period, period, scfg.n_boot, scfg.seed)
        eq = np.exp(np.cumsum(nh))
        peak = np.maximum.accumulate(eq)
        d[f"ci{c}_lo"], d[f"ci{c}_hi"] = float(ci[0]), float(ci[1])
        d[f"cum{c}"] = float(np.sum(nh))
        d[f"mult{c}"] = float(eq[-1])
        d[f"maxdd{c}"] = float(np.min(eq / peak - 1.0))
    d["cum_bh"] = float(np.sum(bh))
    d["mult_bh"] = float(np.exp(np.sum(bh)))
    years = (d["n"] * H * tconfig.K) / tconfig.MINUTES_PER_YEAR
    d["years"] = float(years)
    d["ann_turn"] = float(np.sum(turn) / years) if years > 0 else float("nan")
    return d


# =============================================================================
# gate (iii) — pre-registration
# =============================================================================

def spec_sha256():
    return hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()


def preregister(force=False):
    """Write (or verify) PREREG.json. Idempotent when the file already matches; refuses to
    silently clobber a mismatched registration unless force=True."""
    current = spec_sha256()
    if PREREG_PATH.exists():
        existing = json.loads(PREREG_PATH.read_text())
        if existing.get("spec_sha256") == current:
            print(f"[preregister] already registered, hash matches ({current[:12]}...)")
            return existing
        if not force:
            raise RuntimeError(
                f"PREREG.json exists with spec_sha256={existing.get('spec_sha256', '?')[:12]}..., "
                f"but SPEC.md's current hash is {current[:12]}... — SPEC.md changed since "
                f"registration. Re-run with force=True only if this drift is deliberate."
            )
    scfg = tconfig.STRATEGY
    reg = {
        "variant": "long_flat_898",
        "spec_file": "SPEC.md",
        "spec_sha256": current,
        "registered_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "rule": {
            "lookbacks": list(scfg.lookbacks), "sizing": scfg.sizing, "signal": scfg.signal,
            "vol_window": scfg.vol_window, "H_rebal": scfg.H_rebal, "gross_cap": scfg.gross_cap,
            "signal_transform": "s' = max(s, 0) -- short replaced by flat",
        },
        "scorer": {
            "costs_bp": list(scfg.costs_bp), "gate_costs": list(scfg.gate_costs),
            "n_boot": scfg.n_boot, "seed": scfg.seed,
            "identity": "trader.backtest.cell_metrics / trader.backtest.window_metrics (unmodified)",
        },
        "windows": {"start": RESEARCH_WINDOW_START, "kinds": ["calendar_year", "half_year", "master"]},
    }
    PREREG_PATH.write_text(json.dumps(reg, indent=2) + "\n")
    print(f"[preregister] wrote {PREREG_PATH} (hash {current[:12]}...)")
    return reg


def require_preregistered():
    """Gate (iii) enforcement: raises if PREREG.json is missing or stale, and `run()` calls it
    BEFORE it constructs the variant at all — so no variant number in any artifact this harness
    writes can exist without a matching registration. Narrowly: that is a property of `run()`,
    the only producer of results JSON, not of the module's importable surface
    (`build_long_flat_positions` et al. are plain functions the tests call directly). See
    SPEC.md gate (iii)."""
    if not PREREG_PATH.exists():
        raise RuntimeError(
            "gate (iii) violation: experiments/898_trend_decay/PREREG.json does not exist — "
            "run `python decay_harness.py preregister` before scoring the long-flat variant."
        )
    reg = json.loads(PREREG_PATH.read_text())
    current = spec_sha256()
    if reg.get("spec_sha256") != current:
        raise RuntimeError(
            f"gate (iii) violation: SPEC.md's current hash ({current[:12]}...) does not match "
            f"PREREG.json's registered hash ({reg.get('spec_sha256', '?')[:12]}...) — SPEC.md "
            f"changed after pre-registration. Re-run preregister deliberately if intended."
        )
    return reg


# =============================================================================
# gate (ii) — data-epoch assertion / stale-or-short-table discipline
# =============================================================================

def table_epoch(db_path):
    """MIN/MAX(timestamp) over the WHOLE ohlcv_daily table, independent of which assets survive
    the USD-quote/liquidity/PIT filters -- the literal table epoch this row must assert."""
    if not tdata.ohlcv_daily_exists(db_path):
        raise RuntimeError(f"gate (ii): {db_path} has no ohlcv_daily table -- nothing to score")
    con = tdata._connect(db_path, read_only=True)
    try:
        lo, hi = con.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv_daily").fetchone()
    finally:
        con.close()
    return int(lo), int(hi)


def _fmt_date(ts):
    return _dt.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")


def assess_epoch(db_path, now_utc=None, allow_stale=False):
    """Gate (ii): assert the table epoch, compute staleness vs now, and either refuse loudly,
    caveat loudly, or pass silently-but-recorded, per SPEC.md's thresholds. Returns the epoch
    block to embed in every result artifact. Raises RuntimeError on hard refusal."""
    tbl_lo, tbl_hi = table_epoch(db_path)
    now = now_utc or _dt.datetime.now(_dt.timezone.utc)
    now_ts = int(now.timestamp())
    staleness_days = (now_ts - tbl_hi) / 86400.0
    table_span_days = (tbl_hi - tbl_lo) / 86400.0     # the table's OWN coverage, not vs 2021
    block = {
        "db_path": str(db_path),
        "table_min_ts": tbl_lo, "table_min_date": _fmt_date(tbl_lo),
        "table_max_ts": tbl_hi, "table_max_date": _fmt_date(tbl_hi),
        "asserted_now_utc": now.isoformat(),
        "staleness_days": round(staleness_days, 2),
        "table_span_days": round(table_span_days, 1),
        "stale": False, "short": False, "refused": False,
    }
    is_short = table_span_days < MIN_SPAN_DAYS
    hard = staleness_days > HARD_REFUSE_STALE_DAYS or is_short
    warn = staleness_days > WARN_STALE_DAYS
    if hard and not allow_stale:
        block["refused"] = True
        block["reason"] = (
            f"staleness_days={staleness_days:.1f} > {HARD_REFUSE_STALE_DAYS} "
            f"and/or table_span_days={table_span_days:.1f} < {MIN_SPAN_DAYS} -- loud refusal "
            "(gate ii). Pass --allow-stale to proceed with a recorded caveat instead."
        )
        raise RuntimeError(f"[898 gate ii] REFUSED: {block['reason']} "
                            f"(table_max={block['table_max_date']}, now={now.date().isoformat()})")
    if hard or warn:
        block["stale"] = staleness_days > HARD_REFUSE_STALE_DAYS or (warn and not is_short)
        block["short"] = is_short
        notes = []
        if warn or block["stale"]:
            notes.append(
                f"DATA ROT (G6): table_max={block['table_max_date']}, staleness="
                f"{staleness_days:.1f}d (now={now.date().isoformat()}). Every number below is "
                "'recent as of the table', not as of today -- do not read this as a "
                "current-conditions claim.")
        if is_short:
            notes.append(
                f"SHORT TABLE: only {table_span_days:.1f}d of total coverage (< {MIN_SPAN_DAYS}) "
                "-- decay windows past the table's coverage are not scoreable and are marked "
                "partial/absent; any number reported here is under-powered.")
        block["epoch_caveat"] = " ".join(notes)
    return block


def assert_panel_epoch(block, ts_sub):
    """Gate (ii)'s SECOND promised assertion: the LOADED, FILTERED panel's own `ts_sub`
    min/max — which may be NARROWER than the raw table epoch if the scored universe (USD-quote
    / liquidity / PIT / source filters) does not cover the table's full span. Without this the
    instrument gives no signal at all when the scoreable universe stops short of the table,
    which is exactly the failure gate (ii) exists to catch. Mutates and returns `block`."""
    ts_sub = np.asarray(ts_sub)
    if ts_sub.size == 0:
        block["panel_epoch"] = {"panel_n_rows": 0, "narrower_than_table": True,
                                "note": "PANEL EMPTY: the filtered universe scores nothing."}
        return block
    p_lo, p_hi = int(ts_sub[0]), int(ts_sub[-1])
    head_gap = (p_lo - block["table_min_ts"]) / 86400.0     # >0 => panel starts LATER
    tail_gap = (block["table_max_ts"] - p_hi) / 86400.0     # >0 => panel ends EARLIER
    pe = {
        "panel_min_ts": p_lo, "panel_min_date": _fmt_date(p_lo),
        "panel_max_ts": p_hi, "panel_max_date": _fmt_date(p_hi),
        "panel_n_rows": int(ts_sub.size),
        "panel_span_days": round((p_hi - p_lo) / 86400.0, 1),
        "head_gap_days": round(head_gap, 1), "tail_gap_days": round(tail_gap, 1),
        "narrower_than_table": bool(head_gap > 0 or tail_gap > 0),
    }
    if pe["narrower_than_table"]:
        pe["note"] = (
            f"FILTERED PANEL NARROWER THAN TABLE: the scored universe covers "
            f"{pe['panel_min_date']}..{pe['panel_max_date']} while ohlcv_daily covers "
            f"{block['table_min_date']}..{block['table_max_date']} "
            f"(head {head_gap:.1f}d, tail {tail_gap:.1f}d). Every window bound below is clipped "
            f"to the PANEL, not the table — read 'through {block['table_max_date']}' claims "
            f"against {pe['panel_max_date']}.")
    else:
        pe["note"] = "filtered panel covers the full table epoch (no narrowing)."
    block["panel_epoch"] = pe
    return block


def with_cash_bh_flags(d, scfg):
    """Scope 3 annotation ('report both against B&H and against the cash floor'), applied
    identically to baseline and variant dicts AFTER scoring — never inside the scorer itself,
    so `score_window`'s dict shape stays byte-identical to `window_metrics`'s (see gate i,
    CHECK 1). Cash floor = 0 return / 0 Sharpe (holding cash); `net{c} > 0` is the comparison.
    Mutates and returns `d`; a no-op (returns None unchanged) if `d` is None (thin window)."""
    if d is None:
        return None
    for c in scfg.costs_bp:
        key = f"net{c}"
        if key not in d:
            continue
        d[f"cash_beat{c}"] = bool(np.isfinite(d[key]) and d[key] > 0)
        d[f"bh_beat{c}"] = bool(np.isfinite(d[key]) and np.isfinite(d.get("bh", float("nan")))
                                 and d[key] > d["bh"])
    return d


# =============================================================================
# window enumeration (Scope 1/2)
# =============================================================================

def calendar_year_windows(start_year, end_ts):
    end_year = _dt.datetime.utcfromtimestamp(end_ts).year
    out = []
    for y in range(start_year, end_year + 1):
        s = f"{y}-01-01"
        e = f"{y + 1}-01-01"
        out.append((f"{y}", "calendar_year", bt._to_epoch(s), bt._to_epoch(e)))
    return out


def half_year_windows(start_year, end_ts):
    end_year = _dt.datetime.utcfromtimestamp(end_ts).year
    out = []
    for y in range(start_year, end_year + 1):
        out.append((f"{y}H1", "half_year", bt._to_epoch(f"{y}-01-01"), bt._to_epoch(f"{y}-07-01")))
        out.append((f"{y}H2", "half_year", bt._to_epoch(f"{y}-07-01"), bt._to_epoch(f"{y + 1}-01-01")))
    return out


# =============================================================================
# run: build panel, score baseline + variant over every window + the fixed REGIMES table
# =============================================================================

def run(db_path, allow_stale=False, now_utc=None, source_filter="kraken_rest",
        include_dead=False):
    tconfig.set_backend("numpy")   # determinism reference (893/819/820 convention)
    epoch_block = assess_epoch(db_path, now_utc=now_utc, allow_stale=allow_stale)
    prereg = require_preregistered()

    ts_sub, close_d, ids, syms, dinfo = tdata.load_db_backtest_panel(
        db_path=db_path, source_filter=source_filter, include_dead=include_dead)
    # gate (ii), second assertion: the FILTERED panel's own coverage, not just the table's
    assert_panel_epoch(epoch_block, ts_sub)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where(close_d > 0, close_d, np.nan))
    scfg = tconfig.STRATEGY

    pos_base, pres_base, _, warm_base = tstrat.positions_from_config(logc, scfg)
    pos_var, pres_var, _, warm_var = build_long_flat_positions(logc, scfg)
    pos_lm, lm_scale, lm_matchable = build_leverage_matched_positions(pos_base, pos_var)
    # the exposure grid below is derived once from (pres_base, warm_base) and reused for all
    # three series; that is only valid because the variant changes the SIGNAL, never the
    # present-mask or the warmup. Fail loudly rather than silently mis-align if that drifts.
    if warm_var != warm_base or not np.array_equal(np.asarray(pres_var), np.asarray(pres_base)):
        raise RuntimeError(
            "long-flat variant changed the present-mask or warmup — the shared decision grid "
            "used for exposure reporting would no longer be the scorer's grid for every series.")

    start_year = int(RESEARCH_WINDOW_START[:4])
    windows = (calendar_year_windows(start_year, epoch_block["table_max_ts"])
               + half_year_windows(start_year, epoch_block["table_max_ts"])
               + [("MASTER_2021_to_table_end", "master", bt._to_epoch(RESEARCH_WINDOW_START),
                   None)])

    decay_rows = []
    for name, kind, s_ts, e_ts in windows:
        grid = decision_grid(logc.shape, pres_base, ts_sub, scfg.H_rebal, warm_base, s_ts, e_ts)
        base_m = annotate_exposure(
            with_cash_bh_flags(bt.window_metrics(logc, ts_sub, s_ts, e_ts, scfg), scfg),
            decision_exposure(pos_base, grid))
        var_m = annotate_exposure(
            with_cash_bh_flags(
                score_window(logc, pos_var, pres_var, ts_sub, scfg.H_rebal, warm_var, scfg,
                             s_ts, e_ts), scfg),
            decision_exposure(pos_var, grid))
        lm_m = annotate_exposure(
            with_cash_bh_flags(
                score_window(logc, pos_lm, pres_var, ts_sub, scfg.H_rebal, warm_var, scfg,
                             s_ts, e_ts), scfg),
            decision_exposure(pos_lm, grid))
        lm_m = annotate_exposure(lm_m, leverage_match_stats(lm_scale, lm_matchable, grid))
        e_row = (e_ts if e_ts is not None else epoch_block["table_max_ts"])
        partial = e_row > epoch_block["table_max_ts"]
        decay_rows.append({
            "window": name, "kind": kind,
            "start": _fmt_date(s_ts), "end": _fmt_date(e_row),
            "partial": bool(partial),
            "leverage": _leverage_summary(base_m, var_m, lm_m),
            "baseline": base_m,
            "variant_long_flat": var_m,
            "variant_long_flat_lev_matched": lm_m,
        })

    cell_base = bt.cell_metrics(logc, pos_base, pres_base, ts_sub, scfg.H_rebal, warm_base, scfg)
    cell_var = bt.cell_metrics(logc, pos_var, pres_var, ts_sub, scfg.H_rebal, warm_var, scfg)
    cell_lm = bt.cell_metrics(logc, pos_lm, pres_var, ts_sub, scfg.H_rebal, warm_var, scfg)
    regime_table = {}
    for reg_name, reg_a, reg_b in tconfig.REGIMES:
        # the SAME bounds `bt.regime_rows` derives for this regime (searchsorted on the same
        # UTC-midnight epochs), so the exposure grid is the cell's own scored grid
        grid = decision_grid(logc.shape, pres_base, ts_sub, scfg.H_rebal, warm_base,
                             bt._to_epoch(reg_a), bt._to_epoch(reg_b))
        b = annotate_exposure(with_cash_bh_flags(cell_base["reg"][reg_name], scfg),
                              decision_exposure(pos_base, grid))
        v = annotate_exposure(with_cash_bh_flags(cell_var["reg"][reg_name], scfg),
                              decision_exposure(pos_var, grid))
        m = annotate_exposure(with_cash_bh_flags(cell_lm["reg"][reg_name], scfg),
                              decision_exposure(pos_lm, grid))
        m = annotate_exposure(m, leverage_match_stats(lm_scale, lm_matchable, grid))
        regime_table[reg_name] = {
            "leverage": _leverage_summary(b, v, m),
            "baseline": b, "variant_long_flat": v, "variant_long_flat_lev_matched": m,
        }

    out = {
        "task": "TASK_898", "emits_verdict": False,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "gate_i_scorer": {
            "identity": "trader.backtest.cell_metrics / trader.backtest.window_metrics (unmodified)",
            "seed": scfg.seed, "n_boot": scfg.n_boot,
            "costs_bp": list(scfg.costs_bp), "gate_costs": list(scfg.gate_costs),
        },
        "gate_ii_data_epoch": epoch_block,
        "gate_iii_preregistration": {"spec_sha256": prereg["spec_sha256"],
                                      "registered_at_utc": prereg["registered_at_utc"],
                                      "amendment_note": SPEC_AMENDMENT_NOTE},
        "panel": {"n_assets": int(logc.shape[1]), "n_days": int(logc.shape[0]),
                  "source_filter": source_filter, "include_dead": include_dead,
                  "skip_info": dinfo},
        "series_legend": {
            "baseline": "the frozen 819/820 rule, unmodified (long AND short).",
            "variant_long_flat": (
                "PRIMARY pre-registered variant: shorts replaced by flat. This is the honest "
                "reading of 'shorts replaced by flat' and the only book deployable without "
                "perps — so it is deliberately left DE-LEVERAGED: the clipped short budget "
                "sits in cash, it is NOT reallocated onto longs. Read its Sharpe together with "
                "`gross_exposure_mean` and the lev-matched series."),
            "variant_long_flat_lev_matched": (
                "DIAGNOSTIC comparator, not a deployable book: the long-flat positions rescaled "
                "row-wise to the baseline's own realized gross exposure, so signal quality can "
                "be separated from sitting in cash. `lm_unmatched_frac` reports the decisions "
                "where every present asset was short and the long-only book could not be "
                "matched at all (residual, disclosed confound)."),
            "gross_exposure_*": (
                "realized Sum_i|pos[t,i]| over exactly the decisions the scorer scored "
                "(same H-strided grid, same warmup, same window bounds)."),
        },
        "decay_windows": decay_rows,
        "regime_table": regime_table,
    }
    return out


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="TASK_898 trend-decay + long-flat variant instrument")
    sub = p.add_subparsers(dest="cmd", required=True)
    pp0 = sub.add_parser("preregister")
    pp0.add_argument("--force", action="store_true",
                      help="deliberately re-register after SPEC.md changed (drift, not silent)")
    pr = sub.add_parser("run")
    pr.add_argument("--db", type=str, default=str(_REPO_ROOT / "fixtures" / "trading_fixture.db"))
    pr.add_argument("--out", type=str, default=None)
    pr.add_argument("--allow-stale", action="store_true")
    pr.add_argument("--now-utc", type=str, default=None, metavar="YYYY-MM-DD",
                     help="override wall-clock 'now' for the staleness check (testing only)")
    pr.add_argument("--include-dead", action="store_true")
    pr.add_argument("--source-filter", type=str, default="kraken_rest")
    args = p.parse_args(argv)

    if args.cmd == "preregister":
        preregister(force=args.force)
        return 0

    now_utc = None
    if args.now_utc:
        now_utc = _dt.datetime.strptime(args.now_utc, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
    try:
        out = run(args.db, allow_stale=args.allow_stale, now_utc=now_utc,
                  source_filter=args.source_filter, include_dead=args.include_dead)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    out_path = Path(args.out) if args.out else (_HERE / "results" /
                                                 f"decay_{Path(args.db).stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"[898] wrote {out_path}")
    eb = out["gate_ii_data_epoch"]
    print(f"[898] TABLE EPOCH: {eb['table_min_date']} .. {eb['table_max_date']}  "
          f"staleness={eb['staleness_days']}d")
    pe = eb.get("panel_epoch", {})
    print(f"[898] PANEL EPOCH (filtered ts_sub): {pe.get('panel_min_date')} .. "
          f"{pe.get('panel_max_date')}  rows={pe.get('panel_n_rows')}  "
          f"narrower_than_table={pe.get('narrower_than_table')}")
    if pe.get("narrower_than_table"):
        print(f"[898] PANEL COVERAGE: {pe.get('note')}")
    if "epoch_caveat" in eb:
        print(f"[898] CAVEAT: {eb['epoch_caveat']}")
    full = out["regime_table"].get("FULL", {}).get("leverage", {})
    if full.get("variant_over_baseline") is not None:
        print(f"[898] LEVERAGE (FULL): baseline gross={full['baseline_gross_mean']:.3f}  "
              f"long-flat={full['variant_gross_mean']:.3f} "
              f"({100.0 * full['variant_over_baseline']:.1f}% of baseline)  "
              f"lev-matched={full['lev_matched_gross_mean']:.3f}  "
              f"unmatched_frac={full.get('lm_unmatched_frac')}")
        print("[898] the long-flat book's missing exposure is IMPLICIT CASH -- compare its "
              "Sharpe against the lev-matched series, not the baseline alone.")
    print(f"[898] PREREG: {out['gate_iii_preregistration']['spec_sha256'][:12]}... "
          f"(AMENDED + re-registered; see gate_iii_preregistration.amendment_note)")
    print("[898] NO VERDICT EMITTED -- this is the instrument, not the ruling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

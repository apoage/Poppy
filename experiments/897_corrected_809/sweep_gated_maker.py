#!/usr/bin/env python3
"""
TASK_897 scope 2/3/4 — the `gated_maker` sweep harness + artifact writer, on the HONEST
per-decision estimator (`honest_estimator.py`), over BOTH OOS windows and the full
fee x adverse-selection grid.

WHAT THIS SCRIPT IS AND IS NOT
-------------------------------
- IS the instrument the planner-scheduled gpu-server run launches (D54): it builds the
  gated_maker decision series once per (H, k, variant, tau, N, window) via the audited
  806/808/809 lineage (`probe.py`/`backtest.py`/`decide.py`, imported verbatim, never
  reimplemented), costs it under every (fee, adv) pair, and estimates Sharpe with the
  honest per-decision estimator — never per-bar annualization anywhere in this file.
- Is NOT the real-data run. TASK_897's scope note: "expensive real-data validation is
  SCHEDULED, never polled in-loop." Verification here is at fixture/synthetic scale only
  (`--db-path fixtures/trading_fixture.db`, tiny custom date windows, `--smoke`).
- Does NOT compute or print a CELL CLEARS / ALL FAIL / MIXED verdict. That three-way
  read is the planner's, from this artifact, after the srv run. This script only records.

GATES ENFORCED BY THIS FILE
----------------------------
(ii)  Every cell of the grid appears in the artifact — no silent drops. Each
      (H, k, variant, tau, N, window) combo's decision-engine call is wrapped; on failure
      ALL 9 (fee, adv) sub-cells for that combo are still emitted with status="dropped" and
      a reason. Same treatment one level up for a failed per-H fit (all of that H's cells
      dropped-with-reason, not omitted). A cell that computes but yields degenerate/NaN
      metrics (e.g. an all-flat book) is status="ok" — NaN is a legitimate measurement,
      omission is not.
(iv)  `--attainable-maker-fee-bp` is REQUIRED and un-defaulted — the harness raises before
      touching any data if it is missing, empty, or unparseable. This is the operator's
      Kraken-reachable maker-fee tier set (Fork-C venue ruling: zero-maker-fee is out of
      reach); the harness never invents, guesses, or defaults it, and records it in the
      artifact's metadata block BEFORE the cell table.

`--db-path` IS LIKEWISE REQUIRED AND UN-DEFAULTED, on gate (iv)'s defence-in-depth
precedent. It previously defaulted to the real 24GB `trading_data.db` "so the srv launch
needs no code change" — which made the source of record reachable BY OMISSION from any
interactive invocation. Two guards now stand between an accidental run and that file:
  * no default at all, so forgetting the flag refuses instead of picking the real DB; and
  * `assert_db_path_allowed`, which refuses any path resolving to the protected real DB
    unless the caller ALSO passes the explicit, self-describing
    `--i-am-the-scheduled-real-data-run` flag (the scheduled srv launch passes both; nothing
    else has any reason to).
The real DB is read-only source of record in every case — this harness never writes to it.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
sys.path.insert(0, str(_REPO / "runs" / "806_predictability_probe"))
sys.path.insert(0, str(_REPO / "runs" / "808_linear_backtest"))
sys.path.insert(0, str(_REPO / "runs" / "809_decision_rule_rescue"))
import probe    # noqa: E402  (806 lineage — verbatim, never reimplemented)
import backtest  # noqa: E402  (808 lineage)
import decide    # noqa: E402  (809 lineage — chunked fit/predict, _run_decision)

sys.path.insert(0, str(_THIS.parent))
import honest_estimator as hest  # noqa: E402

# =============================================================================
# PRE-REGISTERED REAL GRID (TASK_897 scope 2/3 — do NOT add cells post-hoc)
# =============================================================================
GRID_H: Tuple[int, ...] = (1, 5, 20)
GRID_K: Tuple[int, ...] = (16, 32)
GRID_VARIANT: Tuple[str, ...] = ("long_short", "long_only")
GRID_TAU: Tuple[float, ...] = (0.75, 0.9)
GRID_N: Tuple[int, ...] = (0, 60)
GRID_FEE_BP: Tuple[float, ...] = (0.0, 8.0, 16.0)
GRID_ADV_BP: Tuple[float, ...] = (0.0, 5.0, 10.0)

# The two OOS windows are sub-ranges of ONE continuous test split under the SAME frozen
# fit (same train/val as the original 809 Config) — window (b) is simply more unseen data
# the minute table now carries, not a re-fit. See report/docstring below for why.
WINDOW_A = ("a_2024-11_2025-04", "2024-11-01", "2025-04-01")
WINDOW_B = ("b_2025-04_2026-03", "2025-04-01", "2026-03-01")
REAL_WINDOWS: Tuple[Tuple[str, str, str], ...] = (WINDOW_A, WINDOW_B)


def _derive_window_label(prefix: str, start: str, end: str) -> str:
    """Derive a window's label from its ACTUAL start/end dates, so the label can never
    disagree with the dates recorded beside it. Format matches the pre-registered default
    labels above: `{prefix}_{YYYY-MM}_{YYYY-MM}`.

    Defect this guards against: the CLI used to tag every window with the hardcoded
    WINDOW_A[0]/WINDOW_B[0] default label regardless of what --window-a/--window-b actually
    supplied, so an overridden window's artifact row could assert a label that contradicted
    the dates recorded beside it in the same metadata line."""
    return f"{prefix}_{start[:7]}_{end[:7]}"


def build_windows(
    a_start: str, a_end: str, b_start: str, b_end: str
) -> Tuple[Tuple[str, str, str], Tuple[str, str, str]]:
    """Build the (name, start, end) window tuples the sweep runs over, deriving each name
    from its own actual dates rather than a hardcoded constant. With the default dates this
    reproduces WINDOW_A/WINDOW_B byte-for-byte."""
    return (
        (_derive_window_label("a", a_start, a_end), a_start, a_end),
        (_derive_window_label("b", b_start, b_end), b_start, b_end),
    )

# Smaller SMOKE grid (fixture/synthetic-scale verification only) — a DIFFERENT, explicitly
# smaller pre-registered grid, not a silent subset of the real one.
SMOKE_GRID_H: Tuple[int, ...] = (5,)
SMOKE_GRID_K: Tuple[int, ...] = (16,)
SMOKE_GRID_VARIANT: Tuple[str, ...] = ("long_short", "long_only")
SMOKE_GRID_TAU: Tuple[float, ...] = (0.75, 0.9)
SMOKE_GRID_N: Tuple[int, ...] = (0, 60)


@dataclass
class CellRecord:
    window: str
    H: int
    k: int
    variant: str
    tau: float
    N: int
    fee_bp: float
    adv_bp: float
    status: str            # "ok" | "dropped" | "failed"
    reason: str            # non-empty iff status != "ok"
    n_bars: int
    n_decisions: int
    decisions_per_year: float
    mean_hold_bars: float
    turnover_mean: float
    fill_rate: float
    sharpe_decision: float
    ci_lo_UNCALIBRATED: float
    ci_hi_UNCALIBRATED: float
    pnl: float
    maxdd: float
    sharpe_per_bar_diagnostic: float


def _to_epoch(date_str: str) -> int:
    return probe._to_epoch(date_str)


def _window_slice(dec_rows: np.ndarray, grid_ts: np.ndarray, start: str, end: str) -> np.ndarray:
    """Boolean mask selecting decision rows whose grid timestamp lies in [start, end)."""
    ts = grid_ts[dec_rows]
    return (ts >= _to_epoch(start)) & (ts < _to_epoch(end))


def _empty_cells(window: str, H: int, ks: Sequence[int], variants: Sequence[str],
                  taus: Sequence[float], ns: Sequence[int], fees_bp: Sequence[float],
                  advs_bp: Sequence[float], status: str, reason: str) -> List[CellRecord]:
    """Materialize every (k, variant, tau, N, fee, adv) cell for one (window, H) as a
    dropped/failed record — the gate-(ii) no-silent-drop path for an upstream failure."""
    out = []
    for k, var, tau, N in itertools.product(ks, variants, taus, ns):
        for fee_bp, adv_bp in itertools.product(fees_bp, advs_bp):
            out.append(CellRecord(
                window=window, H=H, k=k, variant=var, tau=tau, N=N, fee_bp=fee_bp, adv_bp=adv_bp,
                status=status, reason=reason, n_bars=0, n_decisions=0, decisions_per_year=float("nan"),
                mean_hold_bars=float("nan"), turnover_mean=float("nan"), fill_rate=float("nan"),
                sharpe_decision=float("nan"), ci_lo_UNCALIBRATED=float("nan"),
                ci_hi_UNCALIBRATED=float("nan"),
                pnl=float("nan"), maxdd=float("nan"), sharpe_per_bar_diagnostic=float("nan"),
            ))
    return out


def run_sweep(
    attainable_maker_fee_bp: Sequence[float],
    db_path: str,
    train_start: str = "2023-05-16",
    val_start: str = "2024-06-01",
    windows: Sequence[Tuple[str, str, str]] = REAL_WINDOWS,
    grid_h: Sequence[int] = GRID_H,
    grid_k: Sequence[int] = GRID_K,
    grid_variant: Sequence[str] = GRID_VARIANT,
    grid_tau: Sequence[float] = GRID_TAU,
    grid_n: Sequence[int] = GRID_N,
    grid_fee_bp: Sequence[float] = GRID_FEE_BP,
    grid_adv_bp: Sequence[float] = GRID_ADV_BP,
    seed: int = 897,
    n_boot: int = 2000,
    verify_inherited_ic: bool = True,
) -> Tuple[List[CellRecord], Dict]:
    # --- GATE (iv): the attainable maker-fee tier set is a REQUIRED operator input. -------
    # Never invented, guessed, or defaulted — refuse before touching any data.
    if attainable_maker_fee_bp is None or len(list(attainable_maker_fee_bp)) == 0:
        raise ValueError(
            "GATE (iv) REFUSAL: --attainable-maker-fee-bp is required and was not provided. "
            "This is the OPERATOR's Kraken-reachable maker-fee tier set (Fork-C venue ruling: "
            "zero-maker-fee is out of reach) — this harness will not invent, guess, or default "
            "it. Pass e.g. --attainable-maker-fee-bp 8,16 and re-run."
        )
    attainable_maker_fee_bp = [float(x) for x in attainable_maker_fee_bp]

    t0 = time.time()
    cfg = probe.Config(
        db_path=db_path, train_start=train_start, val_start=val_start,
        test_start=windows[0][1], test_end=windows[-1][2],
    )
    # decide.py's chunked fit/predict helpers (fit_xs_signal_chunked, _chunked_predict,
    # inherited_signal_check_chunked) read splits/W/seed/ridge_alphas off the MODULE-LEVEL
    # `decide.CFG` global (set at decide.py import time to probe.Config()'s REAL-data
    # defaults) rather than taking a Config parameter — so this harness's own train/val/
    # window dates would otherwise be silently ignored. Point decide.CFG at OUR cfg for the
    # duration of this sweep (restored in `finally`); this is the only way to reuse those
    # functions verbatim (no reimplementation) at a non-canonical (fixture/synthetic) scale.
    _orig_decide_cfg = decide.CFG
    decide.CFG = cfg
    try:
        return _run_sweep_inner(cfg, attainable_maker_fee_bp, windows, grid_h, grid_k,
                                 grid_variant, grid_tau, grid_n, grid_fee_bp, grid_adv_bp,
                                 seed, n_boot, verify_inherited_ic, db_path, train_start,
                                 val_start, t0)
    finally:
        decide.CFG = _orig_decide_cfg


def _run_sweep_inner(
    cfg, attainable_maker_fee_bp, windows, grid_h, grid_k, grid_variant, grid_tau, grid_n,
    grid_fee_bp, grid_adv_bp, seed, n_boot, verify_inherited_ic, db_path, train_start,
    val_start, t0,
) -> Tuple[List[CellRecord], Dict]:
    bar_sec = probe.report_bar_frequency(cfg)
    symbols, asset_ids, _ = probe.load_universe_trainlocked(cfg)
    ohlcv, grid_ts = probe.build_aligned_ohlcv(cfg, asset_ids, bar_sec)
    high = np.ascontiguousarray(ohlcv[:, :, 1])
    low = np.ascontiguousarray(ohlcv[:, :, 2])
    close = np.ascontiguousarray(ohlcv[:, :, 3])
    del ohlcv
    gc.collect()

    meta = {
        "attainable_maker_fee_bp_tiers": attainable_maker_fee_bp,   # gate (iv): recorded FIRST
        "db_path": db_path,
        "bar_sec": bar_sec,
        "n_assets": len(asset_ids),
        "symbols": symbols,
        "train_start": train_start, "val_start": val_start,
        "windows": [{"name": n, "start": s, "end": e} for (n, s, e) in windows],
        "grid": {"H": list(grid_h), "k": list(grid_k), "variant": list(grid_variant),
                 "tau": list(grid_tau), "N": list(grid_n),
                 "fee_bp": list(grid_fee_bp), "adv_bp": list(grid_adv_bp)},
        "grid_epoch": {"start": int(grid_ts[0]), "end": int(grid_ts[-1])},
        "grid_start_iso": probe._dt.datetime.utcfromtimestamp(int(grid_ts[0])).isoformat() + "Z",
        "grid_end_iso": probe._dt.datetime.utcfromtimestamp(int(grid_ts[-1])).isoformat() + "Z",
        "seed": seed, "n_boot": n_boot,
        "n_cells_expected": len(list(grid_h)) * len(list(grid_k)) * len(list(grid_variant))
        * len(list(grid_tau)) * len(list(grid_n)) * len(windows) * len(list(grid_fee_bp)) * len(list(grid_adv_bp)),
        "estimator": "per-decision, non-overlapping-hold, annualized by realized "
                     "decisions-per-year (experiments/897_corrected_809/honest_estimator.py) "
                     "— NEVER per-bar annualization.",
        "maker_fill_model": "OHLC-touch (limit posted at close(d); fills iff bar d+1's "
                             "low<=limit for buys / high>=limit for sells) + a FLAT "
                             "adverse-selection bps penalty on filled notional. This coupling "
                             "(flat fill-rate x flat adv_sel) is a crude first cut, never "
                             "validated against a real order (out of scope here, per TASK_897).",
    }

    cells: List[CellRecord] = []
    for H in grid_h:
        h_t0 = time.time()
        try:
            _, _, test = probe.make_splits(cfg, grid_ts, H)
            w, mu, sd, alpha = decide.fit_xs_signal_chunked(close, grid_ts, H)
            ic = None
            if verify_inherited_ic:
                ic = decide.inherited_signal_check_chunked(close, grid_ts, H, w, mu, sd)
            dec_rows = np.arange(test.lo, test.hi + 1, dtype=np.int64)
            dec_rows = dec_rows[dec_rows + 2 < close.shape[0]]
            pred = decide._chunked_predict(close, dec_rows, w, mu, sd, cfg.W)
            present = np.isfinite(pred) & np.isfinite(close[dec_rows]) & (close[dec_rows] > 0)
        except Exception as exc:  # noqa: BLE001 — gate (ii): a failed fit drops, never omits
            reason = f"H={H} fit/predict failed: {type(exc).__name__}: {exc}"
            for (wname, wstart, wend) in windows:
                cells.extend(_empty_cells(wname, H, grid_k, grid_variant, grid_tau, grid_n,
                                          grid_fee_bp, grid_adv_bp, "dropped", reason))
            print(f"[897-sweep] H={H} DROPPED ALL CELLS: {reason}", flush=True)
            continue

        meta.setdefault("signal_ic", {})[H] = ic
        print(f"[897-sweep] H={H} alpha={alpha:g} ic={ic} dec_rows={len(dec_rows)} "
              f"fit {time.time()-h_t0:.1f}s", flush=True)

        for (wname, wstart, wend) in windows:
            wmask = _window_slice(dec_rows, grid_ts, wstart, wend)
            for k, variant, tau, N in itertools.product(grid_k, grid_variant, grid_tau, grid_n):
                try:
                    dr_w = dec_rows[wmask]
                    pred_w = pred[wmask]
                    pres_w = present[wmask]
                    gross, turn, fill = decide._run_decision(
                        close, high, low, pred_w, pres_w, dr_w, k, variant, tau, N, "maker"
                    )
                except Exception as exc:  # noqa: BLE001 — gate (ii)
                    reason = (f"window={wname} H={H} k={k} {variant} tau={tau} N={N} decision "
                              f"engine failed: {type(exc).__name__}: {exc}")
                    cells.extend(_empty_cells(wname, H, [k], [variant], [tau], [N],
                                              grid_fee_bp, grid_adv_bp, "dropped", reason))
                    continue

                for fee_bp, adv_bp in itertools.product(grid_fee_bp, grid_adv_bp):
                    try:
                        cost = (fee_bp + adv_bp) * 1e-4
                        m = hest.compute_honest_metrics(gross, turn, cost, bar_sec, seed=seed,
                                                         n_boot=n_boot)
                        cells.append(CellRecord(
                            window=wname, H=H, k=k, variant=variant, tau=tau, N=N,
                            fee_bp=fee_bp, adv_bp=adv_bp, status="ok", reason="",
                            n_bars=m.n_bars, n_decisions=m.n_decisions,
                            decisions_per_year=m.decisions_per_year, mean_hold_bars=m.mean_hold_bars,
                            turnover_mean=m.turnover_mean, fill_rate=fill,
                            sharpe_decision=m.sharpe_decision,
                            ci_lo_UNCALIBRATED=m.ci_lo_UNCALIBRATED,
                            ci_hi_UNCALIBRATED=m.ci_hi_UNCALIBRATED,
                            pnl=m.pnl, maxdd=m.maxdd,
                            sharpe_per_bar_diagnostic=m.sharpe_per_bar_DIAGNOSTIC_DO_NOT_GATE_ON,
                        ))
                    except Exception as exc:  # noqa: BLE001 — gate (ii)
                        reason = f"honest-metrics failed: {type(exc).__name__}: {exc}"
                        cells.extend(_empty_cells(wname, H, [k], [variant], [tau], [N],
                                                  [fee_bp], [adv_bp], "failed", reason))
        del w, mu, sd, pred, present
        gc.collect()

    meta["n_cells_recorded"] = len(cells)
    meta["n_cells_ok"] = sum(1 for c in cells if c.status == "ok")
    meta["n_cells_dropped_or_failed"] = sum(1 for c in cells if c.status != "ok")
    meta["runtime_s"] = time.time() - t0
    return cells, meta


def write_artifact(cells: List[CellRecord], meta: Dict, path: Path) -> None:
    """Write the TASK_897 artifact. Metadata (incl. the gate-(iv) attainable-tier set and the
    data epoch, gate iii) is recorded FIRST, before the cell table. NO verdict is computed or
    printed here — CELL CLEARS / ALL FAIL / MIXED is the planner's read of this artifact."""
    L: List[str] = ["# TASK_897 — gated_maker sweep artifact (HONEST per-decision estimator)", ""]
    L += ["> **No verdict in this file.** Gate (iv): the attainable-tier set below is the "
          "operator's input, recorded before any cell data. Reading CELL CLEARS / ALL FAIL / "
          "MIXED off this table is the planner's job (TASK_897 pre-registered bands), not this "
          "harness's — see `tasks/TASK_897_CORRECTED_809_REMEASURE.md`.", ""]
    L += ["> ## :warning: THE `CI` COLUMN IS **UNCALIBRATED** — THE BAND'S \"CI lo > 0\" "
          "CRITERION CANNOT BE READ OFF THIS TABLE",
          ">",
          "> The pre-registered TASK_897 band has three criteria (post-cost Sharpe >= 1.0, "
          "PnL > 0, **CI lo > 0**). The third one is NOT satisfiable from this artifact as it "
          "stands, and no reader should treat it as if it were. The intervals below are nominal "
          "95% circular-block-bootstrap percentiles whose ACTUAL coverage has been measured "
          "BELOW nominal in every case tested. Two of the three criteria can be read here; the "
          "third is pending a calibration this row did not run.",
          ">",
          "> " + hest.CI_UNCALIBRATED_DISCLOSURE.replace("\n", " "),
          "",
          ]
    L += ["## Metadata (read this BEFORE any cell)",
          f"- **Attainable maker-fee tiers (operator input, gate iv):** "
          f"{meta['attainable_maker_fee_bp_tiers']} bp — the Kraken-reachable subset of the fee "
          "grid below; zero-maker-fee is out of reach per the Fork-C venue ruling.",
          f"- **Data source:** `{meta['db_path']}`; bar_sec={meta['bar_sec']}; "
          f"{meta['n_assets']} assets: {meta['symbols']}.",
          f"- **Data epoch (gate iii):** grid {meta['grid_start_iso']} .. {meta['grid_end_iso']} "
          f"(epoch [{meta['grid_epoch']['start']}, {meta['grid_epoch']['end']}]).",
          "- **877 alias fold: NOT APPLIED by this lineage (gate (iii) is NOT met on this "
          "point).** Verified in code, not assumed: `probe.py`, `backtest.py` and `decide.py` "
          "contain no reference to `asset_aliases`, `normalize_symbol_key`, "
          "`resolve_requested_symbols` or `group_asset_rows`, and "
          "`probe.load_universe_trainlocked` resolves the universe with a plain "
          "`SELECT id, symbol FROM assets` matched by EXACT string. That lineage predates "
          "TASK_877 entirely, so there is nothing for the fold to be inherited from. An earlier "
          "revision of this artifact asserted the opposite — that the fold \"applies transitively "
          "via `probe.load_universe_trainlocked`/`build_aligned_ohlcv` (inherited, never "
          "reimplemented here)\" — and that claim was FALSE; it is retracted here. "
          "CONSEQUENCE: where the pinned minute table carries a feed under twin spelling-variant "
          "`assets` rows (the TASK_868/877 defect: `XBT/CHF` vs `BTC/CHF`, `/USDTEUR` vs "
          "`USDTEUR`), this sweep reads only the exact-spelling row and silently misses the "
          "twin's history — so affected slots can present as sparse or absent over spans where "
          "the twin id holds the real bars. Cells are not known to be free of that effect, and "
          "no per-slot `asset_ids`/row-count disclosure is produced by this lineage. "
          "Implementing the fold is DELIBERATELY out of scope here: TASK_897's own Out-of-scope "
          "forbids modifying the 809 lineage beyond the estimator fix, which conflicts with its "
          "gate (iii); resolving that conflict is a planner decision, and this artifact's job is "
          "to state the gap rather than paper over it.",
          f"- **Train/val:** train_start={meta['train_start']} val_start={meta['val_start']} "
          "(unchanged across both windows — window (b) is more unseen data under the SAME "
          "frozen fit, not a re-fit).",
          f"- **Windows:** {meta['windows']}",
          f"- **Grid:** {meta['grid']}",
          f"- **Signal IC by H (inherited-check; None if `--skip-inherited-ic-check`):** "
          f"{meta.get('signal_ic', {})}",
          f"- **Estimator:** {meta['estimator']}",
          f"- **Maker-fill model (stated inline, per cell, per TASK_897 scope 3):** "
          f"{meta['maker_fill_model']}",
          f"- **Cells expected:** {meta['n_cells_expected']}  **recorded:** "
          f"{meta['n_cells_recorded']}  (ok={meta['n_cells_ok']}, "
          f"dropped/failed={meta['n_cells_dropped_or_failed']})",
          f"- **Runtime:** {meta['runtime_s']:.1f}s", ""]
    if meta["n_cells_recorded"] != meta["n_cells_expected"]:
        L += [f"**GATE (ii) VIOLATION: recorded {meta['n_cells_recorded']} != expected "
              f"{meta['n_cells_expected']}.** This artifact is INCOMPLETE and must not be read.", ""]
    L += ["## Cells (every grid cell — status ok/dropped/failed, no omissions)",
          "| window | H | k | var | tau | N | fee bp | adv bp | status | n_bars | n_dec | "
          "dec/yr | mean_hold | turnover | fill | Sh_decision | CI(UNCALIBRATED) | pnl | maxDD | "
          "Sh_per_bar(diag) | reason |",
          "|---|--:|--:|---|--:|--:|--:|--:|:--:|--:|--:|--:|--:|--:|--:|---|---|--:|--:|--:|---|"]
    for c in cells:
        L.append(
            f"| {c.window} | {c.H} | {c.k} | {c.variant} | {c.tau:g} | {c.N} | {c.fee_bp:g} | "
            f"{c.adv_bp:g} | {c.status} | {c.n_bars} | {c.n_decisions} | "
            f"{c.decisions_per_year:.1f} | {c.mean_hold_bars:.1f} | {c.turnover_mean:.4f} | "
            f"{c.fill_rate:.3f} | {c.sharpe_decision:+.3f} | "
            f"[{c.ci_lo_UNCALIBRATED:+.2f},{c.ci_hi_UNCALIBRATED:+.2f}] | "
            f"{c.pnl:+.4f} | {c.maxdd:+.4f} | {c.sharpe_per_bar_diagnostic:+.2f} | {c.reason} |"
        )
    L += ["", "## Scope honesty",
          "This artifact never computes a pass/fail verdict; the `Sh_per_bar(diag)` column is "
          "the RETRACTED 809 methodology kept only as a visible contrast — never gate on it.",
          "",
          "Two known gaps are stated rather than papered over, and BOTH bear on whether a "
          "CELL CLEARS read is even available from this table:",
          "",
          "1. **The CI is uncalibrated** (banner at the top) — the band's third criterion cannot "
          "be evaluated here.",
          "2. **The 877 alias fold is not applied** by the 806/808/809 lineage (Metadata above) — "
          "gate (iii) is not met on that point, and the fix is a planner decision, not this "
          "row's to make.",
          ""]
    path.write_text("\n".join(L) + "\n")


# The protected read-only source of record (.dark-factory/config.json `protected`). Reaching it
# requires naming it AND acknowledging the scheduled-run flag; omission can never get there.
PROTECTED_REAL_DB = _REPO / "trading_data.db"


def parse_fee_tiers(raw: Optional[str]) -> List[float]:
    """Gate (iv): parse the operator's attainable maker-fee tier list, refusing CLEANLY.

    Every rejection path exits through `SystemExit` with an actionable message. Previously a
    value like `","` slipped past the truthiness check and then blew up as an uncaught
    `ValueError` traceback deeper in `run_sweep` — the same practical refusal, but presented as
    a crash. Same for a non-numeric entry. The refusal is the product here; it should read like
    one."""
    if raw is None or not raw.strip():
        raise SystemExit(
            "GATE (iv) REFUSAL: --attainable-maker-fee-bp is required and has no default. "
            "It is the OPERATOR's Kraken-reachable maker-fee tier set (Fork-C venue ruling: "
            "zero-maker-fee is out of reach) — this harness will not invent or guess it. "
            "Example: --attainable-maker-fee-bp 8,16"
        )
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if not parts:
        raise SystemExit(
            f"GATE (iv) REFUSAL: --attainable-maker-fee-bp {raw!r} contains no tier values "
            "(separators only). Pass a comma-separated bp list, e.g. "
            "--attainable-maker-fee-bp 8,16"
        )
    tiers: List[float] = []
    for p in parts:
        try:
            tiers.append(float(p))
        except ValueError:
            raise SystemExit(
                f"GATE (iv) REFUSAL: --attainable-maker-fee-bp entry {p!r} is not a number "
                f"(from {raw!r}). Pass a comma-separated bp list, e.g. "
                "--attainable-maker-fee-bp 8,16"
            ) from None
    return tiers


def assert_db_path_allowed(db_path: Optional[str], scheduled_real_data_run: bool) -> str:
    """Refuse to touch the protected 24GB source of record unless it was BOTH named explicitly
    AND acknowledged with `--i-am-the-scheduled-real-data-run`. Gate (iv)'s defence-in-depth
    precedent, applied to the data path: an accidental interactive invocation must not be able
    to reach `trading_data.db` BY OMISSION, which the old `--db-path` default allowed."""
    if db_path is None or not str(db_path).strip():
        raise SystemExit(
            "REFUSAL: --db-path is required and has no default (deliberately — it used to "
            "default to the real 24GB trading_data.db, so forgetting the flag reached the "
            "source of record). Pass fixtures/trading_fixture.db (or a scratch copy) for "
            "verification; the scheduled gpu-server run passes the real DB explicitly together "
            "with --i-am-the-scheduled-real-data-run."
        )
    resolved = Path(db_path).expanduser().resolve()
    if resolved == PROTECTED_REAL_DB.resolve() and not scheduled_real_data_run:
        raise SystemExit(
            f"REFUSAL: --db-path {resolved} is the PROTECTED read-only source of record. "
            "TASK_897's scope note: expensive real-data validation is SCHEDULED (gpu-server, "
            "per D54), never polled from a factory or interactive session. If this really IS "
            "the scheduled run, say so explicitly: add --i-am-the-scheduled-real-data-run."
        )
    return str(db_path)


def _parse_window(flag: str, raw: str) -> Tuple[str, str]:
    """`START,END` -> (start, end), refusing cleanly instead of raising a bare unpack ValueError."""
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) != 2 or not all(parts):
        raise SystemExit(
            f"REFUSAL: {flag} {raw!r} must be exactly 'START,END' (ISO dates), "
            f"e.g. {flag} 2024-11-01,2025-04-01"
        )
    return parts[0], parts[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attainable-maker-fee-bp", type=str, default=None,
                     help="REQUIRED comma-separated bp list, e.g. '8,16'. No default (gate iv).")
    ap.add_argument("--db-path", type=str, default=None,
                     help="REQUIRED, no default. Use fixtures/trading_fixture.db (or a scratch "
                          "copy) for verification. The real trading_data.db is reachable ONLY by "
                          "naming it explicitly AND passing --i-am-the-scheduled-real-data-run.")
    ap.add_argument("--i-am-the-scheduled-real-data-run", action="store_true",
                     dest="scheduled_real_data_run",
                     help="Defence-in-depth acknowledgement required to point --db-path at the "
                          "protected real trading_data.db. For the planner-scheduled gpu-server "
                          "launch only (TASK_897/D54); never for a factory or interactive session.")
    ap.add_argument("--train-start", type=str, default="2023-05-16")
    ap.add_argument("--val-start", type=str, default="2024-06-01")
    ap.add_argument("--window-a", type=str, default="2024-11-01,2025-04-01")
    ap.add_argument("--window-b", type=str, default="2025-04-01,2026-03-01")
    ap.add_argument("--out", type=str, default=str(_THIS.parent / "result.md"))
    ap.add_argument("--smoke", action="store_true", help="tiny pre-registered SMOKE grid, not the real one")
    ap.add_argument("--skip-inherited-ic-check", action="store_true",
                     help="Skip the 806/808-inherited signal-IC assertion (its pinned targets are "
                          "real-data magic numbers; skip at fixture/synthetic scale).")
    ap.add_argument("--seed", type=int, default=897)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    tiers = parse_fee_tiers(args.attainable_maker_fee_bp)
    db_path = assert_db_path_allowed(args.db_path, args.scheduled_real_data_run)

    a_start, a_end = _parse_window("--window-a", args.window_a)
    b_start, b_end = _parse_window("--window-b", args.window_b)
    windows = build_windows(a_start, a_end, b_start, b_end)

    if args.smoke:
        grid_h, grid_k, grid_variant, grid_tau, grid_n = (
            SMOKE_GRID_H, SMOKE_GRID_K, SMOKE_GRID_VARIANT, SMOKE_GRID_TAU, SMOKE_GRID_N,
        )
    else:
        grid_h, grid_k, grid_variant, grid_tau, grid_n = GRID_H, GRID_K, GRID_VARIANT, GRID_TAU, GRID_N

    cells, meta = run_sweep(
        attainable_maker_fee_bp=tiers, db_path=db_path, train_start=args.train_start,
        val_start=args.val_start, windows=windows, grid_h=grid_h, grid_k=grid_k,
        grid_variant=grid_variant, grid_tau=grid_tau, grid_n=grid_n, seed=args.seed,
        n_boot=args.n_boot, verify_inherited_ic=not args.skip_inherited_ic_check,
    )
    out = Path(args.out)
    write_artifact(cells, meta, out)
    print(f"[897-sweep] DONE. {meta['n_cells_recorded']}/{meta['n_cells_expected']} cells "
          f"recorded (ok={meta['n_cells_ok']}). Wrote {out}. Runtime {meta['runtime_s']:.1f}s.",
          flush=True)


if __name__ == "__main__":
    main()

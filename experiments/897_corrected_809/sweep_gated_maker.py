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
  CAVEAT ON THE SCOPE OF THAT SENTENCE (D59 fix cycle 2). It is a guarantee about the DB FILE,
  and about SIDECARS only for a ROLLBACK-JOURNAL (`journal_mode=delete`) source. Every DB open
  in this harness goes through `alias_read_layer._connect_ro` (`file:...?mode=ro`), which can
  never write the DB file itself — but a `mode=ro` read of a **WAL-mode** database still CREATES
  `-shm`/`-wal` sidecars where none existed (demonstrated on a WAL copy of the fixture: one
  SELECT created `-shm` 32768 B, `-wal` 0 B). This is NOT live against the protected asset:
  `trading_data.db` is verified rollback-journal today (file-header bytes 18/19 both `1`; `2`
  would mean WAL), and an end-to-end `--smoke` run was checked to leave its directory
  byte-identical (sha256 + full listing). So: no sidecar is created on the real DB — but the
  guarantee is mode-dependent, not unconditional, and would not hold on a WAL source.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import os
import sys
import time
from dataclasses import dataclass
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
import alias_read_layer as alias_layer  # noqa: E402  (TASK_897 D56(a) read-time alias union)
import calibration as cal  # noqa: E402  (TASK_897 D55(a) CI calibration — method reused from
                            # 892, table measured fresh for this Sharpe-ratio/block regime)

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


class _ReadOnlySqlite3ForProbe:
    """D59 fix — stands in for the `sqlite3` module inside `probe`'s namespace for the
    duration of the `probe.report_bar_frequency` call in `_run_sweep_inner` below, so
    `probe.py`'s bare `sqlite3.connect(cfg.db_path)` (READ-WRITE-CREATE — demonstrated: a
    nonexistent `--db-path` got a zero-byte file CREATED at it, DDL committed through the
    exact shipped form) opens read-only instead. `probe.py` stays byte-untouched (809-lineage
    doctrine); this closes the handle from the caller's side by REBINDING THE NAME
    `probe.sqlite3` rather than mutating `sqlite3.connect` in place. `sqlite3` is one cached
    module object shared by every importer, including `alias_read_layer` itself — mutating its
    `connect` attribute globally would ALSO intercept `alias_read_layer._connect_ro`'s own
    internal `sqlite3.connect(...)` call and recurse forever (measured while building this fix:
    a bare `probe.sqlite3.connect = lambda ...: alias_layer._connect_ro(...)` blew the stack).
    Routed through the existing `alias_read_layer._connect_ro` (percent-quoted
    `file:...?mode=ro` URI — the same belt `assert_db_path_allowed` relies on) — no second,
    hand-rolled connect helper. Module-level (not a local class in `_run_sweep_inner`) so the
    test suite can exercise it directly without paying for a full sweep.

    WHAT "READ-ONLY" HERE DOES AND DOES NOT GUARANTEE (D59 fix cycle 2, FINDING 2). The handle
    cannot write the DB FILE, in every case. It leaves the SOURCE DIRECTORY byte-identical only
    for a ROLLBACK-JOURNAL (`journal_mode=delete`) source: on a WAL-mode database a `mode=ro`
    read still CREATES `-shm`/`-wal` sidecars where none existed. Not live against the protected
    asset — `trading_data.db` is verified rollback-journal today (header bytes 18/19 both `1`),
    and an end-to-end `--smoke` run left it byte-identical (sha256 + full listing) — but the
    byte-identity claim is mode-dependent and must not be read as unconditional.

    KNOWN LIMITATION, DELIBERATELY NOT GUARDED: a `--db-path` naming a non-regular file (a FIFO)
    hangs, because a `mode=ro` URI open on one blocks forever where the pre-fix bare connect
    failed fast. A type guard for this was added and then REMOVED on operator call — `--db-path`
    is an operator-supplied path to a real database on a scheduled run, and chasing ever more
    exotic spellings of an implausible input is TASK_891's failure mode. Pass a real sqlite file.
    See `alias_read_layer._connect_ro`'s docstring."""

    @staticmethod
    def connect(db, *a, **k):
        return alias_layer._connect_ro(db)


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
    ci_lo_UNCALIBRATED: float          # the raw nominal-95% bounds — NEVER band-readable alone.
    ci_hi_UNCALIBRATED: float
    ci_advisory: bool                  # True iff this cell's n_decisions stratum is CERTIFIED
                                        # (D55(a)) AND the dependence guard passed — TASK_897 D57
                                        # (2026-08-22): even then, ci_lo_ADVISORY/ci_hi_ADVISORY
                                        # below are NEVER band-readable on this leg. The guard
                                        # behind this flag (honest_estimator.dependence_length,
                                        # tau) is measured inert out-of-family — a near-100% pass
                                        # rate on THIS row's real data is not evidence the
                                        # certifying condition holds. Recorded for TASK_901 to
                                        # re-read retroactively FROM THE SAME ARTIFACTS.
    ci_calibration_k: float            # the width-inflation factor applied (NaN if uncertified).
    ci_calibration_source_n_dec: int   # which calibrated stratum supplied k (-1 if uncertified).
    ci_lo_ADVISORY: float              # ADVISORY bounds — NaN when ci_advisory is False. Even
    ci_hi_ADVISORY: float              # when finite, per D57, NEVER band-readable on this leg.
    block_used: int                    # the block length THIS cell's CI was bootstrapped at.
    measured_dependence_len: float     # this cell's OWN realized per-decision dependence, as READ
                                        # BY TAU (`honest_estimator.dependence_length`) — a
                                        # linear-ACF statistic valid as a range proxy only inside
                                        # the triangular-ACF family the calibration was measured
                                        # against; measured inert on this row's real per-decision
                                        # data (D57). NaN = unmeasurable = refuse regardless. See
                                        # calibration.DEPENDENCE_REFUSAL_MECHANISM.
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
                ci_hi_UNCALIBRATED=float("nan"), ci_advisory=False,
                ci_calibration_k=float("nan"), ci_calibration_source_n_dec=-1,
                ci_lo_ADVISORY=float("nan"), ci_hi_ADVISORY=float("nan"),
                block_used=-1, measured_dependence_len=float("nan"),
                pnl=float("nan"), maxdd=float("nan"), sharpe_per_bar_diagnostic=float("nan"),
            ))
    return out


def apply_ci_calibration(sharpe_decision: float, ci_lo_uncal: float, ci_hi_uncal: float,
                          n_decisions: int, table: Dict,
                          measured_dependence_len: Optional[float] = None,
                          block_used: Optional[int] = None,
                          ) -> Tuple[bool, float, int, float, float]:
    """TASK_897 D55(a) — widen the RAW nominal interval by the calibrated k for this cell's
    n_decisions stratum (nearest-below floor lookup, `calibration.calibrated_k_for`), around the
    POINT ESTIMATE (`sharpe_decision`), never around 0 — the same convention
    `calibration.k_needed_for_trial` widens around when IT measures coverage, so applying k here
    reproduces exactly the interval whose coverage was certified.

    The stratum is selected by `calibration.calibrated_k_for`, which matches the cell's REAL
    BLOCK LENGTH exactly and only then takes the nearest-below n_dec within that block — a cell
    can never borrow a k measured at a different block length (the MAJOR-1 fix; `block_for` is a
    step function of n_dec that saturates at the cap, so n_dec-only borrowing silently crossed
    block regimes).

    Returns (ci_advisory, k, source_n_dec, ci_lo_ADVISORY, ci_hi_ADVISORY). If the pairing was
    never certified (`achieved=False` for every same-block candidate, nothing at or below this
    n_dec at this block, or the input isn't finite), returns `(False, nan, -1, nan, nan)` — the
    qualifier stays on this cell and its `ci_lo_ADVISORY`/`ci_hi_ADVISORY` are NaN, never a silent
    fallback to the raw bounds. TASK_897 D57 (2026-08-22): even `ci_advisory=True` is NEVER
    band-readable on this leg (see the dependence paragraph below and
    `honest_estimator.CI_UNCALIBRATED_DISCLOSURE`) — the returned pair is ADVISORY, recorded for
    TASK_901 to re-read retroactively, not a certification a verdict may be read off.

    FAIL-CLOSED ON A NON-WIDENING RESULT (MINOR 5). The whole point of `k` is to WIDEN, and the
    only direction that must be impossible is a band that got EASIER to clear. If the raw
    bootstrap interval ever failed to bracket its own point estimate (`sharpe < ci_lo_uncal`, or
    `sharpe > ci_hi_uncal`), the arithmetic above would invert and a k>1 would RAISE the lower
    bound; a k<1 would likewise narrow a well-formed interval. Neither has been reproduced
    (~7,000 adversarial trials found no inversion, and every certified k is in [1.2, 1.6]), so
    this is a guard, not a live defect: the widened interval is required to be a SUPERSET of the
    raw one (`ci_lo_ADVISORY <= ci_lo_uncal and ci_hi_ADVISORY >= ci_hi_uncal`), and anything else
    is refused as uncertified rather than published. `calibration.k_needed_for_trial` guards the
    same degeneracy on the measurement side (`d_lo <= 0 -> inf`); this is its consumer-side
    analogue.

    FAIL-CLOSED ON A DEGENERATE (ZERO-WIDTH) RAW INTERVAL (fix cycle 3, MINOR 4). The superset
    guard above admits EQUALITY, so a raw interval with `ci_lo_uncal == ci_hi_uncal == sharpe`
    satisfied it trivially and certified a zero-width band: `apply_ci_calibration(5.0, 5.0, 5.0,
    1000, TABLE)` used to return `ci_advisory=True` with `ci_lo_ADVISORY=+5 > 0`, i.e. the
    pre-registered band's third criterion satisfied off an interval containing no information.
    That is the same class as the original BLOCKER (`n_dec=2`, CI `[+22.93,+22.93]`). It was
    never demonstrated reachable through `compute_honest_metrics` (0/400 adversarial trials), so
    this is a guard rather than a live-defect fix — `ci_hi_uncal > ci_lo_uncal` is now REQUIRED.

    FAIL-CLOSED ON UNCERTIFIED OR OUT-OF-FAMILY DEPENDENCE (fix cycle 3, MAJOR 1(b); narrowed by
    TASK_897 D57, 2026-08-22). Every `k` in the table was certified at `dep == block` ONLY — the
    easiest point of the calibration's own sweep, not the worst
    (`calibration.CERTIFICATION_IS_CONDITIONAL_ON_DEPENDENCE`); at 8x that dependence the SHIPPED
    k covers 0.52-0.55, the magnitude that caused D55 to be raised. `block` is chosen from the
    SAMPLE SIZE (`n_dec // 20`), never from the data, so "dependence == block" would be a
    self-fulfilling assumption if it were merely assumed. `measured_dependence_len`
    (`honest_estimator.dependence_length`, THIS cell's own per-decision return series) ATTEMPTS
    to check that condition via tau — a linear-ACF statistic that equals the dependence's RANGE
    (what the certification actually needs) only inside the triangular-ACF family the table was
    measured against, and demonstrably UNDER-READS out of family
    (`calibration.MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE`). ON THIS ROW'S REAL PER-DECISION DATA
    THE GUARD IS EXPECTED NOT TO BITE (every real series reads tau=1-2 against block=60,
    `calibration.MEASURED_REAL_DATA_TAU_INERT_HALTED_CYCLE`), so a pass here is NOT evidence the
    condition holds. `block_used` is the block its CI was actually bootstrapped at, and MUST equal
    `calibration.block_for(n_decisions)` — a mismatch REFUSES (D57's hardening: the shipped claim
    that the two "cannot drift" is now enforced, not merely asserted). Certification requires
    `measured_dependence_len <= block_used`. Anything else refuses:
      * `block_used`, if supplied, does not equal `calibration.block_for(n_decisions)`;
      * `measured_dependence_len is None` — the DEFAULT, so a caller that forgets to pass the
        measurement gets no certification rather than a silent pass;
      * NaN / non-finite — `dependence_length` returns NaN when the series is too short,
        degenerate, or its autocorrelation never returns to the white-noise band (dependence too
        long to resolve is UNKNOWN dependence, and unknown dependence must never certify);
      * measured dependence strictly greater than the block.
    A refused cell keeps `ci_advisory=False`, NaN `ci_lo_ADVISORY`/`ci_hi_ADVISORY` and the
    UNCALIBRATED qualifier. EITHER WAY — refused or `ci_advisory=True` — the pre-registered band's
    'CI lo > 0' criterion is **UNAVAILABLE-because-uncalibratable** on this leg; a `True` only
    means the guard did not refuse, not that its out-of-family-inert condition was established.
    See `calibration.DEPENDENCE_REFUSAL_MECHANISM` for the measured operating characteristic
    (IN-FAMILY: 0% refusal on i.i.d. series, 0.27-0.42 conservative refusal at dep == block (the
    measured range across the five n_dec rows of `calibration.MEASURED_DEPENDENCE_REFUSAL_OC`:
    0.270 at n_dec=100 up to 0.422 at n_dec=3000 — corrected from "~30-42%", which excluded the
    low end its own record carries and which `calibration.py` already stated correctly), >= 94%
    refusal at dep >= 2x block — none of which generalises out of family) and **TASK_901** (a
    range-based dependence probe, validated out-of-family) for what would re-earn this conjunct,
    retroactively, FROM THE SAME ARTIFACTS.
    """
    if not (np.isfinite(sharpe_decision) and np.isfinite(ci_lo_uncal) and np.isfinite(ci_hi_uncal)
            and n_decisions and np.isfinite(n_decisions) and n_decisions > 0):
        return False, float("nan"), -1, float("nan"), float("nan")
    if not (ci_hi_uncal > ci_lo_uncal):      # MINOR 4: a zero-width (or inverted) raw interval
        return False, float("nan"), -1, float("nan"), float("nan")
    # D57 hardening: `block_used`, when supplied, must equal what the estimator's own rule would
    # select for this n_decisions — the shipped claim ("the two cannot drift") is now enforced
    # rather than merely asserted. Defence-in-depth: the one caller today always passes a
    # matched pair, so this changes no numerics for the shipped path (see
    # test_apply_ci_calibration_refuses_a_block_used_mismatch).
    expected_block = cal.block_for(int(n_decisions))
    # FIX 1: a non-finite `block_used` (nan/inf) must refuse like any other mismatch, not raise
    # out of `int()` — the docstring promises "anything else refuses" and this makes that literal.
    if block_used is not None:
        if not np.isfinite(block_used):
            return False, float("nan"), -1, float("nan"), float("nan")
        if int(block_used) != expected_block:
            return False, float("nan"), -1, float("nan"), float("nan")
    block = expected_block
    # MAJOR 1(b): the certification's own condition, ATTEMPTED-measured — refuse on unknown or
    # excessive dependence BEFORE any k is looked up, so no refused cell can carry a k in its
    # record. (Measured inert out-of-family, D57 — see the docstring above.)
    if measured_dependence_len is None or not np.isfinite(measured_dependence_len):
        return False, float("nan"), -1, float("nan"), float("nan")
    if block <= 0 or float(measured_dependence_len) > block:
        return False, float("nan"), -1, float("nan"), float("nan")
    k, achieved, src_n_dec = cal.calibrated_k_for(int(n_decisions), table)
    if not achieved or k is None or not np.isfinite(k):
        return False, float("nan"), -1, float("nan"), float("nan")
    lo = sharpe_decision - k * (sharpe_decision - ci_lo_uncal)
    hi = sharpe_decision + k * (ci_hi_uncal - sharpe_decision)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= ci_lo_uncal and hi >= ci_hi_uncal
            and lo <= hi):
        return False, float("nan"), -1, float("nan"), float("nan")
    return True, float(k), int(src_n_dec), float(lo), float(hi)


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
    # --- GATE (iv), SECOND HALF: every declared attainable tier must actually BE IN the grid. ---
    # D57 fix cycle 4, item B1. The artifact tells the reader the tier set is "the Kraken-reachable
    # subset of the fee grid below" (see `write_artifact`), and NOTHING enforced that. Pass
    # Kraken's real ladder `--attainable-maker-fee-bp 14,12,10` against `GRID_FEE_BP =
    # (0, 8, 16)` and not ONE declared tier is measured — yet the run completes, the artifact
    # renders the subset sentence, and the planner reads CELL CLEARS off 8bp cells. 8 < 10, i.e.
    # STRICTLY BETTER THAN REALITY: the band is "clears at a maker-fee tier the operator can
    # plausibly reach", so an unenforced subset claim corrupts the verdict in the OPTIMISTIC
    # direction. Refuse before touching any data, naming the offending tiers and the grid.
    grid_fee_set = [float(x) for x in grid_fee_bp]
    missing_tiers = [t for t in attainable_maker_fee_bp if t not in grid_fee_set]
    if missing_tiers:
        raise ValueError(
            f"GATE (iv) REFUSAL: attainable maker-fee tier(s) {missing_tiers} bp are NOT in this "
            f"run's pre-registered fee grid {sorted(set(grid_fee_set))} bp, so NOT ONE of them "
            "would be measured. The artifact would still call the tier set 'the Kraken-reachable "
            "subset of the fee grid', and a verdict read off a CHEAPER grid cell than any "
            "attainable tier is strictly better than reality — optimistic, which is the one "
            "direction this harness must never fail in. Either pass tiers the grid measures "
            f"(available: {sorted(set(grid_fee_set))} bp) or change the pre-registered grid "
            "deliberately (a planner decision, not a flag) and re-run."
        )

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
    # D59 fix — `probe.report_bar_frequency` is this harness's FIRST data touch on every run;
    # scope-patch `probe.sqlite3` to `_ReadOnlySqlite3ForProbe` (defined above, see its
    # docstring for the hazard, the demonstration, and why this is a rebind-the-name patch
    # rather than an in-place mutation of `sqlite3.connect`) for the duration of that ONE call,
    # restored in `finally` — exactly like the `decide.CFG` scope-patch above.
    _orig_probe_sqlite3 = probe.sqlite3
    probe.sqlite3 = _ReadOnlySqlite3ForProbe()
    try:
        bar_sec = probe.report_bar_frequency(cfg)
    finally:
        probe.sqlite3 = _orig_probe_sqlite3

    # --- GATE (iii) / D56(a): the canon alias union, applied as a READ-TIME layer owned by
    # THIS runner (`alias_read_layer.py`) for BOTH the universe ranking AND the panel build —
    # the 809 lineage (`probe.py`) is never edited, and its OWN exact-spelling ranking is also
    # computed here (for free, from the same per-id counts) purely so any universe-rank change
    # under the union read can be named, per D56's explicit disclosure requirement.
    universe = alias_layer.rank_universe_with_alias_union(cfg)
    symbols = universe["symbols"]
    asset_ids = universe["asset_ids"]
    alias_groups = universe["alias_groups"]
    ohlcv, grid_ts, alias_gate_disclosure, per_slot_id_row_counts = (
        alias_layer.build_aligned_ohlcv_with_alias_union(cfg, asset_ids, alias_groups, bar_sec)
    )
    high = np.ascontiguousarray(ohlcv[:, :, 1])
    low = np.ascontiguousarray(ohlcv[:, :, 2])
    close = np.ascontiguousarray(ohlcv[:, :, 3])
    del ohlcv
    gc.collect()

    material_slot_report = alias_layer.alias_material_slot_report(cfg, windows)

    meta = {
        "attainable_maker_fee_bp_tiers": attainable_maker_fee_bp,   # gate (iv): recorded FIRST
        "db_path": db_path,
        "bar_sec": bar_sec,
        "n_assets": len(asset_ids),
        "n_universe": cfg.n_universe,
        "symbols": symbols,
        "alias_union_applied": True,
        "alias_groups": {sym: grp for sym, grp in zip(symbols, alias_groups)},
        "alias_material_slots": list(alias_layer.MATERIAL_ALIAS_SYMBOLS),
        "alias_gate_disclosure": alias_gate_disclosure,
        "alias_per_slot_id_row_counts": {sym: cnt for sym, cnt in
                                          zip(symbols, per_slot_id_row_counts)},
        "alias_material_slot_report": material_slot_report,
        "alias_rank_changes": universe["rank_changes"],
        "alias_top64_entered": universe["top64_entered"],
        "alias_top64_left": universe["top64_left"],
        "alias_unknown_symbols": universe["unknown"],
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
                        ci_adv, ci_k, ci_src, ci_lo_a, ci_hi_a = apply_ci_calibration(
                            m.sharpe_decision, m.ci_lo_UNCALIBRATED, m.ci_hi_UNCALIBRATED,
                            m.n_decisions, cal.MEASURED_CALIBRATION_TABLE,
                            measured_dependence_len=m.measured_dependence_len,
                            block_used=m.block_used)
                        cells.append(CellRecord(
                            window=wname, H=H, k=k, variant=variant, tau=tau, N=N,
                            fee_bp=fee_bp, adv_bp=adv_bp, status="ok", reason="",
                            n_bars=m.n_bars, n_decisions=m.n_decisions,
                            decisions_per_year=m.decisions_per_year, mean_hold_bars=m.mean_hold_bars,
                            turnover_mean=m.turnover_mean, fill_rate=fill,
                            sharpe_decision=m.sharpe_decision,
                            ci_lo_UNCALIBRATED=m.ci_lo_UNCALIBRATED,
                            ci_hi_UNCALIBRATED=m.ci_hi_UNCALIBRATED,
                            ci_advisory=ci_adv, ci_calibration_k=ci_k,
                            ci_calibration_source_n_dec=ci_src,
                            ci_lo_ADVISORY=ci_lo_a, ci_hi_ADVISORY=ci_hi_a,
                            block_used=m.block_used,
                            measured_dependence_len=m.measured_dependence_len,
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


def _cell_block(c: CellRecord) -> Optional[int]:
    """The block length THIS cell's CI was (or would have been) bootstrapped at, or None when no
    interval exists for it at all. Factored out so the banner, the per-cell column and
    `_dependence_ok` cannot read it three slightly different ways (D57 item B2)."""
    return c.block_used if c.block_used and c.block_used > 0 else (
        cal.block_for(c.n_decisions) if c.n_decisions and c.n_decisions > 0 else None)


# The four mutually-exclusive states an `ok` cell can be in with respect to the dependence guard.
# THE BANNER AND THE TABLE BOTH READ THIS ONE FUNCTION, so a banner count can no longer disagree
# with the rows it is counting.
DEP_NO_CI = "no_ci"                  # too few decisions for an interval — the guard never ran
DEP_UNMEASURABLE = "unmeasurable"    # interval exists, dependence could not be measured -> refuse
DEP_REFUSED = "refused"              # measured dependence exceeds the block -> refuse
DEP_PASS = "pass"                    # measured dependence within the block (NOT evidence — D57)


def _dependence_status(c: CellRecord) -> str:
    """Which of the four states above this cell is in.

    D57 fix cycle 4, item B2 — A NO-DATA RUN USED TO RENDER AS THE GUARD WORKING. `_dependence_ok`
    returned False for a cell with `block_used=-1`/`n_decisions=0`, so every such cell landed in
    the banner's `n_dep_refused` count, while the per-cell column rendered it (correctly) as
    "no CI computed". On the fixture run that produced a banner reading "144/144 ok cells were
    REFUSED an advisory interval on dependence grounds (144 of them because the dependence could
    not be measured at all)" against a table in which ZERO rows say `REFUSED ON DEPENDENCE`: the
    reader was told the safety guard bit on every cell — which reads as evidence of health — when
    in fact nothing was measured at all. A mixed real run inflates the same counter on the srv
    leg. A cell with no interval is NOT a dependence refusal and is counted separately."""
    blk = _cell_block(c)
    if blk is None or not c.n_decisions or int(c.n_decisions) < hest.MIN_DECISIONS_FOR_SHARPE:
        return DEP_NO_CI
    if not np.isfinite(c.measured_dependence_len):
        return DEP_UNMEASURABLE
    if float(c.measured_dependence_len) > blk:
        return DEP_REFUSED
    return DEP_PASS


def _dependence_ok(c: CellRecord) -> bool:
    """Did THIS cell's measured per-decision dependence stay inside the block its CI was
    bootstrapped at — the condition every shipped `k` was certified under? Unmeasurable
    dependence is False (fail-closed), never True; so is a cell that has no interval at all."""
    return _dependence_status(c) == DEP_PASS


def write_artifact(cells: List[CellRecord], meta: Dict, path: Path) -> None:
    """Write the TASK_897 artifact. Metadata (incl. the gate-(iv) attainable-tier set and the
    data epoch, gate iii) is recorded FIRST, before the cell table. NO verdict is computed or
    printed here — CELL CLEARS / ALL FAIL / MIXED is the planner's read of this artifact."""
    L: List[str] = ["# TASK_897 — gated_maker sweep artifact (HONEST per-decision estimator)", ""]
    L += ["> **No verdict in this file.** Gate (iv): the attainable-tier set below is the "
          "operator's input, recorded before any cell data. Reading CELL CLEARS / ALL FAIL / "
          "MIXED off this table is the planner's job (TASK_897 pre-registered bands), not this "
          "harness's — see `tasks/TASK_897_CORRECTED_809_REMEASURE.md`.", ""]
    n_ok = sum(1 for c in cells if c.status == "ok")
    n_adv = sum(1 for c in cells if c.status == "ok" and c.ci_advisory)
    # D57 item B2: one classification, counted from the SAME function the table renders from.
    _dep_states = [_dependence_status(c) for c in cells if c.status == "ok"]
    n_dep_unmeasurable = _dep_states.count(DEP_UNMEASURABLE)
    n_dep_refused = n_dep_unmeasurable + _dep_states.count(DEP_REFUSED)
    n_dep_no_ci = _dep_states.count(DEP_NO_CI)
    n_dep_pass = _dep_states.count(DEP_PASS)
    n_dep_measured = n_ok - n_dep_no_ci        # cells the guard actually ran on
    # The banner's own count sentence. Branching on "did the guard run at all" rather than
    # printing "0/0 ok cells were REFUSED" — a true statement that READS like "no refusals, all
    # good" on exactly the run where nothing was measured, which is the misreading item B2 exists
    # to remove.
    if n_dep_measured == 0:
        dep_count_sentence = (
            "**This artifact: THE DEPENDENCE GUARD RAN ON NO CELLS.** Not one `ok` cell here had "
            f"an interval bootstrapped for it ({n_dep_no_ci} of {n_ok} carried fewer than "
            f"{hest.MIN_DECISIONS_FOR_SHARPE} decisions), so nothing in this artifact was either "
            "REFUSED an advisory interval on dependence grounds or passed the guard — an ABSENCE "
            "OF MEASUREMENT, not a clean bill of health. (A measured cell the guard refuses reads "
            "`REFUSED ON DEPENDENCE` in its `CI(ADVISORY)` column and keeps the qualifier.)"
        )
        dep_pass_clause = (
            "with no cell measured there is no pass rate here to mistake for evidence that the "
            "dep<=block condition holds — and a pass would not have been evidence either:"
        )
    else:
        dep_pass_clause = (
            f"{n_dep_pass}/{n_dep_measured} measured ok cells PASSING the guard here is **NOT "
            "evidence the dep<=block condition holds** —"
        )
        dep_count_sentence = (
            f"**This artifact: {n_dep_refused}/{n_dep_measured} ok cells THE GUARD ACTUALLY RAN "
            f"ON were REFUSED an advisory interval on dependence grounds** "
            f"({n_dep_unmeasurable} of them because the dependence could not be measured at all, "
            "which also refuses). A refused cell reads `REFUSED ON DEPENDENCE` in its "
            "`CI(ADVISORY)` column and keeps the qualifier. "
            f"**The guard ran on {n_dep_measured} of the {n_ok} ok cells; the other "
            f"{n_dep_no_ci} carried fewer than {hest.MIN_DECISIONS_FOR_SHARPE} decisions, so no "
            "interval was bootstrapped for them and the dependence guard never ran at all** —"
        )
    # A run in which NOT ONE ok cell carried a decision contains no measurement at all. Stated
    # (never graded — this instrument records, the planner rules) and never as an error: on the
    # fixture, tiny windows legitimately yield zero decisions.
    n_ok_with_decisions = sum(1 for c in cells
                              if c.status == "ok" and c.n_decisions and c.n_decisions > 0)
    no_measurement = bool(n_ok) and n_ok_with_decisions == 0
    if no_measurement:
        L += ["> ## :warning: NO DECISIONS IN THIS RUN — **THIS ARTIFACT CONTAINS NO "
              "MEASUREMENT**",
              ">",
              f"> All {n_ok} `ok` cells recorded ZERO decisions, so no cell has a Sharpe, a PnL, "
              "an interval or a dependence measurement: every per-cell number below is NaN and "
              "NONE of the pre-registered band's three criteria can be read off this file — not "
              "the two D57 leaves readable, and not the third. This is a STATEMENT ABOUT THIS "
              "RUN, not a verdict and not an error: on the small fixture DB, or on any window "
              "narrower than the decision rule's own hold length, zero decisions is the normal "
              "outcome and the run is still a valid record that the pipeline executed end to "
              "end. Check the window dates and the data epoch in the metadata below before "
              "reading anything else. (D57 item B3: a run like this used to render as "
              "`recorded: N (ok=N, dropped/failed=0)` with no gate violation anywhere — the "
              "863 green-looking-artifact shape.)", ""]
    L += ["> ## :warning: TASK_897 D57 (2026-08-22) — THE BAND'S THIRD CRITERION, 'CI lo > 0', "
          "IS **UNAVAILABLE-because-uncalibratable ON THIS LEG**",
          ">",
          "> The pre-registered TASK_897 band has three criteria (post-cost Sharpe >= 1.0, "
          "PnL > 0, **CI lo > 0**). **THIS RUN'S VERDICT READS ON THE FIRST TWO PLUS PER-CELL "
          "POINT ESTIMATES ONLY.** The reason is structural (D49/D55), not a per-cell gap: the "
          "block-bootstrap interval's coverage depends on the data's own dependence structure, "
          "certification is conditional on `dep == block`, and the guard built to check that "
          "condition per cell (`honest_estimator.dependence_length`, tau) is measured "
          "OUT-OF-FAMILY on this data — see the box below. Every interval on this artifact, "
          "calibrated/advisory or not, carries the UNCALIBRATED-class qualifier; none is "
          "band-readable.",
          ">",
          "> ## :warning: THE ADVISORY INTERVAL — PER-CELL, WHERE COMPUTED, **NEVER "
          "BAND-READABLE** (TASK_897 D55(a) mechanism, D57 narrowed claim)",
          ">",
          "> Per D55(a): this row ran 892's calibration METHOD (never its table — different "
          "statistic, different block regime) against THIS estimator's own "
          "`block_bootstrap_sharpe_ci`, raised the block-length cap from 20 to "
          f"{cal.NEW_MAX_BLOCK} by measurement (`calibration.BLOCK_CAP_SELECTION_RATIONALE`), "
          "and built a fresh calibration table keyed on the cell's REAL `(block, n_dec)` pairing "
          "(`calibration.MEASURED_CALIBRATION_TABLE`, reproducible via `python calibration.py` — "
          "NOT a quick command: 67 strata up to n_dec=300000 at 300+300 trials x 500-resample "
          "bootstraps each, single-threaded by default; use `--n-jobs <ncpu>` to parallelize, or "
          "`--n-dec-strata 100 500 3000 --n-trials-calib 50 --n-trials-valid 50 --n-boot 100` for "
          "a ~3s smoke check that the harness runs, NOT a reproduction of this table). "
          "A cell gets an ADVISORY interval ONLY IF a CERTIFIED stratum exists AT ITS OWN BLOCK "
          "at or below its `n_decisions` (certified = validated overlapping-dependence coverage "
          f">= {cal.TARGET_COVERAGE} on an INDEPENDENT batch from the one `k` was chosen on, AND "
          "re-measured at the borrowed non-stratum pairings — `python calibration.py "
          "--verify-borrowed`, measured ~8 minutes single-threaded, `--n-jobs <ncpu>` "
          "parallelizes; prints a per-probe-point progress line to stderr) AND the dependence "
          "guard does not refuse it. It is then WIDENED "
          "by that stratum's `k` around the point estimate and carries `ci_advisory=True`, "
          "`ci_lo_ADVISORY`/`ci_hi_ADVISORY` populated — recorded for **TASK_901** to re-read "
          "retroactively FROM THESE SAME ARTIFACTS, but **NEVER the interval the band's 'CI lo > "
          "0' criterion may be read off, on this leg, regardless of `ci_advisory`.** A cell in an "
          "uncertified (block, n_dec) region, or one the dependence guard refuses, keeps "
          "`ci_advisory=False` and `ci_lo_ADVISORY`/`ci_hi_ADVISORY` are NaN (never a silent "
          "fallback to the raw bounds, and never a k borrowed across a block boundary). "
          f"**This artifact: {n_adv}/{n_ok} ok cells carry an advisory interval; the rest keep "
          "the qualifier with NaN advisory bounds — see the per-cell `advisory` column below. "
          "EITHER WAY the third conjunct stays unavailable.**",
          ">",
          "> ### :warning: THE ADVISORY INTERVAL IS **CONDITIONAL ON MEASURED DEPENDENCE <= "
          "BLOCK, AND THAT MEASUREMENT IS OUT-OF-FAMILY HERE**",
          ">",
          "> " + cal.CERTIFICATION_IS_CONDITIONAL_ON_DEPENDENCE,
          ">",
          "> " + dep_count_sentence + " "
          "A no-interval cell reads `UNCALIBRATED (no CI computed; ...)`, NEVER "
          "`REFUSED ON DEPENDENCE`, and is counted separately BECAUSE it is not "
          "evidence of anything: counting a no-data cell as a dependence refusal makes an empty "
          "run read as the safety guard working (D57 item B2). "
          "**ON THIS ROW'S REAL PER-DECISION DATA THE GUARD IS EXPECTED NOT TO BITE**: tau is a "
          "linear-ACF statistic that equals the dependence's RANGE — what the block bootstrap "
          "actually needs — only inside the triangular-ACF family the guard was calibrated "
          "against, and under-reads badly out of family (`calibration."
          "MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE`: a 4%-of-variance persistent component reads "
          "tau=1, certifies 100%, covers only 0.805-0.376). Every series measured on THIS row's "
          "real data reads tau=1-2 against block=60 "
          "(`calibration.MEASURED_REAL_DATA_TAU_INERT_HALTED_CYCLE`, real-data, "
          "provenance-stamped, NOT re-run in this loop), so "
          + dep_pass_clause +
          " it is the guard measured INERT exactly where it is "
          "used. An empirical containment check on the real long/short H=1 series (97 disjoint "
          "1000-decision chunks) contained the grand Sharpe only 0.639 of the time while the "
          "guard certified ~100%. **TASK_901** (a range-based dependence probe — "
          "first-lag-in-band / variance-ratio / spectral-density-at-zero, validated "
          "out-of-family) is the row that replaces this guard, re-reading this leg's third "
          "conjunct retroactively FROM THESE SAME ARTIFACTS. "
          "**The safe fallback is strictly better than the alternative**: a cell that keeps the "
          "UNCALIBRATED qualifier merely fails to supply the band's third criterion, whereas a "
          "cell certified under a violated (or unverifiable, out-of-family) assumption would "
          "supply a FALSE one — an interval advertised at >= 0.93 coverage while its real "
          "coverage may be the 0.376-0.805 measured above. Withholding a criterion costs a "
          "verdict; asserting a wrong one corrupts it.",
          ">",
          "> " + cal.DEPENDENCE_REFUSAL_MECHANISM,
          ">",
          "> " + cal.CALIBRATION_SCENARIO_CAVEAT,
          ">",
          "> Raw diagnostic (always present, never band-readable alone): "
          "`ci_lo_UNCALIBRATED`/`ci_hi_UNCALIBRATED` are the plain nominal-95% "
          "circular-block-bootstrap percentiles this module has always produced — "
          "`honest_estimator.CI_UNCALIBRATED_DISCLOSURE` (quoted in full in the metadata below) "
          "still applies to THAT pair of columns specifically.",
          "",
          ]
    L += ["## Metadata (read this BEFORE any cell)",
          f"- **Attainable maker-fee tiers (operator input, gate iv):** "
          f"{meta['attainable_maker_fee_bp_tiers']} bp — the Kraken-reachable subset of the fee "
          "grid below; zero-maker-fee is out of reach per the Fork-C venue ruling. **That "
          "'subset' is ENFORCED, not asserted** (D57 item B1): `run_sweep` refuses to run at all "
          "if any declared tier is absent from the fee grid, so every tier named here has "
          "measured cells. Without that check a declared ladder the grid never measures (e.g. "
          "14/12/10 bp against a 0/8/16 grid) would leave the planner reading a verdict off "
          "cheaper cells than any attainable tier — optimistic, the one direction this harness "
          "must not fail in.",
          f"- **Data source:** `{meta['db_path']}`; bar_sec={meta['bar_sec']}; "
          f"{meta['n_assets']} assets: {meta['symbols']}.",
          f"- **Data epoch (gate iii):** grid {meta['grid_start_iso']} .. {meta['grid_end_iso']} "
          f"(epoch [{meta['grid_epoch']['start']}, {meta['grid_epoch']['end']}]).",
          "- **877 alias fold: APPLIED, as a READ-TIME LAYER owned by THIS runner (D56(a)) — "
          "the 806/808/809 lineage itself is UNCHANGED.** `probe.py`/`backtest.py`/`decide.py` "
          "still contain zero references to `asset_aliases`/`normalize_symbol_key`/"
          "`resolve_requested_symbols`/`group_asset_rows` (re-verified every run by "
          "`test_the_lineage_really_has_no_alias_handling`, so this sentence cannot drift back "
          "into false); the union is applied INSTEAD by `alias_read_layer.py`, called from this "
          "script, for BOTH the universe ranking and the panel build, per D56's explicit "
          "requirement that the fold not be limited to patching holes in a fixed set (the "
          "universe is computed from bar counts, so a union read can change top-64 membership). "
          f"**Material slots (the TASK_868/877 identity-split class, D56):** "
          f"{meta['alias_material_slots']}. "
          f"**Universe-rank changes measured under the union read:** "
          f"{len(meta['alias_rank_changes'])} candidate symbol(s) changed rank position; "
          f"entered top-{meta.get('n_universe', 64)}: {meta['alias_top64_entered'] or 'none'}; "
          f"left top-{meta.get('n_universe', 64)}: {meta['alias_top64_left'] or 'none'} "
          "(measured over the TRAIN window that locks the universe — see "
          "`alias_rank_changes`/`top64_entered`/`top64_left` in the raw metadata for the full "
          "list; empirically NONE for this data epoch, because the 6 material slots' twin ids "
          "carry zero rows inside the train window — the identity split happened later, inside "
          "the OOS span, so it cannot move who makes the top-64 here, only what the PANEL reads "
          "for the slots already chosen). **Per-slot alias groups (symbol -> "
          f"[asset_ids, exact-spelling first]):** {meta['alias_groups']}.",
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

    L += ["## CI calibration table (D55(a) — the certified method, measured coverage, chosen "
          "block length, recorded BEFORE any cell is read)",
          f"- **Block-length cap:** raised from {cal.OLD_MAX_BLOCK} to {cal.NEW_MAX_BLOCK} "
          "decisions — measurement/rationale: " + cal.BLOCK_CAP_SELECTION_RATIONALE,
          f"- **Target coverage:** {cal.TARGET_COVERAGE} (two-sided, overlapping-dependence "
          "scenario — the certification bar, same as 892's).",
          "- **Stratification key:** the REAL `(block, n_dec)` pairing, NOT `n_dec` alone. "
          "`block_for` is a STEP function of n_dec that saturates at the cap "
          f"(`max(2, min({cal.NEW_MAX_BLOCK}, n_dec // 20))`), so an n_dec-only nearest-below "
          "lookup could hand a block-60 cell a `k` that was measured at block 50 — measured at "
          "the k that lookup handed out (1.3413, from the PRE-restratification grid; no stratum "
          "in the shipped table below carries it), that borrowing put achieved coverage at "
          "0.9250/0.9400/0.9500 / 0.9550/0.9400/0.9350 / 0.9100/0.9250/0.9200 across n_dec "
          "1200/1350/1500 x seeds 0/1/2 (200 trials x 400 resamples) — mean ~0.933, 4 of 9 cells "
          "UNDER the 0.93 bar, i.e. sitting ON the target at the bottom of the achieved band "
          "(the 62 achieved strata below span 0.9333-0.9833, four of them under 0.943) rather "
          "than inside it. A stratum is now eligible ONLY if it was measured at "
          "the query cell's own block; the strata are generated FROM `block_for` "
          "(`calibration.n_dec_strata_for_grid`) so there is one at every block-transition point.",
          f"- **Certified region:** {cal.certified_region_description()}",
          "- **THE CONDITION ON ALL OF IT (read before any `k` below):** "
          + cal.CERTIFICATION_IS_CONDITIONAL_ON_DEPENDENCE,
          "- **The SHIPPED k, stressed past its certified dependence** (`python calibration.py "
          "--sweep-shipped-k-mismatch`). `k` is held FIXED at the value the table ships for that "
          "n_dec while the real dependence is pushed to 1x/2x/4x/8x the block; two independent "
          "seeds per point. Contrast `measure_dependence_mismatch`, which RE-DERIVES k at each "
          "level and therefore cannot show the shipped interval failing:",
          "", "  | n_dec | block | shipped k | dep multiple | dep | coverage (seed 1 / seed 2) | "
          "trials x boot |",
          "  |--:|--:|--:|--:|--:|--:|--:|"]
    for r in cal.shipped_k_mismatch_rows():
        covs = " / ".join(f"{c:.3f}" for c in r["coverage_by_seed"])
        L.append(f"  | {r['n_dec']} | {r['block']} | {r['shipped_k']:.4f} | "
                 f"{r['dep_multiple']}x | {r['dep']} | {covs} | "
                 f"{r['n_trials']}x{r['n_boot']} |")
    L += ["", "- **Dependence-refusal guard, operating characteristic** (`python calibration.py "
          "--sweep-dependence-refusal`) — what fraction of draws the guard refuses at each real "
          "dependence, measured in BOTH directions so the conservative-error rate is disclosed "
          "rather than hidden:",
          "", "  | n_dec | block | real dependence | refusal rate | of which unmeasurable | "
          "median measured dep |",
          "  |--:|--:|--:|--:|--:|--:|"]
    for nd in sorted(cal.MEASURED_DEPENDENCE_REFUSAL_OC):
        e = cal.MEASURED_DEPENDENCE_REFUSAL_OC[nd]
        for mult in sorted(e["by_dep_multiple"]):
            d = e["by_dep_multiple"][mult]
            L.append(f"  | {nd} | {e['block']} | {d['dep']} | {d['refusal_rate']:.3f} | "
                     f"{d['unmeasurable_rate']:.3f} | {d['median_tau']} |")
    L += ["", "- **How to read the `achieved=False` rows:** "
          + cal.DEMOTION_IS_MC_JITTER_NOT_A_DEMONSTRATED_HOLE,
          "", "  | re-measured | group | recorded in table | coverage by seed | mean | "
          "pooled trials |", "  |--:|---|--:|---|--:|--:|"]
    for group in ("demoted", "achieved_sample"):
        for nd in sorted(cal.MEASURED_TABLE_SEED_JITTER[group]):
            e = cal.MEASURED_TABLE_SEED_JITTER[group][nd]
            seeds = " / ".join(f"{c:.4f}" for c in e["coverage_by_seed"])
            rec = e.get("recorded_in_table")
            L.append(f"  | {nd} | {group} | {rec if rec is not None else '-'} | {seeds} | "
                     f"{e['mean']:.4f} | {e.get('pooled_n_trials', 300 * len(e['coverage_by_seed']))} |")
    L += [""]
    L += [
          "- **Per-stratum table** (block = the length `compute_honest_metrics` "
          "itself selects at that n_dec under the raised cap; `k` = the width-inflation factor; "
          "`achieved` = validated overlapping-dependence coverage >= target on an INDEPENDENT "
          "batch — ONLY `achieved=True` strata AT THE CELL'S OWN BLOCK drop the UNCALIBRATED "
          "qualifier):",
          "", "  | n_dec stratum | block | k | achieved | cov(overlap) | cov(iid) | reason |",
          "  |--:|--:|--:|:--:|--:|--:|---|"]
    for nd in sorted(cal.MEASURED_CALIBRATION_TABLE):
        e = cal.MEASURED_CALIBRATION_TABLE[nd]
        L.append(f"  | {nd} | {e.get('block')} | {e.get('k')!r} | {e.get('achieved')} | "
                 f"{e.get('validation_coverage_overlapping')!r} | "
                 f"{e.get('validation_coverage_iid')!r} | {e.get('reason') or ''} |")
    L += ["", "- **Achieved coverage AT THE BORROWED PAIRINGS** — the certification's second "
          "half, and the one that matters for real cells: a real cell essentially never lands on "
          "a pinned stratum, so each row below is a NON-stratum n_dec run at ITS OWN real block "
          "with the `k` `calibrated_k_for` actually hands it, re-measured on a fresh true-null "
          "batch. A row below target would mean the region around it is NOT certified:",
          "", "  | n_dec (borrower) | block | borrowed k | from stratum | cov(overlap) | "
          "cov(iid) | trials x boot | meets target |",
          "  |--:|--:|--:|--:|--:|--:|--:|:--:|"]
    for nd in sorted(cal.MEASURED_BORROWED_PAIRING_COVERAGE):
        e = cal.MEASURED_BORROWED_PAIRING_COVERAGE[nd]
        L.append(f"  | {nd} | {e.get('block')} | {e.get('borrowed_k')!r} | "
                 f"{e.get('borrowed_from_n_dec')!r} | {e.get('coverage_overlapping')!r} | "
                 f"{e.get('coverage_iid')!r} | {e.get('n_trials')}x{e.get('n_boot')} | "
                 f"{e.get('meets_target')} |")
    L += ["", "> " + cal.CALIBRATION_SCENARIO_CAVEAT, ""]
    L += ["> Raw-interval disclosure (`ci_lo_UNCALIBRATED`/`ci_hi_UNCALIBRATED`, present on every "
          "cell regardless of certification): " + hest.CI_UNCALIBRATED_DISCLOSURE.replace("\n", " "),
          ""]

    L += ["## Alias union gate disclosure (D56(a) — asset_aliases.alias_agreement_gate, run "
          "ONCE over the full grid span for every multi-id slot)"]
    if meta.get("alias_gate_disclosure"):
        for grp in meta["alias_gate_disclosure"]:
            L.append(f"- asset_ids {grp['asset_ids']}: {grp['overlaps']}")
    else:
        L.append("- No multi-id alias group was requested in this universe (nothing to gate).")
    L += ["", "## Material-slot before/after row counts (D56 — the 6 TASK_868/877 "
          "identity-split slots, exact-spelling id alone vs the alias-union total, PER WINDOW "
          "this sweep actually runs)",
          "| symbol | window | exact_id | twin_ids | exact_rows | union_rows | "
          "missing_via_twin | missing % of union |",
          "|---|---|--:|---|--:|--:|--:|--:|"]
    for r in meta.get("alias_material_slot_report", []):
        L.append(f"| {r['symbol']} | {r['window']} | {r['exact_id']} | {r['twin_ids']} | "
                 f"{r['exact_rows']} | {r['union_rows']} | {r['missing_rows_via_twin']} | "
                 f"{r['missing_pct_of_union']:.1f}% |")
    L += [""]

    L += ["## Cells (every grid cell — status ok/dropped/failed, no omissions)",
          "| window | H | k | var | tau | N | fee bp | adv bp | status | n_bars | n_dec | "
          "dec/yr | mean_hold | turnover | fill | Sh_decision | CI(UNCALIBRATED) | block | "
          "dep_len | dep<=block | advisory | "
          "CI(ADVISORY) | pnl | maxDD | Sh_per_bar(diag) | reason |",
          "|---|--:|--:|---|--:|--:|--:|--:|:--:|--:|--:|--:|--:|--:|--:|---|---|--:|--:|:--:|"
          ":--:|---|--:|--:|--:|---|"]
    for c in cells:
        # An uncertified cell names WHY, per D55(a) ("any region you cannot certify keeps the
        # UNCALIBRATED qualifier and says so per-cell") — and, per fix cycle 3 MAJOR 1(b),
        # distinguishes a missing stratum from a REFUSAL on measured dependence. Per D57, even
        # a `True` `ci_advisory` cell is NEVER band-readable on this leg — see `ci_adv_str` below.
        blk = _cell_block(c)
        dep_state = _dependence_status(c)
        dep_ok = dep_state == DEP_PASS
        if c.ci_advisory:
            ci_adv_str = (f"ADVISORY ONLY, NEVER BAND-READABLE (D57): "
                          f"[{c.ci_lo_ADVISORY:+.2f},{c.ci_hi_ADVISORY:+.2f}] "
                          f"(k={c.ci_calibration_k:.3f} from stratum "
                          f"n_dec={c.ci_calibration_source_n_dec}, block={blk}, measured "
                          f"dependence {c.measured_dependence_len:.0f} <= block {blk} — the "
                          f"tau guard is measured inert on this row's real data, so this pass is "
                          f"NOT evidence the condition holds)")
        elif dep_state == DEP_NO_CI:
            # No interval was bootstrapped for this cell at all (fewer than
            # MIN_DECISIONS_FOR_SHARPE decisions), so the dependence guard never ran on it. It is
            # NOT a dependence refusal and the banner does not count it as one (D57 item B2).
            ci_adv_str = f"UNCALIBRATED (no CI computed; n_dec={c.n_decisions})"
        elif dep_state == DEP_UNMEASURABLE:
            ci_adv_str = ("UNCALIBRATED — **REFUSED ON DEPENDENCE**: this cell's realized "
                          "per-decision dependence could NOT be measured (series too short, "
                          "degenerate, or autocorrelation never returns to the white-noise band). "
                          "Unknown dependence never certifies")
        elif dep_state == DEP_REFUSED:
            ci_adv_str = (f"UNCALIBRATED — **REFUSED ON DEPENDENCE**: measured dependence "
                          f"{c.measured_dependence_len:.0f} > block {blk}; every shipped `k` was "
                          f"certified at dep == block ONLY")
        else:
            ci_adv_str = (f"UNCALIBRATED (no certified stratum at block={blk}, "
                          f"n_dec={c.n_decisions})")
        dep_str = (f"{c.measured_dependence_len:.0f}"
                   if np.isfinite(c.measured_dependence_len) else "UNMEASURABLE")
        L.append(
            f"| {c.window} | {c.H} | {c.k} | {c.variant} | {c.tau:g} | {c.N} | {c.fee_bp:g} | "
            f"{c.adv_bp:g} | {c.status} | {c.n_bars} | {c.n_decisions} | "
            f"{c.decisions_per_year:.1f} | {c.mean_hold_bars:.1f} | {c.turnover_mean:.4f} | "
            f"{c.fill_rate:.3f} | {c.sharpe_decision:+.3f} | "
            f"[{c.ci_lo_UNCALIBRATED:+.2f},{c.ci_hi_UNCALIBRATED:+.2f}] | "
            f"{blk if blk is not None else 'n/a'} | {dep_str} | {dep_ok} | "
            f"{c.ci_advisory} | {ci_adv_str} | "
            f"{c.pnl:+.4f} | {c.maxdd:+.4f} | {c.sharpe_per_bar_diagnostic:+.2f} | {c.reason} |"
        )
    L += ["", "## Scope honesty",
          "This artifact never computes a pass/fail verdict; the `Sh_per_bar(diag)` column is "
          "the RETRACTED 809 methodology kept only as a visible contrast — never gate on it.",
          "",
          "**TASK_897 D57 (2026-08-22): the band's third conjunct, 'CI lo > 0', is "
          "UNAVAILABLE-because-uncalibratable on this leg.** The verdict this artifact supports "
          "reads on Sharpe >= 1.0 and PnL > 0 plus per-cell point estimates, never on the CI "
          "column. If a cell clears those two readable conjuncts at an attainable maker tier, "
          "the next step is fill-model validation via small live limit orders — the backtest CI "
          "was always advisory to that gate, not a substitute for it. **TASK_901** (a "
          "range-based dependence probe, validated out-of-family) upgrades this conjunct "
          "retroactively FROM THESE SAME ARTIFACTS.",
          "",
          "Three gaps this artifact used to carry unresolved are now addressed, and all three "
          "remain disclosed rather than papered over:",
          "",
          "1. **The advisory interval is computed PER-CELL, not globally, and keyed on the "
          "cell's REAL `(block, n_dec)` pairing** (banner at the top + the calibration table "
          "above) — a cell's `advisory` column says whether a CERTIFIED stratum existed AT ITS "
          "OWN BLOCK and the dependence guard did not refuse it; ITS `CI(ADVISORY)` COLUMN IS "
          "**NEVER** WHERE THE BAND'S 'CI lo > 0' CRITERION MAY BE READ, on this leg, "
          "`advisory=True` or not — see (3) below for why. A `False`/`UNCALIBRATED` cell names "
          "the (block, n_dec) region that was never certified and still cannot feed the band. "
          "Achieved coverage was re-measured at BORROWED (non-stratum) pairings, not only at the "
          "pinned strata, because a borrowed pairing is what a real cell actually reads.",
          "   **AND THE CERTIFICATION IS CONDITIONAL:** every `k` was measured at "
          "`dep == block` only — the easiest point of the calibration's own dependence sweep, "
          "not the worst — and past it the SHIPPED k's coverage falls to 0.52-0.55 (the "
          "magnitude that raised D55). `honest_estimator.dependence_length` ATTEMPTS to measure "
          "that condition per cell and cells that violate it, or whose dependence cannot be "
          f"measured, are REFUSED the advisory interval — {n_dep_refused} of the "
          f"{n_dep_measured} ok cells the guard ran on here (the remaining {n_dep_no_ci} of "
          f"{n_ok} carried too few decisions for an interval to exist, so the guard never ran "
          "and they are NOT counted as refusals — D57 item B2). "
          "The fallback is strictly better than the alternative: a withheld criterion costs a "
          "verdict, a criterion asserted under a violated (or unverifiable) assumption corrupts "
          "one. See (3) for why this measurement's PASS is not, by itself, evidence either.",
          "   **AND THE PER-STRATUM TABLE IS WEAKER THAN IT READS:** near the 0.93 bar, "
          "achieved-vs-failed at 300 validation trials is largely Monte-Carlo jitter "
          "(re-measurement at 6 named seeds puts four of the five demoted strata at or above the "
          "bar — two of them, 660 and 740, only barely — and 14 re-measured achieved strata have "
          "every 3-seed MEAN at or above it while 2 of their 42 individual seeds fall below). The "
          "demotions stand because they fail closed, not because a coverage hole was demonstrated "
          "at them.",
          "2. **The 877 alias fold IS applied, as a read-time layer owned by this runner** "
          "(Metadata above) — the 806/808/809 lineage is unchanged and re-verified every run "
          "(`test_the_lineage_really_has_no_alias_handling`); gate (iii) is met via the union "
          "read, not via a lineage edit.",
          "3. **THE DEPENDENCE GUARD ITSELF IS OUT-OF-FAMILY AND MEASURED INERT ON THIS ROW'S "
          "REAL DATA (TASK_897 D57, 2026-08-22)** — the third gap, closed here by NARROWING the "
          "claim rather than by re-measuring or replacing the guard. `dependence_length`'s tau "
          "equals dependence RANGE (what a block bootstrap needs) only inside the triangular-ACF "
          "family the calibration was measured against; a synthetic 4%-of-variance persistent "
          "component reads tau=1, certifies 100%, and covers only 0.805-0.376 "
          "(`calibration.MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE`). Every real per-decision "
          "series measured on this row (7 majors, OOS 2024-11 -> 2025-04, H in {1,5,20}, "
          "long-only and long-short) reads tau=1-2 against block=60 "
          "(`calibration.MEASURED_REAL_DATA_TAU_INERT_HALTED_CYCLE`, provenance-stamped, NOT "
          "re-run in this loop) — **THE GUARD IS THEREFORE EXPECTED NOT TO BITE HERE**, and this "
          "artifact's near-100% pass rate is NOT evidence the dep<=block condition holds. An "
          "empirical containment check on the real long/short H=1 series (97 disjoint "
          "1000-decision chunks) contained the grand Sharpe only 0.639 of the time while the "
          "guard certified ~100%. The guard is KEPT — it still fails closed on unmeasurable or "
          "excessive dependence — but its pass is downgraded, here and everywhere it ships, to "
          "'did not refuse', never 'condition measured true'. **TASK_901** (a range-based "
          "dependence probe — first-lag-in-band / variance-ratio / spectral-density-at-zero, "
          "validated out-of-family) is the row that replaces it, re-reading this leg's third "
          "conjunct retroactively FROM THESE SAME ARTIFACTS.",
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
            v = float(p)
        except ValueError:
            raise SystemExit(
                f"GATE (iv) REFUSAL: --attainable-maker-fee-bp entry {p!r} is not a number "
                f"(from {raw!r}). Pass a comma-separated bp list, e.g. "
                "--attainable-maker-fee-bp 8,16"
            ) from None
        # FIX 2: this list is written VERBATIM into the artifact's operator-facing metadata
        # disclosure line ("Attainable maker-fee tiers (operator input, gate iv): ..."), so a
        # fat-fingered tier must not silently corrupt that disclosure. Reject non-finite
        # (nan/inf) and negative entries with the same refuse-cleanly pattern as above. Zero IS
        # allowed: a 0bp maker tier is economically meaningful (a rebate-neutral venue), it is
        # not a parsing failure, and nothing downstream treats 0 as a sentinel for "missing".
        if not np.isfinite(v):
            raise SystemExit(
                f"GATE (iv) REFUSAL: --attainable-maker-fee-bp entry {p!r} is not finite "
                f"(from {raw!r}). Maker-fee tiers must be finite bp values, e.g. "
                "--attainable-maker-fee-bp 8,16"
            )
        if v < 0:
            raise SystemExit(
                f"GATE (iv) REFUSAL: --attainable-maker-fee-bp entry {p!r} is negative "
                f"(from {raw!r}). A maker fee cannot be negative bp (0 is allowed — a "
                "rebate-neutral venue — but a negative tier is not a fee, it's a rebate, and "
                "this harness does not model rebates). Pass a comma-separated non-negative bp "
                "list, e.g. --attainable-maker-fee-bp 8,16"
            )
        tiers.append(v)
    return tiers


DB_PATH_FORBIDDEN_SUBSTRINGS = ("?", "#")
# `//` is here for the same reason `file:` is (D59 fix cycle 3, MAJOR): sqlite's URI parser
# discards a leading `//<authority>` when the authority is empty or exactly `localhost`, so
# `//localhost/home/lukas/Trader/trading_data.db` and `/home/lukas/Trader/trading_data.db` are
# the SAME file to sqlite and DIFFERENT strings to `os.path.realpath` — the prefixed spelling
# cleared this gate and opened the protected DB (demonstrated against a decoy). A DB path in
# this harness is a FILESYSTEM PATH, never a URI, so the URI-only spelling is refused outright.
DB_PATH_FORBIDDEN_PREFIXES = ("file:", "//")


def assert_db_path_allowed(db_path: Optional[str], scheduled_real_data_run: bool) -> str:
    """Refuse to touch the protected 24GB source of record unless it was BOTH named explicitly
    AND acknowledged with `--i-am-the-scheduled-real-data-run`. Gate (iv)'s defence-in-depth
    precedent, applied to the data path: an accidental interactive invocation must not be able
    to reach `trading_data.db` BY OMISSION, which the old `--db-path` default allowed.

    D57 fix cycle 4, item A1 — THE GUARD WAS DEFEATABLE AND BOTH BELTS FAILED ON THE SAME
    STRING. `Path("trading_data.db?x=1")` is a DIFFERENT path object from `Path(
    "trading_data.db")`, so the protected-path comparison below passed it; and
    `alias_read_layer._connect_ro` builds `f"file:{db_path}?mode=ro"`, which for that input is
    `file:trading_data.db?x=1?mode=ro` — sqlite's URI parser reads ONE parameter,
    `x="1?mode=ro"`, leaves `mode` unset, and therefore opens the handle READ-WRITE-CREATE. A
    judge demonstrated a successful `CREATE TABLE` through that path. A DB path in this harness
    is a FILESYSTEM PATH, never a URI: `?`, `#`, a `file:` prefix and a leading `//` (see the
    cycle-3 note below) are refused outright here
    (belt 1) and `_connect_ro` percent-quotes whatever it is handed (belt 2), so neither belt
    depends on the other. Identity is then decided on `os.path.realpath` of the exact string
    sqlite will open, plus an inode/stat check (`os.path.samefile`) so a hardlink or bind-mount
    alias of the protected DB cannot slip past a textual comparison either.

    D59 fix cycle 3, MAJOR — THE SAME CLASS RECURRED ON A DIFFERENT URI SPELLING, AND AGAIN
    BOTH BELTS FAILED ON ONE STRING. With `PROTECTED_REAL_DB` monkeypatched to a decoy:

        assert_db_path_allowed('<decoy>',            scheduled=False) -> REFUSED   (correct)
        assert_db_path_allowed('//localhost<decoy>', scheduled=False) -> PASSED
           ...and `_connect_ro` on the returned string OPENED the protected decoy.

    `os.path.realpath('//localhost/…')` is not the protected realpath (it names a nonexistent
    `/localhost/…`), and the `os.path.samefile` fallback raises `OSError`, which the loop below
    swallows. So `--db-path //localhost/home/lukas/Trader/trading_data.db` WITHOUT
    `--i-am-the-scheduled-real-data-run` cleared this gate — read-only, so the source of record
    is never written, but it defeats the invariant this gate exists for: expensive real-data
    validation is SCHEDULED, never polled inside the factory loop. Two belts now, neither
    depending on the other: the `//` prefix is in `DB_PATH_FORBIDDEN_PREFIXES` (belt 1), and the
    identity check below runs on `alias_read_layer.normalize_db_path(raw)` — the exact string
    sqlite will open — so a future weakening of the prefix list cannot silently reopen this
    (belt 2)."""
    if db_path is None or not str(db_path).strip():
        raise SystemExit(
            "REFUSAL: --db-path is required and has no default (deliberately — it used to "
            "default to the real 24GB trading_data.db, so forgetting the flag reached the "
            "source of record). Pass fixtures/trading_fixture.db (or a scratch copy) for "
            "verification; the scheduled gpu-server run passes the real DB explicitly together "
            "with --i-am-the-scheduled-real-data-run."
        )
    raw = str(db_path)
    bad = [c for c in DB_PATH_FORBIDDEN_SUBSTRINGS if c in raw]
    bad_prefix = [p for p in DB_PATH_FORBIDDEN_PREFIXES if raw.lower().startswith(p)]
    if bad or bad_prefix:
        raise SystemExit(
            f"REFUSAL: --db-path {raw!r} is not a filesystem path. "
            + (f"It contains {', '.join(repr(c) for c in bad)}, which sqlite's URI parser treats "
               "as a query/fragment separator. " if bad else "")
            + (f"It starts with {bad_prefix[0]!r}, i.e. it is a URI, not a path. "
               if bad_prefix else "")
            + ("A leading `//` is a URI AUTHORITY to sqlite: `//localhost/x` and `/x` are the "
               "same file to the parser but different strings to realpath, which let the "
               "prefixed spelling clear this very gate. " if "//" in bad_prefix else "")
            + "This harness opens the DB through a `file:...?mode=ro` URI, so a `?`, `#`, "
              "`file:` or leading `//` in the path SHIFTS the `mode=ro` parameter or the path "
              "itself and silently yields a READ-WRITE-CREATE handle, or a handle on a file "
              "this gate never checked — including the protected 24GB trading_data.db. Pass a "
              "plain path with no URI syntax, e.g. fixtures/trading_fixture.db."
        )
    # FIX 3: `.resolve()`/`realpath` resolves a relative path against the process cwd, so a
    # caller invoked from a staging directory (e.g. the srv leg, which legitimately stages files
    # elsewhere) could type a RELATIVE path that resolves to the protected DB under the repo-root
    # interpretation without ever matching the cwd-relative one, or vice-versa. Check BOTH
    # interpretations and refuse if EITHER lands on the protected DB — strictly more refusing,
    # never less, and no honest path (one that is protected under neither interpretation) is
    # affected.
    protected = os.path.realpath(str(PROTECTED_REAL_DB))
    # BELT 2 (D59 cycle 3, MAJOR): compare identity on the string sqlite will actually OPEN, not
    # on the raw spelling. No-op for every path belt 1 lets through; it is what keeps this gate
    # correct if the prefix list is ever weakened.
    expanded = Path(alias_layer.normalize_db_path(raw)).expanduser()
    candidates = [os.path.realpath(str(expanded))]
    if not expanded.is_absolute():
        candidates.append(os.path.realpath(str(_REPO / expanded)))
    is_protected = any(c == protected for c in candidates)
    if not is_protected:
        # Stat identity, so a hardlink or bind-mount alias of the protected DB (a different
        # string, the same inode) is refused too. Silent when either side does not exist.
        for c in candidates:
            try:
                if os.path.samefile(c, protected):
                    is_protected = True
                    break
            except OSError:
                continue
    resolved_cwd = candidates[0]
    if is_protected and not scheduled_real_data_run:
        raise SystemExit(
            f"REFUSAL: --db-path {resolved_cwd} is the PROTECTED read-only source of record. "
            "TASK_897's scope note: expensive real-data validation is SCHEDULED (gpu-server, "
            "per D54), never polled from a factory or interactive session. If this really IS "
            "the scheduled run, say so explicitly: add --i-am-the-scheduled-real-data-run."
        )
    return str(db_path)


def _parse_window(flag: str, raw: str) -> Tuple[str, str]:
    """`START,END` -> (start, end), refusing cleanly instead of raising a bare unpack ValueError.

    D57 fix cycle 4, item B3 — THIS USED TO ACCEPT GARBAGE AND PRODUCE A GREEN-LOOKING ARTIFACT.
    `--window-a not-a-date,also-not` and `--window-a 2025-04-01,2024-11-01` (inverted) both
    parsed. `_window_slice` then selected ZERO decision rows with no refusal anywhere,
    `build_windows` still labelled the window, and the artifact reported
    `recorded: 144 (ok=144, dropped/failed=0)` with no GATE (ii) violation and exit 0 — the
    documented 863 green-looking-artifact shape, reachable by one operator typo on the srv leg.
    Both spellings are refused HERE, at parse time, before any data is touched: parsability
    against the SAME `%Y-%m-%d` reader the slicer uses (`probe._to_epoch`), and strict ordering.
    Exit-code semantics are deliberately untouched (a planner ruling, out of scope); the
    complementary half of this item is the artifact's own no-decisions statement in
    `write_artifact`."""
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) != 2 or not all(parts):
        raise SystemExit(
            f"REFUSAL: {flag} {raw!r} must be exactly 'START,END' (ISO dates), "
            f"e.g. {flag} 2024-11-01,2025-04-01"
        )
    epochs = []
    for p in parts:
        try:
            epochs.append(_to_epoch(p))
        except (ValueError, TypeError) as exc:
            raise SystemExit(
                f"REFUSAL: {flag} {raw!r} — {p!r} is not an ISO date (YYYY-MM-DD): {exc}. An "
                "unparseable bound is NOT a narrow window: the slicer would select zero "
                "decisions, every cell would still be recorded `ok`, and the artifact would "
                f"read green with no measurement in it. Pass {flag} 2024-11-01,2025-04-01."
            ) from None
    if epochs[0] >= epochs[1]:
        raise SystemExit(
            f"REFUSAL: {flag} {raw!r} — START must be strictly BEFORE END "
            f"({parts[0]!r} >= {parts[1]!r}). An inverted or empty window selects zero decisions "
            "silently, which renders as a complete, gate-clean artifact containing no "
            "measurement at all."
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

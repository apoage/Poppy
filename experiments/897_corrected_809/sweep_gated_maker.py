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
import sqlite3
import sys
import time
import traceback
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


class GateRefusal(ValueError):
    """A GOVERNED refusal raised by this harness's own gates — a designed precondition failure,
    NEVER an unhandled defect.

    TASK_902 fix cycle 2, BLOCKER. `main()` used to wrap `run_sweep` in a bare
    `except Exception` and render EVERY escape as the item-2b "CONTROLLED REFUSAL" artifact, in
    byte-identical prose. Reproduced by injecting a `KeyError`/`TypeError` into
    `probe.report_bar_frequency`: an ordinary code defect produced a file titled `REFUSED BEFORE
    ANY CELL WAS MEASURED` whose body says "This file is a CONTROLLED REFUSAL record" — the same
    output a missing DB produces. Before item 2b that same bug produced a loud uncaught traceback,
    which is strictly MORE diagnosable, so the fix had made the failure mode worse.
    `type(exc).__name__` was the only capture and `traceback.format_exc()` was never called
    anywhere, so the call stack was destroyed and logged nowhere.

    A refusal is now identified by TYPE, not by where it happened to be caught:
    `REFUSAL_EXCEPTIONS` below is the closed set of exception classes the first data touch can
    legitimately raise as a governed refusal. Anything else is a DEFECT and renders as one
    (`write_internal_error_artifact`), traceback included. This class exists so the harness's own
    deliberate gate refusals (which are semantically refusals but arrive as `ValueError`) are not
    forced to share a type with the numpy/stdlib `ValueError` a real bug raises.

    Subclasses `ValueError` so it is a strictly compatible narrowing: every existing caller and
    test that expects `ValueError` from `run_sweep`'s gate (iv) still catches it."""


# The closed set of exception types a governed refusal may arrive as. Everything else escaping
# `run_sweep` is an unhandled DEFECT and must never be rendered as a refusal.
#   * `GateRefusal`         — this harness's own gates (above).
#   * `sqlite3.DatabaseError`— the item-2b case: an absent, unreadable, or not-actually-a-database
#                      `--db-path` raises `sqlite3.OperationalError` ("unable to open database
#                      file") or `sqlite3.DatabaseError` ("file is not a database") at the first
#                      data touch. NOT the broader `sqlite3.Error`, which also covers
#                      `InterfaceError` (sqlite3 API misuse — a defect here, not a precondition).
#   * `OSError`      — the same class of precondition on the filesystem side (a `--db-path` that is
#                      a directory, permission denied, a dangling symlink). Chosen knowing it is
#                      the widest member of this set; it is here because a first-data-touch
#                      `OSError` is overwhelmingly a path precondition.
# `SystemExit`/`KeyboardInterrupt` are deliberately ABSENT: they are `BaseException`, not
# `Exception`, so the handler in `main()` cannot see them and both pass through unwrapped
# (verified by test_baseexceptions_pass_through_unwrapped_and_write_no_artifact).
REFUSAL_EXCEPTIONS: Tuple[type, ...] = (GateRefusal, sqlite3.DatabaseError, OSError)

# ...MINUS these, which MATCH the set above by inheritance but are DEFECTS, not refusals. In
# CPython's sqlite3, `ProgrammingError` is a SUBCLASS of `DatabaseError` (`ProgrammingError ->
# DatabaseError -> Error`), so narrowing the tuple alone could not exclude it — a misuse of the
# sqlite3 API inside this harness or the read layer (a closed cursor, the wrong number of binds)
# would otherwise be rendered to the operator as "your invocation was wrong". It is not: it is a
# bug here. The fail-safe direction for anything ambiguous is DEFECT (loud, with a traceback),
# never a routine-looking refusal, so the exclusion is applied explicitly by
# `_is_governed_refusal` rather than left to the exception hierarchy.
NOT_A_REFUSAL_DESPITE_MATCHING: Tuple[type, ...] = (sqlite3.ProgrammingError,)


def _is_governed_refusal(exc: BaseException) -> bool:
    """Is `exc` one of this harness's DESIGNED precondition failures (-> the CONTROLLED REFUSAL
    artifact), or an unhandled defect (-> the INTERNAL ERROR artifact)? A single predicate rather
    than a chain of `except` clauses, because the taxonomy is not expressible as a tuple: see
    `NOT_A_REFUSAL_DESPITE_MATCHING`. Pinned by
    test_gate_refusal_is_a_valueerror_subclass_so_existing_callers_are_unaffected."""
    return (isinstance(exc, REFUSAL_EXCEPTIONS)
            and not isinstance(exc, NOT_A_REFUSAL_DESPITE_MATCHING))


def _refusal_exception_names() -> str:
    """`REFUSAL_EXCEPTIONS` as operator-readable names for the artifacts. Module-qualified where
    the bare name would be ambiguous — `sqlite3.Error.__name__` is just `Error`, which tells a
    reader nothing."""
    out = []
    for t in REFUSAL_EXCEPTIONS:
        mod = getattr(t, "__module__", "")
        out.append(t.__qualname__ if mod in ("builtins", "", __name__)
                   else f"{mod}.{t.__qualname__}")
    return "`" + "`, `".join(out) + "`"


# =============================================================================
# ECHO CLIPPING (TASK_902 item 11) — defined HERE, above every refusal path in this file, so
# every one of them can reach it.
# =============================================================================
# A 900k-char `--db-path` produced a 900k-char refusal message (measured): every rejection path
# in this file echoes the operator's own string back so they can see what was rejected, and
# nothing bounded how long that echo could be. Clip while keeping the value NAMED (a real prefix,
# never a placeholder) and stating how much was dropped — TASK_888's precedent (1.6M chars ->
# <4k) for the same class of unbounded echo.
#
# FIX CYCLE 2 (TASK_902 MAJOR): the first pass applied this at only 4 of the >=11 echoing sites.
# `parse_fee_tiers` and `_parse_window` were still unbounded, and measured WORSE than the defect
# item 11 cites as its own precedent: a 900,000-char `--attainable-maker-fee-bp` produced a
# 1,800,148-char `SystemExit` (the value echoed TWICE per message) and a 900,000-char
# `--window-a` produced 2,700,344 chars (THREE times). Every echoing refusal in this module now
# goes through `_clip_echo`; the enumeration is in `test_every_echoing_refusal_site_is_clipped`.
ECHO_CLIP_LEN = 2000
# A traceback is machine-generated, multi-frame, and the thing an operator actually needs; it
# gets its own (much larger) budget, but it is still BOUNDED — a defect whose exception message
# is itself 900k chars would otherwise write a 900k-char "defect" artifact.
TRACEBACK_CLIP_LEN = 20000


def _clip_echo(s, limit: int = ECHO_CLIP_LEN) -> str:
    """Bound an operator-supplied value before it is echoed into a refusal message.

    CANNOT ITSELF RAISE (TASK_902 fix cycle 2 — a clipper that raises inside a refusal path turns
    a clean refusal into a traceback, which is the whole failure class this file is closing):
      * non-str input (None, bytes, int, list, Path, an arbitrary object) goes through `str()`;
      * an object whose `__str__` itself raises is rendered as an unprintable placeholder rather
        than being allowed to propagate;
      * lone surrogates (reachable from a real argv on Linux, where undecodable bytes arrive as
        `surrogateescape` code points) are escaped, because they are legal in a `str` but raise
        `UnicodeEncodeError` at the moment the message is WRITTEN to a file or to stderr — i.e.
        the clipper would look fine and the refusal would still die. The round-trip is a no-op
        for every well-formed string, including non-ASCII.
    Verified by `test_clip_echo_cannot_raise_on_hostile_input`."""
    try:
        s = str(s)
    except Exception:  # noqa: BLE001 — an object with a broken __str__ must not break a refusal.
        return f"<unprintable {type(s).__name__}>"
    try:
        s = s.encode("utf-8", "backslashreplace").decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — belt-and-braces; not reachable for a `str`.
        pass
    if len(s) <= limit:
        return s
    return f"{s[:limit]}...[CLIPPED: {len(s) - limit} more chars, {len(s)} total]"


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
    # TASK_902 item 2c — REALIZED book width, mean nonzero slots in `decide._decide_step`'s
    # target book across every bar of THIS cell's decision-engine run. `decide._decide_step`
    # (809 lineage, byte-frozen) opens with `if npr < (2*k if long_short else k): return tgt`
    # (the empty target) whenever the present universe is narrower than the grid's `k` demands,
    # so a cell can silently realize ZERO decisions while still taking the "ok" path (not a fit
    # failure, so it never takes the gate-(ii) drop path) — an empty measurement that used to
    # render indistinguishably from a real one. Separately, τ CAN BIND BEFORE k DOES:
    # `n_elig = max(1, int(round((1.0-tau)*npr)))` means a narrow universe realizes a book far
    # thinner than `k` implies even when npr clears the `2*k`/`k` floor above. `n_decisions` was
    # already recorded (existing field); this adds the OTHER half of "record realized n_decisions
    # and realized book width per cell" — a cell's `k` column is a REQUEST, not a report of what
    # was actually held, and a reader should not assume the book the run held from it alone.
    # Default NaN (never measured) rather than a required field, so every existing construction
    # site (this row must never move a number) keeps working unchanged.
    #
    # THE FIXTURE CANNOT COVER THE NONZERO-DECISION PATH OF THIS FIELD (stated here because this
    # is where a reader looks for its coverage). The fixture universe is 7 assets and never clears
    # `k=16`'s `2k` floor for `long_short` (nor the `k` floor for `long_only`), so NO cell on the
    # fixture ever realizes `n_decisions > 0`: every fixture cell takes `decide._decide_step`'s
    # empty-target early return, and `realized_book_width_mean` is therefore measured only on
    # all-zero books there. The gate-(ii) fixture rerun (the 144-cell SMOKE grid, k=16) is
    # consequently SILENT on this field for nonzero-decision cells. That path is covered ONLY by
    # `test_realized_book_width_is_recorded_and_the_instrumentation_cleans_up_after_itself`, which
    # reaches it by dropping to `grid_k=(1,)` AND stubbing `decide._chunked_predict` to return
    # reliably-positive predictions — i.e. a partly-stubbed unit test, not the shipped grid. A
    # fixture-level gate-(ii) pass must NOT be over-read as covering the nonzero-decision case;
    # only the real 64-asset leg exercises it as shipped.
    realized_book_width_mean: float = float("nan")


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
    # TASK_902 fix cycle 2, MINOR 1 — EVERYTHING PAST THIS POINT IS KEYWORD-ONLY, and
    # `scheduled_real_data_run` is why. It is the acknowledgement that unlocks the PROTECTED 24GB
    # source of record (D57: the flag must reflect explicit operator intent, never an accident),
    # and it used to be the 16th positional-or-keyword parameter of this function. This very
    # hardening row APPENDED it there — which is exactly the mechanism by which a future
    # parameter insertion anywhere above would silently shift a positional argument into it and
    # grant protected-DB access nobody asked for. Keyword-only removes the whole class: the flag
    # can now only ever be set by NAMING it. `seed`/`n_boot`/`verify_inherited_ic` are inside the
    # barrier too, so a future trailing insert lands in keyword-only territory by default rather
    # than extending the positional tail again. Every call site in the tree already passes these
    # by keyword (audited), so this narrows the API without moving a number.
    *,
    seed: int = 897,
    n_boot: int = 2000,
    verify_inherited_ic: bool = True,
    scheduled_real_data_run: bool = False,
) -> Tuple[List[CellRecord], Dict]:
    """Run the gated_maker sweep and return `(cells, meta)`. See the module docstring for what
    this instrument is and is not.

    RAISES: `GateRefusal` (a ValueError subclass) for a governed gate refusal; `SystemExit` from
    `assert_db_path_allowed` for a protected-DB / URI-syntax refusal; `sqlite3.Error`/`OSError`
    from the first data touch for an absent or unreadable `--db-path`. `main()` renders those as
    a CONTROLLED REFUSAL artifact. Anything else escaping this function is an unhandled DEFECT
    and `main()` renders it as one, with the traceback — never as a refusal.

    NOT THREAD-SAFE / NOT REENTRANT — DO NOT RUN TWO SWEEPS CONCURRENTLY IN ONE PROCESS
    (TASK_902 fix cycle 2, MINOR 3; pre-existing in kind, documented rather than redesigned —
    a locking redesign is explicitly out of this row's scope). This function and
    `_run_sweep_inner` reuse the 806/808/809 lineage VERBATIM (never reimplemented), and the only
    way to do that at a non-canonical scale is to scope-patch MODULE-GLOBAL attributes for the
    duration of a call, restoring each in a `finally`. There are five such patches:

        `decide.CFG`                     (run_sweep, around the whole inner call)
        `probe.sqlite3`                  (_run_sweep_inner, x2: bar-frequency + pin-era panel)
        `backtest.SIGNAL_806_IC_FULL`    (_run_sweep_inner, per-H sweep-panel IC measurement)
        `decide._decide_step`            (_run_sweep_inner, per-cell book-width instrumentation)

    None of them is thread-safe, and none is reentrant: two concurrent `run_sweep` calls in one
    process (the 898-class reuse surface this instrument is explicitly pointed at) would
    interleave save/restore pairs, so one call can restore a value the other had already
    replaced — leaving a module attribute permanently mispointed for the rest of the process
    (e.g. `decide.CFG` stuck on a finished sweep's Config, or `probe.sqlite3` stuck on the
    read-only shim, or — the dangerous direction — restored to the bare READ-WRITE `sqlite3`
    while a sweep is still reading the protected DB). Run sweeps SEQUENTIALLY in-process, or in
    separate processes. Each patch site carries the same warning locally."""
    # --- GATE (iv): the attainable maker-fee tier set is a REQUIRED operator input. -------
    # Never invented, guessed, or defaulted — refuse before touching any data.
    if attainable_maker_fee_bp is None or len(list(attainable_maker_fee_bp)) == 0:
        raise GateRefusal(
            "GATE (iv) REFUSAL: --attainable-maker-fee-bp is required and was not provided. "
            "This is the OPERATOR's Kraken-reachable maker-fee tier set (Fork-C venue ruling: "
            "zero-maker-fee is out of reach) — this harness will not invent, guess, or default "
            "it. Pass e.g. --attainable-maker-fee-bp 8,16 and re-run."
        )
    # A programmatic caller (the CLI is already covered by `parse_fee_tiers`) can hand this a
    # non-numeric tier. That is a REFUSAL, not a defect in the instrument, so raise it as one
    # rather than letting a bare `float()` ValueError/TypeError escape and be rendered as an
    # internal error by `main()`'s defect handler.
    try:
        attainable_maker_fee_bp = [float(x) for x in attainable_maker_fee_bp]
    except (TypeError, ValueError) as exc:
        raise GateRefusal(
            "GATE (iv) REFUSAL: --attainable-maker-fee-bp contains a non-numeric tier "
            f"({_clip_echo(exc)}); the tier set is written verbatim into the artifact's "
            "operator-facing disclosure line, so it must be a list of finite bp numbers."
        ) from None
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
        # item 11 (fix cycle 2): both echoes are caller-supplied SEQUENCES, so both are unbounded
        # in length through the public API — clipped like every other echo in this file.
        raise GateRefusal(
            f"GATE (iv) REFUSAL: attainable maker-fee tier(s) {_clip_echo(missing_tiers)} bp are "
            f"NOT in this run's pre-registered fee grid {_clip_echo(sorted(set(grid_fee_set)))} "
            f"bp, so NOT ONE of them "
            "would be measured. The artifact would still call the tier set 'the Kraken-reachable "
            "subset of the fee grid', and a verdict read off a CHEAPER grid cell than any "
            "attainable tier is strictly better than reality — optimistic, which is the one "
            "direction this harness must never fail in. Either pass tiers the grid measures "
            f"(available: {sorted(set(grid_fee_set))} bp) or change the pre-registered grid "
            "deliberately (a planner decision, not a flag) and re-run."
        )

    # TASK_902 item 9 — the protected-DB gate used to be bound ONLY in `main()`; `run_sweep` (and
    # `_run_sweep_inner`) never called `assert_db_path_allowed` themselves, so a PROGRAMMATIC
    # caller of the public API — the exact reuse surface D60 already points the 898-class decay
    # work at — could read `trading_data.db` without ever passing the acknowledgement flag. Bind
    # the guard here too, at the entry to the actual data-touch layer, so `main()`'s own
    # pre-check (kept, for a clean CLI-level refusal before any argument-derived work happens) is
    # defence-in-depth rather than the ONLY belt — the same "two belts, neither depending on the
    # other" pattern this module already uses for the URI-syntax check. This also closes item 10
    # (TOCTOU) for every caller: `assert_db_path_allowed` now returns the RESOLVED path, which is
    # what `cfg.db_path` becomes below and therefore what every downstream open actually uses.
    db_path = assert_db_path_allowed(db_path, scheduled_real_data_run)

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
    # NOT REENTRANT (MINOR 3): this is a MODULE-GLOBAL rebind, so a second concurrent `run_sweep`
    # in this process would interleave with this save/restore pair and could leave `decide.CFG`
    # permanently pointing at a finished sweep's Config. See `run_sweep`'s docstring.
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
    # NOT REENTRANT (MINOR 3): module-global rebind. Two concurrent sweeps could interleave these
    # save/restore pairs and leave `probe.sqlite3` restored to the BARE read-write module while
    # another sweep is still reading — the dangerous direction, on the protected DB. Sequential
    # in-process use only; see `run_sweep`'s docstring.
    _orig_probe_sqlite3 = probe.sqlite3
    probe.sqlite3 = _ReadOnlySqlite3ForProbe()
    try:
        bar_sec = probe.report_bar_frequency(cfg)
    finally:
        probe.sqlite3 = _orig_probe_sqlite3

    # =========================================================================================
    # TASK_897 D60 — PIN-ERA INHERITED-IC GUARD (re-based to the pins' own claim).
    # -----------------------------------------------------------------------------------------
    # D59's guard asserted 806's June-era full-slice test-IC pins against THIS harness's SWEEP
    # panel, whose test slice runs a year past the pins' basis (test_end 2025-04-01) — measured
    # (planner, srv, D60) to be the ENTIRE cause of the srv leg #1 ok=0 (chunked == unchunked
    # exactly at every H; the pins reproduce to -4.5e-5 on a pin-era panel built from TODAY's
    # table). The guard's actual claim is LINEAGE FIDELITY — does this harness reproduce 806's
    # audited signal on 806's OWN basis — so it is re-based to a SEPARATE pin-era panel, built
    # and checked ONCE here, BEFORE the sweep panel exists. The sweep-panel IC becomes a
    # disclosed MEASUREMENT (the decay signal), never gated — see the per-H loop below (item 3).
    #
    # `cfg_pin` is `probe.Config`'s OWN split defaults — train_start=2023-05-16,
    # val_start=2024-06-01, test_start=2024-11-01, test_end=2025-04-01 (verified: these ARE the
    # pins' basis) — built from ONLY `db_path`, deliberately NOT from this run's --train-start /
    # --val-start / --window-a / --window-b overrides, so an operator override of any of those
    # can never move the guard off the pins' own basis.
    #
    # The panel is the EXACT-SPELLING 806 read (`load_universe_trainlocked` + `build_aligned_
    # ohlcv`), NOT the D56(a) alias union below — the union is a measured +0.003-max,
    # OPPOSITE-SIGN effect the pins never contained, and D60 specifies exact-spelling.
    #
    # Both probe calls open the DB via a bare `sqlite3.connect` (the exact D59 hazard already
    # closed on `report_bar_frequency` above — READ-WRITE-CREATE), so this whole read is wrapped
    # in the SAME scope patch, restored in `finally`. Re-introducing an RW handle here would be a
    # BLOCKER-class regression on the protected 24GB DB.
    inherited_ic_pin_era: Dict[int, Dict] = {}
    if verify_inherited_ic:
        cfg_pin = probe.Config(db_path=db_path)
        pin_panel_error: Optional[str] = None
        ohlcv_pin = close_pin = grid_ts_pin = None
        # NOT REENTRANT (MINOR 3) — same module-global hazard as the patch above.
        _orig_probe_sqlite3_pin = probe.sqlite3
        probe.sqlite3 = _ReadOnlySqlite3ForProbe()
        try:
            _symbols_pin, asset_ids_pin, _ranking_pin = probe.load_universe_trainlocked(cfg_pin)
            ohlcv_pin, grid_ts_pin = probe.build_aligned_ohlcv(cfg_pin, asset_ids_pin, bar_sec)
            close_pin = np.ascontiguousarray(ohlcv_pin[:, :, 3])
            del ohlcv_pin
            gc.collect()
        except Exception as exc:  # noqa: BLE001 — gate (ii): the PANEL BUILD itself can fail;
            # every H must still be recorded (dropped), never silently omitted.
            pin_panel_error = f"pin-era PANEL BUILD failed: {type(exc).__name__}: {exc}"
            # CLEANUP SYMMETRY with the success path's `del ohlcv_pin; gc.collect()`: an
            # exception raised AFTER the `ohlcv_pin, grid_ts_pin = ...` bind but BEFORE the
            # `del` (e.g. anything the `[:, :, 3]` slice can raise) would otherwise leave the
            # full pin-era panel alive in THIS frame's locals for the rest of the function —
            # straight through the much larger sweep-panel build immediately below, the worst
            # moment for the scheduled leg's `safe_run --mem 64G` budget. Rebinding the frame
            # locals is what actually frees them (the caught exception's traceback holds THIS
            # frame), so drop them here rather than relying on function exit.
            ohlcv_pin = close_pin = grid_ts_pin = None
            gc.collect()
        finally:
            probe.sqlite3 = _orig_probe_sqlite3_pin

        if pin_panel_error is not None:
            for H in grid_h:
                inherited_ic_pin_era[H] = {"ic": None, "status": "failed",
                                            "reason": pin_panel_error}
        else:
            _orig_decide_cfg_pin = decide.CFG  # == the sweep cfg here (run_sweep already
            decide.CFG = cfg_pin               # patched decide.CFG to `cfg` before this call)
            try:
                for H in grid_h:
                    try:
                        # SECOND ROOT CAUSE of the same vacuity class TASK_902 item 2 names
                        # (that item scopes only the NaN cause; this one is independent of it).
                        # `decide.inherited_signal_check_chunked` reads its target as
                        # `tgt = backtest.SIGNAL_806_IC_FULL.get(H)` and compares ONLY
                        # `if tgt is not None and ...` — so for ANY H outside the pinned set
                        # {1, 5, 20} the comparison short-circuits and the function returns a
                        # real, finite, NEVER-COMPARED IC. Unguarded, that lands as
                        # {"ic": <real number>, "status": "pass"} and the artifact calls it the
                        # lineage-fidelity proof — a vacuous pass, the exact failure mode the
                        # `np.isfinite` rider below closes for the OTHER cause. Not reachable
                        # from today's CLI (GRID_H / SMOKE_GRID_H are subsets of the pinned set)
                        # but fully reachable through the public `run_sweep(grid_h=...)` API,
                        # which is the reuse surface D60 points the 898-class decay work at.
                        # Read the pins the SAME way the lineage function does — the live module
                        # attribute, at the same moment — so the two can never disagree.
                        tgt_pin = backtest.SIGNAL_806_IC_FULL.get(H)
                        if tgt_pin is None:
                            raise AssertionError(
                                f"pin-era inherited-IC guard H={H}: NOT-EVALUABLE — H={H} "
                                "carries no 806 IC pin (pinned H: "
                                f"{sorted(backtest.SIGNAL_806_IC_FULL)}), so "
                                "`decide.inherited_signal_check_chunked` would return an IC it "
                                "never compared against anything and LINEAGE FIDELITY WAS NOT "
                                "ESTABLISHED for this H. An uncompared IC is a GUARD FAILURE, "
                                "never a pass (TASK_897 D60 rider — second root cause of "
                                "TASK_902 item 2's vacuity class, which scopes only the "
                                "non-finite cause).")
                        w_pin, mu_pin, sd_pin, _alpha_pin = decide.fit_xs_signal_chunked(
                            close_pin, grid_ts_pin, H)
                        ic_pin = decide.inherited_signal_check_chunked(
                            close_pin, grid_ts_pin, H, w_pin, mu_pin, sd_pin)
                        # D60 rider on TASK_902's vacuous-on-NaN item: a non-finite IC must never
                        # read as a pass (`nan > tol` is False, so an unguarded compare would let
                        # a NaN slip straight through as "the guard did not refuse").
                        if not np.isfinite(ic_pin):
                            raise AssertionError(
                                f"pin-era inherited-IC guard H={H}: IC is non-finite "
                                f"({ic_pin!r}) — a non-finite IC is a GUARD FAILURE, never a "
                                "pass (TASK_897 D60 rider on TASK_902's vacuous-on-NaN item).")
                        inherited_ic_pin_era[H] = {"ic": float(ic_pin), "status": "pass"}
                    except Exception as exc:  # noqa: BLE001 — gate (ii): drop, never crash
                        inherited_ic_pin_era[H] = {
                            "ic": None, "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
            finally:
                decide.CFG = _orig_decide_cfg_pin
            del close_pin, grid_ts_pin
            gc.collect()
    # else: --skip-inherited-ic-check — skip this WHOLE block (no pin-era panel build, no cost);
    # the fixture path requires this since the pins are real-data magic numbers (item 5).

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
        # TASK_897 D60 — the GATED basis (pin-era) and its inputs, recorded so the artifact can
        # show the guard's own claim (pins + tol) beside its per-H outcome.
        "inherited_ic_pin_era": inherited_ic_pin_era,
        "inherited_ic_pins": dict(backtest.SIGNAL_806_IC_FULL),
        "inherited_ic_tol": backtest.SIGNAL_IC_TOL,
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

        # TASK_897 D60, item 2 — the pin-era guard already ran ABOVE, once, before the sweep
        # panel was even built. A failed or absent (never-ran) result for this H drops every
        # cell for this H here — gate (ii): drop, never crash, never silently omit — mirroring
        # the fit/predict except-drop shape one level down.
        if verify_inherited_ic:
            pin_result = inherited_ic_pin_era.get(H)
            if pin_result is None or pin_result.get("status") != "pass":
                reason = ("pin-era inherited-IC GUARD FAILED (TASK_897 D60 basis: "
                          "probe.Config defaults, test_end=2025-04-01, tol "
                          f"{backtest.SIGNAL_IC_TOL} — NOT the sweep panel): "
                          + (pin_result or {}).get(
                              "reason", "pin-era guard never ran for this H"))
                for (wname, wstart, wend) in windows:
                    cells.extend(_empty_cells(wname, H, grid_k, grid_variant, grid_tau, grid_n,
                                              grid_fee_bp, grid_adv_bp, "dropped", reason))
                print(f"[897-sweep] H={H} DROPPED ALL CELLS (pin-era guard): {reason}",
                      flush=True)
                continue

        try:
            _, _, test = probe.make_splits(cfg, grid_ts, H)
            w, mu, sd, alpha = decide.fit_xs_signal_chunked(close, grid_ts, H)
            # TASK_897 D60, item 3 — sweep-panel IC is a disclosed MEASUREMENT ONLY, never
            # gated. Computed through the SAME lineage function (no reimplementation);
            # non-asserting: scope-patch `backtest.SIGNAL_806_IC_FULL` to `{}` for this one call
            # so `decide.inherited_signal_check_chunked`'s own `tgt = backtest.
            # SIGNAL_806_IC_FULL.get(H)` misses and the IC returns unchecked. REBINDS the module
            # attribute to a NEW dict object — never mutates the shipped dict in place (the D59
            # lesson) — and is restored in `finally` regardless of outcome.
            # NOT REENTRANT (MINOR 3): rebinding `backtest.SIGNAL_806_IC_FULL` to `{}` is a
            # module-global change. A concurrent sweep in this process could observe the EMPTY
            # dict during this window — and the pin-era guard reads exactly this attribute, so it
            # would see "no pin for this H" and drop cells for a reason that is not true. The
            # `tgt_pin is None` guard makes that fail CLOSED (drop, never a vacuous pass), which
            # is why this is a correctness-of-claims hazard rather than a fail-open one.
            # Sequential in-process use only; see `run_sweep`'s docstring.
            ic = None
            if verify_inherited_ic:
                _orig_pins = backtest.SIGNAL_806_IC_FULL
                backtest.SIGNAL_806_IC_FULL = {}
                try:
                    ic = decide.inherited_signal_check_chunked(close, grid_ts, H, w, mu, sd)
                finally:
                    backtest.SIGNAL_806_IC_FULL = _orig_pins
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

        meta.setdefault("signal_ic_sweep_panel", {})[H] = ic
        print(f"[897-sweep] H={H} alpha={alpha:g} ic={ic} dec_rows={len(dec_rows)} "
              f"fit {time.time()-h_t0:.1f}s", flush=True)

        for (wname, wstart, wend) in windows:
            wmask = _window_slice(dec_rows, grid_ts, wstart, wend)
            for k, variant, tau, N in itertools.product(grid_k, grid_variant, grid_tau, grid_n):
                try:
                    dr_w = dec_rows[wmask]
                    pred_w = pred[wmask]
                    pres_w = present[wmask]
                    # TASK_902 item 2c — instrument the REALIZED book width (nonzero slots in
                    # `decide._decide_step`'s per-bar target) without editing `decide.py` (809
                    # lineage, byte-frozen): scope-patch the module attribute for the duration of
                    # this ONE `_run_decision` call, the SAME rebind-the-name pattern this file
                    # already uses for `decide.CFG`/`probe.sqlite3` — the original function still
                    # runs, unmodified, for every bar; this only OBSERVES its return value. Never
                    # changes `gross`/`turn`/`fill` (this cell's actual numbers).
                    # NOT REENTRANT (MINOR 3): module-global rebind, and the one with the widest
                    # blast radius — two concurrent sweeps interleaving these save/restore pairs
                    # can leave `decide._decide_step` permanently pointing at a wrapper closed
                    # over a FINISHED cell's `_book_widths` list, so every later cell in the other
                    # sweep would append its widths into the wrong list (a wrong
                    # `realized_book_width_mean`, i.e. a corrupted rendered claim) — or, if the
                    # restore lands the other way, stack wrappers. Sequential in-process use only;
                    # see `run_sweep`'s docstring.
                    _orig_decide_step = decide._decide_step
                    _book_widths: List[int] = []

                    def _instrumented_decide_step(*_a, _orig=_orig_decide_step,
                                                   _widths=_book_widths, **_k):
                        _tgt = _orig(*_a, **_k)
                        _widths.append(int(np.count_nonzero(_tgt)))
                        return _tgt

                    decide._decide_step = _instrumented_decide_step
                    try:
                        gross, turn, fill = decide._run_decision(
                            close, high, low, pred_w, pres_w, dr_w, k, variant, tau, N, "maker"
                        )
                    finally:
                        decide._decide_step = _orig_decide_step
                    realized_book_width_mean = (float(np.mean(_book_widths))
                                                 if _book_widths else float("nan"))
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
                            realized_book_width_mean=realized_book_width_mean,
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


def _grid_coverage_label(meta: Dict) -> str:
    """TASK_902 item 12 — the artifact must state WHICH pre-registered grid ran: the SMOKE grid
    (fixture/synthetic-scale verification only, normally 144 cells) or the FULL grid (the real
    pre-registered sweep, normally 864 cells) — today it states neither, and a recorded/expected
    count must never be readable as full coverage on its own. Derived from `meta['grid']`
    (recorded regardless of caller) by shape comparison against the module's own pre-registered
    `GRID_*`/`SMOKE_GRID_*` H/k/variant/tau/N constants, so it stays correct even for a custom
    grid passed directly to `run_sweep` (e.g. a test) — those cases name themselves CUSTOM rather
    than guessing. Cites THIS run's own `n_cells_expected` rather than a hardcoded 144/864 —
    those counts also fold in `windows`/`fee_bp`/`adv_bp`, which this shape check does not
    compare, so a hardcoded number could be wrong for a caller that varies only those."""
    g = meta.get("grid") or {}
    shape = tuple(tuple(g.get(k, [])) for k in ("H", "k", "variant", "tau", "N"))
    smoke_shape = (SMOKE_GRID_H, SMOKE_GRID_K, SMOKE_GRID_VARIANT, SMOKE_GRID_TAU, SMOKE_GRID_N)
    full_shape = (GRID_H, GRID_K, GRID_VARIANT, GRID_TAU, GRID_N)
    n = meta.get("n_cells_expected", "?")
    if shape == smoke_shape:
        return (f"**SMOKE** grid — the fixture/synthetic-scale pre-registered grid "
                f"(`SMOKE_GRID_*`; normally 144 cells with the default windows/fee/adv grid, "
                f"{n} cells expected on THIS run), NOT the full pre-registered grid.")
    if shape == full_shape:
        return (f"**FULL** grid — the real-data pre-registered sweep grid (`GRID_*`; normally "
                f"864 cells with the default windows/fee/adv grid, {n} cells expected on THIS "
                "run).")
    return (f"**CUSTOM** grid — neither the SMOKE nor the FULL pre-registered H/k/variant/tau/N "
            f"shape ({n} cells expected on THIS run; H={g.get('H')} k={g.get('k')} "
            f"variant={g.get('variant')} tau={g.get('tau')} N={g.get('N')}).")


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
    # TASK_902 item 4 — the "Scope honesty" section below used to assert "this artifact's
    # near-100% pass rate is NOT evidence ..." UNCONDITIONALLY, even on a run (e.g. the empty
    # fixture run) that recorded no measured pass rate at all — a claim the run itself does not
    # back (errs conservative: it understates how bad the guard's own numbers might be here, but
    # it is still a specific numeric claim this run cannot support). State the REAL pass rate,
    # computed from the SAME `n_dep_pass`/`n_dep_measured` the banner box above already counts,
    # never a hardcoded "near-100%".
    if n_dep_measured > 0:
        _scope_pass_rate_clause = (
            f"this artifact's {n_dep_pass}/{n_dep_measured} pass rate on the ok cells the guard "
            "ran on"
        )
    else:
        _scope_pass_rate_clause = (
            "this artifact recorded NO measured pass rate at all (the guard ran on zero cells)"
        )
    # A run in which NOT ONE ok cell carried a decision contains no measurement at all. Stated
    # (never graded — this instrument records, the planner rules) and never as an error: on the
    # fixture, tiny windows legitimately yield zero decisions.
    n_ok_with_decisions = sum(1 for c in cells
                              if c.status == "ok" and c.n_decisions and c.n_decisions > 0)
    no_measurement = bool(n_ok) and n_ok_with_decisions == 0
    # TASK_902 item 6 — SYMMETRY FOR A 100%-FAILED RUN. A run where every `ok` cell recorded zero
    # decisions gets the large banner below (`no_measurement`); a run where every cell's decision
    # engine or an upstream guard FAILED before any cell could even reach "ok" status used to get
    # NONE — the only trace was the per-cell `reason` column, easy to miss on a table this wide.
    # Both are "this artifact holds no measurement", so both get a top-level statement. Gate (ii)
    # is untouched by either: a 100%-dropped run can still satisfy `n_cells_recorded ==
    # n_cells_expected` (every cell was accounted for, just none of them computed).
    n_cells_total = len(cells)
    all_cells_failed = n_cells_total > 0 and n_ok == 0
    # TASK_902 fix cycle 2, MAJOR — A ZERO-CELL RUN USED TO READ AS HEALTHY FROM TOP TO BOTTOM.
    # `run_sweep(grid_k=())` (API-reachable; not CLI-reachable — the same surface item 2's second
    # root cause lived on) produces `n_cells_expected == n_cells_recorded == n_cells_ok == 0` and
    # `len(cells) == 0`. Every existing loud path missed it: the gate-(ii) completeness check is
    # satisfied TRIVIALLY by `0 == 0`, `no_measurement` requires `bool(n_ok)` (False),
    # `all_cells_failed` requires `n_cells_total > 0` (False), and `exit_code_for` returned 0. So
    # the artifact rendered `expected: 0 recorded: 0 (ok=0, dropped/failed=0)` with no banner and
    # no warning — a degenerate run indistinguishable from a healthy one. This is precisely the
    # "neither banner fires but the run is degenerate anyway" case item 6's doctrine asks about.
    # It gets its OWN banner (this is not an "every cell failed" run — there were no cells to
    # fail) and a NON-ZERO exit (`exit_code_for`, which documents the D58 reasoning: cells
    # recorded + none measured = honest empty = exit 0; NO cells at all = the instrument did
    # nothing = non-zero).
    zero_cell_run = n_cells_total == 0 or int(meta.get("n_cells_expected") or 0) <= 0
    if zero_cell_run:
        L += ["> ## :warning: ZERO CELLS — **THE GRID ITSELF WAS EMPTY; THIS RUN MEASURED "
              "NOTHING AND PROMISED NOTHING**",
              ">",
              f"> This run recorded {n_cells_total} cells against "
              f"{meta.get('n_cells_expected')} expected: at least one grid dimension was EMPTY "
              "(`H`/`k`/`variant`/`tau`/`N`/`windows`/`fee_bp`/`adv_bp` — see the Grid line in "
              "the metadata below), so the product over the grid is zero and there was never a "
              "cell to measure. **This is NOT an honest empty result** (that is a run where "
              "every cell the grid promised WAS recorded and none of them measured anything — "
              "the normal fixture state, which exits 0). Here the grid promised nothing, so "
              "**gate (ii) has nothing to check: its `recorded == expected` test is satisfied "
              "trivially by `0 == 0` and provides no assurance whatsoever on this run.** The "
              "instrument failed to do anything, which is a different thing from measuring "
              "nothing, so the process exits NON-ZERO. Fix the grid and re-run; read no "
              "conclusion of any kind off this file.", ""]
    if all_cells_failed:
        n_dropped = sum(1 for c in cells if c.status == "dropped")
        n_failed = sum(1 for c in cells if c.status == "failed")
        L += ["> ## :warning: EVERY CELL FAILED — **THIS ARTIFACT CONTAINS NO `ok` CELL AT ALL**",
              ">",
              f"> All {n_cells_total} recorded cells are dropped/failed ({n_dropped} `dropped`, "
              f"{n_failed} `failed`) — not one cell reached `status=\"ok\"`, so this artifact "
              "holds no measurement of any kind, not even a zero-decision one. This is a "
              "STATEMENT ABOUT THIS RUN, not a verdict and not, by itself, an error: gate (ii) "
              "only requires `n_cells_recorded == n_cells_expected` (every cell accounted for), "
              "which a 100%-failed run can still satisfy. Check the per-cell `reason` column and "
              "the metadata above (in particular the inherited-IC guard result) before reading "
              "anything else.", ""]
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
          f"- **Inherited-IC GUARD (pin-era basis, TASK_897 D60 — the lineage-fidelity proof; "
          f"GATED; tol UNCHANGED at {meta.get('inherited_ic_tol')}):** basis = `probe.Config`'s "
          "own split defaults, exact-spelling read (train_start=2023-05-16, "
          "val_start=2024-06-01, test_start=2024-11-01, test_end=2025-04-01 — the pins' own "
          "basis, NOT this run's --train-start/--val-start/--window-a/--window-b, which can "
          f"never move it). Pins: {meta.get('inherited_ic_pins', {})}. Per-H result (empty if "
          f"`--skip-inherited-ic-check`): {meta.get('inherited_ic_pin_era', {})}.",
          "- **Signal IC, SWEEP-PANEL basis (disclosed MEASUREMENT, TASK_897 D60 — NEVER gated "
          "against the June pins on this basis):** per-H chunked-fit IC over THIS run's own "
          "test span, computed through the SAME lineage function "
          "(`decide.inherited_signal_check_chunked`) with the pins scope-patched out for that "
          f"one call: {meta.get('signal_ic_sweep_panel', {})} (None per H if "
          "`--skip-inherited-ic-check` — the dict is still keyed by every H that reached the "
          "fit, with a None value; it is the pin-era GUARD dict above that is empty under that "
          "flag). A LOWER value here than the pin-era guard's own is the "
          "signal DECAY the per-window cells measure properly — not a lineage-fidelity failure.",
          f"- **Estimator:** {meta['estimator']}",
          f"- **Maker-fill model (stated inline, per cell, per TASK_897 scope 3):** "
          f"{meta['maker_fill_model']}",
          f"- **Grid coverage (TASK_902 item 12 — the cell count below must never be readable "
          f"as full coverage on its own):** {_grid_coverage_label(meta)}",
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
    L += ["", "> " + cal.DEPENDENCE_REFUSAL_OC_GRID_GAP, ""]
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
          "batch. A row below target would mean the region around it is NOT certified. "
          "TASK_902 item 3: this table renders its OWN `reason` column now — pairing 639 CANNOT "
          "REPRODUCE by construction (stratum 620 was demoted by this row) and the `False` in "
          "its `meets target` cell used to be the only signal of that; the per-stratum table "
          "above already carried a `reason` column, this one did not:",
          "", "  | n_dec (borrower) | block | borrowed k | from stratum | cov(overlap) | "
          "cov(iid) | trials x boot | meets target | reason |",
          "  |--:|--:|--:|--:|--:|--:|--:|:--:|---|"]
    for nd in sorted(cal.MEASURED_BORROWED_PAIRING_COVERAGE):
        e = cal.MEASURED_BORROWED_PAIRING_COVERAGE[nd]
        L.append(f"  | {nd} | {e.get('block')} | {e.get('borrowed_k')!r} | "
                 f"{e.get('borrowed_from_n_dec')!r} | {e.get('coverage_overlapping')!r} | "
                 f"{e.get('coverage_iid')!r} | {e.get('n_trials')}x{e.get('n_boot')} | "
                 f"{e.get('meets_target')} | {e.get('reason') or ''} |")
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
          "dec/yr | mean_hold | turnover | fill | book_w(mean) | Sh_decision | "
          "CI(UNCALIBRATED) | block | "
          "dep_len | dep<=block | advisory | "
          "CI(ADVISORY) | pnl | maxDD | Sh_per_bar(diag) | reason |",
          "|---|--:|--:|---|--:|--:|--:|--:|:--:|--:|--:|--:|--:|--:|--:|--:|---|---|--:|--:|"
          ":--:|:--:|---|--:|--:|--:|---|"]
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
        # TASK_902 item 5: a cell the guard never ran on (DEP_NO_CI — no interval was ever
        # bootstrapped for it) rendered a bare `False` in this column, indistinguishable from a
        # cell the guard DID run and refused. `False` now means "the guard ran and refused";
        # `not-run` means "no interval existed for this cell, so the guard never evaluated it" —
        # the same distinction `_dependence_status`/DEP_NO_CI already draws for the CI(ADVISORY)
        # column beside it (`UNCALIBRATED (no CI computed; ...)`, never `REFUSED ON DEPENDENCE`).
        dep_ok_str = "not-run" if dep_state == DEP_NO_CI else str(dep_ok)
        book_w_str = (f"{c.realized_book_width_mean:.2f}"
                      if np.isfinite(c.realized_book_width_mean) else "n/a")
        # TASK_902 item 2c, second half — a cell can reach `status="ok"` (not a fit/engine
        # failure, so it never takes the gate-(ii) drop path) and still realize ZERO decisions
        # (`decide._decide_step`'s `npr < (2*k if long_short else k)` empty-target floor), which
        # used to render indistinguishably from a real measurement: `n_dec=0` sat quietly among
        # 20+ other columns, blank `reason`, and every metric NaN. Say so explicitly, in the same
        # column an engine failure already names its reason in — `reason` stays "" in the DATA
        # (the CellRecord field, per its own docstring convention: non-empty iff status != "ok"),
        # this is a RENDER-time-only synthesized note for exactly this case.
        reason_str = c.reason
        if c.status == "ok" and (not c.n_decisions or c.n_decisions == 0):
            reason_str = ("EMPTY (0 decisions — ok, not an engine failure; universe narrower "
                          "than k, or tau bound the book to nothing eligible; see n_dec/book_w)")
        L.append(
            f"| {c.window} | {c.H} | {c.k} | {c.variant} | {c.tau:g} | {c.N} | {c.fee_bp:g} | "
            f"{c.adv_bp:g} | {c.status} | {c.n_bars} | {c.n_decisions} | "
            f"{c.decisions_per_year:.1f} | {c.mean_hold_bars:.1f} | {c.turnover_mean:.4f} | "
            f"{c.fill_rate:.3f} | {book_w_str} | {c.sharpe_decision:+.3f} | "
            f"[{c.ci_lo_UNCALIBRATED:+.2f},{c.ci_hi_UNCALIBRATED:+.2f}] | "
            f"{blk if blk is not None else 'n/a'} | {dep_str} | {dep_ok_str} | "
            f"{c.ci_advisory} | {ci_adv_str} | "
            f"{c.pnl:+.4f} | {c.maxdd:+.4f} | {c.sharpe_per_bar_diagnostic:+.2f} | {reason_str} |"
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
          "re-run in this loop) — **THE GUARD IS THEREFORE EXPECTED NOT TO BITE HERE**, and "
          + _scope_pass_rate_clause + " is NOT evidence the dep<=block condition holds. An "
          "empirical containment check on the real long/short H=1 series (97 disjoint "
          "1000-decision chunks) contained the grand Sharpe only 0.639 of the time while the "
          "guard certified ~100%. The guard is KEPT — it still fails closed on unmeasurable or "
          "excessive dependence — but its pass is downgraded, here and everywhere it ships, to "
          "'did not refuse', never 'condition measured true'. **TASK_901** (a range-based "
          "dependence probe — first-lag-in-band / variance-ratio / spectral-density-at-zero, "
          "validated out-of-family) is the row that replaces it, re-reading this leg's third "
          "conjunct retroactively FROM THESE SAME ARTIFACTS.",
          ""]
    # TASK_902 item 7: `--out` now defaults into the untracked `runs/` tree, whose directory may
    # not exist yet on a fresh checkout — create it rather than fail on a missing parent.
    path.parent.mkdir(parents=True, exist_ok=True)
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
    one.

    TASK_902 item 11, FIX CYCLE 2 — EVERY echo below is clipped. This function was the WORST
    unbounded-echo site in the file, worse than the `--db-path` case item 11 was written for:
    each message echoes `raw` (and often the offending part) so a 900,000-char
    `--attainable-maker-fee-bp` produced a MEASURED 1,800,148-char `SystemExit` message, against
    the 900k->1.6M TASK_888 precedent item 11 cites."""
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
            f"GATE (iv) REFUSAL: --attainable-maker-fee-bp {_clip_echo(raw)!r} contains no tier "
            "values "
            "(separators only). Pass a comma-separated bp list, e.g. "
            "--attainable-maker-fee-bp 8,16"
        )
    tiers: List[float] = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            raise SystemExit(
                f"GATE (iv) REFUSAL: --attainable-maker-fee-bp entry {_clip_echo(p)!r} is not a "
                f"number (from {_clip_echo(raw)!r}). Pass a comma-separated bp list, e.g. "
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
                f"GATE (iv) REFUSAL: --attainable-maker-fee-bp entry {_clip_echo(p)!r} is not "
                f"finite (from {_clip_echo(raw)!r}). Maker-fee tiers must be finite bp values, "
                "e.g. "
                "--attainable-maker-fee-bp 8,16"
            )
        if v < 0:
            raise SystemExit(
                f"GATE (iv) REFUSAL: --attainable-maker-fee-bp entry {_clip_echo(p)!r} is "
                f"negative (from {_clip_echo(raw)!r}). A maker fee cannot be negative bp (0 is "
                "allowed — a "
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
            f"REFUSAL: --db-path {_clip_echo(raw)!r} is not a filesystem path. "
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
            f"REFUSAL: --db-path {_clip_echo(resolved_cwd)} is the PROTECTED read-only source "
            "of record. TASK_897's scope note: expensive real-data validation is SCHEDULED "
            "(gpu-server, per D54), never polled from a factory or interactive session. If this "
            "really IS the scheduled run, say so explicitly: add "
            "--i-am-the-scheduled-real-data-run."
        )
    # TASK_902 item 10 — TOCTOU residual: this function used to `return str(db_path)`, the RAW,
    # UNRESOLVED operator spelling, while every check above (including the protected-identity
    # check) ran on a `realpath`-resolved string computed here and then discarded. A symlink
    # component in the raw spelling could therefore be re-targeted between THIS check and the
    # caller's later actual open (`_connect_ro`), which re-resolves the raw string from scratch —
    # the classic check/use gap. Resolve ONCE, here, and feed the SAME resolved value to both
    # this check (already done, above) and every caller — never the raw spelling again.
    if is_protected:
        # The protected DB is a fixed, known constant — return ITS canonical realpath directly
        # rather than whichever candidate happened to match (cwd-relative or _REPO-relative);
        # both name the same file once resolved, and this removes any ambiguity about which one.
        return protected
    # Not protected: `resolved_cwd` is the cwd-relative realpath — the interpretation Python/
    # sqlite will actually use to open a relative path at runtime, resolved at the SAME moment
    # this check ran.
    return resolved_cwd


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
    `write_artifact`.

    TASK_902 item 11, FIX CYCLE 2 — EVERY echo below is clipped, including the echoes INSIDE the
    caught `exc` (`datetime.strptime`'s own message quotes the offending value back, so `{exc}`
    is a third copy of the operator's string). Measured before the fix: a 900,000-char
    `--window-a` produced a 2,700,344-char `SystemExit` message — larger than the 900k->1.6M
    TASK_888 defect item 11 cites as its own precedent."""
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) != 2 or not all(parts):
        raise SystemExit(
            f"REFUSAL: {flag} {_clip_echo(raw)!r} must be exactly 'START,END' (ISO dates), "
            f"e.g. {flag} 2024-11-01,2025-04-01"
        )
    epochs = []
    for p in parts:
        try:
            epochs.append(_to_epoch(p))
        except (ValueError, TypeError) as exc:
            raise SystemExit(
                f"REFUSAL: {flag} {_clip_echo(raw)!r} — {_clip_echo(p)!r} is not an ISO date "
                f"(YYYY-MM-DD): {_clip_echo(exc)}. An "
                "unparseable bound is NOT a narrow window: the slicer would select zero "
                "decisions, every cell would still be recorded `ok`, and the artifact would "
                f"read green with no measurement in it. Pass {flag} 2024-11-01,2025-04-01."
            ) from None
    if epochs[0] >= epochs[1]:
        raise SystemExit(
            f"REFUSAL: {flag} {_clip_echo(raw)!r} — START must be strictly BEFORE END "
            f"({_clip_echo(parts[0])!r} >= {_clip_echo(parts[1])!r}). An inverted or empty "
            "window selects zero decisions "
            "silently, which renders as a complete, gate-clean artifact containing no "
            "measurement at all."
        )
    return parts[0], parts[1]


def exit_code_for(meta: Dict) -> int:
    """TASK_902 item 1 / D58(a) — the exit-code contract. Exit NON-ZERO on a gate-(ii)
    completeness violation (`n_cells_recorded != n_cells_expected` — the instrument failing its
    OWN gate: some cell the grid promised was silently never recorded). Exit 0 for an HONEST
    empty result (`n_cells_ok == 0`, every cell accounted for but none reached `status="ok"` —
    e.g. the fixture's inherited-IC guard failing as expected, or a window with zero decisions):
    that is a legitimate, disclosed measurement outcome the artifact states plainly, and on the
    fixture it is the NORMAL state — a non-zero exit there would trade a false-green for a
    false-red that fires on every verification run, training the next operator to ignore the
    exit code, which is strictly worse than the status quo (D58's ruling). Orchestration still
    PARSES THE ARTIFACT rather than gating on this code alone (the standing contract, pre-902);
    this code is a second, cheap signal for callers that do check it.

    TASK_902 fix cycle 2, MAJOR — A ZERO-CELL RUN IS LOUD, AND EXITS NON-ZERO. `run_sweep(
    grid_k=())` (API-reachable; not CLI-reachable — the same surface item 2's second root cause
    lived on) yielded `n_cells_expected == n_cells_recorded == 0`, so the completeness check above
    was satisfied TRIVIALLY by `0 == 0` and returned 0, while `no_measurement` (needs `bool(n_ok)`)
    and `all_cells_failed` (needs `n_cells_total > 0`) BOTH failed to fire: a degenerate run read
    as healthy from top to bottom.

    WHICH SIDE OF D58 THIS FALLS ON, and why. D58 protects the HONEST EMPTY RESULT: cells were
    recorded, every cell the grid promised is accounted for, and none of them measured anything —
    that is a real, disclosed outcome (the NORMAL fixture state), so it stays exit 0 and a
    non-zero code there would train operators to ignore the code. A run with NO CELLS AT ALL is a
    different thing: the grid promised nothing, so nothing can be honestly empty and gate (ii)
    has nothing to check. It is the instrument failing to do anything — much closer to the
    completeness violation above than to an empty measurement — so it exits NON-ZERO and carries
    its own banner in the artifact (`write_artifact`). Note this cannot fire on any grid that
    names at least one value per dimension, so it never touches the honest-empty fixture path."""
    if meta["n_cells_recorded"] != meta["n_cells_expected"]:
        return 1
    if int(meta.get("n_cells_expected") or 0) <= 0 or int(meta.get("n_cells_recorded") or 0) <= 0:
        return 2
    return 0


def _write_or_shout(path: Path, text: str, what: str) -> bool:
    """Write `text` to `path`, and if THAT write itself fails, make sure the operator still gets
    the whole record — on stderr — instead of nothing at all. Returns True iff the file was
    written.

    TASK_902 fix cycle 2, MINOR 2 — `write_refusal_artifact`'s OWN failure was unguarded and
    OUTSIDE `main()`'s `except`. Its `path.parent.mkdir(parents=True, exist_ok=True)` raises
    `FileExistsError` when a path component is an existing plain file (reproducible with
    `--out <plainfile>/result.md`), and `write_text` can raise `PermissionError`,
    `IsADirectoryError`, `OSError` (ENOSPC) or `UnicodeEncodeError`. Any of those inside the
    failure handler produced NO artifact and a traceback out of the handler — reviving exactly the
    "traceback, exit 1, nothing to parse" state item 2b closed, and losing the ORIGINAL exception
    behind a second, unrelated one. So: the double fault is caught here, both faults are named,
    and the artifact body is dumped to stderr so it is never simply lost.

    Deliberately NOT used by `write_artifact` (the success-path writer): a COMPLETED sweep whose
    artifact could not be written must fail loudly and let the exception propagate, because there
    the file IS the product. Here the file is a failure record and the process is already exiting
    non-zero, so degrading to stderr is strictly better than raising over the top of the real
    cause."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return True
    except Exception as write_exc:  # noqa: BLE001 — see docstring: never lose the original.
        print(
            f"[897-sweep] DOUBLE FAULT: could not write the {what} to {_clip_echo(str(path))!r} "
            f"({type(write_exc).__name__}: {_clip_echo(write_exc)}). The record follows on stderr "
            "so it is not lost; NOTHING WAS WRITTEN AT --out, so an orchestrator parsing the "
            "artifact will see NO FILE — treat that as this failure, not as a run that never "
            "started.",
            file=sys.stderr, flush=True)
        print("---- BEGIN UNWRITTEN " + what.upper() + " ----", file=sys.stderr, flush=True)
        try:
            print(text, file=sys.stderr, flush=True)
        except Exception as print_exc:  # noqa: BLE001 — a triple fault still says something.
            print(f"(record itself could not be printed: {type(print_exc).__name__})",
                  file=sys.stderr, flush=True)
        print("---- END UNWRITTEN " + what.upper() + " ----", file=sys.stderr, flush=True)
        return False


def write_internal_error_artifact(path: Path, reason: str, traceback_text: str, db_path: str,
                                  attainable_maker_fee_bp: Sequence[float],
                                  windows: Sequence[Tuple[str, str, str]],
                                  grid_h: Sequence[int], grid_k: Sequence[int],
                                  grid_variant: Sequence[str], grid_tau: Sequence[float],
                                  grid_n: Sequence[int],
                                  grid_fee_bp: Sequence[float] = GRID_FEE_BP,
                                  grid_adv_bp: Sequence[float] = GRID_ADV_BP) -> None:
    """TASK_902 fix cycle 2, BLOCKER — the artifact for an UNHANDLED DEFECT, deliberately sharing
    NOTHING with `write_refusal_artifact`'s prose.

    A governed refusal (a missing DB, a gate this harness designed) and a bug in this harness's
    own code are completely different events for both readers of this file: the operator, who must
    know whether to fix their invocation or fix the instrument, and the orchestrator, whose
    standing contract (D58, pre-902) is to PARSE THE ARTIFACT and never the exit code alone. Item
    2b's fix made them byte-identical — an injected `KeyError`/`TypeError` in
    `probe.report_bar_frequency` produced a file titled `REFUSED BEFORE ANY CELL WAS MEASURED`
    saying "This file is a CONTROLLED REFUSAL record". This file's title, banner and body all say
    DEFECT instead, and it carries the full traceback, which the refusal path never had at all.

    WHY AN ARTIFACT AND NOT A BARE TRACEBACK. Letting the defect propagate would give the loud,
    diagnosable traceback the pre-2b code had, but it would also write NOTHING at `--out`, and an
    absent artifact is exactly the unreadable state item 2b closed — the orchestrator cannot tell
    it from a run that never started. So both: an artifact that is unmistakably NOT a refusal, AND
    the traceback in it AND on stderr."""
    n_cells_expected = (len(list(grid_h)) * len(list(grid_k)) * len(list(grid_variant))
                         * len(list(grid_tau)) * len(list(grid_n)) * len(list(windows))
                         * len(list(grid_fee_bp)) * len(list(grid_adv_bp)))
    L = [
        "# TASK_897 -- gated_maker sweep -- UNEXPECTED INTERNAL ERROR (**A DEFECT, NOT A "
        "GOVERNED REFUSAL**)", "",
        "> ## :warning: UNEXPECTED INTERNAL ERROR -- **THIS IS A DEFECT IN THE INSTRUMENT, NOT A "
        "GOVERNED REFUSAL**",
        ">",
        "> **This file is NOT a refusal record and NOT a sweep.** `run_sweep` raised an exception "
        "of a type this harness does not recognise as any of its own governed refusal conditions "
        f"({_refusal_exception_names()}). That means the instrument "
        "hit a code path it was not written for -- a BUG here, not a bad invocation. Nothing was "
        "measured, and nothing about the data, the grid, or the operator's arguments may be "
        "inferred from this run.",
        ">",
        "> **Do not read this as a refusal, and do not retry it as if a precondition were "
        "missing.** A governed refusal (missing/unreadable `--db-path`, a gate this harness "
        "designed) writes a DIFFERENT artifact with a different title and banner, and says so in "
        "its own words. This one says DEFECT, and it carries the full traceback below (also "
        "printed to stderr) so it can be diagnosed WITHOUT re-running -- TASK_902 fix cycle 2's "
        "BLOCKER was that the two used to be byte-identical prose and the traceback was destroyed "
        "and logged nowhere. **Neither file quotes the other's banner text**, deliberately: an "
        "orchestrator greps these artifacts, so a cross-reference in the prose would make the two "
        "kinds match each other's search strings and re-create exactly the confusion this split "
        "exists to remove.",
        "",
        f"- **Exception:** `{_clip_echo(reason)}`",
        f"- **db_path (as supplied to this harness):** `{_clip_echo(db_path)}`",
        f"- **attainable_maker_fee_bp_tiers:** {_clip_echo(list(attainable_maker_fee_bp))}",
        f"- **windows:** {_clip_echo([{'name': n, 'start': s, 'end': e} for (n, s, e) in windows])}",
        f"- **grid:** H={_clip_echo(list(grid_h))} k={_clip_echo(list(grid_k))} "
        f"variant={_clip_echo(list(grid_variant))} tau={_clip_echo(list(grid_tau))} "
        f"N={_clip_echo(list(grid_n))} fee_bp={_clip_echo(list(grid_fee_bp))} "
        f"adv_bp={_clip_echo(list(grid_adv_bp))}",
        f"- **Cells expected:** {n_cells_expected}  **recorded:** 0  (no cell was reached)",
        "",
        "## Traceback (the whole point of this file -- an operator must be able to diagnose "
        "without re-running)",
        "",
        "```",
        _clip_echo(traceback_text, TRACEBACK_CLIP_LEN).rstrip("\n"),
        "```",
        "",
    ]
    _write_or_shout(path, "\n".join(L) + "\n", "internal-error artifact")


def write_refusal_artifact(path: Path, reason: str, db_path: str,
                            attainable_maker_fee_bp: Sequence[float],
                            windows: Sequence[Tuple[str, str, str]],
                            grid_h: Sequence[int], grid_k: Sequence[int],
                            grid_variant: Sequence[str], grid_tau: Sequence[float],
                            grid_n: Sequence[int],
                            grid_fee_bp: Sequence[float] = GRID_FEE_BP,
                            grid_adv_bp: Sequence[float] = GRID_ADV_BP) -> None:
    """TASK_902 item 2b — a failure at the FIRST data touch (`probe.report_bar_frequency`,
    reached deep inside `run_sweep`, before `write_artifact` is ever callable) used to crash
    uncaught with NO artifact written: a traceback, exit 1, nothing at `--out`. Since
    orchestration PARSES THE ARTIFACT and never the exit code alone (D58, pre-902), that is
    indistinguishable from a run that never started. This writes a small, clearly-labelled,
    artifact-SHAPED refusal record to the SAME `--out` path instead: no cell table (none was
    ever computed — inventing a full dropped-row table for a failure that happened before the
    universe or even the grid size were resolved would fabricate structure this run never
    reached), but every input the caller supplied, the reason, and an explicit `n_cells_recorded:
    0` against the expected count so a completeness-checking reader sees the same gate-(ii)-style
    violation a partial run would show.

    TASK_902 fix cycle 2, BLOCKER — THIS FILE IS NOW ONLY EVER WRITTEN FOR A GOVERNED REFUSAL.
    `main()` used to route EVERY `Exception` here, so an injected `KeyError`/`TypeError` in
    `probe.report_bar_frequency` — an ordinary code defect — produced this exact artifact,
    byte-identical prose to the legitimate `sqlite3.OperationalError` case, and the traceback was
    thrown away. Discrimination is now by TYPE (`REFUSAL_EXCEPTIONS`), and anything outside that
    closed set gets `write_internal_error_artifact` instead, which says DEFECT everywhere this
    file says REFUSAL and carries the traceback."""
    n_cells_expected = (len(list(grid_h)) * len(list(grid_k)) * len(list(grid_variant))
                         * len(list(grid_tau)) * len(list(grid_n)) * len(list(windows))
                         * len(list(grid_fee_bp)) * len(list(grid_adv_bp)))
    L = [
        "# TASK_897 -- gated_maker sweep artifact -- REFUSED BEFORE ANY CELL WAS MEASURED", "",
        "> ## :warning: THIS RUN REFUSED AT THE FIRST DATA TOUCH -- **NO CELL IN THIS ARTIFACT "
        "WAS EVER MEASURED**",
        ">",
        "> `run_sweep` raised before `write_artifact` was ever reachable (TASK_902 item 2b: this "
        "used to be an uncaught traceback with NO artifact written at all -- indistinguishable, "
        "to an orchestrator that parses the artifact and never the exit code alone, from a run "
        "that never started). This file is a CONTROLLED REFUSAL record, not a completed sweep: "
        "it has no cell table because no cell was ever reached, and the process exits non-zero "
        "(TASK_902 item 1's exit-code contract treats 0 of the expected cells recorded as the "
        "gate-(ii) violation it is).",
        ">",
        "> **A GOVERNED REFUSAL, i.e. a PRECONDITION this harness deliberately checks** -- the "
        "exception was one of "
        f"{_refusal_exception_names()} (a missing/unreadable "
        "`--db-path`, or one of this harness's own gates). It is NOT an instrument defect: a "
        "defect writes a DIFFERENT artifact, with a different title and banner and a full "
        "traceback in it. Fix the invocation, not the instrument. (TASK_902 fix cycle 2's BLOCKER "
        "was that a shipped `TypeError` and a missing DB used to produce byte-identical prose "
        "here. Neither file quotes the other's banner text, deliberately: an orchestrator greps "
        "these artifacts, so a cross-reference would make the two kinds match each other's "
        "search strings.)",
        "",
        # item 11 (fix cycle 2): `db_path` and `reason` were clipped; these four were NOT, and
        # every one of them is a caller-supplied SEQUENCE — unbounded in length through the public
        # API (e.g. `run_sweep(grid_h=range(10**6))`), which is the same surface item 2's second
        # root cause lived on. All echoes in this record are clipped now.
        f"- **db_path (as supplied to this harness):** `{_clip_echo(db_path)}`",
        f"- **attainable_maker_fee_bp_tiers:** {_clip_echo(list(attainable_maker_fee_bp))}",
        f"- **windows:** {_clip_echo([{'name': n, 'start': s, 'end': e} for (n, s, e) in windows])}",
        f"- **grid:** H={_clip_echo(list(grid_h))} k={_clip_echo(list(grid_k))} "
        f"variant={_clip_echo(list(grid_variant))} tau={_clip_echo(list(grid_tau))} "
        f"N={_clip_echo(list(grid_n))} fee_bp={_clip_echo(list(grid_fee_bp))} "
        f"adv_bp={_clip_echo(list(grid_adv_bp))}",
        f"- **Cells expected:** {n_cells_expected}  **recorded:** 0",
        f"- **Refusal reason:** {_clip_echo(reason)}",
        "",
    ]
    # MINOR 2: this write is itself guarded — a `--out` whose parent is an existing plain file
    # used to raise `FileExistsError` out of `mkdir` here, INSIDE main()'s failure handler and
    # outside its `except`, leaving no artifact at all and burying the original refusal.
    _write_or_shout(path, "\n".join(L) + "\n", "refusal artifact")


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
    ap.add_argument(
        "--out", type=str,
        default=str(_REPO / "runs" / "897_corrected_809" / "result.md"),
        help="TASK_902 item 7: defaults into the UNTRACKED runs/ tree ("
             "runs/*/* is gitignored), never into the tracked experiments/897_corrected_809/ "
             "directory a bare run used to write into by default (result.md there is tracked, "
             "so a bare run's output was one `git add -A` away from being committed as source — "
             "every 897 brief carried a 'never use the default --out' instruction discipline "
             "because of this). Override freely; the scheduled gpu-server leg always passes its "
             "own timestamped path explicitly.")
    ap.add_argument("--smoke", action="store_true", help="tiny pre-registered SMOKE grid, not the real one")
    ap.add_argument("--skip-inherited-ic-check", action="store_true",
                     help="Skip the pin-era inherited-IC GUARD (TASK_897 D60: gated against 806's "
                          "pins on a SEPARATE pin-era panel built from probe.Config's own split "
                          "defaults, test_end=2025-04-01 — the pins are real-data magic numbers, "
                          "so this skips building that panel entirely, no cost, at "
                          "fixture/synthetic scale). Also skips the sweep-panel signal-IC "
                          "MEASUREMENT (recorded, never gated even when this flag is absent).")
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

    out = Path(args.out)
    # TASK_902 item 2b — `probe.report_bar_frequency` (called deep inside `run_sweep`) is this
    # harness's FIRST data touch on every run, and it has no exception handling of its own: an
    # absent/unreadable `--db-path` (or any other failure this harness did not anticipate) used
    # to crash UNCAUGHT — a traceback, exit 1, and NO artifact written at all. The standing
    # orchestration contract (D58, pre-902) is that orchestration PARSES THE ARTIFACT, never the
    # exit code — a refusal that produces nothing to parse is indistinguishable from a run that
    # never started. Catch it here and write a legible, clearly-labelled refusal record to the
    # SAME `--out` path instead. `SystemExit` (every clean refusal already in this file, e.g.
    # `assert_db_path_allowed`/`parse_fee_tiers`/gate (iv)) is a `BaseException`, NOT an
    # `Exception`, so it is untouched by the `except` below and propagates exactly as before —
    # same for `KeyboardInterrupt`.
    #
    # TASK_902 fix cycle 2, BLOCKER — TWO HANDLERS, NOT ONE, AND THE TRACEBACK IS NEVER
    # DESTROYED. The single `except Exception` this replaced rendered EVERY escape as the
    # governed-refusal artifact above: an injected `KeyError('nonexistent_key')` or
    # `TypeError("unsupported operand type(s) for +: 'NoneType' and 'int'")` in
    # `probe.report_bar_frequency` produced a file titled `REFUSED BEFORE ANY CELL WAS MEASURED`
    # saying "This file is a CONTROLLED REFUSAL record", byte-identical prose to the legitimate
    # missing-DB case, with exit 1 — so a shipped defect and a bad invocation were
    # indistinguishable to the operator AND to the orchestrator (whose standing contract is to
    # parse the artifact, never the exit code alone). Worse than the loud uncaught traceback it
    # replaced. `reason = f"{type(exc).__name__}: {exc}"` was the only capture and
    # `traceback.format_exc()` was called nowhere in this file, so the call stack was destroyed.
    #
    # Discrimination is by TYPE (`_is_governed_refusal` over the closed `REFUSAL_EXCEPTIONS` set,
    # minus the inherited-but-defect classes), not by where the exception was caught, and BOTH
    # paths now capture `traceback.format_exc()`.
    try:
        cells, meta = run_sweep(
            attainable_maker_fee_bp=tiers, db_path=db_path, train_start=args.train_start,
            val_start=args.val_start, windows=windows, grid_h=grid_h, grid_k=grid_k,
            grid_variant=grid_variant, grid_tau=grid_tau, grid_n=grid_n, seed=args.seed,
            n_boot=args.n_boot, verify_inherited_ic=not args.skip_inherited_ic_check,
            scheduled_real_data_run=args.scheduled_real_data_run,
        )
    except Exception as exc:  # noqa: BLE001 — see the block comment above; the REFUSAL-vs-DEFECT
        # discrimination is `_is_governed_refusal`, by TYPE, not by which clause caught it.
        reason = f"{type(exc).__name__}: {exc}"
        tb_text = traceback.format_exc()          # NEVER destroyed — the BLOCKER's second half.
        if _is_governed_refusal(exc):
            write_refusal_artifact(out, reason, db_path, tiers, windows, grid_h, grid_k,
                                    grid_variant, grid_tau, grid_n)
            print(f"[897-sweep] REFUSED before any cell was measured: {_clip_echo(reason)} Wrote "
                  f"refusal artifact to {out}.", file=sys.stderr, flush=True)
            # The refusal artifact is self-explanatory and the message above names the cause, but
            # the stack is still the cheapest way to see WHICH data touch refused — surfaced on
            # stderr only (never in the artifact, which stays the small legible record it was
            # designed as).
            print("[897-sweep] refusal traceback (diagnostic only — this was a governed refusal, "
                  f"not a defect):\n{_clip_echo(tb_text, TRACEBACK_CLIP_LEN)}",
                  file=sys.stderr, flush=True)
        else:
            write_internal_error_artifact(out, reason, tb_text, db_path, tiers, windows, grid_h,
                                           grid_k, grid_variant, grid_tau, grid_n)
            print(f"[897-sweep] UNEXPECTED INTERNAL ERROR — THIS IS A DEFECT IN THE INSTRUMENT, "
                  f"NOT A GOVERNED REFUSAL: {_clip_echo(reason)} Wrote internal-error artifact to "
                  f"{out}. Traceback follows:", file=sys.stderr, flush=True)
            print(_clip_echo(tb_text, TRACEBACK_CLIP_LEN), file=sys.stderr, flush=True)
        sys.exit(1)

    write_artifact(cells, meta, out)
    print(f"[897-sweep] DONE. {meta['n_cells_recorded']}/{meta['n_cells_expected']} cells "
          f"recorded (ok={meta['n_cells_ok']}). Wrote {out}. Runtime {meta['runtime_s']:.1f}s.",
          flush=True)
    # TASK_902 item 1 / D58(a) — exit-code contract: non-zero on a gate-(ii) completeness
    # violation, exit 0 for an honest empty result. See `exit_code_for`'s docstring.
    code = exit_code_for(meta)
    if code == 2:
        # fix cycle 2, MAJOR — a zero-cell run is loud on stderr too, not only in the artifact.
        print("[897-sweep] ZERO CELLS: at least one grid dimension was EMPTY, so no cell was "
              "ever measured and gate (ii)'s recorded==expected check is satisfied trivially by "
              "0==0. This is the instrument doing nothing, NOT an honest empty result — exiting "
              "2. See the ZERO CELLS banner in the artifact.", file=sys.stderr, flush=True)
    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()

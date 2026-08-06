"""
TASK_872 — regression test for the LIVE Kraken V2 OHLC timestamp parse
(`mcts_trader/data/stream.py::KrakenWebSocket._handle_message`).

Two defects are pinned here.

(A) The TASK_872 defect: the adapter strips a confirmed 'Z' suffix off Kraken's
    `interval_begin` and then calls `.timestamp()` on the now-NAIVE datetime. Python
    resolves a naive datetime's `.timestamp()` in the HOST's local timezone, so every
    live candle's epoch was shifted by the machine's UTC offset (a different,
    DST-sensitive number on every host). Fix: attach `timezone.utc` explicitly to any
    naive parse result before `.timestamp()`, at BOTH conversion sites.

(B) A PRE-EXISTING defect in the nanosecond-truncation block (present unchanged at
    67c8176, i.e. not introduced by the TASK_872 diff, but closed with it because the
    diff re-touched that block): the truncation runs AFTER the 'Z' strip and slices the
    fractional FIELD, which for an offset-bearing string also contains the offset.
    With >=6 fractional digits the offset was deleted outright (no exception at all --
    the primary branch parsed a silently naive string); with fewer it was cut in half
    (ValueError -> the fallback's `split('.')[0]` discarded it too). Either way (A)'s
    `tzinfo is None` guard then stamped UTC onto a wall-clock reading that was never
    UTC, so the guard is precisely what made the error SILENT. Fix: detach the offset
    suffix before truncation and re-attach it at parse time, so `fromisoformat` gets an
    aware string and the naive guard simply does not fire.

Every expectation below is computed INDEPENDENTLY, from the actual instant the input
string denotes (an explicit aware `datetime` per case) -- never by reusing the UTC
constant from a different instant.

Drives `_handle_message()` directly with synthetic V2 payloads -- no network, no
websocket connection, no exchange credentials. Run:
    JAX_PLATFORMS=cpu pytest tools/tests/test_stream_utc.py -v
"""

import asyncio
import importlib.util
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# Robust, machine-independent import shim (matches tools/tests/test_vector_env.py):
# mcts_trader/ is a PEP-420 namespace package, so the repo root must be on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]  # tools/tests/ -> tools/ -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcts_trader.data.stream import KrakenWebSocket  # noqa: E402  (post-fix code under test)

# The last commit before TASK_872's fix -- pinned by hash, NOT the literal ref
# "HEAD", because once this fix is committed HEAD will point PAST it and the
# "vacuous probe" checks below would silently start comparing the fixed code
# against itself. If stream.py's pre-fix history is ever rewritten/squashed past
# this hash, re-pin to whatever commit still has the naive `.timestamp()` calls.
_PRE_FIX_REV = "67c8176493d196388d86fefd72bea8a502301ebc"

TZS = ["UTC", "Europe/Prague", "Asia/Kathmandu", "Pacific/Chatham"]

_IST = timezone(timedelta(hours=5, minutes=30))   # +05:30, non-whole-hour
_PST = timezone(timedelta(hours=-8))              # -08:00, negative

# 2024-07-15 is DST-interior for Europe/Prague (CEST, UTC+2) -- deliberately not a
# winter date, so the pre-fix local-time bug is live rather than accidentally dormant.
# Kraken's documented V2 OHLC format: 'Z' suffix + 9-digit (nanosecond) fraction.
PRIMARY_TS = "2024-07-15T12:34:56.123456789Z"
# Independently derived from the instant PRIMARY_TS denotes. The adapter truncates
# nanoseconds to microseconds (.123456789 -> .123456) and emits str(int(epoch)).
PRIMARY_INSTANT = datetime(2024, 7, 15, 12, 34, 56, 123456, tzinfo=timezone.utc)
EXPECT_EPOCH = int(PRIMARY_INSTANT.timestamp())

# (ts_string, the aware datetime that string actually denotes after the adapter's
# documented ns->us truncation). Each expectation is built from its OWN instant --
# deliberately not from EXPECT_EPOCH, which is a different instant (12:34:56Z).
# Fractional-digit counts 0 / 3 / 6 / 9 for both a positive non-whole-hour offset and
# a negative one: pre-fix, n=0 was correct and every n>=1 was wrong by exactly the
# offset (19800 s / -28800 s).
OFFSET_CASES = [
    ("2024-07-15T12:34:56+05:30",           datetime(2024, 7, 15, 12, 34, 56, 0,      tzinfo=_IST)),
    ("2024-07-15T12:34:56.123+05:30",       datetime(2024, 7, 15, 12, 34, 56, 123000, tzinfo=_IST)),
    ("2024-07-15T12:34:56.123456+05:30",    datetime(2024, 7, 15, 12, 34, 56, 123456, tzinfo=_IST)),
    ("2024-07-15T12:34:56.123456789+05:30", datetime(2024, 7, 15, 12, 34, 56, 123456, tzinfo=_IST)),
    ("2024-07-15T12:34:56-08:00",           datetime(2024, 7, 15, 12, 34, 56, 0,      tzinfo=_PST)),
    ("2024-07-15T12:34:56.123-08:00",       datetime(2024, 7, 15, 12, 34, 56, 123000, tzinfo=_PST)),
    ("2024-07-15T12:34:56.123456-08:00",    datetime(2024, 7, 15, 12, 34, 56, 123456, tzinfo=_PST)),
    ("2024-07-15T12:34:56.123456789-08:00", datetime(2024, 7, 15, 12, 34, 56, 123456, tzinfo=_PST)),
]

# Python 3.10's fromisoformat() requires a rigid 10-char 'YYYY-MM-DD' date but accepts
# ANY single character as the date/time separator: lowercase 't', a space, even 'X' or
# '_' all parse. A first pass at the offset detach located the time field by searching
# for a literal 'T'/' ', so every other separator skipped the detach entirely and
# reinstated the exact silent-mis-stamp defect this file exists to close (found by
# review, never committed). The detach is now POSITIONAL (index 11), so these are
# covered to keep it that way.
NONCANONICAL_SEPARATORS = ["t", " ", "X", "_"]

# Forms with NO offset must come through the detach completely untouched -- this is the
# guard on the backward scan never reaching the date's own hyphens (indices 4 and 7) or
# the separator (index 10). Expectations derived per-instant, as everywhere else.
NO_OFFSET_CASES = [
    ("2024-07-15T12:34:56.123456789Z", datetime(2024, 7, 15, 12, 34, 56, 123456, tzinfo=timezone.utc)),
    ("2024-07-15T12:34:56Z",           datetime(2024, 7, 15, 12, 34, 56, tzinfo=timezone.utc)),
    ("2024-07-15T12:34:56",            datetime(2024, 7, 15, 12, 34, 56, tzinfo=timezone.utc)),
    ("2024-07-15",                     datetime(2024, 7, 15, 0, 0, 0, tzinfo=timezone.utc)),
    # NOT a date carrying an offset: fromisoformat reads the '+' at index 10 as the
    # SEPARATOR and '05:30' as the TIME. Detaching it would corrupt a string that
    # currently parses, which is exactly why the scan starts at index 11 and why a
    # too-short timestamp skips the detach rather than failing loud.
    ("2024-07-15+05:30",               datetime(2024, 7, 15, 5, 30, 0, tzinfo=timezone.utc)),
]

# The ValueError fallback (site 2) is NOT reachable by any well-formed Kraken shape:
# swept over fractional-second lengths 0-12 with 'Z' and with real offsets, the
# truncation step always yields exactly 6 digits and the offset is now re-attached
# intact, so `fromisoformat` always accepts on the first try. (This contradicts the
# task file's premise that a nanosecond payload takes the fallback -- see report.)
# The fallback is a MALFORMED-payload robustness path, so it is exercised with a
# corrupt fractional field, once with 'Z' and once with a real offset -- the latter
# proving the offset also survives INTO the fallback rather than being dropped there.
FALLBACK_TS_Z = "2024-07-15T12:34:56.12x456789Z"
FALLBACK_Z_INSTANT = datetime(2024, 7, 15, 12, 34, 56, tzinfo=timezone.utc)
FALLBACK_TS_OFFSET = "2024-07-15T12:34:56.12x456789+05:30"
FALLBACK_OFFSET_INSTANT = datetime(2024, 7, 15, 12, 34, 56, tzinfo=_IST)

# Colon-less offsets are not accepted by Python 3.10's fromisoformat in EITHER branch.
# The required behaviour is to FAIL LOUD (outer handler logs, candle dropped) rather
# than emit a silently mis-stamped epoch.
UNPARSEABLE_TS = "2024-07-15T12:34:56.123+0530"


def _run_ohlc(ws_cls, interval_begin, symbol="XBT/USD", require=True):
    """Feed one synthetic V2 ohlc/update message through _handle_message and return
    the produced V1-style epoch (first element of the [time, etime, ...] tuple).
    With require=False, returns None if the adapter dropped the item instead."""
    captured = {}

    async def on_ohlc(msg):
        captured["epoch"] = int(msg[1][0])

    async def on_ticker(_data):
        pass

    ws = ws_cls(on_ticker=on_ticker, on_ohlc=on_ohlc)
    msg = {
        "channel": "ohlc",
        "type": "update",
        "data": [{
            "symbol": symbol, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
            "vwap": 1.0, "volume": 1.0, "trades": 1,
            "interval_begin": interval_begin,
        }],
    }
    asyncio.run(ws._handle_message(msg))
    if require:
        assert "epoch" in captured, "on_ohlc callback never fired (adapter swallowed the item)"
    return captured.get("epoch")


def _with_tz(tz, fn):
    """Run fn() with the process TZ actually changed (TZ env var + time.tzset(),
    POSIX). This is the exact channel the bug -- and the fix -- runs through:
    naive-datetime .timestamp() resolves via the C library's localtime(), which
    reads the OS tzdata via TZ. Restores the previous TZ afterward."""
    old = os.environ.get("TZ")
    os.environ["TZ"] = tz
    time.tzset()
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def _load_head_stream_module():
    """Load the PRE-FIX mcts_trader/data/stream.py directly from git at
    _PRE_FIX_REV (the real committed bytes, not a hand-reproduction), so the
    control below exercises the actual pre-fix defect, not an approximation of it."""
    result = subprocess.run(
        ["git", "show", f"{_PRE_FIX_REV}:mcts_trader/data/stream.py"],
        cwd=str(_REPO_ROOT), capture_output=True, check=True, text=True,
    )
    spec = importlib.util.spec_from_loader("stream_head_control", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(result.stdout, "stream_head_control.py", "exec"), mod.__dict__)
    return mod


def test_primary_branch_tz_invariant_and_matches_independent_utc():
    """Site 1 (primary fromisoformat parse), documented Kraken 'Z' + 9-digit shape.
    Post-fix epoch must be IDENTICAL across all four zones AND equal the
    independently computed UTC epoch."""
    results = {tz: _with_tz(tz, lambda: _run_ohlc(KrakenWebSocket, PRIMARY_TS)) for tz in TZS}
    assert len(set(results.values())) == 1, \
        f"post-fix epoch must be TZ-invariant across {TZS}, got {results}"
    assert results["UTC"] == EXPECT_EPOCH, \
        f"post-fix epoch must equal the independently computed UTC value: " \
        f"got {results['UTC']}, expected {EXPECT_EPOCH}"


def test_head_control_is_not_vacuous():
    """The probe must be able to fail: the pre-fix code, loaded from git at
    _PRE_FIX_REV, must produce DIFFERENT epochs across these same timezones for the
    identical payload -- otherwise this whole TZ-invariance proof would check nothing."""
    head_mod = _load_head_stream_module()
    head_results = {tz: _with_tz(tz, lambda: _run_ohlc(head_mod.KrakenWebSocket, PRIMARY_TS))
                    for tz in TZS}
    assert len(set(head_results.values())) > 1, \
        f"pre-fix control must differ across timezones, got {head_results} " \
        f"-- if it doesn't, this probe is vacuous"
    assert head_results["UTC"] == EXPECT_EPOCH, \
        "pre-fix control must be correct in UTC (isolates the bug to non-UTC hosts only)"

    fixed_results = {tz: _with_tz(tz, lambda: _run_ohlc(KrakenWebSocket, PRIMARY_TS))
                     for tz in TZS}
    assert len(set(fixed_results.values())) == 1, \
        f"post-fix module must NOT reproduce the pre-fix spread, got {fixed_results}"


@pytest.mark.parametrize("ts_str,instant", OFFSET_CASES,
                         ids=[c[0] for c in OFFSET_CASES])
def test_offset_bearing_payload_yields_the_instant_it_denotes(ts_str, instant):
    """An `interval_begin` carrying a REAL numeric offset must resolve to the epoch of
    the instant it actually denotes -- computed here from that instant alone, never
    from the UTC constant. Pre-fix, the truncation block destroyed the offset inside
    the string and the naive guard stamped UTC on the leftover wall clock, so every
    case with >=1 fractional digit was wrong by exactly the offset."""
    expected = int(instant.timestamp())
    naive_as_utc = int(instant.replace(tzinfo=timezone.utc).timestamp())

    results = {tz: _with_tz(tz, lambda: _run_ohlc(KrakenWebSocket, ts_str)) for tz in TZS}
    assert len(set(results.values())) == 1, \
        f"{ts_str}: epoch must be TZ-invariant across {TZS}, got {results}"
    assert results["UTC"] == expected, \
        f"{ts_str}: expected {expected} (the instant it denotes); got {results['UTC']}. " \
        f"Off by {results['UTC'] - expected} s -- {naive_as_utc} would mean the offset " \
        f"was dropped and the wall clock mis-stamped as UTC."
    # Guard the guard: these fixtures must actually distinguish the two answers,
    # otherwise the assertion above could pass on a mis-stamped value.
    assert expected != naive_as_utc, \
        f"{ts_str}: fixture is vacuous -- offset-stripped value coincides with the truth"


@pytest.mark.parametrize("ts_str,instant", OFFSET_CASES,
                         ids=[c[0] for c in OFFSET_CASES])
def test_offset_control_pre_fix_is_wrong(ts_str, instant):
    """Non-vacuity for the offset cases: the pre-fix code (real committed bytes at
    _PRE_FIX_REV) must MISS the correct epoch for every fractional case, and must miss
    it by exactly the offset -- which is what makes the fix above load-bearing. n=0
    carries no fractional field for the truncation to slice, so it was already correct
    pre-fix; it is asserted correct rather than wrong."""
    head_mod = _load_head_stream_module()
    expected = int(instant.timestamp())
    naive_as_utc = int(instant.replace(tzinfo=timezone.utc).timestamp())
    got = _with_tz("UTC", lambda: _run_ohlc(head_mod.KrakenWebSocket, ts_str))

    if "." not in ts_str:
        assert got == expected, \
            f"{ts_str}: no fractional field, so the pre-fix truncation never ran -- " \
            f"expected this case to be correct pre-fix, got {got}"
    else:
        assert got == naive_as_utc and got != expected, \
            f"{ts_str}: pre-fix control should have mis-stamped the wall clock as UTC " \
            f"({naive_as_utc}, off by {naive_as_utc - expected} s), got {got}"


@pytest.mark.parametrize("sep", NONCANONICAL_SEPARATORS, ids=lambda s: f"sep={s!r}")
@pytest.mark.parametrize("ts_str,instant", OFFSET_CASES, ids=[c[0] for c in OFFSET_CASES])
def test_offset_detach_is_separator_agnostic(sep, ts_str, instant):
    """The offset detach must not depend on the date/time separator being an uppercase
    'T'. Python 3.10 accepts any single character there, so a separator-MATCHED scan
    would skip the detach for 't'/' '/'X'/'_' and silently mis-stamp the wall clock as
    UTC -- the round-1 defect, reintroduced through a different door. Each case must
    equal the instant it denotes AND agree with its canonical 'T' spelling."""
    swapped = ts_str.replace("T", sep, 1)
    assert swapped != ts_str, "fixture did not actually swap the separator"

    expected = int(instant.timestamp())
    naive_as_utc = int(instant.replace(tzinfo=timezone.utc).timestamp())
    assert expected != naive_as_utc, "fixture is vacuous -- offset makes no difference"

    results = {tz: _with_tz(tz, lambda: _run_ohlc(KrakenWebSocket, swapped)) for tz in TZS}
    assert len(set(results.values())) == 1, \
        f"{swapped!r}: epoch must be TZ-invariant across {TZS}, got {results}"
    assert results["UTC"] == expected, \
        f"{swapped!r}: expected {expected} (the instant it denotes); got {results['UTC']}, " \
        f"off by {results['UTC'] - expected} s -- {naive_as_utc} would mean the " \
        f"separator {sep!r} caused the offset detach to be skipped"

    canonical = _with_tz("UTC", lambda: _run_ohlc(KrakenWebSocket, ts_str))
    assert results["UTC"] == canonical, \
        f"separator {sep!r} changed the result: {results['UTC']} vs {canonical} for 'T'"


@pytest.mark.parametrize("ts_str,instant", NO_OFFSET_CASES,
                         ids=[c[0] for c in NO_OFFSET_CASES])
def test_offset_detach_leaves_offset_free_forms_untouched(ts_str, instant):
    """Regression guard on the backward scan: it must never chew into the date's own
    hyphens or the separator. Offset-free forms (including a bare date, and the
    '2024-07-15+05:30' form whose '+' is a SEPARATOR rather than an offset) must
    resolve to their documented UTC instant, unchanged and TZ-invariant."""
    expected = int(instant.timestamp())
    results = {tz: _with_tz(tz, lambda: _run_ohlc(KrakenWebSocket, ts_str)) for tz in TZS}
    assert set(results.values()) == {expected}, \
        f"{ts_str!r}: expected {expected} in every zone, got {results}"


def test_fallback_branch_is_actually_reached_and_fixed():
    """Site 2 (the ValueError fallback). Proves, by COUNTING real
    datetime.fromisoformat() invocations (not by code inspection), that the fallback
    genuinely fires -- then that it produces the correct instant, is TZ-invariant, and
    retains a real offset instead of dropping it."""
    import datetime as _dt_mod

    # datetime.datetime is an immutable C type -- patch.object() cannot set an
    # attribute directly on it. Subclass it instead and patch the MODULE's
    # `datetime` binding (a plain, mutable module namespace); stream.py's
    # `from datetime import datetime` is a LOCAL import re-executed on every call
    # to _handle_message, so it picks up the patched subclass each time.
    counts = {"n": 0}
    _real_datetime = _dt_mod.datetime  # captured BEFORE patching -- avoids the
    # subclass's fromisoformat recursing into itself via the patched module name.

    class _CountingDatetime(_real_datetime):
        @classmethod
        def fromisoformat(cls, s):
            counts["n"] += 1
            return _real_datetime.fromisoformat(s)

    def _spy_and_count(ws_cls, ts):
        counts["n"] = 0
        with patch.object(_dt_mod, "datetime", _CountingDatetime):
            epoch = _run_ohlc(ws_cls, ts)
        return epoch, counts["n"]

    # Sanity: the PRIMARY payload takes exactly 1 fromisoformat call (succeeds first
    # try) -- confirms the counting mechanism itself discriminates.
    _, n_primary = _with_tz("UTC", lambda: _spy_and_count(KrakenWebSocket, PRIMARY_TS))
    assert n_primary == 1, \
        f"primary payload must resolve in a single fromisoformat call, got {n_primary}"

    # 'Z' + corrupt fraction: 2 calls (first raises, retry after stripping the
    # fraction succeeds), landing naive -> must be stamped UTC.
    z_expected = int(FALLBACK_Z_INSTANT.timestamp())
    epoch_z, n_z = _with_tz("UTC", lambda: _spy_and_count(KrakenWebSocket, FALLBACK_TS_Z))
    assert n_z == 2, \
        f"fallback payload must take exactly 2 fromisoformat calls " \
        f"(primary raises, fallback retries), got {n_z} -- this payload no longer " \
        f"exercises the branch under test"
    assert epoch_z == z_expected, \
        f"fallback branch ('Z') must resolve to {z_expected}, got {epoch_z}"

    # Real offset + corrupt fraction: also 2 calls, and the offset must SURVIVE into
    # the fallback parse rather than being discarded with the fraction.
    off_expected = int(FALLBACK_OFFSET_INSTANT.timestamp())
    off_naive_as_utc = int(FALLBACK_OFFSET_INSTANT.replace(tzinfo=timezone.utc).timestamp())
    epoch_off, n_off = _with_tz("UTC", lambda: _spy_and_count(KrakenWebSocket, FALLBACK_TS_OFFSET))
    assert n_off == 2, \
        f"offset fallback payload must take exactly 2 fromisoformat calls, got {n_off}"
    assert epoch_off == off_expected, \
        f"fallback branch must keep the +05:30 offset: expected {off_expected}, got " \
        f"{epoch_off} ({off_naive_as_utc} would mean the offset was dropped)"

    # Both fallback payloads must be TZ-invariant post-fix.
    for ts, exp in ((FALLBACK_TS_Z, z_expected), (FALLBACK_TS_OFFSET, off_expected)):
        results = {tz: _with_tz(tz, lambda: _run_ohlc(KrakenWebSocket, ts)) for tz in TZS}
        assert set(results.values()) == {exp}, \
            f"{ts}: post-fix fallback must be TZ-invariant and equal {exp}, got {results}"

    # Non-vacuity for the fallback: pre-fix, the same payloads spread across zones.
    head_mod = _load_head_stream_module()
    head_z = {tz: _with_tz(tz, lambda: _run_ohlc(head_mod.KrakenWebSocket, FALLBACK_TS_Z))
              for tz in TZS}
    assert len(set(head_z.values())) > 1, \
        f"pre-fix control for the fallback branch must differ across timezones, " \
        f"got {head_z} -- if it doesn't, this probe is vacuous"
    head_off = _with_tz("UTC", lambda: _run_ohlc(head_mod.KrakenWebSocket, FALLBACK_TS_OFFSET))
    assert head_off != off_expected, \
        f"pre-fix control must have dropped the offset in the fallback, got {head_off}"


def test_unparseable_offset_fails_loud_rather_than_mis_stamping():
    """A colon-less offset is rejected by Python 3.10's fromisoformat in BOTH branches.
    The adapter must then DROP the candle (its outer handler logs the error) -- it must
    never fall through to a silently mis-stamped epoch, which is the failure mode this
    whole file exists to prevent."""
    got = _with_tz("UTC", lambda: _run_ohlc(KrakenWebSocket, UNPARSEABLE_TS, require=False))
    assert got is None, \
        f"unparseable offset must be dropped loudly, not resolved to an epoch; got {got}"


def test_ticker_and_other_handlers_do_no_naive_timestamp_parsing():
    """Module sweep (task scope 3): confirm the ticker handler and the rest of the
    adapter's surface perform no date-string parsing at all -- they forward the raw
    payload verbatim -- so there is no other naive .timestamp()/fromisoformat site
    on this file to convert. This is a structural check on the source, not a
    property of any one payload."""
    import ast
    import inspect

    src = inspect.getsource(sys.modules["mcts_trader.data.stream"])
    call_sites = [
        node for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fromisoformat"
    ]
    # Only the two known, now-fixed call sites should exist on this module's surface.
    assert len(call_sites) == 2, \
        f"module sweep: expected exactly the two known fromisoformat call sites, " \
        f"found {len(call_sites)} -- a new one appeared; give it the same " \
        f"tzinfo=utc treatment or an explicit intentionally-local note"

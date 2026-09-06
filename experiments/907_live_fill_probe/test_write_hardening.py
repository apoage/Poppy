"""TASK_913 (WRITE_HARDENING_SPEC.md) — the write-hardening: the dedicated writer thread behind
one bounded queue (§2.0), priority vs periodic fsync (§2.1/§2.2), queue overflow disclosure
(§2.0.2), R6's controlled shutdown (§2.0.1), and the `clock_jump` -> `emission_silence` rename
(§5). §7's eleven tests, in order, plus two extra gates the orchestrator brief names explicitly
(R1 proven rather than asserted; old files with `clock_jump`/`book_checksum_mismatch` still
parse).

DETERMINISM (§7's own note): every test synchronizes on `CaptureWriter.drain()`, never on
`time.sleep()`. A worker that woke on a timer instead of draining the queue would still pass a
sleep-based test — that is exactly the mutation `drain()` exists to make impossible to hide
behind.
"""
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture as capture_mod  # noqa: E402
from capture import (  # noqa: E402
    OVERFLOW_RESERVE, CaptureWriter, GapRecord, read_capture,
)
from health_probe import analyze_capture  # noqa: E402
from recorder import (  # noqa: E402
    CaptureWriteFailed, LiveRecorder, RecorderConfig,
)


async def _fast_sleep(_seconds):
    return None


class _AsyncCM:
    """Minimal async context manager stand-in for `websockets.connect(...)` (mirrors
    `test_recorder_reconnect.py`'s own fake, redefined here to keep this file independent).
    """

    def __init__(self, ws=None, exc=None):
        self._ws, self._exc = ws, exc

    async def __aenter__(self):
        if self._exc is not None:
            raise self._exc
        return self._ws

    async def __aexit__(self, *a):
        return False


class _NoOpWriter:
    """A writer stub that never fails — used where a test needs SOME writer object but the
    scenario under test never depends on what it actually does (test 11 part (d): the
    never-launder CLASSIFICATION, isolated from R6's write-guard behaviour).
    """

    def write_heartbeat(self, *a, **k):
        pass

    def write_book(self, *a, **k):
        pass

    def write_trade(self, *a, **k):
        pass

    def write_counters(self, *a, **k):
        pass

    def write_gap(self, *a, **k):
        pass

    def queue_depth(self):
        return 0

    def close(self, clean=True):
        pass


# --- §7 test 1 --------------------------------------------------------------------------------

def test_gap_and_meta_records_get_a_priority_fsync(tmp_path, monkeypatch):
    """§2.1: pins durability of the three named priority call sites (gap, meta_start, meta_end).

    Correlates fsync CALLS to the gap/meta writes' own timing (not merely their eventual
    presence), with `fsync_interval_s` set absurdly large so the periodic path cannot fire and
    contaminate the count. Fails the mutation where a future implementation 'simplifies' by
    removing the gap/meta special-case and lets everything ride the periodic-only path — that
    mutation still eventually fsyncs (an "fsync happened at some point" check would pass), but it
    decouples the fsync from the gap/meta write's own timing, which this test specifically checks.
    """
    p = tmp_path / "cap.jsonl"
    fsync_calls = []
    real_fsync = capture_mod.os.fsync

    def counting_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(capture_mod.os, "fsync", counting_fsync)

    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1", fsync_interval_s=10_000.0)
    w.drain()
    n_after_start = len(fsync_calls)
    assert n_after_start >= 1, "meta_start (priority) must fsync immediately"

    w.write_book("BTC/USD", 1.0, 100.0, 100.2, 1.0, 1.0)
    w.drain()
    assert len(fsync_calls) == n_after_start, "book is NOT priority; the interval never elapses"

    w.write_trade("BTC/USD", 1.5, 100.1, 0.1, "buy")
    w.drain()
    assert len(fsync_calls) == n_after_start, "trade is NOT priority either"

    w.write_gap(GapRecord(2.0, 3.0, "disconnect", detail="x"))
    w.drain()
    assert len(fsync_calls) == n_after_start + 1, "the gap write must fsync immediately"

    w.close(clean=True)
    assert len(fsync_calls) == n_after_start + 2, "meta_end (clean close) must fsync immediately"


# --- §7 test 2 --------------------------------------------------------------------------------

def test_event_lines_are_not_individually_fsynced(tmp_path, monkeypatch):
    """§2.0 PRIMARY-fix mutation-poison test: a naive/incomplete refactor could leave the old
    per-line fsync in place for events while ADDING a redundant periodic policy elsewhere,
    satisfying "there is a periodic policy" while never actually removing the per-line cost this
    task exists to remove. Freezes the (injected) monotonic clock so the periodic deadline never
    comes due, then asserts fsync fires ZERO times across a burst of non-priority writes.
    """
    p = tmp_path / "cap.jsonl"
    fixed_clock = {"t": 1_000_000.0}
    fsync_calls = []
    real_fsync = capture_mod.os.fsync

    def counting_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(capture_mod.os, "fsync", counting_fsync)

    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1", fsync_interval_s=30.0,
                       monotonic_clock=lambda: fixed_clock["t"])
    w.drain()
    n0 = len(fsync_calls)  # meta_start's own priority fsync

    for i in range(50):
        w.write_book("BTC/USD", float(i), 100.0, 100.2, 1.0, 1.0)
        w.write_trade("BTC/USD", float(i) + 0.25, 100.1, 0.1, "buy")
        w.write_heartbeat(float(i) + 0.5)
    w.drain()

    assert len(fsync_calls) == n0, (
        "zero fsync calls across a burst of book/trade/heartbeat lines with the clock frozen — "
        "only flush() should have happened per write")
    w.close(clean=False)


# --- §7 test 3 --------------------------------------------------------------------------------

def test_flush_still_happens_once_per_record(tmp_path, monkeypatch):
    """§6's PROVISO, poison-the-weak-point: exactly one `flush()` per record, independent of the
    fsync timer, checked mid-burst — BEFORE any `close()` — so a mutation that only makes the
    data visible via `close()`'s own flush cannot hide behind "the file eventually contains it".
    """
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
    w.drain()

    flush_calls = {"n": 0}
    real_flush = w._fh.flush

    def counting_flush():
        flush_calls["n"] += 1
        return real_flush()

    monkeypatch.setattr(w._fh, "flush", counting_flush)

    n = 25
    for i in range(n):
        w.write_book("BTC/USD", float(i), 100.0, 100.2, 1.0, 1.0)
        w.write_trade("BTC/USD", float(i) + 0.25, 100.1, 0.1, "buy")
        w.write_heartbeat(float(i) + 0.5)
        w.write_counters(float(i) + 0.75, 0, 0, 0)
    w.drain()  # BEFORE any close()

    assert flush_calls["n"] == n * 4, "exactly one flush() per record, independent of the timer"

    with open(p) as fh:
        line_count = sum(1 for line in fh if line.strip())
    assert line_count == 1 + n * 4, (
        "meta_start plus every record must already be VISIBLE with the queue drained and no "
        "close() yet — fails a mutation that batches flush() together with fsync(), or one "
        "where the worker accumulates on a timer instead of draining the queue")

    w.close(clean=False)


# --- §7 test 4 --------------------------------------------------------------------------------

def test_periodic_fsync_fires_after_deadline_not_before(tmp_path, monkeypatch):
    """§2.2/§6.1: pins the cadence as a real boundary. Injected fake monotonic clock (a
    constructor parameter — §2.2's "N is a constructor parameter defaulting to 30.0" note) and a
    small N so the mechanism, not the production constant, is what is pinned.
    """
    p = tmp_path / "cap.jsonl"
    fake_clock = {"t": 1_000.0}
    fsync_calls = []
    real_fsync = capture_mod.os.fsync

    def counting_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(capture_mod.os, "fsync", counting_fsync)

    n = 5.0
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1", fsync_interval_s=n,
                       monotonic_clock=lambda: fake_clock["t"])
    w.drain()
    n0 = len(fsync_calls)  # meta_start's own priority fsync

    # Just before the deadline: no periodic fsync.
    fake_clock["t"] += (n - 0.001)
    w.write_book("BTC/USD", 1.0, 100.0, 100.2, 1.0, 1.0)
    w.drain()
    assert len(fsync_calls) == n0, "must not fire before the deadline"

    # Cross the deadline: exactly one fsync fires for this write.
    fake_clock["t"] += 1.0
    w.write_book("BTC/USD", 2.0, 100.0, 100.2, 1.0, 1.0)
    w.drain()
    assert len(fsync_calls) == n0 + 1, "exactly one fsync once the deadline passed"

    # Immediately after, clock unchanged: must NOT fire again — the deadline must have RESET,
    # not merely stayed "expired since the original start" (the mutation that degrades back to
    # per-line-after-warmup), and it must not have gone permanently silent either.
    w.write_book("BTC/USD", 3.0, 100.0, 100.2, 1.0, 1.0)
    w.drain()
    assert len(fsync_calls) == n0 + 1, "the deadline must have reset on the previous fsync"

    fake_clock["t"] += n + 0.001
    w.write_book("BTC/USD", 4.0, 100.0, 100.2, 1.0, 1.0)
    w.drain()
    assert len(fsync_calls) == n0 + 2, "and it must still fire once a FULL interval elapses again"

    w.close(clean=False)


# --- §7 test 5 --------------------------------------------------------------------------------

def test_torn_tail_tolerated_with_multikb_unflushed_gap_ahead_of_it(tmp_path):
    """§3's caveat: extends existing single-line torn-tail coverage to the shape a
    periodic-fsync writer actually produces on crash — many flushed-not-fsynced records ahead of
    one torn tail, not a single truncated line.
    """
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1", fsync_interval_s=10_000.0)
    n_well_formed = 250  # several KB of well-formed, flush()ed-but-never-fsync()ed records
    for i in range(n_well_formed):
        w.write_trade("BTC/USD", float(i), 100.0 + i * 0.001, 0.1, "buy")
    w.drain()  # guarantee every record is actually write()+flush()ed before we hand-append

    with open(p, "a") as fh:
        fh.write('{"type": "trade", "pair": "BTC/USD", "ts": 999.0, "pri')  # truncated, no \n

    meta = read_capture(str(p))
    assert meta.truncated_tail
    assert meta.n_trade == n_well_formed, "every well-formed record ahead of the tear is counted"

    report = analyze_capture(str(p), wall_now=1_000_000.0)
    assert report["truncated_tail"]
    assert report["per_pair"]["BTC/USD"]["n_trade"] == n_well_formed


# --- §7 test 6 --------------------------------------------------------------------------------

class _OldStyleSyncWriter:
    """A faithful, minimal reproduction of the PRE-913 `CaptureWriter._write` (write+flush+fsync
    inline, per line) — needed ONLY as the regression baseline for this test, since the real
    per-line-fsync code path no longer exists in `capture.py` after this task ships.
    """

    def __init__(self, path, pairs, capture_id):
        self._fh = open(path, "a", buffering=1)
        self._write({"type": "meta_start", "capture_id": capture_id, "pairs": list(pairs),
                     "format_version": capture_mod.CURRENT_FORMAT_VERSION,
                     "clock_basis": capture_mod.CLOCK_BASIS})

    def _write(self, obj):
        self._fh.write(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")
        self._fh.flush()
        capture_mod.os.fsync(self._fh.fileno())

    def write_book(self, pair, ts, bid, ask, bid_sz, ask_sz):
        self._write({"type": "book", "pair": pair, "ts": ts, "bid": bid, "ask": ask,
                     "bid_sz": bid_sz, "ask_sz": ask_sz})

    def write_trade(self, pair, ts, price, volume, side):
        self._write({"type": "trade", "pair": pair, "ts": ts, "price": price,
                     "volume": volume, "side": side})

    def close(self, ts):
        self._write({"type": "meta_end", "ts": ts})
        self._fh.close()


def test_split_feed_verdict_identical_under_periodic_fsync(tmp_path, monkeypatch):
    """Acceptance (iii)'s direct regression pin: TASK_912's split-feed detector must see
    byte-identical `per_pair`/verdict output whether the capture was written with per-line fsync
    (the old design) or priority+periodic fsync (this task) — fsync cadence changes nothing the
    detector reads (§6). Fails the mutation where periodic batching inadvertently changes
    VISIBILITY timing (e.g. batching flush() after all, or holding records in an in-process list
    instead of writing them line-by-line).
    """
    pair = "BTC/USD"
    wall_now = 1_000_000.0
    events = []
    t = wall_now - 120.0
    for i in range(20):
        events.append(("book", t, 100.0 + i * 0.01, 100.2 + i * 0.01))
        events.append(("trade", t + 0.001, 100.05 + i * 0.01))
        t += 5.0
    close_ts = t

    old_path = tmp_path / "old.jsonl"
    old = _OldStyleSyncWriter(str(old_path), [pair], "old")
    for kind, ts, *rest in events:
        if kind == "book":
            bid, ask = rest
            old.write_book(pair, ts, bid, ask, 1.0, 1.0)
        else:
            (price,) = rest
            old.write_trade(pair, ts, price, 0.1, "buy")
    old.close(ts=close_ts)

    monkeypatch.setattr(capture_mod, "_now", lambda: close_ts)
    new_path = tmp_path / "new.jsonl"
    new = CaptureWriter(str(new_path), [pair], "new")
    for kind, ts, *rest in events:
        if kind == "book":
            bid, ask = rest
            new.write_book(pair, ts, bid, ask, 1.0, 1.0)
        else:
            (price,) = rest
            new.write_trade(pair, ts, price, 0.1, "buy")
    new.close(clean=True)  # drains + writes meta_end at the SAME close_ts via monkeypatched _now

    old_report = analyze_capture(str(old_path), wall_now=wall_now)
    new_report = analyze_capture(str(new_path), wall_now=wall_now)

    assert old_report["per_pair"] == new_report["per_pair"]
    assert old_report["verdict"] == new_report["verdict"]


# --- §7 test 7 --------------------------------------------------------------------------------

def test_writer_thread_preserves_record_order_under_load(tmp_path):
    """R3, the invariant the whole gap ledger rests on. Fails the mutation where gap/meta records
    are given a second queue or written inline from the caller — either lets a gap overtake
    events it must follow, silently mis-attributing which span is unaccounted for. Also fails a
    multi-consumer worker pool (order would no longer match call order under load).
    """
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
    w.drain()

    call_order = []
    for i in range(500):
        if i % 7 == 0:
            w.write_gap(GapRecord(float(i), float(i) + 0.5, "disconnect", detail=f"g{i}"))
            call_order.append(("gap", f"g{i}"))
        else:
            w.write_book("BTC/USD", float(i), 100.0, 100.2, 1.0, 1.0)
            call_order.append(("book", float(i)))
    w.drain()

    rows = [json.loads(line) for line in open(p) if line.strip()]
    data_rows = [r for r in rows if r["type"] in ("book", "gap")]
    assert len(data_rows) == len(call_order)
    for expected, actual in zip(call_order, data_rows):
        if expected[0] == "gap":
            assert actual["type"] == "gap" and actual["detail"] == expected[1], (
                "a gap record overtook (or fell behind) the events it must follow")
        else:
            assert actual["type"] == "book" and actual["ts"] == expected[1]

    w.close(clean=False)


# --- §7 test 8 --------------------------------------------------------------------------------

def test_worker_error_is_re_raised_on_the_next_write_and_on_close(tmp_path, monkeypatch):
    """R4, fail-closed on a box with 32,220 logged `ioerr_cnt`. Fails the mutation where the
    worker catches and logs an unrecoverable error instead of latching it — the single worst
    outcome available to this design (a silently truncated capture on a failing disk), which the
    OLD synchronous code could not produce (it would have raised straight out of `_write`).
    """
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
    w.drain()

    real_fsync = capture_mod.os.fsync
    calls = {"n": 0}

    def failing_fsync(fd):
        calls["n"] += 1
        raise OSError(5, "simulated device I/O error")

    monkeypatch.setattr(capture_mod.os, "fsync", failing_fsync)

    # A gap record is priority (§2.1) -> guaranteed immediate fsync -> guaranteed failure, no
    # need to wait out the periodic interval.
    w.write_gap(GapRecord(0.0, 1.0, "disconnect", detail="trigger"))
    w.drain()  # the worker processes it, hits failing_fsync, latches the error, and dies
    assert calls["n"] == 1

    # (a) the exception surfaces on a SUBSEQUENT call from the caller's thread.
    with pytest.raises(OSError, match="simulated device I/O error"):
        w.write_book("BTC/USD", 2.0, 100.0, 100.2, 1.0, 1.0)

    # (c) the writer does NOT keep silently accepting records after the worker has died — the
    # SAME latched error surfaces again, not a fresh admission.
    with pytest.raises(OSError, match="simulated device I/O error"):
        w.write_trade("BTC/USD", 3.0, 100.0, 0.1, "buy")
    assert calls["n"] == 1, "fsync must not be attempted again — the worker is already dead"

    # (b) close() also raises rather than exiting cleanly.
    with pytest.raises(OSError, match="simulated device I/O error"):
        w.close(clean=True)


# --- §7 test 9 --------------------------------------------------------------------------------

def test_close_terminates_when_the_worker_is_blocked_and_does_not_close_the_fd(tmp_path,
                                                                                monkeypatch,
                                                                                caplog):
    """R5 and the fd-reuse hazard §2.0 names. Fails the mutation where `close()` does an
    unbounded `join()` (hangs the process one shutdown in ten, at the measured 10% duty cycle) and
    the mutation where it closes the fd out from under the still-writing worker (a cross-file
    corruption bug on the next `open()` far worse than the tail it was trying to save).
    """
    p = tmp_path / "cap.jsonl"
    release = threading.Event()
    entered = threading.Event()
    real_fsync = capture_mod.os.fsync

    def blocking_fsync(fd):
        entered.set()
        release.wait(timeout=10.0)
        return real_fsync(fd)

    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1", close_join_timeout_s=0.3)
    w.drain()  # settle meta_start with the REAL fsync before the blocking patch goes in

    monkeypatch.setattr(capture_mod.os, "fsync", blocking_fsync)
    caplog.set_level("CRITICAL", logger="907_live_fill_probe.capture")

    w.write_gap(GapRecord(0.0, 1.0, "disconnect", detail="will block"))
    assert entered.wait(timeout=5.0), "the worker must have entered fsync before close() runs"

    t0 = time.monotonic()
    w.close(clean=True)
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, "close() must not block unboundedly"
    assert not w._fh.closed, (
        "R5: must not close the fd while the worker may still be inside fsync(fd)/write(fd)")
    assert w.queue_depth() >= 1, "meta_end (and the stop sentinel) are still queued behind it"
    assert any("still blocked" in rec.message for rec in caplog.records), (
        "the undrained-record situation must be logged loudly")

    release.set()
    w._worker.join(timeout=5.0)
    assert not w._worker.is_alive(), "the daemon thread must still finish once unblocked"


# --- §7 test 10 -------------------------------------------------------------------------------

def test_queue_overflow_discloses_a_writer_overflow_gap_on_the_falling_edge(tmp_path,
                                                                             monkeypatch):
    """§2.0.2's ONE pinned mechanism — the one place this design can lose data that the old one
    could not. Fails the mutation where overflow does `except queue.Full: pass` (the 893 failure
    mode reintroduced through a new door) and equally the mutation where it quietly takes the
    rejected unconditional-fail-closed branch instead.
    """
    p = tmp_path / "cap.jsonl"
    real_fsync = capture_mod.os.fsync

    release = threading.Event()
    entered = threading.Event()

    def blocking_fsync(fd):
        entered.set()
        release.wait(timeout=10.0)
        return real_fsync(fd)

    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1", queue_maxsize=4)
    w.drain()  # settle meta_start with the real fsync first

    monkeypatch.setattr(capture_mod.os, "fsync", blocking_fsync)

    # Block the worker inside fsync via one priority (gap) write. The admission gate only cares
    # about qsize(), not what the worker is currently doing with an item it already dequeued.
    w.write_gap(GapRecord(0.0, 0.5, "disconnect", detail="blocker"))
    assert entered.wait(timeout=5.0), "the worker must be blocked before we probe admission"

    # maxsize=4, OVERFLOW_RESERVE=1 -> the admission line is qsize() < 3: exactly 3 ordinary
    # records admit.
    t_admit = [10.0, 11.0, 12.0]
    for ts in t_admit:
        t0 = time.monotonic()
        w.write_book("BTC/USD", ts, 100.0, 100.2, 1.0, 1.0)
        assert time.monotonic() - t0 < 1.0, "admission must never block (R1)"

    assert w.queue_depth() == 4 - OVERFLOW_RESERVE, (
        "the reserved slot must never be consumed by an ordinary record — fails the mutation "
        "where the admission line is written as `maxsize` and the disclosure has nowhere to go")

    # (a) push past the bound: these must all return normally — no raise, no block.
    dropped_ts = [20.0, 21.0, 22.0]
    for ts in dropped_ts:
        t0 = time.monotonic()
        w.write_book("BTC/USD", ts, 200.0, 200.2, 1.0, 1.0)
        assert time.monotonic() - t0 < 1.0, "an over-bound write must not block (R1)"

    # (b) still capped at maxsize - 1 while blocked — the dropped records never touched the queue.
    assert w.queue_depth() == 4 - OVERFLOW_RESERVE

    # (c)/(d) release the worker, drain the backlog deterministically, THEN admit one more
    # record — the falling edge — and check the disclosure it produces.
    release.set()
    w.drain()
    w.write_book("BTC/USD", 30.0, 300.0, 300.2, 1.0, 1.0)
    w.drain()

    rows = [json.loads(line) for line in open(p) if line.strip()]
    overflow_gaps = [r for r in rows if r["type"] == "gap" and r["reason"] == "writer_overflow"]
    assert len(overflow_gaps) == 1
    g = overflow_gaps[0]
    assert g["start_ts"] == dropped_ts[0]
    assert g["end_ts"] == dropped_ts[-1]
    assert str(len(dropped_ts)) in g["detail"]

    idx_gap = rows.index(g)
    idx_last_before = next(i for i, r in enumerate(rows) if r.get("ts") == t_admit[-1])
    idx_first_after = next(i for i, r in enumerate(rows) if r.get("ts") == 30.0)
    assert idx_last_before < idx_gap < idx_first_after, (
        "the disclosure must sit strictly between the last admitted record before the episode "
        "and the first admitted record after it (R3)")

    book_ts = [r["ts"] for r in rows if r["type"] == "book"]
    assert book_ts == t_admit + [30.0], "the dropped records must never reach the file at all"

    # (e) a second, SEPARATE overflow episode must produce a second, separate coalesced gap —
    # never merged with the first, and never one gap per dropped record.
    release2 = threading.Event()
    entered2 = threading.Event()

    def blocking_fsync2(fd):
        entered2.set()
        release2.wait(timeout=10.0)
        return real_fsync(fd)

    monkeypatch.setattr(capture_mod.os, "fsync", blocking_fsync2)
    w.write_gap(GapRecord(40.0, 40.5, "disconnect", detail="blocker2"))
    assert entered2.wait(timeout=5.0)

    t_admit2 = [41.0, 42.0, 43.0]
    for ts in t_admit2:
        w.write_book("BTC/USD", ts, 400.0, 400.2, 1.0, 1.0)
    assert w.queue_depth() == 4 - OVERFLOW_RESERVE

    dropped_ts2 = [44.0, 45.0]
    for ts in dropped_ts2:
        w.write_book("BTC/USD", ts, 500.0, 500.2, 1.0, 1.0)
    assert w.queue_depth() == 4 - OVERFLOW_RESERVE

    release2.set()
    w.drain()
    w.write_book("BTC/USD", 46.0, 600.0, 600.2, 1.0, 1.0)
    w.drain()

    rows2 = [json.loads(line) for line in open(p) if line.strip()]
    overflow_gaps2 = [r for r in rows2 if r["type"] == "gap" and r["reason"] == "writer_overflow"]
    assert len(overflow_gaps2) == 2, (
        "two separate episodes must produce two separate coalesced gaps — not one merged span, "
        "and not one gap per dropped record")
    g2 = overflow_gaps2[1]
    assert g2["start_ts"] == dropped_ts2[0]
    assert g2["end_ts"] == dropped_ts2[-1]

    w.close(clean=False)


# --- §7 test 11 -------------------------------------------------------------------------------

def test_write_failure_inside_run_s_own_handler_shuts_down_without_escalating(tmp_path,
                                                                               monkeypatch):
    """R6 (§2.0.1) — the case the draft left as a TODO and that test 8 does not reach (test 8
    exercises an isolated `_write`/`close()` raise, never a raise triggered from INSIDE `run()`'s
    own exception handler, which is the path an unrecoverable EIO actually takes).

    Fails the mutation where `run()`'s handler leaves `_record_gap` unguarded (the pre-913
    shape): the write error would re-raise from inside the `except` body, escape `run()`, then a
    THIRD raise out of `main()`'s `finally` would strand the writer's worker thread unjoined.
    Also fails the lazier mutation where the handler swallows the write error and lets the
    reconnect loop continue writing into a dead writer.
    """
    p = tmp_path / "cap.jsonl"

    def always_raise_fsync(fd):
        raise OSError(5, "simulated device I/O error")

    # Patch BEFORE construction, so `meta_start`'s OWN priority fsync (§2.1) is what fails: its
    # write()+flush() succeed (the bytes land in the file) but the fsync attempt raises and
    # latches. Draining guarantees the writer is provably dead by the time `run()` does anything
    # — and the file contains exactly one line (meta_start), with no synthetic extra record
    # needed to prime the failure.
    monkeypatch.setattr(capture_mod.os, "fsync", always_raise_fsync)
    writer = CaptureWriter(str(p), ["BTC/USD"], "capture-t11")
    writer.drain()
    assert writer._worker_error is not None, "sanity: the writer must already be dead"

    def connect_raises_transport_error(url):
        return _AsyncCM(exc=ConnectionError("simulated transport failure"))

    sleep_calls = {"n": 0}

    async def counting_sleep(seconds):
        sleep_calls["n"] += 1

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path="unused", capture_id="t11",
                          heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_raises_transport_error, writer=writer,
                        sleep_fn=counting_sleep)

    with pytest.raises(CaptureWriteFailed) as excinfo:
        asyncio.run(rec.run(max_iterations=1))
    exc = excinfo.value

    # (a) exactly one exception escapes, and it is CaptureWriteFailed, not a bare OSError.
    assert isinstance(exc.__cause__, OSError), "the original write fault must be __cause__"
    assert "simulated transport failure" in str(exc), (
        "the transport exception it was trying to disclose must be named in the message")

    # (b) no retry against a dead writer.
    assert rec._stop is True
    assert sleep_calls["n"] == 0, "no backoff/retry must happen once the write path is dead"

    # (c) the honest negative: no gap record for the disconnect, and the last line is the last
    # record actually written before the fault.
    rows = [json.loads(line) for line in open(p) if line.strip()]
    assert all(r["type"] != "gap" for r in rows), (
        "no gap record for the disconnect — the write path was already dead when it was tried")
    assert rows[-1]["type"] == "meta_start", "the last line is the last record before the fault"

    # (d) CaptureWriteFailed is re-raised, not laundered, if it reaches the never-launder clause
    # — isolated from R6's write-guard behaviour with a writer that never fails, so this checks
    # ONLY the classification tuple at recorder.py's broad except.
    def connect_raises_capture_write_failed(url):
        raise CaptureWriteFailed("synthetic — some other path raising it inside the try body")

    sleep_calls2 = {"n": 0}

    async def counting_sleep2(seconds):
        sleep_calls2["n"] += 1

    cfg2 = RecorderConfig(pairs=["BTC/USD"], out_path="unused", capture_id="t11d",
                           heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec2 = LiveRecorder(cfg2, connect_raises_capture_write_failed, writer=_NoOpWriter(),
                         sleep_fn=counting_sleep2)
    with pytest.raises(CaptureWriteFailed):
        asyncio.run(rec2.run(max_iterations=5))
    assert sleep_calls2["n"] == 0, (
        "CaptureWriteFailed must propagate directly — laundering it would retry with a backoff")

    # (e) close() must still reach CaptureWriter.close() (worker stop-signalled and joined) even
    # though _flush_counters raises on the way in, and the latched write error must still surface
    # rather than close() exiting cleanly (R4). This test asserts NOTHING about R6's CRITICAL log
    # — it takes no caplog fixture — so it does not certify the "undrained count reported" half;
    # `test_R6_critical_log_names_the_errno_the_transport_error_and_the_undrained_depth` below is
    # the guard for that, against a NON-ZERO depth.
    with pytest.raises(OSError, match="simulated device I/O error"):
        rec.close(clean=True)
    assert writer._fh.closed, (
        "CaptureWriter.close() was reached and ran to completion despite _flush_counters "
        "raising on the way in — the R6 point-3 reorder")


# --- Gate 4 (orchestrator brief): R1 proven, not asserted --------------------------------------

def test_R1_fsync_never_issued_on_the_calling_thread(tmp_path, monkeypatch):
    """R1, proven rather than asserted: whatever thread calls `write_*`/`write_gap`/`close` must
    NEVER be the thread `os.fsync()` actually runs on. Fails the mutation where a future
    "simplification" calls `os.fsync()` inline from `_write`/`_enqueue` on the caller's thread —
    every count-based test above would still pass (the call still happens, at the same times),
    but this is the one that would actually catch WHO makes it.
    """
    p = tmp_path / "cap.jsonl"
    caller_thread_ident = threading.get_ident()
    offending = []
    real_fsync = capture_mod.os.fsync

    def checked_fsync(fd):
        if threading.get_ident() == caller_thread_ident:
            offending.append(threading.current_thread().name)
        return real_fsync(fd)

    monkeypatch.setattr(capture_mod.os, "fsync", checked_fsync)

    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1")  # meta_start's own priority fsync
    w.drain()
    w.write_gap(GapRecord(0.0, 1.0, "disconnect", detail="x"))  # priority -> guaranteed fsync
    w.drain()
    w.close(clean=True)  # meta_end -> another guaranteed priority fsync

    assert not offending, f"os.fsync() ran on the calling thread — R1 violated: {offending}"


# --- Gate 5 (orchestrator brief): old files with clock_jump / book_checksum_mismatch parse -----

def test_old_gap_reasons_clock_jump_and_book_checksum_mismatch_still_parse(tmp_path):
    """`GAP_REASONS` is additive-only (§5, §2.0.2 point 7): a capture written by an OLDER
    recorder using `clock_jump`, or one containing `book_checksum_mismatch` (a sibling reason,
    unrelated to this rename, added by run-45's Addendum A), must still read back without
    raising `CaptureFormatError`.
    """
    p = tmp_path / "cap.jsonl"
    with open(p, "w") as fh:
        fh.write(json.dumps({"type": "meta_start", "capture_id": "old", "pairs": ["BTC/USD"],
                              "format_version": 2, "clock_basis": "local_receipt"}) + "\n")
        fh.write(json.dumps({"type": "trade", "pair": "BTC/USD", "ts": 1.0, "price": 1.0,
                              "volume": 1.0, "side": "buy"}) + "\n")
        fh.write(json.dumps({"type": "gap", "start_ts": 1.0, "end_ts": 2.0, "duration_s": 1.0,
                              "reason": "clock_jump", "detail": "old-format"}) + "\n")
        fh.write(json.dumps({"type": "trade", "pair": "BTC/USD", "ts": 2.0, "price": 1.0,
                              "volume": 1.0, "side": "buy"}) + "\n")
        fh.write(json.dumps({"type": "gap", "start_ts": 2.0, "end_ts": 3.0, "duration_s": 1.0,
                              "reason": "book_checksum_mismatch", "detail": "old-format"}) + "\n")
        fh.write(json.dumps({"type": "trade", "pair": "BTC/USD", "ts": 3.0, "price": 1.0,
                              "volume": 1.0, "side": "buy"}) + "\n")
        fh.write(json.dumps({"type": "meta_end", "ts": 3.0}) + "\n")

    meta = read_capture(str(p))  # must not raise
    assert {g.reason for g in meta.gaps} == {"clock_jump", "book_checksum_mismatch"}


# ==================================================================================================
# Run-55 cycle-2 fix pass: the five findings the verification panel raised against cycle 1.
# Each test below is written to FAIL against the cycle-1 artifact (D83(i): the poison is planted
# where the mechanism is BLIND, not where it is strong — that is precisely why the two most
# serious of these shipped past thirteen green tests).
# ==================================================================================================

def _park_the_worker_in_fsync(monkeypatch, entered, release):
    """Patch `os.fsync` so the writer thread blocks inside it until `release` is set, signalling
    `entered` on the way in. The queue-admission gate only reads `qsize()`, so a worker parked on
    an item it has ALREADY dequeued is how every overflow scenario here is built deterministically
    (no sleeps — §7's determinism note).
    """
    real_fsync = capture_mod.os.fsync

    def blocking_fsync(fd):
        entered.set()
        release.wait(timeout=20.0)
        return real_fsync(fd)

    monkeypatch.setattr(capture_mod.os, "fsync", blocking_fsync)


def _build_an_active_overflow_episode(w, n_admitted, n_dropped, base_ts=100.0):
    """Fill `w`'s queue to the admission line and then push `n_dropped` records past it, leaving
    an overflow episode ACTIVE (unresolved) and the queue still full. Returns the dropped ts list.
    """
    for i in range(n_admitted):
        w.write_book("BTC/USD", base_ts + i, 100.0, 100.2, 1.0, 1.0)
    assert w.queue_depth() == w._queue.maxsize - OVERFLOW_RESERVE
    dropped_ts = [base_ts + 100.0 + i for i in range(n_dropped)]
    for ts in dropped_ts:
        w.write_book("BTC/USD", ts, 200.0, 200.2, 1.0, 1.0)
    assert w.queue_depth() == w._queue.maxsize - OVERFLOW_RESERVE, (
        "the dropped records must never have touched the queue")
    assert w._overflow_active, "sanity: the episode must be ACTIVE going into close()"
    return dropped_ts


# --- BLOCKER 1 ---------------------------------------------------------------------------------

def test_close_during_an_active_overflow_episode_discloses_it_instead_of_dropping_silently(
        tmp_path, monkeypatch):
    """BLOCKER (run-55 cycle-2). §2.0.2's binding rule — "on overflow the implementation MUST NOT
    drop silently" — applied to the ONE path §7 test 10 never exercises: an episode still ACTIVE
    at close() time.

    Test 10 only ever resolves an episode with a SUBSEQUENT `write_book`, i.e. through the normal
    admission gate, which is the only place cycle 1 called `_enqueue_overflow_gap`. `close()`'s
    own `meta_end` goes through that same gate, so at a full queue it was itself folded into the
    drop count, and the `_STOP` sentinel — not a record — resolved nothing. Measured against the
    cycle-1 artifact: nine dropped books PLUS meta_end vanished with no gap, no exception and no
    log naming them. That is the 893 failure mode (missing data presented as observed time)
    coming back through a new door.

    Fails against cycle 1 on both counts below: no `writer_overflow` gap and no `meta_end`.
    """
    p = tmp_path / "cap.jsonl"
    release, entered = threading.Event(), threading.Event()

    w = CaptureWriter(str(p), ["BTC/USD"], "cap-close-overflow", queue_maxsize=4)
    w.drain()  # settle meta_start with the REAL fsync before the blocking patch goes in
    _park_the_worker_in_fsync(monkeypatch, entered, release)

    w.write_gap(GapRecord(0.0, 0.5, "disconnect", detail="parks the worker"))
    assert entered.wait(timeout=5.0)

    dropped_ts = _build_an_active_overflow_episode(w, n_admitted=3, n_dropped=9)

    # Release the worker from a helper thread the instant close() starts, so close()'s BOUNDED
    # wait for meta_end admission can be satisfied and everything enqueued still reaches the file.
    # Correctness does not depend on this race resolving either way: `_overflow_active` is only
    # ever cleared by an admitted record or by close() itself, and nothing else writes here.
    closing = threading.Event()

    def _releaser():
        closing.wait(timeout=10.0)
        release.set()

    t = threading.Thread(target=_releaser, daemon=True)
    t.start()
    closing.set()

    t0 = time.monotonic()
    w.close(clean=True)
    assert time.monotonic() - t0 < 10.0, "close() must stay bounded (R5)"
    release.set()
    w._worker.join(timeout=10.0)

    rows = [json.loads(line) for line in open(p) if line.strip()]
    types = [r["type"] for r in rows]

    overflow_gaps = [r for r in rows
                     if r["type"] == "gap" and r.get("reason") == "writer_overflow"]
    assert len(overflow_gaps) == 1, (
        f"the episode active at close() must be disclosed exactly once; record types were "
        f"{types!r}")
    g = overflow_gaps[0]
    assert g["start_ts"] == dropped_ts[0]
    assert g["end_ts"] == dropped_ts[-1]
    assert str(len(dropped_ts)) in g["detail"], "the disclosure must name the dropped count"

    assert types[-1] == "meta_end", (
        f"meta_end must survive the close-time disclosure, not be folded into the drop count; "
        f"record types were {types!r}")
    assert rows.index(g) < len(rows) - 1, "the disclosure must precede meta_end (R3)"

    # The reserve invariant §2.0.2 point 2 rests on is NOT enlarged by this fix.
    assert OVERFLOW_RESERVE == 1
    assert read_capture(str(p)).closed_cleanly is True


def test_close_that_cannot_admit_meta_end_says_so_loudly_instead_of_silently(tmp_path,
                                                                              monkeypatch,
                                                                              caplog):
    """The DEGRADE half of the same BLOCKER. If the worker never comes back, `meta_end` genuinely
    cannot be admitted and `closed_cleanly` reading False is the honest answer — but silence is
    the defect. R5's bound must still hold, and the disclosure must NAME the undrained depth and
    the dropped count.

    Fails against cycle 1, which produced no such log at all: the only line it emitted on this
    path was the pre-existing "still blocked" one, which says nothing about the ten records that
    just disappeared.
    """
    p = tmp_path / "cap.jsonl"
    release, entered = threading.Event(), threading.Event()

    w = CaptureWriter(str(p), ["BTC/USD"], "cap-close-loud", queue_maxsize=4,
                       close_join_timeout_s=0.3)
    w.drain()
    _park_the_worker_in_fsync(monkeypatch, entered, release)
    caplog.set_level("CRITICAL", logger="907_live_fill_probe.capture")

    w.write_gap(GapRecord(0.0, 0.5, "disconnect", detail="parks the worker"))
    assert entered.wait(timeout=5.0)
    _build_an_active_overflow_episode(w, n_admitted=3, n_dropped=9)

    t0 = time.monotonic()
    w.close(clean=True)
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, f"R5: close() must stay bounded; took {elapsed:.2f}s"

    messages = [rec.getMessage() for rec in caplog.records if rec.levelname == "CRITICAL"]
    meta_end_lines = [m for m in messages if "meta_end" in m]
    assert meta_end_lines, (
        f"a meta_end that could not be admitted MUST be disclosed loudly; CRITICAL lines were "
        f"{messages!r}")
    line = meta_end_lines[0]
    assert "undrained" in line
    assert "9 record(s) already dropped" in line, (
        f"the log must name the dropped count, not just the fact of a degrade: {line!r}")
    assert "closed_cleanly will read False" in line

    release.set()
    w._worker.join(timeout=10.0)
    assert read_capture(str(p)).closed_cleanly is False, "the honest answer, loudly disclosed"


# --- MAJOR 2 -----------------------------------------------------------------------------------

def test_a_construction_failure_strands_no_worker_thread_fd_or_flock(tmp_path):
    """MAJOR (run-55 cycle-2). `__init__` used to start the worker BEFORE the validation and
    serialization that can raise. The thread's target is the bound method `self._worker_loop`, so
    a started thread keeps `self` and `self._fh` alive; with nothing enqueued and no `_STOP`
    sent, it parked on `self._queue.get()` for the life of the process, holding the fd AND the
    exclusive flock.

    Reachable in practice, which is why this is not theoretical: `--heartbeat-timeout-s` is
    `type=float` with no range check and `RecorderConfig.__post_init__` never validates it, so an
    operator typo `nan` reaches `_declaration_fields` -> `_require_finite` -> `ValueError`.

    The blindest spot, and the assertion that fails against cycle 1: the SECOND writer. A leaked
    worker self-deadlocks the process against its own flock — "another process already holds the
    exclusive capture lock" when no other process exists.
    """
    p = tmp_path / "cap.jsonl"
    before = {t.ident for t in threading.enumerate()}

    with pytest.raises(ValueError, match="heartbeat_timeout_s"):
        CaptureWriter(str(p), ["BTC/USD"], "cap-nan", heartbeat_timeout_s=float("nan"))

    leaked = [t for t in threading.enumerate()
              if t.ident not in before and t.name == "CaptureWriter-worker"]
    assert not leaked, (
        f"a failed construction must strand no worker thread; leaked {[t.name for t in leaked]}")

    # The flock and the fd must be gone too — proven by USING them, not by inspecting them.
    w2 = CaptureWriter(str(p), ["BTC/USD"], "cap-second")
    try:
        w2.write_book("BTC/USD", 1.0, 100.0, 100.2, 1.0, 1.0)
        w2.drain()
    finally:
        w2.close(clean=True)
    rows = [json.loads(line) for line in open(p) if line.strip()]
    assert rows[0]["type"] == "meta_start", (
        "the failed construction must have written nothing at all")
    assert [r["type"] for r in rows] == ["meta_start", "book", "meta_end"]


# --- MAJOR 3 -----------------------------------------------------------------------------------

def test_drain_returns_promptly_when_the_worker_died_with_a_backlog_behind_it(tmp_path,
                                                                               monkeypatch):
    """MAJOR (run-55 cycle-2), and D83(i) applied to an instrument's own prose. On a fatal write
    error the worker calls `task_done()` only for the item that RAISED and returns; everything
    still queued is never dequeued and never marked done, so `unfinished_tasks` never reaches
    zero and cycle 1's bare `self._queue.join()` blocked forever — while its docstring claimed it
    "block[s] until the worker has applied every record enqueued so far".

    Production is NOT exposed (`CaptureWriter.close()` joins the worker THREAD, not the queue).
    This is a harness hazard: the next test author writing the natural "burst of writes, inject a
    failure, then `drain()`" pattern gets a silent indefinite hang instead of a fast legible
    failure. `drain()` is run on a helper thread here precisely so the hang is a FAILURE rather
    than a hung suite.
    """
    p = tmp_path / "cap.jsonl"
    release, entered = threading.Event(), threading.Event()

    def blocking_then_failing_fsync(fd):
        entered.set()
        release.wait(timeout=20.0)
        raise OSError(5, "simulated device I/O error")

    w = CaptureWriter(str(p), ["BTC/USD"], "cap-drain")
    w.drain()  # settle meta_start with the real fsync
    monkeypatch.setattr(capture_mod.os, "fsync", blocking_then_failing_fsync)

    # A priority record parks the worker inside fsync; the backlog piles up BEHIND it.
    w.write_gap(GapRecord(0.0, 1.0, "disconnect", detail="the item that will raise"))
    assert entered.wait(timeout=5.0)
    w.write_book("BTC/USD", 2.0, 100.0, 100.2, 1.0, 1.0)
    w.write_book("BTC/USD", 3.0, 100.0, 100.2, 1.0, 1.0)
    assert w.queue_depth() == 2

    release.set()                       # the worker now raises, latches (R4) and returns
    w._worker.join(timeout=10.0)
    assert not w._worker.is_alive()
    assert w._worker_error is not None

    finished = threading.Event()
    raised = []

    def _call_drain():
        try:
            w.drain()
        except BaseException as exc:    # noqa: BLE001 — recorded, not swallowed
            raised.append(exc)
        finally:
            finished.set()

    t = threading.Thread(target=_call_drain, daemon=True)
    t.start()
    assert finished.wait(timeout=5.0), (
        "drain() blocked on a worker that will never dequeue again — `queue.join()` on a dead "
        "worker never returns, which is a silent indefinite hang, not a test failure")
    assert not raised, f"drain() must return, not raise, on this path: {raised!r}"

    # It must NOT have discarded the backlog: that count is what R6's CRITICAL log and R5's
    # close() disclose as "undrained", and zeroing it would turn an honest number into a lie.
    assert w.queue_depth() == 2, "the undrained backlog must still be reported honestly"

    # R4 is untouched: the latched fault still surfaces on the next caller-thread call.
    with pytest.raises(OSError, match="simulated device I/O error"):
        w.write_book("BTC/USD", 4.0, 100.0, 100.2, 1.0, 1.0)
    with pytest.raises(OSError, match="simulated device I/O error"):
        w.close(clean=True)


# --- MAJOR 4 -----------------------------------------------------------------------------------

def test_meta_resume_is_priority_fsynced_when_no_resume_gap_follows_it(tmp_path, monkeypatch):
    """MAJOR (run-55 cycle-2). §2.1 enumerates "only three points", but its STATED RATIONALE for
    `meta_start` — written once per process lifetime, carries the declared pair/channel set
    `assert_complete` depends on — applies to `meta_resume` verbatim: same `_declaration_fields()`
    call, added by B1/run-45 specifically so a session-2 config change is not silently lost.

    THE BRANCH THIS PINS, and why the obvious test would certify nothing. `os.fsync(fd)` syncs
    the whole file, so in the common case the resume gap written immediately after `meta_resume`
    carries a priority fsync that covers the preceding `meta_resume` line FOR FREE — a test on
    that case passes with or without the fix. The real hole is the branch where NO resume gap is
    written (`if last_ts is not None and now > last_ts`): a previous session that crashed before
    recording any data event at all — repeated failed reconnects, or a kill before the first
    frame. Measured against cycle 1: `fsync calls during session2 init: 0`.
    """
    p = tmp_path / "cap.jsonl"

    # Session 1: meta_start only — NO book/trade/heartbeat/gap, so `_last_event_ts` returns None
    # and session 2 writes no resume gap to mask the missing fsync.
    w1 = CaptureWriter(str(p), ["BTC/USD"], "cap-s1", channels=["book"])
    w1.drain()
    w1.close(clean=False)
    assert [json.loads(line)["type"] for line in open(p) if line.strip()] == ["meta_start"]

    fsync_calls = []
    real_fsync = capture_mod.os.fsync

    def counting_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(capture_mod.os, "fsync", counting_fsync)

    # Session 2: fsync_interval_s absurdly large, so the periodic path cannot contaminate the
    # count — every fsync observed here is a PRIORITY one.
    w2 = CaptureWriter(str(p), ["BTC/USD"], "cap-s1", channels=["book"],
                        fsync_interval_s=10_000.0)
    w2.drain()
    n_init = len(fsync_calls)

    rows = [json.loads(line) for line in open(p) if line.strip()]
    assert [r["type"] for r in rows] == ["meta_start", "meta_resume"], (
        "this test is only meaningful on the NO-resume-gap branch — a gap here would mask the "
        "missing fsync and the test would certify nothing")

    assert n_init >= 1, (
        "meta_resume must be priority-fsynced: with no resume gap behind it, a session-2 startup "
        "otherwise issues ZERO fsyncs and the declared pair/channel set `assert_complete` reads "
        "is not durable at all")

    # And it is genuinely the PRIORITY path, not an incidental one: a following non-priority
    # record must add no fsync at this interval.
    w2.write_book("BTC/USD", 1.0, 100.0, 100.2, 1.0, 1.0)
    w2.drain()
    assert len(fsync_calls) == n_init, "book is not priority; the interval never elapses"
    w2.close(clean=True)


def test_the_priority_record_type_set_is_exactly_the_four_documented_types():
    """Pins the DELIBERATE divergence from §2.1's literal "only three points" — four types, with
    `meta_resume` added for §2.1's own stated reason. A future reader reconciling code against
    the spec text must find the divergence asserted, not discover it as a surprise.
    """
    assert capture_mod._PRIORITY_RECORD_TYPES == frozenset(
        {"gap", "meta_start", "meta_resume", "meta_end"})


# --- MAJOR 5 -----------------------------------------------------------------------------------

def _writer_dead_with_a_backlog(path, monkeypatch, n_backlog=3):
    """A `CaptureWriter` whose worker has latched a fatal fsync error while `n_backlog` records
    remain queued behind it — i.e. `queue_depth()` is NON-ZERO and the next write raises (R4).

    The non-zero depth is the whole point (D83(i)): an R6-log assertion taken against a depth of
    zero would pass equally against a hard-coded 0, certifying nothing about the number the
    operator actually needs.
    """
    release, entered = threading.Event(), threading.Event()

    def blocking_then_failing_fsync(fd):
        entered.set()
        release.wait(timeout=20.0)
        raise OSError(5, "simulated device I/O error")

    w = CaptureWriter(str(path), ["BTC/USD"], "cap-r6")
    w.drain()
    monkeypatch.setattr(capture_mod.os, "fsync", blocking_then_failing_fsync)
    w.write_gap(GapRecord(0.0, 1.0, "disconnect", detail="the item that will raise"))
    assert entered.wait(timeout=5.0)
    for i in range(n_backlog):
        w.write_book("BTC/USD", 2.0 + i, 100.0, 100.2, 1.0, 1.0)
    release.set()
    w._worker.join(timeout=10.0)
    assert w._worker_error is not None
    assert w.queue_depth() == n_backlog
    return w


def test_R6_critical_log_names_the_errno_the_transport_error_and_the_undrained_depth(
        tmp_path, monkeypatch, caplog):
    """MAJOR (run-55 cycle-2). §2.0.1 R6 point 1 requires the CRITICAL log to name (i) the write
    error including its `errno`, (ii) the transport exception, and (iii) the writer's undrained
    queue depth. That log is the ONLY place those numbers reach a human: §2.0.1's disclosure is
    explicitly out-of-band and deferred, because nothing further can reach the file.

    Cycle 1 shipped it with ZERO coverage while §7 test 11's docstring claimed "(e)/(f) …
    undrained count reported" — test 11 takes no `caplog` fixture and asserts nothing about the
    log. Deleting the `logger.critical` call left all 317 tests green. This is the guard.
    """
    p = tmp_path / "cap.jsonl"
    writer = _writer_dead_with_a_backlog(p, monkeypatch, n_backlog=3)

    def connect_raises_transport_error(url):
        return _AsyncCM(exc=ConnectionError("simulated transport failure"))

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path="unused", capture_id="r6log",
                          heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_raises_transport_error, writer=writer,
                        sleep_fn=_fast_sleep)

    caplog.set_level("CRITICAL", logger="907_live_fill_probe.recorder")
    with pytest.raises(CaptureWriteFailed):
        asyncio.run(rec.run(max_iterations=1))

    criticals = [r.getMessage() for r in caplog.records
                 if r.levelname == "CRITICAL" and "WRITE PATH DEAD" in r.getMessage()]
    assert len(criticals) == 1, (
        f"R6 point 1's CRITICAL disclosure must be emitted exactly once; got {criticals!r}")
    line = criticals[0]
    assert "errno=5" in line, f"(i) the write error's errno must be named: {line!r}"
    assert "simulated device I/O error" in line, f"(i) the write error itself: {line!r}"
    assert "simulated transport failure" in line, f"(ii) the transport exception: {line!r}"
    assert "undrained queue depth: 3" in line, (
        f"(iii) the undrained depth must be the REAL one, not zero: {line!r}")

    with pytest.raises(OSError, match="simulated device I/O error"):
        writer.close(clean=False)


# --- MINOR 6 -----------------------------------------------------------------------------------

def test_an_enormous_exception_repr_is_bounded_in_the_log_and_in_the_raised_message(
        tmp_path, monkeypatch, caplog):
    """MINOR (run-55 cycle-2). Both R6 disclosures interpolate `%r`/`{!r}` of exceptions with no
    size bound, and `CaptureWriteFailed` propagates out of `main()` uncaught into the process
    traceback. A multi-megabyte exception message reached both verbatim.

    Stated fairly: cycle 1 built exactly what R6 point 1 mandates — this is a defect in the
    REQUIREMENT, not a deviation from it. The disclosure is what matters, not the megabytes.
    """
    p = tmp_path / "cap.jsonl"
    writer = _writer_dead_with_a_backlog(p, monkeypatch, n_backlog=1)

    huge = "X" * 500_000

    def connect_raises_enormous_transport_error(url):
        return _AsyncCM(exc=ConnectionError(f"simulated transport failure {huge}"))

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path="unused", capture_id="r6big",
                          heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_raises_enormous_transport_error, writer=writer,
                        sleep_fn=_fast_sleep)

    caplog.set_level("CRITICAL", logger="907_live_fill_probe.recorder")
    with pytest.raises(CaptureWriteFailed) as excinfo:
        asyncio.run(rec.run(max_iterations=1))

    message = str(excinfo.value)
    assert len(message) < 10_000, (
        f"an unbounded exception repr reached the uncaught process traceback: {len(message)} "
        f"chars")
    assert "truncated" in message, "the truncation must be EXPLICIT, not silent"
    assert "simulated transport failure" in message, "the disclosure itself must survive"

    logged = [r.getMessage() for r in caplog.records if "WRITE PATH DEAD" in r.getMessage()]
    assert len(logged) == 1
    assert len(logged[0]) < 10_000, f"the log line is unbounded: {len(logged[0])} chars"
    assert "truncated" in logged[0]
    assert "errno=5" in logged[0], "bounding must not cost the disclosure its content"
    assert "undrained queue depth: 1" in logged[0]

    with pytest.raises(OSError, match="simulated device I/O error"):
        writer.close(clean=False)


# --- MINOR 7 -----------------------------------------------------------------------------------

def test_recorder_close_degrades_on_an_UNRELATED_flush_counters_error_without_masking_it(
        tmp_path, caplog):
    """MINOR (run-55 cycle-2). R6 point 3's reorder collapses ANY `_flush_counters` exception into
    "the writer is dead" + `clean=False`, where the pre-913 code crashed loudly. That is what R6
    point 3 mandates and it is a disclosed choice — but nothing exercised it with an UNRELATED
    exception, so nothing showed the degrade is SAFE rather than a mask over a real bug.

    The property that makes it safe is that the degrade is not silent (a CRITICAL with the
    traceback) and not lossy of shutdown (`CaptureWriter.close` still runs, with `clean=False`
    so `closed_cleanly` honestly reads False). Both are asserted here.
    """
    calls = {"close_clean": None, "n_close": 0}

    class _WriterWithABuggyCounterPath(_NoOpWriter):
        def write_counters(self, *a, **k):
            # Deliberately NOT an OSError and NOT the latched write fault: a plain programming
            # bug, the case R6 point 3's broad `except Exception` could mask.
            raise ValueError("unrelated bug in the counters path, not a dead write path")

        def close(self, clean=True):
            calls["close_clean"] = clean
            calls["n_close"] += 1

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path="unused", capture_id="degrade",
                          heartbeat_timeout_s=5.0)
    rec = LiveRecorder(cfg, lambda url: _AsyncCM(), writer=_WriterWithABuggyCounterPath(),
                        sleep_fn=_fast_sleep)

    caplog.set_level("CRITICAL", logger="907_live_fill_probe.recorder")
    rec.close(clean=True)   # must NOT raise — R6 point 3

    assert calls["n_close"] == 1, "CaptureWriter.close() must always run (R6 point 3)"
    assert calls["close_clean"] is False, (
        "no meta_end is possible once the flush failed; closed_cleanly must read False")

    criticals = [r for r in caplog.records if r.levelname == "CRITICAL"]
    assert criticals, "the degrade must be DISCLOSED, not silent — that is what keeps it honest"
    assert any("failed to flush counters" in r.getMessage() for r in criticals)
    assert any(r.exc_info is not None for r in criticals), (
        "the unrelated exception's traceback must reach the log, or a genuine bug IS masked")

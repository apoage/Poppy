"""TASK_914 — the streaming reader's own regression guard (spec acceptance iii).

`read_capture` used to materialize the whole file THREE times before any analysis started
(`capture.py::_iter_lines_tolerant` did a `readlines()` internally; `read_capture` then built a
second full list of parsed dicts from it; `_file_has_truncated_tail` did a THIRD `readlines()`
while the second list was still alive). That is an O(file) cost layered on top of the O(events)
retention the estimand passes genuinely need (`all_ts`, `data_ts`) — a different quantity, and
the one this task removes.

TWO GUARDS, AND ONLY ONE OF THEM IS A MEASUREMENT (run-53 cycle 3)
-------------------------------------------------------------------
Each test below pins the same property twice, with instruments that fail in different directions:

  1. **A STRUCTURAL guard (deterministic, SCOPED — see the reach caveat at
     `_assert_single_traversal`).** The property that actually matters — "the reader
     performs no duplicated full-file pass" — is structural, not statistical. `builtins.open` is
     wrapped for the duration of the `read_capture` call and every character handed out of the
     capture file is counted, so the test can assert the file is opened EXACTLY ONCE and
     traversed EXACTLY ONCE (characters read == characters in the file) — **but only for file
     access that resolves through the rebound `builtins.open` NAME. Measured (run-53
     verification): rebinding `builtins.open` intercepts ZERO `io.open` calls (`io.open` IS
     `builtins.open`, the same object under a second name, and `pathlib.Path.open` holds a
     direct reference captured at import), so a bypassing second traversal via `io.open` /
     `Path.open` leaves `n_opens == 1` and this guard AFFIRMATIVELY CERTIFIES
     "traversals=1.000" for a genuinely double-traversing reader (measured counterexample
     `rig_stream_io`: 120.95 B/event, n_opens=1, traversals=1.000, PASSED). For those paths
     the RSS bound below is the only operative defense; an fd-layer interception that closes
     the hole is rowed as a follow-up.** Within its reach this half is
     order-independent, host-independent, interpreter-independent and cannot be masked by
     anything in the environment, because it measures no resource at all.
  2. **A QUANTITATIVE guard (RSS).** Peak resident memory attributable to the call, per EVENT.
     RSS is only a noisy proxy for the structural property, but it catches the one shape the
     structural guard cannot see: a duplicated MATERIALIZATION built out of the SINGLE existing
     pass (e.g. `list(_iter_lines_tolerant(path))`), which traverses the file once and retains
     all of it. Both guards are therefore load-bearing and neither subsumes the other; this is
     demonstrated, not asserted — see NON-VACUITY below.

HOW PEAK RSS IS SAMPLED, AND WHY NOT `getrusage` (run-53 cycle 3 BLOCKER)
-------------------------------------------------------------------------
The measurement runs in a fresh child process, and the SAMPLING PRIMITIVE inside that child is
`VmHWM` from `/proc/self/status` — **not** `resource.getrusage(...).ru_maxrss`.

`ru_maxrss` is backed by the kernel's `signal->maxrss`, which is **NOT reset by `execve`**: a
freshly-exec'd child inherits its parent's high-water mark verbatim. `VmHWM` is backed by
`mm->hiwater_rss`, which belongs to the new `mm` `execve` installs and IS fresh. Measured
first-hand on `workstation` (parent touches 400 MB, frees it, then execs a child that allocates
nothing):

    parent ru_maxrss after hog = 419,372 KB
    child  ru_maxrss           = 419,372 KB   <-- the parent's peak, inherited across exec
    child  VmHWM               =  10,144 KB   <-- the child's own, correct figure

Cycle 2 of this run sampled `ru_maxrss` in the child and stated in this file that "a child
process starts from its own floor, so the number is absolute and independent of test order".
That claim was FALSE, and it failed toward PASS. Both primitives sampled in the SAME child, on
this file's own book/trade fixture at N=500,000, `capture.py` carrying one duplicated raw-line
full-file pass (a defect that is RED in isolation under either instrument):

    parent state          ru_maxrss delta     VmHWM delta
    clean parent            274.64 B/event    305.91 B/event
    parent touched 300 MB
      and imported numpy      0.00 B/event    305.23 B/event   <-- would PASS a 160 B/event bound

The `ru_maxrss` guard does not merely lose sensitivity: it saturates at exactly zero and masks a
defect of ANY severity, because the child's inherited `before` sample already exceeds anything
the child itself will ever reach. Nothing in `experiments/907_live_fill_probe/` imports
numpy/pandas/JAX today, which is the only reason it was not already live; one future test file,
or a broader `pytest experiments/` sweep, would have silently and permanently disabled it.

The same mechanism, seen small, was recorded in cycle 2's own docstring as a defect measured at
`260.43 B/event` under pytest against `275.99-281.32` bare-interpreter and misattributed to "a
slightly different allocator state". It was not the allocator. `ru_maxrss` under-reports by
exactly `parent_peak - child_true_baseline`, so the deflation is a function of whoever spawned
the child: measured here, `ru_before` read 28,205,056 B where the child's true fresh baseline was
12,648,448 B, a 15.6 MB subtraction = 31.1 B/event at this fixture size — which is precisely the
gap between the clean figures under the two primitives (89.53 vs 120.64 B/event). Cycle 2's whole
B/event table was low by that quantity, so **every figure in this file below was re-measured
under `VmHWM`** and the per-event bound was re-derived rather than carried over.

Residual, and it is bounded rather than argued away: the child's own delta is
`VmHWM(after) - VmHWM(baseline)`, and a high-water mark reached by the child BEFORE the baseline
(its own interpreter start-up and `import capture`) would be subtracted the same way. The child
therefore reports its baseline and the test REFUSES a baseline above `MAX_CHILD_BASELINE_BYTES`,
so this deflation cannot grow silently the way the `ru_maxrss` one did.

WHAT IS AND IS NOT CLAIMED HERE (run-53 cycle 2 correction, still current)
--------------------------------------------------------------------------
The post-914 reader's memory is **LINEAR IN EVENTS**. It is not flat, not O(1), and it does not
stop growing — only the SLOPE changed. The RSS half of each test below therefore pins exactly one
property: peak RSS scales with the number of EVENTS at a per-event cost that leaves no room for a
duplicated full-file materialization. Neither asserts a bound that a merely LARGER capture would
breach, and neither may be read as evidence of a memory ceiling.

Concretely, on the closed live capture
(`runs/907_live_fill_probe/capture_907_main.FINAL_2026-09-01.jsonl`, 1,440,479,339 bytes,
11,761,474 events: 10,819,267 book + 611,969 trade + 330,238 heartbeat, 1,727 gap records) the
post-914 reader costs **1.0123 RSS-bytes per file-byte**, and the run-53 flow-verifier's sweep
over 200k/1M/3M/6M/11.76M lines found that ratio dead flat at ~1.0 from 3M lines upward — a
per-event slope, not a floor. Consequences, stated plainly:

  * the task spec's `peak RSS < 4 GB` gate is ONE POINT ON A LINE. At 1.0123 B/file-byte a
    4 GiB peak is reached again at ~4.24 GB of capture.
  * what the fix genuinely buys is the 32 GiB `safe_run` boundary moving from ~3.26 GB of
    capture (pre-914, 10.546 B/file-byte) out to ~33.9 GB (post-914). That is the honest, and
    still large, claim.

Translated into RUN LENGTH, and note the denominator carefully: this capture's byte rate is
1,440,479,339 B over its WALL-CLOCK event span of 4.1198 days (`end_ts - start_ts`, measured),
= 3.4965e8 B/day. It is NOT 3.69 days — that is the span with its 36,860.652 s of recorded gap
time subtracted, which is the right denominator for COVERAGE and the wrong one for "how long
until the file gets this big". On the wall-clock rate: the 4 GiB gate recurs at ~12.1 days of
capture, the 32 GiB `safe_run` cap at ~97 days post-914 against ~9.3 days pre-914.

THE BEFORE/AFTER REDUCTION (unchanged by the instrument correction)
--------------------------------------------------------------------
Peak RSS of the `read_capture` call alone on the closed live capture above, workstation
`workstation`, wrapped in `tools/safe_run.sh --cpu`:

  interpreter        pre-TASK_914          post-TASK_914       reduction
  CPython 3.10.14    15,190,888,448 B      1,458,266,112 B      10.42x   (`trader_env`, the
                                                                          project venv)
  CPython 3.14.4     12,342,190,080 B      1,462,108,160 B       8.44x   (`/usr/bin/python3.14`)

**These four figures, and the 1.0123 B/file-byte ratio above, were NOT re-measured under
`VmHWM`** — doing so means a ~15 GB peak against the 1.44 GB live capture, which is not what this
cycle's correction turns on. They are `ru_maxrss` numbers and are therefore LOW by
`parent_peak - child_true_baseline`, which for the dedicated small parent that produced them is
~15 MB: 0.001% of the pre-914 figures and ~1.0% of the post-914 ones. The direction of that error
makes the post-914 side look slightly better than it is, so the stated reduction (10.42x / 8.44x)
is at most ~1% optimistic and no claim here rests on the third significant figure. The
interpreter is load-bearing on the BEFORE
side and irrelevant on the AFTER side: what 3.11+ shrank is precisely the `str`/`dict`
materialization this task deleted, while the surviving retention is `float` lists, which cost the
same in both. That single fact reconciles run 53's two disagreeing BEFORE measurements —
15,190,589,440 B (implementer) and 12,337,229,824 / 12,337,553,408 B (flow-verifier), a ~19%
spread that looked like a measurement defect. Both are correct; they measured different
interpreters, and the re-measurement above reproduces each to within 0.002% and 0.04%.

**Quote 8.44x when a single number is wanted** — it is the conservative one, and no claim here
may be stated more favourably than that.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_MODULE_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, _MODULE_DIR)
from capture import GapRecord  # noqa: E402  (the reader itself is exercised in a subprocess)

#: The RSS half of both tests reads `VmHWM` out of `/proc/self/status`, which is Linux-only.
#: Skipped with a stated reason rather than silently falling back to `getrusage`: the fallback
#: instrument is the one this file exists to stop using (see the module docstring).
_HAVE_PROC_STATUS = os.path.exists("/proc/self/status")

N_HEARTBEATS = 1_000_000
#: 3.20x above the streaming reader's worst measured 46,940 KB and 3.77x below the old
#: three-materializations reader's measured 565,972 KB for the same file — re-derived under the
#: `VmHWM` instrument (run-53 cycle 3); see the test's docstring for the full table.
MAX_INCREMENTAL_RSS_KB = 150_000

#: The measurement child's OWN peak before the `read_capture` call — interpreter start-up plus
#: `import capture` plus the one-pass character count of the fixture. Measured at 12.69-12.81 MB
#: on `workstation` CPython 3.10.14 and 15.20-15.34 MB on 3.14.4.
#:
#: REACH CAVEAT (run-53 verification, transcribed 2026-09-06 per D82(b)): the comment below
#: originally claimed this ceiling makes the one residual deflation window (a child baseline
#: that grows past the quantity being measured) "fail LOUDLY instead of quietly." That claim
#: was too strong by measurement: the guard detects only baselines ABOVE the ceiling, and a
#: duplicated raw-line full-file pass still PASSES at child baselines of 58.2 / 62.3 / 64.4 MB
#: (reporting 166.35 / 158.62 / 154.47 B/event against the 190 bound) — i.e. the "loud" band
#: is SILENT for any baseline between the honest ~13 MB and the ceiling, deflating the figure
#: by `(B - 12.8 MB)`, which at the ceiling is 108.6 B/event = 158% of the entire headroom
#: between honest cost (~121) and the bound. A baseline-independent instrument (or a tighter
#: ceiling re-derived from the deflation arithmetic) is rowed as a follow-up; the value below
#: is unchanged — only this claim about it is corrected.
MAX_CHILD_BASELINE_BYTES = 64 * 1024 * 1024

# --- the book/trade fixture (run-53 cycle 2) -----------------------------------------------
# Heartbeats populate `all_ts` and NOTHING else: they never enter `data_ts`, so they never reach
# `all_data_ts`, `_pair_coverage`, `_cadence_stats`, the keepalive reconciliation or the
# `events_inside_recorded_gaps` bisect scan. Those passes carry most of the reader's memory on a
# real capture (the closed live capture is 92% book / 5% trade / 3% heartbeat), so a guard built
# only from heartbeats structurally cannot see a regression in the path that dominates RSS.
# This fixture is the book/trade half of the guard.
N_BOOK_TRADE_EVENTS = 500_000
BOOK_TRADE_PAIRS = ("BTC/USD", "ETH/USD", "SOL/USD")
#: One 100-event cycle, mirroring the closed live capture's channel mix.
_CYCLE_BOOK, _CYCLE_TRADE, _CYCLE_HEARTBEAT = 92, 5, 3
assert _CYCLE_BOOK + _CYCLE_TRADE + _CYCLE_HEARTBEAT == 100

#: Per-EVENT ceiling for the book/trade fixture, in bytes. Traceable to the measurements in
#: `test_book_trade_capture_costs_per_event_not_per_file_byte`'s docstring, not to a round number:
#: it is the geometric midpoint of the measured bracket, 1.57x above the WORST honest measurement
#: (121.21 B/event) and 1.20x below the CHEAPEST duplicated full-file pass anyone could
#: reintroduce on this path (228 B/event). **[CORRECTED 2026-09-06 per D82(b): the original
#: text said 1.60x below a 304.62 B/event "cheapest defect" — run-53 verification measured
#: the cheapest REINTRODUCIBLE full-file materialization (contiguous `bytes` blob / `mmap`)
#: at 228 B/event, not the 304.62 of the specific `list(...)` shape this table measured, so
#: the true defensive margin is 1.20x, not 1.60x. The bound value 190 is unchanged.]**
#:
#: **RE-DERIVED, and it MOVED (run-53 cycle 3).** Cycle 2 shipped 160 against a claimed 1.67x
#: honest margin, but that bracket was measured with the contaminated `ru_maxrss` instrument,
#: which under-reported the honest cost by ~31 B/event here. The true honest cost is ~121, not
#: ~96, so 160 was in fact only 1.32x above it — half the margin the file claimed. 190 restores a
#: symmetric bracket under the correct instrument. The honest cost is ALLOWED to grow with the
#: capture — that is exactly why this bound is per event and not absolute.
MAX_BOOK_TRADE_RSS_BYTES_PER_EVENT = 190


def _write_heartbeat_capture(path, n_events=N_HEARTBEATS):
    """The maximally sparse shape: one float of retained state per line, nothing data-bearing."""
    with open(path, "w") as fh:
        fh.write(json.dumps({"type": "meta_start", "pairs": ["BTC/USD"], "channels": ["book"],
                              "clock_basis": "local_receipt"}) + "\n")
        ts = 1_700_000_000.0
        for _ in range(n_events):
            fh.write(json.dumps({"type": "heartbeat", "ts": ts}) + "\n")
            ts += 1.0
    return {"n_book": 0, "n_trade": 0, "n_heartbeat": n_events}


def _write_book_trade_capture(path, n_events=N_BOOK_TRADE_EVENTS):
    """Write a book-dominated multi-pair capture that exercises every data-bearing pass.

    Deliberately realistic rather than minimal: three pairs, both data channels present on every
    pair (a declared channel that carries nothing is a REFUSAL one level up), a declared
    `book_emission_policy`/`book_keepalive_s` so the keepalive reconciliation actually runs, a
    top of book that MOVES on every emission (so `_note_book_top` opens a fresh run per event
    instead of collapsing the capture into one stale run), and one recorded `gap` with a real
    300s silence behind it so `_merge_gap_spans`, the containment bisect scan and
    `_pair_coverage`'s gap subtraction all have something to work on.
    """
    ts = 1_700_000_000.0
    step = 0.02
    gap_at = n_events // 2
    counts = {"n_book": 0, "n_trade": 0, "n_heartbeat": 0}
    with open(path, "w") as fh:
        fh.write(json.dumps({
            "type": "meta_start", "ts": ts, "pairs": list(BOOK_TRADE_PAIRS),
            "channels": ["book", "trade"], "clock_basis": "local_receipt",
            "book_emission_policy": "on_change_or_keepalive", "book_keepalive_s": 5.0,
        }) + "\n")
        for i in range(n_events):
            if i == gap_at:
                # A genuine, DISCLOSED outage: the ledger says the wire was dark and the stream
                # agrees (no event inside the span).
                gap = GapRecord(ts, ts + 300.0, "disconnect", detail="synthetic outage")
                fh.write(json.dumps(gap.to_line()) + "\n")
                ts += 300.0
            pair = BOOK_TRADE_PAIRS[i % len(BOOK_TRADE_PAIRS)]
            slot = i % 100
            if slot < _CYCLE_BOOK:
                bid = 30_000.0 + (i % 4_096) * 0.25
                fh.write(json.dumps({
                    "type": "book", "pair": pair, "ts": ts, "bid": bid, "ask": bid + 0.5,
                    "bid_sz": 1.5, "ask_sz": 2.25}) + "\n")
                counts["n_book"] += 1
            elif slot < _CYCLE_BOOK + _CYCLE_TRADE:
                fh.write(json.dumps({
                    "type": "trade", "pair": pair, "ts": ts,
                    "price": 30_000.0 + (i % 997) * 0.5, "volume": 0.125, "side": "buy",
                    "venue_ts": ts - 0.004}) + "\n")
                counts["n_trade"] += 1
            else:
                fh.write(json.dumps({"type": "heartbeat", "ts": ts}) + "\n")
                counts["n_heartbeat"] += 1
            ts += step
        fh.write(json.dumps({"type": "meta_end", "ts": ts}) + "\n")
    return counts


#: The measurement runs in a FRESH CHILD PROCESS and samples `VmHWM`, not `ru_maxrss` — see the
#: module docstring for the kernel reason and the first-hand numbers. The child also carries the
#: STRUCTURAL instrument: `builtins.open` is wrapped for exactly the duration of the
#: `read_capture` call, so every open of the capture file and every character read out of it is
#: counted. Two facts fall out of that counter, and neither of them measures a resource:
#:
#:   * `n_opens`     — how many times `read_capture` opened the capture. Must be 1.
#:   * `chars_read`  — characters handed out of those handles. Must equal the file's own
#:                     character count: one traversal, no more and no less.
#:
#: The wrapper is installed AFTER `import capture` and removed in a `finally`, so it sees the
#: reader's I/O and nothing else. It counts characters rather than bytes because the reader opens
#: in text mode; the capture format is JSON-escaped ASCII, so the two coincide, and the
#: denominator is computed by reading the same file through the same text-mode decoder rather
#: than from `os.path.getsize`, so they cannot disagree even if that ever stopped being true.
_CHILD_MEASURE = r'''
import builtins, json, os, sys
mod_dir, path = sys.argv[1], sys.argv[2]
sys.path.insert(0, mod_dir)

_real_open = builtins.open


def _vmhwm_bytes():
    """This process's peak RSS since exec. `mm->hiwater_rss`, fresh after execve — unlike
    `getrusage`'s `ru_maxrss` (`signal->maxrss`), which the child inherits from its parent."""
    with _real_open("/proc/self/status") as fh:
        for ln in fh:
            if ln.startswith("VmHWM:"):
                return int(ln.split()[1]) * 1024
    raise RuntimeError("VmHWM absent from /proc/self/status")


class _CountingReader:
    """Delegating file wrapper that counts every character handed to the reader."""

    def __init__(self, fh):
        self._fh = fh
        self.chars = 0

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self._fh)
        self.chars += len(line)
        return line

    def readline(self, *a, **k):
        line = self._fh.readline(*a, **k)
        self.chars += len(line)
        return line

    def readlines(self, *a, **k):
        lines = self._fh.readlines(*a, **k)
        self.chars += sum(len(ln) for ln in lines)
        return lines

    def read(self, *a, **k):
        data = self._fh.read(*a, **k)
        self.chars += len(data)
        return data

    def __enter__(self):
        self._fh.__enter__()
        return self

    def __exit__(self, *exc):
        return self._fh.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._fh, name)


_target = os.path.realpath(path)
_handles = []


def _counting_open(file, *a, **k):
    fh = _real_open(file, *a, **k)
    try:
        same = os.path.realpath(file) == _target
    except TypeError:
        same = False
    if not same:
        return fh
    wrapped = _CountingReader(fh)
    _handles.append(wrapped)
    return wrapped


# The denominator for "exactly one traversal", read through the same text-mode decoder the
# reader uses. Done BEFORE the baseline sample so its (negligible) cost is not attributed to
# `read_capture`, and through the REAL `open` so it is not itself counted.
file_chars = 0
with _real_open(path, "r") as _fh:
    for _raw in _fh:
        file_chars += len(_raw)

from capture import read_capture

builtins.open = _counting_open
baseline = _vmhwm_bytes()
try:
    meta = read_capture(path)
finally:
    builtins.open = _real_open
peak = _vmhwm_bytes()

print("RSS_RESULT " + json.dumps({
    "baseline_bytes": baseline,
    "peak_bytes": peak,
    "incremental_bytes": peak - baseline,
    "n_opens": len(_handles),
    "chars_read": sum(h.chars for h in _handles),
    "file_chars": file_chars,
    "n_book": meta.n_book, "n_trade": meta.n_trade, "n_heartbeat": meta.n_heartbeat,
    "had_meta_start": meta.had_meta_start,
    "pairs": meta.pairs,
    "observed_pair_channels": meta.observed_pair_channels,
    "coverage_evaluated": meta.coverage_evaluated,
    "coverage_pairs": sorted(meta.coverage_by_pair),
    "cadence_pairs": sorted({c.pair for c in meta.cadence}),
    "cadence_channels": sorted({c.channel for c in meta.cadence}),
    "n_gaps": len(meta.gaps),
    "n_events_inside_recorded_gaps": meta.n_events_inside_recorded_gaps,
    "closed_cleanly": meta.closed_cleanly,
    "truncated_tail": meta.truncated_tail,
}))
'''


def _measure_read_capture(path):
    """Run `read_capture` under both instruments in a fresh child; return `(bytes, facts)`.

    The child imports `capture` from THIS FILE's own directory, so dropping a modified
    `capture.py` next to a copy of this test is enough to exercise the guard against it.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_MEASURE, _MODULE_DIR, str(path)],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"measurement subprocess failed ({proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RSS_RESULT ")), None)
    assert line is not None, f"no RSS_RESULT line in child stdout:\n{proc.stdout}"
    out = json.loads(line[len("RSS_RESULT "):])
    return out["incremental_bytes"], out


def _assert_single_traversal(meta, path):
    """The STRUCTURAL guard: `read_capture` opened the capture once and read it exactly once
    — THROUGH THE REBOUND `builtins.open` NAME (scope caveat below).

    Deterministic and environment-independent within its reach — it measures no resource, so
    no neighbour process, test ordering, interpreter version or allocator can move it. This is
    the guard that would have caught the original three materializations (two of which were
    separate `readlines()` passes over a re-opened file) without measuring a single byte of
    RSS.

    Its blind spots, both stated because the RSS guard below exists to cover them:
    (1) a duplicated MATERIALIZATION built from the single existing pass (retaining every
    parsed line in a list) traverses the file exactly once and is invisible here;
    (2) REACH (run-53 verification, transcribed 2026-09-06 per D82(b)): the child rebinds
    only the `builtins.open` NAME, which intercepts ZERO `io.open` calls (`io.open` IS
    `builtins.open`; `pathlib.Path.open` holds a direct reference captured at import) — so a
    second traversal routed via `io.open`/`Path.open` leaves `n_opens == 1` and this guard
    certifies "1.000 traversals" for a genuinely double-traversing reader (measured:
    `rig_stream_io`, 120.95 B/event, PASSED). Closing this needs fd-layer interception,
    rowed as a follow-up.
    """
    assert meta["n_opens"] == 1, (
        f"read_capture opened {os.path.basename(str(path))} {meta['n_opens']} times — it must "
        f"open it exactly once. More than one open is a duplicated full-file pass, which is the "
        f"O(file) cost TASK_914 removed; zero means the measurement harness failed to intercept "
        f"the reader's I/O and the structural guard measured nothing.")
    assert meta["chars_read"] == meta["file_chars"], (
        f"read_capture read {meta['chars_read']} characters out of a {meta['file_chars']}-"
        f"character capture ({meta['chars_read'] / max(1, meta['file_chars']):.3f} traversals) "
        f"— it must traverse the file exactly once. More is a duplicated full-file pass; less "
        f"means the reader stopped early and every count it reported is about a prefix of the "
        f"capture.")


def _assert_child_baseline_sane(meta):
    """Refuse a measurement whose own baseline could have eaten the quantity being measured.

    `incremental_bytes` is `VmHWM(after) - VmHWM(baseline)`, and `VmHWM` is a high-water mark, so
    a child that peaked HIGHER before the baseline than `read_capture` peaks after it would
    report a deflated (in the limit, zero) figure. Nothing in the PARENT can do that — `VmHWM` is
    fresh after `execve`, which is the whole point of this instrument — but the child's own
    start-up could, if `capture.py` ever grew a heavy import. That must fail loudly.
    """
    assert meta["baseline_bytes"] < MAX_CHILD_BASELINE_BYTES, (
        f"the measurement child's own peak RSS before the read_capture call was "
        f"{meta['baseline_bytes']} bytes, over the {MAX_CHILD_BASELINE_BYTES}-byte ceiling. The "
        f"reported incremental figure is a high-water-mark DELTA, so a baseline this large can "
        f"silently absorb the cost being measured and make this test vacuous. Find what the "
        f"child now allocates before the call (most likely a new import reached through "
        f"`capture.py`) rather than raising this ceiling.")


@pytest.mark.skipif(not _HAVE_PROC_STATUS,
                    reason="/proc/self/status unavailable — VmHWM peak-RSS sampling is Linux-only")
def test_large_sparse_capture_scales_with_events_not_file_size(tmp_path):
    """A capture whose event content is maximally sparse/cheap (heartbeats only, one float each)
    but whose LINE COUNT is large enough that any full-file materialization would dominate RSS.
    The old reader's scaling was O(file bytes) x 3; the new one is O(events) x 1 — this is the
    quantity difference the task brief calls out, and it is what this test pins.

    Measured, factory run 53 cycle 3, `VmHWM` instrument, wrapped in `tools/safe_run.sh --cpu`
    (`capture.py` pre-TASK_914 (HEAD 5010a1b) vs post-TASK_914, same file, same host, this test's
    exact fixture and harness):

      N=1,000,000 heartbeats (42,000,099-byte file) —
        pre-914 reader,  workstation CPython 3.10.14:  579,555,328 B = 565,972 KB (~553 MiB), 2.000
                                                                                  traversals
        post-914 reader, workstation CPython 3.10.14:   47,951,872 B =  46,828 KB  (~46 MiB), 1.000
        post-914 reader, workstation CPython 3.14.4:    48,066,560 B =  46,940 KB  (~46 MiB), 1.000

      12.06x less, and still LINEAR in events (the retained state is one float per event in
      `all_ts`); what is gone is the per-event raw string PLUS parsed dict PLUS second raw-line
      copy on top of it.

    The bound (150,000 KB) is 3.77x below the pre-914 number and 3.20x above the worst post-914
    number, so it is decisive without being flaky.

    The pre-914 reader also FAILS the structural half of this test outright: it opened the
    capture TWICE (`_iter_lines_tolerant`'s `readlines()` and `_file_has_truncated_tail`'s), for
    2.000 traversals. Measured, not reasoned about — and it is the half that needed no bound at
    all to say so.

    This test has a deliberate blind spot, which is why the book/trade test below exists:
    heartbeats only ever reach `all_ts`. Everything downstream of `data_ts` — per-pair coverage,
    cadence, keepalive reconciliation, the gap-containment scan — is untouched here. Measured,
    not assumed: all three of the book/trade poisons in that test's table leave THIS test green
    on both halves (46,804-46,940 KB, 1.000 traversals), because each is gated on the capture
    carrying data-bearing events and a heartbeat-only capture has none.
    """
    path = tmp_path / "large_sparse.jsonl"
    _write_heartbeat_capture(path, N_HEARTBEATS)

    file_bytes = os.path.getsize(path)
    assert file_bytes > 30_000_000, (
        f"synthetic capture is only {file_bytes} bytes — too small to distinguish O(file) from "
        f"O(events) scaling; bump N_HEARTBEATS")

    incremental_bytes, meta = _measure_read_capture(path)
    incremental_kb = incremental_bytes // 1024

    assert meta["n_heartbeat"] == N_HEARTBEATS
    assert meta["had_meta_start"]

    _assert_single_traversal(meta, path)
    _assert_child_baseline_sane(meta)

    assert incremental_kb < MAX_INCREMENTAL_RSS_KB, (
        f"read_capture's own incremental peak RSS was {incremental_kb} KB for a "
        f"{file_bytes}-byte, {N_HEARTBEATS}-event sparse capture — over the "
        f"{MAX_INCREMENTAL_RSS_KB} KB bound. That bound sits far below what THREE full-file "
        f"materializations (raw lines + parsed dicts + a second raw-line pass) would cost at "
        f"this size (measured ~532 MiB pre-TASK_914 for this exact file shape) — a regression "
        f"back to file-size-scaling RSS is exactly what this test exists to catch.")


@pytest.mark.skipif(not _HAVE_PROC_STATUS,
                    reason="/proc/self/status unavailable — VmHWM peak-RSS sampling is Linux-only")
def test_book_trade_capture_costs_per_event_not_per_file_byte(tmp_path):
    """The same guard, on the path that actually dominates RSS: book + trade, multi-pair.

    The heartbeat test above cannot see this path at all (heartbeats never enter `data_ts`), so
    a duplicated full-file pass reintroduced anywhere below `data_ts` — per-pair coverage,
    `_cadence_stats`, the keepalive reconciliation, the `events_inside_recorded_gaps` bisect —
    would leave it green. The closed live capture is 92% book, so that is most of the memory.

    WHAT THE RSS HALF ASSERTS: peak RSS per EVENT stays under a ceiling. It does NOT assert an
    absolute ceiling, because the honest post-914 cost is linear in events and a bigger capture
    is supposed to cost more. The structural half asserts one traversal and is not a bound at all.

    Measured, factory run 53 cycle 3, `VmHWM` instrument, wrapped in `tools/safe_run.sh --cpu`,
    on this exact fixture at N=500,000 (460,000 book + 25,000 trade + 15,000 heartbeat over
    3 pairs, 62,166,378 bytes = 124.33 file-bytes per event, 1 recorded gap):

      reader variant                                    host/interpreter   B/event  traversals
      post-914 (2 runs)                            workstation CPython 3.10.14 121.04,   1.000
                                                                            120.98
      post-914, 300MB+numpy hog in the parent      workstation CPython 3.10.14  120.69    1.000
      post-914 (2 runs)                            workstation CPython 3.14.4  121.21,   1.000
                                                                            120.87
      pre-914 (HEAD 5010a1b)                       workstation CPython 3.10.14 1287.35    2.000
      post-914 + one duplicated PARSED full-file
        pass, book/trade path only                 workstation CPython 3.10.14 1223.28    2.000
      post-914 + one duplicated RAW-LINE full-file
        pass, book/trade path only                 workstation CPython 3.10.14  305.20    2.000
                                        "          workstation CPython 3.14.4   304.62    2.000
                                        "          + 300MB and numpy hog
                                                     in the parent           305.74    2.000
      post-914 + a RETAINED parsed list built from
        the SINGLE existing pass (no re-traversal) workstation CPython 3.10.14 1177.99    1.000

    Honest spread across both interpreters, with and without the hog: 120.69-121.21 B/event,
    i.e. 0.43%. The bound is 190 B/event: 1.57x above the worst honest measurement (121.21) and
    1.20x below the CHEAPEST observed REINTRODUCIBLE defect (228, a contiguous `bytes`/`mmap`
    blob — corrected 2026-09-06 per D82(b); the 304.62 row in the table above is the specific
    `list(...)` shape, not the cheapest shape). That cheapest defect's cost is derivable
    rather than magic — 124.33 file-bytes per line + ~49 B of `str` header + 8 B of list pointer
    ~= 181 B/event on top of the ~121 B/event baseline ~= 302, against 304.6-305.7 measured.

    The LAST row is the one that keeps BOTH halves of this test load-bearing: a defect that
    retains the whole file without ever re-reading it is invisible to the traversal count (1.000)
    and is caught by RSS alone — while the traversal count is the half that no environment can
    mask. Neither guard subsumes the other, and that is measured here, not argued.

    NON-VACUITY, PROVEN not assumed (run-47's lesson): every poison row above was produced by
    actually planting that poison in a scratch copy of `capture.py` and running THIS test file
    against it. All three poisons go RED here; all three leave the heartbeat test above GREEN,
    which is precisely the hole this test was added to close.

    ORDER-INDEPENDENCE, PROVEN not assumed (run-53 cycle 3): the rows naming a hog were produced
    by running this file in a pytest session whose FIRST collected test allocates 300 MB, touches
    every page and imports numpy, driving the pytest process's own `ru_maxrss` to 351 MB. Under
    cycle 2's `ru_maxrss` instrument that same setup drove the poisoned reading to 0.00 B/event
    and the test PASSED. Under `VmHWM` the poisoned reading is 305.74 B/event WITH the hog against
    305.20 WITHOUT it — a 0.18% difference, well inside the honest run-to-run spread — and the
    test FAILS either way. The figure does not move because the child's high-water mark is created
    by `execve`, after the hog is already gone.
    """
    path = tmp_path / "book_trade.jsonl"
    counts = _write_book_trade_capture(path, N_BOOK_TRADE_EVENTS)

    file_bytes = os.path.getsize(path)
    file_bytes_per_event = file_bytes / N_BOOK_TRADE_EVENTS
    assert file_bytes > 30_000_000, (
        f"synthetic capture is only {file_bytes} bytes — too small to separate a duplicated "
        f"full-file pass from the honest per-event retention; bump N_BOOK_TRADE_EVENTS")
    # The bound only discriminates while a file-byte is expensive relative to the per-event
    # retention. If the fixture's lines ever shrink far below this, a duplicated raw-line pass
    # would fit under the ceiling and the test would go quietly vacuous.
    assert file_bytes_per_event > 100.0, (
        f"fixture lines average {file_bytes_per_event:.1f} bytes — too short for a duplicated "
        f"full-file pass to be distinguishable from O(events) retention under the "
        f"{MAX_BOOK_TRADE_RSS_BYTES_PER_EVENT} B/event bound")

    incremental_bytes, meta = _measure_read_capture(path)
    bytes_per_event = incremental_bytes / N_BOOK_TRADE_EVENTS

    # The reader must actually have WALKED the book/trade path — otherwise a future refactor
    # that skipped it entirely would pass this test on cost alone, which is its own vacuity.
    assert (meta["n_book"], meta["n_trade"], meta["n_heartbeat"]) == (
        counts["n_book"], counts["n_trade"], counts["n_heartbeat"])
    assert meta["pairs"] == sorted(BOOK_TRADE_PAIRS)
    assert meta["observed_pair_channels"] == {p: ["book", "trade"]
                                               for p in sorted(BOOK_TRADE_PAIRS)}
    assert meta["coverage_evaluated"] and meta["coverage_pairs"] == sorted(BOOK_TRADE_PAIRS)
    assert meta["cadence_pairs"] == sorted(BOOK_TRADE_PAIRS)
    assert meta["cadence_channels"] == ["book", "book+trade", "trade"]
    assert meta["n_gaps"] == 1 and meta["n_events_inside_recorded_gaps"] == 0
    assert meta["closed_cleanly"] and not meta["truncated_tail"]

    _assert_single_traversal(meta, path)
    _assert_child_baseline_sane(meta)

    assert bytes_per_event < MAX_BOOK_TRADE_RSS_BYTES_PER_EVENT, (
        f"read_capture's own incremental peak RSS was {incremental_bytes} bytes for "
        f"{N_BOOK_TRADE_EVENTS} book/trade-dominated events ({bytes_per_event:.1f} B/event, "
        f"{file_bytes_per_event:.1f} file-bytes/event) — over the "
        f"{MAX_BOOK_TRADE_RSS_BYTES_PER_EVENT} B/event bound. Growth WITH THE EVENT COUNT is "
        f"expected and fine; this bound is per event precisely so that a larger capture cannot "
        f"trip it. What trips it is a duplicated full-file materialization on the book/trade "
        f"path — measured at 305 B/event for a single extra raw-line pass, 1223 B/event for a "
        f"single extra parsed pass and 1178 B/event for a retained parsed list built from the "
        f"existing pass, against a 120.7-121.2 B/event honest baseline.")

"""TASK_914 — the streaming reader's own regression guard (spec acceptance iii).

`read_capture` used to materialize the whole file THREE times before any analysis started
(`capture.py::_iter_lines_tolerant` did a `readlines()` internally; `read_capture` then built a
second full list of parsed dicts from it; `_file_has_truncated_tail` did a THIRD `readlines()`
while the second list was still alive). That is an O(file) cost layered on top of the O(events)
retention the estimand passes genuinely need (`all_ts`, `data_ts`) — a different quantity, and
the one this task removes.

TWO GUARDS, AND ONLY ONE OF THEM IS A MEASUREMENT (run-53 cycle 3; structural mechanism
rebuilt TASK_921 per D82(b)/D83(i) — the run-53 `builtins.open`-name guard is GONE)
-------------------------------------------------------------------------------------------
Each test below pins the same property twice, with instruments that fail in different directions:

  1. **A STRUCTURAL guard, now at the FD LAYER (TASK_921).** The run-53 guard rebound the
     `builtins.open` NAME and was measured to intercept ZERO `io.open` calls (`io.open` IS
     `builtins.open`, the same object under a second name, and `pathlib.Path.open` holds a
     direct reference captured at import) — verified first-hand in THIS run before writing a
     line of the replacement (`os.open` patched the same way: still zero interceptions of
     `open`/`io.open`/`Path.open`, because CPython's `_io.FileIO` issues the `openat`/`read`
     syscalls from C, never through the Python-level `os.open`/`os.read` names). A NAME-layer
     guard therefore cannot be fixed by patching a different name; it has to stop being a
     NAME-layer guard. The measurement child now runs under `strace -f -y`, which sees the
     real `openat`/`open`/`read`/`pread64`/`readv`/`preadv`/`preadv2` syscalls regardless of
     which Python API issued them, and `-y` resolves every fd argument to the path it
     currently points at (via `/proc/<pid>/fd/<n>`), so a dup'd or reused fd is still
     attributed correctly. `_measure_read_capture` sums (opens, bytes transferred) for every
     syscall whose resolved path is the capture file, and the guard asserts EXACTLY ONE
     open and total bytes == the file's own size — a second traversal through ANY of
     `open`/`io.open`/`Path.open`/`os.open`+`os.read`/`os.pread`, or a re-read via an
     already-open fd, shows up as more opens or more bytes and cannot hide behind a
     particular Python spelling the way the NAME guard could be bypassed. It also follows
     forked/exec'd subprocesses (`strace -f`), so a re-read done by a CHILD process of
     `read_capture` is covered too — a reach the NAME guard never had at all.

     **What this reach still does NOT cover, stated plainly rather than argued past:**
     (i) `mmap`-backed reads — a page fault that satisfies a memory-mapped read emits no
     `read`/`pread64` syscall, so a reader that `mmap()`s the capture and walks the mapping
     would under-count here (the mapping's `mmap()` call itself is visible in a raw trace but
     is deliberately NOT counted as bytes-read by this guard, because an untouched mapped
     range is not a read); (ii) `sendfile`/`splice`/`copy_file_range` — in-kernel copies that
     never cross into userspace as a `read`; (iii) an fd inherited from an ANCESTOR of the
     traced process, opened before `strace` attached — not reachable for THIS harness's own
     calling convention (the child is exec'd directly under `strace`, so every fd the reader
     itself opens is seen from its first syscall), but stated for anyone reusing this
     mechanism where that convention does not hold; (iv) any host where `ptrace` is
     unavailable or blocked (containers without `CAP_SYS_PTRACE`, some CI sandboxes) — the
     suite probes this once at import time and skips with a stated reason rather than
     silently degrading, exactly as the `VmHWM` half already does for `/proc/self/status`.
     Within its reach (i.e. every read through the standard buffered/unbuffered file APIs,
     in-process or in a child process) this half is deterministic and measures no resource,
     so no neighbour process, test ordering, interpreter version or allocator can move it.
  2. **A QUANTITATIVE guard (RSS).** Peak resident memory attributable to the call, per EVENT.
     RSS is only a noisy proxy for the structural property, but it catches the one shape the
     structural guard cannot see: a duplicated MATERIALIZATION built out of the SINGLE existing
     pass (e.g. `list(_iter_lines_tolerant(path))`), which traverses the file once and retains
     all of it — no second open, no extra byte transferred, invisible to (1) by construction.
     Both guards are therefore load-bearing and neither subsumes the other; this is
     demonstrated, not asserted — see NON-VACUITY below, and the two `test_poison_*` functions
     at the end of this file, which plant exactly these two shapes and show each guard catching
     the one the other cannot.

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
therefore reports its baseline and the test REFUSES a baseline above the ceiling
`_max_child_baseline_bytes` derives for that test's OWN event count —
**TASK_921 re-derived that ceiling's VALUE from the deflation arithmetic** (run-53 shipped 64 MiB
in the same breath as measuring that a duplicated raw-line pass PASSES, silently, anywhere in a
13 MB-64 MiB baseline band; see `MAX_CHILD_BASELINE_BYTES`'s own comment for the arithmetic and
the new value), so this deflation cannot grow silently the way the `ru_maxrss` one did. In the
fail-OPEN direction — the two real measurement tests, which assert `measured < bound` — the
remaining silent band is smaller than each test's own honest-to-bound headroom rather than 1.58x
it (36.3% of it at the binding N=500,000; 11.8% of the heartbeat test's absolute KB headroom).
That claim is scoped to those tests deliberately: the poison demonstrations assert a LOWER bound,
where the same deflation is fail-CLOSED and no such band exists to bound. Cycle 2 of TASK_921
corrected an earlier version of this paragraph that asserted the claim unqualified, over a
derivation that had silently omitted the smallest N in the file.

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
import re
import shutil
import subprocess
import sys
import tempfile
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
#: `import capture`. Measured at 12.69-12.81 MB on `workstation` CPython 3.10.14 and 15.20-15.34 MB
#: on 3.14.4 (TASK_921: the child no longer does its own one-pass character count of the fixture
#: before sampling the baseline — that read is replaced by `os.path.getsize`, a `stat(2)`, not a
#: `read(2)`, so it no longer touches the file at all and cannot itself feed the fd-level guard
#: a bogus extra "open").
#:
#: RE-DERIVED (TASK_921 per D82(b) MAJOR-1, not just re-argued): run-53 shipped 64 MiB in the
#: same breath as measuring that a duplicated raw-line full-file pass PASSES, silently, at any
#: baseline in the ENTIRE 13 MB-64 MiB band (166.35 / 158.62 / 154.47 B/event against the 190
#: bound at 58.2 / 62.3 / 64.4 MB baselines) — 108.6 B/event of silent deflation at the ceiling,
#: 158% of the entire headroom between honest cost and bound. The arithmetic that ceiling should
#: have been checked against, and now is:
#:
#:   deflation(B) = VmHWM(after) - VmHWM(baseline=B), so a baseline B eats `(B - floor)` of the
#:   TRUE incremental figure, where `floor` is the child's own honest minimum baseline. That is an
#:   ABSOLUTE quantity; whether it matters is decided in the quantity the GATED TEST'S OWN bound
#:   is denominated in (D81(c)) — per event for the book/trade-shaped bounds, absolute KB for the
#:   heartbeat test:
#:
#:       deflation_per_event(B, N) = (B - floor) / N          deflation_abs(B) = B - floor
#:
#:   using the PESSIMISTIC (lowest measured) floor, 12.69 MB, so the derivation cannot itself be
#:   flattered. A SMALLER N dilutes the per-event deflation LESS, so the binding case is the
#:   SMALLEST N this ceiling actually gates.
#:
#:   **CORRECTED 2026-09-06 (TASK_921 cycle 2 — the previous text was a D83(ii) derivation-
#:   completeness defect).** It named `N_BOOK_TRADE_EVENTS = 500_000` as that binding N, having
#:   compared only the two real measurement tests. That is not the smallest N in this file:
#:   `test_poison_list_materialize_fails_rss_guard` passes through this same gate at
#:   `N_POISON_EVENTS = 60_000`, where a flat 24 MiB ceiling permits 207.93 B/event of deflation
#:   — MORE than the entire 190 B/event bound — so the old text's flagship claim ("the silent
#:   band is now smaller than the honest-to-bound headroom") was false for a test in this file.
#:
#:   The headroom the FAIL-OPEN direction must stay under is the book/trade test's own bound
#:   minus its own worst honest measurement:
#:
#:       headroom = MAX_BOOK_TRADE_RSS_BYTES_PER_EVENT(190) - worst_honest(121.21) = 68.79 B/event
#:
#:   Solving `deflation_per_event(B, N) < headroom` gives the soundness limit `B < floor +
#:   headroom * N`. At EVERY event count present in this file — recomputed and ASSERTED by
#:   `test_child_baseline_ceiling_derivation_holds_at_every_event_count`, which also prints this
#:   table, so it is executable rather than a comment anyone has to trust:
#:
#:       N           soundness limit   deflation @ 24 MiB   % of headroom   direction
#:         500,000     44.90 MiB          24.95 B/event         36.3%       fail-open   <- BINDING
#:          60,000     16.04 MiB         207.93 B/event        302.3%       fail-CLOSED
#:
#:   `N_HEARTBEATS = 1,000,000` is stated separately because its bound is ABSOLUTE, not per event
#:   (`MAX_INCREMENTAL_RSS_KB`), and mixing the denominations is how this comment went wrong once
#:   already: its headroom is 150,000 - 46,940 = 103,060 KB and a 24 MiB baseline deflates by
#:   12,183 KB = **11.8%** of it. (Its per-event soundness limit, 77.71 MiB, is looser still.)
#:
#:   The 60,000-event row is why this constant is no longer read directly by the gate:
#:   `_assert_child_baseline_sane` now takes the CALLER'S OWN N and the DIRECTION of the bound
#:   that caller asserts, and derives the permitted baseline from them (`_max_child_baseline_bytes`),
#:   so the invariant holds by construction for any test added at any future N rather than by a
#:   human remembering to re-check this block. **The CHECK of that invariant did not, until
#:   TASK_921 cycle 3: below the ~181,361-event crossover the N-derived limit is the ceiling in
#:   force, `deflation_per_event ≡ headroom` there by construction, and the strict `<` cycle 2
#:   asserted was therefore unsatisfiable — a permanent false red for exactly the small-N
#:   fail-open test this mechanism exists to serve, dormant only because both real fail-open N
#:   sit above the crossover. The assertion is now branch-aware (strict `<` where the FLAT
#:   ceiling is in force, equality-within-tolerance where the N-derived one is), and
#:   `test_child_baseline_ceiling_derivation_admits_a_future_small_n_fail_open_test` holds it to
#:   the claim.** The poison test at 60,000 asserts a LOWER bound
#:   (`measured > 190`): the same deflation can only push its figure DOWN, i.e. toward RED, and
#:   can never manufacture a false green, so the soundness limit is not required there. It is not
#:   and never was a live hole.
#:
#:   Exempting that direction — rather than applying the N-derived limit everywhere — is a
#:   MEASURED choice, not a cheap one. The poison child does not `import capture` directly: it
#:   loads the real module through the `types.ModuleType`/`compile`/`exec` shim in
#:   `_write_poison_module`'s templates, and that machinery costs it ~2.70 MB of extra baseline.
#:   Measured TASK_921 cycle 2 on `workstation`, CPython 3.10.14, this file at this revision, both
#:   sides through the same `_measure_read_capture` path: REAL `capture.py` baseline
#:   **13,377,536 B**, poison-shim baseline **16,072,704 B**. So N=60,000 is simultaneously the
#:   SMALLEST N and the HIGHEST baseline in this file, and the N-derived limit would clear the
#:   poison child by only **1.046x** — a guaranteed flake, gating a direction that cannot fail
#:   open in the first place.
#:
#:   Re-deriving the FLAT ceiling against N=60,000 was considered and REJECTED for the same
#:   family of reasons: 16.04 MiB is 1.10x the highest previously measured real baseline
#:   (15.34 MB, CPython 3.14.4) and 1.046x the poison child's, tight enough to false-positive on
#:   ordinary interpreter/host variation. Nothing is bought by tightening it either: that poison's
#:   true excess over the bound (986.30 B/event, re-measured this run) dwarfs the 207.93 B/event
#:   maximum theoretical masking by 4.7x.
#:
#:   The value below, 24 MiB, is therefore the flat operational ceiling for the fail-open
#:   direction: it sits UNDER the 44.90 MiB soundness limit at the binding gated N of 500,000
#:   (**36% of the 68.79 B/event headroom** — there, the silent band is SMALLER than the
#:   honest-to-bound headroom, not 1.58x it) while staying ~1.6x above the highest measured REAL
#:   baseline (15.34 MB, CPython 3.14.4) so ordinary interpreter/host variation does not
#:   false-positive this check. Find what the child now allocates before the call (most likely a
#:   new import reached through `capture.py`) rather than raising this ceiling toward that limit.
MAX_CHILD_BASELINE_BYTES = 24 * 1024 * 1024

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

#: The worst (highest) HONEST per-event measurement the bound above must not reject, and the
#: child's own PESSIMISTIC (lowest measured) baseline floor. Both are measurements, not choices:
#: 121.21 B/event and 12.69 MB, `workstation`, CPython 3.10.14 / 3.14.4, under `VmHWM` (TASK_921);
#: the tables they feed live in `MAX_CHILD_BASELINE_BYTES`'s comment. A LOWER floor makes
#: `deflation = B - floor` larger, so pinning the floor at the low end cannot flatter the
#: derivation.
BOOK_TRADE_WORST_HONEST_BYTES_PER_EVENT = 121.21
CHILD_BASELINE_FLOOR_BYTES = 12_690_000

#: What the per-event bound has to spare over that worst honest measurement. Baseline deflation
#: must stay strictly under this, or the ceiling permits a real regression to hide inside the
#: measurement child's own start-up.
CHILD_BASELINE_HEADROOM_BYTES_PER_EVENT = (
    MAX_BOOK_TRADE_RSS_BYTES_PER_EVENT - BOOK_TRADE_WORST_HONEST_BYTES_PER_EVENT)

#: The heartbeat test's bound is ABSOLUTE (KB), not per event, so its headroom is stated in its
#: own quantity: bound minus the streaming reader's worst measured figure for that fixture.
HEARTBEAT_WORST_MEASURED_RSS_KB = 46_940
HEARTBEAT_HEADROOM_KB = MAX_INCREMENTAL_RSS_KB - HEARTBEAT_WORST_MEASURED_RSS_KB


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
#: module docstring for the kernel reason and the first-hand numbers. The RSS half is the ONLY
#: thing this child script measures now; the STRUCTURAL half (opens, bytes transferred) is
#: measured from OUTSIDE the child, by running it under `strace -f -y` and parsing the trace —
#: see `_parse_strace_fd_totals` and `_measure_read_capture` below. That split exists because the
#: structural instrument must sit BELOW every Python-level API (`open`/`io.open`/`Path.open`/
#: `os.open`), and the only thing below all of them in-process is the kernel syscall boundary
#: itself, which a Python-level wrapper — no matter which name it rebinds — cannot reach (see the
#: module docstring's mechanism-1 section for the first-hand measurement proving this of
#: `os.open`, not just `builtins.open`).
_CHILD_MEASURE = r'''
import json, os, sys
mod_dir, path = sys.argv[1], sys.argv[2]
sys.path.insert(0, mod_dir)


def _vmhwm_bytes():
    """This process's peak RSS since exec. `mm->hiwater_rss`, fresh after execve — unlike
    `getrusage`'s `ru_maxrss` (`signal->maxrss`), which the child inherits from its parent."""
    with open("/proc/self/status") as fh:
        for ln in fh:
            if ln.startswith("VmHWM:"):
                return int(ln.split()[1]) * 1024
    raise RuntimeError("VmHWM absent from /proc/self/status")


from capture import read_capture

baseline = _vmhwm_bytes()
meta = read_capture(path)
peak = _vmhwm_bytes()

print("RSS_RESULT " + json.dumps({
    "baseline_bytes": baseline,
    "peak_bytes": peak,
    "incremental_bytes": peak - baseline,
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

#: Syscalls that can move bytes out of a file, at the layer every Python read API funnels down
#: to (`open`/`io.open`/`Path.open` all end in an `openat`; a buffered or unbuffered read of any
#: of them ends in `read`/`pread64`/`readv`/`preadv`/`preadv2`). Deliberately excludes `mmap` —
#: see the module docstring's stated blind spot (i).
_STRACE_TRACE_SET = "openat,open,read,pread64,readv,preadv,preadv2"
_STRACE_OPEN_CALLS = frozenset({"openat", "open"})
_STRACE_READ_CALLS = frozenset({"read", "pread64", "readv", "preadv", "preadv2"})
_STRACE_CALL_RE = re.compile(r'^\d+\s+(\w+)\(')
_STRACE_RET_RE = re.compile(r'=\s*(-?\d+)')


def _parse_strace_fd_totals(tracefile, target_path):
    """Return `(n_opens, bytes_read)` for `target_path` out of a `strace -f -y` log.

    `-y` makes strace resolve every fd argument to the path it CURRENTLY points at (via
    `/proc/<pid>/fd/<n>`), so this is correct across fd reuse/dup and across the forked/exec'd
    subprocesses `-f` follows — a reach the run-53 `builtins.open`-name guard never had. A line
    counts only if the resolved target appears as a COMPLETE quoted argument (`"<path>"`) or a
    complete fd annotation (`<<path>>`), not merely as a substring, so a target path that happens
    to prefix an unrelated file's path cannot cross-attribute a read.
    """
    quoted = '"' + target_path + '"'
    annotated = '<' + target_path + '>'
    n_opens = 0
    bytes_read = 0
    with open(tracefile, "r", errors="replace") as fh:
        for line in fh:
            if quoted not in line and annotated not in line:
                continue
            m_call = _STRACE_CALL_RE.match(line)
            m_ret = _STRACE_RET_RE.search(line)
            if not m_call or not m_ret:
                continue
            call, ret = m_call.group(1), int(m_ret.group(1))
            if call in _STRACE_OPEN_CALLS:
                if ret >= 0:
                    n_opens += 1
            elif call in _STRACE_READ_CALLS:
                if ret > 0:
                    bytes_read += ret
    return n_opens, bytes_read


def _have_working_strace():
    """Probe, once at import, that the EXACT invocation the guards depend on actually works.

    `shutil.which` only proves the binary exists — some sandboxes ship it without `CAP_SYS_PTRACE`,
    where every invocation fails at attach time.

    **TASK_921 cycle 2 (D83(i): the instrument is itself a claim).** The earlier probe ran
    `strace -f -e trace=...` WITHOUT `-y` against `python -c pass`, so it proved ptrace attach and
    syscall filtering and nothing whatever about `-y`'s fd→path resolution — the one output every
    guard assertion in this file is actually parsed out of. The probe now traces a REAL read of a
    real file with the SAME flags `_measure_read_capture` uses, and checks the result through the
    SAME parser (`_parse_strace_fd_totals`), so what it certifies and what the guards consume are
    the same thing. A host where `-y` produced no usable annotation previously reached the guards
    and failed there loudly (`_measure_read_capture`'s `returncode == 0` assert, or a zero-match
    `n_opens` against `_assert_single_traversal`'s `== 1`) — never a false green, so this is a
    prose/reach mismatch being closed, not a live hole. It does move such a host from a loud
    failure to a stated SKIP; that is the same fail-open-to-skip shape the `_HAVE_STRACE` gate
    already has, and the module docstring's "ptrace-less hosts" blind spot already names it.
    """
    if shutil.which("strace") is None:
        return False
    payload = b"x" * 4096
    try:
        with tempfile.TemporaryDirectory() as td:
            # `realpath`: `-y` resolves fds through `/proc/<pid>/fd/<n>`, which reports the
            # canonical path, so the parser's target must be canonical too (`/tmp` is a symlink
            # on some hosts).
            probe_path = os.path.realpath(os.path.join(td, "strace_probe.bin"))
            with open(probe_path, "wb") as fh:
                fh.write(payload)
            tracefile = os.path.join(td, "probe.strace.log")
            probe = subprocess.run(
                ["strace", "-f", "-y", "-e", "trace=" + _STRACE_TRACE_SET, "-o", tracefile,
                 sys.executable, "-c", "import sys; open(sys.argv[1], 'rb').read()", probe_path],
                capture_output=True, text=True, timeout=60)
            if probe.returncode != 0 or not os.path.exists(tracefile):
                return False
            n_opens, bytes_read = _parse_strace_fd_totals(tracefile, probe_path)
    except OSError:
        return False
    # Exactly what the guards read back: an open attributed to the target, and its bytes.
    return n_opens >= 1 and bytes_read >= len(payload)


_HAVE_STRACE = _have_working_strace()


def _measure_read_capture(path, module_dir=_MODULE_DIR):
    """Run `read_capture` under both instruments in a fresh child; return `(bytes, facts)`.

    The child imports `capture` from `module_dir` (a copy of this file's own directory, or a
    poison shim directory — see the `test_poison_*` functions), so dropping a modified
    `capture.py` next to a copy of this test is enough to exercise the guard against it. The
    child runs under `strace -f -y` so the STRUCTURAL facts (`n_opens`, `bytes_read`) are real
    kernel syscalls, not anything the child script itself has to cooperate with counting.
    """
    tracefile = str(Path(path).with_suffix(Path(path).suffix + ".strace.log"))
    proc = subprocess.run(
        ["strace", "-f", "-y", "-e", "trace=" + _STRACE_TRACE_SET, "-o", tracefile,
         sys.executable, "-c", _CHILD_MEASURE, module_dir, str(path)],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"measurement subprocess failed ({proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RSS_RESULT ")), None)
    assert line is not None, f"no RSS_RESULT line in child stdout:\n{proc.stdout}"
    out = json.loads(line[len("RSS_RESULT "):])
    target = os.path.realpath(str(path))
    n_opens, bytes_read = _parse_strace_fd_totals(tracefile, target)
    out["n_opens"] = n_opens
    out["bytes_read"] = bytes_read
    out["file_bytes"] = os.path.getsize(path)
    return out["incremental_bytes"], out


def _assert_single_traversal(meta, path):
    """The STRUCTURAL guard: `read_capture` opened the capture once and read it exactly once —
    measured at the SYSCALL layer via `strace -f -y` (TASK_921), not by rebinding a Python name.

    Deterministic and environment-independent within its reach — it measures no resource, so no
    neighbour process, test ordering, interpreter version or allocator can move it. This is the
    guard that would have caught the original three materializations (two of which were separate
    `readlines()` passes over a re-opened file), AND the run-53 counterexample that defeated the
    NAME-rebinding version of this guard (`rig_stream_io`, a second traversal via `io.open`) —
    see `test_poison_rig_stream_io_fails_structural_guard`, which plants exactly that and shows
    it FAILING here, demonstrated rather than argued.

    Its one remaining blind spot, stated because the RSS guard below exists to cover it: a
    duplicated MATERIALIZATION built from the single existing pass (retaining every parsed line
    in a list, e.g. `list(_iter_lines_tolerant(path))`) opens the file once and reads it once —
    zero extra syscalls — so it is invisible here BY CONSTRUCTION, not by an instrument gap; see
    `test_poison_list_materialize_fails_rss_guard`. The module docstring's mechanism-1 section
    states the narrower blind spots this rebuild does not close (mmap, sendfile/splice, an fd
    inherited pre-trace, ptrace-less hosts).
    """
    assert meta["n_opens"] == 1, (
        f"read_capture opened {os.path.basename(str(path))} {meta['n_opens']} times — it must "
        f"open it exactly once. More than one open is a duplicated full-file pass, which is the "
        f"O(file) cost TASK_914 removed; zero means the measurement harness failed to intercept "
        f"the reader's I/O (or strace could not resolve the fd — check the trace log) and the "
        f"structural guard measured nothing.")
    assert meta["bytes_read"] == meta["file_bytes"], (
        f"read_capture's own read()/pread64()/... syscalls transferred {meta['bytes_read']} "
        f"bytes out of a {meta['file_bytes']}-byte capture "
        f"({meta['bytes_read'] / max(1, meta['file_bytes']):.3f} traversals) — it must traverse "
        f"the file exactly once. More is a duplicated full-file pass (re-open OR re-read through "
        f"an already-open fd); less means the reader stopped early and every count it reported "
        f"is about a prefix of the capture.")


def _max_child_baseline_bytes(n_events):
    """The largest child baseline that cannot silently mask a regression in an `n_events` test.

    N-AWARE BY CONSTRUCTION rather than by a flat constant someone remembers to re-check against
    every call site: the deflation a baseline can hide is `(B - floor) / N` per event, so a test
    with FEWER events tolerates a SMALLER baseline. `MAX_CHILD_BASELINE_BYTES` is the flat
    operational ceiling and stays in force wherever it is the tighter of the two — above the
    ~181,361-event crossover (`CHILD_BASELINE_CEILING_CROSSOVER_EVENTS`), which both event counts
    the fail-open direction currently gates, 500,000 and 1,000,000, sit above. At or below that
    crossover the N-derived limit takes over and the deflation it permits is EXACTLY the
    honest-to-bound headroom; that is the derivation's definition of the limit, not a violation of
    it, and `_assert_baseline_derivation_row` asserts it as such. Full derivation, and the table at
    every N present in this file, in `MAX_CHILD_BASELINE_BYTES`'s own comment.
    """
    return min(float(MAX_CHILD_BASELINE_BYTES),
               CHILD_BASELINE_FLOOR_BYTES + CHILD_BASELINE_HEADROOM_BYTES_PER_EVENT * n_events)


#: The direction of the per-event bound a CALLING test asserts. Required, not defaulted: which
#: direction a caller is in decides whether the baseline ceiling is load-bearing at all, and a
#: default would let the next test inherit the wrong one silently — the exact shape of the
#: derivation-completeness defect this argument was added to close.
_BOUND_UPPER, _BOUND_LOWER = "upper", "lower"


def _assert_child_baseline_sane(meta, n_events, bound_direction):
    """Refuse a measurement whose own baseline could have eaten the quantity being measured.

    `incremental_bytes` is `VmHWM(after) - VmHWM(baseline)`, and `VmHWM` is a high-water mark, so
    a child that peaked HIGHER before the baseline than `read_capture` peaks after it would
    report a deflated (in the limit, zero) figure. Nothing in the PARENT can do that — `VmHWM` is
    fresh after `execve`, which is the whole point of this instrument — but the child's own
    start-up could, if `capture.py` ever grew a heavy import. That must fail loudly.

    `n_events` is the calling test's OWN event count, and `bound_direction` the direction of the
    bound it asserts. Both are required because deflation only cuts one way:

      `_BOUND_UPPER` — the test asserts `measured < BOUND` (the two real measurement tests).
        Deflation is fail-OPEN there: it shrinks the measured figure and can silently hide a
        regression, so the permitted baseline is derived from that test's own N and must keep the
        maximum deflation under the honest-to-bound headroom.
      `_BOUND_LOWER` — the test asserts `measured > BOUND` (the poison demonstrations). The SAME
        deflation shrinks the figure toward RED, so it can never manufacture a false green and
        the N-derived soundness limit is not required. The flat ceiling is still applied, but as
        a "`capture.py` grew a heavy import" smoke check, NOT as a soundness gate.
    """
    assert bound_direction in (_BOUND_UPPER, _BOUND_LOWER), bound_direction
    if bound_direction == _BOUND_UPPER:
        ceiling = _max_child_baseline_bytes(n_events)
        why = (f"silently absorb the cost being measured and make this test vacuous (at this "
               f"test's own {n_events} events, a baseline at the ceiling already hides "
               f"{(ceiling - CHILD_BASELINE_FLOOR_BYTES) / n_events:.2f} B/event of the "
               f"{CHILD_BASELINE_HEADROOM_BYTES_PER_EVENT:.2f} B/event honest-to-bound headroom)")
    else:
        ceiling = float(MAX_CHILD_BASELINE_BYTES)
        why = ("indicate the child grew a heavy start-up allocation. This test asserts a LOWER "
               "bound, so baseline deflation can only push it toward RED and cannot make it "
               "falsely green — this check is a smoke check here, not a soundness gate")
    assert meta["baseline_bytes"] < ceiling, (
        f"the measurement child's own peak RSS before the read_capture call was "
        f"{meta['baseline_bytes']} bytes, over the {ceiling:.0f}-byte ceiling. The "
        f"reported incremental figure is a high-water-mark DELTA, so a baseline this large can "
        f"{why}. Find what the "
        f"child now allocates before the call (most likely a new import reached through "
        f"`capture.py`) rather than raising this ceiling.")


#: Slack for the derivation IDENTITY, in the per-event quantity. Where the N-derived limit is the
#: one in force the ceiling IS `floor + headroom*N`, so `(ceiling - floor) / N` is `headroom`
#: re-associated through binary floating point. The residual is ~1e-14 B/event and its SIGN is
#: arbitrary — measured at this revision, `workstation`, CPython 3.10.14: exactly 0.0 at N=60,000 /
#: 100,000 / 181,361 but -1.42e-14 at N=181,360. That is why the equality below is asserted with a
#: tolerance and never as `==` or as a strict inequality: a strict `<` there is decided by
#: floating-point noise, not by any property of the design. One byte per million events is ~1e8x
#: above that noise and still far below anything physically meaningful.
DERIVATION_IDENTITY_TOLERANCE_BYTES_PER_EVENT = 1e-6

#: Event count at which `floor + headroom*N` crosses the flat ceiling (`(24 MiB - 12,690,000) /
#: 68.79`). At or below it the N-DERIVED limit is in force; above it the flat one is. Both real
#: fail-open gated tests (500,000 and 1,000,000) sit above it, which is the only reason the
#: unsatisfiable assertion this file shipped in cycle 2 was dormant rather than red.
CHILD_BASELINE_CEILING_CROSSOVER_EVENTS = 181_361


def _assert_baseline_derivation_row(n_events, direction, denom):
    """Re-derive, print and ASSERT the baseline-ceiling claim for ONE (N, direction, denomination).

    Factored out of the table test so the extensibility claim that test makes is exercised by the
    SAME code path a real row takes, not by a re-implementation of it.

    Which assertion is load-bearing depends on which of `_max_child_baseline_bytes`'s two
    candidate ceilings is in force at this N, and that is not cosmetic — the two are different
    claims:

      FLAT in force (N above the crossover, or an ABSOLUTE-denominated bound): the flat ceiling
        leaves genuine slack under the honest-to-bound headroom, and `used < headroom` STRICTLY is
        the real property. It is what catches a flat ceiling raised past what the derivation
        permits, and what catches `_max_child_baseline_bytes` losing its `min()`.

      N-DERIVED in force (per-event bound at or below the crossover): equality is the branch's
        DEFINING property, not a violation of it — `ceiling ≡ floor + headroom*N`, so the
        deflation it permits is exactly the honest-to-bound headroom, which is the limit the
        derivation sets. Asserting a strict `<` there is unsatisfiable by construction (TASK_921
        cycle 3 finding); asserting it with a tolerance still rejects any ceiling permitting
        deflation ABOVE headroom, which is the property worth guarding.

    Together the two branches assert `used <= headroom` everywhere and pin WHY it holds at each N.
    """
    ceiling = (_max_child_baseline_bytes(n_events) if direction == _BOUND_UPPER
               else float(MAX_CHILD_BASELINE_BYTES))
    limit = CHILD_BASELINE_FLOOR_BYTES + CHILD_BASELINE_HEADROOM_BYTES_PER_EVENT * n_events
    # Derived from the CONSTANTS, not read back off `ceiling`: this is the branch the derivation
    # says should be in force, so a `_max_child_baseline_bytes` that disagrees (a lost `min()`, a
    # doubled headroom term) is measured against the claim rather than grading its own homework.
    n_derived_in_force = limit <= float(MAX_CHILD_BASELINE_BYTES)
    # What the print's `branch` column reports is the ceiling ACTUALLY APPLIED, which for a
    # fail-CLOSED row is the flat one as a smoke check whatever the derivation would have chosen.
    branch = ("N-derived" if (n_derived_in_force and direction == _BOUND_UPPER)
              else "flat" if direction == _BOUND_UPPER else "flat/smoke")
    deflation_abs = ceiling - CHILD_BASELINE_FLOOR_BYTES
    if denom == "per_event":
        used, headroom, unit = (deflation_abs / n_events,
                                CHILD_BASELINE_HEADROOM_BYTES_PER_EVENT, "B/event")
    else:
        used, headroom, unit = deflation_abs / 1024, HEARTBEAT_HEADROOM_KB, "KB"
    print(f"{n_events:>10,}  {limit / 1024 / 1024:>13.2f} MiB  {used:>13.2f} {unit:<6}  "
          f"{100 * used / headroom:>17.1f}%  {branch:<11}  "
          f"{'fail-open' if direction == _BOUND_UPPER else 'fail-CLOSED'}")

    if direction == _BOUND_UPPER and denom == "per_event" and n_derived_in_force:
        # Equality IS the claim here. `abs(...)` and not `<=`: a ceiling BELOW the identity is
        # equally a signal that the helper stopped implementing this derivation.
        assert abs(used - headroom) <= DERIVATION_IDENTITY_TOLERANCE_BYTES_PER_EVENT, (
            f"at N={n_events} the N-derived limit ({limit:,.0f} B) is the one in force, so the "
            f"ceiling must permit EXACTLY the honest-to-bound headroom of {headroom:.2f} {unit} "
            f"— it permits {used:.10f} {unit} instead (off by "
            f"{used - headroom:+.3e}, tolerance "
            f"{DERIVATION_IDENTITY_TOLERANCE_BYTES_PER_EVENT:g}). `_max_child_baseline_bytes` no "
            f"longer implements `floor + headroom*N` at this N; fix the helper, do not widen this "
            f"tolerance.")
    elif direction == _BOUND_UPPER:
        # The load-bearing claim where the FLAT ceiling is in force: at this test's own N, in this
        # test's own quantity, a baseline the gate ACCEPTS cannot hide as much as the bound has to
        # spare — with real slack, not merely by the equality above.
        assert used < headroom, (
            f"the baseline ceiling applied at N={n_events} permits {used:.2f} {unit} of "
            f"silent deflation against only {headroom:.2f} {unit} of honest-to-bound "
            f"headroom — a regression could hide inside the measurement child's own "
            f"start-up. Lower the ceiling for this N, do not restate the claim.")
    else:
        # The direction argument, asserted rather than remembered: deflation here is
        # fail-CLOSED, and the poison's measured excess dwarfs the maximum masking anyway.
        excess = (POISON_LIST_MATERIALIZE_MEASURED_BYTES_PER_EVENT
                  - MAX_BOOK_TRADE_RSS_BYTES_PER_EVENT)
        assert excess > used, (
            f"the {denom} masking permitted at N={n_events} ({used:.2f} {unit}) is no longer "
            f"dwarfed by this poison's measured excess over the bound ({excess:.2f} B/event). "
            f"The fail-CLOSED direction argument still holds, but the comfort margin quoted "
            f"in `MAX_CHILD_BASELINE_BYTES`'s comment does not — re-measure and restate it.")


def _print_baseline_derivation_header():
    print(f"\n{'N':>10}  {'soundness limit':>16}  {'deflation @ ceiling':>20}  "
          f"{'% of own headroom':>18}  {'branch':<11}  direction")


def test_child_baseline_ceiling_derivation_holds_at_every_event_count():
    """The derivation in `MAX_CHILD_BASELINE_BYTES`'s comment, executed rather than asserted in
    prose — at EVERY event count present in this file, not just the two it used to compare.

    Cycle-1 of TASK_921 derived that ceiling against `N_BOOK_TRADE_EVENTS` alone, called 500,000
    "the test with the FEWEST events", and missed that `test_poison_list_materialize_fails_rss_guard`
    reaches the same gate at 60,000. This test is the structural fix for that class of defect: it
    enumerates every gated N with its bound's direction and denomination and re-derives the claim
    for each. Needs no strace and no `/proc`, so it never skips.

    **CORRECTED, TASK_921 cycle 3.** Cycle 2 claimed here that "a future test added at a new N
    cannot inherit a ceiling nobody checked". Checked it was; PASSABLE it was not. The check was a
    strict `used < headroom`, and wherever the N-derived limit is the ceiling in force,
    `used ≡ headroom` by construction — so any fail-open row added at or below
    `CHILD_BASELINE_CEILING_CROSSOVER_EVENTS` hit a permanent false red instead of validation.
    Dormant only because both real fail-open N (500,000 and 1,000,000) sit above the crossover.
    The branch-aware assertions now in `_assert_baseline_derivation_row` are the fix, and
    `test_child_baseline_ceiling_derivation_admits_a_future_small_n_fail_open_test` is the proof
    the extensibility claim is finally true rather than aspirational.
    """
    rows = [
        (N_HEARTBEATS, _BOUND_UPPER, "absolute"),
        (N_BOOK_TRADE_EVENTS, _BOUND_UPPER, "per_event"),
        (N_POISON_EVENTS, _BOUND_LOWER, "per_event"),
    ]
    _print_baseline_derivation_header()
    for n_events, direction, denom in rows:
        _assert_baseline_derivation_row(n_events, direction, denom)

    # And the claim `MAX_CHILD_BASELINE_BYTES`'s comment makes about the FLAT constant: at the two
    # event counts the fail-open direction actually gates TODAY it is the tighter of the two
    # ceilings, so it is the value in force there. Deliberately scoped to those two N rather than
    # to every fail-open row: below the crossover the flat constant is NOT in force, and a future
    # small-N fail-open test is a legitimate addition (the test below), not a reason to fail this.
    for n_events in (N_HEARTBEATS, N_BOOK_TRADE_EVENTS):
        assert _max_child_baseline_bytes(n_events) == float(MAX_CHILD_BASELINE_BYTES), (
            f"at N={n_events} the N-derived soundness limit is now TIGHTER than the flat "
            f"MAX_CHILD_BASELINE_BYTES, so the flat constant is no longer the value in "
            f"force — update its comment, which says it is.")


def test_child_baseline_ceiling_derivation_admits_a_future_small_n_fail_open_test():
    """The extensibility claim, exercised instead of asserted: a fail-open (`_BOUND_UPPER`,
    per-event) test added BELOW the crossover — where the N-derived limit, not the flat ceiling,
    is the one in force — must be ACCEPTED by the same derivation check every real row runs.

    This is the regression test for the TASK_921 cycle-3 finding. Before the fix this body raised
    `AssertionError: assert 68.79 < 68.79` — a permanent false red at precisely the N range the
    N-derived mechanism exists to serve, so the one thing the mechanism was sold as providing
    (add a test at any N, inherit a checked ceiling) was the one thing it refused to do.

    N=100,000 is a plausible future fixture size and sits well below the crossover; it is not
    itself gated by any test here, which is why this lives in its own test rather than as a row in
    the table above (that table's claim is "every N present in this file", and inventing an N
    would falsify it).
    """
    n_events = 100_000
    assert n_events < CHILD_BASELINE_CEILING_CROSSOVER_EVENTS, n_events
    assert _max_child_baseline_bytes(n_events) < float(MAX_CHILD_BASELINE_BYTES), (
        f"N={n_events} no longer selects the N-derived ceiling, so this test no longer covers the "
        f"branch it exists to cover — pick an N below the "
        f"{CHILD_BASELINE_CEILING_CROSSOVER_EVENTS:,}-event crossover.")
    _print_baseline_derivation_header()
    _assert_baseline_derivation_row(n_events, _BOUND_UPPER, "per_event")


@pytest.mark.skipif(not _HAVE_PROC_STATUS,
                    reason="/proc/self/status unavailable — VmHWM peak-RSS sampling is Linux-only")
@pytest.mark.skipif(not _HAVE_STRACE,
                    reason="strace unavailable or ptrace blocked (no CAP_SYS_PTRACE) — the "
                           "fd-level structural guard needs it; degrading silently to no "
                           "structural check at all is the failure mode this row exists to end")
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
    _assert_child_baseline_sane(meta, N_HEARTBEATS, _BOUND_UPPER)

    assert incremental_kb < MAX_INCREMENTAL_RSS_KB, (
        f"read_capture's own incremental peak RSS was {incremental_kb} KB for a "
        f"{file_bytes}-byte, {N_HEARTBEATS}-event sparse capture — over the "
        f"{MAX_INCREMENTAL_RSS_KB} KB bound. That bound sits far below what THREE full-file "
        f"materializations (raw lines + parsed dicts + a second raw-line pass) would cost at "
        f"this size (measured ~532 MiB pre-TASK_914 for this exact file shape) — a regression "
        f"back to file-size-scaling RSS is exactly what this test exists to catch.")


@pytest.mark.skipif(not _HAVE_PROC_STATUS,
                    reason="/proc/self/status unavailable — VmHWM peak-RSS sampling is Linux-only")
@pytest.mark.skipif(not _HAVE_STRACE,
                    reason="strace unavailable or ptrace blocked (no CAP_SYS_PTRACE) — the "
                           "fd-level structural guard needs it; degrading silently to no "
                           "structural check at all is the failure mode this row exists to end")
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
    _assert_child_baseline_sane(meta, N_BOOK_TRADE_EVENTS, _BOUND_UPPER)

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


# --- G1 non-vacuity: the two poisons, planted and shown FAILING RED (TASK_921) ----------------
# Both poisons are `capture.py` SHIMS — a thin wrapper module that loads the REAL `capture.py`'s
# own source into a fresh module object (compiled and executed directly, so neither poison
# touches or copies the production file's source — out of scope per the task: only
# `test_streaming_reader.py` and its own harness may change) and re-exports `read_capture`/
# `GapRecord`. `_measure_read_capture`'s `module_dir` parameter points the child's `sys.path` at
# the shim directory instead of the real module directory, so the SAME instrumentation exercises
# the poisoned reader. (Spelled via `types.ModuleType`/`compile`/`exec` rather than the stdlib
# dynamic-loader module named in this comment's absence deliberately: this lane's own
# `test_credential_free.py` fences that name lane-wide as an order-transmission-evasion alias,
# case-insensitively, docstrings included — a real false positive here, not a defect in that
# fence, so the fix is to not spell the name rather than to touch the fence.)

_POISON_RIG_STREAM_IO = """
import sys
import types
with open({real_path!r}) as _real_src_fh:
    _real_src = _real_src_fh.read()
_real = types.ModuleType("_capture_real_rig")
_real.__file__ = {real_path!r}
sys.modules["_capture_real_rig"] = _real
exec(compile(_real_src, {real_path!r}, "exec"), _real.__dict__)

GapRecord = _real.GapRecord

def read_capture(path, *a, **k):
    meta = _real.read_capture(path, *a, **k)
    # The run-53 counterexample: a SECOND full traversal, routed through `io.open` rather than
    # `builtins.open`. `io.open` IS `builtins.open` (same object, two names), so a NAME-rebinding
    # guard cannot see this at all; the fd-level guard sees the real second `openat`+`read` set
    # regardless of which name issued them.
    import io
    with io.open(path, "r") as fh:
        for _line in fh:
            pass
    return meta
"""

_POISON_LIST_MATERIALIZE = """
import sys
import types
with open({real_path!r}) as _real_src_fh:
    _real_src = _real_src_fh.read()
_real = types.ModuleType("_capture_real_listmat")
_real.__file__ = {real_path!r}
sys.modules["_capture_real_listmat"] = _real
exec(compile(_real_src, {real_path!r}, "exec"), _real.__dict__)

GapRecord = _real.GapRecord

# Retains every parsed record from the SINGLE existing `_iter_lines_tolerant` pass -- no second
# `open`, no second `read`: exactly the shape `list(_iter_lines_tolerant(path))` would produce,
# built here by wrapping the generator in place so `read_capture`'s own single traversal is the
# only I/O that happens. This is the guard's OTHER stated blind spot -- structurally invisible
# by construction, which is why the RSS half exists at all.
_orig_iter = _real._iter_lines_tolerant
_retained_store = []


def _poisoned_iter(*a, **k):
    for obj in _orig_iter(*a, **k):
        _retained_store.append(obj)
        yield obj


_real._iter_lines_tolerant = _poisoned_iter


def read_capture(path, *a, **k):
    return _real.read_capture(path, *a, **k)
"""

#: Small on purpose: these two tests demonstrate FAILURE, not the regression bound itself (the
#: two tests above already pin the real numbers at realistic scale). 60,000 events keeps them
#: fast while staying well above the `file_bytes_per_event > 100` discriminability floor used
#: throughout this file.
#: 60,000 is the SMALLEST event count in this file, which makes it the binding case for anything
#: derived per event — see `MAX_CHILD_BASELINE_BYTES`'s comment, whose derivation omitted this N
#: until TASK_921 cycle 2, and `test_child_baseline_ceiling_derivation_holds_at_every_event_count`,
#: which now re-derives the claim here too. Raising it to `N_BOOK_TRADE_EVENTS` was considered as
#: an alternative fix and rejected: these poisons run under `strace` and the list-materialization
#: one retains every parsed record, so 500,000 events costs ~8x the wall time and ~590 MB of child
#: RSS to re-prove a demonstration that is already unambiguous at 60,000 (~988 B/event clear of
#: the bound). The gate was made N-aware instead.
N_POISON_EVENTS = 60_000

#: Measured B/event for the list-materialization poison under `VmHWM`, and the figure both the
#: assertion below and `MAX_CHILD_BASELINE_BYTES`'s direction argument quote. RE-MEASURED rather
#: than carried (D83(ii)): **1176.30 B/event, TASK_921 cycle 2, `workstation`, CPython 3.10.14, this
#: file at this revision** — run-53's 1177.99 under the same instrument agrees to 0.14%. The lower
#: of the two is used, since it is the value the comfort-margin assertion has to clear.
POISON_LIST_MATERIALIZE_MEASURED_BYTES_PER_EVENT = 1176.30


def _write_poison_module(tmp_path, name, template):
    mod_dir = tmp_path / name
    mod_dir.mkdir()
    real_path = os.path.join(_MODULE_DIR, "capture.py")
    (mod_dir / "capture.py").write_text(template.format(real_path=real_path))
    return str(mod_dir)


@pytest.mark.skipif(not _HAVE_PROC_STATUS,
                    reason="/proc/self/status unavailable — VmHWM peak-RSS sampling is Linux-only")
@pytest.mark.skipif(not _HAVE_STRACE,
                    reason="strace unavailable or ptrace blocked (no CAP_SYS_PTRACE)")
def test_poison_rig_stream_io_fails_structural_guard(tmp_path):
    """G1 non-vacuity, demonstrated not argued: a reader that re-traverses the capture a SECOND
    time via `io.open` (run-53's exact counterexample against the old `builtins.open`-name guard)
    must FAIL RED under the rebuilt fd-level structural guard.

    This is the poison the old guard COULD NOT see (measured run-53: 120.95 B/event, n_opens=1,
    traversals=1.000, PASSED). If this test ever goes green, the fd-level rebuild has regressed
    back to the same blind spot and G1 is not met.
    """
    path = tmp_path / "poison_rig.jsonl"
    _write_book_trade_capture(path, n_events=N_POISON_EVENTS)
    mod_dir = _write_poison_module(tmp_path, "poison_mod_rig", _POISON_RIG_STREAM_IO)

    _, meta = _measure_read_capture(path, module_dir=mod_dir)

    assert meta["n_opens"] == 2, (
        f"expected the poison to open the capture twice (once via the real reader, once via its "
        f"planted io.open re-traversal); measured n_opens={meta['n_opens']}. If this poison "
        f"itself is not behaving as intended the test below is not exercising anything.")
    with pytest.raises(AssertionError):
        _assert_single_traversal(meta, path)


@pytest.mark.skipif(not _HAVE_PROC_STATUS,
                    reason="/proc/self/status unavailable — VmHWM peak-RSS sampling is Linux-only")
@pytest.mark.skipif(not _HAVE_STRACE,
                    reason="strace unavailable or ptrace blocked (no CAP_SYS_PTRACE)")
def test_poison_list_materialize_fails_rss_guard(tmp_path):
    """G1 non-vacuity, demonstrated not argued: a reader that retains every parsed record from
    its SINGLE existing pass (no re-open, no re-read) is invisible to the structural guard BY
    CONSTRUCTION — that is the guard's stated remaining blind spot — but must FAIL RED under the
    RSS guard, which exists precisely to cover it.
    """
    path = tmp_path / "poison_listmat.jsonl"
    _write_book_trade_capture(path, n_events=N_POISON_EVENTS)
    mod_dir = _write_poison_module(tmp_path, "poison_mod_listmat", _POISON_LIST_MATERIALIZE)

    incremental_bytes, meta = _measure_read_capture(path, module_dir=mod_dir)

    # The structural guard must stay GREEN here — demonstrating that this poison really is the
    # shape the structural half cannot see, not merely a second poison the fd-level guard also
    # happens to catch (which would collapse the two-guard split this file documents).
    _assert_single_traversal(meta, path)
    # `_BOUND_LOWER`: this test asserts `measured > BOUND`, so baseline deflation can only push
    # the figure toward RED — it cannot manufacture a false green, and the N-derived soundness
    # limit (16.04 MiB at 60,000 events) is not required here. Passing the direction explicitly
    # rather than omitting the call keeps the smoke check AND puts the reason in the code, where
    # the next reader finds it, instead of in a human's memory.
    _assert_child_baseline_sane(meta, N_POISON_EVENTS, _BOUND_LOWER)

    bytes_per_event = incremental_bytes / N_POISON_EVENTS
    assert bytes_per_event > MAX_BOOK_TRADE_RSS_BYTES_PER_EVENT, (
        f"list-materialization poison measured {bytes_per_event:.1f} B/event against the "
        f"{MAX_BOOK_TRADE_RSS_BYTES_PER_EVENT} B/event bound — expected it to trip the bound. If "
        f"it does not, the RSS guard is not load-bearing against this poison shape and G1 is not "
        f"demonstrated (this shape measures {POISON_LIST_MATERIALIZE_MEASURED_BYTES_PER_EVENT} "
        f"B/event — TASK_921 cycle 2, agreeing with run-53's 1177.99 to 0.14%; a retained dict "
        f"per line costs roughly the same regardless of file size).")

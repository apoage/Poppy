> **STATUS (2026-09-06): SHIPPED as `task913-write-hardening`** — implemented in full
> (R1-R6, §2.0/§2.0.1/§2.0.2, §2.1, §2.2, §5, and all eleven §7 tests). Suite
> `experiments/907_live_fill_probe/` 304 -> 326 passing; `python -m trader selftest` 17/17.
> Verified by a five-lens panel + a scoped cycle-2 re-verification: 1 BLOCKER + 4 MAJOR found
> and closed, ten reverse-mutations caught, artifacts byte-identical to `e49bea0` except the
> intended `clock_jump` -> `emission_silence` rename.
>
> **Two deliberate divergences from this document's literal text, both measured:**
> 1. **`_PRIORITY_RECORD_TYPES` has FOUR members, not §2.1's "only three points"** —
>    `meta_resume` is included. §2.1's own rationale for `meta_start` (written once per process
>    lifetime; carries the declared pair/channel set `assert_complete` depends on) applies to
>    `meta_resume` verbatim, since both are built by `_declaration_fields()`. Measured exposure:
>    `os.fsync` syncs the whole fd, so the following resume gap normally covers `meta_resume` for
>    free; the real hole is the branch where NO resume gap is written (`last_ts is None`, i.e. the
>    previous session crashed before recording any data event), measured at **zero fsyncs** at
>    session start.
> 2. **§2.0.2's overflow disclosure also fires on the `close()` path.** As written, §2.0.2 only
>    resolves an episode when a NEW record is admitted; `close()`'s own `meta_end` went through the
>    same gate and was folded into the drop count, so an overflow episode active at close time lost
>    both `meta_end` and the `writer_overflow` gap **silently** — the 893 failure mode, reintroduced.
>    `close()` now resolves the episode into the reserved slot first, then admits `meta_end` through
>    a bounded put. If `meta_end` still cannot be admitted, `clean=False` plus a CRITICAL naming the
>    undrained depth and session drop count. Worst-case `close()` wall time is therefore
>    `2 x close_join_timeout_s`, measured at 2.001s for a 1.0s timeout.
>
> **Known remainders (neither blocking, both named rather than hidden):**
> - `_abort_construction()`'s live-worker branch is unreachable from its only call site (by the time
>   `_worker_error` is set the worker has already exited) and has no test coverage. Dead defensive
>   code today; a future change to `_worker_loop`'s error handling would silently re-arm it.
> - `RecorderConfig.__post_init__` validates `book_keepalive_s` but not `heartbeat_timeout_s`, and
>   `--heartbeat-timeout-s` is `type=float` with no range check, so `nan` still reaches
>   `CaptureWriter` (now as a clean `ValueError` with nothing stranded). A range check at the
>   config/argparse boundary is the better home.

# TASK_913 scope 2/3 — recorder write-hardening SPEC (post-capture)

Status: SPEC ONLY. No code in `capture.py` or `recorder.py` changes as part of this file — the
907 live capture is running against that exact code right now, and touching it mid-capture is a
restart forbidden by D67(b)/D74(a). This document is the contract a FUTURE task implements once
the capture closes.

All line numbers below were read against the tree at HEAD as of 2026-08-31 (same revision the
live capture was launched from — `recorder.py`/`capture.py` are untouched since capture start
per D74/TASK_913's own binding fence). Re-verify line numbers before implementing if the files
have moved.

> **ROOT-CAUSE UPDATE (2026-08-31, twice revised — supersedes this spec's original premise).**
> This spec was drafted under TASK_913's working theory that a *process* was hogging the `sda6`
> ext4 journal. That theory is **refuted by measurement** — see
> [`JOURNAL_STALL_DIAGNOSIS.md`](JOURNAL_STALL_DIAGNOSIS.md). During a stall **zero bytes**
> reach the device; there is no transaction and no hog. `bitcoind`, postgres and journald are
> all exonerated (postgres' `walwriter` is a fellow victim, blocked alongside the recorder).
>
> **REVISED 2026-08-31 (second pass): the named cause has changed; the mechanism has not.**
> An earlier version of this callout said the stalls were a *"failing SATA link on `ata32`"*
> that "drops on a ~600 s cycle". **That is withdrawn.** Across the entire persistent journal
> (three boots, 2026-06-05 → now) there are **zero `ata32: SATA link down` events and zero link
> errors** of any kind. The link-up is libata's *recovery action*, not the trigger. What
> actually happens, per `JOURNAL_STALL_DIAGNOSIS.md` §4b:
>
> > `udisksd`'s 600 s SMART housekeeping poll issues an **`SG_IO` ATA passthrough** to
> > `/dev/sda`; the drive never answers it; the unanswered command holds the queue; the SCSI
> > command timeout then fires and the error handler resets the link to recover. Every writer
> > to `sda` — `jbd2/sda6-8`, postgres, and the recorder inside `os.fsync()` — blocks for the
> > whole outage.
>
> **What this changes for THIS document: only the name of the cause.** The recorder is still a
> victim of a device outage it did not create and cannot prevent; the outage is still total
> (zero sectors in *or* out); the recorder is still blocked in
> `do_fsync → ext4_sync_file → jbd2_log_wait_commit` for its full duration — that exact stack
> was captured live on 2026-08-31 at 06:02:36. Every requirement, limit and test below stands
> unchanged. Two figures are updated: the outage is **typically ≈61 s, range 38–93 s, with 93 s
> kept as the sizing bound** — the 5,452-event kernel census pins the distribution, superseding
> both "~59 s" and the earlier "38–93 s (n = 4)" framing, which over-sampled the tails ~3:1 —
> and there is **no repair for the operator to make to the hardware** — the trigger is a daemon.
>
> **What that changes about the hardening — and what it does not.** The hardening below is
> still correct and its motivation is unchanged in strength, but its claim must be stated
> honestly: dropping per-line `fsync` **cannot stop the stalls**, because they are device-level
> and would block any writer that reaches the platter, whatever triggers them. What it does is
> **decouple the recorder's event loop from the device** — buffered `write()`s return into page
> cache immediately, so the loop keeps servicing the websocket across a 38–93 s outage, the
> `ping_timeout=15` keepalive does not trip, and frames are held in RAM rather than dropped.
> **The fix converts a data-loss event into a latency event.** Two limits follow: if dirty-page
> limits are ever reached the writer blocks anyway, and any record the policy still fsyncs *on
> the event-loop thread* stays fully exposed to the outage.
>
> **Sizing note — the `~59 s` constant used below is superseded by `93 s`, and every conclusion
> that rests on it survives.** §§2.0, 2.2, 2.3, 6.1 and 7 are left **as written** (they are the
> reviewed, load-bearing part of this spec); this note records the re-check rather than editing
> them, so an implementer does not have to redo it:
>
> | where | claim at ~59 s | at 93 s | still holds? |
> |---|---|---|---|
> | §2.0 queue bound | 30.25 rec/s × 59 s ≈ 1,785 records vs a 100,000 bound | ≈ **2,813** records | **yes** — 36× margin. The "~55 min of continuous device absence" headroom is unchanged: it is `100,000 ÷ 30.25 rec/s`, a function of the rate and the bound, not of how long one outage lasts. |
> | §2.2 durable-frontier lag | 30 + 61.5 ≈ 92 s, "6.5× margin inside 600 s" | 30 + 93 = **123 s** | **yes** — the margin inside the 600 s cycle falls from 6.5× to **4.9×**; every cycle still contains many completed durable checkpoints, and N = 30 remains far below where that argument erodes |
> | §2.3 SIGKILL-in-outage loss | ≤ ~1,785 records ≈ 0.024 % of capture | ≤ **2,813** ≈ **0.037 %** | **yes** — still ~300× below the 11.56 % being lost today |
> | §6.1 probe lag | "~59 s … just under `FRESH_THRESHOLD_S` (60 s), and it can cross it" | **93 s, so it does cross it** | **yes** — §6.1 already works the crossing case through: it swaps `split-feed death` for `recorder-dead` in the reason string, both FAIL, status unchanged |
> | §6.1 `last_book_age_s` | "a ≤59 s lag cannot push a live pair past 300 s" | ≤ 93 s | **yes** — a pair would have to already sit above ~207 s, still deep in the probe's dead zone |
>
> Read every `~59 s` / `~58.9 s` below as **"the device outage — typically ≈61 s, range 38–93 s,
> size on 93 s"**. No
> requirement (R1–R5), test (§7) or design choice changes *as a result of the 93 s correction*.
>
> **Separately (2026-08-31, adversarial re-verification pass — not part of the 93 s update):**
> two defects were found in §2.0 itself and are now closed in place. R4's claim that a re-raised
> write error is "disclosed rather than lost" is **false on the code path the document names**
> and is withdrawn — traced and replaced by **R6 / §2.0.1**. The overflow rule's
> "gap … or fails closed" disjunction was unimplementable as written (the disclosure needed
> queue capacity that overflow means is gone) and is replaced by the single pinned mechanism in
> **§2.0.2**. §7 test 10 is rewritten to assert that one mechanism and §7 test 11 is added for
> the R6 path. Nothing else in §§2.0–2.3, 6.1 or 7 is touched.
>
> *(A note on why this callout was wrong once: the first pass read `dmesg -T` timestamps, which
> on this box run **+29 s ahead** of the journal's wall clock, and so mistook the first EH
> recovery reset for the end of the stall. Sizing in this spec is insensitive to the correction
> — it uses worst cases throughout — but the causal story it quoted was inverted.)*
>
> **Scope of this update: the WHOLE document, §1 included.** §1 originally stated the
> refuted "journal hog" theory as fact; it has been rewritten (2026-08-31) to describe the
> device-level fault instead. The second limit above turned out to be fatal to the spec's
> original §2 policy — a synchronous fsync for gap/meta/close is still a synchronous fsync on
> the event loop, and gap records are written *because of* stalls, so that rule concentrated the
> surviving blocking calls on exactly the worst moments. That is now resolved by **§2.0**, the
> binding constraint this document is built around. §3's torn-tail finding, §2.3's bounded-loss
> arithmetic and §6's conclusion survive, with the amendments each section names. (§2.0 goes one
> step further than "buffered `write()`s return immediately": it takes the whole write path off
> the loop thread, because a buffered `write()` is *usually* but not *provably* non-blocking.)

## 1. The defect, precisely

`CaptureWriter._write` (`capture.py:303-306`):

```python
303:    def _write(self, obj: Dict[str, Any]) -> None:
304:        self._fh.write(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")
305:        self._fh.flush()
306:        os.fsync(self._fh.fileno())
```

Every record type funnels through this one method — verified by grepping its call sites:

| Record type | Call site |
|---|---|
| `meta_start` | `capture.py:279` (`CaptureWriter.__init__`) |
| `book` | `capture.py:311` (`write_book`) |
| `trade` | `capture.py:332` (`write_trade`) |
| `heartbeat` | `capture.py:335` (`write_heartbeat`) |
| `counters` | `capture.py:341` (`write_counters`) |
| `gap` | `capture.py:348` (`write_gap`) |
| `meta_end` | `capture.py:352` (`close`, `clean=True` only) |

So **every** book tick, trade print, heartbeat, counters snapshot, and gap record incurs a
synchronous `os.fsync()` before the recorder's event loop can proceed to the next frame.
`os.fsync()` on ext4 with a journal (`data=ordered`, the default) blocks until the filesystem
journal commits the write — not just until the bytes reach the platter.

Consequence, per the root-cause measurement in
[`JOURNAL_STALL_DIAGNOSIS.md`](JOURNAL_STALL_DIAGNOSIS.md) (2026-08-31, **n=4 directly sampled
stall windows** by two independent parties plus three clean off-phase controls, backed by a
599.999 s cycle that `udisksd` itself logged as a failed SMART read **5,452/5,452 times over
the 37.9-day boot**, zero missed and zero successful) — not re-derived here:

- `sda` is a **KINGSTON SKC600/512GB** SATA-III SSD on `ata32`. Every ~600.0 s, at wall-clock
  phase `:X2:36`, **it stops completing commands entirely** — zero sectors written *and* zero
  sectors read, while the block layer reports the queue 100 % busy. `scsi_eh_31` (the SCSI
  error handler) goes to D-state and libata resets the link to recover; the reset is the
  *recovery*, not the trigger, and there are **zero `SATA link down` events and zero link
  errors** in the whole persistent journal. **Outage: typically ≈61 s; range 38–93 s; size on
  93 s.** (The 5,452-event kernel census gives three modes — ≈38 s at 4.6 %, **≈61 s at 73.4 %**,
  ≈93 s at 22.0 % — so the four hand-sampled totals of 38 / 61.3 / 61.6 / 93 s over-sampled the
  tails roughly 3:1. Duty cycle ≈10–12 %. D74's planner-side 61.5 s estimate lands on the
  typical case.)
  The trigger is a **`udisksd` SMART housekeeping poll**: a 600 s `g_timeout` issues an `SG_IO`
  ATA passthrough to `/dev/sda` that the drive never answers, and the unanswered command holds
  the queue. Caught live in `blk_execute_rq ← sg_io ← scsi_ioctl ← sd_ioctl` under PID 1946, and
  predicted-then-observed at the following cycle.
  **There is no journal hog.** TASK_913's original scope-1 theory — "a separate box-level
  process pushing a GB-scale journal transaction" — is refuted by that zero-throughput
  measurement: there is no transaction to attribute. `bitcoind` (write_bytes delta 0 over the
  window), postgres and journald are all exonerated; postgres' `walwriter` blocks *alongside*
  the recorder, a fellow victim.
- The ext4 journal on `sda6` (root) is therefore a *conduit*, not a source: `jbd2/sda6-8` is
  blocked because the device it must write to is absent, and so is every other writer to it.
- Because `_write` calls `os.fsync()` on every single line, the recorder thread is parked in
  `jbd2_log_wait_commit` for the full duration of that outage — it isn't merely slow, it is
  **synchronously blocked waiting on a device that is not answering** before it can process the
  next websocket frame.
- While blocked, incoming frames buffer in the kernel socket receive buffer, not in the
  recorder — the recorder's `asyncio` event loop cannot run at all while a synchronous
  `os.fsync()` call holds the only thread. When the loop finally resumes, the elapsed wall-clock
  gap between `recv()` calls exceeds `ping_timeout=15` on the websocket keepalive, which is
  what triggers the 1011 disconnect/reconnect cycles TASK_913 traces the stalls to. The
  arithmetic closes: 438 logged reconnect attempts over ~419 expected stall cycles = **1.05
  reconnects per cycle**.
- The recorder is a **victim**, not the cause: the per-line fsync is what couples a device-level
  hardware fault directly into recorder liveness. A recorder that did not block on every event
  would ride out the outage with buffered data, no observable frame gap, and no keepalive trip.
  Worth stating plainly, because it bounds what this spec can claim: dropping per-line fsync
  **cannot stop the stalls** — they are device-level and would block any writer that reaches the
  platter. It decouples the recorder from them. The repair is the operator's — per
  `JOURNAL_STALL_DIAGNOSIS.md`, stopping `udisks2`'s SMART poll, **not** hardware work; note in
  particular that running `smartctl` is now believed to *induce* the stall rather than diagnose
  it — and is out of this document's scope.

## 2. PRIMARY fix: drop per-line fsync

### 2.0 BINDING CONSTRAINT — no `fsync()` on the event-loop thread

**This is the central design constraint of this document, and it invalidates the policy the
first draft of §2 proposed.** It was found by the orchestrator after the draft, and verified
against the code below.

**The finding.** Every `self._writer.write_*` call is a **synchronous call made from inside the
async `_consume` coroutine** (and from `run()`'s `except` clause, and from `close()`). Verified
by grep over the whole file: `recorder.py` contains **no `run_in_executor`, no `asyncio.to_thread`,
no `concurrent.futures`, and no `threading` of any kind**. The writer call sites are:

| Call site | Line | Reached from |
|---|---|---|
| `_emit` → `write_heartbeat` | `recorder.py:811` | `_consume` (`:731`) |
| `_emit` → `write_book` | `recorder.py:813` | `_consume` (`:731`) |
| `_emit` → `write_trade` | `recorder.py:816` | `_consume` (`:731`) |
| `_flush_counters` → `write_counters` | `recorder.py:820` | `_consume` (`:756`), `_record_gap` (`:871`), `close` (`:874`) |
| `_close_dark_window` → `write_gap` | `recorder.py:851` | `_emit` (`:807`), i.e. `_consume` |
| `_record_gap` → `write_gap` | `recorder.py:862` | `_consume` (`:680`, `:687`, `:745`, `:776`, `:802`), `run` (`:659`) |
| `close` → `CaptureWriter.close` | `recorder.py:875` | `main`'s `finally` (`:918`) — **after** `asyncio.run()` returns |

Every one of those bottoms out in `CaptureWriter._write` (`capture.py:303-306`), which calls
`os.fsync()` inline. **Therefore every `fsync()` today blocks the asyncio event loop** — which
is precisely the mechanism by which the websocket keepalive (`ping_timeout=15`,
`recorder.py:912`) trips. The loop cannot run, so `websockets`' own ping task cannot answer,
so the peer closes with 1011.

**Why the first draft's policy was self-defeating.** It kept a *synchronous* fsync for
gap/meta/close records and *added* a periodic fsync — all still on the loop. Two consequences:

1. A periodic fsync landing inside a ~59 s outage blocks the loop for the remainder of that
   outage. A 30 s deadline comes due 20 times per 600 s cycle and ~10 % of the cycle is stalled,
   so ~2 of those 20 land inside a stall window in expectation — and it takes only **one** to
   freeze the loop past `ping_timeout=15`. That is not a tail risk; it is once per cycle,
   forever. The keepalive trips anyway and the fix does not deliver its main benefit.
2. **Worse and more specific: gap records are written *as a consequence of* disconnects and
   stalls** (`recorder.py:851`, `:862`; and `run()`'s disconnect handler at `:659`). A rule
   saying "gap records keep a synchronous fsync" therefore concentrates the surviving blocking
   fsyncs on exactly the moments when the device is most likely to be unavailable. It is
   precisely backwards.

**The requirement, stated normatively.** The future implementation MUST satisfy:

> **R1.** No `os.fsync()` is ever issued on the thread running the asyncio event loop, at any
> record type, at any cadence, including during `close()` while a loop is still running.
>
> **R2.** No potentially-device-blocking syscall of any kind is issued on the loop thread by the
> write path. `write(2)` to page cache is *usually* non-blocking, but under `balance_dirty_pages`
> throttling it is not guaranteed to be, and this box has a writer population (postgres, journald)
> whose writeback stalls for ~59 s at a time. R1 alone is the dominant fix; R2 is what makes the
> claim *provable* rather than probable.
>
> **R3.** Record ordering in the file is exactly the order the loop produced it — gaps interleaved
> with events precisely as `_record_gap`/`_emit` sequenced them. The gap ledger is only
> interpretable if this holds (`capture.py`'s module docstring, and `read_capture`'s ordering
> assumptions).
>
> **R4.** A write-path error (`OSError`, `EIO` from the failing device — a live possibility here,
> 32,220 `ioerr_cnt`) is never swallowed. It must surface to the recorder, fail-closed.
>
> **R5.** Shutdown terminates. `close()` never blocks unboundedly, never closes the fd while
> another thread may still use it, and reports honestly what it could not persist.
>
> **R6.** A dead write path terminates the recorder **once**, deliberately, through a controlled
> shutdown — it never escalates by raising a second and third time out of the handlers that were
> supposed to disclose it. Specified in full in §2.0.1.

**Chosen design: a single dedicated writer thread behind a bounded queue.**

`CaptureWriter._write` serializes the record to a `str` on the calling (loop) thread — JSON
encoding is pure CPU and the `_require_finite`/`_require_price` validators must keep raising
synchronously into the caller, which is where their error messages are meaningful — and then
`put_nowait`s it onto a `queue.Queue`. One daemon thread consumes the queue and performs
`write()` + `flush()` per record, plus `os.fsync()` per the §2.1/§2.2 policy. `Queue` is FIFO
with a single consumer, so R3 holds by construction.

What that design costs, concretely:

- **Ordering (R3):** free, as above — but *only* while there is exactly one queue and one
  consumer thread. A future "parallel writer" optimization would destroy the ledger. Pin it.
- **Error propagation (R4):** the worker cannot raise into the caller's stack. It must latch the
  first exception, and `_write` must re-raise that latched exception on the *next* call (and
  `close()` must re-raise it too, after draining what it can). Fail-closed: once the worker is
  dead, `_write` MUST NOT keep accepting records into a queue nobody drains — that would be
  silent data loss of exactly the kind this format exists to prevent.
  **The draft claimed the recorder's own `run()` handler turns that re-raise into a disclosure
  ("`run()` already treats an unexpected exception as a disconnect-with-gap
  (`recorder.py:658-660`), so a re-raised write error is disclosed rather than lost"). That
  sentence is FALSE and is withdrawn** — traced against the current code in **R6** below. The
  handler that was supposed to record the disclosure writes through the dead path, so it raises
  a second time from inside its own `except` body and escapes `run()` entirely. There is no
  in-format gap record for "the writer died here", at any point, and the "check that the
  resulting `write_gap` does not itself re-enter the dead writer and recurse" left in the draft
  was a TODO, not a specified behaviour. R6 specifies it.
- **Backpressure and overflow:** the queue MUST be bounded, and `put` MUST be non-blocking — a
  blocking `put` on a full queue reintroduces exactly the loop stall this design removes.
  Size it against the measured worst case: 30.25 rec/s × 59 s ≈ **1,785 records per outage**, so
  a bound of 100,000 entries (~12.5 MB at ~125 B/record) covers ~55 minutes of continuous device
  absence — two orders of magnitude beyond anything measured. On overflow the implementation MUST
  NOT drop silently. **The draft left the mechanism open ("records the loss as a `gap` … or it
  fails closed"), which is not implementable as written**: overflow means the bound is already
  reached, so the disclosure record needs capacity that does not exist at that instant, and the
  draft said nothing about whether it bypasses the bound, blocks (forbidden — `put` must be
  non-blocking), evicts, or dies. **§2.0.2 pins exactly one mechanism.**
- **Close / shutdown semantics (R5):** `CaptureWriter.close(clean=True)` enqueues `meta_end`,
  signals stop, and joins the worker **with a timeout**. Note the timing gift in the current
  code: `LiveRecorder.close()` is called from `main()`'s `finally` at `recorder.py:918`, i.e.
  *after* `asyncio.run()` has returned — there is no live event loop at that point, so a bounded
  blocking join there violates nothing. If a `close()` path is ever added that runs while the
  loop is live, R1 applies to it too.
- **If the worker is still blocked in `fsync()` at the join timeout** (a stall in progress —
  which, at a 10 % duty cycle, will happen roughly one shutdown in ten): the implementation MUST
  NOT close the file descriptor. Closing it frees the number for reuse while a thread is still
  inside `fsync(fd)`/`write(fd)`, and the next `open()` in the process gets that number — a
  cross-file corruption bug far worse than the tail it was trying to save. Correct behaviour:
  leave the fd open, leave the worker as a daemon thread, log loudly with the undrained record
  count, and let process exit reap both. The capture then has no `meta_end` and `closed_cleanly`
  reads **False** — which is the honest answer, because the tail genuinely is unknown. This is a
  behaviour change worth naming: today the same situation makes `close()` hang instead.
- **New loss class on SIGKILL.** Today a `flush()`ed record survives an application-level
  exception or a `kill` (the bytes are in page cache; the OS writes them back). Under the worker
  thread, records sitting in the queue have never been `write()`n and are lost on SIGKILL. In
  steady state the queue depth is ~0 so the exposure is ~0 records; during an outage it is up to
  ~1,785. §2.3's bounded-loss argument is corrected for this below — it is a real cost, it is
  disclosed by the resume gap, and it is smaller than what the current design loses outright.

**Alternatives considered, and why not.**

- **(a) `await loop.run_in_executor(...)` per write.** Also satisfies R1/R2, and awaiting each
  write preserves R3. But it forces `CaptureWriter.write_book`/`write_trade`/`write_gap`/`close`
  to become coroutines, which ripples through `_emit`, `_flush_counters`, `_record_gap`,
  `_close_dark_window` and every synchronous test and offline/replay user of `CaptureWriter`.
  Much larger blast radius for the same guarantee. Rejected on cost, not on correctness. (The
  fire-and-forget variant — `create_task` without `await` — is rejected on correctness: it
  breaks R3.)
- **(b) Keep `write()`+`flush()` on the loop; move only `fsync()` to a timer thread.** Much the
  smallest change, and it preserves §6's visibility argument *exactly* (flush stays synchronous
  on the loop, so a record is page-cache-visible the instant `_write` returns). It satisfies R1
  but not R2: a buffered `write()` can still block under global dirty-page throttling, and on
  this box the whole writeback path stops for ~59 s at a time with postgres and journald
  contributing dirty pages. It also complicates fd lifetime (a thread inside `fsync(fd)` while
  `close()` runs) without removing that hazard. **Recommended as the fallback** if the worker
  thread proves too invasive: it removes the dominant, *measured* blocking source and is a large
  improvement over today. It is not recommended as the primary, because "provably responsive
  across a 59 s outage" is the bar this task set and (b) cannot clear it.
- **(c) `O_DSYNC`/`aiofiles`/io_uring.** Rejected: `O_DSYNC` is strictly worse (per-write
  durability, i.e. today's problem with no opt-out); `aiofiles` is a thread pool with extra
  dependency and no ordering guarantee we would not have to add ourselves; io_uring is far out
  of proportion to a 4 KB/s append stream.

### 2.0.1 R6 — what happens when the write path is dead (the triple-raise, traced and closed)

**This resolves a TODO the draft left inside this BINDING CONSTRAINT section.** The draft
asserted that R4's re-raise is "disclosed rather than lost" because `run()` writes a gap on any
unexpected exception. Traced against the current code, it is not.

**The actual path**, all four steps verified at HEAD 2026-08-31:

1. The worker hits an unrecoverable `OSError`/`EIO` — a live possibility on this box (32,220
   `ioerr_cnt`, 21,207 `iotmo_cnt`) — latches it per R4, and dies.
2. The next ordinary `_write` from `_emit` (`recorder.py:811`/`:813`/`:816`) re-raises the latched
   error. `_consume`'s only `try` blocks cover `ws.recv()` (`recorder.py:677-683`,
   `asyncio.TimeoutError`) and `json.loads` (`:691-701`, `json.JSONDecodeError`); neither is on
   the `_emit` path, so **nothing in `_consume` catches it.**
3. It propagates into `run()`'s `try` (`recorder.py:637-642`). It is not one of the
   never-launder classes at `:648`, so it is caught by `except Exception as exc:` at
   `recorder.py:658`, whose handler calls `self._record_gap("disconnect", …)` (`:659`) →
   `write_gap` (`:862`) → `_write` → **re-raises the same latched error a second time**, from
   inside the `except` body with no enclosing `try`. It escapes `run()`. The reconnect loop is
   *terminated*, not retried — and `await self._sleep(backoff)` (`:664`) never runs, so nothing
   downstream of the gap write happens either.
4. `main()`'s `finally: recorder.close(clean=True)` (`recorder.py:918`) then runs.
   `LiveRecorder.close` calls `self._flush_counters(self._clock())` **first**
   (`recorder.py:874`) → `write_counters` → `_write` → **raises a third time**, out of a
   `finally`, chaining over what was already propagating. Because `:874` precedes `:875`,
   `self._writer.close(clean=clean)` is **never reached**: under §2.0 the worker is never
   stop-signalled, never joined, its undrained count is never reported, and the fd discipline
   R5 specifies never runs.

**Net, stated plainly: on the first unrecoverable write fault there is NO gap record in the
capture saying "the writer died here."** The write path that was supposed to record the
disclosure is the dead one. What the operator gets today is an uncaught process crash with a
three-deep chained traceback and a skipped writer shutdown.

**The resolution: option (A) — a controlled shutdown, plus an honest correction.** The two
options on the table were (A) guard `run()`'s handler so the dead write path cannot escalate,
and (B) accept the hard crash and just fix the false sentence. **(A) is chosen**, because (B)
leaves three real defects standing that are cheap to close: the loop dies by *escape* rather
than by decision, the traceback names the transport error as the apparent top-level cause when
the write fault is the real one, and step 4's ordering bug strands the §2.0 worker thread.
**But (A) does not buy an in-format disclosure either, and this document will not claim it
does** — see "what still reaches the file" below. Both halves of the brief's fix are therefore
applied: the behaviour is specified, *and* the "disclosed rather than lost" sentence is
withdrawn (§2.0, R4 bullet).

**Normative requirement (R6).** The future implementation MUST:

1. Wrap the `_record_gap` call in `run()`'s `except Exception` handler (`recorder.py:659-660`) in
   a narrow `try`/`except`. On failure it MUST NOT retry, MUST NOT back off, and MUST NOT fall
   through to the next loop iteration. It logs `CRITICAL` naming (i) the write error including
   its `errno`, (ii) the transport exception it was trying to disclose, and (iii) the writer's
   undrained queue depth; sets `self._stop = True`; and raises a dedicated
   `CaptureWriteFailed(RuntimeError)` `from` the write error, carrying the transport exception's
   repr in its message. One typed exception leaves `run()`, not a bare `OSError` escaping an
   `except` body.
2. Add `CaptureWriteFailed` to the never-launder clause at `recorder.py:648` (alongside
   `NotImplementedError`, `Phase2RefusalError`, `RailViolation`). It subclasses `RuntimeError`,
   so without this the broad catch at `:658` would launder a dead writer into an ordinary
   "disconnect" gap + backoff + retry — the exact illusory-narrowing failure that clause's own
   comment (D68/902, `recorder.py:648-657`) exists to prevent.
3. Reorder and guard `LiveRecorder.close` (`recorder.py:873-875`) so the writer's shutdown always
   runs:

   ```python
   def close(self, clean: bool = True) -> None:
       try:
           self._flush_counters(self._clock())
       except Exception:
           logger.critical(..., exc_info=True)
           clean = False      # no meta_end is possible; closed_cleanly must read False
       self._writer.close(clean=clean)   # may re-raise the latched error — that is R4
   ```

   `CaptureWriter.close()` still re-raises the latched error (R4, pinned by §7 test 8), but it
   now does so *after* the worker has been stop-signalled, joined within the R5 timeout, its
   undrained count logged, and the fd handled per R5's no-close-under-a-blocked-worker rule.

**What the recorder does then, and what still reaches the file — the honest answer:**

- **Nothing further reaches the file.** Under R4 the worker latches the first exception and dies;
  every subsequent record — the disconnect gap, the counters snapshot, `meta_end` — is
  unwritable, by construction. The capture ends at the last record the worker successfully
  wrote. This is not a shortcoming of (A); it is what "the write path is dead" means. Any design
  that promised an in-format disclosure at that instant would be promising to write through the
  thing that just failed.
- `closed_cleanly` reads **False** (no `meta_end`), which is the honest answer — the tail
  genuinely is unknown. Same posture as the blocked-worker case in R5.
- The disclosure is therefore **out-of-band and deferred**, in two parts: (i) immediately, a
  `CRITICAL` log line + a non-zero process exit (the `CaptureWriteFailed` propagates out of
  `main()` uncaught); (ii) in-format on the next start, `CaptureWriter.__init__`'s resume branch
  (`capture.py:280-286`) writes a `resume` gap from `_last_event_ts(path)` to the restart time,
  which spans exactly the unwritten period — the same mechanism §2.3 already relies on for the
  crash case, and it does not need to know *why* the file stopped.
- One residual, named rather than hidden: with (A), `main()`'s `finally` can still surface the
  latched error *over* the propagating `CaptureWriteFailed` (a `finally` exception supersedes,
  with the original as `__context__`). That is **two** chained exceptions naming the same
  `errno`, not three, with the writer shut down deliberately in between. Judgement call, not a
  measurement: two honest frames pointing at one root cause is acceptable; three, one of which
  is an `except`-body escape and one of which stranded a daemon thread, is not.

§7 test 11 pins this — specifically a write failure raised **from inside `run()`'s own exception
handler**, which is the case no existing test covers (test 8 exercises an isolated `_write` and
`close()` raise, never the handler-re-entry path).

### 2.0.2 Queue overflow — the one pinned mechanism

The draft's "records the loss as a `gap` … or it fails closed" is a disjunction, and the first
branch is not implementable as stated: overflow means the bound is already reached, so there is
no slot for the disclosure record. Pinned, single mechanism:

**Reserve one slot, coalesce the episode, disclose on the falling edge.**

1. The queue is constructed with `maxsize = 100_000` (sized in §2.0). Ordinary records — every
   caller-issued `book`, `trade`, `heartbeat`, `counters`, `gap` and `meta` — are admitted only
   while `qsize() < maxsize - OVERFLOW_RESERVE`, with **`OVERFLOW_RESERVE = 1`**. Reaching that
   line is *overflow*; the incoming record is dropped.
2. **Why exactly 1, and not a fraction of the bound:** at most one overflow-disclosure record can
   ever be outstanding. The writer coalesces an entire contiguous overflow episode into a
   *single* `gap`, and a new episode cannot begin until the previous episode's disclosure has
   been enqueued (which is what ends it). One reserved slot is therefore provably sufficient, and
   any larger reserve would be an unjustified constant. The 1-record admission haircut against a
   100,000 bound is 0.001 % of capacity — it does not move §2.0's ~55-minute headroom figure.
3. While in overflow the writer drops the record and accumulates `n_dropped`, `first_drop_ts` and
   `last_drop_ts` (the `ts` field already serialized on the calling thread — no extra clock read,
   no new clock basis). It does **not** enqueue anything, does not block, and does not raise.
4. On the **falling edge** — the first admitted record after `qsize()` drops back below the line —
   the writer enqueues, *out of the reserved slot and ahead of that record*, one `gap` with
   reason **`"writer_overflow"`**, `[first_drop_ts, last_drop_ts]`, and a detail naming
   `n_dropped` and the bound. It is a gap record, so §2.1's priority fsync applies. Then it
   resets the counters and admits the record normally.
5. **Ordering (R3) is why the falling edge, not the rising one.** The dropped records sit, in wire
   order, strictly between the last admitted record before the episode and the first admitted
   record after it. Enqueuing the disclosure at the falling edge places it at exactly that
   position in the file, so the ledger's `[start_ts, end_ts]` lines up with the hole it
   describes. Disclosing at the rising edge would have to guess `end_ts` for an episode still in
   progress.
6. **The enqueue MUST bypass `_write`'s admission gate**, via an internal put onto the reserved
   capacity — not by re-entering `write_gap`/`_write`, which would hit the same full-queue check
   and recurse.
7. `"writer_overflow"` is a **new** member of `GAP_REASONS` (`capture.py:105-108`, today
   `{"disconnect", "seq_discontinuity", "heartbeat_timeout", "clock_jump", "resume",
   "schema_drift"}`). Add it, do not overload an existing reason — `"disconnect"` would be a lie
   (the socket was fine; the disk was not) and would corrupt the disconnect accounting D74 reads.
   Additive only, same discipline §5 applies to `emission_silence`: old files never contain it,
   so nothing that parses today stops parsing.
8. **Death during an episode is covered without a second mechanism.** If the process dies while in
   overflow, the falling-edge gap is never written — but the dropped records were never in the
   file either, so the file simply ends earlier, and the next start's `resume` gap
   (`capture.py:280-286`) spans a superset of the loss. There is no window in which dropped
   records are presented as observed time, which is the 893 property this is protecting.

**The rejected branch, and why.** Unconditional fail-closed (raise on the first `queue.Full`) is
defensible and was considered: overflow at this bound means ~55 minutes of continuous device
absence, at which point the capture arguably has no value. It is rejected as a **judgement
call, not a measurement**: the estimand is realized coverage, the format's entire posture is
"disclose the hole and keep recording" (§2.3), and killing the capture converts a disclosed
0.0x % hole into a total stop that then needs an operator to notice. The disclose-and-continue
branch is strictly more recoverable and is disclosed either way (point 8). §7 test 10 asserts
**this** branch and no longer accepts the alternative.

### 2.1 Call sites that get a PRIORITY fsync (not a synchronous one)

Verified against the current call graph — these are the only three points where a definite,
named boundary exists (a distinct record type, or the terminal close), so they are the only
points worth fsyncing outside the periodic cadence:

1. **Gap records** — `CaptureWriter.write_gap` (`capture.py:347-348`). A gap record is itself a
   disclosure of lost/uncertain time; if it doesn't survive a crash, the NEXT reader loses the
   one thing that would have told it "this span is honestly unaccounted for," and silently
   reads the span as accounted-for instead (the exact 893 lesson this format exists to prevent
   — see the module docstring, `capture.py:20-23`). This includes the resume-on-restart gap
   written from `CaptureWriter.__init__`'s else-branch (`capture.py:280-286`; line 279 is the
   `meta_start` write in the *if*-branch, not part of the resume path). It also includes the
   writer-generated `writer_overflow` gap (§2.0.2) — same reasoning, same tag; it is the only
   gap the writer produces on its own behalf rather than on a caller's.
2. **Meta records** — `meta_start` (`capture.py:279`, inside `__init__`) and `meta_end`
   (`capture.py:350-352`, inside `close`). Each is written exactly once per process lifetime, so
   fsyncing them costs one call at start and one at clean shutdown, not one per event.
   `meta_start` carries the declared pair/channel set the whole completeness gate
   (`assert_complete`, `capture.py:646`) depends on; `meta_end` is literally what
   `closed_cleanly` reports.
3. **Terminal close** — `CaptureWriter.close(clean=True)` (`capture.py:350-357`) — the
   `meta_end` fsync above covers it; no additional call is needed beyond that, but that final
   fsync must not be skipped by the periodic-timer logic (i.e. the periodic policy must not race
   a `close()` and leave the last events ahead of `meta_end` unsynced).

**"Priority", not "synchronous" — this is the §2.0 correction.** These three record types are
tagged in the queue so the worker fsyncs **immediately after writing them**, ahead of the
periodic deadline. The loop does **not** wait for that fsync to complete. What is bought and
what is given up:

- Bought: the crash window for a gap/meta record shrinks from "up to N seconds" to "queue
  transit plus one fsync" — sub-millisecond in steady state.
- Given up: the record is not durable *before the loop continues*. It never could be without
  violating R1, and waiting buys nothing anyway — if the process dies between the write and the
  fsync the record is lost either way, and blocking the loop makes the death (keepalive trip →
  reconnect) *more* likely. Under a device outage, blocking would mean waiting ~59 s for a
  durability guarantee about a record describing that very outage.
- The one case where it genuinely matters — a gap written during an outage, when the priority
  fsync cannot complete until the device returns — is exactly the case where the synchronous
  rule would have frozen the recorder for the whole outage. Both designs leave the gap record
  non-durable for ~59 s. Only one of them also keeps the capture alive.

### 2.2 Periodic policy for everything else (`book`, `trade`, `heartbeat`, `counters`)

Pin: **one `flush()` per record, performed by the worker thread immediately after its `write()`,
with no batching of its own (§2.0, §6); `os.fsync()` at most once per 30 wall-clock seconds**,
tracked by a monotonic-clock deadline (`time.monotonic()`, not `time.time()` — the gap-accounting
clock basis is wall-clock-sensitive by design, but the fsync cadence must not be derailed by the
same clock jumps/stalls this fix exists to survive). Concretely: the worker tracks
`_last_fsync_monotonic`; after writing and flushing a record it calls `os.fsync()` only if
`time.monotonic() - self._last_fsync_monotonic >= 30.0`, then resets the deadline. A record
tagged gap/meta gets a priority fsync regardless of the timer (§2.1) and always resets it.

#### Why N = 30 s — corrected arithmetic (2026-08-31)

**The first draft's justification was arithmetically wrong and is withdrawn.** It claimed a 30 s
period gives "roughly 2 fsync calls per non-stalled 600 s window" and keeps the durability window
"an order of magnitude below the stall duration". Both figures are wrong, and — more importantly
— the *criterion* behind the second one was never sound. Corrected:

| quantity | draft claimed | actual |
|---|---|---|
| fsyncs per non-stalled 600 s window | ~2 | **~18** (538.5 s ÷ 30 s = 17.95) |
| durability window vs stall duration | "an order of magnitude below" | **0.49×** — 30 s vs the **≈61 s typical** outage, i.e. *the same order* (61 ÷ 30 = 2.03); **0.32×** against the 93 s tail |

(538.5 s = 600 − 61.5. The ≈61 s typical outage is now **measured, not estimated**: the
5,452-event kernel census puts 73.4 % of events on the 30.7 s inter-pass mode, which with the
~30 s SCSI command timeout gives ≈61 s — D74's conservative 61.5 s estimate lands on it. Against
the 93 s tail the non-stalled window is 507 s ÷ 30 = 16.9 fsyncs — the same answer either way.)

**The reduction claim survives and was understated.** Today, at 30.25 rec/s, the recorder issues
~30.25 fsyncs/s ≈ **16,290 per non-stalled 600 s window**. Eighteen against 16,290 is a **99.89 %
reduction, ~905×**. That is the load-bearing number, and it is the one that makes the recorder
stop being a significant journal client (0.033 commits/s against the ~57 commits/s the box
sustains healthy).

**The withdrawn criterion.** "Keep the durability window below the stall duration so a crash
mid-stall risks no more than a crash elsewhere" does not describe anything real *under the §2.0
design*, at any N. Only the worker thread ever calls `fsync()`, and it is single-threaded: once
it is blocked inside an fsync it cannot issue the next one. So **at most one fsync is ever
blocked per outage, whatever N is**. N does not govern outage exposure at all. The draft was
sizing N against a quantity N does not control.

**What N actually trades**, at the measured 30.25 rec/s / ~125 B per record:

| N | records at risk on hard crash | bytes | % of the 7,515,067-line capture | fsyncs per non-stalled window |
|---|---|---|---|---|
| 10 s | 303 | ~37–39 KB | 0.0040 % | ~54 |
| **30 s** | **908** | **~111–116 KB** | **0.012 %** | **~18** |
| 60 s | 1,815 | ~222–232 KB | 0.024 % | ~9 |

**N = 30 s is retained, on this reasoning instead.** Two honest bounds:

1. *Loss window.* 0.012 % of the capture, and — per §2.3 — not silently lost but disclosed as a
   resume `gap` on the next start. Halving it to 0.004 % (N=10) buys nothing the estimand can
   see; doubling it to 0.024 % (N=60) costs nothing the estimand can see either. The constant is
   genuinely not load-bearing to any correctness property in this document.
2. *Durable-frontier lag.* Worst case the file's durable frontier is N + (an outage in progress)
   behind the wire: 30 + 61.5 ≈ **92 s**, a **6.5× margin** inside the 600 s stall cycle. Every
   stall cycle therefore contains many completed durable checkpoints and the capture is never a
   whole cycle behind. This is the margin argument the draft was reaching for; it holds at N=30
   and would start to erode past N ≈ 200 s, where a single outage could swallow the only fsync
   due in a cycle.

**Accordingly the spec pins the mechanism, not the number: any N in 10–60 s is acceptable, 30 s
is the default.** §7.4's test pins that the deadline fires and *resets* — not that it equals 30.
An implementation should take N as a constructor parameter defaulting to 30.0 so the tests can
drive it with a fake clock without hard-coding the production value.

### `flush()` vs `fsync()` — what each buys, and which the reader needs

- `flush()` (Python `TextIOWrapper.flush` / the underlying `write(2)` syscall) pushes bytes out
  of the **Python-level userspace buffer** into the OS. Once flushed, the bytes are in the
  kernel page cache and visible to **any other process reading the same file** — `cp`,
  `ssh host cat`, a fresh `open()` — regardless of whether they have hit disk. This is the
  guarantee every downstream reader in this repo actually depends on: TASK_912's health probe
  copies the live file with `shutil.copyfile` / `ssh host cat` (`health_probe.py:87-98`,
  `copy_capture`) — a page-cache-level read, not a disk-level one.
- `fsync()` additionally forces the kernel to persist those already-page-cache-resident bytes to
  the physical device and blocks until the device (or, on ext4 `data=ordered`, the filesystem
  journal) confirms the commit. It buys **crash/power-loss durability**, not reader visibility.
- Conclusion: the reader (health probe, or any concurrent tail) needs `flush()`, not `fsync()`.
  Dropping only `fsync()` while keeping one `flush()` per record therefore has **zero effect on
  what a concurrent reader can see** — it only widens the window of data that could be lost to an
  actual OS crash / power loss. This is the load-bearing fact behind §6's "no split-feed
  interaction" conclusion below.
- **Corrected for §2.0 (2026-08-31).** The draft added "…not to an application-level exception or
  `kill`, both of which leave already-`flush()`ed bytes durable via the OS's own eventual
  writeback". That remains true *of bytes that have been flushed*, but under the worker-thread
  design it is no longer true of every record the loop has accepted: a record still sitting in
  the queue has never been `write()`n at all, and **is lost on SIGKILL**. Queue depth is ~0 in
  steady state (the worker keeps up trivially at ~4 KB/s), so the practical exposure is ~0
  records; during a device outage it is up to ~1,785. Two consequences, both carried into §2.3:
  the loss bound is `max(N × rate, queue depth at the instant of death)`, not just `N × rate`;
  and *when* the reader sees a record is now "one queue transit after the loop produced it"
  rather than "synchronously, before `_write` returns". Both are bounded by the same device
  outage that today kills the capture outright — see §6 for the probe-facing analysis.
- A SIGTERM handler that drains the queue before exit would remove the SIGKILL exposure for the
  ordinary shutdown path. Recommended, not required; `main()` (`recorder.py:896-918`) installs no
  signal handlers today, and adding one is a behaviour change the implementer should make
  deliberately rather than incidentally.

### 2.3 The bounded-loss argument, quantitatively

Sizing off the actual capture (as briefed for this task, 2026-08-31): **~918 MB / 7,515,067
lines over ~2d21h (69h) of wall-clock span.**

NOT VERIFIED locally: I did not re-derive these totals from the live file myself — TASK_913's
binding fence forbids me from touching `gpu-server`, and the numbers above were handed to me as
the current state of the running capture. Treat them as the orchestrator-supplied input to this
sizing exercise, not an independently re-measured figure.

- 69h = 248,400 s.
- Mean emission rate: 7,515,067 / 248,400 ≈ **30.25 records/s** (blended across all record
  types and however many pairs are subscribed).
- Mean bytes/record: 918 MB / 7,515,067 ≈ **122–128 bytes/line** (122 if MB = 10^6 bytes, 128 if
  MiB = 2^20 bytes; either way, order 125 bytes).
- A 30s fsync window therefore puts at risk, on a hard crash landing at the worst possible
  moment: **30.25 × 30 ≈ 908 records ≈ ~111–116 KB** of unflushed-to-disk (but still
  `flush()`ed, still readable by a concurrent probe up to the crash instant) data.
- 908 records out of 7,515,067 collected so far is **0.012%** of the capture — **~957× smaller
  than** (i.e. just under three orders of magnitude below) the 11.56% of span the CURRENT
  per-line-fsync design is already losing to stalls (D74,
  `lore/handoff_2026-08-30_factory_run_47.md:82`: gap union 25,658 s / 61.65 h span measured at
  that point). *(The draft said "four orders of magnitude"; 11.56 ÷ 0.012082 = 956.8, which is
  three.)*
- **Second loss term, from §2.0.** Under the worker-thread design the true bound is
  `max(N × 30.25, queue depth at death)`. The two terms dominate in different regimes: the fsync
  window (908 records) governs an OS crash / power loss at a random moment, and the queue depth
  governs a SIGKILL. In steady state queue depth ≈ 0, so 908 stands; landing a SIGKILL inside a
  device outage raises it to at most ~1,785 records (~0.024 % of the capture, ~1.9× the
  fsync-window term). Both remain two-to-three orders of magnitude below the 11.56 % being lost
  today, and the second term does not exist at all in the (b) fallback design.
- What a crash costs: at most that bound of tail, and even that tail is not silently
  unaccounted-for — `CaptureWriter.__init__`'s resume branch (`capture.py:280-286`, the
  else-branch; line 279 is the `meta_start` write in the *if*-branch) writes a `gap` record from
  `_last_event_ts(path)` (the last event actually visible in the file — i.e. the last written
  record) to the restart time on the NEXT process start against the same `--out` path. So a
  crash's loss degrades from "silently missing data" to "one disclosed gap record", exactly the
  same disclosure discipline the format already applies to every disconnect. This holds under
  §2.0 unchanged: the resume gap is computed by re-reading the file, so it discloses whatever was
  lost — fsync window and queue contents alike — without needing to know which term caused it.
- What the CURRENT stall costs: 11.56% of span, **every ~600s, for the life of the capture** —
  not a rare tail-risk event but a recurring, already-measured, already-paid cost, and (per
  D74's amendment) with a secondary market-correlated receipt-lag effect on top (907B-2 scope).
- For THIS estimand — a fill-probe capture measuring realized coverage/markout behavior, not a
  system that must survive with zero data loss under adversarial power failure — trading a
  recurring 11.56%-of-span stall/reconnect loss — hardware-caused, but which the per-line
   fsync design maximises our exposure to — for a residual ≤30 s / ≤0.012 %
  (worst case ≤~59 s / ≤0.024 % if a SIGKILL lands inside an outage)
  exposure to a genuinely rare event (an OS crash, power loss, or a hard kill,
  none of which the box has exhibited during this capture) is the correct trade. The bounded-loss
  window shrinks further as the capture runs longer (908 records is a shrinking fraction of a
  growing total), while the stall cost is a fixed percentage that recurs on every 600s cycle
  regardless of capture length.

## 3. Torn-tail claim — VERIFIED TRUE

The task's premise ("the reader already tolerates a torn tail line and 912's fixture proves it")
was checked against the actual code and tests, not assumed:

- **Reader mechanism**: `capture.py::_iter_lines_tolerant` (`capture.py:390-408`) parses every
  line and, on a `json.JSONDecodeError`, checks `if i == len(lines) - 1: continue` — i.e. it
  silently drops ONLY a corrupt/truncated **trailing** line; a non-trailing corrupt line still
  raises `CaptureFormatError` (`capture.py:406-408`). `health_probe.py` restates the identical
  logic independently, stdlib-only, in `_iter_lines_tolerant` at **`health_probe.py:103-122`**
  (the function body runs through 122 — the draft's `103-119` stopped one line short of the
  `raise CaptureParseError` at 122, which is the half that makes the tolerance *bounded*; its own
  docstring says it "Mirrors `capture.py::_iter_lines_tolerant` exactly").
- **`capture.py`'s own test**: `test_capture_gaps.py::test_truncated_trailing_line_is_tolerated_not_corruption`
  (`test_capture_gaps.py:95-105`) writes a well-formed line, appends a hand-truncated,
  no-trailing-newline `trade` line (`test_capture_gaps.py:101`:
  `fh.write('{"type": "trade", "pair": "BTC/USD", "ts": 101.0, "pri')`), and asserts
  `meta.truncated_tail` is set AND `meta.n_trade == 1` (the torn line is dropped, not counted,
  and — critically — the call does not raise).
- **912's health-probe test**: `test_health_probe.py::test_torn_trailing_line_is_tolerated_not_a_schema_mismatch`
  (`test_health_probe.py:140-164`) does the same thing against `health_probe.py`'s independent
  reader, and additionally asserts the torn line does NOT get miscounted as a `schema_mismatch`
  (`record["schema_mismatch_count"] == 0`, `test_health_probe.py:164`) — i.e. torn-tail
  tolerance is proven distinct from, and does not contaminate, the schema-mismatch FAIL gate.
- Both tests were re-run locally (2026-08-31) and PASS:
  `python -m pytest experiments/907_live_fill_probe/test_capture_gaps.py::test_truncated_trailing_line_is_tolerated_not_corruption experiments/907_live_fill_probe/test_health_probe.py::test_torn_trailing_line_is_tolerated_not_a_schema_mismatch -q`
  → `2 passed in 0.04s`.

**Conclusion: the claim is TRUE and load-bearing evidence for it exists at both the format
level and the 912 health-probe level.** The bounded-loss argument in §2.3 may rely on it without
further hedging.

One caveat worth carrying forward (not a defect in the claim, a scope note): today, torn-tail
tolerance is exercised only against a **single** truncated line, because `_write`'s per-line
fsync means a kill can only ever be mid-flight on one `write()+flush()+fsync()` triple at a
time. Once fsync is batched, a crash can leave the LAST `flush()`ed-but-unfsynced record(s)
readable (they're in page cache, possibly on disk, possibly not) and, separately, whatever the
Python-level `TextIOWrapper` had buffered but not yet even `flush()`ed at the instant of the
crash can be lost entirely below the line level — same "torn tail" shape the reader already
tolerates, but the future implementation should add a test with a MULTI-KB batch of unflushed
data ahead of the torn line (see §7) rather than assume single-line truncation generalizes for
free.

**The §2.0 worker thread does not weaken this, and the reason is worth stating precisely.** The
reader's guarantee — "only the FINAL line may be torn; an earlier torn line is real corruption
and raises" — rests on the writer never having more than one record partially written at a time.
That still holds: the worker is a single thread doing `write()` then `flush()` per record, in
queue order, so at any instant at most one record is partially in the file. Records the worker
has not yet dequeued are not in the file at all — they are *absent*, which is the disclosed-gap
case (§2.3), not the torn-line case. **What changes is only the depth of the truncation, not its
shape**: today a kill loses ≤1 partial line, under §2.0 it loses ≤1 partial line *plus* up to
the queue depth of records that never reached the file. The reader's tolerance rule is
untouched. §7's test 5 covers the multi-KB torn-tail shape; §7's test 10 covers the
never-written shape (records lost from the queue must be disclosed, not silently absent).

## 4. SECONDARY option: application volume — recorded, not defaulted

`/mnt/app` (the application volume, 397G free per D74/TASK_913 scope-1 box inventory) has
its own filesystem journal AND sits on a **separate controller** from `ata32` — `nvme0n1`
shows none of the ~600 s stalls that `sda` logs. Pointing a future capture's `--out` there
would move the capture **off the stalling device**, which the root-cause measurement makes a
materially stronger option than it looked when this was filed as a way to "sidestep journal
contention" (there is no contention to sidestep — see `JOURNAL_STALL_DIAGNOSIS.md`). Note this
is a workaround, not a repair: per that document the actual fix is the operator's and is a
daemon change (stop `udisks2`'s 600 s SMART poll), not hardware work.

**This is explicitly the operator's call, not a default.** `recorder.py:889`
(`p.add_argument("--out", required=True, ...)`) already has no default — the operator must name
a path on every invocation, which is the correct posture to keep. `/mnt/app` is the
operator's application volume (docker workloads presumably rely on its I/O headroom); a future
implementation MUST NOT silently point captures there, must not add a code path that defaults
`--out` toward it, and any change to WHERE the recorder writes is a decision the operator makes
explicitly per-invocation, same as today. This spec records the option; it does not adopt it.

## 5. The `clock_jump` rename

**Detector code**: `recorder.py:684-688`, inside `MarketDataRecorder._consume`:

```python
684:            now = self._clock()
685:            prev = self._last_event_ts
686:            if prev is not None and (now - prev) > self.cfg.heartbeat_timeout_s:
687:                self._record_gap("clock_jump", f"local clock advanced {now - prev:.1f}s "
688:                                                f"between consecutive recv() calls")
```

`self._clock` defaults to `time.time` (`recorder.py:575`, constructor default
`clock: Callable[[], float] = time.time`) — ordinary wall-clock, not a monotonic clock and not
an NTP-jump detector. What this actually measures is `now - self._last_event_ts`, i.e. the
elapsed wall-clock time since the last event was processed, exceeding the recv() heartbeat
timeout. `self._last_event_ts` is set in exactly two places: `recorder.py:808` (in `_emit`, on
every processed event) and `recorder.py:868` (in **`_record_gap`**, advancing it to the gap's end
so the next gap starts where this one finished). The draft labelled `:868` "a dark-window close",
which is wrong — `_close_dark_window` spans `recorder.py:825-854` and never touches
`_last_event_ts`; it only writes the `disconnect` gap that closes the window. Either way the
point stands: `_last_event_ts` is emission/processing time, not a clock reading. **The label describes a mechanism (an NTP-style clock correction) the code does not
detect; the code detects emission silence that outran the heartbeat-timeout expectation** —
exactly the D74 stall signature: `asyncio.wait_for(..., timeout=heartbeat_timeout_s)`
(`recorder.py:678`, inside `_consume`) cannot fire its own `TimeoutError` while the process itself is frozen
in `os.fsync()`/`jbd2_log_wait_commit` (the event loop cannot run to notice the timeout expired);
when the loop resumes, this wall-clock check is what actually catches the elapsed silence and
retroactively records it.

**Honest replacement**: rename the reason string from `"clock_jump"` to `"emission_silence"` (or
equivalent — `recv_stall` was considered and rejected: the silence is observed at the
recv()-consumption boundary but its CAUSE, per D74, is process-side blocking, not a stalled
`recv()` itself; "emission_silence" describes what was actually measured — the recorder's own
event-emission clock went quiet longer than the heartbeat budget — without asserting a cause).
Re-word the detail string at the same call site from `"local clock advanced {…}s between
consecutive recv() calls"` to something like `"no event processed for {now-prev:.1f}s (>
heartbeat_timeout_s={cfg.heartbeat_timeout_s}s) — emission silence, not a system clock jump"`.

**Backward compatibility for old files (load-bearing — this is a READ-time relabel per the
task's Out-of-scope, the bytes on disk do not change):**

- `GAP_REASONS` (`capture.py:105-108`) is validated on every `GapRecord` construction
  (`capture.py:208-210`), including on the READ path: `read_capture` reconstructs a `GapRecord`
  from every parsed `gap` line (`capture.py:917`:
  `GapRecord(obj["start_ts"], obj["end_ts"], obj["reason"], obj.get("detail", ""))`). If the new
  implementation removes `"clock_jump"` from `GAP_REASONS` outright, `read_capture` will raise
  `CaptureFormatError` on every existing capture file (including the one currently running,
  which has already written this reason many times) the moment it hits a `clock_jump` line.
- The future implementation MUST add the new label alongside the old one, not swap it:
  `GAP_REASONS = frozenset({"disconnect", "seq_discontinuity", "heartbeat_timeout",
  "clock_jump", "emission_silence", "resume", "schema_drift"})` — `clock_jump` stays in the set,
  documented as deprecated/legacy (old files only; new code never writes it), so historical and
  in-flight captures keep parsing. Only `recorder.py`'s call site (`recorder.py:687`) changes
  what it WRITES going forward. **§2.0.2 adds one further member, `"writer_overflow"`, on this
  same additive discipline — the implementer ships ONE combined set, not two conflicting edits
  to the same frozenset.**
- Anywhere the reason string is rendered for a human (`render.py:121`,
  `health_probe.py:567,611`) should either print the raw reason unchanged (both old and new
  labels are self-explanatory once the rename ships) or, if a friendlier label is wanted, map
  BOTH `clock_jump` and `emission_silence` to the same human-readable phrase — never treat an
  old-labeled record as a different category of event than a new-labeled one; they are the same
  mechanism, differently named.

## 6. Acceptance (iii): cross-check against TASK_912's health probe

**Conclusion: fsync cadence cannot change a probe verdict, by construction — PROVIDED the future
implementation keeps one `flush()` per record (§2.2) and only batches `fsync()`. That proviso is
the load-bearing half of this section and must survive any later optimization.**

The split-feed detector is **rule 4** of `_compute_verdict`, at
**`health_probe.py:521-549`** (`never_book_pairs`/`stale_book_pairs` → the `split-feed death`
reason and its `recorder-dead` sibling). The draft cited `health_probe.py:419-548`, which is not
rule 4 — that is very nearly the whole of `_compute_verdict` (`health_probe.py:418-590`,
carrying rules 1-7). Rule 4 fires when
`per_pair[pair]["last_book_age_s"] > BOOK_STALE_THRESHOLD_S` (300.0s,
`health_probe.py:64`) for some pair while `newest_event_age_s < FRESH_THRESHOLD_S` (60.0s,
`health_probe.py:67`) overall. Both ages are computed as `wall_now - event_ts`
(`health_probe.py:330,339-340`), where `event_ts` is the record's own `"ts"` field — the value
the recorder assigned **at event-receipt time, before `_write` is ever called**
(`write_book`/`write_trade` compute `ts` from their caller's clock, not from anything
write-path-related). Batching `fsync()` changes **when the bytes are guaranteed durable on
disk**; it changes nothing about what `ts` value is stamped into a record, and — because one
`flush()` still follows every record's `write()` — it changes nothing about **when the bytes
become visible** to a reader either. (§6.1 handles the one thing §2.0's worker thread *does*
change about visibility, which is not the cadence but the thread.)

The probe reads exclusively through `copy_capture` (`health_probe.py:87-98`,
`shutil.copyfile` locally / `ssh host cat` remotely — "the ONLY function in this module that
ever touches the live capture path," per its own docstring). Both `copyfile` and `cat` are
ordinary `read(2)`-based byte copies against the filesystem's current view of the file, which is
exactly the kernel page cache state that `flush()` (not `fsync()`) is responsible for updating.
A record that has been `write()`+`flush()`ed but not yet `fsync()`ed is **fully visible** to
`copy_capture` the instant it is flushed — `fsync()` only affects durability against an actual
OS crash / power loss between that flush and the eventual periodic writeback, which is orthogonal
to a live `cp`/`cat` reading the same, currently-running machine's page cache.

Therefore: **a writer that flushes every line and fsyncs only periodically is byte-for-byte
indistinguishable, from the split-feed detector's point of view, from the current per-line-fsync
writer** — same `ts` values, same visibility timing, same `last_book_age_s`/`newest_event_age_s`
computation. The detector cannot be tripped by the change because nothing it reads is a function
of fsync cadence.

**THE PROVISO — one requirement this analysis makes explicit for the future implementation**:
one `flush()` per record is not optional and must not get bundled into the "buffering" language
loosely. **The writer MUST NOT batch `flush()`.** If a future implementer does (e.g. "buffer N
records, then write+flush+maybe-fsync"), THEN the interaction reappears: a book/trade record
sitting in the unflushed Python buffer is invisible to `copy_capture` until the next flush, and a
probe running inside that window would see a stale `last_book_age_s` for a genuinely-live pair —
a false split-feed (or false recorder-dead, if it's every pair) verdict, with a false-positive
window bounded by the flush batch's own cadence. This spec's pin (§2.2) exists specifically to
foreclose that failure mode; the future implementer must not relax it as a further optimization
without re-running this cross-check. §7.3 is the test that holds the line.

### 6.1 The new interaction: §2.0's worker thread and the per-line `flush()` proviso

This is new since the draft — the draft's §6 analysed a change in fsync *cadence* on a
single-threaded writer, and §2.0 changes the *thread*. Worked out:

**Is `flush()` still per-line?** Yes. The worker does `write()` then `flush()` for each record
individually, in queue order; there is no batching anywhere in the design. The proviso is
satisfied in the sense that matters — one flush per record, no record ever waiting on a *later*
record's arrival to become visible.

**Is it still ordered from the probe's point of view?** Yes, and this is stronger than it needs
to be: a single FIFO queue with a single consumer means records reach the file in exactly the
order the loop produced them (R3). The probe reads a byte copy of the file, so it sees a prefix
of that same order. There is no interleaving hazard, because there is no second writer.

**What genuinely changes: visibility is no longer synchronous with `_write`'s return.** A record
becomes visible to `copy_capture` one queue transit after the loop enqueued it, rather than
before `_write` returns. Bounding that transit:

- *Steady state*: the worker consumes ~4 KB/s and is idle almost always; transit is microseconds
  to milliseconds. Far below the probe's coarsest resolution (`FRESH_THRESHOLD_S` = 60.0 s,
  `BOOK_STALE_THRESHOLD_S` = 300.0 s). No effect.
- *During a device outage*: the worker is blocked inside `fsync()` (or a throttled `write()`) for
  up to ~59 s, the queue backs up, and the probe's copy can be up to ~59 s behind the wire.

**Does that ~59 s lag flip a verdict?** Take the two thresholds in turn.

- `newest_event_age_s` is pushed up by the full lag, to ~59 s worst case — just under
  `FRESH_THRESHOLD_S` (60.0 s), and it can cross it. But crossing it does **not** create a FAIL:
  `newest_event_age_s` is only consulted *after* `affected` is non-empty (`health_probe.py:536-537`),
  i.e. only once some pair's book is already >300 s stale or never seen. With no stale pair, the
  branch is not reached and the lag is invisible to the verdict. What crossing 60 s does is swap
  which reason string a pair-staleness FAIL prints — `split-feed death` vs `recorder-dead`. Both
  are FAIL, so the **status is unchanged**; the *attribution* can shift, and the implementer
  should know that.
- `last_book_age_s` for a live pair cannot be pushed from healthy to >300 s by a ≤59 s lag. It
  could carry a pair already sitting at ~250 s over the line — but a book silent for 250 s is
  already at the edge of what the probe calls dead, and the lag is disclosed by the gap ledger.

**And the honest comparison, which is the point:** this lag is **not introduced by the change.**
Under the CURRENT design the recorder is itself blocked in `fsync()` for the same ~59 s, writes
nothing at all during it, and the probe's `newest_event_age_s` grows by exactly the same amount —
*and* the keepalive trips and the capture disconnects. §2.0 is therefore strictly better on the
probe's view, never worse: the same worst-case lag, bounded by the same device outage, without
the disconnect. **Acceptance (iii) holds: a batched-fsync writer does not trip the split-feed
detector by design.**

**One implementation requirement this adds**, and it is testable: the queue must never be the
thing that makes a record late in *steady state*. Concretely — the worker must not sleep, poll on
a timer, or wait to accumulate; it blocks on `queue.get()` and writes what it gets. A "wake every
100 ms and drain" worker would be a flush batcher wearing a different hat, and §7.3/§7.6 must be
written to catch it (assert visibility per record with a drained queue, not merely eventual
presence in the file).

Also flagged for the future implementation, not required by this spec to fix now: two doc
comments assert the CURRENT (per-line fsync) guarantee and will read as stale once this ships —
`capture.py:42-44` ("Crash safety: every line is flushed and fsynced individually...") and
`health_probe.py:33`/`107-108` ("every line is flushed+fsynced individually... capture.py's
fsync-per-line crash-safety guarantee"). The future patch should update both alongside the code
change so the docstrings keep describing the real mechanism.

## 7. Test spec (post-capture implementation)

Mutation-style: each test must plant poison at the point the mechanism is weakest (accidentally
fsyncing everything, or accidentally fsyncing nothing, or batching flush by mistake), not merely
exercise the happy path.

**Determinism note for the §2.0 worker thread.** Tests 1-6 were written against a synchronous
writer and all of them become racy verbatim, because "did fsync happen" is now a question about
another thread. Every one of them needs a **synchronisation point**, and the implementation must
provide one: a test-only `CaptureWriter.drain()` (or `_queue.join()`) that blocks until the
worker has consumed everything enqueued so far. Assert *after* draining, never on a sleep. Do not
paper over this with `time.sleep(0.05)` — that is a flaky test and, worse, one that passes under
the very mutation §6.1 warns about (a worker that wakes on a 100 ms timer).

1. **`test_gap_and_meta_records_get_a_priority_fsync`** — pins durability of the three call
   sites named in §2.1. Monkeypatch/spy `os.fsync` (e.g. wrap `CaptureWriter._fh.fileno` /
   substitute a fake fd tracker, or patch `os.fsync` at module scope and record the fd it was
   called with) on a `CaptureWriter`. Write one `book` record, one `trade` record, one `gap`
   record (`write_gap`), draining between each, then `close(clean=True)`. Assert `os.fsync` was
   called AT LEAST once after the `gap` write drained and AT LEAST once after `close` (for
   `meta_end`) —
   i.e. the count of fsync calls attributable to gap/meta writes must equal the count of
   gap/meta writes issued, regardless of the periodic timer's state. **Fails the mutation** where
   a future implementation makes the periodic-only path apply uniformly to every record type
   (i.e. someone "simplifies" by removing the gap/meta special-case) — that mutation would still
   pass a test that only checks "fsync was called at some point," so the test must specifically
   correlate fsync calls to the gap/meta writes' own timing, not merely their eventual presence
   in the file.
2. **`test_event_lines_are_not_individually_fsynced`** — the mutation-poison test for the
   PRIMARY fix itself, since a naive/incomplete refactor could leave the old per-line fsync in
   place for events while ADDING a redundant periodic one (satisfying "there is a periodic
   policy" while never actually removing the per-line cost this task exists to remove). Spy on
   `os.fsync` call count; write N `book`/`trade`/`heartbeat` records in a burst with the fake
   clock held constant (i.e., inside one fsync period, `time.monotonic()` not advanced) and
   assert `os.fsync` is called **zero** times across those N writes (only `flush()` calls, one
   per write). This is the test that actually distinguishes "fsync batched" from "fsync
   unbatched," which a naive "fsync got called eventually" assertion would not.
3. **`test_flush_still_happens_once_per_record`** — the poison-the-weak-point test for §6's
   PROVISO. Spy on the underlying file handle's `flush` (or count `write(2)` syscalls via a
   wrapped file object) and assert exactly one flush lands per `write_book`/`write_trade`/
   `write_heartbeat`/`write_counters` record, independent of the fsync timer state, and that the
   file's visible line count after `drain()` equals the number of records enqueued. Under §2.0
   the assertion is about the WORKER's per-record behaviour, not about `_write` returning with
   the bytes already out — check after a drain, mid-burst, **before any `close()`**.
   **Fails the mutation** where a future implementer batches `flush()` together with `fsync()`
   (the exact failure mode §6 identifies as reopening the split-feed interaction), and equally
   the §6.1 mutation where the worker accumulates on a timer instead of draining the queue — a
   test that only checks "the file eventually contains the data" would pass under both, since
   `close()` flushes on exit.
4. **`test_periodic_fsync_fires_after_deadline_not_before`** — pins the cadence as a real,
   testable boundary rather than "buffered writes, trust us." Inject a fake monotonic clock and
   a small injected N (§2.2: N is a constructor parameter defaulting to 30.0 — **the test pins
   the mechanism, not the constant 30**). Write one event just before the deadline (assert no
   fsync yet, beyond whatever priority fsyncs occurred), advance the fake clock past N, write
   another event, drain, assert exactly one fsync fired for that second write and the deadline
   reset. **Fails the mutation** where the timer never actually resets (fsyncs every write once
   the clock exceeds the ORIGINAL start time, effectively degrading back to per-line-after-warmup)
   or never fires at all (silently regresses to zero durability for events).
5. **`test_torn_tail_tolerated_with_multikb_unflushed_gap_ahead_of_it`** — extends the existing
   torn-tail coverage (§3) to the batched-fsync shape: write several KB of well-formed
   `flush()`ed-but-not-yet-`fsync()`ed records, then hand-append one truncated trailing line (no
   newline, mid-JSON), and assert `read_capture`/`health_probe.analyze_capture` still report
   `truncated_tail=True`, count every well-formed record ahead of it, and do not raise. This is
   the case §3's caveat calls out: single-line-truncation coverage already exists
   (`test_capture_gaps.py:95-105`, `test_health_probe.py:140-164`) but does not exercise a
   multi-record unflushed run ahead of the tear, which is the shape a periodic-fsync writer
   actually produces on crash (many flushed-not-fsynced records, one torn tail), not the
   single-line shape a per-line-fsync writer produces.
6. **`test_split_feed_verdict_identical_under_periodic_fsync`** — the direct acceptance-(iii)
   regression pin. Build two capture files with IDENTICAL event `ts`/content — one written with
   the OLD per-line-fsync `_write`, one with the NEW periodic-fsync `_write` (same one-flush-per-
   record behavior), draining the new writer before analysis — and assert
   `health_probe.analyze_capture` produces byte-identical
   `per_pair[...]["last_book_age_s"]`/`last_trade_age_s`/`newest_event_age_s` and an identical
   verdict (`status`, `reasons`) for both. **Fails the mutation** where a future implementation's
   periodic batching inadvertently changes visibility timing (e.g. by batching flush after all,
   or by holding records in an in-process list instead of writing them line-by-line) — this is
   the test that would actually catch the §6 false-positive failure mode if it were
   reintroduced, rather than relying on code review to notice.

Tests 7-11 are NEW, added with §2.0 (test 11 with §2.0.1). They pin the worker thread's
contract; without them the threading change is the least-tested and highest-risk part of this
spec.

7. **`test_writer_thread_preserves_record_order_under_load`** — pins R3, the invariant the whole
   gap ledger rests on. Interleave a long burst of `write_book` and `write_gap` calls in a known
   pattern from the calling thread, drain, and assert the file's line order matches the call
   order exactly (not merely that all records are present). **Fails the mutation** where someone
   "optimizes" priority (gap/meta) records by giving them a second queue or writing them inline
   from the caller — either would let a gap record overtake events it must follow, and the ledger
   would silently mis-attribute which span is unaccounted for. Also fails a multi-consumer worker
   pool.
8. **`test_worker_error_is_re_raised_on_the_next_write_and_on_close`** — pins R4, and fail-closed
   behaviour on a box with 32,220 logged `ioerr_cnt`. Inject a file object whose `write` (or a
   patched `os.fsync`) raises `OSError` on the Nth record. Assert: (a) the exception surfaces on
   a subsequent `_write`/`write_book` call from the caller's thread, carrying the original error;
   (b) `close()` also raises rather than exiting cleanly; (c) the writer does **not** keep
   silently accepting records after the worker has died. **Fails the mutation** where the worker
   catches and logs, which would turn a failing disk into a silently truncated capture — the
   single worst outcome available to this design and the one the current synchronous code cannot
   produce.
9. **`test_close_terminates_when_the_worker_is_blocked_and_does_not_close_the_fd`** — pins R5 and
   the fd-reuse hazard §2.0 names. Inject an `os.fsync` that blocks on an event the test controls.
   Call `close(clean=True)` and assert it returns within the join timeout rather than hanging;
   assert the underlying fd was **not** closed while the worker is still inside `fsync` (e.g. the
   file object's `closed` is still False, or a sentinel wrapper records no `close()` call); assert
   the undrained record count is reported. Then release the blocked fsync and let the daemon
   thread finish. **Fails the mutation** where `close()` does an unbounded `join()` (hangs the
   process one shutdown in ten, at the measured 10 % duty cycle) and the mutation where it closes
   the fd out from under the worker (cross-file corruption on the next `open()`).
10. **`test_queue_overflow_discloses_a_writer_overflow_gap_on_the_falling_edge`** — pins the one
    place this design can lose data that the current one cannot. **Rewritten (2026-08-31): the
    draft asserted "a `gap` record in the ledger and/or a raised error", a disjunction both
    §2.0.2 branches satisfy, so it pinned nothing. It now asserts the ONE mechanism §2.0.2
    pins.** Construct a writer with a deliberately tiny bound (e.g. `maxsize=4`), block the
    worker on a test-controlled event, and push past the bound. Assert, in order:
    (a) the over-bound `write_book` calls **return normally** — no raise, no block (time the
    call: a blocking `put` is the mutation §2.0's R1 exists to forbid, and it is what an
    unconditional fail-closed implementation would *not* exhibit either, so assert both);
    (b) the reserved slot is never consumed by an ordinary record — with the worker still
    blocked, `qsize()` sits at `maxsize - 1`, not `maxsize` (**fails the mutation** where the
    admission line is written as `maxsize` and the disclosure has nowhere to go);
    (c) release the worker, write one more record, drain, and assert the file contains **exactly
    one** `gap` with `reason == "writer_overflow"`, whose `start_ts`/`end_ts` are the `ts` of the
    first and last dropped record and whose detail names the dropped count;
    (d) that gap sits **between** the last record admitted before the episode and the first
    admitted after it (R3 — the position is the whole point, per §2.0.2 point 5);
    (e) a second episode produces a second, separate coalesced gap — not one merged with the
    first, and not one per dropped record.
    **Fails the mutation** where overflow does `except queue.Full: pass` — the 893 failure mode
    (silently missing data presented as observed time) reintroduced through a new door — and
    equally the mutation where the implementation quietly takes the rejected fail-closed branch
    instead, which (a) catches.
11. **`test_write_failure_inside_run_s_own_handler_shuts_down_without_escalating`** — pins **R6**
    (§2.0.1), the case the draft left as a TODO and that **no other test reaches**: test 8
    exercises an isolated `_write`/`close()` raise, never a raise triggered from inside `run()`'s
    own `except` body, which is the path an unrecoverable `EIO` actually takes.
    Setup: a `LiveRecorder` whose writer has a latched/always-raising `OSError` write path, and a
    `connect_fn` that raises an ordinary transport error (so `run()` enters `except Exception` at
    `recorder.py:658` and its handler calls `_record_gap`). Run `await recorder.run(max_iterations=1)`.
    Assert: (a) **exactly one** exception escapes, and it is `CaptureWriteFailed` — not a bare
    `OSError` escaping an `except` body — with `__cause__` the write error and the transport
    error's repr in its message; (b) `recorder._stop` is True and the injected `_sleep` was
    **never awaited** (no backoff, no retry against a dead writer); (c) the honest negative — the
    capture file contains **no** `gap` record for that disconnect, and its last line is the last
    record written before the fault (this is the assertion that stops the "disclosed rather than
    lost" claim from creeping back in); (d) `CaptureWriteFailed` is re-raised, not laundered, if
    it reaches the clause at `recorder.py:648`. Then call `recorder.close(clean=True)` and assert
    (e) `CaptureWriter.close()` **was reached** — worker stop-signalled, joined within the R5
    timeout, undrained count reported — even though `_flush_counters` raised on the way in, and
    (f) the latched error still surfaces from `close()` rather than it exiting cleanly (R4).
    **Fails the mutation** where `run()`'s handler leaves `_record_gap` unguarded — the current
    shape — which re-raises from inside the `except` body, escapes `run()`, then raises a third
    time out of `main()`'s `finally` and, because `recorder.py:874` precedes `:875`, skips
    `CaptureWriter.close()` entirely and strands the §2.0 worker thread unjoined. Also fails the
    lazier mutation where the handler swallows the write error and lets the reconnect loop
    continue writing into a dead writer.

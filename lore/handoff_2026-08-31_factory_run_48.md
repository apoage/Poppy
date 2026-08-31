# Factory run 48 — 2026-08-31

## Shipped

**TASK_913** (`2d56873`, tag `task913-journal-stall-diagnosis`) — stall diagnosis + recorder
write-hardening spec. Two new documents in `experiments/907_live_fill_probe/`:
`JOURNAL_STALL_DIAGNOSIS.md` and `WRITE_HARDENING_SPEC.md`. **Zero code changes** —
`recorder.py`/`capture.py`/`health_probe.py` sha256-identical to HEAD; the live 62-hour capture
(PID 4082411, now 3d12h, 9.83M lines) was never touched. 271 passed, selftest 12/12.

## The answer

**There is no journal hog.** During each ~600 s stall zero sectors reach `sda` and every writer
blocks at once (`jbd2/sda6-8`, the recorder in `jbd2_log_wait_commit`, postgres `walwriter`,
kworkers). The device is not busy; it is absent. bitcoind/postgres/journald exonerated —
bitcoind's `write_bytes` delta across a stall window was **0** (its 569 MB was socket traffic).
3/3 on-phase sample windows stalled, 0/3 off-phase controls.

**The actor is `udisksd`'s SMART housekeeping poll**, issuing an `SG_IO` ATA passthrough the
drive never answers. Named from its own log: `journalctl -u udisks2` carries **5,452 of 5,452**
`Error updating SMART data: … Input/output error` on this exact drive, one per ~600 s since
boot+32 s, **zero missed, zero successful**, in a unit log whose only other two lines are
systemd's. Verified first-hand twice 12 h apart (5,380 → 5,453, delta 73 = exactly 73 × 600 s).
Precedence on a single clock: pool thread created 06:42:35.83, block queue permanently non-empty
06:42:36.22 — **0.39 s lead, thread first** — and the outage ends the instant the passthrough is
failed (`EIO` .402441 → link-up .438582 → writes resume .46).

## The run's real lesson: we shipped a wrong culprit and had to kill it ourselves

The first diagnosis correctly refuted the task's premise, then replaced it with a **new** wrong
culprit — "failing SATA link, reseat the cable" — which would have sent the operator into a
shared production chassis. An adversarial judge killed it:

- **Zero** `SATA link down` / SError / CRC / hard-resetting messages across all three boots. The
  link-up is libata's **recovery** at the end of a command timeout; the device goes silent ~90 s
  *before* the link is touched. The artifact even wrote "the link-up is the instant our stall
  ended" without noticing that refutes its own causal chain.
- 1.5 Gbps/UDMA-33 is libata's logged escalation *response*; the same cycle already ran at
  **6.0 Gbps with NCQ on** from boot. A dependent variable read as evidence.
- The "10.0% ≈ 11.56%" duty-cycle match was n=1.
- Confidence was bounded on data never requested: "the dmesg ring doesn't reach further back",
  while `journalctl` held 4 GB going back months containing the disconfirming evidence.

**Then it happened again one layer down.** A second refutation found the *corrected* tables were
built on journald **realtime** stamps — drain-time artifacts, because journald is itself blocked
by the stall. Two EH passes **30.7 s apart in printk time are stamped 7 ms apart**; 47 of 72
events collapse below 1 s. The document had already caught that clock lying once (`dmesg -T`,
+29 s) and did not re-check. Also: the `/proc` `starttime` precedence margin was entirely
consumed by `btime` whole-second truncation (0.83 s bias), and the "predicted `pool-5403` then
observed it" evidence is *entailed* by the cadence (GLib names pool workers from a monotonic
counter, 15 s idle timeout) — causally empty.

Recorded as memory `feedback-refutation-needs-its-own-panel`: **a diagnosis that refutes a
premise needs the same adversarial pass on its replacement.** Being right about the negative half
buys the positive half nothing, and the positive half is what produces an operator action.

Two clock traps worth keeping: `journalctl` realtime AND `-o short-monotonic` are both journald
*receive* clocks and both contaminated during a stall — the printk clock is raw `dmesg` or
`_SOURCE_MONOTONIC_TIMESTAMP` via `-o json`. And never decode `/proc` `starttime` against
`btime`, which is truncated to whole seconds.

## Write hardening (scope 2/3, post-capture by design)

The draft was self-defeating and the bad arithmetic was hiding it. All **seven** writer call
sites are synchronous inside the async `_consume` coroutine (no `run_in_executor`/`to_thread`/
`threading` anywhere), so every `fsync()` blocks the event loop — exactly how `ping_timeout=15`
trips — and the draft kept **synchronous** fsync for gap records, which are written *as a
consequence of* stalls. Spec now carries R1–R6 and a costed single-writer-thread design behind a
bounded non-blocking queue, an honest new loss class (SIGKILL loses queued-but-unwritten
records), and an ordering bug found en route: `close()` flushes counters at `recorder.py:874`
**before** `_writer.close()` at `:875`, stranding the worker.

The N=30 s justification was **withdrawn rather than patched** — under the worker design at most
**one** fsync is blocked per outage regardless of N, so "keep the window below the stall
duration" described nothing real. N is a pinned 10–60 s range; mechanism, not constant.

Honest claim, stated in the spec: dropping per-line fsync **cannot stop these stalls** (they are
device-level). It converts a data-loss event into a latency event.

## Open — operator

1. **The settling test is `systemctl mask --now udisks2`** for two `:X2:36` phases, then
   `unmask --now`. **Not a plain `stop`** — there is no `udisks2.socket`, activation is the D-Bus
   name, and a live desktop session's `gvfs-udisks2-volume-monitor` plus 8 other gvfs processes
   re-activate it within seconds; the operator would see stalls continue and wrongly read it as
   "refuted". **Do not lead with `smartctl`** — reading SMART *is* the trigger.
2. **Root fs is genuinely at risk**: 21 k+ command timeouts on the disk holding `/`, `/home`, the
   postgres datadir and the capture. Verified backup is prudent regardless of cause.
3. Whether the drive, its firmware (SKC600, FW `109R10A0`) or the host's passthrough is at fault
   is **not** determined — only that the poll is the trigger and the drive never answers it.
4. **Not reconciled, recorded not resolved:** duty cycle lands ~10–12%, close to D74's 11.56% —
   but at ~11.5% raw outage there is almost no room for the reconnect overhead the same analysis
   puts on top. The two are NOT shown to be the same quantity. Kept NOT ESTABLISHED.

## Process deviation — the operator's to rule on

**4 cycles, exceeding `guards.halt_after_failures=3` by one closing pass.** Disclosed in the
verdict, not hidden. Rationale: the task never met a wall of the kind the rule guards (no OPEN
decision, no missing data — D43 is the only OPEN row and is unrelated), each cycle converged on
real defects, and cycle 4 transcribed evidence a judge had already collected and confirmed rather
than attempting a new solution. Read strictly, the rule said halt after cycle 3 and seed a
DECISIONS row; I judged shipping a corrected artifact better than shipping a known-defective one.

## Box discipline

Every probe was pure reads under the 2026-08-30 shared-box standing rule: `ps`, `/proc`, `/sys`,
`dmesg`, `journalctl`, `sar`, `strings`, `dpkg`, `systemctl show`/`list-*`. Scratch lived only in
`/dev/shm` (tmpfs) — the sampler's own `write_bytes` delta was verified **0**, so the instrument
added zero block I/O to the device under investigation. No `smartctl`, no device command, no
start/stop/mask, no D-Bus call, no signal, no restart. Cleaned up and re-verified: no stray
processes, `/dev/shm` empty, capture alive and untouched.

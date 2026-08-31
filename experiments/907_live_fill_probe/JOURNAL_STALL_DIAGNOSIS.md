# 907 capture stalls — diagnosis (TASK_913 scope item 1)

**Measured on `gpu-server` by the dark-factory orchestrator under the operator's 2026-08-30
shared-box fence: pure reads only** (`ps`, `/proc`, `/sys`, `dmesg`, `journalctl`, `sar`,
`strings`, `dpkg`, `systemctl show`/`list-*`). No write probes, no device commands, no restarts,
no signals, no D-Bus calls.

| pass | window (CEST) | what it did |
|---|---|---|
| first pass | 2026-08-31 **04:59 → 05:27:56** | 1 s-cadence `/proc` sampler: 3 on-phase windows + 3 off-phase controls |
| adversarial review | 2026-08-31 ~05:30 | independent judge; sampled a 4th stall, read the persistent journal |
| second pass | 2026-08-31 **05:39 → 06:14** | full 3-boot journal census, 600 s period fit, 3 further live stall probes, trigger identified |
| adversarial re-review | 2026-08-31 ~06:40 – 06:55 | attacked the `udisksd` conclusion; it survived; found the clock defect in the timing tables and measured the precedence properly |
| **closing corrective pass** | 2026-08-31 **~19:00 – 19:20** | **evidence chain rebuilt: udisks2 unit-log census made primary; every timing figure re-derived on the printk clock; §1's reset times recovered; §3's speed table recomputed; §4b's precedence argument replaced** |

The samplers wrote exclusively to their own stdout over the SSH channel or to `/dev/shm`
(tmpfs), so the instruments contributed **zero** block I/O to the device under investigation —
verified: the first-pass sampler's own `write_bytes` delta over its window was `0`. All probes ran
under a hard remote `timeout` and are confirmed exited; the server was re-checked afterwards for
leftover processes and found clean. The closing pass was read-only throughout (`journalctl`,
`dmesg`, `/proc`, `/sys`, `systemctl show`, `dpkg`, `strings`, `findmnt`), staged its extracts in
`/dev/shm` and deleted them. Nothing was left running.

> **This document was confidently wrong on its first pass, and its second pass got the *duration*
> arithmetic wrong for the same reason the first pass got the *direction* wrong: an unreconciled
> clock.** Both corrections are recorded in full below, before the evidence, because for the next
> reader they are the most useful thing here.

---

## Verdict

**1. SETTLED — the 600 s actor is `udisksd`'s SMART housekeeping poll. It says so itself, 5,452
times.**

`sudo journalctl -b 0 -u udisks2` contains **5,452** instances of exactly one line:

```
udisksd[1946]: Error performing housekeeping for drive
  /org/freedesktop/UDisks2/drives/KINGSTON_SKC600_2f512GB_YS20220148:
  Error updating SMART data: Error reading SMART data from device:
  Input/output error (udisks-error-quark, 0)
```

The entire unit log for the current 37.9-day boot is **5,454 lines: those 5,452 plus two systemd
lines** (`Starting…`, `Started…`). First failure `2026-07-24 22:22:51.083`, **boot + 31.25 s**.
Last `2026-08-31 18:53:37`. Over the 3,270,646 s span, 600 s of cadence predicts **5,451.08**
intervals; there are **5,451**. Inter-entry gaps: min 536.0 s, median 600.1 s, max 663.6 s,
**zero gaps above 700 s**. Fitted period **599.998991 s** (−1.7 ppm).

**Zero cycles missed. Zero cycles succeeded.** The daemon has been issuing one SMART read every
600 s since 32 seconds after boot, and every single one has failed with `EIO`. That count is not
an inference about who the actor is — it is the actor's own log, obtained with no intervention,
and it matches the kernel's link-up event count (5,452 two-pass events, §3) exactly, one for one.

This is a strictly better instrument than the `ata32` link-up census the second pass led with: it
counts the same events, and it **names the process**. Corroborated at kernel-stack level in §5,
where a `udisksd` thread is caught mid-stall inside the passthrough ioctl.

**What remains missing is one thing only: the counterfactual.** Nobody has stopped the poll and
watched the stalls stop. That intervention is forbidden to the factory by the shared-box standing
rule; it is Recommendation 1 and it is the operator's.

**2. SETTLED — there is no journal hog. TASK_913 scope item 1's premise is dead.**
Scope 1 asks us to *"identify the process pushing the GB-scale transaction (a 61.5 s commit on
an SSD is a big write, not device contention)"*, naming bitcoind, postgres and journald.
**No such process exists.** During a stall the device is not busy with someone else's write — it
is **absent**: zero sectors read, zero sectors written, while `/proc/diskstats` field 10
(`ms_doing_io`) accumulates at the full 1000 ms/s, i.e. the queue is 100 % utilised and
completing nothing. Every writer blocks at once — `jbd2/sda6-8`, the recorder in
`jbd2_log_wait_commit`, `postgres: main: walwriter`, the writeback kworkers. bitcoind, postgres
and journald are exonerated; postgres is a fellow victim.

**3. SETTLED — the link reset is the RECOVERY ACTION, not the trigger.**
The first pass had this exactly backwards. The device stops completing commands first; the
host's SCSI error handler resets the link seconds-to-a-minute later to try to recover it.
Across the **entire persistent journal — all three boots, 2026-06-05 to now — there are ZERO
`ata32: SATA link down` events and ZERO `Emask` / `SError` / `hard resetting` / `failed command`
/ `CRC` / `BadCRC` lines** (§4). The link is never observed to fail. It is only ever observed to
come *up*, which is what the tail of a command-timeout recovery looks like. Corrected chain:

> `udisksd` issues the 600 s SMART `SG_IO` passthrough (t₀, wall-clock phase `:X2:36`)
> → the drive accepts it and never answers
> → the SCSI command timeout (`/sys/block/sda/device/timeout` = 30, `eh_timeout` = 10) expires
> → `scsi_eh_31` wakes, libata EH runs a first recovery pass ending in a link reset
>   → `ata32: SATA link up` + `ata32.00: configured for UDMA/33` **(this is the recovery)**
> → a **second** EH pass follows — in **every** one of 5,452 observed events, never one, never three
> → the passthrough is failed back to `udisksd` as `EIO`, which logs it, and I/O resumes
>   **at that same instant** (the two coincide to within ~20 ms, §5)
> → **typical outage ≈ 61 s; observed range 38 – 93 s.**

Downstream of that, the recorder-victim chain is unchanged and still well-evidenced: per-line
`fsync` → `jbd2_log_wait_commit` → blocked asyncio event loop → `ping_timeout=15` → `1011`
keepalive close → reconnect.

**This inverts the first pass's conclusion twice over.** There is no hardware repair indicated,
and the thing to change is a daemon, not a cable. It also makes recommendation 1 of the first
pass actively dangerous: **reading SMART is what triggers the stall.**

**What we deliberately do NOT claim: that this is a cable, a connector, a backplane or a power
rail.** The first pass inferred that from the 1.5 Gbps / UDMA-33 link speed. That inference is
withdrawn — see §6, the speed is an *output* of the phenomenon. There is at present **no
evidence of a physical-layer fault at all.**

---

## What this artifact got wrong, and why

### First pass

| # | first pass said | actually |
|---|---|---|
| 1 | the link "drops and re-establishes"; link drop → stall | **zero link-downs and zero link errors exist in the entire journal.** The device goes quiet ~30 s *before* the link is touched. The link-up is libata's recovery action. |
| 2 | 1.5 Gbps / UDMA-33 corroborates a marginal cable | link speed is **kernel-imposed**, cleared at every boot and re-earned. The identical 600 s cycle ran at **6.0 Gbps / UDMA/133 with NCQ enabled**, from 32 s after the 2026-07-24 boot. |
| 3 | "Nothing on the process table can be moved, tuned or rescheduled to fix this" | an over-claim, and **false**. The trigger *is* on the process table: `udisksd`'s 600 s SMART housekeeping poll. It can be stopped with one `systemctl` command. |
| 4 | "NOT VERIFIED: the pre-2026-08-18 history — the dmesg ring does not reach further back" | `journalctl` held **4 GB of persistent journal back to 2026-06-05, across three boots** — containing the 6.0 Gbps boot probe, the speed-down ladder and the boot-anchored restart. Every piece of disconfirming evidence was one command away. |
| 5 | "60 s per 600 s = 10.0 %" matches the capture's 11.56 % loss — "the arithmetic closes" | n = 1, and it excluded a second link-up 31 s later that the same paragraph quoted. See §8: the duty cycle is now derivable at n = 5,452 and lands near 11 %, but the agreement with 11.56 % is still **not established** — and for a new reason. |
| 6 | a self-contradiction: "a single 12,288-byte blip at 05:03:07" *and* "zero bytes for the whole 60 s" | the blip is **real and informative**: it lands at the *first* EH recovery pass, which drains one command before the device goes away again. Reproduced twice more (§3). |
| 7 | header: "Measured 2026-08-31 04:59–05:08" | the sampler ran until **05:27:56**. Fixed. |

### Second pass

| # | second pass said | actually |
|---|---|---|
| 8 | the actor identification is "GOOD confidence, one test short of settled" | **understated.** 5,452 self-logged `EIO` failures out of 5,452 cycles, from the daemon's own unit log, settle *who*. Only the counterfactual is missing, and it is forbidden to us. |
| 9 | per-epoch duration table showing the outage distribution is "speed-invariant", computed from `journalctl` **realtime** stamps | the clock was wrong. Rebuilt on the printk clock in §6. The **conclusion survives and strengthens** — the median span is 30.72 s in *every* speed epoch, to three decimal places — but the table as printed is withdrawn and replaced. |
| 10 | "same three modes (≈0 s, ≈31 s, ≈62 s, i.e. one, two or three EH passes)" | the **≈0 s mode does not exist**; it is journald collapse (§2). And **every** event has exactly **two** passes — 5,452 of 5,452. The three modes are in the *interval between* the two passes: ≈7.5 s, ≈30.7 s, ≈62 s. |
| 11 | precedence proved by decoding the pool thread's `/proc` `starttime` to "06:02:35, one second before the 06:02:36 onset" | that margin **is the bug**. `/proc/stat` `btime` is truncated to whole seconds, so every such decode lands ~0.84 s early on this box (§2). Replaced with a single-clock measurement in §5. |
| 12 | `udisksd` starts at boot **+5 s**; the cycle anchors at boot **+14.5 / +16.5 s**; there is a gap to explain | both numbers are wrong. `udisksd` execs at boot **+13.35 s**; the boot-0 link-up offsets were anchored to the first journal entry of the boot, not to boot. Corrected chain in §5 — **there is no gap.** |
| 13 | `pool-5402 → pool-5403` one cycle apart was "predicted in advance, then observed" — treated as strong confirmation | **entailed by the cadence, not a risky prediction.** GLib names thread-pool workers from a monotonically increasing per-process counter and retires them on a 15 s idle timeout, so one job per 600 s *necessarily* gets a fresh, next-numbered thread. It would look identical under the inversion. Demoted (§5). |
| 14 | the pool thread is "gone once the device recovers" | it **lingers 15.4 s past recovery** (GLib idle timeout) — measured: thread 914910 survived the 06:43:37.4 recovery until 06:43:52.913. Removed as corroboration. |
| 15 | outage is "38–93 s (n = 4)"; "the duration is not rigid" | over-read from four hand-samples when n = 5,452 was in the log. Properly clocked: the second-pass interval is 30.7 s in **73.4 %** of all events and **91.7 %** of the last 12 h → **typical outage ≈ 61 s**. 93 s is a real tail worth keeping as a *sizing* bound; the hand-samples over-sampled the tails roughly 3:1. |
| 16 | period "600.000337 s (≈0.56 ± 0.3 ppm)" | over-precise, and computed on a contaminated clock. The `udisks2` `EIO` series — on the NTP-corrected wall clock — fits **599.998991 s (−1.7 ppm)**. Safe claim: **a host timer at 600 s to within a few ppm.** |
| 17 | `systemctl stop udisks2` (with `udisks2.socket` "if present") is the confirming test | **there is no `udisks2.socket`**, and a plain `stop` will be undone within seconds by D-Bus activation from the live desktop session's `gvfs-udisks2-volume-monitor`. The operator would see stalls continue and wrongly conclude "refuted". Corrected protocol in Recommendation 1. |

### The mechanisms behind all of this — three, and the third is a repeat of the second

**(a) A load-bearing claim was bounded by data we never asked for.** Row 4 is the expensive one.
The first pass wrote "NOT VERIFIED — the ring does not reach further back" and then reasoned
*inside* that self-imposed boundary, treating an unexamined limit as a fact about the world. One
`journalctl --list-boots` would have dissolved it, and the evidence beyond that boundary reverses
the conclusion. This lane's recurring failure mode is precisely this: **an unverified
load-bearing claim, carried forward because checking it looked like scope.** Checking the
boundary of your evidence is not out of scope; it is the cheapest thing in the budget.

**(b) A clock was 29 seconds wrong and nobody checked it.** The first pass quoted `dmesg -T`
timestamps. On this box **`dmesg -T` runs ~29 s ahead of the journal's wall clock** — verified
twice live by the second pass, and reproduced independently by the closing pass at 28.93–28.95 s
across several events. So the link-up the first pass called "the instant our sampled stall ended"
— dmesg `05:03:35` — is really `05:03:06`, *thirty-one seconds before* the stall ended, and it
coincides with the 12,288-byte blip the first pass had already recorded and dismissed as noise.
The evidence that would have overturned the causal direction was sitting in the document,
mis-stamped. **When two clocks appear in one argument, reconcile them before reasoning across
them.**

**(c) The same clock was still lying in the other direction, and the document that caught it did
not re-check.** This is the expensive one this time, and it is the *third* instance of this
artifact's own named failure mode. Having discovered that `dmesg -T` and journal wall clock
disagree, the second pass adopted the journal's realtime stamps and built its duration tables on
them — without asking whether *that* clock was fit for measuring durations **of an event that
blocks journald itself**. It is not. journald's realtime stamp is applied when it drains the
kernel ring, and during a stall journald is a victim like everything else, so it stamps a whole
burst at the moment it recovers. 40 of the last 72 events collapse to under one second of
apparent duration. Catching a clock lying about *A* is not evidence it is telling the truth about
*B*; the fix for a clock problem is to name which clock is valid for which quantity, which §2
now does.

---

## Evidence

### 1. The actor's own log — 5,452 self-reported failures (primary)

`sudo journalctl -b 0 -u udisks2`, complete:

| quantity | value |
|---|---|
| total lines in the unit log, current boot | **5,454** |
| `Error performing housekeeping … Error reading SMART data … Input/output error` | **5,452** |
| every other line | **2** (systemd `Starting udisks2.service`, `Started udisks2.service`) |
| successful SMART updates | **0** |
| first failure | `2026-07-24 22:22:51.083` = **boot + 31.246 s** |
| last failure (at time of writing) | `2026-08-31 18:53:37` |
| span | 3,270,646.4 s = **37.855 d** |
| intervals expected at 600 s | **5,451.08** |
| intervals observed | **5,451** — **zero missed** |
| gap min / median / max | **536.0 / 600.1 / 663.6 s** |
| gaps > 700 s | **0** |
| fitted period (least squares on cycle index) | **599.998991 s** (**−1.7 ppm**) |
| residual sd | 15.1 s (max 58.6 s) — the poll's own latency spread, not trigger jitter |

The drive is named in the message: `KINGSTON_SKC600_2f512GB_YS20220148` — the `sda` device under
investigation, and the only `sd*` device on the box.

Two things this settles that nothing else in the artifact does:

- **It names the process, with no probe and no intervention.** Every other line of evidence
  observes the *consequence* (a link reset, a blocked writer) and reasons back. This is the actor
  logging its own failure, 5,452 consecutive times.
- **Zero successes means the poll has never once worked.** Not "it usually works and occasionally
  hangs" — the SMART read on this drive has failed with `EIO` on every attempt since 32 seconds
  after boot. Whatever is wrong with the drive's handling of this command is total and constant.

The count is also the tightest cross-check available: **5,452 `EIO` failures and 5,452 kernel
two-pass link-up events over the same boot** (§3). One recovery per poll, one poll per cycle.

*(The census is re-runnable and the count advances on cadence: 5,380 at 06:53:37, 5,452 at
18:53:37 — exactly 72 more in 12 h. Any figure below quoted as "5,452" is a snapshot of a
counter that increments every 600 s.)*

### 2. Which clock is valid for what — read this before any timing figure

This box has three clocks in play and **each is wrong for one of the two questions** we ask of
it. Getting this wrong produced the second pass's duration tables and its precedence argument.

| clock | how to read it | valid for | **invalid for** |
|---|---|---|---|
| **printk** (`local_clock`) | raw `dmesg`, or `_SOURCE_MONOTONIC_TIMESTAMP` from `journalctl -o json` | **durations / intervals between kernel events** — stamped by the kernel at the moment of printk, unaffected by anything downstream | **absolute wall clock.** Not NTP-corrected; its anchor has drifted **≈29 s over 37.9 days (≈9 ppm)**. This is the `dmesg -T` offset of correction (b). |
| **journald realtime** (`__REALTIME_TIMESTAMP`) | `journalctl -o short-iso`, default output | **wall-clock attribution** — NTP-corrected (`NTPSynchronized=yes`), so "when did this happen" | **durations of a stall.** Stamped at *drain* time, and journald is blocked by the stall. Collapses bursts. |
| **journald monotonic** (`__MONOTONIC_TIMESTAMP`) | `journalctl -o short-monotonic` | nothing here | **both.** It is the *receive* monotonic clock, so it carries the same drain contamination as realtime while looking like a kernel clock. |

**The third row is a trap and it caught this pass.** `journalctl -o short-monotonic` looks like
the printk clock and is not. Verified on one event, all three clocks side by side:

| line | printk (`_SOURCE_MONOTONIC`) | journald monotonic | journald realtime |
|---|---|---|---|
| `ata32: SATA link up` (1st pass) | **3225052.681123** | 3225023.618139 | 06:12:43.450331 |
| `ata32.00: configured for UDMA/33` | **3225052.690533** (+9 ms) | 3225054.415661 (+30.798 s) | 06:12:43.450331 + 0 |
| `ata32: SATA link up` (2nd pass) | **3225083.385959** (+30.705 s) | 3225054.415723 | 06:13:14.247 |
| `ata32.00: configured for UDMA/33` | **3225083.392962** (+7 ms) | 3225054.430660 | 06:13:14.262 |

The printk column is coherent — link-up, then `configured` 7–9 ms later, twice, 30.705 s apart.
The journald-monotonic column reports a `configured` line arriving **30.8 s after its own
link-up**, which never happens; raw `dmesg` for the same four lines matches the printk column
byte for byte.

**What journald realtime does to the duration distribution.** The same 72 events from the last
12 h, measured under both clocks:

| span source | median | mean | ≥ 60 s share | histogram (rounded s) |
|---|---|---|---|---|
| **printk (truth)** | **30.713 s** | 31.81 | **5.6 %** | `{7.5: 2, 30.7: 66, 62: 4}` |
| journald realtime | **0.147 s** | 13.38 | 1.4 % | `{0: 40, 1: 1, 3: 1, 31: 27, 33: 2, 62: 1}` |

**40 of 72 events collapse to under one second.** The "≈0 s mode" in the second pass's tables is
not physical — it is the instrument being blocked by the thing it is measuring. Worked example
from the same window: the 05:13 event's two passes are **30.713 s** apart on the printk clock and
**0.002 s** apart on journald realtime.

**And the `starttime` trap.** `/proc/<pid>/stat` field 22 is in ticks since boot; converting it to
wall clock requires a boot epoch, and `/proc/stat`'s `btime` is **truncated to whole seconds**:

```
btime                        = 1784924539
(now_realtime − uptime)      = 1784924539.837122      → btime is 0.837 s early
```

So *every* `starttime` decode against `btime` on this box lands ~0.84 s early — which is the
entire margin the second pass's precedence claim rested on. Confirmed live on the 06:42:36 cycle:
the pool thread was absent from `/proc/1946/task` at 06:42:35.78 and present at 06:42:35.88
(created ≈ 06:42:35.83), while its `starttime` decoded to 06:42:35.02.

**Rule: never decode `starttime` against truncated `btime`.** Observe the transition directly, or
subtract `(now_realtime − uptime) − btime` as an offset correction.

*Every timing figure below is labelled with the clock it came from.*

### 3. The kernel's view — 5,452 events, every one exactly two passes

Full boot-0 `ata32` link-up series on the **printk clock** (`_SOURCE_MONOTONIC_TIMESTAMP`,
cross-checked against raw `dmesg` for the 12.9 days the ring still holds):

| quantity | value |
|---|---|
| `ata32: SATA link up` lines, boot 0 | **10,909** |
| recovery events (clustering link-ups > 200 s apart) | **5,452** + the boot probe |
| link-ups per event | **2 — in 5,452 of 5,452** |
| event-start gaps: min / median / max | 538.6 / 600.1 / 661.5 s; **zero > 700 s** |
| first-pass → second-pass interval, median | **30.721 s** |
| … mean | 36.71 s |

The **inter-pass interval is trimodal**, and this is where the "three modes" the second pass saw
actually live — not in the number of passes:

| mode | n | share | median | ⇒ outage (with a ~30 s first-timeout term) |
|---|---|---|---|---|
| ≈ 7.5 s | 250 | 4.6 % | 7.55 s | ≈ 38 s |
| **≈ 30.7 s** | **4,000** | **73.4 %** | **30.72 s** | **≈ 61 s** |
| ≈ 62 s | 1,202 | 22.0 % | 62.48 s | ≈ 93 s |

Over the last 12 h the mix is tighter still: 66/72 at 30.7 s (91.7 %), 2 at 7.5 s, 4 at 62 s.

**`ata32` is `sda`**, verified by topology:
`/sys/block/sda` → `/sys/devices/pci0000:40/0000:40:08.3/0000:4b:00.0/**ata32**/host31/target31:0:0/31:0:0:0/block/sda`.
`nvme0n1` is a separate device on a separate controller and shows none of this.

#### 3a. The four directly-sampled stalls, reset times re-derived

Four stalls were sampled at the `/proc` level by two independent parties. All four onset at
wall-clock phase **`:X2:36`**. Onset and resume are from the `/proc` samplers (an uncontaminated
clock); the reset times are journald **realtime** (correct for *when*); the inter-pass interval is
**printk** (correct for *how long*):

| # | sampled by | onset | 1st EH pass | 2nd EH pass | interval (printk) | I/O resumes | **outage** |
|---|---|---|---|---|---|---|---|
| A | first pass | 05:02:36.8 | 05:03:06.562 | 05:03:37.274 | **30.713 s** | 05:03:38.4 | **61.6 s** |
| B | judge | 05:32:36 | 05:33:38.482 | 05:34:09.246 | **30.721 s** | 05:34:09 | **93 s** |
| C | second pass | 05:52:36.5 | 05:53:06.887 | 05:53:40.016 | **30.713 s** | 05:53:37.8 | **61.3 s** |
| D | second pass | 06:12:36 | 06:12:43.450 | 06:13:14.247 | **30.705 s** | 06:13:14 | **38 s** |

**The printk clock dissolves the anomaly the second pass had to hedge.** It noted that window C's
second reset appeared to land *after* the `/proc`-sampled resume (05:53:40.0 vs 05:53:37.8) and
put that down to journal resolution. It was drain lag: 05:53:06.887 + the true 30.713 s interval =
**05:53:37.600**, which is 0.2 s *before* resume, as it must be. Applied to all four windows, the
last EH pass now lands at or just before resume in every case.

**Decomposition.** Outage = (onset → 1st pass) + (inter-pass interval). The second term is known
at n = 5,452 (above). The first term is only measurable with a `/proc` sampler and is **7.4, 29.8,
30.4 and 62.5 s** across these four — and note those are the *same three values* as the inter-pass
modes. `/sys/block/sda/device/timeout` = 30 and `eh_timeout` = 10, so ~30 s is the mechanistically
expected first term and 2 of 4 samples hit it; the other two show it is not rigid. This first term
is the honest limit on how far n = 5,452 can pin the outage — the kernel log gives the second term
at high n and says nothing directly about the first.

**Typical outage ≈ 61 s. Observed range 38 – 93 s (n = 4). Use 93 s for sizing.**

#### 3b. Window C in full shape (0.4 s cadence — the cleanest sample)

| time | Δ sectors written | Δ sectors read | Δ `ms_doing_io` (per 0.4 s) |
|---|---|---|---|
| 05:52:35.3 | 64 | 0 | 17 |
| **05:52:36.5** | **0** | **0** | **111** ← onset |
| 05:52:37.3 … 05:53:06.5 | **0** | **0** | **400–401** (i.e. 100 % busy) |
| **05:53:06.9** | **80** | 0 | 705 ← **the blip: 1st EH pass** |
| 05:53:07.3 … 05:53:37.4 | **0** | **0** | **400–401** |
| **05:53:37.8** | **40** | 0 | 793 ← recovery |
| 05:53:38.2 onward | 80, 72, …, 480 | 0 | resumed |

Three things this makes plain:

- **`ms_doing_io` pins at 400 per 400 ms.** The block layer thinks I/O is in flight the entire
  time. This is not an idle device and not a queueing effect — it is a device that has accepted
  commands and stopped answering. (Note: `/proc/diskstats` field 9, "I/Os in progress", reads `0`
  throughout on this box and is not a usable stall signal; field 10 is.)
- **The blip is the first EH pass, not noise.** First-pass contradiction #6 resolved. One command
  drains (80 sectors here; 24 sectors / 12,288 bytes in window A), and the device goes away again
  until the second pass.
- **Reads are zero too.** The first pass only tracked writes; the device is not answering
  *anything*.

**Also blocked, simultaneously, in window A:** `jbd2/sda6-8`, the recorder, `postgres: main:
walwriter`, `kworker/*+flush-8:0`, and `scsi_eh_31` — the SCSI *error handler* thread, which only
runs when a command has timed out and recovery is executing. Every writer to the device stops at
once. That is what a device-level fault looks like and what a per-process hog does not.

#### 3c. Replicated 3/3 on-phase, with 3 clean negative controls

| window | phase | result |
|---|---|---|
| w1 05:02 | on | **STALL 05:02:37.96 → 05:03:36.90** |
| c1 05:06 | off (control) | **no stall** — +6,438 tx, 124.0 MB written |
| w2 05:12 | on | **STALL 05:12:37.72 → 05:13:36.66** |
| c2 05:16 | off (control) | **no stall** |
| w3 05:22 | on | **STALL 05:22:37.88 → 05:23:36.81** |
| c3 05:26 | off (control) | **no stall** |

**3/3 on-phase windows stall; 0/3 off-phase controls do.** Perfect separation. Onset-to-onset:
**599.757 s** and **600.163 s**. Off-phase the device is entirely healthy — the controls sustain
~2× the transaction rate and ~2.7× the write throughput of the stall windows. This is an
*episodic total outage*, not degradation.

*Precision caveat:* the first-pass sampler's grid is ~1.13 s and is phase-aligned across rounds,
so the three durations agreeing closely is partly a quantisation artifact. Honest resolution on
both onset and duration in that pass is **±1.1 s**. Window C is 0.4 s-resolved.

### 4. The kernel journal — three boots, 2026-06-05 to now, and no link failure in any of them

`journalctl --disk-usage` = **4 G**, persistent, three boots:

| boot | first entry | last entry |
|---|---|---|
| −2 | 2026-06-05 08:26 | 2026-07-24 21:45 |
| −1 | 2026-07-24 21:49 | 2026-07-24 22:20 |
| 0 | 2026-07-24 22:22 | now (37.9 d uptime) |

Complete census of every `ata32` kernel line, per boot:

| boot | total `ata32` lines | `SATA link up` | `SATA link down` | `Emask`/`SError`/`hard resetting`/`failed command`/`CRC`/`BadCRC` |
|---|---|---|---|---|
| −2 | 25,296 | 12,648 | **0** | **0** |
| −1 | 21 | 9 | **0** | **0** |
| 0 | 21,826 | 10,909 | **0** | **0** |

Apart from `SATA link up` and its paired `configured for UDMA/N`, the **only** other `ata32`
lines in 47,143 are **eleven** one-shot messages — eight in boot 0 (three port/identify lines
plus the five escalation-ladder lines in §6), three in boot −1, none retained in boot −2.
There is no error reporting on this link at all. A failing SATA link produces `SError`, `Emask`,
`failed command` and `SATA link down`; this one produces none of them, ever, over 88 days of
journal.

Device error counters are **live and growing**, and the closing pass pins the per-cycle rate the
second pass could only bracket (both readings' wall-clock instants are now recorded):

| counter | 05:44 | 19:14 | Δ over 81.0 cycles | **per cycle** |
|---|---|---|---|---|
| `ioerr_cnt` | 32,238 | **32,730** (`0x7fda`) | +492 | **6.07** |
| `iotmo_cnt` | 21,219 | **21,540** (`0x5424`) | +321 | **3.96** |
| `iodone_cnt` | 57,928,792 | 69,214,208 (`0x4202000`) | +11,285,416 | 139,326 |

So roughly **6 fresh I/O errors and 4 fresh timeouts per 600 s cycle** — a single hung passthrough
does not produce one timeout, it produces a small handful as EH retries and re-queues.

### 5. The trigger, caught live at kernel-stack level

Second-pass probe, 06:02:36 stall, root `/proc/*/stack` reads. Three stacks tell the whole story:

```
[udisksd tid=865397 "pool-5402"]
    blk_execute_rq <- sg_io <- scsi_ioctl <- sd_ioctl <- blkdev_ioctl
                   <- __x64_sys_ioctl <- do_syscall_64            ← THE TRIGGER

[jbd2/sda6-8 tid=868]
    __wait_on_buffer <- jbd2_journal_commit_transaction <- kjournald2   ← blocked behind it

[python tid=4082411]   (the 907 recorder)
    jbd2_log_wait_commit <- jbd2_complete_transaction <- ext4_fc_commit
                         <- ext4_sync_file <- do_fsync <- __x64_sys_fsync  ← the victim
```

The third stack is the recorder-victim chain confirmed at kernel-stack level: `os.fsync()` →
`ext4_sync_file` → `jbd2_log_wait_commit`, exactly as `WRITE_HARDENING_SPEC.md` describes it.

`blk_execute_rq` is the **passthrough / non-filesystem** command path — `SG_IO`, ATA passthrough,
`TEST UNIT READY`, `sd_revalidate_disk` — **not** ordinary writeback, which goes through
`submit_bio`. No victim of a journal stall is ever on this path.

The identification rests on four independent facts:

| # | fact | evidence |
|---|---|---|
| 1 | the stuck thread belongs to `udisksd` | tid `865397` is under `/proc/**1946**/task/`, PID 1946 = `udisksd` (`ExecMainPID=1946`) |
| 2 | it is a **passthrough**, not filesystem I/O | `sg_io` ← `scsi_ioctl` ← `sd_ioctl` — a raw `SG_IO` ioctl on the block device. Nothing to do with the journal. |
| 3 | it is **SMART housekeeping** | `udisks2` 2.10.91-1ubuntu2 depends on `libblockdev-smart3` 3.4.0-1 (and is its **only** installed reverse-dependency). The `udisksd` binary carries `housekeeping_all_drives`, `udisks_linux_drive_object_housekeeping`, `perform_initial_housekeeping_for_drive`, and the exact format string of the 5,452 logged failures: `Error performing housekeeping for drive %s: %s (%s, %d)`. |
| 4 | **the daemon logs the failure itself, 5,452 times, once per cycle, zero successes** | §1 |

Fact 4 is the load-bearing one and it is not an inference. Facts 1–3 explain *how* the poll stalls
the device; fact 4 establishes that it is this poll and nothing else, on every cycle since boot.

#### 5a. Precedence — trigger before victim, on one clock

This is the question the document has to get right, having got it wrong once. The second pass's
`starttime` decode is withdrawn (§2: it is biased 0.84 s early by `btime` truncation, which was
the whole claimed margin). Replaced with a single-clock live measurement of the 06:42:36 cycle:

| t (wall clock) | observation | source |
|---|---|---|
| **06:42:35.83** | `udisksd` pool thread created (absent from `/proc/1946/task` at 35.78, present at 35.88) | `/proc` transition, directly observed |
| **06:42:36.22** | block queue goes permanently non-empty — the 0.5 s bucket ending 36.284 shows `dsect_wr=0, dms_io=68`, i.e. the queue was occupied only for its last 68 ms | `/proc/diskstats`, same sampler, same clock |
| 06:42:36.2 → 06:43:37.4 | 100 % busy, **zero sectors** in either direction, for 61 s | same |
| **06:43:37.402441** | `udisksd` logs `EIO` — the passthrough is failed back to it | journald realtime |
| 06:43:37.438582 | kernel `ata32: SATA link up` (2nd EH pass) | journald realtime |
| **06:43:37.46** | writes resume | `/proc/diskstats` |

**Lead ≈ 0.39 s, thread first**, both endpoints on the same sampler's clock — no cross-clock
arithmetic, no `btime`. And the outage **terminates at the instant the passthrough is failed**:
`EIO`, link-up and resume land within 60 ms of each other. The same coincidence is visible at boot
on the printk clock (§5b), where the second link-up and the first logged `EIO` agree to 12 ms.

A victim thread would appear *after* onset, as `jbd2` and the recorder do. This one appears
before, on the passthrough path, with the daemon's own cadence.

#### 5b. The boot chain — corrected, and there is no gap to explain

The second pass reported `udisksd` starting at boot +5 s against a cycle anchored at +14.5/+16.5 s
and treated the ~9.5 s difference as an open puzzle. Both numbers were wrong. Measured:

| boot offset | event | clock / source |
|---|---|---|
| +4.074 s | `ata32: SATA link up 6.0 Gbps` — the normal boot probe; `ata32.00: ATA-9: KINGSTON SKC600/512GB, 109R10A0, max UDMA/133`, `LBA48 NCQ (depth 32), AA` | printk |
| **+13.349 s** | `udisksd` exec — `ExecMainStartTimestampMonotonic=13348935` µs | systemd |
| +13.34 s | same, cross-checked: `/proc/1946/stat` field 22 = **1334** ticks ÷ `CLK_TCK` 100 | `/proc` |
| ≈ +14 s | coldplug → `perform_initial_housekeeping_for_drive` issues the first SMART passthrough | inferred (bracketed by the two rows above and below) |
| +23.666 s | 1st EH pass — `ata32: SATA link up 6.0 Gbps` | printk |
| +31.234 s | 2nd EH pass (interval **7.568 s** — the ≈7.5 s mode of §3) | printk |
| **+31.246 s** | `udisksd` logs the first `EIO` — **12 ms after the second pass** | journald realtime |
| +623.767 / +631.319 s | next cycle's two EH passes | printk |
| +631.302 s | next cycle's `EIO`; **`EIO`-to-`EIO` = 600.055 s** | journald realtime |

The second pass's quoted offsets (+0.9 / +16.5 / +24.1 / +624.1) are each ~7.15 s low because they
were anchored to the first *journal entry* of the boot rather than to boot; the kernel's own clock
gives 4.07 / 23.67 / 31.23 / 623.77. **The chain closes: exec +13.35 → coldplug → first
passthrough ≈ +14 → EH +23.67 → EH +31.23 → `EIO` +31.25. No gap.**

Two caveats on this anchor, both worth stating: it should not be quoted to 0.1 s, because the
`+31.25 s` event is a *recovery* lagging its trigger by ~17 s; and the phenomenon is running at
**6.0 Gbps / UDMA/133 with NCQ enabled** here, 32 seconds into the boot, which is the observation
that kills the cable theory (§6).

#### 5c. The `pool-5402 → pool-5403` "prediction" — demoted

The second pass predicted that a new `udisksd` pool thread would appear in `sg_io` at the next
600 s phase, observed it, and treated the consecutive thread names as strong confirmation. **It
is not; it is a restatement of the cadence.** GLib names thread-pool workers from a monotonically
increasing per-process counter — the format string `pool-%d` is in `libglib-2.0.so.0`
(`libglib2.0-0t64` 2.88.0-1) — and GLib retires idle pool threads on a default 15 s timeout. So
one job per 600 s **necessarily** gets a fresh thread with the next number. Measured directly:
thread 914910 survived the 06:43:37.4 recovery until **06:43:52.913** — 15.4 s, the idle timeout,
not the device event.

Two consequences. First, the observation would look **identical** under the inversion (if
something else stalled the device and `udisksd` merely blocked on it, it would still spawn one
next-numbered pool thread per cycle), so it carries almost no causal content and cannot be part
of the confidence. Second, the second pass's claim that the thread "is gone once the device
recovers" is wrong — it lingers 15.4 s past recovery — and is removed as corroboration.

The confidence that was resting here now rests on §1's census instead, which is both stronger and
cheaper.

#### 5d. Ruled out as the actor

- systemd timers: the only ~10-minute unit is `sysstat-collect.timer`, which fires at `:X0:00–01`
  — **154 s off** the stalls' `:X2:36` phase, and its phase would follow wall clock, not boot.
  All 19 timers were listed; no other is near 600 s.
- `cron`: root and user crontabs are empty apart from `@reboot nvidia-smi -pm 1`. `/etc/cron.d`
  holds only `anacron` and `e2scrub_all`.
- no `smartd` is installed or running. **Read this one carefully: the absence of `smartd` is
  exactly what made "something is polling SMART" easy to dismiss. `udisksd` polls SMART too, and
  it *is* running — and saying so in its own log 5,452 times.**
- no `libata.force` on the kernel command line.
- ALPM unavailable: `link_power_management_policy` = `max_performance` and
  `link_power_management_supported` = **`0`** — it cannot be the cause and cannot be tuned.
- `/sys/block/sda/device/power/control` = `on` → runtime autosuspend excluded.
- `/sys/block/sda/removable` = `0`, `events_poll_msecs` = `-1` → block-layer media-change polling
  is **off** for `sda`.
- backup / scrub / fstrim: `e2scrub_all.timer` weekly (last 2026-08-30 03:10), `fstrim.timer`
  weekly (last 2026-08-31 00:33), `logrotate` daily. None is 10-minutely and none matches phase.
- the recorder itself has no 600 s behaviour.
- the port's `ext` (external / hot-plug, ESP) marking in `ata32: SATA max UDMA/133 … lpm-pol 1
  ext` — its empty sibling `ata31` carries the same flag and does nothing.
- the ten Docker containers (a Supabase stack) and `incusd`, none of which touched `sda` in the
  pre-onset window.

### 6. Link speed is a DEPENDENT VARIABLE, not evidence of a cable

The kernel logged the whole escalation as libata reacting to timeouts, in boot 0:

| when | kernel line |
|---|---|
| 2026-07-24 22:22 (boot +4.07 s) | `ata32: SATA max UDMA/133 … lpm-pol 1 ext` / `ata32.00: ATA-9: KINGSTON SKC600/512GB, 109R10A0, max UDMA/133` |
| 2026-07-25 00:02 | `ata32.00: NCQ disabled due to excessive errors` |
| 2026-07-28 07:33 | `ata32: limiting SATA link speed to 3.0 Gbps` |
| 2026-07-30 03:13 | `ata32: limiting SATA link speed to 1.5 Gbps` |
| 2026-07-30 21:23 | `ata32.00: limiting speed to UDMA/100:PIO4` |
| 2026-07-31 17:53 | `ata32.00: limiting speed to UDMA/33:PIO4` |

`link32/sata_spd_limit` = `1.5 Gbps` is kernel-set by `sata_down_spd_limit()`, is **cleared at
every boot**, and is then re-earned. `/proc/cmdline` carries no `libata.force`.

**The speed-is-an-output conclusion stands, on three legs. Two are strong; the third was
mis-computed and is now rebuilt.**

**(a) The ladder is logged as libata's own reaction to timeouts.** Every line above is the kernel
announcing a decision it made because commands were not completing. None of them is a report of a
link error — there are zero of those (§4).

**(b) The phenomenon predates the entire ladder, at full speed.** The `udisksd` `EIO` series
begins at **boot + 31.25 s** on 2026-07-24 (§1, §5b), while the link was at **6.0 Gbps /
UDMA/133 with NCQ enabled**, and it has **zero gaps** since — 5,451 consecutive 600 s intervals
spanning the whole downgrade ladder. There is no "before the fault" state to find.

**(c) The duration distribution does not worsen as the link slows** — table rebuilt on the printk
clock, grouping each event by the speed reported in its own link-up line:

| negotiated speed | n events | median interval | mean | share ≥ 60 s |
|---|---|---|---|---|
| 6.0 Gbps | 488 | **30.729 s** | 41.38 | **36.9 %** |
| 3.0 Gbps | 227 | **30.722 s** | 39.21 | **30.0 %** |
| 1.5 Gbps (current) | 4,738 | **30.721 s** | 36.11 | **20.1 %** |

> **The second pass's version of this table is withdrawn as computed** (its six date-window
> epochs, "share ≥ 60 s = 40.0 / 24.7 / 19.8 / 11.9 / 13.0 / 14.4 %"). It was built on journald
> realtime, whose collapse varies with write load — which is one of the things that changed
> between epochs — so the confound tracked the variable of interest. Recomputed on the printk
> clock the confound is gone entirely, and the finding is *cleaner*: the median inter-pass
> interval is **30.72 s at every link speed, to three decimal places**, and the ≥ 60 s share
> *falls* monotonically as the link slows. That is the opposite sign to a feedback loop.
>
> Read the ≥ 60 s column with its limits: the three groups are unbalanced (488 / 227 / 4,738) and
> are different calendar periods under different load, so it is an observational comparison, not a
> controlled one. The **median** is the robust claim, and it is flat.

The command timeout is 30 s regardless of link rate, which is the mechanistic reason to expect
exactly this. *(This also disposes of a hypothesis raised in review — that events cost ~8 s at
6.0 Gbps and 60–95 s at UDMA/33. The ≈7.5 s mode exists at every speed, 24 of 488 events at
6.0 Gbps and 215 of 4,738 at 1.5 Gbps; it is a mode of the distribution, not a property of an
epoch.)*

**So there is no speed feedback loop, and nothing here that a link renegotiation would fix. The
cable / connector / power-delivery inference is withdrawn.**

### 7. The metronome

Two independent fits, and their disagreement is itself a check on the clocks:

| series | clock | n | fitted period | offset from 600.000 |
|---|---|---|---|---|
| `udisks2` `EIO` log entries | journald realtime (NTP-corrected) | 5,452 | **599.998991 s** | **−1.7 ppm** |
| kernel link-up event starts | printk (not NTP-corrected) | 5,453 | 600.004465 s | +7.4 ppm |

The **9.1 ppm difference between the two is the printk clock's own measured drift** (≈29 s over
37.9 days, §2) — so the two series agree once the drift is accounted for, and the honest statement
is the one on the corrected clock: **a host timer at 600 s to within a few ppm.**

> **The second pass's "600.000337 s (≈0.56 ± 0.3 ppm)" is withdrawn as over-precise.** It was
> fitted on onset times reconstructed from a contaminated clock, and its stated uncertainty was
> an order of magnitude smaller than the clock error it was unaware of.

**It re-anchors to boot, not to wall clock.** The `EIO` series' phase zero is boot **+31.25 s** on
boot 0 (§5b) — a recovery event lagging the poll by ~17 s, so the poll itself anchors at ≈ boot
+14 s. Boot −1 shows the same structure in its 31 minutes of life. (Boot −2's kernel log begins
2026-06-10, five days into that boot, so its anchor cannot be checked; its cycle is present and
running from the first retained line.)

A drive's internal crystal would drift tens of ppm over 37.9 days. A few ppm over 5,451
consecutive cycles, re-anchored at boot, with **zero missed cycles**, reads as a host timer — and
§1 and §5 name the timer: udisks2's `housekeeping_all_drives` loop.

### 8. What the arithmetic does — and does not — close

**Stands, re-checked on the printk clock:**

| quantity | value |
|---|---|
| link-up events in the `dmesg` ring / ring span | 3,706 / 308.686 h = **12.006 per hour** (2 per event → **6.003 events/h** → 599.7 s) |
| capture duration at time of measurement | 69.9 h → 69.9 × 6 = **419.4** expected cycles |
| recorder reconnect attempts logged | **438** |
| reconnects per cycle | 438 / 419.4 = **1.044** |

The reconnect storm being ~1:1 with the stall cycle is solid and is the link between the device
outage and the capture's own damage.

**The duty cycle, now derivable at n = 5,452 rather than n = 4:**

outage = (onset → 1st EH pass) + (inter-pass interval). The second term is measured at n = 5,452:
mean **36.71 s** over the boot, **31.81 s** over the last 12 h. The first term is measured only at
n = 4 (7.4 / 29.8 / 30.4 / 62.5 s, mean 32.5 s), with `timeout` = 30 s as its mechanistic
expectation. Combining:

| basis | mean outage | duty cycle |
|---|---|---|
| whole boot 0 (5,452 events) | ≈ 69 s | **≈ 11.5 %** |
| last 12 h (72 events) | ≈ 64 s | **≈ 10.7 %** |

and by mode: ≈ 38 s (4.6 % of events), **≈ 61 s (73.4 %)**, ≈ 93 s (22.0 %).

**Still NOT ESTABLISHED — that this accounts for the capture's 11.56 % loss**, and the reason has
changed. It is no longer "we only have four samples spanning 38–93 s": we now have 5,452 events
and a duty cycle of ≈ 10–12 %, which is much closer to 11.56 % than the second pass's 6–16 %
sketch. But two things block calling it a match:

- the onset→first-pass term is still n = 4. It carries the whole difference between 10 % and 12 %.
- **the arithmetic now leaves almost no room for the overhead we know exists.** The second pass
  argued that some of the 11.56 % is reconnect-and-resubscribe time stacked *on top of* the raw
  outage. If the raw outage is already ≈ 11.5 %, that overhead has nowhere to go. Either the
  overhead is small, or the outage and the reconnect window overlap substantially, or one of the
  two measurements is off. This is recorded as an unreconciled arithmetic note, not a hypothesis;
  it does not affect the causal conclusion, and closing it needs the capture-side accounting, not
  more device measurement.

---

## What this refutes

- **"A GB-scale transaction from some process."** Refuted directly: zero sectors written **and
  zero sectors read** for the whole outage, while the queue reads 100 % busy. There is no
  transaction to attribute.
- **bitcoind as prime suspect.** Over the sampled window bitcoind's `write_bytes` delta was
  **0**; its ~569 MB of `wchar` is syscall traffic (a peer-serving daemon's socket writes), not
  block I/O. It is not the hog, because there is no hog.
- **postgres / journald.** `walwriter` enters D-state *after* the stall begins — a fellow victim.
  journald is a victim badly enough that its own timestamps are collateral damage (§2).
- **Load-driven contention generally.** Device-level `await` was measured healthy all day (D74);
  the stalls are a hard stop, not a queueing effect.
- **A failing SATA link** (this artifact's own first-pass conclusion). Refuted by §4: zero
  link-downs and zero link errors across three boots and 88 days of journal.
- **A marginal cable / connector / power rail.** Not refuted, but **unsupported** — the only
  evidence ever offered for it was the link speed, which §6 shows is an output of the phenomenon
  and which did not change the outage distribution when it fell.

Worth recording plainly: in the sampled window the **largest real disk writer on the box was the
recorder itself** (11.2 MB `write_bytes` / 108 s, ~57 journal commits/s from its per-line
`fsync`). That does not cause the stalls, but it maximises the recorder's exposure to them.

---

## Recommendations — for the operator only. This row took no action.

Ranked. All are box-level and belong to the box's owner; the shared-box fence means the factory
measures and recommends, never acts.

### 1. The one settling test: `mask` the daemon for two cycles, then `unmask`

**This is the only thing missing from the diagnosis, and it must be run as `mask`, not `stop`.**

> **A plain `systemctl stop udisks2` will silently defeat itself.** Verified on the box:
>
> - **There is no `udisks2.socket`.** `systemctl list-unit-files 'udisks*'` returns exactly one
>   unit, `udisks2.service`, and `TriggeredBy=` is empty. The second pass's hedge ("also stop
>   `udisks2.socket` if present") is inapplicable — there is nothing to stop.
> - **Activation is the D-Bus name**, via `/usr/share/dbus-1/system-services/org.freedesktop.UDisks2.service`
>   (`Name=org.freedesktop.UDisks2`, `SystemdService=udisks2.service`).
> - **A live desktop session is holding a client.** `gvfs-udisks2-volume-monitor` runs as PID
>   **22550**, user `operator`, alongside eight other `gvfs` processes (nine in total). Any of them can
>   re-activate `udisksd` over D-Bus within seconds of a `stop`.
>
> The failure mode is the dangerous one: the operator stops the service, `udisksd` is re-activated
> a few seconds later, the stalls continue on schedule, and the *correct* conclusion is recorded
> as **refuted**.

Protocol:

```bash
sudo systemctl mask --now udisks2          # mask, so D-Bus activation cannot bring it back
pgrep -x udisksd                           # must be EMPTY
# watch across two :X2:36 phases (~21 min):
sudo journalctl -kf -o short-monotonic | grep ata32     # expect ZERO link-ups
# and/or poll /proc/diskstats for sda at 0.5 s and expect no 100%-busy/zero-sector window
pgrep -x udisksd                            # re-check: still EMPTY at BOTH phases
sudo systemctl unmask --now udisks2         # expect link-ups to resume at the very next phase
```

- **Verify `pgrep -x udisksd` is empty at both `:X2:36` phases**, not just once at the start —
  that is what distinguishes "the poll stopped" from "the poll was re-activated".
- **Prediction: zero `ata32` link-ups and zero stalls while masked; both resume at the first
  `:X2:36` after `unmask`.** If they continue while `pgrep -x udisksd` is confirmed empty at both
  phases, this diagnosis is wrong and the artifact should be reopened.
- **Nothing else on the box needs udisks2 while it is masked.** The only mount under `/media` is
  an `/etc/fstab` entry (`<redacted mount entry>`) — not udisks-created, not
  controlled by it, and not on `sda`; there are no `/run/media` automounts. Masking unmounts
  nothing. The gvfs processes lose volume-monitor service in the desktop session, which is
  cosmetic on a training server.
- Free, reversible, no reboot, ~25 minutes.
- If confirmed, leaving it masked ends the stalls outright. **Do not reach for "remove
  `libblockdev-smart3`" instead:** `udisks2` is that package's only installed reverse-dependency,
  so removing it would drag `udisks2` out with it — the same outcome as masking, by a more
  destructive route.

### 2. CAUTION — reading SMART is the trigger, not a safe diagnostic

The first pass's top recommendation was `smartctl -a /dev/sda`. §1 and §5 make that the *cause*:
`smartctl` issues the same ATA passthrough that `udisksd` has failed on 5,452 consecutive times.
Two consequences:

- Expect running it to induce a **38–93 s** device stall on a box holding `/`, `/home`, the
  postgres datadir and the live capture. Do not run it casually, and not while the 907 capture
  matters.
- Used deliberately it is a **second** independent test: run `smartctl -a /dev/sda` **off-phase**
  (e.g. at `:X7:00`) and watch whether a stall and an `ata32: SATA link up` appear at that instant
  instead of at `:X2:36`. It is the operator's call and the operator's command; this row will not
  run it. Note it is strictly weaker than Recommendation 1 — it shows a SMART read *can* cause a
  stall, while (1) shows the 600 s stalls *are* this poll — and unlike (1) it is not free.

If it does complete, attribute **199 `UDMA_CRC_Error_Count`** is still the discriminator for the
withdrawn cable theory — §4 predicts near-zero. And firmware **`109R10A0`** on a KINGSTON
SKC600/512GB is worth checking against Kingston's known-issue and update lists: a firmware that
fails every SMART read with `EIO` and hangs the queue for ~61 s doing it is exactly what the
measurements describe.

### 3. Do NOT open the chassis on this evidence

There is no measured physical-layer fault: zero link-downs, zero link errors, and a duration
distribution whose median is identical (30.72 s) at 6.0, 3.0 and 1.5 Gbps. Reseating a cable on a
shared production box, on the strength of a link speed the kernel itself set, would be work done
against a withdrawn inference.

### 4. Treat the root filesystem as at risk regardless

21.5 k command timeouts on the disk that holds `/`, `/home`, the postgres datadir and the capture
is a data-durability problem well beyond this capture, and it stays true whatever the trigger
turns out to be — a drive that fails every status query and hangs its queue for ~61 s doing so is
not a healthy drive even if nothing ever queries it. A verified backup is prudent now, and is a
sensible prerequisite for (1) and (2).

### 5. Do not chase the software suspects from TASK_913's scope 1

Moving bitcoind's datadir, tuning postgres checkpoints or rate-limiting journald will not shift a
SCSI command timeout. Those were the task's prime suspects and they are exonerated. Note this is a
*different* claim from (1): scope 1 hypothesised a process generating **write load**; §5 found a
process issuing a **single passthrough command** that the drive fails. The measurements kill the
first and found the second.

---

## Consequence for the hardening spec (scope item 2)

The hardening is **still right, and the case for it is unchanged in strength — only the named
cause changes.** Dropping per-line `fsync` cannot stop the stalls: they are a device-level
outage and would block any writer that reaches the platter, whatever is triggering them. What it
does is **decouple the recorder's event loop from the device**: buffered `write()` calls return
into page cache immediately, so the loop keeps servicing the websocket across the outage, the
`ping_timeout=15` keepalive does not trip, and frames are not dropped — they are held in RAM and
written out when the device returns.

So the honest framing for `WRITE_HARDENING_SPEC.md` is: **the fix converts a data-loss event into
a latency event.** It should not be sold as removing the stalls. Two limits must be stated: the
outage at the measured ~30 records/s is a small dirty-page burden and rides through easily, but if
dirty-page limits are ever reached the writer blocks anyway; and any record that a policy still
fsyncs synchronously on the loop thread (gap/meta/close) remains fully exposed to the outage.

**The outage figures the spec should quote: typical ≈ 61 s, observed range 38–93 s, and 93 s as
the sizing bound.** The second pass told the spec to read every `~59 s` as "38–93 s (n = 4)";
that is now refined — the *typical* case is ≈ 61 s (the 30.7 s inter-pass mode, 73.4 % of 5,452
events), the 93 s tail is real at 22 % of events, and sizing should continue to use 93 s. The
spec's conclusions are insensitive to this because they size on the worst case, but the typical
figure should be right.

---

## Confidence and limits

The second pass was miscalibrated in **both** directions, and this section is corrected in both.

- **SETTLED — the 600 s actor is `udisksd`'s SMART housekeeping poll.** Upgraded from the second
  pass's "GOOD, one test short of settled", which understated it. The daemon logs its own failure
  **5,452 times out of 5,452 cycles, with zero successes and zero missed cycles**, from 32 s after
  boot, naming the drive; the count matches the kernel's 5,452 two-pass link-up events one for
  one; the poll is caught mid-stall in `blk_execute_rq ← sg_io ← scsi_ioctl` under PID 1946; and
  precedence is established on a single clock at 0.39 s lead, with the outage terminating within
  60 ms of the `EIO` being returned. This is an identification from the actor's own log, not an
  inference from consequences. **The one thing missing is the counterfactual** — nobody has
  stopped the poll and watched the stalls stop. That intervention (Recommendation 1) is the
  operator's action and is **forbidden to the factory by the shared-box standing rule**; it is not
  a gap in the evidence we could have closed.
- **HIGH confidence — no journal hog; every writer blocks together on an absent device.** Four
  independent lines agree: 1 s- and 0.4 s-cadence D-state and `jbd2` sampling; zero sectors read
  *and* written with `ms_doing_io` pinned at 100 %; `scsi_eh` activation; 3/3 on-phase vs 0/3
  off-phase separation. Four directly-sampled stalls by three independent parties.
- **HIGH confidence — the link reset is recovery, not trigger.** Zero link-downs and zero link
  errors in 47,143 `ata32` journal lines across three boots and 88 days; the observed ordering
  (poll → silence → EH → link up → `EIO` → resume) in every sampled window.
- **HIGH confidence — the 600 s cadence is a host timer.** 5,451/5,451 intervals, zero missed,
  37.9 days, 599.998991 s (−1.7 ppm) on the NTP-corrected clock, boot-anchored. The two
  independent fits differ by exactly the printk clock's measured drift.
- **HIGH confidence — link speed is an output, not a cause.** The ladder is logged as libata's own
  reaction; the `EIO` series predates all of it at 6.0 Gbps / UDMA-133 / NCQ-on and has no gaps
  since; the median inter-pass interval is 30.72 s at every speed.
- **MEASURED, n = 5,452 — every event has exactly two EH passes**, with the inter-pass interval
  trimodal at ≈7.5 / ≈30.7 / ≈62 s. The second pass's "one, two or three passes" and its "≈0 s
  mode" are withdrawn; the ≈0 s mode was journald collapse.
- **GOOD confidence — typical outage ≈ 61 s; range 38–93 s.** The inter-pass term is n = 5,452;
  the onset→first-pass term is still **n = 4** (7.4 / 29.8 / 30.4 / 62.5 s, against a 30 s SCSI
  timeout). Duty cycle ≈ 10–12 %. The second pass's "the duration is not rigid" and its 6–16 %
  fuzzed duty cycle were over-read from four hand-samples that over-sampled the tails ~3:1, when
  n = 5,452 was sitting in the log.
- **NOT ESTABLISHED — that the outage duty cycle accounts for the capture's 11.56 % loss.** See
  §8. It is now *closer* (≈10–12 % vs 11.56 %) but the fit has a new problem: it leaves almost no
  room for the reconnect overhead the second pass argued sits on top. Unreconciled; needs
  capture-side accounting, not more device measurement.
- **NOT KNOWN — why the drive fails the passthrough.** A KINGSTON SKC600/512GB on firmware
  `109R10A0` that returns `EIO` on **every** SMART read since boot, hanging its queue ~61 s each
  time, is drive-side behaviour; whether it is a documented firmware defect, and whether ordinary
  I/O is at risk once nothing polls it, is unestablished. Stopping the poll would remove the
  symptom without answering this.
- **NOT KNOWN — whether any physical element is at fault, though it now looks unlikely.** SMART
  was deliberately not read — and §5 makes reading it a *perturbation*, not an observation. There
  is **no** evidence for a cable/connector/backplane/PSU fault; the evidence once offered for it
  has been withdrawn; and a physical-layer fault does not explain a defect that presents only on a
  600 s host timer.
- **NOT MEASURED — the pre-2026-06-05 history.** The persistent journal begins 2026-06-05 and the
  phenomenon is present from its first retained `ata32` line. How much older it is, is unknown.
  *(This replaces the first pass's "the dmesg ring does not reach further back", which was a
  statement about a tool we had not finished using.)*

**On this artifact's own method, plainly.** Its named failure mode — an unverified load-bearing
claim carried forward because checking it looked like scope — bit it three times. Once on the
journal boundary (first pass), once on `dmesg -T` (first pass), and once on `journalctl` realtime
(second pass) — where the document had *already caught that clock lying* and did not re-check it
for the other quantity it was using it to compute. The lesson is narrower than "check your
clocks": **catching an instrument lying about one quantity is not evidence about another.** Name
which clock is valid for which measurement (§2) and label every figure with it.

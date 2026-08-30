# TASK_913 — name the journal hog and spec the recorder's write hardening

**Status**: PROPOSED 2026-08-30 (planner; D74 RULED (a) — no restart, diagnose alongside; most
of the diagnosis was executed planner-side before seeding — this row names the culprit and
ships the hardening SPEC; the code change itself is POST-CAPTURE, since touching the recorder
mid-capture is a restart under D67(b)).
**Size: XS (diagnosis) + S (hardening spec + tests, post-capture). Risk: LOW** — read-only
while the capture runs; the write-path change lands only after the capture closes.

## What is already proven (do not re-derive; planner evidence, 2026-08-30)

1. The capture's ~61.5s `clock_jump` stalls (11.56% of span, D74) are **external ext4 journal
   stalls on sda6 (root), every 600.0s at wall-clock phase `:X2:35–40`**: an independent
   fsync prober on a different file blocked 61.0s at exactly the phase; `jbd2/sda6-8` sits in
   D-state through the windows. Evidence on srv: `/home/operator/fsync_probe_2232.log`,
   `recorder_sample_2242.log`, `dstate_sample_2242.log`.
2. The recorder is a victim via **per-line `flush()`+`fsync()`** (`capture.py:303-306`): it
   lives in `jbd2_log_wait_commit` continuously; when the journal stalls, emission stalls,
   frames buffer in the kernel, and the recovery usually trips the websocket keepalive
   (`ping_timeout=15`) — the 1011 disconnect/reconnect cycles are this defect downstream.
3. NOT the cause (all measured): srv clock (venue-offset median 0.13s stable), reconnect
   backoff (0.5% overlap), machine-wide freeze (dashboard serves through stall windows),
   device-level disk saturation (`sda` await ≤ ~4ms all day; sda is an SSD).
4. Box inventory relevant to the hunt: **bitcoind (snap — datadir on root), PostgreSQL
   (walwriter seen in D), a python dashboard polling bitcoind**, sysstat, journald. Root is
   66% full; the nvme is a separate application volume (`/mnt/app`, own journal,
   unaffected by these stalls).

## Scope

1. **NAME the journal hog** (read-only, can run while the capture lives): through one or two
   stall windows (`:X2:35`), sample `ps` D-state with wchan + `/proc/<pid>/io` write_byte
   deltas + `jbd2` stats (`/proc/fs/jbd2/sda6-8/info`), and identify the process pushing the
   GB-scale transaction (a 61.5s commit on an SSD is a big write, not device contention).
   Prime suspects: bitcoind's periodic persistence; a postgres checkpoint if its datadir is on
   root; journald flush. The artifact names the process with measurements, and states whether
   it can be moved, tuned, or rescheduled WITHOUT touching the capture.
2. **Spec the recorder write hardening for FUTURE captures** (the code lands post-capture):
   - PRIMARY (device-independent): **drop per-line fsync.** Buffered writes; fsync at the
     moments durability actually matters — gap/meta records and terminal close; a periodic
     (e.g. 30s) fsync otherwise. Per-event durability is not a requirement: the reader already
     tolerates a torn tail line (912's fixture proves it) and a crash loses ≤ the tail window.
     The spec must state the bounded-loss argument explicitly.
   - SECONDARY (operator's call, recorded not defaulted): `--out` on a device whose journal
     the hog does not share — the application volume has 397G free but is the operator's
     application volume; do not silently write there.
   - The `clock_jump` gap reason is RENAMED to an honest emission-silence label, and the
     detector's claim is re-worded to what it measures (`now - last_emission_ts`).
3. **Tests** (post-capture): a fixture proving per-event durability is not required for reader
   correctness (torn tail tolerated — 912 already carries this; reference it); a test pinning
   the new fsync policy (gap/meta records are durable, event lines are not each fsynced).

## Out of scope

Any change to the running capture or recorder process (D67(b): restart = new capture; D74(a)
ruled no restart). Relabeling gaps in the EXISTING capture file (the artifact may re-label
at READ time; the bytes on disk are the record). Fixing bitcoind/postgres placement (that is
the operator's box-level call; this row only measures and recommends).

## Acceptance

- (i) The hog is NAMED with measurement (process, bytes-per-window, cadence), and the artifact
  states the remediation options at box level.
- (ii) The hardening spec lands with the bounded-loss argument and the fsync-policy pin;
  no recorder code changes before the capture closes.
- (iii) The 912 health probe's verdict logic is cross-checked against the new policy (a
  batched writer must not trip the split-feed detector by design).

# Factory run 47 — 2026-08-30

## Startup incident: git was not a repository

`git status` failed with "not a repository". Cause: `.git/objects` is a symlink to
`/run/mnt/storage/Trader_versionning/git/objects` and `storage disk` was not mounted
(no fstab entry). Fix: `udisksctl mount -b /dev/sdb1`. Operator confirmed. This is the
documented failure mode; it costs a run start every time the box reboots.

## Shipped

TASK_912 — capture health probe. `experiments/907_live_fill_probe/health_probe.py` +
`test_health_probe.py`. Stdlib-only, read-only-by-copy, verdict PASS/WARN/FAIL.

## Verification record (2 cycles, 6 verification passes)

Cycle 1 panel (correctness / fail-closed / flow-verifier) FAILED the change:
- BLOCKER, found INDEPENDENTLY by two lenses and reproduced first-hand by the orchestrator:
  a pair declared in `meta_start.pairs` with zero observed events was absent from `per_pair`
  entirely, so it could never FAIL -> verdict PASS. That is precisely the failure the row
  exists to catch. Root cause worth remembering: the probe is stdlib-only BY DESIGN, so it
  reimplements capture parsing from scratch, and in doing so dropped the declared-vs-observed
  defense that the sibling `test_declared_pairs.py` already encodes. Every fixture happened
  to declare exactly the pairs it emitted, so the suite was green.
- MAJOR: `gap_histogram` merged all pairs and all event types into ONE timeline, so any outage
  short of every-channel-at-once counted as zero gaps. A real 150s single-pair outage scored
  `{gt_5s:0,...}` and PASS.
- MAJOR: absent `counters` record read as verified-clean, same reason string as a real zero.
- MINOR: `--watch` log append outside the try/except killed the unattended loop on a write hiccup.
- MINOR: no fixture had a silent declared pair or a native `gap` record.

Fix pass ran on the escalated model (opus) per `.dark-factory/config.json` models policy.
All five closed, each with a regression test verified RED before / green after.
Suite 223 -> 235 passed (+12), `trader selftest` 12/12.

## Acceptance, closed first-hand by the orchestrator against the LIVE capture

- (i) Probe run against the real 744MB capture on gpu-server via copy: PASS in ~27s,
  per-pair ages sub-second, `schema_mismatch_count 0`, `counters_present true`,
  `declared_pairs` = the four subscribed pairs, `malformed 0`.
- (ii) 235 passed.
- (iii) `--watch` survived two full cycles on srv, exit 0, two independently-parseable
  appended JSON lines, `events_total`/`file_bytes` both advancing.

Cross-validation of the gap_histogram rewrite: the fixed probe reported
`n_gap_records=1094, >5s=433, >60s=364, >600s=0, max=95.2777 (clock_jump)`.
An INDEPENDENT hand count by the orchestrator ten minutes earlier gave
`1092 / 431 / 362 / 0 / 95.3`. The +2/+2/+2 delta is exactly the two gaps accrued in between.
Two methods, same answer.

Known, accepted, NOT fixed: the probe does two full `readlines()` per analysis (pre-existing),
so RSS is ~1.06 GB on a 747MB capture. Fine on srv today; it scales with the capture and
violates the spirit of the memory-hierarchy rule. Candidate XS row: stream both passes.


## Cycle 2/3 (added after the draft above)

Cycle 2 FAILED too. Both lenses converged independently on the module's ONLY unvalidated field,
`int(last_counters.get("schema_mismatch", 0))`: `int(0.9)` truncated a real mismatch to 0 and the
verdict then asserted "schema_mismatch verified 0" (orchestrator-reproduced); `int("corrupt")`
crashed one-shot mode, which is the path acceptance (i) names. Correctness added a MAJOR: the
reasons list truncated at the first FAIL, hiding concurrent WARN facts, while the malformed-records
check two lines below was not gated the same way. flow-verifier came back CLEAN.

Cycle 3 (opus) closed all three; 235 -> 255 passed, RED-first verified (17 failed pre-fix).
Orchestrator re-ran both of its own repros against the final tree: both now FAIL correctly.

Process note worth keeping: EVERY cycle-1 and cycle-2 pass found something the implementing
agent's own report had not mentioned, and the two BLOCKERs were both "probe returns PASS on a sick
capture" — the failure mode a health probe must never have. Reproducing each BLOCKER first-hand
before accepting it was worth the time; it also caught that my first adjacency test for the capture
finding measured the wrong interval edge.

---

# The out-of-band finding (seeded as D74)

# Run 47 out-of-band finding — the live 907 capture has a rate-dependent receipt-timing defect

## Measured (first-hand, read-only, on gpu-server, capture_907_main.jsonl @ 742MB / 61.65h span)

- Gap union (overlap-deduped): 25,658 s = 7.13 h = **11.56% of span**.
  Queue/handoff records "~1.4% cumulative loss" — off by ~8x.
- Gap seconds by reason: clock_jump 23,824 s (6.62 h) | disconnect 1,670 s (0.46 h)
  | heartbeat_timeout 286 s. The dominant reason is NOT disconnects.
- 397 clock_jump gaps, 686 disconnects, 9 heartbeat_timeouts. Max gap 95.3 s; none >600 s.
- Local-vs-venue clock offset on 354,164 trades (venue_ts is an INDEPENDENT clock):
  min -0.06 | p01 -0.02 | **p50 0.13** | **p99 61.40** | max 135.36 s.
- Per-hour: p50 offset is 0.02-0.17 s in quiet hours but rises to **2.45 / 8.14 / 5.62 / 8.73 s
  in hours 8-11**, exactly the hours where trade count spikes (10,994 / 22,015 / 13,557 / 14,413).
  p99 is ~60 s in EVERY one of the 61 hours.
- Across a clock_jump window: local elapsed 126.00 s vs venue elapsed 1.12 s (worst case).

## Established by measurement (each an independent elimination)

1. **Loss is 11.56% of span, not the recorded ~1.4%.** Overlap-deduped gap union 7.13 h / 61.65 h.
2. **`clock_jump` is the dominant reason** — 6.62 h of the 7.13 h; disconnects only 0.46 h.
3. **NOT an srv clock step.** vs the INDEPENDENT venue clock (`venue_ts`), median local-venue
   offset is 0.13 s and stable across all 61 h. An NTP step shifts the median and persists.
   => the `clock_jump` LABEL is a misdiagnosis. `recorder.py:674-677` fires it on
   `now - _last_event_ts > heartbeat_timeout_s` (30 s), and `_last_event_ts` (recorder.py:797)
   updates only when an event is EMITTED — so ANY emission stall is mislabeled a clock jump.
4. **NOT reconnect/backoff**, despite `backoff_max_s = 60.0` matching the duration almost exactly:
   clock_jump gaps overlap a disconnect gap in 0.5% of cases and CONTAIN one in 0.0%.
5. **NOT load/throughput driven.** p99 offset is ~60 s in EVERY hour — identical in a 2,892-trade
   hour and a 22,015-trade hour. 397 occurrences, ~6.4/h.
6. **Near-deterministic duration**: p10 60.7 / p50 61.5 / p90 62.1 s. A timer, not contention.
7. **Frames keep arriving throughout the window.** recv() has a 30 s timeout and would raise
   `heartbeat_timeout` if the socket went quiet — only 9 of those fired in 61 h. So for ~61.5 s
   the recorder is receiving frames and emitting NO event, with `book_keepalive_s = 5.0`
   supposedly forcing an emission every 5 s per pair.

## THE OPEN QUESTION (sharp, and the whole value of this finding)

What stops event EMISSION for a near-deterministic ~61.5 s, 397 times, while frames flow and a
5 s keepalive is armed? Every cheap external explanation is eliminated above. Remaining suspects
are all inside the parse/emit path (`recorder.py` `_consume` -> parse -> `_emit`) and none is
confirmed. Note the recorder's own counters read json_decode_errors=0, schema_mismatch=0,
unrecognized_frames=0 — so the frames are not being rejected as malformed either, which makes
the silence harder to explain, not easier.

## A SECOND, DISTINCT effect (do not conflate with the above)

Median receipt lag degrades from ~0.02 s to **2.45 / 8.14 / 5.62 / 8.73 s in hours 8-11**, the
four highest-volume hours (10,994-22,015 trades/h). This one IS load-dependent and is consistent
with `capture.py:303-306` — `_write()` does `flush()` + `os.fsync()` on EVERY line inside an
asyncio recorder, so a blocking fsync stalls the event loop including `ws.recv()`. Plausible,
NOT proven; srv `sda` showed w_await 2.86 ms / %util 10.2 / load 0.78 during a quiet hour, which
neither confirms nor refutes peak-hour behaviour.

## Why it is decision-changing for 907B

capture.py calls `clock_basis == "local_receipt"` load-bearing and "decided once". 907B's
estimand-1 is book-mid markouts timed on exactly that basis. ~1% of events per hour carry
receipts 60 s+ late, and in peak hours the MEDIAN event is 2.4-8.7 s late, against markout
horizons denominated in seconds-to-minutes.

## Scope discipline

NOT fixed in this run. TASK_912 explicitly forbids recorder changes and the capture is live;
D67(b) makes a restart a NEW capture. This is an operator call -> seed a DECISIONS row.

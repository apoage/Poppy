# TASK_914 — streaming reader for the 907 capture format

**Status**: PROPOSED 2026-09-01 (planner; seeded from run 49's flag: the reader loads the whole
file — **11.8 GB RSS for a 1.24 GB capture**; a 7-day file implies ~24 GB against the 32 G
safe_run cap).
**Size: S. Risk: LOW** — pure reader refactor; every gate is byte-identity of derived results.

## Why

`read_capture` materializes the full JSONL (events + gaps + metadata lists) before any analysis.
The live capture closes at ~1.45 GB (~14 GB RSS — under the cap, so the FINAL analysis of THIS
capture is not blocked), but any future capture longer than ~2.7 GB of file breaches 32 G, and
B2's book-freshness pass wants per-pair event streams it currently gets as giant lists.

## Scope

1. Read the capture INCREMENTALLY: line-by-line JSON decode, one pass, O(1) memory per
   aggregation (counters, gap spans, per-pair book/trade timestamps kept as needed — the
   estimand passes are streamable by construction: markouts need the fill candidates and a
   mid lookup; a bounded deque per pair suffices).
2. **Byte-identity gate**: the streaming reader produces byte-identical artifacts to the
   current reader on (a) the run-46/47 fixture captures and (b) the closed live capture —
   NO NUMBER MOVES (B-iv's proof shape).
3. RSS is measured in the artifact (peak RSS for the closed capture stated in the run log);
   the gate is peak < 4 GB for the 1.45 GB file.

## Out of scope

Any change to the recorder, the capture format, or the estimand definitions. (The gap-record
ts-sampling fixes from B1 are shipped and are the reader's contract.)

## Acceptance

- (i) Byte-identical artifacts on fixtures + the closed live capture.
- (ii) Peak RSS < 4 GB on the closed capture, measured and stated.
- (iii) The suite (302) green; the reader's own tests gain a synthetic large-capture case
  that would OOM under the old reader's scaling (constructible cheaply with a sparse stream).

# TASK_912 — capture health probe (the T+1h span check, made into tooling)

**Status**: PROPOSED 2026-08-29 (planner; seeded from run-46 ask #3 — D67(b)'s span check was
assigned to a planner and there is no operator on the server, so capture health currently
depends on a human remembering to ssh in).
**Size: XS/S. Risk: LOW** — read-only by construction; its one failure mode (drifting into
estimand territory) is fenced by scope below.

## Why this row exists

The 907 capture is a multi-day unattended recording whose value is concentrated in exactly the
failure mode it cannot report itself: **one pair's book feed dying while the others keep
flowing.** That is the realistic Kraken failure (run-46 measured it: a book subscription dying
at t=1800 while trades continue reads `uncovered_fraction = 0.0000`) and it is invisible to a
naive "file is growing" check, because the file IS growing — on the healthy pairs. Today it
would be caught days later, by the analysis, as a contaminated estimand. The probe's job is to
catch it within minutes, from outside the recorder process.

## Scope

1. **A read-only probe that works from a COPY.** `health_probe.py --capture <path>` first
   byte-copies the in-progress file (local `cp`, or `ssh gpu-server 'cat …'` into a local temp —
   the copy step is part of the tool, never the caller's job), then parses only the copy. It
   must never open the live file for analysis and never write anywhere under `runs/` except its
   own output path. Stdlib only — no project imports, no numpy/jax (it runs on srv in whatever
   python3 is present).
2. **Emits one JSON object + a human summary** with exactly:
   - `wall_now`, `file_bytes`, `capture_id` (from the run header if present; else absent/null —
     no guessing);
   - `events_total`, `schema_mismatch_count` (Addendum A's invariant: emitted top-of-book must
     reconcile with the CRC-verified state — a nonzero count here is FAIL, full stop);
   - `gap_histogram`: count of inter-event gaps >5s, >60s, >600s, and the max gap with its
     wall-clock timestamp;
   - per pair: `n_book`, `n_trade`, `last_book_age_s`, `last_trade_age_s`;
   - a `verdict` field: **PASS / WARN / FAIL** with reasons.
3. **Verdict rules (fail-closed, conservative):**
   - FAIL if any pair's `last_book_age_s` exceeds 300s while the capture's newest event of ANY
     type is under 60s old — this is the split-feed death detector, the reason the row exists;
   - FAIL if `schema_mismatch_count` > 0;
   - WARN if max gap > 600s or any pair's `last_trade_age_s` > 600s during capture hours;
   - PASS otherwise. A stale-everything capture (all feeds old — recorder dead) is FAIL with
     the recorder-dead reason, not the split-feed reason; distinguish them in the reason string.
4. **A loop runner for unattended use**: `--watch --interval 900` appends one JSON line per
   interval to a health log under `runs/907_live_fill_probe/health/` (that directory is the
   probe's ONLY write target), so the whole capture's health history is one `grep FAIL` away.
   Launch under tmux on srv alongside the recorder.
5. **Tests** (fixture captures, synthesized — no live dependency): a healthy capture passes; a
   capture with one pair's book feed dead and the rest flowing FAILS with the split-feed reason
   (this test IS the row — it encodes run-46's measured failure); a recorder-dead capture FAILs
   with the recorder-dead reason; a torn last line (copy taken mid-append) is tolerated without
   a schema-mismatch false positive.

## Out of scope

Anything estimand-shaped: no markouts, no fill modeling, no coverage fractions, no mid-age
statistics beyond the split-feed detector. Those are 907B's job and this probe must not grow a
second opinion about analysis. No writes to the live capture, no recorder changes, no restarts
(the probe observes; recovery is a human/planner decision — D67(b)'s "restarts are a new
capture" rule stands).

## Acceptance

- (i) Probe run against the LIVE capture (via copy) reports its current true state — cross-
  checked by hand once: event counts and per-pair ages match an independent `tail`/`wc` pass.
- (ii) The four fixture tests pass, including the split-feed death detector.
- (iii) The `--watch` loop survives one full interval cycle on srv writing valid JSON lines.

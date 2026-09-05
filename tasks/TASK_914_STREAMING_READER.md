> **STATUS (2026-09-05): HALTED at the cycle limit (run 53) — NOT because the code is wrong.**
> The reader itself is **CLEAN and independently verified**: correctness found nothing across 16
> resume-path fixtures, 11 read_capture fixtures and 400 fuzz trials; the flow-verifier proved
> **artifact byte-identity on the real 1.44 GB capture** and **selftest 12/12** full-repo; the
> regression lens independently re-derived field-level identity and proved it non-vacuous by
> mutation. Measured **peak RSS 12.34 GB -> 1.46 GB** (8.44x on CPython 3.14, 10.42x on 3.10 —
> the spread is the interpreter, both figures reproduce).
>
> **Two MAJORs remain, BOTH confined to the new `test_streaming_reader.py`, none in `capture.py`:**
> (1) `:162` the child-baseline ceiling is ~5x above the point where deflation eats the bound —
> a real duplicated full-file pass PASSES at 154-166 B/event under a 58-64 MB child baseline;
> (2) `:332-342` the structural guard patches only `builtins.open`, but **`io.open` IS
> `builtins.open`**, so a bypassing second traversal certifies `traversals=1.000`.
>
> Cycle budget spent 3/3 (MAJOR -> BLOCKER -> 2 MAJOR). **D76's closing-pass exception does not
> apply** — closing (2) needs a new mechanism, i.e. a new solution attempt. **Gated on D82**
> (three scope options, operator's call). Work preserved:
> `lore/handoff_2026-09-05_factory_run_53_914_halt.patch` (sha256 `a2577008...`, apply-check OK
> vs `5010a1b`). Full record: `lore/handoff_2026-09-05_factory_run_53.md`.
>
> **Corrected in-flight (do not re-inherit the old numbers):** the spec's *"O(1) memory per
> aggregation"* is **FALSE** — `all_ts`/`data_ts` are irreducibly O(events); growth is **LINEAR
> in events**, measured, and only the slope moved. The `<4 GB` gate is therefore ONE POINT on a
> line that recrosses it at roughly a 4 GiB capture (~12.1 days at the measured wall-clock rate);
> the 32 GiB safe_run cap moves from ~9.3 days out to ~97 days. Acceptance (iii)'s "302" is the
> correct HEAD baseline; the suite is 304 with both new tests.

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

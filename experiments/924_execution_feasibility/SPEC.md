# TASK_924 SPEC — execution-feasibility instrument over the closed 907 capture

Pre-registered 2026-09-28, BEFORE the first full-capture pass (G4). This file's own sha256,
recorded in `results.json` as `spec_sha256`, is written into `results.json` before the full
pass begins; any output added after seeing full-pass data is labelled POST-HOC in
`RESULTS.md` and excluded from the go/no-go table.

## 0. The wall (D81, binding)

This instrument IMPORTS ONLY from `experiments/907_live_fill_probe/capture.py`
(`read_capture`, `_iter_lines_tolerant`, `GapRecord`, `BAR_INTERVAL_S`,
`STALE_BOOK_MIN_RUN_S`, `DEFAULT_COVERAGE_WINDOW_S`, `DEFAULT_GAP_DETECT_THRESHOLD_S`,
`DEFAULT_MAX_UNCOVERED_FRACTION`). Nothing under `experiments/907_live_fill_probe/` is
modified. `read_capture()` is called WITHOUT `assert_complete()` — this instrument reads the
reconciled `CaptureMetadata` (gaps, per-pair coverage, stale-run seconds) as data; it never
asserts the capture "complete" and never re-litigates 907's markout gate. No DB access, no
network, no exchange API, read-only over the capture file.

## 1. Inputs

- Capture: `/home/operator/Trader/runs/907_live_fill_probe/capture_907_main.jsonl` on
  `gpu-server`, read in place, read-only. Recorded before the full pass (G2):
  size, mtime, sha256 — see `results.json.g2_before`.
- Smoke/fixture inputs for pre-full-pass testing: `runs/907_live_fill_probe/smoke_60s.jsonl`
  and `resmoke_75s.jsonl` (same server dir, read-only) plus synthetic JSONL built in this
  instrument's own test file (`test_instrument.py`), never written into the 907 tree.
- 907 capture module: imported from a workstation-HEAD copy of `capture.py` staged
  read-only in the execution scratch dir (`/var/tmp/924-implementer/` on gpu-server). The
  server's own `~/Trader` checkout is stale (856e6ad) and is not imported from.

## 2. Definitions (each stated in the quantity its consequence line consumes — D81(c))

### 2.1 Capture span and accounted seconds (REUSED from 907, never re-derived)

`read_capture(path)` returns `CaptureMetadata`. Its recorded gaps
(`meta.gaps`, coalesced by 907's own `_merge_gap_spans` into `meta.merged_gap_spans`) and its
capture span (`meta.start_ts`, `meta.end_ts`) are read off `meta` as-is. For every pair,
`meta.coverage_by_pair[pair].accounted_s` (= capture span minus the union of recorded gaps,
GLOBAL — identical across pairs by construction, since `PairCoverage` is denominated over the
capture's whole span, not the pair's own data span — see `capture.py:1072-1084`) is the
denominator this SPEC uses everywhere below described as "accounted seconds", UNLESS stated
otherwise. This is 907's own number, reused verbatim — the instrument never recomputes the
gap ledger.

### 2.2 G1 — frozen-mid reproduction (BLOCKER if it fails)

B1's shipped definition (`capture.py:186-232`): a run of CONSECUTIVE IDENTICAL `(bid, ask)`
book-price-pairs for one pair, lasting longer than `STALE_BOOK_MIN_RUN_S` (60.0s, one bar
interval), counts as a frozen-price run; `stale_book_s` sums such runs' seconds (each run's
FIRST event is a real observation and is excluded from its own run duration count exactly as
`capture.py:_close_book_run` does). The reported quantity is
`stale_book_s / accounted_s` for BTC/USD, a TIME fraction over ACCOUNTED (non-gap) seconds.

Computed two ways, both required to agree before the full pass is trusted, and BOTH computed
on the smoke/synthetic fixtures pre-registration and on the full capture:

  - **(i) shipped**: `meta.coverage_by_pair["BTC/USD"].stale_book_s` /
    `meta.coverage_by_pair["BTC/USD"].accounted_s`, read directly off `read_capture()`'s
    output — no reimplementation, the number 907 itself produced.
  - **(ii) independent**: this instrument's OWN streaming run-length tracker
    (`_RunTracker` in `instrument.py`), which re-derives run boundaries from the raw
    `(bid, ask)` stream itself (same rule: consecutive-identical run > 60s, first event of a
    run excluded), and its own `accounted_s` computed by subtracting `meta.merged_gap_spans`
    (REUSED, not re-derived) from `meta.start_ts`/`meta.end_ts`. This is a genuinely separate
    code path from 907's `_note_book_top`/`_close_book_run`/`_pair_coverage`, sharing only the
    gap ledger and the span, so it catches a logic bug either implementation might carry.

**Tolerance** (stated and justified BEFORE seeing the full-pass numbers): the two
BTC/USD fractions, and the two `accounted_s` values, must each agree within
`max(60.0s, 0.001 * accounted_s)` absolute seconds. Justification: the two implementations
sum durations in different traversal/bookkeeping order over ~2.28M BTC book events; the only
expected source of disagreement is IEEE-754 summation-order noise, which is many orders of
magnitude below one second at this event count. The bound is set at one bar interval (60s,
907's own resolution floor) OR 0.1% of the ~273,233s accounted span (≈273s), whichever is
larger — generous enough to absorb floating-point noise, tight enough that a real
run-detection or gap-handling bug (which would misclassify a material fraction of ~2.28M
events) trips it. A mismatch beyond this bound is a BLOCKER per the task file — reported, not
tuned away.

The reproduction target from the shipped B1 measurement
(`capture.py:213-214`, comment, not re-derived): `127,071s / 273,233s = 46.48%` for BTC/USD.

**Precondition discovered while writing the pre-registration tests (`test_instrument.py`),
stated here because it is load-bearing for G1**: the shipped `PairCoverage.stale_book_s` is
UNCONDITIONALLY `0.0` for any capture whose `meta_start` does not declare BOTH
`book_emission_policy` and `book_keepalive_s` (`capture.py:2143-2146`,
`declares_keepalive` gate). `capture_907_main.jsonl`'s own `meta_start` (verified directly,
first line of the file on gpu-server) declares
`"book_emission_policy":"change_or_keepalive","book_keepalive_s":5.0`, so the shipped number
is live for the real capture. The instrument's OWN independent `RunTracker` (2.2(ii)) carries
no such gate — it always tracks raw `(bid, ask)` runs — so the two implementations are only
comparable on a capture that declares the policy; `test_instrument.py`'s synthetic fixtures
declare it explicitly for this reason, and `smoke_60s.jsonl`/`resmoke_75s.jsonl` are checked
for the same declaration before their G1 numbers are read as meaningful (a `0.0` from either
implementation on a capture that does NOT declare the policy is not a cross-check failure).

### 2.3 Item 1 — rel-spread surface (EVENT-weighted; stated and justified)

For every `book` row (bid/ask price always present in this format —
`CaptureWriter.write_book` requires both, `capture.py:788-796` — so no row is ever "book
present, price missing"), `rel_half_spread_bp = (ask - bid) / 2 / mid * 1e4` where
`mid = (bid + ask) / 2`. Bucketed by pair and hour-of-day (UTC, `0..23`, `datetime.utcfromtimestamp(ts).hour`
— a DIURNAL view pooling all 3.586 days, distinct from item 5's sequential clock-hour
buckets). Reported: median and IQR (25/75) per (pair, hour-of-day) cell.

This is EVENT-weighted, not time-weighted, and stated as such: each row is one sample
regardless of how long that quote persisted before the next update, so a pair whose book
updates faster in some hours than others will have those hours' rows over-represented
relative to wall-clock time. The natural per-row summary answers "what did an observer
sampling a live tick of this feed typically see", not "what fraction of wall-clock time carried
this spread" (item 5's AS(1h) answers that, time-weighted). Rows are drawn only from time the
recorder was connected (gap time never carries an event row), so no separate gap-exclusion
step is needed for this item; the "D74 gap-exclusion" language in the task item is satisfied
structurally.

### 2.4 Item 2 — mid-staleness (same definition as G1, reused directly)

Per pair: frozen-mid fraction = `stale_book_s / accounted_s` from 2.2(i). Streak-duration
distribution = the set of individual run durations `> STALE_BOOK_MIN_RUN_S` collected by the
SAME independent `_RunTracker` used for 2.2(ii) (median/p95/max reported).

**Both B1 disclosed limits are inherited verbatim and restated in `RESULTS.md`
(D81(c)/task instruction):**
  1. The run key is `(bid, ask)` only. A literal repeat can be either (a) a keepalive timer
     firing with no wire frame behind it, or (b) a real wire frame in which only the SIZES
     moved. This reader (and this instrument, which reuses the same key) cannot distinguish
     them; "frozen" means "the mid carried no new information", not "the wire was not
     observed".
  2. The detector counts ONE shape: a run of CONSECUTIVE IDENTICAL `(bid, ask)`. Any change,
     including a change BACK, ends the run. A mid that OSCILLATES over a small set of values
     without ever discovering a new price is NOT counted by this measure and is NOT detected
     by this instrument, exactly as it is not detected anywhere in B1
     (`capture.py:217-231`). `test_instrument.py` plants this shape and asserts it reads
     zero, carrying the same disclosure forward as a passing regression test rather than a
     claim.

### 2.5 Item 3 — book presence (EVENT-ROW denominated, stated explicitly per D81(c))

Per pair, over ALL rows of that pair (book + trade), each row classified once:
  - `two_sided`: `type == "book"` and `bid_sz > 0` and `ask_sz > 0`.
  - `one_sided`: `type == "book"` and exactly one of `bid_sz > 0` / `ask_sz > 0` (a
    price level with zero displayed size on one side — the schema always carries both price
    fields, so "one-sided" here means "no depth on one side", not "one price field absent").
  - `print_only`: `type == "trade"`.
  Denominator = count of (book + trade) rows for that pair. This is DELIBERATELY
  event-row-denominated, distinct from the TIME-denominated "two-sided book present" fraction
  below (2.6), because 907 defines no prior convention for row classification and the task's
  own item 3 language ("fraction of event rows") names rows as the unit. The
  TIME-denominated two-sided fraction — the one the go/no-go table and the "< ~50% of active
  time" consequence line actually consume — is reported separately (2.6), never conflated
  with this row fraction.
  Depth-at-best sample statistics: median/p95 of `bid_sz`, `ask_sz` per pair, over `two_sided`
  rows only.

### 2.6 Two-sided book presence (TIME-weighted; the quantity the go/no-go table consumes)

Per pair: the book state (last known `bid_sz > 0 and ask_sz > 0`) is forward-filled from each
book event to the pair's NEXT book event, clipped against `meta.merged_gap_spans` (REUSED,
not re-derived — gap seconds are never attributed to any state). The sum of clipped,
two-sided-flagged seconds is the numerator; the denominator is
`meta.coverage_by_pair[pair].accounted_s` (2.1) — the SAME denominator 907 itself uses,
reused directly rather than recomputed a second way, so this fraction is directly comparable
to 907's own coverage numbers. Time before a pair's first event or after its last event
within the capture span is NOT attributed to any state (contributes 0 to the numerator,
already included in the shared denominator) — this is the conservative direction: a pair that
comes online late or goes dead early is measured as LESS covered, not excluded from the
denominator, matching 907's own run-45 rationale (`capture.py:1072-1084`).

### 2.7 Item 4 — quote-update rate and survival

`quote_updates_per_active_minute = n_book_events[pair] / (accounted_s / 60)`.
"How long a stale quote persists" = the distribution of ALL `_RunTracker` run durations for
that pair (not filtered to `> 60s` — every run, including sub-threshold ones), i.e. the time
between one `(bid, ask)` change and the next. Reported as a survival-style summary: median,
p75, p90, p95, max run duration per pair. This is the same run-tracking pass as 2.2(ii)/2.4,
reused rather than a third implementation.

### 2.8 Item 5 — AS(1h), pre-registered BEFORE the full pass (binding form)

The task's structural correction is adopted: the longest gap-free stretch in this capture is
561.8s (907 record), so "AS(1h) over 1h windows requiring a whole gap-free hour" is
structurally 0-sample and is NOT what is computed.

**Binding (time-weighted) form**: per pair, per elapsed CLOCK-HOUR bucket
`k = floor((ts - meta.start_ts) / 3600)` (sequential buckets across the whole capture
timeline — distinct from item 1's hour-OF-DAY buckets), the time-weighted mean of
`rel_half_spread_bp` over seconds that are (a) accounted (not inside a recorded gap — 2.1) AND
(b) two-sided-book-present (2.6), using the SAME forward-filled state as 2.6. Each bucket also
reports its OWN covered-seconds fraction: `(accounted-and-two-sided seconds in this bucket) /
3600`, so a bucket built from a thin sliver of real coverage is visible next to its number
rather than presented as equally strong as a fully-covered hour.

**Non-binding (event-weighted) form, reported alongside, explicitly labelled**: per pair per
same bucket, the unweighted mean of `rel_half_spread_bp` over `two_sided` book ROWS whose `ts`
falls in that bucket (2.5's row definition, restricted to `two_sided` rows). `RESULTS.md`
states the size of the time-weighted-vs-event-weighted difference per pair (pooled over all
buckets) — expected to be material given ~88% of BTC book events repeat the prior `(bid, ask)`
(`capture.py:213`), so event-weighting over-samples whatever spread happened to be quoted at
high update-frequency moments while time-weighting reflects wall-clock exposure.

Reported against the **≤ 2 bp** Phase-B gate line, labelled DESCRIPTIVE ONLY, exactly as the
task file states: it bounds what a passive fill COULD cost; it is not 907's markout gate and
does not by itself unlock Phase B.

## 3. Gates (binary, per task file)

- **G1 REPRO**: 2.2, both implementations agree with each other within tolerance, and (i)
  agrees with the shipped B1 comment number (46.48%) within the same tolerance. Mismatch =
  BLOCKER.
- **G2 READ-ONLY PROOF**: capture file sha256 + size + mtime recorded before AND after the
  full pass, asserted byte-identical, in `results.json.g2_before` / `g2_after`.
- **G3 RSS**: peak RSS (VmHWM of the measuring process, sampled the same way 914's own guard
  samples it — see `test_streaming_reader.py`'s rationale for `VmHWM` over `ru_maxrss`)
  strictly below 4 GiB through the full pass, recorded in `results.json.g3_peak_rss_bytes`.
- **G4 PRE-REGISTRATION**: this file's sha256 recorded in `results.json.spec_sha256` before
  the first full-capture pass starts (`results.json.spec_registered_before_full_pass: true`,
  with the wall-clock timestamps of both events proving the order).

## 4. Go/no-go table (pre-committed, from the task file, restated here verbatim in substance)

- Two-sided book present (2.6, time-weighted) `< ~50%` of active time across pairs, OR AS(1h)
  descriptive (2.8, time-weighted) `>> 2bp` everywhere → Option B's maker premise is
  measured-dead on this venue; recomposition record says so, B closes to "taker-only
  planner" (the 916 arithmetic already prices this).
- Otherwise → B's build row (planner-as-executor spec) is unblocked WITH numbers, and the next
  row cites this artifact's tables.
- The ≤2bp AS(1h) line is DESCRIPTIVE ONLY, per the task file — no "unlocks Phase B" language
  is used anywhere in this artifact regardless of which side of the line the numbers land.

## 5. Tests (`test_instrument.py`, run on gpu-server, never bare on the workstation)

Synthetic JSONL built entirely in the test file (never touching the 907 tree):
  - gap handling: a recorded `gap` record's seconds must not appear in `accounted_s` nor in
    any numerator built from state that would otherwise forward-fill across it.
  - two-sided time-weighting: a planted sequence of book events with known inter-event
    gaps produces the hand-computed two-sided-seconds fraction.
  - frozen-run detection agreeing with the shipped definition on a planted `>60s` identical
    `(bid, ask)` run — cross-checked against 907's own `read_capture()` on the same synthetic
    file (imported, not reimplemented, for this one assertion) so the test itself is an
    instance of gate G1's cross-check, not just the module code.
  - a planted OSCILLATING mid (alternating between two `(bid, ask)` values for the whole
    file) that the frozen detector does NOT flag — asserting the disclosed blind spot
    (2.4.2) stays disclosed and does not silently start passing.

## ERRATA (POST-HOC, added AFTER the full pass — sections 0-6 above are UNCHANGED, pre-registered text)

**Recorded**: SPEC sha256 BEFORE this errata: `56212530e6f90958c5dbc2a97e025ade939efded1374575cdeb9292dd4fa7c72`
(the sha `results.json` cites for every pre-registration event — smoke, resmoke, and the
BLOCKER-triggering full pass on `capture_907_main.jsonl`, all run before this section existed).
SPEC sha256 AFTER this errata is recorded in `results.json.spec_sha256_after_errata`.

**What happened**: the full pass (`capture_907_main.jsonl`, 1,440,479,339 bytes, sha256
`8b3f748dfc5f8a06a22d7484ae0396761a2ec2663fda53f56adabe6537389153`) produced G1 BTC/USD
`138,448.72s / 319,092.10s = 43.39%` from BOTH implementations, agreeing with EACH OTHER
EXACTLY (`stale_s_diff = 0.0`, `accounted_s_diff = 0.0`) but not with the pre-registered
`127,071 / 273,233 = 46.48%` target beyond tolerance — a BLOCKER by 2.2's own rule, reported
rather than tuned away.

**Cause, verified from bytes, not narrative**: the B1 46.5% number was measured on a DIFFERENT
FILE REVISION than the one this SPEC named. Two capture files exist on the workstation
(`runs/907_live_fill_probe/`, gitignored, not previously read by this instrument):
  - `capture_907_main.snapshot_2026-08-31.jsonl` — 1,248,413,957 bytes, span 309,784s (the
    file B1's 46.5%/`stale_book_s=127,071`/`accounted_s=273,233` and the "longest gap-free
    stretch 561.8s" figure were both measured on; `CHANGELOG.md:433` ("BTC/USD reads 0.126%
    un-observed beside 127,071 s of 273,233 s (46.5%) frozen mid"),
    `lore/handoff_2026-09-01_factory_run_49.md:96` (same figures, same capture),
    `lore/handoff_2026-09-02_factory_run_50.md:38` (span/gap table, confirming 309,784s /
    1688 gaps / 561.8s for the snapshot). `lore/handoff_2026-09-05_factory_run_53.md:181-188`
    independently documents the SAME class of error one level up (the capture's "3.586 days"
    figure was ALSO lifted from `CHANGELOG.md`'s B1 entry and is the snapshot's span, not the
    final file's 4.1198 days) — a second, already-recorded instance of this exact provenance
    failure, not the source of the number itself.
  - `capture_907_main.jsonl` on gpu-server (the one this task named and this instrument read,
    identical to `capture_907_main.FINAL_2026-09-01.jsonl` on the workstation) — 1,440,479,339
    bytes, span 355,953s. Its own recorded longest gap-free stretch is 9,670.9s
    (`lore/handoff_2026-09-02_factory_run_50.md`), not 561.8s.
  - The two are the SAME recording at two points in time: the snapshot is byte-identical to the
    FIRST 1,248,413,957 bytes of the final file — verified directly,
    `sha256(head -c 1248413957 capture_907_main.jsonl) == sha256(capture_907_main.snapshot_2026-08-31.jsonl)
    == 7aede82eb782bee73df520423eae98b3eb527d84c503b342ab9408dfba8ed75e`, both sides, on
    gpu-server and the workstation independently.
  - The task file's own "1,440,479,339 bytes, mtime Sep 1 10:00" description names the FINAL
    file (verified — that is exactly `capture_907_main.jsonl`'s stat on gpu-server), so this
    instrument correctly read the file the task named; the B1 TARGET NUMBER quoted alongside it
    was carried over from the earlier snapshot's own measurement. The error is in the number
    that was cited to check against, not in which file was read for the deliverable.

**Correction to G1 (2.2)**: the pre-registered target (127,071/273,233) is re-scoped to the
SNAPSHOT PREFIX, not the final file. G1 is now checked on
`head -c 1248413957 capture_907_main.jsonl` (asserted byte-identical to the snapshot by sha256
before use) rather than on the full final file. This is the BINDING G1 gate. The full-file
shipped-vs-independent agreement (`43.39%`, exact cross-implementation match) is ALSO recorded
in `results.json` as a secondary, non-gating cross-check, because it is a real, useful
confirmation that the two run-tracking implementations agree with each other on the actual
deliverable file even though neither was ever expected to reproduce a different file's number.

**Correction to 2.8's rationale paragraph** (the AS(1h) DEFINITION itself is UNCHANGED and
remains binding — only the justifying prose about gap-free-stretch length is corrected): the
claim "the longest gap-free stretch in this capture is 561.8s... so AS(1h) over 1h windows
requiring a whole gap-free hour is structurally 0-sample" was true of the SNAPSHOT, not of the
FINAL file this instrument actually measures against — the final file's longest gap-free
stretch is 9,670.9s (≈2.7 hours), so a naive whole-gap-free-hour requirement would NOT be
uniformly zero-sample on the final file. This does not change the chosen definition: the
time-weighted, per-elapsed-clock-hour-bucket construction over `accounted_s` (2.1) was already
the more general, always-well-defined form (it does not require an ENTIRE hour to be gap-free,
only sums the accounted-and-two-sided seconds that exist within each bucket, reporting the
bucket's own covered-seconds fraction beside it) — it was chosen for that generality, not merely
because the narrower form was zero-sample on the file believed at pre-registration time to be
the target. It remains the correct choice on the final file regardless.

## 6. Compute plan

Everything on `gpu-server`, wrapped: `tools/safe_run.sh --cpu --mem 4G` (staged copy),
`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`. Smoke/synthetic tests first
(seconds). The full 1.44 GB pass runs inside `tmux` (expected several minutes: one
`read_capture()` pass + one independent streaming pass over ~10.2M lines), polled sparingly.
No writes anywhere outside `/var/tmp/924-implementer/` (execution scratch) and this
experiment's own output directory.

## ERRATA 2 (POST-HOC, cycle-2 fix pass 2026-09-28, added AFTER the full pass and AFTER a verification panel — every section above, including ERRATA 1, is UNCHANGED text)

**Recorded**: SPEC sha256 BEFORE this section:
`1d3b6b6e1c02733ab9909ac1840a575973e25592bcb91623b874c885224702a9` (= the "after ERRATA 1"
sha). SPEC sha256 AFTER this section: recorded in `results.json.spec_sha256_after_errata_2`
and `RESULTS.md` (a file cannot carry its own hash). Everything this section introduces is
POST-HOC and is excluded from the pre-registered go/no-go computation (section 4 / G4).

**Finding 1 (BLOCKER, correctness) — the frozen-mid numerator is gap-blind.** The shipped
907 function clips stale runs to the capture span only (`capture.py:2433-2436`,
`stale_clipped`), while its sibling `covered_s` subtracts recorded gaps; section 2.2(ii)'s
independent `RunTracker` (as first shipped) was never given the gap spans either. The G1
cross-check therefore certified two implementations agreeing on a SHARED blind spot. The
denominator (`accounted_s`) excludes gap seconds; the numerator does not — a numerator/
denominator denomination mismatch. Panel measurement, BTC/USD, final capture: 32,167.73 s
of the 138,448.72 stale seconds lie inside recorded gap spans; 496 of 1,222 runs > 60 s
straddle a gap; the gap-clipped numerator over the unchanged `accounted_s` is 33.31%
(vs 43.39% shipped definition). This is the same gap-blindness class D81 already measured on
this capture (`tasks/DECISIONS.md` D81: "63 of 67 FLAGGED runs overlap a recorded gap").
The streak and survival statistics of 2.4/2.7 share the same run path and the same defect.

**Disposition of G1.** G1 (2.2 + ERRATA 1) reproduces the SHIPPED (gap-blind) definition.
It remains the binding reproduction check OF THE SHIPPED NUMBER and passes as such — it
proves this instrument computes what 907 computes, not that 907's quantity is the correct
frozen-mid fraction. 924's OWN frozen-mid is from here on reported GAP-CLIPPED, beside the
shipped-definition value, never replacing it for G1. D81 wall: the shipped 907 code is NOT
changed; the corrected quantity is measured in 924's own code and disclosed.

**Gap-clipped definition (924's own, post-hoc).** Numerator = seconds inside identical-
`(bid, ask)` runs longer than `STALE_BOOK_MIN_RUN_S` (60 s), intersected with the accounted
segments (capture span minus `meta.merged_gap_spans`, REUSED); denominator = `accounted_s`
(2.1, unchanged). **Run rule at a gap**: a run that straddles a recorded gap is SPLIT at the
gap — the recorder was not observing inside it, so the same price seen after the gap is a new
observation, not evidence the quote survived through it. The > 60 s threshold applies to EACH
observed piece separately. (A reconciliation value that drops gap seconds but keeps the
shipped per-run threshold — the quantity the panel measured — is also reported, labelled
RECONCILIATION.) Streak statistics are recomputed over the gap-clipped pieces.

**Finding 2 (MAJOR) — survival was count-weighted.** 2.7's survival summary weights each run
once (BTC median 0.026 s over 141,513 runs). The maker-relevant statistic is TIME-weighted
(length-biased): the duration of the run a random instant falls in (panel recomputation,
before gap-clipping: BTC median 42.15 s, mean 77.4 s). Post-hoc, reported per pair:
time-weighted p10/p25/p50/p75/p90 + mean over gap-clipped run pieces (quantile rule: sorted
ascending, first duration whose cumulative duration reaches p% of the total; mean =
sum(d^2)/sum(d)). The count-weighted form is kept as SECONDARY, labelled.

**Finding 3 (MAJOR, gate design at brief/task level) — the pre-committed go/no-go cannot
discriminate on this capture.** (a) The recorder never emits a one-sided book row
(`recorder.py:527-531`: `if top is None: continue`), so section 2.6's two-sided presence is
~1.0 BY CONSTRUCTION, not by measurement. (b) "AS(1h)" as defined in 2.8 is a time-weighted
mean relative HALF-SPREAD — a spread cost — not adverse selection / post-fill markout; it
cannot see what a resting order suffers after it fills. (c) The frozen-mid fraction — this
row's own motivating warning — is absent from section 4's table. The pre-registered verdict
is still computed exactly as pre-registered (section 4) and reported, but flagged
`preregistered_gate_discriminating: false`; the decision is the operator's.

**Finding 4 (MINOR)** — the suite never planted a write to prove G2 fires. Cycle-2 tests add
planted-write cases (append; same-size in-place write with mtime restored).

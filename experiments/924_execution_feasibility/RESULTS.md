# TASK_924 RESULTS — execution-feasibility instrument over the closed 907 capture

## HEADLINE (cycle 2, POST-HOC corrections — read this before any table below)

- **status: `PREREG_UNBLOCKED__GATE_NON_DISCRIMINATING`** · **decision_owner: operator**
- **`preregistered_gate_discriminating: false`.** The pre-registered verdict is still
  `UNBLOCKED_WITH_NUMBERS`, computed exactly as pre-registered, but neither pre-committed
  disqualifier could fire on this capture: (1) the recorder never emits a one-sided book row
  (`907_live_fill_probe/recorder.py:527-531`, `if top is None: continue`), so two-sided
  presence is ~1.0 by construction; (2) "AS(1h)" as specified is a mean relative half-spread,
  a spread cost, not adverse selection / markout; (3) the frozen-mid fraction, the row's own
  motivating warning, is not in the pre-registered table.
- **Established**: quoted relative half-spreads are sub-bp to <1 bp when the book is live.
  **NOT established**: that resting-order adverse selection is small. Nothing here measures
  markout, fill probability or queue position. This artifact makes no maker-viability claim.
- Frozen-mid, instrument = `instrument.py` cycle 2, final capture sha `8b3f748d…9153`,
  denominator `accounted_s` = 319,092.10 s (SPEC ERRATA 2):

| pair | frozen-mid, shipped def. (gap-blind, G1 qty) | frozen-mid, gap-clipped (924's own) | quote survival time-wtd p50 / p90 (s, gap-clipped) | count-wtd p50 (s, secondary) | median rel. half-spread (bp) | AS(1h) time-wtd (bp, = spread) |
|---|---|---|---|---|---|---|
| BTC/USD | 43.39% | **30.42%** | **32.5** / 160.5 | 0.026 | 0.0064 | 0.0136 |
| ETH/USD | 23.36% | **12.44%** | **11.4** / 94.4 | 0.025 | 0.2248 | 0.1125 |
| SOL/USD | 14.38% | **3.66%** | **5.46** / 33.4 | 0.121 | 0.9403 | 0.6535 |
| XRP/USD | 9.79% | **1.01%** | **2.36** / 20.4 | 0.017 | 0.7550 | 0.5267 |

- G1 reproduces the SHIPPED (gap-blind) definition and passes as such (snapshot prefix:
  46.5064% vs target 46.5065%, both implementations exact). It certifies agreement with 907,
  not that 907's quantity is gap-correct: 907 clips stale runs to the span only
  (`capture.py:2433-2436`) while dividing by gap-excluded `accounted_s`.


Instrument: `instrument.py` (this directory). Definitions: `SPEC.md`, pre-registered
2026-09-28 (sha256 `56212530e6f90958c5dbc2a97e025ade939efded1374575cdeb9292dd4fa7c72` before
the full pass; `1d3b6b6e1c02733ab9909ac1840a575973e25592bcb91623b874c885224702a9` after the
POST-HOC ERRATA section — see "File-revision correction" below). Full data: `results.json`.

## Files read (read-only throughout; D81 — nothing under `experiments/907_live_fill_probe/`
was ever opened for writing)

| file | bytes | sha256 | role |
|---|---|---|---|
| `capture_907_main.jsonl` (srv) = `capture_907_main.FINAL_2026-09-01.jsonl` (workstation) | 1,440,479,339 | `8b3f748d…9153` | **the deliverable** — every per-pair number below is measured on this file |
| `capture_907_main.snapshot_2026-08-31.jsonl` = `head -c 1248413957` of the final file | 1,248,413,957 | `7aede82e…8ed75` | G1's reproduction target's own file revision (see errata) — read ONLY to satisfy G1 |

Verified: `sha256(head -c 1248413957 <final>) == sha256(<snapshot>)` on both gpu-server and
the workstation, independently. The capture file's sha256/size/mtime were fingerprinted
before AND after BOTH passes (G2) and once more, fresh, after the whole two-session episode
(~5 hours apart) — all four readings identical.

## File-revision correction (POST-HOC, disclosed — SPEC.md's own ERRATA section, sha recorded
before AND after)

The task brief's G1 target (127,071s / 273,233s = 46.48% BTC/USD frozen-mid) was measured on
the **1.248 GB snapshot capture**, not the 1.44 GB final capture this task named and this
instrument reads. Both are the same recording at two points in time (the snapshot is
byte-identical to the final file's first 1,248,413,957 bytes). The brief's number was carried
across from `CHANGELOG.md`'s B1 entry, which describes the snapshot
(`lore/handoff_2026-09-01_factory_run_49.md:96`); the task's own "1,440,479,339 bytes"
description correctly named the final file. **G1 is therefore checked on the snapshot prefix
(binding — this is the file the target number actually describes), with the final-file
cross-implementation agreement recorded separately as a non-binding confirmation.**

## G1 — frozen-mid reproduction

**Binding (snapshot prefix, byte-identical to `CHANGELOG.md`'s target file) — PASS:**

| | shipped (907, imported) | independent (this module) | target (B1, `capture.py:213-214`) |
|---|---|---|---|
| BTC/USD stale_book_s | 127,070.878 | 127,070.878 | 127,071 |
| BTC/USD accounted_s | 273,233.044 | 273,233.044 | 273,233 |
| fraction | 46.5064% | 46.5064% | 46.5065% |

Tolerance: `max(60s, 0.1%*accounted_s)` = 273.2s (SPEC.md 2.2). All three diffs are < 0.2s —
essentially exact, both cross-implementation and against the pre-registered target.

**Non-binding (final capture, the deliverable file) — the two implementations agree with
EACH OTHER exactly (not checked against the snapshot's number — no capture is expected to
reproduce a different capture's measurement):**

| pair | shipped stale_book_s | independent stale_book_s | accounted_s | fraction | impls agree |
|---|---|---|---|---|---|
| BTC/USD | 138,448.724 | 138,448.724 | 319,092.099 | 43.39% | yes (diff 0.0s) |
| ETH/USD | 74,542.798 | 74,542.798 | 319,092.099 | 23.36% | yes (diff 0.0s) |
| SOL/USD | 45,876.483 | 45,876.483 | 319,092.099 | 14.38% | yes (diff 0.0s) |
| XRP/USD | 31,246.223 | 31,246.223 | 319,092.099 | 9.79% | yes (diff 0.0s) |

All four pairs: `stale_s_diff == 0.0`, `accounted_s_diff == 0.0` between the shipped 907
reader and this module's independent run tracker on the final capture — a genuine, exact
cross-check of the deliverable file, corroborated independently by
`lore/handoff_2026-09-02_factory_run_50.md` ("BTC's 138,449s of frozen mid" on the same final
capture, matching this measurement to within 0.3s).

## G2 / G3 / G4

- **G2 (read-only)**: capture sha256/size/mtime identical before the full pass, after the
  full pass, before the snapshot-prefix pass, after the snapshot-prefix pass, and on a fresh
  independent fingerprint taken ~5 hours later while assembling this report. Four
  fingerprints, one file, one hash: `8b3f748dfc5f8a06a22d7484ae0396761a2ec2663fda53f56adabe6537389153`.
- **G3 (RSS)**: full-capture pass peak RSS **1.452 GiB** (VmHWM); snapshot-prefix pass peak
  RSS **1.311 GiB**. Both under the 4 GiB contract, measured via the same primitive
  `test_streaming_reader.py` justifies (`/proc/self/status` `VmHWM`, not `ru_maxrss`).
- **G4 (pre-registration)**: `SPEC.md` sha256 `56212530…` recorded in every `results.json`
  produced before this errata (smoke, resmoke, full pass, snapshot pass) — all four cite the
  identical pre-errata sha, proving the definitions did not move between any of those runs.
  The errata (this file-revision correction) was added AFTER the full pass, is clearly
  labelled POST-HOC in `SPEC.md`, and does not alter any pre-registered definition — only the
  target-number/rationale-prose correction.

## Per-pair table (final capture — the deliverable)

| pair | two-sided (time-wtd) | frozen-mid (shipped def., gap-blind — see HEADLINE) | median rel. half-spread (bp) | AS(1h) time-wtd (bp) | AS(1h) event-wtd (bp, non-binding) | quote survival median (s, count-wtd, gap-blind — see HEADLINE) | quote updates/active min |
|---|---|---|---|---|---|---|---|
| BTC/USD | 99.9999% | 43.39% | 0.0064 | 0.0136 | 0.0524 | 0.026 | 496.5 |
| ETH/USD | 99.9999% | 23.36% | 0.2248 | 0.1125 | 0.3137 | 0.025 | 420.3 |
| SOL/USD | 99.9999% | 14.38% | 0.9403 | 0.6535 | 0.8209 | 0.120 | 416.6 |
| XRP/USD | 99.9999% | 9.79% | 0.7550 | 0.5267 | 0.7926 | 0.017 | 700.9 |

- "Two-sided" is DEPTH-denominated (`bid_sz > 0 and ask_sz > 0`, SPEC.md 2.5/2.6) — a
  DIFFERENT axis from frozen-mid (PRICE-denominated, SPEC.md 2.2/2.4). Every book row in this
  capture carries genuine displayed size on both sides (`row_two_sided_fraction == 1.0` for
  all four pairs, `n_one_sided_rows == 0`; depth medians are non-degenerate — BTC bid/ask
  median 0.10/0.13, XRP 644/744, see `results.json`), so this fraction saturates near 100% for
  every pair regardless of price staleness. **The task file's own "~50%" go/no-go threshold
  appears to have informally conflated "two-sided presence" with "book freshness" (the
  frozen-mid number); this instrument's pre-registered definitions (written before any data
  was read) keep the two separate, and both are reported.**
- AS(1h) time-weighted is well under the 2bp descriptive line for all four pairs (max 0.65bp,
  SOL/USD). Event-weighted values run 2-4x higher than time-weighted for every pair (BTC
  3.8x, ETH 2.8x, SOL 1.3x, XRP 1.5x) — material, in the expected direction: ~88% of BTC book
  events repeat the prior `(bid, ask)` (`capture.py:213`), so time-weighting (which credits
  each SECOND once) discounts high-update-frequency moments relative to event-weighting
  (which credits each ROW once); the two disagree most where update frequency and spread are
  both elevated intermittently.
- Full per-pair detail (depth stats, quote-survival p75/p90/p95, hour-of-day rel-spread
  quartiles, per-bucket AS(1h) with covered-fraction) is in `results.json.per_pair_final_capture`.


## POST-HOC cycle-2 detail (SPEC.md ERRATA 2 — excluded from the pre-registered go/no-go)

SPEC sha256 before ERRATA 2 `1d3b6b6e…a9`; after `2ef517cc316703e9364e507e32dd5a7fa16a3c6b43be102cdbdb03531251f01d`.
Cycle-2 passes ran with the pre-ERRATA-2 SPEC (sha `1d3b6b6e…`); the pre-registered per-pair
fields reproduced bit-identically against cycle 1 (0 of all fields differ).

**Gap-clipped frozen-mid — run rule at a gap.** A run straddling a recorded gap is split at
the gap (the recorder was not observing; the price after the gap is a new observation), and
the > 60 s threshold applies to each observed piece. Numerator: seconds in > 60 s pieces;
denominator unchanged `accounted_s`.

| pair | stale s shipped | stale s inside recorded gaps | >60 s runs straddling a gap | unsplit intersection (reconciliation) | split rule (reported) | streak max shipped → gap-clipped (s) | streak n shipped → gap-clipped |
|---|---|---|---|---|---|---|---|
| BTC/USD | 138,448.72 | 32,167.73 | 496 / 1,222 | 33.31% | 30.42% | 778.98 → 501.18 | 1,222 → 848 |
| ETH/USD | 74,542.80 | 30,088.47 | 459 / 773 | 13.93% | 12.44% | 402.70 → 326.50 | 773 → 372 |
| SOL/USD | 45,876.48 | 30,858.03 | 467 / 600 | 4.71% | 3.66% | 243.26 → 202.72 | 600 → 140 |
| XRP/USD | 31,246.22 | 26,640.51 | 404 / 441 | 1.44% | 1.01% | 240.22 → 143.14 | 441 → 39 |

**Reconciliation with the panel's 33.31% (BTC).** The panel's number is the UNSPLIT
intersection (shipped per-run threshold, gap seconds removed); this instrument reproduces it
exactly (33.31%, 32,167.73 s in gaps, 496 straddling runs). The split rule the fix brief
specified also drops pieces that fall below 60 s once split, and reads **30.42%** — 2.9 pt
below the panel figure, outside the brief's 0.1 pt reproduction tolerance. This is a
definitional difference, not a computational one. Reported as is, not tuned. Both values are
in `results.json.cycle2_post_hoc`.

**Quote survival — time-weighted (length-biased), gap-clipped** = duration of the run a
random in-run instant falls in (quantile: first duration whose cumulative duration reaches
p% of total; mean = Σd²/Σd). Count-weighted is secondary.

| pair | p10 | p25 | p50 | p75 | p90 | mean | gap-blind p50 / mean (panel check) | count-wtd p50 / mean (secondary) |
|---|---|---|---|---|---|---|---|---|
| BTC/USD | 4.75 | 12.72 | 32.49 | 78.79 | 160.52 | 62.34 | 42.15 / 77.45 | 0.026 / 2.15 |
| ETH/USD | 0.72 | 3.31 | 11.42 | 36.54 | 94.40 | 32.35 | 15.79 / 43.21 | 0.025 / 0.64 |
| SOL/USD | 0.48 | 1.71 | 5.46 | 15.12 | 33.37 | 13.25 | 7.01 / 20.68 | 0.121 / 0.84 |
| XRP/USD | 0.14 | 0.58 | 2.36 | 7.55 | 20.39 | 7.95 | 3.83 / 18.80 | 0.017 / 0.21 |

The gap-blind time-weighted BTC values (42.15 s / 77.45 s) reproduce the panel's recomputation.

**Cycle-2 gates.** G1 PASS on the shipped definition (snapshot prefix sha `7aede82e…8ed75`
verified before use; BTC shipped = independent = 46.5064%, target 46.5065%; final capture,
all four pairs, `stale_s_diff = 0.0`). G2 PASS (final capture sha/size/mtime identical
before/after; snapshot prefix likewise). G3 PASS (peak RSS 1.499 GiB full pass, 1.314 GiB
snapshot pass). New tests prove G2 goes false on a planted append and on a same-size
in-place write with the mtime restored (only the sha256 component catches the latter).

## Go/no-go (SPEC.md section 4, pre-registered before any data was read)

*Pre-registered computation, reproduced unchanged. See HEADLINE: this gate is non-discriminating on this capture (`preregistered_gate_discriminating: false`), and the decision is the operator's.*

- Two-sided book present < ~50% of active time, any pair? **NO** (all four ≈100%, see caveat
  above on denomination).
- AS(1h) time-weighted ≫ 2bp, everywhere? **NO** (max 0.65bp, SOL/USD; all four pairs).
- **Verdict (pre-registered): UNBLOCKED_WITH_NUMBERS.** Neither pre-committed disqualifying
  condition is met, and on this capture neither could be (HEADLINE). Status:
  `PREREG_UNBLOCKED__GATE_NON_DISCRIMINATING`, decision owner: operator. The ≤2bp AS(1h) line remains DESCRIPTIVE ONLY — this is not 907's
  markout gate and does not by itself certify a fill model; it bounds what a passive fill
  COULD cost under a wall-clock-weighted reading of the quoted spread.

## What this does NOT establish

0. **That resting-order adverse selection is small.** "AS(1h)" here is quoted half-spread, not markout. Two-sided presence is ~1.0 by recorder construction and carries no information about book liveness.

1. **Both B1 disclosed limits are inherited by the frozen-mid measurement above, verbatim in
   substance** (`capture.py:186-232`, SPEC.md 2.4):
   - The run key is `(bid, ask)` only. A literal repeat can be either (a) a keepalive timer
     firing with no wire frame behind it, or (b) a real wire frame in which only the SIZES
     moved. Neither this instrument nor 907's own reader can distinguish them; "frozen" means
     "the mid carried no new information", not "the wire was not observed".
   - The detector counts ONE shape: a run of CONSECUTIVE IDENTICAL `(bid, ask)`. A mid that
     OSCILLATES over a small set of values without ever discovering a new price is NOT
     detected by this measure — `test_instrument.py::test_oscillating_mid_is_NOT_flagged_frozen`
     pins this staying true rather than silently starting to pass.
2. **This is not a fill-rate or markout estimate.** AS(1h) bounds what a passive fill at the
   quoted half-spread COULD cost; it says nothing about the probability of getting filled at
   that price, adverse selection conditional on a fill, or queue position — that is 907's
   gated markout lane (D81, out of scope here) and the shadow-engine/reference-fill-model
   instruments, not this one.
3. **"Two-sided" ≠ "fresh".** As stated above, the ~100% two-sided figure is a depth-presence
   measure and says nothing about how often the quoted price actually changed; read it
   alongside the frozen-mid fraction, never as a substitute for it.
4. **No venue comparison, no order-book-signal claim, no DB read.** Out of scope per the task
   file and D81; nothing here touches `trading_data.db`, any exchange API, or any other
   venue's data.
5. **[SUPERSEDED by cycle 2 — the 43.39% below is the gap-blind shipped definition; 924's gap-clipped BTC frozen-mid is 30.42%, see HEADLINE]** **File-revision note**: the frozen-mid figure most relevant to Option B's actual execution
   surface is the FINAL capture's 43.39% (not the snapshot's 46.48% quoted in the task brief)
   — a real, if modest, downward revision of how "frozen" BTC/USD's book has been over the
   full 4.12-day record versus the 3.59-day snapshot the brief's number came from.

## Compute / provenance

- All computation ran on `gpu-server`, CPU-only, wrapped `tools/safe_run.sh --cpu --mem 4G`,
  `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`. `trader_env` python 3.10.20.
- 907's `capture.py` was staged read-only in a private scratch dir
  (`/var/tmp/924-implementer/907_live_fill_probe/capture.py`), verified sha256-identical to
  the workstation's `experiments/907_live_fill_probe/capture.py`
  (`355a88e86f2da0cc2363dcbb3680510998214b9890a76b79d48b2b05a92f1d48`) before any use, and
  never opened for writing.
- Unit tests: 13/13 passed on gpu-server (wrapped, cycle 2: 7 pre-registration + 6 post-hoc),
  synthetic inputs only, never touching the 907 tree.

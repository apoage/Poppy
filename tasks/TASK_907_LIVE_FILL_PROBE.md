> **STATUS (2026-08-27, factory run 44): CYCLE 4 BUILT AND GREEN ON ITS OWN SUITE (173 passed),
> BUT HALTED AGAIN — NOT SHIPPED. D66's PRE-COMMITTED STOP HAS FIRED. See D67.**
> Work preserved at `lore/handoff_2026-08-27_factory_run_44_907_cycle4_halt.patch` (6237 lines,
> 21 files) and left in the tree, untracked. **No green verdict was written and none is writable:
> BLOCKERs stand, including one against cycle 4's OWN named fix.**
>
> **D66 authorised ONE cycle for the named fix, with a pre-committed stop: any NEW absence
> granularity demonstrated by a lens → straight to (b), no fifth cycle. FOUR were demonstrated,
> by THREE independent parties.** The mandated Condition-2 lens was briefed BLIND (told nothing of
> what the implementer or orchestrator had found) and converged anyway. The capture-set defect was
> independently reproduced FIVE times. The stop is not a close call.
>
> | # | new absence granularity | axis | severity | demonstrated by |
> |---|---|---|---|---|
> | 1 | **pre-registered horizon with ZERO samples is omitted entirely** (`estimands.py:148` `if not raw: continue`; `render.py:229,258`) | pair × horizon | BLOCKER | implementer, fail-closed lens, orchestrator |
> | 2 | **cross-capture same-pair clobber** (`cli.py:77-80` plain assignment keyed on pair alone, inside the loop over a repeatable `--capture`) | capture set | BLOCKER | implementer, fail-closed, correctness, flow-verifier, orchestrator |
> | 3 | **the RESUME path re-opens the halting MAJOR** (`capture.py:252/266` — `meta_start` is written only on a fresh file; on resume the `pairs` argument is silently discarded) | pair, via a path the fix does not cover | BLOCKER | correctness lens |
> | 4 | **the data span can be absent relative to the capture span, and the aggregate bound is denominated in the very span that collapsed** (`capture.py:914-919`) | time | BLOCKER | flow-verifier |
>
> **Finding 3 is decisive on its own, independently of D66.** `declared_pairs` is a snapshot of the
> FIRST session's subscription config. Session 2 adds `ETH/USD`, dead from t=0 → `is_complete()`
> True, `assert_complete()` PASSED, `ETH/USD` appears **0 times** in the artifact — *worse* than
> pre-cycle-4, where the declared∪observed union at least surfaced it on the provenance line. So
> **D66 Condition 1 is closed on the fresh-capture path and OPEN on the resume path**, and a 3–7
> day capture is precisely where process restarts happen. Cycle 4 does not fully close its own
> named MAJOR. The mirror direction also misdiagnoses: an operator who ADDS a pair on restart and
> receives it cleanly gets a refusal reading "a number backed by a pair no subscription asked for"
> — the subscription did ask; only the metadata is stale.
>
> **Finding 4 is the deepest thing this row has produced, and it generalises the whole family.**
> Coverage is computed over `[data_start_ts, data_end_ts]` and never reconciled against
> `[start_ts, end_ts]`, so when book/trade go dark while heartbeats keep flowing, **the
> denominator shrinks with the failure — the check is denominated in the quantity it is checking.**
> Measured on a 7200s capture carrying data for only its first 130s, heartbeats every 10s
> throughout: all four defenses clean, `uncovered_fraction = 0.000%` against a true un-observed
> fraction of **98.188%**, and the artifact renders `1m: median -0.64bp (n_used=1) — 0bp:
> meets/exceeds, 5bp: meets/exceeds, 10bp: meets/exceeds` — this row's own PREFERRED CONCLUSION,
> off n=1, from a capture that observed the market for 130 of 7200 seconds. The mechanism:
> `_find_unrecorded_gaps` iterates strictly BETWEEN consecutive events, so a hole at an EDGE of the
> data timeline has no consecutive pair spanning it. Interior hole → REFUSED (cycle 2/3 closed
> that); leading, trailing and both-edge → PASS. **This is exactly the symptom the run-43 banner
> named as the thing to watch — "connected but zero events recorded" — with the recorder staying up
> and writing no gap, because heartbeats never let `heartbeat_timeout_s` expire.** Related and
> demonstrated false: `PairCoverage.uncovered_fraction` (`capture.py:437-443`) returns `0.0` when
> `accounted_s <= 0` under a comment asserting "`assert_complete`'s other checks own that case" —
> they do not; `accounted_s=0` passes with 99.993% of the capture un-observed.
>
> **Two SAFETY findings, outside the absence family, on the row whose phase 2 sends real money:**
> - **MAJOR — the "no code path can transmit an order" invariant is pinned by NO test.** It is
>   prose, re-argued after every cycle. Demonstrated: appending a market `add_order` frame carrying
>   a token to the send loop leaves the suite GREEN; switching `WS_URL` to Kraken's authenticated
>   endpoint leaves the suite GREEN. Orchestrator-verified: no test anywhere references `WS_URL`,
>   `wss://` or `ws.kraken`, and the cycle-4 coupling test filters to `method == "subscribe"`
>   frames BEFORE counting (`test_declared_pairs.py:303-305`), so `len(subscribed) == 2` places no
>   constraint on the total frames sent. Directions: `assert len(sent) == 2`; assert the url handed
>   to `connect_fn`; pin the `WS_URL` literal.
> - **MAJOR — acceptance gate (ii) is NOT MET: the notional caps are opt-in and default OFF.**
>   `check_arming` demands `--budget-cap-usd` and then **no rail ever reads it**;
>   `phase2_rails.py:189-192` guard both caps with `if ... is not None`, and both default to
>   `None`. Demonstrated: a **$1,000,000** order with $9,999,999 already spent, against a declared
>   **$50** cap, clears EVERY rail. Inert today only because `place_order` raises — i.e. the entire
>   notional protection rests on the one conjunct the phase-2 increment exists to remove.
>   Pre-existing, untouched by cycle 4, and gate (ii) claims the caps fire.
> - Minor, same family: `recorder.py:297`'s `except Exception` also catches `Phase2RefusalError`
>   and `RailViolation` (both subclass `RuntimeError`), so if the rails are ever called from inside
>   the connect loop a rail violation is laundered into a "transport failure" gap and the loop
>   retries. And the credential fence (`test_credential_free.py:10-14`) is a hardcoded FILENAME
>   allowlist — `__init__.py` is unscanned and any NEW source file is born outside it;
>   demonstrated by adding a `phase2_live.py` reading both credential env vars and building an
>   `add_order` frame: **3 passed**. `FORBIDDEN` contains no env-access token at all. Direction:
>   glob `*.py`; add `os.environ`, `getenv`, `add_order`, `AddOrder`.
>
> **The banner's absolute claim "nothing anywhere catches `NotImplementedError`" is FALSE and is
> corrected here.** `recorder.py:297` is a broad `except Exception`, and
> `NotImplementedError → RuntimeError → Exception`. Orchestrator- and lens-verified that it wraps
> only connect/subscribe/consume with no order path inside it, and that `phase2_rails` is imported
> by no production module in the lane — so **the safety claim holds in substance today**, but it is
> a reading, not a pin, and the task file states phase 2's orders are "placed by the same engine",
> whose natural wiring point is inside that `try`.
>
> **What cycle 4 got RIGHT — verified independently, and worth keeping through the restructure:**
> `declared_pairs` is populated from `meta_start` ONLY, never unioned with observed
> (`capture.py:807`); `meta.pairs` keeps its old union meaning and its sole consumer is
> `render.py:80` (grep-verified across `estimands.py`, `shadow_engine.py`, `cli.py`) — nothing was
> silently re-defined. Condition 1's WRITE side is genuinely closed by a test that drives the real
> `LiveRecorder.run` with the production writer and asserts the subscribe frames' symbols ==
> `meta_start.pairs` == `read_capture(...).declared_pairs`. `observed_pairs` is a read-only
> property over `observed_pair_channels`, so pair-level and cell-level views cannot be made to
> disagree. Fail-CLOSED on a missing OR empty declared set, at BOTH format versions, with the
> bypass named in the refusal ("declared := observed" would let a capture declare nothing and
> reconcile perfectly against whatever it contained) — and the reasoning correctly distinguishes
> `pairs` (written unconditionally → absence is not a producible shape → refuse) from `channels`
> and `counters` (absence IS producible → disclose as undeclared/UNKNOWN). The **(pair × channel)**
> cell is closed via `declared_channel_without_data`, and the `if self.channels:` softening reaches
> synthetic captures only, because the production recorder passes
> `channels=list(SUBSCRIBED_CHANNELS)` unconditionally (`recorder.py:249`, verified). A genuine
> pre-existing latent bug was fixed: a bare string in `meta_start.pairs` was iterated
> character-by-character into a declared set of single letters. 20 of 21 new tests fail against
> pre-fix code (the survivor correctly being the no-CLI-flag test), and the load-bearing ones fail
> BEHAVIOURALLY (`assert not is_complete()`) rather than on `AttributeError` — a real mutation
> proof, not a tautology. The edited pre-existing fixture is honest: `diff -u` shows
> `test_capture_gaps.py` gained exactly 8 lines (6 comment + `declared_pairs=["X"]` +
> `observed_pair_channels={"X":["trade"]}`) matching the coverage record it already built for pair
> `X`; no bound moved, no expectation weakened. End-to-end the fix is real: pre-fix, the dead pair
> renders on exactly ONE line (the provenance) and `is_complete()` is True; post-fix the identical
> capture refuses and NAMES the pair. Gate (iii)'s "cannot reopen 897's ALL FAIL" disclaimer
> survives at line 3 of all 14 rendered artifacts; a gate-(iv) refusal emits no artifact and
> creates no directory; the banner's book-dies-partway ↔ mid-source-split interlock genuinely
> holds (`MATERIAL: 61.4% ... rest on a trade print`); cycle 3's pair-dies-partway defense still
> refuses correctly.
>
> **Also found, lower severity, recorded so the restructure inherits them:** the twins DO diverge —
> `capture.py:624` uses `if self.had_meta_start is False:`, an identity test, so
> `had_meta_start=0` gives `is_complete()` falsy while `assert_complete()` PASSES (unsafe
> direction); the "structurally impossible" claim is really a 12-case enumeration that omits the
> one field not helper-routed. `channels: []` disables both channel checks with no refusal, while
> `declared_pairs: []` is refused — the identical shape one field over. The channel-side mirror
> (`observed_channel_undeclared`) was not implemented, so a capture declaring `channels=["trade"]`
> and full of `book` events passes while every primary estimand number rests on an undeclared
> channel. Resume also discards the resumed session's `channels` and `heartbeat_timeout_s`, so the
> artifact prints a liveness knob that did not govern the whole file. Symbol-spelling drift
> (`btc/usd` vs `BTC/USD`) is caught only at READ time, fail-closed but after the capture is over,
> and Kraken's subscribe-ack — which echoes the venue's own symbol and is the one free signal in
> the first second — is discarded at `recorder.py:214`. `--pairs` is a THIRD operator declaration
> reconciled against nothing: `--pairs SOL/USD` on a capture without it yields exit 0 and a
> ~2900-char gate-passing artifact with the full disclosure box, full provenance and coverage
> tables, and ZERO estimand sections, with no line saying a filter was applied.
>
> **NOT re-litigated and still true:** phase 1 vs phase 2 scope split (confirmed); `place_order()`
> raises `NotImplementedError` as its first statement; the only network primitive is
> `websockets.connect` against the hardcoded public `wss://ws.kraken.com/v2` sending two public
> subscribe frames; `kraken_client`/`add_order`/`_private_request` never imported; no credential
> read, logged or referenced by any current source file. Cycle 4 touched exactly four lane files
> and added no `open`, no `write`, no `subprocess`, no `os.environ`, and no new `except`.
> **The regression lens cleared the two things most likely to be quiet self-accommodation, and it
> did so by mutation rather than by reading.** The edited pre-existing fixture is HONEST: exactly
> 8 lines (+8/−0) in one file, 6 of them comment, the other two stating precisely what the
> fixture's own coverage record already asserted (`declared_pairs=["X"]`,
> `observed_pair_channels={"X": ["trade"]}`); the other nine pre-existing test files are
> md5-identical. Reverting just those two lines breaks exactly one test, and on the fixture's own
> hand-built incompleteness — the same situation its PRE-EXISTING `coverage_evaluated = True` line
> already handled. The masking hypothesis was tested directly and refuted: restoring cycle-2's raw
> (un-merged) gap-span defect turns the union tests RED under BOTH the edited and unedited
> fixture, so the edit costs nothing in detection power. And the **NO NUMBER MOVED** proof is real:
> a healthy 3-pair 2-hour capture built with the production writer renders **byte-identical
> artifacts pre- and post-cycle-4 except for the two added provenance fields on one line** — every
> number, table and count unchanged. Node-ID sets confirm nothing was removed or renamed (152 → 173,
> `comm -23` empty, all 21 additions in `test_declared_pairs.py`), every cycle-3 constant and pinned
> relationship is intact, and the new defense has genuine mutation coverage (three mutants, all RED
> — including `declared_pairs := meta.pairs`, the circular union D66 forbids).
>
> **But the regression lens also found that the new defense OVER-refuses, in a direction this
> lane's own suite is structurally unable to see:**
> - **MAJOR — a legitimately empty `trade` cell is treated as identical to a dead `book` cell, and
>   refuses the whole render.** `capture.py:581-586` is direction-symmetric. `book` absent while
>   trades flow is near-certainly a broken subscription (an actively-printing pair must have a
>   book); `trade` absent while `book` flows may simply mean **no prints in the window**, and that
>   pair's own continuous book flow *proves* the socket and symbol are alive. Run end-to-end: a
>   healthy 3-pair 2h capture plus a 120s tail file in which `SOL/USD` delivered 25 book events and
>   zero prints → `rc=1`, **no artifact**, and every number from the good capture discarded. The
>   refusal's own stated premise does not even hold for that pair — it has a coverage record, a
>   cadence row and book-derived mids, so it is not "silently absent from every table". There is no
>   escape hatch: `--pairs` filters `run_shadow`'s results (`shadow_engine.py:247`) but
>   `render.py:68` asserts over the whole `CaptureMetadata`. **Critically, this fires only when
>   `channels` IS declared — i.e. only on real recorder captures, never on a synthetic fixture — so
>   the 173-test suite cannot reach it.** Direction: make the per-cell check asymmetric (missing
>   `book` on an observed pair refuses; missing `trade` on a pair whose `book` cell is present and
>   covered is DISCLOSED, not refused).
> - **MAJOR — the reconciliation runs DAYS too late, with no in-run signal.** The declared-vs-observed
>   comparison exists only at read/render time (`capture.py:686` is the sole site); `recorder.py`
>   never compares an observed wire `symbol` (`recorder.py:170,198`, taken verbatim with no
>   normalisation) against `cfg.pairs`. Demonstrated with Kraken's legacy `XBT/USD` spelling against
>   a config declaring `BTC/USD`: the capture refuses wholesale at read time while
>   `n_schema_mismatch = 0` and `n_unrecognized_frames = 0` throughout — **the recorder saw no drift
>   at all.** Refusing is the right call; the defect is the timing. Three to seven days of recording
>   arrive complete, unrenderable and not cheaply repeatable, which is exactly the pressure that
>   later gets relieved by widening a bound. Direction: run `pair_reconciliation_gaps()` at capture
>   START — after the first heartbeat cycle — abort loudly, and write the outcome into the capture so
>   it reaches the artifact.
> - **MINOR — the refusal's prescribed remedy is unachievable and self-contradictory for the thin-cell
>   case** (`capture.py:706-711`): re-cutting to a *narrower* window can only reduce a print count, so
>   that remedy is monotonically wrong, and "re-run the recorder against a subscription that does
>   deliver" **is** dropping the pair from the declaration, which the very next sentence forbids. The
>   operator's only reachable action is the one the message calls laundering. Remedy text needs
>   splitting per failure kind.
> - **MINOR — fixing the above requires editing a defense test** (`test_declared_pairs.py:184-210`
>   asserts the per-cell direction symmetrically), which on this lane reads as weakening. Named now
>   so the fix is not later blocked by its own suite; add the mirror case as an explicit expectation
>   whichever way it is ruled.
> - NIT: the implementer's "21 tests (26 with parametrization)" is 21 collected node IDs (16
>   functions, one 6-way parametrized).
>
>
> **DO NOT LAUNCH THE CAPTURE on the strength of cycle 4.** The Kraken v2 schema is still
> unverified against the live feed (planner-side, blocking, unchanged from D66) — and finding 4
> means the reader would not reliably tell you if the feed went dark after a brief success, which
> is the single most likely first-launch failure. D67 carries a recommendation that separates
> RECORDING (now the well-verified half) from ANALYSIS (where every open BLOCKER lives).

> **STATUS (2026-08-27, factory run 43): HALTED ON THE THREE-CYCLE RULE — NOT SHIPPED. Work is
> BUILT, GREEN ON ITS OWN SUITE (152 passed), and LEFT IN THE TREE UNCOMMITTED. See D66.**
> Preserved at `lore/handoff_2026-08-27_factory_run_43_907_halt.patch` (5553 lines, 20 files).
>
> **Phase 1 (shadow) is complete and phase 2 exists only as its refusal gate and rails — an
> orchestrator scope split, flagged for the operator.** `place_order()` raises `NotImplementedError`
> as its first statement; **no code path in this work can transmit an order**, re-enumerated after
> every cycle (the only network primitive is `websockets.connect` against a hardcoded public
> `wss://ws.kraken.com/v2`, sending exactly two public subscribe frames; `kraken_client`'s
> `add_order`/`_private_request` are never imported; nothing anywhere catches `NotImplementedError`).
> No credential is read, logged or referenced. Live order placement was deliberately deferred to its
> own verified increment.
>
> **Why it halted, and the pattern that is the actual finding.** Three cycles, three real defects,
> and every one was the SAME failure family — *silent absence presented as completeness* — at a
> successively finer granularity:
>
> | cycle | defect | granularity |
> |---|---|---|
> | 1 | NaN markouts rendered as real numbers **and** a false `BELOW measured` verdict (`tier >= nan` is always False); overlapping recorded gaps double-counted coverage, so `assert_complete()` passed over a real 200s hole | per-sample, per-gap |
> | 2 | sub-threshold accumulation: events 119s apart (just under a 120s threshold) for 10 days → `is_complete() → True`, full artifact, `recorded_gaps=0`. And `_find_unrecorded_gaps` counted **heartbeats** as events, so a live socket with a throttled book/trade subscription defeated both defenses at once | cumulative, per-capture |
> | 3 | a declared pair with **ZERO** data-bearing events for the entire capture evades all three defenses: it never enters `data_ts`, so there is no timeline to find a jump in and no coverage record to be "degraded". Named in the top-line provenance, then silently absent from every downstream table | per-pair, total absence |
>
> Cycle 3's own docstring shows it reasoned carefully about a pair *degrading partway* (fixed by
> measuring coverage over the capture's **global** span) — the case it did not reach is a pair dead
> from t=0. **Each fix was genuinely better than the last and each was still one granularity short.**
> That is why this is a decision rather than a fourth patch.
>
> **What cycle 3 built and got right** (and it is substantial — reuse it, do not rebuild):
> three defenses that do not subsume one another (per-jump over all timestamps = "was the socket
> dark?"; per-jump per-pair over book/trade only = "did we observe the market?"; cumulative
> `coverage_by_pair`). Recorded gaps leave the **denominator** rather than being double-punished,
> since `covers()` already drops markouts across them, so the gated quantity is the *undisclosed*
> remainder. Union arithmetic throughout, so overlapping spans are never summed. Coverage window is
> `BAR/4` and **must** be tighter than the gap threshold — with window == threshold, any cadence just
> under the threshold reads as fully covered and the aggregate check becomes **structurally unable to
> fire**; that relationship is pinned by a test. `is_complete()` is fail-CLOSED on
> `coverage_evaluated`. `read_capture` refuses `gap_detect_threshold_s > BAR_INTERVAL_S` and the
> default drops 120 → 60, because a *loose reader* threshold is the dangerous direction while a
> stricter recorder is safe. The coverage bound is deliberately **not** a CLI flag — "a dial on a
> fail-loud bound is a bypass with extra steps" — and that is tested. A single declared clock basis
> (local receipt), with `read_capture` refusing any capture declaring another. Schema drift split into
> three counters, persisted and surfaced, with a capture lacking counters saying the counts are
> **UNKNOWN, not zero**. And the trade-print fallback is **excluded** from estimand 1 and reported as
> a labelled secondary table, because after a fill goes badly aggressor flow is one-sided with the
> move, so the print biases measured adverse selection **too pessimistic** — the one direction a
> contaminant may not push, since it is the direction of this row's own preferred conclusion.
> *(Orchestrator ruling: that exclusion CONFORMS to the pre-registration rather than amending it —
> "mid price at t+h" never authorised a print as a mid — and both readings remain evaluable from the
> artifact because the excluded samples are reported with full statistics.)*
>
> **The fix for the halting MAJOR is small and specified**, which is why D66 asks a question rather
> than assuming: require every pair in `meta.pairs` (or every pair the recorder was configured to
> subscribe to) to have at least one coverage record / one data-bearing event before
> `assert_complete()` passes. Related and worth folding in: `assert_complete()` currently trusts the
> *shape* of `coverage_by_pair` rather than verifying it covers every pair that should have been
> checked — 4 of 5 hand-crafted metadata shapes bypassed it (not reachable from the real
> `read_capture` path today, but it is the same missing cross-check).
>
> **Also disclosed, deliberately open:** `covers()` still uses the heartbeat-inclusive span, bounded
> downstream by the staleness cap into `n_markouts_dropped_no_mid` (judged adequate — no fabricated
> number results). The aggregate bound is per-pair pooled over book+trade rather than per-channel,
> because the channels have different natural cadences; the shape pooling could hide (book dies,
> trades stay dense) is caught from the other side by the mid-source split, and a test **proves that
> interlock** rather than arguing it. The 2% / 15s bound is a judgement anchored to the estimand, not
> an empirical calibration — no live cadence data exists to check it against, and the refusal message
> says re-cut or re-run, **explicitly not widen the bound**.
>
> **DO NOT LAUNCH THE CAPTURE YET, for a second independent reason:** Kraken v2's message schema is
> **unverified against the live feed** (the agents had no network). The failure symptom to watch in the
> first minutes of any launch is "connected but zero events recorded". Cycle 3's drift detector makes
> that loud rather than silent, but it has never met the real wire.
>
> **Verification:** security ×1 + fail-closed ×3. Suite **152 passed** (53 → 121 → 152), selftest
> 12/12, `py_compile` clean on all 19 lane files. Five mutations RED, plus two proofs needing no
> mutation at all: a 59s-cadence fixture asserting both per-jump defenses are clean so only the
> cumulative check can be firing, and a re-read of that identical fixture with the bound relaxed to
> 0.999 that passes — making the refusal attributable to the bound and nothing else.

# TASK_907 — the fill-model probe: measure realized adverse selection (the record's first empirical anchor)

**Status**: READY 2026-08-25 (planner; seeded by the operator's D65 ruling — "(b) then (a)":
the probe runs, THEN closure is accepted with its results in mind).
**Size: M. Risk: phase 1 ZERO (shadow, no orders); phase 2 is REAL MONEY, micro-notional,
operator-armed, hard-capped.** The phase boundary is the row's core safety feature.

## Why this row exists

The maker-fill model (OHLC-touch + flat adverse selection) has been the load-bearing,
never-validated assumption of the entire 806→809→897 lineage — named "the known weak point"
since 2026-06, and the 897 ALL FAIL verdict is robust to it in the *optimistic* direction
(0/188 cells clear even at adverse selection = 0; OHLC-touch overestimates fills, so reality
is worse). **This probe calibrates the model against reality. It cannot reopen the verdict** —
no realized markout distribution rescues a band whose nearest cell is −5.045 with the penalty
set to ZERO. What it produces is the first EMPIRICAL answer to "how adverse is a maker fill at
minute scale, really?" — the number the record has asserted without measuring since June.

## Pre-registered measurement

- **Estimand 1 — adverse selection:** the markout distribution of maker fills: for each fill
  at price p at time t, side-signed mid-return to t+{1m, 5m, 20m, 1h}. Report the full
  distribution (deciles), not just the mean. The model's flat {0, 5, 10} bp term is compared
  against the measured distribution's location.
- **Estimand 2 — fill rate:** realized fill rate of a limit posted at bar close vs the
  OHLC-touch assumption (a touch is not a fill — queue priority is the whole question).
- **Pairs:** the sweep universe's most liquid — BTC/USD, ETH/USD, plus 2–4 majors by spread.
- **Shadow phase truthfulness bound:** touch-conditional markouts are the OPTIMISTIC bound on
  adverse selection (real fills are selected worse than touches). If even the optimistic bound
  meets/exceeds the model's 5 bp, the model stands calibrated-pessimistic and phase 2 adds the
  queue-priority correction for the record.

## Phase 1 — SHADOW (build + run first; no orders, no keys, no money)

1. A websocket recorder (Kraken public feed: book top + trades) on the chosen pairs, running
   continuous capture in tmux under `safe_run` on gpu-server, **fail-loud on gaps** — every
   capture gap is recorded, never silently interpolated (the 893 lesson).
2. The shadow engine: at each minute bar close, posts a virtual limit at close; marks it
   filled-by-touch per the model's rule; records the subsequent markouts. Duration target:
   ≥3–7 days (thousands of shadow fills across regimes).
3. Artifact: `experiments/907_live_fill_probe/` + `runs/907_live_fill_probe/shadow_<ts>.md`
   (untracked `--out`), with the estimand distributions and the model comparison.

## Phase 2 — REAL (only via explicit operator arming; nothing in phase 1 can leak into it)

Real post-only limit orders at venue minimums (micro-notional), placed by the same engine.
Hard rails, ALL enforced in code, not convention: per-order notional cap, total-notional cap,
max open orders, auto-cancel after 1–2 bars, a kill file checked every cycle, post-only
enforced at the API level, **never a market order, ever**. Arms only when the operator passes
an explicit arm flag AND a budget cap on the command line. Budget parameters (per-order cap,
total cap, pairs, duration) are the operator's input, recorded in the artifact before the first
order. Credentials via the existing `KRAKEN_API_KEY`/`KRAKEN_API_SECRET` env plumbing
(`mcts_trader/data/kraken_client.py`) — keys are never written to any artifact or log.

## Acceptance (gates)

- (i) Shadow artifact produced with the pre-registered estimands; gaps fail-loud.
- (ii) Phase 2 refuses to start without the arm flag AND budget cap; a fixture-mode test proves
  the rails (kill file, caps, auto-cancel, post-only) fire.
- (iii) The artifact states in its first box: **this probe calibrates a model; it cannot reopen
  897's ALL FAIL** (the verdict's robustness argument quoted).
- (iv) No number in the artifact comes from a run whose capture had an unrecorded gap.

## Out of scope

Any strategy trading. Any use of the probe's numbers to re-litigate 897/809's band. Venue
changes. Reopening the minute surface (that is D65's closure, accepted after this row's
results are in hand).

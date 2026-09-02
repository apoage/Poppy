> **STATUS (2026-09-02, factory run 50): ADDENDUM B2 HALTED at cycle 2 of 3. NOT SHIPPED,
> NOTHING COMMITTED. Blocked on operator decisions, not on implementation.
> ⚠️ BOTH DECISIONS RULED 2026-09-02 (operator; D65-amend-2 + D77 in DECISIONS) — THE ROW
> RESUMES: (1) ALL FOUR pre-registered horizons report (the final capture's premise change is
> accepted); (2) the book-silence exclusion is DEFECT-SIGNAL-GATED with a 1.0% numeric
> backstop — silence excludes only with an instrument-defect signal (keepalive overrun beyond
> tolerance, trades through an unchanged book, schema drift); plain silence with clean
> checksums is MARKET STILLNESS, disclosed never excluded (BTC's frozen mid passed
> falsification). The two open BLOCKERs (fail-closed leak; the 4000bp-publishing repro)
> still must close before ship — the rulings unblock, they do not ship. Cycle 3 is available.**
>
> Work preserved in the live tree AND at
> `lore/handoff_2026-09-02_factory_run_50_907B2_halt.patch` (2858 lines, sha256
> `b11c1bbe…`, `git apply --check` OK vs HEAD `96b59eb`). Suite 302 → 342, selftest 12/12,
> fixture `b0f8de8b…` unchanged. Full record: `lore/handoff_2026-09-02_factory_run_50.md`.
>
> **Why it halted.** Five BLOCKERs across two cycles. Every one of B2's three disqualifiers was
> denominated in a quantity that is not the one it needs to gate — the same error that halted
> run 46 and forced the D73 split. Cycle 3 was available and deliberately NOT spent: the row
> implements a question whose qualitative shape D73 ruled and whose **three quantitative bounds
> have never been ruled**, and its governing decision is now refuted (below).
>
> **DECISION-CHANGING 1 — D65-amend's premise is REFUTED by the final capture.** Run 49 recorded
> "20m and 1h are STRUCTURALLY 0-SAMPLE", reasoning the longest gap-free stretch was 561.8s while
> `covers()` needs the whole `[t, t+h]` clean; D65-amend ruled "two horizons suffice" on that.
> Measured on both files with the same code: the SNAPSHOT's longest gap-free stretch is **561.8s**
> (run 49's figure reproduced exactly — its measurement was right for the file it had), the
> **FINAL capture's is 9,670.9s** (top five 9671/7198/6668/5658/4131s). The final capture added
> only 39 gaps across its last 46,179s — the stall regime nearly stopped, consistent with run 48's
> pending `systemctl mask --now udisks2`. **At HEAD, on the FINAL capture, all four pre-registered
> horizons produce samples on all four pairs** (BTC 1m 8557 / 5m 4536 / 20m 962 / 1h 467).
> D65 pre-registered four horizons; this capture supports four.
>
> **DECISION-CHANGING 2 — the probe's verdict turns entirely on three never-ruled bounds.** The
> real capture REFUSES under the shipped constants, every pair excluded for one reason:
> unexplained book silence against a bound of **0.0000%** ("any violation disqualifies").
> Measured: BTC 303.8s = 0.0952%, ETH 561.1s = 0.1758%, SOL 97.8s = 0.0307%, XRP 488.8s = 0.1532%
> of accounted time; worst single interval 15.2–19.6s. **At any bound ≥ 0.2% all four pairs
> publish** — a labelled counterfactual render at 0.2% (scratch lane copy; live constant verified
> still `0.0`) produced a full 350 KB artifact, four pairs, four horizons. The factory did not
> pick that number: it decides D65's answer.
>
> **What cycle 2 got RIGHT, verified on the real capture.** `book_uncovered_fraction` reads
> **0.0019–0.0082%** across all four pairs where cycle 1 read ~46% for BTC. The wire-evidence
> denominator — a pure keepalive echo is an event whose FULL payload `(bid, ask, bid_sz, ask_sz)`
> repeats — correctly separates "the market was still" from "we could not see the market". BTC's
> 138,449s of frozen mid stays disclosed via `stale_book_s` without counting as un-observed.
> **BTC's frozen mid is real market stillness**: no un-freeze catch-up (in-run trade drift
> 0.006bp vs a 0.257bp jump), frozen runs quieter than average and printing outside the touch LESS
> than baseline (10.0% vs 23.3%), ETH freezing precisely when locked at a 1-tick spread.
> Correctly denominated, item 2b became a REAL signal (flags 10/1222, 12/773, 24/600, 10/441
> frozen runs, worst 66.7–94.0% outside vs baselines 16.9–30.9%, `venue_ts`-restricted).
>
> **TWO BLOCKERS REMAIN OPEN IN THE PRESERVED TREE — do not ship it as is.**
> (1) `shadow_engine.py:355-397`: flagged-run membership tests only the fill/anchor trade's
> timestamp, never whether the posted limit (`bar.close`, `:371`) came from a bar composed of
> flagged-run trades. The cycle-1 repro **still publishes `median 4000.52 bp` at rc=0**, now
> dressed with an honest-looking "8 of 16 candidates excluded" line that gives FALSE confidence.
> The shipped test asserts the same criterion the implementation uses — one shared blind spot.
> (2) `cli.py:118-134`: a pair disqualified in EVERY contributing capture triggers a B4 collision
> refusal, discarding every healthy pair in both captures — the D73 over-refusal shape one layer
> up. Also open: FIX 6's decline rationale is mechanistically false and ships in the artifact
> (`render.py:685` — `_build_bars_and_mids` buckets on local receipt `ts`, never `venue_ts`), and
> echo detection is gated on `declares_keepalive` so an undeclared emission policy disables the
> wire-evidence detector (bounded: the production recorder always declares both keys).
>
> **Resume:** operator rules the bounds and re-rules D65-amend → re-apply the patch → close the
> two BLOCKERs → re-render the real capture (14.8 GB peak RSS, 3:29; TASK_914 would cut this).

> **STATUS (2026-09-01, factory run 49): ADDENDUM B1 SHIPPED. B1-vii DISCHARGED BY RENDER —
> THE LIVE CAPTURE ANALYSES.**
>
> Suite **271 → 302** (31 added across 3 cycles, 0 removed or renamed by cycles 2–3; the only 3
> removals vs HEAD are cycle 1's documented renames). selftest 12/12. Fixture `b0f8de8b…`
> unchanged. B-iv proven byte-identical on six independently-built captures across two lenses.
> Boundaries B1/B3/B4/B5, the B2 span fixes, the over-refusal asymmetric fix, terminal reconnect
> and all six in-gate defects are in. The cycle-3 per-pair stale-book EXCLUSION is carved out
> per D73 and remains B2's.
>
> **The BLOCKER that had been holding the whole probe hostage was a ~12 µs artifact of our own
> recorder, not a corrupt ledger.** `_record_gap` sampled a SECOND, strictly later clock for a
> gap's `end_ts` than the one its frame's events were stamped with, so the event that TERMINATED
> a `clock_jump` landed inside its own gap — 584 times, max **32.902 µs**. A further 14 sat in NO
> individual gap: `_merge_gap_spans` unions two gaps that merely TOUCH at an event's timestamp and
> the seam read as containment. The refusal text asserted this shape "is not one it can produce
> under a monotonic clock" — a true condition with a **false cause**. Fixed at both ends; a judge's
> ablation on the real capture forced the split **598 → 584 → 0**.
>
> **D57, fourth instance, closed at the CLAIM not just the code.** The staleness key was
> `(bid, ask, bid_sz, ask_sz)`, finer than the mid it protects, so a frozen price with jittering
> sizes published `1m median 2.48 bp` on a mid that never moved. Re-keyed to `(bid, ask)` — but
> re-keying does NOT make the gate fire (coverage is denominated over the UNION of book and trade;
> measured 0.00132 against a 2% bound), so every shipped string was narrowed instead: the
> whole-capture refusal reaches stale book time ONLY where that pair's trades stopped too, and a
> frozen or silent book masked by live prints is **not caught anywhere in B1**. "Keepalive-driven"
> was itself unsupported and is retired for FROZEN-MID BOOK TIME.
>
> **Two holes are deliberately un-closed and pinned by tests that say so in their docstrings:** a
> book that goes silent, and a mid that OSCILLATES over a small value set, each while trades keep
> flowing. Chasing period-N was refused on purpose — TASK_891's halt pattern. B2's book-mid
> freshness term is the denomination-correct instrument, and is now load-bearing, not polish.
>
> **DECISION-CHANGING, FOR THE OPERATOR — this row's real output:**
> * **20m and 1h are structurally 0-SAMPLE on all four pairs.** Longest gap-free stretch in
>   3.586 days: **561.8 s**; `covers()` needs the whole `[t, t+h]` clean. Rendered as explicit
>   not-evaluable rows (B3 holding). Only 1m and 5m carry numbers.
> * **Gap-straddling candidate drop: BTC 70.63%, ETH 71.20%, SOL 70.91%, XRP 71.30%.**
> * **BTC/USD reads 0.126% un-observed and 127,071 s of 273,233 s (46.5%) FROZEN MID on the same
>   page** — 2,016,348 of 2,279,026 book events repeat the previous `(bid, ask)`, longest frozen
>   run 779 s — with the 1m decile spine (p40 −0.01 / p50 0.01 / p60 0.05) a frozen mid produces.
>   ETH/SOL/XRP do not show it. **D65 pre-registered four horizons; this capture supports two, on
>   a flagship pair whose mid did not move for nearly half its accounted time. Whether that still
>   answers "just to know" is the operator's call.**
> * Stall regime confirmed FROM THE CAPTURE, not dmesg: 566 `clock_jump`s, median inter-start
>   delta **600.0 s exactly**, median duration 61.53 s, 10.9% of span (run 48's udisksd finding).
>
> **NOT CLOSED, named:** `read_capture` materialises the whole file (`capture.py`,
> `lines = list(...)`) — 11.8–12.6 GB RSS for 1.248 GB, ~9.5×, read twice; the capture is still
> growing and a 7-day file implies ~24 GB against a 32 G cap. Its own row. B2 keeps: the
> book-mid-only freshness term, the per-pair stale exclusion, the TRUE mid-age disclosure
> (`mid age` is age by TIMESTAMP, not by VALUE — the artifact now says so), and D74 receipt-lag
> denomination.
>
> Cycles: 3, exactly at `halt_after_failures`, with cycle 3 pre-declared as the last and bounded
> to text plus one pinning test.

> **STATUS (2026-08-29, planner; D73 RULED by operator): ADDENDUM B IS SPLIT IN TWO. B1 SHIPS
> NOW. B2 IS RE-CUT. THE LIVE CAPTURE IS NEVER ANALYSED AT HEAD.**
>
> Run 46 took Addendum B through three cycles (265 tests passing at halt) and halted at the
> pre-committed guard on a denomination error: **the cycle-3 stale-book exclusion is denominated
> in DATA-BEARING coverage, which a trade event satisfies, while estimand-1 is computed from
> BOOK MIDS ONLY.** Measured: a pair whose book feed dies at t=1800 while trades keep flowing
> reads `uncovered_fraction = 0.0000`, is neither degraded nor excluded, and publishes an
> estimand-1 table (`1m | 119 samples | median −325.89 bp`) built entirely on a mid frozen for
> 75% of the capture — with the freshness disclosure reading `mid age p50 = 0.000s` because the
> keepalive makes a two-hour-stale mid look one second old. Same root: the classifier never
> consults the two instrument-defect signals the same read pass already computes
> (`book_keepalive_overruns` — 288 at 25s against a declared 5s floor — and trades walking
> +230bp through an unchanging book). This is D67's second clause violated by the fix built to
> enforce it. **Honest attribution: the pair-level book-freshness hole is PRE-EXISTING** (a
> BTC-only capture passes at HEAD too); cycle 3 removed an accidental whole-capture refusal
> while its replacement guards the wrong quantity.
>
> **THE SPLIT (D73, on the 904A precedent):**
>
> **ADDENDUM B1 — SHIP NOW** (`907B-1`): boundaries **B1** (per-session declarations), **B3**
> (zero-sample pre-registered horizons render as zero-sample rows), **B4** (capture-set
> clobber), **B5** (`--pairs` reconciliation), the **B2 span fixes** (edge-hole anchoring,
> coverage denominated in the CAPTURE span, `accounted_s=0` fail-closed, the `NO CLAIM`
> sentinel), the **over-refusal asymmetric fix** (missing book refuses; missing trade on a
> book-covered pair is disclosed), **terminal reconnect**, and the in-gate defect list (twins'
> identity test, `channels: []` consistency, the channel-side mirror, per-kind remedy text,
> the `book_checksum_mismatch` enum entry). All independently verified across run-46's nine
> passes; B-iv (no number moves on a healthy capture) proven twice against a HEAD worktree.
> **Carve-out: the cycle-3 per-pair stale-book EXCLUSION does not ship in B1.** Interim
> behavior for a stale-book pair is the pre-cycle-3 whole-capture refusal — fail-closed, and
> the artifact/banner must SAY the stale-book exclusion is pending B2 rather than implying
> coverage. Gates B-i…B-vi apply as written, minus the stale-book piece; B1's own gate is
> **B1-vii: the live capture renders under B1 code** (a smoke render of the real file, or a
> documented refusal naming the true reason — either is a pass; silence is not).
>
> **ADDENDUM B2 — RE-CUT** (`907B-2`, small): the stale-book exclusion alone, around the right
> question — **what makes a BOOK-MID estimand trustworthy**: a book-freshness term
> (`stale_book_s / accounted_s`, from BOOK timestamps only, independent of whether trades
> carried coverage) plus the two defect signals (keepalive overruns; trades walking through an
> unchanging book) as exclusion disqualifiers, per-pair, so one quiet pair stops voiding a
> whole multi-day report. Gates: the t=1800 book-death fixture (book dies, trades flow) EXCLUDES
> the pair or refuses — never publishes frozen-mid markouts; the freshness disclosure reads the
> TRUE mid age; the classifier consults both defect signals; no number moves on healthy
> captures (B-iv proof repeated).
>
> **B2 AMENDED 2026-08-30 per D74: receipt-lag denomination joins the scope.** The live
> capture's receipts run 2.4–8.7s late (median) in the highest-volume hours — load-dependent,
> via the recorder's per-line fsync — against markout horizons of 60s+. Every trade event
> carries `venue_ts`, so the distortion is MEASURABLE per event: B2 must report the
> receipt-lag distribution per hour, quantify the markout-timing distortion it implies at each
> horizon, and re-time the trade-anchored diagnostics on `venue_ts` where the field exists.
> Book events have no `venue_ts` in the emitted record — the honest statement is a measured
> bound (trades' lag distribution as the proxy), never an assumed zero. The ~61.5s journal
> stalls themselves need NO new exclusion machinery: they are recorded gaps, wall-clock-locked
> (uncorrelated with market state), and `covers()` already drops markouts across them.
>
> **Resume mechanics:** re-apply `lore/handoff_2026-08-28_factory_run_46_907B_halt.patch` FIRST
> (2108 lines, sha256 `67b6a3d8…`, `git apply --check` verified 2026-08-29), then carve the
> stale-book exclusion out for B1 and re-cut it as B2. The patch is the base for both halves.
>
> **BINDING WARNING: HEAD IS NOT A SAFE RESTING PLACE.** At HEAD the 7200s/130s capture PASSES
> at `0.000%`, horizons vanish silently, captures clobber order-dependently. The analysis of
> the live capture (running since 2026-08-28 07:07 CEST) runs ONLY on B1-shipped code or later
> — never at HEAD, never on a worktree that merely contains the patch unshipped.

> **STATUS (2026-08-27, factory run 45): ADDENDUM A SHIPPED. ADDENDUM B SPECCED AND OPEN.**
> The row is now two pieces and only the first is done.
>
> **ADDENDUM A — the book-channel schema fix (the LAUNCH-GATING half) — SHIPPED.** Built on 5,737
> RAW frames captured from the live Kraken v2 public feed on `gpu-server` (45s, BTC/USD + ETH/USD,
> depth 10, read-only, no credentials), so the fix rests on the wire rather than on recollection.
> Two defects, not one: (1) **5,661 of 5,665 book rows carry an empty side**, so the old
> both-sides-required check rejected 99.93% of the channel — the planner's "3,538 mismatches in
> 75s"; and (2) **Kraken v2 updates are INCREMENTAL price-level deltas, not top-of-book**, so the
> four rows the old code DID accept were recording a wrong top of book as a real observation
> (measured: a delta at 2505.76 while the true best bid was 2505.81). **Fixing only (1) would have
> converted 5,661 honest refusals into 5,661 silently wrong mids — strictly worse than the bug.**
> The book is now a per-pair, per-connection maintained state machine reconciled against the
> venue's own CRC32 on every row (proven 5,665/5,665 against the live wire before a line was
> written; `parse_float=str` is load-bearing, the checksum is over wire STRING forms).
>
> **Panel: 1 BLOCKER + 4 MAJOR + 4 MINOR, all closed.** The BLOCKER is the one worth remembering:
> the checksum reconciled the maintained STATE while the emitted event was a SEPARATE derivation,
> and nothing reconciled the two — so reinstalling the original defect left the suite GREEN while
> corrupting **482 of 590 emitted tops** on real frames. Worse, the single hand-checked frame (300)
> happened to be an empty-bids row whose lone ask delta IS the best ask: the one shape where the
> bug is invisible. **One hand-check is not a proof.** Closed STRUCTURALLY — `_book_view` returns
> the checksum and the top from one ordered structure, so the second derivation no longer exists —
> plus a whole-fixture check of all 590 emitted events against an independent re-derivation.
> Orchestrator re-applied the mutation post-fix on an isolated copy: RED at frame 9
> (ETH/USD 2505.81 → 2505.75). 17-mutation table A–P all RED but one disclosed equivalent mutant.
>
> **A spec error of the orchestrator's, corrected in A.4(4) rather than blamed on the code:** the
> emission clause described the keepalive as a TIMER and justified it with "otherwise a quiet book
> reads as a dark socket". A timer would write a top of book no frame confirmed at that `ts` — an
> inference from absence recorded as an observation, on the row that has halted four times on
> exactly that. Emission is FRAME-DRIVEN and that is correct: a quiet book emits nothing and must
> read as un-observed.
>
> **Verification:** suite **173 → 217 passed**; `trader selftest` 12/12; wire fixture byte-unchanged
> (sha256 `378fa3dc…`); safety invariant re-checked (`except` 7→7, no `open`/`subprocess`/
> `os.environ`, `WS_URL` unchanged, two public subscribe frames, one `ws.send`). Panel:
> correctness + fail-closed (opus) + regression; fix pass at opus.
>
> **STILL GATED — DO NOT LAUNCH THE CAPTURE ON THIS ALONE.** A.4(5)'s live re-smoke
> (`n_schema_mismatch ≈ 0` against the real feed, watched) is planner-side and is NOT discharged
> here; the fixture is 45s of one afternoon, not proof the schema is stable. D67(b)'s four launch
> preconditions all still bind, and two of them are now MORE load-bearing, because the defects they
> defend against live in **Addendum B and are not fixed**: (ii) the hand-computed data-span vs
> capture-span comparison — finding 4 (B2) is open, so `is_complete()` remains blind to an edge
> hole, and it must not be trusted; and (iii) the FIXED pair set — the resume-path declaration loss
> (B1) is open, so the operational discipline is the only defence in place.
>
> **ADDENDUM B — the D67(a) five-boundary restructure — SPECCED, NOT STARTED.** Written below in
> full. **All five boundaries plus four inherited in-gate claims were reproduced FIRST-HAND against
> the current tree this run** (B1 declaration lost across sessions; B2 98.188% un-observed reported
> as 0.000% with the gate PASSING; B3 a pre-registered horizon silently omitted; B4 the LAST
> `--capture` silently wins, order-dependently; B5 `--pairs` exit 0 with a gate-passing artifact and
> zero estimands) — measurements, not inherited claims. Recommended as the next run's row 1.

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
  the rails (kill file, caps, auto-cancel, post-only) fire. **AMENDED 2026-08-27 per D68 (run 44
  measured this NOT MET — the caps were opt-in and defaulted OFF): the caps are MANDATORY at
  arming (`--budget-cap-usd` AND `--max-order-usd` both required, `check_total_cap` called
  unconditionally, enforced at the order boundary); the safety pins land in TASK_910, and until
  910 ships the honest statement is: rails built, cap enforcement pending 910, and phase 2
  cannot arm before both.**
- (iii) The artifact states in its first box: **this probe calibrates a model; it cannot reopen
  897's ALL FAIL** (the verdict's robustness argument quoted).
- (iv) No number in the artifact comes from a run whose capture had an unrecorded gap.

## Out of scope

Any strategy trading. Any use of the probe's numbers to re-litigate 897/809's band. Venue
changes. Reopening the minute surface (that is D65's closure, accepted after this row's
results are in hand).

---

# ADDENDUM A (2026-08-27, factory run 45) — the book-channel schema fix, on MEASURED wire evidence

The D67(a) re-cut's launch-gating half. The planner's 75s smoke measured "3,538+ mismatches";
this run captured **5,737 RAW frames (45s, BTC/USD + ETH/USD, depth 10, srv, public feed, no
credentials)** and characterised them, so the fix below rests on the wire and not on memory.
Raw dump preserved at `experiments/907_live_fill_probe/wire_fixture_kraken_v2_2026-08-27.jsonl`
(first 600 frames, sha256 `378fa3dc8650df23882954f828cd55be2bbf4cfab63ddec28ed53151592cb831`).

## A.1 What the wire actually sends

| frame | count (45s) |
|---|---|
| `book` / `update` | 5,663 |
| `heartbeat` | 44 |
| `trade` / `update` | 23 |
| method-response (no `channel`) | 4 |
| `book` / `snapshot` | **2** |
| `status` / `update` | 1 |

Book row keys are IDENTICAL for snapshot and update: `{symbol, bids, asks, checksum, timestamp}`.

**Cause of the mass mismatch — one side is empty on almost every update.** Distribution of
`(type, len(bids), len(asks))`: `(update,1,0)` ×1811, `(update,0,1)` ×1730, `(update,2,0)` ×977,
`(update,0,2)` ×852, `(update,3,0)` ×197, `(update,0,3)` ×92, `(snapshot,10,10)` ×2, plus 5
mixed. **5,661 of 5,665 book rows have an empty side**; only 4 carry both. `recorder.py:170`
(`if not pair or not bids or not asks: schema_mismatches += 1`) therefore rejects 99.93% of the
book channel. The 2 snapshots and 4 both-sided updates are the "25 parsed book events" the
planner's smoke saw.

## A.2 The SECOND defect, not previously named, and the more dangerous one

**Updates are INCREMENTAL price-level deltas, not top-of-book.** Measured directly: an update
row `bids: [{price: 2505.76, qty: 1.33933603}]` arrived while the snapshot's best bid was
**2505.81** — 2505.76 is the *third* level down. `recorder.py:174` reads `bids[0]["price"]` as
the top of book. So the four frames the current code DOES accept do not merely under-count;
**they record a wrong top-of-book as a real observation.** Same family as everything else on
this row: a declared quantity ("top of book") never reconciled against what was observed.
Fixing only the empty-side rejection would convert 5,661 honest refusals into 5,661 silently
wrong mids — strictly worse than today. The book must be MAINTAINED, not sampled.

## A.3 The venue hands us a free declared-vs-observed check, and it is PROVEN to reconstruct

Every book row carries a `checksum` (CRC32 over the top-10 book). Reconstructing the book from
the snapshot, applying updates (`qty == 0` deletes a level, otherwise set), trimming each side
to depth 10, and hashing — asks top-10 then bids top-10, each level `price+qty` with the
decimal point removed and leading zeros stripped, concatenated — reproduced the venue's value
on **5,665 of 5,665 rows (100%)**. This is the D67 principle at the wire boundary: the venue
declares the book state it believes we hold, and we reconcile our own against it on every
single frame.

**Load-bearing implementation constraint:** the checksum is over the wire's STRING forms.
`json.loads(text)` destroys them (`2.0` ≠ `"2.00000000"`). Frames must be decoded with
`json.loads(text, parse_float=str, parse_int=str)`. Verified: `_finite_float` already accepts
strings (`float("1.23")`), and `isinstance(value, bool)` is unaffected, so the existing trade
path is safe under this change — but it must be tested, not assumed.

## A.4 Required behaviour

1. **Per-pair book state**, keyed by symbol, held across frames on one connection.
   - `snapshot` REPLACES the pair's book. `update` applies deltas as in A.3.
   - An `update` for a pair with **no snapshot yet** is a `schema_mismatch`, never applied.
   - Book state is **per-connection**: a reconnect discards it and waits for fresh snapshots.
2. **Checksum verified on EVERY book row.** A mismatch means our book has desynced and every
   subsequent mid derived from it is fabricated. On mismatch: count a `schema_mismatch`, drop
   the pair's book, **emit no further book events for that pair until a fresh snapshot**, and
   force a reconnect so a snapshot actually arrives. A desynced book must never emit.
3. **`book_depth` >= 10 is required** (the checksum is defined over the top 10). Refuse a
   config below it at construction, loudly — do not silently skip the checksum.
4. **Emission policy (bounded volume, nothing fabricated).** Raw book frames arrive at ~129/s
   for 2 pairs. Measured top-of-book `(bid_px, bid_qty, ask_px, ask_qty)` change rate: BTC/USD
   478/44s, ETH/USD 700/44s — ~4.8× compression. Emit a `book` event when the top of book
   DIFFERS from the last emitted for that pair, **or** when `book_keepalive_s` (default 5.0)
   has elapsed since that pair's last emission **AND A BOOK FRAME FOR THAT PAIR HAS ARRIVED
   TO CARRY IT**. Every emitted event is a true observation of the top of book at its `ts`;
   nothing is interpolated.
   - **CORRECTED 2026-08-27 (run 45, orchestrator).** An earlier draft of this clause read as a
     TIMER ("or when `book_keepalive_s` has elapsed"), and justified it with "otherwise a quiet
     book reads as a dark socket and the gate misfires". **Both were wrong and the second
     argued for the unsafe design.** A timer would write `{"kind":"book","ts":now,...}` carrying
     a top of book that no frame confirmed at that `now` — an inference from absence, recorded
     as an observation, on the row that has halted four times on exactly that. The keepalive is
     a rate-limit RELEASE evaluated when a frame arrives, never an independent clock. A book
     that goes quiet MUST emit nothing and MUST read as un-observed; that is the fail-closed
     direction and it is the behaviour required here.
   - `book_keepalive_s` MUST still be well under the reader's `gap_detect_threshold_s` (default
     60) so the rate limit never coarsens the record past the gate's resolution. **Pin that
     relationship with a test**, in the same spirit as the existing
     coverage-window < gap-threshold pin.
   - The policy and `book_keepalive_s` are DECLARED into `meta_start` so the artifact states
     the cadence floor the numbers were measured under. An undeclared emission policy would be
     a fourth unreconciled declaration.
5. **Counters stay honest.** A legitimately one-sided update is NO LONGER a schema mismatch. A
   row that is genuinely unreadable (non-dict, missing symbol, non-finite/non-positive price,
   unparseable qty) still is. `n_schema_mismatch` must go to ~0 against the real wire — that is
   the re-smoke's pass condition.

## A.5 Acceptance for ADDENDUM A

- (A-i) `experiments/907_live_fill_probe/wire_fixture_kraken_v2_2026-08-27.jsonl` (600 REAL
  frames) is replayed through `_parse_frame` + the book state machine in a test that asserts:
  **every one of its book rows reconciles against the venue checksum**, `schema_mismatches == 0`
  on the book channel, and the emitted top-of-book at a hand-checked frame equals the value
  hand-computed from the snapshot + the intervening updates. This is D67(b)(i)'s raw-frame ↔
  parsed-event normalization-fidelity check, discharged as a permanent test rather than a
  one-off eyeball.
- (A-ii) Mutation proof, each demonstrated RED: (1) revert to `bids[0]` sampling → the
  hand-checked top-of-book assertion fails; (2) skip the `qty == 0` deletion → checksums
  diverge; (3) drop the depth-10 trim → checksums diverge; (4) `json.loads` without
  `parse_float=str` → checksums diverge; (5) let a desynced book keep emitting → the desync
  test fails.
- (A-iii) The trade path is byte-unchanged in behaviour under `parse_float=str`, proven by the
  pre-existing trade tests staying green plus an explicit string-input case.
- (A-iv) No credential, no order path, no new `except`, no new `open`/`subprocess`/`os.environ`
  in `recorder.py`. `test_credential_free.py` still passes.

---

# ADDENDUM B (2026-08-27, factory run 45) — the D67(a) re-cut: declared-vs-observed at all five boundaries

D67(a) is the spec. This addendum names the five boundaries concretely, at file and line, so the
work is bounded and checkable. **Cycle 4's green work (173 tests) is REUSED, not rebuilt** — the
run-44 banner's "What cycle 4 got RIGHT" paragraph lists what must survive intact, and every one
of those properties is a regression target here.

## The principle (D67, binding)

> Every DECLARED quantity is reconciled against what was OBSERVED at EVERY boundary, and no
> fail-loud check may be denominated in the quantity it is checking.

The second clause is the one this lane keeps re-learning. A check whose denominator collapses
along with the thing it measures cannot fire — the 7200s/130s case (`uncovered_fraction = 0.000%`
against a true un-observed fraction of 98.188%) is the canonical instance and every boundary
below is audited for the same shape.

## The five boundaries

### B1 — recorder subscription config → capture (`meta_start`), ACROSS SESSIONS

`capture.py:252` writes `meta_start` **only on a fresh file**; `capture.py:266`'s resume branch
writes a gap and silently discards the `pairs`, `channels` and `heartbeat_timeout_s` it was
handed. `declared_pairs` is therefore a snapshot of the FIRST session's config. Run-44 finding 3,
BLOCKER, and decisive on its own: session 2 adds `ETH/USD`, dead from t=0 → `is_complete()` True,
`assert_complete()` PASSED, `ETH/USD` appears **0 times** in the artifact — strictly worse than
pre-cycle-4, where the union at least surfaced it on the provenance line.

**Required:** every writer session declares its config into the capture (a `meta_resume` record
alongside `meta_start`, or equivalent). The reader reconciles the **union of all declared
sessions** against observed: a pair declared in ANY session must be observed. The mirror
direction must stop misdiagnosing — a pair ADDED on restart and received cleanly is a pair the
subscription DID ask for, and the refusal that reads "a number backed by a pair no subscription
asked for" is wrong about it. Per-session declarations reach the artifact; a capture whose
sessions declare different configs says so rather than presenting the first one as the whole.
Old captures carrying only `meta_start` keep working (that shape is real and this writer produced
it) — but they must not be indistinguishable from a reconciled multi-session capture.

### B2 — capture span → data span (the time axis)

Two mechanisms, both measured, both BLOCKER (run-44 finding 4):

1. **`_find_unrecorded_gaps` (`capture.py:1063`) iterates strictly BETWEEN consecutive events**
   (`for i in range(1, len(sorted_ts))`), so a hole at an EDGE of the timeline has no consecutive
   pair spanning it. Interior hole → REFUSED (cycles 2/3 closed that); leading, trailing and
   both-edge → PASS. **Required:** when a bounding span is known, the scan is anchored to it —
   the interval `[span_start, first_event]` and `[last_event, span_end]` are candidate holes like
   any other. The capture span `[start_ts, end_ts]` (all events, heartbeats included) is the
   bound for the data-bearing scan; heartbeats prove the socket was up, which is exactly what
   makes their span the right denominator for "did we observe the market?".
2. **Coverage is denominated in the span that collapses** (`capture.py:914-919`):
   `coverage_by_pair` is computed over `[data_start_ts, data_end_ts]`. When book/trade go dark
   while heartbeats keep flowing, the denominator shrinks with the failure. **Required:**
   coverage is denominated in the CAPTURE span, reconciled against the data span; the difference
   is itself a reported, gated quantity.
3. **`PairCoverage.uncovered_fraction` (`capture.py:437-443`) returns `0.0` when
   `accounted_s <= 0`**, under a comment claiming "`assert_complete`'s other checks own that
   case". Measured false: `accounted_s = 0` passes with 99.993% of the capture un-observed.
   **Required:** that branch is fail-CLOSED. A pair with no accounted time has made no coverage
   claim and cannot back a number; `0.0` is the one value that must never be returned there.

Regression target: the 7200s-capture-with-130s-of-data fixture must REFUSE, and the refusal must
name the true un-observed fraction (~98.188%), not 0.000%.

### B3 — pre-registered horizons → estimand results

`estimands.py:148` (`if not raw: continue`) and `render.py:229,258` (`if h is None: continue`)
omit a pre-registered horizon that drew ZERO samples. Run-44 finding 1, BLOCKER. `HORIZONS_S`
is a PRE-REGISTRATION: {1m, 5m, 20m, 1h} were declared before any measurement, and a horizon
that produced nothing is a result about the instrument, not an absence of a row.

**Required:** all four pre-registered horizons appear in every estimand-1 table, always. A
zero-sample horizon renders as explicitly zero-sample with no verdict — never omitted, never
rendered as `0`. Note cycle 4 already got the adjacent case right (`if not vals:` emits the
horizon with `n_used=0` when every sample was non-finite); this is the same treatment one branch
earlier. Whatever emits the row, the render must not be able to drop it: `by_horizon.get(h)`
returning `None` is currently a silent `continue` at two sites and must become a stated row.

### B4 — the `--capture` set → the report

`cli.py:77-80` assigns `estimand1_by_pair[pair] = ...` **inside the loop over a repeatable
`--capture`**, keyed on pair alone. Run-44 finding 2, BLOCKER, independently reproduced five
times: two captures both carrying `BTC/USD` → the second silently overwrites the first, and the
artifact presents one capture's numbers over a provenance block listing both.

**Required:** results are keyed by (capture, pair) — or a pair appearing in more than one capture
is a loud refusal naming both captures. Whichever is chosen, **the artifact must state which
capture each number came from**, and the count of captures contributing to a number must be
reconcilable against the count of captures declared on the command line. Do not silently pool:
pooling markouts across captures separated by an outage is a different estimand than the
pre-registered one, and if it is wanted it is declared, not defaulted into.

### B5 — the `--pairs` operator filter → the report

`--pairs` is a THIRD operator declaration reconciled against nothing. Measured: `--pairs SOL/USD`
against a capture without it yields **exit 0** and a ~2900-char gate-passing artifact with the
full disclosure box, full provenance and coverage tables, and ZERO estimand sections — with no
line saying a filter was applied. `shadow_engine.py:247` filters `run_shadow`'s results while
`render.py:350` derives its pair list from whatever survived, so the filter is invisible
downstream.

**Required:** the filter is reconciled against the declared∩observed pair set. A `--pairs` value
naming a pair no capture declared is a loud refusal (it is an operator declaration about the
instrument that the instrument contradicts). The applied filter is stated in the artifact, next
to the declared and observed sets. An artifact with zero estimand sections must never exit 0
carrying a gate-passing disclosure box.

## Also in scope — the in-gate defects the restructure must close

These were found by the run-44 panel and are the same family; the run-44 banner has the full
evidence for each.

1. **The gate OVER-refuses, in a direction this lane's suite is structurally unable to see**
   (regression lens, MAJOR). `capture.py:581-586` is direction-symmetric: a legitimately empty
   `trade` cell is treated as identical to a dead `book` cell and refuses the whole render.
   Measured: a healthy 3-pair 2h capture plus a 120s tail where `SOL/USD` delivered 25 book
   events and zero prints → `rc=1`, no artifact, every number from the good capture discarded.
   **Make the per-cell check asymmetric:** missing `book` on an observed pair REFUSES (an
   actively-printing pair must have a book, so its absence is a broken subscription); missing
   `trade` on a pair whose `book` cell is present and covered is DISCLOSED, not refused (no
   prints in the window is a market fact, and that pair's own continuous book flow proves the
   socket and symbol are alive). **This fires only when `channels` IS declared — i.e. only on
   real recorder captures, never on a synthetic fixture — so the existing 173 tests cannot reach
   it. It needs a test built on a recorder-shaped capture.** Closing this requires editing
   `test_declared_pairs.py:184-210`, which asserts the direction symmetrically; that edit is
   AUTHORISED here so it is not later mistaken for weakening — add the mirror case as an explicit
   expectation stating which direction refuses and why.
2. **The reconciliation runs DAYS too late, with no in-run signal** (regression lens, MAJOR).
   `capture.py:686` is the sole comparison site — read/render time. `recorder.py` never compares
   an observed wire `symbol` (`recorder.py:170,198`, taken verbatim, no normalisation) against
   `cfg.pairs`. Demonstrated with Kraken's legacy `XBT/USD` spelling against a config declaring
   `BTC/USD`: the capture refuses wholesale at read time while `n_schema_mismatch = 0` and
   `n_unrecognized_frames = 0` throughout — the recorder saw no drift at all. Refusing is right;
   the TIMING is the defect. Three to seven days of recording arriving complete, unrenderable and
   not cheaply repeatable is exactly the pressure that later gets relieved by widening a bound.
   **Required:** reconcile at capture START, after the first heartbeat cycle, abort loudly, and
   write the outcome into the capture so it reaches the artifact. Kraken's subscribe-ack echoes
   the venue's own symbol and is the one free signal available in the first second — it is
   currently discarded at `recorder.py:214`.
3. **The twins DO diverge.** `capture.py:624` uses `if self.had_meta_start is False:` — an
   IDENTITY test — so `had_meta_start = 0` makes `is_complete()` falsy while `assert_complete()`
   PASSES. That is the unsafe direction. The "structurally impossible" claim is a 12-case
   enumeration that omits the one field not helper-routed; either route it or drop the claim.
4. **`channels: []` disables both channel checks with no refusal**, while `declared_pairs: []` is
   refused — the identical shape one field over. Rule consistently and state the rule.
5. **The channel-side mirror `observed_channel_undeclared` was never implemented**, so a capture
   declaring `channels=["trade"]` and full of `book` events passes while every primary estimand
   number rests on an undeclared channel.
6. **The refusal's prescribed remedy is unachievable and self-contradictory for the thin-cell
   case** (`capture.py:706-711`): re-cutting to a NARROWER window can only reduce a print count,
   so that remedy is monotonically wrong, and "re-run the recorder against a subscription that
   does deliver" IS dropping the pair from the declaration, which the next sentence forbids. The
   operator's only reachable action is the one the message calls laundering. Split the remedy
   text per failure kind.

## Acceptance for ADDENDUM B

- (B-i) Each of B1–B5 has a test that FAILS against the current tree and passes after — the
  demonstration is the deliverable, not the patch. Run each one RED first and report the
  assertion.
- (B-ii) The 7200s/130s capture REFUSES and its refusal names the true un-observed fraction.
- (B-iii) The over-refusal repro (healthy 3-pair 2h capture + a book-only 120s tail) now RENDERS,
  with `SOL/USD`'s empty trade cell DISCLOSED in the artifact, and the numbers from the good
  capture are byte-identical to what that capture renders alone.
- (B-iv) **NO NUMBER MOVES on a healthy capture.** Cycle 4 proved this by rendering a healthy
  3-pair 2-hour capture built with the production writer pre- and post-change and diffing:
  byte-identical except the intended new provenance lines. Reproduce that proof for this change.
  Every cycle-3 constant and pinned relationship stays intact (the coverage-window < gap-threshold
  pin especially — with window == threshold the aggregate check becomes structurally unable to
  fire).
- (B-v) Everything the run-44 banner lists under "What cycle 4 got RIGHT" still holds, checked
  item by item: `declared_pairs` from `meta_start` ONLY and never unioned with observed;
  `meta.pairs` keeps its union/provenance meaning with `render.py:80` its sole consumer;
  `observed_pairs` remains a read-only property over `observed_pair_channels`; fail-CLOSED on a
  missing OR empty declared set at BOTH format versions; the bare-string `pairs` fix; the
  coverage bound still NOT a CLI flag.
- (B-vi) Node-ID accounting: report added/removed/renamed test node IDs explicitly (`comm -23`
  against the baseline set). Nothing may be removed or renamed without being named and justified.

## Orchestrator reproductions (factory run 45, first-hand — these are measurements, not inherited claims)

Three of the five boundaries were re-measured against the current tree before this spec was
written. Repro scripts preserved under the run's scratchpad; each is a seed for the test the
acceptance criteria demand.

**B2 / run-44 finding 4 — REPRODUCED.** A 7200s capture built through the production writer
carrying book+trade for its first 130.5s only, heartbeats every 10s throughout:

```
capture span : 7200.0s
data span    : 130.5s
TRUE un-observed fraction of the CAPTURE span: 98.188%
  reported uncovered_fraction[BTC/USD] = 0.000%  (accounted_s=130, span_s=130)
unrecorded_gaps      : 0
unrecorded_data_gaps : 0
is_complete()        : True
>>> assert_complete() PASSED <<<
```

`accounted_s=130, span_s=130` is the diagnosis in two numbers: the denominator IS the collapsed
quantity. The banner's 98.188% is confirmed to three decimals.

**B4 / run-44 finding 2 — REPRODUCED, and sharper than "clobber".** Two single-pair BTC/USD
captures through the production writer, one drifting UP and one DOWN so their markout
distributions differ, rendered alone and together:

| invocation | 1m median bp rendered |
|---|---|
| `--capture A` | 22.03 |
| `--capture B` | 26.34 |
| `--capture A --capture B` | **26.34** (= B alone, every row, every decile) |
| `--capture B --capture A` | **22.03** (= A alone, every row, every decile) |

**The LAST `--capture` on the command line silently wins and the other is discarded whole**, under
a provenance block that names BOTH files. Order-dependence is the sharp edge: the same two
captures produce two different artifacts depending on argument order, and neither says so.

**B5 / the `--pairs` filter — REPRODUCED.** `--pairs SOL/USD` against a healthy BTC/USD capture:
`rc=0`, a 3020-byte artifact carrying the full gate-(iii) disclosure box, **zero** estimand
sections, `SOL/USD` appearing nowhere in the output, and no line anywhere stating that a filter
was applied.

**B1 / run-44 finding 3 — REPRODUCED.** Session 1 declares `[BTC/USD]` and delivers it; session 2
reopens the same capture declaring `[BTC/USD, ETH/USD]` with ETH dead from t=0:

```
declared_pairs : ['BTC/USD']      <-- session 2's declaration DISCARDED entirely
observed_pairs : ['BTC/USD']
is_complete()  : True
assert_complete() PASSED
'ETH/USD' occurrences in the rendered artifact: 0
```

The reader never learns that ETH/USD was subscribed. The two sets reconcile perfectly *because*
the declaration was lost — a reconciliation that passes by forgetting one of its two operands.

**B3 / run-44 finding 1 — REPRODUCED.** A 1500s (25-minute) capture, in which the pre-registered
1h markout can never form:

```
pre-registered horizon  1m rendered as a table row: True
pre-registered horizon  5m rendered as a table row: True
pre-registered horizon 20m rendered as a table row: True
pre-registered horizon  1h rendered as a table row: False   <-- silently absent
```

`rc=0`, artifact renders clean. A reader is shown three horizons and is never told that a fourth
was pre-registered and drew nothing — the difference between "we measured three horizons" and
"we pre-registered four and one was unmeasurable" is exactly the claim this row exists to make
honestly.

**All five boundaries are therefore first-hand measured at the current tree, not inherited.** Each
repro script is the seed of the test (B-i) requires; none of them is the test itself.

**The OVER-refusal MAJOR — REPRODUCED**, and the refusal's own premise is measurably false. A
120s tail capture in which `SOL/USD` delivered a live book and zero prints:

```
observed_pair_channels: {'BTC/USD': ['book','trade'], 'ETH/USD': ['book','trade'],
                         'SOL/USD': ['book']}
SOL/USD has a coverage record: True
rc=1, artifact written: False
Unreconciled: declared_channel_without_data=['SOL/USD:trade']
```

The refusal text asserts the pair is "named in this artifact's own provenance and then absent
from every table below it". For `SOL/USD` that is false: it HAS a coverage record, a cadence row
and book-derived mids. The gate is refusing a pair it can see. `rc=1` with no artifact at all
discards every number from every healthy pair in the capture.

**The three inherited lower-severity claims — ALL CONFIRMED:**

```
had_meta_start=0    -> is_complete()=False, assert_complete()=PASSED   <-- twins diverge, UNSAFE
channels=[]         -> assert_complete() PASSED   (both channel checks silently disabled)
declared_pairs=[]   -> assert_complete() REFUSED  (identical shape, one field over)
channels=['trade'] on a book-full capture -> reconciliation gaps: {}   <-- mirror missing
```

The last line is the sharpest of the three: a capture may declare `channels=['trade']`, contain
nothing but `book` events, and reconcile perfectly — so every primary estimand number rests on a
channel the capture never declared, and the gate is silent.

## Note for the implementer taking this addendum

Addendum A (run 45) landed FIRST and changed two things you inherit:

1. `recorder.py` now maintains a per-pair order book and emits book events on
   **top-of-book change OR a `book_keepalive_s` (5.0s) floor**, declaring
   `book_emission_policy` and `book_keepalive_s` into `meta_start`. Those two keys are additive;
   the reader does not yet consult them. **B2 must reckon with the keepalive**: a keepalive event
   is a real observation of the top of book, but it is emitted on a timer rather than on wire
   traffic, so the coverage machinery must not treat a keepalive stream as proof the market was
   observed if the underlying book has gone stale. Reconcile the declared emission policy against
   the observed event cadence — that is the D67 principle applied to the key Addendum A added.
2. A checksum-mismatch desync currently writes a gap with `reason="schema_drift"`, disambiguated
   only by free text in `detail`, because `GAP_REASONS` is a closed enum in `capture.py` and
   Addendum A's scope fence forbade touching it. **This pass owns `capture.py`, so add the proper
   enum entry** (e.g. `book_checksum_mismatch`) and make the reader able to distinguish "sustained
   unparseable traffic" from "our book desynced" without parsing prose.

Also inherited, disclosed by Addendum A and NOT fixed there (out of its scope, in yours or the
planner's):

- `_check_schema_drift`'s `_frames_since_last_event` counter is not reset when a new connection
  starts (pre-existing).
- The real wire sends virtually one book row per frame (590 rows across 588 update frames + 2
  snapshots), so the fixture replay does NOT exercise multi-row-per-frame paths — partial desync
  within one frame, mixed valid/invalid rows. Those rest on hand-built tests. Do not read the
  fixture coverage as proving more than it does.

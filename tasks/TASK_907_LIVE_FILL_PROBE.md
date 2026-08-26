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

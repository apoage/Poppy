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

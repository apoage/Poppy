# TASK_907 — pre-registration (written BEFORE any analysis code)

Written first, per the 906 lesson (`experiments/906_oos_nonstationarity/PREREGISTRATION.md`
precedent): pre-registration ordering is itself evidence. This file is created before
`estimands.py`, `render.py`, or any other analysis code in this directory exists — verify with
`git log --follow` / file mtimes if this claim is ever challenged. Nothing below is retro-edited
after analysis code lands; amendments get a dated addendum at the bottom, never a silent edit.

## Framing (load-bearing — read before the estimands)

The maker-fill model (OHLC-touch + a flat adverse-selection term) has been the load-bearing,
never-validated assumption of the 806→809→897 lineage since 2026-06. TASK_897 returned **ALL
FAIL**: 0/576 attainable-tier cells clear, 0/864 at any nonzero fee, and **0/188 clear even with
adverse selection set to ZERO**. This probe **calibrates the model. It cannot reopen the
verdict.** No realized markout distribution rescues a band whose nearest cell is −5.045 with the
penalty at zero. This probe is an instrument that produces an honest number, not an attempt to
find an edge.

## Estimand 1 — adverse selection

For each shadow fill (a virtual limit posted at minute-bar close, marked filled-by-touch per the
declared reference fill model — `reference_fill_model.py`), compute the **side-signed** mid-return
from the fill price to mid price at t+{1m, 5m, 20m, 1h}:

```
signed_return_bp(h) = side * (mid[t_fill + h] - fill_price) / fill_price * 1e4
adverse_selection_bp(h) = -signed_return_bp(h)
```

where `side = +1` for a buy-side virtual fill, `-1` for a sell-side virtual fill. A positive
`adverse_selection_bp` means the market moved against the maker after the fill (the classical
adverse-selection direction).

**Report the full distribution (deciles: 10/20/.../90th percentile) at each horizon, not just the
mean.** Compare the model's flat {0, 5, 10} bp term against the measured distribution's LOCATION
(median, and the decile spread around it) — not against a single point estimate.

**Truthfulness bound:** touch-conditional markouts are the OPTIMISTIC bound on adverse selection.
A resting limit order in a real queue is filled only when price trades THROUGH it after clearing
whatever size sits ahead in the queue — by which point the market has already moved further than
the bar's mere OHLC touch implies. Real fills are therefore selected worse (higher realized
adverse selection) than this probe's touch-conditional measurement. **If even this optimistic
bound meets or exceeds the model's 5 bp assumption, the model stands calibrated-pessimistic**, and
phase 2's only remaining job is the queue-priority correction for the record.

## Estimand 2 — fill rate

Realized fill proxy rate of a limit posted at bar close vs the OHLC-touch assumption the model
embeds. **A touch is not a fill — queue priority is the whole question**, and the public feed
(book top + trades, no venue-side queue position) cannot resolve queue priority. Two proxies are
reported, both explicitly labeled as OPTIMISTIC UPPER BOUNDS on the true fill rate (neither can
be tighter than the true rate, because neither accounts for queue position ahead of a resting
order):

1. **OHLC-touch rate** — the model's own assumption: fraction of posted virtual limits where the
   subsequent bar's high/low (reconstructed from captured trades in that bar) touches or crosses
   the limit price.
2. **Trade-cross rate** — a stronger proxy: fraction of posted virtual limits where a captured
   REAL TRADE prints at or through the limit price (not merely a synthesized high/low touch).
   Still optimistic — it does not know whether our hypothetical resting size would have been
   ahead of or behind the print in the real queue.

No claim of a measured TRUE fill rate is made anywhere in this artifact; both numbers are
reported as bounds, and the artifact says so next to every number.

## Pairs

The sweep universe's most liquid: BTC/USD, ETH/USD, plus 2–4 majors selected by spread
(operator/orchestrator picks the concrete list at launch time; this file pre-registers the
SELECTION RULE — most liquid by spread — not a hardcoded list, since the concrete list depends on
live spread data this file cannot see in advance).

## Gate obligations this file exists to satisfy

- Gate (iii): the rendered artifact's first box states, verbatim, that this probe calibrates a
  model and cannot reopen 897's ALL FAIL, quoting the 0/188-at-zero-penalty robustness argument.
  Enforced as a tested property of `render.py`, not a hand-written line.
- Gate (iv): no number in the artifact may come from a run whose capture had an unrecorded gap.
  The renderer refuses (raises / emits an explicit REFUSED section) when capture metadata is
  absent, incomplete, or internally inconsistent with the raw event stream.

## Out of scope (restated from the task file)

Any strategy trading. Any use of this probe's numbers to re-litigate 897/809's band. Venue
changes. Reopening the minute surface (that is D65's closure, accepted after this row's results
are in hand).

---

## Amendment 2026-08-27 — DISCLOSURE fields added (estimands unchanged)

**This is a disclosure addition, not a change of estimand.** Nothing above is edited or
withdrawn: the formulas, the horizons, the deciles, the tier comparison and the optimistic-bound
framing are exactly as pre-registered. What changes is what the artifact must SAY about the
samples behind each number.

Verification of the first implementation found that a non-finite value (Python's `json` accepts
a bare `NaN` literal, so a venue error frame or a schema-drift bug can produce one) reached the
artifact as if it were a measurement: one `NaN` among eleven markouts left `n` reported as 11,
produced `mean`/`median` of `nan`, left several deciles rendering *plausible finite numbers*
derived from a `sorted()` ordering corrupted by the `NaN`, and — because `tier >= nan` is always
`False` — emitted the confidently-worded verdict `0bp: BELOW measured, 5bp: BELOW measured,
10bp: BELOW measured`. That is a number presented as if it had been measured, which is the one
outcome this instrument may not produce.

Accordingly, each horizon now additionally reports:

- **`n_samples`** — markout samples at that horizon before any filtering.
- **`n_used`** — the samples that actually back the reported mean/median/deciles. This is the
  number to read as "n"; it is the pre-registered estimand computed on the valid subset.
- **`n_dropped_nonfinite`** — samples excluded for being non-finite. Reported, never silent: a
  silent drop is the same defect one layer down from a silently admitted `NaN`.

And two reporting rules are made explicit:

1. A non-finite sample never enters a mean, a median, a decile, or a tier comparison.
2. When no finite sample exists at a horizon, the location and every tier comparison render
   **`not-evaluable`** — never a verdict. A check that did not execute a comparison is not a
   passed check, and `tier >= nan` returning `False` is not evidence about the model. The
   horizon is still reported (rather than omitted) precisely so the artifact discloses that
   samples existed and none were usable.

Also disclosed in the artifact, for the same reason (they bound what the numbers cover, without
altering any estimand): the recorder's own health counters (JSON-decode failures, schema
mismatches, unrecognized frames), per-capture gap-ledger normalization counts, the declared
gap-accounting clock basis, and the per-reason counts of dropped markout candidates
(gap-straddling / no-mid / non-finite).

---

## Amendment 2026-08-27 (b) — RESOLUTION and MID SOURCE

Nothing above is edited or withdrawn. Two additions, and the second one contains a judgement
call that is flagged as such rather than buried.

### (b.1) Realized resolution is disclosed, and a declared bound gates it — DISCLOSURE ONLY

Verification found that gate (iv)'s only defense was a PER-JUMP threshold check with no
cumulative protection, fed a timestamp list that included `heartbeat` events. A capture whose
book+trade events were spaced 119s apart for ten continuous days — with heartbeats flowing
normally, so the recorder's own `heartbeat_timeout_s` never expired and no gap was ever written —
reported `is_complete() -> True`, rendered a full artifact (242 posted, 121 touch-fills, 484
markout samples, `recorded_gaps=0`), and disclosed the degraded cadence nowhere. A capture that
never observed the market faster than once per ~2 minutes presented as fully covered, backing an
estimand denominated in 1-minute bars.

The estimands are unchanged. What is added is what the artifact must SAY about the RESOLUTION of
the observations behind each number, plus a bound that refuses rather than reports when that
resolution is not there:

- **Realized data-bearing cadence** — median / p95 / max inter-event interval, per pair, per
  channel, computed from `book`/`trade` events ONLY. Heartbeats are excluded because a heartbeat
  proves the socket is alive, not that the market was observed; that distinction was the defect.
- **Aggregate coverage** — per pair, the cumulative fraction of the capture's *accounted* time
  (its data span minus the union of its RECORDED gaps) that no data-bearing event came within
  **15 seconds** of. Declared bound: **at most 2%**. Above it, the capture is REFUSED and no
  number is rendered from it.
- **Both gap knobs and the bound are printed** next to the numbers: the reader's per-jump
  gap-detect threshold (now hard-clamped to one 60s bar interval — a wider threshold could let a
  jump swallow a whole bar unflagged) and the recorder's own declared `heartbeat_timeout_s`.
- **Per-sample mid staleness** — p50/p95 of how old the backing observation was at each markout
  horizon, and the staleness cap tightened from 300s to one bar interval. A mid observed five
  minutes before its horizon must not be presented as the mid AT that horizon when the shortest
  pre-registered horizon is 60 seconds.

Why 15s and 2%, against this estimand: the shortest pre-registered horizon is 60s, so a 15s
resolution bounds the timing error on the 1m markout at 25% and on the 5m markout at 5%; and the
deliverable is a decile report, where a contaminated fraction *f* can displace a reported decile
by at most *f* in quantile-rank terms — so *f* = 0.02 means the printed p50 is, worst case, the
true p48/p52, below the resolution at which any decile here is read.

### (b.2) The trade-print fallback is excluded from estimand 1 — A JUDGEMENT CALL, stated

The shadow engine's as-of mid preferred top-of-book mid and fell back to the last trade print,
with no source tag anywhere: `MarkoutSample` carried no source field and nothing wired one into
the artifact. The fallback was disclosed only in a code comment, which this artifact's reader
never sees.

**A trade print is not a mid.** It is struck at the aggressor's side of the spread, so
substituting one for a mid mixes two different quantities into one reported adverse-selection
distribution. The bias is not mean-zero where it matters: in the moments after a fill that is
going badly, aggressor flow is one-sided with the move, so the print sits on the far side of the
mid and measured adverse selection comes out too **HIGH (pessimistic)**. That is the direction of
this row's own preferred conclusion — "if even the OPTIMISTIC bound meets the model's 5 bp, the
model stands calibrated-pessimistic" — and it is therefore the one direction a contaminant may
not be allowed to push.

Accordingly: every markout sample is tagged with its mid source and the two populations are
counted separately (`n_markouts_from_book_mid` / `n_markouts_from_trade_fallback`, both
rendered). **Estimand 1 above is computed from book-mid samples only.** Trade-print-backed
samples are reported as a clearly labeled SECONDARY distribution with the bias direction stated,
and a prominent banner fires when the fallback share is material (>= 5%).

**Stated plainly, because it is the one item here that is more than a disclosure field:** the
estimand-1 *formula* is untouched, but *which samples enter it* is now narrower than the first
implementation's. The reading taken is that this CONFORMS to the definition above rather than
amending it — that definition says "mid price at t+h" and never authorized a print as a mid, so
the blend was an implementation liberty and confining it restores the pre-registered quantity.
A reader who disagrees and considers this a change of scope has everything needed to say so: the
excluded samples are not discarded, they are reported separately with their own median, deciles
and count, so both readings can be evaluated from the artifact itself.

### Not fixed, still disclosed

Kraken v2's live schema remains UNVERIFIED by this row (no network access). Every number this
probe produces covers only what was parsed, and the recorder's schema-mismatch counters are the
measure of that.

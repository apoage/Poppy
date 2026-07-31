# Factory run 23 — 2026-07-31 — TASK_865 SHIPPED, D31 seeded OPEN

Neutral technical continuity record. Run 22 stopped TASK_865 on D30; D30 was ruled opt (i)
(forward-fill + disclose); run 23 resumed the verified stash, landed the amendments, survived two
fix rounds, and shipped. The real leg is blocked again — on a new wall this task's own work exposed.

## What shipped

The minute-framed frame pin (D29 opt (a) / D30 opt (i)). Recorded 64-slot frame read positionally
from the minute `ohlcv` table at a `[::bar_minutes]` stride, **62/64 live** on the leg-1 window
against the daily path's 33/64 — the only basis comparable to the `+0.839` bar on universe AND
source AND breadth at once. Stale tails forward-filled and disclosed. Flag-gated
`--panel-frame-source minute`; the shipped 853 daily pin stays the default.

Plus the three D30 riders: (a) price-validity mask threaded into `compute_teacher_floor`,
(b) `--require-active-slots` breadth refusal — now on BOTH pin paths, (c) `scorecard.py` plumbs
`--panel-frame-source` through the manifest + operator pin or REFUSES.

**Suites, final tree:** selftest 12/12 · 863 13/13 (twice) · 865 25/25 · 865 real-DB leg 17/17 ·
861 e2e 10/10 · scorecard 8/8.

## The assertion that was missing, and now is not

Run 22's build had 19 green legs, all asserting ACTIONS, none asserting PORTFOLIO VALUE — which is
exactly how a PnL-fabricating defect passed them. `0_pv_conservation` now ships first-class: flat
prices, fee 0, one mid-window delisting, a LONG book and a SHORT book must each end at exactly the
initial cash. Measured **100,000.00 / 100,000.00 to the cent**, with a structural control on the
pre-fix zeroed tail reproducing D30's ground truth **75,002.50 / 124,997.50** in opposite
directions. The orchestrator reproduced both independently, and separately reproduced the
guard-inertness result at clean HEAD (tail zeroed with and without the force-HOLD guard is
byte-identical — the fix all three run-22 judges prescribed genuinely does nothing).

## Two BLOCKERs, both demonstrated, both fixed

**1. The loaders opened the 24 GB source of record READ-WRITE.** No write SQL is needed: SQLite
performs hot-journal recovery on the first READ through a read-write handle. Demonstrated on a copy
carrying a genuine hot journal — the loader changed the file's sha256 AND size and unlinked the
journal, while a `mode=ro` handle refused and left it byte-identical, against a docstring reading
"Read-only, own connection." Reachable because this change pins `panel_frame_source: minute` +
`db: trading_data.db` into the tracked manifest (7 windows), a writer exists
(`tools/fetch_daily_kraken.py:360`), the handle is held across **73 M rows** for minutes, and the
only backup predates the 858 backfill. All 11 sites now route through `_ro_connect`. Orchestrator
re-verified after the fix: hot-journal DB **sha256 unchanged**, probe raises `attempt to write a
readonly database`, absent path **refused with nothing created**.

**2. Interior frozen spans were tradeable, undisclosed, and outside the STOP gate.** Staleness was
keyed off `last_real_full`, so a slot RESUMING before the window end appeared in no list and read
`covered_rows == n_rows`. Now disclosed as `frozen_spans` on both pin paths, counted in an honest
INERT/FROZEN/LIVE partition, gated on its own axis.

## D31 — the new wall, and why it is bigger than 865

Opt (i) was justified by "a constant series cannot fabricate PnL — a round trip nets exactly 0 at
fee 0". **True for a TRAILING span, false for an INTERIOR span that re-anchors**: the book is filled
at a price no market offered and handed the whole gap move in one bar. On the real leg-1 window the
minute table is near-empty **2025-03-31 → 2025-10-01 on 11 of 64 slots**, re-anchor jumps
ETHBTC **+64.52%**, BTCCAD **+33.88%**, BTCGBP **+32.69%** — sign-indeterminate, each slot
weight-eligible to ~6.25% of PV, landing in the exact scalar graded against `> +0.839`.

**And the canon UNPINNED loader (`vectorbt_env.py:645-655`) fills identically** — so the `+0.839`
bar, the `+0.898` OOS read and the WF quarters are all likely contaminated the same way. That is
what makes it a program-level ruling rather than an 865 bug: fixing it inside 865 would make the
comparison *less* interpretable, not more.

**Late evidence that changes the diagnosis, orchestrator-confirmed on the real DB:** 6 of the 11
suspect slots are an **INGEST IDENTITY SPLIT**, not missing data. Twin `assets` rows under spelling
variants (`/USDTEUR`→`USDTEUR` 1144, `/USDTCHF`→1172, `/USDC/USD`→1145, `/USDCGBP`→1160,
`/USDC/USDT`→1153, `/USDTAUD`→1176) carry **204k–328k minute rows spanning exactly
2025-01-01..2025-09-30**, and `/USDTEUR` vs `USDTEUR` overlap on **125,737 rows with max relative
close diff 0.0** — same feed, split key. The genuine interior-gap case is **5 illiquid crosses**
(ETHBTC, BTCGBP, BTCCHF, BTCCAD, XRPBTC), ~1% of the grid.

## Numbers corrected where they were authored

- D30's `walk_net21 ≈ +17.1242` → **SUPERSEDED by +14.9497**. +17.1242 is what replaying the
  PRE-FIX action stream against filled prices yields; a real re-run lets the search operator
  re-decide. It never lived in a tracked file.
- D30's `scorecard.py:246` → `OPERATOR_ARGS, scorecard.py:297`.
- Inert share ~3% → 9.6% → 8.79% was **all head+tail only**; honest figure **10.67%**. The
  orchestrator's own independent 8.79% shared the harness's blind spot and was therefore *not*
  independent corroboration.
- "The fixture is 25% undisclosed-frozen (192/400 live)" — **RETRACTED**, that was the exact-minute
  reading. Under the shipped bar-window definition the fixture has 6 frozen cells and 294 live. The
  real-leg finding is unaffected.
- `sanity_net21` −4.2446 → +12.0134 is **not** "the tail became tradeable": the pure valuation
  channel is **−1.00 Sharpe**, opposite in sign and ~1/16 the magnitude. The swing is the operator
  re-deciding plus the student retraining.

## Rider (a) is inert, and now says so

`book_target_signs` returns sign 0 on a constant column, so the ladder HOLDs a flat-zero slot, and
`_ffill_interior_only` makes the `real→0` transition its docstring blamed impossible. Masked ==
unmasked bit-identical on both fixture pins. The recorded `−15.83 / bankrupt_at 45` measures the
**zeroed-tail build**, not pre-listing heads. It ships as defence-in-depth with an honest docstring.

## Open remainders

- **D31 (OPEN) — gates the 861 real leg.** Recommendation: repair the identity split first (it turns
  fabricated cells into real prices), mask the ~1% genuine residue, then ONE re-base wave before any
  reference is used as a bar.
- **Row A** — the daily-framed teacher floor still walks UNMASKED (same class as rider (a); fixing it
  moves shipped daily numerics).
- **Row B** — `_to_epoch` resolves `--start/--end` in LOCAL time, so a window's extent depends on the
  host. Measured 308-row CEST offset; explains the 1734-vs-1735-class disagreements between parties.
- **Row C** — fixture has no interior gap under the shipped definition; that proof rides on a
  synthetic DB.
- **Row D** — the canon panel path (`vectorbt_env.py:515/:731`) still opens the DB read-write.
- **TASK_868/869** exist as UNAUTHORISED drafts (see below), gated on D31.

## Process facts worth carrying

- **Two of five panel lenses never finished.** An API session limit killed three mid-run; after
  resuming them, the orchestrator stopped regression and flow-verifier EARLY because the fix pass was
  about to invalidate any measurement taken on the pre-fix tree. Both partials were clean. Acceptance
  (i) therefore rests on four independent sources, not a dedicated regression battery.
- **`/tmp` (27 G tmpfs) filled to 100% mid-run** and broke tooling. Cause: `scorecard_smoke.py`
  defaulted its work dir there — the exact MINOR the fresh judge had flagged. Fixed to `/var/tmp`;
  9 G of stale prior-run scratch reclaimed; all agents moved to disk-backed scratch.
- **An unauthorised D31 ruling appeared in `tasks/DECISIONS.md`** (OPEN → DECIDED, "RULED …
  (PLANNER)") along with TASK_868/869, at 21:49–21:50. The orchestrator first attributed this to the
  round-2 fix agent; **that attribution is withdrawn as unsubstantiated** — the agent denies it and
  the run log records no Write/Edit calls at all today. The ruling was reverted to OPEN with its
  evidence retained. No agent is accused.
- **`chosen_vectors_fixture.json` was clobbered FOUR times** by bare suite runs, each a stamp-only
  `created_utc` diff. Reverted every time; mechanism deliberately untouched — it is TASK_867's.
- Acceptance (i) is **qualified**: the 0.9 breadth fraction rounds up, so ≤10-slot frames need 100%
  and the shipped 863 canonical daily-pinned fixture invocation now refuses at the default. Direction
  is fail-closed; default unchanged; caveat stated in the code.

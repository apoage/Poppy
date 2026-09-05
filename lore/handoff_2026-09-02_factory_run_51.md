# Factory run 51 — TASK_907 Addendum B2: HALTED at cycle 3 of 3

Date: 2026-09-02. Base HEAD `72cb91e`. Nothing committed, nothing deployed, nothing pushed.

## Outcome

**HALTED on the 3-cycle rule, budget fully spent.** Cycle 3 produced four BLOCKERs from three
independent sources (two panel lenses convergently, one lens alone, one from the orchestrator's
own real-capture measurement). The row is not blocked on an operator decision — D65-amend-2 and
D77 were both ruled 2026-09-02 and both were implementable. It is blocked on the fact that three
defect *shapes* recur faster than they are patched.

Work preserved two ways: left in the live tree, and copied to
`lore/handoff_2026-09-02_factory_run_51_907B2_halt.patch` (3749 lines, sha256
`785577cb635ce65c78ce1e15cb2d71e010fab506ffbcfa4164d56acb24a9f6f8`, `git apply --check` verified
OK against HEAD `72cb91e` in a detached worktree). Supersedes run 50's patch, which it contains.

Suite 342 → 362, selftest 12/12, fixture `b0f8de8b…` unchanged, 10 halt-preservation stashes
intact, no deploy, no push.

**The tree is NOT shippable.** Four BLOCKERs are open in it.

## What cycle 3 got right, independently verified

- **B-iv passes.** The regression lens built 17 fixtures across fresh tmpdirs, rendered each
  through a detached HEAD worktree and the live tree, rebuilt every render twice to defeat
  caching: **0 moved numbers, 44 estimand-1 rows across 10 fixtures, cell-level.** The only HEAD
  lines without byte-identical counterparts are three permitted in-scope items (the MAJOR_4
  withdrawal prose, the two new `mid VALUE age` columns, the `mid age` paragraph). The
  flow-verifier's independent B-iv agrees.
- **B-v holds item by item** (`declared_pairs` never unioned with observed; `meta.pairs` semantics;
  `observed_pairs` read-only; fail-closed on missing/empty declared set at both format versions;
  coverage bound still not a CLI flag — `build_argparser` byte-identical to HEAD).
- **B-vi node accounting clean:** vs cycle 2, removed 0 / added exactly 20; vs HEAD, removed 0 /
  added 60. No renames.
- **B1's `book_keepalive_overruns` / `_max_interval_s` keep exactly their B1 semantics**, A/B'd
  across 16 fixtures with empty diff, including the 200s-silence/150s-gap/50s-residual shape that
  cycle 1 destroyed.
- **Security: clean.** The TASK_910 credential fence was proven by MUTATION over all seven
  touched/new files — RED every time, each naming the file. Rails absent from the diff and
  mutation-tested anyway (narrowing the re-raise clause, repointing `WS_URL`, adding a third
  subscribe frame, making `place_order()` return — each turned the suite red). `cli.py` has no
  broad catch. `venue_ts` fuzzing (NaN/Inf/string/bool/1e400/null) fails closed via
  `_require_finite` on all 11 cases.
- **BLOCKER_4's limit-provenance half is genuinely closed** for the geometry it was reported in
  (`_Bar.close_ts`/`close_venue_ts`; orchestrator repro 7996.97 bp → none, drops 26 → 30).
- **BLOCKER_5's three ruled cases all behave**, verified independently by three lenses, including
  the reconciliation line and the per-capture notice. The cycle-2 MAJOR is not reintroduced.
- **D65-amend-2 confirmed:** four horizons render always, zero-sample ones as `not-evaluable`
  sentinels rather than `0`. Lane prose scanned mechanically and clean.

## The four open BLOCKERs

### 1. The flagged span ends one keepalive frame before the book unfreezes (CONVERGENT — 2 lenses)

`capture.py:1879-1881` (`_close_book_run`) appends `(run["start"], run["last"])`. `run["last"]` is
the receipt ts of the last book frame that *repeated* the frozen top; the book stays frozen until
the *next* frame, one declared keepalive later (5.0 s in production). Every print in that trailing
window walks through the same frozen book and falls OUTSIDE the span, so `limit_flagged`,
`anchor_flagged` and the `target_ts` test all inherit an under-covered span.

Reproduced end to end by two lenses independently, rc=0, artifact written:
published estimand-1 1m mean **−0.00 bp → +106.63 bp** by moving one print 5 seconds later,
beneath a disclosure reading `30 of 324 markout candidates excluded = 9.259%`.

This is the root cause of the run-50 symptom the run-51 brief noticed and mis-attributed: the
orchestrator observed `fill_ts=2710.0` landing 5 s past the flagged run's end at 2705.0 and
recorded it as evidence the *predicate* was looking at the wrong quantity. The 5 s was real and
was the signal, but the fault was the span's ENDPOINT, not the predicate. The implementer fixed
the predicate; the shipped repro then passed because it exercised only one geometry.

Direction: close the run at the first frame that CHANGED the book, not the last that repeated it.

### 2. The as-of mid's provenance is never gated (correctness lens; fail-closed lens concurs)

`shadow_engine.py:435` tests `target_ts` for flagged-run membership, but `mid_at` returns the last
book observation at or before `target_ts` within `MAX_MID_STALENESS_S = 60 s`, and `MidQuote`
carries no observation timestamp at all. A target landing seconds after a flagged run therefore
resolves to that run's own frozen quote — the quote the detector concluded trades were walking
through.

Measured: `20m fill_ts=880.0 target=2080.0 bp=482.33 mid_age_s=5.0` ⇒ backing observation at
ts 2075.0, inside the flagged span `(2000.0, 2075.0)`. Anchor clean, limit clean, target clean, so
nothing drops it. Moves published 5m/20m medians by 8.00 and 10.87 bp. A second geometry gives
−510 bp. The `mid VALUE age` p95 columns cannot surface it (the artifact's own documented
near-inertness), so no disclosure column shows it.

Direction: give `MidQuote` the observation's own ts (and `venue_ts`) and OR it into the predicate —
the same provenance treatment `close_ts` just received. Fixing BLOCKER 1's endpoint closes some
geometries but not the mechanism.

### 3. An undeclared emission policy switches off the ENTIRE 2b detector (fail-closed lens)

`capture.py:2174-2200`: when `declares_keepalive` is false, `stale_runs = {}`. `outside_book_runs`
rides on `stale_runs` (`capture.py:2264`), so `outside_book_flagged_runs()` is empty,
`flagged_spans` is empty, and there is no gating at all. The original 8000 bp leak returns:

```
declared policy/keepalive : None None
flagged runs              : {}     stale_book_s: 0.0
book_uncovered_fraction   : 0.0    b2 disqualified: {}
n_dropped_outside_book_run: 0      published |markout|>1000bp: 6  (|bp| up to 7996.97, 42550 asym)
cli rc = 0
```

Four terms that made no measurement all read as the strongest possible claim. This refutes the
implementer's stated ground for its MAJOR_5 deviation (it substituted measurement for the briefed
fail-closed refusal, arguing measurement fully closes the defect). Full-payload identity does not
reach a `(bid, ask)`-frozen book whose SIZES move — the module's own "case A", deliberately scored
as observed. 21 of the lane's B2 fixtures declare the policy, so the leaking shape is the one the
suite does not exercise for 2b.

### 4. D77's gate is denominated at the wrong GRANULARITY — all four real pairs excluded (ORCHESTRATOR, measured)

Measured first-hand on the real 4.117-day capture with the shipped cycle-3 code, one capped read
(peak RSS 14.83 GB, `MAX_UNEXPLAINED_BOOK_SILENCE_FRACTION = 0.01`,
`OUTSIDE_BOOK_EXCESS_RESPONSE = 'run'`):

| pair | flagged runs | unexplained silence | bound | verdict |
|---|---|---|---|---|
| BTC/USD | 10 of 1222 | 303.8 s = 0.0952% | 1.0000% | EXCLUDED — gate fired |
| ETH/USD | 12 of 773 | 561.1 s = 0.1758% | 1.0000% | EXCLUDED — gate fired |
| SOL/USD | 24 of 600 | 97.8 s = 0.0307% | 1.0000% | EXCLUDED — gate fired |
| XRP/USD | 10 of 441 | 488.8 s = 0.1532% | 1.0000% | EXCLUDED — gate fired |

`pairs disqualified: 4 of 4`. **The probe publishes nothing** — the exact outcome D77 was ruled to
end. The 1.0% backstop is never reached; silence runs 5–30× below it and the GATE does all the
excluding.

`unexplained_silence_defect_signals` (`capture.py:1345`) keys the "trades walking through an
unchanged book" signal off `outside_book_flagged_runs()` — the PER-RUN flags — with `any()` over
the whole multi-day pair and no temporal scoping to the silence interval. Three independent
objections:

1. **The same evidence is spent twice.** Under `OUTSIDE_BOOK_EXCESS_RESPONSE == "run"` (active),
   a flagged run already has its localized remedy: its candidates are dropped and counted, pair
   publishes. D77's gate then uses the same 10–24 runs to void the whole pair.
2. **It restores the `"pair"` response through a side door.** `capture.py`'s own docstring for that
   constant says `"pair"` "disqualifies ALL FOUR pairs … and the probe publishes nothing at all,
   which is the exact failure this whole row exists to fix." 10 flagged runs of 1222 (0.8%) void
   the other 1212.
3. **It excludes the pair D77's text names as its counterexample.** The ruling justifies itself on
   BTC's frozen mid passing falsification (no un-freeze catch-up; drift 0.006 bp against a
   0.257 bp jump) — i.e. BTC is the named case of market stillness that must be disclosed. BTC has
   10 flagged runs, so the gate excludes it.

The fail-closed lens reached the same conclusion by inference and could not confirm it (barred from
the capture); the orchestrator's render confirms it by measurement.

**Cause note, recorded against the orchestrator:** the run-51 brief specified the D77 signals
without specifying granularity and without saying that the `"run"` response already owns that
evidence. Run 50's orchestrator recorded the identical omission about the identical signal ("I
omitted from my own cycle-2 brief what a flagged frozen run should DO"). The omission recurred in
a new place one run later.

## MAJORs

- **Estimand 2 is fully contaminated by flagged-run bars** while the artifact says "The flagged
  runs' markout candidates are excluded from every published number": published touch-fill rate
  0.6235 against an honest 0.5000, +12.35pp. `n_posted`/`n_touch_fills`/`n_trade_cross_fills` are
  incremented before and independently of the flagged-run test (`shadow_engine.py:411-418`).
  Estimand 2 is one of the two PRE-REGISTERED estimands.
- **A THIN capture counts as "publishable"** (`cli.py:152,167`): a capture producing zero markouts
  and a NO-CLAIM fill rate still collides with a genuinely publishing capture and voids the whole
  invocation. The refusal text asserts "The check is denominated in PUBLISHABILITY, not in mere
  appearance", which is false for this input. Same outcome at HEAD, so not a regression — but it
  is the D73 over-refusal BLOCKER_5 was raised to remove, unclosed, plus a false claim about the
  code's own behaviour.
- **`flagged_run_exclusions` is pair-keyed inside a per-capture loop** (`cli.py:126,193` →
  `render.py:242`): every capture's block prints the winning capture's numbers. Measured: capA's
  block shows a table of 3 flagged runs and, four lines below, "1 flagged run(s); 30 of 1980
  candidates excluded" — capB's numbers, for a pair that contributed ZERO candidates from capA
  because it was excluded there. **This is B4's original defect** (`estimand1_by_pair[pair] = …`
  assigned inside a loop over `--capture`, keyed on pair, last capture wins) reintroduced in the
  block the same cycle authored while rebuilding B4's collision check.
- **The D77 `verdict` column contradicts the artifact 50 lines later** (`render.py:501,512-519`):
  it is computed from one of the three ORed disqualifiers, so a pair excluded on book-mid freshness
  with clean silence renders `DISCLOSED — market stillness, publishes` and then
  `EXCLUDED EVERYWHERE`. Every existing D77 test uses a single-disqualifier fixture, so none can
  see it.
- **The `mid VALUE age` prose is false in the trade-print secondary table** (`render.py:677-690`
  shared by both tables): `value_age_s = age_s` by construction for that branch, so the two columns
  are equal and "read the two columns together" can never surface anything.

## MINORs worth carrying

- `capture.py:220-230` still titles the block "THE UNRULED BOUNDS", still says "NOT ONE OF THEM IS
  RULED", and still says "At today's value … NO pair publishes" — all three false post-D77. The
  same cycle added a mechanical prose scanner for the D65-amend-2 half and nothing equivalent here.
- `capture.py:258-266` carries flagged-run counts (87/22/17/4 of 996/272/105/24) labelled a
  receipt-time UPPER BOUND. The shipped code is `venue_ts`-restricted and measures 10/12/24/10 of
  1222/773/600/441. The fail-closed lens reasoned from the stale numbers. False claims have now
  migrated from rendered text into code comments.
- `capture.py:572` still ships "`venue_ts` … recorded for the record and **used by nothing**" — the
  same false sentence fixed once in `render.py`, one file over. The regression test greps only the
  rendered report.
- **The D77 backstop is aggregate-only.** D77's justification cites "worst single interval
  15.2–19.6 s" but only the summed fraction is gated. Scaled to the real capture the rule admits a
  single ~58-minute undisclosed book blackout as market stillness. Whether a worst-interval bound
  was intended is an operator question.
- Duplicated `--capture X --capture X` flipped rc=0 → rc=1 (`cli.py:167` appends per iteration, not
  per distinct path) with a remedy text that does not apply.
- `read_capture` peak RSS +10.3% vs HEAD (549,984 → 604,428 KB on a 400k-record synthetic) from
  `trade_price_series` + `trade_lag_events` + `by_hour` + `all_lags` holding three copies of the
  lag list. ~+1.2 GB on the real capture, against a render already at 14.83 GB. TASK_914 territory.
- `render.py:546,689` point readers to a "STALE BOOK TIME line"; the block is headed
  "FROZEN-MID BOOK TIME". No artifact contains that string.
- Estimand-2's drop enumeration omits `n_markouts_dropped_outside_book_run` under a
  "Disclosed, not silent" framing.

## The finding that matters more than any single BLOCKER

Three defect SHAPES have now each recurred inside the very cycle that fixed the previous instance:

| shape | instances across cycles 1–3 |
|---|---|
| a check denominated in the wrong quantity or granularity | book-freshness term; outside-book baseline; limit price; as-of mid; flagged-span endpoint; D77 gate |
| a rendered sentence the code does not back (D57) | "used by nothing" (fixed in `render.py`, still live in `capture.py:572`); estimand-2 "excluded from every published number"; `mid VALUE age` in the trade-print table; the D77 verdict column; "STALE BOOK TIME"; the undeclared-capture self-contradiction; and now the same shape inside code COMMENTS (`capture.py:220-230,258-266`) |
| pair-keyed state mutated inside a loop over `--capture` | B4's original clobber; `flagged_run_exclusions` |

Six, seven and two instances. Cycle 3 fixed one instance of each shape and created another of each.
The row has been patching instances for three cycles at ~5 BLOCKERs per cycle. The recurrence rate
is not falling.

This is what the halt rule exists to surface. A fourth cycle patching four more instances is
predicted to produce a fifth cycle. **Seeded as D78** — the structural question is whether every
input to a published number should carry its own provenance and be checked uniformly (one
mechanism), rather than each input growing its own ad-hoc predicate as it is discovered.

## Process notes

- **Orchestration error, recorded.** The five-lens panel ran in parallel, and the security lens
  performs MUTATION testing (it plants and restores edits in the live tree). Its
  `WS_URL = 'wss://evil.example/v2'` mutation landed inside the regression lens's suite window and
  produced a failure in a file not in the diff. The security lens restored correctly and verified
  sha256 on all 25 lane files; the regression lens correctly diagnosed the collision rather than
  reporting a false finding. It could have gone the other way. **A mutation-testing lens must run
  alone, never concurrently with suite-running lenses on a shared tree.**
- The orchestrator held the D77 correction rather than applying it while five judges were reading
  the tree, on the shared-tree-flicker ground run 50 recorded. That was correct and is why the
  panel's findings are trustworthy.
- The orchestrator reproduced BLOCKER_4 first-hand before briefing rather than inheriting the
  claim, and independently re-measured D65-amend-2's premise (longest gap-free stretch 9,670.9 s,
  1727 gap records, span 355,962.8 s, 39 records in the last 46,179 s — all reproducing run 50
  exactly, from separate streaming code). Both were worth the cost; the first still under-specified
  the fix, per BLOCKER 1's cause note.
- The single capped real-capture read was spent on the one question that decides the row (does the
  shipped code publish?) rather than on a full render. It returned the answer that halts the row.
- `schema_mismatch`, `unrecognized_frames` and `json_decode_errors` are **0 across all 1,161
  counters records** of the final capture, and no gap carries `schema_drift` /
  `book_checksum_mismatch`. D77's schema-drift signal is provably inert on this capture; the
  outside-book signal is the only one that can fire, which is why BLOCKER 4 decides the verdict.
- Cycle 3's implementer disclosed its two deviations plainly rather than burying them. One
  (D77's overrun-as-precondition) the correctness lens checked and confirmed defensible — the
  subset claim is true analytically and empirically. The other (MAJOR_5) the fail-closed lens
  refuted. Disclosure is what made both checkable.

## Next

The row resumes only after D78 is ruled. Resume mechanics: re-apply
`lore/handoff_2026-09-02_factory_run_51_907B2_halt.patch`, then close the four BLOCKERs — but the
D78 ruling should decide whether they are closed individually or by the single provenance
mechanism, because four individual closes is what the last three cycles did.

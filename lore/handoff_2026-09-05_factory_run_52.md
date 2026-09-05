# Factory run 52 — TASK_907 Addendum B2 PATCH ROUND 4: HALTED PERMANENTLY (D78's stop fired)

Date: 2026-09-04/05. Base HEAD `5ca8698`; HEAD moved to `1cd2c70` mid-run (planner queue grooming, `tasks/` only) and the preserved patch was re-verified `git apply --check` OK against it. Nothing committed, nothing deployed, nothing pushed.

## Outcome

**The pre-committed stop attached to D78 has FIRED.** Round 4 was declared the LAST patch round;
verification demonstrated NEW instances of the named recurring shapes, from three independent
panel lenses plus the orchestrator's own real-capture measurement. **The row halts permanently to
the provenance re-cut. There is no fifth round, and none was attempted** — attempting one would be
the same error run 51's orchestrator caught itself about to make in cycle-3 clothes.

Work preserved: `lore/handoff_2026-09-05_factory_run_52_907B2_round4.patch` (5596 lines, sha256
`e0d8f5d5a201e2582fad3b0808ac47bc0353dc08e6192995a5bc38849682a7e2`, verified byte-equal to the live
tree and `git apply --check` OK vs `5ca8698` in a detached worktree). Also left in the live tree.
Suite 362 → 387, selftest 12/12, fixture `b0f8de8b…` unchanged, 10 halt stashes intact.

**The tree is NOT shippable.** Four BLOCKER-grade findings are open in it.

## What round 4 achieved — real, measured, and reusable on the re-cut

The orchestrator's capped real-capture render (one read, rc=0, 3m32s, peak RSS 14.83 GB under a
22 G cgroup cap) **reverses the measurement that halted run 51**:

| pair | unexplained intervals | worst | measured silence | of accounted | verdict |
|---|---|---|---|---|---|
| BTC/USD | 26 | 15.6s | 303.8s | 0.0952% | DISCLOSED — market stillness, publishes |
| ETH/USD | 47 | 17.2s | 561.1s | 0.1758% | DISCLOSED — market stillness, publishes |
| SOL/USD | 8 | 15.2s | 97.8s | 0.0307% | DISCLOSED — market stillness, publishes |
| XRP/USD | 40 | 19.6s | 488.8s | 0.1532% | DISCLOSED — market stillness, publishes |

Run 51 measured `4 of 4 EXCLUDED` and the probe published **nothing**. Round 4 publishes all four.

- **The 8000 bp leak is gone on the real data**: largest |markout| in any estimand-1 cell across
  four pairs × four horizons is **9.95 bp** (run 51 published 7996.97 bp).
- **D65-amend-2 holds**: all four pre-registered horizons populated for every pair
  (BTC 8524/4509/956/465 · ETH 7356/3842/835/411 · SOL 7873/4130/890/437 · XRP 6910/3613/782/394),
  zero `not-evaluable` cells.
- **Internal reconciliation exact on all four pairs**: published + every drop reason = 4 × touch
  fills, to the unit; `posted + posts_excluded = bars posted`.
- **B-iv**: regression lens on 7 of its own fixtures, flow-verifier on 2 more, each rendered
  through base and live and rebuilt twice — cell-level identical but for the four permitted
  in-scope line shapes. B1 field identity byte-exact, including the run-51 200s/150s/50s shape.
- **B-v/B-vi**: `build_argparser` sha256 identical across base, live and bare HEAD; nodes
  362 → 387 (removed 2, added 27), edited test bodies diffed and none loosened.
- **Shape 3 did NOT recur** — three lenses independently re-derived the `--capture`-loop dict
  enumeration; all capture-keyed or append-only. The run-51 `flagged_run_exclusions` clobber is
  genuinely closed.
- BLOCKERs 1–3 are substantively closed: the trailing-window leak (7997.47 bp at base) no longer
  publishes, the undeclared-capture 2b detector exists and flags, `MidQuote` carries required
  provenance and its gate is proven non-vacuous.

## Why it halted — the new instances

### 1. The frozen-top run is BLIND to recorded gaps; 94% of the real defect signal is disconnect (ORCHESTRATOR, measured — SHAPE 1)

A frozen-top run is built from consecutive book frames with no reference to the gap ledger, so its
span can be mostly time the instrument was **not observing at all**.

Measured on the real capture (gap ledger extracted from the capture itself, 1727 records, merged;
run spans parsed from the round-4 artifact):

- **63 of 67 FLAGGED runs overlap a recorded gap**, and **63 of 67 have more than half their span
  inside one** — typically 96–99%, worst **99.7%** (a 61.8s run of which 61.6s is disclosed
  outage). By pair: BTC 13/17, ETH 12/12, SOL 27/27, XRP 11/11 — every flagged run on three pairs.
- The recurring `61.5s` signature is D74's own journal-stall `clock_jump` regime.

So D77's PRIMARY defect signal — "trades walking through an unchanged book" — is, on the real data,
overwhelmingly *"the book looked unchanged because we were disconnected"*. The artifact's excluded
posts (82/54/122/52) and dropped markouts (101/48/104/36) rest on it, and nothing tells a reader
that a flagged run's span is mostly outage.

Base A/B on an orchestrator fixture (one disclosed disconnect, matching quotes either side,
30 post-reconnect prints outside the book):

| tree | run | verdict | n | flagged |
|---|---|---|---|---|
| base (run-51 state) | (1495.0, 1625.0) | NOT ASSESSED | 1 | none |
| round 4 | (1495.0, 1645.0) | **FLAGGED** | 31 | yes |

`stale_book_s` is 130.0 in **both**. The gap-spanning construction is inherited from `stale_runs`;
the **flagging consequence is created by this round**. Independently surfaced on fixtures by the
regression lens, which also showed it adds an un-enumerated artifact line on an otherwise healthy,
fully-disclosed capture (a B-iv breach).

### 2. The endpoint fix inverts the baseline RATE and un-flags defective runs (correctness lens — SHAPE 1)

Widening every run from `(start, last_repeat)` to `(start, first_changing_frame)` also widens the
**baseline exclusion**. The brief's fail-closed argument — "superset of before → baseline shrinks →
fail-closed direction" — is denominated in the baseline **population size**, while the gate divides
by the baseline **rate**. Removing predominantly inside-the-book trades *raises* that rate, raises
the threshold, and **un-flags** runs.

Measured, same fixture, both trees: base `baseline_rate 0.4783` (n=460) FLAGS a run at ratio 2.09;
round 4 `baseline_rate 0.5500` (n=400) calls the same run **`clean`** at ratio 1.82. 74 markout
samples the base tree excluded are published at rc=0 under an outside-book table reading `clean`;
published 1h mean moves −471.80 → −450.15 bp. **Fail-open, data-dependent, unbounded in sign.**
No test touches `outside_book_baseline_rate`, and all four B-iv fixtures have a top of book that
moves every frame — zero frozen runs — so none can reach it.

### 3. The defect-silence exclusion gates 2 of 4 inputs (fail-closed lens — SHAPES 1 AND 2, on new code)

The flagged-run gate tests four inputs (limit print, fill print, target instant, as-of
observation). The new defect-silence exclusion tests only the horizon window and the evaluation
bar — never the **limit's own print**. Measured: a limit priced at a close inside a defect-flagged
interval publishes **+2298.27 bp at rc=0**, moving the published 1m mean **0.453 → 8.689 bp**,
across the disclosure box's own 5 bp decision line and in the direction of the row's preferred
conclusion. The artifact simultaneously asserts the exclusion "reaches EVERY published number" and
enumerates the limit print among the inputs tested. The overrun threshold is 10s, so any 11–59s
interval fits inside one posting bar — the real capture's worst intervals are 15.2–19.6s.

### 4. The new "none" sentence contradicts the exclusion, 30 lines apart (CONVERGENT — 2 lenses — SHAPE 2)

The defect-interval block's empty branch prints *"nothing is excluded on account of silence and the
pairs publish in full"* whenever no interval carries a defect signal — precisely the state of a pair
excluded by the two-sided **backstop**. One rc=0 artifact contains that sentence, the `**EXCLUDED**`
verdict row, and `EXCLUDED EVERYWHERE`. This is the run-51 D77-verdict-column contradiction
**re-created three lines below the cell that was fixed to remove it**, in the block written this
round.

## The panel split, recorded honestly

| lens | model | verdict |
|---|---|---|
| correctness | opus | DO NOT SHIP — 2 BLOCKER (both new shape instances), 2 MINOR, 1 NIT |
| fail-closed | opus | DO NOT SHIP — 1 BLOCKER (shapes 1+2 on new code), 2 MAJOR, 2 MINOR |
| regression | sonnet | DO NOT SHIP — 1 BLOCKER; B-iv/B-v/B-vi/B1 all independently CLEAN |
| flow-verifier | sonnet | **SHIP**, with a self-stated mandate caveat |
| security (mutation) | — | NOT RUN: the row halted before its solo slot; disclosed |

The flow-verifier's dissent is real and is not suppressed: all nine of its end-to-end flows pass by
positive reproduction, including hand-cut estimand-2 equality. It states its own limit — "my mandate
was behavioral verification of the 9 named flows, not a full line-by-line code review". **That is
exactly why the round looks shippable from the artifact alone**, and why a green suite plus nine
green flows cannot answer a wrongly denominated gate.

## Corrections against the orchestrator (recorded)

1. **The brief shipped the defect the correctness lens found.** Section 1 of the round-4 brief
   states: *"Baseline exclusion (a) uses the same runs (superset of before → baseline shrinks →
   fail-closed direction)"*. That argument is in the wrong quantity — population, not rate. The
   implementer built what the brief asserted and repeated it in its self-audit. **This is the third
   consecutive run whose brief mis-stated or under-specified a DENOMINATION**: run 50 ("I omitted
   what a flagged frozen run should DO"), run 51 (D77's signals without their granularity), run 52
   (a fail-closed direction argued in the wrong quantity). The recurrence is the orchestrator's,
   not the implementer's.
2. **The brief never considered recorded gaps.** It specified the endpoint in terms of book frames
   alone and never asked what a run means across an interval with no wire evidence — though
   `covers()` embodies exactly that principle one layer down. Its B-iv instruction said "healthy"
   and produced four fixtures with **zero** `GapRecord`s; the finding surfaced only because the
   panel brief asked the regression lens for a gap fixture.
3. **The panel preamble did not warn that the orchestrator's own sanctioned capped read was in
   flight.** The flow-verifier, correctly reading "the orchestrator does that read", attempted
   `kill -9` on it; the environment's permission classifier blocked the kill and the 3m32s job
   completed. Future preambles must name any orchestrator job running concurrently.
4. Process that worked: the mutation-testing security lens was scheduled to run ALONE (run-51's
   finding), the four suite-running lenses ran concurrently without collision, and the deciding
   real-capture analysis was taken first-hand rather than inherited.

## Next — the re-cut has a concrete first requirement

Seeded as **D81** (OPEN). NB: D80 was claimed by a concurrent planner grooming (`1cd2c70`, Kraken fee tier / goalpost restatement / 907 phase-2 rails), which landed in `tasks/QUEUE.md` mid-run; that row is the planner's to write into DECISIONS. The re-cut's first requirement is now measured rather than theorised:

> **Every interval the instrument reasons about must carry whether the instrument was OBSERVING
> across it.** 94% of the real capture's defect signal is disclosed disconnect. `covers()` already
> applies this to markouts; the frozen-run detector, the outside-book baseline and the silence gate
> do not.

A provenance mechanism attaching observation-state to every span subsumes all three of this round's
shape-1 findings: the gap-blindness directly; the baseline-rate inversion (the baseline would
exclude on observation-state rather than on a widened span); and the un-gated limit print (one
mechanism gating every input to a published number, which is D78's original question).

**An intermediate the operator may prefer**, if the D65 number is wanted sooner than the re-cut can
deliver it: publish estimand 1 with gap-aware exclusion and **without** the 2b defect-signal
machinery, disclosing that the frozen-run signal is withdrawn as unreliable. The probe demonstrably
publishes all four pairs and four horizons with a 9.95 bp maximum. That is a scope decision, not a
fifth patch round, and it should be ruled explicitly rather than drifted into.

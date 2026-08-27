# Factory run 44 — handoff (2026-08-27)

Planner lane armed the factory on the post-run-43 queue. **One row shipped, one row halted again on a
pre-committed stop, three decision rows seeded.** Nothing pushed. Tree clean apart from the
deliberately-uncommitted 907 lane.

| row | outcome | verification |
|---|---|---|
| **TASK_909** | **SHIPPED** — but as a RECOVERY, not the dead-test cleanup the row expected | 897 suite `3 failed, 250 passed` → **`253 passed`**, zero skips; selftest 12/12; correctness + regression panel, plus an escalated re-verify of the orchestrator's own fix |
| **TASK_907** | **HALTED AGAIN — D66's pre-committed stop FIRED.** Cycle 4 green on its own suite but not shippable | own suite **173 passed** (was 152); 5-lens panel at opus: fail-closed, correctness, regression, security, flow-verifier |

Seeded: **D67** (907's restructure — direction confirmed, scope must widen), **D68** (the 907 lane's two
safety findings; one means acceptance gate (ii) is not met), **D69** (a verified-deployed gitignored
artifact silently regressed; mechanism unclosed).

---

## 1. TASK_907 — the stop fired, and (b) as written does not fit what was found

D66 authorised ONE cycle for the named fix with a pre-committed stop: any NEW absence granularity
demonstrated by a lens → straight to (b), no fifth cycle. **Four were demonstrated by three
independent parties.** The mandated Condition-2 lens was briefed BLIND — told nothing of what the
implementer or the orchestrator had found — and converged anyway. The capture-set defect was
reproduced five times over. This was not a close call.

1. **per-horizon total absence** (`estimands.py:148`) — and the code violates its own
   pre-registration: `PREREGISTRATION.md:117-121` commits in writing that an unevaluable horizon is
   "still reported (rather than omitted)". The live case: a 3-hour capture with three ordinary 120s
   reconnect gaps ~50 min apart makes *every* 1h window straddle a gap, so the 1h estimand is never
   measured and the artifact reads complete. **Harm runs the forbidden direction** — adverse selection
   grows with horizon, so silently dropping the longest horizon deletes the evidence most adverse to
   the model, which is this row's own preferred conclusion.
2. **cross-capture same-pair clobber** (`cli.py:77-80`) — plain dict assignment keyed on pair alone,
   inside a loop over a repeatable `--capture`. Two pairs measured over disjoint windows **27 hours
   apart** rendered in one artifact, one pair's whole sample destroyed, both captures certified in the
   provenance.
3. **the RESUME path re-opens the halting MAJOR** (`capture.py:252/266`) — `meta_start` is written only
   on a fresh file; on resume the `pairs` argument is silently discarded, so `declared_pairs` is a
   snapshot of session 1. A pair added on restart and dead from t=0 gives `is_complete()` True and
   appears **zero** times in the artifact — *worse* than pre-cycle-4, where the union at least put it
   on the provenance line. **So D66 Condition 1 is closed on the fresh-capture path and OPEN on
   resume**, and a 3–7 day capture is exactly where process restarts happen. Cycle 4 does not fully
   close its own named MAJOR — which settles the row independently of D66.
4. **the aggregate bound is denominated in the span that collapsed** (`capture.py:914-919`) — the
   deepest finding of the row, and the one that generalises the family: coverage is computed over
   `[data_start_ts, data_end_ts]`, never reconciled against `[start_ts, end_ts]`, so **the denominator
   shrinks with the failure**. A 7200s capture carrying data only in its first 130s, heartbeats
   throughout: all four defenses clean, `uncovered_fraction 0.000%` against a true 98.188%
   un-observed, and it renders `1m: median -0.64bp (n_used=1) — 0bp: meets/exceeds…`. Mechanism:
   `_find_unrecorded_gaps` iterates strictly BETWEEN events, so EDGE holes have no spanning pair —
   interior refuses, leading/trailing/both-edge pass. **This is precisely the symptom the run-43 banner
   named as the thing to watch — "connected but zero events recorded" — with the recorder staying up.**

**The unifying principle, and the real content of the restructure (D67):** every DECLARED quantity must
be reconciled against what was observed at EVERY boundary, and **no fail-loud check may be denominated
in the quantity it is checking.** Two of the four escapes are not in the gate at all (estimands, cli),
so D66's "(b) = gate restructure" would fix neither — that is why D67 exists rather than a silent
execution of (b).

**Cycle 4 is nonetheless real work; reuse it.** Preserved at
`lore/handoff_2026-08-27_factory_run_44_907_cycle4_halt.patch` (21 files, 6111 insertions) and left
untracked in the tree. It closed the halting MAJOR on the fresh path and the (pair × channel) cell,
with Condition 1 shut at BOTH ends there (the write-side test drives the real `LiveRecorder.run` with
the production writer). `meta.pairs` keeps its old union meaning; nothing was silently re-defined.
Fail-CLOSED on a missing OR empty declared set at both format versions, with the bypass named in the
refusal. **NO NUMBER MOVED, proven**: a healthy 3-pair capture renders byte-identical artifacts pre/post
except two added provenance fields. The edited pre-existing fixture is honest — 8 lines, 6 of them
comment — and mutation-proven to cost nothing in detection power. 20 of 21 new tests fail against
pre-fix code, behaviourally rather than on `AttributeError`.

**Also found, and the restructure must not inherit it:** the new per-cell check OVER-refuses, treating a
legitimately empty `trade` cell (no prints, book flowing and proving the socket alive) as identical to a
dead `book` cell, discarding every number from every other capture in the same render — and it fires
only when `channels` is declared, i.e. only on real recorder captures, **so the 173-test suite is
structurally blind to it**. Plus the reconciliation runs at read time only, so drift surfaces after
3–7 days of unrepeatable recording, with the recorder's own drift counters reading zero throughout.

## 2. Two safety findings on the row whose phase 2 sends real money (D68)

- **The "no code path can transmit an order" invariant is pinned by NO test.** It is prose, re-argued
  each cycle. Demonstrated: appending a market `add_order` frame carrying a token to the send loop
  leaves the suite GREEN; switching `WS_URL` to Kraken's authenticated endpoint leaves it GREEN.
  Orchestrator-verified: no test references `WS_URL`, `wss://` or `ws.kraken`, and cycle 4's coupling
  test filters to `method == "subscribe"` **before** counting, so `len(subscribed) == 2` constrains
  nothing about the total frames sent.
- **Acceptance gate (ii) is NOT MET.** `check_arming` demands `--budget-cap-usd`, then no rail ever
  reads it; `phase2_rails.py:189-192` guard both notional caps with `if ... is not None`, both
  defaulting to `None`. A **$1,000,000** order against a declared **$50** cap clears every rail. Inert
  today only because `place_order` raises — the entire notional protection rests on the one conjunct
  the phase-2 increment exists to remove.
- Corrected in the record: the run-43 banner's "nothing anywhere catches `NotImplementedError`" is
  **false** — `recorder.py:297` is a broad `except Exception`. Verified twice that it wraps only
  connect/subscribe/consume with no order path inside, so the safety claim holds in substance; but it
  is a reading, not a pin, and the task file says phase 2's orders are "placed by the same engine".

## 3. TASK_909 — the reds were a live detector, not dead tests (D69)

Cycle 1 inspected the stale local file, found `single_cell_gate_lines` nowhere on disk, concluded it
"never existed", and chose delete-with-prejudice. A correctness lens refuted the premise from the
tracked record: `TASK_897_CORRECTED_809_REMEASURE.md:300-303` attests the gitignored
`recompute_best.py` repair was sha256-verified identical on both hosts on 2026-08-23. **srv still held
it** (7780 B, `2026-08-22 01:46`, sha256 `783b2ea3…`); the workstation had reverted to the June pre-fix
revision (4549 B, `2026-06-22 22:11`). Recovered, installed, tests green: **253 passed**.

**An orchestrator misreading is retracted in the record:** mid-run I concluded the CHANGELOG asserted a
repair that never landed. Wrong — I reasoned from disk state alone. The repair landed and was deployed;
the working tree lied. The 2026-08-22 entry had even named the hazard ("invisible to the clean-tree
gate") two bullets after asserting the fix.

**Mechanism NOT closed.** The June copy retained its mtime *with* June content → an mtime-preserving
restore, not an edit. **All ~120 `runs/*` directory mtimes were rewritten in one sweep
`2026-08-25 22:24:50 → 22:31:19`, outside any logged factory window** (`run-20260825.jsonl` ends
`21:48:02`). The `.pyc` from the 08-22 test run is gone; `decide.cpython-310.pyc` (June) survived. Two
out-of-repo trees hold copies, one bit-identical to the pre-fix revision, one a third divergent
revision. **Hypothesis, explicitly unverified:** 2026-08-25 is the run-42 incident date, and a recovery
restoring gitignored `runs/` artifacts from an older tree with preserved mtimes would explain all of it
at once — i.e. the run-42 *recovery* may itself have re-introduced stale artifacts. **The same event
would have reverted any other gitignored `runs/` artifact, and most have no srv counterpart to diff
against.**

**Gate (i) narrowly missed and named:** greenness rests on a gitignored file, so on a fresh clone the
three tests SKIP. The fix is not reproducible from the repo; the CHANGELOG's sha256 is its only tracked
trace. Blast radius measured and bounded to exactly one file — srv's three other differing hashes match
exact git blobs (`e5c3864`, `5ee6381`), i.e. srv is strictly behind, and all of `runs/806_*`,
`runs/808_*`, `runs/809_*` is now hash-identical host-to-host.

## 4. Process notes worth keeping

- **A red test at HEAD is evidence, not noise.** 909's premise (red suites train red-fatigue) is right,
  but the cure it reached for was deletion — the one remedy that destroys the evidence instead of
  reading it. Consult the tracked record BEFORE any delete-with-prejudice; "this never existed" is a
  claim about history that only git history and the written record can settle.
- **Briefing an adversarial lens BLIND is what made the Condition-2 finding worth anything.** The
  orchestrator held its own candidate (the pair × channel cell) back rather than handing it over;
  cycle 4 turned out to have closed that one, and the blind lens found four others.
- Piping pytest into `tail`/`head` masks its exit code (you get the pager's status). `safe_run.sh`
  itself propagates exit codes correctly — verified, returns 7 for `exit 7`. Parse the
  `N failed, N passed` line.
- Concurrency held: peak 7 agents, load peaked ~3.9 — well inside the run-40 lesson (8 agents → 17.6).
  Every suite run was wrapped in `safe_run.sh`.

## 5. QUEUE DRIFT — planner action needed

`tasks/QUEUE.md` still reads *"Next takeable: TASK_907 per D66(a)"* and lists 907 as row 1. That is now
stale: 907 is halted to D67 and must not be taken as a patch row. **The factory did not edit QUEUE.md**
— it is planner-maintained, and the contract says banners beat the queue on conflict and drift gets
noted here. Both task banners are authoritative. Suggested re-rank once D67–D69 are ruled: 881E → 905 →
904B → 908 → 906, with 907 gated behind D67 and the D68 safety row inserted wherever the operator wants
the phase-1 invariant pinned.

## 6. State

- **OPEN decisions: D43, D67, D68, D69.**
- srv reachable; used read-only this run (the 909 recovery + hash comparisons). Nothing launched there.
- The 907 shadow capture is still NOT launched. Kraken v2 schema smoke remains planner-side and
  blocking — and D67 sharpens it: finding 4 means the watched smoke must compare the DATA span against
  the CAPTURE span by hand rather than trusting `is_complete()`.
- Nothing pushed. `guards.allow_push` false, as always.

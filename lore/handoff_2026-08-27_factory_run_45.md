# Factory run 45 — handoff (2026-08-27 → 08-28)

Planner lane armed the factory with no steering args; queue taken from the top.

## 1. Shipped

| commit | item | verification |
|---|---|---|
| `28e8d34` | **TASK_905** — `validate --fold-root` audited the ORIGINAL tree; a false PASS on a corrupted copy, closed | 3-lens panel → 1 MAJOR + 2 MINOR + 1 NIT, all closed by an opus fix pass. Suite 66→69, selftest 12/12, smoke PASS, gates (i)–(iv) re-proved after the fix |
| `30c5ce1` | **TASK_907 ADDENDUM A** — the Kraken v2 book channel could not read the live wire | 3-lens panel (fail-closed at opus) → **1 BLOCKER** + 4 MAJOR + 4 MINOR, all closed. Suite 173→217, selftest 12/12 |
| `86150d9` | **TASK_911** — identity pins for load-bearing gitignored files (D69) | correctness panel → 1 MAJOR + 3 MINOR, all closed. 897 suite 253→256, module suite 30→54, selftest 12/12 |

## 2. The finding of the run

**Addendum A's BLOCKER, and why it nearly shipped green.** The recorder reconciles its maintained
order book against the venue's CRC32 on every row. But the *emitted* top-of-book was a SEPARATE
derivation from that state, and nothing reconciled the two. Reinstalling the original defect left
the suite **fully green** while corrupting **482 of 590 emitted tops** on real frames — one ~18 bp
wrong. The acceptance criterion had explicitly demanded that mutation go RED; it did not.

Worse, the single hand-checked frame (300) happened to be an empty-bids row whose lone ask delta
IS the best ask — the one row shape on which the defect is invisible. The verification landed, by
chance, on the only place the bug could not be seen.

**One hand-check is not a proof.** When the fixture is already being replayed, check every event.
Closed structurally rather than by test: `_book_view` now returns the checksum and the top from one
ordered structure, so the second derivation does not exist.

## 3. Evidence added to the record

**The live Kraken v2 wire, measured.** 5,737 raw frames on `gpu-server` (45s, BTC/USD + ETH/USD,
depth 10, public, no credentials). 600 preserved as a tracked fixture (`378fa3dc…`).

- `book/update` 5,663 · `heartbeat` 44 · `trade/update` 23 · method-response 4 · `book/snapshot` **2**
- **5,661 of 5,665 book rows carry an empty side** — the planner's "3,538 mismatches in 75s".
- **Updates are INCREMENTAL deltas, not top-of-book** (a delta at 2505.76 while the best bid was
  2505.81). The four rows the old code accepted were recording a wrong top of book as real.
  **Fixing only the rejection would have produced 5,661 silently wrong mids — worse than the bug.**
- **The venue's CRC32 reconstructs 5,665/5,665** from snapshot + deltas + depth-10 trim. It is over
  wire STRING forms, so `parse_float=str` is load-bearing.
- Top-of-book change rate: BTC 478/44s, ETH 700/44s — ~4.8× compression vs the raw frame rate.

## 4. Decisions seeded

- **D70** — TASK_881E's authority binding shipped **default-OFF** behind a flag. The invariant does
  say new behaviour ships flag-gated, but this row closes a MEASURED path where any manifest can
  authorize a delete, and it gates lifting the `--apply` freeze on a tool that deletes remotely.
  Default-OFF leaves the path that actually runs as vulnerable as D64 found it. The reason given was
  about **test fixtures**, not production safety. The measurement that would settle it (how many
  tests break with the binding ON, split by real-protection vs placeholder-convenience; and gate
  (ii)'s coverage diff) was requested and never returned.
- **D71** — `runs/875_wf/orch_wf875.sh` is gone from **both hosts**, with no sha256 ever recorded
  (`notes_2026-08-19.md:200-203` says "sha256-verified identical" and never wrote the value down).
  Found by TASK_911's own manifest on its first run. Second confirmed load-bearing gitignored file
  lost after the 2026-08-25 restore sweep.

## 5. Halted

**TASK_881E** — halted, work preserved, tree reverted, **no gate verified and no panel ever ran.**
The cause was NOT a failed cycle: the agent entered a wait-loop and stopped six times across 379k
tokens without finalizing. Preserved at `handoff_2026-08-27_factory_run_45_881E_halt.patch` (419
lines, **verified to apply cleanly against HEAD in a throwaway worktree before the revert**) and
`..._881E_test.py.txt` (392 lines). Reverting rather than leaving it dirty was deliberate:
`tools/ckpt_gc.py` DELETES, and a half-modified deleter in a shared live tree is its own hazard.

## 6. Process lessons

1. **The never-stash rule belongs in EVERY brief, not just judge briefs.** An implementer ran
   `git stash` on the live tree with five agents in it. It scoped the stash to its own two paths, so
   nothing was stripped — but a blanket stash would have torn through the 905 fix mid-write. It is a
   CONCURRENCY hazard, not a review hazard. The standing memory note is written as the latter, which
   is why it was briefed to judges and omitted from implementers. Memory updated.
   Recovery facts: a scoped `git stash push <paths>` still prints the generic `WIP on master:`
   message, so the message does not reveal blast radius — use `git stash show --stat`. And this repo
   carries ~10 halt-preservation stashes holding the ONLY record of several halted tasks; an
   accidental stash shifts every index, so fence a recovering agent to `stash@{0}` explicitly.
2. **A mutating judge needs its own worktree.** Three judges shared the 907 lane and one was told to
   mutate. The suite flickered `3 failed/182` → `2 failed/183` → `185 passed`; the correctness lens
   avoided reporting phantoms only because it pinned findings to md5-stable content. Read-only lenses
   may share a tree; a mutating lens may not.
3. **"Agent completed" ≠ "agent finished"** when it is waiting on a background job. The orchestrator
   restored a corrupt gitignored file under an agent that was mid-mutation-proof and intended to
   restore it itself. No harm — the agent re-restored explicitly and the final hash verified — but it
   briefly looked to that agent like an unattributable background write, which is the exact class D69
   exists to catch. Check for an in-flight proof before repairing state that looks broken.
4. **One hand-checked sample is not a proof** — see §2.
5. **A spec error is the orchestrator's to own.** Addendum A.4(4) described the keepalive as a TIMER
   and justified it with "otherwise a quiet book reads as a dark socket". A timer would write a top of
   book no frame confirmed at that `ts` — an inference from absence recorded as an observation, on the
   row that has halted four times on exactly that. The implementation was right; the spec was wrong,
   and was corrected in the task file rather than charged to the code. Separately, the
   "additive `meta_start` only" fence on `capture.py` is what made the declaration inert — a fence
   that forbade the only code able to fulfil the requirement.

## 7. State

- **OPEN decisions: D43, D70, D71.**
- **The 907 shadow capture is still NOT launched**, and Addendum A does not by itself authorise it.
  A.4(5)'s watched live re-smoke (`n_schema_mismatch ≈ 0` against the real feed) is planner-side and
  undischarged; the fixture is 45s of one afternoon, not proof the schema is stable. **D67(b)'s four
  preconditions all still bind, and two are now MORE load-bearing because the defects they defend
  against are ADDENDUM B and are not fixed:** (ii) the hand-computed data-span vs capture-span
  comparison — B2 is open, so `is_complete()` is still blind to an edge hole and must not be trusted;
  and (iii) the FIXED pair set — B1 is open, so operational discipline is the only defence.
- `python -m tools.verify_gitignored_pins` exits **1** on this workstation until D71 is ruled.
- srv reachable; used read-only plus one 45s public-feed capture. Nothing launched there.
- Nothing pushed. `guards.allow_push` false, as always.

## 8. Queue drift — planner action needed

The factory did not edit `tasks/QUEUE.md` (planner-maintained; run 44 set the precedent). Banners
and this document are authoritative.

| row | ranked | actual |
|---|---|---|
| 1 `TASK_907` | Ready #1 | **SPLIT.** Addendum A SHIPPED; **Addendum B specced, NOT started** |
| 2 `TASK_910` | Ready #2 | not started |
| 3 `TASK_881E` | Ready #3 | **HALTED → D70** |
| 4 `TASK_905` | Ready #4 | **SHIPPED** |
| 5 `TASK_904B` | Ready #5 | not started |
| 6 `TASK_908` | Ready #6 | not started |
| 7 `TASK_911` | Ready #7 | **SHIPPED** (i)+(ii); **(iii) is planner work** |
| 8 `TASK_906` | Ready #8 | not started |

**The structural change to absorb: row 1 is now two rows.** ADDENDUM B is written into
`TASK_907` in full, and **all five declaration boundaries plus four inherited in-gate claims were
reproduced FIRST-HAND against the current tree this run** — measurements, not inherited claims:

- **B1** session 2's declaration discarded; `ETH/USD` appears **0 times**, gate PASSES
- **B2** 98.188% of the capture un-observed, reported **0.000%**, `assert_complete()` PASSES
  (`accounted_s=130, span_s=130` — the denominator IS the collapsed quantity)
- **B3** pre-registered `1h` silently omitted from a 25-minute capture
- **B4** the **LAST** `--capture` silently wins: `A B` → 26.34, `B A` → 22.03, both provenanced as two
  captures. Order-dependence, not just clobber
- **B5** `--pairs SOL/USD` → exit 0, 3020-byte gate-passing artifact, zero estimands, filter invisible
- plus: the gate OVER-refuses a book-alive/print-empty pair (`rc=1`, no artifact, and the refusal's
  own stated premise is false for that pair); the twins diverge (`had_meta_start=0` → `is_complete()`
  False while `assert_complete()` PASSES); `channels: []` disables both channel checks while
  `declared_pairs: []` refuses; and `channels=['trade']` on a book-full capture reconciles perfectly

Suggested re-rank: **907-Addendum-B → 910 → 904B → 908 → 906**, 881E gated behind D70, 911 gate (iii)
planner-side after D71.

**Addendum B was deliberately not started at the tail of this run.** It is a large single-lane
restructure touching `capture.py`, `estimands.py`, `render.py` and `cli.py` together, on the row that
has halted four times — each time on a fix that was better than the last and still one granularity
short. Starting it tired, with an unverified Addendum A fix in flight in the same lane, is how a
fifth halt gets manufactured. D67(b) decouples it from the calendar: the capture launches on A.

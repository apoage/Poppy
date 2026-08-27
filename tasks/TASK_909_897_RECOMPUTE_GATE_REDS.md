> **STATUS (2026-08-27, factory run 44): SHIPPED — but NOT as this row expected. The three reds were
> NOT dead tests. A verified-deployed artifact had silently REGRESSED, and the fix was RECOVERY, not
> deletion.** 897 suite: `3 failed, 250 passed` → **`253 passed`, zero skips** (orchestrator-rerun,
> re-verified independently). `selftest` 12/12. Gate (ii) MET (see the CHANGELOG entry and the commit
> message). **Gate (i) is narrowly MISSED and named below.** Mechanism → **D69**.
>
> **The row's fork was answered wrong first, and the record is what caught it.** Cycle 1 inspected the
> stale local `runs/809_decision_rule_rescue/recompute_best.py`, found `single_cell_gate_lines` in no
> `.py` file anywhere on disk, concluded it "never existed", and chose delete-with-prejudice. A
> correctness lens refuted the premise from the tracked record:
> `tasks/TASK_897_CORRECTED_809_REMEASURE.md:300-303` attests that on 2026-08-23 the 13
> launch-critical files **"incl. the gitignored `recompute_best.py` repair"** were sha256-verified
> IDENTICAL on both hosts post-deploy. The delete was reversed and the real artifact recovered.
>
> **What was actually wrong.** Workstation: 4549 bytes, mtime **2026-06-22 22:11** — the June pre-fix
> revision, two-criterion gate still at line 90 (`sh_reb >= 1.0 and lo > 0`, no `PnL > 0`).
> `gpu-server`: 7780 bytes, mtime **2026-08-22 01:46**, sha256
> `783b2ea35754d057bed2509b6ea3ffa1f3022473eb9247c706cd0091f4931ffc`, carrying the repaired
> three-criterion gate whose own docstring names
> `test_recompute_gate.py::test_volatility_drag_cell_is_failed_by_the_pnl_criterion` as its pin.
> Recovered, sha256 re-verified after transfer, installed; the pre-fix copy retained out-of-tree.
>
> **An earlier reading is RETRACTED.** Mid-run the orchestrator concluded that CHANGELOG:570-573 and
> this lane's banner asserted a repair that never landed. That was wrong — it reasoned from disk state
> alone. **The repair landed and was deployed; the working tree was what lied.** Notably the
> 2026-08-22 entry named this exact hazard two bullets after asserting the fix ("staging it is a
> manual step invisible to the clean-tree gate"), and that is the hazard that fired.
>
> **Blast radius bounded by measurement, not assumption.** Every `*.py` under
> `experiments/897_corrected_809/` and `runs/809_decision_rule_rescue/` was sha256-compared across
> hosts. Three others differ; all three are tracked, clean vs HEAD, last modified by today's
> `aa2a802` (TASK_902), and their srv hashes match **exact git-history blobs** (`e5c3864`, `5ee6381`)
> — srv is strictly BEHIND, not divergent. Every `.py` in `runs/806_*`, `runs/808_*`, `runs/809_*` is
> now hash-identical host-to-host. **The only anomalous file was the one git cannot protect.**
>
> **GATE (i) MISSED — stated, not papered over.** The gate demands "fully green at HEAD with zero
> skips of this file's tests". Greenness rests on a gitignored file (`.gitignore:54: runs/*/*`, never
> tracked at any point per `git log --all --diff-filter=A`), so on a fresh clone or a new host the
> three tests **SKIP** (measured). The fix is **not reproducible from the repository** — it is
> srv-anchored, and the sha256 recorded in the CHANGELOG is its only durable tracked trace. D69
> carries the remedy options (a tracked identity pin, reusing the `FIXTURE_IDENTITY.json` /
> `frame_symbols_sha256` pattern; or force-tracking; or accepting srv as the second copy).
>
> **This row's tracked diff is EMPTY, and that is the finding.** The test file was restored to its
> HEAD state and the repair is gitignored, so git cannot record this fix at all.
>
> **The revert mechanism is NOT closed** — forensics in D69 and the CHANGELOG entry: an
> mtime-preserving replacement, and all ~120 `runs/*` directory mtimes rewritten in one sweep
> `2026-08-25 22:24:50 → 22:31:19`, outside any logged factory window. Hypothesis (explicitly
> unverified): the run-42 recovery itself re-introduced stale gitignored artifacts. **The same event
> would have reverted any other gitignored `runs/` artifact, and most have no srv counterpart to diff
> against.**
>
> **Standing lesson, worth carrying into every future brief: a red test at HEAD is evidence, not
> noise.** This row's premise — red suites train red-fatigue — is correct, but the cure it reached for
> was deletion, and deleting a red test is the one remedy that destroys the evidence instead of
> reading it. "This function never existed" is a claim about history that only git history and the
> written record can settle, never a stale working file.
>
> **Remainders (pre-existing, faithfully restored, not introduced here):** both `recompute_best.py:56`
> and `test_recompute_gate.py:9` cite "TASK_897 line 45" for the band's criteria, which now live at
> `:254` while `:45` reads unrelated text (the TASK_902 honesty-debt class);
> `test_recompute_gate.py:44`'s "import is side-effect-free" is true of the DB but the module prints
> at import; the frozen artifact reads `ci_lo_UNCALIBRATED` while the pipeline now publishes
> `ci_lo_ADVISORY` — compatible today, exercised by no test, since nothing in the suite runs `main()`.

# TASK_909 — the 897 suite's 3 standing red tests at HEAD

**Status**: READY 2026-08-27 (planner; seeded from run-43's open item 4).
**Size: XS. Risk: LOW, but red-fatigue is the hazard** — a suite that stays red at HEAD trains
every future verification lane to discount red, which is exactly how the next real regression
gets waved through.

## The defect

`experiments/897_corrected_809/test_recompute_gate.py` has 3 failing tests at HEAD, targeting
`recompute_best.py::single_cell_gate_lines` (runs/809 lineage, gitignored). Pre-existing,
unowned, and out of scope of every row that touched the suite (902 shipped green with "3
pre-existing" exclusions named).

## Scope

Own them, one way or the other: **fix** the tests/code if the gate semantics are still live, or
**remove-with-prejudice** (a named `SKIP: superseded by X` is not acceptable — DELETE the test
or repair it; a skipped test that never runs again is a lie of omission in the suite). The
decision turns on whether `recompute_best.py`'s gate still has a consumer after the 897 verdict
(ALL FAIL, lane closed) — if nothing reads it, deleting the dead gate's tests AND saying so in
the commit is the honest close. **AMENDED per D69 (run 44 — this row's delete path nearly
destroyed the evidence of a silent regression): a red test at HEAD is EVIDENCE, not noise;
consult the tracked record (CHANGELOG, banners, git history) BEFORE any deletion, and "the
function never existed" is a claim about history that only the record can settle, never a stale
working file. (Outcome for the record: the reds were a LIVE detector — the deployed repair had
silently regressed on the workstation; recovered from srv; tests green with nothing deleted.)**

## Acceptance (gates)

- (i) `pytest experiments/897_corrected_809/` is fully green at HEAD with zero skips of this
  file's tests.
- (ii) The commit message records which semantics were judged dead vs live, and why.

## Out of scope

Any change to the 897 artifact or its numbers. The 809 lineage beyond the failing gate's
surface.
